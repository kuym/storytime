// storytime: Kokoro-82M TTS CLI.
//
// Reads text or IPA phonemes from stdin, runs Kokoro via ONNX Runtime
// (CoreML EP on Apple Silicon, CPU fallback), and either writes a WAV
// file (with `-o PATH`) or plays the audio directly to the default
// output device (no `-o`).
//
// Text mode shells out to `espeak-ng` for grapheme->IPA conversion.
// IPA mode (--ipa) skips that step so the tool composes in a POSIX pipeline.
//
// Playback uses OS-native APIs with no third-party crates:
//   - macOS: AudioToolbox's AudioQueue (C API, system framework).
//   - Linux: ALSA's libasound (the lowest common denominator).

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use hound::{SampleFormat, WavSpec, WavWriter};
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

#[cfg(feature = "mlx")]
mod mlx;

mod clone;
mod dsp;
mod script;

// ort::Error doesn't implement std::error::Error; bridge via Display.
fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::Deserialize;

const NATIVE_SR: u32 = 24_000;
const STYLE_DIM: usize = 256;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Backend {
    /// ONNX Runtime with the CoreML execution provider.
    Onnx,
    /// Interpret kokoro.onnx directly on MLX (Metal GPU if available, else CPU).
    /// Requires a build with `--features mlx`; the default in that case.
    Mlx,
}
// Most voices are 511 rows; some are 510. Loader accepts either.
#[allow(dead_code)]
const MAX_VOICE_ROWS: usize = 511;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BitDepth {
    #[value(name = "16")]
    I16,
    #[value(name = "24")]
    I24,
    #[value(name = "32")]
    I32,
    #[value(name = "float32")]
    F32,
}

#[derive(Parser, Debug)]
#[command(
    name = "storytime",
    about = "Kokoro-82M text-to-speech (ONNX Runtime + CoreML).",
    long_about = "Reads UTF-8 from stdin. Default mode expects raw text and \
                  pipes through `espeak-ng` for IPA phonemization. With --ipa, \
                  stdin is treated as IPA phonemes directly (as espeak-ng -q --ipa=3 \
                  would emit), and espeak-ng is not required."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,

    #[command(flatten)]
    synth: Args,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Create a new voicepack from a short reference recording of a speaker
    /// (see README "Voice cloning").
    Clone(clone::CloneArgs),
}

#[derive(clap::Args, Debug)]
struct Args {
    /// Input text (or IPA with --ipa) file path. If omitted or `-`, read
    /// from stdin.
    #[arg(short = 'i', long)]
    input: Option<PathBuf>,

    /// Voice name (see --list-voices), e.g. af_bella.
    #[arg(long, default_value = "af_heart")]
    voice: String,

    /// Output WAV sample rate in Hz. Model is 24000 Hz; other rates are resampled.
    #[arg(long, default_value_t = 24_000)]
    sample_rate: u32,

    /// Output PCM bit depth.
    #[arg(long, value_enum, default_value_t = BitDepth::I16)]
    bit_depth: BitDepth,

    /// Speaking rate multiplier.
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Pitch shift in semitones (+ raises, − lowers), e.g. `--pitch 4` or
    /// `--pitch -3`. Tempo is preserved: the model re-times the speech and the
    /// audio is resampled to restore the original duration. In --script mode
    /// this is the default for any character without a `pitch=` cast annotation.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pitch: f32,

    /// Treat stdin as IPA phonemes rather than text (no espeak-ng invocation).
    #[arg(long)]
    ipa: bool,

    /// Override the espeak-ng voice (`-v`) used for grapheme→IPA conversion,
    /// e.g. `en-gb`. By default it is chosen automatically from the storytime
    /// voice's language prefix (American voices → `en-us`, British → `en-gb`,
    /// etc.). In --script mode this overrides the per-character choice for all.
    #[arg(long)]
    espeak_voice: Option<String>,

    /// Disable markdown preprocessing. By default markdown formatting is
    /// interpreted: bold (`**`/`__`) becomes emphasized speech, italic
    /// (`*`/`_`) becomes stressed speech, and other markers (headings aside)
    /// are stripped so they are not vocalized. With this flag the input is
    /// taken as literal text. Implied by --ipa (input is already phonemes).
    #[arg(long)]
    no_markdown: bool,

    /// Directory holding kokoro.onnx, tokens.json, voices/*.bin.
    /// Defaults to ../assets relative to the binary.
    #[arg(long)]
    assets: Option<PathBuf>,

    /// List available voices and exit.
    #[arg(long)]
    list_voices: bool,

    /// Silence inserted between chunker-forced splits within a paragraph, in ms.
    /// (Chunker splits have no textual equivalent, so silence is used.)
    #[arg(long, default_value_t = 120)]
    chunk_gap_ms: u32,

    /// Silence inserted on entry to and exit from a quoted span (dialogue),
    /// in ms. Default 0: the quote characters themselves drive prosody via
    /// Kokoro's trained punctuation tokens, and narration/dialogue stays in
    /// one inference call for better throughput. Set > 0 to re-enable
    /// explicit quote-aware pauses.
    #[arg(long, default_value_t = 0)]
    quote_gap_ms: u32,

    /// Silence inserted between paragraphs (blank-line separated), in ms.
    /// Default 0: a textual pause marker (`--paragraph-marker`) is inserted
    /// instead and Kokoro generates the pause from its own learned prosody.
    #[arg(long, default_value_t = 0)]
    paragraph_gap_ms: u32,

    /// Silence inserted between sections (≥2 blank lines or `## ` heading), in ms.
    /// Default 0: a marker (`--section-marker`) is inserted instead.
    #[arg(long, default_value_t = 0)]
    section_gap_ms: u32,

    /// Silence inserted between chapters (`# ` heading), in ms.
    #[arg(long, default_value_t = 1200)]
    chapter_gap_ms: u32,

    /// Textual pause marker inserted between paragraphs when `--paragraph-gap-ms` is 0.
    /// Default `. … ` uses a period (sentence-ending prosody) plus an ellipsis
    /// (sustained-pause token) so the model generates a clean pause rather
    /// than vocalizing the marker.
    #[arg(long, default_value_t = String::from(". \u{2026} "))]
    paragraph_marker: String,

    /// Textual pause marker inserted between sections when `--section-gap-ms` is 0.
    #[arg(long, default_value_t = String::from(". \u{2026} \u{2026} "))]
    section_marker: String,


    /// Linear fade-in/out applied at every chunk seam, in ms (avoids clicks).
    #[arg(long, default_value_t = 10)]
    fade_ms: u32,

    /// Amplitude below which leading/trailing samples are trimmed from each
    /// chunk before the gap silence is inserted. Set to 0 to disable.
    #[arg(long, default_value_t = 0.005)]
    trim_threshold: f32,

    /// Directory to cache the CoreML-compiled model between runs.
    /// Defaults to `$HOME/Library/Caches/storytime/coreml` on macOS.
    #[arg(long)]
    coreml_cache: Option<PathBuf>,

    /// Disable the CoreML compiled-model cache (forces recompilation each run).
    #[arg(long)]
    no_coreml_cache: bool,

    /// Inference backend. `onnx` uses ONNX Runtime + CoreML EP; `mlx`
    /// interprets the same kokoro.onnx on MLX (Metal GPU if available, else
    /// CPU). Defaults to `mlx` when built with `--features mlx`, else `onnx`.
    /// Both use the same model and voice assets.
    #[arg(long, value_enum)]
    backend: Option<Backend>,

    /// Output WAV path. If omitted, audio plays directly to the default
    /// output device via the OS-native audio API. Use `-` to stream the
    /// WAV to stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Screenplay mode: parse the input as a `NAME: dialogue` script with a
    /// `# Cast` header that assigns each character a voice, synthesize each
    /// speech in its character's voice, and mix the result. Incompatible with
    /// --ipa. See README "Script / multi-voice".
    #[arg(long)]
    script: bool,

    /// Read the cast (dramatis personae) from a separate file instead of (or in
    /// addition to) an inline `# Cast` block. Only meaningful with --script.
    #[arg(long)]
    cast: Option<PathBuf>,

    /// Voice for narration lines that have no speaker in script mode. Defaults
    /// to --voice. A `NARRATOR` cast entry overrides this.
    #[arg(long)]
    narrator: Option<String>,

    /// Overlap, in ms, applied when one speech interrupts another (a speech
    /// ending in `--`/`—`). The interrupter starts this many ms before the
    /// interrupted speech ends. Script mode only.
    #[arg(long, default_value_t = 250)]
    overlap_ms: u32,

    /// Linear gain applied to an interrupted speech's tail as it is overrun by
    /// the interrupter (1.0 = no ducking, 0.0 = silenced). Script mode only.
    #[arg(long, default_value_t = 0.4)]
    duck_gain: f32,

    /// Silence inserted between consecutive (non-overlapping) speeches in script
    /// mode, in ms.
    #[arg(long, default_value_t = 120)]
    line_gap_ms: u32,
}

/// Boundary between two adjacent audio pieces, ordered by strength.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Boundary {
    None = 0,
    Chunk = 1,
    Quote = 2,
    Paragraph = 3,
    Section = 4,
    Chapter = 5,
}

/// One structural block of input — a paragraph of running text, or a heading.
/// `gap_before` is the silence gap to insert before rendering this block.
struct Block {
    text: String,
    gap_before: Boundary,
}

#[derive(Deserialize)]
struct TokensFile {
    vocab: HashMap<String, i64>,
    #[allow(dead_code)]
    n_token: usize,
}

/// Read input text from a file path, or stdin if the path is `None` or `-`.
fn read_input(path: Option<&Path>) -> Result<String> {
    match path {
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            Ok(buf)
        }
        Some(p) if p.as_os_str() == "-" => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            Ok(buf)
        }
        Some(p) => fs::read_to_string(p)
            .with_context(|| format!("reading input file {}", p.display())),
    }
}

/// Resolve the CoreML compiled-model cache directory.
/// Returns `Ok(None)` when caching is explicitly disabled.
fn resolve_coreml_cache(no_cache: bool, explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    if no_cache {
        return Ok(None);
    }
    if let Some(p) = explicit {
        return Ok(Some(p.to_path_buf()));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let default = if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library/Caches/storytime/coreml"))
    } else {
        // CoreML only runs on Apple platforms; elsewhere the flag is a no-op.
        None
    };
    Ok(default)
}

fn assets_dir(cli_override: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = cli_override {
        return Ok(p.to_path_buf());
    }
    // Look relative to the executable, then relative to CWD.
    if let Ok(exe) = std::env::current_exe() {
        for up in [1, 2, 3] {
            if let Some(mut d) = exe.ancestors().nth(up).map(Path::to_path_buf) {
                d.push("assets");
                if d.join("kokoro.onnx").exists() {
                    return Ok(d);
                }
            }
        }
    }
    let cwd = std::env::current_dir()?.join("assets");
    if cwd.join("kokoro.onnx").exists() {
        return Ok(cwd);
    }
    bail!("could not locate assets/ directory (pass --assets)")
}

/// Enumerate available voice names (the stems of `voices/*.bin`), sorted.
fn voice_names(assets: &Path) -> Result<Vec<String>> {
    let dir = assets.join("voices");
    let mut names: Vec<_> = fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("bin") {
                p.file_stem().and_then(|s| s.to_str()).map(String::from)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    Ok(names)
}

fn list_voices(assets: &Path) -> Result<()> {
    for n in voice_names(assets)? {
        println!("{n}");
    }
    // In-progress clones (voices/<name>.bin.temp) are usable for preview under
    // their bare name; flag them so they're discoverable.
    let dir = assets.join("voices");
    if let Ok(entries) = fs::read_dir(&dir) {
        let mut training: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .and_then(|f| f.strip_suffix(".bin.temp"))
                    .map(String::from)
            })
            .collect();
        training.sort();
        for n in training {
            println!("{n} (training)");
        }
    }
    Ok(())
}

fn load_tokens(assets: &Path) -> Result<HashMap<String, i64>> {
    let raw = fs::read_to_string(assets.join("tokens.json"))?;
    let t: TokensFile = serde_json::from_str(&raw)?;
    Ok(t.vocab)
}

/// Tokenize IPA input by character lookup against the vocab.
/// Unknown characters are silently dropped (matches upstream behavior).
fn tokenize(ipa: &str, vocab: &HashMap<String, i64>) -> Vec<i64> {
    let mut out = Vec::with_capacity(ipa.len());
    for ch in ipa.chars() {
        let mut buf = [0u8; 4];
        let key = ch.encode_utf8(&mut buf);
        if let Some(&id) = vocab.get(key) {
            out.push(id);
        }
    }
    out
}

fn count_tokens(s: &str, vocab: &HashMap<String, i64>) -> usize {
    s.chars()
        .filter(|c| {
            let mut buf = [0u8; 4];
            vocab.contains_key(c.encode_utf8(&mut buf))
        })
        .count()
}

/// Split IPA into chunks that each tokenize to at most `max_tokens`.
///
/// Strategy, in order of preference:
///   1. Cut at sentence boundaries (`. ! ? ; …`) — punctuation stays with the
///      preceding chunk so prosody is preserved.
///   2. If a single sentence exceeds the budget, fall back to whitespace
///      (word) boundaries within that sentence.
///   3. If a single word still exceeds the budget, hard-split on characters.
///
/// Adjacent short sentences are greedily packed into one chunk up to the
/// budget to minimize the number of inference calls.
fn chunk_ipa(ipa: &str, vocab: &HashMap<String, i64>, max_tokens: usize) -> Vec<String> {
    assert!(max_tokens > 0);

    // Fast path: fits in one chunk.
    if count_tokens(ipa, vocab) <= max_tokens {
        return vec![ipa.to_string()];
    }

    // 1. Split into sentences, keeping terminators with the preceding text.
    let sentences = split_keep(ipa, |c| matches!(c, '.' | '!' | '?' | ';' | '…'));

    // 2. For any sentence that's still too long, expand into word/char pieces.
    let mut pieces: Vec<String> = Vec::new();
    for s in sentences {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            continue;
        }
        if count_tokens(trimmed, vocab) <= max_tokens {
            pieces.push(trimmed.to_string());
        } else {
            pieces.extend(split_long_sentence(trimmed, vocab, max_tokens));
        }
    }

    // 3. Greedily pack adjacent pieces up to the budget.
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_tokens = 0usize;
    for p in pieces {
        let p_tokens = count_tokens(&p, vocab);
        let join_tokens = if cur.is_empty() { 0 } else { 1 }; // joining space
        if cur_tokens + join_tokens + p_tokens <= max_tokens && !cur.is_empty() {
            cur.push(' ');
            cur.push_str(&p);
            cur_tokens += join_tokens + p_tokens;
        } else {
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            cur = p;
            cur_tokens = p_tokens;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Split `s` into substrings, cutting after any char matching `is_sep`.
/// The separator character is retained at the end of the preceding piece.
fn split_keep(s: &str, is_sep: impl Fn(char) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        cur.push(ch);
        if is_sep(ch) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Break a sentence that exceeds the budget into word-sized (or smaller) pieces.
fn split_long_sentence(
    sentence: &str,
    vocab: &HashMap<String, i64>,
    max_tokens: usize,
) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut cur = String::new();
    let mut cur_tokens = 0usize;
    for word in sentence.split_whitespace() {
        let w_tokens = count_tokens(word, vocab);
        if w_tokens > max_tokens {
            // Flush the current buffer, then hard-split the oversized word.
            if !cur.is_empty() {
                pieces.push(std::mem::take(&mut cur));
                cur_tokens = 0;
            }
            pieces.extend(hard_split(word, vocab, max_tokens));
            continue;
        }
        let join_tokens = if cur.is_empty() { 0 } else { 1 };
        if cur_tokens + join_tokens + w_tokens > max_tokens {
            pieces.push(std::mem::take(&mut cur));
            cur_tokens = 0;
        }
        if !cur.is_empty() {
            cur.push(' ');
            cur_tokens += 1;
        }
        cur.push_str(word);
        cur_tokens += w_tokens;
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    pieces
}

/// Last resort: split a whitespace-free run of characters by codepoint count.
fn hard_split(s: &str, vocab: &HashMap<String, i64>, max_tokens: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_tokens = 0usize;
    for ch in s.chars() {
        let mut buf = [0u8; 4];
        let counts = vocab.contains_key(ch.encode_utf8(&mut buf));
        if counts && cur_tokens + 1 > max_tokens {
            out.push(std::mem::take(&mut cur));
            cur_tokens = 0;
        }
        cur.push(ch);
        if counts {
            cur_tokens += 1;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Punctuation characters preserved across the espeak-ng call so they
/// reach Kokoro's token stream. espeak-ng with `--ipa=3` silently strips
/// all of these, but they are first-class vocab entries in Kokoro's
/// config and the model is trained to produce prosody on them. Missing
/// punctuation is the single largest contributor to "flat" / "monotone"
/// synthesis on longer inputs.
///
/// Chars here match Misaki's `PUNCTS` set (the upstream Python G2P used
/// by Kokoro's authors) plus parentheses, which are also in the vocab.
const PRESERVED_PUNCT: &str = ";:,.!?—…\"()\u{201C}\u{201D}";

/// Normalize punctuation to the forms Kokoro's vocab prefers:
///   - Runs of 3+ ASCII dots (`...`, `....`) collapse to a single `…`
///     (U+2026), which Kokoro has as a dedicated token (ID 10).
///   - Straight `"` characters are paired off into alternating curly
///     `\u{201C}` (open) / `\u{201D}` (close), which are Kokoro's
///     dedicated open/close tokens (IDs 14 and 15). Leaving them as
///     straight `"` maps everything to a single undifferentiated token
///     (ID 11) and loses the open/close distinction.
///
/// Applied before quote-aware splitting and before phonemization, so
/// downstream code sees the normalized form.
fn normalize_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut quote_open = true;
    let mut dots = 0usize;

    let flush_dots = |dots: &mut usize, out: &mut String| {
        if *dots >= 3 {
            out.push('\u{2026}');
        } else {
            for _ in 0..*dots {
                out.push('.');
            }
        }
        *dots = 0;
    };

    for ch in text.chars() {
        if ch == '.' {
            dots += 1;
            continue;
        }
        flush_dots(&mut dots, &mut out);

        if ch == '"' {
            out.push(if quote_open { '\u{201C}' } else { '\u{201D}' });
            quote_open = !quote_open;
        } else {
            out.push(ch);
        }
    }
    flush_dots(&mut dots, &mut out);
    out
}

// ----------------------------------------------------------------------------
// Markdown preprocessing
//
// Markdown formatting is interpreted before structural parsing: emphasis is
// translated into IPA stress (so bold/italic produce emphasized/stressed
// speech), and every other marker is stripped so it is never vocalized.
//
// Emphasis can't be applied here directly — stress lives in phoneme space,
// downstream of espeak-ng — so emphasized spans are wrapped in private-use
// sentinel characters. The sentinels ride untouched through `parse_structure`
// and `normalize_punctuation` (they are neither whitespace, heading prefixes,
// `.`, nor `"`), and `run_espeak` finally consumes them, applying the matching
// IPA stress to the enclosed phonemes. They are never sent to espeak-ng or the
// model and never reach the token stream.
// ----------------------------------------------------------------------------

/// Strength of emphasis on a span, derived from markdown markers:
/// `*italic*`/`_italic_` -> Mild (extra stress), `**bold**`/`__bold__` -> Strong.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Emph {
    None,
    Mild,
    Strong,
}

const SENT_BOLD_OPEN: char = '\u{E000}';
const SENT_BOLD_CLOSE: char = '\u{E001}';
const SENT_ITAL_OPEN: char = '\u{E002}';
const SENT_ITAL_CLOSE: char = '\u{E003}';

/// Strip markdown formatting from `input`, translating emphasis into emphasis
/// sentinels and removing every other marker so it is not spoken. Heading
/// markers (`# `/`## `/`### `) and blank-line structure are deliberately left
/// intact for `parse_structure`. Operates line-by-line so block constructs
/// (fenced code, rules, blockquotes, list bullets) are handled before inline
/// ones (links/images, code spans, emphasis, strikethrough).
fn preprocess_markdown(input: &str) -> String {
    let input = input.replace("\r\n", "\n");
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in input.split('\n') {
        let trimmed = line.trim_start();
        // Fenced code delimiters (``` or ~~~): drop the fence line; keep the
        // body as plain text (markers stripped, content preserved).
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            out.push(String::new());
            continue;
        }
        if in_fence {
            out.push(line.to_string());
            continue;
        }
        // Horizontal rule (--- / *** / ___): becomes a blank line so it acts as
        // a structural break rather than three spoken characters.
        if is_hr(trimmed) {
            out.push(String::new());
            continue;
        }
        let l = strip_blockquote(line);
        let l = strip_list_marker(&l);
        out.push(inline_md(&l));
    }
    out.join("\n")
}

/// True if `s` is a markdown horizontal rule: three or more of the same
/// `- `, `*`, or `_` character, whitespace aside.
fn is_hr(s: &str) -> bool {
    let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.len() < 3 {
        return false;
    }
    let first = compact.chars().next().unwrap();
    matches!(first, '-' | '*' | '_') && compact.chars().all(|c| c == first)
}

/// Remove leading blockquote markers (`>`), including nested ones.
fn strip_blockquote(line: &str) -> String {
    let mut l = line;
    loop {
        let t = l.trim_start();
        match t.strip_prefix('>') {
            Some(rest) => l = rest,
            None => return t.to_string(),
        }
    }
}

/// Remove a leading list bullet (`- `, `+ `, `* `) or ordered marker
/// (`1.`/`1)` followed by a space). The trailing space is required so that
/// line-leading emphasis like `*italic*` is not mistaken for a bullet.
fn strip_list_marker(line: &str) -> String {
    let t = line.trim_start();
    for m in ['-', '+', '*'] {
        if let Some(rest) = t.strip_prefix(m) {
            if let Some(after) = rest.strip_prefix(' ') {
                return after.trim_start().to_string();
            }
        }
    }
    let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 {
        let rest = &t[digits..];
        if let Some(r) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) {
            if let Some(after) = r.strip_prefix(' ') {
                return after.trim_start().to_string();
            }
        }
    }
    t.to_string()
}

/// Apply inline markdown transforms to a single line. Code spans are detected
/// first by splitting on backticks so their contents are emitted verbatim
/// (markers dropped) and never reinterpreted as emphasis.
fn inline_md(line: &str) -> String {
    let mut out = String::new();
    for (i, seg) in line.split('`').enumerate() {
        if i % 2 == 1 {
            // Inside a code span: emit content verbatim, sans backticks.
            out.push_str(seg);
        } else {
            out.push_str(&transform_inline(seg));
        }
    }
    out
}

/// Collapse links/images to their text, drop strikethrough markers, then turn
/// emphasis into sentinels. Order matters: links are resolved first so URLs
/// (which may contain `_`) are gone before emphasis scanning.
fn transform_inline(seg: &str) -> String {
    let s = strip_links(seg);
    let s = s.replace("~~", "");
    parse_emphasis(&s)
}

/// Replace `[text](url)`, `[text][ref]`, and `![alt](url)` with their visible
/// text. Brackets that don't form a link are left as literal characters.
fn strip_links(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        // Image: drop the leading '!', leaving the [alt](url) for the next step.
        if c == '!' && i + 1 < n && chars[i + 1] == '[' {
            i += 1;
            continue;
        }
        if c == '[' {
            if let Some(close) = matching(&chars, i, '[', ']') {
                let after = close + 1;
                // Inline `[text](url)` or reference `[text][id]`: keep the text,
                // drop the brackets and the destination/label that follows.
                let link_end = match chars.get(after) {
                    Some('(') => find_from(&chars, after, ')'),
                    Some('[') => find_from(&chars, after, ']'),
                    _ => None,
                };
                if let Some(end) = link_end {
                    out.extend(chars[i + 1..close].iter());
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Index of the close char that balances the open char at `open`, honoring
/// nesting. Returns None if unbalanced.
fn matching(chars: &[char], open: usize, oc: char, cc: char) -> Option<usize> {
    let mut depth = 0i32;
    for (i, &c) in chars.iter().enumerate().skip(open) {
        if c == oc {
            depth += 1;
        } else if c == cc {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

fn find_from(chars: &[char], start: usize, target: char) -> Option<usize> {
    (start..chars.len()).find(|&i| chars[i] == target)
}

/// Map the (minimum) length of a matched `*`/`_` delimiter run to a strength:
/// one marker is italic (Mild), two or more is bold (Strong).
fn emph_level(run_len: usize) -> Emph {
    match run_len {
        0 => Emph::None,
        1 => Emph::Mild,
        _ => Emph::Strong,
    }
}

/// Rewrite markdown emphasis (`*`/`_` runs) into emphasis sentinels. A
/// CommonMark-lite delimiter matcher: runs are matched closer-to-opener with a
/// stack, the strength taken from the shorter run; underscores only emphasize
/// at word boundaries (so `snake_case` is left alone), and unmatched runs are
/// emitted as literal characters.
fn parse_emphasis(s: &str) -> String {
    #[derive(Clone, Copy)]
    enum Run {
        Text(usize, usize),
        Delim {
            ch: char,
            len: usize,
            can_open: bool,
            can_close: bool,
        },
    }

    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();

    // 1. Tokenize into text spans and maximal delimiter runs.
    let mut runs: Vec<Run> = Vec::new();
    let mut i = 0;
    let mut text_start = 0;
    while i < n {
        let c = chars[i];
        if c == '*' || c == '_' {
            if text_start < i {
                runs.push(Run::Text(text_start, i));
            }
            let mut j = i;
            while j < n && chars[j] == c {
                j += 1;
            }
            let prev = i.checked_sub(1).map(|k| chars[k]);
            let next = chars.get(j).copied();
            let left_flank = next.is_some_and(|x| !x.is_whitespace());
            let right_flank = prev.is_some_and(|x| !x.is_whitespace());
            let (can_open, can_close) = if c == '_' {
                (
                    left_flank && prev.is_none_or(|x| !x.is_alphanumeric()),
                    right_flank && next.is_none_or(|x| !x.is_alphanumeric()),
                )
            } else {
                (left_flank, right_flank)
            };
            runs.push(Run::Delim {
                ch: c,
                len: j - i,
                can_open,
                can_close,
            });
            i = j;
            text_start = j;
        } else {
            i += 1;
        }
    }
    if text_start < n {
        runs.push(Run::Text(text_start, n));
    }

    // 2. Match closers to openers; record each matched run's role + strength.
    #[derive(Clone, Copy)]
    enum Role {
        None,
        Open(Emph),
        Close(Emph),
    }
    let mut roles = vec![Role::None; runs.len()];
    let mut stack: Vec<usize> = Vec::new();
    for idx in 0..runs.len() {
        if let Run::Delim {
            ch, len, can_open, can_close, ..
        } = runs[idx]
        {
            if can_close {
                if let Some(si) = stack
                    .iter()
                    .rposition(|&oi| matches!(runs[oi], Run::Delim { ch: o, .. } if o == ch))
                {
                    let oidx = stack[si];
                    let olen = match runs[oidx] {
                        Run::Delim { len, .. } => len,
                        _ => 1,
                    };
                    let level = emph_level(olen.min(len));
                    roles[oidx] = Role::Open(level);
                    roles[idx] = Role::Close(level);
                    stack.truncate(si); // discard openers left unmatched inside
                    continue;
                }
            }
            if can_open {
                stack.push(idx);
            }
        }
    }

    // 3. Rebuild: matched delimiters become sentinels, the rest stay literal.
    let mut out = String::new();
    for (idx, run) in runs.iter().enumerate() {
        match *run {
            Run::Text(a, b) => out.extend(chars[a..b].iter()),
            Run::Delim { ch, len, .. } => match roles[idx] {
                Role::Open(Emph::Strong) => out.push(SENT_BOLD_OPEN),
                Role::Open(Emph::Mild) => out.push(SENT_ITAL_OPEN),
                Role::Close(Emph::Strong) => out.push(SENT_BOLD_CLOSE),
                Role::Close(Emph::Mild) => out.push(SENT_ITAL_CLOSE),
                _ => (0..len).for_each(|_| out.push(ch)),
            },
        }
    }
    out
}

/// IPA vowel symbols espeak-ng emits for en-us, used to find the nucleus of the
/// primary-stressed syllable so emphasis can lengthen it.
const IPA_VOWELS: &str = "iɪeɛæəɐʌɜɚɝɑɒɔoɵʊuʉayøœ";

fn is_ipa_vowel(c: char) -> bool {
    IPA_VOWELS.contains(c)
}

/// Apply emphasis to one phonemized word: ensure a primary stress mark is
/// present, then lengthen the primary-stressed vowel by `extra` length marks
/// (`ː`). espeak already marks lexical stress, so the lengthening is what makes
/// the word audibly stand out; an unstressed function word with no mark gets a
/// primary mark prepended at its onset.
fn emphasize_word(ipa: &str, extra: usize) -> String {
    let mut s = ipa.to_string();
    if !s.contains('ˈ') {
        if let Some(pos) = s.find('ˌ') {
            s.replace_range(pos..pos + 'ˌ'.len_utf8(), "ˈ");
        } else {
            s.insert(0, 'ˈ');
        }
    }
    if extra == 0 {
        return s;
    }
    let Some(stress) = s.find('ˈ') else {
        return s;
    };
    let after = stress + 'ˈ'.len_utf8();
    // End of the first contiguous vowel run after the stress mark.
    let mut vowel_end: Option<usize> = None;
    for (off, c) in s[after..].char_indices() {
        if is_ipa_vowel(c) {
            vowel_end = Some(after + off + c.len_utf8());
        } else if vowel_end.is_some() {
            break;
        }
    }
    if let Some(end) = vowel_end {
        let marks: String = std::iter::repeat('ː').take(extra).collect();
        s.insert_str(end, &marks);
    }
    s
}

/// Apply emphasis across a phonemized span (one or more space-separated words),
/// lengthening each word's stressed vowel: Mild adds one `ː`, Strong two.
fn apply_emphasis_to_ipa(ipa: &str, level: Emph) -> String {
    let extra = match level {
        Emph::None => return ipa.to_string(),
        Emph::Mild => 1,
        Emph::Strong => 2,
    };
    ipa.split(' ')
        .map(|w| {
            if w.is_empty() {
                String::new()
            } else {
                emphasize_word(w, extra)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The espeak-ng voice (`-v`) for a storytime voice, from its `{lang}{gender}_`
/// prefix: `af_bella` → American English, `bm_george` → British English, etc.
/// This matters because espeak-ng phonemizes, say, British vowels differently
/// from American (non-rhotic, different `a`/`o` qualities); feeding a `b*` voice
/// `en-us` IPA mangles its accent. Names that don't follow the convention
/// (e.g. cloned voices) fall through to the default English.
fn espeak_voice_for(voice_name: &str) -> &'static str {
    let b = voice_name.as_bytes();
    if b.len() >= 3 && b[2] == b'_' {
        match b[0] {
            b'a' => return "en-us",  // American English
            b'b' => return "en-gb",  // British English
            b'e' => return "es",     // Spanish
            b'f' => return "fr-fr",  // French
            b'h' => return "hi",     // Hindi
            b'i' => return "it",     // Italian
            b'j' => return "ja",     // Japanese
            b'p' => return "pt-br",  // Brazilian Portuguese
            b'z' => return "cmn",    // Mandarin Chinese
            _ => {}
        }
    }
    "en-us"
}

/// Run espeak-ng for grapheme->IPA conversion, preserving punctuation.
///
/// `espeak_voice` is the espeak-ng voice id (`-v`, e.g. `en-us` / `en-gb`),
/// chosen per storytime voice so each accent is phonemized correctly.
///
/// The strategy is: split the input on every preserved punctuation
/// character, send only the text segments to espeak-ng (one per line),
/// read one IPA line back per segment, then interleave the punctuation
/// back into the output. One espeak invocation per call, same as a
/// naive pass-through — but the resulting IPA contains `?`, `:`, `;`,
/// etc., which espeak would otherwise drop.
fn run_espeak(text: &str, espeak_voice: &str) -> Result<String> {
    #[derive(Debug)]
    enum Piece<'a> {
        Text(&'a str, Emph),
        Punct(char),
    }

    // 1. Split input on preserved-punctuation and emphasis-sentinel boundaries.
    //    Sentinels are consumed here — they never reach espeak-ng or the model;
    //    instead they toggle the emphasis level carried by the text that
    //    follows, which becomes IPA stress in step 3.
    let mut pieces: Vec<Piece> = Vec::new();
    let mut cur = 0usize;
    let mut bold = 0i32;
    let mut ital = 0i32;
    let level = |b: i32, i: i32| -> Emph {
        if b > 0 {
            Emph::Strong
        } else if i > 0 {
            Emph::Mild
        } else {
            Emph::None
        }
    };
    for (i, ch) in text.char_indices() {
        let sentinel =
            matches!(ch, SENT_BOLD_OPEN | SENT_BOLD_CLOSE | SENT_ITAL_OPEN | SENT_ITAL_CLOSE);
        if sentinel || PRESERVED_PUNCT.contains(ch) {
            if cur < i {
                pieces.push(Piece::Text(&text[cur..i], level(bold, ital)));
            }
            if sentinel {
                match ch {
                    SENT_BOLD_OPEN => bold += 1,
                    SENT_BOLD_CLOSE => bold = (bold - 1).max(0),
                    SENT_ITAL_OPEN => ital += 1,
                    SENT_ITAL_CLOSE => ital = (ital - 1).max(0),
                    _ => {}
                }
            } else {
                pieces.push(Piece::Punct(ch));
            }
            cur = i + ch.len_utf8();
        }
    }
    if cur < text.len() {
        pieces.push(Piece::Text(&text[cur..], level(bold, ital)));
    }

    // 2. Collect non-empty text segments, one per line, for espeak-ng.
    //    espeak-ng emits one IPA line per input line in --ipa=3 mode, which
    //    gives us a clean 1:1 mapping back to the segment list.
    let text_segments: Vec<&str> = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Text(s, _) if !s.trim().is_empty() => Some(*s),
            _ => None,
        })
        .collect();

    let ipa_lines: Vec<String> = if text_segments.is_empty() {
        Vec::new()
    } else {
        let joined = text_segments
            .iter()
            .map(|s| s.trim())
            .collect::<Vec<_>>()
            .join("\n");
        let mut child = Command::new("espeak-ng")
            .args(["-q", "--ipa=3", "-v", espeak_voice])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn espeak-ng (install it or use --ipa)")?;
        child.stdin.as_mut().unwrap().write_all(joined.as_bytes())?;
        let out = child.wait_with_output()?;
        if !out.status.success() {
            bail!("espeak-ng failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        String::from_utf8(out.stdout)?
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    // 3. Interleave punctuation back into the IPA stream. Add a space
    //    after each punct when more text follows, so natural word spacing
    //    survives the rebuild (espeak output has no leading space).
    let mut result = String::new();
    let mut ipa_iter = ipa_lines.into_iter();
    let mut prev_trailing_ws = false;
    for (i, piece) in pieces.iter().enumerate() {
        match piece {
            Piece::Text(s, emph) => {
                if s.trim().is_empty() {
                    continue;
                }
                // A sentinel can split mid-phrase where the only separator was a
                // space (e.g. `say **boo**`); the trim above would drop it, so
                // re-insert a single gap between adjacent text pieces when the
                // original had whitespace at the boundary.
                let leading_ws = s.starts_with(char::is_whitespace);
                if !result.is_empty()
                    && !result.ends_with(' ')
                    && (prev_trailing_ws || leading_ws)
                {
                    result.push(' ');
                }
                if let Some(ipa) = ipa_iter.next() {
                    result.push_str(&apply_emphasis_to_ipa(&ipa, *emph));
                }
                prev_trailing_ws = s.ends_with(char::is_whitespace);
            }
            Piece::Punct(c) => {
                result.push(*c);
                let more_text_follows = pieces[i + 1..]
                    .iter()
                    .any(|p| matches!(p, Piece::Text(s, _) if !s.trim().is_empty()));
                if more_text_follows {
                    result.push(' ');
                }
                prev_trailing_ws = false;
            }
        }
    }
    Ok(result.trim().to_string())
}

struct Voice {
    data: Vec<f32>,
    rows: usize,
}

/// Resolve a `--voice` argument to a voicepack file. A bare voice name maps to
/// `voices/<name>.bin`, falling back to the in-progress `voices/<name>.bin.temp`
/// of a clone that is still training (so a partial voice can be previewed under
/// the same name it will eventually have). A value that looks like a path
/// (contains a separator or is absolute) is used verbatim, which also lets a
/// `.bin.temp` be loaded explicitly.
fn resolve_voice_path(assets: &Path, name: &str) -> PathBuf {
    if name.contains('/') || name.contains('\\') || Path::new(name).is_absolute() {
        return PathBuf::from(name);
    }
    let dir = assets.join("voices");
    let final_path = dir.join(format!("{name}.bin"));
    if final_path.exists() {
        return final_path;
    }
    let temp_path = dir.join(format!("{name}.bin.temp"));
    if temp_path.exists() {
        eprintln!(
            "storytime: voice '{name}' is still training; using in-progress {name}.bin.temp"
        );
        return temp_path;
    }
    final_path // fall through to the familiar "voice not found" error
}

fn load_voice(assets: &Path, name: &str) -> Result<Voice> {
    let path = resolve_voice_path(assets, name);
    let bytes = fs::read(&path).with_context(|| format!("voice not found: {}", path.display()))?;
    if bytes.len() % (STYLE_DIM * 4) != 0 {
        bail!("voice {name}: unexpected size {} bytes", bytes.len());
    }
    let rows = bytes.len() / (STYLE_DIM * 4);
    let mut data = Vec::with_capacity(rows * STYLE_DIM);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Voice { data, rows })
}

/// Select the style row for a given token length. Reference runtime indexes
/// by the number of phoneme tokens (unpadded), clamped to the voice length.
fn select_style(voice: &Voice, n_tokens: usize) -> Vec<f32> {
    let idx = n_tokens.min(voice.rows - 1);
    let start = idx * STYLE_DIM;
    voice.data[start..start + STYLE_DIM].to_vec()
}

fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(samples.to_vec());
    }
    resample_ratio(samples, to as f64 / from as f64)
}

/// Resample so the output has `ratio ×` the samples (output_len ≈ input_len ×
/// ratio). Used for sample-rate conversion (ratio = to/from) and for pitch
/// shifting (ratio = 1/r, reinterpreted at the same rate to scale frequencies).
fn resample_ratio(samples: &[f32], ratio: f64) -> Result<Vec<f32>> {
    if (ratio - 1.0).abs() < 1e-9 || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut r = SincFixedIn::<f32>::new(ratio, 2.0, params, samples.len(), 1)?;
    let out = r.process(&[samples.to_vec()], None)?;
    Ok(out.into_iter().next().unwrap())
}

/// Frequency multiplier for a pitch shift of `semitones` (12 semitones → 2×).
fn pitch_ratio(semitones: f32) -> f32 {
    2f32.powf(semitones / 12.0)
}

/// Stream a WAV to any writer without seeking.
///
/// Because stdout isn't seekable, the RIFF/data size fields can't be
/// back-patched after the sample count is known. Per the user's request,
/// they're set to `0xFFFFFFFF` (the maximum `u32`); downstream decoders
/// that support streaming WAVs (ffmpeg, sox, etc.) either honor the
/// sentinel or read until EOF.
fn write_wav_stream<W: std::io::Write>(
    mut w: W,
    samples: &[f32],
    sr: u32,
    depth: BitDepth,
) -> Result<()> {
    let (bits, fmt_code): (u16, u16) = match depth {
        BitDepth::I16 => (16, 1),
        BitDepth::I24 => (24, 1),
        BitDepth::I32 => (32, 1),
        BitDepth::F32 => (32, 3), // WAVE_FORMAT_IEEE_FLOAT
    };
    let channels: u16 = 1;
    let block_align: u16 = channels * (bits / 8);
    let byte_rate: u32 = sr * block_align as u32;
    let max = u32::MAX;

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&max.to_le_bytes())?; // file size - 8 (streaming sentinel)
    w.write_all(b"WAVE")?;

    // fmt chunk (16-byte PCM / IEEE float form)
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&fmt_code.to_le_bytes())?;
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sr.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits.to_le_bytes())?;

    // data chunk header
    w.write_all(b"data")?;
    w.write_all(&max.to_le_bytes())?; // data size (streaming sentinel)

    // payload
    match depth {
        BitDepth::F32 => {
            for &s in samples {
                w.write_all(&s.clamp(-1.0, 1.0).to_le_bytes())?;
            }
        }
        BitDepth::I16 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                w.write_all(&v.to_le_bytes())?;
            }
        }
        BitDepth::I24 => {
            let peak = ((1i32 << 23) - 1) as f32;
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * peak) as i32;
                w.write_all(&v.to_le_bytes()[..3])?;
            }
        }
        BitDepth::I32 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) as f64 * i32::MAX as f64) as i32;
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

/// Parse stdin into structural blocks so paragraph / section / chapter
/// boundaries survive to the synthesis stage (espeak-ng otherwise flattens
/// all whitespace). Rules:
///   - One blank line between non-empty paragraphs -> Paragraph boundary.
///   - Two or more blank lines -> Section boundary.
///   - A block starting with `# `   -> Chapter boundary (marker stripped).
///   - A block starting with `## `  -> Section boundary (marker stripped).
///   - A block starting with `### ` -> Section boundary (marker stripped).
/// Internal newlines within a paragraph are collapsed to spaces so
/// line-wrapped prose reads naturally.
fn parse_structure(input: &str) -> Vec<Block> {
    let input = input.replace("\r\n", "\n");
    let mut blocks: Vec<Block> = Vec::new();

    let mut current: Vec<String> = Vec::new();
    let mut blanks_before_current: usize = 0;
    let mut pending_blanks: usize = 0;

    let flush = |current: &mut Vec<String>, blanks_before: usize, blocks: &mut Vec<Block>| {
        let joined = current.join(" ");
        let trimmed = joined.trim();
        if trimmed.is_empty() {
            current.clear();
            return;
        }
        let (text, heading) = if let Some(s) = trimmed.strip_prefix("# ") {
            (s.trim().to_string(), Some(Boundary::Chapter))
        } else if let Some(s) = trimmed.strip_prefix("## ") {
            (s.trim().to_string(), Some(Boundary::Section))
        } else if let Some(s) = trimmed.strip_prefix("### ") {
            (s.trim().to_string(), Some(Boundary::Section))
        } else {
            (trimmed.to_string(), None)
        };
        let structural = match blanks_before {
            0 => Boundary::None,
            1 => Boundary::Paragraph,
            _ => Boundary::Section,
        };
        let gap_before = if blocks.is_empty() {
            Boundary::None
        } else {
            match heading {
                Some(h) => structural.max(h),
                None => structural,
            }
        };
        blocks.push(Block { text, gap_before });
        current.clear();
    };

    for line in input.split('\n') {
        if line.trim().is_empty() {
            if current.is_empty() {
                blanks_before_current += 1;
            } else {
                pending_blanks += 1;
            }
        } else {
            if pending_blanks > 0 {
                flush(&mut current, blanks_before_current, &mut blocks);
                blanks_before_current = pending_blanks;
                pending_blanks = 0;
            }
            current.push(line.trim().to_string());
        }
    }
    flush(&mut current, blanks_before_current, &mut blocks);
    blocks
}

fn gap_samples_for(b: Boundary, args: &Args, sr: u32) -> usize {
    let ms = match b {
        Boundary::None => 0,
        Boundary::Chunk => args.chunk_gap_ms,
        Boundary::Quote => args.quote_gap_ms,
        Boundary::Paragraph => args.paragraph_gap_ms,
        Boundary::Section => args.section_gap_ms,
        Boundary::Chapter => args.chapter_gap_ms,
    };
    (sr as usize * ms as usize) / 1000
}

/// Split a block of text into pieces at every transition across a
/// double-quote boundary. Straight `"` toggles the inside/outside state;
/// curly `\u{201C}` / `\u{201D}` are treated as unambiguous open/close.
/// Single quotes are intentionally NOT handled because they're
/// indistinguishable from apostrophes in plain text.
///
/// The quote characters themselves are retained with the quoted piece,
/// so the model still sees them in its phoneme stream and produces the
/// usual punctuation-driven prosody — the inserted Quote gap is purely
/// additive and gives dialogue the extra "beat" a narrator would use.
fn split_quotes(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut inside = false;

    let flush = |cur: &mut String, out: &mut Vec<String>| {
        let t = cur.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
        cur.clear();
    };

    for ch in text.chars() {
        match ch {
            '"' => {
                if !inside {
                    flush(&mut cur, &mut out);
                    cur.push(ch);
                    inside = true;
                } else {
                    cur.push(ch);
                    flush(&mut cur, &mut out);
                    inside = false;
                }
            }
            '\u{201C}' => {
                flush(&mut cur, &mut out);
                cur.push(ch);
                inside = true;
            }
            '\u{201D}' => {
                cur.push(ch);
                flush(&mut cur, &mut out);
                inside = false;
            }
            _ => cur.push(ch),
        }
    }
    flush(&mut cur, &mut out);
    if out.is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

/// Trim leading/trailing silence using windowed RMS energy.
///
/// Instead of reacting to the first individual sample above `threshold`,
/// this requires a sliding window of WINDOW samples to have an RMS above
/// `threshold`. This makes the onset detector immune to isolated spikes
/// from the iSTFT decoder's settling noise, which are brief (a few
/// samples) and don't sustain across a full window.
fn trim_silence(samples: &[f32], threshold: f32) -> &[f32] {
    if threshold <= 0.0 || samples.is_empty() {
        return samples;
    }
    const WINDOW: usize = 128; // ~5.3ms at 24kHz
    let threshold_sq = threshold * threshold;

    let rms_above = |start: usize| -> bool {
        if start + WINDOW > samples.len() {
            return false;
        }
        let sum_sq: f32 = samples[start..start + WINDOW]
            .iter()
            .map(|s| s * s)
            .sum();
        sum_sq / WINDOW as f32 > threshold_sq
    };

    let start = (0..samples.len().saturating_sub(WINDOW))
        .find(|&i| rms_above(i))
        .unwrap_or(samples.len());

    let end = (0..samples.len().saturating_sub(WINDOW))
        .rfind(|&i| rms_above(i))
        .map(|i| (i + WINDOW).min(samples.len()))
        .unwrap_or(start);

    if start >= end {
        return &[];
    }
    &samples[start..end]
}

/// Apply an in-place linear fade-in over the first `fade` samples and a
/// fade-out over the last `fade` samples. Clamped to half the buffer.
fn apply_fade(samples: &mut [f32], fade: usize) {
    let n = samples.len();
    let f = fade.min(n / 2);
    if f == 0 {
        return;
    }
    for i in 0..f {
        let gain = (i as f32 + 1.0) / (f as f32 + 1.0);
        samples[i] *= gain;
        samples[n - 1 - i] *= gain;
    }
}

fn synthesize_chunk(
    session: &mut Session,
    tokens: &[i64],
    style: Vec<f32>,
    speed: f32,
) -> Result<Vec<f32>> {
    let mut padded = Vec::with_capacity(tokens.len() + 2);
    padded.push(0);
    padded.extend_from_slice(tokens);
    padded.push(0);

    let ids_t = Tensor::from_array((vec![1_i64, padded.len() as i64], padded)).map_err(ort_err)?;
    let style_t =
        Tensor::from_array((vec![1_i64, STYLE_DIM as i64], style)).map_err(ort_err)?;
    let speed_t = Tensor::from_array((vec![1_i64], vec![speed])).map_err(ort_err)?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_t,
            "style" => style_t,
            "speed" => speed_t,
        ])
        .map_err(ort_err)?;

    let (_shape, audio) = outputs["audio"]
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    Ok(audio.to_vec())
}

// The MLX backend lives in its own module (cli/src/mlx/): it interprets
// kokoro.onnx directly on MLX (Metal GPU or CPU) via mlx-c.

fn write_wav(path: &Path, samples: &[f32], sr: u32, depth: BitDepth) -> Result<()> {
    let (bits, fmt) = match depth {
        BitDepth::I16 => (16u16, SampleFormat::Int),
        BitDepth::I24 => (24, SampleFormat::Int),
        BitDepth::I32 => (32, SampleFormat::Int),
        BitDepth::F32 => (32, SampleFormat::Float),
    };
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: bits,
        sample_format: fmt,
    };
    let mut w = WavWriter::create(path, spec)?;
    match depth {
        BitDepth::F32 => {
            for &s in samples {
                w.write_sample(s.clamp(-1.0, 1.0))?;
            }
        }
        BitDepth::I16 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                w.write_sample(v)?;
            }
        }
        BitDepth::I24 => {
            let peak = ((1i32 << 23) - 1) as f32;
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * peak) as i32;
                w.write_sample(v)?;
            }
        }
        BitDepth::I32 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) as f64 * i32::MAX as f64) as i32;
                w.write_sample(v)?;
            }
        }
    }
    w.finalize()?;
    Ok(())
}

/// An initialized inference backend: an ONNX Runtime session (CoreML EP) or
/// an MLX runtime. Both take the same `(tokens, style, speed)` and return
/// 24 kHz mono f32 samples.
struct Runtime {
    backend: Backend,
    onnx_session: Option<Session>,
    #[cfg(feature = "mlx")]
    mlx_rt: Option<mlx::MlxRuntime>,
}

impl Runtime {
    fn init(backend: Backend, assets: &Path, coreml_cache: Option<PathBuf>) -> Result<Self> {
        #[cfg(feature = "mlx")]
        let mut mlx_rt: Option<mlx::MlxRuntime> = None;
        let onnx_session: Option<Session> = match backend {
            Backend::Onnx => {
                ort::init().with_name("storytime").commit().map_err(ort_err)?;
                let mut coreml = CoreMLExecutionProvider::default()
                    .with_model_format(CoreMLModelFormat::NeuralNetwork)
                    .with_compute_units(CoreMLComputeUnits::All)
                    .with_low_precision_accumulation_on_gpu(true);
                if let Some(dir) = coreml_cache.as_ref() {
                    fs::create_dir_all(dir)?;
                    coreml = coreml.with_model_cache_dir(dir.display().to_string());
                    eprintln!("storytime: backend=onnx, coreml cache: {}", dir.display());
                } else {
                    eprintln!("storytime: backend=onnx, coreml cache disabled");
                }
                Some(
                    Session::builder()
                        .map_err(ort_err)?
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(ort_err)?
                        .with_execution_providers([coreml.build()])
                        .map_err(ort_err)?
                        .commit_from_file(assets.join("kokoro.onnx"))
                        .map_err(ort_err)?,
                )
            }
            Backend::Mlx => {
                #[cfg(not(feature = "mlx"))]
                bail!("MLX backend not compiled in. Rebuild with: cargo build --features mlx");

                #[cfg(feature = "mlx")]
                {
                    let device = if mlx::gpu_available() {
                        mlx::Device::Gpu
                    } else {
                        mlx::Device::Cpu
                    };
                    eprintln!("storytime: backend=mlx, device={:?}", device);
                    mlx_rt = Some(mlx::MlxRuntime::new(&assets.join("kokoro.onnx"), device)?);
                    None
                }
            }
        };
        Ok(Runtime {
            backend,
            onnx_session,
            #[cfg(feature = "mlx")]
            mlx_rt,
        })
    }

    /// Run one chunk through the selected backend.
    fn synth(&mut self, tokens: &[i64], style: Vec<f32>, speed: f32) -> Result<Vec<f32>> {
        match self.backend {
            Backend::Onnx => {
                synthesize_chunk(self.onnx_session.as_mut().unwrap(), tokens, style, speed)
            }
            Backend::Mlx => {
                #[cfg(feature = "mlx")]
                {
                    self.mlx_rt.as_ref().unwrap().synthesize(tokens, &style, speed)
                }
                #[cfg(not(feature = "mlx"))]
                {
                    let _ = (tokens, style, speed);
                    bail!("MLX backend not compiled in")
                }
            }
        }
    }
}

/// Phonemize `text` (unless `is_ipa`), chunk it under the style-tensor cap, and
/// synthesize each chunk in `voice`, returning one trimmed+faded buffer per
/// chunk. Shared by the single-voice path and script mode.
#[allow(clippy::too_many_arguments)]
fn synthesize_voice_chunks(
    text: &str,
    is_ipa: bool,
    voice: &Voice,
    vocab: &HashMap<String, i64>,
    max_tokens: usize,
    rt: &mut Runtime,
    speed: f32,
    pitch_semitones: f32,
    espeak_voice: &str,
    trim_threshold: f32,
    fade: usize,
) -> Result<Vec<Vec<f32>>> {
    let ipa = if is_ipa {
        text.to_string()
    } else {
        run_espeak(text, espeak_voice)?
    };
    let chunks = chunk_ipa(&ipa, vocab, max_tokens);

    // Pitch shift, tempo-preserving: synthesize at `speed / r` (the model's own
    // time-stretch, which keeps pitch constant) so the audio is `r ×` longer,
    // then resample to `1/r` length — scaling every frequency by `r` and
    // restoring the original duration. r = 1 (pitch 0) is a no-op on both.
    let r = pitch_ratio(pitch_semitones);
    let synth_speed = speed / r;

    let mut bufs = Vec::new();
    for chunk in &chunks {
        let tokens = tokenize(chunk, vocab);
        if tokens.is_empty() {
            continue;
        }
        let style = select_style(voice, tokens.len());
        let audio = rt.synth(&tokens, style, synth_speed)?;
        let audio = if (r - 1.0).abs() > 1e-6 {
            resample_ratio(&audio, 1.0 / r as f64)?
        } else {
            audio
        };
        let mut buf = trim_silence(&audio, trim_threshold).to_vec();
        apply_fade(&mut buf, fade);
        bufs.push(buf);
    }
    Ok(bufs)
}

/// Resample to the requested rate and write/stream/play, per `--output`.
fn emit_audio(samples: &[f32], args: &Args) -> Result<()> {
    let resampled = resample(samples, NATIVE_SR, args.sample_rate)?;
    eprintln!(
        "storytime: {} samples @ {} Hz ({:.2}s)",
        resampled.len(),
        args.sample_rate,
        resampled.len() as f32 / args.sample_rate as f32
    );
    match &args.output {
        Some(path) if path.as_os_str() == "-" => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            write_wav_stream(&mut handle, &resampled, args.sample_rate, args.bit_depth)?;
            eprintln!("storytime: wrote WAV stream to stdout");
        }
        Some(path) => {
            write_wav(path, &resampled, args.sample_rate, args.bit_depth)?;
            eprintln!("storytime: wrote {}", path.display());
        }
        None => {
            eprintln!("storytime: playing to default output device");
            playback::play(&resampled, args.sample_rate)?;
        }
    }
    Ok(())
}

/// A synthesized speech ready for placement on the mix timeline.
struct TurnAudio {
    samples: Vec<f32>,
    interrupts_prev: bool,
}

/// Apply a linear gain ramp from 1.0 down to `floor` across `out[start..end]`,
/// ducking an interrupted speaker's tail under the interrupter.
fn duck_region(out: &mut [f32], start: usize, end: usize, floor: f32) {
    if end <= start || end > out.len() {
        return;
    }
    let n = (end - start) as f32;
    for (k, s) in out[start..end].iter_mut().enumerate() {
        let t = k as f32 / n;
        *s *= 1.0 - (1.0 - floor) * t;
    }
}

/// Mix turns onto a single mono timeline. Non-interrupting turns are laid end to
/// end separated by `line_gap` samples; an interrupting turn starts `overlap`
/// samples before the previous turn ends, summed in, with the previous turn's
/// overlapped tail ducked to `duck_gain`.
fn mix_timeline(turns: &[TurnAudio], line_gap: usize, overlap: usize, duck_gain: f32) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    let mut prev_end: usize = 0;
    for (i, t) in turns.iter().enumerate() {
        let start = if i == 0 {
            0
        } else if t.interrupts_prev {
            prev_end.saturating_sub(overlap)
        } else {
            prev_end + line_gap
        };
        if i > 0 && t.interrupts_prev && start < prev_end {
            duck_region(&mut out, start, prev_end, duck_gain);
        }
        let end = start + t.samples.len();
        if out.len() < end {
            out.resize(end, 0.0);
        }
        for (j, &s) in t.samples.iter().enumerate() {
            out[start + j] += s;
        }
        prev_end = end;
    }
    out
}

/// Screenplay mode: parse the script, resolve each character to a voice,
/// synthesize every speech in its voice, and mix the timeline (with overlapping
/// interruptions). Reuses the same per-text synthesis path as single-voice mode.
fn run_script(
    args: &Args,
    assets: &Path,
    raw_input: &str,
    vocab: &HashMap<String, i64>,
    rt: &mut Runtime,
    fade: usize,
) -> Result<()> {
    // 1. Parse the script (with an optional separate --cast file).
    let extra_cast = match &args.cast {
        Some(p) => Some(
            fs::read_to_string(p).with_context(|| format!("reading cast file {}", p.display()))?,
        ),
        None => None,
    };
    let parsed = script::parse(raw_input, extra_cast.as_deref())?;

    // 2. Resolve voices. Collect distinct speakers in first-appearance order so
    //    any label without a cast entry still gets a voice.
    let available = voice_names(assets)?;
    if available.is_empty() {
        bail!("no voices found under {}/voices", assets.display());
    }
    let narrator_default = args.narrator.clone().unwrap_or_else(|| args.voice.clone());
    let mut speakers: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for sp in &parsed.speeches {
        if seen.insert(sp.character.clone()) {
            speakers.push(sp.character.clone());
        }
    }
    let (mapping, warnings) =
        script::resolve_voices(&parsed.cast, &speakers, &available, &narrator_default);
    for w in &warnings {
        eprintln!("storytime: warning: {w}");
    }
    eprintln!(
        "storytime: script mode: {} speech(es), {} character(s)",
        parsed.speeches.len(),
        speakers.len()
    );
    // Per-character pitch: an explicit `pitch=` cast annotation, else the global
    // --pitch default. Keyed by the lowercased character name (matching
    // `Speech::character`).
    let mut char_pitch: HashMap<String, f32> = HashMap::new();
    for e in &parsed.cast {
        if let Some(p) = e.pitch {
            char_pitch.insert(e.name.trim().to_lowercase(), p);
        }
    }
    let pitch_for = |character: &str| char_pitch.get(character).copied().unwrap_or(args.pitch);
    for s in &speakers {
        let p = pitch_for(s);
        let pstr = if p != 0.0 { format!(", pitch {p:+}") } else { String::new() };
        let vid = mapping.get(s).map_or("?", |v| v);
        let ev = args.espeak_voice.as_deref().unwrap_or_else(|| espeak_voice_for(vid));
        eprintln!("storytime:   {s} -> {vid}{pstr} (espeak {ev})");
    }

    // 3. Load each distinct voice once. The style cap is the smallest across
    //    loaded voices so no chunk overruns any voice's row count.
    let mut voices: HashMap<String, Voice> = HashMap::new();
    for vid in mapping.values() {
        if !voices.contains_key(vid) {
            voices.insert(vid.clone(), load_voice(assets, vid)?);
        }
    }
    let max_tokens = voices
        .values()
        .map(|v| v.rows - 1)
        .min()
        .expect("at least one voice");

    // 4. Coalesce consecutive same-voice, non-interrupting speeches into one
    //    turn (joined by the paragraph pause marker) so they become a single,
    //    longer inference call — the same throughput win the narrator path gets
    //    from marker merging.
    struct Turn {
        voice_id: String,
        text: String,
        interrupts_prev: bool,
        pitch: f32,
    }
    let mut turns: Vec<Turn> = Vec::new();
    for sp in &parsed.speeches {
        let vid = mapping
            .get(&sp.character)
            .cloned()
            .unwrap_or_else(|| narrator_default.clone());
        let pitch = pitch_for(&sp.character);
        // Only merge consecutive speeches that share both voice and pitch.
        let mergeable = !sp.interrupts_prev
            && turns.last().is_some_and(|t| {
                t.voice_id == vid && (t.pitch - pitch).abs() < 1e-6 && !t.interrupts_prev
            });
        if mergeable {
            let last = turns.last_mut().unwrap();
            last.text.push_str(&args.paragraph_marker);
            last.text.push_str(&sp.text);
        } else {
            turns.push(Turn {
                voice_id: vid,
                text: sp.text.clone(),
                interrupts_prev: sp.interrupts_prev,
                pitch,
            });
        }
    }

    // 5. Synthesize each turn into a single buffer (its chunks joined by the
    //    chunk gap, as in single-voice mode).
    let chunk_gap = (NATIVE_SR as usize * args.chunk_gap_ms as usize) / 1000;
    let mut turns_audio: Vec<TurnAudio> = Vec::new();
    for (idx, turn) in turns.iter().enumerate() {
        let voice = &voices[&turn.voice_id];
        // Per-character espeak voice: an explicit --espeak-voice overrides all,
        // else derive it from this character's resolved voice (so a British
        // character is phonemized en-gb even mid-script).
        let espeak_voice = args
            .espeak_voice
            .as_deref()
            .unwrap_or_else(|| espeak_voice_for(&turn.voice_id));
        let text = if args.no_markdown {
            turn.text.clone()
        } else {
            preprocess_markdown(&turn.text)
        };
        let text = normalize_punctuation(&text);
        let bufs = synthesize_voice_chunks(
            &text,
            false,
            voice,
            vocab,
            max_tokens,
            rt,
            args.speed,
            turn.pitch,
            espeak_voice,
            args.trim_threshold,
            fade,
        )?;
        let mut samples: Vec<f32> = Vec::new();
        for (i, b) in bufs.iter().enumerate() {
            if i > 0 {
                samples.resize(samples.len() + chunk_gap, 0.0);
            }
            samples.extend_from_slice(b);
        }
        if samples.is_empty() {
            continue;
        }
        eprintln!(
            "storytime: turn {}/{} [{}]: {} samples{}",
            idx + 1,
            turns.len(),
            turn.voice_id,
            samples.len(),
            if turn.interrupts_prev { " (overlap)" } else { "" }
        );
        turns_audio.push(TurnAudio {
            samples,
            interrupts_prev: turn.interrupts_prev,
        });
    }
    if turns_audio.is_empty() {
        bail!("no audio produced from script");
    }

    // 6. Mix onto a mono timeline and emit.
    let line_gap = (NATIVE_SR as usize * args.line_gap_ms as usize) / 1000;
    let overlap = (NATIVE_SR as usize * args.overlap_ms as usize) / 1000;
    let mixed = mix_timeline(&turns_audio, line_gap, overlap, args.duck_gain);
    emit_audio(&mixed, args)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(Cmd::Clone(clone_args)) = cli.command {
        return clone::run(clone_args);
    }
    let args = cli.synth;

    let assets = assets_dir(args.assets.as_deref())?;

    if args.list_voices {
        return list_voices(&assets);
    }

    let raw_input = read_input(args.input.as_deref())?;
    if raw_input.trim().is_empty() {
        bail!("input was empty");
    }

    if args.script && args.ipa {
        bail!("--script and --ipa are mutually exclusive (script mode needs text for espeak-ng)");
    }

    let vocab = load_tokens(&assets)?;

    // Resolve the backend: explicit `--backend` wins, else default to MLX when
    // this binary was built with `--features mlx`, otherwise ONNX.
    let backend = args.backend.unwrap_or(if cfg!(feature = "mlx") {
        Backend::Mlx
    } else {
        Backend::Onnx
    });

    // Initialize the chosen inference backend (shared by both paths).
    let cache_dir = resolve_coreml_cache(args.no_coreml_cache, args.coreml_cache.as_deref())?;
    let mut rt = Runtime::init(backend, &assets, cache_dir)?;

    let fade = (NATIVE_SR as usize * args.fade_ms as usize) / 1000;

    // Screenplay mode is a separate front-end: parse the cast + speeches,
    // resolve voices, synthesize per character, and mix (with overlapping
    // interruptions). It shares the per-text synthesis path below.
    if args.script {
        return run_script(&args, &assets, &raw_input, &vocab, &mut rt, fade);
    }

    // ---- Single-voice path ----
    // Interpret markdown (unless disabled or in IPA mode, where the input is
    // already phonemes). Emphasis is carried forward as sentinels that become
    // IPA stress in `run_espeak`; all other markers are stripped here so they
    // are never spoken.
    let input_text = if args.no_markdown || args.ipa {
        raw_input
    } else {
        preprocess_markdown(&raw_input)
    };

    let voice = load_voice(&assets, &args.voice)?;
    let max_tokens = voice.rows - 1;

    // espeak-ng voice: an explicit --espeak-voice wins, else derive it from the
    // storytime voice's language prefix so the accent is phonemized correctly.
    let espeak_voice = args
        .espeak_voice
        .clone()
        .unwrap_or_else(|| espeak_voice_for(&args.voice).to_string());

    // Parse structure first so paragraph / section / chapter breaks survive
    // espeak-ng (which would otherwise flatten all whitespace).
    let blocks = parse_structure(&input_text);
    if blocks.is_empty() {
        bail!("no content after structural parsing");
    }
    eprintln!(
        "storytime: {} block(s), voice={}, speed={}, espeak={}",
        blocks.len(),
        args.voice,
        args.speed,
        espeak_voice
    );

    // Accumulate (preceding_gap, audio) pieces so we can insert the right
    // amount of silence between each.
    let mut pieces: Vec<(Boundary, Vec<f32>)> = Vec::new();

    // Flatten blocks into units. A block becomes one unit unless the user
    // has asked for explicit quote-aware silence (--quote-gap-ms > 0), in
    // which case it's split at quote boundaries into multiple units.
    struct Unit {
        text: String,
        gap_before: Boundary,
    }
    let mut units: Vec<Unit> = Vec::new();
    for block in blocks.iter() {
        let normalized = normalize_punctuation(&block.text);
        let sub_pieces = if args.quote_gap_ms > 0 {
            split_quotes(&normalized)
        } else {
            vec![normalized]
        };
        for (i, sub) in sub_pieces.into_iter().enumerate() {
            let gap = if i == 0 { block.gap_before } else { Boundary::Quote };
            units.push(Unit { text: sub, gap_before: gap });
        }
    }

    // Group consecutive units that are separated by a "soft" boundary
    // (Paragraph / Section / Quote) whose silence gap is zero. Adjacent
    // units in the same group are concatenated with a textual pause
    // marker so Kokoro generates the pause itself — this reduces the
    // number of inference calls substantially (fewer, longer inputs
    // amortize the model's fixed per-call overhead). Boundaries with
    // gap_ms > 0, Chapter, and Chunk always force a new group.
    struct Group {
        text: String,
        gap_before: Boundary,
    }
    let paragraph_marker = &args.paragraph_marker;
    let section_marker = &args.section_marker;
    let gap_ms_for = |b: Boundary| -> u32 {
        match b {
            Boundary::None => 0,
            Boundary::Chunk => args.chunk_gap_ms,
            Boundary::Quote => args.quote_gap_ms,
            Boundary::Paragraph => args.paragraph_gap_ms,
            Boundary::Section => args.section_gap_ms,
            Boundary::Chapter => args.chapter_gap_ms,
        }
    };
    let mut groups: Vec<Group> = Vec::new();
    for unit in units {
        let soft = matches!(
            unit.gap_before,
            Boundary::Paragraph | Boundary::Section | Boundary::Quote
        );
        let ms = gap_ms_for(unit.gap_before);
        if groups.is_empty() || !soft || ms > 0 {
            groups.push(Group {
                text: unit.text,
                gap_before: unit.gap_before,
            });
        } else {
            let marker = match unit.gap_before {
                Boundary::Paragraph => paragraph_marker.as_str(),
                Boundary::Section => section_marker.as_str(),
                _ => " ",
            };
            let last = groups.last_mut().unwrap();
            last.text.push_str(marker);
            last.text.push_str(&unit.text);
        }
    }

    eprintln!(
        "storytime: {} block(s) -> {} group(s) after marker merging",
        blocks.len(),
        groups.len()
    );

    for (group_idx, group) in groups.iter().enumerate() {
        let bufs = synthesize_voice_chunks(
            &group.text,
            args.ipa,
            &voice,
            &vocab,
            max_tokens,
            &mut rt,
            args.speed,
            args.pitch,
            &espeak_voice,
            args.trim_threshold,
            fade,
        )?;
        for (chunk_idx, buf) in bufs.into_iter().enumerate() {
            let gap_before = if pieces.is_empty() {
                Boundary::None
            } else if chunk_idx == 0 {
                group.gap_before
            } else {
                Boundary::Chunk
            };
            eprintln!(
                "storytime: group {}/{} chunk {}: {} samples ({:?} gap)",
                group_idx + 1,
                groups.len(),
                chunk_idx + 1,
                buf.len(),
                gap_before,
            );
            pieces.push((gap_before, buf));
        }
    }

    if pieces.is_empty() {
        bail!("no valid phoneme tokens produced from input");
    }

    let total_samples: usize = pieces.iter().map(|(b, a)| gap_samples_for(*b, &args, NATIVE_SR) + a.len()).sum();
    let mut samples: Vec<f32> = Vec::with_capacity(total_samples);
    for (gap, audio) in &pieces {
        let gap_n = gap_samples_for(*gap, &args, NATIVE_SR);
        samples.resize(samples.len() + gap_n, 0.0);
        samples.extend_from_slice(audio);
    }

    emit_audio(&samples, &args)
}

// ----------------------------------------------------------------------------
// Playback: OS-native, no third-party crates.
// ----------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod playback {
    //! macOS playback via AudioToolbox's AudioQueue (C API, system framework).
    //!
    //! AudioQueue is the "simplest modern" in-memory-PCM path that doesn't
    //! require Objective-C or file I/O: allocate a buffer, copy packed-float
    //! samples in, enqueue, start, wait for the buffer to drain, dispose.
    use anyhow::{anyhow, Result};
    use std::os::raw::c_void;
    use std::ptr;
    use std::time::Duration;

    // AudioStreamBasicDescription — laid out to match <CoreAudioTypes/CoreAudioBaseTypes.h>.
    #[repr(C)]
    struct AudioStreamBasicDescription {
        sample_rate: f64,
        format_id: u32,
        format_flags: u32,
        bytes_per_packet: u32,
        frames_per_packet: u32,
        bytes_per_frame: u32,
        channels_per_frame: u32,
        bits_per_channel: u32,
        _reserved: u32,
    }

    #[repr(C)]
    struct AudioQueueBuffer {
        audio_data_bytes_capacity: u32,
        audio_data: *mut c_void,
        audio_data_byte_size: u32,
        user_data: *mut c_void,
        packet_description_capacity: u32,
        packet_descriptions: *mut c_void,
        packet_description_count: u32,
    }

    // 'lpcm' FourCC in big-endian packing — AudioToolbox's format constant.
    const K_AUDIO_FORMAT_LINEAR_PCM: u32 = 0x6C70636D;
    const K_FLAG_IS_FLOAT: u32 = 1;
    const K_FLAG_IS_PACKED: u32 = 1 << 3;

    type AudioQueueRef = *mut c_void;

    // Buffer-done callback. We ship one buffer, so we just note completion.
    // The callback is invoked on the queue's internal thread.
    extern "C" fn on_buffer_done(
        user_data: *mut c_void,
        _queue: AudioQueueRef,
        _buffer: *mut AudioQueueBuffer,
    ) {
        unsafe {
            let done = &*(user_data as *const std::sync::atomic::AtomicBool);
            done.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[link(name = "AudioToolbox", kind = "framework")]
    extern "C" {
        fn AudioQueueNewOutput(
            format: *const AudioStreamBasicDescription,
            callback: extern "C" fn(*mut c_void, AudioQueueRef, *mut AudioQueueBuffer),
            user_data: *mut c_void,
            callback_run_loop: *mut c_void,
            callback_run_loop_mode: *mut c_void,
            flags: u32,
            out_aq: *mut AudioQueueRef,
        ) -> i32;
        fn AudioQueueAllocateBuffer(
            aq: AudioQueueRef,
            buffer_byte_size: u32,
            out_buffer: *mut *mut AudioQueueBuffer,
        ) -> i32;
        fn AudioQueueEnqueueBuffer(
            aq: AudioQueueRef,
            buffer: *mut AudioQueueBuffer,
            num_packet_descs: u32,
            packet_descs: *const c_void,
        ) -> i32;
        fn AudioQueueStart(aq: AudioQueueRef, start_time: *const c_void) -> i32;
        fn AudioQueueStop(aq: AudioQueueRef, immediate: u8) -> i32;
        fn AudioQueueDispose(aq: AudioQueueRef, immediate: u8) -> i32;
    }

    pub fn play(samples: &[f32], sample_rate: u32) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let fmt = AudioStreamBasicDescription {
            sample_rate: sample_rate as f64,
            format_id: K_AUDIO_FORMAT_LINEAR_PCM,
            format_flags: K_FLAG_IS_FLOAT | K_FLAG_IS_PACKED,
            bytes_per_packet: 4,
            frames_per_packet: 1,
            bytes_per_frame: 4,
            channels_per_frame: 1,
            bits_per_channel: 32,
            _reserved: 0,
        };

        let done = Box::new(std::sync::atomic::AtomicBool::new(false));
        let done_ptr = &*done as *const _ as *mut c_void;

        let mut queue: AudioQueueRef = ptr::null_mut();
        let status = unsafe {
            AudioQueueNewOutput(
                &fmt,
                on_buffer_done,
                done_ptr,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut queue,
            )
        };
        if status != 0 {
            return Err(anyhow!("AudioQueueNewOutput failed: OSStatus {status}"));
        }

        let byte_size = (samples.len() * std::mem::size_of::<f32>()) as u32;
        let mut buffer: *mut AudioQueueBuffer = ptr::null_mut();
        let status = unsafe { AudioQueueAllocateBuffer(queue, byte_size, &mut buffer) };
        if status != 0 {
            unsafe { AudioQueueDispose(queue, 1) };
            return Err(anyhow!("AudioQueueAllocateBuffer failed: OSStatus {status}"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                samples.as_ptr() as *const u8,
                (*buffer).audio_data as *mut u8,
                byte_size as usize,
            );
            (*buffer).audio_data_byte_size = byte_size;
        }

        let status = unsafe { AudioQueueEnqueueBuffer(queue, buffer, 0, ptr::null()) };
        if status != 0 {
            unsafe { AudioQueueDispose(queue, 1) };
            return Err(anyhow!("AudioQueueEnqueueBuffer failed: OSStatus {status}"));
        }

        let status = unsafe { AudioQueueStart(queue, ptr::null()) };
        if status != 0 {
            unsafe { AudioQueueDispose(queue, 1) };
            return Err(anyhow!("AudioQueueStart failed: OSStatus {status}"));
        }

        // Wait for the buffer-done callback. Poll briefly; the queue runs on
        // its own thread. Cap the total wait at duration + 2s as a safeguard.
        let duration = Duration::from_secs_f64(samples.len() as f64 / sample_rate as f64);
        let deadline = std::time::Instant::now() + duration + Duration::from_secs(2);
        while !done.load(std::sync::atomic::Ordering::SeqCst) {
            if std::time::Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        // Small tail so the device finishes flushing its own buffer.
        std::thread::sleep(Duration::from_millis(80));

        unsafe {
            AudioQueueStop(queue, 0);
            AudioQueueDispose(queue, 1);
        }
        drop(done);
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod playback {
    //! Linux playback via ALSA (libasound).
    //!
    //! ALSA is present on every desktop/server Linux install; PipeWire and
    //! PulseAudio both provide ALSA-compatible PCM devices through the
    //! `default` alias, so opening "default" works regardless of the
    //! higher-level sound server in use.
    use anyhow::{anyhow, Result};
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_long, c_uint, c_ulong, c_void};

    const SND_PCM_STREAM_PLAYBACK: c_int = 0;
    const SND_PCM_ACCESS_RW_INTERLEAVED: c_int = 3;
    const SND_PCM_FORMAT_FLOAT_LE: c_int = 14;

    #[link(name = "asound")]
    extern "C" {
        fn snd_pcm_open(
            pcm: *mut *mut c_void,
            name: *const c_char,
            stream: c_int,
            mode: c_int,
        ) -> c_int;
        fn snd_pcm_set_params(
            pcm: *mut c_void,
            format: c_int,
            access: c_int,
            channels: c_uint,
            rate: c_uint,
            soft_resample: c_int,
            latency: c_uint,
        ) -> c_int;
        fn snd_pcm_writei(pcm: *mut c_void, buffer: *const c_void, size: c_ulong) -> c_long;
        fn snd_pcm_recover(pcm: *mut c_void, err: c_int, silent: c_int) -> c_int;
        fn snd_pcm_drain(pcm: *mut c_void) -> c_int;
        fn snd_pcm_close(pcm: *mut c_void) -> c_int;
        fn snd_strerror(err: c_int) -> *const c_char;
    }

    fn err_string(code: c_int) -> String {
        unsafe {
            let p = snd_strerror(code);
            if p.is_null() {
                format!("code {code}")
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    pub fn play(samples: &[f32], sample_rate: u32) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut pcm: *mut c_void = std::ptr::null_mut();
        let name = CString::new("default").unwrap();
        let rc = unsafe { snd_pcm_open(&mut pcm, name.as_ptr(), SND_PCM_STREAM_PLAYBACK, 0) };
        if rc < 0 {
            return Err(anyhow!("snd_pcm_open: {}", err_string(rc)));
        }
        let rc = unsafe {
            snd_pcm_set_params(
                pcm,
                SND_PCM_FORMAT_FLOAT_LE,
                SND_PCM_ACCESS_RW_INTERLEAVED,
                1,           // mono
                sample_rate, // rate
                1,           // allow soft resampling
                500_000,     // target latency in microseconds (~0.5s)
            )
        };
        if rc < 0 {
            unsafe { snd_pcm_close(pcm) };
            return Err(anyhow!("snd_pcm_set_params: {}", err_string(rc)));
        }

        // Write in a loop: snd_pcm_writei may return fewer frames than requested
        // or a negative error (e.g. -EPIPE on xrun) which we recover from.
        let mut remaining = samples.len();
        let mut cursor = samples.as_ptr();
        while remaining > 0 {
            let n = unsafe { snd_pcm_writei(pcm, cursor as *const c_void, remaining as c_ulong) };
            if n < 0 {
                let rec = unsafe { snd_pcm_recover(pcm, n as c_int, 1) };
                if rec < 0 {
                    unsafe { snd_pcm_close(pcm) };
                    return Err(anyhow!("snd_pcm_writei: {}", err_string(rec)));
                }
                continue;
            }
            let written = n as usize;
            cursor = unsafe { cursor.add(written) };
            remaining -= written;
        }

        unsafe {
            snd_pcm_drain(pcm);
            snd_pcm_close(pcm);
        }
        Ok(())
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod playback {
    use anyhow::{bail, Result};
    pub fn play(_samples: &[f32], _sample_rate: u32) -> Result<()> {
        bail!("direct playback is only implemented on macOS and Linux; pass -o PATH to write a WAV instead");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience: render the sentinels visibly so assertions read clearly.
    fn show(s: &str) -> String {
        s.replace(SENT_BOLD_OPEN, "<b>")
            .replace(SENT_BOLD_CLOSE, "</b>")
            .replace(SENT_ITAL_OPEN, "<i>")
            .replace(SENT_ITAL_CLOSE, "</i>")
    }

    #[test]
    fn emphasis_to_sentinels() {
        assert_eq!(show(&parse_emphasis("a **bold** b")), "a <b>bold</b> b");
        assert_eq!(show(&parse_emphasis("a *italic* b")), "a <i>italic</i> b");
        assert_eq!(show(&parse_emphasis("a __bold__ b")), "a <b>bold</b> b");
        assert_eq!(show(&parse_emphasis("a _italic_ b")), "a <i>italic</i> b");
        // Triple markers collapse to the strong (bold) span.
        assert_eq!(show(&parse_emphasis("a ***x*** b")), "a <b>x</b> b");
        // Multi-word spans keep their inner spaces.
        assert_eq!(
            show(&parse_emphasis("the **big bad wolf**")),
            "the <b>big bad wolf</b>"
        );
    }

    #[test]
    fn underscores_only_emphasize_at_word_boundaries() {
        // snake_case must survive untouched.
        assert_eq!(parse_emphasis("call do_a_thing now"), "call do_a_thing now");
        // A stray, unmatched marker is emitted literally.
        assert_eq!(parse_emphasis("2 * 3 = 6"), "2 * 3 = 6");
    }

    #[test]
    fn strips_links_images_and_inline_code() {
        assert_eq!(inline_md("see [the docs](https://x.io) now"), "see the docs now");
        assert_eq!(inline_md("![a cat](cat.png) sat"), "a cat sat");
        assert_eq!(inline_md("run `cargo build` first"), "run cargo build first");
        // Reference-style link.
        assert_eq!(inline_md("see [the docs][1]"), "see the docs");
        // Bare brackets are not a link and stay literal.
        assert_eq!(inline_md("a [b] c"), "a [b] c");
    }

    #[test]
    fn strips_block_markers_but_keeps_headings() {
        let md = "# Title\n\n> quoted line\n\n- one\n- two\n\n---\n\nplain";
        let out = preprocess_markdown(md);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines.contains(&"# Title")); // heading marker preserved
        assert!(lines.contains(&"quoted line")); // blockquote stripped
        assert!(lines.contains(&"one")); // bullet stripped
        assert!(lines.contains(&"two"));
        assert!(lines.contains(&"plain"));
        assert!(!out.contains('>'));
        assert!(!out.contains("---"));
        assert!(!out.contains("- ")); // no bullet markers remain
    }

    #[test]
    fn emphasize_word_lengthens_stressed_vowel() {
        // Mild adds one length mark after the primary-stressed vowel.
        assert_eq!(emphasize_word("bˈuː", 1), "bˈuːː");
        // Strong adds two.
        assert_eq!(emphasize_word("bˈæd", 2), "bˈæːːd");
        // A word lacking any stress mark gets a primary mark prepended.
        assert!(emphasize_word("ðə", 1).starts_with('ˈ'));
        // A secondary mark is promoted to primary rather than prepended.
        let promoted = emphasize_word("ˌæbc", 0);
        assert!(promoted.contains('ˈ') && !promoted.contains('ˌ'));
        // None level is a no-op.
        assert_eq!(apply_emphasis_to_ipa("bˈuː", Emph::None), "bˈuː");
    }

    #[test]
    fn legacy_flag_invocation_parses_without_subcommand() {
        use clap::Parser as _;
        let cli = Cli::try_parse_from([
            "storytime", "--voice", "af_bella", "--speed", "1.1", "-o", "/tmp/x.wav",
        ])
        .expect("legacy invocation must keep parsing");
        assert!(cli.command.is_none());
        assert_eq!(cli.synth.voice, "af_bella");
        assert_eq!(cli.synth.output.as_deref(), Some(Path::new("/tmp/x.wav")));
    }

    #[test]
    fn espeak_voice_is_derived_from_language_prefix() {
        assert_eq!(espeak_voice_for("af_bella"), "en-us"); // American
        assert_eq!(espeak_voice_for("am_adam"), "en-us");
        assert_eq!(espeak_voice_for("bf_emma"), "en-gb"); // British
        assert_eq!(espeak_voice_for("bm_george"), "en-gb");
        assert_eq!(espeak_voice_for("ef_dora"), "es"); // Spanish
        assert_eq!(espeak_voice_for("ff_siwis"), "fr-fr"); // French
        assert_eq!(espeak_voice_for("jf_alpha"), "ja"); // Japanese
        assert_eq!(espeak_voice_for("zf_xiaobei"), "cmn"); // Mandarin
        // Names not following {lang}{gender}_ (e.g. a clone) default to English.
        assert_eq!(espeak_voice_for("myvoice"), "en-us");
        assert_eq!(espeak_voice_for("british_dad"), "en-us"); // no `_` at index 2
        assert_eq!(espeak_voice_for(""), "en-us");
    }

    #[test]
    fn pitch_ratio_and_resample_round_trip() {
        // 0 semitones is identity; an octave doubles/halves the ratio.
        assert!((pitch_ratio(0.0) - 1.0).abs() < 1e-6);
        assert!((pitch_ratio(12.0) - 2.0).abs() < 1e-5);
        assert!((pitch_ratio(-12.0) - 0.5).abs() < 1e-5);
        // resample_ratio scales the sample count by the ratio (the pitch/tempo
        // axis). The sinc resampler drops up to ~sinc_len/2 tail samples of
        // filter latency, negligible for real multi-second chunks.
        let sig: Vec<f32> = (0..24_000)
            .map(|n| (2.0 * std::f32::consts::PI * 200.0 * n as f32 / 24_000.0).sin())
            .collect();
        let up = resample_ratio(&sig, 0.5).unwrap(); // pitch up an octave (~half length)
        assert!((up.len() as i32 - 12_000).abs() <= 128, "got {}", up.len());
        assert_eq!(resample_ratio(&sig, 1.0).unwrap().len(), sig.len()); // identity
    }

    #[test]
    fn pitch_flag_parses_negative() {
        use clap::Parser as _;
        let cli = Cli::try_parse_from(["storytime", "--pitch", "-3.5", "--voice", "af_bella"])
            .expect("negative --pitch must parse");
        assert!((cli.synth.pitch + 3.5).abs() < 1e-6);
    }

    #[test]
    fn resolve_voice_path_falls_back_to_in_progress_temp() {
        let dir = std::env::temp_dir().join("storytime-resolve-test");
        let voices = dir.join("voices");
        std::fs::create_dir_all(&voices).unwrap();
        let final_p = voices.join("v.bin");
        let temp_p = voices.join("v.bin.temp");
        let _ = std::fs::remove_file(&final_p);
        let _ = std::fs::remove_file(&temp_p);

        // Neither exists -> the final path (so the error names the expected file).
        assert_eq!(resolve_voice_path(&dir, "v"), final_p);
        // Only the in-progress temp exists -> preview from it.
        std::fs::write(&temp_p, b"x").unwrap();
        assert_eq!(resolve_voice_path(&dir, "v"), temp_p);
        // A completed voice wins over an in-progress one.
        std::fs::write(&final_p, b"x").unwrap();
        assert_eq!(resolve_voice_path(&dir, "v"), final_p);
        // Path-like arguments pass through verbatim (lets a .temp be loaded explicitly).
        assert_eq!(resolve_voice_path(&dir, "/abs/x.bin"), PathBuf::from("/abs/x.bin"));
        assert_eq!(resolve_voice_path(&dir, "d/x.bin.temp"), PathBuf::from("d/x.bin.temp"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clone_subcommand_parses() {
        use clap::Parser as _;
        let cli = Cli::try_parse_from([
            "storytime", "clone", "--ref", "/tmp/me.wav", "--name", "kuy",
            "--init", "af_bella,af_heart", "--steps", "100",
        ])
        .expect("clone invocation must parse");
        match cli.command {
            Some(Cmd::Clone(c)) => {
                assert_eq!(c.name.as_deref(), Some("kuy"));
                assert_eq!(c.init, vec!["af_bella", "af_heart"]);
                assert_eq!(c.steps, 100);
            }
            other => panic!("expected clone subcommand, got {other:?}"),
        }
        // --print-script needs neither --ref nor --name.
        assert!(Cli::try_parse_from(["storytime", "clone", "--print-script"]).is_ok());
        // Without --print-script, --ref and --name are required.
        assert!(Cli::try_parse_from(["storytime", "clone"]).is_err());
    }

    #[test]
    fn apply_emphasis_handles_multiword_spans() {
        // Each word's stressed vowel is lengthened independently.
        assert_eq!(
            apply_emphasis_to_ipa("bˈɪɡ bˈæd", Emph::Strong),
            "bˈɪːːɡ bˈæːːd"
        );
    }
}
