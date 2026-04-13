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
use clap::{Parser, ValueEnum};
use hound::{SampleFormat, WavSpec, WavWriter};
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

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

    /// Treat stdin as IPA phonemes rather than text (no espeak-ng invocation).
    #[arg(long)]
    ipa: bool,

    /// Directory holding kokoro.onnx, tokens.json, voices/*.bin.
    /// Defaults to ../assets relative to the binary.
    #[arg(long)]
    assets: Option<PathBuf>,

    /// List available voices and exit.
    #[arg(long)]
    list_voices: bool,

    /// Silence inserted between chunker-forced splits within a paragraph, in ms.
    #[arg(long, default_value_t = 120)]
    chunk_gap_ms: u32,

    /// Silence inserted on entry to and exit from a quoted span (dialogue),
    /// in ms. Set to 0 to disable quote-aware pauses.
    #[arg(long, default_value_t = 250)]
    quote_gap_ms: u32,

    /// Silence inserted between paragraphs (separated by a blank line), in ms.
    #[arg(long, default_value_t = 400)]
    paragraph_gap_ms: u32,

    /// Silence inserted between sections (multiple blank lines or `## ` heading), in ms.
    #[arg(long, default_value_t = 700)]
    section_gap_ms: u32,

    /// Silence inserted between chapters (`# ` heading), in ms.
    #[arg(long, default_value_t = 1200)]
    chapter_gap_ms: u32,

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

    /// Output WAV path. If omitted, audio plays directly to the default
    /// output device via the OS-native audio API. Use `-` to stream the
    /// WAV to stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
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
fn resolve_coreml_cache(args: &Args) -> Result<Option<PathBuf>> {
    if args.no_coreml_cache {
        return Ok(None);
    }
    if let Some(p) = args.coreml_cache.as_ref() {
        return Ok(Some(p.clone()));
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

fn list_voices(assets: &Path) -> Result<()> {
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
    for n in names {
        println!("{n}");
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

fn run_espeak(text: &str) -> Result<String> {
    let mut child = Command::new("espeak-ng")
        .args(["-q", "--ipa=3", "-v", "en-us"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn espeak-ng (install it or use --ipa)")?;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(text.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("espeak-ng failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(String::from_utf8(out.stdout)?
        .chars()
        .filter(|c| !c.is_control() || *c == '\n')
        .collect::<String>()
        .replace('\n', " ")
        .trim()
        .to_string())
}

struct Voice {
    data: Vec<f32>,
    rows: usize,
}

fn load_voice(assets: &Path, name: &str) -> Result<Voice> {
    let path = assets.join("voices").join(format!("{name}.bin"));
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
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut r = SincFixedIn::<f32>::new(
        to as f64 / from as f64,
        2.0,
        params,
        samples.len(),
        1,
    )?;
    let out = r.process(&[samples.to_vec()], None)?;
    Ok(out.into_iter().next().unwrap())
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

/// Trim leading/trailing samples whose magnitude is below `threshold`.
/// Returns an empty slice if nothing clears the threshold.
fn trim_silence(samples: &[f32], threshold: f32) -> &[f32] {
    if threshold <= 0.0 || samples.is_empty() {
        return samples;
    }
    let start = samples.iter().position(|s| s.abs() > threshold);
    let end = samples.iter().rposition(|s| s.abs() > threshold);
    match (start, end) {
        (Some(s), Some(e)) => &samples[s..=e],
        _ => &[],
    }
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

fn main() -> Result<()> {
    let args = Args::parse();

    let assets = assets_dir(args.assets.as_deref())?;

    if args.list_voices {
        return list_voices(&assets);
    }

    let input_text = read_input(args.input.as_deref())?;
    if input_text.trim().is_empty() {
        bail!("input was empty");
    }

    let vocab = load_tokens(&assets)?;
    let voice = load_voice(&assets, &args.voice)?;
    let max_tokens = voice.rows - 1;

    // Parse structure first so paragraph / section / chapter breaks survive
    // espeak-ng (which would otherwise flatten all whitespace).
    let blocks = parse_structure(&input_text);
    if blocks.is_empty() {
        bail!("no content after structural parsing");
    }
    eprintln!(
        "storytime: {} block(s), voice={}, speed={}",
        blocks.len(),
        args.voice,
        args.speed
    );

    // Build ONNX session with CoreML EP (falls back to CPU automatically).
    // Tuning choices:
    //   - NeuralNetwork format: the MLProgram format trips an MPSGraph
    //     verification error on Kokoro's dynamic-shape matmuls on Apple
    //     Silicon (mps.matmul contracting-dim mismatch under dynamic token
    //     lengths). NeuralNetwork is the older but stable path.
    //   - compute units All: lets CoreML schedule across CPU + GPU + ANE.
    //   - model cache dir: CoreML otherwise recompiles the model on every
    //     session build (several seconds on a cold run). Caching makes
    //     subsequent starts fast.
    //   - fp16 accumulation on GPU: typically faster, negligible quality cost.
    ort::init().with_name("storytime").commit().map_err(ort_err)?;

    let cache_dir = resolve_coreml_cache(&args)?;
    let mut coreml = CoreMLExecutionProvider::default()
        .with_model_format(CoreMLModelFormat::NeuralNetwork)
        .with_compute_units(CoreMLComputeUnits::All)
        .with_low_precision_accumulation_on_gpu(true);
    if let Some(dir) = cache_dir.as_ref() {
        fs::create_dir_all(dir)?;
        coreml = coreml.with_model_cache_dir(dir.display().to_string());
        eprintln!("storytime: coreml cache: {}", dir.display());
    } else {
        eprintln!("storytime: coreml cache disabled");
    }

    let mut session = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_execution_providers([coreml.build()])
        .map_err(ort_err)?
        .commit_from_file(assets.join("kokoro.onnx"))
        .map_err(ort_err)?;

    let fade = (NATIVE_SR as usize * args.fade_ms as usize) / 1000;

    // Accumulate (preceding_gap, audio) pieces so we can insert the right
    // amount of silence between each.
    let mut pieces: Vec<(Boundary, Vec<f32>)> = Vec::new();

    // Flatten blocks into quote-aware units: each block is split at every
    // double-quote boundary so narration/dialogue transitions get an extra
    // typed pause. The first unit of a block carries the block's own gap
    // (paragraph/section/chapter); subsequent units within the same block
    // use the Quote gap.
    struct Unit<'a> {
        text: String,
        gap_before: Boundary,
        block_idx: usize,
        block_count: usize,
        within_block_idx: usize,
        within_block_count: usize,
        block: &'a Block,
    }
    let mut units: Vec<Unit> = Vec::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        let sub_pieces = if args.quote_gap_ms > 0 {
            split_quotes(&block.text)
        } else {
            vec![block.text.clone()]
        };
        let count = sub_pieces.len();
        for (i, sub) in sub_pieces.into_iter().enumerate() {
            let gap = if i == 0 { block.gap_before } else { Boundary::Quote };
            units.push(Unit {
                text: sub,
                gap_before: gap,
                block_idx,
                block_count: blocks.len(),
                within_block_idx: i,
                within_block_count: count,
                block,
            });
        }
    }
    // Silence the unused-field warnings for `block` — it's kept for clarity.
    let _ = &units.first().map(|u| u.block);

    for unit in &units {
        let ipa = if args.ipa {
            unit.text.clone()
        } else {
            run_espeak(&unit.text)?
        };
        let chunks = chunk_ipa(&ipa, &vocab, max_tokens);
        if chunks.is_empty() {
            continue;
        }
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let tokens = tokenize(chunk, &vocab);
            if tokens.is_empty() {
                continue;
            }
            let style = select_style(&voice, tokens.len());
            let audio = synthesize_chunk(&mut session, &tokens, style, args.speed)?;

            // Trim model-produced leading/trailing silence so the explicit
            // typed gap below is the only thing the listener perceives.
            let trimmed = trim_silence(&audio, args.trim_threshold).to_vec();
            let mut buf = trimmed;
            apply_fade(&mut buf, fade);

            let gap_before = if pieces.is_empty() {
                Boundary::None
            } else if chunk_idx == 0 {
                unit.gap_before
            } else {
                Boundary::Chunk
            };

            eprintln!(
                "storytime: block {}/{} piece {}/{} chunk {}/{}: {} tokens -> {} samples ({:?} gap)",
                unit.block_idx + 1,
                unit.block_count,
                unit.within_block_idx + 1,
                unit.within_block_count,
                chunk_idx + 1,
                chunks.len(),
                tokens.len(),
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

    eprintln!(
        "storytime: {} samples total @ {} Hz ({:.2}s)",
        samples.len(),
        NATIVE_SR,
        samples.len() as f32 / NATIVE_SR as f32
    );

    let resampled = resample(&samples, NATIVE_SR, args.sample_rate)?;
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
