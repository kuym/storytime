# storytime

A local command-line text-to-speech utility built on [hexgrad/Kokoro-82M][kokoro].
Reads UTF-8 text (or IPA phonemes) from stdin, synthesizes speech on-device using
ONNX Runtime with Apple's CoreML execution provider, and writes a WAV file with
configurable sample rate and bit depth.

No Python or PyTorch is involved at runtime — they're used only as a one-shot
export step to convert the upstream `.pth` checkpoint into an ONNX graph plus
plain float32 voice tensors that the Rust CLI consumes.

[kokoro]: https://huggingface.co/hexgrad/Kokoro-82M

---

## Features

- **File or speaker output.** With `-o PATH` writes a WAV file; without `-o`
  plays directly to the default output device using OS-native audio APIs
  (AudioToolbox's AudioQueue on macOS, ALSA on Linux) — no third-party
  audio crates.
- **Local & offline.** Model, voices, and runtime all live on disk. No network.
- **Hardware-accelerated.** Runs on the Apple Neural Engine / GPU via CoreML
  (`MLComputeUnits=All`, fp16 GPU accumulation), with automatic CPU fallback
  for ops CoreML doesn't cover. The compiled CoreML model is cached between
  runs in `~/Library/Caches/storytime/coreml`, cutting cold-start by ~50%
  after the first invocation.
- **Two input modes.**
  - *Text mode* (default): stdin is raw text; the tool shells out to
    `espeak-ng` for grapheme-to-phoneme conversion.
  - *IPA mode* (`--ipa`): stdin is IPA phonemes directly. No `espeak-ng`
    dependency on this process — composes in a POSIX pipeline with any G2P.
- **Automatic chunking.** Inputs longer than the model's ~510-token style
  limit are automatically split at sentence boundaries (falling back to word,
  then character boundaries if a sentence or word is itself too long),
  synthesized chunk-by-chunk, and concatenated with a short silence gap
  between pieces.
- **Structural pause handling.** Paragraph, section, and chapter boundaries
  are preserved from the input and translated into longer silence gaps
  (configurable via flags). Each chunk's leading/trailing model-produced
  silence is trimmed and a short linear fade is applied at the seams so the
  resulting audio has clean, realistically-spaced transitions.
- **54 voices** from the Kokoro-82M release (en-US, en-GB, Spanish, French,
  Italian, Hindi, Japanese, Brazilian Portuguese, Mandarin).
- **Configurable output.** 16/24/32-bit PCM or IEEE float32, any sample rate
  (resampled from the model's native 24 kHz via a high-quality sinc resampler).
- **Adjustable speaking rate** via `--speed`, and **pitch shifting** via `--pitch`
  (semitones, tempo-preserving) — global, or per character in script mode.
- **Voice cloning.** `storytime clone` builds a new voicepack from a 10–20 s
  recording of a real speaker via offline style-space optimization — see
  [Voice cloning](#voice-cloning).

---

## Inference backends

`storytime` supports two backends, selectable via `--backend`. **Both use the
same `kokoro.onnx` model and `voices/*.bin` assets** — there is no separate set
of MLX weights.

| backend | flag | acceleration | notes |
|---|---|---|---|
| **ONNX** | `--backend onnx` | CoreML EP (ANE/GPU/CPU) | available in every build |
| **MLX** | `--backend mlx` | Metal GPU, else CPU | requires `--features mlx`; **the default in that case** |

### MLX backend

The MLX backend interprets `kokoro.onnx` **directly** on Apple's
[MLX](https://github.com/ml-explore/mlx) (via the [mlx-c](https://github.com/ml-explore/mlx-c)
C API): it parses the ONNX graph natively in Rust and runs each operator as an
MLX op on the Metal GPU. It is verified numerically equivalent to ONNX Runtime
CPU, op-for-op, to float32 epsilon (the only residual is the inherent f32
conditioning of the vocoder's harmonic oscillator — see
`docs/onnx-to-mlx-plan.md`).

The same interpreter is reused for the GE2E speaker encoder in
[voice cloning](#voice-cloning), so an `--features mlx` build runs the entire
clone loop — synthesis and embedding — on the GPU.

```sh
# Build with MLX support (requires Xcode + the Metal toolchain + macOS 14+).
# First build clones and compiles mlx-c (~5 min); cached afterwards.
cd cli
cargo build --release --features mlx
# (if Metal compilation fails: xcodebuild -downloadComponent MetalToolchain)

# Run — MLX is the default backend in an mlx build; GPU is auto-selected.
echo "Hello." | storytime -o hello.wav            # backend=mlx, device=Gpu
echo "Hello." | storytime --backend onnx -o h.wav # force the ONNX backend
```

With `--features mlx`, the default backend is **MLX on the Metal GPU** when a
compatible GPU is present, falling back to **MLX on CPU** otherwise. Without the
feature, only the ONNX backend is compiled in and is the default. The MLX build
vendors nothing — `build.rs` fetches and builds mlx-c at build time.

## Architecture

```
                   ┌──────────────────── one-shot (Python) ────────────────────┐
                   │                                                            │
   kokoro-v1_0.pth │   export.py   │→  kokoro.onnx                              │
   voices/*.pt    ─┼─▶  (kokoro +  │→  voices/*.bin  (float32, [N,1,256])      │
   config.json     │    PyTorch)   │→  tokens.json   (IPA char → token-id)     │
   GE2E weights    │               │→  spk_encoder.onnx (voice-cloning scorer) │
                   └────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
  ┌──────────────────────────── runtime (Rust binary) ────────────────────────┐
  │                                                                            │
  │  stdin  ──▶  [espeak-ng subprocess]* ──▶  IPA  ──▶  [token ids]            │
  │                                                        │                   │
  │                                                        ▼                   │
  │  voice .bin ─▶ [style row select] ─────▶  [ONNX Runtime + CoreML EP] ─▶ f32│
  │                                                        │                   │
  │                                                        ▼                   │
  │                                               [resample + WAV encode] ─▶ out.wav
  │                                                                            │
  │  * skipped when --ipa is passed                                            │
  └────────────────────────────────────────────────────────────────────────────┘
```

### Why ONNX + CoreML (and not XNNPACK / TFLite / Executorch)?

The original request mentioned XNNPACK or comparable runtimes. Of the viable
options on Apple Silicon:

| runtime | export effort | CPU perf | ANE/GPU | notes |
|---|---|---|---|---|
| **ONNX Runtime + CoreML EP** | low (single `torch.onnx.export`) | good (MLAS kernels, AVX/NEON) | ✅ via CoreML EP | chosen |
| ONNX Runtime + XNNPACK EP | low | good | ❌ CPU only | no reason to skip CoreML on this Mac |
| TFLite + XNNPACK | high (torch → onnx → tf → tflite, lossy) | good | partial via CoreML delegate | more fragile pipeline |
| ExecuTorch | high (custom export, immature for this model class) | good | ✅ MPS backend | not yet worth the setup cost |

ONNX Runtime's CoreML execution provider gives us ANE/GPU acceleration for
essentially zero additional work, with a graceful CPU fallback for ops CoreML
doesn't support. The Rust binding (`ort`) bundles a matched runtime, so there's
no separate shared library to install.

### Why espeak-ng (as a subprocess)?

Kokoro is not trained on raw text — it consumes a fixed vocabulary of IPA
phoneme tokens (178 tokens, defined in the model's `config.json`). A
grapheme-to-phoneme (G2P) step is required before inference.

Kokoro was trained using [`espeak-ng`][espeak]'s IPA output specifically, so
using a different G2P degrades pronunciation quality. The alternatives all
have significant tradeoffs:

- **Bundle/link libespeak-ng statically.** Possible, but espeak-ng is GPLv3 —
  statically linking makes the whole binary GPLv3.
- **Port Misaki (the official Kokoro G2P) to Rust.** Misaki is Python-only and
  large; a Rust port is a separate multi-week project, and for non-English
  languages Misaki falls back to espeak-ng anyway.
- **Shell out to `espeak-ng` as a subprocess.** Keeps the license boundary at
  the process boundary, works immediately, and is trivially replaceable with
  `--ipa` for users who already have IPA from another source.

This tool takes the subprocess approach and additionally exposes `--ipa` so
espeak-ng becomes optional when piping IPA from elsewhere.

[espeak]: https://github.com/espeak-ng/espeak-ng

---

## Setup

### Prerequisites

- macOS on Apple Silicon (tested on macOS 15, arm64) or Debian/Ubuntu Linux.
  On macOS the CoreML EP is used; elsewhere it falls back to CPU.
- Rust toolchain (stable, ≥ 1.85) — see https://rustup.rs.
- A system package manager: `brew` on macOS, `apt` on Linux. `setup.sh`
  installs everything else it needs (Python, `espeak-ng`).

### 1. One-time: download + export the model

```sh
./setup.sh
```

This downloads the Kokoro-82M checkpoint and voices from HuggingFace
(`hexgrad/Kokoro-82M`, pinned to a known-good revision), installs the export
dependencies into `export/.venv`, and converts everything into `assets/`. It is
safe to re-run.

<details>
<summary>Manual export (if you already have Python set up)</summary>

```sh
cd export
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export.py                      # downloads from HuggingFace, then exports
python export.py --snapshot /path/to/local/snapshot   # or use a local snapshot
```
</details>

Either path writes:

```
assets/
├── kokoro.onnx       # ~325 MB, the full model
├── tokens.json       # IPA char → token-id vocab
├── spk_encoder.onnx  # ~6 MB GE2E speaker encoder (only used by `storytime clone`)
└── voices/
    ├── af_alloy.bin
    ├── af_bella.bin
    └── ... (54 files)
```

Notes:

- The export uses the legacy TorchScript exporter (`dynamo=False`) because the
  new TorchDynamo-based exporter fails on the transformers library's SDPA
  attention path.
- `transformers` is pinned to `4.47.1`. Newer versions introduce a
  `create_bidirectional_mask` helper that doesn't survive tracing.
- Voice tensors are stored as raw little-endian float32 with shape
  `[N, 1, 256]`, where `N` is 510 or 511 depending on the voice. The Rust
  loader handles either.

### 2. Build the CLI

```sh
cd cli
cargo build --release
# binary: cli/target/release/storytime
```

The `ort` crate is configured with `download-binaries`, so a matched
ONNX Runtime is fetched at build time. No `brew install onnxruntime` is
needed.

### 3. (Optional) Install espeak-ng for text mode

```sh
brew install espeak-ng
```

Skip this if you'll always pipe pre-computed IPA via `--ipa`.

---

## Usage

### Basic

```sh
# Read from a file instead of stdin
storytime -i story.txt -o story.wav

# Play directly through the speakers (no -o)
echo "Hello, world." | storytime --voice af_bella

# Text in, WAV out (requires espeak-ng)
echo "Hello, world." | storytime --voice af_bella -o hello.wav

# IPA in, WAV out (no espeak-ng required on this process)
echo "həlˈoʊ wˈɜːld." | storytime --ipa -o hello.wav

# Compose with an external G2P
espeak-ng -q --ipa=3 -v en-us "Hello, world." \
  | storytime --ipa -o hello.wav
```

### Punctuation normalization

Before anything else, each block's text is normalized to the forms
Kokoro's vocab prefers:

- **Ellipses.** Runs of three or more ASCII dots (`...`, `....`) are
  collapsed to a single `…` (U+2026). Kokoro has a dedicated token
  for `…`; leaving it as three separate `.` tokens gives three clipped
  pauses instead of one sustained one.
- **Quote pairs.** Straight `"` characters are paired off into
  alternating curly `"` (open, U+201C) and `"` (close, U+201D).
  Kokoro has distinct open/close tokens with different learned
  prosody; mapping everything to the undifferentiated straight
  `"` token loses that distinction.

Input that already uses `…` or curly quotes passes through unchanged.

### Punctuation prosody

Kokoro's training vocab includes `; : , . ! ? — … " ( )` as first-class
tokens, and the model is trained to produce pauses and intonation
changes on them. espeak-ng in `--ipa=3` mode, however, silently
**strips every one of these** from its output, so a naive text →
espeak → Kokoro pipeline loses all punctuation prosody.

To fix this without switching to Misaki (the upstream Python G2P,
which would mean adding a Python runtime dependency), `storytime`
splits each block on preserved-punctuation boundaries, feeds only the
text segments to espeak-ng (one per line), and interleaves the
punctuation back into the IPA output before tokenizing. One espeak
invocation per block, same as a naive pass-through — but now `?`,
`:`, `;`, `—`, `…`, commas, and quote marks all reach the model and
drive its prosody.

If you see unnaturally flat delivery on a specific passage, check the
input punctuation is present; the model uses it heavily.

### Structural pauses

Kokoro produces natural prosody (short pauses on `.`/`,`/`;`/etc.) inside a
block of running text, but espeak-ng collapses all whitespace before the
model sees anything — so without help, paragraph and chapter boundaries
get the same pause as a comma.

`storytime` parses the input into structural blocks *before* phonemization
and inserts a typed silence gap at each boundary:

| boundary | how it's detected | default |
|---|---|---|
| paragraph | one blank line between non-empty lines | inline marker `" — — — "` |
| section | two or more blank lines, or a `## `/`### ` heading | inline marker `" — — — — — "` |
| chapter | a `# ` heading | 1200 ms silence |
| quote | entry to / exit from a `"..."` span | rely on quote tokens |
| within-paragraph chunk split | forced by the 510-token limit | 120 ms silence |

**Two mechanisms** drive pauses: **textual markers** (inserted into the
text before phonemization so Kokoro generates the pause itself from its
trained prosody) and **explicit silence** (zero samples spliced in after
synthesis). Markers are strictly faster because multiple blocks merge
into a single inference call — Kokoro has a fixed per-call overhead
that amortizes over longer inputs, so collapsing 7 short blocks into
one ~500-token call is visibly faster than 7 short calls.

By default paragraph and section boundaries use markers (no inference
split), quote boundaries use neither (the `"` tokens alone drive
prosody), and chapter boundaries use explicit silence (the pause is
too long to express cleanly as inline punctuation).

Markdown heading markers (`# `, `## `, `### `) are stripped from the spoken
text but their presence upgrades the boundary strength. So this input:

```
# Chapter One

The night was dark and stormy.

Suddenly, a shot rang out.

## A Pause

It was quiet again.

# Chapter Two

The end.
```

…produces "Chapter One" → 1200 ms → paragraph → 400 ms → paragraph →
700 ms ("A Pause") → 400 ms → paragraph → 1200 ms → "Chapter Two" → 400 ms
→ "The end." (and the `#` markers themselves are not spoken).

Within any paragraph, the text is *also* split at every entry/exit of a
double-quote span (straight `"..."` or curly `"..."`), so dialogue gets
a small pause before and after the character's line — e.g. `She said,
"Hello." Then she left.` becomes three pieces with Quote gaps at the
transitions. Single quotes are not used as boundaries (they're
indistinguishable from apostrophes in plain text).

The flags `--paragraph-gap-ms`, `--section-gap-ms`, `--chapter-gap-ms`,
`--chunk-gap-ms`, `--quote-gap-ms` tune the durations. Set any of them
to `0` to disable that boundary type. Additionally, before inserting each
gap the synthesized chunk is:

1. **Trimmed** of leading/trailing near-silence (`--trim-threshold`,
   default 0.005), so the typed gap above is the *only* silence the
   listener hears at that boundary — no stacked model tail.
2. **Fade in/out** applied linearly over `--fade-ms` (default 10 ms) at
   both ends, which removes the clicks you'd otherwise hear when
   non-zero-crossing samples sit next to inserted silence (particularly
   audible at long chapter gaps).

In `--ipa` mode, structural parsing still runs: preserve blank lines in
your piped IPA to get paragraph/section gaps, or use `# ` / `## `
prefixes to mark chapter/section boundaries.

### Markdown formatting

Text input is treated as Markdown by default. Formatting markers are
*interpreted* and then stripped, so they are never spoken, and emphasis is
translated into Kokoro's phoneme-level stress:

| Markdown | Effect on speech |
|---|---|
| `*italic*` / `_italic_` | **Stressed** — the word's primary-stressed vowel is lengthened (one `ː`). |
| `**bold**` / `__bold__` | **Emphasized** — a stronger, more drawn-out lengthening (two `ː`). |
| `# ` / `## ` / `### ` headings | Marker stripped; boundary strength upgraded (see above). |
| `[text](url)`, `[text][ref]`, `![alt](url)` | Collapsed to their visible text (`text` / `alt`); the URL is dropped. |
| `` `code` `` and fenced ``` ``` ``` ``` blocks | Backticks/fences removed; the contents are kept and spoken as plain text. |
| `- ` / `* ` / `+ ` / `1. ` list bullets | Bullet removed; item text kept. |
| `> ` blockquotes, `---`/`***`/`___` rules, `~~strike~~` | Markers removed. |

Emphasis works because Kokoro consumes IPA and is trained on the stress
(`ˈ`/`ˌ`) and length (`ː`) tokens that the upstream Misaki G2P produces.
storytime ensures the emphasized word carries a primary-stress mark and then
lengthens its stressed vowel — so `She was **very** happy` phonemizes the
emphasized word as `vˈɛːːɹi` rather than a flat `vˈɛɹi`. Underscores only
emphasize at word boundaries, so identifiers like `do_a_thing` are left alone.

Pass `--no-markdown` to take the input as literal text instead (no stripping,
no emphasis). `--ipa` mode implies `--no-markdown`, since the input is already
phonemes.

### Streaming to stdout

Passing `-o -` writes the WAV to stdout so the output can be piped
directly into another process without a temporary file:

```sh
echo "Streamed." | storytime -o - | ffplay -autoexit -nodisp -
echo "Streamed." | storytime -o - | ffmpeg -i - out.mp3
```

Because stdout is not seekable, the RIFF/data size fields in the
header can't be back-patched after the fact and are set to the maximum
`u32` value (`0xFFFFFFFF`). Streaming-aware decoders (ffmpeg, sox, VLC,
the majority of media players) either honor that sentinel or read
until EOF. A few strict parsers that validate the declared sizes may
reject these streams — save to a file with `-o out.wav` in that case.

### Direct playback

If you don't pass `-o`, the synthesized audio is played through the default
output device using the OS-native audio API:

- **macOS** — AudioToolbox's AudioQueue (linked via the `AudioToolbox`
  system framework). No extra install; works on any supported macOS.
- **Linux** — ALSA (`libasound`). Install the ALSA runtime if it isn't
  already present (`sudo apt install libasound2` on Debian/Ubuntu, etc.).
  Opening the `default` PCM device works transparently through PipeWire
  and PulseAudio as well.

Both paths are pure FFI — no `cpal`, `rodio`, or other third-party audio
crates in the dependency tree. The `--sample-rate` and `--bit-depth` flags
still apply to file output; playback always uses the resampled float32
stream at the chosen `--sample-rate`.

### Controlling the output format

```sh
echo "Test." | storytime \
  --voice am_michael \
  --sample-rate 48000 \
  --bit-depth 24 \
  --speed 1.1 \
  -o test.wav
```

### Listing voices

```sh
storytime --list-voices
```

### Script / multi-voice

With `--script`, storytime reads a screenplay where different characters speak
in different voices. The format is the universal `NAME: dialogue` convention any
LLM (local or hosted) already produces reliably, plus a `# Cast` header that
assigns each character a voice. storytime segments the input by speaker,
synthesizes each speech in that character's voice, and mixes the result —
including genuine overlap when one character interrupts another.

```sh
storytime --script -i play.md -o play.wav
```

A complete example:

```markdown
# Cast
ALICE: female, american, young
BOB: male, british, gruff
NARRATOR: af_heart            # an explicit voice id also works
GIANT: bm_george pitch=-5     # optional per-character pitch shift

---

NARRATOR: They stood at the door, neither willing to move first.

ALICE: Are you sure about this? I really think we should wait and--
BOB: Stop. We're going in, and that's final.

ALICE: *Fine.* But if this goes wrong, it's on you.
```

The format (matching is case-insensitive and forgiving, so models don't have to
be precise):

- **Cast block** — the lines under a `Cast` or `Dramatis Personae` heading, up to
  the next heading or `---` rule. Each entry is `NAME: voice`, where `voice` is
  either an explicit voice id (`af_bella`) or a **trait list**. It is never spoken.
- **Traits** — `gender` (`female`/`male`) and `accent`/language (`american`,
  `british`, `spanish`, `french`, `hindi`, `italian`, `japanese`, `portuguese`,
  `mandarin`) are honored exactly, since they are encoded in the voice catalog.
  Other traits (`young`, `gruff`, `warm`, `deep`, …) are best-effort: they keep
  each character's voice distinct and make the assignment reproducible, but the
  catalog can't guarantee a specific timbre. Each character gets a different voice.
- **Speech** — a line beginning `NAME:` (a declared character, or any name in
  screenplay all-caps) starts that character's turn; following non-speaker lines,
  up to a blank line or the next speaker, belong to it.
- **Pitch** — a cast entry may add `pitch=<semitones>` (e.g. `pitch=-5` to drop a
  giant's voice, `pitch=+8` to lift a mouse's). It shifts that character's pitch
  while preserving tempo. Characters without one use the global `--pitch` (default
  `0`), so `--script --pitch 2` lifts the whole scene except where overridden.
- **Narration** — lines with no speaker are spoken by `NARRATOR` (or `--narrator`
  / `--voice` if no narrator is cast).
- **Interruptions** — end a speech with `--` or `—` and the next speech overlaps
  it: the interrupter begins `--overlap-ms` before the first finishes, and the
  interrupted tail is ducked under it (`--duck-gain`).
- **Parentheticals** — `(stage directions)` are stripped, not spoken.
- Markdown inside a speech (`*emphasis*`, quotes, punctuation) works as usual.

Output is mono. The cast can also live in a separate file via `--cast cast.md`,
leaving the body as pure dialogue. `--script` is incompatible with `--ipa`.

### Flags

| flag | default | description |
|---|---|---|
| `-i, --input PATH` | *(stdin)* | read input from file; `-` or omitted means stdin |
| `--voice NAME` | `af_heart` | voice name from `--list-voices` |
| `--sample-rate HZ` | `24000` | output sample rate; model native is 24 kHz |
| `--bit-depth {16,24,32,float32}` | `16` | PCM bit depth |
| `--speed FLOAT` | `1.0` | speaking rate multiplier |
| `--pitch SEMITONES` | `0` | pitch shift in semitones (+ up / − down), tempo preserved; script-mode default (see below) |
| `--ipa` | off | treat stdin as IPA (skip espeak-ng) |
| `--assets PATH` | `../assets` | location of exported assets |
| `--list-voices` | — | list available voices and exit |
| `-o, --output PATH` | *(unset)* | write WAV here; `-` streams WAV to stdout; if omitted, play to default output device |
| `--chunk-gap-ms` | `120` | silence between chunker-forced splits inside a paragraph |
| `--quote-gap-ms` | `0` | silence at quote transitions; `> 0` forces a quote-aware split |
| `--paragraph-gap-ms` | `0` | silence between paragraphs; `> 0` forces a split (overrides marker) |
| `--section-gap-ms` | `0` | silence between sections; `> 0` forces a split |
| `--chapter-gap-ms` | `1200` | silence between chapters (`# ` heading) |
| `--paragraph-marker` | `". … "` | inline marker between paragraphs (period = sentence-ending prosody, ellipsis = sustained pause) |
| `--section-marker` | `". … … "` | inline marker between sections |
| `--fade-ms` | `10` | linear fade-in/out at every chunk seam (avoids clicks) |
| `--trim-threshold` | `0.005` | amplitude below which per-chunk leading/trailing silence is trimmed (`0` disables) |
| `--coreml-cache PATH` | `~/Library/Caches/storytime/coreml` | where CoreML stores its compiled model between runs |
| `--no-coreml-cache` | off | disable the cache (forces recompilation each run) |
| `--script` | off | screenplay mode: `NAME: dialogue` with a `# Cast` header (see [Script / multi-voice](#script--multi-voice)) |
| `--cast PATH` | *(unset)* | read the cast from a separate file (script mode) |
| `--narrator NAME` | *(= `--voice`)* | voice for unattributed narration (script mode) |
| `--overlap-ms` | `250` | overlap when one speech interrupts another (script mode) |
| `--duck-gain` | `0.4` | gain applied to an interrupted speech's tail under the interrupter (script mode) |
| `--line-gap-ms` | `120` | silence between consecutive non-overlapping speeches (script mode) |

### Voices

Naming convention: `{lang}{gender}_{name}`.

- `af_*`, `am_*` — American female / male
- `bf_*`, `bm_*` — British female / male
- `ef_*`, `em_*` — Spanish
- `ff_*` — French
- `hf_*`, `hm_*` — Hindi
- `if_*`, `im_*` — Italian
- `jf_*`, `jm_*` — Japanese
- `pf_*`, `pm_*` — Brazilian Portuguese
- `zf_*`, `zm_*` — Mandarin Chinese

Match your voice to your input language — pronunciation quality depends on
both the voice embedding and the G2P output that produced the IPA.

---

## Voice cloning

`storytime clone` creates a **new voicepack from a short recording of a real
speaker** — record yourself once, then narrate any story in (an approximation
of) your own voice, fully offline:

```sh
# 1. See the paragraph you'll need to read aloud.
storytime clone --print-script

# 2. Record yourself reading it (10–20 s, quiet room, any recorder), then
#    convert to mono WAV:
ffmpeg -i raw.m4a -ar 24000 -ac 1 ref.wav

# 3. Optimize a new voicepack against the recording (hours; interruptible —
#    audition mid-run and stop early, or extend later with --resume).
storytime clone --ref ref.wav --name myvoice --budget-min 60

# 4. Use it like any stock voice, forever, at zero extra runtime cost.
echo "Once upon a time..." | storytime --voice myvoice -o story.wav
```

### How it works (and what to expect)

Kokoro ships no reference/style encoder — hexgrad deliberately withheld it —
so a recording cannot be *encoded* into the model's 256-dim style space.
Instead, `clone` runs a gradient-free hill-climb over that space (after
[KVoiceWalk](https://github.com/RobViren/kvoicewalk)): it starts from a blend
of the stock voices closest to your recording, then repeatedly perturbs the
style vector, synthesizes two fixed test utterances, and scores the audio
against your recording — a weighted harmonic mean of **speaker-embedding
similarity** (a GE2E encoder exported to `assets/spk_encoder.onnx`),
**cross-text self-similarity** (stability), and a low-weight **acoustic
feature** guard. Improvements are kept; the result is an ordinary
`voices/<name>.bin`. Details in `docs/voice-cloning.md`.

Set expectations accordingly: Kokoro is an 82 M-param model trained on a few
hundred hours — the clone will be *recognizably you-ish*, not a studio-grade
voice double. Results are stochastic; a different `--seed` can land a
noticeably better (or worse) voice, and runs are cheap to repeat. Each step
synthesizes one or two ~13 s test utterances, so throughput is backend-bound —
on the **MLX GPU backend the whole loop (synthesis *and* the speaker encoder)
runs on the Metal GPU at ~2 steps/s**, versus ~0.2 steps/s on ONNX/CPU. Even at
the faster rate the default `--steps 2000` is a long walk, which is why
`--budget-min`, mid-run auditioning, and `--resume` exist. The two backends are
verified to score identically (the embedding agrees to cosine > 0.999), so the
choice is purely speed.

### Practical knobs

| flag | default | meaning |
|---|---|---|
| `--steps` | `2000` | maximum optimization steps |
| `--budget-min` | `0` (off) | wall-clock cap in minutes; stops at whichever of `--steps`/budget hits first |
| `--init` | auto | starting blend, e.g. `--init af_bella,af_heart`; default ranks all English voices against your recording and blends the top 3 |
| `--seed` | `0` | RNG seed (best-effort reproducibility: CoreML/Metal inference is not bit-deterministic; use `--backend onnx` for stricter runs) |
| `--resume` | | continue an interrupted walk from its saved state (`voices/<name>.bin.temp.json`) |
| `--ref-text FILE` | built-in script | transcript, if you recorded something other than `--print-script` (needs espeak-ng) |
| `--backend` | `mlx` if built with `--features mlx`, else `onnx` | runs both synthesis and the speaker encoder; `mlx` keeps the whole loop on the GPU |

#### Pause, resume, and preview (like a browser download)

Training is incremental and crash-safe. While it runs, the in-progress voice
lives in `voices/<name>.bin.temp` (a partial-download file), with its training
state in a `voices/<name>.bin.temp.json` sidecar; both are rewritten atomically
every ~50 steps or 60 seconds. The final `voices/<name>.bin` appears **only when
training completes** (the temp file is then renamed into place and the sidecar
removed). So you can:

```sh
# Start training. Press Ctrl-C any time to stop gracefully: it finishes the
# current step, writes a checkpoint, and exits (press Ctrl-C again to abort
# immediately). A kill/reboot loses at most the last checkpoint interval.
storytime clone --ref ref.wav --name myvoice --steps 2000

# Resume from exactly where it left off (continues toward the original --steps):
storytime clone --ref ref.wav --name myvoice --resume

# Preview the partial voice from ANOTHER terminal while it's still training —
# the bare name resolves to the in-progress .bin.temp until the final exists:
echo "Once upon a time..." | storytime --voice myvoice -o preview.wav
```

`--budget-min N` is the same pause built in: it stops after N minutes leaving a
resumable temp, so `--budget-min 5` then `--resume` (repeatedly) walks a long
clone in short sessions. `--list-voices` shows in-progress clones as
`<name> (training)`. Recording tips: read the script naturally at your normal
pitch, use a quiet room, and avoid clipping; clean input matters more than
length. Cloning targets **English voices** (the speaker-similarity scorer is
English-trained).

One-time prerequisite: `assets/spk_encoder.onnx`, produced by `./setup.sh`
(or `python export/export.py --skip-model --skip-voices` on an existing
setup). espeak-ng is *not* needed at clone time — the test utterances ship
pre-phonemized.

---

## Repository layout

```
storytime/
├── export/                  # one-shot Python export (not used at runtime)
│   ├── export.py
│   ├── verify_spk.py        # speaker-encoder parity checks (voice cloning)
│   └── requirements.txt
├── cli/                     # Rust CLI (runtime)
│   ├── Cargo.toml
│   ├── build.rs             # fetches+builds mlx-c under --features mlx
│   └── src/
│       ├── main.rs
│       ├── clone.rs         # `storytime clone` voice-cloning subcommand
│       ├── dsp.rs           # audio analysis for cloning (spectrogram, YIN, ...)
│       ├── script.rs        # screenplay / multi-voice mode
│       └── mlx/             # MLX backend: native ONNX-graph interpreter
├── docs/                    # design + verification notes
├── assets/                  # produced by export.py (gitignored)
│   ├── kokoro.onnx
│   ├── tokens.json
│   ├── spk_encoder.onnx     # GE2E speaker encoder (voice cloning)
│   └── voices/*.bin
└── README.md
```

The `assets/` directory is gitignored — it's large (~325 MB model plus ~27 MB
of voice tensors) and reproducible from the upstream snapshot via `export.py`.

---

## Troubleshooting

**`espeak-ng: command not found`** — either install it (`brew install espeak-ng`)
or use `--ipa` and provide phonemes from another source.

**`could not locate assets/ directory`** — run `export.py` first, or pass
`--assets /path/to/assets`.

**`spk_encoder.onnx not found` (voice cloning)** — your `assets/` predates the
cloning feature. Re-run `./setup.sh`, or on an existing setup:
`cd export && source .venv/bin/activate && python export.py --skip-model --skip-voices`.

**`Context leak detected, msgtracer returned -1`** — cosmetic noise from
macOS's CoreML stack. Inference still runs correctly. Ignore.

**Long inputs** — the model's style tensor has a fixed maximum length
(510–511 phoneme tokens, depending on voice). The CLI handles this
automatically: long inputs are split at sentence boundaries (`.!?;…`),
then at word boundaries inside any sentence that's still too long, then
at character boundaries as a last resort. Each chunk is synthesized
independently and the results are concatenated with a ~150 ms silence gap.
Progress is printed per chunk on stderr.

**Pronunciation is wrong** — check that your `--voice` language matches the
input language, and that the IPA being fed to the model looks reasonable.
Run `espeak-ng -q --ipa=3 -v en-us "your text"` to inspect what the model
actually sees.

---

## License

This repository's code is available under the Apache 2.0 license.

The Kokoro-82M model weights are distributed by hexgrad under Apache 2.0.
`espeak-ng`, when used, is GPLv3; this tool invokes it as a separate process,
so there is no linking relationship between `storytime` and espeak-ng.
