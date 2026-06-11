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
- **Adjustable speaking rate** via `--speed`.

---

## Inference backends

`storytime` supports two backends, selectable via `--backend`:

| backend | flag | acceleration | weights format | notes |
|---|---|---|---|---|
| **ONNX** (default) | `--backend onnx` | CoreML EP (ANE/GPU/CPU) | `kokoro.onnx` (exported via `export.py`) | Stable, tested |
| **MLX** | `--backend mlx` | Metal GPU (Apple Silicon) | safetensors (from [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)) | Requires `--features mlx` at build time |

### MLX backend setup

```sh
# 1. Build with MLX support (requires Xcode + macOS 14+)
cd cli
MACOSX_DEPLOYMENT_TARGET=14.0 cargo build --release --features mlx

# 2. Download MLX-format weights
# Place config.json + *.safetensors + voices/*.safetensors in assets/mlx/
# or pass --mlx-weights /path/to/weights

# 3. Run
echo "Hello." | storytime --backend mlx --mlx-weights /path/to/Kokoro-82M-bf16
```

The MLX backend uses Apple's [MLX framework](https://github.com/ml-explore/mlx)
(vendored as a git submodule at `vendor/mlx-swift`) for Metal GPU-accelerated
inference. The Kokoro model architecture is ported to Swift using mlx-swift's
`MLXNN` module, and exposed to Rust via `@_cdecl` C-callable bridge functions.

Without `--features mlx`, `--backend mlx` prints an error and the binary
has no MLX dependency at all.

## Architecture

```
                   ┌──────────────────── one-shot (Python) ────────────────────┐
                   │                                                            │
   kokoro-v1_0.pth │   export.py   │→  kokoro.onnx                              │
   voices/*.pt    ─┼─▶  (kokoro +  │→  voices/*.bin  (float32, [N,1,256])      │
   config.json     │    PyTorch)   │→  tokens.json   (IPA char → token-id)     │
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

- macOS on Apple Silicon (tested on macOS 15, arm64). CoreML EP is used; an
  Intel Mac would fall back to CPU only.
- Rust toolchain (stable, ≥ 1.85).
- Python 3.10+ with `pip` and `venv`.
- The Kokoro-82M snapshot already downloaded locally. The export script
  defaults to:
  ```
  /Users/kuy/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/
    snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987
  ```
  Pass `--snapshot` to override.

### 1. One-time: export the model

```sh
cd export
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export.py
```

This writes:

```
assets/
├── kokoro.onnx      # ~325 MB, the full model
├── tokens.json      # IPA char → token-id vocab
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

### Flags

| flag | default | description |
|---|---|---|
| `-i, --input PATH` | *(stdin)* | read input from file; `-` or omitted means stdin |
| `--voice NAME` | `af_heart` | voice name from `--list-voices` |
| `--sample-rate HZ` | `24000` | output sample rate; model native is 24 kHz |
| `--bit-depth {16,24,32,float32}` | `16` | PCM bit depth |
| `--speed FLOAT` | `1.0` | speaking rate multiplier |
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

## Repository layout

```
storytime/
├── export/                  # one-shot Python export (not used at runtime)
│   ├── export.py
│   └── requirements.txt
├── cli/                     # Rust CLI (runtime)
│   ├── Cargo.toml
│   └── src/main.rs
├── assets/                  # produced by export.py (gitignored)
│   ├── kokoro.onnx
│   ├── tokens.json
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

This repository's code is available under the MIT license.

The Kokoro-82M model weights are distributed by hexgrad under Apache 2.0.
`espeak-ng`, when used, is GPLv3; this tool invokes it as a separate process,
so there is no linking relationship between `storytime` and espeak-ng.
