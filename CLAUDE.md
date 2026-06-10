# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, offline CLI (`storytime`) that synthesizes speech from text or IPA using the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model. It reads UTF-8 from stdin/file and
writes a WAV (or plays directly via OS-native audio). Python is **not** a runtime dependency — it
only runs once, offline, to export the model. See `README.md` for the full user-facing manual; it is
unusually detailed and is the source of truth for flag semantics and prosody behavior.

## Commands

All Rust work happens in `cli/` (it is its own crate, **not** a workspace — there is no top-level
`Cargo.toml`).

```sh
cd cli
cargo build --release            # binary: cli/target/release/storytime
cargo test                       # unit tests live at the bottom of src/main.rs
cargo test parse_emphasis        # run a single test by name substring
cargo clippy

# MLX backend (optional, Apple Silicon + Xcode + macOS 14+):
MACOSX_DEPLOYMENT_TARGET=14.0 cargo build --release --features mlx
```

One-time model export (regenerates the gitignored `assets/`):

```sh
cd export
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export.py                 # writes assets/{kokoro.onnx, tokens.json, voices/*.bin}
```

Quick manual run:

```sh
echo "Hello, world." | cargo run --release -- --voice af_bella -o /tmp/hello.wav
echo "həlˈoʊ." | cargo run --release -- --ipa -o /tmp/h.wav   # skips espeak-ng
```

`espeak-ng` must be on PATH for text mode (`brew install espeak-ng`); `--ipa` mode does not need it.

## Pipeline architecture

Nearly all runtime logic is in the single file `cli/src/main.rs` (~2100 lines). The flow in `main()`
is a linear text→audio pipeline; understanding the *order* of stages is the key to working here,
because each stage assumes the previous one's invariants:

1. **`preprocess_markdown`** — input is Markdown by default. Block/inline markers are stripped so
   they are never spoken. Emphasis (`*`/`_`) can't be applied here because stress lives in *phoneme*
   space (downstream of espeak), so emphasized spans are wrapped in **private-use sentinel chars**
   (`SENT_*`, U+E000–E003) that ride untouched through later text stages. Skipped under `--ipa` or
   `--no-markdown`.
2. **`parse_structure`** — splits text into `Block`s separated by typed `Boundary`s (paragraph /
   section / chapter / quote), detected from blank lines and `#`/`##` headings. Boundaries become
   either inline textual pause markers (merged into one inference call — faster) or explicit silence
   spliced in post-synthesis.
3. **`normalize_punctuation`** — collapses `...`→`…` and pairs straight quotes into curly open/close,
   because Kokoro has distinct learned tokens for those.
4. **`run_espeak`** — the trickiest stage. espeak-ng's `--ipa=3` *strips* punctuation, so this splits
   each block on `PRESERVED_PUNCT` boundaries, feeds only text segments to one espeak invocation, and
   **interleaves the punctuation back** into the IPA. It also **consumes the emphasis sentinels** here,
   converting them to IPA stress via `apply_emphasis_to_ipa`/`emphasize_word` (lengthen the
   primary-stressed vowel with `ː`). Sentinels never reach espeak or the model.
5. **`chunk_ipa`** — the model's style tensor caps at ~510 tokens. Long IPA is split at sentence
   (`.!?;…`) → word → char boundaries (`split_long_sentence`/`hard_split`) so no chunk exceeds the cap.
6. **`tokenize`** → **synthesis** (`synthesize_chunk` for ONNX, `synthesize_mlx` for MLX) → per-chunk
   `trim_silence` + `apply_fade` + typed silence gap → concat → `resample` → `write_wav`/playback.

When changing any stage, keep these contracts intact: sentinels must survive stages 2–3; punctuation
in `PRESERVED_PUNCT` must round-trip through stage 4; nothing downstream of stage 4 should see raw
graphemes.

## Backends and conditional compilation

Two inference backends selected by `--backend`:

- **ONNX (default)** — `ort` crate with the CoreML execution provider (ANE/GPU, CPU fallback). The
  compiled CoreML model is cached under `~/Library/Caches/storytime/coreml`. `ort` is configured with
  `download-binaries`, so no system ONNX Runtime install is needed.
- **MLX (`--features mlx`)** — Metal GPU via Apple's MLX. The Kokoro model is **ported to Swift** in
  `mlx-backend/Sources/KokoroMLX/` (`Model.swift`, `Modules.swift`, `ISTFTNet.swift`) and exposed to
  Rust through `@_cdecl` C functions (`Bridge.swift`, header `include/kokoro_mlx.h`). `cli/build.rs`
  shells out to `swift build`, links the resulting static lib plus the Swift runtime / Metal /
  Accelerate frameworks. mlx-swift is a git submodule at `vendor/mlx-swift` (`git submodule update
  --init`).

`main.rs` uses `cfg`-gated duplicate modules — `mod mlx_ffi` (real FFI under `feature = "mlx"`, else a
stub that errors) and three `mod playback` blocks (macOS AudioToolbox / Linux ALSA / unsupported
fallback), all pure FFI with no third-party audio crates. When editing one variant, update the others
so every `cfg` configuration still compiles.

## Conventions

- **Voice tensors** are raw little-endian f32 of shape `[N,1,256]` where N is 510 or 511; the loader
  handles either. Voice names follow `{lang}{gender}_{name}` (`af_*` American female, `bm_*` British
  male, etc.).
- **Model is native 24 kHz**; any other `--sample-rate` is reached via the `rubato` sinc resampler.
- `assets/` and `*.wav` are gitignored and large; never commit them. Stray `bedtime-story-*.wav/.txt`
  and `demo/*.txt` files in the repo root are scratch output, not source.
- Unit tests are colocated in the `#[cfg(test)] mod tests` block at the end of `main.rs`.
