# Voice cloning: `storytime clone`

Design doc for cloning a speaker from a short reference recording into a new Kokoro voicepack.
Status: implemented (June 2026) — `storytime clone`, see README "Voice cloning" for usage.

## Goal

Record a clean 10–20 s WAV of a speaker reading a known reference paragraph, then have
`storytime` synthesize stories in (an approximation of) that voice — fully offline, with zero new
runtime cost. The output of cloning is an ordinary `assets/voices/<name>.bin` voicepack, so after
the one-time clone step, `--voice <name>` works everywhere a stock voice does.

## Why this approach (investigation summary)

**The "real" path is blocked by design.** Kokoro-82M is a StyleTTS2-family decoder. Its 256-d
style vectors split as dims 0–127 = acoustic/timbre style (fed to the decoder) and dims 128–255 =
prosody style (fed to the duration/prosody predictor). In StyleTTS2 those come from reference
encoders run over example audio — but hexgrad deliberately withheld Kokoro's encoder ("Decoder
only: no diffusion, no encoder release"), and has said it generalizes poorly out-of-distribution
anyway. The released StyleTTS2 LibriTTS encoder is from a different training run; its style space
is incompatible with Kokoro's. Nobody has published a recovered or retrained Kokoro-compatible
style encoder.

**kokoclone was investigated and rejected.** Despite the name,
[kokoclone](https://github.com/Ashish-Patnaik/kokoclone) never produces a Kokoro style vector. It
synthesizes with a hardcoded *stock* voice, then runs a separate voice-conversion stage over the
output audio: [kanade-tokenizer](https://github.com/frothywater/kanade-tokenizer) (frozen
WavLM-base+ → content tokens + a 128-d speaker embedding from the reference wav, mel decoder
conditioned via AdaLN-Zero) followed by a Vocos vocoder. That adds ~230 M params / ~850 MB of
models running on **every** synthesis call, is PyTorch-only (no ONNX exports exist), and the VC
model is trained on English-only LibriTTS. Porting it would rival the MLX backend in scope and
break this project's single-model offline design. Kanade's 128-d embedding has no relationship to
Kokoro's 256-d style space.

**The approach that fits: style-space optimization.**
[KVoiceWalk](https://github.com/RobViren/kvoicewalk) (Apache-2.0) demonstrates gradient-free
hill-climbing of the style tensor itself: start from a blend of stock voices, perturb, synthesize
test utterances, score against the reference recording, keep improvements. Reported results:
Resemblyzer speaker similarity 71 % (best stock voice) → 93 % after ~10 k steps. The artifact is a
genuine voicepack tensor — byte-compatible with our existing loader.

**Expectations.** Kokoro (82 M params, a few hundred training hours) cannot reach XTTS/F5-class
cloning fidelity; even hexgrad's own in-training reconstruction from 3 minutes of audio was "a
crude reconstruction". The result is "recognizably you", stochastic across runs, and costs roughly
1–2 h of one-time compute on Apple Silicon.

## Architecture

```
storytime clone --ref me.wav --name kuy [--steps N] [--budget-min M] [--init v1,v2] [--seed S] [--resume]

ref.wav ─► mono / resample 16 kHz ─► reference embedding (assets/spk_encoder.onnx via ort, CPU)
                                 └─► reference acoustic stats (F0 / RMS / spectral)

init = blend of stock voicepacks (auto-ranked by embedding similarity to the reference, or --init)
loop (greedy hill-climb over a single 256-d delta shared across all rows):
    delta' = best_delta + N(0,1) ⊙ per-dim-std(stock voices) ⊙ U(0.01, 0.15)
    synth target utterance (baked IPA — the same text the user read) via the existing backend
    early-exit if target-similarity ≤ 0.98 × current best        # skips the second synthesis
    synth other utterance
    score = weighted harmonic mean(target_sim 0.48, self_sim 0.50, feat_sim 0.02) × 100
    accept iff score > best; checkpoint best .bin periodically (audition mid-run with --voice)

write assets/voices/<name>.bin:  out[r][c] = clamp(base[r][c] + delta[c])   # exact load_voice format
```

### Scoring

The weighted **harmonic** mean is deliberate (KVoiceWalk's validated formula): it allows one
component to backslide if the combined score improves, while punishing any component near zero.

- **target_similarity** — cosine between the speaker embedding of the candidate's synthesis of the
  *reference-script text* and the embedding of the reference recording. Text-matched comparison:
  same words, different voice.
- **self_similarity** — cosine between the candidate's embeddings of two *different* texts.
  A stability term: without it the walk converges to audio that scores well but sounds (per the
  KVoiceWalk author) "like a metal basket of tools being thrown down stairs".
- **feature_similarity** — NaN-safe relative differences over a small acoustic feature set
  (F0 mean/std + voiced ratio, RMS mean/std, spectral centroid mean/std, rolloff(0.85), ZCR).
  Weight is only 0.02 — it exists to keep the walk in-bounds, not to drive it.

### Speaker embedding

Resemblyzer's GE2E encoder (~1.4 M params, 256-d L2-normalized output, exactly what KVoiceWalk
scores with). Exported once, offline, by `export/export.py` to `assets/spk_encoder.onnx` (~5.5 MB).

It runs on **whichever backend the clone is using**: with `--backend mlx` the encoder runs through
the same native ONNX→MLX interpreter as Kokoro synthesis (`crate::mlx::MlxRuntime`), so the entire
optimization loop is GPU-resident with no per-step CPU bounce; otherwise it runs on ONNX Runtime
(CPU). The interpreter needed only two new ops for this graph — `Relu` and `ReduceL2` — on top of
the set Kokoro already exercises (`LSTM`, `MatMul`, `Gemm`, …). The two paths are verified to agree
to cosine > 0.999 (`spk_embedding_mlx_matches_ort`), so the cloning score does not depend on which
backend produced the embedding. `MlxRuntime::run(inputs, output)` is the generic graph-execution
entry point added so a second model can reuse the interpreter; `synthesize()` is now a thin
Kokoro-specific wrapper over it.

**Parity trick:** the ONNX graph's input is the 201-bin *power spectrogram* (n_fft = 400,
hop = 160, hann, centered/reflect-padded, 16 kHz) and the librosa Slaney 40-mel filterbank is
baked into the graph as a constant. Rust then only has to reproduce a windowed rFFT power
spectrum — trivially fixture-testable — while the fussy mel construction stays in Python where
librosa generates it. Remaining Resemblyzer semantics reimplemented in Rust: −30 dBFS
increase-only loudness normalization, partial windows of 160 mel frames every 77 frames (drop the
last if < 75 % covered), mean of partial embeddings, L2-normalize. We skip Resemblyzer's webrtcvad
silence trim and use the pipeline's existing `trim_silence` on both sides (small accepted
deviation, covered by the parity gate below).

### Reference script UX

The user records themselves reading a **canonical reference paragraph shipped with storytime**
(`storytime clone --print-script` prints it; ~2 phonetically rich sentences, 100–200 tokens). Its
IPA is baked in as a constant, plus a second baked "other text" for self-similarity. Consequences:

- espeak-ng is **not** required at clone time (IPA was generated once at development time);
- the target embedding comparison is text-matched (KVoiceWalk semantics);
- token counts are fixed → only ~2 input shapes ever hit the backend (warm ONNX/CoreML compile
  cache from iteration 1; on MLX the recurring shapes likewise keep the buffer cache hot).

`--ref-text <file>` allows a custom transcript (this path does run espeak-ng).

## Notes from the KVoiceWalk port (deliberate deviations)

1. **Shared per-row delta instead of full-tensor walk.** Upstream perturbs the entire
   `[510,1,256]` tensor each step, but with fixed test texts only 1–2 rows are ever *evaluated*
   (Kokoro selects the style row by token count). The other ~508 rows random-walk with zero
   selection pressure and degrade. We optimize one 256-d delta applied on top of a fixed base
   blend — preserving the per-length row structure exactly and shrinking the search space 510×.
2. **NaN-safe feature diffs.** Upstream computes `abs((v - t)/t)` against target features that are
   legitimately zero (beat/pitch stats), yielding inf/NaN penalties. We use
   `abs(v - t)/max(|t|, ε)` and drop the fragile features (tempo/beat/tonnetz/chroma) entirely.
3. **Seedable RNG** (SplitMix64 + Box-Muller; upstream is unseeded by design). Caveat: CoreML/Metal
   inference is not bit-deterministic, so accepted paths can still diverge; `--backend onnx` on
   CPU gives best-effort strict reproducibility.
4. **Exposed constants.** Upstream hardcodes the 0.98 early-exit gate and the U(0.01, 0.15)
   diversity range; we keep the same defaults but name them at the top of `clone.rs`.
5. **Guard rails:** per-dimension `[min, max]` envelope across all installed voicepacks (clamp),
   plus a duration-sanity guard rejecting candidates whose audio length deviates > 30 % from the
   base voice's for the same tokens.

## File layout

| file | role |
|---|---|
| `cli/src/main.rs` | `Cli { Option<Cmd>, flatten Args }` wrapper (legacy flag-only invocation unchanged); `Runtime { init, synth }` wraps the ONNX/MLX backend so clone reuses the exact synthesis path |
| `cli/src/clone.rs` | `CloneArgs`, walk loop, scoring, init blend / auto-rank, incremental `.bin.temp` checkpoint + `--resume` + `finalize`, `write_voice_bin`/`atomic_write`, baked IPA constants |
| `cli/src/main.rs` (loader) | `resolve_voice_path` — `--voice <name>` falls back to an in-progress `<name>.bin.temp`; `--list-voices` flags `(training)` |
| `cli/src/dsp.rs` | WAV reading (hound `WavReader`), resampling, power spectrogram (realfft), YIN F0, acoustic stats, `SpeakerEncoder` (ort-CPU or MLX-GPU per backend), SplitMix64 RNG |
| `cli/src/mlx/{mod,ops}.rs` | `MlxRuntime::run(inputs, output)` generic graph entry point (the encoder reuses it); `Relu` + `ReduceL2` ops added for the GE2E graph |
| `export/export.py` | `+ GE2EWrapper` export → `assets/spk_encoder.onnx` (mel filterbank baked in); weights from the `resemblyzer` pip package or a pinned HF mirror; `--skip-spk` flag |
| `export/verify_spk.py` | parity script: Resemblyzer vs exported ONNX cosine (expect > 0.999); `--dump-fixture` emits the Rust spectrogram test fixture |
| `cli/Cargo.toml` | `+ realfft` (the only new crate; no `rand`, no pitch crate) |

### Incremental / resumable training (browser-download model)

Training never touches the final `<name>.bin` until it is finished. While it runs, two paired temp
files live in `voices/`:

- `<name>.bin.temp` — the in-progress voicepack, raw f32 in exactly `load_voice` format, so it is
  directly usable for **preview** while training continues;
- `<name>.bin.temp.json` — the resume state (init blend, delta, step, seed, accepted, best score).

Both are rewritten **atomically** (write-sibling-then-rename, via `atomic_write`) every
`CHECKPOINT_EVERY_STEPS` (50) **or** `CHECKPOINT_EVERY_SECS` (60) — whichever trips first, plus once
before the first step and once at exit. The time trigger matters because a CPU run is slow enough
that a step-only cadence could miss a short session; with it, a 5-minute run always leaves a
recent, resumable checkpoint regardless of backend speed. On completion, `finalize()` renames
`<name>.bin.temp` → `<name>.bin` and deletes the sidecar; a non-completing stop (`--budget-min`,
Ctrl-C, kill) leaves the temps in place.

**Ctrl-C / SIGTERM** is handled by the shared `interrupt` module (`cli/src/interrupt.rs`).
`clone::run` calls `interrupt::install_graceful()` **after** `Runtime::init` (so a backend that
installs its own SIGINT handler during init can't clobber ours); the first interrupt flips an atomic
flag the walk polls — finishing the current step, writing a checkpoint, and exiting 0 via the same
"paused" path as `--budget-min` — and a second hard-exits (130). Installing the handler also
*reclaims* SIGINT when the process was launched with it inherited-ignored (`SIG_IGN`), and the
module `unblock`s SIGINT/SIGTERM from the signal mask in case a library blocked them — these two
launch conditions are the usual reason "Ctrl-C does nothing." Everything the handler does is
async-signal-safe (an atomic store / `_exit`); raw libc FFI, no new crate. (One-shot synthesis in
`main.rs` uses the same module's `install_abort`, which exits immediately on any interrupt.)

`--resume` reads the sidecar and continues toward the original `--steps` (seed re-derived as
`seed + step` so it doesn't replay the same perturbations). Preview resolution: a bare `--voice
<name>` loads `<name>.bin` if it exists, else falls back to `<name>.bin.temp` (with a note);
`--list-voices` flags in-progress clones as `<name> (training)`. A second `storytime` process
previewing the partial voice is safe against the trainer's concurrent writes because every update
is an atomic rename.

## Verification

1. **Unit tests** — spectrogram vs a librosa-generated fixture; YIN on synthetic sines (± 2 Hz);
   seeded RNG stability; `write_voice_bin` ↔ `load_voice` round-trip; blend math; legacy CLI parse
   still resolves to no-subcommand; checkpoint → loadable `.bin.temp` → `finalize` promotion; the
   `resolve_voice_path` in-progress fallback.
2. **Embedding parity gates** (run before trusting any walk):
   - Rust ONNX path vs Python: `python export/verify_spk.py ref.wav` → Resemblyzer-vs-ONNX
     cosine > 0.999; then `cargo test spk_embedding_matches_python_frontend -- --ignored` for the
     Rust frontend vs Python (> 0.999).
   - MLX vs ONNX: `cargo test --features mlx spk_embedding_mlx_matches_ort -- --ignored` (> 0.999),
     so the MLX-GPU encoder scores identically to the ONNX one.
3. **Integration smoke test** (skips politely when gitignored assets are absent): synthesize a fake
   "reference" from `af_bella`, run 20 clone steps initialized from `am_adam`, assert a finite
   weakly-improving score and a loadable `.bin`.
4. **Self-clone sanity**: clone toward a stock voice's own output — should converge fast and sound
   near-identical.
5. **End-to-end**:
   ```sh
   ffmpeg -i raw.m4a -ar 24000 -ac 1 ref.wav        # 15 s natural read of the reference script
   storytime clone --ref ref.wav --name kuy --budget-min 45
   echo "Once upon a time..." | storytime --voice kuy -o story.wav
   ```
6. **Pause / resume / preview** (verified manually): start a clone, `kill` it mid-walk → only
   `<name>.bin.temp{,.json}` remain (no final `.bin`); preview with `--voice <name>` (loads the
   temp); `--resume` continues from the saved step and, on reaching `--steps`, promotes to
   `<name>.bin` and removes the temps.

## Known limitations

- English-focused: GE2E is English-trained; init ranking defaults to American/British voices.
- Quality is stochastic; re-running with a new `--seed` is cheap and sometimes wins big.
- Not in scope (possible follow-ups): runtime voice blending (`--voice a:0.6,b:0.4`), and a
  Kanade-style VC stage as a fidelity escape hatch if optimization disappoints.
