# Plan: replace the Swift MLX backend with an ONNX→MLX graph interpreter

Status: **Phase-2 Metal GPU backend working; CPU≡GPU verified** · Last updated: 2026-06-10

## TL;DR

storytime currently carries **two** definitions of the Kokoro model:

1. `assets/kokoro.onnx` — the exported compute graph, run via ONNX Runtime + the
   CoreML execution provider. Correct, but in practice runs **on CPU** on Apple
   Silicon (CoreML EP can't take this graph's ops onto ANE/GPU, so it fragments
   and falls back).
2. `mlx-backend/` — a **hand-written re-implementation** of the same model in
   Swift on top of MLXNN (~1130 LOC of model code + the MLXNN dependency +
   ~90 lines of fragile Swift-runtime linking in `cli/build.rs`), so that the
   model can run on the Metal GPU.

Maintaining two definitions of the same model is the problem. The plan is to
**delete the second definition** and instead **interpret the ONNX graph directly
against MLX core** via [mlx-c](https://github.com/ml-explore/mlx-c), loading the
weights that already live in the `.onnx`. The `.onnx` becomes the single source
of truth; MLXNN and the Swift port are no longer needed.

This works because **the ONNX graph is already lowered below the layer
abstraction** — there is no `Linear`/`Conv1d`/`LSTM` *layer* to re-implement,
only primitive ops (`Gemm`, `MatMul`, `Conv`, `InstanceNormalization`, …). We
write an **op-level interpreter** (~55 op kernels, most trivial), not a model.

## Why MLX (and why CoreML fails)

The graph has **dynamic sequence length** (`input_ids[1, tokens]` →
`audio[samples]`), an **`LSTM`**, **control flow** (`Loop`/`If`), and **Sequence
(list-of-tensor)** values. CoreML's EP supports a limited static op set and bails
to CPU on the rest; once the graph fragments, the whole thing is effectively CPU.
MLX runs everything on the Metal GPU in eager mode with native dynamic shapes —
which is exactly the capability gap that motivated the Swift backend in the first
place. We keep that capability but drop the redundant model definition.

## Homework: what's actually in `kokoro.onnx`

Inspected with `onnx` (in `export/.venv`). See "Reproduce the analysis" below.

### Shape
- `ir_version 8`, single `ai.onnx` **opset 17**, **no custom domains**.
- **4765 nodes**, **60 distinct op types**, **5 subgraphs** (control-flow bodies).
- Inputs: `input_ids[1, tokens]`, `style[1, 256]`, `speed[1]`. Output: `audio[samples]`.

### Weights
- **443 initializers, 81,148,592 params, all fp32, all internal** (no external
  data). The `.onnx` is the single weight source — 81M × 4 B = ~325 MB. The
  per-voice `style[1,256]` tensor is a runtime input (today's `voices/*.bin`),
  not a graph weight.

### Op inventory (full, by frequency)
```
1403 Constant      466 Add        435 Mul        289 Shape       237 Gather
196 Identity       182 Slice      169 Unsqueeze  154 Reshape     107 Transpose
102 MatMul          92 Concat      92 Div         90 Conv         79 Cast
 73 Gemm            70 InstanceNormalization      62 Pow          51 Sin
 48 Reciprocal      46 ConstantOfShape            39 Where        37 Sqrt
 32 Expand          31 LayerNormalization         29 Equal        28 LeakyRelu
 22 Range           13 Tanh        12 Softmax      7 ScatterND     7 ConvTranspose
  6 LSTM             6 Resize       5 Sub          5 Squeeze       4 TopK
  4 ReduceProd       4 ScatterElements             3 Greater       2 SplitToSequence
  2 SequenceAt       2 If           2 Pad          2 Less          1 ReduceMax
  1 Not              1 Sigmoid      1 ReduceSum     1 Round         1 Clip
  1 SequenceEmpty    1 Loop         1 SequenceInsert 1 ConcatFromSequence
  1 Floor            1 RandomUniformLike            1 CumSum        1 RandomNormalLike
  1 And              1 Atan         1 Exp           1 Cos
```

Notes:
- **No `STFT`/`DFT`/`MelSpectrogram` op.** The ISTFTNet vocoder is decomposed into
  `Conv`/`MatMul`/`Sin`/`Cos`; noise excitation is `RandomNormalLike`/
  `RandomUniformLike`. We don't even need `MLXFFT`.

### Control flow (the part that decides "interpreter vs flat translator")
Turned out **bounded and simple**:
- **1 `Loop`** — body is 18 nodes (`SequenceAt`/`Expand`/`Where`/`Reshape`/
  `SequenceInsert`/…). This is the **duration-based length regulator**: trip
  count = token count (bounded, known at runtime), expands each token by its
  predicted duration into a sequence accumulator, then `ConcatFromSequence`. No
  nesting, no unbounded autoregression.
- **2 `If`** — both degenerate: `{Constant, Squeeze}` vs `{Identity}` (a
  conditional squeeze of a size-1 dim).
- **Sequence ops** (`SplitToSequence`/`SequenceEmpty`/`SequenceAt`/
  `SequenceInsert`/`ConcatFromSequence`) implement the list-of-tensors
  accumulator around the loop.

⇒ The interpreter needs only: bounded-loop execution, a list-of-tensors value
type, and a trivial conditional. A few hundred lines, **not** a general ONNX
control-flow engine.

## mlx-c coverage verification

Checked the mlx-c headers (`mlx/c/ops.h`, `fast.h`, `random.h`) — repo at MLX
**0.31.2** bindings. Every non-trivial op maps:

| ONNX op | mlx-c | Notes |
|---|---|---|
| `Conv` | `mlx_conv1d` / `mlx_conv_general` | direct |
| `ConvTranspose` | `mlx_conv_transpose1d` / `mlx_conv_general` | `conv_general` exposes `stride`, asymmetric `padding_lo`/`padding_hi`, `kernel_dilation`, **`input_dilation`** → covers every ONNX Conv/ConvTranspose attr combo incl. groups |
| `InstanceNormalization` | `mlx_mean`+`mlx_var`+`mlx_rsqrt` | trivial compose (no fused op) |
| `LayerNormalization` | `mlx_fast_layer_norm` | **fused**; nullable weight/bias + eps matches ONNX |
| `LSTM` | `mlx_addmm`/`matmul`+`sigmoid`+`tanh`+`split`/`slice`/`concatenate` | hand-written recurrence |
| `ScatterND` | `mlx_scatter` | |
| `ScatterElements` | `mlx_put_along_axis` | |
| `TopK` | `mlx_topk` | direct |
| `CumSum` | `mlx_cumsum` | |
| `Where` | `mlx_where` | |
| `RandomNormalLike` / `RandomUniformLike` | `mlx_random_normal` / `mlx_random_uniform` | |
| `Expand` | `mlx_broadcast_to` | |
| `Pad` | `mlx_pad` | |
| `Gather` | `mlx_take` / `mlx_take_axis` | (really trivial tier) |
| **`Resize`** | **only gap — decomposes** | see below |

### The `Resize` gap collapses
No native interpolation op in mlx-c (it's an MLXNN `Upsample` layer, not a core
op). The 6 actual nodes are benign:
- **4 nodes**: `nearest`/`asymmetric`/`floor`, integer scale ×2 or ×300 on the
  last axis → **`mlx_repeat`**, a one-liner.
- **2 nodes**: `linear`/`half_pixel`, but **constant** scale (1/300 and 300).
  Constant scale ⇒ precompute a fixed interpolation matrix once and apply as
  **`mlx_matmul`**.

So even the gap is a small, fully-static kernel — no dynamic interpolation engine.

### Caveat
Header inspection proves the ops **exist**, not that semantics are **identical**
(LSTM gate ordering, scatter index conventions, conv pad ordering, `half_pixel`
math). Those are pinned down by per-op parity tests in Phase 0, where existence
becomes proven-correctness.

## Translation effort: three tiers

| Tier | Share | Work |
|---|---|---|
| Trivial 1:1 | ~85% of nodes | `Add`, `Mul`, `MatMul`, `Gemm`, `Transpose`, `Reshape`, `Concat`, `Slice`, `Gather`, `Cast`, `Sqrt`, `Pow`, `Sin`, `Where`, `Range`, `Softmax`, `Tanh`, `LeakyRelu`, … → one mlx-c call each |
| Real kernel (~15 ops) | — | `Conv`/`ConvTranspose`, `InstanceNormalization`, `LayerNormalization`, `Resize`, `LSTM`, scatter/`TopK`/`CumSum` — composed; attribute mapping is where care goes |
| Control flow (~5 ops) | — | `Loop` (bounded), `If` (trivial), Sequence value type — small interpreter |

## Target architecture

- **Rust + mlx-c**, via `bindgen`. The interpreter lives in the Rust CLI; **no
  Swift, no C++ written by us, no MLXNN**. `@_cdecl` and the Swift-runtime
  linking in `build.rs` are deleted.
- Parse the `.onnx` protobuf (Rust: `prost`/`protobuf`, or shell the existing
  Python once to pre-bake initializers — TBD in Phase 0).
- Materialize the 443 initializers as `mlx_array`s once at init.
- Topologically interpret nodes → `mlx_array` ops; handle the loop/if/sequence
  values explicitly.
- Reuse the existing Rust `tokenize()` for `input_ids` (don't re-derive the vocab).
- `build.rs` collapses to: build/link mlx-c + the mlx core static lib + Metal
  frameworks. (mlx-c links its own matching mlx core, so the `vendor/mlx-swift`
  submodule version stops mattering and can eventually be removed.)

### Why this beats the alternatives
- vs **Swift status quo**: single source of truth (`.onnx`), no model drift, no
  MLXNN, no Swift toolchain in the build, generalizes to future Kokoro re-exports.
- vs **C++ bridge** (earlier idea): that was blocked by "MLX has no C++ nn
  library, so reimplement MLXNN." The interpreter **dissolves that blocker** —
  no nn library needed in any language, because the graph is pre-lowered.

## Phase-0 spike results (PASSED)

Run on an Apple M2 Max. Spike lives in `spike/` (gitignored); reproduce as below.

**Outcome: GO.** Both highest-risk kernels reproduce ONNX Runtime CPU to float32
epsilon, and Rust links mlx-c cleanly.

| Check | Result |
|---|---|
| Build mlx-c + mlx core (CPU-only, `MLX_BUILD_METAL=OFF`) | ✅ `libmlxc.a` + `libmlx.a`, example runs |
| ONNX Runtime CPU reference: 4756 intermediate tensors dumped for a fixed input | ✅ `spike/ref_tensors.npz` (audio rms 0.050) |
| **Conv1d** parity (`/F0.0/conv1/Conv`, NCL↔NLC + weight `[Cout,Cin,K]→[Cout,K,Cin]`) | ✅ **rel 6.5e-7** |
| **LSTM** parity (`/text_encoder/lstms.0`, bidirectional, hidden=256, `iofc` gate order) | ✅ **rel 2.7e-7** |
| Rust → mlx-c link + compute (`mlx_add`, hand-declared externs) | ✅ verified |

### Gotchas discovered (carry into Phase 1–2)
- **Lazy transposes are strided views.** `mlx_array_data_float32` exposes the
  underlying (pre-transpose) buffer, so a transposed result reads back wrong.
  Call `mlx_contiguous(..., allow_col_major=false)` before reading raw data (or
  before any step that assumes row-major memory). This was the entire initial
  Conv mismatch.
- **`mlx_dtype` enum: `MLX_FLOAT32 = 10`** (9 is `MLX_FLOAT16`). bindgen will get
  this right; don't hand-hardcode.
- **`mlx_array`/`mlx_stream` are single-`void*` structs** passed/returned by
  value — maps to a `#[repr(C)]` one-pointer struct in Rust; the AArch64 ABI
  handles it.
- **Conv weight layout**: ONNX `[Cout,Cin,K]` → MLX `[Cout,K,Cin]` (transpose
  axes `0,2,1`); input ONNX `[N,Cin,L]` → MLX `[N,L,Cin]`. MLX conv is
  cross-correlation (matches ONNX), no kernel flip.
- **LSTM**: ONNX gate order is **`iofc`**; per-direction `bias = Wb + Rb`;
  backward direction processes timesteps reversed; `Y` is `[seq, num_dir, batch,
  hidden]`. Confirmed exact.
- **GPU build needs the Metal toolchain**: `xcodebuild -downloadComponent
  MetalToolchain` (separate, on recent Xcode). CPU-only build skips it and is
  sufficient for parity work; the production backend needs the GPU build.

### Reproduce the spike
```sh
# build mlx-c CPU-only
cp -R <mlx-c clone> spike/mlx-c && cd spike/mlx-c
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_METAL=OFF
cmake --build build -j
# reference + per-op tensors
export/.venv/bin/pip install onnxruntime
export/.venv/bin/python spike/ref_dump.py          # -> spike/ref_tensors.npz
# (extraction of conv/lstm tensors -> spike/td/*.bin; see git history of this work)
# C parity harness:
clang -O2 -Ispike/mlx-c spike/parity.c \
  spike/mlx-c/build/libmlxc.a spike/mlx-c/build/_deps/mlx-build/libmlx.a \
  -framework Accelerate -framework Foundation -lc++ -o spike/parity && ./spike/parity
# Rust link test:
cd spike/rust-link && cargo run --release
```

## Phase-1 results (interpreter complete, CPU)

A full ONNX→MLX graph interpreter now runs the **entire** `kokoro.onnx` on MLX
CPU and produces audio. Lives in `spike/` (source tracked; build artifacts and
generated safetensors gitignored).

**Pipeline:**
- `spike/lower.py` — one ONNX Runtime CPU pass: folds Constants, emits
  `graph.json` (topo node IR incl. Loop/If subgraphs), `weights.safetensors`
  (1892 tensors, scalars kept rank-0), and `ref.safetensors` (4694 intermediates
  for node-by-node parity).
- `spike/interp/` — Rust + bindgen over mlx-c. Loads the IR + weights, runs each
  node on mlx-c, validates **every** float32 *and* int output against the ONNX
  reference, stops/logs on divergence. All 56 op types implemented (elementwise,
  shape/structural, Gemm/Conv/ConvTranspose/Norms/LSTM, Gather/Scatter/TopK,
  Resize, Pad, and the `SplitToSequence`/`Loop`/`If`/`ConcatFromSequence` length
  regulator).

**Parity status:** the entire deterministic network — embeddings, the PL-BERT
encoder, both bidirectional LSTMs, the prosody/duration predictor, F0/N
predictors, the length-regulator Loop, and the decoder up to the vocoder — matches
ONNX Runtime CPU to **< 5e-4 relative**. Final audio is **rel 1.9e-2**.

The residual is isolated to the **harmonic-source oscillator + iSTFT phase** in
the vocoder: `sin()` of a few-hundred-radian accumulated phase amplifies ~6e-6
relative phase error to ~3.5e-3, and the `atan2` phase reconstruction (expressed
as `Div`→`Atan`→`Where`) further amplifies it where the STFT real part is near
zero. This is **inherent f32 non-associativity between two independent
implementations, not a bug** — it cannot be removed without bit-matching ONNX's
exact summation/rounding through the oscillator. Perceptually 1.9e-2 (≈ −34 dB)
is indistinguishable.

**Proof of isolation:** with the oscillator (`m_source`) outputs injected from
the reference (`INJECT=m_source`), the final audio matches ONNX Runtime CPU to
**rel 7.2e-6** with **zero** diverging intermediates. So every one of the 56 ops
and the entire rest of the vocoder (STFT, `atan2` phase, upsampling, noise convs,
iSTFT) is numerically exact to float32 epsilon; the 1.9e-2 is *entirely* the
oscillator's conditioning.

**Bugs found and fixed along the way (each caught by node-level parity):**
- `np.ascontiguousarray` forces ndim≥1 → silently turned rank-0 scalar constants
  (Gather indices) into `(1,)`, corrupting ONNX rank propagation. Fixed in lower.py.
- mlx-c safetensors **iterator** value handle aliases across iterations → fetch
  by name with `get` instead.
- `mlx_softmax` reduces the whole array; need `mlx_softmax_axis`.
- `mlx_concatenate` has no axis; need `mlx_concatenate_axis`.
- ONNX `Div` on **integer** inputs truncates toward zero; `mlx_divide` is float
  (this was the seed of nearly all upstream drift — fixing it dropped the whole
  network to <5e-4).
- ONNX `Gather` = `np.take(axis)`; implemented on host (mlx take/take_axis
  semantics didn't match for multi-dim data).
- Negative-step `Slice` (reverse): ONNX's "past the beginning" end sentinel
  doesn't map to mlx slice stop → host slice for negative steps.
- `ConvTranspose` grouped weight layout: ONNX `[Cin, Cout/g, K]` → MLX
  `[Cout, K, Cin/g]` needs a grouped reshape+permute, not a plain transpose.
- `mlx_pad` supports only constant/edge → host implementation for `reflect`.
- RNG nodes (`RandomUniformLike`/`RandomNormalLike`) are unseeded/nondeterministic
  → inject the captured reference outputs for parity (production uses mlx_random).

**Still to do for a production CPU backend:** native ONNX parsing in Rust (drop
the Python lower step), array lifetime management (the spike leaks), use
mlx_scatter/device ops instead of host fallbacks (Gather/Scatter/Slice/Pad/TopK),
and wire it into the CLI behind `--backend mlx`.

## Phase-2 results (Metal GPU)

The interpreter targets a `mlx_stream`, so GPU was a small change: `--gpu`
selects `mlx_default_gpu_stream_new()`, and `--compare` runs the whole graph on
both devices (with identical injected noise) and diffs every node.

**Build:** Xcode's `xcodebuild -downloadComponent MetalToolchain` initially
failed (broken `DVTDownloads`/`IDESimulatorFoundation` plugin from a version
skew); `xcodebuild -runFirstLaunch` repaired it, then the toolchain downloaded
and mlx-c rebuilt with `MLX_BUILD_METAL=ON`. The interpreter's `build.rs` links
Metal / MetalPerformanceShaders / MetalPerformanceShadersGraph / QuartzCore.
One runtime fix: the safetensors `Load` op has no GPU eval, so weights are loaded
and `mlx_array_eval`'d on a CPU stream before any GPU compute uses them.

**Numeric equivalence — CPU ≡ GPU (Apple M2 Max):**
- Deterministic graph (oscillator injected on both devices): audio **rel 5.4e-6**,
  worst node 3.2e-5, **0 nodes >1e-3** — every op runs identically on Metal and CPU.
- Full pipeline (oscillator computed): audio rel **1.7e-2** — same magnitude and
  same root cause as CPU-vs-ONNX (CPU and GPU `sin`/`atan2` transcendentals differ
  slightly; the `atan2` iSTFT phase amplifies it). Worst node is again the
  `m_source` atan2 `Div`. Not a GPU bug.
- GPU-synthesized audio is statistically identical to CPU (rms 0.0527, peak 0.325).

**Performance:** GPU synth of a 27-token / 2.12s clip ran in **~1.3s wall vs ~22s
on CPU (~17×)** — and that's *with* the spike's host-fallback ops (Gather/Scatter/
Slice/Pad/LSTM/TopK read back to host every call, forcing CPU↔GPU syncs). Moving
those to device ops would widen the gap further.

## Phased plan

- **Phase 0 — spike / go-no-go. ✅ DONE — PASSED** (see "Phase-0 spike results").
  Built mlx-c CPU-only, dumped the ONNX CPU reference, and proved Conv1d + the
  bidirectional LSTM match to float32 epsilon, plus Rust↔mlx-c linkage. The
  kill-switch (shaky LSTM/Conv parity) did not trigger.
- **Phase 1 — op kernels + interpreter. ✅ DONE** (see "Phase-1 results"). Full
  graph runs on MLX CPU; deterministic network matches ONNX to <5e-4, audio 1.9e-2
  (residual = inherent oscillator/iSTFT f32 conditioning).
- **Phase 2 — graph interpreter. ✅ folded into Phase 1.** Topological executor +
  bounded `Loop`/`If`/Sequence support implemented and verified.
- **Phase 3 — integrate + strip Swift.** Replace `synthesize_mlx`/`mlx_ffi` to
  call the interpreter; rewrite `build.rs`; keep `mlx-backend/` behind the flag
  until parity passes.
- **Phase 4 — parity gate.** RMS/byte-diff vs current Swift output **and** vs ONNX
  across a fixed phoneme/voice matrix. On pass, delete `mlx-backend/` and the
  Swift linking.

**Effort estimate:** Phases 1–2 dominate; ~1.5–3 weeks, most of it parity-chasing
on Conv/ConvTranspose, LSTM, and the `Resize` linear kernel.

## Open questions / risks
- ONNX parsing in Rust (`prost` vs pre-baking initializers via Python) — decide in Phase 0.
- mlx-c (0.31.2) vs `vendor/mlx-swift` (0.31.4) version reconciliation — pin mlx-c to its matching core; submodule becomes irrelevant.
- Exact semantic parity on LSTM gate order, scatter conventions, `half_pixel` Resize — covered by per-op tests.
- mlx-c build packaging (static lib vendoring vs CMake fetch in `build.rs`).

## Reproduce the analysis
```sh
# Op inventory, weights, control-flow bodies:
export/.venv/bin/python  # load assets/kokoro.onnx with onnx, walk graph + subgraphs
# mlx-c coverage:
git clone --depth 1 https://github.com/ml-explore/mlx-c   # grep mlx/c/{ops,fast,random}.h
```
