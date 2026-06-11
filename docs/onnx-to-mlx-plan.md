# Plan: replace the Swift MLX backend with an ONNX→MLX graph interpreter

Status: **approved, Phase-0 spike in progress** · Last updated: 2026-06-10

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

## Phased plan

- **Phase 0 — spike / go-no-go (in progress).** bindgen over mlx-c; build mlx-c +
  core; load initializers as `mlx_array`; interpret a linear sub-slice of the
  graph (text encoder: `Gemm`/`Conv`/`LayerNormalization`/`LSTM`) and **diff
  against onnxruntime CPU** on the same `input_ids`. Kill-switch: if `LSTM`/`Conv`
  parity is shaky here, stop. First concrete sub-step: pin the mlx-c/mlx build
  (CMake vs prebuilt) and a minimal end-to-end "add two arrays from Rust" link.
- **Phase 1 — op kernels.** Implement the ~15 non-trivial kernels, each unit-
  tested against numpy/onnxruntime reference on random inputs.
- **Phase 2 — graph interpreter.** Topological executor + the bounded `Loop`/`If`/
  Sequence support; wire initializers + `input_ids`/`style`/`speed`.
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
