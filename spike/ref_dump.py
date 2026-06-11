#!/usr/bin/env python3
"""Phase-0: dump ONNX Runtime CPU reference tensors for parity checks.

Runs kokoro.onnx with every top-level node output exposed, so the MLX
interpreter spike can diff each intermediate and find the first divergence.
"""
import json, sys
import numpy as np
import onnx
import onnxruntime as ort

MODEL = "assets/kokoro.onnx"
OUT = "spike/ref_tensors.npz"

# 1. Fixed input.
vocab = json.load(open("assets/tokens.json"))
vocab = vocab.get("vocab", vocab)
ipa = "həlˈoʊ"  # həlˈoʊ
ids = [vocab[c] for c in ipa if c in vocab]
input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
n_tokens = len(ids)
voice = np.fromfile("assets/voices/af_heart.bin", dtype="<f4").reshape(510, 256)
style = voice[min(n_tokens, 509)][None, :].astype(np.float32)
speed = np.array([1.0], dtype=np.float32)
print(f"input_ids {input_ids.shape} {input_ids.tolist()}  style {style.shape}  n_tokens {n_tokens}")

# 2. Expose every top-level node output as a graph output.
m = onnx.load(MODEL)
existing = {o.name for o in m.graph.output}
extra = []
for node in m.graph.node:
    for o in node.output:
        if o and o not in existing:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
            extra.append(o)
print(f"exposed {len(extra)} extra intermediate outputs")

# 3. Run on CPU.
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep node names/shapes
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
want = [o.name for o in sess.get_outputs()]
res = sess.run(want, {"input_ids": input_ids, "style": style, "speed": speed})

# 4. Save tensors that are arrays (skip sequence/optional types).
saved = {}
for name, val in zip(want, res):
    if isinstance(val, np.ndarray):
        saved[name.replace("/", "__")] = val
saved["__input_ids"] = input_ids
saved["__style"] = style
saved["__speed"] = speed
np.savez(OUT, **saved)
audio = saved.get("audio")
print(f"saved {len(saved)} tensors to {OUT}")
if audio is not None:
    print(f"audio: shape={audio.shape} rms={np.sqrt((audio**2).mean()):.5f} "
          f"min={audio.min():.4f} max={audio.max():.4f}")
