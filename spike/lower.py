#!/usr/bin/env python3
"""Phase-1: lower kokoro.onnx into a Rust-interpretable IR + safetensors.

Emits into spike/art/:
  graph.json            topo-ordered node IR (Constants folded out; Loop/If
                        subgraphs nested), plus input/output specs.
  weights.safetensors   all initializers + folded Constant tensors, by name.
  ref.safetensors       every ndarray intermediate from an ONNX Runtime CPU
                        run, keyed by original tensor name, for node-by-node
                        parity checks in the interpreter.
"""
import json, os
import numpy as np
import onnx
from onnx import numpy_helper as nh
import onnxruntime as ort
from safetensors.numpy import save_file

ART = "spike/art"
os.makedirs(ART, exist_ok=True)

m = onnx.load("assets/kokoro.onnx")

# ---- fixed input (matches spike/ref_dump.py) ----
vocab = json.load(open("assets/tokens.json"))
vocab = vocab.get("vocab", vocab)
ids = [vocab[c] for c in "həlˈoʊ" if c in vocab]
input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
n_tokens = len(ids)
voice = np.fromfile("assets/voices/af_heart.bin", dtype="<f4").reshape(510, 256)
style = voice[min(n_tokens, 509)][None, :].astype(np.float32)
speed = np.array([1.0], dtype=np.float32)

weights = {}  # name -> np.ndarray  (initializers + folded constants)
for t in m.graph.initializer:
    weights[t.name] = nh.to_array(t)


def attr_to_py(a):
    """Serialize a non-graph attribute to a JSON-friendly value."""
    T = onnx.AttributeProto
    if a.type == T.INT:    return ("i", a.i)
    if a.type == T.INTS:   return ("ints", list(a.ints))
    if a.type == T.FLOAT:  return ("f", a.f)
    if a.type == T.FLOATS: return ("floats", list(a.floats))
    if a.type == T.STRING: return ("s", a.s.decode())
    if a.type == T.STRINGS:return ("strings", [s.decode() for s in a.strings])
    if a.type == T.TENSOR: return ("t", None)  # handled by caller (folded into weights)
    return ("unknown", str(a.type))


def fold_constant(node):
    """Return (name, ndarray) for a Constant node, or None."""
    for a in node.attribute:
        if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
            return node.output[0], nh.to_array(a.t)
        if a.name == "value_float":  return node.output[0], np.array(a.f, np.float32)
        if a.name == "value_int":    return node.output[0], np.array(a.i, np.int64)
        if a.name == "value_floats": return node.output[0], np.array(list(a.floats), np.float32)
        if a.name == "value_ints":   return node.output[0], np.array(list(a.ints), np.int64)
    return None


def emit_graph(g, prefix=""):
    """Recursively turn a GraphProto into node IR, folding Constants into weights."""
    nodes = []
    for n in g.node:
        if n.op_type == "Constant":
            r = fold_constant(n)
            if r is not None:
                weights[r[0]] = r[1]
                continue
        node = {
            "op": n.op_type,
            "name": n.name,
            "input": list(n.input),
            "output": list(n.output),
            "attr": {},
        }
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                node["attr"][a.name] = {
                    "_subgraph": True,
                    "input": [vi.name for vi in a.g.input],
                    "output": [vi.name for vi in a.g.output],
                    "nodes": emit_graph(a.g, prefix + n.name + "/"),
                }
            elif a.type == onnx.AttributeProto.TENSOR:
                key = prefix + n.name + "/attr/" + a.name
                weights[key] = nh.to_array(a.t)
                node["attr"][a.name] = ["t", key]
            else:
                node["attr"][a.name] = list(attr_to_py(a))
        nodes.append(node)
    return nodes


ir = {
    "inputs": [vi.name for vi in m.graph.input],
    "outputs": [vi.name for vi in m.graph.output],
    "nodes": emit_graph(m.graph),
}
json.dump(ir, open(f"{ART}/graph.json", "w"))
print(f"graph.json: {len(ir['nodes'])} top-level nodes (Constants folded)")

# ---- weights.safetensors ----
# Make contiguous WITHOUT bumping rank: np.ascontiguousarray forces ndim>=1,
# which would silently turn rank-0 scalar constants (e.g. Gather indices) into
# shape (1,) and corrupt ONNX rank propagation downstream. Preserve shape.
def cont(a):
    a = np.asarray(a)
    return np.ascontiguousarray(a).reshape(a.shape)

wsave = {k: cont(v) for k, v in weights.items()}
save_file(wsave, f"{ART}/weights.safetensors")
print(f"weights.safetensors: {len(wsave)} tensors")

# ---- reference run: expose every node output, save intermediates ----
mm = onnx.load("assets/kokoro.onnx")
existing = {o.name for o in mm.graph.output}
for node in mm.graph.node:
    for o in node.output:
        if o and o not in existing:
            mm.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(mm.SerializeToString(), so, providers=["CPUExecutionProvider"])
want = [o.name for o in sess.get_outputs()]
res = sess.run(want, {"input_ids": input_ids, "style": style, "speed": speed})
ref = {}
for name, val in zip(want, res):
    if isinstance(val, np.ndarray) and val.dtype in (np.float32, np.int64, np.float64):
        ref[name] = cont(val.astype(np.float32) if val.dtype == np.float64 else val)
ref["__input_ids"] = input_ids
ref["__style"] = style
ref["__speed"] = speed
save_file(ref, f"{ART}/ref.safetensors")
audio = ref.get("audio")
print(f"ref.safetensors: {len(ref)} tensors | audio rms={np.sqrt((audio**2).mean()):.5f}")
