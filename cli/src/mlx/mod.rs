//! MLX backend: interprets `kokoro.onnx` directly on MLX (Metal GPU or CPU)
//! via the mlx-c API. Parses the ONNX graph natively (no derived artifacts),
//! folds constants, and runs each node as an mlx-c op. Verified numerically
//! equivalent to ONNX Runtime CPU (see docs/onnx-to-mlx-plan.md).
#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    static_mut_refs
)]

mod sys {
    include!(concat!(env!("OUT_DIR"), "/mlx_bindings.rs"));
}
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use sys::*;

// ---- dtype enum values (mlx/c/array.h) ----
const BOOL: u32 = mlx_dtype__MLX_BOOL;
const INT32: u32 = mlx_dtype__MLX_INT32;
const INT64: u32 = mlx_dtype__MLX_INT64;
const UINT32: u32 = mlx_dtype__MLX_UINT32;
const F32: u32 = mlx_dtype__MLX_FLOAT32;
const F16: u32 = mlx_dtype__MLX_FLOAT16;

static mut STREAM: mlx_stream = mlx_stream {
    ctx: std::ptr::null_mut(),
};
fn st() -> mlx_stream {
    unsafe { STREAM }
}
fn newarr() -> mlx_array {
    unsafe { mlx_array_new() }
}
fn rng_key(seed: u64) -> mlx_array {
    let mut r = newarr();
    unsafe { mlx_random_key(&mut r, seed) };
    track(r)
}

// Per-synthesize arenas: every array/vector created while interpreting a chunk
// is registered here and freed when the chunk completes. mlx_array is a Copy
// pointer struct with no Drop, so without this the interpreter leaks every
// intermediate. Weights are created at load time (arena inactive) so they are
// never registered and persist across chunks.
static mut ARENA: Option<Vec<mlx_array>> = None;
static mut VARENA: Option<Vec<mlx_vector_array>> = None;

fn track(a: mlx_array) -> mlx_array {
    unsafe {
        if let Some(ar) = ARENA.as_mut() {
            ar.push(a);
        }
    }
    a
}
fn track_vec(v: mlx_vector_array) -> mlx_vector_array {
    unsafe {
        if let Some(ar) = VARENA.as_mut() {
            ar.push(v);
        }
    }
    v
}

/// Generic op call: `$f(&mut res, args.., stream)` -> res.
macro_rules! op {
    ($f:ident $(, $a:expr )* ) => {{
        let mut r = newarr();
        let rc = unsafe { $f(&mut r as *mut _, $($a,)* st()) };
        assert_eq!(rc, 0, concat!("rc!=0 in ", stringify!($f)));
        track(r)
    }};
}

// ---- array helpers ----
fn ndim(a: mlx_array) -> usize {
    unsafe { mlx_array_ndim(a) }
}
fn shape(a: mlx_array) -> Vec<i32> {
    unsafe {
        let p = mlx_array_shape(a);
        (0..ndim(a)).map(|i| *p.add(i)).collect()
    }
}
fn dtype(a: mlx_array) -> u32 {
    unsafe { mlx_array_dtype(a) }
}
fn contig(a: mlx_array) -> mlx_array {
    let mut r = newarr();
    unsafe {
        mlx_contiguous(&mut r, a, false, st());
        mlx_array_eval(r);
    }
    track(r)
}
fn ctxof(a: mlx_array) -> usize {
    a.ctx as usize
}
fn read_f32(a: mlx_array) -> Vec<f32> {
    let a = if dtype(a) == F32 { a } else { op!(mlx_astype, a, F32) };
    let c = contig(a);
    let n = unsafe { mlx_array_size(c) };
    unsafe {
        let p = mlx_array_data_float32(c);
        (0..n).map(|i| *p.add(i)).collect()
    }
}
fn read_i64(a: mlx_array) -> Vec<i64> {
    let a = if dtype(a) == INT64 { a } else { op!(mlx_astype, a, INT64) };
    let c = contig(a);
    let n = unsafe { mlx_array_size(c) };
    unsafe {
        let p = mlx_array_data_int64(c);
        (0..n).map(|i| *p.add(i)).collect()
    }
}
fn from_f32(data: &[f32], shp: &[i32]) -> mlx_array {
    track(unsafe { mlx_array_new_data(data.as_ptr() as *const _, shp.as_ptr(), shp.len() as i32, F32) })
}
fn from_i64(data: &[i64], shp: &[i32]) -> mlx_array {
    track(unsafe { mlx_array_new_data(data.as_ptr() as *const _, shp.as_ptr(), shp.len() as i32, INT64) })
}
fn from_i32(data: &[i32], shp: &[i32]) -> mlx_array {
    track(unsafe { mlx_array_new_data(data.as_ptr() as *const _, shp.as_ptr(), shp.len() as i32, INT32) })
}
fn scalar_f32(v: f32) -> mlx_array {
    track(unsafe { mlx_array_new_data(&v as *const _ as *const _, std::ptr::null(), 0, F32) })
}
fn bool_scalar(b: bool) -> mlx_array {
    let v: u8 = b as u8;
    track(unsafe { mlx_array_new_data(&v as *const u8 as *const _, std::ptr::null(), 0, BOOL) })
}
fn transpose(a: mlx_array, perm: &[i32]) -> mlx_array {
    op!(mlx_transpose_axes, a, perm.as_ptr(), perm.len())
}
fn reshape(a: mlx_array, shp: &[i32]) -> mlx_array {
    op!(mlx_reshape, a, shp.as_ptr(), shp.len())
}
fn onnx_dtype(t: i64) -> u32 {
    match t {
        1 => F32,
        10 => F16,
        6 => INT32,
        7 => INT64,
        9 => BOOL,
        11 => F32, // double -> f32
        _ => panic!("unhandled onnx dtype {t}"),
    }
}

// ---- graph IR ----
enum Attr {
    I(i64),
    Ints(Vec<i64>),
    F(f64),
    S(String),
    T(mlx_array),
    G(Subgraph),
}
struct Node {
    op: String,
    name: String,
    input: Vec<String>,
    output: Vec<String>,
    attr: HashMap<String, Attr>,
}
struct Subgraph {
    input: Vec<String>,
    output: Vec<String>,
    nodes: Vec<Node>,
}
impl Node {
    fn ai(&self, k: &str, d: i64) -> i64 {
        match self.attr.get(k) {
            Some(Attr::I(n)) => *n,
            _ => d,
        }
    }
    fn aints(&self, k: &str) -> Vec<i64> {
        match self.attr.get(k) {
            Some(Attr::Ints(v)) => v.clone(),
            _ => Vec::new(),
        }
    }
    fn af(&self, k: &str, d: f64) -> f64 {
        match self.attr.get(k) {
            Some(Attr::F(n)) => *n,
            Some(Attr::I(n)) => *n as f64,
            _ => d,
        }
    }
    fn s(&self, k: &str) -> Option<String> {
        match self.attr.get(k) {
            Some(Attr::S(s)) => Some(s.clone()),
            _ => None,
        }
    }
    fn tensor(&self, k: &str) -> Option<mlx_array> {
        match self.attr.get(k) {
            Some(Attr::T(a)) => Some(*a),
            _ => None,
        }
    }
    fn subgraph(&self, k: &str) -> &Subgraph {
        match self.attr.get(k) {
            Some(Attr::G(g)) => g,
            _ => panic!("missing subgraph {k}"),
        }
    }
}

#[derive(Clone)]
enum Val {
    A(mlx_array),
    Seq(Vec<mlx_array>),
}
type Env = HashMap<String, Val>;

// ---- liveness (for per-chunk garbage collection) ----

// Every value name referenced anywhere inside a control-flow subgraph (at any
// nesting depth). Used so a value captured only by an If/Loop body is kept
// alive until the controlling node has run.
fn collect_subgraph_refs(sg: &Subgraph, set: &mut std::collections::HashSet<String>) {
    for n in &sg.nodes {
        for inp in &n.input {
            if !inp.is_empty() {
                set.insert(inp.clone());
            }
        }
        for a in n.attr.values() {
            if let Attr::G(inner) = a {
                collect_subgraph_refs(inner, set);
            }
        }
    }
}

// last_use[name] = index of the last top-level node that reads `name` (either
// directly or, for If/Loop nodes, anywhere in a nested subgraph). A value whose
// last use is node i can be freed once node i has executed. Names produced but
// never consumed (e.g. the graph output "audio") get no entry and are kept
// until the chunk-end sweep.
fn compute_last_use(nodes: &[Node]) -> HashMap<String, usize> {
    let mut lu = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for inp in &n.input {
            if !inp.is_empty() {
                lu.insert(inp.clone(), i);
            }
        }
        for a in n.attr.values() {
            if let Attr::G(sg) = a {
                let mut refs = std::collections::HashSet::new();
                collect_subgraph_refs(sg, &mut refs);
                for r in refs {
                    lu.insert(r, i);
                }
            }
        }
    }
    lu
}

fn ga(env: &Env, name: &str) -> mlx_array {
    match env.get(name) {
        Some(Val::A(a)) => *a,
        Some(Val::Seq(_)) => panic!("expected array, got sequence: {name}"),
        None => panic!("missing tensor: {name}"),
    }
}
fn gseq<'a>(env: &'a Env, name: &str) -> &'a Vec<mlx_array> {
    match env.get(name) {
        Some(Val::Seq(v)) => v,
        _ => panic!("expected sequence: {name}"),
    }
}
fn has(env: &Env, name: &str) -> bool {
    !name.is_empty() && env.contains_key(name)
}

include!("onnx.rs");
include!("ops.rs");

// ---- public API ----

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

/// True if a Metal GPU is available.
pub fn gpu_available() -> bool {
    let mut count = 0i32;
    unsafe { mlx_device_count(&mut count, mlx_device_type__MLX_GPU) };
    count > 0
}

pub struct MlxRuntime {
    graph: Graph,
    // Index of the last top-level node that reads each value (see compute_last_use).
    last_use: HashMap<String, usize>,
    // ctx pointers of the persistent weight arrays — these are never freed by
    // the per-chunk garbage collector.
    weight_ctx: HashSet<usize>,
}

impl MlxRuntime {
    /// Load and lower `kokoro.onnx`, materializing weights on a CPU stream.
    pub fn new(model_path: &Path, device: Device) -> Result<Self> {
        // Set the compute stream. Weights are always built/eval'd on CPU (the
        // safetensors/Load path has no GPU eval), then compute runs on `device`.
        unsafe { STREAM = mlx_default_cpu_stream_new() };
        let bytes = std::fs::read(model_path)
            .with_context(|| format!("reading {}", model_path.display()))?;
        let graph = Graph::load(&bytes)?;
        unsafe {
            STREAM = match device {
                Device::Gpu => mlx_default_gpu_stream_new(),
                Device::Cpu => mlx_default_cpu_stream_new(),
            };
            // Cap MLX's reuse cache so freed buffers are returned to the OS
            // within a chunk instead of accumulating to the chunk's high-water
            // mark. The live working set is well under this, so reuse stays hot.
            let mut prev = 0usize;
            mlx_set_cache_limit(&mut prev, 512 * 1024 * 1024);
        };
        let last_use = compute_last_use(&graph.nodes);
        let weight_ctx: HashSet<usize> = graph.weights.values().map(|a| ctxof(*a)).collect();
        Ok(Self {
            graph,
            last_use,
            weight_ctx,
        })
    }

    /// Synthesize one chunk. `tokens` are phoneme token ids (no padding),
    /// `style` is the 256-dim voice row, `speed` the rate multiplier.
    pub fn synthesize(&self, tokens: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        self.with_arena(|| {
            let mut ids = Vec::with_capacity(tokens.len() + 2);
            ids.push(0);
            ids.extend_from_slice(tokens);
            ids.push(0);
            let mut env = self.weights_env();
            env.insert("input_ids".into(), Val::A(from_i64(&ids, &[1, ids.len() as i32])));
            env.insert("style".into(), Val::A(from_f32(style, &[1, style.len() as i32])));
            env.insert("speed".into(), Val::A(from_f32(&[speed], &[1])));
            let (audio, _shape) = self.eval_graph(env, "audio")?;
            Ok(audio)
        })
    }

    /// Run the loaded graph generically: bind each `(name, row-major data,
    /// shape)` input and return the named `output`'s host data + shape. Lets a
    /// second model — the GE2E speaker encoder used by `storytime clone` — reuse
    /// the exact same MLX interpreter and Metal stream as Kokoro synthesis,
    /// keeping the whole cloning loop GPU-resident.
    pub fn run(
        &self,
        inputs: &[(&str, &[f32], &[i32])],
        output: &str,
    ) -> Result<(Vec<f32>, Vec<i32>)> {
        self.with_arena(|| {
            let mut env = self.weights_env();
            for (name, data, shp) in inputs {
                env.insert((*name).to_string(), Val::A(from_f32(data, shp)));
            }
            self.eval_graph(env, output)
        })
    }

    /// A fresh env seeded with the persistent weight arrays.
    fn weights_env(&self) -> Env {
        let mut env: Env = HashMap::new();
        for (k, v) in &self.graph.weights {
            env.insert(k.clone(), Val::A(*v));
        }
        env
    }

    /// Activate the per-node temporary arena (track/track_vec register here),
    /// run `f`, then tear the arena down and return idle GPU buffers to the OS.
    /// Calls must not nest — the arena is process-global, and the clone loop
    /// synthesizes then embeds strictly in turn, never concurrently.
    fn with_arena<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        unsafe {
            ARENA = Some(Vec::new());
            VARENA = Some(Vec::new());
        }
        let result = f();
        unsafe {
            ARENA = None;
            VARENA = None;
            // Return cached (idle) buffers to the OS so peak footprint reflects
            // one call's working set, not the high-water mark of every call.
            mlx_clear_cache();
        }
        result
    }

    /// Interpret the graph to completion: eval_graph clears the temp arena after
    /// every node (freeing that node's intermediates) and frees long-lived values
    /// via liveness analysis, so only the live working set is resident — not the
    /// whole forward pass. Reads `output` to a host Vec and frees the rest.
    fn eval_graph(&self, mut env: Env, output: &str) -> Result<(Vec<f32>, Vec<i32>)> {
        let wctx = &self.weight_ctx;
        // Force evaluation every `eval_every` nodes (see below). Larger values
        // submit more GPU work per flush (fewer sync round-trips) at the cost of
        // a larger live working set; override via env for a memory/speed trade.
        let eval_every: usize = std::env::var("STORYTIME_MLX_EVAL_EVERY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        for (i, n) in self.graph.nodes.iter().enumerate() {
            // Start a fresh temp arena for this node. (Anything created before
            // the loop, e.g. input_ids, is untracked here but lives in env and
            // is reclaimed by the liveness sweep below.)
            unsafe {
                ARENA.as_mut().unwrap().clear();
                VARENA.as_mut().unwrap().clear();
            }

            let outs = run_node(n, &mut env);

            // Collect this node's output arrays, then force evaluation every
            // `eval_every` nodes. Evaluating collapses the lazy graph so freed
            // intermediates are actually released (otherwise the whole forward
            // pass stays one graph that pins everything until the final read).
            // We only ever evaluate freshly-produced outputs (never a handle a
            // later sweep may free), so this is safe at any cadence.
            let mut out_ctx: HashSet<usize> = HashSet::new();
            let mut out_arrays: Vec<mlx_array> = Vec::new();
            for (_, v) in &outs {
                match v {
                    Val::A(a) => {
                        out_ctx.insert(ctxof(*a));
                        out_arrays.push(*a);
                    }
                    Val::Seq(s) => {
                        for a in s {
                            out_ctx.insert(ctxof(*a));
                            out_arrays.push(*a);
                        }
                    }
                }
            }
            if i % eval_every == 0 && !out_arrays.is_empty() {
                unsafe {
                    // Async eval: schedule the batch on the GPU and keep building
                    // the next nodes on the CPU instead of blocking. Host reads
                    // (read_f32/contig) and the final audio read sync as needed.
                    let vec = mlx_vector_array_new_data(out_arrays.as_ptr(), out_arrays.len());
                    mlx_async_eval(vec);
                    mlx_vector_array_free(vec);
                }
            }

            // Free this node's internal temporaries: everything created while
            // running it except the arrays it returns (and weights, which can
            // flow through unchanged via e.g. Identity).
            unsafe {
                let arr = ARENA.as_mut().unwrap();
                let mut seen: HashSet<usize> = HashSet::with_capacity(arr.len());
                for &a in arr.iter() {
                    let c = ctxof(a);
                    if out_ctx.contains(&c) || wctx.contains(&c) {
                        continue;
                    }
                    if seen.insert(c) {
                        mlx_array_free(a);
                    }
                }
                arr.clear();
                let varr = VARENA.as_mut().unwrap();
                for &v in varr.iter() {
                    mlx_vector_array_free(v);
                }
                varr.clear();
            }

            for (k, v) in outs {
                env.insert(k, v);
            }

            // Liveness sweep: free any env value whose last use was this node.
            // Candidates are exactly the names this node read (directly or via a
            // subgraph) — only those can have last_use == i.
            let mut dead_names: Vec<String> = Vec::new();
            for inp in &n.input {
                if self.last_use.get(inp) == Some(&i) {
                    dead_names.push(inp.clone());
                }
            }
            for a in n.attr.values() {
                if let Attr::G(sg) = a {
                    let mut refs = HashSet::new();
                    collect_subgraph_refs(sg, &mut refs);
                    for r in refs {
                        if self.last_use.get(&r) == Some(&i) {
                            dead_names.push(r);
                        }
                    }
                }
            }
            let mut dead: Vec<mlx_array> = Vec::new();
            for name in &dead_names {
                if let Some(v) = env.remove(name) {
                    match v {
                        Val::A(a) => dead.push(a),
                        Val::Seq(s) => dead.extend(s),
                    }
                }
            }
            free_dead(&dead, &env, wctx);
        }

        let out_arr = match env.get(output) {
            Some(Val::A(a)) => *a,
            _ => bail!("MLX interpreter produced no output '{output}'"),
        };
        let out_shape = shape(out_arr);
        let out = read_f32(out_arr); // copies to a host Vec

        // Final cleanup: free the read temporaries and every remaining non-weight
        // value (the output buffer and any produced-but-unconsumed outputs).
        unsafe {
            let mut seen: HashSet<usize> = HashSet::new();
            for &a in ARENA.as_mut().unwrap().iter() {
                let c = ctxof(a);
                if !wctx.contains(&c) && seen.insert(c) {
                    mlx_array_free(a);
                }
            }
            ARENA.as_mut().unwrap().clear();
            for &v in VARENA.as_mut().unwrap().iter() {
                mlx_vector_array_free(v);
            }
            VARENA.as_mut().unwrap().clear();
            for v in env.values() {
                match v {
                    Val::A(a) => {
                        let c = ctxof(*a);
                        if !wctx.contains(&c) && seen.insert(c) {
                            mlx_array_free(*a);
                        }
                    }
                    Val::Seq(s) => {
                        for a in s {
                            let c = ctxof(*a);
                            if !wctx.contains(&c) && seen.insert(c) {
                                mlx_array_free(*a);
                            }
                        }
                    }
                }
            }
        }
        Ok((out, out_shape))
    }
}

// Free arrays in `dead` that are not weights and are not still referenced by any
// remaining value in `env`. Deduped by ctx so aliased handles (e.g. an Identity
// output sharing a ctx with its input) are freed at most once and never while a
// live name still points at them.
fn free_dead(dead: &[mlx_array], env: &Env, wctx: &HashSet<usize>) {
    if dead.is_empty() {
        return;
    }
    let mut live: HashSet<usize> = HashSet::new();
    for v in env.values() {
        match v {
            Val::A(a) => {
                live.insert(ctxof(*a));
            }
            Val::Seq(s) => {
                for a in s {
                    live.insert(ctxof(*a));
                }
            }
        }
    }
    let mut freed: HashSet<usize> = HashSet::new();
    for &a in dead {
        let c = ctxof(a);
        if wctx.contains(&c) || live.contains(&c) {
            continue;
        }
        if freed.insert(c) {
            unsafe { mlx_array_free(a) };
        }
    }
}
