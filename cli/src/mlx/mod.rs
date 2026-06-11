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
    r
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
            }
        };
        Ok(Self { graph })
    }

    /// Synthesize one chunk. `tokens` are phoneme token ids (no padding),
    /// `style` is the 256-dim voice row, `speed` the rate multiplier.
    pub fn synthesize(&self, tokens: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        // Activate per-call arenas so every array created while interpreting
        // this chunk is freed at the end (weights, created at load, are not
        // registered and persist). Without this the interpreter leaks all
        // intermediates and OOMs on multi-chunk inputs.
        unsafe {
            ARENA = Some(Vec::new());
            VARENA = Some(Vec::new());
        }
        let result = self.synthesize_inner(tokens, style, speed);
        unsafe {
            if let Some(arr) = ARENA.take() {
                let mut seen: HashSet<usize> = HashSet::with_capacity(arr.len());
                for a in arr {
                    if seen.insert(a.ctx as usize) {
                        mlx_array_free(a);
                    }
                }
            }
            if let Some(varr) = VARENA.take() {
                for v in varr {
                    mlx_vector_array_free(v);
                }
            }
        }
        result
    }

    fn synthesize_inner(&self, tokens: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(0);
        ids.extend_from_slice(tokens);
        ids.push(0);
        let input_ids = from_i64(&ids, &[1, ids.len() as i32]);
        let style_a = from_f32(style, &[1, style.len() as i32]);
        let speed_a = from_f32(&[speed], &[1]);

        let mut env: Env = HashMap::new();
        for (k, v) in &self.graph.weights {
            env.insert(k.clone(), Val::A(*v));
        }
        env.insert("input_ids".into(), Val::A(input_ids));
        env.insert("style".into(), Val::A(style_a));
        env.insert("speed".into(), Val::A(speed_a));

        for n in &self.graph.nodes {
            let outs = run_node(n, &mut env);
            for (k, v) in outs {
                env.insert(k, v);
            }
        }
        match env.get("audio") {
            // read_f32 copies to a host Vec, so the audio array can be freed
            // with the rest of the arena after this returns.
            Some(Val::A(a)) => Ok(read_f32(*a)),
            _ => bail!("MLX interpreter produced no audio output"),
        }
    }
}
