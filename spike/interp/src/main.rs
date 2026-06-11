#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]
//! ONNX -> MLX (CPU) graph interpreter. Loads the lowered graph.json + weights,
//! runs each node on mlx-c, and validates every float32 output against the ONNX
//! Runtime CPU reference (ref.safetensors). Stops at the first real divergence.
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use serde::Deserialize;
use std::collections::HashMap;
use std::os::raw::c_char;
use sys::*;

// dtype enum values (mlx/c/array.h)
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

// Generic op call: $f(&mut res, args.., stream) ; returns res.
macro_rules! op {
    ($f:ident $(, $a:expr )* ) => {{
        let mut r = newarr();
        let rc = unsafe { $f(&mut r as *mut _, $($a,)* st()) };
        assert_eq!(rc, 0, concat!("rc!=0 in ", stringify!($f)));
        r
    }};
}

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
    unsafe {
        mlx_array_new_data(
            data.as_ptr() as *const _,
            shp.as_ptr(),
            shp.len() as i32,
            F32,
        )
    }
}
fn from_i64(data: &[i64], shp: &[i32]) -> mlx_array {
    unsafe {
        mlx_array_new_data(
            data.as_ptr() as *const _,
            shp.as_ptr(),
            shp.len() as i32,
            INT64,
        )
    }
}
fn from_i32(data: &[i32], shp: &[i32]) -> mlx_array {
    unsafe {
        mlx_array_new_data(
            data.as_ptr() as *const _,
            shp.as_ptr(),
            shp.len() as i32,
            INT32,
        )
    }
}
fn scalar_f32(v: f32) -> mlx_array {
    unsafe { mlx_array_new_data(&v as *const _ as *const _, std::ptr::null(), 0, F32) }
}
fn transpose(a: mlx_array, perm: &[i32]) -> mlx_array {
    op!(mlx_transpose_axes, a, perm.as_ptr(), perm.len())
}
fn reshape(a: mlx_array, shp: &[i32]) -> mlx_array {
    op!(mlx_reshape, a, shp.as_ptr(), shp.len())
}

// ---- ONNX dtype (TensorProto) -> mlx dtype ----
fn onnx_dtype(t: i64) -> u32 {
    match t {
        1 => F32,
        10 => F16,
        6 => INT32,
        7 => INT64,
        9 => BOOL,
        11 => F32, // double -> f32 (spike)
        _ => panic!("unhandled onnx dtype {t}"),
    }
}

// ---- graph IR ----
#[derive(Deserialize)]
struct Graph {
    inputs: Vec<String>,
    outputs: Vec<String>,
    nodes: Vec<Node>,
}
#[derive(Deserialize)]
struct Node {
    op: String,
    name: String,
    input: Vec<String>,
    output: Vec<String>,
    attr: HashMap<String, serde_json::Value>,
}
impl Node {
    fn ai(&self, k: &str, d: i64) -> i64 {
        self.attr.get(k).map(|v| v[1].as_i64().unwrap()).unwrap_or(d)
    }
    fn aints(&self, k: &str) -> Vec<i64> {
        self.attr
            .get(k)
            .map(|v| {
                v[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            })
            .unwrap_or_default()
    }
    fn af(&self, k: &str, d: f64) -> f64 {
        self.attr.get(k).map(|v| v[1].as_f64().unwrap()).unwrap_or(d)
    }
    fn at(&self, k: &str) -> Option<String> {
        self.attr.get(k).map(|v| v[1].as_str().unwrap().to_string())
    }
}

#[derive(Deserialize)]
struct Subgraph {
    input: Vec<String>,
    output: Vec<String>,
    nodes: Vec<Node>,
}

#[derive(Clone)]
enum Val {
    A(mlx_array),
    Seq(Vec<mlx_array>),
}
type Env = HashMap<String, Val>;

fn gseq<'a>(env: &'a Env, name: &str) -> &'a Vec<mlx_array> {
    match env.get(name) {
        Some(Val::Seq(v)) => v,
        _ => panic!("expected sequence: {name}"),
    }
}
fn bool_scalar(b: bool) -> mlx_array {
    let v: u8 = b as u8;
    unsafe { mlx_array_new_data(&v as *const u8 as *const _, std::ptr::null(), 0, BOOL) }
}

fn ga(env: &Env, name: &str) -> mlx_array {
    match env.get(name) {
        Some(Val::A(a)) => *a,
        Some(Val::Seq(_)) => panic!("expected array, got sequence: {name}"),
        None => panic!("missing tensor: {name}"),
    }
}
fn has(env: &Env, name: &str) -> bool {
    !name.is_empty() && env.contains_key(name)
}

fn load_safetensors(path: &str) -> HashMap<String, mlx_array> {
    let mut map = unsafe { mlx_map_string_to_array_new() };
    let mut meta = unsafe { mlx_map_string_to_string_new() };
    let cpath = std::ffi::CString::new(path).unwrap();
    let rc = unsafe { mlx_load_safetensors(&mut map, &mut meta, cpath.as_ptr(), st()) };
    assert_eq!(rc, 0, "load_safetensors {path}");
    // Collect keys via the iterator, then fetch each value by name with `get`
    // (the iterator's value handle aliases across iterations).
    let mut keys: Vec<String> = Vec::new();
    unsafe {
        let it = mlx_map_string_to_array_iterator_new(map);
        let mut key: *const c_char = std::ptr::null();
        let mut val = newarr();
        while mlx_map_string_to_array_iterator_next(&mut key, &mut val, it) == 0 {
            keys.push(std::ffi::CStr::from_ptr(key).to_str().unwrap().to_string());
        }
        mlx_map_string_to_array_iterator_free(it);
    }
    let mut out = HashMap::new();
    for k in keys {
        let ck = std::ffi::CString::new(k.as_str()).unwrap();
        let mut val = newarr();
        let rc = unsafe { mlx_map_string_to_array_get(&mut val, map, ck.as_ptr()) };
        assert_eq!(rc, 0, "map get {k}");
        out.insert(k, val);
    }
    out
}

fn main() {
    unsafe { STREAM = mlx_default_cpu_stream_new() };
    let g: Graph = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open("spike/art/graph.json").unwrap(),
    ))
    .unwrap();
    let weights = load_safetensors("spike/art/weights.safetensors");
    let refmap = load_safetensors("spike/art/ref.safetensors");

    println!("loaded {} weights, {} ref tensors", weights.len(), refmap.len());
    for k in [
        "kmodel.decoder.generator.noise_res.1.adain1.0.norm.bias",
        "onnx::LSTM_7356",
        "onnx::Conv_7580",
    ] {
        match weights.get(k) {
            Some(a) => println!("DBG {k} shape={:?} dtype={}", shape(*a), dtype(*a)),
            None => println!("DBG {k} MISSING"),
        }
    }
    let mut env: Env = HashMap::new();
    for (k, v) in &weights {
        env.insert(k.clone(), Val::A(*v));
    }
    // graph inputs from the reference run
    env.insert("input_ids".into(), Val::A(refmap["__input_ids"]));
    env.insert("style".into(), Val::A(refmap["__style"]));
    env.insert("speed".into(), Val::A(refmap["__speed"]));

    let mut worst = 0f64;
    let mut diverged = 0u32;
    let verbose = std::env::var("V").is_ok();
    let inject = std::env::var("INJECT").ok();
    for (ni, n) in g.nodes.iter().enumerate() {
        if verbose {
            let ins: Vec<Vec<i32>> = n
                .input
                .iter()
                .map(|i| match env.get(i) {
                    Some(Val::A(a)) => shape(*a),
                    Some(Val::Seq(_)) => vec![-9],
                    None => vec![-1],
                })
                .collect();
            eprintln!("[{ni}] {} '{}' ins={:?}", n.op, n.name, ins);
        }
        let outs = run_node(n, &mut env, &refmap);
        for (name, a) in outs {
            // Diagnostic: override matching float outputs with the ref value to
            // isolate where the audio error originates (INJECT=<substr>).
            let a = match (&inject, &a) {
                (Some(sub), Val::A(arr))
                    if dtype(*arr) == F32 && name.contains(sub.as_str()) && refmap.contains_key(&name) =>
                {
                    Val::A(refmap[&name])
                }
                _ => a,
            };
            env.insert(name.clone(), a.clone());
            if let Val::A(arr) = a {
                if let Some(r) = refmap.get(&name) {
                    let dt = dtype(arr);
                    if dt == F32 {
                        let rel = compare(arr, *r);
                        if rel > worst {
                            worst = rel;
                        }
                        if rel > 1e-2 {
                            diverged += 1;
                            if diverged <= 12 {
                                println!(
                                    "[{ni}] {} '{}' float rel={:.3e} shape={:?}",
                                    n.op, name, rel, shape(arr)
                                );
                            }
                        }
                    } else if dt == INT64 || dt == INT32 || dt == BOOL {
                        // integer/bool intermediates: exact match
                        let g = read_i64(arr);
                        let rf = read_i64(*r);
                        if g.len() != rf.len() || g.iter().zip(&rf).any(|(a, b)| a != b) {
                            println!(
                                "[{ni}] {} '{}' INT DIVERGES shape={:?} got{:?} ref{:?}",
                                n.op,
                                name,
                                shape(arr),
                                &g[..g.len().min(6)],
                                &rf[..rf.len().min(6)]
                            );
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
        if ni % 250 == 0 {
            println!("[{ni}/{}] {} ok (worst rel so far {:.2e})", g.nodes.len(), n.op, worst);
        }
    }
    // final audio
    if let Some(Val::A(audio)) = env.get("audio") {
        let rel = compare(*audio, refmap["audio"]);
        println!(
            "\nFINAL audio rel={:.3e} (worst intermediate {:.2e}, {} float nodes >1e-2)",
            rel, worst, diverged
        );
        println!("{}", if rel < 1e-3 { "PARITY OK" } else { "PARITY FAIL" });
    } else {
        println!("no audio produced");
    }
}

fn compare(got: mlx_array, refa: mlx_array) -> f64 {
    let g = read_f32(got);
    let r = read_f32(refa);
    if g.len() != r.len() {
        return 1e9;
    }
    let mut se = 0f64;
    let mut rs = 0f64;
    for i in 0..g.len() {
        let d = g[i] as f64 - r[i] as f64;
        se += d * d;
        rs += (r[i] as f64) * (r[i] as f64);
    }
    let rmse = (se / g.len() as f64).sqrt();
    let rrms = (rs / g.len() as f64).sqrt();
    if rrms > 1e-9 {
        rmse / rrms
    } else {
        rmse
    }
}

include!("ops.rs");
