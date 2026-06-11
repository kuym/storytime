//! Minimal proof that Cargo links mlx-c and can call it: add two arrays on the
//! CPU stream and read the result back. Hand-declared externs (production path
//! is bindgen over mlx/c/*.h); `mlx_array`/`mlx_stream` are single-pointer
//! structs passed by value.
use std::ffi::c_void;
use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone)]
struct MlxArray {
    ctx: *mut c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct MlxStream {
    ctx: *mut c_void,
}

const MLX_FLOAT32: c_int = 10; // mlx_dtype enum index from mlx/c/array.h

extern "C" {
    fn mlx_default_cpu_stream_new() -> MlxStream;
    fn mlx_array_new_data(data: *const c_void, shape: *const c_int, dim: c_int, dtype: c_int)
        -> MlxArray;
    fn mlx_array_new() -> MlxArray;
    fn mlx_add(res: *mut MlxArray, a: MlxArray, b: MlxArray, s: MlxStream) -> c_int;
    fn mlx_array_eval(arr: MlxArray) -> c_int;
    fn mlx_array_data_float32(arr: MlxArray) -> *const f32;
    fn mlx_array_size(arr: MlxArray) -> usize;
}

fn main() {
    unsafe {
        let s = mlx_default_cpu_stream_new();
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [10.0f32, 20.0, 30.0, 40.0];
        let shape = [4i32];
        let a = mlx_array_new_data(a_data.as_ptr() as *const c_void, shape.as_ptr(), 1, MLX_FLOAT32);
        let b = mlx_array_new_data(b_data.as_ptr() as *const c_void, shape.as_ptr(), 1, MLX_FLOAT32);
        let mut r = mlx_array_new();
        assert_eq!(mlx_add(&mut r, a, b, s), 0);
        assert_eq!(mlx_array_eval(r), 0);
        let n = mlx_array_size(r);
        let data = std::slice::from_raw_parts(mlx_array_data_float32(r), n);
        println!("mlx_add result: {:?}", data);
        assert_eq!(data, &[11.0, 22.0, 33.0, 44.0]);
        println!("OK: Rust -> mlx-c link + compute verified");
    }
}
