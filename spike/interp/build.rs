// Generate mlx-c bindings (bindgen) and link the prebuilt CPU-only mlx-c + core.
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mlxc = manifest.parent().unwrap().join("mlx-c");
    let build = mlxc.join("build");

    // --- link ---
    println!("cargo:rustc-link-search=native={}", build.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("_deps/mlx-build").display()
    );
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // Metal GPU backend (libmlx built with MLX_BUILD_METAL=ON)
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    println!("cargo:rustc-link-lib=framework=QuartzCore");

    // --- bindgen ---
    // CommandLineTools libclang, in case clang-sys can't auto-locate it.
    if env::var("LIBCLANG_PATH").is_err() {
        env::set_var(
            "LIBCLANG_PATH",
            "/Library/Developer/CommandLineTools/usr/lib",
        );
    }
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", mlxc.display()))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("write bindings");
    println!("cargo:rerun-if-changed=wrapper.h");
}
