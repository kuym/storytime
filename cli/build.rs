// Build script.
//
// Without `--features mlx` this is a no-op (the ONNX backend's `ort` crate
// fetches its own runtime). With `--features mlx` it builds the mlx-c C API
// (which FetchContent-pulls and builds the MLX C++/Metal core), links it, and
// generates Rust bindings with bindgen — no vendored sources or submodules.
use std::env;
use std::path::PathBuf;
use std::process::Command;

const MLX_C_REPO: &str = "https://github.com/ml-explore/mlx-c.git";
// Pinned mlx-c commit (regenerated bindings for MLX 0.31.2).
const MLX_C_REV: &str = "fba4470b89073180056c9ea46c443051375f7399";

fn main() {
    if env::var("CARGO_FEATURE_MLX").is_err() {
        return;
    }
    println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=14.0");
    env::set_var("MACOSX_DEPLOYMENT_TARGET", "14.0");

    // Build mlx-c into a stable cache dir (survives crate-hash changes; under
    // target/, which is gitignored). Skip the build if the lib already exists.
    let cache = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("target")
        .join("mlx-c");
    let src = cache.join("src");
    let build = cache.join("build");
    let lib = build.join("libmlxc.a");

    if !src.join("CMakeLists.txt").exists() {
        std::fs::create_dir_all(&cache).unwrap();
        run(Command::new("git").args(["clone", MLX_C_REPO, src.to_str().unwrap()]));
        run(Command::new("git").args(["-C", src.to_str().unwrap(), "checkout", MLX_C_REV]));
    }
    if !lib.exists() {
        run(Command::new("cmake").args([
            "-G",
            "Ninja",
            "-S",
            src.to_str().unwrap(),
            "-B",
            build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMLX_BUILD_METAL=ON",
            "-DMLX_C_BUILD_EXAMPLES=OFF",
        ]));
        run(Command::new("cmake").args(["--build", build.to_str().unwrap(), "-j"]));
    }

    // Link mlx-c + the MLX core + system frameworks.
    println!("cargo:rustc-link-search=native={}", build.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("_deps/mlx-build").display()
    );
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=c++");
    for fw in [
        "Accelerate",
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
        "QuartzCore",
    ] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }

    // Clang compiler-runtime: provides ___isPlatformVersionAtLeast used by the
    // @available checks in MLX's C++ core.
    if let Ok(out) = Command::new("xcrun").args(["--find", "clang"]).output() {
        let clang = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if let Some(usr) = std::path::Path::new(&clang).parent().and_then(|p| p.parent()) {
            if let Ok(entries) = std::fs::read_dir(usr.join("lib/clang")) {
                for e in entries.flatten() {
                    let darwin = e.path().join("lib/darwin");
                    if darwin.join("libclang_rt.osx.a").exists() {
                        println!("cargo:rustc-link-search=native={}", darwin.display());
                        println!("cargo:rustc-link-lib=static=clang_rt.osx");
                        break;
                    }
                }
            }
        }
    }

    // bindgen over the mlx C API.
    if env::var("LIBCLANG_PATH").is_err() {
        env::set_var("LIBCLANG_PATH", "/Library/Developer/CommandLineTools/usr/lib");
    }
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", src.display()))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("mlx_bindings.rs"))
        .expect("write bindings");
    println!("cargo:rerun-if-changed=wrapper.h");
}

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {cmd:?}: {e}"));
    assert!(status.success(), "command failed: {cmd:?}");
}
