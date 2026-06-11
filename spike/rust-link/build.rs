// Link the prebuilt mlx-c + mlx core static libs (CPU-only spike build).
use std::path::PathBuf;

fn main() {
    let spike = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("mlx-c/build");
    println!("cargo:rustc-link-search=native={}", spike.display());
    println!(
        "cargo:rustc-link-search=native={}",
        spike.join("_deps/mlx-build").display()
    );
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    // mlx core is C++ and uses Accelerate's BLAS on macOS.
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
