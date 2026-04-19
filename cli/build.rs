use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_MLX").is_ok() {
        // MLX requires macOS 14.0+
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=14.0");
        env::set_var("MACOSX_DEPLOYMENT_TARGET", "14.0");
        let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
        let mlx_backend = PathBuf::from(&manifest).join("../mlx-backend");

        // Build the Swift package in release mode.
        let status = Command::new("swift")
            .args(["build", "-c", "release"])
            .current_dir(&mlx_backend)
            .status()
            .expect("failed to run swift build");
        assert!(status.success(), "swift build failed");

        let lib_dir = mlx_backend.join(".build/release");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=KokoroMLX");

        // MLX's C++ core needs the C++ stdlib and system frameworks.
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");

        // The Swift runtime libraries must also be linked.
        // On macOS with Xcode, they're in the toolchain.
        if let Ok(output) = Command::new("xcrun")
            .args(["--show-sdk-path"])
            .output()
        {
            let sdk = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rustc-link-search=native={sdk}/usr/lib/swift");
        }
        if let Ok(output) = Command::new("swift")
            .args(["-print-target-info"])
            .output()
        {
            // Parse runtimeLibraryPaths from swift -print-target-info JSON
            let info = String::from_utf8_lossy(&output.stdout);
            for line in info.lines() {
                let trimmed = line.trim().trim_matches('"').trim_end_matches(',');
                if trimmed.contains("lib/swift/macos") || trimmed.contains("lib/swift-5") {
                    let path = trimmed.trim_matches('"');
                    if std::path::Path::new(path).exists() {
                        println!("cargo:rustc-link-search=native={path}");
                    }
                }
            }
        }

        // Also link the mlx-swift dependencies. The static lib archive
        // includes the Cmlx (C++ mlx core) objects, but we still need
        // to tell the linker about its transitive deps.
        let cmlx_lib_dir = mlx_backend.join(".build/release");
        println!("cargo:rustc-link-search=native={}", cmlx_lib_dir.display());

        // Clang compiler runtime — provides ___isPlatformVersionAtLeast
        // used by @available checks in MLX's C++ core.
        if let Ok(output) = Command::new("xcrun")
            .args(["--toolchain", "default", "-f", "clang"])
            .output()
        {
            let clang_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let toolchain_usr = PathBuf::from(&clang_path)
                .parent().unwrap()  // bin
                .parent().unwrap()  // usr
                .to_path_buf();
            let clang_lib = toolchain_usr.join("lib/clang");
            if let Ok(entries) = std::fs::read_dir(&clang_lib) {
                for entry in entries.flatten() {
                    let darwin = entry.path().join("lib/darwin");
                    if darwin.join("libclang_rt.osx.a").exists() {
                        println!("cargo:rustc-link-search=native={}", darwin.display());
                        println!("cargo:rustc-link-lib=static=clang_rt.osx");
                        break;
                    }
                }
            }
        }

        // Rerun if Swift sources change.
        println!("cargo:rerun-if-changed=../mlx-backend/Sources/");
        println!("cargo:rerun-if-changed=../mlx-backend/Package.swift");
    }
}
