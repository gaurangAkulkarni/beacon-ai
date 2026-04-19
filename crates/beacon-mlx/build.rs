//! Build script for beacon-mlx.
//!
//! 1. Builds the C++ shim (`shim/`) via `cmake` crate → static library.
//! 2. Runs `bindgen` on `shim/include/beacon_shim.h` to generate Rust FFI.
//! 3. Emits the correct `cargo:rustc-link-*` directives so downstream crates
//!    can link the shim (and transitively, MLX + system frameworks on macOS).

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shim_dir = manifest_dir.join("../../shim");
    let shim_dir = shim_dir.canonicalize().expect("shim/ directory must exist");

    // --- CMake build ----------------------------------------------------------
    // The `cmake` crate's `build()` runs cmake configure+build and returns a
    // path (`dst`). When there is no `install` target the libraries stay in
    // `dst/build/` rather than `dst/lib/`. We search both locations.
    let dst = cmake::Config::new(&shim_dir)
        .define("BEACON_SHIM_BUILD_TESTS", "OFF")
        .build_target("beacon_shim")
        .build();

    let build_dir = dst.join("build");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=beacon_shim");

    // On Apple, also link MLX and system frameworks.
    if cfg!(target_os = "macos") {
        // MLX static lib built by the cmake subdirectory.
        let mlx_dir = build_dir.join("third_party/mlx");
        println!("cargo:rustc-link-search=native={}", mlx_dir.display());
        println!("cargo:rustc-link-lib=static=mlx");

        // MLX's gguflib.
        let gguflib_dir = mlx_dir.join("mlx/io");
        if gguflib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", gguflib_dir.display());
            println!("cargo:rustc-link-lib=static=gguflib");
        }

        // System frameworks required by MLX.
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // C++ standard library.
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rerun if shim sources change.
    println!(
        "cargo:rerun-if-changed={}",
        shim_dir.join("include").display()
    );
    println!("cargo:rerun-if-changed={}", shim_dir.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        shim_dir.join("CMakeLists.txt").display()
    );

    // --- Bindgen --------------------------------------------------------------
    let header = shim_dir.join("include/beacon_shim.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .allowlist_function("beacon_.*")
        .allowlist_type("Beacon.*")
        .allowlist_var("BEACON_.*")
        // Generate Rust enums for C enums.
        .rustified_enum("BeaconStatus")
        .rustified_enum("BeaconDtype")
        // Layout tests are noisy and platform-dependent; the C ABI is stable.
        .layout_tests(false)
        .generate()
        .expect("failed to generate bindings from beacon_shim.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("ffi.rs"))
        .expect("failed to write ffi.rs");
}
