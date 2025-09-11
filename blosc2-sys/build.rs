use std::path::{Path, PathBuf};

fn main() {
    generate_bindings();

    // Build and link
    let lib_name = build_c_lib();
    println!("cargo::rustc-link-lib=static={lib_name}");

    build_c_lib_extension();
}

fn generate_bindings() {
    let c_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("c");
    println!("cargo::rerun-if-changed={}", c_dir.display());

    let builder = bindgen::Builder::default()
        .use_core()
        .header("c/bindings.h")
        .allowlist_file(".*blosc2.h")
        .allowlist_file(".*b2nd.h")
        .allowlist_file(".*blosc2_rs.h")
        .allowlist_recursively(false)
        .default_enum_style(bindgen::EnumVariation::Consts)
        .generate_cstr(true)
        .blocklist_item("blosc2_get_blosc2_stdio_mmap_defaults")
        .opaque_type("blosc_timestamp_t")
        .blocklist_item("BLOSC2_CPARAMS_DEFAULTS")
        .blocklist_item("BLOSC2_DPARAMS_DEFAULTS")
        .blocklist_item("BLOSC2_IO_DEFAULTS")
        .blocklist_item("BLOSC2_STORAGE_DEFAULTS")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Failed to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_c_lib() -> String {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    let blosc_orig_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("third-party")
        .join("c-blosc2");
    println!("cargo::rerun-if-changed={}", blosc_orig_dir.display());

    // copy blosc_dir to OUT_DIR.
    // Required because the build process modify the sources, and we are not allowed to modify any
    // files outside of OUT_DIR.
    let blosc_dir = out_dir.join("c-blosc2");
    if !blosc_dir.exists() {
        copy_recursively(&blosc_orig_dir, &blosc_dir).unwrap();
    }

    let mut build = cmake::Config::new(&blosc_dir);
    let bool2opt = |b: bool| if b { "ON" } else { "OFF" };
    build
        .define("BUILD_STATIC", "ON")
        .define("BUILD_SHARED", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_FUZZERS", "OFF")
        .define("BUILD_BENCHMARKS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("DEACTIVATE_ZLIB", bool2opt(!cfg!(feature = "zlib")))
        .define("DEACTIVATE_ZSTD", bool2opt(!cfg!(feature = "zstd")))
        .out_dir(out_dir.join("c-blosc2-build"));
    let profile = build.get_profile().to_string();
    let blosc_build_dir = build.build();
    let blosc_build_dir = blosc_build_dir.join("build");

    let target = std::env::var("TARGET").unwrap();
    let (lib_dir, libname) = if target.contains("windows") && target.contains("msvc") {
        (blosc_build_dir.join("blosc").join(profile), "libblosc2")
    } else {
        (blosc_build_dir.join("blosc"), "blosc2")
    };

    println!(
        "cargo::rustc-link-search=native={}",
        lib_dir.to_str().unwrap()
    );
    libname.to_string()
}

fn build_c_lib_extension() {
    let sources_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("c");
    println!("cargo::rerun-if-changed={}", sources_dir.display());

    cc::Build::new()
        .file(sources_dir.join("blosc2_rs.c"))
        .include(sources_dir)
        .compile(&format!("blosc2_rs_c_ext_{}", env!("CARGO_PKG_VERSION")));
}

fn copy_recursively(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_file() {
        std::fs::copy(src, dst)?;
    } else {
        std::fs::create_dir(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            copy_recursively(&entry.path(), &dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}
