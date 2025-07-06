use std::path::PathBuf;
use std::process::Command;

fn main() {
    generate_bindings();

    // Build and link
    if std::env::var("DOCS_RS").is_err() {
        let lib_name = build_c_lib();
        println!("cargo::rustc-link-lib=static={lib_name}");
    }
}

fn generate_bindings() {
    let builder = bindgen::Builder::default()
        .use_core()
        .header("c/bindings.h")
        .allowlist_file(".*/blosc2.h")
        .allowlist_recursively(false)
        .default_enum_style(bindgen::EnumVariation::Consts)
        .blocklist_item("blosc2_get_blosc2_stdio_mmap_defaults")
        .opaque_type("blosc_timestamp_t")
        .blocklist_item("BLOSC2_CPARAMS_DEFAULTS")
        .blocklist_item("BLOSC2_DPARAMS_DEFAULTS")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Failed to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_c_lib() -> String {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    let blosc_dir = out_dir.join("c-blosc2");
    if !blosc_dir.exists() {
        Command::new("git")
            .arg("--version")
            .status()
            .expect("git not found");
        std::fs::create_dir_all(&blosc_dir).expect("Failed to create c-blosc2 directory");
        Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "--branch",
                "v2.19.0",
                "https://github.com/Blosc/c-blosc2.git",
                ".",
            ])
            .current_dir(&blosc_dir)
            .status()
            .inspect_err(|_| {
                let _ = std::fs::remove_dir_all(&blosc_dir);
            })
            .expect("Failed to clone c-blosc2 repository");
    }

    let mut build = cmake::Config::new(&blosc_dir);
    build
        .define("BUILD_STATIC", "ON")
        .define("BUILD_SHARED", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_FUZZERS", "OFF")
        .define("BUILD_BENCHMARKS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
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
