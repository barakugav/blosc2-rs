fn main() {
    let rustv = rustc_version();
    let check_cfg = rustv.map(|v| v >= 80).unwrap_or(false);

    println!("cargo::rerun-if-env-changed=BLOSC2_RS_DENY_WARNINGS");
    let deny_warnings = std::env::var("BLOSC2_RS_DENY_WARNINGS").as_deref() == Ok("1");
    if check_cfg {
        println!("cargo:rustc-check-cfg=cfg(deny_warnings)");
    }
    if deny_warnings {
        println!("cargo:rustc-cfg=deny_warnings");
    }
}

fn rustc_version() -> Option<u32> {
    // Code copied from cxx crate

    let rustc = std::env::var_os("RUSTC")?;
    let output = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()?;
    let version = String::from_utf8(output.stdout).ok()?;
    let mut pieces = version.split('.');
    if pieces.next() != Some("rustc 1") {
        return None;
    }
    let minor = pieces.next()?.parse().ok()?;
    Some(minor)
}
