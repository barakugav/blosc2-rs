use blosc2::{CParams, DParams};

#[test]
fn params_default_initializes_global_state() {
    // Regression test: `CParams::default` / `DParams::default` should be safe as the first API
    // calls in a process.
    let cparams = CParams::default();
    let dparams = DParams::default();
    assert!(cparams.get_typesize() > 0);
    assert!(dparams.get_nthreads() >= 1);
}

