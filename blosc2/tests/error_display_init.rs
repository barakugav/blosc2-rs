#[test]
fn error_display_initializes_global_state() {
    // Regression test: formatting an Error should be safe as the first interaction with the crate
    // (e.g. when an error is constructed without calling into Blosc2 first).
    let s = format!("{}", blosc2::Error::InvalidParam);
    assert!(!s.is_empty());
}

