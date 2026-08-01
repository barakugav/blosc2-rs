#[test]
fn list_compressors_initializes_global_state() {
    // Regression test: calling into the misc API should work even if Blosc2 was never initialized
    // in this process yet.
    let compressors: Vec<&'static str> = blosc2::list_compressors().collect();
    assert!(!compressors.is_empty());
}

