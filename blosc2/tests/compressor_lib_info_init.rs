use blosc2::CompressAlgo;

#[test]
fn compressor_lib_info_initializes_global_state() {
    // Regression test: calling into the misc API should work even if Blosc2 was never initialized
    // in this process yet.
    let (lib, version) = blosc2::compressor_lib_info(CompressAlgo::Blosclz);
    assert!(!lib.is_empty());
    assert!(!version.is_empty());
}

