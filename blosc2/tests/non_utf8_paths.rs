#[test]
fn nul_in_paths_returns_error_instead_of_panicking() {
    use std::path::PathBuf;

    use blosc2::chunk::SChunk;
    use blosc2::{CParams, DParams, Error};

    // Paths with interior NUL cannot be represented as C strings; ensure we don't panic.
    let urlpath = PathBuf::from("bad\0path");

    let err = match SChunk::new_on_disk(&urlpath, CParams::default(), DParams::default()) {
        Ok(_) => panic!("expected invalid path error"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidParam));
}
