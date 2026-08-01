use blosc2::chunk::Chunk;
use blosc2::util::CowVec;

const CHUNK_HEX: &str = "05010701080000000800000028000000000000000001000000000000000000000102030405060708";

fn decode_hex(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0);
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = (bytes[i] as char).to_digit(16).unwrap();
        let lo = (bytes[i + 1] as char).to_digit(16).unwrap();
        out.push(((hi << 4) | lo) as u8);
    }
    out
}

#[test]
fn chunk_from_compressed_initializes_global_state() {
    // Regression test: `Chunk::from_compressed` must work even if Blosc2 was never initialized
    // in this process yet.
    let compressed = decode_hex(CHUNK_HEX);
    let chunk = Chunk::from_compressed(CowVec::from(compressed)).unwrap();
    assert_eq!(chunk.decompress().unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

