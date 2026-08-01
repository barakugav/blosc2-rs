use blosc2::chunk::SChunk;
use blosc2::util::CowVec;

const SCHUNK_BUF_HEX: &str = concat!(
    "9ea862326672616d6500d200000061cf00000000000000d4a412005003d30000000000000008",
    "d30000000000000028d200000008d200000008d200000008d10000d10001c2d8060000000000",
    "010000000000000000000093cd0007de0000dc00000501070808000000080000002800000000",
    "0000000001000000000000000000000102030405060708050107080800000008000000280000",
    "00000000000001000000000000000000000000000000000000940193cd0006de0000dc0000",
    "ce00000023d80000000000000000000000000000000000",
);

fn decode_hex(hex: &str) -> Vec<u8> {
    assert!(
        hex.len() % 2 == 0,
        "hex string must have even length, got {}",
        hex.len()
    );
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = (bytes[i] as char)
            .to_digit(16)
            .unwrap_or_else(|| panic!("invalid hex char {:?}", bytes[i] as char));
        let lo = (bytes[i + 1] as char)
            .to_digit(16)
            .unwrap_or_else(|| panic!("invalid hex char {:?}", bytes[i + 1] as char));
        out.push(((hi << 4) | lo) as u8);
    }
    out
}

#[test]
fn from_buffer_initializes_global_state() {
    // Regression test: `SChunk::from_buffer` must work even if Blosc2 was never initialized in
    // this process yet (i.e. no prior `SChunk::new`/`open` calls).
    let buffer = decode_hex(SCHUNK_BUF_HEX);

    let mut schunk = SChunk::from_buffer(CowVec::from(buffer)).unwrap();
    assert_eq!(schunk.num_chunks(), 1);
    assert_eq!(schunk.get_chunk(0).unwrap().decompress().unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}
