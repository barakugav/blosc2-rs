# blosc2-rs

[![Crates.io](https://img.shields.io/crates/v/blosc2.svg)](https://crates.io/crates/blosc2/)
[![Documentation](https://docs.rs/blosc2/badge.svg)](https://docs.rs/blosc2/)
![License](https://img.shields.io/crates/l/blosc2)

Rust bindings for blosc2 - a fast, compressed, persistent binary data store library.

Provide a safe interface to the [blosc2](https://github.com/Blosc/c-blosc2) library, which is a
new major version of the original [blosc](https://github.com/barakugav/blosc-rs) library.

### Getting Started

To use this library, add the following to your `Cargo.toml`:
```toml
[dependencies]
blosc2 = "0.1"
```

In the following example we compress a vector of integers into a chunk of bytes and then
decompress it back:
```rust
use blosc2::{CParams, Chunk, DParams, Decoder, Encoder};

let data: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
let i32len = std::mem::size_of::<i32>();
let data_bytes =
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * i32len) };

// Compress the data into a Chunk
let cparams = CParams::default()
    .typesize(i32len.try_into().unwrap())
    .clevel(5)
    .nthreads(2)
    .clone();
let chunk: Chunk = Encoder::new(cparams)
    .unwrap()
    .compress(&data_bytes)
    .unwrap();
let chunk_bytes: &[u8] = chunk.as_bytes();

// Decompress the Chunk
let dparams = DParams::default();
let decompressed = Decoder::new(dparams)
    .unwrap()
    .decompress(chunk_bytes)
    .unwrap();

// Check that the decompressed data matches the original
assert_eq!(data_bytes, decompressed);

// A chunk support random access to individual items
assert_eq!(&data_bytes[0..4], chunk.item(0).expect("failed to get the 0-th item"));
assert_eq!(&data_bytes[12..16], chunk.item(3).expect("failed to get the 3-th item"));
assert_eq!(&data_bytes[4..20], chunk.items(1..5).expect("failed to get items 1 to 4"));
```

## Super Chunk
In addition to the basic `Chunk`, this library provides a `SChunk` (Super Chunk) that treat
multiple chunks as a single entity. A super chunk can be saved or loaded from or to files, and
it supports random access to chunks or individual items across all chunks.
```rust
use blosc2::{CParams, DParams, Encoder, SChunk};

let i32len = std::mem::size_of::<i32>();
let cparams = CParams::default()
    .typesize(i32len.try_into().unwrap())
    .clone();
let mut schunk = SChunk::new(cparams.clone(), DParams::default()).unwrap();

// Create two data arrays
let data1: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
let data2: [i32; 7] = [8, 9, 10, 11, 12, 13, 14];
let data1_bytes =
    unsafe { std::slice::from_raw_parts(data1.as_ptr() as *const u8, data1.len() * i32len) };
let data2_bytes =
    unsafe { std::slice::from_raw_parts(data2.as_ptr() as *const u8, data2.len() * i32len) };

// Append the first data array to the SChunk, which will be compressed using SChunk's CParams
schunk.append(data1_bytes).unwrap();
assert_eq!(schunk.num_chunks(), 1);
assert_eq!(7, schunk.items_num());

// Append the second data array to the SChunk, as already compressed data
let data2_cparams = CParams::default()
    .typesize(i32len.try_into().unwrap()) // typesize must match the SChunk's CParams
    .clevel(9)
    .clone();
let data2_chunk = Encoder::new(data2_cparams)
    .unwrap()
    .compress(data2_bytes)
    .unwrap();
schunk.append_chunk(data2_chunk.shallow_clone()).unwrap();
assert_eq!(schunk.num_chunks(), 2);
assert_eq!(14, schunk.items_num());

// Random access a whole chunk within the super-chunk
assert_eq!(
    data2_chunk.decompress().unwrap(),
    schunk.get_chunk(1).unwrap().decompress().unwrap()
);

// Random access individual items within the super-chunk
assert_eq!(5, i32::from_ne_bytes(schunk.item(4).unwrap().try_into().unwrap()));
assert_eq!(12, i32::from_ne_bytes(schunk.item(11).unwrap().try_into().unwrap()));
```

## Features
Cargo features enable or disable support for various compression codecs such as `zstd` and
`zlib`.
