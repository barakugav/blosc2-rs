#![cfg_attr(deny_warnings, deny(warnings))]
#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

//! Rust bindings for blosc2 - a fast, compressed, persistent binary data store library.
//!
//! Provide a safe interface to the [blosc2](https://github.com/Blosc/c-blosc2) library, which is a
//! new major version of the original [blosc](https://github.com/barakugav/blosc-rs) library.
//!
//! The library provides [`Ndarray`](crate::nd::Ndarray), an n-dimensional array implementation with compressed
//! storage, and some lower level utilities for working with compressed data such as [`Chunk`](crate::chunk::Chunk)
//! and [`SChunk`](crate::chunk::SChunk).
//!
//!
//! ### Getting Started
//!
//! To use this library, add the following to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! blosc2 = "0.1"
//! ```
//!
//! The [`Ndarray`](crate::nd::Ndarray) is an n-dimensional array with compressed storage, that support random access
//! to items or slices.
//! In the following example, we create a new `Ndarray` from an array from the `ndarray` crate, and than access its
//! elements and slice:
//! ```rust
//! use blosc2::nd::{Ndarray, NdarrayParams};
//!
//! let arr = Ndarray::from_ndarray(
//!     &ndarray::array!([1_i32, 2, 3], [4, 5, 6], [7, 8, 9]),
//!     NdarrayParams::default()
//!         .chunkshape(Some(&[2, 2]))
//!         .blockshape(Some(&[1, 1])),
//! ).unwrap();
//!
//! assert_eq!(4, arr.get::<i32>(&[1, 0]).unwrap());
//! assert_eq!(9, arr.get::<i32>(&[2, 2]).unwrap());
//!
//! let slice_arr: ndarray::Array2<i32> = arr.slice(&[0..2, 1..3]).unwrap();
//! assert_eq!(slice_arr, ndarray::array![[2, 3], [5, 6]]);
//! ```
//!
//! The library provides a high level [`Ndarray`](crate::nd::Ndarray) struct, which is built upon the
//! [`SChunk`](crate::chunk::SChunk), that in itself is built upon [`Chunk`](crate::chunk::Chunk).
//! Most users will probably interact only with the `Ndarray`, but the chunks types are also available.
//!
//!
//! ## Features
//! - `zlib`: Enable support for the zlib compression codec.
//! - `zstd`: Enable support for the zstd compression codec.
//! - `ndarray`: Enable conversions between blosc2's [`Ndarray`](crate::nd::Ndarray) and the `ndarray` crate's
//!   `ArrayBase` types.
//! - `half`: Add a dependency to the `half` crate, and implement the [`Dtyped`](crate::nd::Dtyped) trait for
//!   `half::f16`. See the [`util`] module for the info.
//! - `num-complex`: Add a dependency to the `num-complex` crate, and implement the [`Dtyped`](crate::nd::Dtyped) trait
//!   for `num_complex::Complex<f32>` and `num_complex::Complex<f64>`. See the [`util`] module for the info.
//!
//! ## Error Handling
//! The library follow the C API and returns error codes. In addition, if the environment variable
//! `BLOSC_TRACE` is set, it will print detailed trace during failures which is useful for
//! debugging.

mod params;
pub use params::*;

pub mod chunk;
mod global;
pub mod nd;
pub mod util;

mod error;
pub use error::Error;

mod tracing;
pub(crate) use tracing::trace;

mod misc;
pub use misc::*;

/// The version of the crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The version of the underlying C-blosc2 library used by this crate.
pub const BLOSC2_C_VERSION: &str = {
    match blosc2_sys::BLOSC2_VERSION_STRING.to_str() {
        Ok(s) => s,
        Err(_) => unreachable!(),
    }
};

#[cfg(feature = "ndarray")]
pub use ndarray;
