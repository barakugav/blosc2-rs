//! Basic [`Chunk`] and [`SChunk`] types.
//!
//! This module contains the most basic compressed type of the library, [`Chunk`], which is a contiguous block of
//! memory that was compressed as a single unit and contains repeated "items" of the same typesize.
//! The super chunk, [`SChunk`], is a collection of multiple chunks forming a single logical list of items,
//! but it allows modifying, inserting, and deleting entire chunks while preserving the overall structure.
//!
//! The [`SChunk`] is used as a building block by higher-level data structures like [`Ndarray`](crate::nd::Ndarray),
//! which provide a more user-friendly interface for working with n-dimensional arrays.

mod encode;
pub use encode::*;

#[allow(clippy::module_inception)]
mod chunk;
pub use chunk::*;

mod schunk;
pub use schunk::*;
