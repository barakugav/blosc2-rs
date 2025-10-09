//! Utility functions and types.
//!
//! This module define the [`f16`] and [`Complex`] types. If the `half` or `num-complex` features are enabled,
//! the corresponding types are re-exported from the `half` and `num-complex` crates respectively. Otherwise,
//! simple structs are defined, providing a minimal implementation with conversions from/to bits, without any
//! arithmetic operations.

use std::ffi::CString;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::path::Path;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::Error;

/// A vector similar to `Vec<T>`, but allocated using C's `malloc` and freed using `free`.
///
/// This type is useful for returning buffers allocated by C code to Rust.
/// The struct does not support resizing.
pub struct FfiVec<T> {
    ptr: NonNull<T>,
    len: usize,
}
impl<T> FfiVec<T> {
    /// Creates a new `FfiVec` from a raw pointer and length.
    ///
    /// Note that the `len` is not the length of allocation, but rather the number of valid elements
    /// in the vector. The "capacity" is not passed, as it is not required by C's `free` and resizing
    /// is not supported.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is valid and points to a memory region of at least `len` elements of
    /// type `T`. The memory must have be allocated using `malloc` and it will be freed using `free` when the `FfiVec`
    /// is dropped.
    pub unsafe fn from_raw_parts(ptr: NonNull<T>, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Get a slice of the elements in the vector.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    /// Get a mutable slice of the elements in the vector.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
impl<T> Drop for FfiVec<T> {
    fn drop(&mut self) {
        unsafe { std::ptr::drop_in_place(self.as_mut_slice()) };
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}
impl FfiVec<u8> {
    /// Creates a new `FfiVec` from a byte slice by allocating memory using `malloc` and copying
    /// the bytes data.
    pub fn copy_of(buf: &[u8]) -> Self {
        let ptr = unsafe { libc::malloc(buf.len()) };
        let ptr = NonNull::new(ptr as *mut u8).unwrap();
        let mut vec = Self {
            ptr,
            len: buf.len(),
        };
        vec.as_mut_slice().copy_from_slice(buf);
        vec
    }
}
impl Clone for FfiVec<u8> {
    fn clone(&self) -> Self {
        Self::copy_of(self.as_slice())
    }
}
impl<T> std::fmt::Debug for FfiVec<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_slice(), f)
    }
}

/// A super type of either a `Vec<T>`, an `FfiVec<T>` or a borrowed slice of type `&[T]`.
///
/// This type is similar to `Cow<[T]>`, but it also has a variant for `FfiVec<T>`.
pub enum CowVec<'a, T> {
    /// An owned `Vec<T>`.
    OwnedRust(Vec<T>),
    /// An owned `FfiVec<T>`, allocated using C's `malloc`.
    OwnedFfi(FfiVec<T>),
    /// A borrowed slice of type `&[T]`.
    Borrowed(&'a [T]),
}
impl<T> CowVec<'_, T> {
    pub(crate) unsafe fn from_c_buf(ptr: NonNull<T>, len: usize, needs_free: bool) -> Self {
        if needs_free {
            Self::OwnedFfi(unsafe { FfiVec::from_raw_parts(ptr, len) })
        } else {
            Self::Borrowed(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) })
        }
    }

    /// Get a slice of the elements in the vector.
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::OwnedRust(vec) => vec.as_slice(),
            Self::OwnedFfi(bytes) => bytes.as_slice(),
            Self::Borrowed(slice) => slice,
        }
    }

    /// Convert the `CowVec` into a `Vec<T>`, cloning the data if necessary.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        match self {
            Self::OwnedRust(vec) => vec,
            Self::Borrowed(_) | Self::OwnedFfi(_) => self.as_slice().to_vec(),
        }
    }
}

impl<T> From<Vec<T>> for CowVec<'_, T> {
    fn from(value: Vec<T>) -> Self {
        Self::OwnedRust(value)
    }
}
impl<T> From<FfiVec<T>> for CowVec<'_, T> {
    fn from(value: FfiVec<T>) -> Self {
        Self::OwnedFfi(value)
    }
}
impl<'a, T> From<&'a [T]> for CowVec<'a, T> {
    fn from(value: &'a [T]) -> Self {
        Self::Borrowed(value)
    }
}
impl<T> AsRef<[T]> for CowVec<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T> std::ops::Deref for CowVec<'_, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl Clone for CowVec<'_, u8> {
    fn clone(&self) -> Self {
        match self {
            Self::OwnedRust(vec) => Self::OwnedRust(vec.clone()),
            Self::OwnedFfi(ffi_vec) => Self::OwnedFfi(ffi_vec.clone()),
            Self::Borrowed(slice) => Self::Borrowed(slice),
        }
    }
}

pub(crate) struct BytesMaybePassOwnershipToC<'a>(ManuallyDrop<CowVec<'a, u8>>);
impl<'a> BytesMaybePassOwnershipToC<'a> {
    pub(crate) fn new(bytes: CowVec<'a, u8>) -> Self {
        Self(ManuallyDrop::new(bytes))
    }
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
    pub(crate) fn ownership_moved(&self) -> bool {
        match &*self.0 {
            CowVec::Borrowed(_) | CowVec::OwnedRust(_) => false,
            CowVec::OwnedFfi(_) => true, // We move the ownership of the C allocated buffer to the C library
        }
    }
}
impl Drop for BytesMaybePassOwnershipToC<'_> {
    fn drop(&mut self) {
        // If we didnt need to copy, the ownership moved to the C library.
        // Only drop in case we needed to copy the data and therefore we own the data.
        if !self.ownership_moved() {
            unsafe { ManuallyDrop::drop(&mut self.0) };
        }
    }
}

pub(crate) fn validate_compressed_buf_and_get_sizes(src: &[u8]) -> Result<(i32, i32, i32), Error> {
    let mut nbytes = MaybeUninit::uninit();
    let mut cbytes = MaybeUninit::uninit();
    let mut blocksize = MaybeUninit::uninit();
    unsafe {
        blosc2_sys::blosc2_cbuffer_sizes(
            src.as_ptr().cast(),
            &mut nbytes as *mut MaybeUninit<i32> as *mut i32,
            &mut cbytes as *mut MaybeUninit<i32> as *mut i32,
            &mut blocksize as *mut MaybeUninit<i32> as *mut i32,
        )
        .into_result()?;
    }
    Ok((
        unsafe { nbytes.assume_init() },
        unsafe { cbytes.assume_init() },
        unsafe { blocksize.assume_init() },
    ))
}

pub(crate) fn path2cstr(path: &Path) -> CString {
    path.to_str()
        .and_then(|p| CString::new(p).ok())
        .expect("failed to convert path to cstr")
}

pub(crate) struct ArrayVec<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
}
impl<T, const N: usize> ArrayVec<T, N> {
    fn new() -> Self {
        Self {
            data: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }
    #[allow(unused)]
    pub(crate) fn from_slice(slice: &[T]) -> Option<Self>
    where
        T: Copy,
    {
        Self::from_slice_fn(slice, |item| *item)
    }
    pub(crate) fn from_slice_fn<U>(slice: &[U], f: impl Fn(&U) -> T) -> Option<Self> {
        if slice.len() > N {
            return None;
        }
        let mut arr = Self::new();
        for (i, item) in slice.iter().enumerate() {
            arr.data[i].write(f(item));
        }
        arr.len = slice.len();
        Some(arr)
    }
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }
}
impl<T, const N: usize> Drop for ArrayVec<T, N> {
    fn drop(&mut self) {
        for i in 0..self.len {
            unsafe {
                self.data[i].assume_init_drop();
            }
        }
    }
}
impl<T, const N: usize> Clone for ArrayVec<T, N>
where
    T: Copy,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            len: self.len,
        }
    }
}
impl<T, const N: usize> std::fmt::Debug for ArrayVec<T, N>
where
    T: std::fmt::Debug + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.as_slice(), f)
    }
}

/// Memory-mapped mode for opening a super chunk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum MmapMode {
    /// Open existing file for reading only.
    Read,
    /// Open existing file for reading and writing.
    ReadWrite,
    // /// Create or overwrite existing file for reading and writing.
    // WriteOverride,
    /// Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
    Cow,
}

cfg_if::cfg_if! { if #[cfg(feature = "half")] {
    pub use half::f16;
} else {
        /// A 16-bit floating point type implementing the IEEE 754-2008 standard [`binary16`] a.k.a "half"
        /// format.
        ///
        /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
        /// Enable the `half` feature to get a fully functional `f16` type.
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(transparent)]
        #[allow(non_camel_case_types)]
        pub struct f16(u16);
        impl f16 {
            #[doc = concat!("Creates a new `f16` from its raw bit representation.")]
            pub const fn from_bits(bits: u16) -> Self {
                Self(bits)
            }
            #[doc = concat!("Get the raw bit representation of the `f16`.")]
            pub const fn to_bits(&self) -> u16 {
                self.0
            }
        }
} }

cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
    pub use num_complex::Complex;
} else {
    /// A complex number in Cartesian form.
    ///
    /// Doesn't provide any arithmetic operations, but expose the real and imaginary parts.
    /// Enable the `num-complex` feature to get a fully functional `Complex` type.
    ///
    /// `Complex<T>` is memory layout compatible with an array `[T; 2]`, which is compatible with
    /// libc, numpy, etc.
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct Complex<T> {
        /// Real portion of the complex number
        pub re: T,
        /// Imaginary portion of the complex number
        pub im: T,
    }
} }

#[cfg(test)]
pub(crate) mod tests {
    use rand::distr::weighted::WeightedIndex;
    use rand::prelude::*;

    use crate::util::CowVec;
    use crate::{CParams, CompressAlgo, DParams, Filter, SplitMode};

    #[test]
    fn cow_vec_covariant() {
        #[allow(unused)]
        fn assert_covariant<'a, 'b: 'a, T>(x: CowVec<'b, T>) -> CowVec<'a, T> {
            x
        }
    }

    pub(crate) fn rand_src_len(typesize: usize, rand: &mut impl Rng) -> usize {
        if typesize == 0 {
            return 0;
        }

        let (max_lens, weights): (Vec<_>, Vec<_>) = [
            (0x1, 1),
            (0x10, 4),
            (0x100, 64),
            (0x1000, 64),
            (0x10000, 4),
            (0x100000, 1),
        ]
        .into_iter()
        .unzip();
        let dist = WeightedIndex::new(weights).unwrap();
        let max_len = max_lens[dist.sample(rand)];
        let len = rand.random_range(0..=max_len);
        ceil_to_multiple(len, typesize)
    }

    pub(crate) fn rand_cparams(rand: &mut impl Rng) -> CParams {
        rand_cparams_with_itemsize(None, false, rand)
    }

    pub(crate) fn rand_cparams_with_itemsize(
        itemsize: Option<usize>,
        lossy: bool,
        rand: &mut impl Rng,
    ) -> CParams {
        let mut params = CParams::default();

        let compressors = [
            None,
            Some(CompressAlgo::Blosclz),
            Some(CompressAlgo::Lz4),
            Some(CompressAlgo::Lz4hc),
            #[cfg(feature = "zlib")]
            Some(CompressAlgo::Zlib),
            #[cfg(feature = "zstd")]
            Some(CompressAlgo::Zstd),
        ];
        if let Some(compressor) = compressors.choose(rand).unwrap() {
            params.compressor(compressor.clone());
        }

        let levels = [None, Some(rand.random_range(0..=9))];
        if let Some(clevel) = levels.choose(rand).unwrap() {
            params.clevel(*clevel);
        }

        if let Some(itemsize) = itemsize {
            params.typesize(itemsize).unwrap();
        } else {
            let typesizes = [
                None,
                Some(4),
                Some(8),
                Some(rand.random_range(1..=16)),
                Some(rand.random_range(1..=64)),
            ];
            if let Some(typesize) = typesizes.choose(rand).unwrap() {
                params.typesize(*typesize).unwrap();
            }
        }

        let nthreads = [None, Some(rand.random_range(0..=64))];
        if let Some(nthreads) = nthreads.choose(rand).unwrap() {
            params.nthreads(*nthreads);
        }

        let blocksizes = [
            None,
            Some(None),
            Some(Some(rand.random_range(1..=64))),
            Some(Some(1024)),
            Some(Some(32 * 1024)),
            Some(Some(512 * 1024)),
            Some(Some(4 * 1024 * 1024)),
        ];
        if let Some(blocksize) = blocksizes.choose(rand).unwrap() {
            params.blocksize(*blocksize);
        }

        let splitmodes = [
            None,
            Some(SplitMode::Auto),
            Some(SplitMode::Always),
            Some(SplitMode::Never),
            Some(SplitMode::ForwardCompat),
        ];
        if let Some(splitmode) = splitmodes.choose(rand).unwrap() {
            params.splitmode(*splitmode);
        }

        let mut basic_filters = vec![Filter::ByteShuffle, Filter::BitShuffle];
        if false {
            // TODO: https://github.com/Blosc/c-blosc2/issues/701
            basic_filters.push(Filter::Delta);
        }
        if lossy && [4, 8].contains(&params.get_typesize()) {
            basic_filters.push(Filter::TruncPrecision {
                prec_bits: -rand.random_range(1..=3),
            });

            // TODO: values are wrong when both trunc precision and byte shuffle are used
            basic_filters.retain(|f| !matches!(f, Filter::ByteShuffle));
        }
        let filters = basic_filters
            .iter()
            .map(|f| Some(vec![f.clone()]))
            .chain([None, Some(Vec::new()), {
                let mut basic_filters = basic_filters.clone();
                basic_filters.shuffle(rand);
                let filters = basic_filters
                    .into_iter()
                    .take(rand.random_range(1..=6))
                    .collect();
                Some(filters)
            }])
            .collect::<Vec<_>>();
        if let Some(filters) = filters.choose(rand).unwrap() {
            params.filters(filters).unwrap();
        }

        params
    }

    pub(crate) fn rand_dparams(rand: &mut impl Rng) -> DParams {
        let mut params = DParams::default();

        let nthreads = [None, Some(rand.random_range(0..=2))];
        if let Some(nthreads) = nthreads.choose(rand).unwrap() {
            params.nthreads(*nthreads);
        }

        params
    }

    pub(crate) fn ceil_to_multiple(x: usize, m: usize) -> usize {
        assert!(m > 0);
        x.div_ceil(m) * m
    }
}
