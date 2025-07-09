use std::ffi::CString;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::path::Path;
use std::ptr::NonNull;

use crate::Error;
use crate::error::ErrorCode;

pub struct FfiBytes {
    ptr: NonNull<u8>,
    len: usize,
}
impl FfiBytes {
    pub unsafe fn new(ptr: NonNull<u8>, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn copy_of(src: &[u8]) -> Self {
        let len = src.len();
        let ptr = unsafe { libc::malloc(len) };
        let ptr = NonNull::new(ptr as *mut u8).unwrap();
        let mut self_ = Self { ptr, len };
        self_.as_mut_slice().copy_from_slice(src);
        self_
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
impl Drop for FfiBytes {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}
impl Clone for FfiBytes {
    fn clone(&self) -> Self {
        Self::copy_of(self.as_slice())
    }
}

#[derive(Clone)]
pub enum CowBytes<'a> {
    OwnedRust(Vec<u8>),
    OwnedFfi(FfiBytes),
    Borrowed(&'a [u8]),
}
impl<'a> CowBytes<'a> {
    pub(crate) unsafe fn from_c_buf(ptr: NonNull<u8>, len: usize, needs_free: bool) -> Self {
        if needs_free {
            Self::OwnedFfi(FfiBytes { ptr, len })
        } else {
            Self::Borrowed(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) })
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::OwnedRust(vec) => vec.as_slice(),
            Self::OwnedFfi(bytes) => bytes.as_slice(),
            Self::Borrowed(slice) => slice,
        }
    }

    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::OwnedRust(vec) => vec,
            Self::Borrowed(_) | Self::OwnedFfi(_) => self.as_slice().to_vec(),
        }
    }

    pub fn to_mut(&mut self) -> &mut Vec<u8> {
        match self {
            Self::OwnedRust(vec) => vec,
            Self::OwnedFfi(_) | Self::Borrowed(_) => {
                *self = Self::OwnedRust(self.as_slice().to_vec());
                match self {
                    Self::OwnedRust(vec) => vec,
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn shallow_clone(&self) -> CowBytes {
        CowBytes::Borrowed(self.as_slice())
    }
}

impl<'a> From<Vec<u8>> for CowBytes<'a> {
    fn from(value: Vec<u8>) -> Self {
        Self::OwnedRust(value)
    }
}
impl<'a> From<FfiBytes> for CowBytes<'a> {
    fn from(value: FfiBytes) -> Self {
        Self::OwnedFfi(value)
    }
}
impl<'a> From<&'a [u8]> for CowBytes<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self::Borrowed(value)
    }
}
impl<'a> AsRef<[u8]> for CowBytes<'a> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

pub(crate) struct BytesMaybePassOwnershipToC<'a>(ManuallyDrop<CowBytes<'a>>);
impl<'a> BytesMaybePassOwnershipToC<'a> {
    pub(crate) fn new(bytes: CowBytes<'a>) -> Self {
        Self(ManuallyDrop::new(bytes))
    }
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
    pub(crate) fn ownership_moved(&self) -> bool {
        match &*self.0 {
            CowBytes::Borrowed(_) | CowBytes::OwnedRust(_) => false,
            CowBytes::OwnedFfi(_) => true, // We move the ownership of the C allocated buffer to the C library
        }
    }
}
impl<'a> Drop for BytesMaybePassOwnershipToC<'a> {
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

#[cfg(test)]
pub(crate) mod tests {
    use rand::distr::weighted::WeightedIndex;
    use rand::prelude::*;

    pub(crate) fn rand_src_len(rand: &mut StdRng) -> usize {
        let (max_lens, weights): (Vec<_>, Vec<_>) = [
            (0x1, 1),
            (0x10, 4),
            (0x100, 8),
            (0x1000, 16),
            (0x10000, 4),
            (0x100000, 1),
        ]
        .into_iter()
        .unzip();
        let dist = WeightedIndex::new(&weights).unwrap();
        let max_len = max_lens[dist.sample(rand)];
        rand.random_range(0..=max_len)
    }
}
