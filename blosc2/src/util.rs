use std::ffi::CString;
use std::mem::MaybeUninit;
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

    pub fn into_vec(self) -> Vec<u8> {
        match self {
            Self::OwnedRust(vec) => vec,
            Self::Borrowed(_) | Self::OwnedFfi(_) => self.as_slice().to_vec(),
        }
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

pub(crate) fn path2cstr(path: &Path) -> CString {
    path.to_str()
        .and_then(|p| CString::new(p).ok())
        .expect("failed to convert path to cstr")
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
