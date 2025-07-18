use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::util::FfiVec;
use crate::CompressAlgo;

/// Get a list of compressors names supported in the current build.
pub fn list_compressors() -> impl Iterator<Item = &'static str> {
    let compressors = unsafe { blosc2_sys::blosc2_list_compressors() };
    let len = unsafe { libc::strlen(compressors) };
    let slice: &'static [u8] = unsafe { std::slice::from_raw_parts(compressors.cast(), len + 1) };
    let compressors = std::ffi::CStr::from_bytes_with_nul(slice).unwrap();
    let compressors = compressors.to_str().unwrap();
    compressors.split(',')
}

/// Get info from a compression library included in the current build.
///
/// # Arguments
///
/// * `compressor`: The compressor algorithm to get info from.
///
/// # Returns
///
/// A tuple containing the compression library name and its version.
pub fn compressor_lib_info(compressor: CompressAlgo) -> (String, String) {
    let mut compname = MaybeUninit::<*const core::ffi::c_char>::uninit();
    unsafe { blosc2_sys::blosc2_compcode_to_compname(compressor as _, compname.as_mut_ptr()) };
    let compname = unsafe { compname.assume_init() };
    assert!(!compname.is_null());

    let mut complib = MaybeUninit::<*mut core::ffi::c_char>::uninit();
    let mut version = MaybeUninit::<*mut core::ffi::c_char>::uninit();
    unsafe {
        blosc2_sys::blosc2_get_complib_info(compname, complib.as_mut_ptr(), version.as_mut_ptr())
    };
    let complib = NonNull::new(unsafe { complib.assume_init() }).unwrap();
    let version = NonNull::new(unsafe { version.assume_init() }).unwrap();
    let complib = unsafe { FfiVec::from_raw_parts(complib, libc::strlen(complib.as_ptr()) + 1) };
    let version = unsafe { FfiVec::from_raw_parts(version, libc::strlen(version.as_ptr()) + 1) };

    let complib_bytes =
        unsafe { std::mem::transmute::<&[core::ffi::c_char], &[u8]>(complib.as_slice()) };
    let complib = CStr::from_bytes_with_nul(complib_bytes).unwrap();
    let complib = complib.to_str().unwrap().to_string();
    let version_bytes =
        unsafe { std::mem::transmute::<&[core::ffi::c_char], &[u8]>(version.as_slice()) };
    let version = CStr::from_bytes_with_nul(version_bytes).unwrap();
    let version = version.to_str().unwrap().to_string();

    (complib, version)
}
