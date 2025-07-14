// Set the BLOSC_TRACE environment variable
//  * for getting more info on what is happening. If the error is not related with
//  * wrong params, please report it back together with the buffer data causing this,
//  * as well as the compression params used.

mod error;
pub use error::Error;

mod chunk;
mod global;
mod util;

mod tracing;
pub(crate) use tracing::trace;

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::{FfiVec, validate_compressed_buf_and_get_sizes};

/// The version of the crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The version of the underlying C-blosc2 library used by this crate.
pub const BLOSC2_C_VERSION: &str = {
    match blosc2_sys::BLOSC2_VERSION_STRING.to_str() {
        Ok(s) => s,
        Err(_) => unreachable!(),
    }
};

struct Context(NonNull<blosc2_sys::blosc2_context>);
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { blosc2_sys::blosc2_free_ctx(self.0.as_ptr()) }
    }
}
pub struct Encoder(Context);
impl Encoder {
    pub fn new(params: CParams) -> Result<Self, Error> {
        let ctx = unsafe { blosc2_sys::blosc2_create_cctx(params.0) };
        let ctx = NonNull::new(ctx).ok_or(Error::Failure)?;
        Ok(Self(Context(ctx)))
    }

    pub fn compress(&mut self, src: &[u8]) -> Result<Vec<u8>, Error> {
        let dst_max_len = src.len() + blosc2_sys::BLOSC2_MAX_OVERHEAD as usize;
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_max_len);
        unsafe { dst.set_len(dst_max_len) };

        let len = self.compress_into(src, dst.as_mut_slice())?;
        assert!(len <= dst_max_len);
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    pub fn compress_into(
        &mut self,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        let status = unsafe {
            blosc2_sys::blosc2_compress_ctx(
                self.0.0.as_ptr(),
                src.as_ptr().cast(),
                src.len() as _,
                dst.as_mut_ptr().cast(),
                dst.len() as _,
            )
        };
        match status {
            len if len > 0 => {
                debug_assert!(len as usize <= dst.len());
                Ok(len as usize)
            }
            0 => Err(Error::WriteBuffer),
            _ => {
                debug_assert!(status < 0);
                Err(Error::from_int(status))
            }
        }
    }
}
pub struct Decoder(Context);
impl Decoder {
    pub fn new(params: DParams) -> Result<Self, Error> {
        let ctx = unsafe { blosc2_sys::blosc2_create_dctx(params.0) };
        let ctx = NonNull::new(ctx).ok_or(Error::Failure)?;
        Ok(Self(Context(ctx)))
    }

    pub fn decompress(&mut self, src: &[u8]) -> Result<Vec<u8>, Error> {
        if src.len() < blosc2_sys::BLOSC_MIN_HEADER_LENGTH as usize {
            return Err(Error::ReadBuffer);
        }
        let (nbytes, _cbytes, _blocksize) = validate_compressed_buf_and_get_sizes(src)?;
        let dst_len = nbytes as usize;

        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_len);
        unsafe { dst.set_len(dst_len) };
        let len = self.decompress_into(src, dst.as_mut_slice())?;

        assert!(len <= dst_len);
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    pub fn decompress_into(
        &mut self,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        let len = unsafe {
            blosc2_sys::blosc2_decompress_ctx(
                self.0.0.as_ptr(),
                src.as_ptr().cast(),
                src.len() as _,
                dst.as_mut_ptr().cast(),
                dst.len() as _,
            )
            .into_result()? as usize
        };
        debug_assert!(len <= dst.len());
        Ok(len)
    }
}

#[derive(Clone)]
pub struct CParams(pub(crate) blosc2_sys::blosc2_cparams);
impl Default for CParams {
    fn default() -> Self {
        Self(unsafe { blosc2_sys::blosc2_get_blosc2_cparams_defaults() })
    }
}
impl CParams {
    pub fn compressor(&mut self, compressor: CompressAlgo) -> &mut Self {
        self.0.compcode = compressor as _;
        self
    }
    pub fn get_compressor(&self) -> CompressAlgo {
        match self.0.compcode as u32 {
            blosc2_sys::BLOSC_BLOSCLZ => CompressAlgo::Blosclz,
            blosc2_sys::BLOSC_LZ4 => CompressAlgo::Lz4,
            blosc2_sys::BLOSC_LZ4HC => CompressAlgo::Lz4hc,
            blosc2_sys::BLOSC_ZLIB => CompressAlgo::Zlib,
            blosc2_sys::BLOSC_ZSTD => CompressAlgo::Zstd,
            unknown_code => panic!("Unknown compressor code: {unknown_code}"),
        }
    }

    pub fn clevel(&mut self, clevel: u32) -> &mut Self {
        self.0.clevel = clevel as u8;
        self
    }
    pub fn get_clevel(&self) -> u32 {
        self.0.clevel as u32
    }

    pub fn typesize(&mut self, typesize: NonZeroUsize) -> &mut Self {
        self.0.typesize = typesize.get() as i32;
        self
    }
    pub fn get_typesize(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.typesize as usize).unwrap()
    }

    pub fn nthreads(&mut self, mut nthreads: usize) -> &mut Self {
        if nthreads == 0 {
            nthreads = 1;
        }
        self.0.nthreads = nthreads as i16;
        self
    }
    pub fn get_nthreads(&self) -> usize {
        self.0.nthreads as usize
    }

    pub fn blocksize(&mut self, blocksize: Option<usize>) -> &mut Self {
        self.0.blocksize = match blocksize {
            None => 0, // auto
            Some(0) => 1,
            Some(blocksize) => blocksize as i32,
        };
        self
    }
    pub fn get_blocksize(&self) -> Option<usize> {
        (self.0.blocksize > 0).then_some(self.0.blocksize as usize)
    }

    pub fn splitmode(&mut self, splitmode: SplitMode) -> &mut Self {
        self.0.splitmode = splitmode as _;
        self
    }
    pub fn get_splitmode(&self) -> SplitMode {
        match self.0.splitmode as u32 {
            blosc2_sys::BLOSC_ALWAYS_SPLIT => SplitMode::Always,
            blosc2_sys::BLOSC_NEVER_SPLIT => SplitMode::Never,
            blosc2_sys::BLOSC_AUTO_SPLIT => SplitMode::Auto,
            blosc2_sys::BLOSC_FORWARD_COMPAT_SPLIT => SplitMode::ForwardCompat,
            unknown_mode => panic!("Unknown split mode: {unknown_mode}"),
        }
    }

    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self, Error> {
        if filters.len() > 6 {
            return Err(Error::InvalidParam);
        }
        if filters.len() > 2 {
            println!("Warning, more than two filters was not tested and seems buggy!")
        }
        self.0.filters = [blosc2_sys::BLOSC_NOFILTER as _; 6];
        self.0.filters_meta = [0; 6];
        for (i, filter) in filters.iter().enumerate() {
            let (filter, meta) = match filter {
                Filter::Shuffle => (blosc2_sys::BLOSC_SHUFFLE, 0),
                Filter::BitShuffle => (blosc2_sys::BLOSC_BITSHUFFLE, 0),
                Filter::Delta => (blosc2_sys::BLOSC_DELTA, 0),
                Filter::TruncPrecision { prec_bits } => {
                    (blosc2_sys::BLOSC_TRUNC_PREC, *prec_bits as u8)
                }
            };
            self.0.filters[i] = filter as _;
            self.0.filters_meta[i] = meta;
        }
        Ok(self)
    }
    pub fn get_filters(&self) -> impl Iterator<Item = Filter> {
        self.0
            .filters
            .iter()
            .zip(self.0.filters_meta)
            .filter_map(|(f, meta)| {
                Some(match *f as u32 {
                    blosc2_sys::BLOSC_NOFILTER => return None,
                    blosc2_sys::BLOSC_SHUFFLE => Filter::Shuffle,
                    blosc2_sys::BLOSC_BITSHUFFLE => Filter::BitShuffle,
                    blosc2_sys::BLOSC_DELTA => Filter::Delta,
                    blosc2_sys::BLOSC_TRUNC_PREC => Filter::TruncPrecision {
                        prec_bits: meta as _,
                    },
                    unknown_filter => panic!("Unknown filter code: {unknown_filter}"),
                })
            })
    }
}
impl std::fmt::Debug for CParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CParams")
            .field("compressor", &self.get_compressor())
            .field("clevel", &self.get_clevel())
            .field("typesize", &self.get_typesize())
            .field("nthreads", &self.get_nthreads())
            .field("blocksize", &self.get_blocksize())
            .field("splitmode", &self.get_splitmode())
            .field("filters", &self.get_filters().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressAlgo {
    Blosclz = blosc2_sys::BLOSC_BLOSCLZ as _,
    Lz4 = blosc2_sys::BLOSC_LZ4 as _,
    Lz4hc = blosc2_sys::BLOSC_LZ4HC as _,
    Zlib = blosc2_sys::BLOSC_ZLIB as _,
    Zstd = blosc2_sys::BLOSC_ZSTD as _,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Filter {
    Shuffle,
    BitShuffle,
    Delta,
    TruncPrecision {
        // Positive value will set absolute precision bits, whereas negative
        // value will reduce the precision bits (similar to Python slicing convention).
        prec_bits: i8,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SplitMode {
    Always = blosc2_sys::BLOSC_ALWAYS_SPLIT as _,
    Never = blosc2_sys::BLOSC_NEVER_SPLIT as _,
    Auto = blosc2_sys::BLOSC_AUTO_SPLIT as _,
    ForwardCompat = blosc2_sys::BLOSC_FORWARD_COMPAT_SPLIT as _,
}

#[derive(Clone)]
pub struct DParams(pub(crate) blosc2_sys::blosc2_dparams);
impl Default for DParams {
    fn default() -> Self {
        Self(unsafe { blosc2_sys::blosc2_get_blosc2_dparams_defaults() })
    }
}
impl DParams {
    pub fn nthreads(&mut self, mut nthreads: usize) -> &mut Self {
        if nthreads == 0 {
            nthreads = 1;
        }
        self.0.nthreads = nthreads as i16;
        self
    }
    pub fn get_nthreads(&self) -> usize {
        self.0.nthreads as usize
    }
}
impl std::fmt::Debug for DParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DParams")
            .field("nthreads", &self.get_nthreads())
            .finish()
    }
}

pub fn list_compressors() -> impl Iterator<Item = &'static str> {
    let compressors = unsafe { blosc2_sys::blosc2_list_compressors() };
    let len = unsafe { strlen(compressors) };
    let slice: &'static [u8] = unsafe { std::slice::from_raw_parts(compressors.cast(), len + 1) };
    let compressors = std::ffi::CStr::from_bytes_with_nul(slice).unwrap();
    let compressors = compressors.to_str().unwrap();
    compressors.split(',')
}

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
    let complib = unsafe { FfiVec::new(complib, strlen(complib.as_ptr()) + 1) };
    let version = unsafe { FfiVec::new(version, strlen(version.as_ptr()) + 1) };

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

unsafe fn strlen(s: *const core::ffi::c_char) -> usize {
    let mut len = 0;
    // SAFETY: Outer caller has provided a pointer to a valid C string.
    while unsafe { *s.add(len) } != 0 {
        len += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use crate::util::tests::{rand_cparams, rand_dparams, rand_src_len};
    use crate::{Decoder, Encoder};

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0x83a9228e9af47dec);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let src_len = rand_src_len(cparams.get_typesize().get(), &mut rand);
            let src = (&mut rand).random_iter().take(src_len).collect::<Vec<u8>>();

            let compressed = Encoder::new(cparams).unwrap().compress(&src).unwrap();

            let decompressed = Decoder::new(rand_dparams(&mut rand))
                .unwrap()
                .decompress(&compressed)
                .unwrap();
            assert_eq!(src, decompressed);
        }
    }
}
