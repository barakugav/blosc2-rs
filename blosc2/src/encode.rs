use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::validate_compressed_buf_and_get_sizes;
use crate::Error;

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

    pub(crate) fn ctx_ptr(&self) -> *mut blosc2_sys::blosc2_context {
        self.0 .0.as_ptr()
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
                self.ctx_ptr(),
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

    pub fn params(&self) -> CParams {
        let mut params = MaybeUninit::<blosc2_sys::blosc2_cparams>::uninit();
        unsafe {
            blosc2_sys::blosc2_ctx_get_cparams(self.ctx_ptr(), params.as_mut_ptr())
                .into_result()
                .unwrap()
        };
        let params = unsafe { params.assume_init() };
        CParams(params)
    }
}
pub struct Decoder(Context);
impl Decoder {
    pub fn new(params: DParams) -> Result<Self, Error> {
        let ctx = unsafe { blosc2_sys::blosc2_create_dctx(params.0) };
        let ctx = NonNull::new(ctx).ok_or(Error::Failure)?;
        Ok(Self(Context(ctx)))
    }

    pub(crate) fn ctx_ptr(&self) -> *mut blosc2_sys::blosc2_context {
        self.0 .0.as_ptr()
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
                self.ctx_ptr(),
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

    pub fn params(&self) -> DParams {
        let mut params = MaybeUninit::<blosc2_sys::blosc2_dparams>::uninit();
        unsafe {
            blosc2_sys::blosc2_ctx_get_dparams(self.ctx_ptr(), params.as_mut_ptr())
                .into_result()
                .unwrap()
        };
        let params = unsafe { params.assume_init() };
        DParams(params)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressAlgo {
    Blosclz = blosc2_sys::BLOSC_BLOSCLZ as _,
    Lz4 = blosc2_sys::BLOSC_LZ4 as _,
    Lz4hc = blosc2_sys::BLOSC_LZ4HC as _,
    #[cfg(feature = "zlib")]
    Zlib = blosc2_sys::BLOSC_ZLIB as _,
    #[cfg(feature = "zstd")]
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
        match self.0.compcode as _ {
            blosc2_sys::BLOSC_BLOSCLZ => CompressAlgo::Blosclz,
            blosc2_sys::BLOSC_LZ4 => CompressAlgo::Lz4,
            blosc2_sys::BLOSC_LZ4HC => CompressAlgo::Lz4hc,
            #[cfg(feature = "zlib")]
            blosc2_sys::BLOSC_ZLIB => CompressAlgo::Zlib,
            #[cfg(feature = "zstd")]
            blosc2_sys::BLOSC_ZSTD => CompressAlgo::Zstd,
            unknown_code => panic!("Unknown compressor code: {unknown_code}"),
        }
    }

    pub fn clevel(&mut self, clevel: u32) -> &mut Self {
        self.0.clevel = clevel as _;
        self
    }
    pub fn get_clevel(&self) -> u32 {
        self.0.clevel as u32
    }

    pub fn typesize(&mut self, typesize: NonZeroUsize) -> &mut Self {
        self.0.typesize = typesize.get() as _;
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
            Some(blocksize) => blocksize as _,
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
        match self.0.splitmode as _ {
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
        let filters = self.0.filters;
        let filters_meta = self.0.filters_meta;
        filters
            .into_iter()
            .zip(filters_meta)
            .filter_map(|(f, meta)| {
                Some(match f as _ {
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

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use super::{Decoder, Encoder};
    use crate::util::tests::{rand_cparams, rand_dparams, rand_src_len};

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
