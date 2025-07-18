use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::validate_compressed_buf_and_get_sizes;
use crate::{Chunk, Error};

struct Context(NonNull<blosc2_sys::blosc2_context>);
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { blosc2_sys::blosc2_free_ctx(self.0.as_ptr()) }
    }
}

/// An encoder for compressing data.
pub struct Encoder(Context);
impl Encoder {
    /// Create a new `Encoder` with the given compression parameters.
    pub fn new(params: CParams) -> Result<Self, Error> {
        let ctx = unsafe { blosc2_sys::blosc2_create_cctx(params.0) };
        let ctx = NonNull::new(ctx).ok_or(Error::Failure)?;
        Ok(Self(Context(ctx)))
    }

    pub(crate) fn ctx_ptr(&self) -> *mut blosc2_sys::blosc2_context {
        self.0 .0.as_ptr()
    }

    /// Compress the given bytes into a new allocated `Chunk`.
    ///
    /// Note that this function allocates a new `Vec<u8>` for the compressed data with the maximum possible size
    /// required for it (uncompressed size + 32), which may be larger than whats actually needed. If this function is
    /// used in a critical performance path, consider using `compress_into` instead, allowing you to provide a
    /// pre-allocated buffer which can be used repeatedly without the overhead of allocations.
    pub fn compress(&mut self, src: &[u8]) -> Result<Chunk<'static>, Error> {
        let dst_max_len = src.len() + blosc2_sys::BLOSC2_MAX_OVERHEAD as usize;
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_max_len);
        unsafe { dst.set_len(dst_max_len) };

        let len = self.compress_into(src, dst.as_mut_slice())?;
        assert!(len <= dst_max_len);
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };

        Ok(unsafe {
            Chunk::from_compressed_unchecked(vec.into(), src.len(), self.params().get_typesize())
        })
    }

    /// Compress the given bytes into a pre-allocated buffer.
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer.
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

    /// Compress a repeated value into a new allocated `Chunk`.
    ///
    /// blosc2 can create chunks of repeated values in a very efficient way without actually
    /// storing the repeated values many times.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of times the value should be repeated.
    /// * `value` - The value to repeat.
    ///
    /// # Returns
    ///
    /// A `Chunk` containing the compressed repeated value.
    pub fn compress_repeatval(
        &self,
        count: usize,
        value: &RepeatedValue,
    ) -> Result<Chunk<'static>, Error> {
        let header_size = blosc2_sys::BLOSC_EXTENDED_HEADER_LENGTH as usize;
        let dst_len = match &value {
            RepeatedValue::Zero | RepeatedValue::Nan | RepeatedValue::Uninit => header_size,
            RepeatedValue::Value(value) => header_size + value.len(),
        };
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_len);
        unsafe { dst.set_len(dst_len) };

        let len = self.compress_repeatval_into(count, value, dst.as_mut_slice())?;
        assert_eq!(len, dst_len);
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };

        let typesize = self.params().get_typesize();
        Ok(unsafe { Chunk::from_compressed_unchecked(vec.into(), count * typesize, typesize) })
    }

    /// Compress a repeated value into a pre-allocated buffer.
    ///
    /// This function is similar to [`Encoder::compress_repeatval`], but allows you to provide a
    /// pre-allocated buffer to store the compressed data.
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer.
    pub fn compress_repeatval_into(
        &self,
        count: usize,
        value: &RepeatedValue,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        let params = self.params();
        let typesize = params.get_typesize();
        let nbytes = typesize * count;
        let status = match value {
            RepeatedValue::Zero => unsafe {
                blosc2_sys::blosc2_chunk_zeros(
                    params.0,
                    nbytes as _,
                    dst.as_mut_ptr().cast(),
                    dst.len() as _,
                )
            },
            RepeatedValue::Nan => unsafe {
                blosc2_sys::blosc2_chunk_nans(
                    params.0,
                    nbytes as _,
                    dst.as_mut_ptr().cast(),
                    dst.len() as _,
                )
            },
            RepeatedValue::Value(value) => {
                if value.len() != typesize {
                    crate::trace!(
                        "Repeated value size doesn't match CParams: {} != {}",
                        value.len(),
                        typesize
                    );
                    return Err(Error::InvalidParam);
                }
                unsafe {
                    blosc2_sys::blosc2_chunk_repeatval(
                        params.0,
                        nbytes as _,
                        dst.as_mut_ptr().cast(),
                        dst.len() as _,
                        value.as_ptr().cast(),
                    )
                }
            }
            RepeatedValue::Uninit => unsafe {
                blosc2_sys::blosc2_chunk_uninit(
                    params.0,
                    nbytes as _,
                    dst.as_mut_ptr().cast(),
                    dst.len() as _,
                )
            },
        };
        Ok(status.into_result()? as usize)
    }

    /// Get the compression parameters used by this encoder.
    pub fn params(&self) -> CParams {
        let mut params = MaybeUninit::uninit();
        unsafe {
            blosc2_sys::blosc2_ctx_get_cparams(self.ctx_ptr(), params.as_mut_ptr())
                .into_result()
                .unwrap()
        };
        let params = unsafe { params.assume_init() };
        CParams(params)
    }
}

/// Represents a repeated value that can be compressed.
///
/// This enum is used as an argument to [`Encoder::compress_repeatval`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RepeatedValue<'a> {
    /// Repeated zeros.
    Zero,
    /// Repeated NaN values (for types that support NaN, like `f32` and `f64`).
    Nan,
    /// Uninitialized values.
    Uninit,
    /// A specific value to repeat.
    ///
    /// The value must have the same size as the `typesize` used in the compression parameters.
    Value(&'a [u8]),
}

/// A decoder for decompressing data.
pub struct Decoder(Context);
impl Decoder {
    /// Create a new `Decoder` with the given decompression parameters.
    pub fn new(params: DParams) -> Result<Self, Error> {
        let ctx = unsafe { blosc2_sys::blosc2_create_dctx(params.0) };
        let ctx = NonNull::new(ctx).ok_or(Error::Failure)?;
        Ok(Self(Context(ctx)))
    }

    pub(crate) fn ctx_ptr(&self) -> *mut blosc2_sys::blosc2_context {
        self.0 .0.as_ptr()
    }

    /// Decompress the given bytes into a new allocated `Vec<u8>`.
    ///
    /// Note that the returned vector may not be aligned to the original data type's alignment, and the caller should
    /// ensure that the alignment is correct before transmuting it to original type. If the alignment does not match
    /// the original data type, the bytes should be copied to a new aligned allocation before transmuting, otherwise
    /// undefined behavior may occur. Alternatively, the caller can use [`Self::decompress_into`] and provide an already
    /// aligned destination buffer.
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

    /// Decompress the given bytes into a pre-allocated buffer.
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer.
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

    /// Get the decompression parameters used by this decoder.
    pub fn params(&self) -> DParams {
        let mut params = MaybeUninit::uninit();
        unsafe {
            blosc2_sys::blosc2_ctx_get_dparams(self.ctx_ptr(), params.as_mut_ptr())
                .into_result()
                .unwrap()
        };
        let params = unsafe { params.assume_init() };
        DParams(params)
    }
}

/// Compression algorithms supported by blosc2.
///
/// The library itself always uses some "backend" compression algorithm, such as `blosclz`, `lz4`,
/// `zlib`, or `zstd`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressAlgo {
    /// Blosc's own compression algorithm, `blosclz`.
    Blosclz = blosc2_sys::BLOSC_BLOSCLZ as _,
    /// LZ4 compression algorithm.
    Lz4 = blosc2_sys::BLOSC_LZ4 as _,
    /// LZ4HC compression algorithm.
    Lz4hc = blosc2_sys::BLOSC_LZ4HC as _,
    /// Zlib compression algorithm.
    #[cfg(feature = "zlib")]
    Zlib = blosc2_sys::BLOSC_ZLIB as _,
    /// Zstandard compression algorithm.
    #[cfg(feature = "zstd")]
    Zstd = blosc2_sys::BLOSC_ZSTD as _,
}

/// Filters that can be applied to the data before compression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Filter {
    /// Byte shuffle filter.
    ///
    /// Given an array of bytes, representing N elements of a type with S bytes, the filter rearrange the bytes from
    /// `[1_1, 1_2, ..., 1_S, 2_1, 2_2, ..., 2_S, ..., N_1, N_2, ..., N_S]` to
    /// `[1_1, 2_1, ..., N_1, 1_2, 2_2, ..., N_2, ..., 1_S, 2_S, ..., N_S]`,
    /// where `i_j` is the j-th byte of the i-th element.
    ByteShuffle,
    /// Bit shuffle filter.
    ///
    /// Similar to `ByteShuffle`, but operates on bits instead of bytes.
    BitShuffle,
    /// Delta filter.
    ///
    /// This filter encodes the data as differences between consecutive elements.
    Delta,
    /// Truncation precision filter for floating point data.
    ///
    /// This filter reduces the precision of floating point numbers by truncating the least
    /// significant bits.
    ///
    /// This filter is only supported for floating point types (e.g., `f32`, `f64`). This can not
    /// be enforced by the library, there it is only checked that the typesize is 4 or 8 bytes.
    TruncPrecision {
        /// The number of bits to truncate.
        ///
        /// Positive value will set absolute precision bits, whereas negative
        /// value will reduce the precision bits (similar to Python slicing convention).
        prec_bits: i8,
    },
}

/// A split mode option for encoders.
#[allow(missing_docs)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SplitMode {
    Always = blosc2_sys::BLOSC_ALWAYS_SPLIT as _,
    Never = blosc2_sys::BLOSC_NEVER_SPLIT as _,
    Auto = blosc2_sys::BLOSC_AUTO_SPLIT as _,
    ForwardCompat = blosc2_sys::BLOSC_FORWARD_COMPAT_SPLIT as _,
}

/// Compression parameters for encoders.
#[derive(Clone)]
pub struct CParams(pub(crate) blosc2_sys::blosc2_cparams);
impl Default for CParams {
    fn default() -> Self {
        Self(unsafe { blosc2_sys::blosc2_get_blosc2_cparams_defaults() })
    }
}
impl CParams {
    /// Set the compressor to use.
    ///
    /// By default, the compressor is set to `Blosclz`.
    pub fn compressor(&mut self, compressor: CompressAlgo) -> &mut Self {
        self.0.compcode = compressor as _;
        self
    }
    /// Get the compressor currently set in the parameters.
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

    /// Set the compression level, in range [0, 9].
    ///
    /// By default, the compression level is set to 5.
    pub fn clevel(&mut self, clevel: u32) -> &mut Self {
        self.0.clevel = clevel as _;
        self
    }
    /// Get the compression level currently set in the parameters.
    pub fn get_clevel(&self) -> u32 {
        self.0.clevel as u32
    }

    /// Set the typesize of the data to compress (in bytes).
    ///
    /// By default, the typesize is set to 8 bytes.
    pub fn typesize(&mut self, typesize: NonZeroUsize) -> &mut Self {
        self.0.typesize = typesize.get() as _;
        self
    }
    /// Get the typesize currently set in the parameters.
    pub fn get_typesize(&self) -> usize {
        debug_assert!(self.0.typesize > 0);
        self.0.typesize as usize
    }

    /// Set the number of threads to use for compression.
    ///
    /// By default, the number of threads is set to 1.
    pub fn nthreads(&mut self, mut nthreads: usize) -> &mut Self {
        if nthreads == 0 {
            nthreads = 1;
        }
        self.0.nthreads = nthreads as i16;
        self
    }
    /// Get the number of threads currently set in the parameters.
    pub fn get_nthreads(&self) -> usize {
        self.0.nthreads as usize
    }

    /// Set the block size for compression.
    ///
    /// None means automatic block size.
    ///
    /// By default, an automatic block size is used.
    pub fn blocksize(&mut self, blocksize: Option<usize>) -> &mut Self {
        self.0.blocksize = match blocksize {
            None => 0, // auto
            Some(0) => 1,
            Some(blocksize) => blocksize as _,
        };
        self
    }
    /// Get the block size currently set in the parameters.
    pub fn get_blocksize(&self) -> Option<usize> {
        (self.0.blocksize > 0).then_some(self.0.blocksize as usize)
    }

    /// Set the split mode for the encoder.
    ///
    /// By default, the split mode is set to `ForwardCompat`.
    pub fn splitmode(&mut self, splitmode: SplitMode) -> &mut Self {
        self.0.splitmode = splitmode as _;
        self
    }
    /// Get the split mode currently set in the parameters.
    pub fn get_splitmode(&self) -> SplitMode {
        match self.0.splitmode as _ {
            blosc2_sys::BLOSC_ALWAYS_SPLIT => SplitMode::Always,
            blosc2_sys::BLOSC_NEVER_SPLIT => SplitMode::Never,
            blosc2_sys::BLOSC_AUTO_SPLIT => SplitMode::Auto,
            blosc2_sys::BLOSC_FORWARD_COMPAT_SPLIT => SplitMode::ForwardCompat,
            unknown_mode => panic!("Unknown split mode: {unknown_mode}"),
        }
    }

    /// Set the filters to apply before compression.
    ///
    /// The maximum number of filters is 6.
    ///
    /// By default, a single `ByteShuffle` filter is applied.
    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self, Error> {
        if filters.len() > 6 {
            crate::trace!("Too many filters, maximum is 6");
            return Err(Error::InvalidParam);
        }
        if filters.len() > 2 {
            println!("Warning, more than two filters was not tested and seems buggy!")
        }
        self.0.filters = [blosc2_sys::BLOSC_NOFILTER as _; 6];
        self.0.filters_meta = [0; 6];
        for (i, filter) in filters.iter().enumerate() {
            let (filter, meta) = match filter {
                Filter::ByteShuffle => (blosc2_sys::BLOSC_SHUFFLE, 0),
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
    /// Get the filters currently set in the parameters.
    pub fn get_filters(&self) -> impl Iterator<Item = Filter> {
        let filters = self.0.filters;
        let filters_meta = self.0.filters_meta;
        filters
            .into_iter()
            .zip(filters_meta)
            .filter_map(|(f, meta)| {
                Some(match f as _ {
                    blosc2_sys::BLOSC_NOFILTER => return None,
                    blosc2_sys::BLOSC_SHUFFLE => Filter::ByteShuffle,
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

/// Decompression parameters for decoders.
#[derive(Clone)]
pub struct DParams(pub(crate) blosc2_sys::blosc2_dparams);
impl Default for DParams {
    fn default() -> Self {
        Self(unsafe { blosc2_sys::blosc2_get_blosc2_dparams_defaults() })
    }
}
impl DParams {
    /// Set the number of threads to use for decompression.
    ///
    /// By default, the number of threads is set to 1.
    pub fn nthreads(&mut self, mut nthreads: usize) -> &mut Self {
        if nthreads == 0 {
            nthreads = 1;
        }
        self.0.nthreads = nthreads as i16;
        self
    }
    /// Get the number of threads currently set in the parameters.
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
    use crate::RepeatedValue;

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0x83a9228e9af47dec);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let src_len = rand_src_len(cparams.get_typesize(), &mut rand);
            let src = (&mut rand).random_iter().take(src_len).collect::<Vec<u8>>();

            let compressed = Encoder::new(cparams).unwrap().compress(&src).unwrap();

            let decompressed = Decoder::new(rand_dparams(&mut rand))
                .unwrap()
                .decompress(compressed.as_bytes())
                .unwrap();
            assert_eq!(src, decompressed);
        }
    }

    #[test]
    fn repeatedval() {
        let mut rand = StdRng::seed_from_u64(0x83a9228e9af47dec);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let typesize = cparams.get_typesize();

            let mut element_buf = Vec::new();
            let value = {
                element_buf.clear();
                element_buf.extend(
                    (&mut rand)
                        .random_iter()
                        .take(typesize)
                        .collect::<Vec<u8>>(),
                );

                let mut variants = Vec::new();
                variants.push(RepeatedValue::Zero);
                variants.push(RepeatedValue::Uninit);
                variants.push(RepeatedValue::Value(&element_buf));
                if [4, 8].contains(&typesize) {
                    variants.push(RepeatedValue::Nan);
                }

                variants.choose(&mut rand).unwrap().clone()
            };
            let src_len = rand_src_len(typesize, &mut rand);

            let compressed = Encoder::new(cparams)
                .unwrap()
                .compress_repeatval(src_len / typesize, &value)
                .unwrap();

            let decompressed = Decoder::new(rand_dparams(&mut rand))
                .unwrap()
                .decompress(compressed.as_bytes())
                .unwrap();
            assert_eq!(src_len, decompressed.len());
            for item in decompressed.chunks_exact(typesize) {
                match value {
                    RepeatedValue::Zero => assert!(item.iter().all(|&b| b == 0)),
                    RepeatedValue::Nan => match typesize {
                        4 => assert!(f32::from_ne_bytes(item.try_into().unwrap()).is_nan()),
                        8 => assert!(f64::from_ne_bytes(item.try_into().unwrap()).is_nan()),
                        _ => panic!("Unexpected typesize for NaN: {}", typesize),
                    },
                    RepeatedValue::Uninit => {}
                    RepeatedValue::Value(v) => assert_eq!(item, v),
                }
            }
        }
    }
}
