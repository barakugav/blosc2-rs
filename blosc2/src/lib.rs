use std::mem::MaybeUninit;
use std::ptr::NonNull;

pub fn compress(src: &[u8], context: &mut Context) -> Result<Vec<u8>, CompressError> {
    let dst_max_len = src.len() + blosc2_sys::BLOSC2_MAX_OVERHEAD as usize;
    let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_max_len);
    unsafe { dst.set_len(dst_max_len) };

    let len = compress_into(src, dst.as_mut_slice(), context)?;
    assert!(len <= dst_max_len);
    unsafe { dst.set_len(len) };
    // SAFETY: every element from 0 to len was initialized
    let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
    Ok(vec)
}

pub fn compress_into(
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    context: &mut Context,
) -> Result<usize, CompressError> {
    let status = unsafe {
        blosc2_sys::blosc2_compress_ctx(
            context.0.as_ptr(),
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
        0 => Err(CompressError::DestinationBufferTooSmall),
        _ => {
            debug_assert!(status < 0);
            Err(CompressError::InternalError(status))
        }
    }
}

/// Error that can occur during compression.
#[derive(Debug)]
pub enum CompressError {
    /// Error indicating that the destination buffer is too small to hold the compressed data.
    DestinationBufferTooSmall,
    /// blosc internal error.
    InternalError(i32),
}
impl std::fmt::Display for CompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressError::DestinationBufferTooSmall => {
                f.write_str("destination buffer is too small")
            }
            CompressError::InternalError(status) => write!(f, "blosc internal error: {status}"),
        }
    }
}
impl std::error::Error for CompressError {}

pub fn decompress(src: &[u8], context: &mut Context) -> Result<Vec<u8>, DecompressError> {
    if src.len() < blosc2_sys::BLOSC_MIN_HEADER_LENGTH as usize {
        return Err(DecompressError::DecompressingError);
    }
    let mut nbytes = MaybeUninit::uninit();
    let mut cbytes = MaybeUninit::uninit();
    let mut blocksize = MaybeUninit::uninit();
    let status = unsafe {
        blosc2_sys::blosc2_cbuffer_sizes(
            src.as_ptr().cast(),
            &mut nbytes as *mut MaybeUninit<i32> as *mut i32,
            &mut cbytes as *mut MaybeUninit<i32> as *mut i32,
            &mut blocksize as *mut MaybeUninit<i32> as *mut i32,
        )
    };
    if status < 0 {
        return Err(DecompressError::InternalError(status));
    }
    let dst_len = unsafe { nbytes.assume_init() } as usize;

    let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_len);
    unsafe { dst.set_len(dst_len) };
    let len = decompress_into(src, dst.as_mut_slice(), context)?;

    assert!(len <= dst_len);
    unsafe { dst.set_len(len) };
    // SAFETY: every element from 0 to len was initialized
    let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
    Ok(vec)
}

pub fn decompress_into(
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    context: &mut Context,
) -> Result<usize, DecompressError> {
    let status = unsafe {
        blosc2_sys::blosc2_decompress_ctx(
            context.0.as_ptr(),
            src.as_ptr().cast(),
            src.len() as _,
            dst.as_mut_ptr().cast(),
            dst.len() as _,
        )
    };
    match status {
        len if len >= 0 => {
            debug_assert!(len as usize <= dst.len());
            Ok(len as usize)
        }
        _ => Err(DecompressError::InternalError(status)),
    }
}

/// Error that can occur during decompression.
#[derive(Debug)]
pub enum DecompressError {
    /// Error indicating that the destination buffer is too small to hold the decompressed data.
    DestinationBufferTooSmall,
    /// Error indicating that the data could not be decompressed.
    DecompressingError,
    /// blosc internal error.
    InternalError(i32),
    // /// An I/O error occurred while reading the compressed data.
    // IoError(std::io::Error),
}
// impl From<std::io::Error> for DecompressError {
//     fn from(err: std::io::Error) -> Self {
//         DecompressError::IoError(err)
//     }
// }
impl std::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressError::DestinationBufferTooSmall => {
                f.write_str("destination buffer is too small")
            }
            DecompressError::DecompressingError => f.write_str("failed to decompress the data"),
            DecompressError::InternalError(status) => write!(f, "blosc internal error: {status}"),
            // DecompressError::IoError(err) => write!(f, "I/O error: {err}"),
        }
    }
}
impl std::error::Error for DecompressError {}

#[derive(Debug)]
pub struct Context(NonNull<blosc2_sys::blosc2_context>);
impl Context {
    pub fn new_compress(params: CParams) -> Option<Self> {
        let ctx = unsafe { blosc2_sys::blosc2_create_cctx(params.0) };
        Some(Context(NonNull::new(ctx)?))
    }
    pub fn new_decompress(params: DParams) -> Option<Self> {
        let ctx = unsafe { blosc2_sys::blosc2_create_dctx(params.0) };
        Some(Context(NonNull::new(ctx)?))
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { blosc2_sys::blosc2_free_ctx(self.0.as_ptr()) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CParams(blosc2_sys::blosc2_cparams);
impl Default for CParams {
    fn default() -> Self {
        Self(blosc2_sys::blosc2_cparams {
            compcode: blosc2_sys::BLOSC_BLOSCLZ as _,
            compcode_meta: 0,
            clevel: 5,
            use_dict: 0,
            typesize: 8,
            nthreads: 1,
            blocksize: 0,
            splitmode: SplitMode::ForwardCompat as _,
            schunk: std::ptr::null_mut(),
            filters: [0, 0, 0, 0, 0, blosc2_sys::BLOSC_SHUFFLE as _],
            filters_meta: [0, 0, 0, 0, 0, 0],
            prefilter: None,
            preparams: std::ptr::null_mut(),
            tuner_params: std::ptr::null_mut(),
            tuner_id: 0,
            instr_codec: false,
            codec_params: std::ptr::null_mut(),
            filter_params: [std::ptr::null_mut(); 6],
        })
    }
}
impl CParams {
    #[must_use]
    pub fn compressor(mut self, compressor: CompressAlgo) -> Self {
        self.0.compcode = compressor as _;
        self
    }
    #[must_use]
    pub fn clevel(mut self, clevel: u32) -> Self {
        self.0.clevel = clevel as u8;
        self
    }
    #[must_use]
    pub fn typesize(mut self, typesize: usize) -> Self {
        self.0.typesize = typesize as i32;
        self
    }
    #[must_use]
    pub fn nthreads(mut self, nthreads: usize) -> Self {
        self.0.nthreads = nthreads as i16;
        self
    }
    #[must_use]
    pub fn blocksize(mut self, blocksize: usize) -> Self {
        self.0.blocksize = blocksize as i32;
        self
    }
    #[must_use]
    pub fn splitmode(mut self, splitmode: SplitMode) -> Self {
        self.0.splitmode = splitmode as _;
        self
    }
    #[must_use]
    pub fn filters(mut self, filters: &[Filter]) -> Option<Self> {
        if filters.len() > 6 {
            return None;
        }
        self.0.filters = [blosc2_sys::BLOSC_NOFILTER as _; 6];
        self.0.filters_meta = [0; 6];
        for (i, filter) in filters.iter().enumerate() {
            let (filter, meta) = match filter {
                Filter::Shuffle => (blosc2_sys::BLOSC_SHUFFLE, 0),
                Filter::BitShuffle => (blosc2_sys::BLOSC_BITSHUFFLE, 0),
                Filter::Delta => (blosc2_sys::BLOSC_DELTA, 0),
                Filter::TruncPrecision(prec_bits) => {
                    (blosc2_sys::BLOSC_TRUNC_PREC, *prec_bits as u8)
                }
            };
            self.0.filters[i] = filter as _;
            self.0.filters_meta[i] = meta;
        }
        Some(self)
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
    // Positive values of prec_bits will set absolute precision bits, whereas negative
    // values will reduce the precision bits (similar to Python slicing convention).
    TruncPrecision(i8),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SplitMode {
    Always = blosc2_sys::BLOSC_ALWAYS_SPLIT as _,
    Never = blosc2_sys::BLOSC_NEVER_SPLIT as _,
    Auto = blosc2_sys::BLOSC_AUTO_SPLIT as _,
    ForwardCompat = blosc2_sys::BLOSC_FORWARD_COMPAT_SPLIT as _,
}

#[derive(Clone, Copy, Debug)]
pub struct DParams(blosc2_sys::blosc2_dparams);
impl Default for DParams {
    fn default() -> Self {
        Self(blosc2_sys::blosc2_dparams {
            nthreads: 1,
            schunk: std::ptr::null_mut(),
            postfilter: None,
            postparams: std::ptr::null_mut(),
        })
    }
}
impl DParams {
    #[must_use]
    pub fn nthreads(mut self, nthreads: usize) -> Self {
        self.0.nthreads = nthreads as i16;
        self
    }
}

pub fn list_compressors() -> impl IntoIterator<Item = &'static str> {
    let compressors = unsafe { blosc2_sys::blosc2_list_compressors() };
    let len = unsafe { strlen(compressors) };
    let slice = unsafe { std::slice::from_raw_parts(compressors.cast(), len + 1) };
    let compressors = std::ffi::CStr::from_bytes_with_nul(slice).unwrap();
    let compressors = compressors.to_str().unwrap();
    compressors.split(',')
}

unsafe fn strlen(s: *const ::core::ffi::c_char) -> usize {
    let mut len = 0;
    // SAFETY: Outer caller has provided a pointer to a valid C string.
    while unsafe { *s.add(len) } != 0 {
        len += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::{CParams, Context};

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0x83a9228e9af47dec);

        for _ in 0..100 {
            let src_len = {
                let max_lens = [0x1, 0x10, 0x100, 0x1000, 0x10000, 0x100000];
                let max_len = max_lens[rand.random_range(0..max_lens.len())];
                rand.random_range(0..=max_len)
            };
            let src = (0..rand.random_range(0..=src_len))
                .map(|_| rand.random_range(0..=255) as u8)
                .collect::<Vec<u8>>();

            let compressed = crate::compress(
                &src,
                &mut Context::new_compress(CParams::default()).unwrap(),
            )
            .unwrap();

            let decompressed = crate::decompress(
                &compressed,
                &mut Context::new_decompress(crate::DParams::default()).unwrap(),
            )
            .unwrap();
            assert_eq!(src, decompressed);
        }
    }
}
