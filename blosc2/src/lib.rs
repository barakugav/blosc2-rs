use std::mem::MaybeUninit;
use std::ptr::NonNull;

pub fn compress(src: &[u8], context: &mut Context) -> Result<Vec<u8>, Error> {
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
) -> Result<usize, Error> {
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
        0 => Err(Error::WriteBuffer),
        _ => {
            debug_assert!(status < 0);
            Err(Error::from_int(status))
        }
    }
}

pub fn decompress(src: &[u8], context: &mut Context) -> Result<Vec<u8>, Error> {
    if src.len() < blosc2_sys::BLOSC_MIN_HEADER_LENGTH as usize {
        return Err(Error::ReadBuffer);
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
        return Err(Error::from_int(status));
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
) -> Result<usize, Error> {
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
        _ => Err(Error::from_int(status)),
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Generic failure
    Failure = blosc2_sys::BLOSC2_ERROR_FAILURE as _,
    /// Bad stream
    Stream = blosc2_sys::BLOSC2_ERROR_STREAM as _,
    /// Invalid data
    Data = blosc2_sys::BLOSC2_ERROR_DATA as _,
    /// Memory alloc/realloc failure
    MemoryAlloc = blosc2_sys::BLOSC2_ERROR_MEMORY_ALLOC as _,
    /// Not enough space to read
    ReadBuffer = blosc2_sys::BLOSC2_ERROR_READ_BUFFER as _,
    /// Not enough space to write
    WriteBuffer = blosc2_sys::BLOSC2_ERROR_WRITE_BUFFER as _,
    /// Codec not supported
    CodecSupport = blosc2_sys::BLOSC2_ERROR_CODEC_SUPPORT as _,
    /// Invalid parameter supplied to codec
    CodecParam = blosc2_sys::BLOSC2_ERROR_CODEC_PARAM as _,
    /// Codec dictionary error
    CodecDict = blosc2_sys::BLOSC2_ERROR_CODEC_DICT as _,
    /// Version not supported
    VersionSupport = blosc2_sys::BLOSC2_ERROR_VERSION_SUPPORT as _,
    /// Invalid value in header
    InvalidHeader = blosc2_sys::BLOSC2_ERROR_INVALID_HEADER as _,
    /// Invalid parameter supplied to function
    InvalidParam = blosc2_sys::BLOSC2_ERROR_INVALID_PARAM as _,
    /// File read failure
    FileRead = blosc2_sys::BLOSC2_ERROR_FILE_READ as _,
    /// File write failure
    FileWrite = blosc2_sys::BLOSC2_ERROR_FILE_WRITE as _,
    /// File open failure
    FileOpen = blosc2_sys::BLOSC2_ERROR_FILE_OPEN as _,
    /// Not found
    NotFound = blosc2_sys::BLOSC2_ERROR_NOT_FOUND as _,
    /// Bad run length encoding
    RunLength = blosc2_sys::BLOSC2_ERROR_RUN_LENGTH as _,
    /// Filter pipeline error
    FilterPipeline = blosc2_sys::BLOSC2_ERROR_FILTER_PIPELINE as _,
    /// Chunk insert failure
    ChunkInsert = blosc2_sys::BLOSC2_ERROR_CHUNK_INSERT as _,
    /// Chunk append failure
    ChunkAppend = blosc2_sys::BLOSC2_ERROR_CHUNK_APPEND as _,
    /// Chunk update failure
    ChunkUpdate = blosc2_sys::BLOSC2_ERROR_CHUNK_UPDATE as _,
    /// Sizes larger than 2gb not supported
    TwoGbLimit = blosc2_sys::BLOSC2_ERROR_2GB_LIMIT as _,
    /// Super-chunk copy failure
    SchunkCopy = blosc2_sys::BLOSC2_ERROR_SCHUNK_COPY as _,
    /// Wrong type for frame
    FrameType = blosc2_sys::BLOSC2_ERROR_FRAME_TYPE as _,
    /// File truncate failure
    FileTruncate = blosc2_sys::BLOSC2_ERROR_FILE_TRUNCATE as _,
    /// Thread or thread context creation failure
    ThreadCreate = blosc2_sys::BLOSC2_ERROR_THREAD_CREATE as _,
    /// Postfilter failure
    Postfilter = blosc2_sys::BLOSC2_ERROR_POSTFILTER as _,
    /// Special frame failure
    FrameSpecial = blosc2_sys::BLOSC2_ERROR_FRAME_SPECIAL as _,
    /// Special super-chunk failure
    SChunkSpecial = blosc2_sys::BLOSC2_ERROR_SCHUNK_SPECIAL as _,
    /// IO plugin error
    PluginIO = blosc2_sys::BLOSC2_ERROR_PLUGIN_IO as _,
    /// Remove file failure
    FileRemove = blosc2_sys::BLOSC2_ERROR_FILE_REMOVE as _,
    /// Pointer is null
    NullPointer = blosc2_sys::BLOSC2_ERROR_NULL_POINTER as _,
    /// Invalid index
    InvalidIndex = blosc2_sys::BLOSC2_ERROR_INVALID_INDEX as _,
    /// Metalayer has not been found
    MetalayerNotFound = blosc2_sys::BLOSC2_ERROR_METALAYER_NOT_FOUND as _,
    /// Max buffer size exceeded
    MaxBufsizeExceeded = blosc2_sys::BLOSC2_ERROR_MAX_BUFSIZE_EXCEEDED as _,
    /// Tuner failure
    Tuner = blosc2_sys::BLOSC2_ERROR_TUNER as _,
}
impl Error {
    fn from_int(code: core::ffi::c_int) -> Self {
        match code {
            blosc2_sys::BLOSC2_ERROR_FAILURE => Error::Failure,
            blosc2_sys::BLOSC2_ERROR_STREAM => Error::Stream,
            blosc2_sys::BLOSC2_ERROR_DATA => Error::Data,
            blosc2_sys::BLOSC2_ERROR_MEMORY_ALLOC => Error::MemoryAlloc,
            blosc2_sys::BLOSC2_ERROR_READ_BUFFER => Error::ReadBuffer,
            blosc2_sys::BLOSC2_ERROR_WRITE_BUFFER => Error::WriteBuffer,
            blosc2_sys::BLOSC2_ERROR_CODEC_SUPPORT => Error::CodecSupport,
            blosc2_sys::BLOSC2_ERROR_CODEC_PARAM => Error::CodecParam,
            blosc2_sys::BLOSC2_ERROR_CODEC_DICT => Error::CodecDict,
            blosc2_sys::BLOSC2_ERROR_VERSION_SUPPORT => Error::VersionSupport,
            blosc2_sys::BLOSC2_ERROR_INVALID_HEADER => Error::InvalidHeader,
            blosc2_sys::BLOSC2_ERROR_INVALID_PARAM => Error::InvalidParam,
            blosc2_sys::BLOSC2_ERROR_FILE_READ => Error::FileRead,
            blosc2_sys::BLOSC2_ERROR_FILE_WRITE => Error::FileWrite,
            blosc2_sys::BLOSC2_ERROR_FILE_OPEN => Error::FileOpen,
            blosc2_sys::BLOSC2_ERROR_NOT_FOUND => Error::NotFound,
            blosc2_sys::BLOSC2_ERROR_RUN_LENGTH => Error::RunLength,
            blosc2_sys::BLOSC2_ERROR_FILTER_PIPELINE => Error::FilterPipeline,
            blosc2_sys::BLOSC2_ERROR_CHUNK_INSERT => Error::ChunkInsert,
            blosc2_sys::BLOSC2_ERROR_CHUNK_APPEND => Error::ChunkAppend,
            blosc2_sys::BLOSC2_ERROR_CHUNK_UPDATE => Error::ChunkUpdate,
            blosc2_sys::BLOSC2_ERROR_2GB_LIMIT => Error::TwoGbLimit,
            blosc2_sys::BLOSC2_ERROR_SCHUNK_COPY => Error::SchunkCopy,
            blosc2_sys::BLOSC2_ERROR_FRAME_TYPE => Error::FrameType,
            blosc2_sys::BLOSC2_ERROR_FILE_TRUNCATE => Error::FileTruncate,
            blosc2_sys::BLOSC2_ERROR_THREAD_CREATE => Error::ThreadCreate,
            blosc2_sys::BLOSC2_ERROR_POSTFILTER => Error::Postfilter,
            blosc2_sys::BLOSC2_ERROR_FRAME_SPECIAL => Error::FrameSpecial,
            blosc2_sys::BLOSC2_ERROR_SCHUNK_SPECIAL => Error::SChunkSpecial,
            blosc2_sys::BLOSC2_ERROR_PLUGIN_IO => Error::PluginIO,
            blosc2_sys::BLOSC2_ERROR_FILE_REMOVE => Error::FileRemove,
            blosc2_sys::BLOSC2_ERROR_NULL_POINTER => Error::NullPointer,
            blosc2_sys::BLOSC2_ERROR_INVALID_INDEX => Error::InvalidIndex,
            blosc2_sys::BLOSC2_ERROR_METALAYER_NOT_FOUND => Error::MetalayerNotFound,
            blosc2_sys::BLOSC2_ERROR_MAX_BUFSIZE_EXCEEDED => Error::MaxBufsizeExceeded,
            blosc2_sys::BLOSC2_ERROR_TUNER => Error::Tuner,
            unknown => {
                eprintln!("Unknown blosc2 error code: {unknown}");
                Error::Failure
            }
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Failure => f.write_str("generic failure"),
            Error::Stream => f.write_str("bad stream"),
            Error::Data => f.write_str("invalid data"),
            Error::MemoryAlloc => f.write_str("memory alloc/realloc failure"),
            Error::ReadBuffer => f.write_str("not enough space to read"),
            Error::WriteBuffer => f.write_str("not enough space to write"),
            Error::CodecSupport => f.write_str("codec not supported"),
            Error::CodecParam => f.write_str("invalid parameter supplied to codec"),
            Error::CodecDict => f.write_str("codec dictionary error"),
            Error::VersionSupport => f.write_str("version not supported"),
            Error::InvalidHeader => f.write_str("invalid value in header"),
            Error::InvalidParam => f.write_str("invalid parameter supplied to function"),
            Error::FileRead => f.write_str("file read failure"),
            Error::FileWrite => f.write_str("file write failure"),
            Error::FileOpen => f.write_str("file open failure"),
            Error::NotFound => f.write_str("not found"),
            Error::RunLength => f.write_str("bad run length encoding"),
            Error::FilterPipeline => f.write_str("filter pipeline error"),
            Error::ChunkInsert => f.write_str("chunk insert failure"),
            Error::ChunkAppend => f.write_str("chunk append failure"),
            Error::ChunkUpdate => f.write_str("chunk update failure"),
            Error::TwoGbLimit => f.write_str("sizes larger than 2gb not supported"),
            Error::SchunkCopy => f.write_str("super-chunk copy failure"),
            Error::FrameType => f.write_str("wrong type for frame"),
            Error::FileTruncate => f.write_str("file truncate failure"),
            Error::ThreadCreate => f.write_str("thread or thread context creation failure"),
            Error::Postfilter => f.write_str("postfilter failure"),
            Error::FrameSpecial => f.write_str("special frame failure"),
            Error::SChunkSpecial => f.write_str("special super-chunk failure"),
            Error::PluginIO => f.write_str("IO plugin error"),
            Error::FileRemove => f.write_str("remove file failure"),
            Error::NullPointer => f.write_str("pointer is null"),
            Error::InvalidIndex => f.write_str("invalid index"),
            Error::MetalayerNotFound => f.write_str("metalayer has not been found"),
            Error::MaxBufsizeExceeded => f.write_str("max buffer size exceeded"),
            Error::Tuner => f.write_str("tuner failure"),
        }
    }
}
impl std::error::Error for Error {}

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
    pub fn compressor(&mut self, compressor: CompressAlgo) -> &mut Self {
        self.0.compcode = compressor as _;
        self
    }
    #[must_use]
    pub fn clevel(&mut self, clevel: u32) -> &mut Self {
        self.0.clevel = clevel as u8;
        self
    }
    #[must_use]
    pub fn typesize(&mut self, typesize: usize) -> &mut Self {
        self.0.typesize = typesize as i32;
        self
    }
    #[must_use]
    pub fn nthreads(&mut self, nthreads: usize) -> &mut Self {
        self.0.nthreads = nthreads as i16;
        self
    }
    #[must_use]
    pub fn blocksize(&mut self, blocksize: usize) -> &mut Self {
        self.0.blocksize = blocksize as i32;
        self
    }
    #[must_use]
    pub fn splitmode(&mut self, splitmode: SplitMode) -> &mut Self {
        self.0.splitmode = splitmode as _;
        self
    }
    #[must_use]
    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self, Error> {
        if filters.len() > 6 {
            return Err(Error::InvalidParam);
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
        Ok(self)
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
    pub fn nthreads(&mut self, nthreads: usize) -> &mut Self {
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
