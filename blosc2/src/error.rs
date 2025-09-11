/// Error codes for the blosc2 library.
///
/// Note that the error are codes that do not contain any additional information.
/// For debugging purposes, set the environment variable `BLOSC_TRACE` to get many more
/// trace prints that can help understand what went wrong.

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
    pub(crate) fn from_int(code: core::ffi::c_int) -> Self {
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
                crate::trace!("Unknown blosc2 error code: {}", unknown);
                Error::Failure
            }
        }
    }

    pub(crate) fn to_int(&self) -> core::ffi::c_int {
        match self {
            Error::Failure => blosc2_sys::BLOSC2_ERROR_FAILURE,
            Error::Stream => blosc2_sys::BLOSC2_ERROR_STREAM,
            Error::Data => blosc2_sys::BLOSC2_ERROR_DATA,
            Error::MemoryAlloc => blosc2_sys::BLOSC2_ERROR_MEMORY_ALLOC,
            Error::ReadBuffer => blosc2_sys::BLOSC2_ERROR_READ_BUFFER,
            Error::WriteBuffer => blosc2_sys::BLOSC2_ERROR_WRITE_BUFFER,
            Error::CodecSupport => blosc2_sys::BLOSC2_ERROR_CODEC_SUPPORT,
            Error::CodecParam => blosc2_sys::BLOSC2_ERROR_CODEC_PARAM,
            Error::CodecDict => blosc2_sys::BLOSC2_ERROR_CODEC_DICT,
            Error::VersionSupport => blosc2_sys::BLOSC2_ERROR_VERSION_SUPPORT,
            Error::InvalidHeader => blosc2_sys::BLOSC2_ERROR_INVALID_HEADER,
            Error::InvalidParam => blosc2_sys::BLOSC2_ERROR_INVALID_PARAM,
            Error::FileRead => blosc2_sys::BLOSC2_ERROR_FILE_READ,
            Error::FileWrite => blosc2_sys::BLOSC2_ERROR_FILE_WRITE,
            Error::FileOpen => blosc2_sys::BLOSC2_ERROR_FILE_OPEN,
            Error::NotFound => blosc2_sys::BLOSC2_ERROR_NOT_FOUND,
            Error::RunLength => blosc2_sys::BLOSC2_ERROR_RUN_LENGTH,
            Error::FilterPipeline => blosc2_sys::BLOSC2_ERROR_FILTER_PIPELINE,
            Error::ChunkInsert => blosc2_sys::BLOSC2_ERROR_CHUNK_INSERT,
            Error::ChunkAppend => blosc2_sys::BLOSC2_ERROR_CHUNK_APPEND,
            Error::ChunkUpdate => blosc2_sys::BLOSC2_ERROR_CHUNK_UPDATE,
            Error::TwoGbLimit => blosc2_sys::BLOSC2_ERROR_2GB_LIMIT,
            Error::SchunkCopy => blosc2_sys::BLOSC2_ERROR_SCHUNK_COPY,
            Error::FrameType => blosc2_sys::BLOSC2_ERROR_FRAME_TYPE,
            Error::FileTruncate => blosc2_sys::BLOSC2_ERROR_FILE_TRUNCATE,
            Error::ThreadCreate => blosc2_sys::BLOSC2_ERROR_THREAD_CREATE,
            Error::Postfilter => blosc2_sys::BLOSC2_ERROR_POSTFILTER,
            Error::FrameSpecial => blosc2_sys::BLOSC2_ERROR_FRAME_SPECIAL,
            Error::SChunkSpecial => blosc2_sys::BLOSC2_ERROR_SCHUNK_SPECIAL,
            Error::PluginIO => blosc2_sys::BLOSC2_ERROR_PLUGIN_IO,
            Error::FileRemove => blosc2_sys::BLOSC2_ERROR_FILE_REMOVE,
            Error::NullPointer => blosc2_sys::BLOSC2_ERROR_NULL_POINTER,
            Error::InvalidIndex => blosc2_sys::BLOSC2_ERROR_INVALID_INDEX,
            Error::MetalayerNotFound => blosc2_sys::BLOSC2_ERROR_METALAYER_NOT_FOUND,
            Error::MaxBufsizeExceeded => blosc2_sys::BLOSC2_ERROR_MAX_BUFSIZE_EXCEEDED,
            Error::Tuner => blosc2_sys::BLOSC2_ERROR_TUNER,
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = unsafe { blosc2_sys::blosc2_error_string(self.to_int()) };
        assert!(!err_str.is_null());
        let len = unsafe { libc::strlen(err_str) };
        let err_str: &'static [u8] = unsafe { std::slice::from_raw_parts(err_str.cast(), len + 1) };
        let err_str = std::ffi::CStr::from_bytes_with_nul(err_str).unwrap();
        let err_str = err_str.to_str().unwrap();
        f.write_str(err_str)
    }
}
impl std::error::Error for Error {}

pub(crate) trait ErrorCode: Sized {
    fn into_result(self) -> Result<Self, Error>;
}
macro_rules! impl_status_check {
    ($t:ty) => {
        impl ErrorCode for $t {
            fn into_result(self) -> Result<Self, Error> {
                if self < 0 {
                    Err(Error::from_int(self as _))
                } else {
                    Ok(self)
                }
            }
        }
    };
}
impl_status_check!(i32);
impl_status_check!(i64);
