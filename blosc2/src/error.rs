/// Error codes for the blosc2 library.
///
/// Note that the error are codes that do not contain any additional information.
/// For debugging purposes, you can set the environment variable `BLOSC_TRACE` to get many more
/// trace prints that can help you understand what went wrong.

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
                eprintln!("Unknown blosc2 error code: {unknown}");
                Error::Failure
            }
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Failure => f.write_str("Generic failure"),
            Error::Stream => f.write_str("Bad stream"),
            Error::Data => f.write_str("Invalid data"),
            Error::MemoryAlloc => f.write_str("Memory alloc/realloc failure"),
            Error::ReadBuffer => f.write_str("Not enough space to read"),
            Error::WriteBuffer => f.write_str("Not enough space to write"),
            Error::CodecSupport => f.write_str("Codec not supported"),
            Error::CodecParam => f.write_str("Invalid parameter supplied to codec"),
            Error::CodecDict => f.write_str("Codec dictionary error"),
            Error::VersionSupport => f.write_str("Version not supported"),
            Error::InvalidHeader => f.write_str("Invalid value in header"),
            Error::InvalidParam => f.write_str("Invalid parameter supplied to function"),
            Error::FileRead => f.write_str("File read failure"),
            Error::FileWrite => f.write_str("File write failure"),
            Error::FileOpen => f.write_str("File open failure"),
            Error::NotFound => f.write_str("Not found"),
            Error::RunLength => f.write_str("Bad run length encoding"),
            Error::FilterPipeline => f.write_str("Filter pipeline error"),
            Error::ChunkInsert => f.write_str("Chunk insert failure"),
            Error::ChunkAppend => f.write_str("Chunk append failure"),
            Error::ChunkUpdate => f.write_str("Chunk update failure"),
            Error::TwoGbLimit => f.write_str("Sizes larger than 2gb not supported"),
            Error::SchunkCopy => f.write_str("Super-chunk copy failure"),
            Error::FrameType => f.write_str("Wrong type for frame"),
            Error::FileTruncate => f.write_str("File truncate failure"),
            Error::ThreadCreate => f.write_str("Thread or thread context creation failure"),
            Error::Postfilter => f.write_str("Postfilter failure"),
            Error::FrameSpecial => f.write_str("Special frame failure"),
            Error::SChunkSpecial => f.write_str("Special super-chunk failure"),
            Error::PluginIO => f.write_str("IO plugin error"),
            Error::FileRemove => f.write_str("Remove file failure"),
            Error::NullPointer => f.write_str("Pointer is null"),
            Error::InvalidIndex => f.write_str("Invalid index"),
            Error::MetalayerNotFound => f.write_str("Metalayer has not been found"),
            Error::MaxBufsizeExceeded => f.write_str("Maximum buffersize exceeded"),
            Error::Tuner => f.write_str("Tuner failure"),
        }
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
