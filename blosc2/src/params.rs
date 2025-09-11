use crate::Error;

/// Compression algorithms supported by blosc2.
///
/// The library itself always uses some "backend" compression algorithm, such as `blosclz`, `lz4`,
/// `zlib`, or `zstd`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum SplitMode {
    Always = blosc2_sys::BLOSC_ALWAYS_SPLIT as _,
    Never = blosc2_sys::BLOSC_NEVER_SPLIT as _,
    Auto = blosc2_sys::BLOSC_AUTO_SPLIT as _,
    ForwardCompat = blosc2_sys::BLOSC_FORWARD_COMPAT_SPLIT as _,
}

/// Compression parameters, used by [`Encoder`](crate::chunk::Encoder), [`SChunk`](crate::chunk::SChunk) and [`Ndarray`](crate::nd::Ndarray).
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
    /// The typesize must be in range [1, 255].
    ///
    /// By default, the typesize is set to 8 bytes.
    pub fn typesize(&mut self, typesize: usize) -> Result<&mut Self, Error> {
        if !(1..=blosc2_sys::BLOSC_MAX_TYPESIZE as usize).contains(&typesize) {
            crate::trace!(
                "Itemsize {} is greater than BLOSC_MAX_TYPESIZE {}",
                typesize,
                blosc2_sys::BLOSC_MAX_TYPESIZE
            );
            return Err(Error::InvalidParam);
        }
        self.0.typesize = typesize as _;
        Ok(self)
    }
    /// Get the typesize currently set in the parameters.
    pub fn get_typesize(&self) -> usize {
        debug_assert!(self.0.typesize > 0);
        self.0.typesize as usize
    }

    /// Set the number of threads to use for compression.
    ///
    /// By default, the number of threads is set to 1.
    pub fn nthreads(&mut self, nthreads: usize) -> &mut Self {
        self.0.nthreads = nthreads.max(1) as i16;
        self
    }
    /// Get the number of threads currently set in the parameters.
    pub fn get_nthreads(&self) -> usize {
        self.0.nthreads as usize
    }

    /// Set the block size for compression.
    ///
    /// `None` means automatic block size.
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
    ///
    /// `None` means automatic block size.
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

/// Decompression parameters, used by [`Decoder`](crate::chunk::Decoder), [`SChunk`](crate::chunk::SChunk) and [`Ndarray`](crate::nd::Ndarray).
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
    pub fn nthreads(&mut self, nthreads: usize) -> &mut Self {
        self.0.nthreads = nthreads.max(1) as i16;
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
