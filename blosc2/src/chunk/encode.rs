use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::chunk::Chunk;
use crate::error::{Error, ErrorCode};
use crate::util::validate_compressed_buf_and_get_sizes;
use crate::{CParams, DParams};

struct Context(NonNull<blosc2_sys::blosc2_context>);
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { blosc2_sys::blosc2_free_ctx(self.0.as_ptr()) }
    }
}

/// An encoder for compressing bytes into [`Chunk`].
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
    ///
    /// # Arguments
    ///
    /// * `src` - The source bytes to compress. Must be a multiple of the item size.
    ///
    /// # Returns
    ///
    /// A `Chunk` containing the compressed data.
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
    /// # Arguments
    ///
    /// * `src` - The source bytes to compress. Must be a multiple of the item size.
    /// * `dst` - The destination buffer to write the compressed data into. After the function call the valid part of
    ///   the buffer will contain the compressed data, and can be interpreted as a `Chunk`.
    ///   See [`Chunk::from_compressed`].
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
    /// * `value` - The value to repeat. See [`RepeatedValue`] for details.
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
#[non_exhaustive]
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

/// A decoder for decompressing bytes from a [`Chunk`].
///
/// Functions of the decoder expect bytes slices, rather than an actual `Chunk` struct, but the bytes are expected to
/// be a `Chunk`s bytes.
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
    ///
    /// # Arguments
    ///
    /// * `src` - The source bytes to decompress. Should be a [`Chunk`]'s bytes.
    ///
    /// # Returns
    ///
    /// A vector containing the decompressed bytes, of size `itemsize * items_num`.
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
    /// # Arguments
    ///
    /// * `src` - The source bytes to decompress. Should be a [`Chunk`]'s bytes.
    /// * `dst` - The destination buffer to write the decompressed data into.
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer, `itemsize * items_num`.
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

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use super::{Decoder, Encoder, RepeatedValue};
    use crate::util::tests::{rand_cparams, rand_dparams, rand_src_len};

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
                        _ => panic!("Unexpected typesize for NaN: {typesize}"),
                    },
                    RepeatedValue::Uninit => {}
                    RepeatedValue::Value(v) => assert_eq!(item, v),
                }
            }
        }
    }
}
