use std::cell::RefCell;
use std::mem::MaybeUninit;

use crate::error::ErrorCode;
use crate::util::{validate_compressed_buf_and_get_sizes, CowVec};
use crate::{DParams, Decoder, Error};

/// A chunk of compressed data.
///
/// A chunk is a thin wrapper around `CowVec<u8>`, with some additional metadata, and the creation
/// of a `Chunk` through the [`Self::from_compressed`] is very cheap.
pub struct Chunk<'a> {
    pub(crate) buffer: CowVec<'a, u8>,
    pub(crate) nbytes: usize,
    pub(crate) typesize: usize,
    pub(crate) decoder: RefCell<Option<Decoder>>,
}
impl<'a> Chunk<'a> {
    /// Create a new `Chunk` from a compressed bytes buffer.
    pub fn from_compressed(bytes: CowVec<'a, u8>) -> Result<Self, Error> {
        let (nbytes, _cbytes, _blocksize) =
            validate_compressed_buf_and_get_sizes(bytes.as_slice())?;

        let mut typesize = MaybeUninit::uninit();
        let mut flags = MaybeUninit::uninit();
        unsafe {
            blosc2_sys::blosc1_cbuffer_metainfo(
                bytes.as_slice().as_ptr().cast(),
                typesize.as_mut_ptr(),
                flags.as_mut_ptr(),
            );
        }
        let typesize = unsafe { typesize.assume_init() };
        // let flags = unsafe { flags.assume_init() };
        if typesize == 0 {
            return Err(Error::Failure);
        }

        Ok(unsafe { Self::from_compressed_unchecked(bytes, nbytes as usize, typesize) })
    }

    pub(crate) unsafe fn from_compressed_unchecked(
        bytes: CowVec<'a, u8>,
        nbytes: usize,
        typesize: usize,
    ) -> Self {
        Self {
            buffer: bytes,
            nbytes,
            typesize,
            decoder: RefCell::new(None),
        }
    }

    /// Get a reference to the underlying bytes buffer.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// Convert the chunk into a bytes buffer.
    pub fn into_bytes(self) -> CowVec<'a, u8> {
        self.buffer
    }

    fn decoder_mut(&self) -> Result<std::cell::RefMut<'_, Decoder>, Error> {
        let mut decoder = self.decoder.borrow_mut();
        if decoder.is_none() {
            *decoder = Some(Decoder::new(Default::default())?);
        }
        Ok(std::cell::RefMut::map(decoder, |d| d.as_mut().unwrap()))
    }

    /// Get the current decompression parameters.
    pub fn get_dparams(&self) -> DParams {
        self.decoder
            .borrow()
            .as_ref()
            .map(|decoder| decoder.params())
            .unwrap_or_default()
    }

    /// Set the decompression parameters.
    pub fn set_dparams(&self, params: DParams) -> Result<(), Error> {
        *self.decoder.borrow_mut() = Some(Decoder::new(params)?);
        Ok(())
    }

    /// Decompress the whole chunk into a new allocated bytes vector.
    ///
    /// For decompressing the data into an already allocated buffer, create a `Decoder`
    /// (with other params or [`Self::get_dparams`]) and use [`Decoder::decompress_into`].
    pub fn decompress(&self) -> Result<Vec<u8>, Error> {
        self.decoder_mut()?.decompress(self.buffer.as_slice())
    }

    /// Get the number of bytes in the *uncompressed* data.
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    /// Get the size of each item in the chunk.
    pub fn typesize(&self) -> usize {
        self.typesize
    }

    /// Get the number of items in the chunk.
    pub fn items_num(&self) -> usize {
        self.nbytes / self.typesize
    }

    /// Get an item at the specified index.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that the returned vector may not be aligned to the original data type's alignment, and the caller should
    /// ensure that the alignment is correct before transmuting it to original type. If the alignment does not match
    /// the original data type, the bytes should be copied to a new aligned allocation before transmuting, otherwise
    /// undefined behavior may occur. Alternatively, the caller can use [`Self::item_into`] and provide an already
    /// aligned destination buffer.
    ///
    pub fn item(&self, idx: usize) -> Result<Vec<u8>, Error> {
        self.items(idx..idx + 1)
    }

    /// Get an item at the specified index and copy it into the provided destination buffer.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that if the destination buffer is not aligned to the original data type's alignment, the caller should
    /// not transmute the decompressed data to original type, as this may lead to undefined behavior.
    pub fn item_into(&self, idx: usize, dst: &mut [MaybeUninit<u8>]) -> Result<usize, Error> {
        self.items_into(idx..idx + 1, dst)
    }

    /// Get a range of items specified by the index range.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that the returned vector may not be aligned to the original data type's alignment, and the caller should
    /// ensure that the alignment is correct before transmuting it to original type. If the alignment does not match
    /// the original data type, the bytes should be copied to a new aligned allocation before transmuting, otherwise
    /// undefined behavior may occur. Alternatively, the caller can use [`Self::items_into`] and provide an already
    /// aligned destination buffer.
    pub fn items(&self, idx: std::ops::Range<usize>) -> Result<Vec<u8>, Error> {
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(self.typesize * idx.len());
        unsafe { dst.set_len(self.typesize * idx.len()) };
        let len = self.items_into(idx, &mut dst)?;

        assert!(len <= dst.len());
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        Ok(unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) })
    }

    /// Get a range of items specified by the index range and copy them into the provided destination buffer.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that if the destination buffer is not aligned to the original data type's alignment, the caller should
    /// not transmute the decompressed data to original type, as this may lead to undefined behavior.
    pub fn items_into(
        &self,
        idx: std::ops::Range<usize>,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        Ok(unsafe {
            blosc2_sys::blosc2_getitem_ctx(
                self.decoder_mut()?.ctx_ptr(),
                self.buffer.as_slice().as_ptr().cast(),
                self.buffer.as_slice().len() as _,
                idx.start as _,
                idx.len() as _,
                dst.as_mut_ptr().cast(),
                dst.len() as _,
            )
            .into_result()? as usize
        })
    }

    /// Create a shallow clone of the chunk without re-allocating the internal buffer.
    pub fn shallow_clone(&self) -> Chunk {
        Chunk {
            buffer: CowVec::Borrowed(self.buffer.as_slice()),
            nbytes: self.nbytes,
            typesize: self.typesize,
            decoder: RefCell::new(self.copy_decoder()),
        }
    }

    fn copy_decoder(&self) -> Option<Decoder> {
        self.decoder
            .borrow()
            .as_ref()
            .and_then(|decoder| Decoder::new(decoder.params()).ok( /* On error, ignore and dont copy the decoder */))
    }
}
impl Clone for Chunk<'_> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            nbytes: self.nbytes,
            typesize: self.typesize,
            decoder: RefCell::new(self.copy_decoder()),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use rand::prelude::*;

    use crate::chunk::Chunk;
    use crate::util::tests::{rand_cparams, rand_dparams, rand_src_len};
    use crate::{CParams, Encoder};

    #[test]
    fn get_item() {
        let mut rand = StdRng::seed_from_u64(0xb47b5287627f3d57);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let len = rand_src_len(cparams.get_typesize(), &mut rand);
            let data = rand_chunk_data(len, &mut rand);
            let chunk = Encoder::new(cparams.clone())
                .unwrap()
                .compress(&data)
                .unwrap();

            chunk.set_dparams(dparams).unwrap();
            assert_eq!(cparams.get_typesize(), chunk.typesize());
            assert_eq!(len, chunk.nbytes());
            assert_eq!(len / chunk.typesize(), chunk.items_num());
            if chunk.items_num() > 0 {
                for _ in 0..10 {
                    let idx = rand.random_range(0..chunk.items_num());
                    let item = chunk.item(idx).unwrap();
                    assert_eq!(
                        item,
                        data[idx * chunk.typesize()..(idx + 1) * chunk.typesize()]
                    );
                }
                for _ in 0..10 {
                    let start = rand.random_range(0..chunk.items_num());
                    let end = rand.random_range(start..chunk.items_num());
                    let items = chunk.items(start..end).unwrap();
                    assert_eq!(
                        items,
                        data[start * chunk.typesize()..end * chunk.typesize()]
                    );
                }
            }
        }
    }

    pub(crate) fn rand_chunk(size: usize, params: CParams, rand: &mut StdRng) -> Chunk<'static> {
        let data = rand_chunk_data(size, rand);
        Encoder::new(params).unwrap().compress(&data).unwrap()
    }

    pub(crate) fn rand_chunk_data(size: usize, rand: &mut StdRng) -> Vec<u8> {
        rand.random_iter().take(size).collect()
    }
}
