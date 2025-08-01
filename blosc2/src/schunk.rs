use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::{path2cstr, BytesMaybePassOwnershipToC, CowVec};
use crate::{CParams, Chunk, DParams, Error};

/// A super chunk (SChunk) is a collection of compressed chunks that are treated as a single entity.
///
/// It can be stored in memory or on disk, and support inserting, updating, deleting and appending
/// chunks or data to it. It support random access to either chunks or items in the chunks.
pub struct SChunk(NonNull<blosc2_sys::blosc2_schunk>);
impl SChunk {
    fn from_ptr(ptr: *mut blosc2_sys::blosc2_schunk) -> Result<Self, Error> {
        Ok(Self(NonNull::new(ptr).ok_or(Error::Failure)?))
    }

    /// Create a new in-memory super chunk.
    ///
    /// The created super chunk will not be contiguous. See [`Self::new`] for more details.
    pub fn new_in_memory(cparams: CParams, dparams: DParams) -> Result<Self, Error> {
        Self::new(false, None, cparams, dparams)
    }

    /// Create a new on-disk super chunk at the given path.
    ///
    /// The created super chunk will not be contiguous. See [`Self::new`] for more details.
    pub fn new_on_disk(urlpath: &Path, cparams: CParams, dparams: DParams) -> Result<Self, Error> {
        Self::new(false, Some(urlpath), cparams, dparams)
    }

    /// Create a new super chunk with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `contiguous` - If true, the super chunk will be stored in a contiguous memory block.
    ///   Note that contiguous super chunks are very inefficient for update operations.
    /// * `urlpath` - If `Some(path)`, the super chunk will be stored on disk at the specified path.
    ///   If `None`, the super chunk will be stored in memory.
    /// * `cparams` - Compression parameters used to compress new chunks added to the super chunk.
    /// * `dparams` - Decompression parameters used to decompress chunks from the super chunk.
    pub fn new(
        contiguous: bool,
        urlpath: Option<&Path>,
        cparams: CParams,
        dparams: DParams,
    ) -> Result<Self, Error> {
        crate::global::global_init();

        let urlpath = urlpath.map(path2cstr);
        let urlpath = urlpath
            .as_ref()
            .map(|p| p.as_ptr().cast_mut())
            .unwrap_or(std::ptr::null_mut());

        let mut storage = blosc2_sys::blosc2_storage {
            contiguous,
            urlpath,
            cparams: (&cparams.0 as *const blosc2_sys::blosc2_cparams).cast_mut(),
            dparams: (&dparams.0 as *const blosc2_sys::blosc2_dparams).cast_mut(),
            io: std::ptr::null_mut(),
        };
        Self::from_ptr(unsafe { blosc2_sys::blosc2_schunk_new(&mut storage as *mut _) })
    }

    /// Open an existing super chunk from the specified path.
    pub fn open(urlpath: &Path) -> Result<Self, Error> {
        Self::open_with_offset(urlpath, 0)
    }

    /// Open an existing super chunk from the specified path with an offset.
    pub fn open_with_offset(urlpath: &Path, offset: usize) -> Result<Self, Error> {
        let urlpath = path2cstr(urlpath);
        Self::from_ptr(unsafe {
            blosc2_sys::blosc2_schunk_open_offset(urlpath.as_ptr().cast_mut(), offset as _)
        })
    }

    /// Create a super chunk from an existing in-memory buffer.
    pub fn from_buffer(buffer: CowVec<u8>) -> Result<Self, Error> {
        let buffer = BytesMaybePassOwnershipToC::new(buffer);
        let schunk = unsafe {
            blosc2_sys::blosc2_schunk_from_buffer(
                buffer.as_slice().as_ptr().cast_mut(),
                buffer.as_slice().len() as _,
                !buffer.ownership_moved(),
            )
        };
        Self::from_ptr(schunk)
    }

    /// Serialize the super chunk to an in-memory buffer.
    pub fn to_buffer(&mut self) -> Result<CowVec<u8>, Error> {
        let mut buffer = MaybeUninit::uninit();
        let mut needs_free = MaybeUninit::uninit();
        let len = unsafe {
            blosc2_sys::blosc2_schunk_to_buffer(
                self.0.as_ptr(),
                buffer.as_mut_ptr(),
                needs_free.as_mut_ptr(),
            )
            .into_result()? as usize
        };

        let buffer = NonNull::new(unsafe { buffer.assume_init() }).ok_or(Error::Failure)?;
        let needs_free = unsafe { needs_free.assume_init() };
        Ok(unsafe { CowVec::from_c_buf(buffer, len, needs_free) })
    }

    /// Serialize the super chunk to a file.
    ///
    /// # Arguments
    /// * `urlpath` - The path to the file where the super chunk will be saved.
    /// * `append` - If true, the super chunk will be appended to the file. If false, the file
    ///   should not exist, otherwise an error will be returned.
    pub fn to_file(&mut self, urlpath: &Path, append: bool) -> Result<(), Error> {
        let urlpath = path2cstr(urlpath);
        unsafe {
            if append {
                blosc2_sys::blosc2_schunk_append_file(self.0.as_ptr(), urlpath.as_ptr().cast_mut())
                    .into_result()?;
            } else {
                blosc2_sys::blosc2_schunk_to_file(self.0.as_ptr(), urlpath.as_ptr().cast_mut())
                    .into_result()?;
            }
        }
        Ok(())
    }

    /// Append (uncompressed) data to the super chunk as a new chunk.
    ///
    /// The data will be compressed using the compression parameters of the super chunk.
    pub fn append(&mut self, data: &[u8]) -> Result<(), Error> {
        if data.is_empty() {
            crate::trace!("Empty chunk is not allowed");
            return Err(Error::ReadBuffer);
        }

        // the size of chunks must be the same for every chunk
        unsafe {
            blosc2_sys::blosc2_schunk_append_buffer(
                self.0.as_ptr(),
                data.as_ptr().cast(),
                data.len() as _,
            )
            .into_result()?;
        };
        Ok(())
    }

    /// Append a compressed chunk to the super chunk.
    pub fn append_chunk(&mut self, chunk: Chunk) -> Result<(), Error> {
        self.check_chunk_compatible(&chunk)?;
        let chunk = BytesMaybePassOwnershipToC::new(chunk.into_bytes());
        unsafe {
            blosc2_sys::blosc2_schunk_append_chunk(
                self.0.as_ptr(),
                chunk.as_slice().as_ptr().cast_mut(),
                !chunk.ownership_moved(),
            )
            .into_result()?;
        }
        Ok(())
    }

    /// Update an existing chunk in the super chunk.
    ///
    /// The new chunk will replace the existing chunk at the specified index.
    pub fn update_chunk(&mut self, index: usize, chunk: Chunk) -> Result<(), Error> {
        self.check_chunk_compatible(&chunk)?;
        let chunk = BytesMaybePassOwnershipToC::new(chunk.into_bytes());
        unsafe {
            blosc2_sys::blosc2_schunk_update_chunk(
                self.0.as_ptr(),
                index as _,
                chunk.as_slice().as_ptr().cast_mut(),
                !chunk.ownership_moved(),
            )
            .into_result()?;
        }
        Ok(())
    }

    /// Insert a new chunk at the specified index in the super chunk.
    pub fn insert_chunk(&mut self, index: usize, chunk: Chunk) -> Result<(), Error> {
        self.check_chunk_compatible(&chunk)?;
        let chunk = BytesMaybePassOwnershipToC::new(chunk.into_bytes());
        unsafe {
            blosc2_sys::blosc2_schunk_insert_chunk(
                self.0.as_ptr(),
                index as _,
                chunk.as_slice().as_ptr().cast_mut(),
                !chunk.ownership_moved(),
            )
            .into_result()?;
        }
        Ok(())
    }

    fn check_chunk_compatible(&self, chunk: &Chunk) -> Result<(), Error> {
        if chunk.typesize() != self.typesize() {
            crate::trace!(
                "Chunk typesize {} does not match schunk typesize {}",
                chunk.typesize(),
                self.typesize()
            );
            return Err(Error::InvalidParam);
        }
        Ok(())
    }

    /// Delete a chunk at the specified index from the super chunk.
    pub fn delete_chunk(&mut self, index: usize) -> Result<(), Error> {
        unsafe {
            blosc2_sys::blosc2_schunk_delete_chunk(self.0.as_ptr(), index as _).into_result()?;
        }
        Ok(())
    }

    /// Get a chunk at the specified index from the super chunk.
    pub fn get_chunk(&mut self, index: usize) -> Result<Chunk, Error> {
        let mut ptr = MaybeUninit::uninit();
        let mut needs_free = MaybeUninit::uninit();
        let len = unsafe {
            blosc2_sys::blosc2_schunk_get_chunk(
                self.0.as_ptr(),
                index as _,
                ptr.as_mut_ptr(),
                needs_free.as_mut_ptr(),
            )
            .into_result()? as usize
        };
        let ptr = NonNull::new(unsafe { ptr.assume_init() }).ok_or(Error::Failure)?;
        let needs_free = unsafe { needs_free.assume_init() };
        let buf = unsafe { CowVec::from_c_buf(ptr, len, needs_free) };
        let chunk = Chunk::from_compressed(buf)?;
        chunk.set_dparams(self.dparams())?;
        Ok(chunk)
    }

    /// Decompress a chunk at the specified index into a new allocated bytes vector.
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer.
    pub fn decompress_chunk_into(
        &mut self,
        index: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        let len = unsafe {
            blosc2_sys::blosc2_schunk_decompress_chunk(
                self.0.as_ptr(),
                index as _,
                dst.as_mut_ptr().cast(),
                dst.len() as _,
            )
            .into_result()? as usize
        };
        debug_assert!(len <= dst.len());
        Ok(len)
    }

    /// Copy the super chunk to a new in-memory super chunk.
    ///
    /// The created super chunk will not be contiguous. See [`Self::copy`] for more details.
    pub fn copy_to_memory(&self, cparams: CParams, dparams: DParams) -> Result<SChunk, Error> {
        self.copy(false, None, cparams, dparams)
    }

    /// Copy the super chunk to a new on-disk super chunk at the specified path.
    ///
    /// The created super chunk will not be contiguous. See [`Self::copy`] for more details.
    pub fn copy_to_disk(
        &self,
        urlpath: &Path,
        cparams: CParams,
        dparams: DParams,
    ) -> Result<SChunk, Error> {
        self.copy(false, Some(urlpath), cparams, dparams)
    }

    /// Copy the super chunk to a new super chunk with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `contiguous` - If true, the super chunk will be stored in a contiguous memory block.
    ///   Note that contiguous super chunks are very inefficient for update operations.
    /// * `urlpath` - If `Some(path)`, the super chunk will be stored on disk at the specified path.
    ///   If `None`, the super chunk will be stored in memory.
    /// * `cparams` - Compression parameters used to compress new chunks added to the super chunk.
    /// * `dparams` - Decompression parameters used to decompress chunks from the super chunk.
    pub fn copy(
        &self,
        contiguous: bool,
        urlpath: Option<&Path>,
        cparams: CParams,
        dparams: DParams,
    ) -> Result<SChunk, Error> {
        crate::global::global_init();

        let urlpath = urlpath.map(path2cstr);
        let urlpath = urlpath
            .as_ref()
            .map(|p| p.as_ptr().cast_mut())
            .unwrap_or(std::ptr::null_mut());

        let mut storage = blosc2_sys::blosc2_storage {
            contiguous,
            urlpath,
            cparams: (&cparams.0 as *const blosc2_sys::blosc2_cparams).cast_mut(),
            dparams: (&dparams.0 as *const blosc2_sys::blosc2_dparams).cast_mut(),
            io: std::ptr::null_mut(),
        };

        let schunk = unsafe { blosc2_sys::blosc2_schunk_copy(self.0.as_ptr(), &mut storage) };
        Self::from_ptr(schunk)
    }

    /// Get the number of chunks in the super chunk.
    pub fn num_chunks(&self) -> usize {
        unsafe { self.0.as_ref() }.nchunks as usize
    }

    /// Get the compression parameters used by this super chunk.
    pub fn cparams(&self) -> CParams {
        let mut params = MaybeUninit::uninit();
        unsafe {
            blosc2_sys::blosc2_schunk_get_cparams(self.0.as_ptr(), params.as_mut_ptr())
                .into_result()
                .unwrap();
        }
        let params = unsafe { params.assume_init() };

        let mut ret = unsafe { *params };
        ret.schunk = std::ptr::null_mut();
        unsafe { libc::free(params as *mut _) };
        CParams(ret)
    }

    /// Get the decompression parameters used by this super chunk.
    pub fn dparams(&self) -> DParams {
        let mut params = MaybeUninit::uninit();
        unsafe {
            blosc2_sys::blosc2_schunk_get_dparams(self.0.as_ptr(), params.as_mut_ptr())
                .into_result()
                .unwrap();
        }
        let params = unsafe { params.assume_init() };

        let mut ret = unsafe { *params };
        ret.schunk = std::ptr::null_mut();
        unsafe { libc::free(params as *mut _) };
        DParams(ret)
    }

    // pub fn nbytes(&self) -> usize {
    //     self.nbytes
    // }

    /// Get the size of each item in the super chunk.
    pub fn typesize(&self) -> usize {
        unsafe { self.0.as_ref() }.typesize as usize
    }

    /// Get the number of items in the super chunk.
    ///
    /// The returned number is the total number of items across all chunks in the super chunk.
    /// All of the `item(s)(_into)` and `set_item(s)` methods accept indices that are zero-based
    /// in the range `0..items_num()`.
    pub fn items_num(&self) -> usize {
        self.num_chunks() * unsafe { self.0.as_ref() }.chunksize as usize / self.typesize()
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
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer.
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
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(self.typesize() * idx.len());
        unsafe { dst.set_len(self.typesize() * idx.len()) };
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
    ///
    /// # Returns
    ///
    /// The number of bytes copied into the destination buffer.
    pub fn items_into(
        &self,
        idx: std::ops::Range<usize>,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        if idx.start > idx.end || idx.end > self.items_num() {
            crate::trace!(
                "Invalid index range: {}..{} for items_num: {}",
                idx.start,
                idx.end,
                self.items_num()
            );
            return Err(Error::InvalidParam);
        }
        let required_len = idx.len() * self.typesize();
        if dst.len() < required_len {
            crate::trace!(
                "Destination buffer is too small: {} bytes required, {} bytes provided",
                required_len,
                dst.len()
            );
            return Err(Error::WriteBuffer);
        }
        unsafe {
            blosc2_sys::blosc2_schunk_get_slice_buffer(
                self.0.as_ptr(),
                idx.start as _,
                idx.end as _,
                dst.as_mut_ptr().cast(),
            )
            .into_result()?;
        }
        Ok(required_len)
    }

    /// Set an item at the specified index.
    ///
    /// Note that this will re-compress the affected chunk(s) in the super chunk.
    pub fn set_item(&mut self, idx: usize, value: &[u8]) -> Result<(), Error> {
        self.set_items(idx..idx + 1, value)
    }

    /// Set a range of items specified by the index range.
    ///
    /// Note that this will re-compress the affected chunk(s) in the super chunk.
    pub fn set_items(&mut self, idx: std::ops::Range<usize>, values: &[u8]) -> Result<(), Error> {
        let expected_length = idx.len() * self.typesize();
        if expected_length != values.len() {
            crate::trace!(
                "Expected {} bytes for {} items, got {} bytes",
                expected_length,
                idx.len(),
                values.len()
            );
            return Err(Error::InvalidParam);
        }
        unsafe {
            blosc2_sys::blosc2_schunk_set_slice_buffer(
                self.0.as_ptr(),
                idx.start as _,
                idx.end as _,
                values.as_ptr().cast_mut().cast(),
            )
            .into_result()?
        };
        Ok(())
    }
}
impl Drop for SChunk {
    fn drop(&mut self) {
        let res = unsafe { blosc2_sys::blosc2_schunk_free(self.0.as_ptr()) }.into_result();
        if let Err(err) = res {
            eprintln!("Failed to free schunk: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::mem::MaybeUninit;

    use rand::prelude::*;

    use crate::chunk::tests::{rand_chunk, rand_chunk_data};
    use crate::chunk::Chunk;
    use crate::util::tests::{ceil_to_multiple, rand_cparams, rand_dparams, rand_src_len};
    use crate::util::{CowVec, FfiVec};
    use crate::{CParams, DParams, Encoder, SChunk};

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0xbe1392d28cdfb3ec);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data(cparams.get_typesize(), &mut rand);
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for data_chunk in &data_chunks {
                schunk.append(data_chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), None);
        }
    }

    #[test]
    fn append_chunk() {
        let mut rand = StdRng::seed_from_u64(0x612356293fbd4da9);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data(cparams.get_typesize(), &mut rand);
            let chunks = data2chunks(&data_chunks, cparams.clone());
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            assert_eq!(0, schunk.num_chunks());
            for (idx, chunk) in chunks.iter().enumerate() {
                let chunk = rand_chunk_ownership(chunk, &mut rand);
                schunk.append_chunk(chunk).unwrap();
                assert_eq!(idx + 1, schunk.num_chunks());
            }
            assert_eq_chunks(schunk, Some(&data_chunks), Some(&chunks));
        }
    }

    #[test]
    fn update_chunk() {
        let mut rand = StdRng::seed_from_u64(0xd21532770bf89aaf);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data_non_empty(cparams.get_typesize(), &mut rand);
            let mut chunks = data2chunks(&data_chunks, cparams.clone());
            let mut schunk = new_schunk(cparams.clone(), dparams, &mut rand);
            let schunk = schunk.as_mut();
            for chunk in &chunks {
                let chunk = rand_chunk_ownership(chunk, &mut rand);
                schunk.append_chunk(chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), Some(&chunks));

            for _ in 0..5 {
                let index = rand.random_range(0..chunks.len());
                let new_chunk = {
                    let old_chunk = schunk.get_chunk(index).unwrap();
                    let len = old_chunk.decompress().unwrap().len();
                    rand_chunk(len, cparams.clone(), &mut rand)
                };
                let new_chunk2 = rand_chunk_ownership(&new_chunk, &mut rand);
                schunk.update_chunk(index, new_chunk2).unwrap();
                chunks[index] = new_chunk;
                assert_eq_chunks(schunk, None, Some(&chunks));
            }
        }
    }

    #[test]
    fn insert_chunk() {
        let mut rand = StdRng::seed_from_u64(0xf8180c0622cf7183);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let chunks_unordered = data2chunks(
                &rand_chunks_data(cparams.get_typesize(), &mut rand),
                cparams.clone(),
            );
            let mut chunks = Vec::new();
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            assert_eq!(0, schunk.num_chunks());
            for (idx, orig_chunk) in chunks_unordered.into_iter().enumerate() {
                let insert_index = rand.random_range(0..=chunks.len());
                let chunk = rand_chunk_ownership(&orig_chunk, &mut rand);
                schunk.insert_chunk(insert_index, chunk).unwrap();
                chunks.insert(insert_index, orig_chunk);
                assert_eq!(idx + 1, schunk.num_chunks());
            }
            assert_eq_chunks(schunk, None, Some(&chunks));
        }
    }

    #[test]
    fn delete_chunk() {
        let mut rand = StdRng::seed_from_u64(0x6f9e85f045f71a7);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data_non_empty(cparams.get_typesize(), &mut rand);
            let mut chunks = data2chunks(&data_chunks, cparams.clone());
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for chunk in &chunks {
                let chunk = rand_chunk_ownership(chunk, &mut rand);
                schunk.append_chunk(chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), Some(&chunks));

            for _ in 0..5 {
                if chunks.is_empty() {
                    break;
                }
                let index = rand.random_range(0..chunks.len());
                schunk.delete_chunk(index).unwrap();
                chunks.remove(index);
                assert_eq_chunks(schunk, None, Some(&chunks));
            }
        }
    }

    #[test]
    fn random_ops() {
        let mut rand = StdRng::seed_from_u64(0x3386d9c773ca92f8);
        for _ in 0..20 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data_non_empty(cparams.get_typesize(), &mut rand);
            let chunk_size = data_chunks.first().unwrap().len();
            let mut chunks = data2chunks(&data_chunks, cparams.clone());
            let mut schunk = new_schunk(cparams.clone(), dparams, &mut rand);
            let schunk = schunk.as_mut();
            for chunk in &chunks {
                let chunk = rand_chunk_ownership(chunk, &mut rand);
                schunk.append_chunk(chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), Some(&chunks));

            for _ in 0..20 {
                match rand.random_range(0..4) {
                    // append
                    0 => {
                        let orig_chunk = rand_chunk(chunk_size, cparams.clone(), &mut rand);
                        if rand.random::<bool>() {
                            schunk.append(&orig_chunk.decompress().unwrap()).unwrap();
                        } else {
                            let chunk = rand_chunk_ownership(&orig_chunk, &mut rand);
                            schunk.append_chunk(chunk).unwrap();
                        }
                        chunks.push(orig_chunk);
                    }
                    // insert
                    1 => {
                        let orig_chunk = rand_chunk(chunk_size, cparams.clone(), &mut rand);
                        let chunk = rand_chunk_ownership(&orig_chunk, &mut rand);
                        let idx = rand.random_range(0..=chunks.len());
                        schunk.insert_chunk(idx, chunk).unwrap();
                        chunks.insert(idx, orig_chunk);
                    }
                    // update
                    2 => {
                        if !chunks.is_empty() {
                            let orig_chunk = rand_chunk(chunk_size, cparams.clone(), &mut rand);
                            let chunk = rand_chunk_ownership(&orig_chunk, &mut rand);
                            let idx = rand.random_range(0..chunks.len());
                            schunk.update_chunk(idx, chunk).unwrap();
                            chunks[idx] = orig_chunk;
                        }
                    }
                    // delete
                    3 => {
                        if !chunks.is_empty() {
                            let idx = rand.random_range(0..chunks.len());
                            schunk.delete_chunk(idx).unwrap();
                            chunks.remove(idx);
                        }
                    }
                    _ => unreachable!(),
                }
                assert_eq_chunks(schunk, None, Some(&chunks));
            }
        }
    }

    #[test]
    fn to_file_and_open() {
        let mut rand = StdRng::seed_from_u64(0xbf070613424edd9f);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data(cparams.get_typesize(), &mut rand);
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for data_chunk in &data_chunks {
                schunk.append(data_chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), None);

            let temp_dir = tempfile::TempDir::new().unwrap();
            let urlpath = temp_dir.path().join("schunk.bin");
            let padding = rand.random::<bool>().then(|| {
                let padding = rand.random_range(0..=1024);
                File::create(&urlpath)
                    .unwrap()
                    .write_all(&vec![0u8; padding])
                    .unwrap();
                padding
            });
            schunk.to_file(&urlpath, padding.is_some()).unwrap();

            let mut schunk2 = if let Some(padding) = padding {
                SChunk::open_with_offset(&urlpath, padding).unwrap()
            } else {
                SChunk::open(&urlpath).unwrap()
            };
            assert_eq_chunks(&mut schunk2, Some(&data_chunks), None);
        }
    }

    #[test]
    fn to_buffer_and_from_buffer() {
        let mut rand = StdRng::seed_from_u64(0x917636160c63a6b7);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data(cparams.get_typesize(), &mut rand);
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for data_chunk in &data_chunks {
                schunk.append(data_chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), None);

            let buffer = schunk.to_buffer().unwrap();
            let mut schunk2 = SChunk::from_buffer(buffer).unwrap();
            assert_eq_chunks(&mut schunk2, Some(&data_chunks), None);
        }
    }

    #[test]
    fn get_item_() {
        let mut rand = StdRng::seed_from_u64(0x8b1956871d0fa980);
        for _ in 0..20 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);
            let typesize = cparams.get_typesize();

            let data_chunks = rand_chunks_data(typesize, &mut rand);
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for data_chunk in &data_chunks {
                schunk.append(data_chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), None);

            assert_eq!(
                schunk.items_num(),
                data_chunks
                    .iter()
                    .map(|c| c.len() / typesize)
                    .sum::<usize>()
            );
            assert_eq!(schunk.typesize(), typesize);
            if schunk.items_num() > 0 {
                let items_data = data_chunks
                    .iter()
                    .flat_map(|c| c.iter().cloned())
                    .collect::<Vec<u8>>()
                    .chunks_exact(typesize)
                    .map(|c| c.to_vec())
                    .collect::<Vec<_>>();

                for _ in 0..10 {
                    let idx = rand.random_range(0..schunk.items_num());
                    let item = schunk.item(idx).unwrap();
                    assert_eq!(&item, &items_data[idx]);
                }
                for _ in 0..3 {
                    let start = rand.random_range(0..schunk.items_num());
                    let end = rand.random_range(start..schunk.items_num());
                    let items = schunk.items(start..end).unwrap();
                    assert_eq!(
                        items,
                        items_data[start..end]
                            .iter()
                            .flatten()
                            .cloned()
                            .collect::<Vec<u8>>()
                    );
                }
            }
        }
    }

    #[test]
    fn set_item() {
        let mut rand = StdRng::seed_from_u64(0xc675240e4b55357d);
        for _ in 0..20 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);
            let typesize = cparams.get_typesize();

            let data_chunks = rand_chunks_data(typesize, &mut rand);
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for data_chunk in &data_chunks {
                schunk.append(data_chunk).unwrap();
            }

            let items_num = schunk.items_num();
            if items_num > 0 {
                let mut items_data = data_chunks
                    .iter()
                    .flat_map(|c| c.iter().cloned())
                    .collect::<Vec<u8>>()
                    .chunks_exact(typesize)
                    .map(|c| c.to_vec())
                    .collect::<Vec<_>>();
                for _ in 0..10 {
                    let idx = rand.random_range(0..schunk.items_num());
                    let item = (&mut rand)
                        .random_iter()
                        .take(typesize)
                        .collect::<Vec<u8>>();
                    schunk.set_item(idx, &item).unwrap();
                    items_data[idx] = item.clone();

                    assert_eq!(item, schunk.item(idx).unwrap());

                    assert_eq!(schunk.items_num(), items_num);
                    assert_eq!(
                        schunk.items(0..items_num).unwrap(),
                        items_data.iter().flatten().cloned().collect::<Vec<u8>>()
                    );
                }
                for _ in 0..3 {
                    let start = rand.random_range(0..schunk.items_num());
                    let end = rand.random_range(start..schunk.items_num());
                    let items = (0..(end - start))
                        .map(|_| {
                            (&mut rand)
                                .random_iter()
                                .take(typesize)
                                .collect::<Vec<u8>>()
                        })
                        .collect::<Vec<_>>();
                    let items_continous = items.iter().flatten().cloned().collect::<Vec<u8>>();
                    schunk.set_items(start..end, &items_continous).unwrap();
                    for (i, item) in items.iter().enumerate() {
                        items_data[start + i] = item.clone();
                    }

                    #[allow(clippy::needless_range_loop)]
                    for i in start..end {
                        assert_eq!(&items_data[i], &schunk.item(i).unwrap());
                    }
                    assert_eq!(&items_continous, &schunk.items(start..end).unwrap());

                    assert_eq!(schunk.items_num(), items_num);
                    assert_eq!(
                        schunk.items(0..items_num).unwrap(),
                        items_data.iter().flatten().cloned().collect::<Vec<u8>>()
                    );
                }
            }
        }
    }

    #[test]
    fn copy() {
        let mut rand = StdRng::seed_from_u64(0x5d8a7ff46d9529df);
        for _ in 0..30 {
            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);

            let data_chunks = rand_chunks_data(cparams.get_typesize(), &mut rand);
            let mut schunk = new_schunk(cparams, dparams, &mut rand);
            let schunk = schunk.as_mut();
            for data_chunk in &data_chunks {
                schunk.append(data_chunk).unwrap();
            }
            assert_eq_chunks(schunk, Some(&data_chunks), None);

            let cparams = rand_cparams(&mut rand);
            let dparams = rand_dparams(&mut rand);
            let temp_dir = tempfile::TempDir::new().unwrap();
            let urlpath = temp_dir.path().join("schunk-dir2");
            let mut schunk2 = match rand.random_range(0..4) {
                0 => schunk.copy_to_memory(cparams, dparams).unwrap(),
                1 => schunk.copy(true, None, cparams, dparams).unwrap(),
                2 => schunk.copy_to_disk(&urlpath, cparams, dparams).unwrap(),
                3 => schunk.copy(true, Some(&urlpath), cparams, dparams).unwrap(),
                _ => unreachable!(),
            };
            assert_eq_chunks(&mut schunk2, Some(&data_chunks), None);
        }
    }

    fn rand_chunks_data_non_empty(typesize: usize, rand: &mut StdRng) -> Vec<Vec<u8>> {
        assert!(typesize > 0);
        for _ in 0..100 {
            let data = rand_chunks_data(typesize, rand);
            if !data.is_empty() && !data.first().unwrap().is_empty() {
                return data;
            }
        }
        panic!()
    }

    fn rand_chunks_data(typesize: usize, rand: &mut StdRng) -> Vec<Vec<u8>> {
        let chunks_num = rand.random_range(0..512);
        let chunk_size = if chunks_num == 0 {
            0
        } else {
            let max_total_size = rand_src_len(typesize, rand);
            let max_chunk_size = max_total_size.div_ceil(chunks_num).max(1);
            ceil_to_multiple(rand.random_range(1..=max_chunk_size), typesize)
        };
        (0..chunks_num)
            .map(|_| rand_chunk_data(chunk_size, rand))
            .collect()
    }

    fn data2chunks(data: &[impl AsRef<[u8]>], params: CParams) -> Vec<Chunk<'static>> {
        let mut encoder = Encoder::new(params).unwrap();
        data.iter()
            .map(|d| encoder.compress(d.as_ref()).unwrap())
            .collect()
    }

    fn rand_chunk_ownership<'a>(chunk: &'a Chunk, rand: &mut StdRng) -> Chunk<'a> {
        Chunk {
            buffer: rand_bytes_ownership(&chunk.buffer, rand),
            ..chunk.shallow_clone()
        }
    }

    fn rand_bytes_ownership<'a>(bytes: &'a CowVec<u8>, rand: &mut StdRng) -> CowVec<'a, u8> {
        match rand.random_range(0..3) {
            0 => CowVec::Borrowed(bytes.as_slice()),
            1 => CowVec::OwnedRust(bytes.as_slice().to_vec()),
            2 => CowVec::OwnedFfi(FfiVec::copy_of(bytes.as_slice())),
            _ => unreachable!(),
        }
    }

    fn assert_eq_chunks(
        schunk: &mut SChunk,
        data_chunks: Option<&[Vec<u8>]>,
        chunks: Option<&[Chunk]>,
    ) {
        if let Some(data_chunks) = data_chunks {
            assert_eq!(schunk.num_chunks(), data_chunks.len());
        }
        if let Some(chunks) = chunks {
            assert_eq!(schunk.num_chunks(), chunks.len());
        }
        let mut temp_buf = Vec::<MaybeUninit<u8>>::new();
        for i in 0..schunk.num_chunks() {
            let chunk = schunk.get_chunk(i).unwrap();
            if let Some(chunks) = chunks {
                assert_eq!(chunk.as_bytes(), chunks[i].as_bytes());
            }

            let decompressed_chunk = chunk.decompress().unwrap();
            if let Some(data_chunks) = data_chunks {
                assert_eq!(&data_chunks[i], &decompressed_chunk);
            }

            temp_buf.resize(decompressed_chunk.len(), MaybeUninit::uninit());
            let decompressed_chunk2_len = schunk.decompress_chunk_into(i, &mut temp_buf).unwrap();
            let decompressed_chunk2 = unsafe {
                std::slice::from_raw_parts(temp_buf.as_ptr() as *const u8, decompressed_chunk2_len)
            };
            if let Some(data_chunks) = data_chunks {
                assert_eq!(&data_chunks[i], &decompressed_chunk2);
            }
        }
    }

    struct SChunkWrapper {
        #[allow(unused)]
        temp_dir: tempfile::TempDir,
        schunk: SChunk,
    }
    impl AsMut<SChunk> for SChunkWrapper {
        fn as_mut(&mut self) -> &mut SChunk {
            &mut self.schunk
        }
    }

    fn new_schunk(cparams: CParams, dparams: DParams, rand: &mut StdRng) -> SChunkWrapper {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let urlpath = temp_dir.path().join("schunk-dir");
        let schunk = match rand.random_range(0..4) {
            0 => SChunk::new_in_memory(cparams, dparams).unwrap(),
            1 => SChunk::new(true, None, cparams, dparams).unwrap(),
            2 => SChunk::new_on_disk(&urlpath, cparams, dparams).unwrap(),
            3 => SChunk::new(true, Some(&urlpath), cparams, dparams).unwrap(),
            _ => unreachable!(),
        };
        SChunkWrapper { temp_dir, schunk }
    }
}
