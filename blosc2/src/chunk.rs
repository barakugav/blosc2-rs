use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::{
    BytesMaybePassOwnershipToC, CowBytes, path2cstr, validate_compressed_buf_and_get_sizes,
};
use crate::{CParams, DParams, Error};

pub struct SChunk(NonNull<blosc2_sys::blosc2_schunk>);
impl SChunk {
    fn from_ptr(ptr: *mut blosc2_sys::blosc2_schunk) -> Result<Self, Error> {
        Ok(Self(NonNull::new(ptr).ok_or(Error::Failure)?))
    }

    pub fn new(
        contiguous: bool,
        urlpath: Option<&Path>,
        cparams: CParams,
        dparams: DParams,
    ) -> Result<Self, Error> {
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

    pub fn new_in_memory(cparams: CParams, dparams: DParams) -> Result<Self, Error> {
        Self::new(false, None, cparams, dparams)
    }

    pub fn new_on_disk(urlpath: &Path, cparams: CParams, dparams: DParams) -> Result<Self, Error> {
        Self::new(false, Some(urlpath), cparams, dparams)
    }

    pub fn open(urlpath: &Path) -> Result<Self, Error> {
        Self::open_with_offset(urlpath, 0)
    }

    pub fn open_with_offset(urlpath: &Path, offset: usize) -> Result<Self, Error> {
        let urlpath = path2cstr(urlpath);
        Self::from_ptr(unsafe {
            blosc2_sys::blosc2_schunk_open_offset(urlpath.as_ptr().cast_mut(), offset as _)
        })
    }

    pub fn from_buffer(buffer: CowBytes) -> Result<Self, Error> {
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

    pub fn to_buffer(&mut self) -> Result<CowBytes, Error> {
        let mut buffer = MaybeUninit::<*mut u8>::uninit();
        let mut needs_free = MaybeUninit::<bool>::uninit();
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
        Ok(unsafe { CowBytes::from_c_buf(buffer, len, needs_free) })
    }

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

    pub fn append(&mut self, data: &[u8]) -> Result<(), Error> {
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

    pub fn append_chunk(&mut self, chunk: Chunk) -> Result<(), Error> {
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

    pub fn update_chunk(&mut self, index: usize, chunk: Chunk) -> Result<(), Error> {
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

    pub fn insert_chunk(&mut self, index: usize, chunk: Chunk) -> Result<(), Error> {
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

    pub fn delete_chunk(&mut self, index: usize) -> Result<(), Error> {
        unsafe {
            blosc2_sys::blosc2_schunk_delete_chunk(self.0.as_ptr(), index as _).into_result()?;
        }
        Ok(())
    }

    pub fn get_chunk(&mut self, index: usize) -> Result<Chunk, Error> {
        let mut ptr = MaybeUninit::<*mut u8>::uninit();
        let mut needs_free = MaybeUninit::<bool>::uninit();
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
        let buf = unsafe { CowBytes::from_c_buf(ptr, len, needs_free) };
        Ok(unsafe { Chunk::from_compressed_unchecked(buf, Some(self)) })
    }

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

    pub fn num_chunks(&self) -> usize {
        unsafe { self.0.as_ref() }.nchunks as usize
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
#[derive(Clone)]
pub struct Chunk<'a> {
    buffer: CowBytes<'a>,
    schunk: Option<&'a SChunk>,
}
impl<'a> Chunk<'a> {
    pub fn from_data(data: &[u8]) -> Result<Self, Error> {
        let buffer = crate::Encoder::new(Default::default())?.compress(data)?;
        Ok(unsafe { Self::from_compressed_unchecked(buffer.into(), None) })
    }

    pub fn from_compressed(bytes: CowBytes<'a>) -> Result<Self, Error> {
        // validate, dont care about the sizes
        validate_compressed_buf_and_get_sizes(bytes.as_slice())?;
        Ok(unsafe { Self::from_compressed_unchecked(bytes, None) })
    }

    unsafe fn from_compressed_unchecked(buffer: CowBytes<'a>, schunk: Option<&'a SChunk>) -> Self {
        Self { buffer, schunk }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    pub fn into_bytes(self) -> CowBytes<'a> {
        self.buffer
    }

    pub fn decompress(&self) -> Result<Vec<u8>, Error> {
        let dparams = self
            .schunk
            .as_ref()
            .map(|schunk| DParams(unsafe { *schunk.0.as_ref().storage.as_ref().unwrap().dparams }))
            .unwrap_or_default();
        crate::Decoder::new(dparams)?.decompress(self.buffer.as_slice())
    }

    pub fn shallow_clone(&self) -> Chunk {
        Chunk {
            buffer: self.buffer.shallow_clone(),
            schunk: self.schunk,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use crate::chunk::{Chunk, SChunk};
    use crate::util::tests::rand_src_len;
    use crate::util::{CowBytes, FfiBytes};

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0xbe1392d28cdfb3ec);
        for _ in 0..30 {
            let data_chunks = rand_chunks_data(&mut rand);
            let mut schunk = SChunk::new_in_memory(Default::default(), Default::default()).unwrap();
            for data_chunk in &data_chunks {
                schunk.append(&data_chunk).unwrap();
            }
            assert_eq_chunks(&mut schunk, &data_chunks, None);
        }
    }

    #[test]
    fn append_chunk() {
        let mut rand = StdRng::seed_from_u64(0x612356293fbd4da9);
        for _ in 0..30 {
            let data_chunks = rand_chunks_data(&mut rand);
            let chunks = data2chunks(&data_chunks);
            let mut schunk = SChunk::new_in_memory(Default::default(), Default::default()).unwrap();
            for chunk in rand_chunks_ownerships(&chunks, &mut rand) {
                schunk.append_chunk(chunk).unwrap();
            }
            assert_eq_chunks(&mut schunk, &data_chunks, Some(&chunks));
        }
    }

    fn rand_chunks_data(rand: &mut StdRng) -> Vec<Vec<u8>> {
        let chunks_num = rand.random_range(0..4096);
        let chunk_size = if chunks_num == 0 {
            0
        } else {
            let max_total_size = rand_src_len(rand);
            let max_chunk_size = max_total_size.div_ceil(chunks_num);
            rand.random_range(0..=max_chunk_size)
        };
        (0..chunks_num)
            .map(|_| rand.random_iter().take(chunk_size).collect::<Vec<u8>>())
            .collect::<Vec<_>>()
    }

    fn data2chunks(data: &[impl AsRef<[u8]>]) -> Vec<Chunk<'static>> {
        data.iter()
            .map(|d| Chunk::from_data(d.as_ref()).unwrap())
            .collect::<Vec<_>>()
    }

    fn rand_chunks_ownerships<'a>(chunks: &'a [Chunk], rand: &mut StdRng) -> Vec<Chunk<'a>> {
        chunks
            .iter()
            .map(|chunk| rand_chunk_ownership(chunk, rand))
            .collect::<Vec<_>>()
    }

    fn rand_chunk_ownership<'a>(chunk: &'a Chunk, rand: &mut StdRng) -> Chunk<'a> {
        Chunk {
            buffer: rand_bytes_ownership(&chunk.buffer, rand),
            schunk: chunk.schunk,
        }
    }

    fn rand_bytes_ownership<'a>(bytes: &'a CowBytes, rand: &mut StdRng) -> CowBytes<'a> {
        match rand.random_range(0..3) {
            0 => CowBytes::Borrowed(bytes.as_slice()),
            1 => CowBytes::OwnedRust(bytes.as_slice().to_vec()),
            2 => CowBytes::OwnedFfi(FfiBytes::copy_of(bytes.as_slice())),
            _ => unreachable!(),
        }
    }

    fn assert_eq_chunks(schunk: &mut SChunk, data_chunks: &[Vec<u8>], chunks: Option<&[Chunk]>) {
        assert_eq!(schunk.num_chunks(), data_chunks.len());
        for (i, data_chunk) in data_chunks.iter().enumerate() {
            let chunk = schunk.get_chunk(i).unwrap();
            if let Some(chunks) = chunks {
                assert_eq!(chunk.as_bytes(), chunks[i].as_bytes());
            }
            let decompressed_chunk = chunk.decompress().unwrap();
            assert_eq!(data_chunk, &decompressed_chunk);
        }
    }
}
