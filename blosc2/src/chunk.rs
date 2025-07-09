use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::{CowBytes, path2cstr, validate_compressed_buf_and_get_sizes};
use crate::{CParams, DParams, Error};

pub struct SChunk<'a>(NonNull<blosc2_sys::blosc2_schunk>, PhantomData<&'a ()>);
impl<'a> SChunk<'a> {
    fn from_ptr(ptr: *mut blosc2_sys::blosc2_schunk) -> Result<Self, Error> {
        let ptr = NonNull::new(ptr).ok_or(Error::Failure)?;
        Ok(Self(ptr, PhantomData))
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
        let bytes = buffer.as_slice();
        let copy = match &buffer {
            CowBytes::Borrowed(_) | CowBytes::OwnedRust(_) => true,
            CowBytes::OwnedFfi(_) => false, // We move the ownership of the C allocated buffer to the C library
        };

        let schunk = unsafe {
            blosc2_sys::blosc2_schunk_from_buffer(bytes.as_ptr().cast_mut(), bytes.len() as _, copy)
        };

        if copy {
            drop(buffer);
        } else {
            // Forget the buffer to avoid calling its drop() and freeing it as its ownership was moved to the C library
            std::mem::forget(buffer);
        }

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

    pub fn append_chunk(&mut self, chunk: &Chunk) -> Result<(), Error> {
        unsafe { self.append_chunk_impl(chunk, true) }
    }

    pub fn append_chunk_nocopy(&mut self, chunk: &'a Chunk) -> Result<(), Error> {
        unsafe { self.append_chunk_impl(chunk, false) }
    }

    unsafe fn append_chunk_impl(&mut self, chunk: &Chunk, copy: bool) -> Result<(), Error> {
        unsafe {
            blosc2_sys::blosc2_schunk_append_chunk(
                self.0.as_ptr(),
                chunk.as_bytes().as_ptr().cast_mut(),
                copy,
            )
            .into_result()?;
        }
        Ok(())
    }

    pub fn update_chunk(&mut self, index: usize, chunk: &Chunk) -> Result<(), Error> {
        unsafe { self.update_chunk_impl(index, chunk, true) }
    }

    pub fn update_chunk_nocopy(&mut self, index: usize, chunk: &'a Chunk) -> Result<(), Error> {
        unsafe { self.update_chunk_impl(index, chunk, false) }
    }

    unsafe fn update_chunk_impl(
        &mut self,
        index: usize,
        chunk: &Chunk,
        copy: bool,
    ) -> Result<(), Error> {
        unsafe {
            blosc2_sys::blosc2_schunk_update_chunk(
                self.0.as_ptr(),
                index as _,
                chunk.as_bytes().as_ptr().cast_mut(),
                copy,
            )
            .into_result()?;
        }
        Ok(())
    }

    pub fn insert_chunk(&mut self, index: usize, chunk: &Chunk) -> Result<(), Error> {
        unsafe { self.insert_chunk_impl(index, chunk, true) }
    }

    pub fn insert_chunk_nocopy(&mut self, index: usize, chunk: &'a Chunk) -> Result<(), Error> {
        unsafe { self.insert_chunk_impl(index, chunk, false) }
    }

    unsafe fn insert_chunk_impl(
        &mut self,
        index: usize,
        chunk: &Chunk,
        copy: bool,
    ) -> Result<(), Error> {
        unsafe {
            blosc2_sys::blosc2_schunk_insert_chunk(
                self.0.as_ptr(),
                index as _,
                chunk.as_bytes().as_ptr().cast_mut(),
                copy,
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
        let mut chunk = Chunk::from_compressed(buf)?;
        chunk.schunk = Some(self);
        Ok(chunk)
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
impl Drop for SChunk<'_> {
    fn drop(&mut self) {
        let res = unsafe { blosc2_sys::blosc2_schunk_free(self.0.as_ptr()) }.into_result();
        if let Err(err) = res {
            eprintln!("Failed to free schunk: {err}");
        }
    }
}
pub struct Chunk<'a> {
    buffer: CowBytes<'a>,
    schunk: Option<&'a SChunk<'a>>,
}
impl<'a> Chunk<'a> {
    pub fn from_data(data: &[u8]) -> Result<Self, Error> {
        let buffer = crate::Encoder::new(Default::default())?
            .compress(data)?
            .into();
        Ok(Self {
            buffer,
            schunk: None,
        })
    }

    pub fn from_compressed(bytes: CowBytes<'a>) -> Result<Self, Error> {
        // validate, dont care about the sizes
        validate_compressed_buf_and_get_sizes(bytes.as_slice())?;

        Ok(Self {
            buffer: bytes,
            schunk: None,
        })
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
}
