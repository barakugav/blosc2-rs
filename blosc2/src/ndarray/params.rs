use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use crate::ndarray::DimVec;
use crate::{CParams, DParams, Dtype, Error};

#[derive(Debug, Clone)]
pub struct NdarrayParams {
    pub(crate) cparams: CParams,
    pub(crate) dparams: DParams,
    shape: Option<DimVec<i64>>,
    chunksize: Option<DimVec<i32>>,
    blockshape: Option<DimVec<i32>>,
    dtype: Option<(Dtype, CString)>,
}
impl Default for NdarrayParams {
    fn default() -> Self {
        Self::new()
    }
}
impl NdarrayParams {
    pub fn new() -> Self {
        Self {
            cparams: CParams::default(),
            dparams: DParams::default(),
            shape: None,
            chunksize: None,
            blockshape: None,
            dtype: None,
        }
    }
    pub fn cparams(&mut self, cparams: CParams) -> &mut Self {
        self.cparams = cparams;
        self
    }
    pub fn dparams(&mut self, dparams: DParams) -> &mut Self {
        self.dparams = dparams;
        self
    }
    pub fn shape(&mut self, shape: &[i64]) -> Result<&mut Self, Error> {
        let shape = DimVec::from_slice(shape).ok_or_else(|| {
            crate::trace!("Too many dimensions: {}", shape.len());
            Error::InvalidParam
        })?;
        self.shape = Some(shape);
        Ok(self)
    }
    pub(crate) fn shape_required(&self) -> Result<&DimVec<i64>, Error> {
        self.shape.as_ref().ok_or_else(|| {
            crate::trace!("Shape is required");
            Error::InvalidParam
        })
    }
    pub fn chunksize(&mut self, chunksize: &[i32]) -> Result<&mut Self, Error> {
        let chunksize = DimVec::from_slice(chunksize).ok_or_else(|| {
            crate::trace!("Too many dimensions: {}", chunksize.len());
            Error::InvalidParam
        })?;
        self.chunksize = Some(chunksize);
        Ok(self)
    }
    pub(crate) fn chunksize_required(&self) -> Result<&DimVec<i32>, Error> {
        self.chunksize.as_ref().ok_or_else(|| {
            crate::trace!("Chunkshape is required");
            Error::InvalidParam
        })
    }
    pub fn blockshape(&mut self, blockshape: &[i32]) -> Result<&mut Self, Error> {
        let blockshape = DimVec::from_slice(blockshape).ok_or_else(|| {
            crate::trace!("Too many dimensions: {}", blockshape.len());
            Error::InvalidParam
        })?;
        self.blockshape = Some(blockshape);
        Ok(self)
    }
    pub(crate) fn blockshape_required(&self) -> Result<&DimVec<i32>, Error> {
        self.blockshape.as_ref().ok_or_else(|| {
            crate::trace!("Blockshape is required");
            Error::InvalidParam
        })
    }
    pub fn dtype(&mut self, dtype: &str) -> Result<&mut Self, Error> {
        let dtype_cstr = CString::new(dtype).map_err(|_| Error::InvalidParam)?;
        let dtype = Dtype::try_from(dtype).map_err(|_| {
            crate::trace!("Invalid dtype: {}", dtype);
            Error::InvalidParam
        })?;
        self.dtype = Some((dtype, dtype_cstr));
        Ok(self)
    }
    pub(crate) fn dtype_required(&self) -> Result<(&Dtype, &CString), Error> {
        let dtype = self.dtype.as_ref().ok_or_else(|| {
            crate::trace!("Dtype is required");
            Error::InvalidParam
        })?;
        Ok((&dtype.0, &dtype.1))
    }
}

pub(crate) struct Ctx(NonNull<blosc2_sys::b2nd_context_t>);
impl Ctx {
    pub(crate) fn new(
        storage: &blosc2_sys::blosc2_storage,
        shape: &[i64],
        chunkshape: &[i32],
        blockshape: &[i32],
        dtype: &CStr,
        dtype_format: i8,
    ) -> Result<Self, Error> {
        let ndim = shape.len();
        if chunkshape.len() != ndim || blockshape.len() != ndim {
            crate::trace!(
                "Chunkshape {} and blockshape {} lengths must match shape dimension: {}",
                chunkshape.len(),
                blockshape.len(),
                ndim
            );
            return Err(Error::InvalidParam);
        }

        let metalayers = [];
        let ctx = unsafe {
            blosc2_sys::b2nd_create_ctx(
                storage as *const _,
                ndim as i8,
                shape.as_ptr(),
                chunkshape.as_ptr(),
                blockshape.as_ptr(),
                dtype.as_ptr(),
                dtype_format,
                metalayers.as_ptr(),
                metalayers.len() as i32,
            )
        };
        Ok(Ctx(NonNull::new(ctx.cast()).ok_or(Error::Failure)?))
    }

    pub(crate) fn as_ptr(&self) -> *mut blosc2_sys::b2nd_context_t {
        self.0.as_ptr()
    }
}
impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { blosc2_sys::b2nd_free_ctx(self.0.as_ptr()) };
    }
}
