use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use crate::ndarray::DimVec;
use crate::{CParams, DParams, Dtype, Error};

/// Parameters for an [`Ndarray`](crate::Ndarray).
///
/// The parameters control the dtype and shape of created arrays, along side compression/decompression settings.
/// These parameters are required to create any ndarray object, whether a new array is created or a slice or
/// a copy is created from an existing array.
/// See for example [`Ndarray::zeros`](crate::Ndarray::zeros) or [`Ndarray::from_ndarray`](crate::Ndarray::from_ndarray).
///
/// Functions that create arrays may require or ignore specific parameters, for example the dtype is ignored when we
/// create a blosc ndarray from an existing [`ndarray::ArrayBase`], and some of the compression parameters are optional
/// when we slice an existing array. Each such function document which parameters are required and which are ignored.
#[derive(Debug, Clone)]
pub struct NdarrayParams {
    pub(crate) cparams: CParams,
    pub(crate) dparams: DParams,
    pub(crate) shape: Option<DimVec<i64>>,
    pub(crate) chunkshape: Option<DimVec<i32>>,
    pub(crate) blockshape: Option<DimVec<i32>>,
    pub(crate) dtype: Option<(Dtype, CString)>,
}
impl Default for NdarrayParams {
    fn default() -> Self {
        Self::new()
    }
}
impl NdarrayParams {
    /// Create a new set of parameters for an Ndarray.
    ///
    /// The compression and decompression params are initialized to their default values.
    pub fn new() -> Self {
        Self {
            cparams: CParams::default(),
            dparams: DParams::default(),
            shape: None,
            chunkshape: None,
            blockshape: None,
            dtype: None,
        }
    }
    /// Set the compression parameters.
    pub fn cparams(&mut self, cparams: CParams) -> &mut Self {
        self.cparams = cparams;
        self
    }
    /// Set the decompression parameters.
    pub fn dparams(&mut self, dparams: DParams) -> &mut Self {
        self.dparams = dparams;
        self
    }
    /// Set the shape of the array.
    ///
    /// This parameter is usually respected when creating a new array, but may be ignored when slicing or copying an
    /// existing array.
    pub fn shape(&mut self, shape: &[usize]) -> &mut Self {
        let shape = DimVec::from_slice_fn(shape, |s| *s as i64).expect("Too many dimensions");
        self.shape = Some(shape);
        self
    }
    pub(crate) fn shape_required(&self) -> Result<&DimVec<i64>, Error> {
        self.shape.as_ref().ok_or_else(|| {
            crate::trace!("Shape is required");
            Error::InvalidParam
        })
    }
    /// Set the chunk shape of the array.
    ///
    /// Internally blosc partitions the data with a two layer partition into chunks and blocks,
    /// and compress each slice independently. This allows for a faster random access time to an element
    /// or a slice. The `chunkshape` parameter controls the size of these chunks, and its length
    /// should match the number of dimensions of the created array.
    pub fn chunkshape(&mut self, chunkshape: &[usize]) -> &mut Self {
        let chunkshape =
            DimVec::from_slice_fn(chunkshape, |s| *s as i32).expect("Too many dimensions");
        self.chunkshape = Some(chunkshape);
        self
    }
    pub(crate) fn chunkshape_required(&self) -> Result<&DimVec<i32>, Error> {
        self.chunkshape.as_ref().ok_or_else(|| {
            crate::trace!("Chunkshape is required");
            Error::InvalidParam
        })
    }
    /// Set the block shape of the array.
    ///
    /// Internally blosc partitions the data with a two layer partition into chunks and blocks,
    /// and compress each slice independently. This allows for a faster random access time to an element
    /// or a slice. The `blockshape` parameter controls the size of these blocks, and its length
    /// should match the number of dimensions of the created array.
    pub fn blockshape(&mut self, blockshape: &[usize]) -> &mut Self {
        let blockshape =
            DimVec::from_slice_fn(blockshape, |s| *s as i32).expect("Too many dimensions");
        self.blockshape = Some(blockshape);
        self
    }
    pub(crate) fn blockshape_required(&self) -> Result<&DimVec<i32>, Error> {
        self.blockshape.as_ref().ok_or_else(|| {
            crate::trace!("Blockshape is required");
            Error::InvalidParam
        })
    }
    /// Set the data type of the array.
    ///
    /// The dtype string should follow the numpy definitions, see
    /// https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
    /// for the full definitions, and the [`dtype_numpy_str()`](crate::Dtyped::dtype_numpy_str) function
    /// for examples.
    pub fn dtype(&mut self, dtype: &str) -> Result<&mut Self, Error> {
        let dtype_str = dtype;
        let dtype_cstr = CString::new(dtype_str).map_err(|_| Error::InvalidParam)?;
        let dtype = Dtype::try_from(dtype_str).map_err(|e| {
            crate::trace!("Invalid dtype: '{}', error: {}", dtype, e);
            Error::InvalidParam
        })?;
        if dtype.itemsize > blosc2_sys::BLOSC_MAX_TYPESIZE as usize {
            crate::trace!(
                "Itemsize {} is greater than BLOSC_MAX_TYPESIZE {}: {}",
                dtype.itemsize,
                blosc2_sys::BLOSC_MAX_TYPESIZE,
                dtype_str
            );
            return Err(Error::InvalidParam);
        }
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
