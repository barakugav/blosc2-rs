use std::ffi::CStr;
use std::ptr::NonNull;

use crate::nd::DimVec;
use crate::{CParams, DParams, Error};

/// Compression/decompression parameters for an [`Ndarray`](crate::nd::Ndarray).
///
/// These parameters are required to create any ndarray object, whether a new array is created or a slice or
/// a copy is created from an existing array.
/// See for example [`Ndarray::zeros`](crate::nd::Ndarray::zeros) or [`Ndarray::from_ndarray`](crate::nd::Ndarray::from_ndarray).
///
/// Functions that create arrays may require or ignore specific parameters, for example when slicing an existing array
/// the chunkshape and blockshape are optional, overriding the original array's parameters if provided.
/// Each such function document which parameters are required and which are optional/ignored.
#[derive(Debug, Clone, Default)]
pub struct NdarrayParams {
    pub(crate) cparams: CParams,
    pub(crate) dparams: DParams,
    pub(crate) chunkshape: Option<DimVec<i32>>,
    pub(crate) blockshape: Option<DimVec<i32>>,
}
impl NdarrayParams {
    /// Create a new set of parameters for an Ndarray.
    ///
    /// The compression and decompression params are initialized to their default values.
    pub fn new() -> Self {
        Self::default()
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
    /// Set the chunk shape of the array.
    ///
    /// Internally blosc partitions the data with a two layer partition into chunks and blocks,
    /// and compress each slice independently. This allows for a faster random access time to an element
    /// or a slice. The `chunkshape` parameter controls the size of these chunks, and its length
    /// should match the number of dimensions of the created array.
    ///
    /// # Arguments
    ///
    /// * `chunkshape` - The chunk shape of the array, `None` means automatic chunk shape
    ///   (not supported in all functions).
    ///
    /// # Panics
    ///
    /// This function will panic if the shape has more than [`MAX_DIM`](crate::nd::MAX_DIM) dimensions.
    pub fn chunkshape(&mut self, chunkshape: Option<&[usize]>) -> &mut Self {
        self.chunkshape = chunkshape
            .map(|shape| DimVec::from_slice_fn(shape, |s| *s as i32).expect("Too many dimensions"));
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
    ///
    /// # Arguments
    ///
    /// * `blockshape` - The block shape of the array, `None` means automatic block shape
    ///   (not supported in all functions).
    ///
    /// # Panics
    ///
    /// This function will panic if the shape has more than [`MAX_DIM`](crate::nd::MAX_DIM) dimensions.
    pub fn blockshape(&mut self, blockshape: Option<&[usize]>) -> &mut Self {
        self.blockshape = blockshape
            .map(|shape| DimVec::from_slice_fn(shape, |s| *s as i32).expect("Too many dimensions"));
        self
    }
    pub(crate) fn blockshape_required(&self) -> Result<&DimVec<i32>, Error> {
        self.blockshape.as_ref().ok_or_else(|| {
            crate::trace!("Blockshape is required");
            Error::InvalidParam
        })
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
