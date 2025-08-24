//! n-dimensional arrays with compressed storage (blosc2 NDarray, b2nd).
//!
//! The [`Ndarray`] struct is the main struct in this module, representing an n-dimensional array with
//! compressed storage. See its documentation for more details.

mod dtype;
pub use dtype::*;

mod params;
pub use params::*;

#[cfg(feature = "pyo3")]
mod python;
#[cfg(feature = "pyo3")]
pub use python::*;

use std::ffi::CString;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;

use crate::chunk::{SChunk, SChunkOpenOptions, SChunkStorageParams};
use crate::error::{Error, ErrorCode};
use crate::util::{path2cstr, ArrayVec, CowVec};

/// The maximum number of dimensions for an Ndarray.
pub const MAX_DIM: usize = blosc2_sys::B2ND_MAX_DIM as usize;
type DimVec<T> = ArrayVec<T, MAX_DIM>;

/// An n-dimensional array with compressed storage (blosc2 NDarray, b2nd).
///
/// The array stores elements of a specific data type, in an n-dimensional layout.
/// Differing from [`ndarray::ArrayBase`], both the type and the number of dimensions are managed at runtime and are
/// not known at compile time.
/// The data is stored in compressed format, either in memory or on disk. The data is compressed in chunks so that
/// random access to elements or slices does not require decompression of the entire array.
///
/// Construction of a new array can be done by many ways such as initializing a new array with a repeated value,
/// reading an array from disk, copying from existing [`ndarray::ArrayBase`] or blosc ndarray, etc.
/// Many of these functions accept an [`NdarrayParams`] struct that specify the compression/decompression settings.
///
/// Some of the functions of the struct are used to convert from/to arrays from the `ndarray` crate,
/// which may introduce some confusion about the term "ndarray". In most cases we refer to the
/// `ndarray` crate when we say "ndarray", while we refer to "blosc ndarray" or "b2nd" when
/// talking about the compressed arrays in this module.
/// For example, [`Ndarray::from_ndarray`] and [`Ndarray::to_ndarray`] are conversion from/to [`ndarray::ArrayBase`]
/// to/from a blosc ndarray, and [`Ndarray::slice_blosc`] returns a slice as a blosc ndarray.
///
/// ```rust
/// use blosc2::nd::{Ndarray, NdarrayParams};
///
/// let arr = Ndarray::from_ndarray(
///     &ndarray::array!([1_i32, 2, 3], [4, 5, 6], [7, 8, 9]),
///     NdarrayParams::default()
///         .chunkshape(Some(&[2, 2]))
///         .blockshape(Some(&[1, 1])),
/// ).unwrap();
///
/// assert_eq!(4, arr.get::<i32>(&[1, 0]).unwrap());
/// assert_eq!(9, arr.get::<i32>(&[2, 2]).unwrap());
///
/// let slice_arr: ndarray::Array2<i32> = arr.slice(&[0..2, 1..3]).unwrap();
/// assert_eq!(slice_arr, ndarray::array![[2, 3], [5, 6]]);
/// ```
pub struct Ndarray {
    ptr: NonNull<blosc2_sys::b2nd_array_t>,
    dtype: Dtype,
}
impl Ndarray {
    fn new_impl(
        value: InitArgs,
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        let mut cparams = params.cparams.clone();
        let mut dparams = params.dparams.clone();

        let dtype2;
        let dtype_cstr;
        let shape2;
        let chunkshape;
        let blockshape;
        match &value {
            InitArgs::Zeros(dtype, shape)
            | InitArgs::Nans(dtype, shape)
            | InitArgs::Uninit(dtype, shape)
            | InitArgs::RepeatedValue(dtype, _, shape)
            | InitArgs::CopyFromValuesBuf(dtype, _, shape) => {
                dtype2 = dtype;
                dtype_cstr = std::borrow::Cow::Owned(CString::new(dtype.to_numpy_str()).unwrap());
                shape2 = shape.as_slice();
                chunkshape = params.chunkshape_required()?.as_slice();
                blockshape = params.blockshape_required()?.as_slice();
            }
            InitArgs::CopyFromNdarray(ndarray) => {
                dtype2 = &ndarray.dtype;
                dtype_cstr = ndarray.dtype_cstr().into();
                shape2 = ndarray.shape();
                chunkshape = params
                    .chunkshape
                    .as_ref()
                    .map(|c| c.as_slice())
                    .unwrap_or_else(|| ndarray.chunkshape());
                blockshape = params
                    .blockshape
                    .as_ref()
                    .map(|b| b.as_slice())
                    .unwrap_or_else(|| ndarray.blockshape());
            }
        }
        let dtype = dtype2;
        let shape = shape2;

        cparams.typesize(dtype.itemsize()).inspect_err(|_| {
            crate::trace!("Invalid dtype: {}", dtype_cstr.to_str().unwrap());
        })?;

        if shape.is_empty() {
            crate::trace!("Zero-dim ndarray are not supported");
            return Err(Error::InvalidParam);
        }

        crate::global::global_init();

        let urlpath = storage.urlpath.map(path2cstr);
        let urlpath = urlpath
            .as_ref()
            .map(|p| p.as_ptr().cast_mut())
            .unwrap_or(std::ptr::null_mut());
        let storage = blosc2_sys::blosc2_storage {
            contiguous: storage.contiguous,
            urlpath,
            cparams: &mut cparams.0 as *mut blosc2_sys::blosc2_cparams,
            dparams: &mut dparams.0 as *mut blosc2_sys::blosc2_dparams,
            io: std::ptr::null_mut(),
        };

        let ctx = Ctx::new(
            &storage,
            shape,
            chunkshape,
            blockshape,
            dtype_cstr.as_ref(),
            blosc2_sys::DTYPE_NUMPY_FORMAT as _,
        )?;

        let mut array = MaybeUninit::<*mut blosc2_sys::b2nd_array_t>::uninit();
        let res = match value {
            InitArgs::Zeros(_, _) => unsafe {
                blosc2_sys::b2nd_zeros(ctx.as_ptr(), array.as_mut_ptr())
            },
            InitArgs::Nans(_, _) => unsafe {
                blosc2_sys::b2nd_nans(ctx.as_ptr(), array.as_mut_ptr())
            },
            InitArgs::Uninit(_, _) => unsafe {
                blosc2_sys::b2nd_uninit(ctx.as_ptr(), array.as_mut_ptr())
            },
            InitArgs::RepeatedValue(_, value, _) => {
                if value.len() != dtype.itemsize() {
                    crate::trace!(
                        "Repeated value length {} does not match dtype itemsize {}",
                        value.len(),
                        dtype.itemsize()
                    );
                    return Err(Error::InvalidParam);
                }
                unsafe {
                    blosc2_sys::b2nd_full(ctx.as_ptr(), array.as_mut_ptr(), value.as_ptr().cast())
                }
            }
            InitArgs::CopyFromValuesBuf(_, items, _) => {
                let expected_length =
                    dtype.itemsize() * shape.iter().map(|s| *s as usize).product::<usize>();
                if items.len() != expected_length {
                    crate::trace!(
                        "Items buffer length {} does not match expected length {} ({} * {:?})",
                        items.len(),
                        expected_length,
                        dtype.itemsize(),
                        shape
                    );
                    return Err(Error::InvalidParam);
                }
                unsafe {
                    blosc2_sys::b2nd_from_cbuffer(
                        ctx.as_ptr(),
                        array.as_mut_ptr(),
                        items.as_ptr().cast(),
                        items.len() as i64,
                    )
                }
            }
            InitArgs::CopyFromNdarray(ndarray) => unsafe {
                blosc2_sys::b2nd_copy(ctx.as_ptr(), ndarray.as_ptr(), array.as_mut_ptr())
                    .into_result()?
            },
        };
        res.into_result()?;
        let array = unsafe { array.assume_init() };

        unsafe { Self::from_raw_ptr(array.cast()) }
    }

    /// Creates a new ndarray in memory filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the new ndarray.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    ///
    /// For example, the following code creates a new ndarray of shape `[100, 20, 4]` filled with 32 bit integer zeros:
    /// ```rust
    /// use blosc2::nd::{Ndarray, NdarrayParams};
    ///
    /// let arr = Ndarray::zeros::<i32>(
    ///     &[100, 20, 4],
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[10, 4, 2]))
    ///         .blockshape(Some(&[2, 2, 2])),
    /// );
    /// assert!(arr.is_ok());
    /// ```
    pub fn zeros<T>(shape: &[usize], params: &NdarrayParams) -> Result<Self, Error>
    where
        T: Dtyped,
    {
        Self::new(NdarrayInitValue::zero::<T>(), shape, params)
    }

    /// Creates a new ndarray in memory filled with a given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value for all elements in the array.
    /// * `shape` - The shape of the new ndarray.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    ///
    /// For example, the following code creates a new ndarray of shape `[64, 32, 32]` filled with 64 bit float ones:
    /// ```rust
    /// use blosc2::nd::{Ndarray, NdarrayParams};
    ///
    /// let arr = Ndarray::full(
    ///     1.0_f64,
    ///     &[64, 32, 32],
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[8, 4, 16]))
    ///         .blockshape(Some(&[4, 4, 1])),
    /// );
    /// assert!(arr.is_ok());
    /// ```
    pub fn full<T>(value: T, shape: &[usize], params: &NdarrayParams) -> Result<Self, Error>
    where
        T: Dtyped,
    {
        Self::new(NdarrayInitValue::value::<T>(value), shape, params)
    }

    /// Creates a new ndarray in memory with a special value (zero/nan/uninit) or a repeated value.
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value for all elements in the array.
    /// * `shape` - The shape of the new ndarray.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    ///
    /// For example, the following code creates a new ndarray of shape `[12, 24, 8]` filled with uninitialized
    /// complex f64 values:
    /// ```rust
    /// use blosc2::nd::{Ndarray, NdarrayInitValue, NdarrayParams};
    ///
    /// let arr = Ndarray::new(
    ///     unsafe { NdarrayInitValue::uninit::<blosc2::util::Complex::<f64>>()},
    ///     &[12, 24, 8],
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[6, 8, 1]))
    ///         .blockshape(Some(&[3, 8, 1])),
    /// );
    /// assert!(arr.is_ok());
    /// ```
    pub fn new(
        value: NdarrayInitValue,
        shape: &[usize],
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_at(value, shape, SChunkStorageParams::in_memory(), params)
    }

    /// Creates a new ndarray on disk with a special value (zero/nan/uninit) or a repeated value.
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value for all elements in the array.
    /// * `shape` - The shape of the new ndarray.
    /// * `urlpath` - The path to the new file on disk.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    pub fn new_on_disk(
        value: NdarrayInitValue,
        shape: &[usize],
        urlpath: &Path,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_at(value, shape, SChunkStorageParams::on_disk(urlpath), params)
    }

    /// Creates a new ndarray at the given storage with a special value (zero/nan/uninit) or a repeated value.
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value for all elements in the array.
    /// * `shape` - The shape of the new ndarray.
    /// * `storage` - The storage parameters for the ndarray. This is the most flexible option to specify the
    ///   location (in memory or on disk) and whether the storage is contiguous or not. See [`SChunkStorageParams`]
    ///   for more details.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    pub fn new_at(
        value: NdarrayInitValue,
        shape: &[usize],
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        let dtype = value.dtype;
        let shape = DimVec::from_slice_fn(shape, |s| *s as i64).expect("Too many dimensions");
        Self::new_impl(
            match &value.value {
                InitValueInner::Zero => InitArgs::Zeros(dtype, shape),
                InitValueInner::Nan => InitArgs::Nans(dtype, shape),
                InitValueInner::Uninit => InitArgs::Uninit(dtype, shape),
                InitValueInner::Value(v) => InitArgs::RepeatedValue(dtype, v, shape),
            },
            storage,
            params,
        )
    }

    /// Opens an existing ndarray from disk.
    pub fn open(urlpath: &Path) -> Result<Self, Error> {
        Self::open_with_options(urlpath, &SChunkOpenOptions::default())
    }

    /// Opens an existing ndarray from disk with the given options.
    pub fn open_with_options(urlpath: &Path, options: &SChunkOpenOptions) -> Result<Self, Error> {
        crate::global::global_init();

        let schunk = SChunk::open_with_options(urlpath, options)?;
        let schunk = schunk.into_raw_ptr() as *mut blosc2_sys::blosc2_schunk;

        let mut array = MaybeUninit::<*mut blosc2_sys::b2nd_array_t>::uninit();
        unsafe {
            blosc2_sys::b2nd_from_schunk(schunk, array.as_mut_ptr()).into_result()?;
        };
        let array = unsafe { array.assume_init() };

        unsafe { Self::from_raw_ptr(array.cast()) }
    }

    /// Creates a new in memory ndarray from a contiguous buffer of items.
    ///
    /// # Arguments
    ///
    /// * `items` - The buffer containing the item data, with `typesize * prod(shape)` bytes.
    /// * `shape` - The shape of the new ndarray.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    ///
    /// The items buffer size must match the shape.
    /// The elements should be laid out in memory according to the standard strides of the shape.
    ///
    /// ```rust
    /// use blosc2::nd::{Ndarray, NdarrayParams};
    ///
    /// let arr = Ndarray::from_items(
    ///     &[1_32, 2, 3, 4, 5, 6],
    ///     &[2, 3],
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[2, 1]))
    ///         .blockshape(Some(&[1, 1])),
    /// );
    /// assert!(arr.is_ok());
    /// ```
    pub fn from_items<T>(
        items: &[T],
        shape: &[usize],
        params: &NdarrayParams,
    ) -> Result<Self, Error>
    where
        T: Dtyped,
    {
        let items_buf = unsafe {
            std::slice::from_raw_parts(items.as_ptr().cast::<u8>(), std::mem::size_of_val(items))
        };
        Self::from_items_bytes(items_buf, T::dtype(), shape, params)
    }

    /// Creates a new in memory ndarray from a contiguous buffer of items.
    ///
    /// # Arguments
    ///
    /// * `items` - The buffer containing the item data, with `typesize * prod(shape)` bytes.
    /// * `dtype` - The data type of the items.
    /// * `shape` - The shape of the new ndarray.
    /// * `params` - The parameters for the ndarray. The chunkshape and blockshape are required.
    ///
    /// The items buffer size must match the dtype and shape.
    /// The elements should be laid out in memory according to the standard strides of the shape.
    ///
    /// ```rust
    /// use blosc2::nd::{Dtyped, Ndarray, NdarrayParams};
    ///
    /// let data = [1_32, 2, 3, 4, 5, 6];
    /// let data_buf = unsafe {
    ///     std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(&data))
    /// };
    /// let arr = Ndarray::from_items_bytes(
    ///     data_buf,
    ///     i32::dtype(),
    ///     &[2, 3],
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[2, 1]))
    ///         .blockshape(Some(&[1, 1])),
    /// );
    /// assert!(arr.is_ok());
    /// ```
    pub fn from_items_bytes(
        items: &[u8],
        dtype: Dtype,
        shape: &[usize],
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::from_items_bytes_at(
            items,
            dtype,
            shape,
            SChunkStorageParams::in_memory(),
            params,
        )
    }

    /// Creates a new ndarray at the given storage from a contiguous buffer of items.
    ///
    /// # Arguments
    ///
    /// * `items` - The buffer containing the item data, with `typesize * prod(shape)` bytes.
    /// * `dtype` - The data type of the items.
    /// * `shape` - The shape of the new ndarray.
    /// * `storage` - The storage parameters for the ndarray. See [`SChunkStorageParams`] for more details.
    /// * `params` - The parameters for the ndarray. The chunkshape, blockshape and dtype are required.
    ///
    /// The items buffer size must match the dtype and shape.
    /// The elements should be laid out in memory according to the standard strides of the shape.
    pub fn from_items_bytes_at(
        items: &[u8],
        dtype: Dtype,
        shape: &[usize],
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        let shape = DimVec::from_slice_fn(shape, |s| *s as i64).expect("Too many dimensions");
        Self::new_impl(
            InitArgs::CopyFromValuesBuf(dtype, items, shape),
            storage,
            params,
        )
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut array = MaybeUninit::<*mut blosc2_sys::b2nd_array_t>::uninit();
        unsafe {
            blosc2_sys::b2nd_from_cframe(
                bytes.as_ptr().cast_mut(),
                bytes.len() as _,
                true,
                array.as_mut_ptr(),
            )
            .into_result()?
        };
        let array = unsafe { array.assume_init() };
        unsafe { Self::from_raw_ptr(array.cast()) }
    }

    /// Creates a new blosc ndarray from an existing [`ndarray::ArrayBase`].
    ///
    /// # Arguments
    ///
    /// * `ndarray` - The source ndarray to copy data from.
    /// * `params` - The parameters for the new ndarray. The chunkshape and blockshape are required.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, specifying its data type.
    ///
    /// ```rust
    /// use blosc2::nd::{Ndarray, NdarrayParams};
    ///
    /// let arr = Ndarray::from_ndarray(
    ///     &ndarray::array!([1_i32, 2, 3], [4, 5, 6], [7, 8, 9]),
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[2, 2]))
    ///         .blockshape(Some(&[1, 1])),
    /// ).unwrap();
    /// let slice_arr: ndarray::Array2<i32> = arr.slice(&[0..2, 1..3]).unwrap();
    /// assert_eq!(slice_arr, ndarray::array![[2, 3], [5, 6]]);
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn from_ndarray<S, T, D>(
        ndarray: &ndarray::ArrayBase<S, D>,
        params: &NdarrayParams,
    ) -> Result<Self, Error>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        Self::from_ndarray_at(ndarray, SChunkStorageParams::in_memory(), params)
    }

    /// Creates a new blosc ndarray at the given storage from an existing [`ndarray::ArrayBase`].
    ///
    /// # Arguments
    ///
    /// * `ndarray` - The source ndarray to copy data from.
    /// * `storage` - The storage parameters for the new ndarray.
    /// * `params` - The parameters for the new ndarray. The chunkshape and blockshape are required.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, specifying its data type.
    #[cfg(feature = "ndarray")]
    pub fn from_ndarray_at<S, T, D>(
        ndarray: &ndarray::ArrayBase<S, D>,
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        if ndarray.ndim() > MAX_DIM {
            crate::trace!(
                "ndarray has {} dimensions, but maximum supported is {}",
                ndarray.ndim(),
                MAX_DIM
            );
            return Err(Error::InvalidParam);
        }

        let data = ndarray.as_standard_layout();
        let data = data
            .as_slice()
            .expect("arr.as_standard_layout() should be contiguous");
        let data_buf = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };

        Self::from_items_bytes_at(data_buf, T::dtype(), ndarray.shape(), storage, params)
    }

    /// Creates a new blosc ndarray from a raw pointer to a `blosc2_sys::b2nd_array_t`.
    ///
    /// The ownership of the underlying memory is transferred to the new ndarray, and it will be freed once
    /// the ndarray is dropped using `blosc2_sys::b2nd_free`.
    ///
    /// This function can be useful if a user wants to accept an ndarray across ffi boundaries.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is valid and points to a valid `blosc2_sys::b2nd_array_t`, and that
    /// no other references to the same memory exist.
    pub unsafe fn from_raw_ptr(ptr: *mut ()) -> Result<Self, Error> {
        let ptr: NonNull<blosc2_sys::b2nd_array_t> =
            NonNull::new(ptr.cast()).ok_or(Error::Failure)?;

        let dtype_cstr = unsafe { std::ffi::CStr::from_ptr((*ptr.as_ptr()).dtype) };
        let dtype = dtype_cstr.to_str().unwrap();
        let dtype = Dtype::from_numpy_str(dtype).map_err(|e| {
            crate::trace!("Invalid dtype: '{}', error: {}", dtype, e);
            Error::InvalidParam
        })?;

        Ok(Self { ptr, dtype })
    }
}

/// Represents initial value use to initialize an ndarray.
///
/// This struct is used by [`Ndarray::new`] and its variants, providing an initial value for all elements, and also
/// determining the dtype of the created array.
#[derive(Clone, Debug)]
pub struct NdarrayInitValue {
    dtype: Dtype,
    value: InitValueInner,
}
impl NdarrayInitValue {
    fn new(value: InitValueInner, dtype: Dtype) -> Self {
        Self { dtype, value }
    }

    /// A zero value of type `T`.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and it will determine the created ndarray dtype.
    pub fn zero<T>() -> Self
    where
        T: Dtyped,
    {
        Self::zero_of(T::dtype())
    }

    /// A zero value of the given dtype.
    pub fn zero_of(dtype: Dtype) -> Self {
        Self::new(InitValueInner::Zero, dtype)
    }

    /// A NaN value of type `T` (for types that support NaN, like `f32` and `f64`).
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and it will determine the created ndarray dtype.
    pub fn nan<T>() -> Self
    where
        T: Dtyped,
    {
        Self::nan_of(T::dtype())
    }

    /// A NaN value of the given dtype (for types that support NaN, like `f32` and `f64`).
    pub fn nan_of(dtype: Dtype) -> Self {
        Self::new(InitValueInner::Nan, dtype)
    }

    /// An uninitialized value of type `T`.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and it will determine the created ndarray dtype.
    ///
    /// # Safety
    ///
    /// An ndarray created with uninit values does not enforce the unsafe properties of its elements using the type
    /// system, and all the regular functions are marked as safe, but may lead to UB if not used carefully. Therefore,
    /// this function is marked unsafe for the caller to acknowledge the potential risks.
    /// The caller must ensure to write to all elements before reading them, after an array with uninit values is
    /// created.
    pub unsafe fn uninit<T>() -> Self
    where
        T: Dtyped,
    {
        Self::uninit_of(T::dtype())
    }

    /// An uninitialized value of the given dtype.
    ///
    /// # Safety
    ///
    /// An ndarray created with uninit values does not enforce the unsafe properties of its elements using the type
    /// system, and all the regular functions are marked as safe, but may lead to UB if not used carefully. Therefore,
    /// this function is marked unsafe for the caller to acknowledge the potential risks.
    /// The caller must ensure to write to all elements before reading them, after an array with uninit values is
    /// created.
    pub unsafe fn uninit_of(dtype: Dtype) -> Self {
        Self::new(InitValueInner::Uninit, dtype)
    }

    /// A specific value of type `T`.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and it will determine the created ndarray dtype.
    pub fn value<T>(value: T) -> Self
    where
        T: Dtyped,
    {
        let value_bytes = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };
        Self::value_bytes(value_bytes.to_vec(), T::dtype()).unwrap()
    }

    /// A specific value of the given dtype.
    ///
    /// The value must have the same size as specified by the dtype.
    pub fn value_bytes(value: Vec<u8>, dtype: Dtype) -> Result<Self, Error> {
        if value.len() != dtype.itemsize() {
            crate::trace!(
                "Value length {} does not match dtype itemsize {}",
                value.len(),
                dtype.itemsize()
            );
            return Err(Error::InvalidParam);
        }
        Ok(Self {
            dtype,
            value: InitValueInner::Value(value),
        })
    }
}
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum InitValueInner {
    Zero,
    Nan,
    Uninit,
    Value(Vec<u8>),
}

enum InitArgs<'a> {
    Zeros(Dtype, DimVec<i64>),
    Nans(Dtype, DimVec<i64>),
    Uninit(Dtype, DimVec<i64>),
    RepeatedValue(Dtype, &'a [u8], DimVec<i64>),
    CopyFromValuesBuf(Dtype, &'a [u8], DimVec<i64>),
    CopyFromNdarray(&'a Ndarray),
}

impl Ndarray {
    fn arr(&self) -> &blosc2_sys::b2nd_array_t {
        unsafe { self.ptr.as_ref() }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.arr().ndim as usize
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.arr().shape.as_ptr(), self.ndim()) }
    }

    /// Get the data type of the array.
    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn dtype_cstr(&self) -> &std::ffi::CStr {
        unsafe { std::ffi::CStr::from_ptr(self.arr().dtype) }
    }

    /// Get the size of each element in bytes.
    pub fn typesize(&self) -> usize {
        debug_assert_eq!(
            self.dtype.itemsize(),
            unsafe { &*self.arr().sc }.typesize as usize
        );
        self.dtype.itemsize()
    }

    /// Get the shape of chunks in the ndarray.
    pub fn chunkshape(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.arr().chunkshape.as_ptr(), self.ndim()) }
    }

    /// Get the shape of blocks in the ndarary.
    pub fn blockshape(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.arr().blockshape.as_ptr(), self.ndim()) }
    }

    /// Check whether the array is contiguous or sparse.
    pub fn is_contiguous(&self) -> bool {
        let storage = unsafe { self.arr().sc.as_ref().unwrap().storage.as_ref().unwrap() };
        storage.contiguous
    }

    /// Get an [`ndarray::Array`] with the data copied from this ndarray.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    #[cfg(feature = "ndarray")]
    pub fn to_ndarray<T, D>(&self) -> Result<ndarray::Array<T, D>, Error>
    where
        T: Dtyped,
        D: ndarray::Dimension,
    {
        self.check_dtype::<T>()?;
        // Safety: we have checked the dtype
        unsafe { self.to_ndarray_without_dtype_check() }
    }

    /// Get an [`ndarray::Array`] with the data copied from this ndarray without checking the dtype.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the dtype of `T` matches the dtype of the ndarray. The size, alignment, and inner
    /// fields should all be compatible. See [`Dtyped`] for more information.
    #[cfg(feature = "ndarray")]
    unsafe fn to_ndarray_without_dtype_check<T, D>(&self) -> Result<ndarray::Array<T, D>, Error>
    where
        T: Copy + 'static,
        D: ndarray::Dimension,
    {
        self.check_dimension_type_for_read::<D>()?;

        assert_eq!(std::mem::size_of::<T>(), self.typesize());
        let shape = self.shape().iter().map(|s| *s as usize).collect::<Vec<_>>();
        let buf_len = shape.iter().product::<usize>();
        let mut buf = Vec::<MaybeUninit<T>>::with_capacity(buf_len);
        unsafe { buf.set_len(buf_len) };

        self.to_items_bytes_into(std::slice::from_raw_parts_mut(
            buf.as_mut_ptr().cast(),
            buf_len * std::mem::size_of::<T>(),
        ))?;
        let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(buf) };

        let ndim = self.ndim();
        let mut res_shape = D::zeros(ndim);
        for i in 0..ndim {
            res_shape[i] = shape[i];
        }
        Ok(ndarray::Array::from_shape_vec(res_shape, buf).unwrap())
    }

    /// Get a vector with all the elements of the array.
    ///
    /// The returned array is of size `prod(shape)`, and the elements are laid out in memory according to the
    /// standard strides.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    pub fn to_items<T>(&self) -> Result<Vec<T>, Error>
    where
        T: Dtyped,
    {
        self.check_dtype::<T>()?;
        let buf_len = self.shape().iter().map(|s| *s as usize).product::<usize>();
        let mut buf = Vec::<MaybeUninit<T>>::with_capacity(buf_len);
        unsafe {
            buf.set_len(buf_len);
        }
        self.to_items_bytes_into(unsafe {
            std::slice::from_raw_parts_mut(
                buf.as_mut_ptr().cast(),
                buf_len * std::mem::size_of::<T>(),
            )
        })?;
        let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(buf) };
        Ok(buf)
    }

    /// Get a bytes vector with all the elements of the array.
    ///
    /// The returned array is of size `itemsize * prod(shape)`, and the elements are laid out in memory according to the
    /// standard strides.
    pub fn to_items_bytes(&self) -> Result<Vec<u8>, Error> {
        let buf_len = self.typesize() * self.shape().iter().map(|s| *s as usize).product::<usize>();
        let mut buf = Vec::<MaybeUninit<u8>>::with_capacity(buf_len);
        unsafe {
            buf.set_len(buf_len);
        }
        self.to_items_bytes_into(&mut buf)?;
        let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(buf) };
        Ok(buf)
    }

    /// Write the bytes of all the array elements to a given buffer.
    ///
    /// The given buffer should have at least `itemsize * prod(shape)` bytes allocated, and the elements will be written
    /// according to the standard strides.
    ///
    /// # Returns
    ///
    /// The number of bytes written to the buffer.
    pub fn to_items_bytes_into(&self, buf: &mut [MaybeUninit<u8>]) -> Result<usize, Error> {
        unsafe {
            blosc2_sys::b2nd_to_cbuffer(self.as_ptr(), buf.as_mut_ptr().cast(), buf.len() as i64)
                .into_result()?;
        }
        Ok(self.typesize() * self.shape().iter().map(|s| *s as usize).product::<usize>())
    }

    /// Get a single element from the array.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    pub fn get<T>(&self, idx: &[usize]) -> Result<T, Error>
    where
        T: Dtyped,
    {
        self.check_dtype::<T>()?;
        let mut buf = MaybeUninit::<T>::uninit();
        self.get_buf_into(idx, unsafe {
            std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), std::mem::size_of::<T>())
        })?;
        let buf = unsafe { buf.assume_init() };
        Ok(buf)
    }

    /// Get a single element from the array as bytes vector.
    pub fn get_buf(&self, idx: &[usize]) -> Result<Vec<u8>, Error> {
        let buf_len = self.typesize();
        let mut buf = Vec::<MaybeUninit<u8>>::with_capacity(buf_len);
        unsafe { buf.set_len(buf_len) };
        self.get_buf_into(idx, &mut buf)?;
        let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(buf) };
        Ok(buf)
    }

    /// Get a single element from the array and write it to the provided buffer.
    pub fn get_buf_into(&self, idx: &[usize], dst: &mut [MaybeUninit<u8>]) -> Result<usize, Error> {
        if idx.len() != self.ndim() {
            crate::trace!(
                "Index length {} must match array dimension {}",
                idx.len(),
                self.ndim()
            );
            return Err(Error::InvalidParam);
        }

        let slice = DimVec::from_slice_fn(idx, |&i| i..i + 1);
        // Safety: checked idx.len()==self.ndim() , and we know self.ndim()<=MAX_DIM
        let slice = unsafe { slice.unwrap_unchecked() };

        self.slice_buf_into(slice.as_slice(), dst)
    }

    pub fn to_bytes(&self) -> Result<CowVec<u8>, Error> {
        let mut buffer = MaybeUninit::uninit();
        let mut buffer_len = MaybeUninit::uninit();
        let mut needs_free = MaybeUninit::uninit();
        unsafe {
            blosc2_sys::b2nd_to_cframe(
                self.as_ptr(),
                buffer.as_mut_ptr(),
                buffer_len.as_mut_ptr(),
                needs_free.as_mut_ptr(),
            )
            .into_result()?
        };

        let buffer = NonNull::new(unsafe { buffer.assume_init() }).ok_or(Error::Failure)?;
        let buffer_len = unsafe { buffer_len.assume_init() };
        let needs_free = unsafe { needs_free.assume_init() };
        Ok(unsafe { CowVec::from_c_buf(buffer, buffer_len as usize, needs_free) })
    }

    /// Get a slice of the array as an [`ndarray::Array`].
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    ///
    /// # Returns
    ///
    /// A new [`ndarray::Array`] representing the sliced portion of the array.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    ///
    /// ```rust
    /// use blosc2::nd::{Ndarray, NdarrayParams};
    ///
    /// let arr = Ndarray::from_ndarray(
    ///     &ndarray::array!([1_i32, 2, 3], [4, 5, 6], [7, 8, 9]),
    ///     NdarrayParams::default()
    ///         .chunkshape(Some(&[2, 2]))
    ///         .blockshape(Some(&[1, 1])),
    /// ).unwrap();
    /// let slice_arr: ndarray::Array2<i32> = arr.slice(&[0..2, 1..3]).unwrap();
    /// assert_eq!(slice_arr, ndarray::array![[2, 3], [5, 6]]);
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn slice<T, D>(
        &self,
        slice: &[std::ops::Range<usize>],
    ) -> Result<ndarray::Array<T, D>, Error>
    where
        T: Dtyped,
        D: ndarray::Dimension,
    {
        self.check_dtype::<T>()?;
        // Safety: we have already checked the dtype
        unsafe { self.slice_without_dtype_check(slice) }
    }

    /// Get a slice of the array as an [`ndarray::Array`] without checking the dtype.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    ///
    /// # Returns
    ///
    /// A new [`ndarray::Array`] representing the sliced portion of the array.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the dtype of `T` matches the dtype of the ndarray. The size, alignment, and inner
    /// fields should all be compatible. See [`Dtyped`] for more information.
    #[cfg(feature = "ndarray")]
    unsafe fn slice_without_dtype_check<T, D>(
        &self,
        slice: &[std::ops::Range<usize>],
    ) -> Result<ndarray::Array<T, D>, Error>
    where
        T: Copy + 'static,
        D: ndarray::Dimension,
    {
        self.check_dimension_type_for_read::<D>()?;

        assert_eq!(std::mem::size_of::<T>(), self.typesize());
        let buf_len = slice.iter().map(|r| r.len()).product::<usize>();
        let mut buf = Vec::<MaybeUninit<T>>::with_capacity(buf_len);
        unsafe { buf.set_len(buf_len) };

        self.slice_buf_into(
            slice,
            std::slice::from_raw_parts_mut(
                buf.as_mut_ptr().cast(),
                buf_len * std::mem::size_of::<T>(),
            ),
        )?;
        let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(buf) };

        let ndim = self.ndim();
        let mut res_shape = D::zeros(ndim);
        for i in 0..ndim {
            res_shape[i] = slice[i].len();
        }
        Ok(ndarray::Array::from_shape_vec(res_shape, buf).unwrap())
    }

    /// Get a slice of the array as a blosc [`Ndarray`].
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    /// * `params` - The parameters to use for the new array. The chunkshape and blockshape are optional, overriding
    ///   the original array's parameters.
    ///
    /// # Returns
    ///
    /// A new blosc [`Ndarray`] representing the sliced portion of the array.
    pub fn slice_blosc(
        &self,
        slice: &[std::ops::Range<usize>],
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        self.check_slice_arg(slice)?;
        let shape = DimVec::from_slice_fn(slice, |r| r.len() as i64);
        let start = DimVec::from_slice_fn(slice, |r| r.start as i64);
        let end = DimVec::from_slice_fn(slice, |r| r.end as i64);
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let start = unsafe { start.unwrap_unchecked() };
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let end = unsafe { end.unwrap_unchecked() };

        let mut cparams = params.cparams.clone();
        let mut dparams = params.dparams.clone();
        let chunkshape = params
            .chunkshape
            .as_ref()
            .map(|c| c.as_slice())
            .unwrap_or_else(|| self.chunkshape());
        let blockshape = params
            .blockshape
            .as_ref()
            .map(|b| b.as_slice())
            .unwrap_or_else(|| self.blockshape());
        cparams.typesize(self.dtype.itemsize()).inspect_err(|_| {
            crate::trace!("Invalid dtype: {}", self.dtype_cstr().to_str().unwrap());
        })?;

        let storage = blosc2_sys::blosc2_storage {
            contiguous: false,
            urlpath: std::ptr::null_mut(),
            cparams: &mut cparams.0 as *mut blosc2_sys::blosc2_cparams,
            dparams: &mut dparams.0 as *mut blosc2_sys::blosc2_dparams,
            io: std::ptr::null_mut(),
        };

        let ctx = Ctx::new(
            &storage,
            shape.as_slice(),
            chunkshape,
            blockshape,
            self.dtype_cstr(),
            blosc2_sys::DTYPE_NUMPY_FORMAT as _,
        )?;

        let mut sliced = MaybeUninit::<*mut blosc2_sys::b2nd_array_t>::uninit();
        unsafe {
            blosc2_sys::b2nd_get_slice(
                ctx.as_ptr(),
                sliced.as_mut_ptr(),
                self.as_ptr(),
                start.as_slice().as_ptr(),
                end.as_slice().as_ptr(),
            )
            .into_result()?;
        };
        let sliced = unsafe { sliced.assume_init() };
        unsafe { Self::from_raw_ptr(sliced.cast()) }
    }

    /// Get a slice of the array as a contiguous buffer of items.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    ///
    /// # Returns
    ///
    /// A new vector containing the extracted items, of size `prod(slice_shape)`. The items are laid out in memory
    /// according to the standard strides of the slice shape.
    pub fn slice_items<T>(&self, slice: &[std::ops::Range<usize>]) -> Result<Vec<T>, Error>
    where
        T: Dtyped,
    {
        self.check_dtype::<T>()?;

        let buf_len = slice.iter().map(|r| r.len()).product::<usize>();
        let mut dst = Vec::<MaybeUninit<T>>::with_capacity(buf_len);
        unsafe { dst.set_len(buf_len) };
        self.slice_buf_into(slice, unsafe {
            std::slice::from_raw_parts_mut(
                dst.as_mut_ptr().cast(),
                buf_len * std::mem::size_of::<T>(),
            )
        })?;
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(dst) };
        Ok(vec)
    }

    /// Get a slice of the array as a contiguous buffer of items and copy it into the provided buffer.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    /// * `buf` - the buffer to copy the extracted data into.
    ///
    /// # Returns
    ///
    /// The number of items written to the buffer.
    ///
    /// The given buffer should have at least `prod(slice_shape)` elements allocated, and the elements will be
    /// written according to the standard strides of the slice shape.
    pub fn slice_items_into<T>(
        &self,
        slice: &[std::ops::Range<usize>],
        dst: &mut [MaybeUninit<T>],
    ) -> Result<usize, Error>
    where
        T: Dtyped,
    {
        self.check_dtype::<T>()?;

        self.slice_buf_into(slice, unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), std::mem::size_of_val(dst))
        })?;
        Ok(slice.iter().map(|r| r.len()).product::<usize>())
    }

    /// Get a slice of the array as a contiguous buffer of items as bytes.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    ///
    /// # Returns
    ///
    /// A contiguous buffer of bytes representing the extracted portion of the array. The length of the returned buffer
    /// will be `itemsize * prod(slice_shape)`. The items are laid out in memory according to the standard strides of
    /// the slice shape.
    pub fn slice_buf(&self, slice: &[std::ops::Range<usize>]) -> Result<Vec<u8>, Error> {
        let buf_len = self.typesize() * slice.iter().map(|r| r.len()).product::<usize>();
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(buf_len);
        unsafe { dst.set_len(buf_len) };
        self.slice_buf_into(slice, &mut dst)?;
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    /// Get a slice of the array as a contiguous buffer of items as bytes, and copy it into the provided buffer.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to extract.
    /// * `buf` - the buffer to copy the extracted data into.
    ///
    /// # Returns
    ///
    /// The number of bytes written to the buffer.
    ///
    /// The given buffer should have at least `itemsize * prod(slice_shape)` bytes allocated, and the elements will be
    /// written according to the standard strides. The items are laid out in memory according to the standard strides of
    /// the slice shape.
    pub fn slice_buf_into(
        &self,
        slice: &[std::ops::Range<usize>],
        buf: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        self.check_slice_arg(slice)?;
        let shape = DimVec::from_slice_fn(slice, |r| r.len());
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };
        self.slice_buf_into_with_shape(slice, buf, shape.as_slice())
    }

    fn slice_buf_into_with_shape(
        &self,
        slice: &[std::ops::Range<usize>],
        buf: &mut [MaybeUninit<u8>],
        buf_shape: &[usize],
    ) -> Result<usize, Error> {
        self.check_slice_arg(slice)?;
        if buf_shape.len() != self.ndim() {
            crate::trace!(
                "Buffer shape length {} must match array dimension {}",
                buf_shape.len(),
                self.ndim()
            );
            return Err(Error::InvalidParam);
        }
        let start = DimVec::from_slice_fn(slice, |r| r.start as i64);
        let end = DimVec::from_slice_fn(slice, |r| r.end as i64);
        let buf_shape = DimVec::from_slice_fn(buf_shape, |s| *s as i64);
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let start = unsafe { start.unwrap_unchecked() };
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let end = unsafe { end.unwrap_unchecked() };
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let buf_shape = unsafe { buf_shape.unwrap_unchecked() };

        unsafe {
            blosc2_sys::b2nd_get_slice_cbuffer(
                self.as_ptr(),
                start.as_slice().as_ptr(),
                end.as_slice().as_ptr(),
                buf.as_mut_ptr().cast(),
                buf_shape.as_slice().as_ptr(),
                buf.len() as i64,
            )
            .into_result()?;
        }
        Ok(self.typesize()
            * buf_shape
                .as_slice()
                .iter()
                .map(|s| *s as usize)
                .product::<usize>())
    }

    /// Set all elements in a slice of the array, copying from the provided [`ndarray::ArrayBase`].
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to modify.
    /// * `data` - the source data to copy from, as an [`ndarray::ArrayBase`]. The shape of `data` must match the shape
    ///   of the slice.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    #[cfg(feature = "ndarray")]
    pub fn set_slice<S, T, D>(
        &mut self,
        slice: &[std::ops::Range<usize>],
        data: &ndarray::ArrayBase<S, D>,
    ) -> Result<(), Error>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        self.check_dtype::<T>()?;
        // Safety: we have checked the dtype
        unsafe { self.set_slice_without_dtype_check(slice, data) }
    }

    /// Set all elements in a slice of the array, copying from the provided [`ndarray::ArrayBase`], without checking
    /// the dtype.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to modify.
    /// * `data` - the source data to copy from, as an [`ndarray::ArrayBase`]. The shape of `data` must match the shape
    ///   of the slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the dtype of `T` matches the dtype of the ndarray. The size, alignment, and inner
    /// fields should all be compatible. See [`Dtyped`] for more information.
    #[cfg(feature = "ndarray")]
    unsafe fn set_slice_without_dtype_check<S, T, D>(
        &mut self,
        slice: &[std::ops::Range<usize>],
        data: &ndarray::ArrayBase<S, D>,
    ) -> Result<(), Error>
    where
        T: Copy + 'static,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        if slice.len() != data.ndim()
            || slice
                .iter()
                .zip(data.shape())
                .any(|(s, &dim)| s.len() != dim)
        {
            crate::trace!(
                "Shape mismatch: slice indices {:?}, set data {:?}",
                slice,
                data.shape()
            );
            return Err(Error::InvalidParam);
        }

        let data = data.as_standard_layout();
        let data = data
            .as_slice()
            .expect("arr.as_standard_layout() should be contiguous");
        let data_buf = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        Self::set_slice_buf(self, slice, data_buf)
    }

    /// Set all elements in a slice of the array, copying from the provided contiguous buffer of items.
    ///
    /// The type `T` must implement the [`Dtyped`] trait, and the function will succeed only if `T` matches the dtype
    /// of the ndarray.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to modify.
    /// * `items` - the source data to copy from, as a contiguous buffer of `T` elements.
    ///   The length of `items` must be equal to `prod(slice_shape)` and the elements should be laid out according to
    ///   the standard strides of the slice shape.
    pub fn set_slice_items<T>(
        &mut self,
        slice: &[std::ops::Range<usize>],
        items: &[T],
    ) -> Result<(), Error>
    where
        T: Dtyped,
    {
        self.check_slice_arg(slice)?;

        self.set_slice_buf(slice, unsafe {
            std::slice::from_raw_parts(items.as_ptr().cast(), std::mem::size_of_val(items))
        })
    }

    /// Set all elements in a slice of the array, copying from the provided contiguous buffer of items bytes.
    ///
    /// # Arguments
    ///
    /// * `slice` - a range per dimension, representing the portion of the array to modify.
    /// * `items` - the source data to copy from, as a contiguous buffer of bytes. The length of `items` must be equal
    ///   to `itemsize * prod(slice_shape)` and the elements should be laid out according to the standard strides
    ///   of the slice shape.
    pub fn set_slice_buf(
        &mut self,
        slice: &[std::ops::Range<usize>],
        items: &[u8],
    ) -> Result<(), Error> {
        self.check_slice_arg(slice)?;
        let shape = DimVec::from_slice_fn(slice, |r| r.len());
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };
        self.set_slice_buf_with_shape(slice, items, shape.as_slice())
    }

    fn set_slice_buf_with_shape(
        &mut self,
        slice: &[std::ops::Range<usize>],
        items: &[u8],
        items_shape: &[usize],
    ) -> Result<(), Error> {
        self.check_slice_arg(slice)?;
        if items_shape.len() != self.ndim() {
            crate::trace!(
                "Buffer shape length {} must match array dimension {}",
                items_shape.len(),
                self.ndim()
            );
            return Err(Error::InvalidParam);
        }
        let start = DimVec::from_slice_fn(slice, |r| r.start as i64);
        let end = DimVec::from_slice_fn(slice, |r| r.end as i64);
        let items_shape = DimVec::from_slice_fn(items_shape, |s| *s as i64);
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let start = unsafe { start.unwrap_unchecked() };
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let end = unsafe { end.unwrap_unchecked() };
        // Safety: checked slice.len()==self.ndim() in check_slice_arg, and we know self.ndim()<=MAX_DIM
        let items_shape = unsafe { items_shape.unwrap_unchecked() };

        unsafe {
            blosc2_sys::b2nd_set_slice_cbuffer(
                items.as_ptr().cast(),
                items_shape.as_slice().as_ptr(),
                items.len() as i64,
                start.as_slice().as_ptr(),
                end.as_slice().as_ptr(),
                self.as_ptr(),
            )
            .into_result()?;
        }
        Ok(())
    }

    fn check_slice_arg(&self, slice: &[std::ops::Range<usize>]) -> Result<(), Error> {
        if slice.len() != self.ndim()
            || slice
                .iter()
                .zip(self.shape())
                .any(|(s, &dim)| s.start > s.end || s.end > dim as usize)
        {
            crate::trace!("Invalid slice {:?} for shape {:?}", slice, self.shape());
            return Err(Error::InvalidParam);
        }
        Ok(())
    }

    #[cfg(feature = "ndarray")]
    fn check_dimension_type_for_read<D>(&self) -> Result<(), Error>
    where
        D: ndarray::Dimension,
    {
        let ndim = self.ndim();
        if D::NDIM.is_some() && D::NDIM.unwrap() != ndim {
            crate::trace!(
                "Dimension mismatch: ndim {}, requested {}",
                self.ndim(),
                D::NDIM.unwrap()
            );
            return Err(Error::InvalidParam);
        }
        Ok(())
    }

    fn as_ptr(&self) -> *mut blosc2_sys::b2nd_array_t {
        self.ptr.as_ptr()
    }

    /// Returns a raw pointer to the underlying `blosc2_sys::b2nd_array_t` object.
    pub fn as_raw_ptr(&self) -> *const () {
        self.as_ptr().cast()
    }

    /// Consumes the `Ndarray` and returns a raw pointer to the underlying `blosc2_sys::b2nd_array_t` object.
    ///
    /// The ownership over the object is passed to the caller, which should call `blosc2_sys::b2nd_free` when there
    /// is no more need for it.
    pub fn into_raw_ptr(self) -> *mut () {
        let ptr = self.as_ptr().cast();
        std::mem::forget(self);
        ptr
    }
}

impl Ndarray {
    /// Saves the array to the specified file.
    ///
    /// Either a single file or a directory will be created at `urlpath`, depending if the array is sparse or
    /// contiguous. See [`SChunkStorageParams`]
    ///
    /// # Arguments
    ///
    /// * `urlpath` - The path to the file where the array should be saved.
    /// * `append` - Whether to append to the file or write a new one. If this is `true`, the file should already
    ///   exists.
    ///
    /// # Returns
    ///
    /// The offset of the saved array in the file.
    pub fn save(&self, urlpath: &Path, append: bool) -> Result<u64, Error> {
        let urlpath = path2cstr(urlpath);
        let offset = if !append {
            unsafe { blosc2_sys::b2nd_save(self.as_ptr(), urlpath.as_ptr().cast_mut()) }
                .into_result()?;
            0
        } else {
            unsafe { blosc2_sys::b2nd_save_append(self.as_ptr(), urlpath.as_ptr()) }
                .into_result()? as u64
        };
        Ok(offset)
    }

    /// Create a copy of this array in memory.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to use for the new array. The chunkshape and blockshape are optional, overriding
    ///   the original array's parameters.
    pub fn copy(&self, params: &NdarrayParams) -> Result<Self, Error> {
        self.copy_to(SChunkStorageParams::in_memory(), params)
    }

    /// Copy the array to the given storage.
    ///
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage parameters to use for the new array.
    /// * `params` - The parameters to use for the new array. The chunkshape and blockshape are optional, overriding
    ///   the original array's parameters.
    pub fn copy_to(
        &self,
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_impl(InitArgs::CopyFromNdarray(self), storage, params)
    }

    #[allow(unused)]
    fn check_dtype<T>(&self) -> Result<(), Error>
    where
        T: Dtyped,
    {
        let t_dtype = T::dtype();
        if self.dtype != t_dtype {
            crate::trace!("Dtype mismatch: {:?} != {:?}", self.dtype, t_dtype);
            return Err(Error::InvalidParam);
        }
        Ok(())
    }
}
impl Drop for Ndarray {
    fn drop(&mut self) {
        unsafe {
            blosc2_sys::b2nd_free(self.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::mem::MaybeUninit;
    use std::path::PathBuf;

    use rand::distr::weighted::WeightedIndex;
    use rand::prelude::*;

    use super::{Ndarray, NdarrayInitValue, NdarrayParams};
    use crate::nd::InitValueInner;
    use crate::nd::{
        Dtype, DtypeKind, DtypeScalarKind, SChunkOpenOptions, SChunkStorageParams, MAX_DIM,
    };
    use crate::util::tests::{ceil_to_multiple, rand_cparams, rand_dparams};
    use crate::util::{f16, Complex, MmapMode};

    #[cfg(feature = "ndarray")]
    use super::Dtyped;

    #[cfg(feature = "ndarray")]
    #[test]
    fn new_simple() {
        let array1 = ndarray::array!([[1_i32, 2], [3, 4],]);
        let b2nd = Ndarray::from_ndarray(
            &array1,
            NdarrayParams::default()
                .blockshape(Some(&[1, 1, 1]))
                .chunkshape(Some(&[1, 1, 1])),
        )
        .unwrap();

        let array2: ndarray::ArrayD<i32> = b2nd.to_ndarray().unwrap();
        assert_eq!(array1.view().into_dyn(), array2.view().into_dyn());

        let array3: ndarray::Array3<i32> = b2nd.to_ndarray().unwrap();
        assert_eq!(array1.view().into_dyn(), array3.view().into_dyn());
    }

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0xae360b0cc77f052f);
        for _ in 0..30 {
            let (shape, dtype, params) = rand_params(&mut rand);
            let data = rand_data(&dtype, &shape, &mut rand);
            let storage = rand_storage(&mut rand);
            let array =
                Ndarray::from_items_bytes_at(&data, dtype, &shape, storage.params(), &params)
                    .unwrap();

            assert_eq!(
                shape,
                array
                    .shape()
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>()
            );
            let array_data = array.to_items_bytes().unwrap();
            assert_eq!(data, array_data);
        }
    }

    #[test]
    fn new_repeated_value() {
        let mut rand = StdRng::seed_from_u64(0x4a551bfb4793837b);
        for _ in 0..30 {
            let (shape, dtype, params) = rand_params(&mut rand);
            let storage = rand_storage(&mut rand);
            let value = (&mut rand)
                .random_iter()
                .take(dtype.itemsize())
                .collect::<Vec<_>>();
            let value = {
                let values = [
                    NdarrayInitValue::zero_of(dtype.clone()),
                    unsafe { NdarrayInitValue::uninit_of(dtype.clone()) },
                    NdarrayInitValue::value_bytes(value, dtype.clone()).unwrap(),
                ];
                values.choose(&mut rand).unwrap().clone()
            };
            let array = Ndarray::new_at(value.clone(), &shape, storage.params(), &params).unwrap();

            assert_eq!(
                shape,
                array
                    .shape()
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>()
            );
            let array_data = array.to_items_bytes().unwrap();
            assert_eq!(
                array_data.len(),
                dtype.itemsize() * shape.iter().product::<usize>()
            );
            match value.value {
                InitValueInner::Zero => assert!(array_data.iter().all(|&b| b == 0)),
                InitValueInner::Uninit => {}
                InitValueInner::Value(items) => assert!(array_data
                    .chunks_exact(dtype.itemsize())
                    .all(|chunk| chunk == items)),
                InitValueInner::Nan => unreachable!(),
            }
        }
    }

    #[test]
    fn new_from_items() {
        fn test_impl<T>(repeat: usize, rand: &mut impl Rng)
        where
            T: Dtyped + PartialEq + std::fmt::Debug,
        {
            for _ in 0..repeat {
                let dtype = T::dtype();
                let (shape, params) = rand_params_with_dtype(&dtype, rand);
                let items = rand_data_typed::<T>(&shape, rand);
                let array = Ndarray::from_items(&items, &shape, &params).unwrap();

                assert_eq!(
                    shape,
                    array
                        .shape()
                        .iter()
                        .map(|s| *s as usize)
                        .collect::<Vec<_>>()
                );
                let items_new = array.to_items::<T>().unwrap();
                assert_eq!(items, items_new);
            }
        }

        let mut rand = StdRng::seed_from_u64(0xa3eae3352da3f602);
        for i in 0..=17 {
            match i {
                0 => test_impl::<i8>(1, &mut rand),
                2 => test_impl::<i16>(1, &mut rand),
                4 => test_impl::<i32>(8, &mut rand),
                6 => test_impl::<i64>(8, &mut rand),
                1 => test_impl::<u8>(1, &mut rand),
                3 => test_impl::<u16>(1, &mut rand),
                5 => test_impl::<u32>(1, &mut rand),
                7 => test_impl::<u64>(1, &mut rand),
                8 => {
                    cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                        // need the half crate to implement PartialEq for f16
                        test_impl::<f16>(1, &mut rand);
                    } }
                }
                9 => test_impl::<f32>(8, &mut rand),
                10 => test_impl::<f64>(8, &mut rand),
                11 => test_impl::<Complex<f32>>(2, &mut rand),
                12 => test_impl::<Complex<f64>>(2, &mut rand),
                13 => test_impl::<bool>(1, &mut rand),
                14 => test_impl::<Point>(3, &mut rand),
                15 => test_impl::<Person>(2, &mut rand),
                16 => test_impl::<PersonAligned>(2, &mut rand),
                17 => test_impl::<AudioSample>(2, &mut rand),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn new_from_items_bytes() {
        let mut rand = StdRng::seed_from_u64(0x5ac98edb0400b82f);
        for _ in 0..30 {
            let (shape, dtype, params) = rand_params(&mut rand);
            let storage = rand_storage(&mut rand);
            let data = rand_data(&dtype, &shape, &mut rand);
            let array =
                Ndarray::from_items_bytes_at(&data, dtype, &shape, storage.params(), &params)
                    .unwrap();

            assert_eq!(
                shape,
                array
                    .shape()
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>()
            );
            let array_data = array.to_items_bytes().unwrap();
            assert_eq!(data, array_data);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn new_from_ndarray() {
        fn test_impl<T>(repeat: usize, rand: &mut impl Rng)
        where
            T: Dtyped + PartialEq + std::fmt::Debug,
        {
            for _ in 0..repeat {
                let dtype = T::dtype();
                let (shape, params) = rand_params_with_dtype(&dtype, rand);
                let storage = rand_storage(rand);
                let array_orig = rand_ndarray::<T>(&shape, rand);
                let array =
                    Ndarray::from_ndarray_at(&array_orig, storage.params(), &params).unwrap();

                assert_eq!(
                    shape,
                    array
                        .shape()
                        .iter()
                        .map(|s| *s as usize)
                        .collect::<Vec<_>>()
                );
                let array_new: ndarray::ArrayD<T> = array.to_ndarray().unwrap();
                assert_arr_eq_nan!(&array_orig, &array_new);
            }
        }

        let mut rand = StdRng::seed_from_u64(0x74970cbc6e4c8b4b);
        for i in 0..=17 {
            match i {
                0 => test_impl::<i8>(1, &mut rand),
                2 => test_impl::<i16>(1, &mut rand),
                4 => test_impl::<i32>(8, &mut rand),
                6 => test_impl::<i64>(8, &mut rand),
                1 => test_impl::<u8>(1, &mut rand),
                3 => test_impl::<u16>(1, &mut rand),
                5 => test_impl::<u32>(1, &mut rand),
                7 => test_impl::<u64>(1, &mut rand),
                8 => {
                    cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                        // need the half crate to implement PartialEq for f16
                        test_impl::<f16>(1, &mut rand);
                    } }
                }
                9 => test_impl::<f32>(8, &mut rand),
                10 => test_impl::<f64>(8, &mut rand),
                11 => test_impl::<Complex<f32>>(2, &mut rand),
                12 => test_impl::<Complex<f64>>(2, &mut rand),
                13 => test_impl::<bool>(1, &mut rand),
                14 => test_impl::<Point>(3, &mut rand),
                15 => test_impl::<Person>(2, &mut rand),
                16 => test_impl::<PersonAligned>(2, &mut rand),
                17 => test_impl::<AudioSample>(2, &mut rand),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn save_open() {
        let mut rand = StdRng::seed_from_u64(0xe62c3a15347c0cf0);
        for _ in 0..30 {
            let temp_dir = tempfile::TempDir::new().unwrap();
            let path = temp_dir.path().join("test.b2nd");

            let (shape, dtype, params) = rand_params(&mut rand);
            let data = rand_data(&dtype, &shape, &mut rand);

            let padding = rand.random::<bool>().then(|| {
                let padding = rand.random_range(0..=4096);
                std::fs::write(&path, vec![0; padding]).unwrap();
                padding
            });
            let array = Ndarray::from_items_bytes_at(
                &data,
                dtype,
                &shape,
                SChunkStorageParams {
                    contiguous: rand.random::<bool>(),
                    urlpath: None,
                },
                &params,
            )
            .unwrap();
            let written_offset = array.save(&path, padding.is_some()).unwrap();
            assert_eq!(written_offset, padding.unwrap_or(0) as u64);

            let offset = padding.unwrap_or(0) as u64;
            let mmap = (array.is_contiguous() && rand.random::<bool>()).then(|| {
                *[MmapMode::Read, MmapMode::ReadWrite, MmapMode::Cow]
                    .choose(&mut rand)
                    .unwrap()
            });

            let array = match (offset, mmap) {
                (0, None) => Ndarray::open(&path).unwrap(),
                (_, _) => Ndarray::open_with_options(&path, unsafe {
                    SChunkOpenOptions::new().offset(offset).mmap(mmap)
                })
                .unwrap(),
            };

            assert_eq!(
                shape,
                array
                    .shape()
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>()
            );
            let array_data = array.to_items_bytes().unwrap();
            assert_eq!(data, array_data);
        }
    }

    #[test]
    fn copy() {
        let mut rand = StdRng::seed_from_u64(0xa2c6877c4f542c7e);
        for _ in 0..30 {
            let (shape, dtype, params) = rand_params(&mut rand);
            let storage = rand_storage(&mut rand);
            let data = rand_data(&dtype, &shape, &mut rand);
            let array1 =
                Ndarray::from_items_bytes_at(&data, dtype, &shape, storage.params(), &params)
                    .unwrap();

            let mut params2 = NdarrayParams::default();
            if rand.random::<bool>() {
                let (chunkshape2, blockshape2) = rand_chunk_block_shapes(&shape, &mut rand);
                params2.chunkshape(Some(&chunkshape2));
                params2.blockshape(Some(&blockshape2));
            };

            let storage2 = rand_storage(&mut rand);
            let array2 = array1.copy_to(storage2.params(), &params2).unwrap();

            assert_eq!(array1.shape(), array2.shape());
            assert_eq!(array1.dtype(), array2.dtype());
            assert_eq!(
                array1.to_items_bytes().unwrap(),
                array2.to_items_bytes().unwrap()
            );
        }
    }

    #[test]
    fn get() {
        fn test_impl<T>(repeat: usize, rand: &mut impl Rng)
        where
            T: Dtyped + PartialEq + std::fmt::Debug,
        {
            for _ in 0..repeat {
                let dtype = T::dtype();
                let (shape, params) = rand_params_with_dtype(&dtype, rand);
                let storage = rand_storage(rand);
                let array_orig = rand_ndarray::<T>(&shape, rand);
                let array =
                    Ndarray::from_ndarray_at(&array_orig, storage.params(), &params).unwrap();

                for _ in 0..30 {
                    let idx = shape
                        .iter()
                        .map(|&s| rand.random_range(0..s))
                        .collect::<Vec<_>>();
                    let item = array.get::<T>(&idx).unwrap();
                    assert_arr_eq_nan!(
                        // we use the array_eq to compare nans properly
                        &ndarray::array![item],
                        &ndarray::array![array_orig[idx.as_slice()]]
                    );
                }
            }
        }

        let mut rand = StdRng::seed_from_u64(0xc885561584f1500c);
        for i in 0..=17 {
            match i {
                0 => test_impl::<i8>(1, &mut rand),
                2 => test_impl::<i16>(1, &mut rand),
                4 => test_impl::<i32>(8, &mut rand),
                6 => test_impl::<i64>(8, &mut rand),
                1 => test_impl::<u8>(1, &mut rand),
                3 => test_impl::<u16>(1, &mut rand),
                5 => test_impl::<u32>(1, &mut rand),
                7 => test_impl::<u64>(1, &mut rand),
                8 => {
                    cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                        // need the half crate to implement PartialEq for f16
                        test_impl::<f16>(1, &mut rand);
                    } }
                }
                9 => test_impl::<f32>(8, &mut rand),
                10 => test_impl::<f64>(8, &mut rand),
                11 => test_impl::<Complex<f32>>(2, &mut rand),
                12 => test_impl::<Complex<f64>>(2, &mut rand),
                13 => test_impl::<bool>(1, &mut rand),
                14 => test_impl::<Point>(3, &mut rand),
                15 => test_impl::<Person>(2, &mut rand),
                16 => test_impl::<PersonAligned>(2, &mut rand),
                17 => test_impl::<AudioSample>(2, &mut rand),
                _ => unreachable!(),
            }
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn slice() {
        fn test_impl<T>(repeat: usize, rand: &mut impl Rng)
        where
            T: Dtyped + PartialEq + std::fmt::Debug,
        {
            for _ in 0..repeat {
                let dtype = T::dtype();
                let (shape, params) = rand_params_with_dtype(&dtype, rand);
                let storage = rand_storage(rand);
                let array_orig = rand_ndarray::<T>(&shape, rand);
                let array =
                    Ndarray::from_ndarray_at(&array_orig, storage.params(), &params).unwrap();

                for _ in 0..10 {
                    let slice = shape
                        .iter()
                        .map(|&s| {
                            let begin = rand.random_range(0..=s);
                            let end = rand.random_range(begin..=s);
                            begin..end
                        })
                        .collect::<Vec<_>>();
                    let slice_data: ndarray::ArrayD<T> = array.slice(&slice).unwrap();

                    let slice_data_expected = ndarray::ArrayD::from_shape_fn(
                        slice.iter().map(|r| r.len()).collect::<Vec<_>>(),
                        |indices| {
                            let index = slice
                                .iter()
                                .enumerate()
                                .map(|(i, r)| r.start + indices[i])
                                .collect::<Vec<_>>();
                            array_orig[index.as_slice()]
                        },
                    );
                    assert_eq!(slice_data_expected.shape(), slice_data.shape());
                    assert_eq!(slice_data_expected, slice_data);
                }
            }
        }

        let mut rand = StdRng::seed_from_u64(0x1a55f4aca9291cef);
        for i in 0..=17 {
            match i {
                0 => test_impl::<i8>(1, &mut rand),
                2 => test_impl::<i16>(1, &mut rand),
                4 => test_impl::<i32>(8, &mut rand),
                6 => test_impl::<i64>(8, &mut rand),
                1 => test_impl::<u8>(1, &mut rand),
                3 => test_impl::<u16>(1, &mut rand),
                5 => test_impl::<u32>(1, &mut rand),
                7 => test_impl::<u64>(1, &mut rand),
                8 => {
                    cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                        // need the half crate to implement PartialEq for f16
                        test_impl::<f16>(1, &mut rand);
                    } }
                }
                9 => test_impl::<f32>(8, &mut rand),
                10 => test_impl::<f64>(8, &mut rand),
                11 => test_impl::<Complex<f32>>(2, &mut rand),
                12 => test_impl::<Complex<f64>>(2, &mut rand),
                13 => test_impl::<bool>(1, &mut rand),
                14 => test_impl::<Point>(3, &mut rand),
                15 => test_impl::<Person>(2, &mut rand),
                16 => test_impl::<PersonAligned>(2, &mut rand),
                17 => test_impl::<AudioSample>(2, &mut rand),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn slice_items() {
        fn test_impl<T>(repeat: usize, rand: &mut impl Rng)
        where
            T: Dtyped + PartialEq + std::fmt::Debug,
        {
            for _ in 0..repeat {
                let dtype = T::dtype();
                let (shape, params) = rand_params_with_dtype(&dtype, rand);
                let storage = rand_storage(rand);
                let array_orig = rand_ndarray::<T>(&shape, rand);
                let array =
                    Ndarray::from_ndarray_at(&array_orig, storage.params(), &params).unwrap();

                for _ in 0..10 {
                    let slice = shape
                        .iter()
                        .map(|&s| {
                            let begin = rand.random_range(0..=s);
                            let end = rand.random_range(begin..=s);
                            begin..end
                        })
                        .collect::<Vec<_>>();
                    let slice_data = array.slice_items::<T>(&slice).unwrap();
                    let slice_data = ndarray::ArrayD::from_shape_vec(
                        slice.iter().map(|r| r.len()).collect::<Vec<_>>(),
                        slice_data,
                    )
                    .unwrap();

                    let slice_data_expected = ndarray::ArrayD::from_shape_fn(
                        slice.iter().map(|r| r.len()).collect::<Vec<_>>(),
                        |indices| {
                            let index = slice
                                .iter()
                                .enumerate()
                                .map(|(i, r)| r.start + indices[i])
                                .collect::<Vec<_>>();
                            array_orig[index.as_slice()]
                        },
                    );
                    assert_eq!(slice_data_expected.shape(), slice_data.shape());
                    assert_arr_eq_nan!(&slice_data_expected, &slice_data);
                }
            }
        }

        let mut rand = StdRng::seed_from_u64(0x63820ee4a404a912);
        for i in 0..=17 {
            match i {
                0 => test_impl::<i8>(1, &mut rand),
                2 => test_impl::<i16>(1, &mut rand),
                4 => test_impl::<i32>(8, &mut rand),
                6 => test_impl::<i64>(8, &mut rand),
                1 => test_impl::<u8>(1, &mut rand),
                3 => test_impl::<u16>(1, &mut rand),
                5 => test_impl::<u32>(1, &mut rand),
                7 => test_impl::<u64>(1, &mut rand),
                8 => {
                    cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                        // need the half crate to implement PartialEq for f16
                        test_impl::<f16>(1, &mut rand);
                    } }
                }
                9 => test_impl::<f32>(8, &mut rand),
                10 => test_impl::<f64>(8, &mut rand),
                11 => test_impl::<Complex<f32>>(2, &mut rand),
                12 => test_impl::<Complex<f64>>(2, &mut rand),
                13 => test_impl::<bool>(1, &mut rand),
                14 => test_impl::<Point>(3, &mut rand),
                15 => test_impl::<Person>(2, &mut rand),
                16 => test_impl::<PersonAligned>(2, &mut rand),
                17 => test_impl::<AudioSample>(2, &mut rand),
                _ => unreachable!(),
            }
        }
    }

    fn default_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
        let mut strides = shape
            .iter()
            .rev()
            .scan(itemsize, |acc, s| {
                let stride = *acc;
                *acc *= s;
                Some(stride as isize)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        strides
    }

    fn index2offset(index: &[usize], strides: &[isize]) -> isize {
        index
            .iter()
            .zip(strides)
            .map(|(idx, stride)| (*idx as isize) * stride)
            .sum()
    }

    #[test]
    fn slice_blosc() {
        let mut rand = StdRng::seed_from_u64(0x548631dfed77f37a);
        for _ in 0..30 {
            let (shape, dtype, params) = rand_params(&mut rand);
            let storage = rand_storage(&mut rand);
            let data = rand_data(&dtype, &shape, &mut rand);
            let array = Ndarray::from_items_bytes_at(
                &data,
                dtype.clone(),
                &shape,
                storage.params(),
                &params,
            )
            .unwrap();
            let orig_strides = default_strides(&shape, dtype.itemsize());

            for _ in 0..10 {
                let slice = shape
                    .iter()
                    .map(|&s| {
                        let begin = rand.random_range(0..=s);
                        let end = rand.random_range(begin..=s);
                        begin..end
                    })
                    .collect::<Vec<_>>();
                let mut slice_params = NdarrayParams::default();
                if rand.random::<bool>() {
                    let (chunkshape2, blockshape2) = rand_chunk_block_shapes(&shape, &mut rand);
                    slice_params.chunkshape(Some(&chunkshape2));
                    slice_params.blockshape(Some(&blockshape2));
                };
                let slice_ndarray = array.slice_blosc(&slice, &slice_params).unwrap();

                let slice_shape = slice_ndarray
                    .shape()
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>();
                assert_eq!(
                    slice.iter().map(|r| r.len()).collect::<Vec<_>>(),
                    slice_shape
                );
                assert_eq!(slice_ndarray.typesize(), dtype.itemsize());
                let slice_strides = default_strides(&slice_shape, dtype.itemsize());
                let slice_data_buf = slice_ndarray.to_items_bytes().unwrap();

                let mut iter = ArrayIter::new(&slice_shape, &slice_strides);
                while let Some((index, _)) = iter.next() {
                    let orig_index = index
                        .iter()
                        .zip(slice.iter())
                        .map(|(&i, r)| i + r.start)
                        .collect::<Vec<_>>();
                    let orig_offset = index2offset(&orig_index, &orig_strides) as usize;
                    let slice_offset = index2offset(index, &slice_strides) as usize;
                    assert_eq!(data[orig_offset], slice_data_buf[slice_offset]);
                }
            }
        }
    }

    pub(crate) fn rand_params(rand: &mut impl Rng) -> (Vec<usize>, Dtype, NdarrayParams) {
        let dtype = rand_dtype(rand);
        let (shape, params) = rand_params_with_dtype(&dtype, rand);
        (shape, dtype, params)
    }

    fn rand_params_with_dtype(dtype: &Dtype, rand: &mut impl Rng) -> (Vec<usize>, NdarrayParams) {
        let shape = loop {
            let shape = rand_shape(rand);
            if dtype.itemsize() * shape.iter().product::<usize>() < 1 << 28 {
                break shape;
            }
        };

        let (chunkshape, blockshape) = rand_chunk_block_shapes(&shape, rand);
        let mut params = NdarrayParams::default();
        params
            .cparams(rand_cparams(rand))
            .dparams(rand_dparams(rand))
            .chunkshape(Some(&chunkshape))
            .blockshape(Some(&blockshape));
        (shape, params)
    }

    fn rand_shape(rand: &mut impl Rng) -> Vec<usize> {
        let ndim = rand_ndim(rand);
        let max_dim_len = if ndim < 4 { 8 } else { 4 };
        let mut dim_dist = usize_dist_most_likely_small(1..max_dim_len + 1, rand);
        (0..ndim).map(|_| dim_dist()).collect::<Vec<_>>()
    }

    fn rand_ndim(rand: &mut impl Rng) -> usize {
        usize_dist_most_likely_small(1..MAX_DIM + 1, rand)()
    }

    fn rand_dtype(rand: &mut impl Rng) -> Dtype {
        fn rand_dtype_impl(depth: usize, rand: &mut impl Rng) -> Dtype {
            if rand.random_range(0..=depth) == 0 {
                let kinds = [
                    DtypeScalarKind::I8,
                    DtypeScalarKind::U8,
                    DtypeScalarKind::I16,
                    DtypeScalarKind::U16,
                    DtypeScalarKind::I32,
                    DtypeScalarKind::U32,
                    DtypeScalarKind::I64,
                    DtypeScalarKind::U64,
                    DtypeScalarKind::F16,
                    DtypeScalarKind::F32,
                    DtypeScalarKind::F64,
                    DtypeScalarKind::ComplexF32,
                    DtypeScalarKind::ComplexF64,
                    DtypeScalarKind::Bool,
                ];
                let kind = *kinds.choose(rand).unwrap();
                let mut dtype = Dtype::of_scalar(kind);
                if rand.random_range(0..=depth) == 0 {
                    let shape_len = usize_dist_most_likely_small(1..4, rand)();
                    let shape = (0..shape_len)
                        .map(|_| usize_dist_most_likely_small(1..4, rand)())
                        .collect::<Vec<_>>();
                    dtype = dtype.with_shape(shape).unwrap();
                }
                dtype
            } else {
                let fields_count = usize_dist_most_likely_small(1..4, rand)();
                let aligned = rand.random::<bool>();
                let mut struct_size = 0;
                let mut fields = HashMap::new();
                for i in 0..fields_count {
                    let name = format!("field_{}", i);
                    let dtype = rand_dtype_impl(depth - 1, rand);
                    if aligned {
                        struct_size = ceil_to_multiple(struct_size, dtype.alignment());
                    }
                    let offset = struct_size;
                    struct_size += dtype.itemsize();
                    fields.insert(name, (offset, dtype));
                }
                let alignment = if aligned {
                    let alignment = fields
                        .values()
                        .map(|(_offset, dtype)| dtype.alignment())
                        .max()
                        .unwrap_or(1);
                    struct_size = ceil_to_multiple(struct_size, alignment);
                    alignment
                } else {
                    1
                };
                let mut shape = Vec::new();
                if rand.random_range(0..=depth - 1) == 0 {
                    let shape_len = usize_dist_most_likely_small(1..4, rand)();
                    shape = (0..shape_len)
                        .map(|_| usize_dist_most_likely_small(1..4, rand)())
                        .collect::<Vec<_>>();
                    struct_size *= shape.iter().product::<usize>();
                }
                Dtype::new(DtypeKind::Struct { fields }, shape, struct_size, alignment).unwrap()
            }
        }
        loop {
            let dtype = rand_dtype_impl(2, rand);
            if dtype.itemsize() <= blosc2_sys::BLOSC_MAX_TYPESIZE as usize {
                return dtype;
            }
        }
    }

    fn rand_data_typed<T>(shape: &[usize], rand: &mut impl Rng) -> Vec<T>
    where
        T: Dtyped,
    {
        rand_data(&T::dtype(), shape, rand)
            .chunks_exact(std::mem::size_of::<T>())
            .map(|chunk| {
                let mut value = MaybeUninit::<T>::uninit();
                unsafe {
                    value
                        .as_mut_ptr()
                        .cast::<u8>()
                        .copy_from_nonoverlapping(chunk.as_ptr(), std::mem::size_of::<T>())
                };
                unsafe { value.assume_init() }
            })
            .collect()
    }

    pub(crate) fn rand_data(dtype: &Dtype, shape: &[usize], rand: &mut impl Rng) -> Vec<u8> {
        struct GenOp<R> {
            rand: R,
        }
        impl<R: Rng> UnaryOp for GenOp<R> {
            fn i8(&mut self, ptr: *mut i8, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn i16(&mut self, ptr: *mut i16, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn i32(&mut self, ptr: *mut i32, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn i64(&mut self, ptr: *mut i64, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn u8(&mut self, ptr: *mut u8, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn u16(&mut self, ptr: *mut u16, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn u32(&mut self, ptr: *mut u32, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn u64(&mut self, ptr: *mut u64, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn f16(&mut self, ptr: *mut f16, _idx: &[usize]) {
                cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                    let value = f16::from_f32(self.rand.random());
                } else {
                    let value = f16::from_bits(self.rand.random());
                } }
                unsafe { ptr.write(value) };
            }
            fn f32(&mut self, ptr: *mut f32, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn f64(&mut self, ptr: *mut f64, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
            fn complex_f32(&mut self, ptr: *mut Complex<f32>, _idx: &[usize]) {
                let (re, im) = (self.rand.random(), self.rand.random());
                unsafe { ptr.write(Complex { re, im }) };
            }
            fn complex_f64(&mut self, ptr: *mut Complex<f64>, _idx: &[usize]) {
                let (re, im) = (self.rand.random(), self.rand.random());
                unsafe { ptr.write(Complex { re, im }) };
            }
            fn bool(&mut self, ptr: *mut bool, _idx: &[usize]) {
                unsafe { ptr.write(self.rand.random()) };
            }
        }

        let mut buf = vec![0; dtype.itemsize() * shape.iter().product::<usize>()];
        unsafe {
            unary_op(
                buf.as_mut_ptr().cast(),
                shape,
                &default_strides(shape, dtype.itemsize()),
                dtype,
                &mut GenOp { rand },
            )
        };
        buf
    }

    trait UnaryOp {
        fn i8(&mut self, ptr: *mut i8, idx: &[usize]);
        fn i16(&mut self, ptr: *mut i16, idx: &[usize]);
        fn i32(&mut self, ptr: *mut i32, idx: &[usize]);
        fn i64(&mut self, ptr: *mut i64, idx: &[usize]);
        fn u8(&mut self, ptr: *mut u8, idx: &[usize]);
        fn u16(&mut self, ptr: *mut u16, idx: &[usize]);
        fn u32(&mut self, ptr: *mut u32, idx: &[usize]);
        fn u64(&mut self, ptr: *mut u64, idx: &[usize]);
        fn f16(&mut self, ptr: *mut f16, idx: &[usize]);
        fn f32(&mut self, ptr: *mut f32, idx: &[usize]);
        fn f64(&mut self, ptr: *mut f64, idx: &[usize]);
        fn complex_f32(&mut self, ptr: *mut Complex<f32>, idx: &[usize]);
        fn complex_f64(&mut self, ptr: *mut Complex<f64>, idx: &[usize]);
        fn bool(&mut self, ptr: *mut bool, idx: &[usize]);
    }
    unsafe fn unary_op(
        data_ptr: *mut (),
        shape: &[usize],
        strides: &[isize],
        dtype: &Dtype,
        op: &mut impl UnaryOp,
    ) {
        let scalar_fields = extract_inner_scalar_fields(dtype);
        for (scalar_kind, field_offset) in scalar_fields {
            let mut iter = ArrayIter::new(shape, strides);

            macro_rules! handle_scalar_kind {
                ($method:ident, $type:ty) => {
                    while let Some((index, offset)) = iter.next() {
                        let ptr =
                            unsafe { data_ptr.offset(offset).add(field_offset).cast::<$type>() };
                        op.$method(ptr, index);
                    }
                };
            }

            match scalar_kind {
                DtypeScalarKind::I8 => handle_scalar_kind!(i8, i8),
                DtypeScalarKind::I16 => handle_scalar_kind!(i16, i16),
                DtypeScalarKind::I32 => handle_scalar_kind!(i32, i32),
                DtypeScalarKind::I64 => handle_scalar_kind!(i64, i64),
                DtypeScalarKind::U8 => handle_scalar_kind!(u8, u8),
                DtypeScalarKind::U16 => handle_scalar_kind!(u16, u16),
                DtypeScalarKind::U32 => handle_scalar_kind!(u32, u32),
                DtypeScalarKind::U64 => handle_scalar_kind!(u64, u64),
                DtypeScalarKind::F16 => handle_scalar_kind!(f16, f16),
                DtypeScalarKind::F32 => handle_scalar_kind!(f32, f32),
                DtypeScalarKind::F64 => handle_scalar_kind!(f64, f64),
                DtypeScalarKind::ComplexF32 => handle_scalar_kind!(complex_f32, Complex<f32>),
                DtypeScalarKind::ComplexF64 => handle_scalar_kind!(complex_f64, Complex<f64>),
                DtypeScalarKind::Bool => handle_scalar_kind!(bool, bool),
            }
        }
    }

    trait BinaryOp {
        fn i8(&mut self, ptr1: *mut i8, ptr2: *mut i8, idx: &[usize]);
        fn i16(&mut self, ptr1: *mut i16, ptr2: *mut i16, idx: &[usize]);
        fn i32(&mut self, ptr1: *mut i32, ptr2: *mut i32, idx: &[usize]);
        fn i64(&mut self, ptr1: *mut i64, ptr2: *mut i64, idx: &[usize]);
        fn u8(&mut self, ptr1: *mut u8, ptr2: *mut u8, idx: &[usize]);
        fn u16(&mut self, ptr1: *mut u16, ptr2: *mut u16, idx: &[usize]);
        fn u32(&mut self, ptr1: *mut u32, ptr2: *mut u32, idx: &[usize]);
        fn u64(&mut self, ptr1: *mut u64, ptr2: *mut u64, idx: &[usize]);
        fn f16(&mut self, ptr1: *mut f16, ptr2: *mut f16, idx: &[usize]);
        fn f32(&mut self, ptr1: *mut f32, ptr2: *mut f32, idx: &[usize]);
        fn f64(&mut self, ptr1: *mut f64, ptr2: *mut f64, idx: &[usize]);
        fn complex_f32(&mut self, ptr1: *mut Complex<f32>, ptr2: *mut Complex<f32>, idx: &[usize]);
        fn complex_f64(&mut self, ptr1: *mut Complex<f64>, ptr2: *mut Complex<f64>, idx: &[usize]);
        fn bool(&mut self, ptr1: *mut bool, ptr2: *mut bool, idx: &[usize]);
    }
    unsafe fn binary_op(
        data_ptr1: *mut (),
        data_ptr2: *mut (),
        shape: &[usize],
        strides1: &[isize],
        strides2: &[isize],
        dtype: &Dtype,
        op: &mut impl BinaryOp,
    ) {
        let scalar_fields = extract_inner_scalar_fields(dtype);
        for (scalar_kind, field_offset) in scalar_fields {
            let mut iter1 = ArrayIter::new(shape, strides1);
            let mut iter2 = ArrayIter::new(shape, strides2);

            macro_rules! handle_scalar_kind {
                ($method:ident, $type:ty) => {
                    loop {
                        match (iter1.next(), iter2.next()) {
                            (Some((index1, offset1)), Some((index2, offset2))) => {
                                debug_assert_eq!(index1, index2);
                                let ptr1 = unsafe {
                                    data_ptr1.offset(offset1).add(field_offset).cast::<$type>()
                                };
                                let ptr2 = unsafe {
                                    data_ptr2.offset(offset2).add(field_offset).cast::<$type>()
                                };
                                op.$method(ptr1, ptr2, index1);
                            }
                            (None, None) => break,
                            _ => unreachable!(),
                        }
                    }
                };
            }

            match scalar_kind {
                DtypeScalarKind::I8 => handle_scalar_kind!(i8, i8),
                DtypeScalarKind::I16 => handle_scalar_kind!(i16, i16),
                DtypeScalarKind::I32 => handle_scalar_kind!(i32, i32),
                DtypeScalarKind::I64 => handle_scalar_kind!(i64, i64),
                DtypeScalarKind::U8 => handle_scalar_kind!(u8, u8),
                DtypeScalarKind::U16 => handle_scalar_kind!(u16, u16),
                DtypeScalarKind::U32 => handle_scalar_kind!(u32, u32),
                DtypeScalarKind::U64 => handle_scalar_kind!(u64, u64),
                DtypeScalarKind::F16 => handle_scalar_kind!(f16, f16),
                DtypeScalarKind::F32 => handle_scalar_kind!(f32, f32),
                DtypeScalarKind::F64 => handle_scalar_kind!(f64, f64),
                DtypeScalarKind::ComplexF32 => handle_scalar_kind!(complex_f32, Complex<f32>),
                DtypeScalarKind::ComplexF64 => handle_scalar_kind!(complex_f64, Complex<f64>),
                DtypeScalarKind::Bool => handle_scalar_kind!(bool, bool),
            }
        }
    }

    fn extract_inner_scalar_fields(dtype: &Dtype) -> Vec<(DtypeScalarKind, usize)> {
        let mut inner_fields = Vec::new();
        match dtype.kind() {
            DtypeKind::Scalar {
                kind,
                endianness: _,
            } => inner_fields.push((*kind, 0)),
            DtypeKind::Struct { fields } => {
                for (offset, field) in fields.values() {
                    let mut subtype_scalars = extract_inner_scalar_fields(field);
                    for (_, subtype_scalar_offset) in subtype_scalars.iter_mut() {
                        *subtype_scalar_offset += offset;
                    }
                    inner_fields.extend(subtype_scalars);
                }
            }
        }
        if !dtype.shape().is_empty() {
            let repeated = dtype.shape().iter().product::<usize>();
            if repeated == 0 {
                return Vec::new();
            }
            assert_eq!(dtype.itemsize() % repeated, 0);
            let base_itemsize = dtype.itemsize() / repeated;
            assert!(inner_fields
                .iter()
                .all(|(_kind, offset)| (0..base_itemsize).contains(offset)));
            inner_fields = (0..repeated)
                .flat_map(|r| {
                    let base_offset = r * base_itemsize;
                    inner_fields
                        .iter()
                        .map(|(kind, offset)| (*kind, base_offset + offset))
                        .collect::<Vec<_>>()
                })
                .collect();
        }
        inner_fields
    }

    pub(crate) fn rand_storage(rand: &mut impl Rng) -> StorageWrapper {
        let contiguous = rand.random::<bool>();
        let path = rand.random::<bool>().then(|| {
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().join("test.b");
            (dir, path)
        });
        StorageWrapper {
            path,
            contiguous: contiguous,
        }
    }
    pub(crate) struct StorageWrapper {
        path: Option<(tempfile::TempDir, PathBuf)>,
        contiguous: bool,
    }
    impl StorageWrapper {
        pub(crate) fn params(&self) -> SChunkStorageParams<'_> {
            SChunkStorageParams {
                urlpath: self.path.as_ref().map(|(_, p)| p.as_path()),
                contiguous: self.contiguous,
            }
        }
    }

    fn rand_chunk_block_shapes(shape: &[usize], rand: &mut impl Rng) -> (Vec<usize>, Vec<usize>) {
        let log2 = |x: usize| (usize::BITS as usize - x.leading_zeros() as usize);
        let chunkshape = shape
            .iter()
            .map(|s| 1 << rand.random_range(0..log2(*s)))
            .collect::<Vec<_>>();
        let blockshape = chunkshape
            .iter()
            .map(|s| 1 << rand.random_range(0..log2(*s)))
            .collect::<Vec<_>>();
        (chunkshape, blockshape)
    }

    fn usize_dist_most_likely_small(
        range: std::ops::Range<usize>,
        rand: &mut impl Rng,
    ) -> impl FnMut() -> usize + '_ {
        let dist = WeightedIndex::new((0..range.len()).map(|i| range.len() - i)).unwrap();
        move || range.start + dist.sample(rand)
    }

    #[allow(unused)]
    fn array_equal_impl(
        data1_ptr: *const (),
        data2_ptr: *const (),
        dtype: &Dtype,
        shape: &[usize],
        strides1: &[isize],
        strides2: &[isize],
        equal_nan: bool,
    ) -> bool {
        struct ArrayEq {
            equal_nan: bool,
            result: bool,
        }
        impl BinaryOp for ArrayEq {
            fn i8(&mut self, ptr1: *mut i8, ptr2: *mut i8, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn i16(&mut self, ptr1: *mut i16, ptr2: *mut i16, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn i32(&mut self, ptr1: *mut i32, ptr2: *mut i32, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn i64(&mut self, ptr1: *mut i64, ptr2: *mut i64, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn u8(&mut self, ptr1: *mut u8, ptr2: *mut u8, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn u16(&mut self, ptr1: *mut u16, ptr2: *mut u16, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn u32(&mut self, ptr1: *mut u32, ptr2: *mut u32, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn u64(&mut self, ptr1: *mut u64, ptr2: *mut u64, _idx: &[usize]) {
                self.result &= unsafe { ptr1.read() == ptr2.read() };
            }
            fn f16(&mut self, ptr1: *mut f16, ptr2: *mut f16, _idx: &[usize]) {
                let (val1, val2) = unsafe { (ptr1.read(), ptr2.read()) };

                cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                    let eq = (val1 == val2) || (self.equal_nan && val1.is_nan() && val2.is_nan());
                } else {
                    let eq = val1.to_bits() == val2.to_bits();
                } }
                self.result &= eq;
            }
            fn f32(&mut self, ptr1: *mut f32, ptr2: *mut f32, _idx: &[usize]) {
                let (val1, val2) = unsafe { (ptr1.read(), ptr2.read()) };
                self.result &= (val1 == val2) || (self.equal_nan && val1.is_nan() && val2.is_nan());
            }
            fn f64(&mut self, ptr1: *mut f64, ptr2: *mut f64, _idx: &[usize]) {
                let (val1, val2) = unsafe { (ptr1.read(), ptr2.read()) };
                self.result &= (val1 == val2) || (self.equal_nan && val1.is_nan() && val2.is_nan());
            }
            fn complex_f32(
                &mut self,
                ptr1: *mut Complex<f32>,
                ptr2: *mut Complex<f32>,
                _idx: &[usize],
            ) {
                let (val1, val2) = unsafe { (ptr1.read(), ptr2.read()) };
                self.result &= (val1.re == val2.re)
                    || (self.equal_nan && val1.re.is_nan() && val2.re.is_nan());
                self.result &= (val1.im == val2.im)
                    || (self.equal_nan && val1.im.is_nan() && val2.im.is_nan());
            }
            fn complex_f64(
                &mut self,
                ptr1: *mut Complex<f64>,
                ptr2: *mut Complex<f64>,
                _idx: &[usize],
            ) {
                let (val1, val2) = unsafe { (ptr1.read(), ptr2.read()) };
                self.result &= (val1.re == val2.re)
                    || (self.equal_nan && val1.re.is_nan() && val2.re.is_nan());
                self.result &= (val1.im == val2.im)
                    || (self.equal_nan && val1.im.is_nan() && val2.im.is_nan());
            }
            fn bool(&mut self, ptr1: *mut bool, ptr2: *mut bool, _idx: &[usize]) {
                let (val1, val2) = unsafe { (ptr1.read(), ptr2.read()) };
                self.result &= val1 == val2;
            }
        }

        let mut eq = ArrayEq {
            equal_nan,
            result: true,
        };
        unsafe {
            binary_op(
                data1_ptr.cast_mut(),
                data2_ptr.cast_mut(),
                shape,
                strides1,
                strides2,
                dtype,
                &mut eq,
            )
        };
        eq.result
    }

    struct ArrayIter<'a> {
        shape: &'a [usize],
        // strides in bytes
        strides: &'a [isize],
        index: Vec<usize>,
        offset: isize,
        state: ArrayIterState,
    }
    enum ArrayIterState {
        First,
        InProgress,
        Exhausted,
    }
    impl<'a> ArrayIter<'a> {
        fn new(
            shape: &'a [usize],
            // strides in bytes
            strides: &'a [isize],
        ) -> Self {
            assert_eq!(shape.len(), strides.len());
            let state = if shape.iter().any(|&s| s == 0) {
                ArrayIterState::Exhausted
            } else {
                ArrayIterState::First
            };
            Self {
                shape,
                strides,
                index: vec![0; shape.len()],
                offset: 0,
                state,
            }
        }

        fn next(&mut self) -> Option<(&[usize], isize)> {
            match self.state {
                ArrayIterState::First => {
                    self.state = ArrayIterState::InProgress;
                    return Some((&self.index, self.offset));
                }
                ArrayIterState::InProgress => {}
                ArrayIterState::Exhausted => return None,
            }
            for i in (0..self.shape.len()).rev() {
                let size = self.shape[i];
                let stride = self.strides[i];
                let idx = &mut self.index[i];

                *idx += 1;
                if *idx < size {
                    self.offset += stride;
                    return Some((&self.index, self.offset));
                }
                *idx = 0;
                self.offset -= (size as isize - 1) * stride;
            }
            self.state = ArrayIterState::Exhausted;
            None
        }
    }

    #[cfg(feature = "ndarray")]
    use vanilla_ndarray::*;
    #[cfg(feature = "ndarray")]
    mod vanilla_ndarray {
        use std::collections::HashMap;

        use rand::prelude::*;

        use crate::nd::tests::array_equal_impl;
        use crate::nd::{Dtype, DtypeScalarKind, Dtyped};

        pub(crate) fn rand_ndarray<T>(shape: &[usize], rand: &mut impl Rng) -> ndarray::ArrayD<T>
        where
            T: Dtyped,
        {
            use std::mem::MaybeUninit;

            let len = shape.iter().product::<usize>();
            let mut buf = Vec::<MaybeUninit<T>>::with_capacity(len);
            unsafe { buf.set_len(len) };
            {
                let buf_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        buf.as_mut_ptr().cast::<u8>(),
                        len * std::mem::size_of::<T>(),
                    )
                };
                for (buf_elm, val) in buf_data.iter_mut().zip(rand.random_iter::<u8>()) {
                    *buf_elm = val;
                }
            }
            let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(buf) };
            ndarray::ArrayD::from_shape_vec(shape, buf).unwrap()
        }

        macro_rules! assert_arr_eq_nan {
            ($left:expr, $right:expr $(,)?) => {{
                assert!(crate::nd::tests::ndarray_eq_nan($left, $right), )
            }};
            ($left:expr, $right:expr, $($arg:tt)*) => {{
                assert!(crate::nd::tests::ndarray_eq_nan($left, $right), $($arg)+)
            }};
        }
        pub(crate) use assert_arr_eq_nan;

        pub(crate) fn ndarray_eq_nan<S1, S2, D1, D2, T>(
            arr1: &ndarray::ArrayBase<S1, D1>,
            arr2: &ndarray::ArrayBase<S2, D2>,
        ) -> bool
        where
            S1: ndarray::Data<Elem = T>,
            S2: ndarray::Data<Elem = T>,
            D1: ndarray::Dimension,
            D2: ndarray::Dimension,
            T: Dtyped,
        {
            if arr1.shape() != arr2.shape() {
                return false;
            }
            array_equal_impl(
                arr1.as_ptr().cast(),
                arr2.as_ptr().cast(),
                &T::dtype(),
                arr1.shape(),
                arr1.strides(),
                arr2.strides(),
                true,
            )
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(C, packed)]
        pub(crate) struct Point {
            x: i32,
            y: u32,
            z: i32,
        }
        unsafe impl Dtyped for Point {
            fn dtype() -> Dtype {
                Dtype::of_struct(HashMap::from([
                    ("x".into(), (0, Dtype::of_scalar(DtypeScalarKind::I32))),
                    ("y".into(), (4, Dtype::of_scalar(DtypeScalarKind::U32))),
                    ("z".into(), (8, Dtype::of_scalar(DtypeScalarKind::I32))),
                ]))
                .unwrap()
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(C, packed)]
        pub(crate) struct Person {
            height: i32,
            weight: i64,
        }
        unsafe impl Dtyped for Person {
            fn dtype() -> Dtype {
                Dtype::of_struct(HashMap::from([
                    ("height".into(), (0, Dtype::of_scalar(DtypeScalarKind::I32))),
                    ("weight".into(), (4, Dtype::of_scalar(DtypeScalarKind::I64))),
                ]))
                .unwrap()
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(C)]
        pub(crate) struct PersonAligned {
            height: i32,
            weight: i64,
        }
        unsafe impl Dtyped for PersonAligned {
            fn dtype() -> Dtype {
                Dtype::of_struct(HashMap::from([
                    ("height".into(), (0, Dtype::of_scalar(DtypeScalarKind::I32))),
                    ("weight".into(), (8, Dtype::of_scalar(DtypeScalarKind::I64))),
                ]))
                .unwrap()
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(C)]
        pub(crate) struct AudioSample([[f32; 2]; 16]);
        unsafe impl Dtyped for AudioSample {
            fn dtype() -> Dtype {
                Dtype::of_scalar(DtypeScalarKind::F32)
                    .with_shape(vec![16, 2])
                    .unwrap()
            }
        }
    }
}
