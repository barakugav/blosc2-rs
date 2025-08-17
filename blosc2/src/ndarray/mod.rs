mod dtype;
pub use dtype::*;

mod ast;

mod params;
pub use params::*;

use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::path::Path;
use std::ptr::NonNull;

use crate::error::ErrorCode;
use crate::util::{path2cstr, ArrayVec};
use crate::{Error, SChunkStorageParams};

pub const MAX_DIM: usize = blosc2_sys::B2ND_MAX_DIM as usize;
type DimVec<T> = ArrayVec<T, MAX_DIM>;

pub struct Ndarray {
    ptr: NonNull<blosc2_sys::b2nd_array_t>,
    dtype: Dtype,
}
impl Ndarray {
    fn new_impl(
        value: &InitArgs,
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        let mut cparams = params.cparams.clone();
        let mut dparams = params.dparams.clone();
        let chunkshape = params.chunksize_required()?;
        let blockshape = params.blockshape_required()?;

        let (shape, dtype, dtype_cstr) = match &value {
            InitArgs::Zeros
            | InitArgs::Nans
            | InitArgs::Uninit
            | InitArgs::RepeatedValue(_)
            | InitArgs::CopyFromValuesBuf(_) => {
                let shape = params.shape_required()?;
                let (dtype, dtype_cstr) = params.dtype_required()?;
                (shape.as_slice(), dtype, dtype_cstr.as_c_str())
            }
            InitArgs::CopyFromNdarray(ndarray) => {
                let dtype = &ndarray.dtype;
                let shape = ndarray.shape();
                (shape, dtype, ndarray.dtype_cstr())
            }
        };

        let itemsize = NonZeroUsize::new(dtype.itemsize).ok_or_else(|| {
            crate::trace!("Zero itemsize is not supported");
            Error::InvalidParam
        })?;
        cparams.typesize(itemsize);

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
            chunkshape.as_slice(),
            blockshape.as_slice(),
            dtype_cstr,
            blosc2_sys::DTYPE_NUMPY_FORMAT as _,
        )?;

        let mut array = MaybeUninit::<*mut blosc2_sys::b2nd_array_t>::uninit();
        let res = match value {
            InitArgs::Zeros => unsafe { blosc2_sys::b2nd_zeros(ctx.as_ptr(), array.as_mut_ptr()) },
            InitArgs::Nans => unsafe { blosc2_sys::b2nd_nans(ctx.as_ptr(), array.as_mut_ptr()) },
            InitArgs::Uninit => unsafe {
                blosc2_sys::b2nd_uninit(ctx.as_ptr(), array.as_mut_ptr())
            },
            InitArgs::RepeatedValue(value) => {
                if value.len() != dtype.itemsize {
                    crate::trace!(
                        "Repeated value length {} does not match dtype itemsize {}",
                        value.len(),
                        dtype.itemsize
                    );
                    return Err(Error::InvalidParam);
                }
                unsafe {
                    blosc2_sys::b2nd_full(ctx.as_ptr(), array.as_mut_ptr(), value.as_ptr().cast())
                }
            }
            InitArgs::CopyFromValuesBuf(items) => {
                let expected_length =
                    dtype.itemsize * shape.iter().map(|s| *s as usize).product::<usize>();
                if items.len() != expected_length {
                    crate::trace!(
                        "Items buffer length {} does not match expected length {} ({} * {:?})",
                        items.len(),
                        expected_length,
                        dtype.itemsize,
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

    pub fn new(value: &NdarrayInitValue, params: &NdarrayParams) -> Result<Self, Error> {
        Self::new_at(value, SChunkStorageParams::in_memory(), params)
    }

    pub fn new_on_disk(
        value: &NdarrayInitValue,
        urlpath: &Path,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_at(value, SChunkStorageParams::on_disk(urlpath), params)
    }

    pub fn new_at(
        value: &NdarrayInitValue,
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_impl(
            &match value {
                NdarrayInitValue::Zeros => InitArgs::Zeros,
                NdarrayInitValue::Nans => InitArgs::Nans,
                NdarrayInitValue::Uninit => InitArgs::Uninit,
                NdarrayInitValue::RepeatedValue(v) => InitArgs::RepeatedValue(v),
            },
            storage,
            params,
        )
    }

    pub fn open(urlpath: &Path) -> Result<Self, Error> {
        Self::open_with_offset(urlpath, 0)
    }

    pub fn open_with_offset(urlpath: &Path, offset: usize) -> Result<Self, Error> {
        crate::global::global_init();

        let urlpath = path2cstr(urlpath);
        let mut array = MaybeUninit::<*mut blosc2_sys::b2nd_array_t>::uninit();
        unsafe {
            blosc2_sys::b2nd_open_offset(
                urlpath.as_ptr().cast_mut(),
                array.as_mut_ptr(),
                offset as _,
            )
            .into_result()?;
        };
        let array = unsafe { array.assume_init() };
        unsafe { Self::from_raw_ptr(array.cast()) }
    }

    pub fn from_items_buf(items: &[u8], params: &NdarrayParams) -> Result<Self, Error> {
        Self::from_items_buf_at(items, SChunkStorageParams::in_memory(), params)
    }

    pub fn from_items_buf_at(
        items: &[u8],
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_impl(&InitArgs::CopyFromValuesBuf(items), storage, params)
    }

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
        let shape = DimVec::from_slice_fn(ndarray.shape(), |s| *s as i64);
        // Safety: we know ndim <= MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };

        let mut params = params.clone();
        params.dtype(T::dtype_numpy_str())?;
        params.shape(shape.as_slice())?;

        let data = ndarray.as_standard_layout();
        let data = data
            .as_slice()
            .expect("arr.as_standard_layout() should be contiguous");
        let data_buf = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };

        Self::from_items_buf_at(data_buf, storage, &params)
    }

    pub unsafe fn from_raw_ptr(ptr: *mut ()) -> Result<Self, Error> {
        let ptr: NonNull<blosc2_sys::b2nd_array_t> =
            NonNull::new(ptr.cast()).ok_or(Error::Failure)?;

        let dtype_cstr = unsafe { std::ffi::CStr::from_ptr((&*ptr.as_ptr()).dtype) };
        let dtype = dtype_cstr.to_str().unwrap();
        let dtype = Dtype::try_from(dtype).map_err(|_| {
            crate::trace!("Invalid dtype: {}", dtype);
            Error::InvalidParam
        })?;

        Ok(Self { ptr, dtype })
    }
}

/// Represents a repeated value that can be compressed.
///
/// This enum is used as an argument to [`Encoder::compress_repeatval`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NdarrayInitValue<'a> {
    /// Repeated zeros.
    Zeros,
    /// Repeated NaN values (for types that support NaN, like `f32` and `f64`).
    Nans,
    /// Uninitialized values.
    Uninit,
    /// A specific value to repeat.
    ///
    /// The value must have the same size as the `typesize` used in the compression parameters.
    RepeatedValue(&'a [u8]),
}

enum InitArgs<'a> {
    Zeros,
    Nans,
    Uninit,
    RepeatedValue(&'a [u8]),
    CopyFromValuesBuf(&'a [u8]),
    CopyFromNdarray(&'a Ndarray),
}

impl Ndarray {
    fn arr(&self) -> &blosc2_sys::b2nd_array_t {
        unsafe { self.ptr.as_ref() }
    }

    pub fn ndim(&self) -> usize {
        self.arr().ndim as usize
    }

    pub fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.arr().shape.as_ptr(), self.ndim()) }
    }

    pub fn dtype_str(&self) -> &str {
        self.dtype_cstr().to_str().unwrap()
    }

    fn dtype_cstr(&self) -> &std::ffi::CStr {
        unsafe { std::ffi::CStr::from_ptr(self.arr().dtype) }
    }

    pub fn typesize(&self) -> usize {
        debug_assert_eq!(
            self.dtype.itemsize,
            unsafe { &*self.arr().sc }.typesize as usize
        );
        self.dtype.itemsize
    }

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

    #[cfg(feature = "ndarray")]
    pub unsafe fn to_ndarray_without_dtype_check<T, D>(&self) -> Result<ndarray::Array<T, D>, Error>
    where
        T: Copy + 'static,
        D: ndarray::Dimension,
    {
        self.check_ndarray_ndim::<D>()?;

        assert_eq!(std::mem::size_of::<T>(), self.typesize());
        let shape = self.shape().iter().map(|s| *s as usize).collect::<Vec<_>>();
        let buf_len = shape.iter().product::<usize>();
        let mut buf = Vec::<MaybeUninit<T>>::with_capacity(buf_len);
        unsafe { buf.set_len(buf_len) };

        self.to_items_into(std::slice::from_raw_parts_mut(
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

    pub fn to_items(&self) -> Result<Vec<u8>, Error> {
        let buf_len = self.typesize() * self.shape().iter().map(|s| *s as usize).product::<usize>();
        let mut buf = Vec::<MaybeUninit<u8>>::with_capacity(buf_len);
        unsafe {
            buf.set_len(buf_len);
        }
        self.to_items_into(&mut buf)?;
        let buf = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(buf) };
        Ok(buf)
    }

    pub fn to_items_into(&self, buf: &mut [MaybeUninit<u8>]) -> Result<(), Error> {
        unsafe {
            blosc2_sys::b2nd_to_cbuffer(self.as_ptr(), buf.as_mut_ptr().cast(), buf.len() as i64)
                .into_result()?;
        }
        Ok(())
    }

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

    #[cfg(feature = "ndarray")]
    pub unsafe fn slice_without_dtype_check<T, D>(
        &self,
        slice: &[std::ops::Range<usize>],
    ) -> Result<ndarray::Array<T, D>, Error>
    where
        T: Copy + 'static,
        D: ndarray::Dimension,
    {
        self.check_ndarray_ndim::<D>()?;

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

    pub fn slice_blosc(
        &self,
        slice: &[std::ops::Range<usize>],
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        self.check_slice_arg(slice)?;
        let shape = DimVec::from_slice_fn(slice, |r| r.len() as i64);
        let start = DimVec::from_slice_fn(slice, |r| r.start as i64);
        let end = DimVec::from_slice_fn(slice, |r| r.end as i64);
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let start = unsafe { start.unwrap_unchecked() };
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let end = unsafe { end.unwrap_unchecked() };

        let mut cparams = params.cparams.clone();
        let mut dparams = params.dparams.clone();
        let chunkshape = params.chunksize_required()?;
        let blockshape = params.blockshape_required()?;
        let itemsize = NonZeroUsize::new(self.dtype.itemsize).ok_or_else(|| {
            crate::trace!("Zero itemsize is not supported");
            Error::InvalidParam
        })?;
        cparams.typesize(itemsize);

        let storage = blosc2_sys::blosc2_storage {
            contiguous: false,
            urlpath: std::ptr::null_mut(),
            cparams: &mut cparams.0 as *mut blosc2_sys::blosc2_cparams,
            dparams: &mut dparams.0 as *mut blosc2_sys::blosc2_dparams,
            io: std::ptr::null_mut(),
        };

        let ctx = Ctx::new(
            &storage,
            shape.as_slice(), // ignore shape in params
            chunkshape.as_slice(),
            blockshape.as_slice(),
            self.dtype_cstr(), // ignore dtype in params
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

    pub fn slice_buf(&self, slice: &[std::ops::Range<usize>]) -> Result<Vec<u8>, Error> {
        let len = slice.iter().map(|r| r.len()).product::<usize>();
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(len);
        unsafe { dst.set_len(len) };
        self.slice_buf_into(slice, &mut dst)?;
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    pub fn slice_buf_into(
        &self,
        slice: &[std::ops::Range<usize>],
        buf: &mut [MaybeUninit<u8>],
    ) -> Result<(), Error> {
        self.check_slice_arg(slice)?;
        let shape = DimVec::from_slice_fn(slice, |r| r.len());
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };
        self.slice_buf_into_with_shape(slice, buf, shape.as_slice())
    }

    pub fn slice_buf_into_with_shape(
        &self,
        slice: &[std::ops::Range<usize>],
        buf: &mut [MaybeUninit<u8>],
        buf_shape: &[usize],
    ) -> Result<(), Error> {
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
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let start = unsafe { start.unwrap_unchecked() };
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let end = unsafe { end.unwrap_unchecked() };
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
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
        Ok(())
    }

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

    #[cfg(feature = "ndarray")]
    pub unsafe fn set_slice_without_dtype_check<S, T, D>(
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

    pub fn set_slice_buf(
        &mut self,
        slice: &[std::ops::Range<usize>],
        items: &[u8],
    ) -> Result<(), Error> {
        self.check_slice_arg(slice)?;
        let shape = DimVec::from_slice_fn(slice, |r| r.len());
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let shape = unsafe { shape.unwrap_unchecked() };
        self.set_slice_buf_with_shape(slice, items, shape.as_slice())
    }

    pub fn set_slice_buf_with_shape(
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
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let start = unsafe { start.unwrap_unchecked() };
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
        let end = unsafe { end.unwrap_unchecked() };
        // Safety: slice.len()==self.ndim(), and we know self.ndim()<=MAX_DIM
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
    fn check_ndarray_ndim<D>(&self) -> Result<(), Error>
    where
        D: ndarray::Dimension,
    {
        let ndim = self.ndim();
        if D::NDIM.is_some() && D::NDIM.unwrap() != ndim {
            crate::trace!(
                "Dimension mismatch: expected {}, got {}",
                D::NDIM.unwrap(),
                self.ndim()
            );
            return Err(Error::InvalidParam);
        }
        Ok(())
    }

    fn as_ptr(&self) -> *mut blosc2_sys::b2nd_array_t {
        self.ptr.as_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const () {
        self.as_ptr().cast()
    }

    pub fn into_raw_ptr(self) -> *mut () {
        let ptr = self.as_ptr().cast();
        std::mem::forget(self);
        ptr
    }
}

impl Ndarray {
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

    pub fn copy(&self, params: &NdarrayParams) -> Result<Self, Error> {
        self.copy_to(SChunkStorageParams::in_memory(), params)
    }

    pub fn copy_to(
        &self,
        storage: SChunkStorageParams,
        params: &NdarrayParams,
    ) -> Result<Self, Error> {
        Self::new_impl(&InitArgs::CopyFromNdarray(self), storage, params)
    }

    fn check_dtype<T>(&self) -> Result<(), Error>
    where
        T: Dtyped,
    {
        let t_dtype_str = T::dtype_numpy_str();
        let t_dtype = Dtype::try_from(t_dtype_str).map_err(|e| {
            crate::trace!("Failed to parse T's dtype str: {}", e);
            Error::InvalidParam
        })?;
        if self.dtype != t_dtype {
            crate::trace!(
                "Dtype mismatch: {} != {}",
                self.dtype.to_numpy_str(),
                t_dtype_str
            );
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

    use super::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_new() {
        let array1 = ndarray::array!([[1_i32, 2], [3, 4],]);
        let b2nd = Ndarray::from_ndarray(
            &array1,
            &NdarrayParams::default()
                .blockshape(&[1, 1, 1])
                .unwrap()
                .chunksize(&[1, 1, 1])
                .unwrap(),
        )
        .unwrap();

        let array2: ndarray::ArrayD<i32> = b2nd.to_ndarray().unwrap();
        assert_eq!(array1.view().into_dyn(), array2.view().into_dyn());

        let array3: ndarray::Array3<i32> = b2nd.to_ndarray().unwrap();
        assert_eq!(array1.view().into_dyn(), array3.view().into_dyn());
    }
}
