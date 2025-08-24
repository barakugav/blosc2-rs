use std::mem::ManuallyDrop;

use crate::nd::Ndarray;

use pyo3::types::{PyCapsule, PyType};
use pyo3::{prelude::*, PyTypeInfo};

#[repr(transparent)]
pub struct PyNdarray(PyAny);
unsafe impl PyTypeInfo for PyNdarray {
    const NAME: &'static str = "NDArray";
    const MODULE: Option<&'static str> = Some("blosc2");

    fn type_object_raw(py: Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        py.import("blosc2")
            .unwrap()
            .getattr("NDArray")
            .unwrap()
            .downcast::<PyType>()
            .unwrap()
            .as_type_ptr()
    }
}
pub trait PyNdarrayMethods {
    fn to_array_ref(&self) -> PyResult<PyNdarrayRef>;
}
impl<'py> PyNdarrayMethods for Bound<'py, PyNdarray> {
    fn to_array_ref(&self) -> PyResult<PyNdarrayRef> {
        check_python_blosc_version(self.py())?;

        let raw_ptr = self.as_any().getattr("as_ffi_ptr")?.call0()?;
        let raw_ptr = raw_ptr.downcast::<PyCapsule>()?;
        let raw_ptr =
            unsafe { pyo3::ffi::PyCapsule_GetPointer(raw_ptr.as_ptr(), c"b2nd_array_t*".as_ptr()) };
        let array = unsafe { Ndarray::from_raw_ptr(raw_ptr as _) }.unwrap();
        Ok(PyNdarrayRef {
            arr: ManuallyDrop::new(array),
            phantom: std::marker::PhantomData,
        })
    }
}

pub struct PyNdarrayRef<'a> {
    arr: ManuallyDrop<Ndarray>,
    phantom: std::marker::PhantomData<&'a ()>,
}
impl AsRef<Ndarray> for PyNdarrayRef<'_> {
    fn as_ref(&self) -> &Ndarray {
        &self.arr
    }
}
impl std::ops::Deref for PyNdarrayRef<'_> {
    type Target = Ndarray;

    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}

impl<'py> IntoPyObject<'py> for Ndarray {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        check_python_blosc_version(py)?;

        let array_from_ffi_ptr_fn = py.import("blosc2")?.getattr("array_from_ffi_ptr")?;

        let array = self.into_raw_ptr() as *mut blosc2_sys::b2nd_array_t;
        let array = unsafe {
            let array = pyo3::ffi::PyCapsule_New(array as _, c"b2nd_array_t*".as_ptr(), None);
            Bound::from_owned_ptr_or_err(py, array)?
                .downcast_into::<PyCapsule>()
                .unwrap()
        };
        array_from_ffi_ptr_fn.call1((array,))
    }
}

fn check_python_blosc_version(py: Python) -> PyResult<()> {
    let py_clib_version = python_blosc2_clib_version(py)?;
    let rs_clib_version = blosc2_sys::BLOSC2_VERSION_STRING.to_str().unwrap();
    if py_clib_version != rs_clib_version {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Incompatible versions of underlying C blosc2 library used by Rust ({rs_clib_version}) and Python ({py_clib_version}).
            Choose a different version of Rust blosc2_sys or a different version of Python blosc2."
        )));
    }
    if let Ok(version) = python_blosc2_version(py) {
        if version < (3, 7, 3) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "python blosc2>=3.7.3 is required to convert NDArray from Rust to Python, but found {version:?}"
            )));
        }
    }
    Ok(())
}

fn python_blosc2_version(py: Python) -> PyResult<(u32, u32, u32)> {
    let version_str = py
        .import("blosc2")?
        .getattr("__version__")?
        .extract::<String>()?;
    let version = version_str.split('.').collect::<Vec<_>>();
    if version.len() < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "invalid python blosc2 version: {version_str}"
        )));
    }
    let [major, minor, patch] = version
        .iter()
        .take(3)
        .map(|s| s.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .unwrap();
    Ok((major, minor, patch))
}

fn python_blosc2_clib_version(py: Python) -> PyResult<String> {
    py.import("blosc2")?
        .getattr("VERSION_STRING")?
        .extract::<String>()
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rand::prelude::*;

    use crate::nd::tests::{rand_data, rand_params, rand_storage};
    use crate::nd::{Ndarray, PyNdarray, PyNdarrayMethods};

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0x3a6f329597cc9f6d);
        for _ in 0..30 {
            let (shape, dtype, params) = rand_params(&mut rand);
            let data = rand_data(&dtype, &shape, &mut rand);
            let storage = rand_storage(&mut rand);
            let array =
                Ndarray::from_items_bytes_at(&data, dtype, &shape, storage.params(), &params)
                    .unwrap();

            Python::with_gil(|py| {
                let locals = PyDict::new(py);
                locals.set_item("shape", array.shape()).unwrapy(py);
                locals
                    .set_item("array", array.into_pyobject(py).unwrapy(py))
                    .unwrapy(py);
                py.run(
                    c"
assert array.shape == tuple(shape), f'Expected shape {shape}, got {array.shape}'
array2 = array.copy()
                ",
                    None,
                    Some(&locals),
                )
                .unwrapy(py);
                let array2 = locals.get_item("array2").unwrapy(py).unwrap();
                let array2 = array2.downcast_into::<PyNdarray>().unwrap();
                let array2 = array2.to_array_ref().unwrap();
                assert_eq!(
                    array2.as_ref().shape(),
                    shape
                        .iter()
                        .map(|s| *s as i64)
                        .collect::<Vec<_>>()
                        .as_slice()
                );
            });

            panic!()
            // assert_eq!(
            //     shape,
            //     array
            //         .shape()
            //         .iter()
            //         .map(|s| *s as usize)
            //         .collect::<Vec<_>>()
            // );
            // let array_data = array.to_items().unwrap();
            // assert_eq!(data, array_data);
        }
    }

    trait Unwrapy {
        type Output;
        fn unwrapy(self, py: Python) -> Self::Output;
    }
    impl<T> Unwrapy for PyResult<T> {
        type Output = T;
        #[track_caller]
        fn unwrapy(self, py: Python) -> T {
            self.inspect_err(|err| {
                // print the familiar python stack trace
                err.print_and_set_sys_last_vars(py);
            })
            .unwrap()
        }
    }

    #[ctor::ctor]
    fn init_python() {
        // Usually when writing tests for Python bindings, we need add the modules using `append_to_inittab`, but
        // because we are using the venv, `brain_log_bindings` is already installed and we can import it directly.
        //
        // Note that due to the above, when modifying the bindings, re-install the package to make the changes
        // effective in the tests.
        //
        // pyo3::append_to_inittab!(brain_log_bindings);

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Pyo3 doesn't detect venv on MacOS.
            // See comments in brain-link/.../python_util.rs
            if std::env::var("VIRTUAL_ENV").is_ok() {
                py.run(
                    cr#"
import os, sys
venv_path = os.environ['VIRTUAL_ENV']
if os.name == 'nt':
    packages_path = f"{venv_path}/Lib/site-packages"
else:
    packages_path = f"{venv_path}/lib/python3.12/site-packages"
# DO NOT CHANGE to `.append(..)`
sys.path.insert(0, packages_path)

# DO NOT REMOVE
import site
site.addsitedir(packages_path)
        "#,
                    None,
                    None,
                )
                .unwrap();
            }
        });
    }
}
