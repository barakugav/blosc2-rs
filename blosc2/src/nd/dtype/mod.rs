mod ast;
mod numpy_str;

use std::collections::HashMap;

use crate::nd::dtype::numpy_str::{parse_numpy_dtype_str, DtypeParseError};
use crate::util::{f16, Complex};

/// Description of a type layout and inner fields.
///
/// blosc's [`Ndarray`](crate::nd::Ndarray) maintain the dtype of each array dynamically using a this type.
/// A `Dtype` represent the layout (size and alignment) and inner fields of every element in an ndarray.
/// Such dtype can be either a scalar (one of [`DtypeScalarKind`]) or a struct containing inner fields, each with its
/// own name, offset and dtype.
/// In addition, a `Dtype` has a shape attribute, representing multiples elements of the same "base" type, similar to
/// numpy.
///
/// # Examples
/// ```rust
/// use blosc2::nd::{Dtype, Dtyped, DtypeScalarKind};
/// use std::collections::HashMap;
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// #[repr(C, packed)]
/// pub(crate) struct Point {
///     x: i32,
///     y: u32,
///     z: i32,
/// }
/// unsafe impl Dtyped for Point {
///     fn dtype() -> Dtype {
///         Dtype::of_struct(HashMap::from([
///             ("x".into(), (0, Dtype::of_scalar(DtypeScalarKind::I32))),
///             ("y".into(), (4, Dtype::of_scalar(DtypeScalarKind::U32))),
///             ("z".into(), (8, Dtype::of_scalar(DtypeScalarKind::I32))),
///         ]))
///         .unwrap()
///     }
/// }
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// #[repr(C, packed)]
/// pub(crate) struct Person {
///     height: i32,
///     weight: i64,
/// }
/// unsafe impl Dtyped for Person {
///     fn dtype() -> Dtype {
///         Dtype::of_struct(HashMap::from([
///             ("height".into(), (0, Dtype::of_scalar(DtypeScalarKind::I32))),
///             ("weight".into(), (4, Dtype::of_scalar(DtypeScalarKind::I64))),
///         ]))
///         .unwrap()
///     }
/// }
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// #[repr(C)]
/// pub(crate) struct PersonAligned {
///     height: i32,
///     weight: i64,
/// }
/// unsafe impl Dtyped for PersonAligned {
///     fn dtype() -> Dtype {
///         Dtype::of_struct(HashMap::from([
///             ("height".into(), (0, Dtype::of_scalar(DtypeScalarKind::I32))),
///             ("weight".into(), (8, Dtype::of_scalar(DtypeScalarKind::I64))),
///         ]))
///         .unwrap()
///     }
/// }
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// #[repr(C)]
/// pub(crate) struct AudioSample([[f32; 2]; 16]);
/// unsafe impl Dtyped for AudioSample {
///     fn dtype() -> Dtype {
///         Dtype::of_scalar(DtypeScalarKind::F32)
///             .with_shape(vec![16, 2])
///             .unwrap()
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dtype {
    kind: DtypeKind,
    shape: Vec<usize>,
    itemsize: usize,
    alignment: usize,
}
/// An inner kind of [`Dtype`], either a scalar or a struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtypeKind {
    /// A scalar dtype.
    Scalar {
        /// The kind of the scalar dtype.
        kind: DtypeScalarKind,
        /// The endianness of the scalar dtype.
        endianness: Endianness,
    },
    /// A struct dtype.
    Struct {
        /// The fields of the struct dtype, represented as a map of field names to their (offset, dtype) pairs.
        fields: HashMap<String, (usize, Dtype)>,
    },
}
/// The kind of a scalar dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeScalarKind {
    /// [`i8`] dtype.
    I8,
    /// [`i16`] dtype.
    I16,
    /// [`i32`] dtype.
    I32,
    /// [`i64`] dtype.
    I64,
    /// [`u8`] dtype.
    U8,
    /// [`u16`] dtype.
    U16,
    /// [`u32`] dtype.
    U32,
    /// [`u64`] dtype.
    U64,
    /// [`f16`] dtype.
    F16,
    /// [`f32`] dtype.
    F32,
    /// [`f64`] dtype.
    F64,
    /// [`Complex<f32>`] dtype.
    ComplexF32,
    /// [`Complex<f64>`] dtype.
    ComplexF64,
    /// [`bool`] dtype.
    Bool,
}
/// The endianness of a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// Little-endian.
    Little,
    /// Big-endian.
    Big,
}
impl Dtype {
    /// Creates a new scalar dtype.
    ///
    /// The created dtype will use the native endianness.
    pub fn of_scalar(kind: DtypeScalarKind) -> Self {
        Self {
            kind: DtypeKind::Scalar {
                kind,
                endianness: Endianness::native(),
            },
            shape: Vec::new(),
            itemsize: kind.itemsize(),
            alignment: kind.alignment(),
        }
    }

    /// Creates a new struct dtype from a set of fields definitions.
    ///
    /// # Arguments
    ///
    /// * `fields` - A map of field names to tuple `(offset, dtype)`.
    ///
    /// The fields should be either in packed or aligned offsets, custom offsets are not supported.
    /// There are some cases in which it is ambiguous whether the offsets are packed or aligned, and it may affect the
    /// computed total itemsize of the struct. In these cases, consider using the explicit [`Self::new`].
    pub fn of_struct(fields: HashMap<String, (usize, Dtype)>) -> Result<Self, DtypeError> {
        fn determine_itemsize_and_alignment(
            fields: &HashMap<String, (usize, Dtype)>,
        ) -> Result<(usize, usize), DtypeError> {
            let mut fields_vec = fields
                .iter()
                .map(|(_name, (offset, dtype))| (*offset, dtype.itemsize, dtype.alignment))
                .collect::<Vec<_>>();
            fields_vec.sort_unstable_by_key(|(offset, _itemsize, _alignment)| *offset);

            let mut expected_offset = 0;
            let is_packed = fields_vec.iter().all({
                |(offset, itemsize, _alignment)| {
                    let packed = *offset == expected_offset;
                    expected_offset += itemsize;
                    packed
                }
            });
            if is_packed {
                let itemsize = expected_offset;
                return Ok((itemsize, 1));
            }

            let mut expected_offset = 0;
            let is_aligned = fields_vec.iter().all({
                |(offset, itemsize, alignment)| {
                    expected_offset = ceil_to_multiple(expected_offset, *alignment);
                    let aligned = *offset == expected_offset;
                    expected_offset += itemsize;
                    aligned
                }
            });
            if is_aligned {
                let max_alignment = fields_vec
                    .iter()
                    .map(|(_offset, _itemsize, alignment)| *alignment)
                    .max()
                    .unwrap_or(1);
                let itemsize = ceil_to_multiple(expected_offset, max_alignment);
                return Ok((itemsize, max_alignment));
            }

            Err(DtypeError::InvalidOffsets)
        }

        let (itemsize, alignment) = determine_itemsize_and_alignment(&fields)?;
        Ok(Self {
            kind: DtypeKind::Struct { fields },
            shape: Vec::new(),
            itemsize,
            alignment,
        })
    }

    /// Creates a new dtype by specifying all of the parameters explicitly.
    ///
    /// Thw shape, itemsize and alignment will be validated against the kind. See [`DtypeError`] for their constraints.
    pub fn new(
        kind: DtypeKind,
        shape: Vec<usize>,
        itemsize: usize,
        alignment: usize,
    ) -> Result<Self, DtypeError> {
        let shape_prod = shape.iter().product::<usize>();
        if shape_prod == 0 {
            return Err(DtypeError::InvalidShape);
        }
        if itemsize % shape_prod != 0 {
            return Err(DtypeError::InvalidItemsize);
        }
        let element_itemsize = itemsize / shape_prod;

        match &kind {
            DtypeKind::Scalar {
                kind,
                endianness: _,
            } => {
                if kind.alignment() != alignment {
                    return Err(DtypeError::InvalidAlignment);
                }
                if kind.itemsize() != element_itemsize {
                    return Err(DtypeError::InvalidItemsize);
                }
            }
            DtypeKind::Struct { fields } => {
                let mut fields_vec = fields
                    .iter()
                    .map(|(_name, (offset, dtype))| (*offset, dtype.itemsize, dtype.alignment))
                    .collect::<Vec<_>>();
                fields_vec.sort_unstable_by_key(|(offset, _itemsize, _alignment)| *offset);

                if alignment == 1 {
                    // packed struct

                    let mut expected_offset = 0;
                    let is_packed = fields_vec.iter().all({
                        |(offset, itemsize, _alignment)| {
                            let packed = *offset == expected_offset;
                            expected_offset += itemsize;
                            packed
                        }
                    });
                    if !is_packed {
                        return Err(DtypeError::InvalidOffsets);
                    }
                    let expected_itemsize = expected_offset;
                    if expected_itemsize != element_itemsize {
                        return Err(DtypeError::InvalidItemsize);
                    }
                } else {
                    // aligned struct

                    let max_alignment = fields
                        .values()
                        .map(|(_offset, dtype)| dtype.alignment)
                        .max()
                        .unwrap_or(1);
                    if alignment != max_alignment {
                        return Err(DtypeError::InvalidAlignment);
                    }

                    let mut expected_offset = 0;
                    let is_aligned = fields_vec.iter().all({
                        |(offset, itemsize, alignment)| {
                            expected_offset = ceil_to_multiple(expected_offset, *alignment);
                            let aligned = *offset == expected_offset;
                            expected_offset += itemsize;
                            aligned
                        }
                    });
                    if !is_aligned {
                        return Err(DtypeError::InvalidOffsets);
                    }

                    let expected_itemsize = ceil_to_multiple(expected_offset, max_alignment);
                    if expected_itemsize != element_itemsize {
                        return Err(DtypeError::InvalidItemsize);
                    }
                }
            }
        }

        Ok(Self {
            kind,
            shape,
            itemsize,
            alignment,
        })
    }

    /// Get the kind of the dtype.
    pub fn kind(&self) -> &DtypeKind {
        &self.kind
    }

    /// Get the shape of the dtype.
    ///
    /// Empty shape means a single element of the dtype.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the itemsize of the dtype.
    ///
    /// If this dtype has a shape, the itemsize is the product of the shape dimensions and the base itemsize.
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    /// Get the alignment of the dtype.
    ///
    /// For scalar dtypes, the alignment is the same as [`DtypeScalarKind::alignment()`].
    /// For struct dtypes, the alignment is either `1` for packed dtypes, or the maximum alignment of the inner fields
    /// for aligned structs.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get a dtype with this dtype specs, but with a different shape.
    pub fn with_shape(self, shape: Vec<usize>) -> Result<Self, DtypeError> {
        let current_shape_prod = self.shape.iter().product::<usize>();
        debug_assert!(current_shape_prod > 0);
        debug_assert_eq!(self.itemsize % current_shape_prod, 0);
        let base_itemsize = self.itemsize / current_shape_prod;

        let shape_prod = shape.iter().product::<usize>();
        if shape_prod == 0 {
            return Err(DtypeError::InvalidShape);
        }
        let itemsize = base_itemsize * shape_prod;
        Ok(Self {
            kind: self.kind,
            shape,
            itemsize,
            alignment: self.alignment,
        })
    }

    /// Create a new dtype from a numpy dtype string.
    ///
    /// See <https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing> for the full
    /// definitions.
    ///
    /// # Examples
    /// ```rust
    /// use blosc2::nd::{Dtype, Dtyped};
    ///
    /// // An packed point struct, with size 12 and alignment 1
    /// #[derive(Debug, Clone, Copy, PartialEq)]
    /// #[repr(C, packed)]
    /// struct Point {
    ///     x: i32,
    ///     y: u32,
    ///     z: i32,
    /// }
    /// unsafe impl Dtyped for Point {
    ///     fn dtype() -> Dtype {
    ///         Dtype::from_numpy_str("[('x', '<i4'), ('y', '<u4'), ('z', '<i4')]").unwrap()
    ///     }
    /// }
    ///
    /// // An packed person struct, with size 12 and alignment 1
    /// #[derive(Debug, Clone, Copy, PartialEq)]
    /// #[repr(C, packed)]
    /// struct Person {
    ///     height: i32,
    ///     weight: i64,
    /// }
    /// unsafe impl Dtyped for Person {
    ///     fn dtype() -> Dtype {
    ///         Dtype::from_numpy_str("[('height', '<i4'), ('weight', '<i8')]").unwrap()
    ///     }
    /// }
    ///
    /// // An aligned person struct, with size 16 and alignment 8
    /// #[derive(Debug, Clone, Copy, PartialEq)]
    /// #[repr(C)]
    /// struct PersonAligned {
    ///     height: i32,
    ///     weight: i64,
    /// }
    /// unsafe impl Dtyped for PersonAligned {
    ///     fn dtype() -> Dtype {
    ///         Dtype::from_numpy_str("{'names':['height','weight'], 'formats':['<i4', '<i8'], 'aligned':True}").unwrap()
    ///     }
    /// }
    ///
    /// // An audio sample with two channels and 16 samples, represented as an `f32` 2D array.
    /// #[derive(Debug, Clone, Copy, PartialEq)]
    /// #[repr(C)]
    /// struct AudioSample([[f32; 2]; 16]);
    /// unsafe impl Dtyped for AudioSample {
    ///     fn dtype() -> Dtype {
    ///         Dtype::from_numpy_str("('<f4', (16, 2))").unwrap()
    ///     }
    /// }
    /// ```
    pub fn from_numpy_str(s: &str) -> Result<Self, DtypeParseError> {
        parse_numpy_dtype_str(s)
    }
}

/// Error that can happen when creating a new [`Dtype`]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DtypeError {
    /// Invalid field offsets.
    ///
    /// Currently blosc2 support either packed fields, or aligned fields, while fully custom offsets are not supported.
    InvalidOffsets,
    /// Invalid itemsize.
    ///
    /// In numpy, the itemsize can be arbitrary as long as it is at least the max field offset + field itemsize,
    /// but in blosc2 only "vanilla" packed or aligned dtypes are supported.
    /// - If a dtype has shape, the itemsize must be a multiple of the shape product, and after dividing it by the
    ///   product the "base" itemsize should match the scalar or nested fields definitions of the dtype.
    ///   The rest of the rules assume the dtype has no shape, referring to the base itemsize.
    /// - If a dtype is a scalar, the itemsize must match the scalar definition.
    ///   See [`DtypeScalarKind::itemsize`] for details.
    /// - If a dtype is a struct with packed fields, the itemsize must be exactly the sum of the field itemsize.
    /// - If a dtype is a struct with aligned fields, the itemsize must be the field with the greatest offset plus its
    ///   itemsize, aligned to the dtype alignment (which is the maximum alignment of any field).
    InvalidItemsize,
    /// Invalid alignment.
    ///
    /// - If the dtype is a scalar, the alignment must match the scalar definition.
    ///   See [`DtypeScalarKind::alignment`] for details.
    /// - If the dtype is a struct with packed fields, the alignment must be 1.
    /// - If the dtype is a struct with aligned fields, the alignment must be the maximum alignment of any field.
    InvalidAlignment,
    /// Invalid shape.
    ///
    /// Shape with zero dimension is not allowed.
    InvalidShape,
}
impl std::fmt::Display for DtypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOffsets => write!(f, "Invalid field offsets"),
            Self::InvalidItemsize => write!(f, "Invalid itemsize"),
            Self::InvalidAlignment => write!(f, "Invalid alignment"),
            Self::InvalidShape => write!(f, "Invalid shape"),
        }
    }
}
impl DtypeScalarKind {
    /// Get the size of the scalar in bytes.
    pub fn itemsize(&self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::ComplexF32 => 8,
            Self::ComplexF64 => 16,
            Self::Bool => 1,
        }
    }
    /// Get the alignment of the scalar in bytes.
    pub fn alignment(&self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::ComplexF32 => 4,
            Self::ComplexF64 => 8,
            Self::Bool => 1,
        }
    }
}
impl Endianness {
    /// Get the native endianness.
    pub fn native() -> Self {
        if cfg!(target_endian = "little") {
            Endianness::Little
        } else {
            Endianness::Big
        }
    }
}

///
/// A trait for types that can be represented by a [`Dtype`].
///
/// blosc's [`Ndarray`](crate::nd::Ndarray) maintain the dtype of each array dynamically using a [`Dtype`].
/// For safe conversions between (typed erased) `Ndarray` and other typed arrays (for example [`ndarray::ArrayBase`])
/// the `Dtyped` trait is used to verify type compatibility.
///
/// # Safety
///
/// This trait is very unsafe, and the caller should implement it carefully, matching the type size,
/// alignment and inner fields of the type. Types implementing this should most likely be annotated with `#[repr(C)]`
/// or `#[repr(C, packed)]`, for aligned and packed fields respectively.
pub unsafe trait Dtyped: Copy + 'static {
    /// Get the dtype representing the type layout and inner fields.
    fn dtype() -> Dtype;
}
macro_rules! impl_dtyped_scalar {
    ($ty:ty, $kind:ident) => {
        unsafe impl Dtyped for $ty {
            fn dtype() -> Dtype {
                Dtype::of_scalar(DtypeScalarKind::$kind)
            }
        }
    };
}

impl_dtyped_scalar!(i8, I8);
impl_dtyped_scalar!(i16, I16);
impl_dtyped_scalar!(i32, I32);
impl_dtyped_scalar!(i64, I64);
impl_dtyped_scalar!(u8, U8);
impl_dtyped_scalar!(u16, U16);
impl_dtyped_scalar!(u32, U32);
impl_dtyped_scalar!(u64, U64);
impl_dtyped_scalar!(f16, F16);
impl_dtyped_scalar!(f32, F32);
impl_dtyped_scalar!(f64, F64);
impl_dtyped_scalar!(Complex<f32>, ComplexF32);
impl_dtyped_scalar!(Complex<f64>, ComplexF64);
impl_dtyped_scalar!(bool, Bool);

fn ceil_to_multiple(x: usize, m: usize) -> usize {
    assert!(m > 0);
    x.div_ceil(m) * m
}
