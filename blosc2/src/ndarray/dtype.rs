use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;
use std::num::ParseIntError;

use crate::ndarray::ast::{parse_ast, Node};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dtype {
    pub kind: DtypeKind,
    pub shape: Vec<usize>,
    pub itemsize: usize,
    pub alignment: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DtypeKind {
    Scalar {
        kind: DtypeScalarKind,
        endianness: Endianness,
    },
    Struct {
        fields: HashMap<String, DtypeSubfield>,
    },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DtypeSubfield {
    pub dtype: Dtype,
    pub offset: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DtypeScalarKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    ComplexF32,
    ComplexF64,
    Bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Endianness {
    Little,
    Big,
}

pub unsafe trait Dtyped: Copy + 'static {
    fn dtype_numpy_str() -> &'static str;
}
// trait Dtyped2: Dtyped {
//     fn dtype() -> Dtype {
//         Dtype::try_from(Self::dtype_numpy_str()).unwrap()
//     }
// }
// impl<T: Dtyped> Dtyped2 for T {}

cfg_if::cfg_if! { if #[cfg(target_endian = "little")] {
    macro_rules! endian_prefix {
        ($str:literal) => {
            concat!("<", $str)
        };
    }
} else {
    macro_rules! endian_prefix {
        ($str:literal) => {
            concat!(">", $str)
        };
    }
} }

macro_rules! impl_dtyped {
    ($ty:ty, $str:expr) => {
        unsafe impl Dtyped for $ty {
            fn dtype_numpy_str() -> &'static str {
                $str
            }
        }
    };
}

macro_rules! impl_dtyped_scalar {
    ($ty:ty, $str:literal) => {
        impl_dtyped!($ty, {
            if const { core::mem::size_of::<$ty>() == 1 } {
                $str
            } else {
                endian_prefix!($str)
            }
        });
    };
}
impl_dtyped_scalar!(i8, "i1");
impl_dtyped_scalar!(i16, "i2");
impl_dtyped_scalar!(i32, "i4");
impl_dtyped_scalar!(i64, "i8");
impl_dtyped_scalar!(u8, "u1");
impl_dtyped_scalar!(u16, "u2");
impl_dtyped_scalar!(u32, "u4");
impl_dtyped_scalar!(u64, "u8");
impl_dtyped_scalar!(f16, "f2");
impl_dtyped_scalar!(f32, "f4");
impl_dtyped_scalar!(f64, "f8");
impl_dtyped_scalar!(Complex<f32>, "c4");
impl_dtyped_scalar!(Complex<f64>, "c8");
impl_dtyped_scalar!(bool, "b1");

cfg_if::cfg_if! { if #[cfg(feature = "half")] {
    pub use half::f16;
} else {
        /// A 16-bit floating point type implementing the IEEE 754-2008 standard [`binary16`] a.k.a "half"
        /// format.
        ///
        /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
        /// Enable the `half` feature to get a fully functional `f16` type.
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(transparent)]
        #[allow(non_camel_case_types)]
        pub struct f16(u16);
        impl f16 {
            #[doc = concat!("Creates a new `f16` from its raw bit representation.")]
            pub const fn from_bits(bits: u16) -> Self {
                Self(bits)
            }
            #[doc = concat!("Get the raw bit representation of the `f16`.")]
            pub const fn to_bits(&self) -> u16 {
                self.0
            }
        }
} }

cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
    pub use num_complex::Complex;
} else {
    /// A complex number in Cartesian form.
    ///
    /// Doesn't provide any arithmetic operations, but expose the real and imaginary parts.
    /// Enable the `num-complex` feature to get a fully functional `Complex` type.
    ///
    /// `Complex<T>` is memory layout compatible with an array `[T; 2]`, which is compatible with
    /// libc, numpy, etc.
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(C)]
    pub struct Complex<T> {
        /// Real portion of the complex number
        pub re: T,
        /// Imaginary portion of the complex number
        pub im: T,
    }
} }

impl TryFrom<&str> for Dtype {
    type Error = DtypeParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let mut chars = s.chars();
        let first_char = chars
            .next()
            .ok_or_else(|| DtypeParseError::msg("Unexpected end of input"))?;
        match first_char {
            '[' | '(' | '{' => {
                let ast = parse_ast(s).map_err(|e| {
                    DtypeParseError::new(DtypeParseErrorKind::AstError {
                        msg: e.msg,
                        pos: s.len() - e.pos_from_end,
                    })
                })?;
                ast2dtype(ast)
            }
            _ => parse_scalar_dtype(s),
        }
    }
}
fn parse_scalar_dtype(s: &str) -> Result<Dtype, DtypeParseError> {
    let mut chars = s.chars();
    let first_char = chars
        .next()
        .ok_or(DtypeParseError::msg("Unexpected end of input"))?;
    let native_endianness = if cfg!(target_endian = "little") {
        Endianness::Little
    } else {
        Endianness::Big
    };
    let (endianness, kind_char) = match first_char {
        '<' => (Endianness::Little, None),
        '>' => (Endianness::Big, None),
        '|' | '=' => (native_endianness, None),
        kind_char => (native_endianness, Some(kind_char)),
    };
    let kind_char = match kind_char {
        Some(kind_char) => kind_char,
        None => chars.next().ok_or(DtypeParseError::msg(
            "Unexpected end of input after endianness",
        ))?,
    };
    let size = if !chars.as_str().is_empty() {
        chars
            .as_str()
            .parse::<usize>()
            .map_err(|e| DtypeParseError::new(DtypeParseErrorKind::ParseIntError(e)))
            .context("dtype size")?
    } else {
        1
    };
    let (kind, alignment) = match (kind_char, size) {
        ('i', 1) => (DtypeScalarKind::I8, 1),
        ('i', 2) => (DtypeScalarKind::I16, 2),
        ('i', 4) => (DtypeScalarKind::I32, 4),
        ('i', 8) => (DtypeScalarKind::I64, 8),
        ('i', _) => return Err(DtypeParseError::new(DtypeParseErrorKind::UnsupportedScalar)),
        ('u', 1) => (DtypeScalarKind::U8, 1),
        ('u', 2) => (DtypeScalarKind::U16, 2),
        ('u', 4) => (DtypeScalarKind::U32, 4),
        ('u', 8) => (DtypeScalarKind::U64, 8),
        ('u', _) => return Err(DtypeParseError::new(DtypeParseErrorKind::UnsupportedScalar)),
        ('f', 2) => (DtypeScalarKind::F16, 2),
        ('f', 4) => (DtypeScalarKind::F32, 4),
        ('f', 8) => (DtypeScalarKind::F64, 8),
        ('f', _) => return Err(DtypeParseError::new(DtypeParseErrorKind::UnsupportedScalar)),
        ('c', 8) => (DtypeScalarKind::ComplexF32, 4),
        ('c', 16) => (DtypeScalarKind::ComplexF64, 8),
        ('c', _) => return Err(DtypeParseError::new(DtypeParseErrorKind::UnsupportedScalar)),
        ('b', 1) | ('?', 1) => (DtypeScalarKind::Bool, 1),
        ('b', _) | ('?', _) => {
            return Err(DtypeParseError::new(DtypeParseErrorKind::UnsupportedScalar))
        }
        (_, _) => return Err(DtypeParseError::new(DtypeParseErrorKind::UnsupportedScalar)),
    };
    Ok(Dtype {
        kind: DtypeKind::Scalar { kind, endianness },
        shape: Vec::new(),
        itemsize: size,
        alignment,
    })
}

fn ast2dtype(ast: Node) -> Result<Dtype, DtypeParseError> {
    fn ast2shape(shape_ast: Node) -> Result<Vec<usize>, DtypeParseError> {
        shape_ast
            .expect_tuple()?
            .into_iter()
            .map(|n| {
                n.expect_literal()?
                    .parse::<usize>()
                    .map_err(|e| DtypeParseError::new(DtypeParseErrorKind::ParseIntError(e)))
            })
            .collect()
    }

    match ast {
        Node::Str(field_dtype_str) => parse_scalar_dtype(&field_dtype_str),
        Node::Tuple(nodes) => match nodes.len() {
            1 => {
                let [dtype_ast] = unsafe { nodes.try_into().unwrap_unchecked() };
                ast2dtype(dtype_ast)
            }
            2 => {
                let [dtype_ast, shape] = unsafe { nodes.try_into().unwrap_unchecked() };
                let mut dtype = ast2dtype(dtype_ast).context("dtype")?;
                let mut shape = ast2shape(shape).context("shape")?;

                dtype.itemsize *= shape.iter().product::<usize>();

                shape.extend(dtype.shape);
                dtype.shape = shape;

                Ok(dtype)
            }
            _ => Err(DtypeParseError::msg(
                "Expected 1- or 2-tuple for dtype definition",
            )),
        },
        Node::List(fields_ast) => {
            let mut fields = HashMap::new();
            let mut struct_size = 0;
            for field_ast in fields_ast {
                let field_ast = field_ast.expect_tuple().context("field")?;
                let (name, field_type_ast, shape) = match field_ast.len() {
                    2 => {
                        // Safety: we know the length is 2
                        let [name, field_type_ast] =
                            unsafe { field_ast.try_into().unwrap_unchecked() };
                        (name, field_type_ast, Vec::new())
                    }
                    3 => {
                        // Safety: we know the length is 3
                        let [name, field_type_ast, shape] =
                            unsafe { field_ast.try_into().unwrap_unchecked() };
                        let shape = ast2shape(shape).context("field shape")?;
                        (name, field_type_ast, shape)
                    }
                    _ => {
                        return Err(DtypeParseError::msg(
                            "Expected 2- or 3-tuple for field definition",
                        ))
                    }
                };
                let name = name.expect_str().context("field name")?;
                let mut field_type = ast2dtype(field_type_ast).context("field type")?;
                if !shape.is_empty() {
                    field_type.itemsize *= shape.iter().product::<usize>();

                    let mut combined_shape = shape;
                    combined_shape.extend(field_type.shape);
                    field_type.shape = combined_shape;
                }

                let field_itemsize = field_type.itemsize;
                fields.insert(
                    name,
                    DtypeSubfield {
                        dtype: field_type,
                        offset: struct_size,
                    },
                );
                struct_size += field_itemsize;
            }

            Ok(Dtype {
                kind: DtypeKind::Struct { fields },
                shape: Vec::new(),
                itemsize: struct_size,
                alignment: 1,
            })
        }
        Node::Dict(dtype_dict_ast) => {
            let mut dtype_dict = HashMap::with_capacity(dtype_dict_ast.len());
            for (key, value) in dtype_dict_ast {
                let key = key.expect_str().context("dict key")?;
                if dtype_dict.contains_key(&key) {
                    return Err(DtypeParseError::new(DtypeParseErrorKind::DuplicateKey))
                        .context(key);
                }
                dtype_dict.insert(key, value);
            }

            let names = dtype_dict
                .remove("names")
                .ok_or(DtypeParseError::new(DtypeParseErrorKind::MissingKey(
                    "names",
                )))?
                .expect_list()
                .context("names")?
                .into_iter()
                .map(|n| n.expect_str().context("name"))
                .collect::<Result<Vec<_>, _>>()?;

            let formats = dtype_dict
                .remove("formats")
                .ok_or(DtypeParseError::new(DtypeParseErrorKind::MissingKey(
                    "formats",
                )))?
                .expect_list()
                .context("formats")?;
            if formats.len() != names.len() {
                return Err(DtypeParseError::msg(
                    "Number of names and formats must match",
                ));
            }
            let formats = formats
                .into_iter()
                .zip(names.iter())
                .map(|(format, name)| {
                    ast2dtype(format).context_with(|| format!("format for field '{}'", name))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let aligned = dtype_dict
                .remove("aligned")
                .map(|aligned| match aligned.expect_literal()?.as_str() {
                    "True" => Ok(true),
                    "False" => Ok(false),
                    _ => Err(DtypeParseError::msg("Invalid bool value")),
                })
                .transpose()
                .context("aligned")?
                .unwrap_or(false);

            let offsets = if let Some(offsets) = dtype_dict.remove("offsets") {
                let offsets = offsets.expect_list().context("offsets")?;
                if offsets.len() != names.len() {
                    return Err(DtypeParseError::msg(
                        "Number of offsets must match number of names",
                    ));
                }
                offsets
                    .into_iter()
                    .map(|offset| {
                        offset.expect_literal()?.parse::<usize>().map_err(|e| {
                            DtypeParseError::new(DtypeParseErrorKind::ParseIntError(e))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .context("offsets")?
            } else {
                formats
                    .iter()
                    .scan(0, |acc, f| {
                        if aligned {
                            *acc = ceil_to_multiple(*acc, f.alignment);
                        }
                        let offset = *acc;
                        *acc += f.itemsize;
                        Some(offset)
                    })
                    .collect()
            };

            let mut itemsize = dtype_dict
                .remove("itemsize")
                .map(|itemsize| {
                    let itemsize = itemsize.expect_literal()?;
                    let itemsize = itemsize
                        .parse::<usize>()
                        .map_err(|e| DtypeParseError::new(DtypeParseErrorKind::ParseIntError(e)))?;
                    Ok(itemsize)
                })
                .transpose()
                .context("itemsize")?
                .unwrap_or_else(|| {
                    offsets
                        .iter()
                        .zip(formats.iter())
                        .map(|(offset, format)| *offset + format.itemsize)
                        .max()
                        .unwrap_or(0)
                });

            if !dtype_dict.is_empty() {
                return Err(DtypeParseError::msg(format!(
                    "Unexpected keys in dtype dict: {:?}",
                    dtype_dict.keys()
                )));
            }

            let mut fields = HashMap::new();
            for ((name, format), offset) in names.into_iter().zip(formats).zip(offsets) {
                if fields.contains_key(&name) {
                    return Err(DtypeParseError::new(DtypeParseErrorKind::DuplicateKey))
                        .context(name);
                }
                fields.insert(
                    name,
                    DtypeSubfield {
                        dtype: format,
                        offset,
                    },
                );
            }

            let alignment = if aligned {
                let alignment = fields
                    .values()
                    .map(|f| f.dtype.alignment)
                    .max()
                    .unwrap_or(1);
                itemsize = ceil_to_multiple(itemsize, alignment);
                alignment
            } else {
                1
            };

            Ok(Dtype {
                kind: DtypeKind::Struct { fields },
                shape: Vec::new(),
                itemsize,
                alignment,
            })
        }
        Node::Literal(_) => Err(DtypeParseError::msg(
            "Expected a list, tuple, or dict for ast2dtype",
        )),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DtypeParseError {
    kind: DtypeParseErrorKind,
    backtrace: Vec<Cow<'static, str>>,
}
#[derive(Debug, PartialEq, Eq)]
enum DtypeParseErrorKind {
    AstError { msg: &'static str, pos: usize },
    ParseIntError(ParseIntError),
    UnsupportedScalar,
    ExpectedLiteral,
    ExpectedString,
    ExpectedTuple,
    ExpectedList,
    DuplicateKey,
    MissingKey(&'static str),
    Other(Cow<'static, str>),
}
impl DtypeParseError {
    fn new(kind: DtypeParseErrorKind) -> Self {
        DtypeParseError {
            kind,
            backtrace: Vec::new(),
        }
    }
    fn msg(msg: impl Into<Cow<'static, str>>) -> Self {
        DtypeParseError::new(DtypeParseErrorKind::Other(msg.into()))
    }
}
impl std::fmt::Display for DtypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.backtrace.is_empty() {
            write!(f, "{}, ", self.backtrace.join(" -> "))?;
        };
        match &self.kind {
            DtypeParseErrorKind::AstError { msg, pos } => {
                write!(f, "AST parse error: {msg} at position {pos}")
            }
            DtypeParseErrorKind::ParseIntError(err) => write!(f, "Parse int error: {err}"),
            DtypeParseErrorKind::UnsupportedScalar => write!(f, "Unsupported scalar type"),
            DtypeParseErrorKind::ExpectedLiteral => write!(f, "Expected a literal"),
            DtypeParseErrorKind::ExpectedString => write!(f, "Expected a string"),
            DtypeParseErrorKind::ExpectedTuple => write!(f, "Expected a tuple"),
            DtypeParseErrorKind::ExpectedList => write!(f, "Expected a list"),
            DtypeParseErrorKind::DuplicateKey => write!(f, "Duplicate key found"),
            DtypeParseErrorKind::MissingKey(key) => {
                write!(f, "Missing required key: {key}")
            }
            DtypeParseErrorKind::Other(cow) => f.write_str(cow),
        }
    }
}

trait DtypeParseResultExt {
    fn context(self, msg: impl Into<Cow<'static, str>>) -> Self;
    fn context_with(self, msg: impl FnOnce() -> String) -> Self;
}
impl<T> DtypeParseResultExt for Result<T, DtypeParseError> {
    fn context(self, msg: impl Into<Cow<'static, str>>) -> Self {
        self.map_err(|mut e| {
            e.backtrace.insert(0, msg.into());
            e
        })
    }
    fn context_with(self, msg: impl FnOnce() -> String) -> Self {
        self.map_err(|mut e| {
            e.backtrace.insert(0, msg().into());
            e
        })
    }
}
impl Node {
    fn expect_literal(self) -> Result<String, DtypeParseError> {
        if let Node::Literal(literal) = self {
            Ok(literal)
        } else {
            Err(DtypeParseError::new(DtypeParseErrorKind::ExpectedLiteral))
        }
    }

    fn expect_str(self) -> Result<String, DtypeParseError> {
        if let Node::Str(string) = self {
            Ok(string)
        } else {
            Err(DtypeParseError::new(DtypeParseErrorKind::ExpectedString))
        }
    }

    fn expect_tuple(self) -> Result<Vec<Node>, DtypeParseError> {
        if let Node::Tuple(values) = self {
            Ok(values)
        } else {
            Err(DtypeParseError::new(DtypeParseErrorKind::ExpectedTuple))
        }
    }

    fn expect_list(self) -> Result<Vec<Node>, DtypeParseError> {
        if let Node::List(nodes) = self {
            Ok(nodes)
        } else {
            Err(DtypeParseError::new(DtypeParseErrorKind::ExpectedList))
        }
    }
}

fn ceil_to_multiple(x: usize, m: usize) -> usize {
    assert!(m > 0);
    x.div_ceil(m) * m
}

fn scalar_dtype(kind: DtypeScalarKind) -> Dtype {
    let native_endianness = if cfg!(target_endian = "little") {
        Endianness::Little
    } else {
        Endianness::Big
    };
    let (itemsize, alignment) = match kind {
        DtypeScalarKind::I8 => (1, 1),
        DtypeScalarKind::I16 => (2, 2),
        DtypeScalarKind::I32 => (4, 4),
        DtypeScalarKind::I64 => (8, 8),
        DtypeScalarKind::U8 => (1, 1),
        DtypeScalarKind::U16 => (2, 2),
        DtypeScalarKind::U32 => (4, 4),
        DtypeScalarKind::U64 => (8, 8),
        DtypeScalarKind::F16 => (2, 2),
        DtypeScalarKind::F32 => (4, 4),
        DtypeScalarKind::F64 => (8, 8),
        DtypeScalarKind::ComplexF32 => (8, 4),
        DtypeScalarKind::ComplexF64 => (16, 8),
        DtypeScalarKind::Bool => (1, 1),
    };
    Dtype {
        kind: DtypeKind::Scalar {
            kind,
            endianness: native_endianness,
        },
        shape: Vec::new(),
        itemsize,
        alignment,
    }
}

impl Dtype {
    pub(crate) fn to_numpy_str(&self) -> String {
        struct FormatAsNumpy<'a>(&'a Dtype);

        impl std::fmt::Display for FormatAsNumpy<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.write_as_numpy_str(f, false)
            }
        }

        FormatAsNumpy(self).to_string()
    }

    fn write_as_numpy_str(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        mut nested: bool,
    ) -> std::fmt::Result {
        let mut itemsize = self.itemsize;
        let with_shape = !self.shape.is_empty();
        let shape_size = self.shape.iter().product::<usize>();
        if with_shape {
            f.write_char('(')?;
            if shape_size != 0 {
                assert_eq!(itemsize % shape_size, 0);
                itemsize /= shape_size;
            } else {
                assert_eq!(itemsize, 0);
            }
            nested = true;
        }
        match &self.kind {
            DtypeKind::Scalar { kind, endianness } => {
                let plain_dtype = scalar_dtype(*kind);
                if plain_dtype.itemsize != itemsize && shape_size > 0 {
                    crate::trace!(
                        "Dtype itemsize mismatch: expected {}, got {} ({:?})",
                        plain_dtype.itemsize,
                        itemsize,
                        *kind
                    );
                    return Err(std::fmt::Error);
                }
                if plain_dtype.alignment != self.alignment {
                    crate::trace!(
                        "Dtype alignment mismatch: expected {}, got {} ({:?})",
                        plain_dtype.alignment,
                        self.alignment,
                        *kind
                    );
                    return Err(std::fmt::Error);
                }
                let kind = match kind {
                    DtypeScalarKind::I8 => "i1",
                    DtypeScalarKind::I16 => "i2",
                    DtypeScalarKind::I32 => "i4",
                    DtypeScalarKind::I64 => "i8",
                    DtypeScalarKind::U8 => "u1",
                    DtypeScalarKind::U16 => "u2",
                    DtypeScalarKind::U32 => "u4",
                    DtypeScalarKind::U64 => "u8",
                    DtypeScalarKind::F16 => "f2",
                    DtypeScalarKind::F32 => "f4",
                    DtypeScalarKind::F64 => "f8",
                    DtypeScalarKind::ComplexF32 => "c8",
                    DtypeScalarKind::ComplexF64 => "c16",
                    DtypeScalarKind::Bool => "b1",
                };
                let endianness_prefix = match (itemsize, endianness) {
                    (1, _) => "|",
                    (_, Endianness::Little) => "<",
                    (_, Endianness::Big) => ">",
                };
                if nested {
                    f.write_char('\'')?;
                }
                write!(f, "{}{}", endianness_prefix, kind)?;
                if nested {
                    f.write_char('\'')?;
                }
            }
            DtypeKind::Struct { fields } => {
                f.write_char('{')?;

                let mut fields = fields.iter().collect::<Vec<_>>();
                fields.sort_by_key(|(_, field)| field.offset);

                let aligned = self.alignment > 1;
                if aligned {
                    // make sure offsets are as expected with aligned=True
                    let mut expected_offset = 0;
                    for (_f_name, field) in &fields {
                        expected_offset = ceil_to_multiple(expected_offset, field.dtype.alignment);
                        if field.offset != expected_offset {
                            panic!();
                        }
                        expected_offset += field.dtype.itemsize;
                    }
                }

                f.write_str("'names':[")?;
                for (f_name, _field) in &fields {
                    write!(f, "'{}',", f_name)?;
                }
                f.write_char(']')?;
                f.write_char(',')?;

                f.write_str("'formats':[")?;
                for (_f_name, field) in &fields {
                    field.dtype.write_as_numpy_str(f, true)?;
                    f.write_char(',')?;
                }
                f.write_char(']')?;
                f.write_char(',')?;

                f.write_str("'offsets':[")?;
                for (_f_name, field) in &fields {
                    write!(f, "{},", field.offset)?;
                }
                f.write_char(']')?;
                f.write_char(',')?;

                write!(f, "'itemsize': {itemsize},")?;
                write!(f, "'aligned': {},", if aligned { "True" } else { "False" })?;

                f.write_char('}')?;
            }
        }
        if with_shape {
            f.write_char(',')?;
            f.write_char('(')?;
            for dim in &self.shape {
                write!(f, "{},", dim)?;
            }
            f.write_char(')')?;
            f.write_char(')')?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use crate::{Dtype, DtypeKind, DtypeScalarKind, DtypeSubfield};

    #[test]
    fn dtype_from_str() {
        use super::scalar_dtype as sdtype;
        use DtypeScalarKind as SKind;

        let dtypes = [
            ("|i1", sdtype(SKind::I8)),
            ("<i2", sdtype(SKind::I16)),
            ("<i4", sdtype(SKind::I32)),
            ("<i8", sdtype(SKind::I64)),
            ("|u1", sdtype(SKind::U8)),
            ("<u2", sdtype(SKind::U16)),
            ("<u4", sdtype(SKind::U32)),
            ("<u8", sdtype(SKind::U64)),
            ("<f2", sdtype(SKind::F16)),
            ("<f4", sdtype(SKind::F32)),
            ("<f8", sdtype(SKind::F64)),
            ("<c8", sdtype(SKind::ComplexF32)),
            ("<c16", sdtype(SKind::ComplexF64)),
            ("|b1", sdtype(SKind::Bool)),
            (
                "('<i4', (2, 3, 4))",
                Dtype {
                    shape: vec![2, 3, 4],
                    itemsize: 96,
                    ..sdtype(SKind::I32)
                },
            ),
            (
                "('<i4', (2,))",
                Dtype {
                    shape: vec![2],
                    itemsize: 8,
                    ..sdtype(SKind::I32)
                },
            ),
            (
                "('<i4', (0,))",
                Dtype {
                    shape: vec![0],
                    itemsize: 0,
                    ..sdtype(SKind::I32)
                },
            ),
            ("('<i4')", sdtype(SKind::I32)),
            (
                "('?', (2, 3, 4))",
                Dtype {
                    shape: vec![2, 3, 4],
                    itemsize: 24,
                    ..sdtype(SKind::Bool)
                },
            ),
            (
                "('<i4', (2, 3, 4))",
                Dtype {
                    shape: vec![2, 3, 4],
                    itemsize: 96,
                    ..sdtype(SKind::I32)
                },
            ),
            (
                "[('a', '<i4'), ('b', '<f8')]",
                Dtype {
                    shape: vec![],
                    itemsize: 12,
                    kind: DtypeKind::Struct {
                        fields: HashMap::from([
                            (
                                "a".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::I32),
                                    offset: 0,
                                },
                            ),
                            (
                                "b".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::F64),
                                    offset: 4,
                                },
                            ),
                        ]),
                    },
                    alignment: 1,
                },
            ),
            (
                "{'names': ['a', 'b'], 'formats': ['<i4', '<f8'], 'offsets': [0, 8], 'itemsize': 16, 'aligned': True}",
                Dtype {
                    kind: DtypeKind::Struct {
                        fields: HashMap::from([
                            (
                                "a".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::I32),
                                    offset: 0,
                                },
                            ),
                            (
                                "b".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::F64),
                                    offset: 8,
                                },
                            ),
                        ]),
                    },
                    shape: vec![],
                    itemsize: 16,
                    alignment: 8,
                },
            ),
            (
                "{'names': ['a', 'b'], 'formats': ['<i4', '<f8'], 'offsets': [0, 8], 'itemsize': 16, 'aligned': True, }",
                Dtype {
                    kind: DtypeKind::Struct {
                        fields: HashMap::from([
                            (
                                "a".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::I32),
                                    offset: 0,
                                },
                            ),
                            (
                                "b".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::F64),
                                    offset: 8,
                                },
                            ),
                        ]),
                    },
                    shape: vec![],
                    itemsize: 16,
                    alignment: 8,
                },
            ),
            (
                "[('a', '<i4'), ('b', '<f8', (2, 3, 4))]",
                Dtype {
                    kind: DtypeKind::Struct {
                        fields: HashMap::from([
                            (
                                "a".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::I32),
                                    offset: 0,
                                },
                            ),
                            (
                                "b".to_string(),
                                DtypeSubfield {
                                    dtype: Dtype {
                                        shape: vec![2, 3, 4],
                                        itemsize: 192,
                                        ..sdtype(SKind::F64)
                                    },
                                    offset: 4,
                                },
                            ),
                        ]),
                    },
                    shape: vec![],
                    itemsize: 196,
                    alignment: 1,
                },
            ),
            (
                "{'names': ['a', 'b'], 'formats': ['<i4', ('<f8', (2, 3, 4))], 'offsets': [0, 8], 'aligned': True}",
                Dtype {
                    kind: DtypeKind::Struct {
                        fields: HashMap::from([
                            (
                                "a".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::I32),
                                    offset: 0,
                                },
                            ),
                            (
                                "b".to_string(),
                                DtypeSubfield {
                                    dtype: Dtype {
                                        shape: vec![2, 3, 4],
                                        itemsize: 192,
                                        ..sdtype(SKind::F64)
                                    },
                                    offset: 8,
                                },
                            ),
                        ]),
                    },
                    shape: vec![],
                    itemsize: 200,
                    alignment: 8,
                },
            ),
            (
                "{'names': ['a', 'b'], 'formats': ['<i4', ('<f8', (2, 3, 4))], 'offsets': [0, 8], 'itemsize': 200, 'aligned': True}",
                Dtype {
                    kind: DtypeKind::Struct {
                        fields: HashMap::from([
                            (
                                "a".to_string(),
                                DtypeSubfield {
                                    dtype: sdtype(SKind::I32),
                                    offset: 0,
                                },
                            ),
                            (
                                "b".to_string(),
                                DtypeSubfield {
                                    dtype: Dtype {
                                        shape: vec![2, 3, 4],
                                        itemsize: 192,
                                        ..sdtype(SKind::F64)
                                    },
                                    offset: 8,
                                },
                            ),
                        ]),
                    },
                    shape: vec![],
                    itemsize: 200,
                    alignment: 8,
                },
            ),
        ];

        for (dtype_str, expected_dtype) in dtypes {
            let dtype = Dtype::try_from(dtype_str);
            assert_eq!(dtype, Ok(expected_dtype.clone()));

            let dtype_str2 = dtype.unwrap().to_numpy_str();
            let dtype2 = Dtype::try_from(dtype_str2.as_str());
            assert_eq!(dtype2, Ok(expected_dtype));
        }
    }
}
