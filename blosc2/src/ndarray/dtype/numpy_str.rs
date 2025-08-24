use std::{borrow::Cow, collections::HashMap, fmt::Write, num::ParseIntError};

use crate::ndarray::dtype::ast::{parse_ast, Node};
use crate::ndarray::dtype::ceil_to_multiple;
use crate::{Dtype, DtypeError, DtypeKind, DtypeScalarKind, Endianness};

pub(crate) fn parse_numpy_dtype_str(s: &str) -> Result<Dtype, DtypeParseError> {
    let mut chars = s.chars();
    let first_char = chars
        .next()
        .ok_or_else(|| DtypeParseError::msg("Unexpected end of input"))?;
    match first_char {
        '[' | '(' | '{' => {
            let ast =
                parse_ast(s).map_err(|e| DtypeParseError::new(DtypeParseErrorKind::AstError(e)))?;
            ast2dtype(ast)
        }
        _ => parse_scalar_dtype(s),
    }
}
fn parse_scalar_dtype(s: &str) -> Result<Dtype, DtypeParseError> {
    let mut chars = s.chars();
    let first_char = chars
        .next()
        .ok_or(DtypeParseError::msg("Unexpected end of input"))?;
    let (endianness, kind_char) = match first_char {
        '<' => (Endianness::Little, None),
        '>' => (Endianness::Big, None),
        '|' | '=' => (Endianness::native(), None),
        kind_char => (Endianness::native(), Some(kind_char)),
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
                fields.insert(name, (struct_size, field_type));
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
                fields.insert(name, (offset, format));
            }

            let alignment = if aligned {
                let alignment = fields
                    .values()
                    .map(|(_offset, dtype)| dtype.alignment)
                    .max()
                    .unwrap_or(1);
                itemsize = ceil_to_multiple(itemsize, alignment);
                alignment
            } else {
                1
            };

            Dtype::new(
                DtypeKind::Struct { fields },
                Vec::new(),
                itemsize,
                alignment,
            )
            .map_err(|e| DtypeParseError::new(DtypeParseErrorKind::DtypeError(e)))
        }
        Node::Literal(_) => Err(DtypeParseError::msg(
            "Expected a list, tuple, or dict for ast2dtype",
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtypeParseError {
    kind: DtypeParseErrorKind,
    backtrace: Vec<Cow<'static, str>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum DtypeParseErrorKind {
    AstError(crate::ndarray::dtype::ast::ParseError), // { msg: &'static str, pos: usize },
    ParseIntError(ParseIntError),
    UnsupportedScalar,
    ExpectedLiteral,
    ExpectedString,
    ExpectedTuple,
    ExpectedList,
    DuplicateKey,
    MissingKey(&'static str),
    DtypeError(DtypeError),
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
            DtypeParseErrorKind::AstError(err) => std::fmt::Display::fmt(err, f),
            DtypeParseErrorKind::ParseIntError(err) => write!(f, "Parse int error: {err}"),
            DtypeParseErrorKind::UnsupportedScalar => f.write_str("Unsupported scalar type"),
            DtypeParseErrorKind::ExpectedLiteral => f.write_str("Expected a literal"),
            DtypeParseErrorKind::ExpectedString => f.write_str("Expected a string"),
            DtypeParseErrorKind::ExpectedTuple => f.write_str("Expected a tuple"),
            DtypeParseErrorKind::ExpectedList => f.write_str("Expected a list"),
            DtypeParseErrorKind::DuplicateKey => f.write_str("Duplicate key found"),
            DtypeParseErrorKind::MissingKey(key) => {
                write!(f, "Missing required key: {key}")
            }
            DtypeParseErrorKind::DtypeError(err) => std::fmt::Display::fmt(err, f),
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

    #[allow(unused)]
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
                let plain_dtype = Dtype::of_scalar(*kind);
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
                    (1, _) => '|',
                    (_, Endianness::Little) => '<',
                    (_, Endianness::Big) => '>',
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
                fields.sort_by_key(|(_, (offset, _field))| *offset);

                let aligned = self.alignment > 1;
                if aligned {
                    // make sure offsets are as expected with aligned=True
                    let mut expected_offset = 0;
                    for (f_name, (offset, field)) in &fields {
                        expected_offset = ceil_to_multiple(expected_offset, field.alignment);
                        if *offset != expected_offset {
                            crate::trace!(
                                "Unexpected dtype field offset for field '{}': expected {}, got {}",
                                f_name,
                                expected_offset,
                                offset
                            );
                            return Err(std::fmt::Error);
                        }
                        expected_offset += field.itemsize;
                    }
                }

                f.write_str("'names':[")?;
                for (f_name, _field) in &fields {
                    write!(f, "'{}',", f_name)?;
                }
                f.write_char(']')?;
                f.write_char(',')?;

                f.write_str("'formats':[")?;
                for (_f_name, (offset, field)) in &fields {
                    field.write_as_numpy_str(f, true)?;
                    f.write_char(',')?;
                }
                f.write_char(']')?;
                f.write_char(',')?;

                f.write_str("'offsets':[")?;
                for (_f_name, (offset, field)) in &fields {
                    write!(f, "{},", offset)?;
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

    use crate::{Dtype, DtypeKind, DtypeScalarKind};

    #[test]
    fn dtype_from_str() {
        // use super::Dtype::of_scalar as sdtype;
        use DtypeScalarKind as SKind;

        let sdtype = super::Dtype::of_scalar;

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
                                (0, sdtype(SKind::I32)),
                            ),
                            (
                                "b".to_string(),
                                (4, sdtype(SKind::F64)),
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
                                (0, sdtype(SKind::I32)),
                            ),
                            (
                                "b".to_string(),
                                (8, sdtype(SKind::F64)),
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
                                (0, sdtype(SKind::I32)),
                            ),
                            (
                                "b".to_string(),
                                (8, sdtype(SKind::F64)),
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
                                (0, sdtype(SKind::I32))
                            ),
                            (
                                "b".to_string(),
                                (4, Dtype {
                                    shape: vec![2, 3, 4],
                                    itemsize: 192,
                                    ..sdtype(SKind::F64)
                                }),
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
                                (0, sdtype(SKind::I32))
                            ),
                            (
                                "b".to_string(),
                                (8, Dtype {
                                    shape: vec![2, 3, 4],
                                    itemsize: 192,
                                    ..sdtype(SKind::F64)
                                }),
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
                                (0, sdtype(SKind::I32))
                            ),
                            (
                                "b".to_string(),
                                (8, Dtype {
                                    shape: vec![2, 3, 4],
                                    itemsize: 192,
                                    ..sdtype(SKind::F64)
                                }),
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
            let dtype = Dtype::from_numpy_str(dtype_str);
            assert_eq!(dtype, Ok(expected_dtype.clone()));

            let dtype_str2 = dtype.unwrap().to_numpy_str();
            let dtype2 = Dtype::from_numpy_str(dtype_str2.as_str());
            assert_eq!(dtype2, Ok(expected_dtype));
        }
    }
}
