pub(crate) enum Node {
    Literal(String),
    Str(String),
    List(Vec<Node>),
    Tuple(Vec<Node>),
    Dict(Vec<(Node, Node)>),
}

pub(crate) fn parse_ast(s: &str) -> Result<Node, ParseError> {
    parse_ast_impl(&mut Chars(s.chars())).map_err(|e| ParseError {
        msg: e.msg,
        pos: s.len() - e.pos_from_end,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParseError {
    pub msg: &'static str,
    pub pos: usize,
}
impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AST parse error at position {}: {}", self.pos, self.msg)
    }
}

struct ParseErrorInternal {
    pub msg: &'static str,
    pub pos_from_end: usize,
}

struct Chars<'a>(std::str::Chars<'a>);
impl Chars<'_> {
    fn peek(&mut self) -> Option<char> {
        self.0.clone().next()
    }
    fn next(&mut self) -> Option<char> {
        self.0.next()
    }
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.next();
            } else {
                break;
            }
        }
    }
    fn err(&self, msg: &'static str) -> ParseErrorInternal {
        ParseErrorInternal {
            msg,
            pos_from_end: self.0.as_str().len(),
        }
    }
}

fn parse_ast_impl(s: &mut Chars) -> Result<Node, ParseErrorInternal> {
    let first_char = s.next().ok_or(s.err("Unexpected end of input"))?;
    match first_char {
        '[' | '(' => {
            let expected_end_char = if first_char == '[' { ']' } else { ')' };
            let mut children = Vec::new();

            s.skip_whitespace();
            if s.peek() == Some(expected_end_char) {
                s.next();
            } else {
                loop {
                    children.push(parse_ast_impl(s)?);
                    s.skip_whitespace();

                    match s.next().ok_or(s.err("tuple/list was never closed"))? {
                        ',' => {
                            s.skip_whitespace();
                            if s.peek() == Some(expected_end_char) {
                                s.next();
                                break;
                            }
                        }
                        c if c == expected_end_char => break,
                        _ => return Err(s.err("Expected ',' or end of list/tuple")),
                    }
                }
            }

            Ok(if expected_end_char == ')' {
                Node::Tuple(children)
            } else {
                Node::List(children)
            })
        }

        '{' => {
            let mut children = Vec::new();
            s.skip_whitespace();
            if s.peek() == Some('}') {
                s.next();
            } else {
                loop {
                    let key = parse_ast_impl(s)?;
                    s.skip_whitespace();
                    if s.next() != Some(':') {
                        return Err(s.err("Expected ':'"));
                    }
                    s.skip_whitespace();
                    let value = parse_ast_impl(s)?;
                    children.push((key, value));
                    s.skip_whitespace();

                    match s.next().ok_or(s.err("'{' was never closed"))? {
                        ',' => {
                            s.skip_whitespace();
                            if s.peek() == Some('}') {
                                s.next();
                                break;
                            }
                        }
                        '}' => break,
                        _ => return Err(s.err("Expected ',' or '}'")),
                    }
                }
            }
            Ok(Node::Dict(children))
        }

        '"' | '\'' => {
            let end_char = if first_char == '"' { '"' } else { '\'' };
            let mut value = String::new();
            let mut escape = false;
            loop {
                let char = s.next().ok_or(s.err("unterminated string literal"))?;
                if escape {
                    value.push(char);
                    escape = false;
                } else if char == '\\' {
                    escape = true;
                } else if char == end_char {
                    break;
                } else {
                    value.push(char);
                }
            }
            Ok(Node::Str(value))
        }

        'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
            let mut value = String::from(first_char);
            loop {
                let next_char = s.peek();
                if next_char.is_none()
                    || !matches!(next_char.unwrap(), 'a'..='z' | 'A'..='Z' | '0'..='9' | '_')
                {
                    break;
                }
                value.push(next_char.unwrap());
                s.next();
            }
            Ok(Node::Literal(value))
        }

        _ => Err(s.err("Unexpected token")),
    }
}
