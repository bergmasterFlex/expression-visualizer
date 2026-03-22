/// Arithmetic expression tokenizer and recursive-descent parser.
///
/// Grammar:
///   expr       → ternary
///   ternary    → comparison ('?' ternary ':' ternary)?
///   comparison → additive (('>'|'<'|'>='|'<='|'=='|'!=') additive)?
///   additive   → mult (('+' | '-') mult)*
///   mult       → atom (('*' | '/') atom)*
///   atom       → NUMBER | BOOL | '(' expr ')'

// ── AST node types ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum AstNode {
    NumLiteral(String),
    BoolLiteral(bool),
    BinaryExpr {
        op: char,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
    ComparisonExpr {
        op: String,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
    TernaryExpr {
        condition: Box<AstNode>,
        consequent: Box<AstNode>,
        alternate: Box<AstNode>,
    },
}

impl AstNode {
    /// Short display label for rendering.
    pub fn label(&self) -> String {
        match self {
            AstNode::NumLiteral(v) => v.clone(),
            AstNode::BoolLiteral(b) => if *b { "true" } else { "false" }.into(),
            AstNode::BinaryExpr { op, .. } => op.to_string(),
            AstNode::ComparisonExpr { op, .. } => op.clone(),
            AstNode::TernaryExpr { .. } => "?:".into(),
        }
    }

    /// Type name for UI tooltips.
    pub fn type_name(&self) -> &'static str {
        match self {
            AstNode::NumLiteral(_) => "NumLiteral",
            AstNode::BoolLiteral(_) => "BoolLiteral",
            AstNode::BinaryExpr { .. } => "BinaryExpr",
            AstNode::ComparisonExpr { .. } => "CompareExpr",
            AstNode::TernaryExpr { .. } => "TernaryExpr",
        }
    }
}

// ── Tokenizer ───────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Token {
    Num(String),
    Bool(bool),
    Op(char),        // + - * / ( ) ? :
    Cmp(String),     // > < >= <= == !=
}

fn tokenize(input: &str) -> Vec<Token> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => i += 1,
            c @ ('+' | '-' | '*' | '/' | '(' | ')' | '?' | ':') => {
                tokens.push(Token::Op(c));
                i += 1;
            }
            '>' | '<' | '=' | '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Cmp(format!("{}=", chars[i])));
                    i += 2;
                } else if chars[i] == '>' || chars[i] == '<' {
                    tokens.push(Token::Cmp(chars[i].to_string()));
                    i += 1;
                } else {
                    i += 1;
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                tokens.push(Token::Num(chars[start..i].iter().collect()));
            }
            c if c.is_ascii_alphabetic() => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                match word.as_str() {
                    "true" => tokens.push(Token::Bool(true)),
                    "false" => tokens.push(Token::Bool(false)),
                    _ => {} // ignore unknown identifiers
                }
            }
            _ => i += 1,
        }
    }
    tokens
}

// ── Parser ──────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        tok
    }

    fn peek_op(&self) -> Option<char> {
        match self.peek()? {
            Token::Op(c) => Some(*c),
            _ => None,
        }
    }

    fn peek_cmp(&self) -> Option<String> {
        match self.peek()? {
            Token::Cmp(s) => Some(s.clone()),
            _ => None,
        }
    }

    // expr → ternary
    fn parse_expr(&mut self) -> AstNode {
        self.parse_ternary()
    }

    // ternary → comparison ('?' ternary ':' ternary)?
    fn parse_ternary(&mut self) -> AstNode {
        let cond = self.parse_comparison();
        if self.peek_op() == Some('?') {
            self.consume(); // eat '?'
            let consequent = self.parse_ternary();
            if self.peek_op() == Some(':') {
                self.consume(); // eat ':'
            }
            let alternate = self.parse_ternary();
            AstNode::TernaryExpr {
                condition: Box::new(cond),
                consequent: Box::new(consequent),
                alternate: Box::new(alternate),
            }
        } else {
            cond
        }
    }

    // comparison → additive (cmp_op additive)?
    fn parse_comparison(&mut self) -> AstNode {
        let left = self.parse_additive();
        if let Some(op) = self.peek_cmp() {
            self.consume();
            let right = self.parse_additive();
            AstNode::ComparisonExpr {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            left
        }
    }

    // additive → mult (('+' | '-') mult)*
    fn parse_additive(&mut self) -> AstNode {
        let mut left = self.parse_mult();
        while matches!(self.peek_op(), Some('+') | Some('-')) {
            let op = match self.consume() {
                Some(Token::Op(c)) => c,
                _ => unreachable!(),
            };
            let right = self.parse_mult();
            left = AstNode::BinaryExpr {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        left
    }

    // mult → atom (('*' | '/') atom)*
    fn parse_mult(&mut self) -> AstNode {
        let mut left = self.parse_atom();
        while matches!(self.peek_op(), Some('*') | Some('/')) {
            let op = match self.consume() {
                Some(Token::Op(c)) => c,
                _ => unreachable!(),
            };
            let right = self.parse_atom();
            left = AstNode::BinaryExpr {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        left
    }

    // atom → NUMBER | BOOL | '(' expr ')'
    fn parse_atom(&mut self) -> AstNode {
        match self.peek().cloned() {
            Some(Token::Op('(')) => {
                self.consume();
                let node = self.parse_expr();
                if self.peek_op() == Some(')') {
                    self.consume();
                }
                node
            }
            Some(Token::Num(v)) => {
                self.consume();
                AstNode::NumLiteral(v)
            }
            Some(Token::Bool(b)) => {
                self.consume();
                AstNode::BoolLiteral(b)
            }
            _ => {
                self.consume(); // skip unknown
                AstNode::NumLiteral("?".into())
            }
        }
    }
}

/// Parse an arithmetic expression string into an AST.
pub fn parse(input: &str) -> AstNode {
    let tokens = tokenize(input);
    if tokens.is_empty() {
        return AstNode::NumLiteral("?".into());
    }
    let mut parser = Parser::new(tokens);
    parser.parse_expr()
}
