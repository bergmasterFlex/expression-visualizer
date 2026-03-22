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
