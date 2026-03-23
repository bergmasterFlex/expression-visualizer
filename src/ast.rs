// ── AST node types ──────────────────────────────────────────
//
pub struct Ast {
    pub nodes: std::collections::HashMap<AstNodeId, EAstNode>,
}

impl Ast {
    pub fn empty() -> Self {
        Self {
            nodes: std::collections::HashMap::new(),
        }
    }

    pub fn plus(&self, n: EAstNode) -> Self {
        let mut nodes = self.nodes.clone();
        nodes.insert(self.next_id(), n);
        Self { nodes }
    }

    pub fn next_id(&self) -> AstNodeId {
        AstNodeId(self.nodes.keys().map(|id| id.0 + 1).max().unwrap_or(0))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AstNodeId(usize);

#[derive(Debug, Clone)]
pub enum EAstNode {
    Sink {
        input: Option<AstNodeId>,
    },
    FunctionCall {
        function_declaration_id: FunctionDeclarationId,
        input_arguments: Vec<AstNodeId>,
    },
    NumLiteral(String),
    BoolLiteral(bool),
    MatchTrue {
        input_argument: Option<AstNodeId>,
    },
    MatchFalse {
        input_argument: Option<AstNodeId>,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(usize);

pub struct FunctionDeclaration {
    name: String,
    inputs: Vec<FunctionParameterDeclaration>,
    output_type: EType,
}

pub struct FunctionParameterDeclaration {
    name: String,
    r#type: EType,
}

#[derive(Debug, Clone)]
pub enum EType {
    Number,
    Bool,
    SumType(Vec<EType>),
    Error,
}

impl EAstNode {
    pub fn label(
        &self,
        function_declarations: &std::collections::HashMap<
            FunctionDeclarationId,
            FunctionDeclaration,
        >,
    ) -> String {
        match self {
            EAstNode::Sink { .. } => "sink".to_string(),
            EAstNode::FunctionCall {
                function_declaration_id,
                ..
            } => function_declarations
                .get(&function_declaration_id)
                .unwrap()
                .name
                .to_string(),
            EAstNode::BoolLiteral(b) => b.to_string(),
            EAstNode::NumLiteral(n) => n.to_string(),
            EAstNode::MatchTrue { .. } => "true".to_string(),
            EAstNode::MatchFalse { .. } => "false".to_string(),
        }
    }

    /// Type name for UI tooltips.
    pub fn type_name(&self) -> &'static str {
        match self {
            _ => "<dummy type>",
        }
    }
}
