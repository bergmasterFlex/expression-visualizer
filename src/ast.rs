// ── AST node types ──────────────────────────────────────────
//
#[derive(Clone)]
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
    NumLiteral(f32),
    BoolLiteral(bool),
    MatchTrue {
        input_argument: Option<AstNodeId>,
    },
    MatchFalse {
        input_argument: Option<AstNodeId>,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(pub usize);

pub struct FunctionDeclaration {
    pub name: String,
    pub inputs: Vec<FunctionParameterDeclaration>,
    pub output_type: EType,
}

pub struct FunctionParameterDeclaration {
    pub name: String,
    pub r#type: EType,
}

#[derive(Debug, Clone)]
pub enum EType {
    Number,
    Bool,
    SumType(Vec<EType>),
    Error,
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self {
            EType::Number => "number".to_string(),
            EType::Bool => "bool".to_string(),
            EType::SumType(sub_types) => sub_types
                .iter()
                .map(|sub_type| sub_type.to_string())
                .collect::<Vec<_>>()
                .join("|"),
            EType::Error => "error".to_string(),
        }
    }
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
            EAstNode::MatchTrue { .. } => "match true".to_string(),
            EAstNode::MatchFalse { .. } => "match false".to_string(),
        }
    }

    /// Type name for UI tooltips.
    pub fn eval_type(
        &self,
        ast: &Ast,
        function_declarations: &std::collections::HashMap<
            FunctionDeclarationId,
            FunctionDeclaration,
        >,
    ) -> EType {
        match self {
            EAstNode::Sink { input } => input
                .as_ref()
                .and_then(|i| ast.nodes.get(i))
                .map(|n| n.eval_type(ast, function_declarations))
                .unwrap_or(EType::Error),
            EAstNode::FunctionCall {
                function_declaration_id,
                ..
            } => function_declarations
                .get(&function_declaration_id)
                .unwrap()
                .output_type
                .clone(),
            EAstNode::BoolLiteral(_) => EType::Bool,
            EAstNode::NumLiteral(_) => EType::Number,
            EAstNode::MatchTrue { .. } => EType::Bool,
            EAstNode::MatchFalse { .. } => EType::Bool,
        }
    }
}
