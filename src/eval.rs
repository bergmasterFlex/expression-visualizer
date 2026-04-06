#[derive(Debug, Clone)]
pub enum EType {
    Int(Option<i32>),
    Float(Option<f32>),
    Bool(Option<bool>),
    String(Option<String>),
    Char(Option<char>),
    Any,
    Undefined,
    Exception,
    SumType(Vec<EType>),
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self {
            EType::Int(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "int".to_string()
                }
            }
            EType::Float(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "float".to_string()
                }
            }
            EType::Bool(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "bool".to_string()
                }
            }
            EType::String(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "string".to_string()
                }
            }
            EType::Char(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "char".to_string()
                }
            }
            EType::Any => "any".to_string(),
            EType::Undefined => "undefined".to_string(),
            EType::Exception => "exception".to_string(),
            EType::SumType(sub_types) => sub_types
                .iter()
                .map(|sub_type| sub_type.to_string())
                .collect::<Vec<_>>()
                .join("|"),
        }
    }
}

fn has_duplicates<T: Eq + std::hash::Hash>(v: &[T]) -> bool {
    let mut seen = std::collections::HashSet::new();
    v.iter().any(|item| !seen.insert(item))
}

/// Type name for UI tooltips.
pub fn eval_type(
    node: &crate::ast::node::ENode,
    ast: &crate::ast::Ast,
    function_declarations: &std::collections::HashMap<
        crate::ast::FunctionDeclarationId,
        crate::ast::FunctionDeclaration,
    >,
    visited_nodes: Vec<crate::ast::node::Id>,
) -> Result<EType, String> {
    if has_duplicates(&visited_nodes) {
        return Err("infinite edge loop".to_string());
    }
    match node {
        crate::ast::node::ENode::Sink { input_anchor } => {
            match ast
                .get_connected_nodes_to_anchor(input_anchor.clone())
                .first()
            {
                Some(input_node_id) => eval_type(
                    ast.nodes.get(input_node_id).unwrap(),
                    ast,
                    function_declarations,
                    visited_nodes
                        .iter()
                        .cloned()
                        .chain([input_node_id.clone()])
                        .collect(),
                ),
                None => Err("no edge to sink input".to_string()),
            }
        }
        crate::ast::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        } => Ok(function_declarations
            .get(&function_declaration_id)
            .unwrap()
            .output_type
            .clone()),
        crate::ast::node::ENode::TypeIntroduction { r#type, .. }
        | crate::ast::node::ENode::TypeElimination { r#type, .. } => Ok(match r#type {
            crate::ast::node::EType::Bool { value } => EType::Bool(None),
            crate::ast::node::EType::Int { value } => EType::Int(None),
            crate::ast::node::EType::Float { value } => EType::Float(None),
            crate::ast::node::EType::Char { value } => EType::Char(None),
            crate::ast::node::EType::String { value } => EType::String(None),
            crate::ast::node::EType::Any => EType::Any,
            crate::ast::node::EType::Undefined => EType::Undefined,
            crate::ast::node::EType::Exception { message } => EType::Exception,
        }),
        crate::ast::node::ENode::Match { input_anchor, .. } => {
            match ast
                .get_connected_nodes_to_anchor(input_anchor.clone())
                .first()
            {
                Some(input_node_id) => eval_type(
                    ast.nodes.get(input_node_id).unwrap(),
                    ast,
                    function_declarations,
                    visited_nodes
                        .iter()
                        .cloned()
                        .chain([input_node_id.clone()])
                        .collect(),
                ),
                None => Err("no edge to match input".to_string()),
            }
        }
    }
}
