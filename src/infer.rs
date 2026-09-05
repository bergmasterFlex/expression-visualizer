#[derive(Debug, Clone)]
pub enum EType {
    Int(Option<i32>),
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

/// Convert the AST-level type descriptor into the evaluation-level type,
/// discarding any user-typed value literal.
pub fn ast_type_to_eval_type(t: &crate::model::r#type::EType) -> EType {
    match t {
        crate::model::r#type::EType::Bool { .. } => EType::Bool(None),
        crate::model::r#type::EType::Int { .. } => EType::Int(None),
        crate::model::r#type::EType::Char { .. } => EType::Char(None),
        crate::model::r#type::EType::String { .. } => EType::String(None),
        crate::model::r#type::EType::Any => EType::Any,
        crate::model::r#type::EType::Undefined { .. } => EType::Undefined,
    }
}

/// Recursively flatten a type into its leaf variants. `SumType`s are expanded
/// depth-first; every non-`SumType` variant is emitted as-is.
pub fn flatten_type(t: &EType) -> Vec<EType> {
    match t {
        EType::SumType(sub_types) => sub_types.iter().flat_map(flatten_type).collect(),
        other => vec![other.clone()],
    }
}

fn has_duplicates<T: Eq + std::hash::Hash>(v: &[T]) -> bool {
    let mut seen = std::collections::HashSet::new();
    v.iter().any(|item| !seen.insert(item))
}

/// Type name for UI tooltips.
pub fn eval_type(
    node: &crate::model::node::ENode,
    ast: &crate::model::ast::Ast,
    function_declarations: &std::collections::HashMap<
        crate::model::function_declaration::FunctionDeclarationId,
        crate::model::function_declaration::FunctionDeclaration,
    >,
    visited_nodes: Vec<crate::model::node::Id>,
) -> Result<EType, String> {
    if has_duplicates(&visited_nodes) {
        return Err("infinite edge loop".to_string());
    }
    match node {
        crate::model::node::ENode::Sink { input_anchor } => {
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
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        } => Ok(function_declarations
            .get(&function_declaration_id)
            .unwrap()
            .output_type
            .clone()),
        crate::model::node::ENode::ConstDecl { r#type, .. }
        | crate::model::node::ENode::TypeCast { r#type, .. }
        | crate::model::node::ENode::VarDecl { r#type, .. }
        | crate::model::node::ENode::Pattern { r#type, .. } => Ok(match r#type {
            crate::model::r#type::EType::Bool { value } => EType::Bool(None),
            crate::model::r#type::EType::Int { value } => EType::Int(None),
            crate::model::r#type::EType::Char { value } => EType::Char(None),
            crate::model::r#type::EType::String { value } => EType::String(None),
            crate::model::r#type::EType::Any => EType::Any,
            crate::model::r#type::EType::Undefined { .. } => EType::Undefined,
        }),
        crate::model::node::ENode::Match { .. } => Err("match has no type".to_string()),
        crate::model::node::ENode::Program { .. } => Err("program has no type".to_string()),
    }
}

/// Collect every VarDecl in the AST as (node_id, name).
pub fn collect_var_decls(ast: &crate::model::ast::Ast) -> Vec<(crate::model::node::Id, String)> {
    let mut out: Vec<_> = ast
        .nodes
        .iter()
        .filter_map(|(id, node)| match node {
            crate::model::node::ENode::VarDecl { name, .. } => Some((id.clone(), name.clone())),
            _ => None,
        })
        .collect();
    out.sort_by(|(a, _), (b, _)| a.cmp(b));
    out
}

/// Direction-agnostic neighbours of an anchor: returns every node sharing an
/// edge with `anchor`, regardless of which end the edge was recorded from.
/// Drag-to-connect lets the user start from either anchor, so we accept both.
fn neighbours_of_anchor(
    ast: &crate::model::ast::Ast,
    anchor: &crate::model::anchor::Id,
) -> Vec<crate::model::node::Id> {
    let mut out: Vec<crate::model::node::Id> = ast.get_connected_nodes_to_anchor(anchor.clone());
    if let Some(edges) = ast.edges.get(anchor) {
        for e in edges {
            if let Some(n) = ast.anchor_to_node.get(&e.to) {
                out.push(n.clone());
            }
        }
    }
    out
}

/// True if any Sink has at least one edge on its input anchor.
pub fn sink_has_input(ast: &crate::model::ast::Ast) -> bool {
    ast.nodes.values().any(|node| match node {
        crate::model::node::ENode::Sink { input_anchor } => {
            !neighbours_of_anchor(ast, input_anchor).is_empty()
        }
        _ => false,
    })
}
