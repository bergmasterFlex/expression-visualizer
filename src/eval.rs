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

/// Convert the AST-level type descriptor into the evaluation-level type,
/// discarding any user-typed value literal.
pub fn ast_type_to_eval_type(t: &crate::ast::node::EType) -> EType {
    match t {
        crate::ast::node::EType::Bool { .. } => EType::Bool(None),
        crate::ast::node::EType::Int { .. } => EType::Int(None),
        crate::ast::node::EType::Float { .. } => EType::Float(None),
        crate::ast::node::EType::Char { .. } => EType::Char(None),
        crate::ast::node::EType::String { .. } => EType::String(None),
        crate::ast::node::EType::Any => EType::Any,
        crate::ast::node::EType::Undefined { .. } => EType::Undefined,
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
        crate::ast::node::ENode::ConstDecl { r#type, .. }
        | crate::ast::node::ENode::TypeCast { r#type, .. }
        | crate::ast::node::ENode::VarDecl { r#type, .. }
        | crate::ast::node::ENode::Pattern { r#type, .. } => Ok(match r#type {
            crate::ast::node::EType::Bool { value } => EType::Bool(None),
            crate::ast::node::EType::Int { value } => EType::Int(None),
            crate::ast::node::EType::Float { value } => EType::Float(None),
            crate::ast::node::EType::Char { value } => EType::Char(None),
            crate::ast::node::EType::String { value } => EType::String(None),
            crate::ast::node::EType::Any => EType::Any,
            crate::ast::node::EType::Undefined { .. } => EType::Undefined,
        }),
        crate::ast::node::ENode::Match { .. } => Err("match has no type".to_string()),
        crate::ast::node::ENode::Program { .. } => Err("program has no type".to_string()),
    }
}

// ── Stepwise evaluation (showcase v1) ───────────────────────

pub const RANDOM_POOL: [&str; 6] = ["1", "-3", "1.345", "\"test-value\"", "'x'", "true"];

use rand::seq::IndexedRandom;
use std::collections::HashMap;

fn pick_random(rng: &mut impl rand::Rng) -> String {
    RANDOM_POOL.choose(rng).copied().unwrap_or("?").to_string()
}

/// Collect every VarDecl in the AST as (node_id, name).
pub fn collect_var_decls(ast: &crate::ast::Ast) -> Vec<(crate::ast::node::Id, String)> {
    let mut out: Vec<_> = ast
        .nodes
        .iter()
        .filter_map(|(id, node)| match node {
            crate::ast::node::ENode::VarDecl { name, .. } => Some((id.clone(), name.clone())),
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
    ast: &crate::ast::Ast,
    anchor: &crate::ast::anchor::Id,
) -> Vec<crate::ast::node::Id> {
    let mut out: Vec<crate::ast::node::Id> = ast.get_connected_nodes_to_anchor(anchor.clone());
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
pub fn sink_has_input(ast: &crate::ast::Ast) -> bool {
    ast.nodes.values().any(|node| match node {
        crate::ast::node::ENode::Sink { input_anchor } => {
            !neighbours_of_anchor(ast, input_anchor).is_empty()
        }
        _ => false,
    })
}

/// Step 0 snapshot: seed every VarDecl with the user's typed value (or empty
/// string if missing) and every ConstDecl with a random pool pick.
pub fn initial_values(
    ast: &crate::ast::Ast,
    user_vardecl_values: &HashMap<crate::ast::node::Id, String>,
    rng: &mut impl rand::Rng,
) -> HashMap<crate::ast::node::Id, String> {
    let mut out = HashMap::new();
    for (id, node) in &ast.nodes {
        match node {
            crate::ast::node::ENode::VarDecl { .. } => {
                let v = user_vardecl_values.get(id).cloned().unwrap_or_default();
                out.insert(id.clone(), v);
            }
            crate::ast::node::ENode::ConstDecl { .. } => {
                out.insert(id.clone(), pick_random(rng));
            }
            _ => {}
        }
    }
    out
}

fn node_input_anchors(node: &crate::ast::node::ENode) -> Vec<crate::ast::anchor::Id> {
    node.anchors()
        .into_iter()
        .filter_map(|(aid, a)| match a {
            crate::ast::anchor::EAnchor::Input { .. } => Some(aid),
            _ => None,
        })
        .collect()
}

fn newly_eligible_nodes(
    ast: &crate::ast::Ast,
    current: &HashMap<crate::ast::node::Id, String>,
) -> Vec<crate::ast::node::Id> {
    let mut out = Vec::new();
    for (id, node) in &ast.nodes {
        if current.contains_key(id) {
            continue;
        }
        let inputs = node_input_anchors(node);
        if inputs.is_empty() {
            continue;
        }
        let all_valued = inputs.iter().all(|aid| {
            let sources = neighbours_of_anchor(ast, aid);
            !sources.is_empty() && sources.iter().any(|sid| current.contains_key(sid))
        });
        if all_valued {
            out.push(id.clone());
        }
    }
    out
}

/// Quick check used by the UI to grey out the Next button.
pub fn can_step_next(
    ast: &crate::ast::Ast,
    current: &HashMap<crate::ast::node::Id, String>,
) -> bool {
    !newly_eligible_nodes(ast, current).is_empty()
}

/// Compute next snapshot: every not-yet-valued node whose every input anchor
/// has at least one incoming edge to a source node that already has a value
/// gets a freshly random-picked value. Returns None if nothing would change.
pub fn step_next(
    ast: &crate::ast::Ast,
    current: &HashMap<crate::ast::node::Id, String>,
    rng: &mut impl rand::Rng,
) -> Option<HashMap<crate::ast::node::Id, String>> {
    let eligible = newly_eligible_nodes(ast, current);
    if eligible.is_empty() {
        return None;
    }
    let mut next = current.clone();
    for id in eligible {
        next.insert(id, pick_random(rng));
    }
    Some(next)
}
