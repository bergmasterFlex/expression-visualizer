#[derive(Debug, Clone)]
pub enum EType {
    Int(Option<i32>),
    Bool(Option<bool>),
    String(Option<String>),
    Char(Option<char>),
    /// The inferer could not (yet) decide this output type: a Match whose
    /// branches are incomplete, or a Match/TypeCast with no input edge, where
    /// total-vs-partial cannot be decided. Propagates outward until a node
    /// that fixes its own output type. Never a *final* type — every finished
    /// output anchor resolves to a concrete type.
    Pending,
    None,
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
            EType::Pending => "pending".to_string(),
            EType::None => "none".to_string(),
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
        crate::model::r#type::EType::None { .. } => EType::None,
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

type FunctionDeclarations = std::collections::HashMap<
    crate::model::function_declaration::FunctionDeclarationId,
    crate::model::function_declaration::FunctionDeclaration,
>;

/// The type an anchor carries.
///
/// Output anchors always answer `Some(..)` — a finished graph gives every
/// output a concrete type, an unfinished one gives `Pending`. Input anchors
/// answer `Some(..)` only where the node constrains what may flow in
/// (FunctionCall parameters); `None` means *no constraint* — Sink, Match and
/// TypeCast inputs accept any value. There is no supertype: `None` is the
/// absence of a constraint, not a type that subsumes the others.
///
/// `ast` must be the flattened AST (`LayoutAst::flattened_ast`): every edge —
/// including those inside Pattern branches — lives in the program-level edge
/// table, so inference on a bare sub-AST would see no edges at all.
pub fn anchor_type(
    ast: &crate::model::ast::Ast,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
) -> Option<EType> {
    anchor_type_guarded(
        ast,
        anchor_id,
        function_declarations,
        &mut std::collections::HashSet::new(),
    )
}

/// `visiting` holds the anchors on the current inference path. A user can wire
/// a cycle (A's output feeds B's input feeds A's input), and both the TypeCast
/// and the Match rule recurse through incoming edges — without the guard that
/// recursion never terminates. A cycle is by definition undecidable, so it
/// resolves to `Pending`.
fn anchor_type_guarded(
    ast: &crate::model::ast::Ast,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    if !visiting.insert(anchor_id.clone()) {
        return Some(EType::Pending);
    }
    let result = anchor_type_uncycled(ast, anchor_id, function_declarations, visiting);
    visiting.remove(anchor_id);
    result
}

fn anchor_type_uncycled(
    ast: &crate::model::ast::Ast,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    let node_id = ast.anchor_to_node.get(anchor_id)?;
    match ast.nodes.get(node_id)? {
        // Declared types are fixed: they never depend on what flows in.
        crate::model::node::ENode::ConstDecl {
            r#type,
            output_anchor,
        }
        | crate::model::node::ENode::VarDecl {
            r#type,
            output_anchor,
            ..
        }
        | crate::model::node::ENode::Pattern {
            r#type,
            output_anchor,
            ..
        } => (anchor_id == output_anchor).then(|| ast_type_to_eval_type(r#type)),
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            input_anchors,
            output_anchor,
        } => {
            let declaration = function_declarations.get(function_declaration_id)?;
            if anchor_id == output_anchor {
                return Some(declaration.output_type.clone());
            }
            let index = input_anchors.iter().position(|a| a == anchor_id)?;
            declaration.inputs.get(index).map(|p| p.r#type.clone())
        }
        crate::model::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        } => (anchor_id == output_anchor).then(|| {
            type_cast_output_type(ast, r#type, input_anchor, function_declarations, visiting)
        }),
        crate::model::node::ENode::Match {
            patterns,
            input_anchor,
            output_anchor,
        } => (anchor_id == output_anchor).then(|| {
            match_output_type(ast, patterns, input_anchor, function_declarations, visiting)
        }),
        // Sink input takes anything; Program has no anchors at all.
        crate::model::node::ENode::Sink { .. } | crate::model::node::ENode::Program {} => None,
    }
}

/// A cast to `target` is total when the incoming type already matches, and
/// partial otherwise — a partial cast can fail, which is modelled as
/// `Sum(target, none)`. Which of the two applies cannot be decided before an
/// incoming type is known, so an unconnected (or itself pending) input makes
/// the output `Pending`.
fn type_cast_output_type(
    ast: &crate::model::ast::Ast,
    target: &crate::model::r#type::EType,
    input_anchor: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> EType {
    let target = ast_type_to_eval_type(target);
    match incoming_type(ast, input_anchor, function_declarations, visiting) {
        None | Some(EType::Pending) => EType::Pending,
        Some(incoming) if !types_match(&incoming, &target) => {
            EType::SumType(vec![target, EType::None])
        }
        Some(_) => target,
    }
}

/// A Match yields whatever its selected branch yields, so its output type is
/// the union of the branch types. Every branch must be known: an unwired
/// branch, a branch that is itself pending, or a Match with no patterns at all
/// leaves the union undecided. The Match input matters too — without it the
/// pattern set cannot be checked against what actually flows in.
fn match_output_type(
    ast: &crate::model::ast::Ast,
    patterns: &[crate::model::node::Id],
    input_anchor: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> EType {
    if patterns.is_empty() {
        return EType::Pending;
    }
    match incoming_type(ast, input_anchor, function_declarations, visiting) {
        None | Some(EType::Pending) => return EType::Pending,
        Some(_) => {}
    }
    let mut leaves: Vec<EType> = Vec::new();
    for pattern_id in patterns {
        let branch = match branch_type(ast, pattern_id, function_declarations, visiting) {
            None | Some(EType::Pending) => return EType::Pending,
            Some(t) => t,
        };
        for leaf in flatten_type(&branch) {
            if !leaves.iter().any(|seen| types_match(seen, &leaf)) {
                leaves.push(leaf);
            }
        }
    }
    match leaves.len() {
        0 => EType::Pending,
        1 => leaves.remove(0),
        _ => EType::SumType(leaves),
    }
}

/// Type a Pattern's branch produces: whatever reaches the branch's own Sink.
fn branch_type(
    ast: &crate::model::ast::Ast,
    pattern_id: &crate::model::node::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    let crate::model::node::ENode::Pattern { sink_node_id, .. } = ast.nodes.get(pattern_id)? else {
        return None;
    };
    let crate::model::node::ENode::Sink { input_anchor } = ast.nodes.get(sink_node_id)? else {
        return None;
    };
    incoming_type(ast, input_anchor, function_declarations, visiting)
}

/// Type flowing into `input` from the connected source anchor. `None` when the
/// input is unconnected or the source carries no type.
fn incoming_type(
    ast: &crate::model::ast::Ast,
    input: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    let source = source_anchor_for_input(ast, input)?;
    anchor_type_guarded(ast, &source, function_declarations, visiting)
}

/// Source anchor feeding into `input`, if connected. Drag-to-connect records
/// an edge from either end, so both directions are checked.
pub fn source_anchor_for_input(
    ast: &crate::model::ast::Ast,
    input: &crate::model::anchor::Id,
) -> Option<crate::model::anchor::Id> {
    for (from, edges) in &ast.edges {
        if edges.iter().any(|e| &e.to == input) {
            return Some(from.clone());
        }
    }
    ast.edges
        .get(input)
        .and_then(|edges| edges.first())
        .map(|e| e.to.clone())
}

/// Type a node produces, i.e. the type of its output anchor. `None` for nodes
/// that have no output at all (Sink, Program). Used for the selection display.
pub fn node_output_type(
    ast: &crate::model::ast::Ast,
    node_id: &crate::model::node::Id,
    function_declarations: &FunctionDeclarations,
) -> Option<EType> {
    let output_anchor =
        ast.nodes
            .get(node_id)?
            .anchors()
            .into_iter()
            .find_map(|(id, anchor)| {
                matches!(anchor, crate::model::anchor::EAnchor::Output).then_some(id)
            })?;
    anchor_type(ast, &output_anchor, function_declarations)
}

/// Structural type equality, ignoring any carried value literal. Two `SumType`s
/// match when their leaves match pairwise in order.
pub fn types_match(a: &EType, b: &EType) -> bool {
    match (a, b) {
        (EType::Int(_), EType::Int(_))
        | (EType::Bool(_), EType::Bool(_))
        | (EType::String(_), EType::String(_))
        | (EType::Char(_), EType::Char(_))
        | (EType::None, EType::None)
        | (EType::Pending, EType::Pending)
        | (EType::Exception, EType::Exception) => true,
        (EType::SumType(x), EType::SumType(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| types_match(p, q))
        }
        _ => false,
    }
}
