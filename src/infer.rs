/// A type as the inferer sees it.
///
/// The four value kinds carry the literal they are pinned to, as the text the
/// user typed — the same text `model::r#type::EType` holds, and unparsed for
/// the same reason: half a number is still what stands on the node, and there
/// is no moment at which the graph is allowed to disagree with what is written
/// on it.
///
/// A pinned literal makes the type *narrower*, not merely decorated: `1` and
/// `2` are two types, and a sum of them is not `Integer`. Nothing widens a
/// literal back to its base type except a value set that covers it (see
/// `normalize_leaves`) or an explicit TypeCast.
#[derive(Debug, Clone)]
pub enum EType {
    Int(Option<String>),
    Bool(Option<String>),
    String(Option<String>),
    Char(Option<String>),
    /// The inferer could not (yet) decide this output type: a Match whose
    /// branches are incomplete, or a Match/TypeCast with no input edge, where
    /// total-vs-partial cannot be decided. Propagates outward until a node
    /// that fixes its own output type. Never a *final* type — every finished
    /// output anchor resolves to a concrete type.
    Pending,
    None,
    SumType(Vec<EType>),
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self {
            EType::Int(value) => value.clone().unwrap_or_else(|| "Integer".to_string()),
            EType::Bool(value) => value.clone().unwrap_or_else(|| "Bool".to_string()),
            EType::String(value) => value.clone().unwrap_or_else(|| "String".to_string()),
            EType::Char(value) => value.clone().unwrap_or_else(|| "Char".to_string()),
            EType::Pending => "pending".to_string(),
            EType::None => "None".to_string(),
            EType::SumType(sub_types) => sub_types
                .iter()
                .map(|sub_type| sub_type.to_string())
                .collect::<Vec<_>>()
                .join("|"),
        }
    }
}

/// The literal a leaf is pinned to, if any. `none` answers `None`: its value is
/// its type, so there is nothing pinned *to* it.
pub fn leaf_literal(t: &EType) -> Option<&str> {
    match t {
        EType::Int(value) | EType::Bool(value) | EType::String(value) | EType::Char(value) => {
            value.as_deref()
        }
        _ => Option::None,
    }
}

/// Convert the graph-level type descriptor into the evaluation-level type,
/// literal included.
///
/// The literal used to be dropped here and carried alongside as a string
/// (`anchor_literal`). It is kept now because a Match's output is the union of
/// what its branches produce, and a union that forgets which values it is made
/// of cannot tell `true` from `Bool`.
pub fn graph_type_to_eval_type(t: &crate::model::r#type::EType) -> EType {
    let value = crate::layout::value_of_etype(t);
    match t {
        crate::model::r#type::EType::Bool { .. } => EType::Bool(value),
        crate::model::r#type::EType::Int { .. } => EType::Int(value),
        crate::model::r#type::EType::Char { .. } => EType::Char(value),
        crate::model::r#type::EType::String { .. } => EType::String(value),
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

/// Leaves of `t` that each claim their own row: the four concrete value types
/// plus `none`. Sum types are expanded first; `Pending` claims no row of its
/// own.
///
/// This lives here rather than in the renderer because the row count is an
/// *addressing* fact — it decides how many cells an anchor occupies — which
/// the rendering merely follows.
pub fn row_leaves(t: &EType) -> Vec<EType> {
    flatten_type(t)
        .into_iter()
        .filter(|leaf| {
            matches!(
                leaf,
                EType::Bool(_) | EType::Char(_) | EType::Int(_) | EType::String(_) | EType::None
            )
        })
        .collect()
}

/// The vertical slice of one leaf row that a type claims, as fractions of that
/// row's height: `0.0` is the row's top edge, `1.0` its bottom edge.
///
/// A row is one band and a band is one type, so what a *pattern* takes out of
/// it is the share of that type's inhabitants it stands for. A base type names
/// all of them and takes the whole band. `Integer` has infinitely many, so `42`
/// takes infinitely little — nothing at all — and can only be drawn at an edge
/// rather than across a height. `Bool` has exactly two, so `true` and `false`
/// take half each and together leave no gap.
///
/// This is the arithmetic an exhaustiveness check is made of: the spans a
/// Match's arms claim of a row either cover it, leave a hole — an arm is
/// missing — or overlap, and then one of them is redundant. It lives here
/// rather than in the renderer for the reason `row_leaves` gives just above: it
/// is a fact about the type, which the drawing merely follows. A linter should
/// not have to reach through Bevy to ask it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowSpan {
    pub top: f32,
    pub bottom: f32,
}

impl RowSpan {
    /// The whole band: what a base type claims, and what `none` claims, whose
    /// one value *is* the type.
    pub const FULL: RowSpan = RowSpan {
        top: 0.0,
        bottom: 1.0,
    };
    /// What `true` claims of a Bool band. Bool is the only type small enough
    /// that naming one of its values is worth a share of the band rather than a
    /// line on it.
    pub const TOP_HALF: RowSpan = RowSpan {
        top: 0.0,
        bottom: 0.5,
    };
    /// What `false` claims. Together with `TOP_HALF` the band is covered
    /// exactly once, which is what makes a two-armed Bool match read as
    /// exhaustive without anything having to say so.
    pub const BOTTOM_HALF: RowSpan = RowSpan {
        top: 0.5,
        bottom: 1.0,
    };
    /// No height at all — a *place on the band* rather than a share of it. A
    /// literal of an unbounded type attaches here, at the row's top edge,
    /// because that is the edge a band is read from.
    pub const TOP_EDGE: RowSpan = RowSpan {
        top: 0.0,
        bottom: 0.0,
    };

    pub fn height(&self) -> f32 {
        self.bottom - self.top
    }

    /// True for a span that claims none of the row. Callers that turn a span
    /// into geometry have to know, because a shape of zero height is no shape.
    pub fn is_degenerate(&self) -> bool {
        self.height() <= f32::EPSILON
    }

    /// The part of the row this span leaves alone, or `None` where it leaves
    /// nothing.
    ///
    /// What a cast's target does not claim is what the cast can fail on, so
    /// this is the sad path stated as geometry: `FULL` leaves nothing and
    /// cannot fail, `TOP_HALF` leaves `BOTTOM_HALF` — casting `Bool` to `true`
    /// fails on exactly `false` — and `TOP_EDGE` leaves the whole band, because
    /// naming one value out of infinitely many is a promise that almost every
    /// value breaks.
    ///
    /// Only spans lying against an edge have a single-interval complement, and
    /// only those are ever built: `literal_row_span` makes nothing else. A span
    /// floating in the middle of a band would leave two pieces, and this
    /// answers `None` for it rather than picking one — better a missing strand
    /// than a wrong one, if the model ever grows such a span.
    pub fn complement(&self) -> Option<RowSpan> {
        let touches_top = self.top <= f32::EPSILON;
        let touches_bottom = self.bottom >= 1.0 - f32::EPSILON;
        if self.is_degenerate() {
            return Some(RowSpan::FULL);
        }
        match (touches_top, touches_bottom) {
            (true, true) => Option::None,
            (true, false) => Some(RowSpan {
                top: self.bottom,
                bottom: 1.0,
            }),
            (false, true) => Some(RowSpan {
                top: 0.0,
                bottom: self.top,
            }),
            (false, false) => Option::None,
        }
    }

    /// True when both spans claim some of the same slice.
    ///
    /// Nothing calls this yet: it is the question the exhaustiveness linter
    /// will ask of a Match's arms, and it is stated here, beside the spans it
    /// compares, rather than left to be re-derived somewhere that has no
    /// business knowing how a band is divided.
    ///
    /// Two degenerate spans at the same edge do *not* overlap: neither takes
    /// anything, so neither can take it from the other. `1` and `2` are
    /// distinct arms however close their lines are drawn, and a linter reading
    /// this must not call them redundant.
    #[allow(dead_code)]
    pub fn overlaps(&self, other: &RowSpan) -> bool {
        self.top.max(other.top) < self.bottom.min(other.bottom)
    }
}

/// The share of `leaf`'s band that a value of it claims. A `literal` of `None`
/// asks about the type itself, which is the whole band.
///
/// `literal` is graph-level text — what the user typed — so `Bool` is parsed
/// here rather than trusted. Text that is neither `true` nor `false` names no
/// inhabitant anything could be pointed at, so it claims nothing; that is
/// honest, and it is the only answer that does not panic on a half-typed word.
pub fn literal_row_span(leaf: &EType, literal: Option<&str>) -> RowSpan {
    let Some(literal) = literal else {
        return RowSpan::FULL;
    };
    match leaf {
        // Pinning `none`'s literal to `none` narrows nothing: the value and the
        // type are the same statement, so the row stays wholly claimed.
        EType::None => RowSpan::FULL,
        EType::Bool(_) => match literal {
            "true" => RowSpan::TOP_HALF,
            "false" => RowSpan::BOTTOM_HALF,
            _ => RowSpan::TOP_EDGE,
        },
        // Unbounded: one value out of infinitely many is 0% of the band.
        EType::Int(_) | EType::Char(_) | EType::String(_) => RowSpan::TOP_EDGE,
        // Neither claims a row of its own (`row_leaves`), so neither is ever
        // asked — answered anyway, because a total function is easier to trust
        // than one with a hole in it.
        EType::Pending | EType::SumType(_) => RowSpan::FULL,
    }
}

/// True when `wider` admits every value `narrower` admits.
///
/// This is the question `types_match` cannot answer. `types_match` compares
/// *shapes* and deliberately ignores literals, which is right for asking
/// whether two things are the same kind of thing. Subsumption asks the other
/// question — may this value travel here — and a literal is not the same kind
/// of statement as its base type: `Integer` admits `1`, and `1` does not admit
/// `Integer`.
///
/// Sums are handled per leaf and in no particular order, unlike `types_match`,
/// whose pairwise-in-order rule is about equality rather than admission. So
/// `Integer` subsumes `1|2` — which is what makes a cast from it total.
///
/// `Pending` subsumes nothing and is subsumed by nothing: an undecided type
/// makes no claim either way, and pretending otherwise would let an unfinished
/// graph look finished.
pub fn subsumes(wider: &EType, narrower: &EType) -> bool {
    let wide_leaves = flatten_type(wider);
    flatten_type(narrower).iter().all(|leaf| {
        wide_leaves
            .iter()
            .any(|wide| leaf_subsumes_leaf(wide, leaf))
    })
}

/// Subsumption between two single leaves. Same kind, and either the wider one
/// names no literal — the whole type, so every value of it — or both name the
/// same one.
fn leaf_subsumes_leaf(wider: &EType, narrower: &EType) -> bool {
    if matches!(wider, EType::Pending) || matches!(narrower, EType::Pending) {
        return false;
    }
    if !types_match(wider, narrower) {
        return false;
    }
    match leaf_literal(wider) {
        Option::None => true,
        Some(wide) => leaf_literal(narrower) == Some(wide),
    }
}

/// Collapse a set of leaves into the narrowest type admitting all of them.
///
/// Three rules, applied in order, and all three are the same rule seen from
/// different sides — *never widen a value set that was not asked to be
/// widened*:
///
/// 1. the same leaf twice is one leaf. Two arms yielding `true` yield `true`.
/// 2. a base type swallows the literals of its own kind. `1|Integer` is
///    `Integer`, because `Integer` already admits `1`.
/// 3. literals whose spans cover their band without a gap *are* the base type,
///    and are written as it. In practice only `Bool` can reach this — `true`
///    and `false` take half the band each — which is exactly right: a type with
///    two inhabitants can be exhausted by naming both, and one with infinitely
///    many never can.
///
/// Rule 3 is why this reads the spans rather than special-casing `Bool`. The
/// arithmetic already knows how many values a band holds; saying it twice would
/// be a second place to get it wrong.
pub fn normalize_leaves(leaves: Vec<EType>) -> Vec<EType> {
    // Rule 1.
    let mut out: Vec<EType> = Vec::new();
    for leaf in leaves {
        if !out.iter().any(|seen| leaf_subsumes_leaf(seen, &leaf)) {
            out.push(leaf);
        }
    }
    // Rule 3: a kind whose literals cover their band is that kind, whole. Done
    // before rule 2 so the base type it introduces does the swallowing there.
    let covered: Vec<EType> = out
        .iter()
        .filter(|leaf| leaf_literal(leaf).is_some())
        .filter(|leaf| {
            let spans: Vec<RowSpan> = out
                .iter()
                .filter(|other| types_match(other, leaf))
                .filter_map(|other| leaf_literal(other).map(|v| literal_row_span(other, Some(v))))
                .collect();
            spans_cover_band(&spans)
        })
        .map(|leaf| base_type_of(leaf))
        .collect();
    out.extend(covered);
    // Rule 2.
    let bases: Vec<EType> = out
        .iter()
        .filter(|leaf| leaf_literal(leaf).is_none())
        .cloned()
        .collect();
    out.retain(|leaf| {
        leaf_literal(leaf).is_none() || !bases.iter().any(|base| leaf_subsumes_leaf(base, leaf))
    });
    // Rule 1 again: rule 3 may have introduced the same base type once per
    // literal that asked for it.
    let mut deduped: Vec<EType> = Vec::new();
    for leaf in out {
        if !deduped.iter().any(|seen| leaf_subsumes_leaf(seen, &leaf)) {
            deduped.push(leaf);
        }
    }
    deduped
}

/// The same kind with no literal pinned to it — the whole type the literal was
/// one value of.
pub fn base_type_of(leaf: &EType) -> EType {
    match leaf {
        EType::Int(_) => EType::Int(Option::None),
        EType::Bool(_) => EType::Bool(Option::None),
        EType::String(_) => EType::String(Option::None),
        EType::Char(_) => EType::Char(Option::None),
        other => other.clone(),
    }
}

/// True when `spans` leave no part of the band `0.0..1.0` unclaimed.
///
/// Degenerate spans contribute nothing, which is the point: no number of
/// integer literals ever covers the Integer band, however many are named.
fn spans_cover_band(spans: &[RowSpan]) -> bool {
    let mut ordered: Vec<&RowSpan> = spans.iter().filter(|s| !s.is_degenerate()).collect();
    ordered.sort_by(|a, b| {
        a.top
            .partial_cmp(&b.top)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut reached = 0.0_f32;
    for span in ordered {
        if span.top > reached + f32::EPSILON {
            return false;
        }
        reached = reached.max(span.bottom);
    }
    reached >= 1.0 - f32::EPSILON
}

/// The share of the leaf row `row` that `claimant` takes, or `None` where the
/// two can never describe the same value.
///
/// Both directions matter and they are not symmetric. A claimant that admits
/// everything the row holds takes the row entire — an `Integer` arm against an
/// `Integer` row, but also an `Integer` arm against a `1` row, which has no
/// height to divide. A claimant *narrower* than the row takes only its share of
/// it, which is where `true` gets half a Bool band and `42` gets a line on an
/// Integer one.
pub fn claimed_span(row: &EType, claimant: &EType) -> Option<RowSpan> {
    if leaf_subsumes_leaf(claimant, row) {
        Some(RowSpan::FULL)
    } else if leaf_subsumes_leaf(row, claimant) {
        Some(literal_row_span(row, leaf_literal(claimant)))
    } else {
        Option::None
    }
}

/// How many cells an anchor occupies along Y: one per sum-type member, never
/// fewer than one.
///
/// An input the node does not constrain (Sink, Match, TypeCast) takes the
/// height of whatever is wired into it, so its band lines up with the source's
/// — which means connecting or disconnecting an edge changes the node's
/// footprint and has to be followed by a re-settle.
pub fn anchor_rows(
    graph: &crate::model::term_graph::TermGraph,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
) -> usize {
    let declared = anchor_type(graph, anchor_id, function_declarations);
    let rows = match declared {
        Some(t) => row_leaves(&t).len(),
        None => match graph.anchors.get(anchor_id) {
            Some(crate::model::anchor::EAnchor::Input(_)) => {
                incoming_anchor_type(graph, anchor_id, function_declarations)
                    .map(|t| row_leaves(&t).len())
                    .unwrap_or(0)
            }
            _ => 0,
        },
    };
    rows.max(1)
}

/// Type flowing into `input` from its connected source anchor, if any.
pub fn incoming_anchor_type(
    graph: &crate::model::term_graph::TermGraph,
    input: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
) -> Option<EType> {
    incoming_type(
        graph,
        input,
        function_declarations,
        &mut std::collections::HashSet::new(),
    )
}
/// Collect every Source in the graph as (node_id, name).
pub fn collect_sources(
    graph: &crate::model::term_graph::TermGraph,
) -> Vec<(crate::model::node::Id, String)> {
    let mut out: Vec<_> = graph
        .nodes
        .iter()
        .filter_map(|(id, node)| match node {
            crate::model::node::ENode::Source { name, .. } => Some((id.clone(), name.clone())),
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
    graph: &crate::model::term_graph::TermGraph,
    anchor: &crate::model::anchor::Id,
) -> Vec<crate::model::node::Id> {
    let mut out: Vec<crate::model::node::Id> = graph.get_connected_nodes_to_anchor(anchor.clone());
    if let Some(edges) = graph.edges.get(anchor) {
        for e in edges {
            if let Some(n) = graph.anchor_to_node.get(&e.to) {
                out.push(n.clone());
            }
        }
    }
    out
}

/// True if any Sink has at least one edge on its input anchor.
pub fn sink_has_input(graph: &crate::model::term_graph::TermGraph) -> bool {
    graph.nodes.values().any(|node| match node {
        crate::model::node::ENode::Sink { input_anchor } => {
            !neighbours_of_anchor(graph, input_anchor).is_empty()
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
/// `graph` must be the flattened graph (`LayoutGraph::flattened_graph`): every edge —
/// including those inside Pattern branches — lives in the program-level edge
/// table, so inference on a bare sub-graph would see no edges at all.
pub fn anchor_type(
    graph: &crate::model::term_graph::TermGraph,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
) -> Option<EType> {
    anchor_type_guarded(
        graph,
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
    graph: &crate::model::term_graph::TermGraph,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    if !visiting.insert(anchor_id.clone()) {
        return Some(EType::Pending);
    }
    let result = anchor_type_uncycled(graph, anchor_id, function_declarations, visiting);
    visiting.remove(anchor_id);
    result
}

fn anchor_type_uncycled(
    graph: &crate::model::term_graph::TermGraph,
    anchor_id: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    let node_id = graph.anchor_to_node.get(anchor_id)?;
    match graph.nodes.get(node_id)? {
        // Declared types are fixed: they never depend on what flows in.
        // A Constant *is* its value, so its literal narrows its type: a `42`
        // node has the type `42`, and a Match fed from it sees exactly that.
        crate::model::node::ENode::Constant {
            r#type,
            output_anchor,
        } => (anchor_id == output_anchor).then(|| graph_type_to_eval_type(r#type)),
        // A Source is the opposite case, and the literal on it must *not*
        // narrow anything. What a Source produces comes from the evaluation
        // prompt; a literal written on it is the shape of that answer, not the
        // answer. Letting it through would type the graph on a value that may
        // never arrive.
        crate::model::node::ENode::Source {
            r#type,
            output_anchor,
            ..
        } => (anchor_id == output_anchor).then(|| base_type_of(&graph_type_to_eval_type(r#type))),
        // A branch source hands the matched value into its branch, so it
        // carries its Pattern's declared type — the narrowing the Match
        // performs. It has no type of its own to declare.
        crate::model::node::ENode::BranchSource {
            pattern,
            output_anchor,
        } => {
            if anchor_id != output_anchor {
                return None;
            }
            match graph.nodes.get(pattern)? {
                // An arm that declares nothing yet types nothing: its source
                // carries `Pending`, which is what the branch downstream of it
                // reads and what the renderer draws grey.
                crate::model::node::ENode::Pattern { r#type, .. } => Some(
                    r#type
                        .as_ref()
                        .map(graph_type_to_eval_type)
                        .unwrap_or(EType::Pending),
                ),
                _ => None,
            }
        }
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
            // A parameter with no declared type constrains nothing, and
            // `None` already says exactly that here.
            declaration.inputs.get(index).and_then(|p| p.r#type.clone())
        }
        crate::model::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        } => (anchor_id == output_anchor).then(|| {
            type_cast_output_type(
                graph,
                r#type.as_ref(),
                input_anchor,
                function_declarations,
                visiting,
            )
        }),
        crate::model::node::ENode::Match {
            patterns,
            input_anchor,
            output_anchor,
        } => (anchor_id == output_anchor).then(|| {
            match_output_type(
                graph,
                patterns,
                input_anchor,
                function_declarations,
                visiting,
            )
        }),
        // Sink input takes anything; Pattern and Root have no anchors.
        crate::model::node::ENode::Sink { .. }
        | crate::model::node::ENode::Pattern { .. }
        | crate::model::node::ENode::Root {} => None,
    }
}

/// A cast to `target` is total when the incoming type already matches, and
/// partial otherwise — a partial cast can fail, which is modelled as
/// `Sum(target, none)`. Which of the two applies cannot be decided before an
/// incoming type is known, so an unconnected (or itself pending) input makes
/// the output `Pending`.
///
/// A cast with no target chosen yet is pending before the input is even looked
/// at: there is nothing to be total or partial *to*.
fn type_cast_output_type(
    graph: &crate::model::term_graph::TermGraph,
    target: Option<&crate::model::r#type::EType>,
    input_anchor: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> EType {
    let Some(target) = target else {
        return EType::Pending;
    };
    let target = graph_type_to_eval_type(target);
    match incoming_type(graph, input_anchor, function_declarations, visiting) {
        None | Some(EType::Pending) => EType::Pending,
        // Total exactly when the target already admits everything that arrives.
        // Asked as subsumption rather than as equality because widening is the
        // ordinary use of a cast: `1|2` cast to `Integer` cannot fail, and a
        // `none` hung off it would claim a sad path that does not exist.
        Some(incoming) if !subsumes(&target, &incoming) => {
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
    graph: &crate::model::term_graph::TermGraph,
    patterns: &[crate::model::node::Id],
    input_anchor: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> EType {
    if patterns.is_empty() {
        return EType::Pending;
    }
    match incoming_type(graph, input_anchor, function_declarations, visiting) {
        None | Some(EType::Pending) => return EType::Pending,
        Some(_) => {}
    }
    let mut leaves: Vec<EType> = Vec::new();
    for pattern_id in patterns {
        let branch = match branch_type(graph, pattern_id, function_declarations, visiting) {
            None | Some(EType::Pending) => return EType::Pending,
            Some(t) => t,
        };
        leaves.extend(flatten_type(&branch));
    }
    // The union is narrowed, not flattened: branches all yielding `true` yield
    // `true`, and `1` beside `2` stays `1|2`. Widening to the base type is
    // something the graph has to say out loud — with a TypeCast — or something
    // the values themselves prove by covering the band between them.
    let mut leaves = normalize_leaves(leaves);
    match leaves.len() {
        0 => EType::Pending,
        1 => leaves.remove(0),
        _ => EType::SumType(leaves),
    }
}

/// Type a Pattern's branch produces: whatever reaches the branch's own Sink.
fn branch_type(
    graph: &crate::model::term_graph::TermGraph,
    pattern_id: &crate::model::node::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    let crate::model::node::ENode::Pattern { sink_node_id, .. } = graph.nodes.get(pattern_id)?
    else {
        return None;
    };
    let crate::model::node::ENode::Sink { input_anchor } = graph.nodes.get(sink_node_id)? else {
        return None;
    };
    incoming_type(graph, input_anchor, function_declarations, visiting)
}

/// Type flowing into `input` from the connected source anchor. `None` when the
/// input is unconnected or the source carries no type.
fn incoming_type(
    graph: &crate::model::term_graph::TermGraph,
    input: &crate::model::anchor::Id,
    function_declarations: &FunctionDeclarations,
    visiting: &mut std::collections::HashSet<crate::model::anchor::Id>,
) -> Option<EType> {
    let source = source_anchor_for_input(graph, input)?;
    anchor_type_guarded(graph, &source, function_declarations, visiting)
}

/// Source anchor feeding into `input`, if connected. Drag-to-connect records
/// an edge from either end, so both directions are checked.
pub fn source_anchor_for_input(
    graph: &crate::model::term_graph::TermGraph,
    input: &crate::model::anchor::Id,
) -> Option<crate::model::anchor::Id> {
    for (from, edges) in &graph.edges {
        if edges.iter().any(|e| &e.to == input) {
            return Some(from.clone());
        }
    }
    graph
        .edges
        .get(input)
        .and_then(|edges| edges.first())
        .map(|e| e.to.clone())
}

/// Type a node produces, i.e. the type of its output anchor. `None` for nodes
/// that have no output at all (Sink, Root). Used for the selection display.
pub fn node_output_type(
    graph: &crate::model::term_graph::TermGraph,
    node_id: &crate::model::node::Id,
    function_declarations: &FunctionDeclarations,
) -> Option<EType> {
    let output_anchor =
        graph
            .nodes
            .get(node_id)?
            .anchors()
            .into_iter()
            .find_map(|(id, anchor)| {
                matches!(anchor, crate::model::anchor::EAnchor::Output).then_some(id)
            })?;
    anchor_type(graph, &output_anchor, function_declarations)
}

/// graph-level literal an anchor's type is pinned to, if any.
///
/// A BranchSource borrows its Pattern's whole declaration, literal included:
/// inside the branch the matched value is known to be exactly that literal, so
/// the source shows it rather than the bare type.
///
/// Lives here rather than on `LayoutGraph` because resolving a BranchSource
/// means reaching its Pattern, which sits in the *parent* scope — only the
/// flattened graph has both.
pub fn anchor_literal(
    graph: &crate::model::term_graph::TermGraph,
    anchor_id: &crate::model::anchor::Id,
) -> Option<String> {
    let node_id = graph.anchor_to_node.get(anchor_id)?;
    match graph.nodes.get(node_id)? {
        crate::model::node::ENode::Constant {
            r#type,
            output_anchor,
        }
        | crate::model::node::ENode::Source {
            r#type,
            output_anchor,
            ..
        } => (anchor_id == output_anchor)
            .then(|| crate::layout::value_of_etype(r#type))
            .flatten(),
        crate::model::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        } => (anchor_id == input_anchor || anchor_id == output_anchor)
            .then(|| r#type.as_ref().and_then(crate::layout::value_of_etype))
            .flatten(),
        crate::model::node::ENode::BranchSource {
            pattern,
            output_anchor,
        } => {
            if anchor_id != output_anchor {
                return None;
            }
            match graph.nodes.get(pattern)? {
                crate::model::node::ENode::Pattern { r#type, .. } => {
                    r#type.as_ref().and_then(crate::layout::value_of_etype)
                }
                _ => None,
            }
        }
        // FunctionCall anchors bind to declaration types, which carry no
        // graph-level literal; Match, Pattern, Sink and Root carry no
        // anchored type at all.
        _ => None,
    }
}
/// graph-level literal of whatever feeds `input`, if any.
///
/// `anchor_literal` answers about the anchor it is handed, and an input that
/// constrains nothing — a Sink's, a Match's — owns no type to pin a literal to,
/// so it answers `None` there however concrete the arriving value is. What
/// arrives is a literal or it is not, though, and an anchor that draws what
/// arrives has to know which. So the question is forwarded one hop upstream, to
/// the anchor that does own it.
pub fn incoming_anchor_literal(
    graph: &crate::model::term_graph::TermGraph,
    input: &crate::model::anchor::Id,
) -> Option<String> {
    source_anchor_for_input(graph, input).and_then(|source| anchor_literal(graph, &source))
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
        | (EType::Pending, EType::Pending) => true,
        (EType::SumType(x), EType::SumType(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| types_match(p, q))
        }
        _ => false,
    }
}
