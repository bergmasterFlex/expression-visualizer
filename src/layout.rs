use bevy::prelude::*;

type NodeIdDomain = crate::common::IdDomain<crate::model::node::Id>;
type AnchorIdDomain = crate::common::IdDomain<crate::model::anchor::Id>;
type FunctionDeclarations = std::collections::HashMap<
    crate::model::function_declaration::FunctionDeclarationId,
    crate::model::function_declaration::FunctionDeclaration,
>;

/// What a single cell of a node stands for.
///
/// The whole point of the per-part cell layout is that an address names at
/// most one thing: a specific anchor row, or the node's own property. That is
/// what lets the caret alone decide what an edit acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellRole {
    /// Row `leaf` of input anchor number `index`.
    Input { index: usize, leaf: usize },
    /// Row `leaf` of the output anchor.
    Output { leaf: usize },
    /// The node's own property: the type it declares, or the function it calls.
    Body,
}

/// Node-local cell layout: every cell a node claims, and what each one means.
///
/// Built by `LayoutAst::with_shapes` from the node kind plus its anchor types,
/// and cached on the `LayoutNode`. It never depends on positions, only on
/// types and wiring — which is why it stays valid across a whole
/// `settle_footprints` pass.
#[derive(Debug, Clone)]
pub struct NodeShape {
    cells: Vec<(IVec3, CellRole)>,
}

impl NodeShape {
    pub fn new(cells: Vec<(IVec3, CellRole)>) -> Self {
        Self { cells }
    }

    /// Placeholder for a node whose shape has not been computed yet: a single
    /// body cell at the origin. Every edit path ends in `with_shapes`, so this
    /// only ever survives between a builder and the next re-settle.
    pub fn placeholder() -> Self {
        Self::new(vec![(IVec3::ZERO, CellRole::Body)])
    }

    pub fn cells(&self) -> &[(IVec3, CellRole)] {
        &self.cells
    }

    /// Node-local bounding box. Never empty — a shape always has at least one
    /// cell.
    pub fn bbox(&self) -> AABB {
        self.cells
            .iter()
            .fold(None, |acc, (c, _)| {
                AABB::union_opt(acc, Some(AABB::point(*c)))
            })
            .unwrap_or(AABB::point(IVec3::ZERO))
    }

    /// What the node-local cell `local` stands for, if it belongs to the node.
    /// This is the lookup an edit acts on.
    pub fn role_at(&self, local: IVec3) -> Option<&CellRole> {
        self.cells
            .iter()
            .find(|(c, _)| *c == local)
            .map(|(_, role)| role)
    }
}

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub node_id: crate::model::node::Id,
    pub pos: Vec3,
    /// Cell layout, refreshed by `LayoutAst::with_shapes` whenever types or
    /// wiring change. Independent of `pos`, so it survives the settle pass.
    pub shape: NodeShape,
}

impl LayoutNode {
    /// A node at `pos` whose shape is not known yet (see
    /// `NodeShape::placeholder`).
    pub fn unshaped(node_id: crate::model::node::Id, pos: Vec3) -> Self {
        Self {
            node_id,
            pos,
            shape: NodeShape::placeholder(),
        }
    }
}

/// Match-local Z of the Pattern row. The Match's own input anchor owns local
/// Z=0, so the arms start one cell behind it and their branch volumes one
/// further still (see `LayoutAst::sub_layout_origin`).
pub const PATTERN_LOCAL_Z: f32 = 1.0;
/// Cells an anchor claims: `infer::anchor_rows` of them, growing along +Y from
/// row 0, all in column `x` at depth `z`.
fn anchor_cells(
    flat_ast: &crate::model::ast::Ast,
    function_declarations: &FunctionDeclarations,
    anchor: &crate::model::anchor::Id,
    x: i32,
    z: i32,
    role: impl Fn(usize) -> CellRole,
) -> Vec<(IVec3, CellRole)> {
    (0..crate::infer::anchor_rows(flat_ast, anchor, function_declarations))
        .map(|leaf| (IVec3::new(x, leaf as i32, z), role(leaf)))
        .collect()
}
#[derive(Debug, Clone)]
pub struct LayoutEdge {
    pub from_anchor: LayoutAnchor,
    pub to_anchor: LayoutAnchor,
}

#[derive(Debug, Clone)]
pub struct LayoutAnchor {
    pub anchor_id: crate::model::anchor::Id,
    pub node_id: crate::model::node::Id,
    pub anchor: crate::model::anchor::EAnchor,
    pub pos: Vec3,
}

/// A single entry produced by `LayoutAst::walk_all`. Groups a LayoutNode with
/// enough context (its owning LayoutAst for anchor lookups and the accumulated
/// grid-space offset) so the render layer can place pattern sub-AST nodes
/// correctly without re-doing the traversal.
pub struct WalkedNode<'a> {
    pub layout_ast: &'a LayoutAst,
    pub layout_node: &'a LayoutNode,
    pub extra_offset: Vec3,
}

/// One entry per LayoutAst reached from a root. Used by the per-AST grid
/// renderer to place a dedicated grid mesh per Program/Pattern sub-AST.
pub struct WalkedAst<'a> {
    pub layout_ast: &'a LayoutAst,
    /// Owner path from the walk root to this AST. Empty at the outermost
    /// LayoutAst (the one `walk_all_asts` was called on).
    pub context: Vec<crate::model::node::Id>,
    /// Accumulated grid-space offset from the root to this AST's origin.
    pub extra_offset: Vec3,
}

/// Bounds of an AST grid in that AST's local grid coordinates. Both corners
/// are inclusive. X/Z size the drawn grid plane and X/Y its Z faces; the Y
/// span is the rows the scope owns, which is also what bounds a caret
/// address to its scope.
#[derive(Debug, Clone, Copy)]
pub struct AstGridBounds {
    pub min: IVec3,
    pub max: IVec3,
}

/// Inclusive 3D bounding box in grid cells. Used for match footprint math.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: IVec3,
    pub max: IVec3,
}

/// Layout space is the non-negative octant. Constructing a box that reaches
/// below the origin means some placement or displacement rule leaked a
/// negative address; fail loudly in debug builds rather than propagating it
/// silently into grid bounds, picking and the caret.
fn debug_assert_non_negative(min: IVec3) {
    debug_assert!(
        min.x >= 0 && min.y >= 0 && min.z >= 0,
        "layout AABB reaches into negative space: min={:?}",
        min
    );
}

impl AABB {
    pub fn point(p: IVec3) -> Self {
        debug_assert_non_negative(p);
        Self { min: p, max: p }
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: IVec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: IVec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn union_opt(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        match (a, b) {
            (Some(a), Some(b)) => Some(a.union(b)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    pub fn translated(self, delta: IVec3) -> Self {
        debug_assert_non_negative(self.min + delta);
        Self {
            min: self.min + delta,
            max: self.max + delta,
        }
    }

    pub fn contains(&self, p: IVec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    pub fn cells(&self) -> impl Iterator<Item = IVec3> + '_ {
        let (min, max) = (self.min, self.max);
        (min.z..=max.z).flat_map(move |z| {
            (min.y..=max.y).flat_map(move |y| (min.x..=max.x).map(move |x| IVec3::new(x, y, z)))
        })
    }
}

#[derive(Clone)]
pub struct LayoutAst {
    pub ast: crate::model::ast::Ast,
    pub layout_nodes: std::collections::HashMap<crate::model::node::Id, LayoutNode>,
    /// Per-owner nested layouts. Keyed by the container node id (Program in
    /// Step 1; Pattern in later steps). The `sub_layouts[program_id]` LayoutAst
    /// is the source of truth for the top-level context — the ENode::Program's
    /// own `ast` field is unused in Step 1 and stays empty.
    pub sub_layouts: std::collections::HashMap<crate::model::node::Id, LayoutAst>,
}

impl LayoutAst {
    /// A LayoutAst whose AST already holds its terminating `Sink` (via
    /// `Ast::new`), with a matching LayoutNode placed at the sink's default
    /// grid position. Replaces the former `empty()` + `plus_sink()` pairing.
    pub fn new(
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let (ast, node_id_domain, anchor_id_domain) =
            crate::model::ast::Ast::new(node_id_domain, anchor_id_domain);
        let sink_node_id = ast.sink_node_id.clone();
        let layout = Self {
            ast,
            layout_nodes: std::collections::HashMap::new(),
            sub_layouts: std::collections::HashMap::new(),
        }
        ._plus_layout_node(&sink_node_id, Vec3::new(0.0, 0.0, 4.0));
        (layout, node_id_domain, anchor_id_domain)
    }

    /// Build a root LayoutAst that holds a single Program node plus a
    /// `sub_layouts` entry keyed by the Program's id. The inner LayoutAst is
    /// where user-visible nodes live, and starts with its own terminating sink.
    pub fn empty_with_program() -> (Self, crate::model::node::Id, NodeIdDomain, AnchorIdDomain) {
        let node_id_domain = NodeIdDomain::new();
        let anchor_id_domain = AnchorIdDomain::new();
        let (node_id_domain, program_id) = node_id_domain.next_id();
        // The outer wrapper only carries the Program node; the sink `Ast::new`
        // mandates goes unused here (this AST is never rendered or evaluated —
        // rendering and eval both operate on `sub_layouts[program_id]`).
        let (outer_ast, node_id_domain, anchor_id_domain) =
            crate::model::ast::Ast::new(node_id_domain, anchor_id_domain);
        let outer_ast =
            outer_ast.plus_node(program_id.clone(), crate::model::node::ENode::Program {});
        let (program_sub, node_id_domain, anchor_id_domain) =
            Self::new(node_id_domain, anchor_id_domain);
        let outer = Self {
            ast: outer_ast,
            layout_nodes: std::collections::HashMap::new(),
            sub_layouts: std::collections::HashMap::from([(program_id.clone(), program_sub)]),
        };
        (outer, program_id, node_id_domain, anchor_id_domain)
    }

    pub fn minus_node(&self, node_id: &crate::model::node::Id) -> Self {
        match self.ast.nodes.get(node_id) {
            Some(crate::model::node::ENode::Pattern { parent_match, .. }) => {
                let parent_id = parent_match.clone();
                let remaining: Vec<crate::model::node::Id> = match self.ast.nodes.get(&parent_id) {
                    Some(crate::model::node::ENode::Match { patterns, .. }) => {
                        patterns.iter().filter(|p| *p != node_id).cloned().collect()
                    }
                    _ => vec![],
                };
                let after_pattern = Self {
                    ast: self.ast.minus_node(node_id),
                    layout_nodes: self
                        .layout_nodes
                        .clone()
                        .into_iter()
                        .filter(|(id, _)| id != node_id)
                        .collect(),
                    sub_layouts: self
                        .sub_layouts
                        .clone()
                        .into_iter()
                        .filter(|(id, _)| id != node_id)
                        .collect(),
                };
                if remaining.is_empty() {
                    Self {
                        ast: after_pattern.ast.minus_node(&parent_id),
                        layout_nodes: after_pattern
                            .layout_nodes
                            .into_iter()
                            .filter(|(id, _)| id != &parent_id)
                            .collect(),
                        sub_layouts: after_pattern.sub_layouts,
                    }
                } else {
                    after_pattern
                        ._with_match_patterns(&parent_id, remaining)
                        .recompute_match_pos(&parent_id)
                }
            }
            Some(crate::model::node::ENode::Match { patterns, .. }) => {
                let child_ids: Vec<_> = patterns.clone();
                let after_children = child_ids.iter().fold(
                    Self {
                        ast: self.ast.clone(),
                        layout_nodes: self.layout_nodes.clone(),
                        sub_layouts: self.sub_layouts.clone(),
                    },
                    |acc, pid| Self {
                        ast: acc.ast.minus_node(pid),
                        layout_nodes: acc
                            .layout_nodes
                            .into_iter()
                            .filter(|(id, _)| id != pid)
                            .collect(),
                        sub_layouts: acc
                            .sub_layouts
                            .into_iter()
                            .filter(|(id, _)| id != pid)
                            .collect(),
                    },
                );
                Self {
                    ast: after_children.ast.minus_node(node_id),
                    layout_nodes: after_children
                        .layout_nodes
                        .into_iter()
                        .filter(|(id, _)| id != node_id)
                        .collect(),
                    sub_layouts: after_children.sub_layouts,
                }
            }
            _ => Self {
                ast: self.ast.minus_node(node_id),
                layout_nodes: self
                    .layout_nodes
                    .clone()
                    .into_iter()
                    .filter(|(id, _)| id != node_id)
                    .collect(),
                sub_layouts: self.sub_layouts.clone(),
            },
        }
    }

    /// Returns the node whose footprint contains `pos`, if any.
    /// `Match` containers are excluded — only their `Pattern` children are
    /// selectable, so a click on the envelope never picks the container.
    /// Multi-cell nodes (matches implicitly, function calls with the
    /// minimum-extent rule) are selectable from any cell inside their
    /// `node_footprint`.
    pub fn node_at(&self, pos: IVec3) -> Option<crate::model::node::Id> {
        self.layout_nodes.keys().find_map(|id| {
            if matches!(
                self.ast.nodes.get(id),
                Some(crate::model::node::ENode::Match { .. })
            ) {
                return None;
            }
            let bbox = self.node_footprint(id)?;
            if bbox.contains(pos) {
                Some(id.clone())
            } else {
                None
            }
        })
    }

    /// AABB (in this LayoutAst's local grid coords) that a node claims: the
    /// bounding box of its `NodeShape`, translated to its position.
    ///
    /// A Match delegates to `match_footprint` instead, because its extent also
    /// covers its Patterns and their branch volumes — separate nodes and
    /// sub-layouts that its own shape knows nothing about.
    pub fn node_footprint(&self, id: &crate::model::node::Id) -> Option<AABB> {
        if matches!(
            self.ast.nodes.get(id),
            Some(crate::model::node::ENode::Match { .. })
        ) {
            return self.match_footprint(id);
        }
        let ln = self.layout_nodes.get(id)?;
        Some(ln.shape.bbox().translated(ln.pos.round().as_ivec3()))
    }

    /// Build a `grid position -> node id` lookup over all selectable nodes.
    /// Every node claims every cell of its `node_footprint`. When both a
    /// Pattern and its owning Match cover the same cell (Pattern rows of the
    /// match column) the Pattern id wins, so intra-stack Y-swap still
    /// resolves the Pattern as target — matches are inserted first and
    /// non-matches overwrite them.
    fn occupancy_map(&self) -> std::collections::HashMap<IVec3, crate::model::node::Id> {
        let mut map: std::collections::HashMap<IVec3, crate::model::node::Id> =
            std::collections::HashMap::new();
        for id in self.layout_nodes.keys() {
            if !self.is_match(id) {
                continue;
            }
            let Some(bbox) = self.node_footprint(id) else {
                continue;
            };
            for cell in bbox.cells() {
                map.insert(cell, id.clone());
            }
        }
        for id in self.layout_nodes.keys() {
            if self.is_match(id) {
                continue;
            }
            let Some(bbox) = self.node_footprint(id) else {
                continue;
            };
            for cell in bbox.cells() {
                map.insert(cell, id.clone());
            }
        }
        map
    }

    fn is_pattern(&self, id: &crate::model::node::Id) -> bool {
        matches!(
            self.ast.nodes.get(id),
            Some(crate::model::node::ENode::Pattern { .. })
        )
    }

    fn parent_match_of(&self, id: &crate::model::node::Id) -> Option<crate::model::node::Id> {
        match self.ast.nodes.get(id) {
            Some(crate::model::node::ENode::Pattern { parent_match, .. }) => {
                Some(parent_match.clone())
            }
            _ => None,
        }
    }

    /// Return the sibling Pattern ids of a `Match`.
    fn match_pattern_ids(&self, match_id: &crate::model::node::Id) -> Vec<crate::model::node::Id> {
        match self.ast.nodes.get(match_id) {
            Some(crate::model::node::ENode::Match { patterns, .. }) => patterns.clone(),
            _ => vec![],
        }
    }

    fn is_match(&self, id: &crate::model::node::Id) -> bool {
        matches!(
            self.ast.nodes.get(id),
            Some(crate::model::node::ENode::Match { .. })
        )
    }

    /// Recompute every node's `NodeShape` from the current types and wiring.
    ///
    /// `flat_ast` must be the flattened AST — anchor heights depend on edges,
    /// and only the program-level table holds them. Sub-layouts are shaped
    /// first, because a Match places its output behind its deepest branch and
    /// needs those branches measured already.
    ///
    /// Must run before `settle_footprints`, never during it: shapes decide
    /// footprints, and footprints decide displacement.
    pub fn with_shapes(
        &self,
        flat_ast: &crate::model::ast::Ast,
        function_declarations: &FunctionDeclarations,
    ) -> Self {
        let staged = Self {
            ast: self.ast.clone(),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self
                .sub_layouts
                .iter()
                .map(|(k, v)| (k.clone(), v.with_shapes(flat_ast, function_declarations)))
                .collect(),
        };
        let layout_nodes = staged
            .layout_nodes
            .iter()
            .map(|(id, ln)| {
                let shape = flat_ast
                    .nodes
                    .get(id)
                    .map(|node| staged.node_shape(node, flat_ast, function_declarations))
                    .unwrap_or_else(NodeShape::placeholder);
                (
                    id.clone(),
                    LayoutNode {
                        node_id: ln.node_id.clone(),
                        pos: ln.pos,
                        shape,
                    },
                )
            })
            .collect();
        Self {
            ast: staged.ast,
            layout_nodes,
            sub_layouts: staged.sub_layouts,
        }
    }
    /// Cell layout of one node, in node-local coordinates (`x`, `y`, `z`).
    ///
    /// Every part gets its own cell so that an address names one property:
    ///
    /// | node | cells (`x|z`) |
    /// |---|---|
    /// | Sink | `0\|0` input |
    /// | VarDecl / ConstDecl | `0\|0` body, `0\|1` output |
    /// | TypeCast | `0\|0` input, `0\|1` body, `0\|2` output |
    /// | FunctionCall (n inputs) | `i\|0` input i, `(0..n)\|1..2` body, `0\|3` output |
    /// | Match | `0\|0` input, output directly behind the deepest branch |
    /// | Pattern | one cell |
    /// | BranchSource | `0\|0` output |
    ///
    /// An anchor claims `infer::anchor_rows` cells along +Y from its own row.
    /// A Match's Patterns and branch volumes are separate nodes and
    /// sub-layouts, so they are not part of its shape — `match_footprint`
    /// unions those in.
    fn node_shape(
        &self,
        node: &crate::model::node::ENode,
        flat_ast: &crate::model::ast::Ast,
        fds: &FunctionDeclarations,
    ) -> NodeShape {
        let mut cells: Vec<(IVec3, CellRole)> = Vec::new();
        match node {
            crate::model::node::ENode::Sink { input_anchor } => {
                cells.extend(anchor_cells(flat_ast, fds, input_anchor, 0, 0, |leaf| {
                    CellRole::Input { index: 0, leaf }
                }));
            }
            crate::model::node::ENode::VarDecl { output_anchor, .. }
            | crate::model::node::ENode::ConstDecl { output_anchor, .. } => {
                cells.push((IVec3::ZERO, CellRole::Body));
                cells.extend(anchor_cells(flat_ast, fds, output_anchor, 0, 1, |leaf| {
                    CellRole::Output { leaf }
                }));
            }
            // Mirror of the Sink: one anchor and nothing else. It declares no
            // type of its own — that comes from its Pattern — so no body cell.
            crate::model::node::ENode::BranchSource { output_anchor, .. } => {
                cells.extend(anchor_cells(flat_ast, fds, output_anchor, 0, 0, |leaf| {
                    CellRole::Output { leaf }
                }));
            }
            crate::model::node::ENode::TypeCast {
                input_anchor,
                output_anchor,
                ..
            } => {
                cells.extend(anchor_cells(flat_ast, fds, input_anchor, 0, 0, |leaf| {
                    CellRole::Input { index: 0, leaf }
                }));
                cells.push((IVec3::new(0, 0, 1), CellRole::Body));
                cells.extend(anchor_cells(flat_ast, fds, output_anchor, 0, 2, |leaf| {
                    CellRole::Output { leaf }
                }));
            }
            crate::model::node::ENode::FunctionCall {
                input_anchors,
                output_anchor,
                ..
            } => {
                for (index, anchor) in input_anchors.iter().enumerate() {
                    cells.extend(anchor_cells(
                        flat_ast,
                        fds,
                        anchor,
                        index as i32,
                        0,
                        move |leaf| CellRole::Input { index, leaf },
                    ));
                }
                // Body spans the full input width, two cells deep — it carries
                // the name of the referenced function.
                for x in 0..input_anchors.len().max(1) as i32 {
                    for z in 1..=2 {
                        cells.push((IVec3::new(x, 0, z), CellRole::Body));
                    }
                }
                // Output stays in column 0 so its address does not move when
                // the call is swapped for a function of different arity.
                cells.extend(anchor_cells(flat_ast, fds, output_anchor, 0, 3, |leaf| {
                    CellRole::Output { leaf }
                }));
            }
            crate::model::node::ENode::Match {
                input_anchor,
                output_anchor,
                patterns,
            } => {
                cells.extend(anchor_cells(flat_ast, fds, input_anchor, 0, 0, |leaf| {
                    CellRole::Input { index: 0, leaf }
                }));
                cells.extend(anchor_cells(
                    flat_ast,
                    fds,
                    output_anchor,
                    0,
                    self.match_output_z(patterns),
                    |leaf| CellRole::Output { leaf },
                ));
            }
            // A Pattern owns no anchor: one cell declaring the arm's type.
            // Program is never laid out.
            crate::model::node::ENode::Pattern { .. } | crate::model::node::ENode::Program {} => {
                cells.push((IVec3::ZERO, CellRole::Body));
            }
        }
        NodeShape::new(cells)
    }

    /// Match-local Z of the output anchor, one empty cell behind the deepest
    /// branch.
    ///
    /// Patterns sit at match-local Z=1 and their branch volumes start at Z=2,
    /// so a branch reaching branch-local Z=d ends at match-local `2 + d` — its
    /// Sink. The gap keeps that Sink and the Match's own output from butting
    /// up against each other.
    pub fn match_output_z(&self, patterns: &[crate::model::node::Id]) -> i32 {
        let deepest_sink = patterns
            .iter()
            .filter_map(|pid| self.sub_layouts.get(pid))
            .filter_map(|sub| sub.inner_footprint())
            .map(|b| b.max.z)
            .max()
            .unwrap_or(0);
        2 + deepest_sink + 2
    }

    /// Union of all visible grid cells in this LayoutAst's local coords.
    /// Match nodes contribute their `match_footprint` (recursive over
    /// nested sub-ASTs); every other node contributes its rounded cell.
    /// Sub-layouts contribute their own inner_footprint offset by the owner
    /// node's grid position.
    pub fn inner_footprint(&self) -> Option<AABB> {
        let mut bbox: Option<AABB> = None;
        for (id, ln) in &self.layout_nodes {
            let node_bbox = if self.is_match(id) {
                self.match_footprint(id)
            } else {
                Some(ln.shape.bbox().translated(ln.pos.round().as_ivec3()))
            };
            bbox = AABB::union_opt(bbox, node_bbox);
        }
        for (owner_id, sub) in &self.sub_layouts {
            let owner_pos = self.sub_layout_origin(owner_id).round().as_ivec3();
            let sub_bbox = sub.inner_footprint().map(|b| b.translated(owner_pos));
            bbox = AABB::union_opt(bbox, sub_bbox);
        }
        bbox
    }

    /// How many Y rows a Pattern's branch occupies. A branch holding a nested
    /// Match is as tall as that Match's own arm stack, which is what lets
    /// growth propagate outward.
    fn branch_row_height(&self, pattern_id: &crate::model::node::Id) -> i32 {
        self.sub_layouts
            .get(pattern_id)
            .and_then(|s| s.inner_footprint())
            .map(|b| (b.max.y - b.min.y + 1).max(1))
            .unwrap_or(1)
    }

    /// Re-space a Match's Patterns so each arm clears the branch above it.
    ///
    /// Patterns are only ever *inserted* one row apart, so a branch that grew
    /// — because something nested inside it grew — would overlap its lower
    /// sibling. Re-spacing packs the arms by their actual heights instead.
    ///
    /// `settle_footprints` runs this bottom-up: an inner Match re-spaces its
    /// own arms first, which grows its branch's `inner_footprint`, which grows
    /// the enclosing arm, which re-spaces the outer Match. That chain is what
    /// makes growth cascade out of arbitrarily deep nesting.
    ///
    /// The lowest Pattern keeps its row, so the Match's origin stays put and
    /// the stack only ever grows in +Y.
    fn respace_match_patterns(&self, match_id: &crate::model::node::Id) -> Self {
        let mut ordered: Vec<(crate::model::node::Id, f32)> = self
            .match_pattern_ids(match_id)
            .into_iter()
            .filter_map(|pid| self.layout_nodes.get(&pid).map(|ln| (pid, ln.pos.y)))
            .collect();
        if ordered.is_empty() {
            return self.clone_shape();
        }
        ordered.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut layout_nodes = self.layout_nodes.clone();
        let mut row = ordered[0].1;
        for (pid, _) in &ordered {
            if let Some(ln) = layout_nodes.get_mut(pid) {
                ln.pos.y = row;
            }
            row += self.branch_row_height(pid) as f32;
        }
        Self {
            ast: self.ast.clone(),
            layout_nodes,
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    /// Grid-space AABB (in this LayoutAst's local coords) that a Match
    /// container claims. Aggregates all Pattern arms:
    ///   - Y: `sum over patterns of max(1, subast_y_extent)` stacked at the
    ///     Pattern's own y (patterns are assumed to occupy consecutive rows).
    ///   - X: union of every Pattern's sub-AST X extent, translated by the
    ///     Pattern's outer position.
    ///   - Z: from the Pattern's own Z (the parent-facing side) out to its
    ///     branch: the branch origin sits one cell behind the Pattern, so the
    ///     sub-AST Z range is offset by 1, plus one further cell of padding.
    /// Recursion via `inner_footprint` — inner matches inflate their host
    /// Pattern's sub-AST bbox and thus the outer footprint too.
    pub fn match_footprint(&self, match_id: &crate::model::node::Id) -> Option<AABB> {
        if !self.is_match(match_id) {
            return None;
        }
        let pattern_ids = self.match_pattern_ids(match_id);
        // The Match's own cells (input anchor, and its output behind the
        // branches) come from its shape; the arms are unioned in below.
        let mut bbox: Option<AABB> = self
            .layout_nodes
            .get(match_id)
            .map(|ln| ln.shape.bbox().translated(ln.pos.round().as_ivec3()));
        for pid in &pattern_ids {
            let Some(p_ln) = self.layout_nodes.get(pid) else {
                continue;
            };
            let p_pos = p_ln.pos.round().as_ivec3();
            let sub = self.sub_layouts.get(pid);
            let sub_bbox = sub.and_then(|s| s.inner_footprint());
            let arm_y = self.branch_row_height(pid);
            let (x_min, x_max, z_max) = match sub_bbox {
                Some(b) => (b.min.x, b.max.x, b.max.z),
                None => (0, 0, 0),
            };
            let arm = AABB {
                min: IVec3::new(p_pos.x + x_min, p_pos.y, p_pos.z),
                max: IVec3::new(p_pos.x + x_max, p_pos.y + arm_y - 1, p_pos.z + z_max + 2),
            };
            bbox = AABB::union_opt(bbox, Some(arm));
        }
        bbox
    }

    /// Move `node_id` by `delta_pos`, applying the swap constraint: no two
    /// nodes may share a grid position. Existing nodes on the target are
    /// displaced (mirror delta), cascading recursively through matches.
    /// Returns the updated layout and the effective grid position of the
    /// primary node (which differs from `origin + delta` when the move
    /// jumped over a match).
    pub fn move_node_delta(
        &self,
        node_id: crate::model::node::Id,
        delta_pos: Vec3,
    ) -> (Self, IVec3) {
        let Some(primary_ln) = self.layout_nodes.get(&node_id) else {
            return (self.clone_shape(), IVec3::ZERO);
        };
        let primary_origin = primary_ln.pos;
        // VarDecls are pinned to the source row (Y=0, Z=0) and may only be
        // reordered along X. A BranchSource is pinned outright: it must stay
        // at branch-local (0,0,0). Every other node type has to stay beyond
        // the source row (Z >= 1). Layout space is non-negative, so X and Y
        // are clamped at 0 as well.
        let delta_pos = match self.ast.nodes.get(&node_id) {
            Some(crate::model::node::ENode::VarDecl { .. }) => Vec3::new(delta_pos.x, 0.0, 0.0),
            Some(crate::model::node::ENode::BranchSource { .. }) => Vec3::ZERO,
            _ => {
                // Z=0 is the source row and the sink's Z is the sink's alone,
                // so everything else lives strictly between them.
                let target_z = (primary_origin.z + delta_pos.z).round() as i32;
                let sink_z = self.sink_z();
                if target_z <= 0 || sink_z.is_some_and(|s| target_z >= s) {
                    Vec3::new(delta_pos.x, delta_pos.y, 0.0)
                } else {
                    delta_pos
                }
            }
        };
        let clamp_axis = |origin: f32, delta: f32| {
            if (origin + delta).round() as i32 >= 0 {
                delta
            } else {
                0.0
            }
        };
        let delta_pos = Vec3::new(
            clamp_axis(primary_origin.x, delta_pos.x),
            clamp_axis(primary_origin.y, delta_pos.y),
            delta_pos.z,
        );
        let occupancy = self.occupancy_map();

        let primary_delta = self.jump_delta(&occupancy, &node_id, primary_origin, delta_pos);

        let mut plan: std::collections::HashMap<crate::model::node::Id, Vec3> =
            std::collections::HashMap::new();
        let mut worklist: std::collections::VecDeque<crate::model::node::Id> =
            std::collections::VecDeque::new();
        for (id, d) in self.move_group(&node_id, primary_delta) {
            let origin = self
                .layout_nodes
                .get(&id)
                .map(|ln| ln.pos)
                .unwrap_or(Vec3::ZERO);
            plan.insert(id.clone(), origin + d);
            worklist.push_back(id);
        }

        let mut iterations = 0usize;
        while let Some(cur) = worklist.pop_front() {
            iterations += 1;
            if iterations > 128 {
                warn!("move_node_delta: aborted after 128 iterations");
                return (self.clone_shape(), primary_origin.round().as_ivec3());
            }
            let Some(new_pos) = plan.get(&cur).copied() else {
                continue;
            };
            let key = new_pos.round().as_ivec3();
            let Some(occ) = occupancy.get(&key) else {
                continue;
            };
            if *occ == cur || plan.contains_key(occ) {
                continue;
            }
            let occ_origin = self
                .layout_nodes
                .get(occ)
                .map(|ln| ln.pos)
                .unwrap_or(Vec3::ZERO);
            let cur_origin = self
                .layout_nodes
                .get(&cur)
                .map(|ln| ln.pos)
                .unwrap_or(Vec3::ZERO);
            let cur_delta = new_pos - cur_origin;
            // Section-swap for multi-cell owners: the owner rides in the same
            // direction as the intruder so its footprint fully vacates the
            // target cell (mirror delta would leave residual overlap when
            // the owner is wider than the mover's step).
            let occ_is_multi = self
                .node_footprint(occ)
                .map(|b| b.cells().count() > 1)
                .unwrap_or(false);
            let raw_swap = if occ_is_multi { cur_delta } else { -cur_delta };
            let swap_delta = self.jump_delta(&occupancy, occ, occ_origin, raw_swap);
            for (id, d) in self.move_group(occ, swap_delta) {
                if plan.contains_key(&id) {
                    warn!("move_node_delta: plan conflict, aborted");
                    return (self.clone_shape(), primary_origin.round().as_ivec3());
                }
                let origin = self
                    .layout_nodes
                    .get(&id)
                    .map(|ln| ln.pos)
                    .unwrap_or(Vec3::ZERO);
                plan.insert(id.clone(), origin + d);
                worklist.push_back(id);
            }
        }

        let moved = Self {
            ast: self.ast.clone(),
            layout_nodes: self
                .layout_nodes
                .iter()
                .map(|(id, ln)| {
                    let new_pos = plan.get(id).copied().unwrap_or(ln.pos);
                    (
                        id.clone(),
                        LayoutNode {
                            node_id: id.clone(),
                            pos: new_pos,
                            shape: ln.shape.clone(),
                        },
                    )
                })
                .collect(),
            sub_layouts: self.sub_layouts.clone(),
        };

        let mut match_ids: std::collections::HashSet<crate::model::node::Id> =
            std::collections::HashSet::new();
        for id in plan.keys() {
            if self.is_pattern(id) {
                if let Some(mid) = self.parent_match_of(id) {
                    match_ids.insert(mid);
                }
            }
            if self.is_match(id) {
                match_ids.insert(id.clone());
            }
        }
        let after_recompute = match_ids
            .iter()
            .fold(moved, |acc, mid| acc.recompute_match_pos(mid));

        let effective_primary = plan
            .get(&node_id)
            .copied()
            .unwrap_or(primary_origin)
            .round()
            .as_ivec3();
        (after_recompute, effective_primary)
    }

    fn clone_shape(&self) -> Self {
        Self {
            ast: self.ast.clone(),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    /// Push external nodes out of every multi-cell node footprint, bottom-up.
    /// Applies to any node whose `node_footprint` has volume > 1 — matches
    /// and multi-input function calls. Recurses into `sub_layouts` first so
    /// inner owners settle before their container measures its own footprint.
    /// At each level, for every such owner: scan `layout_nodes` for
    /// non-related nodes whose rounded position falls inside the footprint
    /// and bump them along the axis with the smallest exit distance — +Y for
    /// Y-overlap, ±X toward the near footprint edge, +Z toward the sub-sink
    /// side (never −Z; that side faces the parent wall). Every push target is
    /// clamped to the non-negative octant, so an intruder on the low-X side of
    /// a footprint touching x=0 is pushed out the high side instead. Uses
    /// `move_node_delta` for each bump so cascading collisions resolve
    /// automatically. Iterates until stable (128-step cap).
    ///
    /// Between the recursion and the intruder pass, every Match at this level
    /// re-spaces its arms (`respace_match_patterns`). By then the branches
    /// below have settled, so their heights are final — that ordering is what
    /// carries growth out of nested Matches into their enclosing ones.
    pub fn settle_footprints(&self) -> Self {
        let settled_subs: std::collections::HashMap<crate::model::node::Id, LayoutAst> = self
            .sub_layouts
            .iter()
            .map(|(k, v)| (k.clone(), v.settle_footprints()))
            .collect();
        let layout = Self {
            ast: self.ast.clone(),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: settled_subs,
        };
        // Branch heights are final now, so pack every arm stack before any
        // footprint is measured below.
        let match_ids: Vec<crate::model::node::Id> = layout
            .layout_nodes
            .keys()
            .filter(|id| layout.is_match(id))
            .cloned()
            .collect();
        let mut layout = match_ids.iter().fold(layout, |acc, mid| {
            acc.respace_match_patterns(mid).recompute_match_pos(mid)
        });
        let owner_ids: Vec<crate::model::node::Id> = layout
            .layout_nodes
            .keys()
            .filter(|id| {
                layout
                    .node_footprint(id)
                    .map(|b| b.cells().count() > 1)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        for owner_id in &owner_ids {
            // Patterns of a Match ride along with their parent; skip them
            // as intruders. For non-match owners there are no related ids.
            let related: Vec<crate::model::node::Id> = if layout.is_match(owner_id) {
                layout.match_pattern_ids(owner_id)
            } else {
                vec![]
            };
            for _ in 0..128 {
                let Some(bbox) = layout.node_footprint(owner_id) else {
                    break;
                };
                let intruder = layout.layout_nodes.iter().find_map(|(id, ln)| {
                    if id == owner_id || related.contains(id) {
                        return None;
                    }
                    // A BranchSource cannot be displaced — it is pinned to its
                    // branch origin — so never pick one as the intruder, or the
                    // settle loop would spin until its iteration cap.
                    if matches!(
                        layout.ast.nodes.get(id),
                        Some(crate::model::node::ENode::BranchSource { .. })
                    ) {
                        return None;
                    }
                    let p = ln.pos.round().as_ivec3();
                    if bbox.contains(p) {
                        Some((id.clone(), p))
                    } else {
                        None
                    }
                });
                let Some((intruder_id, ipos)) = intruder else {
                    break;
                };
                // Exit toward the near X edge, unless that would leave the
                // non-negative octant — then out the far edge.
                let center_x = (bbox.min.x + bbox.max.x) / 2;
                let push_x = if ipos.x <= center_x && bbox.min.x - 1 >= 0 {
                    bbox.min.x - 1 - ipos.x
                } else {
                    bbox.max.x + 1 - ipos.x
                };
                let push_y = bbox.max.y + 1 - ipos.y;
                let push_z = bbox.max.z + 1 - ipos.z;
                // VarDecls are pinned to Y=0, Z=0 → only X-push can succeed.
                let is_var_decl = matches!(
                    layout.ast.nodes.get(&intruder_id),
                    Some(crate::model::node::ENode::VarDecl { .. })
                );
                // Priority: Z (deeper) → X (sideways) → Y (downward). Y is a
                // last resort because it crosses row boundaries; XZ keeps
                // the intruder on the same floor.
                let (best_axis, best_dist) = if is_var_decl {
                    (0u8, push_x)
                } else if push_z != 0 {
                    (2u8, push_z)
                } else if push_x != 0 {
                    (0u8, push_x)
                } else {
                    (1u8, push_y)
                };
                let delta = match best_axis {
                    0 => Vec3::new(best_dist as f32, 0.0, 0.0),
                    2 => Vec3::new(0.0, 0.0, best_dist as f32),
                    _ => Vec3::new(0.0, best_dist as f32, 0.0),
                };
                if delta.length_squared() < 0.25 {
                    break;
                }
                let (new_layout, _) = layout.move_node_delta(intruder_id, delta);
                layout = new_layout;
            }
        }
        layout.settle_sink().harmonize_match_sinks()
    }

    /// Push the Sink back (+Z, away from the wall) so the grid encapsulates
    /// the deepest node, regardless of where on X/Y it sits. `deepest_z` is
    /// the maximum over every non-sink node's `node_footprint` max.z (so
    /// multi-cell owners — function calls, Match footprints extending into
    /// +Z — are contained too). The sink lands one cell beyond that
    /// (`deepest_z + 1`), matching the convention the swap constraint already
    /// produces. Expand-only: the sink never moves back toward the wall,
    /// preserving the initial roomy corridor.
    fn settle_sink(&self) -> Self {
        let Some(sink_id) = self.sink_id() else {
            return self.clone_shape();
        };
        let Some(sink_ln) = self.layout_nodes.get(&sink_id) else {
            return self.clone_shape();
        };
        let sink_pos = sink_ln.pos;
        let current_sink_z = sink_pos.round().as_ivec3().z;

        let mut deepest_z = i32::MIN;
        for id in self.layout_nodes.keys() {
            if *id == sink_id {
                continue;
            }
            let Some(fp) = self.node_footprint(id) else {
                continue;
            };
            if fp.max.z > deepest_z {
                deepest_z = fp.max.z;
            }
        }
        if deepest_z == i32::MIN {
            return self.clone_shape();
        }

        let new_sink_z = current_sink_z.max(deepest_z + 1);
        if new_sink_z == current_sink_z {
            return self.clone_shape();
        }

        let mut layout = self.clone_shape();
        if let Some(ln) = layout.layout_nodes.get_mut(&sink_id) {
            ln.pos.z = new_sink_z as f32;
        }
        layout
    }

    /// Layout-Z of this scope's Sink, if it has one. The Sink always sits on
    /// the scope's maximum Z and that row is reserved for it.
    pub fn sink_z(&self) -> Option<i32> {
        let sink_id = self.sink_id()?;
        self.layout_nodes
            .get(&sink_id)
            .map(|ln| ln.pos.round().as_ivec3().z)
    }
    /// The single Sink node id of this LayoutAst, if present.
    pub fn sink_id(&self) -> Option<crate::model::node::Id> {
        self.layout_nodes
            .keys()
            .find(|id| {
                matches!(
                    self.ast.nodes.get(*id),
                    Some(crate::model::node::ENode::Sink { .. })
                )
            })
            .cloned()
    }

    /// Align the Sink of every Pattern under each Match at this level to
    /// a common position: the deepest sibling sink (minimum Z). Copies that
    /// reference sink's (x, z) onto every sibling sink so the match's back wall
    /// is a single flat plane and the sinks stack exactly above each other
    /// (only Y differs). Nested matches are handled by `settle_footprints`'s
    /// recursion, so this only needs to touch matches owned at this level.
    fn harmonize_match_sinks(&self) -> Self {
        let mut layout = self.clone_shape();
        let match_ids: Vec<crate::model::node::Id> = layout
            .layout_nodes
            .keys()
            .filter(|id| layout.is_match(id))
            .cloned()
            .collect();
        for match_id in &match_ids {
            let pattern_ids = layout.match_pattern_ids(match_id);
            // Reference = the sibling sink furthest back (largest Z).
            let mut reference: Option<(f32, f32)> = None;
            for pid in &pattern_ids {
                let Some(sub) = layout.sub_layouts.get(pid) else {
                    continue;
                };
                let Some(sink_id) = sub.sink_id() else {
                    continue;
                };
                let Some(ln) = sub.layout_nodes.get(&sink_id) else {
                    continue;
                };
                if reference.map(|(_, z)| ln.pos.z > z).unwrap_or(true) {
                    reference = Some((ln.pos.x, ln.pos.z));
                }
            }
            let Some((ref_x, ref_z)) = reference else {
                continue;
            };
            for pid in &pattern_ids {
                let Some(sub) = layout.sub_layouts.get_mut(pid) else {
                    continue;
                };
                let Some(sink_id) = sub.sink_id() else {
                    continue;
                };
                if let Some(ln) = sub.layout_nodes.get_mut(&sink_id) {
                    ln.pos.x = ref_x;
                    ln.pos.z = ref_z;
                }
            }
        }
        layout
    }

    /// Adjust a Y-direction delta so the mover jumps over the entire Y range
    /// of any Match footprint it would land on. Cascades through nested
    /// matches. For X/Z deltas or non-Match collisions the nominal delta
    /// is returned unchanged. Same-match sibling collisions also pass through
    /// unchanged so an intra-stack Y-swap can happen.
    fn jump_delta(
        &self,
        occupancy: &std::collections::HashMap<IVec3, crate::model::node::Id>,
        mover: &crate::model::node::Id,
        origin: Vec3,
        nominal_delta: Vec3,
    ) -> Vec3 {
        if nominal_delta.y.abs() < 0.5 {
            return nominal_delta;
        }
        let step = if nominal_delta.y > 0.0 { 1.0 } else { -1.0 };
        let mut target = origin + nominal_delta;
        for _ in 0..32 {
            let key = target.round().as_ivec3();
            let Some(occ) = occupancy.get(&key) else {
                return target - origin;
            };
            if occ == mover {
                return target - origin;
            }
            // Resolve which match's footprint we've hit (occ is either the
            // match itself or a Pattern child of one).
            let match_id = if self.is_match(occ) {
                Some(occ.clone())
            } else if self.is_pattern(occ) {
                self.parent_match_of(occ)
            } else {
                None
            };
            let Some(match_id) = match_id else {
                return target - origin;
            };
            // Same-match intra-stack Y-swap: let it through.
            if self.is_pattern(mover) && self.parent_match_of(mover) == Some(match_id.clone()) {
                return target - origin;
            }
            let Some(bbox) = self.match_footprint(&match_id) else {
                return target - origin;
            };
            // Clamp the upward jump at 0: layout space has no negative row.
            target.y = if step > 0.0 {
                bbox.max.y as f32 + step
            } else {
                (bbox.min.y as f32 + step).max(0.0)
            };
        }
        nominal_delta
    }

    /// Compute the co-moving group for a seed node.
    /// - Match seed: seed plus all its Pattern children with the same
    ///   delta (rigid block; sub-layouts follow via Pattern-local coords).
    /// - Non-Pattern seed: `{seed}`.
    /// - Pattern with Y-only delta: `{seed}` (row change is per-pattern).
    /// - Pattern with XZ delta: seed plus all sibling Patterns of the same
    ///   match, all with the XZ delta (siblings without the Y component).
    fn move_group(
        &self,
        seed: &crate::model::node::Id,
        seed_delta: Vec3,
    ) -> Vec<(crate::model::node::Id, Vec3)> {
        if self.is_match(seed) {
            let mut group: Vec<(crate::model::node::Id, Vec3)> = vec![(seed.clone(), seed_delta)];
            for pid in self.match_pattern_ids(seed) {
                group.push((pid, seed_delta));
            }
            return group;
        }
        if !self.is_pattern(seed) {
            return vec![(seed.clone(), seed_delta)];
        }
        let xz_zero = seed_delta.x.abs() < 0.5 && seed_delta.z.abs() < 0.5;
        if xz_zero {
            return vec![(seed.clone(), seed_delta)];
        }
        let Some(match_id) = self.parent_match_of(seed) else {
            return vec![(seed.clone(), seed_delta)];
        };
        // Sibling patterns co-move on XZ, and the parent Match must follow
        // so its footprint stays aligned during the swap-cascade phase.
        let xz_delta = Vec3::new(seed_delta.x, 0.0, seed_delta.z);
        let mut group: Vec<(crate::model::node::Id, Vec3)> = self
            .match_pattern_ids(&match_id)
            .into_iter()
            .map(|sid| {
                let d = if sid == *seed { seed_delta } else { xz_delta };
                (sid, d)
            })
            .collect();
        group.push((match_id, xz_delta));
        group
    }

    pub fn plus_edge(&self, from: crate::model::anchor::Id, to: crate::model::anchor::Id) -> Self {
        Self {
            ast: self.ast.plus_edge(from, to),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    /// Merge this scene's AST with every nested sub-layout's AST (recursively)
    /// into one flat `Ast`. Pattern sub-scenes keep their nodes in `sub_layouts`
    /// — invisible to this scene's own `ast` — so evaluation, which walks a
    /// single `Ast`, needs this combined view. The result's `sink_node_id` stays
    /// this scene's root sink (folding starts from `self.ast`).
    pub fn flattened_ast(&self) -> crate::model::ast::Ast {
        self.sub_layouts
            .values()
            .fold(self.ast.clone(), |acc, sub| {
                acc.merged_with(sub.flattened_ast())
            })
    }

    pub fn plus_const_decl(
        &self,
        r#type: crate::model::r#type::EType,
        pos: Vec3,
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let (anchor_id_domain, output_anchor_id) = anchor_id_domain.next_id();
        let (node_id_domain, node_id) = node_id_domain.next_id();
        let ast = self.ast.plus_node(
            node_id.clone(),
            crate::model::node::ENode::ConstDecl {
                r#type,
                output_anchor: output_anchor_id,
            },
        );
        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos);
        (layout, node_id_domain, anchor_id_domain)
    }

    pub fn plus_type_cast(
        &self,
        r#type: crate::model::r#type::EType,
        pos: Vec3,
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let (anchor_id_domain, input_anchor_id) = anchor_id_domain.next_id();
        let (anchor_id_domain, output_anchor_id) = anchor_id_domain.next_id();
        let (node_id_domain, node_id) = node_id_domain.next_id();
        let ast = self.ast.plus_node(
            node_id.clone(),
            crate::model::node::ENode::TypeCast {
                r#type,
                input_anchor: input_anchor_id,
                output_anchor: output_anchor_id,
            },
        );
        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos);
        (layout, node_id_domain, anchor_id_domain)
    }

    pub fn plus_function_call(
        &self,
        function_declaration: (
            crate::model::function_declaration::FunctionDeclarationId,
            &crate::model::function_declaration::FunctionDeclaration,
        ),
        pos: Vec3,
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let (anchor_id_domain, input_anchor_ids) =
            function_declaration
                .1
                .inputs
                .iter()
                .fold::<(AnchorIdDomain, Vec<crate::model::anchor::Id>), _>(
                    (anchor_id_domain, vec![]),
                    |(anchor_id_domain, input_anchor_ids), _| {
                        let (anchor_id_domain, new_anchor_id) = anchor_id_domain.next_id();
                        (
                            anchor_id_domain,
                            input_anchor_ids
                                .into_iter()
                                .chain(vec![new_anchor_id])
                                .collect(),
                        )
                    },
                );
        let (anchor_id_domain, output_anchor_id) = anchor_id_domain.next_id();
        let (node_id_domain, node_id) = node_id_domain.next_id();
        let ast = self.ast.plus_node(
            node_id.clone(),
            crate::model::node::ENode::FunctionCall {
                function_declaration_id: function_declaration.0,
                input_anchors: input_anchor_ids,
                output_anchor: output_anchor_id,
            },
        );
        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos);
        (layout, node_id_domain, anchor_id_domain)
    }

    pub fn with_function_call_replaced(
        &self,
        node_id: &crate::model::node::Id,
        new_fn: (
            crate::model::function_declaration::FunctionDeclarationId,
            &crate::model::function_declaration::FunctionDeclaration,
        ),
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let pos = self.layout_nodes.get(node_id).unwrap().pos;
        self.minus_node(node_id)
            .plus_function_call(new_fn, pos, node_id_domain, anchor_id_domain)
    }

    /// Create a `Match` container plus its initial `Pattern` child at `pos`.
    /// The Match's synthetic LayoutNode mirrors the lowest Pattern's pos so
    /// rendering can iterate `layout_nodes` uniformly. The Pattern is created
    /// with a fresh sub-AST (BranchSource + Sink) and a matching entry in
    /// `sub_layouts[pattern_id]`. The branch volume starts one cell behind the
    /// Pattern, so branch-local (0,0,0) — the BranchSource — is the cell
    /// adjoining the Pattern in +Z.
    pub fn plus_match(
        &self,
        pos: Vec3,
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let (anchor_id_domain, match_input_anchor_id) = anchor_id_domain.next_id();
        let (anchor_id_domain, match_output_anchor_id) = anchor_id_domain.next_id();
        let (node_id_domain, match_node_id) = node_id_domain.next_id();
        let ast = self.ast.plus_node(
            match_node_id.clone(),
            crate::model::node::ENode::Match {
                patterns: vec![],
                input_anchor: match_input_anchor_id.clone(),
                output_anchor: match_output_anchor_id.clone(),
            },
        );
        let (node_id_domain, pattern_node_id) = node_id_domain.next_id();
        // The sub-AST draws its sink node and anchor ids from the same shared
        // id domains, so every id in the tree stays globally unique.
        let (node_id_domain, anchor_id_domain, pattern_sub_ast, sub_sink_id, branch_source_id) =
            crate::model::ast::Ast::new_pattern_sub_ast(
                node_id_domain,
                anchor_id_domain,
                pattern_node_id.clone(),
            );
        let ast = ast.plus_node(
            pattern_node_id.clone(),
            crate::model::node::ENode::Pattern {
                parent_match: match_node_id.clone(),
                r#type: crate::model::r#type::EType::Int { value: None },
                sink_node_id: sub_sink_id.clone(),
            },
        );
        let pattern_sub_layout =
            Self::initial_pattern_sub_layout(&pattern_sub_ast, &sub_sink_id, &branch_source_id);
        let ast = ast.with_node_replaced(
            &match_node_id,
            crate::model::node::ENode::Match {
                patterns: vec![pattern_node_id.clone()],
                input_anchor: match_input_anchor_id,
                output_anchor: match_output_anchor_id,
            },
        );
        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self
                .sub_layouts
                .clone()
                .into_iter()
                .chain([(pattern_node_id.clone(), pattern_sub_layout)])
                .collect(),
        }
        // The Match keeps `pos` for its input anchor; the Pattern sits one cell
        // behind it, and its branch one further (`sub_layout_origin`).
        ._plus_layout_node(&pattern_node_id, pos + Vec3::new(0.0, 0.0, PATTERN_LOCAL_Z))
        ._plus_layout_node(&match_node_id, pos);
        (layout, node_id_domain, anchor_id_domain)
    }

    /// LayoutAst for a fresh Pattern's sub-AST: the branch's `BranchSource` at
    /// branch-local (0,0,0) and its Sink at (0,0,2).
    ///
    /// Branch-local (0,0,0) is the Pattern's cell + Z1 (see
    /// `sub_layout_origin`), so the source sits directly behind its Pattern.
    /// The Sink at Z=2 leaves exactly one free working cell at Z=1 from birth;
    /// `settle_sink` pushes it further back as the branch fills up.
    fn initial_pattern_sub_layout(
        sub_ast: &crate::model::ast::Ast,
        sub_sink_id: &crate::model::node::Id,
        branch_source_id: &crate::model::node::Id,
    ) -> Self {
        Self {
            ast: sub_ast.clone(),
            layout_nodes: std::collections::HashMap::from([
                (
                    branch_source_id.clone(),
                    LayoutNode::unshaped(branch_source_id.clone(), Vec3::ZERO),
                ),
                (
                    sub_sink_id.clone(),
                    LayoutNode::unshaped(sub_sink_id.clone(), Vec3::new(0.0, 0.0, 2.0)),
                ),
            ]),
            sub_layouts: std::collections::HashMap::new(),
        }
    }

    /// Insert a new Pattern directly below `selected_pattern_id` in its parent
    /// Match — i.e. into the row at `selected.Y + 1`, since layout `+Y` renders
    /// downward. Works from any Pattern of the stack, not just the topmost.
    ///
    /// Sibling Patterns strictly *after* the selected row shift by +1 to make
    /// room; the selected Pattern itself stays put, so the caret needs no
    /// adjustment. The growing match footprint then displaces any external
    /// neighbours via `settle_footprints`.
    ///
    /// Growth is +Y only, which keeps the stack inside the non-negative octant
    /// and leaves the Pattern at the match's lowest Y as its origin row.
    pub fn plus_pattern_below(
        &self,
        selected_pattern_id: &crate::model::node::Id,
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        let (parent_id, selected_pos) = match self.ast.nodes.get(selected_pattern_id) {
            Some(crate::model::node::ENode::Pattern { parent_match, .. }) => {
                let ln = self.layout_nodes.get(selected_pattern_id).unwrap();
                (parent_match.clone(), ln.pos)
            }
            _ => {
                return (
                    Self {
                        ast: self.ast.clone(),
                        layout_nodes: self.layout_nodes.clone(),
                        sub_layouts: self.sub_layouts.clone(),
                    },
                    node_id_domain,
                    anchor_id_domain,
                )
            }
        };
        let selected_y = selected_pos.y;
        let column_x = selected_pos.x;
        let column_z = selected_pos.z;
        // Shift only sibling Patterns of the same match strictly below the
        // selected row — the selected Pattern keeps its slot and the new one
        // takes the row right after it. External neighbours are handled by
        // settle_footprints once the new Pattern has grown the footprint.
        let match_pattern_ids: std::collections::HashSet<crate::model::node::Id> =
            self.match_pattern_ids(&parent_id).into_iter().collect();
        let shifted_layout_nodes = self
            .layout_nodes
            .iter()
            .map(|(id, ln)| {
                if !match_pattern_ids.contains(id) {
                    return (id.clone(), ln.clone());
                }
                if ln.pos.y > selected_y + 0.001 {
                    (
                        id.clone(),
                        LayoutNode {
                            node_id: id.clone(),
                            pos: ln.pos + Vec3::new(0.0, 1.0, 0.0),
                            shape: ln.shape.clone(),
                        },
                    )
                } else {
                    (id.clone(), ln.clone())
                }
            })
            .collect();
        let shifted = Self {
            ast: self.ast.clone(),
            layout_nodes: shifted_layout_nodes,
            sub_layouts: self.sub_layouts.clone(),
        };
        let sibling_ids: Vec<crate::model::node::Id> = match shifted.ast.nodes.get(&parent_id) {
            Some(crate::model::node::ENode::Match { patterns, .. }) => patterns.clone(),
            _ => vec![],
        };
        let (node_id_domain, new_pattern_id) = node_id_domain.next_id();
        // The sub-AST draws its ids from the same shared domains, keeping every
        // id in the tree globally unique (see plus_match).
        let (
            node_id_domain,
            anchor_id_domain,
            new_pattern_sub_ast,
            new_sub_sink_id,
            new_branch_source_id,
        ) = crate::model::ast::Ast::new_pattern_sub_ast(
            node_id_domain,
            anchor_id_domain,
            new_pattern_id.clone(),
        );
        let ast = shifted.ast.plus_node(
            new_pattern_id.clone(),
            crate::model::node::ENode::Pattern {
                parent_match: parent_id.clone(),
                r#type: crate::model::r#type::EType::Int { value: None },
                sink_node_id: new_sub_sink_id.clone(),
            },
        );
        let new_pattern_sub_layout = Self::initial_pattern_sub_layout(
            &new_pattern_sub_ast,
            &new_sub_sink_id,
            &new_branch_source_id,
        );
        let new_patterns: Vec<crate::model::node::Id> = sibling_ids
            .iter()
            .cloned()
            .chain([new_pattern_id.clone()])
            .collect();
        let (match_input_anchor, match_output_anchor) = match ast.nodes.get(&parent_id) {
            Some(crate::model::node::ENode::Match {
                input_anchor,
                output_anchor,
                ..
            }) => (input_anchor.clone(), output_anchor.clone()),
            _ => return (shifted, node_id_domain, anchor_id_domain),
        };
        let ast = ast.with_node_replaced(
            &parent_id,
            crate::model::node::ENode::Match {
                patterns: new_patterns,
                input_anchor: match_input_anchor,
                output_anchor: match_output_anchor,
            },
        );
        let with_new = Self {
            ast,
            layout_nodes: shifted.layout_nodes,
            sub_layouts: shifted
                .sub_layouts
                .into_iter()
                .chain([(new_pattern_id.clone(), new_pattern_sub_layout)])
                .collect(),
        }
        ._plus_layout_node(
            &new_pattern_id,
            Vec3::new(column_x, selected_y + 1.0, column_z),
        );
        let match_ids: Vec<crate::model::node::Id> = with_new
            .ast
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                if matches!(n, crate::model::node::ENode::Match { .. }) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        let layout = match_ids
            .iter()
            .fold(with_new, |acc, mid| acc.recompute_match_pos(mid));
        (layout, node_id_domain, anchor_id_domain)
    }

    /// Refresh a Match's synthetic LayoutNode: same X and lowest sibling Y as
    /// its Patterns, but one cell *before* them in Z, because the Match's own
    /// input anchor owns that cell (`PATTERN_LOCAL_Z`). Needed after any
    /// add/remove/shift of Patterns so the render pass finds the container at
    /// the correct origin.
    pub fn recompute_match_pos(&self, match_id: &crate::model::node::Id) -> Self {
        let pattern_ids: Vec<crate::model::node::Id> = match self.ast.nodes.get(match_id) {
            Some(crate::model::node::ENode::Match { patterns, .. }) => patterns.clone(),
            _ => {
                return Self {
                    ast: self.ast.clone(),
                    layout_nodes: self.layout_nodes.clone(),
                    sub_layouts: self.sub_layouts.clone(),
                }
            }
        };
        let lowest_pos = pattern_ids
            .iter()
            .filter_map(|pid| self.layout_nodes.get(pid).map(|ln| ln.pos))
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
        let Some(lowest_pos) = lowest_pos else {
            return Self {
                ast: self.ast.clone(),
                layout_nodes: self.layout_nodes.clone(),
                sub_layouts: self.sub_layouts.clone(),
            };
        };
        let new_pos = lowest_pos - Vec3::new(0.0, 0.0, PATTERN_LOCAL_Z);
        Self {
            ast: self.ast.clone(),
            layout_nodes: self
                .layout_nodes
                .iter()
                .map(|(id, ln)| {
                    if id == match_id {
                        (
                            id.clone(),
                            LayoutNode {
                                node_id: id.clone(),
                                pos: new_pos,
                                shape: ln.shape.clone(),
                            },
                        )
                    } else {
                        (id.clone(), ln.clone())
                    }
                })
                .collect(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    fn _with_match_patterns(
        &self,
        match_id: &crate::model::node::Id,
        new_patterns: Vec<crate::model::node::Id>,
    ) -> Self {
        let (input_anchor, output_anchor) = match self.ast.nodes.get(match_id) {
            Some(crate::model::node::ENode::Match {
                input_anchor,
                output_anchor,
                ..
            }) => (input_anchor.clone(), output_anchor.clone()),
            _ => {
                return Self {
                    ast: self.ast.clone(),
                    layout_nodes: self.layout_nodes.clone(),
                    sub_layouts: self.sub_layouts.clone(),
                }
            }
        };
        Self {
            ast: self.ast.with_node_replaced(
                match_id,
                crate::model::node::ENode::Match {
                    patterns: new_patterns,
                    input_anchor,
                    output_anchor,
                },
            ),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    pub fn plus_var_decl(
        &self,
        pos: Vec3,
        node_id_domain: NodeIdDomain,
        anchor_id_domain: AnchorIdDomain,
    ) -> (Self, NodeIdDomain, AnchorIdDomain) {
        // VarDecls live only on the Program wall (Y=0, Z=0). Snap defensively.
        let pos = Vec3::new(pos.x, 0.0, 0.0);
        let (anchor_id_domain, output_anchor_id) = anchor_id_domain.next_id();
        let (node_id_domain, node_id) = node_id_domain.next_id();
        let ast = self.ast.plus_node(
            node_id.clone(),
            crate::model::node::ENode::VarDecl {
                name: "v".to_string(),
                r#type: crate::model::r#type::EType::Int { value: None },
                output_anchor: output_anchor_id,
            },
        );
        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos);
        (layout, node_id_domain, anchor_id_domain)
    }

    /// Grid-space origin of `owner_id`'s sub-layout, in this LayoutAst's own
    /// coordinates.
    ///
    /// A Pattern's branch volume starts one cell *behind* the Pattern (+Z):
    /// the Pattern itself belongs to the Match volume, not to the branch, so
    /// branch-local (0,0,0) — where the `BranchSource` sits — lands at the
    /// Pattern's Z+1. Every other owner (the Program wrapper) contributes no
    /// shift.
    fn sub_layout_origin(&self, owner_id: &crate::model::node::Id) -> Vec3 {
        let base = self
            .layout_nodes
            .get(owner_id)
            .map(|ln| ln.pos)
            .unwrap_or(Vec3::ZERO);
        if matches!(
            self.ast.nodes.get(owner_id),
            Some(crate::model::node::ENode::Pattern { .. })
        ) {
            base + Vec3::new(0.0, 0.0, 1.0)
        } else {
            base
        }
    }

    fn _plus_layout_node(&self, node_id: &crate::model::node::Id, pos: Vec3) -> Self {
        Self {
            ast: self.ast.clone(),
            layout_nodes: self
                .layout_nodes
                .clone()
                .into_iter()
                .chain([(node_id.clone(), LayoutNode::unshaped(node_id.clone(), pos))])
                .collect(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    /// Recursively walk every node in this LayoutAst and its `sub_layouts`.
    /// Each entry carries the containing LayoutAst (for anchor lookups), the
    /// LayoutNode, and the accumulated grid-space offset from the outer root.
    ///
    /// Descent into a sub-layout uses the owner node's grid position as the
    /// additional offset. The outer root's `layout_nodes` is expected to be
    /// empty (Program has no LayoutNode) so `sub_layouts[program_id]` is
    /// entered with offset (0,0,0).
    pub fn walk_all(&self) -> Vec<WalkedNode> {
        let mut out = Vec::new();
        self.walk_all_into(Vec3::ZERO, &mut out);
        out
    }

    fn walk_all_into<'a>(&'a self, offset: Vec3, out: &mut Vec<WalkedNode<'a>>) {
        for layout_node in self.layout_nodes.values() {
            out.push(WalkedNode {
                layout_ast: self,
                layout_node,
                extra_offset: offset,
            });
        }
        for (owner_id, sub_layout) in &self.sub_layouts {
            let sub_offset = offset + self.sub_layout_origin(owner_id);
            sub_layout.walk_all_into(sub_offset, out);
        }
    }

    /// Yield every LayoutAst reachable from `self`, including `self` itself.
    /// Each entry carries the owner path from `self` down to the yielded AST
    /// (empty at `self`) and the accumulated grid-space offset.
    pub fn walk_all_asts(&self) -> Vec<WalkedAst> {
        let mut out = Vec::new();
        self.walk_all_asts_into(Vec::new(), Vec3::ZERO, &mut out);
        out
    }

    fn walk_all_asts_into<'a>(
        &'a self,
        context: Vec<crate::model::node::Id>,
        offset: Vec3,
        out: &mut Vec<WalkedAst<'a>>,
    ) {
        out.push(WalkedAst {
            layout_ast: self,
            context: context.clone(),
            extra_offset: offset,
        });
        for (owner_id, sub_layout) in &self.sub_layouts {
            let sub_origin = self.sub_layout_origin(owner_id);
            let mut sub_context = context.clone();
            sub_context.push(owner_id.clone());
            sub_layout.walk_all_asts_into(sub_context, offset + sub_origin, out);
        }
    }

    /// Compute the grid bounds for this AST in its own local coordinates.
    ///
    /// - Z range: `[0, sink.z]` where `sink.z` is the single Sink's layout-Z.
    ///   Both ends are reserved: local Z=0 is the source row (VarDecls in the
    ///   root scope, the BranchSource in a branch) and `sink.z` belongs to the
    ///   Sink alone. Returns `None` if no Sink exists.
    /// - X range: min/max over every node's footprint X-span. Multi-cell
    ///   footprints (matches, ≥3-input function calls) grow the grid so the
    ///   extra cell is drawable. If the AST has no nodes at all the sink is
    ///   the only node — X min/max = 0 (sink is at X=0).
    /// - Y range: min/max over every node's footprint Y-span. A Match
    ///   contributes its whole Pattern stack, so a scope containing one spans
    ///   every row its patterns occupy. Only `scope_at` reads this — the grid
    ///   mesh is a single plane and uses X/Z alone — but without it a caret on
    ///   any Pattern below the first row would fall outside every scope.
    ///
    /// The bounds follow the nodes and nothing else: the caret cannot widen
    /// them by moving, since `clamp_to_volume` keeps it inside.
    pub fn ast_grid_bounds(&self) -> Option<AstGridBounds> {
        let sink_z =
            self.layout_nodes
                .iter()
                .find_map(|(id, ln)| match self.ast.nodes.get(id) {
                    Some(crate::model::node::ENode::Sink { .. }) => {
                        Some(ln.pos.round().as_ivec3().z)
                    }
                    _ => None,
                })?;
        let z_min = 0;
        let z_max = sink_z;
        let mut x_min = i32::MAX;
        let mut x_max = i32::MIN;
        let mut y_min = i32::MAX;
        let mut y_max = i32::MIN;
        for id in self.layout_nodes.keys() {
            let Some(fp) = self.node_footprint(id) else {
                continue;
            };
            x_min = x_min.min(fp.min.x);
            x_max = x_max.max(fp.max.x);
            y_min = y_min.min(fp.min.y);
            y_max = y_max.max(fp.max.y);
        }
        if x_min == i32::MAX {
            x_min = 0;
            x_max = 0;
        }
        if y_min == i32::MAX {
            y_min = 0;
            y_max = 0;
        }
        Some(AstGridBounds {
            min: IVec3::new(x_min, y_min, z_min),
            max: IVec3::new(x_max, y_max, z_max),
        })
    }

    /// Clamp a global cell address into this AST's volume, i.e. the bounds
    /// every scope nested in it lives inside.
    ///
    /// Caret navigation goes through here: the volume follows the nodes, and
    /// widening it — adding empty cells to move into — is an explicit action,
    /// never a side effect of moving. With no Sink there is no volume yet;
    /// only the non-negative half-space constrains the address then.
    pub fn clamp_to_volume(&self, global: IVec3) -> IVec3 {
        match self.ast_grid_bounds() {
            Some(bounds) => global.clamp(bounds.min, bounds.max),
            None => global.max(IVec3::ZERO),
        }
    }

    pub fn edges(&self) -> Vec<LayoutEdge> {
        self.ast
            .edges
            .iter()
            .flat_map(|(from_anchor_id, edges)| {
                edges.clone().into_iter().map(|edge| LayoutEdge {
                    from_anchor: self.layout_anchor(from_anchor_id.clone()),
                    to_anchor: self.layout_anchor(edge.to.clone()),
                })
            })
            .collect()
    }

    pub fn layout_anchor(&self, anchor_id: crate::model::anchor::Id) -> LayoutAnchor {
        self.try_layout_anchor(&anchor_id).unwrap_or_else(|| {
            panic!(
                "layout_anchor: anchor {:?} not found in any (sub-)ast",
                anchor_id
            )
        })
    }

    fn try_layout_anchor(&self, anchor_id: &crate::model::anchor::Id) -> Option<LayoutAnchor> {
        if let Some(anchor) = self.ast.anchors.get(anchor_id) {
            let node_id = self.ast.anchor_to_node.get(anchor_id).unwrap().clone();
            return Some(LayoutAnchor {
                anchor_id: anchor_id.clone(),
                anchor: anchor.clone(),
                node_id,
                pos: Vec3::splat(1.0),
            });
        }
        for sub in self.sub_layouts.values() {
            if let Some(la) = sub.try_layout_anchor(anchor_id) {
                return Some(la);
            }
        }
        None
    }

    /// Owner path from this LayoutAst down to the LayoutAst whose
    /// `layout_nodes` contains `target`. `Some(vec![])` = target lives in
    /// `self`; `Some(vec![a, b])` = target lives in `self.sub_layouts[a]
    /// .sub_layouts[b]`. `None` = not found.
    pub fn context_of_node(
        &self,
        target: &crate::model::node::Id,
    ) -> Option<Vec<crate::model::node::Id>> {
        if self.layout_nodes.contains_key(target) {
            return Some(vec![]);
        }
        for (owner_id, sub) in &self.sub_layouts {
            if let Some(mut rest) = sub.context_of_node(target) {
                let mut path = vec![owner_id.clone()];
                path.append(&mut rest);
                return Some(path);
            }
        }
        None
    }

    /// Return the LayoutAst that holds `target` in its `ast.nodes` map.
    /// Used by editor handlers to mutate node fields without needing to
    /// know which sub-layout the node lives in.
    pub fn find_node_ast_mut(&mut self, target: &crate::model::node::Id) -> Option<&mut LayoutAst> {
        if self.ast.nodes.contains_key(target) {
            return Some(self);
        }
        for sub in self.sub_layouts.values_mut() {
            if let Some(found) = sub.find_node_ast_mut(target) {
                return Some(found);
            }
        }
        None
    }

    /// Resolve an owner path (as produced by `context_of_node`) to the
    /// corresponding sub-LayoutAst reference. Panics if the path names a
    /// key that no longer exists — callers are expected to have obtained
    /// the path from a fresh lookup in the same frame.
    pub fn resolve_context<'a>(&'a self, path: &[crate::model::node::Id]) -> &'a LayoutAst {
        let mut ast = self;
        for id in path {
            ast = ast.sub_layouts.get(id).unwrap();
        }
        ast
    }

    /// Resolve a global cell address to the scope that owns it.
    ///
    /// Returns the owner path of the **innermost** scope whose volume contains
    /// `global`, together with the address expressed in that scope's own local
    /// coordinates. This is the sole answer to "what does editing act on":
    /// when the caret sits inside a Match branch volume it always resolves to
    /// that branch scope, never to the enclosing parent.
    ///
    /// A scope's volume is its `ast_grid_bounds`, i.e. the whole wall-to-sink
    /// corridor including the cells that are still empty — otherwise the caret
    /// could not be placed where a node is about to be inserted.
    ///
    /// Every scope claims its own source row at local Z=0. A Pattern is not
    /// part of its branch volume — the branch starts one cell behind it — so
    /// the Pattern stays addressable in the Match's scope while branch-local
    /// Z=0 belongs to the branch's BranchSource.
    ///
    /// `None` when the address lies outside every scope; callers treat that as
    /// "nothing to edit here".
    pub fn scope_at(&self, global: IVec3) -> Option<(Vec<crate::model::node::Id>, IVec3)> {
        let mut best: Option<(Vec<crate::model::node::Id>, IVec3)> = None;
        for walked in self.walk_all_asts() {
            let Some(bounds) = walked.layout_ast.ast_grid_bounds() else {
                continue;
            };
            let offset = walked.extra_offset.round().as_ivec3();
            let local = global - offset;
            let inside = local.x >= bounds.min.x
                && local.x <= bounds.max.x
                && local.y >= bounds.min.y
                && local.y <= bounds.max.y
                && local.z >= bounds.min.z
                && local.z <= bounds.max.z;
            if !inside {
                continue;
            }
            // Deeper path wins: sub-scopes are nested inside their parent's
            // volume, and the innermost one is the owner.
            if best
                .as_ref()
                .is_none_or(|(path, _)| walked.context.len() > path.len())
            {
                best = Some((walked.context.clone(), local));
            }
        }
        best
    }

    /// Grid-space origin of the scope named by `path`, i.e. the offset that
    /// turns a scope-local address into a global one.
    pub fn scope_offset(&self, path: &[crate::model::node::Id]) -> IVec3 {
        let mut offset = IVec3::ZERO;
        let mut ast = self;
        for id in path {
            offset += ast.sub_layout_origin(id).round().as_ivec3();
            let Some(next) = ast.sub_layouts.get(id) else {
                break;
            };
            ast = next;
        }
        offset
    }

    /// Mutable counterpart to `resolve_context`.
    pub fn resolve_context_mut<'a>(
        &'a mut self,
        path: &[crate::model::node::Id],
    ) -> Option<&'a mut LayoutAst> {
        let mut ast = self;
        for id in path {
            ast = ast.sub_layouts.get_mut(id)?;
        }
        Some(ast)
    }
}

/// Concrete literal string on an `model::r#type::EType`, if any. Only the
/// value-carrying variants have an `Option<String>`; the `None` type variant
/// yields no literal.
pub fn value_of_etype(t: &crate::model::r#type::EType) -> Option<String> {
    match t {
        crate::model::r#type::EType::Bool { value }
        | crate::model::r#type::EType::Int { value }
        | crate::model::r#type::EType::String { value }
        | crate::model::r#type::EType::Char { value } => value.clone(),
        _ => None,
    }
}
