use bevy::prelude::*;

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub node_id: crate::ast::node::Id,
    pub pos: Vec3,
}

#[derive(Debug, Clone)]
pub struct LayoutEdge {
    pub from_anchor: LayoutAnchor,
    pub to_anchor: LayoutAnchor,
}

#[derive(Debug, Clone)]
pub struct LayoutAnchor {
    pub anchor_id: crate::ast::AnchorId,
    pub node_id: crate::ast::node::Id,
    pub anchor: crate::ast::EAnchor,
    pub pos: Vec3,
}

/// A single entry produced by `LayoutAst::walk_all`. Groups a LayoutNode with
/// enough context (its owning LayoutAst for anchor lookups, the accumulated
/// grid-space offset, and a sink-scale hint) so the render layer can place
/// pattern sub-AST nodes correctly without re-doing the traversal.
pub struct WalkedNode<'a> {
    pub layout_ast: &'a LayoutAst,
    pub layout_node: &'a LayoutNode,
    pub extra_offset: Vec3,
    pub sink_scale: f32,
}

/// One entry per LayoutAst reached from a root. Used by the per-AST grid
/// renderer to place a dedicated grid mesh per Program/Pattern sub-AST.
pub struct WalkedAst<'a> {
    pub layout_ast: &'a LayoutAst,
    /// Owner path from the walk root to this AST. Empty at the outermost
    /// LayoutAst (the one `walk_all_asts` was called on).
    pub context: Vec<crate::ast::node::Id>,
    /// Accumulated grid-space offset from the root to this AST's origin.
    pub extra_offset: Vec3,
}

/// Bounds of an AST grid in that AST's local grid coordinates. Both corners
/// are inclusive; `y = 0` for both.
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

impl AABB {
    pub fn point(p: IVec3) -> Self {
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
    pub ast: crate::ast::Ast,
    pub layout_nodes: std::collections::HashMap<crate::ast::node::Id, LayoutNode>,
    /// Per-owner nested layouts. Keyed by the container node id (Program in
    /// Step 1; Pattern in later steps). The `sub_layouts[program_id]` LayoutAst
    /// is the source of truth for the top-level context — the ENode::Program's
    /// own `ast` field is unused in Step 1 and stays empty.
    pub sub_layouts: std::collections::HashMap<crate::ast::node::Id, LayoutAst>,
}

impl LayoutAst {
    pub fn empty() -> Self {
        Self {
            ast: crate::ast::Ast::empty(),
            layout_nodes: std::collections::HashMap::new(),
            sub_layouts: std::collections::HashMap::new(),
        }
    }

    /// Build a root LayoutAst that holds a single Program node plus an empty
    /// `sub_layouts` entry keyed by the Program's id. The inner LayoutAst is
    /// where user-visible nodes live in Step 1 (mirrors the previous behavior
    /// of a flat `LayoutAst::empty()` root).
    pub fn empty_with_program() -> (Self, crate::ast::node::Id) {
        let (ast, program_id) = crate::ast::Ast::empty().plus(crate::ast::node::ENode::Program {});
        let outer = Self {
            ast,
            layout_nodes: std::collections::HashMap::new(),
            sub_layouts: std::collections::HashMap::from([(program_id.clone(), Self::empty())]),
        };
        (outer, program_id)
    }

    pub fn minus_node(&self, node_id: &crate::ast::node::Id) -> Self {
        match self.ast.nodes.get(node_id) {
            Some(crate::ast::node::ENode::Pattern { parent_match, .. }) => {
                let parent_id = parent_match.clone();
                let remaining: Vec<crate::ast::node::Id> = match self.ast.nodes.get(&parent_id) {
                    Some(crate::ast::node::ENode::MatchNew { patterns, .. }) => {
                        patterns.iter().filter(|p| *p != node_id).cloned().collect()
                    }
                    _ => vec![],
                };
                let after_pattern = Self {
                    ast: self.ast.minus(node_id),
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
                        ast: after_pattern.ast.minus(&parent_id),
                        layout_nodes: after_pattern
                            .layout_nodes
                            .into_iter()
                            .filter(|(id, _)| id != &parent_id)
                            .collect(),
                        sub_layouts: after_pattern.sub_layouts,
                    }
                } else {
                    after_pattern
                        ._with_matchnew_patterns(&parent_id, remaining)
                        .recompute_matchnew_pos(&parent_id)
                }
            }
            Some(crate::ast::node::ENode::MatchNew { patterns, .. }) => {
                let child_ids: Vec<_> = patterns.clone();
                let after_children = child_ids.iter().fold(
                    Self {
                        ast: self.ast.clone(),
                        layout_nodes: self.layout_nodes.clone(),
                        sub_layouts: self.sub_layouts.clone(),
                    },
                    |acc, pid| Self {
                        ast: acc.ast.minus(pid),
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
                    ast: after_children.ast.minus(node_id),
                    layout_nodes: after_children
                        .layout_nodes
                        .into_iter()
                        .filter(|(id, _)| id != node_id)
                        .collect(),
                    sub_layouts: after_children.sub_layouts,
                }
            }
            _ => Self {
                ast: self.ast.minus(node_id),
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

    /// Returns the node whose layout position rounds to `pos`, if any.
    /// `MatchNew` containers are excluded — only their `Pattern` children are
    /// selectable, so a click on the envelope never picks the container.
    pub fn node_at(&self, pos: IVec3) -> Option<crate::ast::node::Id> {
        self.layout_nodes.iter().find_map(|(id, ln)| {
            if ln.pos.round().as_ivec3() != pos {
                return None;
            }
            if matches!(
                self.ast.nodes.get(id),
                Some(crate::ast::node::ENode::MatchNew { .. })
            ) {
                return None;
            }
            Some(id.clone())
        })
    }

    /// Build a `grid position -> node id` lookup over all selectable nodes.
    /// MatchNews claim every cell of their `matchnew_footprint`; other nodes
    /// claim a single cell. When both a Pattern and its owning Match cover the
    /// same cell (Pattern rows of the match column) the Pattern id wins, so
    /// intra-stack Y-swap still resolves the Pattern as target.
    fn occupancy_map(&self) -> std::collections::HashMap<IVec3, crate::ast::node::Id> {
        let mut map: std::collections::HashMap<IVec3, crate::ast::node::Id> =
            std::collections::HashMap::new();
        for (id, _ln) in &self.layout_nodes {
            if !self.is_matchnew(id) {
                continue;
            }
            let Some(bbox) = self.matchnew_footprint(id) else {
                continue;
            };
            for cell in bbox.cells() {
                map.insert(cell, id.clone());
            }
        }
        for (id, ln) in &self.layout_nodes {
            if self.is_matchnew(id) {
                continue;
            }
            map.insert(ln.pos.round().as_ivec3(), id.clone());
        }
        map
    }

    fn is_pattern(&self, id: &crate::ast::node::Id) -> bool {
        matches!(
            self.ast.nodes.get(id),
            Some(crate::ast::node::ENode::Pattern { .. })
        )
    }

    fn parent_match_of(&self, id: &crate::ast::node::Id) -> Option<crate::ast::node::Id> {
        match self.ast.nodes.get(id) {
            Some(crate::ast::node::ENode::Pattern { parent_match, .. }) => {
                Some(parent_match.clone())
            }
            _ => None,
        }
    }

    /// Return the sibling Pattern ids of a `MatchNew`.
    fn match_pattern_ids(&self, match_id: &crate::ast::node::Id) -> Vec<crate::ast::node::Id> {
        match self.ast.nodes.get(match_id) {
            Some(crate::ast::node::ENode::MatchNew { patterns, .. }) => patterns.clone(),
            _ => vec![],
        }
    }

    fn is_matchnew(&self, id: &crate::ast::node::Id) -> bool {
        matches!(
            self.ast.nodes.get(id),
            Some(crate::ast::node::ENode::MatchNew { .. })
        )
    }

    /// Union of all visible grid cells in this LayoutAst's local coords.
    /// MatchNew nodes contribute their `matchnew_footprint` (recursive over
    /// nested sub-ASTs); every other node contributes its rounded cell.
    /// Sub-layouts contribute their own inner_footprint offset by the owner
    /// node's grid position.
    pub fn inner_footprint(&self) -> Option<AABB> {
        let mut bbox: Option<AABB> = None;
        for (id, ln) in &self.layout_nodes {
            let node_bbox = if self.is_matchnew(id) {
                self.matchnew_footprint(id)
            } else {
                Some(AABB::point(ln.pos.round().as_ivec3()))
            };
            bbox = AABB::union_opt(bbox, node_bbox);
        }
        for (owner_id, sub) in &self.sub_layouts {
            let owner_pos = self
                .layout_nodes
                .get(owner_id)
                .map(|ln| ln.pos.round().as_ivec3())
                .unwrap_or(IVec3::ZERO);
            let sub_bbox = sub.inner_footprint().map(|b| b.translated(owner_pos));
            bbox = AABB::union_opt(bbox, sub_bbox);
        }
        bbox
    }

    /// Grid-space AABB (in this LayoutAst's local coords) that a MatchNew
    /// container claims. Aggregates all Pattern arms:
    ///   - Y: `sum over patterns of max(1, subast_y_extent)` stacked at the
    ///     Pattern's own y (patterns are assumed to occupy consecutive rows).
    ///   - X: union of every Pattern's sub-AST X extent, translated by the
    ///     Pattern's outer position.
    ///   - Z: each Pattern's sub-AST Z range padded by 2 additional cells in
    ///     the −Z direction (to contain the sub-sink), upper-bounded by the
    ///     Pattern's own Z (i.e. the parent-facing side).
    /// Recursion via `inner_footprint` — inner matches inflate their host
    /// Pattern's sub-AST bbox and thus the outer footprint too.
    pub fn matchnew_footprint(&self, match_id: &crate::ast::node::Id) -> Option<AABB> {
        if !self.is_matchnew(match_id) {
            return None;
        }
        let pattern_ids = self.match_pattern_ids(match_id);
        let mut bbox: Option<AABB> = None;
        for pid in &pattern_ids {
            let Some(p_ln) = self.layout_nodes.get(pid) else {
                continue;
            };
            let p_pos = p_ln.pos.round().as_ivec3();
            let sub = self.sub_layouts.get(pid);
            let sub_bbox = sub.and_then(|s| s.inner_footprint());
            let (arm_y, x_min, x_max, z_min) = match sub_bbox {
                Some(b) => {
                    let sub_y_extent = b.max.y - b.min.y + 1;
                    let arm_y = sub_y_extent.max(1);
                    (arm_y, b.min.x, b.max.x, b.min.z)
                }
                None => (1, 0, 0, 0),
            };
            let arm = AABB {
                min: IVec3::new(p_pos.x + x_min, p_pos.y, p_pos.z + z_min - 2),
                max: IVec3::new(p_pos.x + x_max, p_pos.y + arm_y - 1, p_pos.z),
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
    pub fn move_node_delta(&self, node_id: crate::ast::node::Id, delta_pos: Vec3) -> (Self, IVec3) {
        let Some(primary_ln) = self.layout_nodes.get(&node_id) else {
            return (self.clone_shape(), IVec3::ZERO);
        };
        let primary_origin = primary_ln.pos;
        // VarDecls are pinned to the Program wall row (Y=0, Z=0);
        // every other node type must stay south of it (Z < 0).
        let delta_pos = match self.ast.nodes.get(&node_id) {
            Some(crate::ast::node::ENode::VarDecl { .. }) => Vec3::new(delta_pos.x, 0.0, 0.0),
            _ => {
                let target_z = (primary_origin.z + delta_pos.z).round() as i32;
                if target_z >= 0 {
                    Vec3::new(delta_pos.x, delta_pos.y, 0.0)
                } else {
                    delta_pos
                }
            }
        };
        let occupancy = self.occupancy_map();

        let primary_delta = self.jump_delta(&occupancy, &node_id, primary_origin, delta_pos);

        let mut plan: std::collections::HashMap<crate::ast::node::Id, Vec3> =
            std::collections::HashMap::new();
        let mut worklist: std::collections::VecDeque<crate::ast::node::Id> =
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
            // Section-swap for MatchNew: match rides in the same direction as
            // the intruder so its multi-cell footprint fully vacates the
            // target cell (mirror delta would leave residual overlap when
            // the match is wider than the mover's step).
            let raw_swap = if self.is_matchnew(occ) {
                cur_delta
            } else {
                -cur_delta
            };
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
                        },
                    )
                })
                .collect(),
            sub_layouts: self.sub_layouts.clone(),
        };

        let mut match_ids: std::collections::HashSet<crate::ast::node::Id> =
            std::collections::HashSet::new();
        for id in plan.keys() {
            if self.is_pattern(id) {
                if let Some(mid) = self.parent_match_of(id) {
                    match_ids.insert(mid);
                }
            }
            if self.is_matchnew(id) {
                match_ids.insert(id.clone());
            }
        }
        let after_recompute = match_ids
            .iter()
            .fold(moved, |acc, mid| acc.recompute_matchnew_pos(mid));

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

    /// Push external nodes out of every MatchNew's footprint, bottom-up.
    /// Recurses into `sub_layouts` first so inner matches settle before their
    /// container measures its own footprint. At each level, for every match:
    /// scan `layout_nodes` for non-child nodes whose rounded position falls
    /// inside the footprint and bump them along the axis with the smallest
    /// exit distance — +Y for Y-overlap, ±X toward the near footprint edge,
    /// −Z toward the sub-sink side (never +Z; that side faces the parent
    /// wall). Uses `move_node_delta` for each bump so cascading collisions
    /// resolve automatically. Iterates until stable (128-step cap).
    pub fn settle_matches(&self) -> Self {
        let settled_subs: std::collections::HashMap<crate::ast::node::Id, LayoutAst> = self
            .sub_layouts
            .iter()
            .map(|(k, v)| (k.clone(), v.settle_matches()))
            .collect();
        let mut layout = Self {
            ast: self.ast.clone(),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: settled_subs,
        };
        let match_ids: Vec<crate::ast::node::Id> = layout
            .layout_nodes
            .keys()
            .filter(|id| layout.is_matchnew(id))
            .cloned()
            .collect();
        for match_id in &match_ids {
            for _ in 0..128 {
                let Some(bbox) = layout.matchnew_footprint(match_id) else {
                    break;
                };
                let pattern_ids = layout.match_pattern_ids(match_id);
                let intruder = layout.layout_nodes.iter().find_map(|(id, ln)| {
                    if id == match_id || pattern_ids.contains(id) {
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
                let center_x = (bbox.min.x + bbox.max.x) / 2;
                let push_x = if ipos.x <= center_x {
                    bbox.min.x - 1 - ipos.x
                } else {
                    bbox.max.x + 1 - ipos.x
                };
                let push_y = bbox.max.y + 1 - ipos.y;
                let push_z = bbox.min.z - 1 - ipos.z;
                // VarDecls are pinned to Y=0, Z=0 → only X-push can succeed.
                let is_var_decl = matches!(
                    layout.ast.nodes.get(&intruder_id),
                    Some(crate::ast::node::ENode::VarDecl { .. })
                );
                // Priority: Z (deeper) → X (sideways) → Y (upward). Y is a
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
        layout
    }

    /// Adjust a Y-direction delta so the mover jumps over the entire Y range
    /// of any Match footprint it would land on. Cascades through nested
    /// matches. For X/Z deltas or non-Match collisions the nominal delta
    /// is returned unchanged. Same-match sibling collisions also pass through
    /// unchanged so an intra-stack Y-swap can happen.
    fn jump_delta(
        &self,
        occupancy: &std::collections::HashMap<IVec3, crate::ast::node::Id>,
        mover: &crate::ast::node::Id,
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
            let match_id = if self.is_matchnew(occ) {
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
            let Some(bbox) = self.matchnew_footprint(&match_id) else {
                return target - origin;
            };
            target.y = if step > 0.0 {
                bbox.max.y as f32 + step
            } else {
                bbox.min.y as f32 + step
            };
        }
        nominal_delta
    }

    /// Compute the co-moving group for a seed node.
    /// - MatchNew seed: seed plus all its Pattern children with the same
    ///   delta (rigid block; sub-layouts follow via Pattern-local coords).
    /// - Non-Pattern seed: `{seed}`.
    /// - Pattern with Y-only delta: `{seed}` (row change is per-pattern).
    /// - Pattern with XZ delta: seed plus all sibling Patterns of the same
    ///   match, all with the XZ delta (siblings without the Y component).
    fn move_group(
        &self,
        seed: &crate::ast::node::Id,
        seed_delta: Vec3,
    ) -> Vec<(crate::ast::node::Id, Vec3)> {
        if self.is_matchnew(seed) {
            let mut group: Vec<(crate::ast::node::Id, Vec3)> = vec![(seed.clone(), seed_delta)];
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
        let mut group: Vec<(crate::ast::node::Id, Vec3)> = self
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

    pub fn plus_edge(&self, from: crate::ast::AnchorId, to: crate::ast::AnchorId) -> Self {
        Self {
            ast: self.ast.plus_edge(from, to),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    pub fn plus_sink_wall(&self) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: input_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, Vec3::new(0.0, 0.0, -4.0))
    }

    pub fn plus_type_introduction(&self, r#type: crate::ast::node::EType, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::TypeIntroduction {
            r#type,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    pub fn plus_type_elimination(&self, r#type: crate::ast::node::EType, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::TypeElimination {
            r#type,
            input_anchor: input_anchor_id,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    pub fn plus_function_call(
        &self,
        function_declaration: (
            crate::ast::FunctionDeclarationId,
            &crate::ast::FunctionDeclaration,
        ),
        pos: Vec3,
    ) -> Self {
        let (ast, input_anchor_ids) =
            function_declaration
                .1
                .inputs
                .iter()
                .fold::<(crate::ast::Ast, Vec<crate::ast::AnchorId>), _>(
                    (self.ast.clone(), vec![]),
                    |(ast, input_anchor_ids), _| {
                        let (ast, new_anchor_id) = ast.with_next_anchor_id();
                        (
                            ast,
                            input_anchor_ids
                                .into_iter()
                                .chain(vec![new_anchor_id])
                                .collect(),
                        )
                    },
                );
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::FunctionCall {
            function_declaration_id: function_declaration.0,
            input_anchors: input_anchor_ids,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    pub fn with_function_call_replaced(
        &self,
        node_id: &crate::ast::node::Id,
        new_fn: (
            crate::ast::FunctionDeclarationId,
            &crate::ast::FunctionDeclaration,
        ),
    ) -> Self {
        let pos = self.layout_nodes.get(node_id).unwrap().pos;
        self.minus_node(node_id).plus_function_call(new_fn, pos)
    }

    pub fn plus_match_front(&self, pos: Vec3) -> Self {
        let (ast, node_id) = self.ast.plus(crate::ast::node::ENode::MatchFront {
            levels: 3,
            width: 2,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    /// Create a `MatchNew` container plus its initial `Pattern` child at `pos`.
    /// The MatchNew's synthetic LayoutNode mirrors the lowest Pattern's pos so
    /// rendering can iterate `layout_nodes` uniformly. The Pattern is created
    /// with a fresh sub-AST (single SinkWall) and a matching entry in
    /// `sub_layouts[pattern_id]` positioning that sink at Pattern-local (0,0,-1).
    pub fn plus_match_new(&self, pos: Vec3) -> Self {
        let (ast, match_input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, pattern_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, match_node_id) = ast.plus(crate::ast::node::ENode::MatchNew {
            patterns: vec![],
            input_anchor: match_input_anchor_id.clone(),
        });
        // Sub-AST bootstrap MUST happen after the Pattern parent-id is
        // reserved so the sub-AST's counter starts past it and every id
        // in the tree stays globally unique.
        let (ast, pattern_node_id) = ast.plus(crate::ast::node::ENode::Pattern {
            parent_match: match_node_id.clone(),
            r#type: crate::ast::node::EType::Int { value: None },
            output_anchor: pattern_output_anchor_id,
        });
        let (ast, pattern_sub_ast, sub_sink_id) =
            crate::ast::Ast::initial_pattern_sub_ast_from(ast);
        let pattern_sub_layout = Self::initial_pattern_sub_layout(&pattern_sub_ast, &sub_sink_id);
        let ast = ast.with_node_replaced(
            &match_node_id,
            crate::ast::node::ENode::MatchNew {
                patterns: vec![pattern_node_id.clone()],
                input_anchor: match_input_anchor_id,
            },
        );
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self
                .sub_layouts
                .clone()
                .into_iter()
                .chain([(pattern_node_id.clone(), pattern_sub_layout)])
                .collect(),
        }
        ._plus_layout_node(&pattern_node_id, pos)
        ._plus_layout_node(&match_node_id, pos)
    }

    /// LayoutAst for a fresh Pattern's sub-AST: registers the initial sink at
    /// Pattern-local grid position (0, 0, -1).
    fn initial_pattern_sub_layout(
        sub_ast: &crate::ast::Ast,
        sub_sink_id: &crate::ast::node::Id,
    ) -> Self {
        Self {
            ast: sub_ast.clone(),
            layout_nodes: std::collections::HashMap::from([(
                sub_sink_id.clone(),
                LayoutNode {
                    node_id: sub_sink_id.clone(),
                    pos: Vec3::new(0.0, 0.0, -1.0),
                },
            )]),
            sub_layouts: std::collections::HashMap::new(),
        }
    }

    /// Insert a new Pattern into `selected_pattern_id`'s parent MatchNew,
    /// occupying the selected slot. Sibling Patterns of the same match at
    /// Y ≥ selected.Y shift up by 1 to make room; the growing match footprint
    /// then displaces any external neighbors via `settle_matches`. The caller
    /// bumps `pick.selected_pos.y` by 1 so selection tracks the originally-
    /// selected Pattern (now one row higher).
    pub fn plus_pattern_above(&self, selected_pattern_id: &crate::ast::node::Id) -> Self {
        let (parent_id, selected_pos) = match self.ast.nodes.get(selected_pattern_id) {
            Some(crate::ast::node::ENode::Pattern { parent_match, .. }) => {
                let ln = self.layout_nodes.get(selected_pattern_id).unwrap();
                (parent_match.clone(), ln.pos)
            }
            _ => {
                return Self {
                    ast: self.ast.clone(),
                    layout_nodes: self.layout_nodes.clone(),
                    sub_layouts: self.sub_layouts.clone(),
                }
            }
        };
        let selected_y = selected_pos.y;
        let column_x = selected_pos.x;
        let column_z = selected_pos.z;
        // Shift only sibling Patterns of the same match at Y ≥ selected.Y.
        // External neighbours are handled by settle_matches once the new
        // Pattern has grown the footprint.
        let match_pattern_ids: std::collections::HashSet<crate::ast::node::Id> =
            self.match_pattern_ids(&parent_id).into_iter().collect();
        let shifted_layout_nodes = self
            .layout_nodes
            .iter()
            .map(|(id, ln)| {
                if !match_pattern_ids.contains(id) {
                    return (id.clone(), ln.clone());
                }
                if ln.pos.y >= selected_y - 0.001 {
                    (
                        id.clone(),
                        LayoutNode {
                            node_id: id.clone(),
                            pos: ln.pos + Vec3::new(0.0, 1.0, 0.0),
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
        let sibling_ids: Vec<crate::ast::node::Id> = match shifted.ast.nodes.get(&parent_id) {
            Some(crate::ast::node::ENode::MatchNew { patterns, .. }) => patterns.clone(),
            _ => vec![],
        };
        let (ast, new_output_anchor_id) = shifted.ast.with_next_anchor_id();
        // Reserve Pattern id first, then bootstrap sub-AST off the bumped
        // parent counter (see plus_match_new for the id-uniqueness rule).
        let (ast, new_pattern_id) = ast.plus(crate::ast::node::ENode::Pattern {
            parent_match: parent_id.clone(),
            r#type: crate::ast::node::EType::Int { value: None },
            output_anchor: new_output_anchor_id,
        });
        let (ast, new_pattern_sub_ast, new_sub_sink_id) =
            crate::ast::Ast::initial_pattern_sub_ast_from(ast);
        let new_pattern_sub_layout =
            Self::initial_pattern_sub_layout(&new_pattern_sub_ast, &new_sub_sink_id);
        let new_patterns: Vec<crate::ast::node::Id> = sibling_ids
            .iter()
            .cloned()
            .chain([new_pattern_id.clone()])
            .collect();
        let match_input_anchor = match ast.nodes.get(&parent_id) {
            Some(crate::ast::node::ENode::MatchNew { input_anchor, .. }) => input_anchor.clone(),
            _ => return shifted,
        };
        let ast = ast.with_node_replaced(
            &parent_id,
            crate::ast::node::ENode::MatchNew {
                patterns: new_patterns,
                input_anchor: match_input_anchor,
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
        ._plus_layout_node(&new_pattern_id, Vec3::new(column_x, selected_y, column_z));
        let match_ids: Vec<crate::ast::node::Id> = with_new
            .ast
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                if matches!(n, crate::ast::node::ENode::MatchNew { .. }) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        match_ids
            .iter()
            .fold(with_new, |acc, mid| acc.recompute_matchnew_pos(mid))
    }

    /// Refresh a MatchNew's synthetic LayoutNode to sit at the lowest sibling
    /// Pattern's grid position (needed after any add/remove/shift of Patterns
    /// so the render pass finds the container at the correct origin).
    pub fn recompute_matchnew_pos(&self, match_id: &crate::ast::node::Id) -> Self {
        let pattern_ids: Vec<crate::ast::node::Id> = match self.ast.nodes.get(match_id) {
            Some(crate::ast::node::ENode::MatchNew { patterns, .. }) => patterns.clone(),
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
        let Some(new_pos) = lowest_pos else {
            return Self {
                ast: self.ast.clone(),
                layout_nodes: self.layout_nodes.clone(),
                sub_layouts: self.sub_layouts.clone(),
            };
        };
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

    fn _with_matchnew_patterns(
        &self,
        match_id: &crate::ast::node::Id,
        new_patterns: Vec<crate::ast::node::Id>,
    ) -> Self {
        let input_anchor = match self.ast.nodes.get(match_id) {
            Some(crate::ast::node::ENode::MatchNew { input_anchor, .. }) => input_anchor.clone(),
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
                crate::ast::node::ENode::MatchNew {
                    patterns: new_patterns,
                    input_anchor,
                },
            ),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    pub fn plus_var_decl(&self, pos: Vec3) -> Self {
        // VarDecls live only on the Program wall (Y=0, Z=0). Snap defensively.
        let pos = Vec3::new(pos.x, 0.0, 0.0);
        let (ast, output_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::VarDecl {
            name: "v".to_string(),
            r#type: crate::ast::node::EType::Any,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    fn _plus_layout_node(&self, node_id: &crate::ast::node::Id, pos: Vec3) -> Self {
        Self {
            ast: self.ast.clone(),
            layout_nodes: self
                .layout_nodes
                .clone()
                .into_iter()
                .chain([(
                    node_id.clone(),
                    LayoutNode {
                        node_id: node_id.clone(),
                        pos,
                    },
                )])
                .collect(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    /// Recursively walk every node in this LayoutAst and its `sub_layouts`.
    /// Each entry carries the containing LayoutAst (for anchor lookups), the
    /// LayoutNode, the accumulated grid-space offset from the outer root, and
    /// a sink-scale hint (1.0 outside patterns, 1/3 inside).
    ///
    /// Descent into a sub-layout uses the owner node's grid position as the
    /// additional offset. The outer root's `layout_nodes` is expected to be
    /// empty (Program has no LayoutNode) so `sub_layouts[program_id]` is
    /// entered with offset (0,0,0) and scale 1.
    pub fn walk_all(&self) -> Vec<WalkedNode> {
        let mut out = Vec::new();
        self.walk_all_into(Vec3::ZERO, 1.0, &mut out);
        out
    }

    fn walk_all_into<'a>(&'a self, offset: Vec3, sink_scale: f32, out: &mut Vec<WalkedNode<'a>>) {
        for layout_node in self.layout_nodes.values() {
            out.push(WalkedNode {
                layout_ast: self,
                layout_node,
                extra_offset: offset,
                sink_scale,
            });
        }
        for (owner_id, sub_layout) in &self.sub_layouts {
            let owner_grid_pos = self
                .layout_nodes
                .get(owner_id)
                .map(|ln| ln.pos)
                .unwrap_or(Vec3::ZERO);
            let sub_offset = offset + owner_grid_pos;
            let sub_scale = if matches!(
                self.ast.nodes.get(owner_id),
                Some(crate::ast::node::ENode::Pattern { .. })
            ) {
                1.0 / 3.0
            } else {
                sink_scale
            };
            sub_layout.walk_all_into(sub_offset, sub_scale, out);
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
        context: Vec<crate::ast::node::Id>,
        offset: Vec3,
        out: &mut Vec<WalkedAst<'a>>,
    ) {
        out.push(WalkedAst {
            layout_ast: self,
            context: context.clone(),
            extra_offset: offset,
        });
        for (owner_id, sub_layout) in &self.sub_layouts {
            let owner_grid_pos = self
                .layout_nodes
                .get(owner_id)
                .map(|ln| ln.pos)
                .unwrap_or(Vec3::ZERO);
            let mut sub_context = context.clone();
            sub_context.push(owner_id.clone());
            sub_layout.walk_all_asts_into(sub_context, offset + owner_grid_pos, out);
        }
    }

    /// Compute the grid bounds for this AST in its own local coordinates.
    ///
    /// - Z range: `[sink.z + 1, -1]` where `sink.z` is the single SinkWall's
    ///   layout-Z. The front wall is treated as Z=0 for both the Program-Ast
    ///   (which has no explicit front node) and Pattern sub-ASTs (where the
    ///   Pattern itself is the implicit front). If the corridor collapses
    ///   (`sink.z >= -1`) or no SinkWall exists, returns `None`.
    /// - X range: min/max over all layout nodes' rounded X, unioned with
    ///   `active_selection.x` when provided. If the AST has no nodes at all
    ///   the sink is the only node — X min/max = 0 (sink is at X=0).
    ///
    /// `active_selection` is the currently-selected cell in this AST's local
    /// coords, passed only when this AST is the active editing context.
    pub fn ast_grid_bounds(&self, active_selection: Option<IVec3>) -> Option<AstGridBounds> {
        let sink_z =
            self.layout_nodes
                .iter()
                .find_map(|(id, ln)| match self.ast.nodes.get(id) {
                    Some(crate::ast::node::ENode::SinkWall { .. }) => {
                        Some(ln.pos.round().as_ivec3().z)
                    }
                    _ => None,
                })?;
        let z_min = sink_z + 1;
        let z_max = -1;
        if z_min > z_max {
            return None;
        }
        let mut x_min = i32::MAX;
        let mut x_max = i32::MIN;
        for ln in self.layout_nodes.values() {
            let x = ln.pos.round().as_ivec3().x;
            if x < x_min {
                x_min = x;
            }
            if x > x_max {
                x_max = x;
            }
        }
        if let Some(sel) = active_selection {
            if sel.x < x_min {
                x_min = sel.x;
            }
            if sel.x > x_max {
                x_max = sel.x;
            }
        }
        if x_min == i32::MAX {
            x_min = 0;
            x_max = 0;
        }
        Some(AstGridBounds {
            min: IVec3::new(x_min, 0, z_min),
            max: IVec3::new(x_max, 0, z_max),
        })
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

    pub fn layout_anchor(&self, anchor_id: crate::ast::AnchorId) -> LayoutAnchor {
        self.try_layout_anchor(&anchor_id).unwrap_or_else(|| {
            panic!(
                "layout_anchor: anchor {:?} not found in any (sub-)ast",
                anchor_id
            )
        })
    }

    fn try_layout_anchor(&self, anchor_id: &crate::ast::AnchorId) -> Option<LayoutAnchor> {
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

    /// Return the `EType` an output anchor produces, or an input anchor
    /// expects — whichever the anchor happens to be. Walks sub-layouts.
    /// `None` for anchors that carry no type (SinkWall input, MatchBack, …).
    pub fn anchor_type(
        &self,
        anchor_id: &crate::ast::AnchorId,
        function_declarations: &std::collections::HashMap<
            crate::ast::FunctionDeclarationId,
            crate::ast::FunctionDeclaration,
        >,
    ) -> Option<crate::eval::EType> {
        if let Some(node_id) = self.ast.anchor_to_node.get(anchor_id) {
            let node = self.ast.nodes.get(node_id)?;
            return anchor_type_from_node(node, anchor_id, function_declarations);
        }
        for sub in self.sub_layouts.values() {
            if let Some(t) = sub.anchor_type(anchor_id, function_declarations) {
                return Some(t);
            }
        }
        None
    }

    /// AST-level literal string attached to this anchor's type, if any.
    /// Walks sub-layouts. Returns `None` when the anchor's node kind doesn't
    /// carry an `ast::node::EType` field, or the field's value is `None`.
    pub fn anchor_ast_value(&self, anchor_id: &crate::ast::AnchorId) -> Option<String> {
        if let Some(node_id) = self.ast.anchor_to_node.get(anchor_id) {
            let node = self.ast.nodes.get(node_id)?;
            return anchor_ast_value_from_node(node, anchor_id);
        }
        for sub in self.sub_layouts.values() {
            if let Some(v) = sub.anchor_ast_value(anchor_id) {
                return Some(v);
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
        target: &crate::ast::node::Id,
    ) -> Option<Vec<crate::ast::node::Id>> {
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
    pub fn find_node_ast_mut(&mut self, target: &crate::ast::node::Id) -> Option<&mut LayoutAst> {
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
    pub fn resolve_context<'a>(&'a self, path: &[crate::ast::node::Id]) -> &'a LayoutAst {
        let mut ast = self;
        for id in path {
            ast = ast.sub_layouts.get(id).unwrap();
        }
        ast
    }

    /// Sum of grid-space owner positions along `path`. Used by crosshair to
    /// place its anchor for sub-AST-selected nodes when no rendered entity
    /// is around to read from.
    pub fn context_offset(&self, path: &[crate::ast::node::Id]) -> Vec3 {
        let mut offset = Vec3::ZERO;
        let mut ast = self;
        for id in path {
            if let Some(ln) = ast.layout_nodes.get(id) {
                offset += ln.pos;
            }
            let Some(next) = ast.sub_layouts.get(id) else {
                break;
            };
            ast = next;
        }
        offset
    }
}

fn anchor_type_from_node(
    node: &crate::ast::node::ENode,
    anchor_id: &crate::ast::AnchorId,
    function_declarations: &std::collections::HashMap<
        crate::ast::FunctionDeclarationId,
        crate::ast::FunctionDeclaration,
    >,
) -> Option<crate::eval::EType> {
    match node {
        crate::ast::node::ENode::TypeIntroduction { r#type, .. }
        | crate::ast::node::ENode::VarDecl { r#type, .. }
        | crate::ast::node::ENode::Pattern { r#type, .. }
        | crate::ast::node::ENode::TypeElimination { r#type, .. } => {
            Some(crate::eval::ast_type_to_eval_type(r#type))
        }
        crate::ast::node::ENode::FunctionCall {
            function_declaration_id,
            input_anchors,
            output_anchor,
        } => {
            let decl = function_declarations.get(function_declaration_id)?;
            if anchor_id == output_anchor {
                return Some(decl.output_type.clone());
            }
            let idx = input_anchors.iter().position(|a| a == anchor_id)?;
            decl.inputs.get(idx).map(|p| p.r#type.clone())
        }
        // Anchors that don't currently carry a rendered type.
        crate::ast::node::ENode::SinkWall { .. }
        | crate::ast::node::ENode::MatchBack { .. }
        | crate::ast::node::ENode::MatchNew { .. }
        | crate::ast::node::ENode::MatchFront { .. }
        | crate::ast::node::ENode::MatchGrid { .. }
        | crate::ast::node::ENode::Program {} => None,
    }
}

/// Concrete literal string on an `ast::node::EType`, if any. Only the
/// value-carrying variants have an `Option<String>`; `Any`, `Undefined`,
/// `Exception` return `None`.
pub fn value_of_etype(t: &crate::ast::node::EType) -> Option<String> {
    match t {
        crate::ast::node::EType::Bool { value }
        | crate::ast::node::EType::Int { value }
        | crate::ast::node::EType::Float { value }
        | crate::ast::node::EType::String { value }
        | crate::ast::node::EType::Char { value } => value.clone(),
        _ => None,
    }
}

/// Return the AST-level literal attached to `anchor_id`'s type, if any.
///
/// Only anchors on nodes whose type is a first-class `ast::node::EType` field
/// (TypeIntroduction, VarDecl, Pattern, TypeElimination) can carry a literal.
/// FunctionCall inputs/outputs bind to `eval::EType` on the declaration, which
/// has no AST-level literal — treated as `None`. Match / SinkWall / Program
/// don't carry types at all.
pub fn anchor_ast_value_from_node(
    node: &crate::ast::node::ENode,
    anchor_id: &crate::ast::AnchorId,
) -> Option<String> {
    match node {
        crate::ast::node::ENode::TypeIntroduction {
            r#type,
            output_anchor,
        }
        | crate::ast::node::ENode::VarDecl {
            r#type,
            output_anchor,
            ..
        }
        | crate::ast::node::ENode::Pattern {
            r#type,
            output_anchor,
            ..
        } => {
            if anchor_id == output_anchor {
                value_of_etype(r#type)
            } else {
                None
            }
        }
        crate::ast::node::ENode::TypeElimination {
            r#type,
            input_anchor,
            output_anchor,
        } => {
            if anchor_id == input_anchor || anchor_id == output_anchor {
                value_of_etype(r#type)
            } else {
                None
            }
        }
        _ => None,
    }
}
