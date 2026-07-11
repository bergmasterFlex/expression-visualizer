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
    pub color: Color,
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
        let (ast, program_id) = crate::ast::Ast::empty().plus(crate::ast::node::ENode::Program {
            ast: crate::ast::Ast::empty(),
        });
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
    /// Excludes `MatchNew` containers (same rule as `node_at`).
    fn occupancy_map(&self) -> std::collections::HashMap<IVec3, crate::ast::node::Id> {
        self.layout_nodes
            .iter()
            .filter_map(|(id, ln)| {
                if matches!(
                    self.ast.nodes.get(id),
                    Some(crate::ast::node::ENode::MatchNew { .. })
                ) {
                    return None;
                }
                Some((ln.pos.round().as_ivec3(), id.clone()))
            })
            .collect()
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
            let swap_delta = self.jump_delta(&occupancy, occ, occ_origin, -cur_delta);
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

    /// Adjust a Y-direction delta so the mover jumps over the entire Y range
    /// of any match whose pattern it would land on. Cascades through nested
    /// matches. For X/Z deltas or non-pattern collisions the nominal delta
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
            if occ == mover || !self.is_pattern(occ) {
                return target - origin;
            }
            if self.is_pattern(mover) && self.parent_match_of(mover) == self.parent_match_of(occ) {
                return target - origin;
            }
            let Some(match_id) = self.parent_match_of(occ) else {
                return target - origin;
            };
            let ys: Vec<f32> = self
                .match_pattern_ids(&match_id)
                .iter()
                .filter_map(|pid| self.layout_nodes.get(pid).map(|ln| ln.pos.y))
                .collect();
            if ys.is_empty() {
                return target - origin;
            }
            let extreme = if step > 0.0 {
                ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            } else {
                ys.iter().cloned().fold(f32::INFINITY, f32::min)
            };
            target.y = extreme + step;
        }
        nominal_delta
    }

    /// Compute the co-moving group for a seed node.
    /// Non-Pattern: just `{seed}`. Pattern with Y-only delta: `{seed}` (row
    /// change is per-pattern). Pattern with XZ delta: seed plus all sibling
    /// patterns of the same match, all with the XZ delta (siblings without
    /// the Y component).
    fn move_group(
        &self,
        seed: &crate::ast::node::Id,
        seed_delta: Vec3,
    ) -> Vec<(crate::ast::node::Id, Vec3)> {
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
        self.match_pattern_ids(&match_id)
            .into_iter()
            .map(|sid| {
                let d = if sid == *seed {
                    seed_delta
                } else {
                    Vec3::new(seed_delta.x, 0.0, seed_delta.z)
                };
                (sid, d)
            })
            .collect()
    }

    pub fn plus_edge(&self, from: crate::ast::AnchorId, to: crate::ast::AnchorId) -> Self {
        Self {
            ast: self.ast.plus_edge(from, to),
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
    }

    pub fn plus_edge_colored(
        &self,
        from: crate::ast::AnchorId,
        to: crate::ast::AnchorId,
        color: Color,
    ) -> Self {
        Self {
            ast: self.ast.plus_edge_colored(from, to, color),
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

    /// Example scene for the "Sink" example button: a SinkWall connected
    /// to an Int(3) TypeIntroduction node sitting at the origin.
    pub fn plus_sink_example(&self) -> Self {
        let (ast, sink_input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, sink_node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: sink_input_anchor_id.clone(),
        });
        let (ast, ti_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, ti_node_id) = ast.plus(crate::ast::node::ENode::TypeIntroduction {
            r#type: crate::ast::node::EType::Int {
                value: Some("3".to_string()),
            },
            output_anchor: ti_output_anchor_id.clone(),
        });
        let ast = ast.plus_edge(ti_output_anchor_id, sink_input_anchor_id);
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&sink_node_id, Vec3::new(0.0, 0.0, -4.0))
        ._plus_layout_node(&ti_node_id, Vec3::new(0.0, 0.0, 0.0))
    }

    /// Example scene for the "ConstDecl" example button: a SinkWall connected
    /// to a Float(3.141) TypeIntroduction node sitting at the origin.
    pub fn plus_constdecl_example(&self) -> Self {
        let (ast, sink_input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, sink_node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: sink_input_anchor_id.clone(),
        });
        let (ast, ti_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, ti_node_id) = ast.plus(crate::ast::node::ENode::TypeIntroduction {
            r#type: crate::ast::node::EType::Float {
                value: Some("3.141".to_string()),
            },
            output_anchor: ti_output_anchor_id.clone(),
        });
        let ast = ast.plus_edge(ti_output_anchor_id, sink_input_anchor_id);
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&sink_node_id, Vec3::new(0.0, 0.0, -4.0))
        ._plus_layout_node(&ti_node_id, Vec3::new(0.0, 0.0, 0.0))
    }

    /// Example scene for the "FuncCall" example button: a SinkWall at the back
    /// wall, a FunctionCall node in the centre using the given declaration
    /// (intended: `charAt`), and one TypeIntroduction per input wired in front
    /// of the call. For inputs named `str` / `i` the TypeIntroductions get the
    /// fixed sample values `"Hello World"` / `4`; any other parameter falls
    /// back to a valueless `Any`.
    pub fn plus_funccall_example(
        &self,
        function_declaration: (
            crate::ast::FunctionDeclarationId,
            &crate::ast::FunctionDeclaration,
        ),
    ) -> Self {
        let (ast, sink_input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, sink_node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: sink_input_anchor_id.clone(),
        });

        let inputs = &function_declaration.1.inputs;
        let (ast, fc_input_anchor_ids) = inputs.iter().fold(
            (ast, Vec::<crate::ast::AnchorId>::new()),
            |(ast, mut acc), _| {
                let (ast, anchor_id) = ast.with_next_anchor_id();
                acc.push(anchor_id);
                (ast, acc)
            },
        );
        let (ast, fc_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, fc_node_id) = ast.plus(crate::ast::node::ENode::FunctionCall {
            function_declaration_id: function_declaration.0,
            input_anchors: fc_input_anchor_ids.clone(),
            output_anchor: fc_output_anchor_id.clone(),
        });
        let ast = ast.plus_edge(fc_output_anchor_id, sink_input_anchor_id);

        let input_count = inputs.len();
        let (ast, ti_layout_specs) = inputs
            .iter()
            .zip(fc_input_anchor_ids.into_iter())
            .enumerate()
            .fold(
                (ast, Vec::<(crate::ast::node::Id, f32)>::new()),
                |(ast, mut acc), (i, (param, fc_input_anchor_id))| {
                    let ty = match param.name.as_str() {
                        "str" => crate::ast::node::EType::String {
                            value: Some("Hello World".to_string()),
                        },
                        "i" => crate::ast::node::EType::Int {
                            value: Some("4".to_string()),
                        },
                        _ => crate::ast::node::EType::Any,
                    };
                    let (ast, ti_output_anchor_id) = ast.with_next_anchor_id();
                    let (ast, ti_node_id) = ast.plus(crate::ast::node::ENode::TypeIntroduction {
                        r#type: ty,
                        output_anchor: ti_output_anchor_id.clone(),
                    });
                    let ast = ast.plus_edge(ti_output_anchor_id, fc_input_anchor_id);
                    let x = (i as f32) * 2.0 - (input_count as f32 - 1.0);
                    acc.push((ti_node_id, x));
                    (ast, acc)
                },
            );

        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&sink_node_id, Vec3::new(0.0, 0.0, -4.0))
        ._plus_layout_node(&fc_node_id, Vec3::new(0.0, 0.0, 0.0));

        ti_layout_specs
            .into_iter()
            .fold(layout, |layout, (node_id, x)| {
                layout._plus_layout_node(&node_id, Vec3::new(x, 0.0, 4.0))
            })
    }

    /// Example scene for the "Match" example button: a MatchFront wall, two
    /// MatchBack pyramids, and three vertically stacked MatchGrids. The nodes
    /// have no input/output anchors — purely visual layout containers.
    pub fn plus_match_example(&self) -> Self {
        let (ast, sink_input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, sink_node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: sink_input_anchor_id.clone(),
        });
        let (ast, mf_node_id) = ast.plus(crate::ast::node::ENode::MatchFront {
            levels: 3,
            width: 2,
        });
        let mb_levels = 3usize;
        let (ast, mb1_input_anchor_ids) = (0..mb_levels).fold(
            (ast, Vec::<crate::ast::AnchorId>::new()),
            |(ast, mut acc), _| {
                let (ast, anchor_id) = ast.with_next_anchor_id();
                acc.push(anchor_id);
                (ast, acc)
            },
        );
        let (ast, mb1_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, mb1_node_id) = ast.plus(crate::ast::node::ENode::MatchBack {
            levels: mb_levels,
            input_anchors: mb1_input_anchor_ids.clone(),
            output_anchor: mb1_output_anchor_id.clone(),
        });
        let (ast, mb2_input_anchor_ids) = (0..mb_levels).fold(
            (ast, Vec::<crate::ast::AnchorId>::new()),
            |(ast, mut acc), _| {
                let (ast, anchor_id) = ast.with_next_anchor_id();
                acc.push(anchor_id);
                (ast, acc)
            },
        );
        let (ast, mb2_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, mb2_node_id) = ast.plus(crate::ast::node::ENode::MatchBack {
            levels: mb_levels,
            input_anchors: mb2_input_anchor_ids.clone(),
            output_anchor: mb2_output_anchor_id.clone(),
        });
        let (ast, mg1_node_id) =
            ast.plus(crate::ast::node::ENode::MatchGrid { width: 3, depth: 2 });
        let (ast, mg2_node_id) =
            ast.plus(crate::ast::node::ENode::MatchGrid { width: 3, depth: 2 });
        let (ast, mg3_node_id) =
            ast.plus(crate::ast::node::ENode::MatchGrid { width: 3, depth: 2 });
        let (ast, vd_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, vd_node_id) = ast.plus(crate::ast::node::ENode::VarDecl {
            name: "s".to_string(),
            r#type: crate::ast::node::EType::String { value: None },
            output_anchor: vd_output_anchor_id.clone(),
        });
        let (ast, te_top_input_anchor_id) = ast.with_next_anchor_id();
        let (ast, te_top_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, te_top_node_id) = ast.plus(crate::ast::node::ENode::TypeElimination {
            r#type: crate::ast::node::EType::Int { value: None },
            input_anchor: te_top_input_anchor_id.clone(),
            output_anchor: te_top_output_anchor_id.clone(),
        });
        let (ast, te_mid_input_anchor_id) = ast.with_next_anchor_id();
        let (ast, te_mid_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, te_mid_node_id) = ast.plus(crate::ast::node::ENode::TypeElimination {
            r#type: crate::ast::node::EType::String {
                value: Some("World".to_string()),
            },
            input_anchor: te_mid_input_anchor_id.clone(),
            output_anchor: te_mid_output_anchor_id.clone(),
        });
        let (ast, te_bot_input_anchor_id) = ast.with_next_anchor_id();
        let (ast, te_bot_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, te_bot_node_id) = ast.plus(crate::ast::node::ENode::TypeElimination {
            r#type: crate::ast::node::EType::String { value: None },
            input_anchor: te_bot_input_anchor_id.clone(),
            output_anchor: te_bot_output_anchor_id.clone(),
        });
        let ast = ast.plus_edge(mb1_output_anchor_id, sink_input_anchor_id.clone());
        let ast = ast.plus_edge(mb2_output_anchor_id, sink_input_anchor_id);
        let ast = ast.plus_edge(vd_output_anchor_id.clone(), te_top_input_anchor_id);
        let ast = ast.plus_edge(vd_output_anchor_id.clone(), te_mid_input_anchor_id);
        let ast = ast.plus_edge(vd_output_anchor_id, te_bot_input_anchor_id);
        let ast = ast.plus_edge(
            te_bot_output_anchor_id.clone(),
            mb1_input_anchor_ids[0].clone(),
        );
        let ast = ast.plus_edge(te_bot_output_anchor_id, mb2_input_anchor_ids[0].clone());
        let ast = ast.plus_edge(
            te_mid_output_anchor_id.clone(),
            mb1_input_anchor_ids[1].clone(),
        );
        let ast = ast.plus_edge(te_mid_output_anchor_id, mb2_input_anchor_ids[1].clone());
        let ast = ast.plus_edge(te_top_output_anchor_id, mb2_input_anchor_ids[2].clone());
        let (ast, ti_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, ti_node_id) = ast.plus(crate::ast::node::ENode::TypeIntroduction {
            r#type: crate::ast::node::EType::Int {
                value: Some("3".to_string()),
            },
            output_anchor: ti_output_anchor_id.clone(),
        });
        let ast = ast.plus_edge(ti_output_anchor_id, mb1_input_anchor_ids[2].clone());
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&sink_node_id, Vec3::new(0.0, 0.0, -12.0))
        ._plus_layout_node(&mf_node_id, Vec3::new(0.0, 0.0, -2.0))
        ._plus_layout_node(&mb1_node_id, Vec3::new(0.0, 0.0, -4.0))
        ._plus_layout_node(&mb2_node_id, Vec3::new(2.0, 0.0, -4.0))
        ._plus_layout_node(&mg1_node_id, Vec3::new(0.0, 0.0, -2.0))
        ._plus_layout_node(&mg2_node_id, Vec3::new(0.0, 1.0, -2.0))
        ._plus_layout_node(&mg3_node_id, Vec3::new(0.0, 2.0, -2.0))
        ._plus_layout_node(&vd_node_id, Vec3::new(0.0, 0.0, 4.0))
        ._plus_layout_node(&te_bot_node_id, Vec3::new(0.0, 0.0, -2.0))
        ._plus_layout_node(&te_mid_node_id, Vec3::new(0.0, 2.0, -2.0))
        ._plus_layout_node(&te_top_node_id, Vec3::new(0.0, 4.0, -2.0))
        ._plus_layout_node(&ti_node_id, Vec3::new(-5.0, 0.0, 0.0))
    }

    /// Example scene for the "VarDecl" example button: a SinkWall at the back
    /// wall plane with three VarDecl nodes ("0": string, "1": int, "2": bool)
    /// sitting at the front wall's z plane, spread across x = -1, 0, +1
    /// (world x = -3, 0, +3), all wired into the SinkWall's input.
    pub fn plus_vardecl_example(&self) -> Self {
        let (ast, sink_input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, sink_node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: sink_input_anchor_id.clone(),
        });

        let vardecl_specs: Vec<(String, crate::ast::node::EType, f32)> = vec![
            (
                "0".to_string(),
                crate::ast::node::EType::String { value: None },
                -1.0,
            ),
            (
                "1".to_string(),
                crate::ast::node::EType::Int { value: None },
                0.0,
            ),
            (
                "2".to_string(),
                crate::ast::node::EType::Bool { value: None },
                1.0,
            ),
        ];

        let (ast, vardecl_nodes) = vardecl_specs.into_iter().fold(
            (ast, Vec::<(crate::ast::node::Id, f32)>::new()),
            |(ast, mut acc), (name, ty, x)| {
                let (ast, vd_output_anchor_id) = ast.with_next_anchor_id();
                let (ast, vd_node_id) = ast.plus(crate::ast::node::ENode::VarDecl {
                    name,
                    r#type: ty,
                    output_anchor: vd_output_anchor_id.clone(),
                });
                let ast = ast.plus_edge(vd_output_anchor_id, sink_input_anchor_id.clone());
                acc.push((vd_node_id, x));
                (ast, acc)
            },
        );

        let layout = Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
            sub_layouts: self.sub_layouts.clone(),
        }
        ._plus_layout_node(&sink_node_id, Vec3::new(0.0, 0.0, -4.0));

        vardecl_nodes
            .into_iter()
            .fold(layout, |layout, (node_id, x)| {
                layout._plus_layout_node(&node_id, Vec3::new(x, 0.0, 4.0))
            })
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
            // Pattern.ast is dead (Step-2 note); real sub-AST lives in
            // sub_layouts. A dummy empty ast is fine here.
            ast: crate::ast::Ast::empty(),
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

    /// Insert a new Pattern below `selected_pattern_id` in its parent MatchNew.
    /// Every node at the match's XZ column with Y ≥ selected.Y is shifted up
    /// by 1 grid unit — that includes the selected Pattern, sibling Patterns
    /// above it, AND any external nodes stacked directly above the match at
    /// the same XZ (so no external node ever gets overlapped by the growing
    /// match). The new Pattern occupies the freed slot with a default
    /// `Int { value: None }` type. The caller is expected to bump
    /// `pick.selected_pos.y` by 1 so selection tracks the originally-selected
    /// Pattern (now one row higher).
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
        let shifted_layout_nodes = self
            .layout_nodes
            .iter()
            .map(|(id, ln)| {
                if matches!(
                    self.ast.nodes.get(id),
                    Some(crate::ast::node::ENode::MatchNew { .. })
                ) {
                    return (id.clone(), ln.clone());
                }
                let same_xz =
                    (ln.pos.x - column_x).abs() < 0.5 && (ln.pos.z - column_z).abs() < 0.5;
                let above_or_at = ln.pos.y >= selected_y - 0.001;
                if same_xz && above_or_at {
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
            ast: crate::ast::Ast::empty(),
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

    pub fn edges(&self) -> Vec<LayoutEdge> {
        self.ast
            .edges
            .iter()
            .flat_map(|(from_anchor_id, edges)| {
                edges.clone().into_iter().map(|edge| LayoutEdge {
                    from_anchor: self.layout_anchor(from_anchor_id.clone()),
                    to_anchor: self.layout_anchor(edge.to.clone()),
                    color: edge.color,
                })
            })
            .collect()
    }

    pub fn layout_anchor(&self, anchor_id: crate::ast::AnchorId) -> LayoutAnchor {
        let anchor = self.ast.anchors.get(&anchor_id).unwrap();
        let node_id = self.ast.anchor_to_node.get(&anchor_id).unwrap().clone();
        LayoutAnchor {
            anchor_id: anchor_id,
            anchor: anchor.clone(),
            node_id,
            pos: Vec3::splat(1.0),
        }
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
