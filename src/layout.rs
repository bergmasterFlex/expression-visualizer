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

pub struct LayoutAst {
    pub ast: crate::ast::Ast,
    pub layout_nodes: std::collections::HashMap<crate::ast::node::Id, LayoutNode>,
}

impl LayoutAst {
    pub fn empty() -> Self {
        Self {
            ast: crate::ast::Ast::empty(),
            layout_nodes: std::collections::HashMap::new(),
        }
    }

    pub fn minus_node(&self, node_id: &crate::ast::node::Id) -> Self {
        Self {
            ast: self.ast.minus(node_id),
            layout_nodes: self
                .layout_nodes
                .clone()
                .into_iter()
                .filter(|(id, _)| id != node_id)
                .collect(),
        }
    }

    pub fn move_node_delta(&self, node_id: crate::ast::node::Id, delta_pos: Vec3) -> Self {
        Self {
            ast: self.ast.clone(),
            layout_nodes: self
                .layout_nodes
                .iter()
                .map(|(id, layout_node)| {
                    (
                        id.clone(),
                        if *id == node_id {
                            LayoutNode {
                                node_id: id.clone(),
                                pos: layout_node.pos + delta_pos,
                            }
                        } else {
                            layout_node.clone()
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn plus_edge(&self, from: crate::ast::AnchorId, to: crate::ast::AnchorId) -> Self {
        Self {
            ast: self.ast.plus_edge(from, to),
            layout_nodes: self.layout_nodes.clone(),
        }
    }

    pub fn plus_sink(&self) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::Sink {
            input_anchor: input_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
        }
        ._plus_layout_node(&node_id, Vec3::new(0.0, 0.0, 0.0))
    }

    pub fn plus_sink_wall(&self) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::SinkWall {
            input_anchor: input_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
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
        let (ast, mb1_input_anchor_ids) =
            (0..mb_levels).fold((ast, Vec::<crate::ast::AnchorId>::new()), |(ast, mut acc), _| {
                let (ast, anchor_id) = ast.with_next_anchor_id();
                acc.push(anchor_id);
                (ast, acc)
            });
        let (ast, mb1_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, mb1_node_id) = ast.plus(crate::ast::node::ENode::MatchBack {
            levels: mb_levels,
            input_anchors: mb1_input_anchor_ids.clone(),
            output_anchor: mb1_output_anchor_id.clone(),
        });
        let (ast, mb2_input_anchor_ids) =
            (0..mb_levels).fold((ast, Vec::<crate::ast::AnchorId>::new()), |(ast, mut acc), _| {
                let (ast, anchor_id) = ast.with_next_anchor_id();
                acc.push(anchor_id);
                (ast, acc)
            });
        let (ast, mb2_output_anchor_id) = ast.with_next_anchor_id();
        let (ast, mb2_node_id) = ast.plus(crate::ast::node::ENode::MatchBack {
            levels: mb_levels,
            input_anchors: mb2_input_anchor_ids.clone(),
            output_anchor: mb2_output_anchor_id.clone(),
        });
        let (ast, mg1_node_id) = ast.plus(crate::ast::node::ENode::MatchGrid {
            width: 3,
            depth: 2,
        });
        let (ast, mg2_node_id) = ast.plus(crate::ast::node::ENode::MatchGrid {
            width: 3,
            depth: 2,
        });
        let (ast, mg3_node_id) = ast.plus(crate::ast::node::ENode::MatchGrid {
            width: 3,
            depth: 2,
        });
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
        let ast = ast.plus_edge(te_bot_output_anchor_id.clone(), mb1_input_anchor_ids[0].clone());
        let ast = ast.plus_edge(te_bot_output_anchor_id, mb2_input_anchor_ids[0].clone());
        let ast = ast.plus_edge(te_mid_output_anchor_id.clone(), mb1_input_anchor_ids[1].clone());
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
        }
        ._plus_layout_node(&node_id, pos)
    }

    pub fn plus_match(&self, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::node::ENode::Match {
            input_anchor: input_anchor_id,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
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
        }
    }

    pub fn edges(&self) -> Vec<LayoutEdge> {
        self.ast
            .edges
            .iter()
            .flat_map(|(from_anchor_id, to_anchor_ids)| {
                to_anchor_ids
                    .clone()
                    .into_iter()
                    .map(|to_anchor_id| LayoutEdge {
                        from_anchor: self.layout_anchor(from_anchor_id.clone()),
                        to_anchor: self.layout_anchor(to_anchor_id.clone()),
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
}

/// Spacing constants for the 3D layout.
const SPACING_X: f32 = 2.0;
const SPACING_Y: f32 = 2.0;
const SPACING_Z: f32 = 3.5;
