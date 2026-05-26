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
