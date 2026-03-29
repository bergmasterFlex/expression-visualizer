use bevy::prelude::*;

use crate::ast::{AstNodeId, FunctionDeclaration, FunctionDeclarationId};

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub node_id: crate::ast::AstNodeId,
    pub pos: Vec3,
}

#[derive(Debug, Clone)]
pub struct LayoutEdge {
    pub from_node_id: crate::ast::AstNodeId,
    pub from_anchor_id: crate::ast::AnchorId,
    pub to_node_id: crate::ast::AstNodeId,
    pub to_anchor_id: crate::ast::AnchorId,
    pub from_pos: Vec3,
    pub to_pos: Vec3,
}

pub struct LayoutAst {
    pub ast: crate::ast::Ast,
    pub layout_nodes: std::collections::HashMap<crate::ast::AstNodeId, LayoutNode>,
}

impl LayoutAst {
    pub fn empty() -> Self {
        Self {
            ast: crate::ast::Ast::empty(),
            layout_nodes: std::collections::HashMap::new(),
        }
    }

    pub fn minus_node(&self, node_id: &AstNodeId) -> Self {
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

    pub fn move_node_delta(&self, node_id: AstNodeId, delta_pos: Vec3) -> Self {
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
        let (ast, node_id) = ast.plus(crate::ast::EAstNode::Sink {
            input_anchor: input_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
        }
        ._plus_layout_node(&node_id, Vec3::new(0.0, 0.0, 0.0))
    }

    pub fn plus_number_literal(&self, value: f32, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::EAstNode::NumLiteral {
            value,
            input_anchor: input_anchor_id,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    pub fn plus_bool_literal(&self, value: bool, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::EAstNode::BoolLiteral {
            value,
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
        function_declaration: (FunctionDeclarationId, &FunctionDeclaration),
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
        let (ast, node_id) = ast.plus(crate::ast::EAstNode::FunctionCall {
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

    pub fn plus_match_true(&self, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::EAstNode::MatchTrue {
            input_anchor: input_anchor_id,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    pub fn plus_match_false(&self, pos: Vec3) -> Self {
        let (ast, input_anchor_id) = self.ast.with_next_anchor_id();
        let (ast, output_anchor_id) = ast.with_next_anchor_id();
        let (ast, node_id) = ast.plus(crate::ast::EAstNode::MatchFalse {
            input_anchor: input_anchor_id,
            output_anchor: output_anchor_id,
        });
        Self {
            ast,
            layout_nodes: self.layout_nodes.clone(),
        }
        ._plus_layout_node(&node_id, pos)
    }

    fn _plus_layout_node(&self, node_id: &AstNodeId, pos: Vec3) -> Self {
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
            .map(|(from_anchor_id, to_anchor_id)| {
                let from_node_id = self.ast.anchor_to_node.get(from_anchor_id).unwrap().clone();
                let to_node_id = self.ast.anchor_to_node.get(to_anchor_id).unwrap().clone();
                LayoutEdge {
                    from_node_id: from_node_id.clone(),
                    from_anchor_id: from_anchor_id.clone(),
                    to_node_id: to_node_id.clone(),
                    to_anchor_id: to_anchor_id.clone(),
                    from_pos: self.layout_nodes.get(&from_node_id).unwrap().pos,
                    to_pos: self.layout_nodes.get(&to_node_id).unwrap().pos,
                }
            })
            .collect()
    }
}

/// Spacing constants for the 3D layout.
const SPACING_X: f32 = 2.0;
const SPACING_Y: f32 = 2.0;
const SPACING_Z: f32 = 3.5;
