use bevy::prelude::*;

use crate::ast::{AstNodeId, FunctionDeclarationId};

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub node_id: crate::ast::AstNodeId,
    pub pos: Vec3,
}

#[derive(Debug, Clone)]
pub struct LayoutEdge {
    pub from_node_id: crate::ast::AstNodeId,
    pub to_node_id: crate::ast::AstNodeId,
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

    pub fn plus_sink(&self) -> Self {
        self._plus_node(
            crate::ast::EAstNode::Sink { input: None },
            Vec3::new(0.0, 0.0, 0.0),
        )
    }

    pub fn plus_number_literal(&self, value: f32, pos: Vec3) -> Self {
        self._plus_node(crate::ast::EAstNode::NumLiteral(value), pos)
    }

    pub fn plus_bool_literal(&self, value: bool, pos: Vec3) -> Self {
        self._plus_node(crate::ast::EAstNode::BoolLiteral(value), pos)
    }

    pub fn plus_function_call(
        &self,
        function_declaration: FunctionDeclarationId,
        pos: Vec3,
    ) -> Self {
        self._plus_node(
            crate::ast::EAstNode::FunctionCall {
                function_declaration_id: function_declaration,
                input_arguments: vec![],
            },
            pos,
        )
    }

    pub fn plus_match_true(&self, pos: Vec3) -> Self {
        self._plus_node(
            crate::ast::EAstNode::MatchTrue {
                input_argument: None,
            },
            pos,
        )
    }

    pub fn plus_match_false(&self, pos: Vec3) -> Self {
        self._plus_node(
            crate::ast::EAstNode::MatchFalse {
                input_argument: None,
            },
            pos,
        )
    }

    fn _plus_node(&self, node: crate::ast::EAstNode, pos: Vec3) -> Self {
        let id = self.ast.next_id();
        Self {
            ast: self.ast.plus(node),
            layout_nodes: self
                .layout_nodes
                .clone()
                .into_iter()
                .chain([(id.clone(), LayoutNode { node_id: id, pos })])
                .collect(),
        }
    }

    pub fn edges(&self) -> Vec<LayoutEdge> {
        self.ast
            .nodes
            .iter()
            .flat_map(|(node_id, node)| match node {
                crate::ast::EAstNode::Sink {
                    input: Some(other_id),
                } => vec![LayoutEdge {
                    from_node_id: other_id.clone(),
                    to_node_id: node_id.clone(),
                    from_pos: self.layout_nodes.get(other_id).unwrap().pos,
                    to_pos: self.layout_nodes.get(node_id).unwrap().pos,
                }],
                crate::ast::EAstNode::FunctionCall {
                    input_arguments, ..
                } => input_arguments
                    .into_iter()
                    .map(|input_argument_node_id| LayoutEdge {
                        from_node_id: input_argument_node_id.clone(),
                        to_node_id: node_id.clone(),
                        from_pos: self.layout_nodes.get(input_argument_node_id).unwrap().pos,
                        to_pos: self.layout_nodes.get(node_id).unwrap().pos,
                    })
                    .collect::<Vec<LayoutEdge>>(),
                crate::ast::EAstNode::MatchTrue {
                    input_argument: Some(other_id),
                }
                | crate::ast::EAstNode::MatchFalse {
                    input_argument: Some(other_id),
                } => vec![LayoutEdge {
                    from_node_id: other_id.clone(),
                    to_node_id: node_id.clone(),
                    from_pos: self.layout_nodes.get(other_id).unwrap().pos,
                    to_pos: self.layout_nodes.get(node_id).unwrap().pos,
                }],
                crate::ast::EAstNode::MatchTrue {
                    input_argument: None,
                }
                | crate::ast::EAstNode::MatchFalse {
                    input_argument: None,
                }
                | crate::ast::EAstNode::BoolLiteral(_)
                | crate::ast::EAstNode::NumLiteral(_)
                | crate::ast::EAstNode::Sink { input: None } => vec![],
            })
            .collect()
    }
}

/// Spacing constants for the 3D layout.
const SPACING_X: f32 = 2.0;
const SPACING_Y: f32 = 2.0;
const SPACING_Z: f32 = 3.5;
