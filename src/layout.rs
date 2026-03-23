use bevy::prelude::*;

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

    pub fn plus_sink(&self) -> Self {
        let id = self.ast.next_id();
        let sink = crate::ast::EAstNode::Sink { input: None };
        LayoutAst {
            ast: self.ast.plus(sink),
            layout_nodes: self
                .layout_nodes
                .clone()
                .into_iter()
                .chain([(
                    id.clone(),
                    LayoutNode {
                        node_id: id,
                        pos: Vec3::new(0.0, 0.0, 0.0),
                    },
                )])
                .collect(),
        }
    }

    pub fn plus_number_literal(&self, value: String, pos: Vec3) -> Self {
        let id = self.ast.next_id();
        let number_literal = crate::ast::EAstNode::NumLiteral(value.to_string());
        LayoutAst {
            ast: self.ast.plus(number_literal),
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
