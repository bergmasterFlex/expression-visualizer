pub mod node;

use bevy::prelude::Color;

#[derive(Clone, Debug)]
pub struct Edge {
    pub to: AnchorId,
    pub color: Color,
}

#[derive(Clone)]
pub struct Ast {
    next_node_id: node::Id,
    next_anchor_id: AnchorId,
    pub nodes: std::collections::HashMap<node::Id, node::ENode>,
    pub anchors: std::collections::HashMap<AnchorId, EAnchor>,
    pub anchor_to_node: std::collections::HashMap<AnchorId, node::Id>,
    pub edges: std::collections::HashMap<AnchorId, Vec<Edge>>,
}

impl Ast {
    pub fn with_next_anchor_id(&self) -> (Self, AnchorId) {
        (
            Self {
                next_node_id: self.next_node_id.clone(),
                next_anchor_id: AnchorId(self.next_anchor_id.0 + 1),
                nodes: self.nodes.clone(),
                anchors: self.anchors.clone(),
                anchor_to_node: self.anchor_to_node.clone(),
                edges: self.edges.clone(),
            },
            self.next_anchor_id.clone(),
        )
    }

    pub fn empty() -> Self {
        Self {
            next_node_id: node::Id(0),
            next_anchor_id: AnchorId(0),
            nodes: std::collections::HashMap::new(),
            anchors: std::collections::HashMap::new(),
            anchor_to_node: std::collections::HashMap::new(),
            edges: std::collections::HashMap::new(),
        }
    }

    pub fn plus_edge(&self, from: AnchorId, to: AnchorId) -> Self {
        self.plus_edge_colored(from, to, crate::colors::WHITE)
    }

    pub fn plus_edge_colored(&self, from: AnchorId, to: AnchorId, color: Color) -> Self {
        let edge = Edge { to, color };
        Self {
            next_node_id: self.next_node_id.clone(),
            next_anchor_id: self.next_anchor_id.clone(),
            anchors: self.anchors.clone(),
            nodes: self.nodes.clone(),
            anchor_to_node: self.anchor_to_node.clone(),
            edges: self
                .edges
                .clone()
                .into_iter()
                .chain(vec![(
                    from.clone(),
                    self.edges.get(&from).map_or(vec![edge.clone()], |edges| {
                        edges.clone().into_iter().chain(vec![edge]).collect()
                    }),
                )])
                .collect(),
        }
    }

    pub fn plus(&self, n: node::ENode) -> (Self, node::Id) {
        let anchors = n.anchors();
        (
            Self {
                next_node_id: node::Id(self.next_node_id.0 + 1),
                next_anchor_id: self.next_anchor_id.clone(),
                anchors: self
                    .anchors
                    .clone()
                    .into_iter()
                    .chain(anchors.clone())
                    .collect(),
                nodes: self
                    .nodes
                    .clone()
                    .into_iter()
                    .chain(vec![(self.next_node_id.clone(), n)])
                    .collect(),
                anchor_to_node: self
                    .anchor_to_node
                    .clone()
                    .into_iter()
                    .chain(
                        anchors
                            .into_iter()
                            .map(|(id, _)| (id, self.next_node_id.clone())),
                    )
                    .collect(),
                edges: self.edges.clone(),
            },
            self.next_node_id.clone(),
        )
    }

    pub fn minus(&self, n_id: &node::Id) -> Self {
        let anchor_ids = self
            .nodes
            .get(n_id)
            .unwrap()
            .anchors()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        Self {
            next_node_id: self.next_node_id.clone(),
            next_anchor_id: self.next_anchor_id.clone(),
            nodes: self
                .nodes
                .clone()
                .into_iter()
                .filter(|(id, _)| id != n_id)
                .collect(),
            anchors: self
                .anchors
                .clone()
                .into_iter()
                .filter(|(id, _)| !anchor_ids.contains(id))
                .collect(),
            anchor_to_node: self
                .anchor_to_node
                .clone()
                .into_iter()
                .filter(|(id, _)| !anchor_ids.contains(id))
                .collect(),
            edges: self
                .edges
                .clone()
                .into_iter()
                .filter(|(from, _)| !anchor_ids.contains(from))
                .filter_map(|(from, edges)| {
                    let kept: Vec<Edge> = edges
                        .into_iter()
                        .filter(|e| !anchor_ids.contains(&e.to))
                        .collect();
                    if kept.is_empty() {
                        None
                    } else {
                        Some((from, kept))
                    }
                })
                .collect(),
        }
    }

    pub fn get_connected_nodes_to_anchor(&self, anchor: AnchorId) -> Vec<node::Id> {
        self.edges
            .iter()
            .flat_map(|(from, edges)| edges.iter().map(|e| (from.clone(), e)))
            .filter_map(|(from, edge)| {
                if edge.to == anchor {
                    Some(self.anchor_to_node.get(&from).unwrap().clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct AnchorId(usize);

#[derive(Clone, Debug)]
pub enum EAnchor {
    Input {
        order_num: usize,
        name: Option<String>,
    },
    Output,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(pub usize);

pub struct FunctionDeclaration {
    pub name: String,
    pub inputs: Vec<FunctionParameterDeclaration>,
    pub output_type: crate::eval::EType,
}

pub struct FunctionParameterDeclaration {
    pub name: String,
    pub r#type: crate::eval::EType,
}
