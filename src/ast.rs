pub mod anchor;
pub mod node;

#[derive(Clone, Debug)]
pub struct Edge {
    pub to: anchor::Id,
}

#[derive(Clone, Debug)]
pub struct Ast {
    pub nodes: std::collections::HashMap<node::Id, node::ENode>,
    pub anchors: std::collections::HashMap<anchor::Id, anchor::EAnchor>,
    pub anchor_to_node: std::collections::HashMap<anchor::Id, node::Id>,
    pub edges: std::collections::HashMap<anchor::Id, Vec<Edge>>,
}

impl Ast {
    pub fn empty() -> Self {
        Self {
            nodes: std::collections::HashMap::new(),
            anchors: std::collections::HashMap::new(),
            anchor_to_node: std::collections::HashMap::new(),
            edges: std::collections::HashMap::new(),
        }
    }

    pub fn new_pattern_sub_ast(
        node_id_domain: crate::common::IdDomain<node::Id>,
        anchor_id_domain: crate::common::IdDomain<anchor::Id>,
    ) -> (
        crate::common::IdDomain<node::Id>,
        crate::common::IdDomain<anchor::Id>,
        Self,
        node::Id,
    ) {
        let (node_id_domain, sink_node_id) = node_id_domain.next_id();
        let (anchor_id_domain, sink_input_anchor_id) = anchor_id_domain.next_id();
        let sub_ast = Self::empty().plus_node(
            sink_node_id.clone(),
            node::ENode::Sink {
                input_anchor: sink_input_anchor_id,
            },
        );
        (node_id_domain, anchor_id_domain, sub_ast, sink_node_id)
    }

    pub fn plus_edge(&self, from: anchor::Id, to: anchor::Id) -> Self {
        let edge = Edge { to };
        Self {
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

    pub fn plus_node(&self, node_id: node::Id, n: node::ENode) -> Self {
        let anchors = n.anchors();
        Self {
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
                .chain(vec![(node_id.clone(), n)])
                .collect(),
            anchor_to_node: self
                .anchor_to_node
                .clone()
                .into_iter()
                .chain(anchors.into_iter().map(|(id, _)| (id, node_id.clone())))
                .collect(),
            edges: self.edges.clone(),
        }
    }

    /// Replace `n_id`'s node with `new_node`. Anchor tables are untouched;
    /// callers are responsible for ensuring the replacement has the same
    /// anchors (used e.g. to update a `Match`'s `patterns` list).
    pub fn with_node_replaced(&self, n_id: &node::Id, new_node: node::ENode) -> Self {
        Self {
            nodes: self
                .nodes
                .clone()
                .into_iter()
                .map(|(id, n)| {
                    if id == *n_id {
                        (id, new_node.clone())
                    } else {
                        (id, n)
                    }
                })
                .collect(),
            anchors: self.anchors.clone(),
            anchor_to_node: self.anchor_to_node.clone(),
            edges: self.edges.clone(),
        }
    }

    pub fn minus_node(&self, n_id: &node::Id) -> Self {
        let anchor_ids = self
            .nodes
            .get(n_id)
            .unwrap()
            .anchors()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        Self {
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

    pub fn get_connected_nodes_to_anchor(&self, anchor: anchor::Id) -> Vec<node::Id> {
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(pub usize);

#[derive(Clone)]
pub struct FunctionDeclaration {
    pub name: String,
    pub inputs: Vec<FunctionParameterDeclaration>,
    pub output_type: crate::eval::EType,
}

#[derive(Clone)]
pub struct FunctionParameterDeclaration {
    pub name: String,
    pub r#type: crate::eval::EType,
}
