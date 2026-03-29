use bevy::render::render_graph::Edge;

// ── AST node types ──────────────────────────────────────────
//
#[derive(Clone)]
pub struct Ast {
    next_node_id: AstNodeId,
    next_anchor_id: AnchorId,
    pub nodes: std::collections::HashMap<AstNodeId, EAstNode>,
    pub anchors: std::collections::HashMap<AnchorId, EAnchor>,
    pub anchor_to_node: std::collections::HashMap<AnchorId, AstNodeId>,
    pub edges: std::collections::HashMap<AnchorId, Vec<AnchorId>>,
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
            next_node_id: AstNodeId(0),
            next_anchor_id: AnchorId(0),
            nodes: std::collections::HashMap::new(),
            anchors: std::collections::HashMap::new(),
            anchor_to_node: std::collections::HashMap::new(),
            edges: std::collections::HashMap::new(),
        }
    }

    pub fn plus_edge(&self, from: AnchorId, to: AnchorId) -> Self {
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
                    self.edges.get(&from).map_or(vec![to.clone()], |anchors| {
                        anchors.clone().into_iter().chain(vec![to]).collect()
                    }),
                )])
                .collect(),
        }
    }

    pub fn plus(&self, n: EAstNode) -> (Self, AstNodeId) {
        let anchors = n.anchors();
        (
            Self {
                next_node_id: AstNodeId(self.next_node_id.0 + 1),
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

    pub fn minus(&self, n_id: &AstNodeId) -> Self {
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
            edges: self.edges.clone(),
        }
    }

    pub fn get_connected_nodes_to_anchor(&self, anchor: AnchorId) -> Vec<AstNodeId> {
        self.edges
            .iter()
            .flat_map(|(from, tos)| tos.iter().map(|to| (from.clone(), to)))
            .filter_map(|(from, to)| {
                if *to == anchor {
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
    Input { order_num: usize, name: String },
    Output,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AstNodeId(usize);

#[derive(Debug, Clone)]
pub enum EAstNode {
    Sink {
        input_anchor: AnchorId,
    },
    FunctionCall {
        function_declaration_id: FunctionDeclarationId,
        input_anchors: Vec<AnchorId>,
        output_anchor: AnchorId,
    },
    NumLiteral {
        value: f32,
        input_anchor: AnchorId,
        output_anchor: AnchorId,
    },
    BoolLiteral {
        value: bool,
        input_anchor: AnchorId,
        output_anchor: AnchorId,
    },
    MatchTrue {
        input_anchor: AnchorId,
        output_anchor: AnchorId,
    },
    MatchFalse {
        input_anchor: AnchorId,
        output_anchor: AnchorId,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(pub usize);

pub struct FunctionDeclaration {
    pub name: String,
    pub inputs: Vec<FunctionParameterDeclaration>,
    pub output_type: EType,
}

pub struct FunctionParameterDeclaration {
    pub name: String,
    pub r#type: EType,
}

#[derive(Debug, Clone)]
pub enum EType {
    Number,
    Bool,
    SumType(Vec<EType>),
    Error,
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self {
            EType::Number => "number".to_string(),
            EType::Bool => "bool".to_string(),
            EType::SumType(sub_types) => sub_types
                .iter()
                .map(|sub_type| sub_type.to_string())
                .collect::<Vec<_>>()
                .join("|"),
            EType::Error => "error".to_string(),
        }
    }
}

impl EAstNode {
    pub fn label(
        &self,
        function_declarations: &std::collections::HashMap<
            FunctionDeclarationId,
            FunctionDeclaration,
        >,
    ) -> String {
        match self {
            EAstNode::Sink { .. } => "sink".to_string(),
            EAstNode::FunctionCall {
                function_declaration_id,
                ..
            } => function_declarations
                .get(&function_declaration_id)
                .unwrap()
                .name
                .to_string(),
            EAstNode::BoolLiteral { value, .. } => value.to_string(),
            EAstNode::NumLiteral { value, .. } => value.to_string(),
            EAstNode::MatchTrue { .. } => "match true".to_string(),
            EAstNode::MatchFalse { .. } => "match false".to_string(),
        }
    }

    /// Type name for UI tooltips.
    pub fn eval_type(
        &self,
        ast: &Ast,
        function_declarations: &std::collections::HashMap<
            FunctionDeclarationId,
            FunctionDeclaration,
        >,
    ) -> EType {
        match self {
            EAstNode::Sink { input_anchor } => {
                match ast
                    .get_connected_nodes_to_anchor(input_anchor.clone())
                    .first()
                {
                    Some(input_node_id) => ast
                        .nodes
                        .get(input_node_id)
                        .unwrap()
                        .eval_type(ast, function_declarations),
                    None => EType::Error,
                }
            }
            EAstNode::FunctionCall {
                function_declaration_id,
                ..
            } => function_declarations
                .get(&function_declaration_id)
                .unwrap()
                .output_type
                .clone(),
            EAstNode::BoolLiteral { .. } => EType::Bool,
            EAstNode::NumLiteral { .. } => EType::Number,
            EAstNode::MatchTrue { .. } => EType::Bool,
            EAstNode::MatchFalse { .. } => EType::Bool,
        }
    }

    pub fn anchors(&self) -> Vec<(AnchorId, EAnchor)> {
        match self {
            EAstNode::BoolLiteral {
                input_anchor,
                output_anchor,
                ..
            }
            | EAstNode::NumLiteral {
                input_anchor,
                output_anchor,
                ..
            }
            | EAstNode::MatchTrue {
                input_anchor,
                output_anchor,
                ..
            }
            | EAstNode::MatchFalse {
                input_anchor,
                output_anchor,
                ..
            } => vec![
                (
                    input_anchor.clone(),
                    EAnchor::Input {
                        order_num: 0,
                        name: "<cont>".to_string(),
                    },
                ),
                (output_anchor.clone(), EAnchor::Output),
            ],
            EAstNode::FunctionCall {
                input_anchors,
                output_anchor,
                ..
            } => input_anchors
                .clone()
                .into_iter()
                .enumerate()
                .map(|(i, anchor_id)| {
                    (
                        anchor_id,
                        EAnchor::Input {
                            order_num: i,
                            name: format!("param{}", i),
                        },
                    )
                })
                .chain(vec![(output_anchor.clone(), EAnchor::Output)])
                .collect(),
            EAstNode::Sink { input_anchor } => vec![(
                input_anchor.clone(),
                EAnchor::Input {
                    order_num: 0,
                    name: "<cont>".to_string(),
                },
            )],
        }
    }
}
