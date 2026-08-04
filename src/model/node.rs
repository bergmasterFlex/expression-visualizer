#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Id(usize);

impl crate::common::TId for Id {
    fn zero() -> Self {
        Self(0)
    }

    fn next_id(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({})", self.0)
    }
}

#[derive(Debug, Clone)]
pub enum ENode {
    Program {},
    Sink {
        input_anchor: super::anchor::Id,
    },
    FunctionCall {
        function_declaration_id: super::function_declaration::FunctionDeclarationId,
        input_anchors: Vec<super::anchor::Id>,
        output_anchor: super::anchor::Id,
    },
    ConstDecl {
        r#type: super::r#type::EType,
        output_anchor: super::anchor::Id,
    },
    TypeCast {
        r#type: super::r#type::EType,
        input_anchor: super::anchor::Id,
        output_anchor: super::anchor::Id,
    },
    VarDecl {
        name: String,
        r#type: super::r#type::EType,
        output_anchor: super::anchor::Id,
    },
    Match {
        patterns: Vec<super::node::Id>,
        input_anchor: super::anchor::Id,
        output_anchor: super::anchor::Id,
    },
    Pattern {
        parent_match: super::node::Id,
        r#type: super::r#type::EType,
        output_anchor: super::anchor::Id,
        sink_node_id: super::node::Id,
    },
}

impl ENode {
    pub fn anchors(&self) -> Vec<(super::anchor::Id, super::anchor::EAnchor)> {
        match self {
            ENode::FunctionCall {
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
                        super::anchor::EAnchor::Input(super::anchor::InputAnchor {
                            order_num: i,
                            name: Some(format!("param{}", i)),
                        }),
                    )
                })
                .chain(vec![(
                    output_anchor.clone(),
                    super::anchor::EAnchor::Output,
                )])
                .collect(),
            ENode::Sink { input_anchor } => vec![(
                input_anchor.clone(),
                super::anchor::EAnchor::Input(super::anchor::InputAnchor {
                    order_num: 0,
                    name: None,
                }),
            )],
            ENode::ConstDecl { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::anchor::EAnchor::Output)]
            }
            ENode::TypeCast {
                input_anchor,
                output_anchor,
                ..
            } => vec![
                (
                    input_anchor.clone(),
                    super::anchor::EAnchor::Input(super::anchor::InputAnchor {
                        order_num: 0,
                        name: None,
                    }),
                ),
                (output_anchor.clone(), super::anchor::EAnchor::Output),
            ],
            ENode::VarDecl { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::anchor::EAnchor::Output)]
            }
            ENode::Match {
                input_anchor,
                output_anchor,
                ..
            } => vec![
                (
                    input_anchor.clone(),
                    super::anchor::EAnchor::Input(super::anchor::InputAnchor {
                        order_num: 0,
                        name: None,
                    }),
                ),
                (output_anchor.clone(), super::anchor::EAnchor::Output),
            ],
            ENode::Pattern { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::anchor::EAnchor::Output)]
            }
            ENode::Program { .. } => vec![],
        }
    }

    pub fn input_anchors(&self) -> Vec<(super::anchor::Id, super::anchor::InputAnchor)> {
        self.anchors()
            .iter()
            .filter_map(|(id, anchor)| {
                if let super::anchor::EAnchor::Input(input_anchor) = anchor {
                    Some((id.clone(), input_anchor.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}
