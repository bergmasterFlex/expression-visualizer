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
    /// The root owner of the layout tree. Not a node kind of the language —
    /// the abstract syntax has no such thing — but the layout needs one object
    /// that owns the outermost scope, and what has to be addressable needs an
    /// entry in the layout. Same category as `Pattern` below.
    Root {},
    Sink {
        input_anchor: super::anchor::Id,
    },
    FunctionCall {
        function_declaration_id: super::function_declaration::FunctionDeclarationId,
        input_anchors: Vec<super::anchor::Id>,
        output_anchor: super::anchor::Id,
    },
    Constant {
        r#type: super::r#type::EType,
        output_anchor: super::anchor::Id,
    },
    TypeCast {
        r#type: super::r#type::EType,
        input_anchor: super::anchor::Id,
        output_anchor: super::anchor::Id,
    },
    Source {
        name: String,
        r#type: super::r#type::EType,
        output_anchor: super::anchor::Id,
    },
    Match {
        patterns: Vec<super::node::Id>,
        input_anchor: super::anchor::Id,
        output_anchor: super::anchor::Id,
    },
    /// One arm of a Match. Lives in the Match's volume, not in its branch:
    /// it declares the type the arm matches and fixes the branch's Y row, but
    /// carries no anchor. The branch reads the matched value from its own
    /// `BranchSource` instead, so no edge crosses the volume boundary.
    Pattern {
        parent_match: super::node::Id,
        r#type: super::r#type::EType,
        sink_node_id: super::node::Id,
    },
    /// The single entry point of a Match branch, at branch-local (0,0,0) —
    /// directly behind its Pattern. Exactly one exists per branch and it is
    /// created with the branch, never by the user. Its output carries the
    /// matched value, typed by `pattern`'s declared type; it is what a
    /// top-level `Source` is to the root scope.
    BranchSource {
        pattern: super::node::Id,
        output_anchor: super::anchor::Id,
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
            ENode::Constant { output_anchor, .. } => {
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
            ENode::Source { output_anchor, .. } => {
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
            ENode::BranchSource { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::anchor::EAnchor::Output)]
            }
            // A Pattern carries no anchor: the branch reads its value from
            // the branch's own BranchSource.
            ENode::Pattern { .. } | ENode::Root { .. } => vec![],
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
