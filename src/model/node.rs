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
        /// The type to cast to, or the literal to produce — `None` until one is
        /// chosen. A node is built before its type is typed, and until then it
        /// has no type rather than a placeholder one: what it declares is
        /// `Pending`, which is drawn grey, and an editor that cancels the entry
        /// takes the node with it rather than leaving a default nobody picked.
        r#type: Option<super::r#type::EType>,
        input_anchor: super::anchor::Id,
        output_anchor: super::anchor::Id,
    },
    Source {
        name: String,
        /// The type every value handed in at this Source will be one of.
        ///
        /// `None` only while the node is still being built. A Source is built
        /// before its type is typed, and a default written here would put a
        /// declaration on the node that nobody made — so it carries none, is
        /// drawn grey, and the editor keeps it marked as unfinished until a
        /// type is given or an Escape takes the whole node away again. A
        /// Source that declares nothing is never left standing: every value
        /// handed in at one has to be a value of *something*, and the
        /// evaluation prompt would have nothing to parse an answer as.
        ///
        /// Optional for the same reason a `TypeCast`'s is, then, but not for
        /// as long: a cast with no target may stay in the graph and fail there.
        r#type: Option<super::r#type::EType>,
        output_anchor: super::anchor::Id,
    },
    Match {
        /// The arms, in the order they are checked — and therefore in the
        /// order they are drawn, top to bottom.
        ///
        /// One list and not two. Arm types may overlap, and a Match takes the
        /// value down the *first* arm that matches
        /// (`eval::eval_pattern_match`), so their order is a decision the
        /// program makes rather than an arrangement of it. That is why it is
        /// stated here and why the layout reads it rather than keeping an
        /// order of its own: `layout::LayoutGraph::respace_match_patterns`
        /// hands out the rows from this list, so what stands higher on screen
        /// is what is asked first, always.
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
        /// The type this arm matches — `None` until one is chosen, for the same
        /// reason a `TypeCast`'s is. An arm that declares nothing matches
        /// nothing, and its `BranchSource` carries `Pending` rather than a type
        /// the branch would then be typed against.
        r#type: Option<super::r#type::EType>,
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
    /// The declared way a value other than the matched one reaches a branch.
    ///
    /// A `BranchSource` carries what the Match narrowed and nothing else, so
    /// everything else a branch needs would otherwise have to be wired in from
    /// the outside straight onto whatever node wanted it. A Tunnel is the one
    /// place that is allowed to happen: its input hangs outside the branch's
    /// front face, where the enclosing graph can reach it, and its output
    /// stands on the branch's entry row like any other source of a value.
    ///
    /// It declares nothing of its own — not a type, not a name, only a
    /// position. Whatever arrives at the input leaves at the output, and with
    /// nothing wired in it carries `Pending`, which is what the branch behind
    /// it then reads. That is the whole of it: a Tunnel is a hole in a wall,
    /// and a hole has no opinion about what goes through.
    Tunnel {
        input_anchor: super::anchor::Id,
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
            // Both are real anchors, unlike the input a Source only *draws*:
            // the whole purpose of a Tunnel is that an edge can end on its
            // input, so it has to be in the tables an edge is resolved
            // through.
            ENode::Tunnel {
                input_anchor,
                output_anchor,
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
