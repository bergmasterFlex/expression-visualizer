#[derive(Clone)]
pub struct State {
    pub node_ids_to_values: std::collections::HashMap<crate::model::node::Id, EValue>,
    /// Why a node produced `none`. `None` says only *that* no value was
    /// produced, never *why*, so the why lives beside the value rather than
    /// inside it: session-only, unreadable by the evaluated program, gone with
    /// the run — the relation a program has to a stack trace.
    ///
    /// Invariant: only ever written by `with_produced`, and only together with
    /// a value. `main.rs` measures a step's progress by the value map alone
    /// and relies on that.
    ///
    /// An entry exists iff the node itself produced the `none`. A node that
    /// merely passes one on carries no entry — the reason belongs to the
    /// operation that failed, and the edges already show the way back to it.
    pub trace: std::collections::HashMap<crate::model::node::Id, String>,
}

#[derive(Clone)]
pub enum EValue {
    Bool(bool),
    Int(i32),
    String(String),
    Char(char),
    None,
}

/// A produced value and, when that value is `none`, the reason for it.
///
/// The reason travels beside the value and never inside it. `reason` is
/// private and `none_because` is the only way to set it, so a reason cannot
/// be attached to anything but a `none`.
pub struct Produced {
    value: EValue,
    reason: Option<String>,
}

impl From<EValue> for Produced {
    fn from(value: EValue) -> Self {
        Self {
            value,
            reason: None,
        }
    }
}

impl Produced {
    fn none_because(reason: String) -> Self {
        Self {
            value: EValue::None,
            reason: Some(reason),
        }
    }
}

impl State {
    fn empty() -> Self {
        Self {
            node_ids_to_values: std::collections::HashMap::new(),
            trace: std::collections::HashMap::new(),
        }
    }

    pub fn new(
        graph: &crate::model::term_graph::TermGraph,
        user_source_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        graph
            .nodes
            .get(&graph.sink_node_id)
            .cloned()
            .ok_or(vec![format!("sink node {} not found", graph.sink_node_id)])
            .and_then(|sink_node| {
                Self::empty().eval_next_step(
                    graph,
                    user_source_values,
                    (graph.sink_node_id.clone(), sink_node),
                    function_declarations,
                )
            })
    }

    pub fn eval_next_step(
        &self,
        graph: &crate::model::term_graph::TermGraph,
        user_source_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        (visitor_node_id, visitor_node): (crate::model::node::Id, crate::model::node::ENode),
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        if self.node_ids_to_values.contains_key(&visitor_node_id) {
            Ok(self.clone())
        } else {
            let input_anchor_ids_to_node_ids =
                graph.get_connected_nodes_to_node_input_anchors(&visitor_node_id);
            let input_anchor_ids_to_values = input_anchor_ids_to_node_ids
                .iter()
                .filter_map(|(anchor_id, node_id)| {
                    self.node_ids_to_values
                        .get(node_id)
                        .map(|value| (anchor_id.clone(), value.clone()))
                })
                .collect::<std::collections::HashMap<_, _>>();
            // Every connected input has arrived: a count of edges on the left
            // and a count of anchors on the right, and they may be compared
            // because an input anchor carries at most one incoming edge.
            // That is a structural invariant of the language rather than an
            // assumption made here, and `LayoutGraph::plus_edge` is where it
            // is kept — a new edge onto an occupied input replaces what was
            // there instead of joining it.
            //
            // Stated rather than defended. A graph that broke the invariant
            // would leave this comparison unsatisfiable and the node
            // unevaluated, so `Next` would go quiet — which is the failure
            // direction to want, since counting anchors on both sides would
            // instead pick one of two arriving values by hash order and
            // answer with it.
            if input_anchor_ids_to_node_ids.len() == input_anchor_ids_to_values.len() {
                self.eval_value_for_node(
                    &visitor_node_id,
                    visitor_node,
                    input_anchor_ids_to_values,
                    user_source_values,
                    graph,
                    function_declarations,
                )
            } else {
                let next_step_sub_evals = input_anchor_ids_to_node_ids
                    .into_iter()
                    .map(|(_, node_id)| match graph.nodes.get(&node_id).cloned() {
                        Some(node) => self.eval_next_step(
                            graph,
                            user_source_values,
                            (node_id, node),
                            function_declarations,
                        ),
                        None => Err(vec![format!("node {} not found", node_id)]),
                    })
                    .collect::<Vec<Result<Self, Vec<String>>>>();
                let errors = next_step_sub_evals
                    .iter()
                    .filter_map(|result| result.as_ref().err())
                    .flatten()
                    .cloned()
                    .collect::<Vec<String>>();
                if errors.is_empty() {
                    Ok(next_step_sub_evals
                        .into_iter()
                        .filter_map(Result::ok)
                        .fold(self.clone(), Self::merged_with))
                } else {
                    Err(errors)
                }
            }
        }
    }

    /// Record what `node_id` produced: the value always, the reason only when
    /// the node itself could not produce one. `Option` is an iterator of zero
    /// or one, so the trace entry needs no branch of its own.
    fn with_produced(&self, node_id: &crate::model::node::Id, produced: Produced) -> Self {
        Self {
            node_ids_to_values: self
                .node_ids_to_values
                .clone()
                .into_iter()
                .chain(vec![(node_id.clone(), produced.value)])
                .collect(),
            trace: self
                .trace
                .clone()
                .into_iter()
                .chain(produced.reason.map(|reason| (node_id.clone(), reason)))
                .collect(),
        }
    }

    /// Record a value that this node did not compute itself — it was passed on
    /// from somewhere else, so there is nothing to explain here.
    fn with_value(&self, node_id: &crate::model::node::Id, value: EValue) -> Self {
        self.with_produced(node_id, value.into())
    }

    fn merged_with(self, other: Self) -> Self {
        Self {
            node_ids_to_values: self
                .node_ids_to_values
                .into_iter()
                .chain(other.node_ids_to_values)
                .collect(),
            trace: self.trace.into_iter().chain(other.trace).collect(),
        }
    }

    pub fn eval_value_for_node(
        &self,
        node_id: &crate::model::node::Id,
        node: crate::model::node::ENode,
        input_anchor_ids_to_values: std::collections::HashMap<crate::model::anchor::Id, EValue>,
        user_source_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        graph: &crate::model::term_graph::TermGraph,
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        match node {
            crate::model::node::ENode::Root {} => {
                Err(vec!["cannot get value of a root node".to_string()])
            }
            crate::model::node::ENode::Sink { input_anchor } => input_anchor_ids_to_values
                .get(&input_anchor)
                .cloned()
                .map(|value| self.with_value(node_id, value))
                .ok_or_else(|| vec!["no value found for input anchor for sink".to_string()]),
            crate::model::node::ENode::FunctionCall {
                function_declaration_id,
                input_anchors,
                ..
            } => {
                let mut sorted_input_anchors = input_anchors.clone();
                sorted_input_anchors.sort_by_key(|anchor_id| match graph.anchors.get(anchor_id) {
                    Some(crate::model::anchor::EAnchor::Input(input_anchor)) => {
                        input_anchor.order_num
                    }
                    _ => usize::MAX,
                });
                sorted_input_anchors
                    .iter()
                    .map(|anchor_id| {
                        input_anchor_ids_to_values
                            .get(anchor_id)
                            .cloned()
                            .ok_or(format!("no value found for input anchor {:?}", anchor_id))
                    })
                    .collect::<Result<Vec<EValue>, String>>()
                    .and_then(|arguments| {
                        function_declarations
                            .get(&function_declaration_id)
                            .ok_or(format!(
                                "function declaration with id {} not found",
                                function_declaration_id.0
                            ))
                            .and_then(|function_declaration| {
                                Self::eval_value_for_function_call(function_declaration, arguments)
                            })
                    })
                    .map(|produced| self.with_produced(node_id, produced))
                    .map_err(|error| vec![error])
            }
            crate::model::node::ENode::Constant { r#type, .. } => Self::eval_value_for_type(r#type)
                .map(|produced| self.with_produced(node_id, produced))
                .map_err(|error| vec![error]),
            crate::model::node::ENode::TypeCast {
                r#type,
                input_anchor,
                ..
            } => r#type
                // A cast with no declared target has nothing to cast *to*. It
                // is a half-built node rather than a failing one, so this is an
                // error of the graph and not a `none` travelling along an edge.
                .ok_or_else(|| vec!["type cast has no declared type".to_string()])
                .and_then(|r#type| {
                    input_anchor_ids_to_values
                        .get(&input_anchor)
                        .ok_or_else(|| {
                            vec!["no value found for input anchor for type cast".to_string()]
                        })
                        .map(|value| {
                            self.with_produced(
                                node_id,
                                Self::eval_value_for_type_cast(value.clone(), r#type),
                            )
                        })
                }),
            crate::model::node::ENode::Source { name, .. } => user_source_values
                .get(node_id)
                .cloned()
                .map(|value| self.with_value(node_id, value))
                .ok_or_else(|| vec![format!("no value for source provided: {}", name)]),
            crate::model::node::ENode::Match {
                patterns,
                input_anchor,
                ..
            } => input_anchor_ids_to_values
                .get(&input_anchor)
                .cloned()
                .ok_or_else(|| vec!["no value found for input anchor for match".to_string()])
                .and_then(|input_value| {
                    Self::eval_pattern_match(patterns, graph, input_value)
                        .map_err(|error| vec![error])
                })
                .and_then(
                    |matching_pattern_id| match graph.nodes.get(&matching_pattern_id) {
                        Some(crate::model::node::ENode::Pattern { sink_node_id, .. }) => match self
                            .node_ids_to_values
                            .get(sink_node_id)
                        {
                            Some(sink_value) => Ok(self.with_value(node_id, sink_value.clone())),
                            None => match graph.nodes.get(sink_node_id).cloned() {
                                Some(sink_node) => self.eval_next_step(
                                    graph,
                                    user_source_values,
                                    (sink_node_id.clone(), sink_node),
                                    function_declarations,
                                ),
                                None => Err(vec![format!("node {} not found", sink_node_id)]),
                            },
                        },
                        Some(_) => Err(vec![format!(
                            "matched node {} is not a pattern",
                            matching_pattern_id
                        )]),
                        None => Err(vec![format!("node {} not found", matching_pattern_id)]),
                    },
                ),
            // A Pattern is a type declaration in the Match's volume, not a
            // value producer — the branch reads from its BranchSource instead,
            // and the Match evaluates the branch's Sink directly.
            crate::model::node::ENode::Pattern { .. } => {
                Err(vec!["a pattern produces no value".to_string()])
            }
            // The branch's entry point: hand through whatever arrived at the
            // owning Match's input anchor. Reached only once that Match has
            // selected this branch, so the value is the matched one.
            crate::model::node::ENode::BranchSource { pattern, .. } => {
                let parent_match = match graph.nodes.get(&pattern) {
                    Some(crate::model::node::ENode::Pattern { parent_match, .. }) => {
                        parent_match.clone()
                    }
                    Some(_) => {
                        return Err(vec![format!("node {} is not a pattern", pattern)]);
                    }
                    None => return Err(vec![format!("pattern {} not found", pattern)]),
                };
                match graph.nodes.get(&parent_match) {
                    Some(crate::model::node::ENode::Match { input_anchor, .. }) => graph
                        .get_connected_nodes_to_anchor(input_anchor.clone())
                        .into_iter()
                        .find_map(|source_node_id| {
                            self.node_ids_to_values.get(&source_node_id).cloned()
                        })
                        .map(|value| self.with_value(node_id, value))
                        .ok_or_else(|| {
                            vec![format!(
                                "no value at parent match input anchor for branch source {}",
                                node_id
                            )]
                        }),
                    Some(_) => Err(vec![format!(
                        "parent match {} of pattern {} is not a match node",
                        parent_match, pattern
                    )]),
                    None => Err(vec![format!("parent match {} not found", parent_match)]),
                }
            }
            // Hand through whatever the enclosing graph put on the input. The
            // simpler cousin of the branch source above: that one has to find
            // its way through its Pattern to the owning Match to learn what
            // arrived, while a Tunnel is wired to its producer directly and
            // need only ask its own anchor.
            crate::model::node::ENode::Tunnel { input_anchor, .. } => graph
                .get_connected_nodes_to_anchor(input_anchor.clone())
                .into_iter()
                .find_map(|source_node_id| self.node_ids_to_values.get(&source_node_id).cloned())
                .map(|value| self.with_value(node_id, value))
                .ok_or_else(|| vec![format!("nothing arrives at tunnel {}", node_id)]),
        }
    }

    fn eval_pattern_match(
        patterns: Vec<crate::model::node::Id>,
        graph: &crate::model::term_graph::TermGraph,
        input_value: EValue,
    ) -> Result<crate::model::node::Id, String> {
        patterns
            .into_iter()
            .find(|pattern_id| match graph.nodes.get(pattern_id) {
                // An arm that declares nothing matches nothing. A Match of
                // nothing but such arms falls through to the same error as one
                // whose arms simply do not cover the value, which is what it
                // is: no arm matched.
                Some(crate::model::node::ENode::Pattern {
                    r#type: Some(r#type),
                    ..
                }) => Self::value_matches_type(&input_value, r#type),
                _ => false,
            })
            .ok_or_else(|| format!("no pattern matched value {}", input_value))
    }

    fn value_matches_type(
        input_value: &EValue,
        pattern_type: &crate::model::r#type::EType,
    ) -> bool {
        match (pattern_type, input_value) {
            (crate::model::r#type::EType::Bool { value }, EValue::Bool(b)) => value
                .as_ref()
                .is_none_or(|v| v.parse::<bool>().ok() == Some(*b)),
            (crate::model::r#type::EType::Int { value }, EValue::Int(i)) => value
                .as_ref()
                .is_none_or(|v| v.parse::<i32>().ok() == Some(*i)),
            (crate::model::r#type::EType::String { value }, EValue::String(s)) => {
                value.as_ref().is_none_or(|v| v == s)
            }
            (crate::model::r#type::EType::Char { value }, EValue::Char(c)) => value
                .as_ref()
                .is_none_or(|v| v.parse::<char>().ok() == Some(*c)),
            (crate::model::r#type::EType::None { .. }, EValue::None) => true,
            _ => false,
        }
    }

    fn eval_value_for_type(r#type: crate::model::r#type::EType) -> Result<Produced, String> {
        match r#type {
            crate::model::r#type::EType::Bool { value } => value
                .ok_or("bool type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<bool>()
                        .map(|v| EValue::Bool(v).into())
                        .map_err(|_| format!("could not parse \"{}\" as Bool", v))
                }),
            crate::model::r#type::EType::Int { value } => value
                .ok_or("int type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<i32>()
                        .map(|v| EValue::Int(v).into())
                        .map_err(|_| format!("could not parse \"{}\" as Integer", v))
                }),
            crate::model::r#type::EType::String { value } => value
                .map(|v| EValue::String(v).into())
                .ok_or("string type did not have a specific value!".to_string()),
            crate::model::r#type::EType::Char { value } => value
                .ok_or("char type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<char>()
                        .map(|v| EValue::Char(v).into())
                        .map_err(|_| format!("could not parse \"{}\" as Char", v))
                }),
            // `none` is a single-symbol type: the value is the type, so there
            // is nothing left to read off and nothing that can be missing.
            crate::model::r#type::EType::None {} => Ok(EValue::None.into()),
        }
    }

    /// Value equality, as `=` and `!=` see it.
    ///
    /// There is exactly one `none`, so two of them are always equal. Values of
    /// different kinds are simply unequal — the comparison is total and never
    /// fails.
    fn values_equal(a: &EValue, b: &EValue) -> bool {
        match (a, b) {
            (EValue::Bool(a), EValue::Bool(b)) => a == b,
            (EValue::Int(a), EValue::Int(b)) => a == b,
            (EValue::String(a), EValue::String(b)) => a == b,
            (EValue::Char(a), EValue::Char(b)) => a == b,
            (EValue::None, EValue::None) => true,
            _ => false,
        }
    }

    /// The text a `Char | String` argument stands for, `None` for any other
    /// kind. `concat` accepts either on both sides.
    fn text_of(value: &EValue) -> Option<String> {
        match value {
            EValue::String(s) => Some(s.clone()),
            EValue::Char(c) => Some(c.to_string()),
            _ => None,
        }
    }

    fn eval_value_for_function_call(
        function_declaration: &crate::model::function_declaration::FunctionDeclaration,
        arguments: Vec<EValue>,
    ) -> Result<Produced, String> {
        match (function_declaration.name.as_str(), arguments.as_slice()) {
            // Arithmetic. Integer overflow wraps rather than trapping: the
            // language has no exceptions, and none is reserved for the failures
            // the output type actually declares.
            ("+", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(a.wrapping_add(*b)).into()),
            ("-", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(a.wrapping_sub(*b)).into()),
            ("*", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(a.wrapping_mul(*b)).into()),
            ("/", [EValue::Int(a), EValue::Int(b)]) => Ok(if *b == 0 {
                Produced::none_because("division by zero".to_string())
            } else {
                EValue::Int(a.wrapping_div(*b)).into()
            }),
            ("mod", [EValue::Int(a), EValue::Int(b)]) => Ok(if *b == 0 {
                Produced::none_because("division by zero".to_string())
            } else {
                EValue::Int(a.wrapping_rem(*b)).into()
            }),
            ("neg", [EValue::Int(a)]) => Ok(EValue::Int(a.wrapping_neg()).into()),
            // Comparison
            ("=", [a, b]) => Ok(EValue::Bool(Self::values_equal(a, b)).into()),
            ("!=", [a, b]) => Ok(EValue::Bool(!Self::values_equal(a, b)).into()),
            ("<", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Bool(a < b).into()),
            (">", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Bool(a > b).into()),
            ("<=", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Bool(a <= b).into()),
            (">=", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Bool(a >= b).into()),
            // Logic. Both operands are already evaluated by the time we get
            // here — a function call asks for every argument, so `&&` and `||`
            // do not short-circuit.
            ("&&", [EValue::Bool(a), EValue::Bool(b)]) => Ok(EValue::Bool(*a && *b).into()),
            ("||", [EValue::Bool(a), EValue::Bool(b)]) => Ok(EValue::Bool(*a || *b).into()),
            ("!", [EValue::Bool(a)]) => Ok(EValue::Bool(!a).into()),
            // String. Indices count characters, not bytes.
            ("len", [EValue::String(s)]) => Ok(EValue::Int(s.chars().count() as i32).into()),
            ("charAt", [EValue::String(s), EValue::Int(i)]) => {
                match usize::try_from(*i).ok().and_then(|i| s.chars().nth(i)) {
                    Some(c) => Ok(EValue::Char(c).into()),
                    None => Ok(Produced::none_because(format!(
                        "charAt: index {} out of bounds for string of length {}",
                        i,
                        s.chars().count()
                    ))),
                }
            }
            ("concat", [left, right]) => match (Self::text_of(left), Self::text_of(right)) {
                (Some(left), Some(right)) => Ok(EValue::String(left + &right).into()),
                _ => Err(Self::argument_error(function_declaration, &arguments)),
            },
            ("substr", [EValue::String(s), EValue::Int(begin), EValue::Int(length)]) => {
                let chars = s.chars().collect::<Vec<char>>();
                // Out of range is a none, not a clamped substring: a shorter
                // string than asked for would be a wrong answer, not a missing
                // one.
                let range = usize::try_from(*begin).ok().and_then(|start| {
                    let end = start.checked_add(usize::try_from(*length).ok()?)?;
                    (end <= chars.len()).then_some(start..end)
                });
                match range {
                    Some(range) => Ok(EValue::String(chars[range].iter().collect()).into()),
                    None => Ok(Produced::none_because(format!(
                        "substr: {}..{} out of bounds for string of length {}",
                        begin,
                        begin.saturating_add(*length),
                        chars.len()
                    ))),
                }
            }
            // Math
            ("min", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(*a.min(b)).into()),
            ("max", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(*a.max(b)).into()),
            ("abs", [EValue::Int(a)]) => Ok(EValue::Int(a.wrapping_abs()).into()),
            // Either the arguments do not fit the signature, or the name is
            // not one of the defined functions at all. Both mean the same
            // thing to the caller: this call cannot be evaluated.
            _ => Err(Self::argument_error(function_declaration, &arguments)),
        }
    }

    fn argument_error(
        function_declaration: &crate::model::function_declaration::FunctionDeclaration,
        arguments: &[EValue],
    ) -> String {
        format!(
            "function {} cannot be applied to ({})",
            function_declaration.name,
            arguments
                .iter()
                .map(EValue::type_name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    /// Cast one value, by the table the picture is drawn from.
    ///
    /// The rules are normative and live in the thesis this program
    /// illustrates — `91-appendix.tex`, "Type casting rules", summarised as a
    /// matrix in `03-concept-design.tex`. `infer::cast_kind` states which of
    /// them are total and which can fail; this is the same table one level
    /// down, at the values. Neither may be changed without the other: what the
    /// strands show is what happens here.
    ///
    /// Three steps, and the order of the first two is the whole of a bug that
    /// used to live here.
    ///
    /// **The sad path passes through, without exception.** `none` arrives as
    /// `none` whatever the target says, and it says so first, before anything
    /// else is looked at. The literal-target branch used to jump this queue,
    /// so a `none` cast to `42` came out as `42` — the one rule the appendix
    /// states *without exception* broken by the one branch that never asked.
    /// The reason stays where it was produced: this node did not fail, it
    /// forwarded.
    ///
    /// **Then the table**, on the target's base type alone.
    ///
    /// **Then, for a literal target, a comparison.** A cast to `42` does not
    /// author a `42` — that would be a Constant with an input anchor bolted
    /// on, which is no kind of node. It converts what arrives and then checks
    /// whether it landed on the value named, handing back `none` when it did
    /// not. The check is `value_matches_type`, the same function a one-armed
    /// Match on that literal asks, so the two cannot answer differently.
    ///
    /// That comparison is what a cast to a literal is *for*, and it is worth
    /// stating against the `=` function it resembles: `=` compares any two
    /// values, of any two types, and yields a `Bool`. A cast makes the same
    /// comparison and types the answer as `literal | none` — the value itself
    /// on the happy path, nothing on the sad one. It is a `Result` where `=`
    /// is a predicate, and that difference is the whole of the node.
    fn eval_value_for_type_cast(
        input_value: EValue,
        target_type: crate::model::r#type::EType,
    ) -> Produced {
        if matches!(input_value, EValue::None) {
            return EValue::None.into();
        }
        let produced = Self::cast_to_base_type(input_value, &target_type);
        // A target naming no literal is the whole of the cast: the base type
        // is what was asked for and the base type is what arrived.
        if crate::layout::value_of_etype(&target_type).is_none() {
            return produced;
        }
        // A conversion that already failed keeps its own reason. Saying the
        // value is not the literal on top of it would name the second of two
        // failures and hide the first.
        if matches!(produced.value, EValue::None) {
            return produced;
        }
        if Self::value_matches_type(&produced.value, &target_type) {
            produced
        } else {
            Produced::none_because(format!(
                "cast produced {} which is not {}",
                produced.value, target_type
            ))
        }
    }

    /// The casting table itself, on the target's base type. A literal on the
    /// target is ignored here and settled by the comparison in
    /// `eval_value_for_type_cast`; `none` never arrives, having been forwarded
    /// there before this is called.
    ///
    /// Two rules are worth naming because the obvious implementation gets them
    /// wrong. A Char casts to the Integer it *spells*, not to its code point:
    /// `'3'` is `3` and never `51`, because the five types are disjoint sets
    /// and a Char is a symbol rather than a number wearing one. And a Bool is
    /// `0`/`1` and `'0'`/`'1'` in both directions, which makes every other
    /// Integer and every other Char no Bool at all — not `true` by virtue of
    /// being non-zero.
    fn cast_to_base_type(
        input_value: EValue,
        target_type: &crate::model::r#type::EType,
    ) -> Produced {
        match target_type {
            crate::model::r#type::EType::Bool { .. } => match input_value {
                EValue::Bool(b) => EValue::Bool(b).into(),
                EValue::Int(0) => EValue::Bool(false).into(),
                EValue::Int(1) => EValue::Bool(true).into(),
                EValue::Int(i) => {
                    Produced::none_because(format!("cannot cast Integer {} to Bool", i))
                }
                EValue::Char('0') => EValue::Bool(false).into(),
                EValue::Char('1') => EValue::Bool(true).into(),
                EValue::Char(c) => {
                    Produced::none_because(format!("cannot cast Char '{}' to Bool", c))
                }
                // The two words are spelled either way round, which is the one
                // place the table is deliberately lenient — text is what a
                // human typed and `TRUE` is not a different claim from `true`.
                // The digits are not: `"0"` is the digit, and there is no case
                // to fold.
                EValue::String(s) => match s.as_str() {
                    "0" => EValue::Bool(false).into(),
                    "1" => EValue::Bool(true).into(),
                    other if other.eq_ignore_ascii_case("false") => EValue::Bool(false).into(),
                    other if other.eq_ignore_ascii_case("true") => EValue::Bool(true).into(),
                    other => {
                        Produced::none_because(format!("cannot cast String \"{}\" to Bool", other))
                    }
                },
                EValue::None => unreachable!("`none` is forwarded before the table is consulted"),
            },
            crate::model::r#type::EType::Int { .. } => match input_value {
                EValue::Bool(b) => EValue::Int(if b { 1 } else { 0 }).into(),
                EValue::Int(i) => EValue::Int(i).into(),
                EValue::Char(c) if c.is_ascii_digit() => {
                    EValue::Int(i32::from(c as u8 - b'0')).into()
                }
                EValue::Char(c) => {
                    Produced::none_because(format!("cannot cast Char '{}' to Integer", c))
                }
                EValue::String(s) => Self::parse_cast_integer(&s)
                    .map(|i| EValue::Int(i).into())
                    .unwrap_or_else(|| {
                        Produced::none_because(format!("cannot cast String \"{}\" to Integer", s))
                    }),
                EValue::None => unreachable!("`none` is forwarded before the table is consulted"),
            },
            crate::model::r#type::EType::Char { .. } => match input_value {
                EValue::Bool(b) => EValue::Char(if b { '1' } else { '0' }).into(),
                EValue::Int(i) if (0..=9).contains(&i) => {
                    EValue::Char(char::from(b'0' + i as u8)).into()
                }
                EValue::Int(i) => {
                    Produced::none_because(format!("cannot cast Integer {} to Char", i))
                }
                EValue::Char(c) => EValue::Char(c).into(),
                // `FromStr for char` succeeds on exactly one scalar value and
                // errors on anything else, which is the rule as written.
                EValue::String(s) => s
                    .parse::<char>()
                    .map(|c| EValue::Char(c).into())
                    .unwrap_or_else(|_| {
                        Produced::none_because(format!("cannot cast String \"{}\" to Char", s))
                    }),
                EValue::None => unreachable!("`none` is forwarded before the table is consulted"),
            },
            // Every kind has a canonical textual form, so this cast is total.
            crate::model::r#type::EType::String { .. } => EValue::String(match input_value {
                EValue::Bool(b) => b.to_string(),
                EValue::Int(i) => i.to_string(),
                EValue::String(s) => s,
                EValue::Char(c) => c.to_string(),
                // Forwarded by `eval_value_for_type_cast` before it could get
                // here. Written out rather than folded into a catch-all so
                // that a `none` reaching this arm is a crash and not a cast
                // rule quietly contradicting the appendix.
                EValue::None => unreachable!("`none` is forwarded before the table is consulted"),
            })
            .into(),
            // A cast to `none` would ignore whatever flows in and hand back a
            // fixed `none`, which is a Constant spelled the long way round.
            // `refuse_none_cast` greys it out at the prompt, so there is no
            // way to build one.
            crate::model::r#type::EType::None {} => {
                unreachable!("a cast to `none` is refused at the prompt")
            }
        }
    }

    /// Decimal integer text as a cast reads it: an optional sign, decimal
    /// digits, no leading zeros, and nothing else at all.
    ///
    /// A shape guard in front of `i32::from_str` rather than a second parser.
    /// `from_str` already refuses whitespace, thousands separators, a decimal
    /// point, `10e2`, and anything outside the representable range, and it
    /// already accepts `+`/`-` and reads `"-0"`, `"0"` and `"+0"` as zero —
    /// all of which the appendix asks for. The one thing it does *not* refuse
    /// is a leading zero.
    ///
    /// Which is the thing worth refusing. `i32::to_string` never writes one,
    /// so refusing to read one is exactly what makes `Integer → String →
    /// Integer` the identity, and what keeps `7` and `07` from being two ways
    /// of writing one value. A number has one spelling here, the way `none`
    /// has one symbol.
    fn parse_cast_integer(text: &str) -> Option<i32> {
        let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if digits.len() > 1 && digits.starts_with('0') {
            return None;
        }
        text.parse::<i32>().ok()
    }

    /// True once the sink node carries a value, i.e. evaluation has reached the
    /// root and stepping further would not add anything. Used to grey out the
    /// `Next` button.
    pub fn is_evaluated(&self, graph: &crate::model::term_graph::TermGraph) -> bool {
        self.node_ids_to_values.contains_key(&graph.sink_node_id)
    }
}

impl std::fmt::Display for EValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EValue::Bool(value) => write!(f, "{}", value),
            EValue::Int(value) => write!(f, "{}", value),
            EValue::String(value) => write!(f, "{}", value),
            EValue::Char(value) => write!(f, "{}", value),
            EValue::None => write!(f, "none"),
        }
    }
}

impl EValue {
    /// Parse a user-typed string into a value of the declared type. Sources
    /// carry their value in `user_source_values` (their type literal is
    /// `None` at declaration time), so this is the path from the prompt modal
    /// into evaluation.
    ///
    /// It is also the editor's own check: the INSERT prompt asks it whether
    /// the literal being typed is a value of the type it names, and offers the
    /// suggestion only when it is. Validating there against the same function
    /// the evaluation runs on is the point — what the prompt accepts cannot
    /// fail later.
    pub fn parse(target: &crate::model::r#type::EType, raw: &str) -> Result<EValue, String> {
        match target {
            crate::model::r#type::EType::Bool { .. } => raw
                .parse::<bool>()
                .map(EValue::Bool)
                .map_err(|_| format!("could not parse \"{}\" as Bool", raw)),
            crate::model::r#type::EType::Int { .. } => raw
                .parse::<i32>()
                .map(EValue::Int)
                .map_err(|_| format!("could not parse \"{}\" as Integer", raw)),
            crate::model::r#type::EType::String { .. } => Ok(EValue::String(raw.to_string())),
            crate::model::r#type::EType::Char { .. } => raw
                .parse::<char>()
                .map(EValue::Char)
                .map_err(|_| format!("could not parse \"{}\" as Char", raw)),
            // A Source declared `none` has exactly one possible value, so
            // whatever was typed there says nothing and is dropped.
            crate::model::r#type::EType::None { .. } => Ok(EValue::None),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            EValue::Bool(_) => "Bool",
            EValue::Int(_) => "Integer",
            EValue::String(_) => "String",
            EValue::Char(_) => "Char",
            EValue::None => "None",
        }
    }
}
