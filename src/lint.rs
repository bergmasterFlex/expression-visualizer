//! What is wrong with a graph, asked before a run rather than during one.
//!
//! Evaluation used to be the only thing that checked anything. `eval_next_step`
//! returns `Result<State, Vec<String>>`, and the driver turned a failure into a
//! modal that replaced the `Running` phase — so the first problem stopped the
//! run, took the state it failed in with it, and named the place as an anchor
//! id in `Debug` form. A half-wired `add` reached the user as
//! `no value found for input anchor Id(7)`.
//!
//! This asks the same questions of the graph standing still. Every answer keeps
//! the node it is about, so the list can say where to go, and they all come out
//! together, so a graph with three holes in it says so once.
//!
//! **Every node is asked every question, reachable or not.** It was gated on
//! reachability for a while, on the reasoning that evaluation never walks past
//! the sink's input and so a node dangling off to one side cannot block it. But
//! a half-built node is half-built wherever it stands, and being told only once
//! it is wired up is being told late — so the hole is reported where it is, and
//! reachability is one more answer beside it rather than a gate in front of it.
//!
//! Being unreachable is itself worth one line: [`Severity::Warning`], at the
//! *end* of each dangling run and not once per node in it. A chain nothing
//! reads is one mistake and not four, and its end is where it reads — the node
//! whose value nothing takes is the reason the ones behind it go unread.
//!
//! Reachability is asked per scope, from each scope's own sink. A Match that
//! nothing reads is a statement about the Match; the arms below it go on being
//! judged by whether they reach the end of their own arm, which is the only end
//! an arm has.
//!
//! Nothing here reaches through Bevy, and nothing here reads a position: a
//! diagnostic names a `node::Id` and the caller turns that into a cell. That
//! is the condition `infer.rs` states for the span arithmetic it lends out —
//! *"A linter should not have to reach through Bevy to ask it."*

type FunctionDeclarations = std::collections::HashMap<
    crate::model::function_declaration::FunctionDeclarationId,
    crate::model::function_declaration::FunctionDeclaration,
>;

/// How much a diagnostic is worth stopping for.
///
/// The order is the one the list sorts by, and `Error` is first because it is
/// the only one that decides anything: a run does not start while one stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The run cannot start. Something is missing rather than merely wrong —
    /// wherever it stands, because a hole is a hole before anything is wired
    /// to it.
    Error,
    /// The run starts and something in it is provably dead or provably
    /// disagrees. Nothing here is about a *value* — that is what a run finds
    /// out, and it is `Note`'s business.
    Warning,
    /// True of the graph and worth saying, but nothing to fix.
    ///
    /// Nothing produces one yet. The severity exists so that the list, the
    /// panel and the sort do not have to grow a third case later: the two
    /// obvious sources are a `CastKind::Partial` cast, and the per-node reasons
    /// a run records in `eval::State::trace`, which are already keyed by node
    /// id and today only visible when the caret stands on the node.
    #[allow(dead_code)]
    Note,
}

impl Severity {
    /// The glyph the panel marks a row with. Kept beside the severity rather
    /// than in the panel: it is what the severity *reads as*, and a second
    /// place to decide it is a second place to disagree.
    pub fn glyph(&self) -> &'static str {
        match self {
            Severity::Error => "●",
            Severity::Warning => "▲",
            Severity::Note => "○",
        }
    }
}

/// One thing that is wrong, and where.
///
/// Comparable so the panel can tell whether the list it is showing is still the
/// list there is, and rebuild its rows only when it is not.
#[derive(Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    /// The node it is about.
    ///
    /// Nothing answers `None` any more. It used to be how E1 said *this is
    /// about the program and not about a place in it* — until the Sink turned
    /// out to be a place like any other, with an id and a cell, and a row that
    /// named it could be walked to like every other row. The `Option` stays
    /// for the same reason [`Severity::Note`] stays: a statement about the
    /// graph as a whole is a thing a linter may yet want to make, and the
    /// panel already reads this as an `Option` on its way to a cell.
    pub node: Option<crate::model::node::Id>,
    pub message: String,
}

/// Everything wrong with `root`, worst first.
///
/// `root` is the outermost `LayoutGraph`; every check runs against its
/// flattened graph, because that is what evaluation runs against. Flattening
/// matters for more than convenience: a Tunnel's input anchor is reached from
/// the *enclosing* scope, so asking the branch's own graph whether it is wired
/// would answer no for every Tunnel there is.
pub fn check(root: &crate::layout::LayoutGraph, decls: &FunctionDeclarations) -> Vec<Diagnostic> {
    let graph = root.flattened_graph();
    let mut out: Vec<Diagnostic> = Vec::new();

    // E1. The one check that already existed, asked of the root graph the way
    // `begin_evaluation` asked it.
    if !crate::infer::sink_has_input(&root.graph) {
        out.push(Diagnostic {
            severity: Severity::Error,
            // The Sink is a node with a cell, so the one statement about the
            // program as a whole still has somewhere to send the reader. It is
            // the root's own sink and not the flattened graph's: `sink_node_id`
            // survives flattening, but what E1 asks about is the outermost one.
            node: Some(root.graph.sink_node_id.clone()),
            message: "Nothing is connected to the sink".to_string(),
        });
    }

    let reachable = reachable_from_sinks(&graph);
    let tails = unreachable_tails(&graph, &reachable);

    // E6. Before anything that recurses through inference, because a cycle
    // makes every type behind it `Pending` and the reports that follow would
    // all be consequences of this one.
    for id in cyclic_nodes(&graph) {
        out.push(Diagnostic {
            severity: Severity::Error,
            message: format!(
                "{} feeds itself, directly or through its inputs",
                label(&graph, &id, decls)
            ),
            node: Some(id),
        });
    }

    // Sorted, because `layout_nodes` is a `HashMap` and a list that reshuffled
    // itself every frame would be unreadable whatever it said.
    let mut ids: Vec<crate::model::node::Id> = graph.nodes.keys().cloned().collect();
    ids.sort();

    for id in ids {
        let Some(node) = graph.nodes.get(&id) else {
            continue;
        };
        // W5. `unreachable_tails` has already decided which of the unread
        // nodes speaks for its run; the rest of that run says nothing, and
        // every node — read or not — is asked the same questions below.
        if tails.contains(&id) {
            out.push(Diagnostic {
                severity: Severity::Warning,
                message: format!("Unreachable node: {}", label(&graph, &id, decls)),
                node: Some(id.clone()),
            });
        }

        check_node(&graph, &id, node, decls, &mut out);
    }

    // A stable order for equal severities: `ids` was sorted, so the only thing
    // left to settle is severity itself.
    out.sort_by_key(|d| d.severity);
    out
}

/// Everything asked of one node, whether or not anything reads it.
fn check_node(
    graph: &crate::model::term_graph::TermGraph,
    id: &crate::model::node::Id,
    node: &crate::model::node::ENode,
    decls: &FunctionDeclarations,
    out: &mut Vec<Diagnostic>,
) {
    let name = label(graph, id, decls);

    // E3. The four kinds that are built before they are typed. A node with no
    // declared type is half-built rather than wrongly built — `eval.rs` calls
    // it "an error of the graph and not a `none` travelling along an edge".
    let untyped = match node {
        crate::model::node::ENode::Source { r#type, .. } => r#type.is_none().then_some("Source"),
        crate::model::node::ENode::TypeCast { r#type, .. } => {
            r#type.is_none().then_some("TypeCast")
        }
        crate::model::node::ENode::Pattern { r#type, .. } => r#type.is_none().then_some("Arm"),
        crate::model::node::ENode::Tunnel { r#type, .. } => r#type.is_none().then_some("Tunnel"),
        _ => None,
    };
    if let Some(kind) = untyped {
        out.push(Diagnostic {
            severity: Severity::Error,
            node: Some(id.clone()),
            message: format!("{} declares no type", kind),
        });
    }

    // E4.
    if let crate::model::node::ENode::Match { patterns, .. } = node {
        if patterns.is_empty() {
            out.push(Diagnostic {
                severity: Severity::Error,
                node: Some(id.clone()),
                message: "Match has no arms".to_string(),
            });
        }
    }

    // E5. Checked before any message that would name the function, because
    // naming it is what would panic.
    if let crate::model::node::ENode::FunctionCall {
        function_declaration_id,
        ..
    } = node
    {
        if !decls.contains_key(function_declaration_id) {
            out.push(Diagnostic {
                severity: Severity::Error,
                node: Some(id.clone()),
                message: "Call names a function the catalogue does not hold".to_string(),
            });
            return;
        }
    }

    // E2. The gap that used to kill a run halfway: `eval_next_step` compares
    // arrived edges against edges with values, so an anchor with no edge
    // satisfies it and the node dies further down instead.
    //
    // The outermost Sink is E1's to report — it is the one unwired input that
    // is about the program rather than about a node in it, and saying it twice
    // in two wordings would read as two faults. E1 now names this very node,
    // so the two rows would stand at the same address as well: the same hole,
    // pointed at twice. A *branch's* Sink is not exempt: an arm that produces
    // nothing is an ordinary hole.
    let inputs = if *id == graph.sink_node_id {
        Vec::new()
    } else {
        node.input_anchors()
    };
    for (anchor_id, anchor) in inputs {
        if crate::infer::source_anchor_for_input(graph, &anchor_id).is_some() {
            continue;
        }
        out.push(Diagnostic {
            severity: Severity::Error,
            node: Some(id.clone()),
            message: match input_name(node, &anchor, decls) {
                Some(param) => format!("{}: input \"{}\" is not connected", name, param),
                None => format!("{}: input is not connected", name),
            },
        });
    }

    match node {
        crate::model::node::ENode::TypeCast {
            r#type: Some(target),
            input_anchor,
            ..
        } => check_cast(graph, id, &name, target, input_anchor, decls, out),
        crate::model::node::ENode::Match {
            patterns,
            input_anchor,
            ..
        } => check_match(graph, id, patterns, input_anchor, decls, out),
        crate::model::node::ENode::Tunnel {
            r#type: Some(declared),
            input_anchor,
            ..
        } => {
            // No `base_type_of`: what a Tunnel is given is a base type — the
            // prompt offers it nothing else — which is the reading `infer`
            // takes of the same field.
            let declared = crate::infer::graph_type_to_eval_type(declared);
            // W4 for a Tunnel. It states what it lets through and converts
            // nothing, so a value the declaration does not cover is a
            // disagreement in the picture — worth saying, and not worth
            // refusing: "a constraint is a statement and not a gate".
            if let Some(arriving) = crate::infer::incoming_anchor_type(graph, input_anchor, decls) {
                if !arriving_agrees(&declared, &arriving) {
                    out.push(Diagnostic {
                        severity: Severity::Warning,
                        node: Some(id.clone()),
                        message: format!(
                            "Tunnel lets through {} but {} arrives",
                            declared.to_string(),
                            arriving.to_string()
                        ),
                    });
                }
            }
        }
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            input_anchors,
            ..
        } => {
            // W4 for a call. A parameter with no declared type is
            // unconstrained rather than typed `None` — `=` and `!=` take two
            // values of any kind — so those are skipped rather than compared.
            let Some(declaration) = decls.get(function_declaration_id) else {
                return;
            };
            for (index, anchor_id) in input_anchors.iter().enumerate() {
                let Some(parameter) = declaration.inputs.get(index) else {
                    continue;
                };
                let Some(expected) = parameter.r#type.as_ref() else {
                    continue;
                };
                let Some(arriving) = crate::infer::incoming_anchor_type(graph, anchor_id, decls)
                else {
                    continue;
                };
                if !arriving_agrees(expected, &arriving) {
                    out.push(Diagnostic {
                        severity: Severity::Warning,
                        node: Some(id.clone()),
                        message: format!(
                            "{}: \"{}\" takes {} but {} arrives",
                            name,
                            parameter.name,
                            expected.to_string(),
                            arriving.to_string()
                        ),
                    });
                }
            }
        }
        _ => {}
    }
}

/// Whether what arrives is covered by what is declared.
///
/// `Pending` is not a disagreement, it is an absence — something further up is
/// unfinished, and that is already its own diagnostic. Reporting it here too
/// would put a second line on every node behind the first hole.
fn arriving_agrees(declared: &crate::infer::EType, arriving: &crate::infer::EType) -> bool {
    if crate::infer::row_leaves(arriving).is_empty() {
        return true;
    }
    crate::infer::subsumes(declared, arriving)
}

/// W3. A cast every one of whose input rows becomes `none`.
///
/// Asked row by row, because that is how a cast is decided: `cast_kind` answers
/// for one leaf row, and `type_cast_output_type` already walks exactly this
/// loop and throws the answer away. Only a cast where *every* row is
/// `AlwaysNone` is reported — one row of several failing is an ordinary partial
/// cast, and the picture already draws it as a strand that lands nowhere.
fn check_cast(
    graph: &crate::model::term_graph::TermGraph,
    id: &crate::model::node::Id,
    name: &str,
    target: &crate::model::r#type::EType,
    input_anchor: &crate::model::anchor::Id,
    decls: &FunctionDeclarations,
    out: &mut Vec<Diagnostic>,
) {
    let target = crate::infer::graph_type_to_eval_type(target);
    let Some(arriving) = crate::infer::incoming_anchor_type(graph, input_anchor, decls) else {
        return;
    };
    let rows = crate::infer::row_leaves(&arriving);
    if rows.is_empty() {
        return;
    }
    if rows
        .iter()
        .all(|row| crate::infer::cast_kind(row, &target) == crate::infer::CastKind::AlwaysNone)
    {
        out.push(Diagnostic {
            severity: Severity::Warning,
            node: Some(id.clone()),
            message: format!(
                "{}: nothing arriving as {} can become {}",
                name,
                arriving.to_string(),
                target.to_string()
            ),
        });
    }
}

/// W1 and W2, off the spans the picture is already drawn from.
///
/// `infer` states the rule and lends the arithmetic: the spans a Match's arms
/// claim of a row "either cover it, leave a hole — an arm is missing — or
/// overlap, and then one of them is redundant". Reading the same functions the
/// renderer reads is the point: a linter that computed its own would sooner or
/// later disagree with the band the user is looking at.
fn check_match(
    graph: &crate::model::term_graph::TermGraph,
    id: &crate::model::node::Id,
    patterns: &[crate::model::node::Id],
    input_anchor: &crate::model::anchor::Id,
    decls: &FunctionDeclarations,
    out: &mut Vec<Diagnostic>,
) {
    let Some(scrutinee) = crate::infer::incoming_anchor_type(graph, input_anchor, decls) else {
        return;
    };
    // An arm with no declared type is E3's business; it claims nothing here
    // rather than being guessed at.
    let arms: Vec<(crate::model::node::Id, crate::infer::EType)> = patterns
        .iter()
        .filter_map(|pattern_id| match graph.nodes.get(pattern_id) {
            Some(crate::model::node::ENode::Pattern {
                r#type: Some(t), ..
            }) => Some((pattern_id.clone(), crate::infer::graph_type_to_eval_type(t))),
            _ => None,
        })
        .collect();
    if arms.is_empty() {
        return;
    }

    for row in crate::infer::row_leaves(&scrutinee) {
        let claimed: Vec<Option<crate::infer::RowSpan>> = arms
            .iter()
            .map(|(_, arm)| crate::infer::claimed_span(&row, arm))
            .collect();

        // E7. An error and not a warning: a value the arms do not claim
        // reaches the Match and there is nothing for it to go down, so the run
        // stops there. That it stops at a value rather than at a hole in the
        // picture is what used to make it read as a warning, but a run that
        // cannot finish is a run that cannot start.
        let spans: Vec<crate::infer::RowSpan> = claimed.iter().flatten().copied().collect();
        if !crate::infer::spans_cover_band(&spans) {
            out.push(Diagnostic {
                severity: Severity::Error,
                node: Some(id.clone()),
                message: format!("Non-exhaustive Match: {} not covered", row.to_string()),
            });
        }

        // W2. An arm is redundant where an *earlier* one already claims the
        // whole of what it claims — a Match takes the first arm that matches,
        // so only what stands above it can take its values away.
        //
        // One earlier arm at a time, deliberately: two arms that jointly cover
        // a third leave that third unreported. Saying less than the truth is
        // the direction to err in, and the containment test below is exact for
        // what it does report.
        for (index, span) in claimed.iter().enumerate() {
            let Some(span) = span else { continue };
            if span.is_degenerate() {
                continue;
            }
            let shadowed = claimed[..index]
                .iter()
                .flatten()
                .any(|earlier| covers(earlier, span));
            if shadowed {
                out.push(Diagnostic {
                    severity: Severity::Warning,
                    node: Some(arms[index].0.clone()),
                    message: format!(
                        "Arm {} is never reached: an arm above it already takes every {}",
                        arms[index].1.to_string(),
                        row.to_string()
                    ),
                });
            }
        }
    }
}

/// Whether `outer` takes the whole of what `inner` takes.
///
/// `RowSpan::overlaps` asks whether two spans share anything, which is the
/// weaker question; shadowing needs containment. Stated here rather than in
/// `infer` because it is this rule's business and not the band's.
fn covers(outer: &crate::infer::RowSpan, inner: &crate::infer::RowSpan) -> bool {
    outer.top <= inner.top + f32::EPSILON && outer.bottom >= inner.bottom - f32::EPSILON
}

/// Every node some sink's input reaches, walking backwards along the edges.
///
/// Every scope has one and answers to it alone, so every one of them is a
/// starting point — see the seeding below. A Match reaching its arms is
/// deliberately *not* followed: what an arm's contents answer to is the end of
/// that arm, and making it depend on whether anything reads the Match is what
/// turned one unread Match into a report per node standing under it.
///
/// Carries its own `seen` set and so terminates on a cycle.
fn reachable_from_sinks(
    graph: &crate::model::term_graph::TermGraph,
) -> std::collections::HashSet<crate::model::node::Id> {
    let mut seen = std::collections::HashSet::new();
    // Every scope answers to its own terminal: the program to the outer Sink,
    // a branch to the Sink it was born with. Seeding all of them at once is
    // what keeps an unreachable Match from making its arms' contents
    // unreachable too — an arm ends where its own Sink is, whatever reads the
    // Match.
    let mut stack = vec![graph.sink_node_id.clone()];
    stack.extend(graph.nodes.values().filter_map(|node| match node {
        crate::model::node::ENode::Pattern { sink_node_id, .. } => Some(sink_node_id.clone()),
        _ => None,
    }));
    // Backwards along input edges and nothing else. The walk crosses a scope
    // boundary exactly where the graph does — a Tunnel is fed from outside —
    // and nowhere else: a Match holding its Patterns is not a value arriving
    // anywhere, so it is not a way to reach one.
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let Some(node) = graph.nodes.get(&id) else {
            continue;
        };
        for (anchor_id, _) in node.input_anchors() {
            let Some(source) = crate::infer::source_anchor_for_input(graph, &anchor_id) else {
                continue;
            };
            if let Some(producer) = graph.anchor_to_node.get(&source) {
                stack.push(producer.clone());
            }
        }
    }
    seen
}

/// Which unreachable nodes are worth a line: the last of each dangling run.
///
/// Three kinds are never one, whatever the walk found. The Root owns the
/// outermost scope and is never wired to anything. A Pattern is reached through
/// its Match rather than through an edge. And a BranchSource is created with
/// its branch rather than by anyone — a branch is free not to want the matched
/// value at all, and one that builds its result from a Tunnel or a literal
/// simply never reads it, so unused is the normal case there.
fn unreachable_tails(
    graph: &crate::model::term_graph::TermGraph,
    reachable: &std::collections::HashSet<crate::model::node::Id>,
) -> std::collections::HashSet<crate::model::node::Id> {
    let speaks_for_itself = |id: &crate::model::node::Id| {
        if reachable.contains(id) {
            return false;
        }
        !matches!(
            graph.nodes.get(id),
            None | Some(
                crate::model::node::ENode::Root { .. }
                    | crate::model::node::ENode::Pattern { .. }
                    | crate::model::node::ENode::BranchSource { .. }
            )
        )
    };
    // A producer whose value another unreachable node takes is not the end of
    // anything: the one that takes it is further down the same dead run.
    let mut feeds_the_dead: std::collections::HashSet<crate::model::node::Id> =
        std::collections::HashSet::new();
    for (to, from) in &graph.incoming_edge {
        let Some(producer) = graph.anchor_to_node.get(from) else {
            continue;
        };
        if !speaks_for_itself(producer) {
            continue;
        }
        if graph
            .anchor_to_node
            .get(to)
            .is_some_and(|consumer| speaks_for_itself(consumer))
        {
            feeds_the_dead.insert(producer.clone());
        }
    }
    graph
        .nodes
        .keys()
        .filter(|id| speaks_for_itself(id) && !feeds_the_dead.contains(*id))
        .cloned()
        .collect()
}

/// Nodes that lie on a cycle of input edges.
///
/// Inference already notices this and says nothing about it: `anchor_type_guarded`
/// keeps a `visiting` set and returns `Pending` on a revisit, so a cycle reaches
/// the user as an undecided type. The evaluator keeps no such set at all and
/// recurses until the stack is gone — so this is the one diagnostic that
/// prevents a crash rather than a confusion.
///
/// Every node, reachable or not. A cycle off to one side is never walked into
/// by a run, but it is walked into by everything below — and what it does to
/// the reports there is the same confusion it would cause anywhere else.
fn cyclic_nodes(graph: &crate::model::term_graph::TermGraph) -> Vec<crate::model::node::Id> {
    let mut on_cycle = std::collections::BTreeSet::new();
    let mut settled = std::collections::HashSet::new();
    let mut ids: Vec<&crate::model::node::Id> = graph.nodes.keys().collect();
    ids.sort();
    for id in ids {
        walk(graph, id, &mut Vec::new(), &mut settled, &mut on_cycle);
    }
    on_cycle.into_iter().collect()
}

fn walk(
    graph: &crate::model::term_graph::TermGraph,
    id: &crate::model::node::Id,
    path: &mut Vec<crate::model::node::Id>,
    settled: &mut std::collections::HashSet<crate::model::node::Id>,
    on_cycle: &mut std::collections::BTreeSet<crate::model::node::Id>,
) {
    if let Some(at) = path.iter().position(|seen| seen == id) {
        // Everything from the revisit onward is the loop itself. Naming all of
        // it rather than one entry point: which node "the" cycle is at is not a
        // question with an answer, and any of them is a place to cut it.
        on_cycle.extend(path[at..].iter().cloned());
        return;
    }
    if settled.contains(id) {
        return;
    }
    let Some(node) = graph.nodes.get(id) else {
        return;
    };
    path.push(id.clone());
    for (anchor_id, _) in node.input_anchors() {
        let Some(source) = crate::infer::source_anchor_for_input(graph, &anchor_id) else {
            continue;
        };
        if let Some(producer) = graph.anchor_to_node.get(&source) {
            walk(graph, producer, path, settled, on_cycle);
        }
    }
    path.pop();
    settled.insert(id.clone());
}

/// What a parameter is called, for a message that has to point at one of
/// several. `None` where the node has exactly one input and naming it would add
/// nothing.
fn input_name(
    node: &crate::model::node::ENode,
    anchor: &crate::model::anchor::InputAnchor,
    decls: &FunctionDeclarations,
) -> Option<String> {
    let crate::model::node::ENode::FunctionCall {
        function_declaration_id,
        ..
    } = node
    else {
        return None;
    };
    decls
        .get(function_declaration_id)
        .and_then(|declaration| declaration.inputs.get(anchor.order_num))
        .map(|parameter| parameter.name.clone())
}

/// How a node is named in a diagnostic.
///
/// It answers for every graph, including the broken ones — a naming that
/// reached into the function catalogue and `unwrap`ed it would panic on
/// exactly the graph E5 exists to report.
fn label(
    graph: &crate::model::term_graph::TermGraph,
    id: &crate::model::node::Id,
    decls: &FunctionDeclarations,
) -> String {
    let Some(node) = graph.nodes.get(id) else {
        return "Node".to_string();
    };
    match node {
        crate::model::node::ENode::Root {} => "Root".to_string(),
        crate::model::node::ENode::Sink { .. } => "Sink".to_string(),
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        } => decls
            .get(function_declaration_id)
            .map(|declaration| declaration.name.clone())
            .unwrap_or_else(|| "Call".to_string()),
        crate::model::node::ENode::Constant { r#type, .. } => r#type.to_string(),
        crate::model::node::ENode::Source { name, .. } if !name.is_empty() => {
            format!("Source \"{}\"", name)
        }
        crate::model::node::ENode::Source { .. } => "Source".to_string(),
        crate::model::node::ENode::TypeCast { .. } => "TypeCast".to_string(),
        crate::model::node::ENode::Match { .. } => "Match".to_string(),
        crate::model::node::ENode::Pattern { .. } => "Arm".to_string(),
        crate::model::node::ENode::BranchSource { .. } => "Branch source".to_string(),
        crate::model::node::ENode::Tunnel { .. } => "Tunnel".to_string(),
    }
}
