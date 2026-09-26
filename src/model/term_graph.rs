#[derive(Clone, Debug)]
pub struct TermGraph {
    pub nodes: std::collections::HashMap<super::node::Id, super::node::ENode>,
    pub anchors: std::collections::HashMap<super::anchor::Id, super::anchor::EAnchor>,
    pub anchor_to_node: std::collections::HashMap<super::anchor::Id, super::node::Id>,
    /// The edge arriving at an input anchor: the key is the input, the value the
    /// output that feeds it.
    ///
    /// Keyed by the arriving end, because that is where the language's one
    /// constraint sits — an input anchor carries *at most one* incoming edge,
    /// and a map keyed this way has nowhere to put a second. A producer is under
    /// no such rule and may feed as many consumers as it likes; those are
    /// several entries sharing one value.
    ///
    /// Edges are recorded output → input and only ever that way: `commit_draft`
    /// is the sole door and turns every pair into that direction before it comes
    /// through. That half is *not* structural — both ends are an `anchor::Id`,
    /// so a swapped insert would type-check — and is guarded by a
    /// `debug_assert!` in `LayoutGraph::plus_edge` instead, which is the one
    /// place that can resolve an anchor across scopes.
    pub incoming_edge: std::collections::HashMap<super::anchor::Id, super::anchor::Id>,
    /// The Sink node that terminates this graph. Always present: an `TermGraph` is born
    /// with its sink in `new`, and every builder carries it forward unchanged.
    pub sink_node_id: super::node::Id,
}

impl TermGraph {
    /// Create an graph that already contains its terminating `Sink` node (with a
    /// fresh input anchor), pointed to by `sink_node_id`. The id domains are
    /// threaded through so every id stays globally unique.
    pub fn new(
        node_id_domain: crate::common::IdDomain<super::node::Id>,
        anchor_id_domain: crate::common::IdDomain<super::anchor::Id>,
    ) -> (
        Self,
        crate::common::IdDomain<super::node::Id>,
        crate::common::IdDomain<super::anchor::Id>,
    ) {
        let (node_id_domain, sink_node_id) = node_id_domain.next_id();
        let (anchor_id_domain, sink_input_anchor_id) = anchor_id_domain.next_id();
        let graph = Self {
            nodes: std::collections::HashMap::new(),
            anchors: std::collections::HashMap::new(),
            anchor_to_node: std::collections::HashMap::new(),
            incoming_edge: std::collections::HashMap::new(),
            sink_node_id: sink_node_id.clone(),
        }
        .plus_node(
            sink_node_id,
            super::node::ENode::Sink {
                input_anchor: sink_input_anchor_id,
            },
        );
        (graph, node_id_domain, anchor_id_domain)
    }

    /// Sub-graph for a Match branch: the terminating Sink plus the branch's
    /// single `BranchSource`, which belongs to `pattern`. Both are
    /// constitutive — a branch is never without them — so they are created
    /// here rather than by any user action.
    pub fn new_pattern_sub_graph(
        node_id_domain: crate::common::IdDomain<super::node::Id>,
        anchor_id_domain: crate::common::IdDomain<super::anchor::Id>,
        pattern: super::node::Id,
    ) -> (
        crate::common::IdDomain<super::node::Id>,
        crate::common::IdDomain<super::anchor::Id>,
        Self,
        super::node::Id,
        super::node::Id,
    ) {
        let (sub_graph, node_id_domain, anchor_id_domain) =
            Self::new(node_id_domain, anchor_id_domain);
        let sink_node_id = sub_graph.sink_node_id.clone();
        let (node_id_domain, branch_source_id) = node_id_domain.next_id();
        let (anchor_id_domain, branch_source_anchor_id) = anchor_id_domain.next_id();
        let sub_graph = sub_graph.plus_node(
            branch_source_id.clone(),
            super::node::ENode::BranchSource {
                pattern,
                output_anchor: branch_source_anchor_id,
            },
        );
        (
            node_id_domain,
            anchor_id_domain,
            sub_graph,
            sink_node_id,
            branch_source_id,
        )
    }

    /// Union another graph's nodes/anchors/edges into this one, keeping this graph's
    /// `sink_node_id` as the root. Node/anchor ids are globally unique across a
    /// (sub-)graph tree, so those maps never collide.
    ///
    /// Neither do the edge tables, and that is worth saying rather than
    /// assuming: every edge is recorded on the root graph, because
    /// `LayoutGraph::plus_edge` is only ever called there — so a sub-graph's
    /// `incoming_edge` is empty and there is nothing here to resolve. Were one
    /// ever not empty, an input held on both sides would be settled last-wins
    /// and an edge would go missing without a word. That is what the assertion
    /// is for: the emptiness is load-bearing, and this is the only place it
    /// would be quietly relied on.
    pub fn merged_with(self, other: Self) -> Self {
        debug_assert!(
            other.incoming_edge.is_empty(),
            "merged_with: only the root graph records edges"
        );
        Self {
            nodes: self.nodes.into_iter().chain(other.nodes).collect(),
            anchors: self.anchors.into_iter().chain(other.anchors).collect(),
            anchor_to_node: self
                .anchor_to_node
                .into_iter()
                .chain(other.anchor_to_node)
                .collect(),
            incoming_edge: self
                .incoming_edge
                .into_iter()
                .chain(other.incoming_edge)
                .collect(),
            sink_node_id: self.sink_node_id,
        }
    }

    /// Wire `from` to `to`, taking the place of whatever arrived at `to`.
    ///
    /// Replacing is not this function being lenient about a rule it could have
    /// enforced: the table holds one source per input and has nowhere to put a
    /// second. `LayoutGraph::plus_edge` is where that is argued for — it is a
    /// decision about what the editor does to a wired input, not a property of
    /// a `HashMap` — and it is also where the direction is asserted.
    pub fn plus_edge(&self, from: super::anchor::Id, to: super::anchor::Id) -> Self {
        Self {
            anchors: self.anchors.clone(),
            nodes: self.nodes.clone(),
            anchor_to_node: self.anchor_to_node.clone(),
            sink_node_id: self.sink_node_id.clone(),
            incoming_edge: self
                .incoming_edge
                .clone()
                .into_iter()
                .chain(vec![(to, from)])
                .collect(),
        }
    }

    /// Keep only the edges both of whose ends are still anchors of something.
    ///
    /// The counterpart to `minus_node` for a scene made of more than one graph.
    /// `minus_node` takes a node's edges with it, but only the ones recorded in
    /// the *same* `TermGraph` — and a branch's nodes live in their own one while
    /// every edge is recorded on the root (`LayoutGraph::plus_edge` is called on
    /// `root_graph()` alone, whatever scope the two anchors are in). So a node
    /// taken out of a branch leaves the root holding edges to anchors that are
    /// gone.
    ///
    /// Asked as a question about what is still there, rather than answered by
    /// listing what a removal took away. The list is the hard part — a Pattern
    /// takes a whole branch volume with it and a Match takes every Pattern —
    /// and a list one entry short leaves an edge pointing at nothing, which
    /// `LayoutGraph::layout_anchor` meets as a panic rather than as a missing
    /// strand. There is nothing for this to be short of.
    pub fn retaining_edges(&self, live: &std::collections::HashSet<super::anchor::Id>) -> Self {
        Self {
            nodes: self.nodes.clone(),
            anchors: self.anchors.clone(),
            anchor_to_node: self.anchor_to_node.clone(),
            sink_node_id: self.sink_node_id.clone(),
            incoming_edge: self
                .incoming_edge
                .clone()
                .into_iter()
                .filter(|(to, from)| live.contains(to) && live.contains(from))
                .collect(),
        }
    }

    pub fn plus_node(&self, node_id: super::node::Id, n: super::node::ENode) -> Self {
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
            incoming_edge: self.incoming_edge.clone(),
            sink_node_id: self.sink_node_id.clone(),
        }
    }

    /// Replace `n_id`'s node with `new_node`. Anchor tables are untouched;
    /// callers are responsible for ensuring the replacement has the same
    /// anchors (used e.g. to update a `Match`'s `patterns` list).
    pub fn with_node_replaced(&self, n_id: &super::node::Id, new_node: super::node::ENode) -> Self {
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
            incoming_edge: self.incoming_edge.clone(),
            sink_node_id: self.sink_node_id.clone(),
        }
    }

    pub fn minus_node(&self, n_id: &super::node::Id) -> Self {
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
            incoming_edge: self
                .incoming_edge
                .clone()
                .into_iter()
                .filter(|(to, from)| !anchor_ids.contains(to) && !anchor_ids.contains(from))
                .collect(),
            sink_node_id: self.sink_node_id.clone(),
        }
    }

    /// The node producing the value that arrives at `input`, if one does.
    ///
    /// An `Option` and not a list, because an input anchor carries at most one
    /// incoming edge and the edge table is shaped to say so. One hop upstream
    /// and no further: `infer::source_anchor_for_input` is the same hop asked
    /// about the anchor rather than the node.
    ///
    /// The `unwrap` stands on `minus_dangling_edges`: an edge whose producer is
    /// not an anchor of anything is a scene that was left unswept, and meeting
    /// that as a panic here is what the sweep exists to prevent. Answering
    /// `None` instead would let it pass as an unwired input.
    pub fn source_node_for_input(&self, input: &super::anchor::Id) -> Option<super::node::Id> {
        self.incoming_edge
            .get(input)
            .map(|from| self.anchor_to_node.get(from).unwrap().clone())
    }

    /// Every input of `node_id` that something arrives at, with the node it
    /// arrives from. Inputs nothing is wired to are left out rather than
    /// carried as absences: what the caller counts is what has arrived.
    pub fn source_nodes_by_input(
        &self,
        node_id: &super::node::Id,
    ) -> Vec<(super::anchor::Id, super::node::Id)> {
        self.nodes
            .get(node_id)
            .into_iter()
            .flat_map(|node| node.input_anchors())
            .filter_map(|(anchor_id, _)| {
                self.source_node_for_input(&anchor_id)
                    .map(|node_id| (anchor_id, node_id))
            })
            .collect()
    }
}
