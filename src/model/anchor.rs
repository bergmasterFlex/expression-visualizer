/// Ordered so a list of anchors can be put in a settled order without anyone
/// reading what is inside one. Two anchors may share a cell — a Tunnel's input
/// hangs outside its own scope and can fall on a cell of the enclosing one —
/// and a candidate list that reshuffled on such a tie would have the editor
/// proposing a different edge from one frame to the next while nothing moved.
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct Id(usize);

impl crate::common::TId for Id {
    fn zero() -> Self {
        Self(0)
    }

    fn next_id(&self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Debug)]
pub enum EAnchor {
    Input(InputAnchor),
    Output,
}

#[derive(Clone, Debug)]
pub struct InputAnchor {
    pub order_num: usize,
}
