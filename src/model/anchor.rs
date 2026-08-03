#[derive(Clone, Debug, Hash, Eq, PartialEq)]
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
    pub name: Option<String>,
}
