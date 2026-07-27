#[derive(Debug, Clone)]
pub struct IdDomain<T: TId> {
    next_id: T,
}

pub trait TId: Clone {
    fn zero() -> Self;

    fn next_id(&self) -> Self;
}

impl<T: TId> IdDomain<T> {
    pub fn new() -> Self {
        Self { next_id: T::zero() }
    }

    pub fn next_id(self) -> (Self, T) {
        (
            Self {
                next_id: self.next_id.next_id(),
            },
            self.next_id,
        )
    }
}
