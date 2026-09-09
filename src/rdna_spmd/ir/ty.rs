#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Ty {
    I1,
    I32,
    I64,
    F32,
    F64,
}
impl Ty {
    pub fn bits(self) -> u32 {
        match self {
            Self::I1 => 1,
            Self::I32 | Self::F32 => 32,
            Self::I64 | Self::F64 => 64,
        }
    }
    pub fn integer(self) -> bool {
        matches!(self, Self::I1 | Self::I32 | Self::I64)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ValueId(pub usize);
