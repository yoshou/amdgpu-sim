use super::Ty;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParameterSource {
    Vgpr(u32),
    Sgpr(u32),
    MaskBit(u32),
    Scc,
}

#[derive(Clone, Debug)]
pub(crate) struct Parameter {
    pub source: ParameterSource,
    pub ty: Ty,
}

pub(crate) fn exec_index(inputs: &[Parameter], exec: u32) -> usize {
    inputs
        .iter()
        .position(|p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec))
        .expect("a program without its EXEC mask among the parameters")
}
