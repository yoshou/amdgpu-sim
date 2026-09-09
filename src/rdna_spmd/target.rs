use super::program::LiftedFunction;

pub(crate) trait Target: Send + Sync {
    fn supports(&self, arch: &str) -> bool;
    fn decode(&self, entry_pc: usize, memory: &[u8]) -> Result<LiftedFunction, String>;
}
