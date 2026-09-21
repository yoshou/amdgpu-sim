use super::program::Program;

pub(crate) trait Target: Send + Sync {
    fn supports(&self, arch: &str) -> bool;
    fn decode(&self, entry_pc: usize, memory: &[u8]) -> Result<Program, String>;
}
