#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub offset: u32,
    pub allocation: u64,
    pub pointer: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Environment {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub kernarg: Vec<u8>,
    pub bindings: Vec<Binding>,
    pub exposed: Vec<u64>,
}

impl Environment {
    pub fn binding(&self, offset: u32) -> Option<&Binding> {
        self.bindings.iter().find(|b| b.offset == offset)
    }

    pub fn kernarg_word(&self, offset: u32, bytes: u32) -> u32 {
        let mut word = [0u8; 4];
        for (k, byte) in word.iter_mut().enumerate().take(bytes as usize) {
            *byte = self.kernarg.get(offset as usize + k).copied().unwrap_or(0);
        }
        u32::from_le_bytes(word)
    }

    pub fn workgroup_size(&self) -> u32 {
        self.block.iter().product()
    }
}
