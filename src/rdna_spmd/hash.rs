use std::hash::{BuildHasherDefault, Hasher};

#[derive(Clone, Copy, Default)]
pub struct Mix(u64);

impl Mix {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for Mix {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }
    fn write_u8(&mut self, x: u8) {
        self.add(x as u64);
    }
    fn write_u32(&mut self, x: u32) {
        self.add(x as u64);
    }
    fn write_u64(&mut self, x: u64) {
        self.add(x);
    }
    fn write_usize(&mut self, x: usize) {
        self.add(x as u64);
    }
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Mix>>;
