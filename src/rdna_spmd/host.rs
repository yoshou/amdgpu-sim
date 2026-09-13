pub(crate) struct Vectors {
    pub bits: u32,
    pub native_masks: bool,
}

impl Vectors {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") { return Self { bits: 512, native_masks: true }; }
            if std::arch::is_x86_feature_detected!("avx2") { return Self { bits: 256, native_masks: false }; }
            return Self { bits: 128, native_masks: false };
        }
        #[cfg(target_arch = "aarch64")]
        { Self { bits: 128, native_masks: false } }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        { Self { bits: 64, native_masks: false } }
    }
    pub fn lanes(&self, element_bits: u32) -> u32 { (self.bits / element_bits).max(1) }
    pub fn scalar_below(&self) -> u32 { 128 }
}
