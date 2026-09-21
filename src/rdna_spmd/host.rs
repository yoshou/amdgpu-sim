pub struct Vectors {
    pub bits: u32,
}

impl Vectors {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Self { bits: 512 };
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self { bits: 256 };
            }
            return Self { bits: 128 };
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self { bits: 128 }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { bits: 64 }
        }
    }
    pub fn lanes(&self, element_bits: u32) -> u32 {
        (self.bits / element_bits).max(1)
    }
}
