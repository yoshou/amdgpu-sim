use super::fiber::Fiber;

fn lanes(width: usize) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}
use super::super::codegen::yields::{Argument, YieldValues};
use super::super::ir::{EffectOp, WaveOp};

impl YieldValues {
    #[inline]
    fn argument(&self, index: usize, lane: usize, width: usize, fibers: &[Fiber]) -> u32 {
        let offset = match self.arguments[index] {
            Argument::Constant(value) => return value,
            Argument::Uniform => (self.base + index) * width,
            Argument::Lane => (self.base + index) * width + lane % width,
        };
        unsafe { *fibers[lane / width].yield_values().add(offset) }
    }

    #[inline(never)]
    fn query_bits(
        &self,
        index: usize,
        width: usize,
        valid: u32,
        fibers: &[Fiber],
        first_only: bool,
    ) -> u32 {

        #[cfg(target_arch = "x86_64")]
        if width >= 4 {
            return self.query_bits_simd(index, width, valid, fibers, first_only);
        }
        self.query_bits_scalar(index, width, valid, fibers, first_only)
    }

    #[inline(never)]
    fn query_bits_scalar(
        &self,
        index: usize,
        width: usize,
        valid: u32,
        fibers: &[Fiber],
        first_only: bool,
    ) -> u32 {
        if let Argument::Constant(value) = self.arguments[index] {
            return if value == 0 { 0 } else { valid };
        }
        let uniform = matches!(self.arguments[index], Argument::Uniform);
        let mut result = 0;
        for (packet, fiber) in fibers.iter().enumerate() {
            let shift = packet * width;
            let mask = (valid >> shift) & lanes(width);
            if mask == 0 {
                continue;
            }
            let ptr = unsafe { fiber.yield_values().add((self.base + index) * width) };
            if uniform {
                if unsafe { *ptr } != 0 {
                    result |= mask << shift;
                }
            } else {
                for lane in 0..width {
                    if mask >> lane & 1 != 0 && unsafe { *ptr.add(lane) } != 0 {
                        result |= 1 << (shift + lane);
                        if first_only {
                            return result;
                        }
                    }
                }
            }
            if first_only && result != 0 {
                return result;
            }
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    fn query_bits_simd(
        &self,
        index: usize,
        width: usize,
        valid: u32,
        fibers: &[Fiber],
        first_only: bool,
    ) -> u32 {
        if let Argument::Constant(value) = self.arguments[index] {
            return if value == 0 { 0 } else { valid };
        }
        let uniform = matches!(self.arguments[index], Argument::Uniform);
        let mut result = 0;
        for (packet, fiber) in fibers.iter().enumerate() {
            let shift = packet * width;
            let mask = (valid >> shift) & lanes(width);
            if mask == 0 {
                continue;
            }
            let ptr = unsafe { fiber.yield_values().add((self.base + index) * width) };
            #[cfg(target_arch = "x86_64")]
            if !uniform && width >= 4 && mask == lanes(width) {

                use std::arch::x86_64::{
                    _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_movemask_ps,
                    _mm_setzero_si128,
                };
                let mut lane = 0;
                while lane < width {
                    let bits = unsafe {
                        let values = _mm_loadu_si128(ptr.add(lane).cast());
                        let zeros = _mm_cmpeq_epi32(values, _mm_setzero_si128());
                        (_mm_movemask_ps(_mm_castsi128_ps(zeros)) as u32) ^ 15
                    };
                    result |= bits << (shift + lane);
                    if first_only && result != 0 {
                        return result;
                    }
                    lane += 4;
                }
                continue;
            }
            if uniform {
                if unsafe { *ptr } != 0 {
                    result |= mask << shift;
                }
            } else {
                for lane in 0..width {
                    if mask >> lane & 1 != 0 && unsafe { *ptr.add(lane) } != 0 {
                        result |= 1 << (shift + lane);
                        if first_only {
                            return result;
                        }
                    }
                }
            }
            if first_only && result != 0 {
                return result;
            }
        }
        result
    }

    pub fn uniform_id(&self, width: usize, valid: u32, fibers: &[Fiber]) -> u32 {
        assert_ne!(valid, 0);
        let first = valid.trailing_zeros() as usize;
        let read = |lane: usize| self.argument(0, lane, width, fibers);
        let id = read(first);
        for lane in 0..32 {
            if valid >> lane & 1 != 0 {
                assert_eq!(read(lane), id, "nonuniform barrier ID");
            }
        }
        id
    }
    pub fn broadcast_result(&self, width: usize, valid: u32, fibers: &[Fiber], value: u32) {
        assert_eq!(self.outputs.len(), 1);
        if self.uniform_result() {

            for (packet, fiber) in fibers.iter().enumerate() {
                if valid >> (packet * width) & lanes(width) != 0 {
                    unsafe {
                        *fiber
                            .yield_values()
                            .add((self.base + self.output_base) * width) = value;
                    }
                }
            }
        } else {
            for lane in 0..32 {
                if valid >> lane & 1 != 0 {
                    unsafe {
                        *fibers[lane / width]
                            .yield_values()
                            .add((self.base + self.output_base) * width + lane % width) = value;
                    }
                }
            }
        }
    }

    pub fn apply_wave(&self, width: usize, valid: u32, fibers: &[Fiber]) {
        assert_eq!(fibers.len(), 32 / width);
        assert_ne!(valid, 0);
        let op = match self.op {
            EffectOp::Wave(op) => op,
            _ => panic!("not a wave effect"),
        };
        let answer = match op {
            WaveOp::Any => Some((self.query_bits(0, width, valid, fibers, true) != 0) as u32),
            WaveOp::Ballot => Some(self.query_bits(0, width, valid, fibers, false)),
            WaveOp::ReadFirstLane => {
                let bits = self.query_bits(1, width, valid, fibers, true);

                let lane = if bits == 0 {
                    0
                } else {
                    bits.trailing_zeros() as usize
                };
                Some(if valid >> lane & 1 != 0 {
                    self.argument(0, lane, width, fibers)
                } else {
                    0
                })
            }
            _ => None,
        };
        if let Some(answer) = answer {

            self.broadcast_result(width, valid, fibers, answer);
            return;
        }
        let arg = |index: usize, lane: usize| {
            if valid >> lane & 1 == 0 {
                0
            } else {
                self.argument(index, lane, width, fibers)
            }
        };

        if op == WaveOp::WriteLane {
            let first = valid.trailing_zeros() as usize;
            let value = arg(0, first);
            let lane = (arg(1, first) & 31) as usize;
            if valid >> lane & 1 != 0 {
                unsafe {
                    *fibers[lane / width]
                        .yield_values()
                        .add((self.base + self.output_base) * width + lane % width) = value;
                }
            }
            return;
        }
        if op == WaveOp::ReadLane && self.uniform_selector {
            let first = valid.trailing_zeros() as usize;
            let value = arg(0, (arg(1, first) & 31) as usize);
            self.broadcast_result(width, valid, fibers, value);
            return;
        }
        if op == WaveOp::Wmma {
            let mut padding = (valid != u32::MAX).then(|| vec![0u32; self.cells() * width]);
            let mut pointers = [std::ptr::null_mut(); 32];
            for packet in 0..fibers.len() {
                pointers[packet] = if valid >> (packet * width) & lanes(width) != 0 {
                    unsafe { fibers[packet].yield_values().add(self.base * width) }
                } else {
                    padding.as_mut().unwrap().as_mut_ptr()
                };
            }
            if valid != u32::MAX {
                for lane in 0..32 {
                    if valid >> lane & 1 == 0 {
                        for arg in 0..self.inputs.len() {
                            unsafe {
                                *pointers[lane / width].add(arg * width + lane % width) = 0;
                            }
                        }
                    }
                }
            }

            unsafe {
                super::wmma::apply_values(width, pointers.as_ptr());
            }
            return;
        }
        assert!(self.inputs.len() <= 3 && self.outputs.len() == 1);
        let result = evaluate(op, valid, |index, lane| {
            self.argument(index, lane, width, fibers)
        });
        if self.uniform_result() {
            self.broadcast_result(width, valid, fibers, result[0]);
        } else {
            for lane in 0..32 {
                if valid >> lane & 1 != 0 {
                    unsafe {
                        *fibers[lane / width]
                            .yield_values()
                            .add((self.base + self.output_base) * width + lane % width) =
                            result[lane];
                    }
                }
            }
        }
    }
}

pub(crate) fn evaluate(op: WaveOp, valid: u32, read: impl Fn(usize, usize) -> u32) -> [u32; 32] {
    assert_ne!(valid, 0, "empty wave");
    let arg = |index, lane| {
        if valid >> lane & 1 != 0 {
            read(index, lane)
        } else {
            0
        }
    };

    let mut out = [0; 32];
    match op {
        WaveOp::Any => out.fill((0..32).any(|lane| arg(0, lane) != 0) as u32),
        WaveOp::Ballot => {
            let mask = (0..32).fold(0, |mask, lane| mask | ((arg(0, lane) != 0) as u32) << lane);
            out.fill(mask);
        }
        WaveOp::ReadFirstLane => {
            let lane = (0..32).find(|&lane| arg(1, lane) != 0).unwrap_or(0);
            out.fill(arg(0, lane));
        }
        WaveOp::ReadLane => {
            out = std::array::from_fn(|lane| arg(0, (arg(1, lane) & 31) as usize));
        }
        WaveOp::WriteLane => {
            let first = valid.trailing_zeros() as usize;
            assert!(first < 32, "empty wave");
            let value = arg(0, first);
            let selector = arg(1, first);
            out = std::array::from_fn(|lane| {
                if valid >> lane & 1 != 0 {
                    assert_eq!(arg(0, lane), value, "nonuniform writelane value");
                    assert_eq!(arg(1, lane), selector, "nonuniform writelane lane");
                }
                arg(2, lane)
            });
            out[(selector & 31) as usize] = value;
        }
        WaveOp::Bpermute | WaveOp::BpermuteFi => {
            out = std::array::from_fn(|lane| {
                let src = ((arg(0, lane) >> 2) & 31) as usize;
                if op == WaveOp::BpermuteFi || arg(2, src) != 0 {
                    arg(1, src)
                } else {
                    0
                }
            });
        }
        WaveOp::Wmma => panic!("WMMA uses the existing fragment lowering"),
    }
    out
}
