use super::fiber::Fiber;

fn lanes(width: usize) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}
use super::super::codegen::{Argument, YieldValues};
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
        valid: u64,
        fibers: &[Fiber],
        first_only: bool,
    ) -> u64 {

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
        valid: u64,
        fibers: &[Fiber],
        first_only: bool,
    ) -> u64 {
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
        valid: u64,
        fibers: &[Fiber],
        first_only: bool,
    ) -> u64 {
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
                        (_mm_movemask_ps(_mm_castsi128_ps(zeros)) as u64) ^ 15
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

    pub fn uniform_id(&self, width: usize, valid: u64, fibers: &[Fiber]) -> u32 {
        assert_ne!(valid, 0);
        let first = valid.trailing_zeros() as usize;
        let read = |lane: usize| self.argument(0, lane, width, fibers);
        let id = read(first);
        for lane in 0..fibers.len() * width {
            if valid >> lane & 1 != 0 {
                assert_eq!(read(lane), id, "nonuniform barrier ID");
            }
        }
        id
    }
    pub fn broadcast_result(&self, width: usize, valid: u64, fibers: &[Fiber], value: u32) {
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
            for lane in 0..fibers.len() * width {
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

    pub fn apply_wave(&self, width: usize, valid: u64, fibers: &[Fiber]) {
        let count = fibers.len() * width;
        assert!(matches!(count, 32 | 64), "a wave of {} lanes", count);
        assert_ne!(valid, 0);
        let op = match self.op {
            EffectOp::Wave(WaveOp::Meet) => return,
            EffectOp::Wave(op) => op,
            _ => panic!("not a wave effect"),
        };
        let answer = match op {
            WaveOp::Any => Some((self.query_bits(0, width, valid, fibers, true) != 0) as u32),
            WaveOp::Ballot { high } => {
                let bits = self.query_bits(0, width, valid, fibers, false);
                Some((bits >> if high { 32 } else { 0 }) as u32)
            }
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
            let lane = (arg(1, first) as usize) & (count - 1);
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
            let value = arg(0, (arg(1, first) as usize) & (count - 1));
            self.broadcast_result(width, valid, fibers, value);
            return;
        }
        if op == WaveOp::Wmma {
            assert_eq!(count, 32, "a matrix multiply in a wave of {} lanes", count);
            let full = lanes(count);
            let mut padding = (valid != full).then(|| vec![0u32; self.cells() * width]);
            let mut pointers = [std::ptr::null_mut(); 32];
            for packet in 0..fibers.len() {
                pointers[packet] = if valid >> (packet * width) & lanes(width) != 0 {
                    unsafe { fibers[packet].yield_values().add(self.base * width) }
                } else {
                    padding.as_mut().unwrap().as_mut_ptr()
                };
            }
            if valid != full {
                for lane in 0..count {
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
        let result = evaluate(op, count, valid, |index, lane| {
            self.argument(index, lane, width, fibers)
        });
        if self.uniform_result() {
            self.broadcast_result(width, valid, fibers, result[0]);
        } else {
            for lane in 0..count {
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

fn evaluate(op: WaveOp, count: usize, valid: u64, read: impl Fn(usize, usize) -> u32) -> [u32; 64] {
    assert_ne!(valid, 0, "empty wave");
    let arg = |index, lane: usize| {
        if valid >> lane & 1 != 0 {
            read(index, lane)
        } else {
            0
        }
    };
    let last = count - 1;
    let mut out = [0; 64];
    match op {
        WaveOp::Any => out.fill((0..count).any(|lane| arg(0, lane) != 0) as u32),
        WaveOp::Ballot { high } => {
            let first = if high { 32 } else { 0 };
            let mask = (0..32).fold(0, |mask, k| mask | ((arg(0, first + k) != 0) as u32) << k);
            out.fill(mask);
        }
        WaveOp::ReadFirstLane => {
            let lane = (0..count).find(|&lane| arg(1, lane) != 0).unwrap_or(0);
            out.fill(arg(0, lane));
        }
        WaveOp::ReadLane => {
            for lane in 0..count {
                out[lane] = arg(0, arg(1, lane) as usize & last);
            }
        }
        WaveOp::WriteLane => {
            let first = valid.trailing_zeros() as usize;
            assert!(first < count, "empty wave");
            let value = arg(0, first);
            let selector = arg(1, first);
            for lane in 0..count {
                if valid >> lane & 1 != 0 {
                    assert_eq!(arg(0, lane), value, "nonuniform writelane value");
                    assert_eq!(arg(1, lane), selector, "nonuniform writelane lane");
                }
                out[lane] = arg(2, lane);
            }
            out[selector as usize & last] = value;
        }
        WaveOp::Bpermute | WaveOp::BpermuteFi => {
            for lane in 0..count {
                let src = (arg(0, lane) >> 2) as usize & last;
                out[lane] = if op == WaveOp::BpermuteFi || arg(2, src) != 0 {
                    arg(1, src)
                } else {
                    0
                };
            }
        }
        WaveOp::Wmma => panic!("WMMA uses the existing fragment lowering"),
        WaveOp::Meet => panic!("a meeting exchanges no values"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lanes_of(count: usize, f: impl Fn(usize, usize) -> u32) -> impl Fn(usize, usize) -> u32 {
        move |index, lane| {
            assert!(lane < count, "read lane {} of a wave of {}", lane, count);
            f(index, lane)
        }
    }

    #[test]
    fn evaluate_answers_every_wave_operation_over_the_lanes_of_a_wave_of_64() {
        let count = 64;
        let valid = u64::MAX >> 3;
        let read = lanes_of(count, |index, lane| match index {
            0 => (lane as u32) * 7 % 5,
            1 => (63 - lane) as u32,
            _ => 1000 + lane as u32,
        });
        let bits: Vec<bool> = (0..count).map(|l| valid >> l & 1 != 0 && (l as u32) * 7 % 5 != 0).collect();
        let low = (0..32).fold(0u32, |m, k| m | (bits[k] as u32) << k);
        let high = (0..32).fold(0u32, |m, k| m | (bits[32 + k] as u32) << k);
        assert_eq!(evaluate(WaveOp::Ballot { high: false }, count, valid, &read)[0], low);
        assert_eq!(evaluate(WaveOp::Ballot { high: true }, count, valid, &read)[0], high);
        assert_eq!(evaluate(WaveOp::Any, count, valid, &read)[0], 1);
        let first = (0..count).find(|&l| valid >> l & 1 != 0 && (63 - l) != 0).unwrap();
        assert_eq!(evaluate(WaveOp::ReadFirstLane, count, valid, &read)[0], (first as u32) * 7 % 5);
        let lanes = evaluate(WaveOp::ReadLane, count, valid, &read);
        for l in (0..count).filter(|&l| valid >> l & 1 != 0) {
            let source = 63 - l;
            let want = if valid >> source & 1 != 0 { (source as u32) * 7 % 5 } else { 0 };
            assert_eq!(lanes[l], want, "readlane at lane {}", l);
        }
        let permuted = evaluate(WaveOp::BpermuteFi, count, valid, |index, lane| match index {
            0 => ((lane + 40) * 4) as u32,
            1 => 500 + lane as u32,
            _ => 1,
        });
        for l in (0..count).filter(|&l| valid >> l & 1 != 0) {
            let source = (l + 40) % 64;
            let want = if valid >> source & 1 != 0 { 500 + source as u32 } else { 0 };
            assert_eq!(permuted[l], want, "bpermute at lane {}", l);
        }
    }

    #[test]
    fn evaluate_writes_the_lane_a_selector_picks_modulo_the_wave() {
        for (count, selector, lane) in [(64, 45u32, 45usize), (64, 109, 45), (32, 45, 13)] {
            let valid = if count == 64 { u64::MAX } else { u32::MAX as u64 };
            let out = evaluate(WaveOp::WriteLane, count, valid, |index, l| match index {
                0 => 77,
                1 => selector,
                _ => l as u32,
            });
            for l in 0..count {
                assert_eq!(out[l], if l == lane { 77 } else { l as u32 }, "wave of {}, selector {}, lane {}", count, selector, l);
            }
        }
    }
}
