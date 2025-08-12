use crate::buffer::*;
use crate::instructions::*;
use crate::processor::*;
use crate::rdna4_decoder::*;
use crate::rdna_instructions::*;
use crate::rdna_translator::*;

struct F64x8 {
    value0: std::arch::x86_64::__m256d,
    value1: std::arch::x86_64::__m256d,
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn read_vector_source_operand_f64x8(
    elem: usize,
    addr: SourceOperand,
    sgprs: *const u32,
    vgprs: *mut u32,
) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    match addr {
        SourceOperand::ScalarRegister(value) => {
            let value = _mm256_castsi256_pd(_mm256_set1_epi64x(
                ((((*sgprs.wrapping_add((value + 1) as usize)) as u64) << 32)
                    | ((*sgprs.wrapping_add(value as usize)) as u64)) as i64,
            ));
            F64x8 {
                value0: value,
                value1: value,
            }
        }
        SourceOperand::VectorRegister(value) => {
            let value_lo = _mm256_loadu_si256(
                (vgprs.wrapping_add(value as usize * 32 + elem)) as *const __m256i,
            );
            let value_hi = _mm256_loadu_si256(
                (vgprs.wrapping_add(value as usize * 32 + elem + 32)) as *const __m256i,
            );
            let value0 = _mm256_castsi256_pd(_mm256_unpacklo_epi32(value_lo, value_hi));
            let value1 = _mm256_castsi256_pd(_mm256_unpackhi_epi32(value_lo, value_hi));
            F64x8 {
                value0: value0,
                value1: value1,
            }
        }
        SourceOperand::LiteralConstant(value) => {
            let value = _mm256_castsi256_pd(_mm256_set1_epi64x((value as i64) << 32));
            F64x8 {
                value0: value,
                value1: value,
            }
        }
        SourceOperand::IntegerConstant(value) => {
            let value = _mm256_castsi256_pd(_mm256_set1_epi64x(value as i64));
            F64x8 {
                value0: value,
                value1: value,
            }
        }
        SourceOperand::FloatConstant(value) => {
            let value = _mm256_set1_pd(value);
            F64x8 {
                value0: value,
                value1: value,
            }
        }
    }
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn read_vgpr_f64x8(elem: usize, addr: usize, vgprs: *mut u32) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value_lo = _mm256_loadu_si256((vgprs.wrapping_add(addr * 32 + elem)) as *const __m256i);
    let value_hi =
        _mm256_loadu_si256((vgprs.wrapping_add(addr * 32 + elem + 32)) as *const __m256i);
    let value0 = _mm256_castsi256_pd(_mm256_unpacklo_epi32(value_lo, value_hi));
    let value1 = _mm256_castsi256_pd(_mm256_unpackhi_epi32(value_lo, value_hi));
    F64x8 {
        value0: value0,
        value1: value1,
    }
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn mask_store_vgpr_f64x8(
    mask: std::arch::x86_64::__m256i,
    elem: usize,
    addr: usize,
    value: F64x8,
    vgprs: *mut u32,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value0 = _mm256_castpd_ps(value.value0);
    let value1 = _mm256_castpd_ps(value.value1);
    let value_lo = _mm256_shuffle_ps::<0x88>(value0, value1);
    let value_hi = _mm256_shuffle_ps::<0xDD>(value0, value1);
    let value_lo = _mm256_castps_si256(value_lo);
    let value_hi = _mm256_castps_si256(value_hi);

    _mm256_maskstore_epi32(
        (vgprs.wrapping_add(addr * 32 + elem)) as *mut i32,
        mask,
        value_lo,
    );
    _mm256_maskstore_epi32(
        (vgprs.wrapping_add(addr * 32 + elem + 32)) as *mut i32,
        mask,
        value_hi,
    );
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn abs_neg_f64x8(value: F64x8, abs: u8, neg: u8, idx: usize) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value0 = value.value0;
    let value1 = value.value1;

    let (value0, value1) = if (abs >> idx) & 1 != 0 {
        (
            _mm256_andnot_pd(value0, _mm256_set1_pd(-0.0)),
            _mm256_andnot_pd(value1, _mm256_set1_pd(-0.0)),
        )
    } else {
        (value0, value1)
    };
    let (value0, value1) = if (neg >> idx) & 1 != 0 {
        (
            _mm256_xor_pd(value0, _mm256_set1_pd(-0.0)),
            _mm256_xor_pd(value1, _mm256_set1_pd(-0.0)),
        )
    } else {
        (value0, value1)
    };
    F64x8 {
        value0: value0,
        value1: value1,
    }
}

#[target_feature(enable = "fma")]
#[cfg(target_arch = "x86_64")]
unsafe fn fmadd_f64x8(s0: F64x8, s1: F64x8, s2: F64x8) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value0 = _mm256_fmadd_pd(s0.value0, s1.value0, s2.value0);
    let value1 = _mm256_fmadd_pd(s0.value1, s1.value1, s2.value1);
    F64x8 {
        value0: value0,
        value1: value1,
    }
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn add_f64x8(s0: F64x8, s1: F64x8) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value0 = _mm256_add_pd(s0.value0, s1.value0);
    let value1 = _mm256_add_pd(s0.value1, s1.value1);
    F64x8 {
        value0: value0,
        value1: value1,
    }
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn mul_f64x8(s0: F64x8, s1: F64x8) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value0 = _mm256_mul_pd(s0.value0, s1.value0);
    let value1 = _mm256_mul_pd(s0.value1, s1.value1);
    F64x8 {
        value0: value0,
        value1: value1,
    }
}

#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
unsafe fn omod_clamp_f64x8(value: F64x8, omod: u8, clamp: bool) -> F64x8 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let value0 = value.value0;
    let value1 = value.value1;

    let value0 = match omod {
        1 => _mm256_mul_pd(value0, _mm256_set1_pd(2.0)),
        2 => _mm256_mul_pd(value0, _mm256_set1_pd(4.0)),
        3 => _mm256_mul_pd(value0, _mm256_set1_pd(0.5)),
        _ => value0,
    };
    let value1 = match omod {
        1 => _mm256_mul_pd(value1, _mm256_set1_pd(2.0)),
        2 => _mm256_mul_pd(value1, _mm256_set1_pd(4.0)),
        3 => _mm256_mul_pd(value1, _mm256_set1_pd(0.5)),
        _ => value1,
    };

    if clamp {
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        F64x8 {
            value0: _mm256_max_pd(zero, _mm256_min_pd(value0, one)),
            value1: _mm256_max_pd(zero, _mm256_min_pd(value1, one)),
        }
    } else {
        F64x8 {
            value0: value0,
            value1: value1,
        }
    }
}

pub trait RegisterFile<T: Copy> {
    fn new(num_elems: usize, count: usize, default: T) -> Self;
    fn get(&self, elem: usize, idx: usize) -> T;

    fn set(&mut self, elem: usize, idx: usize, val: T);

    fn get_vec(&self, idx: usize) -> &[T];

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]);
}

pub struct RegisterFileImpl<T: Copy> {
    num_elems: usize,
    pub regs: Vec<T>,
}

impl<T: Copy> RegisterFile<T> for RegisterFileImpl<T> {
    fn new(num_elems: usize, count: usize, default: T) -> Self {
        RegisterFileImpl {
            num_elems: num_elems,
            regs: vec![default; num_elems * count],
        }
    }

    fn get(&self, elem: usize, idx: usize) -> T {
        if elem >= self.num_elems {
            panic!("Element index out of bounds");
        }
        self.regs[self.num_elems * idx + elem]
    }

    fn set(&mut self, elem: usize, idx: usize, val: T) {
        if elem >= self.num_elems {
            panic!("Element index out of bounds");
        }
        self.regs[self.num_elems * idx + elem] = val
    }

    fn get_vec(&self, idx: usize) -> &[T] {
        let beg = self.num_elems * idx;
        let end = self.num_elems * (idx + 1);
        &self.regs.as_slice()[beg..end]
    }

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]) {
        for elem in 0..self.num_elems {
            if let Some(val) = vals[elem] {
                self.set(elem, idx, val);
            }
        }
    }
}

pub enum Signals {
    None,
    EndOfProgram,
    Switch,
    Unknown,
}

pub trait Processor {
    fn step(&mut self) -> Signals;
}

#[derive(Copy, Clone, Debug)]
struct Context {
    id: usize,
    pc: usize,
    scc: bool,
    scratch_base: u64,
}

struct SIMD32 {
    ctx: Context,
    next_pc: usize,
    insts: Vec<u8>,
    pub sgprs: RegisterFileImpl<u32>,
    pub vgprs: RegisterFileImpl<u32>,
    num_vgprs: usize,
    insts_blocks: HashMap<u64, InstBlock>,
    translator: RDNATranslator,
}

#[inline(always)]
fn u64_from_u32_u32(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

#[inline(always)]
fn add_u32(a: u32, b: u32, c: u32) -> (u32, bool) {
    let d = (a as u64) + (b as u64) + (c as u64);
    (d as u32, d > (u32::MAX as u64))
}

#[inline(always)]
fn mul_u32(a: u32, b: u32) -> u32 {
    let c = (a as u64) * (b as u64);
    (c & 0xFFFFFFFF) as u32
}

fn get_exp_f64(val: f64) -> i16 {
    let bits: u64 = f64::to_bits(val);
    ((bits >> 52) & 0x7ff) as i16
}

fn div_scale_f64(s0: f64, s1: f64, s2: f64) -> (f64, bool) {
    let mut vcc = false;
    let mut d = s0 * s2 / s1;
    let s1_exp = get_exp_f64(s1);
    let s2_exp = get_exp_f64(s2);
    if s2 == 0.0 || s1 == 0.0 {
        d = f64::NAN;
    } else if s2_exp - s1_exp >= 768 {
        // N/D near MAX_FLOAT
        vcc = true;
        if s0 == s1 {
            // Only scale the denominator
            d = s0 * 128f64.exp2();
        }
    } else if !s1.is_normal() {
        d = s0 * 128f64.exp2();
    } else if (!(1.0 / s1).is_normal()) && (!(s2 / s1).is_normal()) {
        vcc = true;
        if s0 == s1 {
            // Only scale the denominator
            d = s0 * 128f64.exp2();
        }
    } else if !(1.0 / s1).is_normal() {
        d = s0 * (-128f64).exp2();
    } else if !(s2 / s1).is_normal() {
        vcc = true;
        if s0 == s2 {
            // Only scale the numerator
            d = s0 * 128f64.exp2();
        }
    } else if s2_exp <= 53 {
        // Numerator is tiny
        d = s0 * 128f64.exp2();
    }
    (d, vcc)
}

fn div_fixup_f64(s0: f64, s1: f64, s2: f64) -> f64 {
    let sign_out = s1.is_sign_negative() != s2.is_sign_negative();
    if s2.is_nan() {
        s2
    } else if s1.is_nan() {
        s1
    } else if s1 == 0.0 && s2 == 0.0 {
        // 0/0
        u64_to_f64(0xfff8000000000000)
    } else if s1.is_infinite() && s2.is_infinite() {
        // inf/inf
        u64_to_f64(0xfff8000000000000)
    } else if s1 == 0.0 || s2.is_infinite() {
        // x/0, or inf/y
        if sign_out {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    } else if s1.is_infinite() || s2 == 0.0 {
        // x/inf, 0/y
        if sign_out {
            -0.0
        } else {
            0.0
        }
    } else if (get_exp_f64(s2) - get_exp_f64(s1)) < -1075 {
        panic!();
    } else if get_exp_f64(s1) == 2047 {
        panic!();
    } else {
        if sign_out {
            -s0.abs()
        } else {
            s0.abs()
        }
    }
}

fn cmp_class_f64(a: f64, mask: u32) -> bool {
    let mut result = false;
    if (mask & 0x01) != 0 {
        // value is a signaling NaN.
        result |= a.is_nan();
    }
    if (mask & 0x02) != 0 {
        // value is a quiet NaN.
        result |= a.is_nan();
    }
    if (mask & 0x04) != 0 {
        // value is negative infinity.
        result |= a.is_infinite() && a.is_sign_negative();
    }
    if (mask & 0x08) != 0 {
        // value is a negative normal value.
        result |= a.is_normal() && a.is_sign_negative();
    }
    if (mask & 0x10) != 0 {
        // value is a negative denormal value.
        result |= !a.is_normal() && a.is_sign_negative();
    }
    if (mask & 0x20) != 0 {
        // value is negative zero.
        result |= a == 0.0 && a.is_sign_negative();
    }
    if (mask & 0x40) != 0 {
        // value is positive zero.
        result |= a == 0.0 && a.is_sign_positive();
    }
    if (mask & 0x80) != 0 {
        // value is a positive denormal value.
        result |= !a.is_normal() && a.is_sign_positive();
    }
    if (mask & 0x100) != 0 {
        // value is a positive normal value.
        result |= a.is_normal() && a.is_sign_positive();
    }
    if (mask & 0x200) != 0 {
        // value is positive infinity.
        result |= a.is_infinite() && a.is_sign_positive();
    }
    result
}

use num_traits::ops::mul_add::MulAdd;

#[inline(always)]
fn fma<T: MulAdd<Output = T>>(a: T, b: T, c: T) -> T {
    a.mul_add(b, c)
}

#[inline(always)]
fn u64_to_f64(value: u64) -> f64 {
    f64::from_bits(value)
}

#[inline(always)]
fn f32_to_u32(value: f32) -> u32 {
    f32::to_bits(value)
}

fn abs_neg<T: num::Float>(value: T, abs: u8, neg: u8, idx: usize) -> T {
    let mut result = value;
    if ((abs >> idx) & 1) != 0 {
        result = result.abs();
    }
    if ((neg >> idx) & 1) != 0 {
        result = -result;
    }
    result
}

#[inline(always)]
fn u32_to_f32_abs_neg(value: u32, abs: u8, neg: u8, idx: usize) -> f32 {
    let result = f32::from_bits(value);
    abs_neg(result, abs, neg, idx)
}

#[inline(always)]
fn f64_to_u64(value: f64) -> u64 {
    f64::to_bits(value)
}

fn clamp_f64(value: f64, min_value: f64, max_value: f64) -> f64 {
    if value < min_value {
        min_value
    } else if value > max_value {
        max_value
    } else {
        value
    }
}

fn f64_to_u64_omod_clamp(value: f64, omod: u8, clamp: bool) -> u64 {
    let value = match omod {
        1 => value * 2.0,
        2 => value * 4.0,
        3 => value * 0.5,
        _ => value,
    };
    let value = if clamp {
        clamp_f64(value, 0.0, 1.0)
    } else {
        value
    };
    f64::to_bits(value)
}

impl SIMD32 {
    pub fn dispatch(&mut self, entry_addr: usize, setup_data: Vec<RegisterSetupData>) {
        let num_wavefronts = setup_data.len();
        let num_sgprs = 128;

        for i in 0..num_sgprs {
            for slot in 0..16 {
                self.sgprs.set(0, slot * num_sgprs + i, 0);
            }
        }
        for i in 0..(num_wavefronts * self.num_vgprs) {
            for elem in 0..32 {
                self.vgprs.set(elem, i, 0);
            }
        }
        for wavefront in 0..num_wavefronts {
            let vgpr_offset = wavefront * self.num_vgprs;
            let sgprs = setup_data[wavefront].sgprs;
            let vgprs = setup_data[wavefront].vgprs;
            let user_sgpr_count = setup_data[wavefront].user_sgpr_count;
            for i in 0..user_sgpr_count {
                self.sgprs.set(0, wavefront * num_sgprs + i, sgprs[i]);
            }
            self.sgprs.set(0, wavefront * num_sgprs + 126, 0xFFFFFFFF); // EXEC_LO
            self.sgprs.set(0, wavefront * num_sgprs + 127, 0xFFFFFFFF); // EXEC_HI
            self.sgprs
                .set(0, wavefront * num_sgprs + 117, sgprs[user_sgpr_count]); // TTMP9
            self.sgprs.set(
                0,
                wavefront * num_sgprs + 115,
                (sgprs[user_sgpr_count + 2] << 16) | sgprs[user_sgpr_count + 1],
            ); // TTMP7
            for i in 0..16 {
                for elem in 0..32 {
                    self.vgprs.set(elem, vgpr_offset + i, vgprs[i][elem]);
                }
            }
        }

        use std::collections::VecDeque;

        let mut ctxs = VecDeque::new();
        for wavefront in 0..num_wavefronts {
            ctxs.push_back(Context {
                id: wavefront,
                pc: entry_addr,
                scc: false,
                scratch_base: setup_data[wavefront].scratch_base,
            })
        }

        let is_signal_none = |signal: &Signals| match signal {
            Signals::None => true,
            _ => false,
        };

        while !ctxs.is_empty() {
            if let Some(ctx) = ctxs.pop_front() {
                self.ctx = ctx;
            }
            let mut signal = self.step();
            while is_signal_none(&signal) {
                signal = self.step();
            }

            match signal {
                Signals::EndOfProgram => {}
                Signals::Switch => ctxs.push_back(self.ctx),
                _ => panic!(),
            }
        }
    }

    fn step(&mut self) -> Signals {
        let inst_stream = InstStream {
            insts: &self.insts[self.ctx.pc..],
        };
        let pc = self.ctx.pc as u64;
        let block = self.insts_blocks.get_mut(&pc);
        if block.is_some() && self.translator.insts.len() == 0 {
            let block = block.unwrap();
            let terminator = block.terminator.clone();
            let terminator_pc = block.terminator_pc;
            let terminator_next_pc = block.terminator_next_pc;

            let sgprs_ptr =
                self.sgprs.regs.as_mut_ptr().wrapping_add(128 * self.ctx.id) as *mut u32;
            let vgprs_ptr = (self
                .vgprs
                .regs
                .as_mut_ptr()
                .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                as *mut u32;
            let scc_ptr = (&mut self.ctx.scc) as *mut bool;

            block.execute(sgprs_ptr, vgprs_ptr, scc_ptr);

            self.next_pc = terminator_next_pc;
            self.set_pc(terminator_pc);

            let result = self.execute_inst(terminator);
            self.set_pc(self.next_pc as u64);
            result
        } else if let Ok((inst, size)) = decode_rdna4(inst_stream) {
            self.next_pc = self.get_pc() as usize + size;

            self.translator.add_inst(self.ctx.pc as u64, inst.clone());
            let result = if is_terminator(&inst) {
                if self.translator.insts.len() > 0 {
                    if !self
                        .insts_blocks
                        .contains_key(&self.translator.get_address().unwrap())
                    {
                        let mut block = self.translator.build();

                        block.terminator_next_pc = self.next_pc;
                        block.terminator_pc = self.get_pc();

                        self.insts_blocks
                            .insert(self.translator.get_address().unwrap(), block);
                    }

                    let block = self
                        .insts_blocks
                        .get_mut(&self.translator.get_address().unwrap())
                        .unwrap();

                    let sgprs_ptr =
                        self.sgprs.regs.as_mut_ptr().wrapping_add(128 * self.ctx.id) as *mut u32;
                    let vgprs_ptr = (self
                        .vgprs
                        .regs
                        .as_mut_ptr()
                        .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                        as *mut u32;
                    let scc_ptr = (&mut self.ctx.scc) as *mut bool;

                    block.execute(sgprs_ptr, vgprs_ptr, scc_ptr);

                    self.translator.clear();
                }
                self.execute_inst(inst)
            } else {
                Signals::None
            };
            self.set_pc(self.next_pc as u64);

            result
        } else {
            let inst = get_u64(&self.insts, self.ctx.pc);
            println!(
                "Unknown instruction 0x{:08X} at PC: 0x{:08X}",
                inst & 0xFFFFFFFF,
                self.ctx.pc
            );
            Signals::Unknown
        }
    }

    fn get_pc(&self) -> u64 {
        (&self.insts[self.ctx.pc] as *const u8) as u64
    }

    fn set_pc(&mut self, value: u64) {
        let base_ptr = (&self.insts[0] as *const u8) as u64;
        self.ctx.pc = (value - base_ptr) as usize;
    }

    fn is_execz(&self) -> bool {
        self.get_exec() == 0
    }

    fn is_execnz(&self) -> bool {
        !self.is_execz()
    }

    fn is_vccz(&self) -> bool {
        self.get_vcc() == 0
    }

    fn is_vccnz(&self) -> bool {
        !self.is_vccz()
    }

    fn read_sgpr(&self, idx: usize) -> u32 {
        if idx == 124 {
            0 // NULL
        } else {
            self.sgprs.get(0, self.ctx.id * 128 + idx)
        }
    }

    fn read_sgpr_pair(&self, idx: usize) -> u64 {
        u64_from_u32_u32(self.read_sgpr(idx), self.read_sgpr(idx + 1))
    }

    fn write_sgpr(&mut self, idx: usize, value: u32) {
        self.sgprs.set(0, self.ctx.id * 128 + idx, value);
    }

    fn read_vgpr(&self, elem: usize, idx: usize) -> u32 {
        if idx >= self.num_vgprs {
            panic!();
        }
        self.vgprs.get(elem, self.num_vgprs * self.ctx.id + idx)
    }

    fn read_vgpr_pair(&self, elem: usize, idx: usize) -> u64 {
        u64_from_u32_u32(self.read_vgpr(elem, idx), self.read_vgpr(elem, idx + 1))
    }

    fn write_vgpr(&mut self, elem: usize, idx: usize, value: u32) {
        if idx >= self.num_vgprs {
            panic!();
        }
        self.vgprs
            .set(elem, self.num_vgprs * self.ctx.id + idx, value);
    }

    fn write_vgpr_pair(&mut self, elem: usize, idx: usize, value: u64) {
        self.write_vgpr(elem, idx, (value & 0xFFFFFFFF) as u32);
        self.write_vgpr(elem, idx + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }

    fn set_sgpr_bit(&mut self, idx: usize, bit: usize, value: bool) {
        if bit >= 32 {
            let mask = 1 << (bit - 32);
            let old_value = self.read_sgpr(idx + 1);
            self.write_sop_dst(
                idx + 1,
                (old_value & !mask) | ((value as u32) << (bit - 32)),
            );
        } else {
            let mask = 1 << bit;
            let old_value = self.read_sgpr(idx);
            self.write_sop_dst(idx, (old_value & !mask) | ((value as u32) << bit));
        }
    }

    fn get_exec(&self) -> u32 {
        self.read_sgpr(126)
    }

    fn set_exec(&mut self, value: u32) {
        self.write_sgpr(126, value);
    }

    fn get_exec_bit(&self, elem: usize) -> bool {
        if elem >= 32 {
            ((self.read_sgpr(127) >> (elem - 32)) & 1) != 0
        } else {
            ((self.read_sgpr(126) >> elem) & 1) != 0
        }
    }

    fn set_exec_bit(&mut self, elem: usize, value: bool) {
        if elem >= 32 {
            let mask = 1 << (elem - 32);
            let old_value = self.read_sgpr(127);
            self.write_sgpr(127, (old_value & !mask) | ((value as u32) << (elem - 32)));
        } else {
            let mask = 1 << elem;
            let old_value = self.read_sgpr(126);
            self.write_sgpr(126, (old_value & !mask) | ((value as u32) << elem));
        }
    }

    fn get_vcc(&self) -> u32 {
        self.read_sgpr(106)
    }

    fn get_vcc_bit(&self, elem: usize) -> bool {
        if elem >= 32 {
            ((self.read_sgpr(107) >> (elem - 32)) & 1) != 0
        } else {
            ((self.read_sgpr(106) >> elem) & 1) != 0
        }
    }

    fn set_vcc_bit(&mut self, elem: usize, value: bool) {
        if elem >= 32 {
            let mask = 1 << (elem - 32);
            let old_value = self.read_sgpr(107);
            self.write_sgpr(107, (old_value & !mask) | ((value as u32) << (elem - 32)));
        } else {
            let mask = 1 << elem;
            let old_value = self.read_sgpr(106);
            self.write_sgpr(106, (old_value & !mask) | ((value as u32) << elem));
        }
    }

    fn write_sop_dst(&mut self, addr: usize, value: u32) {
        match addr {
            0..=105 => self.write_sgpr(addr, value),
            106 => self.write_sgpr(addr, value),
            107 => self.write_sgpr(addr, value),
            108..=123 => self.write_sgpr(addr, value),
            124 => {}                            // NULL
            126 => self.write_sgpr(addr, value), // EXEC_LO
            127 => self.write_sgpr(addr, value), // EXEC_HI
            _ => panic!(),
        }
    }

    fn write_sop_dst_pair(&mut self, addr: usize, value: u64) {
        self.write_sop_dst(addr, (value & 0xFFFFFFFF) as u32);
        self.write_sop_dst(addr + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }

    fn read_scalar_source_operand_u32(&self, addr: SourceOperand) -> u32 {
        match addr {
            SourceOperand::LiteralConstant(value) => value,
            SourceOperand::IntegerConstant(value) => (value & 0xFFFFFFFF) as u32,
            SourceOperand::FloatConstant(value) => f32_to_u32(value as f32),
            SourceOperand::ScalarRegister(value) => self.read_sgpr(value as usize),
            SourceOperand::VectorRegister(_) => panic!(),
        }
    }

    fn read_scalar_source_operand_u64(&self, addr: SourceOperand) -> u64 {
        match addr {
            SourceOperand::LiteralConstant(value) => value as u64,
            SourceOperand::IntegerConstant(value) => value,
            SourceOperand::FloatConstant(value) => f64_to_u64(value),
            SourceOperand::ScalarRegister(value) => self.read_sgpr_pair(value as usize),
            SourceOperand::VectorRegister(_) => panic!(),
        }
    }

    fn read_vector_source_operand_u32(&self, elem: usize, addr: SourceOperand) -> u32 {
        match addr {
            SourceOperand::LiteralConstant(value) => value,
            SourceOperand::IntegerConstant(value) => (value & 0xFFFFFFFF) as u32,
            SourceOperand::FloatConstant(value) => f32_to_u32(value as f32),
            SourceOperand::ScalarRegister(value) => self.read_sgpr(value as usize),
            SourceOperand::VectorRegister(value) => self.read_vgpr(elem, value as usize),
        }
    }

    fn read_vector_source_operand_u64(&self, elem: usize, addr: SourceOperand) -> u64 {
        match addr {
            SourceOperand::LiteralConstant(value) => value as u64,
            SourceOperand::IntegerConstant(value) => value,
            SourceOperand::FloatConstant(value) => f64_to_u64(value),
            SourceOperand::ScalarRegister(value) => self.read_sgpr_pair(value as usize),
            SourceOperand::VectorRegister(value) => self.read_vgpr_pair(elem, value as usize),
        }
    }

    fn read_vector_source_operand_f64(&self, elem: usize, addr: SourceOperand) -> f64 {
        match addr {
            SourceOperand::LiteralConstant(value) => u64_to_f64((value as u64) << 32),
            SourceOperand::IntegerConstant(value) => u64_to_f64(value),
            SourceOperand::FloatConstant(value) => value,
            SourceOperand::ScalarRegister(value) => u64_to_f64(self.read_sgpr_pair(value as usize)),
            SourceOperand::VectorRegister(value) => {
                u64_to_f64(self.read_vgpr_pair(elem, value as usize))
            }
        }
    }

    fn execute_inst(&mut self, inst: InstFormat) -> Signals {
        match inst {
            InstFormat::SOP1(fields) => self.execute_sop1(fields),
            InstFormat::SOP2(fields) => self.execute_sop2(fields),
            InstFormat::SOPC(fields) => self.execute_sopc(fields),
            InstFormat::VOP1(fields) => self.execute_vop1(fields),
            InstFormat::VOP2(fields) => self.execute_vop2(fields),
            InstFormat::VOP3(fields) => self.execute_vop3(fields),
            InstFormat::VOP3SD(fields) => self.execute_vop3sd(fields),
            InstFormat::VOPC(fields) => self.execute_vopc(fields),
            InstFormat::VOPD(fields) => self.execute_vopd(fields),
            InstFormat::SMEM(fields) => self.execute_smem(fields),
            InstFormat::SOPP(fields) => self.execute_sopp(fields),
            InstFormat::VSCRATCH(fields) => self.execute_vscratch(fields),
            InstFormat::VGLOBAL(fields) => self.execute_vglobal(fields),
            _ => unimplemented!(),
        }
    }

    fn execute_sop1(&mut self, inst: SOP1) -> Signals {
        let d = inst.sdst as usize;
        let s0 = inst.ssrc0;

        match inst.op {
            I::S_MOV_B32 => {
                self.s_mov_b32(d, s0);
            }
            I::S_MOV_B64 => {
                self.s_mov_b64(d, s0);
            }
            I::S_AND_SAVEEXEC_B32 => {
                self.s_and_saveexec_b32(d, s0);
            }
            I::S_OR_SAVEEXEC_B32 => {
                self.s_or_saveexec_b32(d, s0);
            }
            I::S_AND_NOT1_SAVEEXEC_B32 => {
                self.s_and_not1_saveexec_b32(d, s0);
            }
            _ => unimplemented!(),
        }

        Signals::None
    }

    fn s_mov_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = s0_value;
        self.write_sop_dst(d, d_value);
    }

    fn s_mov_b64(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let d_value = s0_value;
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_and_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value & exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn s_or_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value | exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn s_and_not1_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value & !exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn execute_sop2(&mut self, inst: SOP2) -> Signals {
        let d = inst.sdst as usize;
        let s0 = inst.ssrc0;
        let s1 = inst.ssrc1;

        match inst.op {
            I::S_ADD_NC_U64 => {
                self.s_add_nc_u64(d, s0, s1);
            }
            I::S_AND_B32 => {
                self.s_and_b32(d, s0, s1);
            }
            I::S_OR_B32 => {
                self.s_or_b32(d, s0, s1);
            }
            I::S_XOR_B32 => {
                self.s_xor_b32(d, s0, s1);
            }
            I::S_AND_NOT1_B32 => {
                self.s_and_not1_b32(d, s0, s1);
            }
            I::S_OR_NOT1_B32 => {
                self.s_or_not1_b32(d, s0, s1);
            }
            I::S_CSELECT_B32 => {
                self.s_cselect_b32(d, s0, s1);
            }
            I::S_MUL_U64 => {
                self.s_mul_u64(d, s0, s1);
            }
            I::S_MUL_I32 => {
                self.s_mul_i32(d, s0, s1);
            }
            I::S_LSHR_B32 => {
                self.s_lshr_b32(d, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn s_add_nc_u64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value.wrapping_add(s1_value);
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_and_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value & s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_or_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value | s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_xor_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value ^ s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_and_not1_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value & !s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_or_not1_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value | !s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_cselect_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = if self.ctx.scc { s0_value } else { s1_value };
        self.write_sop_dst(d, d_value);
    }

    fn s_mul_u64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value * s1_value;
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_mul_i32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        let d_value = s0_value * s1_value;
        self.write_sop_dst(d, d_value as u32);
    }

    fn s_lshr_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        let d_value = s0_value >> (s1_value & 0x1F);
        self.write_sop_dst(d, d_value as u32);
        self.ctx.scc = d_value != 0;
    }

    fn execute_sopc(&mut self, inst: SOPC) -> Signals {
        let s0 = inst.ssrc0;
        let s1 = inst.ssrc1;

        match inst.op {
            I::S_CMP_LG_U32 => {
                self.s_cmp_lg_u32(s0, s1);
            }
            I::S_CMP_LG_U64 => {
                self.s_cmp_lg_u64(s0, s1);
            }
            I::S_CMP_EQ_U64 => {
                self.s_cmp_eq_u64(s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn s_cmp_lg_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value != s1_value;
    }

    fn s_cmp_lg_u64(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        self.ctx.scc = s0_value != s1_value;
    }

    fn s_cmp_eq_u64(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        self.ctx.scc = s0_value == s1_value;
    }

    fn execute_vop1(&mut self, inst: VOP1) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        match inst.op {
            I::V_NOP => {}
            I::V_MOV_B32 => {
                self.v_mov_b32_e32(d, s0);
            }
            I::V_CVT_F64_U32 => {
                self.v_cvt_f64_u32_e32(d, s0);
            }
            I::V_RCP_F64 => {
                self.v_rcp_f64_e32(d, s0);
            }
            I::V_RSQ_F64 => {
                self.v_rsq_f64_e32(d, s0);
            }
            I::V_FRACT_F64 => {
                self.v_fract_f64_e32(d, s0);
            }
            I::V_CVT_I32_F64 => {
                self.v_cvt_i32_f64_e32(d, s0);
            }
            I::V_CVT_F64_I32 => {
                self.v_cvt_f64_i32_e32(d, s0);
            }
            I::V_RNDNE_F64 => {
                self.v_rndne_f64_e32(d, s0);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_mov_b32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cvt_f64_u32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_rcp_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = 1.0 / s0_value;

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_rsq_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = 1.0 / s0_value.sqrt();

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_fract_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value.fract();

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_cvt_i32_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value as i32;

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_cvt_f64_i32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_rndne_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let mut d_value = (s0_value + 0.5).floor();
            if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
                d_value -= 1.0;
            }

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn execute_vop2(&mut self, inst: VOP2) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        let s1 = inst.vsrc1 as usize;
        match inst.op {
            I::V_AND_B32 => {
                self.v_and_b32_e32(d, s0, s1);
            }
            I::V_XOR_B32 => {
                self.v_xor_b32_e32(d, s0, s1);
            }
            I::V_CNDMASK_B32 => {
                self.v_cndmask_b32_e32(d, s0, s1);
            }
            I::V_ADD_NC_U32 => {
                self.v_add_nc_u32_e32(d, s0, s1);
            }
            I::V_MUL_F64 => {
                self.v_mul_f64_e32(d, s0, s1);
            }
            I::V_ADD_F64 => {
                self.v_add_f64_e32(d, s0, s1);
            }
            I::V_MAX_NUM_F64 => {
                self.v_max_num_f64_e32(d, s0, s1);
            }
            I::V_LSHLREV_B32 => {
                self.v_lshlrev_b32_e32(d, s0, s1);
            }
            I::V_LSHRREV_B32 => {
                self.v_lshrrev_b32_e32(d, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_and_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value & s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value ^ s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cndmask_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = if self.get_vcc_bit(elem) {
                s1_value
            } else {
                s0_value
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_nc_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.wrapping_add(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_mul_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        if true {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;

                let bitpos = _mm256_set_epi32(
                    1 << 7,
                    1 << 6,
                    1 << 5,
                    1 << 4,
                    1 << 3,
                    1 << 2,
                    1 << 1,
                    1 << 0,
                );

                let sgprs = self.sgprs.regs.as_ptr().wrapping_add(128 * self.ctx.id) as *const u32;
                let vgprs = (self
                    .vgprs
                    .regs
                    .as_mut_ptr()
                    .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                    as *mut u32;

                for elem in (0..32).step_by(8) {
                    let i = (self.get_exec() >> elem) as i32;
                    let mask =
                        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

                    let s0_value = read_vector_source_operand_f64x8(elem, s0, sgprs, vgprs);
                    let s1_value = read_vgpr_f64x8(elem, s1, vgprs);

                    let d_value = mul_f64x8(s0_value, s1_value);

                    mask_store_vgpr_f64x8(mask, elem, d, d_value, vgprs);
                }
            }
        } else {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let s0_value = self.read_vector_source_operand_f64(elem, s0);
                let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
                let d_value = s0_value * s1_value;
                self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
            }
        }
    }

    fn v_add_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value + s1_value;
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_max_num_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value.max(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_lshlrev_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value << (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshrrev_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn execute_vop3(&mut self, inst: VOP3) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        let s1 = inst.src1;
        let s2 = inst.src2;
        let abs = inst.abs;
        let neg = inst.neg;
        let clamp = inst.cm != 0;
        let omod = inst.omod;
        match inst.op {
            I::V_READLANE_B32 => {
                self.v_readlane_b32(d, s0, s1);
            }
            I::V_WRITELANE_B32 => {
                self.v_writelane_b32(d, s0, s1);
            }
            I::V_AND_B32 => {
                self.v_and_b32_e64(d, s0, s1);
            }
            I::V_BFE_U32 => {
                self.v_bfe_u32(d, s0, s1, s2);
            }
            I::V_CMP_LT_U32 => {
                self.v_cmp_lt_u32_e64(d, s0, s1);
            }
            I::V_MUL_LO_U32 => {
                self.v_mul_lo_u32(d, s0, s1);
            }
            I::V_XOR_B32 => {
                self.v_xor_b32_e64(d, s0, s1);
            }
            I::V_XOR3_B32 => {
                self.v_xor3_b32(d, s0, s1, s2);
            }
            I::V_ADD_NC_U32 => {
                self.v_add_nc_u32_e64(d, s0, s1);
            }
            I::V_ADD3_U32 => {
                self.v_add3_u32(d, s0, s1, s2);
            }
            I::V_CVT_F64_U32 => {
                self.v_cvt_f64_u32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_I32_F64 => {
                self.v_cvt_i32_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_ADD_F64 => {
                self.v_add_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MUL_F64 => {
                self.v_mul_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_RCP_F64 => {
                self.v_rcp_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_RNDNE_F64 => {
                self.v_rndne_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FMA_F64 => {
                self.v_fma_f64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FMAS_F64 => {
                self.v_div_fmas_f64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FIXUP_F64 => {
                self.v_div_fixup_f64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_MIN_NUM_F64 => {
                self.v_min_num_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MAX_NUM_F64 => {
                self.v_max_num_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LT_F64 => {
                self.v_cmp_lt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NGT_F64 => {
                self.v_cmp_ngt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_GT_F64 => {
                self.v_cmp_gt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NLT_F64 => {
                self.v_cmp_nlt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LG_F64 => {
                self.v_cmp_lg_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NGE_F64 => {
                self.v_cmp_nge_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LE_F64 => {
                self.v_cmp_le_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NEQ_F64 => {
                self.v_cmp_neq_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CNDMASK_B32 => {
                self.v_cndmask_b32_e64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_LSHLREV_B32 => {
                self.v_lshlrev_b32_e64(d, s0, s1);
            }
            I::V_LSHRREV_B32 => {
                self.v_lshrrev_b32_e64(d, s0, s1);
            }
            I::V_LSHLREV_B64 => {
                self.v_lshlrev_b64_e64(d, s0, s1);
            }
            I::V_LSHRREV_B64 => {
                self.v_lshrrev_b64(d, s0, s1);
            }
            I::V_OR_B32 => {
                self.v_or_b32_e64(d, s0, s1);
            }
            I::V_LDEXP_F64 => {
                self.v_ldexp_f64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_RSQ_F64 => {
                self.v_rsq_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CMP_CLASS_F64 => {
                self.v_cmp_class_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_XAD_U32 => {
                self.v_xad_u32(d, s0, s1, s2);
            }
            I::V_LSHL_ADD_U32 => {
                self.v_lshl_add_u32(d, s0, s1, s2);
            }
            I::V_CMP_NE_U32 => {
                self.v_cmp_ne_u32_e64(d, s0, s1);
            }
            I::V_CMP_EQ_U32 => {
                self.v_cmp_eq_u32_e64(d, s0, s1);
            }
            I::V_CMP_GT_U32 => {
                self.v_cmp_gt_u32_e64(d, s0, s1);
            }
            I::V_CMP_GT_I32 => {
                self.v_cmp_gt_i32_e64(d, s0, s1);
            }
            I::V_CMP_LT_U64 => {
                self.v_cmp_lt_u64_e64(d, s0, s1);
            }
            I::V_TRIG_PREOP_F64 => {
                self.v_trig_preop_f64(d, s0, s1, abs, neg, clamp, omod);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_readlane_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s1_value = (self.read_scalar_source_operand_u32(s1) as usize) & 0x1F;
        let s0_value = self.read_vector_source_operand_u32(s1_value, s0);
        let d_value = s0_value;
        self.write_sgpr(d, d_value);
    }

    fn v_writelane_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s1_value = (self.read_scalar_source_operand_u32(s1) as usize) & 0x1F;
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = s0_value;
        self.write_vgpr(s1_value, d, d_value);
    }

    fn v_and_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value & s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_bfe_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value >> (s1_value & 0x1F)) & ((1 << (s2_value & 0x1F)) - 1);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cmp_lt_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_mul_lo_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = mul_u32(s0_value, s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value ^ s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor3_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value ^ s1_value) ^ s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_nc_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value.wrapping_add(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add3_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let (d_value, _) = add_u32(s0_value, s1_value, s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cvt_f64_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        _abs: u8,
        _neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_i32_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let d_value = s0_value as i32;

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_add_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        if true {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;

                let bitpos = _mm256_set_epi32(
                    1 << 7,
                    1 << 6,
                    1 << 5,
                    1 << 4,
                    1 << 3,
                    1 << 2,
                    1 << 1,
                    1 << 0,
                );

                let sgprs = self.sgprs.regs.as_ptr().wrapping_add(128 * self.ctx.id) as *const u32;
                let vgprs = (self
                    .vgprs
                    .regs
                    .as_mut_ptr()
                    .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                    as *mut u32;

                for elem in (0..32).step_by(8) {
                    let i = (self.get_exec() >> elem) as i32;
                    let mask =
                        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

                    let s0_value = read_vector_source_operand_f64x8(elem, s0, sgprs, vgprs);
                    let s1_value = read_vector_source_operand_f64x8(elem, s1, sgprs, vgprs);

                    let s0_value = abs_neg_f64x8(s0_value, abs, neg, 0);
                    let s1_value = abs_neg_f64x8(s1_value, abs, neg, 1);

                    let d_value = add_f64x8(s0_value, s1_value);

                    let d_value = omod_clamp_f64x8(d_value, omod, clamp);

                    mask_store_vgpr_f64x8(mask, elem, d, d_value, vgprs);
                }
            }
        } else {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
                let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
                let d_value = s0_value + s1_value;
                self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
            }
        }
    }

    fn v_mul_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value * s1_value;
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rcp_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let d_value = 1.0 / s0_value;
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rndne_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let mut d_value = (s0_value + 0.5).floor();
            if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
                d_value -= 1.0;
            }
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_fma_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        if true {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;

                let bitpos = _mm256_set_epi32(
                    1 << 7,
                    1 << 6,
                    1 << 5,
                    1 << 4,
                    1 << 3,
                    1 << 2,
                    1 << 1,
                    1 << 0,
                );

                let sgprs = self.sgprs.regs.as_ptr().wrapping_add(128 * self.ctx.id) as *const u32;
                let vgprs = (self
                    .vgprs
                    .regs
                    .as_mut_ptr()
                    .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                    as *mut u32;

                for elem in (0..32).step_by(8) {
                    let i = (self.get_exec() >> elem) as i32;
                    let mask =
                        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

                    let s0_value = read_vector_source_operand_f64x8(elem, s0, sgprs, vgprs);
                    let s1_value = read_vector_source_operand_f64x8(elem, s1, sgprs, vgprs);
                    let s2_value = read_vector_source_operand_f64x8(elem, s2, sgprs, vgprs);

                    let s0_value = abs_neg_f64x8(s0_value, abs, neg, 0);
                    let s1_value = abs_neg_f64x8(s1_value, abs, neg, 1);
                    let s2_value = abs_neg_f64x8(s2_value, abs, neg, 2);

                    let d_value = fmadd_f64x8(s0_value, s1_value, s2_value);

                    let d_value = omod_clamp_f64x8(d_value, omod, clamp);

                    mask_store_vgpr_f64x8(mask, elem, d, d_value, vgprs);
                }
            }
        } else {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
                let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
                let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
                let d_value = fma(s0_value, s1_value, s2_value);
                self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
            }
        }
    }

    fn v_div_fmas_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let d_value = if self.get_vcc_bit(elem) {
                64f64.exp2() * fma(s0_value, s1_value, s2_value)
            } else {
                fma(s0_value, s1_value, s2_value)
            };
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_div_fixup_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let d_value = div_fixup_f64(s0_value, s1_value, s2_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_min_num_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value.min(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_max_num_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value.max(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cmp_lt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ngt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nlt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lg_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nge_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value >= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_neq_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cndmask_b32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value =
                u32_to_f32_abs_neg(self.read_vector_source_operand_u32(elem, s0), abs, neg, 0);
            let s1_value =
                u32_to_f32_abs_neg(self.read_vector_source_operand_u32(elem, s1), abs, neg, 1);
            let s2_value = self.read_scalar_source_operand_u32(s2);
            let d_value = if (s2_value >> elem) & 1 != 0 {
                s1_value
            } else {
                s0_value
            };
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_lshlrev_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s1_value << (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshrrev_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshlrev_b64_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s1_value << (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn v_lshrrev_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s1_value >> (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn v_or_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value | s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_ldexp_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as i32;
            let d_value = libm::ldexp(s0_value, s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rsq_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        _abs: u8,
        _neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = 1.0 / s0_value.sqrt();

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_cmp_class_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let s0_values = (0..32)
            .map(|elem| abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0))
            .collect::<Vec<f64>>();
        let s1_values = (0..32)
            .map(|elem| self.read_vector_source_operand_u32(elem, s1))
            .collect::<Vec<u32>>();

        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = s0_values[elem];
            let s1_value = s1_values[elem];
            let d_value = cmp_class_f64(s0_value, s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_xad_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value ^ s1_value).wrapping_add(s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshl_add_u32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value << (s1_value & 0x1F)).wrapping_add(s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cmp_ne_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_i32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as i32;
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_u64_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_trig_preop_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        const TWO_OVER_PI_FRACTION: [u64; 20] = [
            0xBA10AC06608DF8F6,
            0x25D4D7F6BF623F1A,
            0xE2F67A0E73EF14A5,
            0xD45AEA4F758FD7CB,
            0x136E9E8C7ECD3CBF,
            0xDA3EDA6CFD9E4F96,
            0x301FDE5E2316B414,
            0x50763FF12FFFBC0B,
            0x73E93908BF177BF2,
            0xFC827323AC7306A6,
            0x8909D338E04D68BE,
            0x4E7DD1046BEA5D76,
            0x2439FC3BD6396253,
            0xA5C00C925DD413A3,
            0x8AC36E48DC74849B,
            0x2083FCA2C757BD77,
            0xBB81B6C52B327887,
            0x2A53F84EAFA3EA69,
            0x000145F306DC9C88,
            0x0000000000000000,
        ];

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);

            let mut shift = (s1_value & 0x1F) as i32 * 53;
            if get_exp_f64(s0_value) > 1077 {
                shift += get_exp_f64(s0_value) as i32 - 1077;
            }

            let result = get_bits_u64(&TWO_OVER_PI_FRACTION, (1201 - 53 - shift) as usize, 53);
            let mut scale = -53 - shift;

            if get_exp_f64(s0_value) >= 1968 {
                scale += 128;
            }

            let d_value = libm::ldexp(result as f64, scale);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn execute_vop3sd(&mut self, inst: VOP3SD) -> Signals {
        let d0 = inst.vdst as usize;
        let d1 = inst.sdst as usize;
        let s0 = inst.src0;
        let s1 = inst.src1;
        let s2 = inst.src2;
        let neg = inst.neg;
        let clamp = inst.cm != 0;
        let omod = inst.omod;
        match inst.op {
            I::V_ADD_CO_U32 => {
                self.v_add_co_u32(d0, d1, s0, s1);
            }
            I::V_ADD_CO_CI_U32 => {
                self.v_add_co_ci_u32_e64(d0, d1, s0, s1, s2);
            }
            I::V_MAD_CO_U64_U32 => {
                self.v_mad_co_u64_u32(d0, d1, s0, s1, s2);
            }
            I::V_DIV_SCALE_F64 => {
                self.v_div_scale_f64(d0, d1, s0, s1, s2, 0, neg, clamp, omod);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_add_co_u32(&mut self, d0: usize, d1: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (d0_value, d1_value) = s0_value.overflowing_add(s1_value);
            self.write_vgpr(elem, d0, d0_value);
            vcc |= (d1_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d1, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_add_co_ci_u32_e64(
        &mut self,
        d0: usize,
        d1: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_scalar_source_operand_u32(s2);
            let (d0_value, d1_value) = add_u32(s0_value, s1_value, ((s2_value >> elem) & 1) as u32);
            self.write_vgpr(elem, d0, d0_value);
            vcc |= (d1_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d1, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_mad_co_u64_u32(
        &mut self,
        d0: usize,
        d1: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u64;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as u64;
            let s2_value = self.read_vector_source_operand_u64(elem, s2);
            let (d0_value, d1_value) = (s0_value * s1_value).overflowing_add(s2_value);
            self.write_vgpr_pair(elem, d0, d0_value);
            vcc |= (d1_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d1, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_div_scale_f64(
        &mut self,
        d: usize,
        sdst: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let (d_value, flag) = div_scale_f64(s0_value, s1_value, s2_value);

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
            vcc |= (flag as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn execute_vopc(&mut self, inst: VOPC) -> Signals {
        let s0 = inst.src0;
        let s1 = inst.vsrc1 as usize;
        match inst.op {
            I::V_CMP_GT_U32 => {
                self.v_cmp_gt_u32_e32(s0, s1);
            }
            I::V_CMP_NE_U32 => {
                self.v_cmp_ne_u32_e32(s0, s1);
            }
            I::V_CMP_EQ_U32 => {
                self.v_cmp_eq_u32_e32(s0, s1);
            }
            I::V_CMP_LT_U32 => {
                self.v_cmp_lt_u32_e32(s0, s1);
            }
            I::V_CMPX_LT_U32 => {
                self.v_cmpx_lt_u32_e32(s0, s1);
            }
            I::V_CMPX_EQ_U32 => {
                self.v_cmpx_eq_u32_e32(s0, s1);
            }
            I::V_CMPX_LT_I32 => {
                self.v_cmpx_lt_i32_e32(s0, s1);
            }
            I::V_CMP_GT_U64 => {
                self.v_cmp_gt_u64_e32(s0, s1);
            }
            I::V_CMP_GT_F64 => {
                self.v_cmp_gt_f64_e32(s0, s1);
            }
            I::V_CMP_NLT_F64 => {
                self.v_cmp_nlt_f64_e32(s0, s1);
            }
            I::V_CMP_LT_F64 => {
                self.v_cmp_lt_f64_e32(s0, s1);
            }
            I::V_CMP_LE_F64 => {
                self.v_cmp_le_f64_e32(s0, s1);
            }
            I::V_CMP_NGT_F64 => {
                self.v_cmp_ngt_f64_e32(s0, s1);
            }
            I::V_CMPX_NGT_F64 => {
                self.v_cmpx_ngt_f64_e32(s0, s1);
            }
            I::V_CMPX_NGE_F64 => {
                self.v_cmpx_nge_f64_e32(s0, s1);
            }
            I::V_CMPX_NLT_F64 => {
                self.v_cmpx_nlt_f64_e32(s0, s1);
            }
            _ => {
                unimplemented!()
            }
        }
        Signals::None
    }

    fn v_cmp_gt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ne_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_eq_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_i32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nlt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ngt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ngt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nge_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value >= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nlt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn execute_vopd(&mut self, inst: VOPD) -> Signals {
        let mut dual_result0_u32 = [0u32; 32];
        let mut dual_result1_u32 = [0u32; 32];

        let s0 = inst.src0x;
        let s1 = inst.vsrc1x as usize;
        match inst.opx {
            I::V_DUAL_CNDMASK_B32 => {
                self.v_dual_cndmask_b32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_MOV_B32 => {
                self.v_dual_mov_b32(&mut dual_result0_u32, s0);
            }
            _ => unimplemented!(),
        }
        let s0 = inst.src0y;
        let s1 = inst.vsrc1y as usize;
        match inst.opy {
            I::V_DUAL_CNDMASK_B32 => {
                self.v_dual_cndmask_b32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_MOV_B32 => {
                self.v_dual_mov_b32(&mut dual_result1_u32, s0);
            }
            I::V_DUAL_LSHLREV_B32 => {
                self.v_dual_lshlrev_b32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_ADD_NC_U32 => {
                self.v_dual_add_nc_u32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_AND_B32 => {
                self.v_dual_and_b32(&mut dual_result1_u32, s0, s1);
            }
            _ => unimplemented!(),
        }
        let d = inst.vdstx as usize;
        match inst.opx {
            I::V_DUAL_CNDMASK_B32 | I::V_DUAL_MOV_B32 => {
                for elem in 0..32 {
                    if !self.get_exec_bit(elem) {
                        continue;
                    }
                    self.write_vgpr(elem, d, dual_result0_u32[elem]);
                }
            }
            _ => unimplemented!(),
        }
        let d = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as usize;
        match inst.opy {
            I::V_DUAL_CNDMASK_B32
            | I::V_DUAL_MOV_B32
            | I::V_DUAL_LSHLREV_B32
            | I::V_DUAL_ADD_NC_U32
            | I::V_DUAL_AND_B32 => {
                for elem in 0..32 {
                    if !self.get_exec_bit(elem) {
                        continue;
                    }
                    self.write_vgpr(elem, d, dual_result1_u32[elem]);
                }
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_dual_cndmask_b32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = if self.get_vcc_bit(elem) {
                s1_value
            } else {
                s0_value
            };
            d_values[elem] = d_value;
        }
    }

    fn v_dual_mov_b32(&mut self, d_values: &mut [u32], s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value;
            d_values[elem] = d_value;
        }
    }

    fn v_dual_lshlrev_b32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value << (s0_value & 0x1F);
            d_values[elem] = d_value;
        }
    }

    fn v_dual_add_nc_u32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.wrapping_add(s1_value);
            d_values[elem] = d_value;
        }
    }

    fn v_dual_and_b32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value & s1_value;
            d_values[elem] = d_value;
        }
    }

    fn execute_smem(&mut self, inst: SMEM) -> Signals {
        let sdata = inst.sdata as usize;
        let ioffset = inst.ioffset as u64;
        let sbase = (inst.sbase * 2) as usize;
        match inst.op {
            I::S_LOAD_B32 => {
                self.s_load_b32(sdata, sbase, ioffset);
            }
            I::S_LOAD_B64 => {
                self.s_load_b64(sdata, sbase, ioffset);
            }
            I::S_LOAD_B128 => {
                self.s_load_b128(sdata, sbase, ioffset);
            }
            I::S_LOAD_B256 => {
                self.s_load_b256(sdata, sbase, ioffset);
            }
            I::S_LOAD_B512 => {
                self.s_load_b512(sdata, sbase, ioffset);
            }
            I::S_LOAD_B96 => {
                self.s_load_b96(sdata, sbase, ioffset);
            }
            I::S_LOAD_I8 => {
                self.s_load_i8(sdata, sbase, ioffset);
            }
            I::S_LOAD_U8 => {
                self.s_load_u8(sdata, sbase, ioffset);
            }
            I::S_LOAD_I16 => {
                self.s_load_i16(sdata, sbase, ioffset);
            }
            I::S_LOAD_U16 => {
                self.s_load_u16(sdata, sbase, ioffset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn s_load_b32(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const u32;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, data);
    }

    fn s_load_b64(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..2 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b128(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..4 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b256(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..8 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b512(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..16 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b96(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..3 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_i8(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const i8;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, (data as i32) as u32);
    }

    fn s_load_u8(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const u8;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, data as u32);
    }

    fn s_load_i16(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const i16;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, (data as i32) as u32);
    }

    fn s_load_u16(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const u16;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, data as u32);
    }

    fn execute_vscratch(&mut self, inst: VSCRATCH) -> Signals {
        let saddr = inst.saddr as usize;
        let vaddr = inst.vaddr as usize;
        let vsrc = inst.vsrc as usize;
        let vdst = inst.vdst as usize;
        let ioffset = inst.ioffset as u32;
        match inst.op {
            I::SCRATCH_STORE_B32 => {
                self.scratch_store_b32(vaddr, vsrc, saddr, ioffset);
            }
            I::SCRATCH_STORE_B64 => {
                self.scratch_store_b64(vaddr, vsrc, saddr, ioffset);
            }
            I::SCRATCH_LOAD_B32 => {
                self.scratch_load_b32(vaddr, vdst, saddr, ioffset);
            }
            I::SCRATCH_LOAD_B64 => {
                self.scratch_load_b64(vaddr, vdst, saddr, ioffset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn scratch_store_b32(&mut self, _vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, vsrc);
            let vaddr_val = 0u64;
            let saddr_val = self.read_sgpr(saddr) as u64;
            let offset = ((vaddr_val + saddr_val + (ioffset as u64)) / 4 * 32 + (elem as u64)) * 4;
            let addr = self.ctx.scratch_base + offset;

            let ptr = addr as *mut u32;
            unsafe {
                *ptr = data;
            }
        }
    }

    fn scratch_store_b64(&mut self, _vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = ((vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) / 4 * 32
                    + (elem as u64))
                    * 4;
                let addr = self.ctx.scratch_base + offset;

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn scratch_load_b32(&mut self, _vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let vaddr_val = 0u64;
            let saddr_val = self.read_sgpr(saddr) as u64;
            let offset = ((vaddr_val + saddr_val + ioffset as u64) / 4 * 32 + (elem as u64)) * 4;
            let addr = self.ctx.scratch_base + offset;

            let ptr = addr as *mut u32;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn scratch_load_b64(&mut self, _vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = ((vaddr_val + saddr_val + ioffset as u64 + (i as u64 * 4)) / 4 * 32
                    + (elem as u64))
                    * 4;
                let addr = self.ctx.scratch_base + offset;

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn execute_vglobal(&mut self, inst: VGLOBAL) -> Signals {
        let saddr = inst.saddr as usize;
        let vaddr = inst.vaddr as usize;
        let vsrc = inst.vsrc as usize;
        let vdst = inst.vdst as usize;
        let ioffset = inst.ioffset as u32;
        match inst.op {
            I::GLOBAL_STORE_B32 => {
                self.global_store_b32(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_STORE_B64 => {
                self.global_store_b64(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_STORE_B128 => {
                self.global_store_b128(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B32 => {
                self.global_load_b32(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B64 => {
                self.global_load_b64(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B128 => {
                self.global_load_b128(vaddr, vdst, saddr, ioffset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn global_store_b32(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, vsrc);
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u32;
            unsafe {
                *ptr = data;
            }
        }
    }

    fn global_store_b64(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let addr = offset[elem] + (ioffset as u64) + (i as u64 * 4);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn global_store_b128(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let addr = offset[elem] + (ioffset as u64) + (i as u64 * 4);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn global_load_b32(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u32;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn global_load_b64(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let addr = offset[elem] + (ioffset as u64) + (i as u64 * 4);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn global_load_b128(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let ioffset = ((ioffset << 8) as i32) >> 8;
                let addr = offset[elem] as i64 + (ioffset as i64) + (i as i64 * 4);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn execute_sopp(&mut self, inst: SOPP) -> Signals {
        let simm16 = inst.simm16 as i16;
        match inst.op {
            I::S_NOP => {}
            I::S_ENDPGM => return Signals::EndOfProgram,
            I::S_WAIT_ALU => {}
            I::S_WAIT_KMCNT => {}
            I::S_WAIT_LOADCNT => {}
            I::S_CLAUSE => {}
            I::S_DELAY_ALU => {}
            I::S_SENDMSG => {}
            I::S_CBRANCH_EXECZ => {
                if self.is_execz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_EXECNZ => {
                if self.is_execnz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_VCCZ => {
                if self.is_vccz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_VCCNZ => {
                if self.is_vccnz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_SCC0 => {
                if !self.ctx.scc {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_SCC1 => {
                if self.ctx.scc {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_BRANCH => {
                self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
}

#[derive(Debug, Clone)]
struct RegisterSetupData {
    user_sgpr_count: usize,
    sgprs: [u32; 16],
    vgprs: [[u32; 32]; 16],
    scratch_base: u64,
}

fn decode_kernel_desc(kd: &[u8]) -> KernelDescriptor {
    KernelDescriptor {
        group_segment_fixed_size: get_u32(kd, 0) as usize,
        private_segment_fixed_size: get_u32(kd, 4) as usize,
        max_flat_workgroup_size: get_u32(kd, 8) as usize,
        is_dynamic_call_stack: get_bit(kd, 12, 0),
        is_xnack_enabled: get_bit(kd, 12, 1),
        kernel_code_entry_byte_offset: get_u64(kd, 16) as usize,
        enable_sgpr_private_segment_buffer: get_bit(kd, 56, 0),
        enable_sgpr_dispatch_ptr: get_bit(kd, 56, 1),
        enable_sgpr_queue_ptr: get_bit(kd, 56, 2),
        enable_sgpr_kernarg_segment_ptr: get_bit(kd, 56, 3),
        enable_sgpr_dispatch_id: get_bit(kd, 56, 4),
        enable_sgpr_flat_scratch_init: get_bit(kd, 56, 5),
        enable_sgpr_private_segment: get_bit(kd, 56, 6),
        enable_sgpr_grid_workgroup_count_x: get_bit(kd, 56, 7),
        enable_sgpr_grid_workgroup_count_y: get_bit(kd, 57, 0),
        enable_sgpr_grid_workgroup_count_z: get_bit(kd, 57, 1),
        granulated_workitem_vgpr_count: (get_bits(kd, 48, 0, 6) as usize + 1) * 8,
        granulated_wavefront_sgpr_count: 0,
        enable_sgpr_private_segment_wave_offset: get_bit(kd, 52, 0),
        user_sgpr_count: get_bits(kd, 52, 1, 5) as usize,
        enable_trap_handler: get_bit(kd, 52, 6),
        enable_sgpr_workgroup_id_x: get_bit(kd, 52, 7),
        enable_sgpr_workgroup_id_y: get_bit(kd, 52, 8),
        enable_sgpr_workgroup_id_z: get_bit(kd, 52, 9),
        enable_sgpr_workgroup_info: get_bit(kd, 52, 10),
        enable_vgpr_workitem_id: get_bits(kd, 52, 11, 2),
    }
}

struct ComputeUnit {
    simds: Vec<Arc<Mutex<SIMD32>>>,
}

use std::collections::HashMap;

impl ComputeUnit {
    pub fn new(pc: usize, insts: Vec<u8>, num_vgprs: usize) -> Self {
        let mut simds = vec![];
        for _ in 0..2 {
            let num_wave_slot = 16;
            simds.push(Arc::new(Mutex::new(SIMD32 {
                ctx: Context {
                    id: 0,
                    pc: pc,
                    scc: false,
                    scratch_base: 0,
                },
                next_pc: 0,
                insts: insts.clone(),
                sgprs: RegisterFileImpl::new(1, 128 * num_wave_slot, 0),
                vgprs: RegisterFileImpl::new(32, 1536 / 4, 0),
                num_vgprs: num_vgprs,
                insts_blocks: HashMap::new(),
                translator: RDNATranslator::new(),
            })));
        }

        ComputeUnit { simds: simds }
    }
}

struct WorkgroupProcessor {
    cunits: Vec<ComputeUnit>,
}

use std::sync::{Arc, Mutex};

pub struct RDNAProcessor<'a> {
    wgps: Vec<WorkgroupProcessor>,
    entry_address: usize,
    kernel_desc: KernelDescriptor,
    aql_packet_address: u64,
    kernel_args_ptr: u64,
    aql: HsaKernelDispatchPacket<'a>,
    private_seg_buffer: Vec<u8>,
}

unsafe impl<'a> Send for SIMD32 {}

impl<'a> RDNAProcessor<'a> {
    pub fn new(
        aql: &HsaKernelDispatchPacket<'a>,
        num_cunits: usize,
        wavefront_size: usize,
        mem: &Vec<u8>,
    ) -> Self {
        let insts = aql.kernel_object.object.to_vec();
        let kd = aql.kernel_object.offset;
        let kernel_desc = decode_kernel_desc(&insts[kd..(kd + 64)]);
        let aql_packet_address = (aql as *const HsaKernelDispatchPacket) as u64;
        let num_wgps = num_cunits / 2;

        assert!(num_cunits % 2 == 0, "Number of compute units must be even.");
        assert!(wavefront_size == 32, "Wavefront size must be 32.");

        let mut wgps = vec![];
        for _ in 0..num_wgps {
            let mut cunits_in_wgp = vec![];
            for _ in 0..2 {
                let cu = ComputeUnit::new(
                    kd + kernel_desc.kernel_code_entry_byte_offset,
                    mem.clone(),
                    kernel_desc.granulated_workitem_vgpr_count,
                );
                cunits_in_wgp.push(cu);
            }
            let wgp = WorkgroupProcessor {
                cunits: cunits_in_wgp,
            };

            wgps.push(wgp);
        }

        let kernel_args_ptr = aql.kernarg_address.address();
        let entry_address = kd + kernel_desc.kernel_code_entry_byte_offset;

        let private_segment_size = aql.private_segment_size as usize;
        let private_seg_buffer: Vec<u8> =
            vec![0u8; private_segment_size * num_cunits * 2 * 32 * 16];

        // create instance
        RDNAProcessor {
            wgps: wgps,
            kernel_desc: kernel_desc,
            kernel_args_ptr: kernel_args_ptr,
            aql_packet_address: aql_packet_address,
            entry_address: entry_address,
            aql: *aql,
            private_seg_buffer: private_seg_buffer,
        }
    }

    fn dispatch(
        &self,
        workgroup_id_x: u32,
        workgroup_id_y: u32,
        workgroup_id_z: u32,
        workitem_offset: usize,
    ) -> RegisterSetupData {
        let private_seg_ptr = if self.private_seg_buffer.len() > 0 {
            (&self.private_seg_buffer[0] as *const u8) as u64
        } else {
            0
        };

        let kernel_args_ptr = self.kernel_args_ptr;
        let aql_packet_address = self.aql_packet_address;
        let kernel_desc = &self.kernel_desc;
        let private_seg_size = self.aql.private_segment_size as u64;

        // Initialize SGPRS
        let mut sgprs = [0u32; 16];
        let mut sgprs_pos = 0;
        if kernel_desc.enable_sgpr_private_segment_buffer {
            let mut desc_w0 = 0;
            desc_w0 |=
                (private_seg_ptr + workitem_offset as u64 * private_seg_size) & ((1 << 48) - 1);
            desc_w0 |= (private_seg_size & ((1 << 14) - 1)) << 48;
            for i in 0..2 {
                sgprs[sgprs_pos + i] = ((desc_w0 >> (i * 32)) & 0xFFFFFFFF) as u32;
            }
            sgprs_pos += 4;
        }
        if kernel_desc.enable_sgpr_dispatch_ptr {
            sgprs[sgprs_pos] = (aql_packet_address & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((aql_packet_address >> 32) & 0xFFFFFFFF) as u32;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_queue_ptr {
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_kernarg_segment_ptr {
            sgprs[sgprs_pos] = (kernel_args_ptr & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((kernel_args_ptr >> 32) & 0xFFFFFFFF) as u32;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_dispatch_id {
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_flat_scratch_init {
            sgprs[sgprs_pos] = workitem_offset as u32 * self.aql.private_segment_size;
            sgprs[sgprs_pos + 1] = self.aql.private_segment_size;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_x && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_y && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_z && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_x {
            sgprs[sgprs_pos] = workgroup_id_x;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_y {
            sgprs[sgprs_pos] = workgroup_id_y;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_z {
            sgprs[sgprs_pos] = workgroup_id_z;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_info {
            sgprs[sgprs_pos] = 0;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_private_segment_wave_offset {
            sgprs[sgprs_pos] = 0;
        }

        // Initialize VGPRS
        let mut vgprs = [[0u32; 32]; 16];
        let vgprs_pos = 0;
        for i in 0..32 {
            let id_x = (i + workitem_offset) % self.aql.workgroup_size_x as usize;
            vgprs[vgprs_pos][i] = id_x as u32;
        }
        if kernel_desc.enable_vgpr_workitem_id > 0 {
            for i in 0..32 {
                let id_y = ((i + workitem_offset) / self.aql.workgroup_size_x as usize)
                    % self.aql.workgroup_size_y as usize;
                vgprs[vgprs_pos][i] |= (id_y as u32) << 10;
            }
        }
        if kernel_desc.enable_vgpr_workitem_id > 1 {
            for i in 0..32 {
                let id_z = ((i + workitem_offset)
                    / (self.aql.workgroup_size_x * self.aql.workgroup_size_y) as usize)
                    % self.aql.workgroup_size_z as usize;
                vgprs[vgprs_pos][i] |= (id_z as u32) << 20;
            }
        }

        RegisterSetupData {
            user_sgpr_count: kernel_desc.user_sgpr_count,
            sgprs: sgprs,
            vgprs: vgprs,
            scratch_base: private_seg_ptr + workitem_offset as u64 * private_seg_size,
        }
    }

    pub fn execute(&mut self) {
        let workgroup_size_x = self.aql.workgroup_size_x as u32;
        let workgroup_size_y = self.aql.workgroup_size_y as u32;
        let workgroup_size_z = self.aql.workgroup_size_z as u32;

        let workgroup_size = (workgroup_size_x * workgroup_size_y * workgroup_size_z) as usize;

        let num_workgroup_x = (self.aql.grid_size_x + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroup_y = (self.aql.grid_size_y + workgroup_size_y - 1) / workgroup_size_y;
        let num_workgroup_z = (self.aql.grid_size_z + workgroup_size_z - 1) / workgroup_size_z;

        let num_workgroups = num_workgroup_x * num_workgroup_y * num_workgroup_z;

        use indicatif::{ProgressBar, ProgressStyle};
        let bar = ProgressBar::new(num_workgroups as u64);

        bar.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .progress_chars("#>-"));

        let num_wgps = self.wgps.len();

        for workgroup_id_base in (0..num_workgroups).step_by(num_wgps) {
            let mut thread_handles = vec![];
            for wgp_idx in 0..num_wgps {
                let workgroup_id = workgroup_id_base + wgp_idx as u32;
                let workgroup_id_x = workgroup_id % num_workgroup_x;
                let workgroup_id_y = (workgroup_id / num_workgroup_x) % num_workgroup_y;
                let workgroup_id_z =
                    (workgroup_id / (num_workgroup_x * num_workgroup_y)) % num_workgroup_z;

                let entry_address = self.entry_address;

                for cu_idx in 0..2 {
                    for simd_idx in 0..2 {
                        let mut setup_data = vec![];
                        for workitem_id in (0..workgroup_size).step_by(32 * 2 * 2) {
                            setup_data.push(self.dispatch(
                                workgroup_id_x,
                                workgroup_id_y,
                                workgroup_id_z,
                                workitem_id
                                    + cu_idx * 64
                                    + simd_idx * 32
                                    + wgp_idx * workgroup_size,
                            ));
                        }

                        let simd: Arc<Mutex<SIMD32>> =
                            Arc::clone(&self.wgps[wgp_idx].cunits[cu_idx].simds[simd_idx]);

                        let handle = std::thread::spawn(move || {
                            if let Ok(mut v) = simd.lock() {
                                v.dispatch(entry_address, setup_data);
                            }
                        });
                        thread_handles.push(handle);
                    }
                }

                bar.inc(1);
            }

            for t in thread_handles {
                t.join().unwrap();
            }
        }

        let mut sum_block_call_count = HashMap::new();
        let mut sum_block_elapsed_time = HashMap::new();
        let mut sum_instruction_count = HashMap::new();
        for wgp in &self.wgps {
            for cu in &wgp.cunits {
                for simd in &cu.simds {
                    let v = simd.lock().unwrap();
                    for (addr, block) in v.insts_blocks.iter() {
                        *sum_block_call_count.entry(*addr).or_insert(0) += block.call_count;
                        *sum_block_elapsed_time.entry(*addr).or_insert(0) += block.elapsed_time;
                        *sum_instruction_count.entry(*addr).or_insert(0) = block.num_instructions;
                    }
                }
            }
        }

        let mut sorted_blocks: Vec<_> = sum_block_elapsed_time.iter().collect();
        sorted_blocks.sort_by(|a, b| b.1.cmp(a.1));
        println!("Block execution summary:");
        for (addr, elapsed_time) in sorted_blocks {
            let call_count = sum_block_call_count.get(addr).unwrap_or(&0);
            println!(
                "Block at 0x{:08X} executed {} times, total elapsed time: {} ms, instruction count: {}",
                addr,
                call_count,
                (*elapsed_time as f64 / 1_000_000.0),
                sum_instruction_count.get(addr).unwrap_or(&0)
            );
        }

        bar.finish();
    }
}
