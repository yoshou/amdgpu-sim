//! Decode conservative operand-use envelopes and native address forms.
//! SSA analyses consume the corresponding ValueIds recorded by the lifter.
use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};
fn src_vgpr(op: &SourceOperand, out: &mut Vec<u32>) {
    if let SourceOperand::VectorRegister(r) = op {
        out.push(*r as u32);
        out.push(*r as u32 + 1); // over-approx: covers f64/64-bit pair reads
    }
}

/// All VGPRs an instruction may read (over-approximated upward — never under).
pub fn vgpr_reads(inst: &InstFormat) -> Vec<u32> {
    if let Some((reads, _)) = super::half::registers(inst) { return reads; }
    if let Some(memory)=super::memory::instruction(inst){return memory.reads();}
    if let Some(wave)=super::wave::instruction(inst){return wave.io().reads.vgprs().collect();}
    let mut r = Vec::new();
    match inst {
        InstFormat::VOP1(i) => src_vgpr(&i.src0, &mut r),
        InstFormat::VOP2(i) => {
            src_vgpr(&i.src0, &mut r);
            r.push(i.vsrc1 as u32);
            r.push(i.vsrc1 as u32 + 1);
            if matches!(i.op, I::V_FMAC_F32) { r.push(i.vdst as u32); }
        }
        InstFormat::VOP3(i) => {
            src_vgpr(&i.src0, &mut r);
            src_vgpr(&i.src1, &mut r);
            src_vgpr(&i.src2, &mut r);
            if matches!(i.op, I::V_FMAC_F32) { r.push(i.vdst as u32); }
        }
        InstFormat::VOP3SD(i) => {
            src_vgpr(&i.src0, &mut r);
            src_vgpr(&i.src1, &mut r);
            src_vgpr(&i.src2, &mut r);
        }
        InstFormat::VOPC(i) => {
            src_vgpr(&i.src0, &mut r);
            r.push(i.vsrc1 as u32);
            r.push(i.vsrc1 as u32 + 1);
        }
        InstFormat::VOPD(i) => {
            src_vgpr(&i.src0x, &mut r);
            r.push(i.vsrc1x as u32);
            r.push(i.vsrc1x as u32 + 1);
            src_vgpr(&i.src0y, &mut r);
            r.push(i.vsrc1y as u32);
            r.push(i.vsrc1y as u32 + 1);
        }
        InstFormat::VIMAGE(i) => {
            for reg in [i.vaddr0, i.vaddr1, i.vaddr2, i.vaddr3, i.vaddr4] {
                r.push(reg as u32);
                r.push(reg as u32 + 1);
            }
        }
        InstFormat::VSAMPLE(i) => {
            r.push(i.vaddr0 as u32);
            r.push(i.vaddr0 as u32 + 1);
            r.push(i.vaddr1 as u32);
            r.push(i.vaddr1 as u32 + 1);
        }
        InstFormat::VOP3P(i) => {
            src_vgpr(&i.src0, &mut r);
            src_vgpr(&i.src1, &mut r);
            src_vgpr(&i.src2, &mut r);
            if matches!(i.op, I::V_FMA_MIXLO_F16) { r.push(i.vdst as u32); }
        }
        _ => {} // SALU/SMEM/SOPC read SGPRs only
    }
    r
}

/// PRECISE VGPR reads for divergence analysis: a 32-bit operand reads only `r`
/// (not the over-approximated `{r,r+1}` used for liveness soundness), so the
/// 32-bit address arithmetic doesn't pick up a spurious divergent neighbour and
/// a uniform load address stays provably uniform. f64/64-bit operands read the
/// pair. (Over-counting here would only *add* divergence — conservative/sound —
/// but it loses the uniform-gather→broadcast optimization, so we read precisely.)
fn op_is_pair(op: I) -> bool {
    let s = format!("{:?}", op);
    (s.contains("F64") && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32 | I::V_CVT_I32_F64))
        || matches!(op, I::V_MAD_CO_U64_U32 | I::V_LSHLREV_B64 | I::V_ASHR_I64 | I::V_ASHRREV_I64)
}
fn push_src(r: &mut Vec<u32>, o: &SourceOperand, pair: bool) {
    if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); if pair { r.push(*x as u32 + 1); } }
}
fn push_v(r: &mut Vec<u32>, idx: u8, pair: bool) { r.push(idx as u32); if pair { r.push(idx as u32 + 1); } }
pub fn div_reads(inst: &InstFormat) -> Vec<u32> {
    if let Some((reads, _)) = super::half::registers(inst) { return reads; }
    let mut r = Vec::new();
    match inst {
        InstFormat::VOP1(i) => push_src(&mut r, &i.src0, op_is_pair(i.op)),
        InstFormat::VOP2(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_v(&mut r, i.vsrc1, p); }
        InstFormat::VOP3(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_src(&mut r, &i.src1, p); push_src(&mut r, &i.src2, p); }
        InstFormat::VOP3SD(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_src(&mut r, &i.src1, p); push_src(&mut r, &i.src2, p); }
        InstFormat::VOPC(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_v(&mut r, i.vsrc1, p); }
        InstFormat::VOPD(i) => { push_src(&mut r, &i.src0x, false); push_v(&mut r, i.vsrc1x, false); push_src(&mut r, &i.src0y, false); push_v(&mut r, i.vsrc1y, false); }
        InstFormat::VGLOBAL(i) => {
            // address: 64-bit VGPR pair when saddr is null (124), else a 32-bit offset.
            r.push(i.vaddr as u32);
            if i.saddr == 124 { r.push(i.vaddr as u32 + 1); }
            let store_words = match i.op {
                I::GLOBAL_STORE_B32 => 1, I::GLOBAL_STORE_B64 => 2,
                I::GLOBAL_STORE_B96 => 3, I::GLOBAL_STORE_B128 => 4, _ => 0,
            };
            for k in 0..store_words { r.push(i.vsrc as u32 + k); }
        }
        InstFormat::VFLAT(i) => {
            r.push(i.vaddr as u32);
            if i.saddr == 124 { r.push(i.vaddr as u32 + 1); }
        }
        InstFormat::VSCRATCH(i) => {
            if i.sve != 0 { r.push(i.vaddr as u32); }
        }
        InstFormat::VIMAGE(i) => {
            r.extend([
                i.vaddr0 as u32,
                i.vaddr0 as u32 + 1,
                i.vaddr1 as u32,
                i.vaddr2 as u32,
                i.vaddr2 as u32 + 1,
                i.vaddr2 as u32 + 2,
                i.vaddr3 as u32,
                i.vaddr3 as u32 + 1,
                i.vaddr3 as u32 + 2,
                i.vaddr4 as u32,
                i.vaddr4 as u32 + 1,
                i.vaddr4 as u32 + 2,
            ]);
        }
        InstFormat::VSAMPLE(i) => {
            r.push(i.vaddr0 as u32);
            r.push(i.vaddr1 as u32);
        }
        _ => {}
    }
    r
}

pub fn frame_def(inst: &InstFormat) -> Option<(u32, u32)> {
    if let InstFormat::VOP3SD(i) = inst {
        if matches!(i.op, I::V_MAD_CO_U64_U32) {
            let stride = match (&i.src0, &i.src1) {
                (SourceOperand::VectorRegister(0), SourceOperand::IntegerConstant(c)) => Some(*c as u32),
                (SourceOperand::IntegerConstant(c), SourceOperand::VectorRegister(0)) => Some(*c as u32),
                (SourceOperand::VectorRegister(0), SourceOperand::LiteralConstant(c)) => Some(*c),
                (SourceOperand::LiteralConstant(c), SourceOperand::VectorRegister(0)) => Some(*c),
                _ => None,
            };
            let base_uniform = matches!(i.src2, SourceOperand::ScalarRegister(_) | SourceOperand::IntegerConstant(_) | SourceOperand::LiteralConstant(_));
            if let Some(s) = stride {
                if base_uniform && s > 0 && s <= 256 { return Some((i.vdst as u32, s)); }
            }
        }
    }
    None
}

pub fn uses_private(inst: &InstFormat) -> bool {
    use crate::rdna_instructions::InstFormat::*;
    let has = |o: &SourceOperand| matches!(o, SourceOperand::PrivateBase);
    match inst {
        VOP1(i) => has(&i.src0),
        VOP2(i) => has(&i.src0),
        VOP3(i) => has(&i.src0) || has(&i.src1) || has(&i.src2),
        VOP3SD(i) => has(&i.src0) || has(&i.src1) || has(&i.src2),
        VOPC(i) => has(&i.src0),
        // VSCRATCH addresses a distinct private segment for every packed lane,
        // even when its explicit SGPR/VGPR byte offsets are uniform.
        VSCRATCH(_) => true,
        // FLAT may be redirected into the per-lane private aperture at runtime.
        // Conservatively classify its load results as divergent.
        VFLAT(_) => true,
        VIMAGE(_) | VSAMPLE(_) => true,
        _ => false,
    }
}


/// Opcodes that write their `vdst:vdst+1` as an f64 result (set the pair fresh).
fn is_f64_producer(op: I) -> bool {
    matches!(
        op,
        I::V_ADD_F64
            | I::V_MUL_F64
            | I::V_FMA_F64
            | I::V_MAX_NUM_F64
            | I::V_MIN_NUM_F64
            | I::V_FRACT_F64
            | I::V_RSQ_F64
            | I::V_RCP_F64
            | I::V_SQRT_F64
            | I::V_LDEXP_F64
            | I::V_DIV_SCALE_F64
            | I::V_DIV_FMAS_F64
            | I::V_DIV_FIXUP_F64
            | I::V_TRIG_PREOP_F64
            | I::V_RNDNE_F64
            | I::V_CVT_F64_I32
            | I::V_CVT_F64_U32
    )
}

/// Whether a VOP3-encoded op writes its `vdst` to an SGPR mask (a compare) rather
/// than a VGPR — those don't touch VGPR freshness.
fn vop3_writes_mask(op: I) -> bool {
    let s = format!("{:?}", op);
    s.starts_with("V_CMP") || s.starts_with("V_CMPX")
}

/// Opcodes that write a 64-bit (two-register) integer result.
fn is_wide_int(op: I) -> bool {
    matches!(op, I::V_MAD_CO_U64_U32 | I::V_LSHLREV_B64 | I::V_LSHRREV_B64 | I::V_ASHR_I64 | I::V_ASHRREV_I64)
}

/// The f64 pairs this instruction defines (sets fresh). Global loads of ≥2 words
/// are loaded directly as `double`s into the shadow (see emit), so the even pairs
/// they cover are defined too.
pub(super) fn f64_defs(inst: &InstFormat) -> Vec<u32> {
    match inst {
        InstFormat::VOP1(i) if is_f64_producer(i.op) => vec![i.vdst as u32],
        InstFormat::VOP2(i) if is_f64_producer(i.op) => vec![i.vdst as u32],
        InstFormat::VOP3(i) if is_f64_producer(i.op) => vec![i.vdst as u32],
        InstFormat::VOP3SD(i) if is_f64_producer(i.op) => vec![i.vdst as u32],
        InstFormat::VGLOBAL(i) => {
            let words = match i.op {
                I::GLOBAL_LOAD_B64 => 2,
                I::GLOBAL_LOAD_B96 => 3,
                I::GLOBAL_LOAD_B128 => 4,
                _ => 0,
            };
            (0..words / 2).map(|p| i.vdst as u32 + p * 2).collect()
        }
        _ => vec![],
    }
}

/// All VGPR registers this instruction writes (at 32-bit granularity). Used to
/// clear fresh bits; over-approximation is sound.
pub(in crate::rdna_spmd) fn vgpr_writes(inst: &InstFormat) -> Vec<u32> {
    if let Some((_, dst)) = super::half::registers(inst) { return vec![dst]; }
    if let Some(wave)=super::wave::instruction(inst){return wave.io().writes.vgprs().collect();}
    if matches!(inst,InstFormat::VOP3(i) if matches!(i.op,I::V_S_RCP_F32)){return vec![];}
    if let Some(memory) = super::memory::instruction(inst) { return memory.writes(); }
    match inst {
        InstFormat::VOP1(i) => {
            let w = is_f64_producer(i.op) || matches!(i.op, I::V_CVT_F64_I32 | I::V_CVT_F64_U32);
            if w { vec![i.vdst as u32, i.vdst as u32 + 1] } else { vec![i.vdst as u32] }
        }
        InstFormat::VOP2(i) => {
            if is_f64_producer(i.op) || is_wide_int(i.op) { vec![i.vdst as u32, i.vdst as u32 + 1] } else { vec![i.vdst as u32] }
        }
        InstFormat::VOP3(i) => {
            if vop3_writes_mask(i.op) {
                vec![] // writes an SGPR mask, no VGPR
            } else if is_f64_producer(i.op) || is_wide_int(i.op) {
                vec![i.vdst as u32, i.vdst as u32 + 1]
            } else {
                vec![i.vdst as u32]
            }
        }
        InstFormat::VOP3SD(i) => {
            if is_f64_producer(i.op) || is_wide_int(i.op) {
                vec![i.vdst as u32, i.vdst as u32 + 1]
            } else {
                vec![i.vdst as u32]
            }
        }
        InstFormat::VOPD(i) => {
            // Y-op's real VGPR is (vdsty<<1)|((vdstx&1)^1).
            let dx = i.vdstx as u32;
            let dy = ((i.vdsty as u32) << 1) | ((dx & 1) ^ 1);
            vec![dx, dy]
        }
        InstFormat::VIMAGE(i) if matches!(i.op, I::IMAGE_BVH64_INTERSECT_RAY) => {
            (0..4).map(|k| i.vdata as u32 + k).collect()
        }
        InstFormat::VSAMPLE(i) if matches!(i.op, I::IMAGE_SAMPLE_LZ) => {
            vec![i.vdata as u32]
        }
        InstFormat::VOP3P(i) => match i.op {
            // Cross-lane WMMA writes its 8-VGPR f32 accumulator (it is lifted to a
            // wave-level boundary before compilation, but account for its writes so
            // any overlapping f64 shadow is correctly invalidated if it appears).
            I::V_WMMA_F32_16X16X16_F16 => (0..8).map(|k| i.vdst as u32 + k).collect(),
            _ => vec![i.vdst as u32], // V_FMA_MIXLO_F16 and other packed ops
        },
        _ => vec![],
    }
}



/// Whether a SOP op writes a 64-bit (two-register) scalar result.
fn is_sop_u64(op: I) -> bool {
    matches!(
        op,
        I::S_MOV_B64
            | I::S_ADD_NC_U64
            | I::S_MUL_U64
            | I::S_LSHL_B64
            | I::S_LSHR_B64
            | I::S_ASHR_I64
            | I::S_AND_B64
            | I::S_OR_B64
            | I::S_XOR_B64
            | I::S_CSELECT_B64
    )
}

/// SGPR pairs an instruction defines as a 64-bit value (set the i64 shadow).
pub(super) fn sgpr_u64_defs(inst: &InstFormat) -> Vec<u32> {
    match inst {
        InstFormat::SOP1(i) if is_sop_u64(i.op) => vec![i.sdst as u32],
        InstFormat::SOP2(i) if is_sop_u64(i.op) => vec![i.sdst as u32],
        _ => vec![],
    }
}

/// Native f64 views requested by the original operand encoding. This includes
/// encoded unused operands, preserving the existing disjoint-cell policy.
pub(super) fn f64_pairs(inst: &InstFormat) -> Vec<u32> {
    let reads = |op: I| format!("{op:?}").contains("F64")
        && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32);
    let mut pairs = Vec::new();
    let mut source = |op, src: &SourceOperand| {
        if reads(op) { if let SourceOperand::VectorRegister(r) = src { pairs.push(*r as u32); } }
    };
    let (op, dest) = match inst {
        InstFormat::VOP1(i) => { source(i.op, &i.src0); (Some(i.op), Some(i.vdst as u32)) },
        InstFormat::VOP2(i) => { source(i.op, &i.src0); source(i.op, &SourceOperand::VectorRegister(i.vsrc1)); (Some(i.op), Some(i.vdst as u32)) },
        InstFormat::VOPC(i) => { source(i.op, &i.src0); source(i.op, &SourceOperand::VectorRegister(i.vsrc1)); (None, None) },
        InstFormat::VOP3(i) => {
            source(i.op, &i.src0); source(i.op, &i.src1); source(i.op, &i.src2);
            (Some(i.op), (!format!("{:?}",i.op).contains("V_CMP")).then_some(i.vdst as u32))
        },
        InstFormat::VOP3SD(i) => { source(i.op, &i.src0); source(i.op, &i.src1); source(i.op, &i.src2); (Some(i.op), Some(i.vdst as u32)) },
        _ => (None, None),
    };
    if let (Some(op), Some(dst)) = (op, dest) {
        if reads(op) || matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32) { pairs.push(dst); }
    }
    pairs
}

fn normal_f64_pow2_exponent(op: &SourceOperand) -> bool {
    let value = match op {
        SourceOperand::IntegerConstant(value) => *value as u32 as i32,
        SourceOperand::LiteralConstant(value) => *value as i32,
        SourceOperand::FloatConstant(value) => (*value as f32).to_bits() as i32,
        _ => return false,
    };
    (-1022..=1023).contains(&value)
}

fn normal_pow2_cndmask_def(inst: &InstFormat) -> Option<u32> {
    let InstFormat::VOP3(i) = inst else { return None };
    (matches!(i.op, I::V_CNDMASK_B32)
        && i.abs == 0
        && i.neg == 0
        && normal_f64_pow2_exponent(&i.src0)
        && normal_f64_pow2_exponent(&i.src1))
        .then_some(i.vdst as u32)
}


pub(super) fn sqrt_policy(inst: &InstFormat, before: &super::state::Words, after: &super::state::Words) -> crate::rdna_spmd::sqrt_idiom::Policy {
    use crate::rdna_spmd::sqrt_idiom::{Policy, Shape};
    use super::state::Word;
    let pair = |words: &super::state::Words, slot: u32| [words[&Word::Vgpr(slot)], words[&Word::Vgpr(slot+1)]];
    let input = |op: &SourceOperand| if let SourceOperand::VectorRegister(r) = op { Some(pair(before, *r as u32)) } else { None };
    let shape = if let Some(r) = normal_pow2_cndmask_def(inst) {
        Shape::Exponent(after[&Word::Vgpr(r)])
    } else { match inst {
        InstFormat::VOP3(i) if matches!(i.op, I::V_LDEXP_F64) => Shape::Scale {
            input: input(&i.src0),
            exponent: if let SourceOperand::VectorRegister(r) = i.src1 { Some(before[&Word::Vgpr(r as u32)]) } else { None },
            output: pair(after, i.vdst as u32), source: super::input(i.src0, super::Ty::F64), destination: i.vdst as u32,
            unmodified: i.abs == 0 && i.neg == 0, exponent_unmodified: (i.abs | i.neg) & 2 == 0,
        },
        InstFormat::VOP1(i) if matches!(i.op, I::V_SQRT_F64) => input(&i.src0)
            .map_or(Shape::Other, |input| Shape::Sqrt { input, output: pair(after, i.vdst as u32) }),
        InstFormat::VOP3(i) if matches!(i.op, I::V_CMP_CLASS_F64) => input(&i.src0).map_or(Shape::Other, Shape::Class),
        _ => Shape::Other,
    }};
    let steppable = matches!(inst, InstFormat::VOP1(_) | InstFormat::VOP2(_) | InstFormat::VOP3(_)
        | InstFormat::VOP3SD(_) | InstFormat::VOP3P(_) | InstFormat::VOPC(_) | InstFormat::VOPD(_)
        | InstFormat::SOP1(_) | InstFormat::SOP2(_) | InstFormat::SOPC(_) | InstFormat::SOPK(_) | InstFormat::SOPP(_));
    Policy { shape, steppable, replaced: vgpr_writes(inst).into_iter().filter_map(|r| before.get(&Word::Vgpr(r)).copied()).collect() }
}

fn is_f64_op(op: I) -> bool {
    // Ops whose vector source operands are read as f64 *pairs* (r, r+1). i32 ops
    // (V_CNDMASK/V_MOV/V_AND/...) read a single register. Over-including a pair
    // for an f64 op's occasional i32 sub-operand (e.g. V_LDEXP src1) is safe;
    // under-counting an i32 op's single read as a pair was the bug.
    let s = format!("{:?}", op);
    s.contains("F64") && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32)
}

/// VGPR registers read by an instruction (source operands), at correct width.
pub(super) fn math_reads(inst: &InstFormat) -> Vec<u32> {
    if let Some((reads, _)) = super::half::registers(inst) { return reads; }
    let mut r = Vec::new();
    match inst {
        InstFormat::VOP1(i) => {
            if let SourceOperand::VectorRegister(x) = i.src0 {
                r.push(x as u32);
                if is_f64_op(i.op) { r.push(x as u32 + 1); }
            }
        }
        InstFormat::VOP2(i) => {
            let pair = is_f64_op(i.op);
            if let SourceOperand::VectorRegister(x) = i.src0 { r.push(x as u32); if pair { r.push(x as u32 + 1); } }
            r.push(i.vsrc1 as u32);
            if pair { r.push(i.vsrc1 as u32 + 1); }
        }
        InstFormat::VOP3(i) => {
            let pair = is_f64_op(i.op);
            for o in [&i.src0, &i.src1, &i.src2] {
                if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); if pair { r.push(*x as u32 + 1); } }
            }
        }
        InstFormat::VOP3SD(i) => {
            let pair = is_f64_op(i.op);
            for o in [&i.src0, &i.src1, &i.src2] {
                if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); if pair { r.push(*x as u32 + 1); } }
            }
        }
        InstFormat::VOP3P(i) => {
            // Mirror the VOP3P write side in `freshness::vgpr_writes`. Reads are the
            // base VGPR of each source (V_FMA_MIXLO_F16; the wave-wide WMMA is lifted
            // out before this runs, so its multi-register spans never reach here).
            for o in [&i.src0, &i.src1, &i.src2] {
                if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); }
            }
        }
        InstFormat::VOPC(i) => {
            let pair = is_f64_op(i.op);
            if let SourceOperand::VectorRegister(x) = i.src0 { r.push(x as u32); if pair { r.push(x as u32 + 1); } }
            r.push(i.vsrc1 as u32);
            if pair { r.push(i.vsrc1 as u32 + 1); }
        }
        InstFormat::VOPD(i) => {
            // VOPD packs two 32-bit ops; sources are single registers.
            if let SourceOperand::VectorRegister(x) = i.src0x { r.push(x as u32); }
            if let SourceOperand::VectorRegister(x) = i.src0y { r.push(x as u32); }
            r.push(i.vsrc1x as u32);
            r.push(i.vsrc1y as u32);
        }
        InstFormat::VGLOBAL(i) => {
            r.push(i.vaddr as u32);
            r.push(i.vaddr as u32 + 1); // 64-bit address
            for k in 0..4 { r.push(i.vsrc as u32 + k); } // store data (up to B128)
        }
        _ => {}
    }
    r
}
