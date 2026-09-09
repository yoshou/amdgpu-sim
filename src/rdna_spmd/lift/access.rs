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

/// Opcodes that write their `vdst:vdst+1` as an f64 result (set the pair fresh).
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

fn is_f64_op(op: I) -> bool {
    // Ops whose vector source operands are read as f64 *pairs* (r, r+1). i32 ops
    // (V_CNDMASK/V_MOV/V_AND/...) read a single register. Over-including a pair
    // for an f64 op's occasional i32 sub-operand (e.g. V_LDEXP src1) is safe;
    // under-counting an i32 op's single read as a pair was the bug.
    let s = format!("{:?}", op);
    s.contains("F64") && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32)
}

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
