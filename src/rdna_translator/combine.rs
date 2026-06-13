use super::*;

// Instruction combine: per-block dead code elimination over the decoded
// instruction stream, run before IR emission.
//
// Rewriting an anchor instruction to compute its result from original
// operands (e.g. V_DIV_FIXUP_F64 emitting the quotient directly) leaves the
// expansion sequence feeding it dead; this pass removes those instructions so
// their cost (real divides in V_DIV_SCALE_F64/V_RCP_F64 and the FMA
// refinement chain) disappears.
//
// Scope is a single block on purpose: blocks are split at every EXEC write,
// so EXEC is constant within a block and a later full write to a register
// kills an earlier one for all active lanes, while inactive-lane values never
// escape the block (the block-end save merges against the block-incoming
// value). A write is therefore provably dead when every register it defines
// is fully rewritten later in the same block with no read in between — no
// cross-block liveness assumptions are needed.

// SGPRs live at 0.., VGPRs at VGPR_BASE.. in the pass's register numbering.
const VGPR_BASE: u32 = 512;

const SGPR_NULL: u32 = 124;

struct InstEffects {
    reads: Vec<u32>,
    // Registers fully written by the instruction. Must be exact or
    // under-approximated: a partial write (e.g. 16-bit halves) must not be
    // listed here.
    kills: Vec<u32>,
    // Pure VALU instruction whose only effect is writing `kills`.
    removable: bool,
    // false: unknown effects; treated as reading every register.
    known: bool,
}

impl InstEffects {
    fn unknown() -> Self {
        InstEffects {
            reads: Vec::new(),
            kills: Vec::new(),
            removable: false,
            known: false,
        }
    }

    fn known(reads: Vec<u32>, kills: Vec<u32>, removable: bool) -> Self {
        InstEffects {
            reads,
            kills,
            removable,
            known: true,
        }
    }
}

fn read_sgpr(reg: u32, words: u32, reads: &mut Vec<u32>) {
    for i in 0..words {
        if reg + i != SGPR_NULL {
            reads.push(reg + i);
        }
    }
}

fn read_vgpr(reg: u32, words: u32, reads: &mut Vec<u32>) {
    for i in 0..words {
        reads.push(VGPR_BASE + reg + i);
    }
}

fn read_src(op: &SourceOperand, words: u32, reads: &mut Vec<u32>) {
    match op {
        SourceOperand::ScalarRegister(reg) => read_sgpr(*reg as u32, words, reads),
        SourceOperand::VectorRegister(reg) => read_vgpr(*reg as u32, words, reads),
        _ => {}
    }
}

fn kill_sgpr(reg: u32, words: u32, kills: &mut Vec<u32>) {
    for i in 0..words {
        if reg + i != SGPR_NULL {
            kills.push(reg + i);
        }
    }
}

fn kill_vgpr(reg: u32, words: u32, kills: &mut Vec<u32>) {
    for i in 0..words {
        kills.push(VGPR_BASE + reg + i);
    }
}

// Width of each operand in 32-bit words for instructions the pass
// understands; (src_words, dst_words). Reads may be over-approximated,
// kills must not be.
fn vop3_arith_widths(op: &I) -> Option<(u32, u32)> {
    match op {
        I::V_FMA_F64
        | I::V_MUL_F64
        | I::V_ADD_F64
        | I::V_MAX_NUM_F64
        | I::V_MIN_NUM_F64
        | I::V_DIV_FMAS_F64
        | I::V_DIV_FIXUP_F64 => Some((2, 2)),
        I::V_LDEXP_F64 | I::V_TRIG_PREOP_F64 => Some((2, 2)),
        I::V_CVT_U32_F64 | I::V_CVT_I32_F64 | I::V_CVT_F32_F64 => Some((2, 1)),
        I::V_CVT_F64_U32 | I::V_CVT_F64_I32 | I::V_CVT_F64_F32 => Some((1, 2)),
        I::V_CNDMASK_B32
        | I::V_MOV_B32
        | I::V_XOR_B32
        | I::V_AND_B32
        | I::V_OR_B32
        | I::V_XOR3_B32
        | I::V_ADD3_U32
        | I::V_LSHLREV_B32
        | I::V_LSHRREV_B32
        | I::V_ASHRREV_I32
        | I::V_ADD_NC_U32
        | I::V_SUB_NC_U32
        | I::V_MUL_LO_U32
        | I::V_FMA_F32
        | I::V_ADD_F32
        | I::V_MUL_F32 => Some((1, 1)),
        _ => None,
    }
}

fn vop1_widths(op: &I) -> Option<(u32, u32)> {
    match op {
        I::V_RCP_F64 | I::V_RSQ_F64 | I::V_SQRT_F64 | I::V_RNDNE_F64 | I::V_FREXP_MANT_F64 => {
            Some((2, 2))
        }
        I::V_FRACT_F64 => Some((2, 2)),
        I::V_FREXP_EXP_I32_F64 | I::V_CVT_U32_F64 | I::V_CVT_I32_F64 | I::V_CVT_F32_F64 => {
            Some((2, 1))
        }
        I::V_CVT_F64_U32 | I::V_CVT_F64_I32 | I::V_CVT_F64_F32 => Some((1, 2)),
        I::V_MOV_B32 => Some((1, 1)),
        _ => None,
    }
}

fn effects_of(inst: &InstFormat) -> InstEffects {
    match inst {
        InstFormat::SOPP(inst) => match inst.op {
            I::S_DELAY_ALU
            | I::S_WAIT_ALU
            | I::S_WAIT_LOADCNT
            | I::S_WAIT_LOADCNT_DSCNT
            | I::S_WAIT_DSCNT
            | I::S_WAIT_KMCNT
            | I::S_WAIT_STORECNT
            | I::S_WAIT_BVHCNT
            | I::S_WAIT_SAMPLECNT
            | I::S_CLAUSE
            | I::S_NOP
            | I::S_BRANCH
            | I::S_ENDPGM => InstEffects::known(Vec::new(), Vec::new(), false),
            I::S_CBRANCH_VCCZ | I::S_CBRANCH_VCCNZ => {
                InstEffects::known(vec![106], Vec::new(), false)
            }
            I::S_CBRANCH_EXECZ | I::S_CBRANCH_EXECNZ => {
                InstEffects::known(vec![126], Vec::new(), false)
            }
            I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 => {
                InstEffects::known(Vec::new(), Vec::new(), false)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::SOP1(inst) => match inst.op {
            I::S_MOV_B32 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            I::S_MOV_B64 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 2, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 2, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 | I::S_OR_SAVEEXEC_B32 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 1, &mut reads);
                read_sgpr(126, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                kill_sgpr(126, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::SOP2(inst) => match inst.op {
            I::S_AND_B32
            | I::S_OR_B32
            | I::S_XOR_B32
            | I::S_AND_NOT1_B32
            | I::S_CSELECT_B32
            | I::S_LSHL_B32
            | I::S_LSHR_B32
            | I::S_MUL_I32 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 1, &mut reads);
                read_src(&inst.ssrc1, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            I::S_ADD_NC_U64 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 2, &mut reads);
                read_src(&inst.ssrc1, 2, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 2, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::SOPC(inst) => {
            // SOPC compares only write SCC, which this pass does not model.
            let mut reads = Vec::new();
            read_src(&inst.ssrc0, 2, &mut reads);
            read_src(&inst.ssrc1, 2, &mut reads);
            InstEffects::known(reads, Vec::new(), false)
        }
        InstFormat::SOPK(inst) => {
            // Conservatively treat the destination as read.
            let mut reads = Vec::new();
            read_sgpr(inst.sdst as u32, 2, &mut reads);
            InstEffects::known(reads, Vec::new(), false)
        }
        InstFormat::VOPC(inst) => {
            let name = format!("{:?}", inst.op);
            let mut reads = Vec::new();
            read_src(&inst.src0, 2, &mut reads);
            read_vgpr(inst.vsrc1 as u32, 2, &mut reads);
            read_sgpr(126, 1, &mut reads);
            let mut kills = Vec::new();
            if name.starts_with("V_CMPX_") {
                kill_sgpr(126, 1, &mut kills);
            } else {
                kill_sgpr(106, 1, &mut kills);
            }
            InstEffects::known(reads, kills, false)
        }
        InstFormat::VOP1(inst) => match vop1_widths(&inst.op) {
            Some((src_words, dst_words)) => {
                let mut reads = Vec::new();
                read_src(&inst.src0, src_words, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, dst_words, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            None => InstEffects::unknown(),
        },
        InstFormat::VOP2(inst) => match vop3_arith_widths(&inst.op) {
            Some((src_words, dst_words)) => {
                let mut reads = Vec::new();
                read_src(&inst.src0, src_words, &mut reads);
                read_vgpr(inst.vsrc1 as u32, src_words, &mut reads);
                if let I::V_CNDMASK_B32 = inst.op {
                    read_sgpr(106, 1, &mut reads);
                }
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, dst_words, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            None => InstEffects::unknown(),
        },
        InstFormat::VOP3(inst) => {
            let name = format!("{:?}", inst.op);
            if name.starts_with("V_CMPX_") {
                let mut reads = Vec::new();
                read_src(&inst.src0, 2, &mut reads);
                read_src(&inst.src1, 2, &mut reads);
                read_sgpr(126, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(126, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            } else if name.starts_with("V_CMP_") {
                // VOP3-encoded compare: vdst is the destination SGPR.
                let mut reads = Vec::new();
                read_src(&inst.src0, 2, &mut reads);
                read_src(&inst.src1, 2, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.vdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            } else if let I::V_DIV_FIXUP_F64 = inst.op {
                // The emitter computes src2/src1 directly and never reads
                // src0, so the refinement chain feeding src0 can die.
                let mut reads = Vec::new();
                read_src(&inst.src1, 2, &mut reads);
                read_src(&inst.src2, 2, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 2, &mut kills);
                InstEffects::known(reads, kills, true)
            } else {
                match vop3_arith_widths(&inst.op) {
                    Some((src_words, dst_words)) => {
                        let mut reads = Vec::new();
                        let src1_words = match inst.op {
                            I::V_LDEXP_F64 | I::V_TRIG_PREOP_F64 => 1,
                            _ => src_words,
                        };
                        read_src(&inst.src0, src_words, &mut reads);
                        read_src(&inst.src1, src1_words, &mut reads);
                        read_src(&inst.src2, src_words, &mut reads);
                        if let I::V_DIV_FMAS_F64 = inst.op {
                            read_sgpr(106, 1, &mut reads);
                        }
                        let mut kills = Vec::new();
                        kill_vgpr(inst.vdst as u32, dst_words, &mut kills);
                        InstEffects::known(reads, kills, true)
                    }
                    None => InstEffects::unknown(),
                }
            }
        }
        InstFormat::VOP3SD(inst) => match inst.op {
            I::V_DIV_SCALE_F64 => {
                let mut reads = Vec::new();
                read_src(&inst.src0, 2, &mut reads);
                read_src(&inst.src1, 2, &mut reads);
                read_src(&inst.src2, 2, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 2, &mut kills);
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            I::V_MAD_CO_U64_U32 => {
                let mut reads = Vec::new();
                read_src(&inst.src0, 1, &mut reads);
                read_src(&inst.src1, 1, &mut reads);
                read_src(&inst.src2, 2, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 2, &mut kills);
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            I::V_ADD_CO_CI_U32 | I::V_SUB_CO_CI_U32 => {
                let mut reads = Vec::new();
                read_src(&inst.src0, 1, &mut reads);
                read_src(&inst.src1, 1, &mut reads);
                read_src(&inst.src2, 1, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 1, &mut kills);
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::VGLOBAL(inst) => {
            let name = format!("{:?}", inst.op);
            let data_words = if name.ends_with("_B128") {
                4
            } else if name.ends_with("_B96") {
                3
            } else if name.ends_with("_B64") {
                2
            } else {
                1
            };
            let mut reads = Vec::new();
            read_vgpr(inst.vaddr as u32, 2, &mut reads);
            if inst.saddr != SGPR_NULL as u8 {
                read_sgpr(inst.saddr as u32, 2, &mut reads);
            }
            read_sgpr(126, 1, &mut reads);
            if name.starts_with("GLOBAL_LOAD") {
                // Loads only write lanes with EXEC set, so the destination is
                // a partial write: model it as a read, never a kill.
                read_vgpr(inst.vdst as u32, data_words, &mut reads);
                InstEffects::known(reads, Vec::new(), false)
            } else if name.starts_with("GLOBAL_STORE") {
                read_vgpr(inst.vsrc as u32, data_words, &mut reads);
                InstEffects::known(reads, Vec::new(), false)
            } else {
                InstEffects::unknown()
            }
        }
        InstFormat::VOPD(inst) => {
            let half =
                |op: &I, src0: &SourceOperand, vsrc1: u8, vdst: u8| -> Option<(Vec<u32>, Vec<u32>)> {
                    let mut reads = Vec::new();
                    let mut kills = Vec::new();
                    match op {
                        I::V_DUAL_MOV_B32 => {
                            read_src(src0, 1, &mut reads);
                        }
                        I::V_DUAL_CNDMASK_B32 => {
                            read_src(src0, 1, &mut reads);
                            read_vgpr(vsrc1 as u32, 1, &mut reads);
                            read_sgpr(106, 1, &mut reads);
                        }
                        I::V_DUAL_ADD_F32
                        | I::V_DUAL_MUL_F32
                        | I::V_DUAL_AND_B32
                        | I::V_DUAL_ADD_NC_U32
                        | I::V_DUAL_LSHLREV_B32 => {
                            read_src(src0, 1, &mut reads);
                            read_vgpr(vsrc1 as u32, 1, &mut reads);
                        }
                        I::V_DUAL_FMAC_F32 => {
                            read_src(src0, 1, &mut reads);
                            read_vgpr(vsrc1 as u32, 1, &mut reads);
                            read_vgpr(vdst as u32, 1, &mut reads);
                        }
                        _ => return None,
                    }
                    kill_vgpr(vdst as u32, 1, &mut kills);
                    Some((reads, kills))
                };

            match (
                half(&inst.opx, &inst.src0x, inst.vsrc1x, inst.vdstx),
                half(&inst.opy, &inst.src0y, inst.vsrc1y, inst.vdsty),
            ) {
                (Some((rx, kx)), Some((ry, ky))) => {
                    let mut reads = rx;
                    reads.extend(ry);
                    let mut kills = kx;
                    kills.extend(ky);
                    InstEffects::known(reads, kills, false)
                }
                _ => InstEffects::unknown(),
            }
        }
        _ => InstEffects::unknown(),
    }
}

fn vgpr_pair(op: &SourceOperand) -> Option<u32> {
    if let SourceOperand::VectorRegister(reg) = op {
        Some(*reg as u32)
    } else {
        None
    }
}

fn operand_eq(a: &SourceOperand, b: &SourceOperand) -> bool {
    match (a, b) {
        (SourceOperand::ScalarRegister(x), SourceOperand::ScalarRegister(y)) => x == y,
        (SourceOperand::VectorRegister(x), SourceOperand::VectorRegister(y)) => x == y,
        (SourceOperand::FloatConstant(x), SourceOperand::FloatConstant(y)) => {
            x.to_bits() == y.to_bits()
        }
        (SourceOperand::IntegerConstant(x), SourceOperand::IntegerConstant(y)) => x == y,
        (SourceOperand::LiteralConstant(x), SourceOperand::LiteralConstant(y)) => x == y,
        _ => false,
    }
}

fn is_const_one(op: &SourceOperand) -> bool {
    matches!(op, SourceOperand::FloatConstant(v) if v.to_bits() == 1.0f64.to_bits())
}

fn is_fma_f64(
    inst: &InstFormat,
) -> Option<(u32, SourceOperand, SourceOperand, SourceOperand, u8, u8)> {
    if let InstFormat::VOP3(i) = inst {
        if matches!(i.op, I::V_FMA_F64) && i.omod == 0 && i.opsel == 0 && i.cm == 0 {
            return Some((i.vdst as u32, i.src0, i.src1, i.src2, i.neg, i.abs));
        }
    }
    None
}

fn is_mul_f64(inst: &InstFormat) -> Option<(SourceOperand, SourceOperand)> {
    match inst {
        InstFormat::VOP3(i) => {
            if matches!(i.op, I::V_MUL_F64)
                && i.neg == 0
                && i.abs == 0
                && i.omod == 0
                && i.cm == 0
            {
                return Some((i.src0, i.src1));
            }
            None
        }
        InstFormat::VOP2(i) => {
            if matches!(i.op, I::V_MUL_F64) {
                return Some((i.src0, SourceOperand::VectorRegister(i.vsrc1)));
            }
            None
        }
        _ => None,
    }
}

fn is_rcp_f64(inst: &InstFormat, src_reg: u32) -> bool {
    match inst {
        InstFormat::VOP1(i) => {
            matches!(i.op, I::V_RCP_F64) && vgpr_pair(&i.src0) == Some(src_reg)
        }
        InstFormat::VOP3(i) => {
            matches!(i.op, I::V_RCP_F64)
                && i.neg == 0
                && i.abs == 0
                && i.omod == 0
                && vgpr_pair(&i.src0) == Some(src_reg)
        }
        _ => false,
    }
}

fn is_div_scale_f64(
    inst: &InstFormat,
) -> Option<(u32, SourceOperand, SourceOperand, SourceOperand)> {
    if let InstFormat::VOP3SD(i) = inst {
        if matches!(i.op, I::V_DIV_SCALE_F64) && i.neg == 0 && i.omod == 0 && i.cm == 0 {
            return Some((i.sdst as u32, i.src0, i.src1, i.src2));
        }
    }
    None
}

// Finds the instruction defining the full VGPR pair `reg` strictly before
// `before`. Fails on a partial-pair write or an instruction with unknown
// effects, which could also write it.
fn find_def(effects: &[InstEffects], before: usize, reg: u32) -> Option<usize> {
    for j in (0..before).rev() {
        let e = &effects[j];
        if !e.known {
            return None;
        }
        let lo = e.kills.contains(&(VGPR_BASE + reg));
        let hi = e.kills.contains(&(VGPR_BASE + reg + 1));
        if lo && hi {
            return Some(j);
        }
        if lo || hi {
            return None;
        }
    }
    None
}

// Matches the compiler's f64 division expansion feeding a V_DIV_FIXUP_F64 at
// `anchor`:
//
//   a   = v_div_scale_f64 (den, den, num)
//   r0  = v_rcp_f64 a
//   t   = v_fma_f64 -a, r, 1.0            (repeated with r = fma(r, t, r))
//   n_s = v_div_scale_f64 vcc, (num, den, num)
//   q   = v_mul_f64 n_s, r
//   f   = v_fma_f64 -a, q, n_s
//   e   = v_div_fmas_f64 f, r, q
//   dst = v_div_fixup_f64 e, den, num
//
// Returns the instruction indices of the chain (everything but the anchor)
// for removal. The expansion's temporaries are dead past the anchor by
// construction; the only assumption made beyond in-block dataflow is that
// their stale values are not read by later blocks, which holds for
// compiler-generated code because the expansion is emitted as a unit.
fn match_div_f64(
    insts: &[InstFormat],
    effects: &[InstEffects],
    anchor: usize,
) -> Option<Vec<usize>> {
    let (e_reg, den, num) = if let InstFormat::VOP3(i) = &insts[anchor] {
        if !matches!(i.op, I::V_DIV_FIXUP_F64) {
            return None;
        }
        (vgpr_pair(&i.src0)?, i.src1, i.src2)
    } else {
        return None;
    };

    let mut matched = Vec::new();

    // e = div_fmas(f, r, q)
    let i_fmas = find_def(effects, anchor, e_reg)?;
    let (f_reg, r_reg, q_reg) = if let InstFormat::VOP3(i) = &insts[i_fmas] {
        if !matches!(i.op, I::V_DIV_FMAS_F64) || i.neg != 0 || i.abs != 0 {
            return None;
        }
        (
            vgpr_pair(&i.src0)?,
            vgpr_pair(&i.src1)?,
            vgpr_pair(&i.src2)?,
        )
    } else {
        return None;
    };
    matched.push(i_fmas);

    // q = n_s * r
    let i_mul = find_def(effects, i_fmas, q_reg)?;
    let (m0, m1) = is_mul_f64(&insts[i_mul])?;
    let ns_reg = if vgpr_pair(&m1) == Some(r_reg) {
        vgpr_pair(&m0)?
    } else if vgpr_pair(&m0) == Some(r_reg) {
        vgpr_pair(&m1)?
    } else {
        return None;
    };
    matched.push(i_mul);

    // f = fma(-a, q, n_s)
    let i_f = find_def(effects, i_fmas, f_reg)?;
    let (_, f0, f1, f2, neg, abs) = is_fma_f64(&insts[i_f])?;
    if neg != 1 || abs != 0 {
        return None;
    }
    let a_reg = vgpr_pair(&f0)?;
    if vgpr_pair(&f1) != Some(q_reg) || vgpr_pair(&f2) != Some(ns_reg) {
        return None;
    }
    matched.push(i_f);

    // n_s = div_scale(num, den, num)
    let i_dsn = find_def(effects, i_mul.min(i_f), ns_reg)?;
    let (_, d0, d1, d2) = is_div_scale_f64(&insts[i_dsn])?;
    if !operand_eq(&d0, &num) || !operand_eq(&d1, &den) || !operand_eq(&d2, &num) {
        return None;
    }
    matched.push(i_dsn);

    // Newton-Raphson refinement: r = fma(r', t, r'), t = fma(-a, r', 1.0),
    // bottoming out at r = rcp(a).
    let mut i_r = find_def(effects, i_mul, r_reg)?;
    let mut found_rcp = false;
    for _ in 0..8 {
        if is_rcp_f64(&insts[i_r], a_reg) {
            matched.push(i_r);
            found_rcp = true;
            break;
        }
        let (_, r0, r1, r2, neg, abs) = is_fma_f64(&insts[i_r])?;
        if neg != 0 || abs != 0 {
            return None;
        }
        let r_prev = vgpr_pair(&r0)?;
        if vgpr_pair(&r2) != Some(r_prev) {
            return None;
        }
        let t_reg = vgpr_pair(&r1)?;
        matched.push(i_r);

        let i_t = find_def(effects, i_r, t_reg)?;
        let (_, t0, t1, t2, tneg, tabs) = is_fma_f64(&insts[i_t])?;
        if tneg != 1 || tabs != 0 {
            return None;
        }
        if vgpr_pair(&t0) != Some(a_reg) || vgpr_pair(&t1) != Some(r_prev) || !is_const_one(&t2)
        {
            return None;
        }
        matched.push(i_t);

        i_r = find_def(effects, i_t, r_prev)?;
    }
    if !found_rcp {
        return None;
    }

    // a = div_scale(den, den, num)
    let earliest = *matched.iter().min().unwrap();
    let i_dsa = find_def(effects, earliest, a_reg)?;
    let (_, a0, a1, a2) = is_div_scale_f64(&insts[i_dsa])?;
    if !operand_eq(&a0, &den) || !operand_eq(&a1, &den) || !operand_eq(&a2, &num) {
        return None;
    }
    matched.push(i_dsa);

    // The chain's results must not be observable outside the matched set:
    // every register a matched instruction writes may only be read by other
    // matched instructions before being fully rewritten. VGPR temporaries may
    // reach the block end (dead past the anchor by the expansion contract);
    // an SGPR written by something other than div_scale must be rewritten
    // within the block, since branches and later blocks may read it.
    for &i in &matched {
        for &reg in &effects[i].kills {
            let mut killed = false;
            for j in (i + 1)..insts.len() {
                let ej = &effects[j];
                if matched.contains(&j) {
                    if ej.kills.contains(&reg) {
                        killed = true;
                        break;
                    }
                    continue;
                }
                if !ej.known {
                    return None;
                }
                if ej.reads.contains(&reg) {
                    return None;
                }
                if ej.kills.contains(&reg) {
                    killed = true;
                    break;
                }
            }
            // div_scale's SGPR flag is overwritten by the emitter with a
            // constant rather than the architected scale flags, so no correct
            // reader depends on it past the consuming div_fmas; any genuine
            // outside reader is already rejected by the loop above.
            let div_scale_sdst = reg < VGPR_BASE
                && matches!(&insts[i], InstFormat::VOP3SD(d) if matches!(d.op, I::V_DIV_SCALE_F64));
            if !killed && reg < VGPR_BASE && !div_scale_sdst {
                return None;
            }
        }
    }

    Some(matched)
}

fn as_rsq_f64(inst: &InstFormat) -> Option<(u32, u32)> {
    match inst {
        InstFormat::VOP1(i) if matches!(i.op, I::V_RSQ_F64) => {
            Some((i.vdst as u32, vgpr_pair(&i.src0)?))
        }
        InstFormat::VOP3(i)
            if matches!(i.op, I::V_RSQ_F64) && i.neg == 0 && i.abs == 0 && i.omod == 0 =>
        {
            Some((i.vdst as u32, vgpr_pair(&i.src0)?))
        }
        _ => None,
    }
}

// (vdst pair, src0, src1) for an unmodified f64 multiply (VOP2 or VOP3).
fn as_mul_f64(inst: &InstFormat) -> Option<(u32, SourceOperand, SourceOperand)> {
    match inst {
        InstFormat::VOP3(i)
            if matches!(i.op, I::V_MUL_F64)
                && i.neg == 0
                && i.abs == 0
                && i.omod == 0
                && i.cm == 0
                && i.opsel == 0 =>
        {
            Some((i.vdst as u32, i.src0, i.src1))
        }
        InstFormat::VOP2(i) if matches!(i.op, I::V_MUL_F64) => {
            Some((i.vdst as u32, i.src0, SourceOperand::VectorRegister(i.vsrc1)))
        }
        _ => None,
    }
}

// (vdst pair, src0, src1, src2, neg) for an unmodified f64 FMA.
fn as_fma_f64(
    inst: &InstFormat,
) -> Option<(u32, SourceOperand, SourceOperand, SourceOperand, u8)> {
    if let InstFormat::VOP3(i) = inst {
        if matches!(i.op, I::V_FMA_F64) && i.abs == 0 && i.omod == 0 && i.cm == 0 && i.opsel == 0 {
            return Some((i.vdst as u32, i.src0, i.src1, i.src2, i.neg));
        }
    }
    None
}

// First index after `start` whose instruction satisfies `pred`. The chain is a
// strict data dependency, so the matching instruction is uniquely pinned by its
// register operands; interleaved scheduling fillers are skipped.
fn find_forward(
    insts: &[InstFormat],
    start: usize,
    end: usize,
    pred: impl Fn(&InstFormat) -> bool,
) -> Option<usize> {
    (start + 1..end).find(|&j| pred(&insts[j]))
}

fn is_mul_xy(inst: &InstFormat, x: u32, y: u32) -> Option<u32> {
    let (dst, s0, s1) = as_mul_f64(inst)?;
    let (a, b) = (vgpr_pair(&s0), vgpr_pair(&s1));
    if (a == Some(x) && b == Some(y)) || (a == Some(y) && b == Some(x)) {
        Some(dst)
    } else {
        None
    }
}

fn is_mul_half(inst: &InstFormat, r: u32) -> Option<u32> {
    let (dst, s0, s1) = as_mul_f64(inst)?;
    let half = |o: &SourceOperand| matches!(o, SourceOperand::FloatConstant(v) if v.to_bits() == 0.5f64.to_bits());
    if (half(&s0) && vgpr_pair(&s1) == Some(r)) || (half(&s1) && vgpr_pair(&s0) == Some(r)) {
        Some(dst)
    } else {
        None
    }
}

// Matches the rsq + Newton-Raphson f64 sqrt expansion seeded at the
// V_RSQ_F64 `anchor`:
//
//   r  = rsq(X)             [anchor, dst rD]
//   A  = X * rD
//   rD = 0.5 * rD
//   B  = fma(-rD, A, 0.5)
//   A  = fma(A, B, A)
//   rD = fma(rD, B, rD)
//   B  = fma(-A, A, X)
//   A  = fma(B, rD, A)
//   B  = fma(-A, A, X)
//   rD = fma(B, rD, A)      [final, = sqrt(X)]
//
// rD ends holding sqrt(X) so the whole chain collapses to V_SQRT_F64(rD, X):
// returns the final FMA index (to rewrite) and the other chain indices (to
// remove). X is never written by the chain, so it is still live to seed the
// sqrt; this is checked, as is that nothing outside the chain observes the
// temporaries.
struct SqrtMatch {
    final_idx: usize,
    removed: Vec<usize>,
    rd: u32,
    x: u32,
}

fn match_sqrt_f64(insts: &[InstFormat], effects: &[InstEffects], anchor: usize) -> Option<SqrtMatch> {
    let (rd, x) = as_rsq_f64(&insts[anchor])?;
    let n = insts.len();

    // step1: A = X * rD
    let i1 = find_forward(insts, anchor, n, |i| is_mul_xy(i, x, rd).is_some())?;
    let a = is_mul_xy(&insts[i1], x, rd)?;

    // step2: rD = 0.5 * rD
    let i2 = find_forward(insts, i1, n, |i| is_mul_half(i, rd) == Some(rd))?;

    // step3: B = fma(-rD, A, 0.5)
    let i3 = find_forward(insts, i2, n, |i| {
        as_fma_f64(i).map_or(false, |(_, s0, s1, s2, neg)| {
            neg == 1
                && vgpr_pair(&s0) == Some(rd)
                && vgpr_pair(&s1) == Some(a)
                && matches!(s2, SourceOperand::FloatConstant(v) if v.to_bits() == 0.5f64.to_bits())
        })
    })?;
    let b = as_fma_f64(&insts[i3])?.0;

    // step4: A = fma(A, B, A)
    let i4 = find_forward(insts, i3, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == a
                && neg == 0
                && vgpr_pair(&s0) == Some(a)
                && vgpr_pair(&s1) == Some(b)
                && vgpr_pair(&s2) == Some(a)
        })
    })?;

    // step5: rD = fma(rD, B, rD)
    let i5 = find_forward(insts, i4, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == rd
                && neg == 0
                && vgpr_pair(&s0) == Some(rd)
                && vgpr_pair(&s1) == Some(b)
                && vgpr_pair(&s2) == Some(rd)
        })
    })?;

    // step6: B = fma(-A, A, X)
    let i6 = find_forward(insts, i5, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == b
                && neg == 1
                && vgpr_pair(&s0) == Some(a)
                && vgpr_pair(&s1) == Some(a)
                && vgpr_pair(&s2) == Some(x)
        })
    })?;

    // step7: A = fma(B, rD, A)
    let i7 = find_forward(insts, i6, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == a
                && neg == 0
                && vgpr_pair(&s0) == Some(b)
                && vgpr_pair(&s1) == Some(rd)
                && vgpr_pair(&s2) == Some(a)
        })
    })?;

    // step8: B = fma(-A, A, X)
    let i8 = find_forward(insts, i7, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == b
                && neg == 1
                && vgpr_pair(&s0) == Some(a)
                && vgpr_pair(&s1) == Some(a)
                && vgpr_pair(&s2) == Some(x)
        })
    })?;

    // step9: rD = fma(B, rD, A)  -> final sqrt(X)
    let i9 = find_forward(insts, i8, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == rd
                && neg == 0
                && vgpr_pair(&s0) == Some(b)
                && vgpr_pair(&s1) == Some(rd)
                && vgpr_pair(&s2) == Some(a)
        })
    })?;

    let removed = vec![anchor, i1, i2, i3, i4, i5, i6, i7, i8];

    // X must remain unwritten through the final FMA so the rewritten sqrt
    // reads the same input the chain consumed.
    for j in (anchor + 1)..i9 {
        if removed.contains(&j) {
            continue;
        }
        let e = &effects[j];
        if !e.known {
            return None;
        }
        if e.kills.contains(&(VGPR_BASE + x)) || e.kills.contains(&(VGPR_BASE + x + 1)) {
            return None;
        }
    }

    // The chain's temporaries (A, B and the intermediate rD writes) must not be
    // observed outside the matched set before being overwritten. The final FMA
    // is rewritten to read only X, so reads by it no longer count.
    let matched: Vec<usize> = removed.iter().copied().chain(std::iter::once(i9)).collect();
    for &i in &removed {
        for &reg in &effects[i].kills {
            let mut killed = false;
            for j in (i + 1)..n {
                if matched.contains(&j) {
                    if j == i9 {
                        // After rewrite the final FMA neither reads nor writes
                        // the temporaries (only X -> rD), except rD which it
                        // still defines.
                        if reg == VGPR_BASE + rd || reg == VGPR_BASE + rd + 1 {
                            killed = true;
                            break;
                        }
                        continue;
                    }
                    if effects[j].kills.contains(&reg) {
                        killed = true;
                        break;
                    }
                    continue;
                }
                let ej = &effects[j];
                if !ej.known {
                    return None;
                }
                if ej.reads.contains(&reg) {
                    return None;
                }
                if ej.kills.contains(&reg) {
                    killed = true;
                    break;
                }
            }
            // Stricter than the div matcher: a temporary surviving to the block
            // end may be read by a successor block (e.g. the chain's sqrt/rsqrt
            // estimates), so only remove a definition that is provably
            // overwritten within this block.
            if !killed {
                return None;
            }
        }
    }

    Some(SqrtMatch {
        final_idx: i9,
        removed,
        rd,
        x,
    })
}

// Removes instructions whose results are provably dead within the block.
// Returns the number of removed instructions.
pub(crate) fn combine_block(insts: &mut Vec<InstFormat>) -> usize {
    let mut removed_total = 0;

    // Phase 0: collapse rsq + Newton-Raphson f64 sqrt expansions to a single
    // V_SQRT_F64. Rewrites the final FMA in place and deletes the rest.
    {
        let effects: Vec<InstEffects> = insts.iter().map(effects_of).collect();
        let mut remove = vec![false; insts.len()];
        let mut rewrites: Vec<(usize, InstFormat)> = Vec::new();

        for anchor in 0..insts.len() {
            if let Some(m) = match_sqrt_f64(insts, &effects, anchor) {
                if m.removed.iter().all(|&i| !remove[i]) && !remove[m.final_idx] {
                    for &i in &m.removed {
                        remove[i] = true;
                    }
                    rewrites.push((
                        m.final_idx,
                        InstFormat::VOP1(VOP1 {
                            src0: SourceOperand::VectorRegister(m.x as u8),
                            op: I::V_SQRT_F64,
                            vdst: m.rd as u8,
                        }),
                    ));
                }
            }
        }

        for (idx, inst) in rewrites {
            insts[idx] = inst;
        }
        removed_total += remove.iter().filter(|&&r| r).count();
        let mut keep = remove.iter().map(|&r| !r);
        insts.retain(|_| keep.next().unwrap());
    }

    // Phase 1: pattern-matched rewrites. V_DIV_FIXUP_F64 already computes the
    // quotient by itself, so a matched expansion chain is deleted wholesale.
    {
        let effects: Vec<InstEffects> = insts.iter().map(effects_of).collect();
        let mut remove = vec![false; insts.len()];

        for anchor in 0..insts.len() {
            if let Some(matched) = match_div_f64(insts, &effects, anchor) {
                if matched.iter().all(|&i| !remove[i]) {
                    for &i in &matched {
                        remove[i] = true;
                    }
                }
            }
        }

        removed_total += remove.iter().filter(|&&r| r).count();
        let mut keep = remove.iter().map(|&r| !r);
        insts.retain(|_| keep.next().unwrap());
    }

    // Phase 2: generic in-block dead code elimination.
    loop {
        let effects: Vec<InstEffects> = insts.iter().map(effects_of).collect();
        let mut remove = vec![false; insts.len()];
        let n = insts.len();

        // The last instruction is the block terminator (or falls through to
        // the next block); never remove it.
        for i in 0..n.saturating_sub(1) {
            let e = &effects[i];
            if !e.removable || e.kills.is_empty() {
                continue;
            }

            let dead = e.kills.iter().all(|&r| {
                for j in (i + 1)..n {
                    let ej = &effects[j];
                    if !ej.known {
                        return false;
                    }
                    if ej.reads.contains(&r) {
                        return false;
                    }
                    if ej.kills.contains(&r) {
                        return true;
                    }
                }
                // Reaches the end of the block: conservatively live-out.
                false
            });

            if dead {
                remove[i] = true;
            }
        }

        let removed = remove.iter().filter(|&&r| r).count();
        if removed == 0 {
            break;
        }
        removed_total += removed;

        let mut keep = remove.iter().map(|&r| !r);
        insts.retain(|_| keep.next().unwrap());
    }

    removed_total
}
