//! f64-register-typing analysis: which VGPR *pairs* hold a live `double` in the
//! per-pair f64 shadow at each program point.
//!
//! Background: a VGPR is a 32-bit slot, so an f64 occupies a pair `(r, r+1)`.
//! Storing the register file as i32 allocas means an f64 that crosses a basic
//! block is rebuilt from two i32 phis (`zext`/`shl`/`or`/`bitcast`) — measured at
//! ~26% of all executed instructions, which native code (xmm `double`) and the
//! masked backend (amortized over 16 SIMD lanes) don't pay per-lane. LLVM cannot
//! coalesce the paired i32 phis back into one f64 phi (verified: `default<O3>` +
//! `aggressive-instcombine` leave the reconstruction in place).
//!
//! Fix: keep a parallel `double` alloca per pair. emit writes it on f64 stores
//! and reads it instead of reconstructing — *but only where the shadow is
//! provably current* ("fresh"). This forward must-analysis computes the
//! fresh set on entry to each block. A pair is fresh after an f64 write to it and
//! stays fresh until any 32-bit/overlapping write touches reg r or r+1.
//!
//! Soundness: only f64 *writes* set fresh; *every* VGPR write clears the fresh
//! bit of the pairs it overlaps (over-clearing is always safe — it just forces a
//! reconstruction). The meet is intersection (fresh only if fresh on all paths).

use std::collections::BTreeMap;

use crate::instructions::I;
use crate::rdna_instructions::InstFormat;

use super::ir::{ScalarBlock, ScalarProgram, Terminator};
use super::regtype::RegSet;

// VGPRs number up to 256, so pair-freshness sets are [u128; 2] (RegSet) — a
// plain u128 with `& 127` indexing aliases pair p with pair p+128 (e.g. an
// f64 write to v153:v154 would falsely mark pair v25:v26 fresh).
#[inline]
fn pset(s: &mut RegSet, r: u32) {
    let r = (r & 255) as usize;
    s[r / 128] |= 1u128 << (r % 128);
}
#[inline]
fn pclr(s: &mut RegSet, r: u32) {
    let r = (r & 255) as usize;
    s[r / 128] &= !(1u128 << (r % 128));
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
    matches!(op, I::V_MAD_CO_U64_U32 | I::V_LSHLREV_B64 | I::V_ASHR_I64 | I::V_ASHRREV_I64)
}

/// The f64 pairs this instruction defines (sets fresh). Global loads of ≥2 words
/// are loaded directly as `double`s into the shadow (see emit), so the even pairs
/// they cover are defined too.
pub fn f64_defs(inst: &InstFormat) -> Vec<u32> {
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
pub fn vgpr_writes(inst: &InstFormat) -> Vec<u32> {
    match inst {
        InstFormat::VOP1(i) => {
            let w = is_f64_producer(i.op) || matches!(i.op, I::V_CVT_F64_I32 | I::V_CVT_F64_U32);
            if w { vec![i.vdst as u32, i.vdst as u32 + 1] } else { vec![i.vdst as u32] }
        }
        InstFormat::VOP2(i) => {
            if is_f64_producer(i.op) { vec![i.vdst as u32, i.vdst as u32 + 1] } else { vec![i.vdst as u32] }
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
        InstFormat::VGLOBAL(i) => {
            let words = match i.op {
                I::GLOBAL_LOAD_B32 => 1,
                I::GLOBAL_LOAD_B64 => 2,
                I::GLOBAL_LOAD_B96 => 3,
                I::GLOBAL_LOAD_B128 => 4,
                _ => 0, // stores write no VGPR
            };
            (0..words).map(|k| i.vdst as u32 + k).collect()
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

/// Apply a single instruction's effect to the fresh bitmask.
fn transfer_inst(inst: &InstFormat, mut fresh: RegSet) -> RegSet {
    // Every VGPR write clears the fresh bit of every pair overlapping that reg:
    // pair `d` (regs d,d+1) and pair `d-1` (regs d-1,d).
    for d in vgpr_writes(inst) {
        pclr(&mut fresh, d);
        if d > 0 {
            pclr(&mut fresh, d - 1);
        }
    }
    // f64 writes make their pairs fresh.
    for r in f64_defs(inst) {
        pset(&mut fresh, r);
    }
    fresh
}

fn block_exit(block: &ScalarBlock, entry: RegSet) -> RegSet {
    let mut f = entry;
    for inst in &block.body {
        f = transfer_inst(inst, f);
    }
    f
}

// ===================================================================
//  SGPR i64 typing — same idea for scalar register pairs. A loop-carried
//  64-bit value (e.g. a global-load base pointer advanced by `S_ADD_NC_U64`
//  each iteration) is otherwise rebuilt from two i32 phis (`shl 32`+`or`)
//  and re-split every iteration; keeping it in an i64 shadow lets it flow as
//  one i64 phi (a single `add` per iteration).
// ===================================================================

const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;
const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;

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
fn sgpr_u64_defs(inst: &InstFormat) -> Vec<u32> {
    match inst {
        InstFormat::SOP1(i) if is_sop_u64(i.op) => vec![i.sdst as u32],
        InstFormat::SOP2(i) if is_sop_u64(i.op) => vec![i.sdst as u32],
        _ => vec![],
    }
}

/// All SGPRs an instruction writes (32-bit granularity), used to clear the i64
/// shadow. EXEC/VCC are i1-typed (handled separately), never i64-shadowed.
fn sgpr_writes(inst: &InstFormat) -> Vec<u32> {
    let keep = |r: u32| r != EXEC_LO && r != EXEC_HI && r != VCC_LO && r != VCC_HI;
    let wide = |r: u32| vec![r, r + 1].into_iter().filter(|&x| keep(x)).collect::<Vec<_>>();
    let one = |r: u32| if keep(r) { vec![r] } else { vec![] };
    match inst {
        InstFormat::SOP1(i) => if is_sop_u64(i.op) { wide(i.sdst as u32) } else { one(i.sdst as u32) },
        InstFormat::SOP2(i) => if is_sop_u64(i.op) { wide(i.sdst as u32) } else { one(i.sdst as u32) },
        InstFormat::SOPK(i) => one(i.sdst as u32),
        InstFormat::VOP3SD(i) => one(i.sdst as u32),
        InstFormat::SMEM(i) => {
            let words = match i.op {
                I::S_LOAD_B32 => 1,
                I::S_LOAD_B64 => 2,
                I::S_LOAD_B96 => 3,
                I::S_LOAD_B128 => 4,
                I::S_LOAD_B256 => 8,
                I::S_LOAD_B512 => 16,
                _ => 1,
            };
            (0..words).flat_map(|k| one(i.sdata as u32 + k)).collect()
        }
        // VOPC / VOP3 compares write a lane mask into an SGPR (or VCC/EXEC).
        InstFormat::VOPC(_) => vec![], // writes VCC/EXEC (i1)
        InstFormat::VOP3(i) if format!("{:?}", i.op).contains("V_CMP") => one(i.vdst as u32),
        _ => vec![],
    }
}

fn sgpr_transfer(inst: &InstFormat, mut fresh: u128) -> u128 {
    for d in sgpr_writes(inst) {
        fresh &= !(1u128 << (d & 127));
        if d > 0 {
            fresh &= !(1u128 << ((d - 1) & 127));
        }
    }
    for r in sgpr_u64_defs(inst) {
        fresh |= 1u128 << (r & 127);
    }
    fresh
}

fn sgpr_block_exit(block: &ScalarBlock, entry: u128) -> u128 {
    let mut f = entry;
    for inst in &block.body {
        f = sgpr_transfer(inst, f);
    }
    f
}

/// Forward must-analysis of SGPR i64-shadow freshness, per block entry.
pub fn analyze_sgpr(prog: &ScalarProgram) -> BTreeMap<usize, u128> {
    let mut entry: BTreeMap<usize, u128> = prog.blocks.keys().map(|&pc| (pc, u128::MAX)).collect();
    loop {
        let mut incoming: BTreeMap<usize, Option<u128>> = BTreeMap::new();
        incoming.insert(prog.entry_pc, Some(0));
        for (&_pc, block) in &prog.blocks {
            let exit = sgpr_block_exit(block, entry[&_pc]);
            // A Barrier (coroutine yield) discards the SGPR i64 shadow allocas, so
            // the resume block is entered with nothing fresh (see `analyze`).
            let (succs, out): (Vec<usize>, u128) = match &block.term {
                Terminator::Return => (vec![], exit),
                Terminator::Jump(t) => (vec![*t], exit),
                Terminator::Branch { taken, fallthrough, .. } => (vec![*taken, *fallthrough], exit),
                Terminator::Barrier { resume } => (vec![*resume], 0),
            };
            for t in succs {
                let e = incoming.entry(t).or_insert(None);
                *e = Some(match *e { None => out, Some(p) => p & out });
            }
        }
        let mut changed = false;
        for (&pc, _) in &prog.blocks {
            if let Some(Some(v)) = incoming.get(&pc) {
                if entry[&pc] != *v {
                    entry.insert(pc, *v);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    entry
}

/// Forward must-analysis: the fresh bitmask on entry to each block.
pub fn analyze(prog: &ScalarProgram) -> BTreeMap<usize, RegSet> {
    // Optimistic init (all fresh) lowered by intersection; entry has nothing
    // fresh yet (the shadow allocas are uninitialized at function start).
    let mut entry: BTreeMap<usize, RegSet> =
        prog.blocks.keys().map(|&pc| (pc, [u128::MAX; 2])).collect();

    loop {
        let mut incoming: BTreeMap<usize, Option<RegSet>> = BTreeMap::new();
        incoming.insert(prog.entry_pc, Some([0; 2]));

        for (&_pc, block) in &prog.blocks {
            let exit = block_exit(block, entry[&_pc]);
            // A Barrier is a coroutine yield: the f64 shadow allocas are local to
            // one `run()` call and do not survive the yield, so the resume block is
            // (re-)entered with nothing fresh. Propagate an empty set, not `exit`.
            let (succs, out): (Vec<usize>, RegSet) = match &block.term {
                Terminator::Return => (vec![], exit),
                Terminator::Jump(t) => (vec![*t], exit),
                Terminator::Branch { taken, fallthrough, .. } => (vec![*taken, *fallthrough], exit),
                Terminator::Barrier { resume } => (vec![*resume], [0; 2]),
            };
            for t in succs {
                let e = incoming.entry(t).or_insert(None);
                *e = Some(match *e {
                    None => out,
                    Some(p) => [p[0] & out[0], p[1] & out[1]], // meet = intersection
                });
            }
        }

        let mut changed = false;
        for (&pc, _) in &prog.blocks {
            if let Some(Some(v)) = incoming.get(&pc) {
                if entry[&pc] != *v {
                    entry.insert(pc, *v);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    entry
}
