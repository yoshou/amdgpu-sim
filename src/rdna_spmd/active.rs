//! De-SIMT lane-active analysis.
//!
//! The kernel was compiled for a 32-lane wavefront; we execute a *single* lane
//! and recover scalar control flow by projecting branch conditions onto lane-0's
//! bit (see `emit::taken_cond`). With that projection `EXEC` is a 1-bit "is this
//! lane active" flag and the `s_cbranch_execz/execnz` edges carry a *definite*
//! value: the fall-through of `execz` means `EXEC[0] == 1`.
//!
//! This is a forward must-analysis that proves, per program point, whether
//! `EXEC[0]` is guaranteed `1` ("active"). Where the lane is provably active the
//! masked SIMT idioms are inert: vector writes need no `select(EXEC[0],new,old)`
//! predication and compares need no `& EXEC`. Eliminating that mask bookkeeping
//! turns the wavefront code into ordinary single-work-item scalar code, leaving
//! only genuine data-conditional updates.
//!
//! To see through the standard if/loop reconvergence idiom
//! ```text
//!   s_and_saveexec sN, M     ; sN = EXEC ; EXEC = M & EXEC   (enter divergent region)
//!   ... predicated body ...
//!   s_or exec, exec, sN      ; EXEC |= sN                    (reconverge)
//! ```
//! the state also tracks which SGPRs hold a *saved-active* mask (bit0 == 1). When
//! such an SGPR is OR'd back into EXEC, the lane is provably active again.
//!
//! Soundness: the lattice is optimistic (`active` = top, `masks` = all-ones). A
//! point is `active` only if every path proves `EXEC[0] == 1`; the transfer never
//! claims active where it could be 0 (a `V_CMPX`/`saveexec` lowers it, and every
//! scalar write conservatively clears the saved-mask bit). A `false` is always
//! safe (predicate anyway); a `true` is always correct.

use std::collections::BTreeMap;

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};

use super::ir::{Cond, ScalarBlock, ScalarProgram, Terminator};

const EXEC: u32 = 126;

/// Abstract lane state: `active` = `EXEC[0]` provably 1; `masks` bit i = SGPR i
/// provably holds bit0 == 1 (a saved-active wave mask).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct State {
    pub active: bool,
    pub masks: u128,
}

impl State {
    fn meet(self, o: State) -> State {
        State { active: self.active && o.active, masks: self.masks & o.masks }
    }
}

/// SGPRs written by an instruction (over-approximated — used to *clear* tracked
/// saved-active bits, so over-clearing is always sound).
fn scalar_dests(inst: &InstFormat) -> u128 {
    let bit = |r: u32| 1u128 << (r & 127);
    let pair = |r: u32| bit(r) | bit(r + 1);
    match inst {
        InstFormat::SOP1(i) => match i.op {
            I::S_MOV_B64 => pair(i.sdst as u32),
            _ => bit(i.sdst as u32),
        },
        InstFormat::SOP2(i) => match i.op {
            I::S_ADD_NC_U64 | I::S_MUL_U64 | I::S_LSHL_B64 | I::S_AND_B64 | I::S_OR_B64 => {
                pair(i.sdst as u32)
            }
            _ => bit(i.sdst as u32),
        },
        InstFormat::SOPK(i) => bit(i.sdst as u32),
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
            (0..words).fold(0u128, |m, k| m | bit(i.sdata as u32 + k))
        }
        InstFormat::VOP3SD(i) => bit(i.sdst as u32),
        InstFormat::VOP1(i) if matches!(i.op,I::V_READFIRSTLANE_B32) => bit(i.vdst as u32),
        InstFormat::VOP3(i) if matches!(i.op,I::V_READLANE_B32) => bit(i.vdst as u32),
        // VOPC / VOP3 compares write a lane mask (VCC or, for V_CMPX, EXEC).
        InstFormat::VOPC(i) => {
            if format!("{:?}", i.op).starts_with("V_CMPX") { bit(EXEC) } else { bit(106) }
        }
        InstFormat::VOP3(i) if format!("{:?}", i.op).contains("V_CMP") => bit(i.vdst as u32),
        _ => 0,
    }
}

/// Includes implicit EXEC writes (saveexec and cmpx), not just scalar destinations.
pub(super) fn writes_exec(inst: &InstFormat) -> bool {
    scalar_dests(inst) & (1u128 << EXEC) != 0 || matches!(inst,
        InstFormat::SOP1(i) if matches!(i.op,
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 |
            I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32))
}

/// Only mask widening can expose a value from a previously inactive lane.
/// Unknown writes remain widening; recognize only subset-of-old-EXEC forms.
pub(super) fn may_enable_lanes(inst: &InstFormat) -> bool {
    if !writes_exec(inst) { return false; }
    let exec = |s: &SourceOperand| matches!(s, SourceOperand::ScalarRegister(126));
    match inst {
        InstFormat::VOPC(_) => false, // st_cmp: comparison & old EXEC
        InstFormat::VOP3(i) if format!("{:?}", i.op).starts_with("V_CMP") => false,
        InstFormat::SOP1(i) if matches!(i.op, I::S_AND_SAVEEXEC_B32) => false,
        InstFormat::SOP1(i) if i.sdst == 126 && matches!(i.op, I::S_MOV_B32 | I::S_MOV_B64) => {
            !(exec(&i.ssrc0) || matches!(i.ssrc0,
                SourceOperand::IntegerConstant(0) | SourceOperand::LiteralConstant(0)))
        }
        InstFormat::SOP2(i) if i.sdst == 126 && matches!(i.op, I::S_AND_B32 | I::S_AND_B64) => {
            !(exec(&i.ssrc0) || exec(&i.ssrc1))
        }
        InstFormat::SOP2(i) if matches!(i.op, I::S_AND_NOT1_B32) => !exec(&i.ssrc0),
        _ => true,
    }
}

/// Forward transfer of a single instruction.
fn transfer(inst: &InstFormat, st: State) -> State {
    let was_active = st.active;
    // Every scalar write invalidates the saved-active status of its destination.
    let mut masks = st.masks & !scalar_dests(inst);
    let mut active = st.active && !writes_exec(inst);

    match inst {
        // Enter a divergent region: sN = old EXEC, EXEC = (M [& ~]) EXEC.
        InstFormat::SOP1(i)
            if matches!(
                i.op,
                I::S_AND_SAVEEXEC_B32
                    | I::S_AND_NOT1_SAVEEXEC_B32
                    | I::S_OR_SAVEEXEC_B32
                    | I::S_XOR_SAVEEXEC_B32
            ) =>
        {
            // sN holds the saved EXEC, whose bit0 is the pre-op active state.
            if was_active {
                masks |= 1u128 << (i.sdst as u32 & 127);
            }
            active = false;
        }
        // Direct EXEC writes.
        InstFormat::SOP1(i) if i.sdst as u32 == EXEC => {
            active = matches!(i.op, I::S_MOV_B32 | I::S_MOV_B64)
                && match i.ssrc0 {
                    SourceOperand::IntegerConstant(v) => v & 1 != 0,
                    SourceOperand::LiteralConstant(v) => v & 1 != 0,
                    _ => false,
                };
        }
        InstFormat::SOP2(i) if i.sdst as u32 == EXEC => {
            // `s_or exec, exec, sN`: reconverge — active if EXEC was active OR sN
            // is a saved-active mask. Any other EXEC write is not provable.
            if matches!(i.op, I::S_OR_B32) {
                let sn = match i.ssrc1 {
                    SourceOperand::ScalarRegister(r) => Some(r as u32),
                    _ => match i.ssrc0 {
                        SourceOperand::ScalarRegister(r) => Some(r as u32),
                        _ => None,
                    },
                };
                let sn_active = sn.map_or(false, |r| (masks >> (r & 127)) & 1 == 1);
                let reads_exec = matches!(i.ssrc0, SourceOperand::ScalarRegister(126))
                    || matches!(i.ssrc1, SourceOperand::ScalarRegister(126));
                active = (was_active && reads_exec) || sn_active;
            } else {
                active = false;
            }
        }
        // V_CMPX narrows EXEC.
        InstFormat::VOPC(i) if format!("{:?}", i.op).starts_with("V_CMPX") => {
            active = false;
        }
        InstFormat::VOP3(i)
            if format!("{:?}", i.op).starts_with("V_CMPX") && i.vdst as u32 == EXEC =>
        {
            active = false;
        }
        _ => {}
    }

    State { active, masks }
}

/// Per body instruction, whether the lane is provably active *before* it runs.
pub fn body_active_states(block: &ScalarBlock, entry: State) -> Vec<bool> {
    let mut out = Vec::with_capacity(block.body.len());
    let mut cur = entry;
    for inst in &block.body {
        out.push(cur.active);
        cur = transfer(inst, cur);
    }
    out
}

fn block_exit(block: &ScalarBlock, entry: State) -> State {
    let mut cur = entry;
    for inst in &block.body {
        cur = transfer(inst, cur);
    }
    cur
}

/// Forward must-analysis: the entry abstract state of every block.
pub fn analyze_states(prog: &ScalarProgram) -> BTreeMap<usize, State> {
    analyze_states_ex(prog, false)
}

/// As [`analyze_states`], but `sound_for_packing` controls the loop/reconverge
/// relaxations. The scalar backend executes a *single* lane, for which a
/// compiler-generated loop is EXEC-balanced and back-edges / `execz` arms carry
/// no new deactivation — so skipping them (the `false` mode) lets active-ness
/// propagate into loop bodies. For width-W packing
/// (`true`) that is **unsound**: lanes exit a loop at different iterations, so a
/// loop body proven "active" via the skipped back-edge actually runs with some
/// packed lanes masked off. The `true` mode includes back-edges and the
/// EXEC-off arms in the meet, so only genuinely uniform regions (e.g. a counted
/// loop whose EXEC is unchanged) can omit redundant predication, while
/// divergent loops remain predicated.
pub fn analyze_states_ex(prog: &ScalarProgram, sound_for_packing: bool) -> BTreeMap<usize, State> {
    let top = State { active: true, masks: u128::MAX };
    let mut entry: BTreeMap<usize, State> = prog.blocks.keys().map(|&pc| (pc, top)).collect();
    // EXEC[0]==1 on the synthetic function-entry edge; no saved masks yet.
    let ext = State { active: true, masks: 0 };

    loop {
        let mut incoming: BTreeMap<usize, Option<State>> = BTreeMap::new();
        incoming.insert(prog.entry_pc, Some(ext));

        for (&_pc, block) in &prog.blocks {
            let exit = block_exit(block, entry[&_pc]);
            // EXEC-conditional edges pin EXEC[0] but preserve the saved masks.
            let exec_edge = |a: bool| State { active: a, masks: exit.masks };
            // `inactive` edges are the `execz`/`execnz` arms taken when the lane
            // is masked off; they lead to a reconvergence point (or the next
            // sample) where EXEC is restored. They never carry *new* activation,
            // so — like back-edges — they must not lower a merge's active state.
            let edges: Vec<(usize, State, bool)> = match &block.term {
                Terminator::Return => vec![],
                Terminator::Jump(t) => vec![(*t, exit, false)],
                Terminator::Branch { cond, taken, fallthrough } => match cond {
                    Cond::ExecZ => {
                        vec![(*taken, exec_edge(false), true), (*fallthrough, exec_edge(true), false)]
                    }
                    Cond::ExecNz => {
                        vec![(*taken, exec_edge(true), false), (*fallthrough, exec_edge(false), true)]
                    }
                    _ => vec![(*taken, exit, false), (*fallthrough, exit, false)],
                },
                Terminator::Barrier { resume } | Terminator::Yield { resume, .. } => vec![(*resume, exit, false)],
            };
            for (t, st, inactive) in edges {
                if inactive && !sound_for_packing {
                    continue;
                }
                // Compiler-generated loops are EXEC-balanced: the body restores
                // EXEC to its header-entry value each iteration, so a back-edge
                // (target at/below the source) carries no new deactivation — the
                // header's active state is determined by its pre-header. Skipping
                // back-edges in the meet lets active-ness propagate into loop
                // bodies for a single work-item when the loop balances EXEC.
                // For width-W packing this must include back-edges (see
                // `analyze_states_ex` doc).
                if t <= _pc && !sound_for_packing {
                    continue;
                }
                let e = incoming.entry(t).or_insert(None);
                *e = Some(match *e {
                    None => st,
                    Some(p) => p.meet(st),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_instructions::{SOP1, SOP2};
    #[test]
    fn distinguishes_exec_subsets_from_reactivation() {
        for (op, widening) in [(I::S_AND_B32, false), (I::S_AND_NOT1_B32, false), (I::S_OR_B32, true), (I::S_XOR_B32, true)] {
            let inst = InstFormat::SOP2(SOP2 { op, sdst: 126,
                ssrc0: SourceOperand::ScalarRegister(126), ssrc1: SourceOperand::ScalarRegister(4) });
            assert!(writes_exec(&inst));
            assert_eq!(may_enable_lanes(&inst), widening);
        }
        // This ISA form is src & !old_exec in the existing emitter and can
        // enable lanes; do not confuse it with old_exec & !src.
        let inst = InstFormat::SOP1(SOP1 { op: I::S_AND_NOT1_SAVEEXEC_B32, sdst: 4,
            ssrc0: SourceOperand::ScalarRegister(6) });
        assert!(may_enable_lanes(&inst));
    }
}
