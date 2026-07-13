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
        // VOPC / VOP3 compares write a lane mask (VCC or, for V_CMPX, EXEC).
        InstFormat::VOPC(i) => {
            if format!("{:?}", i.op).starts_with("V_CMPX") { bit(EXEC) } else { bit(106) }
        }
        InstFormat::VOP3(i) if format!("{:?}", i.op).contains("V_CMP") => bit(i.vdst as u32),
        _ => 0,
    }
}

/// Forward transfer of a single instruction.
fn transfer(inst: &InstFormat, st: State) -> State {
    let was_active = st.active;
    // Every scalar write invalidates the saved-active status of its destination.
    let mut masks = st.masks & !scalar_dests(inst);
    let mut active = st.active;

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
            active = matches!(i.ssrc0, SourceOperand::IntegerConstant(v) if (v & 1) == 1);
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
                active = active || sn_active;
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
                Terminator::Barrier { resume } => vec![(*resume, exit, false)],
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
