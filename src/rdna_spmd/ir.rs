//! Scalar IR: the LLVM-independent representation of one work-item's control
//! flow and instructions.
//!
//! For a single work-item, a wavefront program is just a control-flow graph
//! whose branches test a 1-bit EXEC (lane active?) / VCC / SCC. This module
//! normalizes the decoded [`RDNAProgram`] into:
//!  - a per-block linear stream of [`InstFormat`] with pure scheduling no-ops
//!    (`s_delay_alu`, `s_wait*`, `s_clause`, `s_nop`, ...) removed, and
//!  - an explicit [`Terminator`] per block.
//!
//! The opcode *semantics* are produced later by `emit.rs`; this layer only
//! decides control flow and which instructions survive.

use std::collections::BTreeMap;

use crate::instructions::I;
use crate::rdna_instructions::InstFormat;
use crate::rdna_translator::{collapse_div_expansions, RDNAProgram};

/// Branch condition recovered from a block's terminating SOPP instruction.
/// For a single lane EXEC/VCC are 1-bit; these become ordinary scalar branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cond {
    ExecZ,
    ExecNz,
    VccZ,
    VccNz,
    Scc0,
    Scc1,
}

/// How a scalar block transfers control.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// `s_endpgm`: return from the work-item function.
    Return,
    /// Unconditional fall-through / `s_branch` to a single successor.
    Jump(usize),
    /// Conditional branch: if `cond` holds go to `taken`, else `fallthrough`.
    Branch {
        cond: Cond,
        taken: usize,
        fallthrough: usize,
    },
    /// Workgroup barrier (`s_barrier_signal`/`s_barrier_wait`): the cooperative
    /// backend yields here so the scheduler can run every other work-item to the
    /// same barrier before any proceeds. `resume` is the pc of the block holding
    /// the post-barrier continuation (a resume entry). Only produced by
    /// [`split_at_barriers`]; the non-cooperative backends never see it.
    Barrier { resume: usize },
}

/// A basic block lowered for scalar execution.
#[derive(Debug, Clone)]
pub struct ScalarBlock {
    pub pc: usize,
    /// Body instructions (terminator removed), scheduling no-ops filtered out.
    pub body: Vec<InstFormat>,
    pub term: Terminator,
}

/// A whole work-item program in Scalar IR form.
#[derive(Debug, Clone)]
pub struct ScalarProgram {
    pub entry_pc: usize,
    pub blocks: BTreeMap<usize, ScalarBlock>,
}

/// Instructions with no architectural effect in emulation: scheduling hints,
/// wait counters, and clause/nop markers. Dropped during lowering.
pub fn is_noop(inst: &InstFormat) -> bool {
    match inst {
        InstFormat::SOPP(i) => matches!(
            i.op,
            I::S_DELAY_ALU
                | I::S_WAIT_ALU
                | I::S_WAIT_LOADCNT
                | I::S_WAIT_KMCNT
                | I::S_WAIT_DSCNT
                | I::S_WAIT_STORECNT
                | I::S_WAIT_STORECNT_DSCNT
                | I::S_WAIT_LOADCNT_DSCNT
                | I::S_WAIT_SAMPLECNT
                | I::S_WAIT_BVHCNT
                | I::S_WAIT_EXPCNT
                | I::S_WAIT_EVENT
                | I::S_WAIT_IDLE
                | I::S_WAITCNT
                | I::S_NOP
                | I::S_CLAUSE
                | I::S_SENDMSG
        ),
        _ => false,
    }
}

/// Build the [`Terminator`] for a block from its last instruction and the CFG
/// successor list (`next_pcs`: `[fallthrough, taken]` for conditional branches).
fn lower_terminator(last: &InstFormat, next_pcs: &[usize]) -> Terminator {
    if let InstFormat::SOPP(i) = last {
        match i.op {
            I::S_ENDPGM => return Terminator::Return,
            I::S_BRANCH => return Terminator::Jump(next_pcs[0]),
            I::S_CBRANCH_EXECZ => {
                return Terminator::Branch {
                    cond: Cond::ExecZ,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_EXECNZ => {
                return Terminator::Branch {
                    cond: Cond::ExecNz,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_VCCZ => {
                return Terminator::Branch {
                    cond: Cond::VccZ,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_VCCNZ => {
                return Terminator::Branch {
                    cond: Cond::VccNz,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_SCC0 => {
                return Terminator::Branch {
                    cond: Cond::Scc0,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_SCC1 => {
                return Terminator::Branch {
                    cond: Cond::Scc1,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            _ => {}
        }
    }
    // Non-terminator last instruction: straight-line fall-through.
    Terminator::Jump(next_pcs[0])
}

/// Lower a decoded [`RDNAProgram`] into Scalar IR.
pub fn build_scalar_program(program: &RDNAProgram) -> ScalarProgram {
    let mut blocks = BTreeMap::new();

    for (&pc, block) in program.blocks() {
        let insts = block.insts();
        let (last, head) = insts.split_last().expect("empty block");

        let term = lower_terminator(last, block.next_pcs());

        // Whether the last instruction is itself a control-flow terminator: if
        // not, it is a normal instruction that must stay in the body.
        let last_is_term = matches!(
            last,
            InstFormat::SOPP(i) if matches!(
                i.op,
                I::S_ENDPGM
                    | I::S_BRANCH
                    | I::S_CBRANCH_EXECZ
                    | I::S_CBRANCH_EXECNZ
                    | I::S_CBRANCH_VCCZ
                    | I::S_CBRANCH_VCCNZ
                    | I::S_CBRANCH_SCC0
                    | I::S_CBRANCH_SCC1
            )
        );

        let body_src: &[InstFormat] = if last_is_term { head } else { insts };
        let mut body: Vec<InstFormat> = body_src.iter().filter(|i| !is_noop(i)).cloned().collect();
        // This backend computes a V_DIV_FIXUP_F64 quotient from the original
        // operands, which leaves the expansion feeding it dead. The other two
        // engines apply the real fixup and keep it.
        collapse_div_expansions(&mut body);

        blocks.insert(pc, ScalarBlock { pc, body, term });
    }

    let mut prog = ScalarProgram {
        entry_pc: program.entry_pc(),
        blocks,
    };
    // Math-idiom combine: collapse rsq+Newton f64 sqrt expansions to V_SQRT_F64.
    super::mathcombine::fold_sqrt(&mut prog);
    prog
}

/// A workgroup barrier instruction (`s_barrier_signal` / `s_barrier_wait`). The
/// split-barrier pair (signal then wait) is one synchronization point.
pub fn is_barrier(inst: &InstFormat) -> bool {
    match inst {
        InstFormat::SOP1(i) => matches!(i.op, I::S_BARRIER_SIGNAL | I::S_BARRIER_SIGNAL_ISFIRST),
        InstFormat::SOPP(i) => matches!(i.op, I::S_BARRIER_WAIT | I::S_BARRIER),
        _ => false,
    }
}

/// Split every block containing a barrier so each barrier becomes a block
/// boundary carrying a [`Terminator::Barrier`]. The instructions *after* a
/// barrier form a fresh block whose pc is a resume entry for the cooperative
/// scheduler; the barrier instructions themselves are dropped (the yield/sync is
/// modeled by the terminator). Blocks without a barrier pass through unchanged,
/// so their pcs — and every existing branch target — are preserved.
pub fn split_at_barriers(program: &ScalarProgram) -> ScalarProgram {
    let mut next_pc = program.blocks.keys().copied().max().unwrap_or(0) + 1;
    let mut blocks: BTreeMap<usize, ScalarBlock> = BTreeMap::new();

    for block in program.blocks.values() {
        if !block.body.iter().any(|i| is_barrier(i)) {
            blocks.insert(block.pc, block.clone());
            continue;
        }
        // Cut the body into segments separated by maximal runs of barrier insts
        // (a signal+wait pair is one run → one boundary).
        let mut segments: Vec<Vec<InstFormat>> = Vec::new();
        let mut cur: Vec<InstFormat> = Vec::new();
        let mut in_barrier = false;
        for inst in &block.body {
            if is_barrier(inst) {
                if !in_barrier {
                    segments.push(std::mem::take(&mut cur));
                    in_barrier = true;
                }
                // drop the barrier instruction itself
            } else {
                in_barrier = false;
                cur.push(inst.clone());
            }
        }
        segments.push(cur); // final (post-last-barrier) segment, possibly empty

        let n = segments.len();
        let mut pcs = Vec::with_capacity(n);
        pcs.push(block.pc);
        for _ in 1..n {
            pcs.push(next_pc);
            next_pc += 1;
        }
        for (i, seg) in segments.into_iter().enumerate() {
            let term = if i + 1 < n {
                Terminator::Barrier { resume: pcs[i + 1] }
            } else {
                block.term.clone()
            };
            blocks.insert(pcs[i], ScalarBlock { pc: pcs[i], body: seg, term });
        }
    }

    ScalarProgram { entry_pc: program.entry_pc, blocks }
}
