//! Decoded register CFG accepted at the SPMD input boundary.
//! Preparation and compilation retain the typed SSA in `Program`.
//!
//! For a single work-item, a wavefront program is just a control-flow graph
//! whose branches test a 1-bit EXEC (lane active?) / VCC / SCC. This module
//! represents decoded instructions as:
//!  - a per-block linear stream of [`InstFormat`] with pure scheduling no-ops
//!    (`s_delay_alu`, `s_wait*`, `s_clause`, `s_nop`, ...) removed, and
//!  - an explicit [`Terminator`] per block.
//!
//! `lift` produces typed SSA semantics for analysis and code generation; this layer only
//! normalizes control flow and removes scheduling no-ops. Optimization order
//! belongs to [`super::compiler::Compiler`].

use std::collections::BTreeMap;

use crate::instructions::I;
use crate::rdna_instructions::InstFormat;

pub(super) mod typed;

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
    /// A typed wave or workgroup effect, with its continuation.
    Yield { resume: usize, action: Box<super::lift::wave::YieldAction> },
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

/// Normalize one instruction block without applying optimization passes.
/// `insts` still contains its final instruction so fallthrough blocks retain it.
pub(super) fn lower_block(pc: usize, insts: &[InstFormat], next_pcs: &[usize]) -> ScalarBlock {
    let (last, head) = insts.split_last().expect("empty block");

    let term = lower_terminator(last, next_pcs);

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
    let body: Vec<InstFormat> = body_src.iter().filter(|i| !is_noop(i)).cloned().collect();
    ScalarBlock { pc, body, term }
}

/// Split barriers into typed signal/wait yields, preserving IDs, signal-is-first
/// results and the original instruction order. Resume entries retain the body
/// after each effect; existing branch targets remain unchanged.
pub fn split_at_barriers(program: &impl super::CompilationInput) -> super::Program {
    program.to_ssa().split(|action|!action.is_wave()).0
}
