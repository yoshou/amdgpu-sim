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
use crate::instructions::I;
use crate::rdna4_decoder::{decode_rdna4, InstStream};
use crate::rdna_instructions::InstFormat;
use std::collections::{BTreeMap, BTreeSet};

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
pub(super) struct DecodedBlock {
    pub insts: Vec<InstFormat>,
    pub next_pcs: Vec<usize>,
}

pub(super) struct Decoded {
    pub entry_pc: usize,
    pub blocks: BTreeMap<usize, DecodedBlock>,
}

fn is_terminator(inst: &InstFormat) -> bool {
    matches!(inst, InstFormat::SOPP(i) if matches!(i.op,
        I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 | I::S_CBRANCH_VCCZ | I::S_CBRANCH_VCCNZ | I::S_CBRANCH_EXECZ
        | I::S_CBRANCH_EXECNZ | I::S_BRANCH | I::S_BARRIER_WAIT | I::S_ENDPGM))
}

fn successors(pc: usize, inst: &InstFormat) -> Vec<usize> {
    let target = |simm16: u16| ((pc as i64) + (simm16 as i16 as i64) * 4) as usize;
    match inst {
        InstFormat::SOPP(i) => match i.op {
            I::S_CBRANCH_EXECZ | I::S_CBRANCH_EXECNZ | I::S_CBRANCH_VCCZ | I::S_CBRANCH_VCCNZ | I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 => vec![pc, target(i.simm16)],
            I::S_BRANCH => vec![target(i.simm16)],
            I::S_ENDPGM => vec![],
            _ => vec![pc],
        },
        _ => vec![pc],
    }
}

fn decode_at(memory: &[u8], pc: usize) -> Result<(InstFormat, usize), String> {
    if pc + 8 > memory.len() { return Err(format!("instruction at {pc:#x} is outside the loaded object")); }
    decode_rdna4(InstStream { insts: &memory[pc..] }).map_err(|_| format!("undecodable instruction at {pc:#x}"))
}

struct Search<'a> {
    memory: &'a [u8],
    ranges: BTreeSet<(usize, usize)>,
}

impl Search<'_> {
    fn containing(&self, pc: usize) -> Option<(usize, usize)> {
        self.ranges.iter().copied().find(|&(start, end)| pc >= start && pc < end)
    }
    fn walk(&mut self, start: usize) -> Result<(), String> {
        let mut pc = start;
        let mut last;
        loop {
            let (inst, size) = decode_at(self.memory, pc)?;
            pc += size;
            let stop = is_terminator(&inst) || self.containing(pc).is_some() || super::lift::control::writes_exec(&inst);
            last = inst;
            if stop { break; }
        }
        let next = successors(pc, &last);
        self.ranges.insert((start, pc));
        for next_pc in next {
            if let Some((range_start, range_end)) = self.containing(next_pc) {
                if range_start < next_pc {
                    self.ranges.remove(&(range_start, range_end));
                    self.ranges.insert((range_start, next_pc));
                    self.ranges.insert((next_pc, range_end));
                }
            } else {
                self.walk(next_pc)?;
            }
        }
        Ok(())
    }
}

pub(super) fn program(entry_pc: usize, memory: &[u8]) -> Result<Decoded, String> {
    let mut search = Search { memory, ranges: BTreeSet::new() };
    search.walk(entry_pc)?;
    let ranges: Vec<_> = search.ranges.into_iter().collect();
    for pair in ranges.windows(2) {
        if pair[0].1 != pair[1].0 {
            return Err(format!("decoded ranges are not contiguous: {:#x}..{:#x} and {:#x}..{:#x}", pair[0].0, pair[0].1, pair[1].0, pair[1].1));
        }
    }
    let mut blocks = BTreeMap::new();
    for (start, end) in ranges {
        let mut insts = Vec::new();
        let mut pc = start;
        while pc < end {
            let (inst, size) = decode_at(memory, pc)?;
            insts.push(inst);
            pc += size;
        }
        let next_pcs = successors(end, insts.last().ok_or_else(|| format!("empty range at {start:#x}"))?);
        blocks.insert(start, DecodedBlock { insts, next_pcs });
    }
    Ok(Decoded { entry_pc, blocks })
}

#[cfg(test)]
pub(super) fn load_object(path: &str, descriptor_symbol: &str) -> (usize, Vec<u8>) {
    use object::{Object, ObjectSegment};
    let data = std::fs::read(path).unwrap();
    let elf = object::File::parse(data.as_slice()).unwrap();
    let mut memory = Vec::<u8>::new();
    for segment in elf.segments() {
        let offset = segment.address() as usize;
        let size = segment.size() as usize;
        memory.resize(memory.len().max(offset + size), 0);
        let bytes = segment.data();
        memory[offset..offset + bytes.len().min(size)].copy_from_slice(&bytes[..bytes.len().min(size)]);
    }
    let descriptor_address = elf.symbols().find(|symbol| symbol.name() == Some(descriptor_symbol)).unwrap().address() as usize;
    let descriptor = crate::processor::decode_kernel_desc(&memory[descriptor_address..descriptor_address + 64]);
    (descriptor_address + descriptor.kernel_code_entry_byte_offset, memory)
}

#[cfg(test)]
pub(super) const OBJECTS: &[(&str, &str)] = &[
    ("examples/smallpt/kernel_gfx1200.o", "_ZN7smallptL6kernelEPKNS_6SphereEmjjPNS_7Vector3Ej.kd"),
    ("examples/raytracing/kernel_gfx1200.o", "_Z24ambient_occlusion_kernelP14_hiprtGeometryPh15HIP_vector_typeIiLj2EEf.kd"),
    ("examples/texture/kernel_gfx1200.o", "_Z16histogram_kernelPjjjjP13__hip_texture.kd"),
    ("examples/histogram/kernel_gfx1200.o", "_Z18histogram256_blockPhPji.kd"),
    ("examples/simple_hgemm/kernel_gfx1200.o", "_Z15hgemm_rocwmma_djjjPKDF16_S0_S0_PDF16_jjjjff.kd"),
    ("examples/warp_shuffle/kernel_gfx1200.o", "_Z23matrix_transpose_kernelPfPKfj.kd"),
    ("examples/bitonic_sort/kernel_gfx1200.o", "_Z19bitonic_sort_kernelPjjjb.kd"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_object_decodes_to_the_shared_translator_cfg() {
        for &(path, symbol) in OBJECTS {
            let (entry, memory) = load_object(path, symbol);
            let ours = program(entry, &memory).unwrap();
            let theirs = crate::rdna_translator::RDNAProgram::new(entry, &memory);
            assert_eq!(ours.entry_pc, theirs.entry_pc(), "{path}");
            let mut expected: Vec<_> = theirs.blocks().keys().copied().collect();
            expected.sort_unstable();
            assert_eq!(ours.blocks.keys().copied().collect::<Vec<_>>(), expected, "{path}");
            for (pc, block) in &ours.blocks {
                let other = &theirs.blocks()[pc];
                assert_eq!(block.next_pcs, other.next_pcs(), "{path} block {pc:#x}");
                assert_eq!(format!("{:?}", block.insts), format!("{:?}", other.insts()), "{path} block {pc:#x}");
            }
        }
    }

    #[test]
    fn truncated_input_is_a_structured_error() {
        let (entry, memory) = load_object(OBJECTS[0].0, OBJECTS[0].1);
        assert!(program(entry, &memory[..entry + 8]).is_err());
        assert!(program(memory.len() + 4, &memory).is_err());
    }
}
