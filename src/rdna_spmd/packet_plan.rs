//! Prepare the existing packet optimizations before LLVM emission.
//!
//! Owns the typed SSA function and its representation decisions. Analyses run
//! on that function before native emission, including every clustered member.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use crate::instructions::I;
#[cfg(test)]
use crate::rdna_instructions::InstFormat;
#[cfg(test)]
use crate::rdna_instructions::VGLOBAL;

use super::boundary::BoundaryIo;
use super::ir::{Cond, ScalarProgram, Terminator};
#[cfg(test)]
use super::ir::ScalarBlock;
use super::load_cluster::{self, VgCluster};
use super::regtype::RegSet;
use super::sqrt_idiom::{self, SqrtCollapse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlobalLoad {
    Gather,
    Broadcast,
    Frame { stride_words: u32, offset_words: u32 },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum InstructionAction {
    Emit,
    Cluster(VgCluster),
    ClusterMember,
}

pub(super) struct InstructionPlan {
    pub elide_predicate: bool,
    pub entry_exec_unchanged: bool,
    pub nonempty_exec: bool,
    pub sqrt: Option<SqrtCollapse>,
    pub global_load: GlobalLoad,
    pub action: InstructionAction,
}

pub(super) struct BlockPlan {
    pub instructions: Vec<InstructionPlan>,
    pub fresh: RegSet,
    pub stale: RegSet,
    pub specialize: bool,
}

pub(super) struct MaskRegion {
    pub header: usize,
    pub body: BTreeSet<usize>,
    pub registers: Vec<u32>,
    pub exits: Vec<(usize, usize)>,
}

pub(super) struct PacketPlan<'a> {
    pub width: u32,
    pub function: super::lift::function::Function,
    pub boundary: Option<&'a BTreeMap<usize, BoundaryIo>>,
    pub blocks: BTreeMap<usize, BlockPlan>,
    pub f64_pairs: RegSet,
    pub mask_region: Option<MaskRegion>,
}

impl<'a> PacketPlan<'a> {
    #[cfg(test)]
    pub fn new(program: &ScalarProgram, width: u32, boundary: Option<&'a BTreeMap<usize, BoundaryIo>>) -> Self {
        Self::with_registry(std::sync::Arc::new(super::dialect::DialectRegistry::rdna4()), program, width, boundary)
    }
    pub fn with_registry(
        registry: std::sync::Arc<super::dialect::DialectRegistry>,
        program: &ScalarProgram,
        width: u32,
        boundary: Option<&'a BTreeMap<usize, BoundaryIo>>,
    ) -> Self {
        Self::with_return_state(registry,program,width,boundary,true)
    }
    pub fn with_return_state(
        registry: std::sync::Arc<super::dialect::DialectRegistry>,
        program: &ScalarProgram,
        width: u32,
        boundary: Option<&'a BTreeMap<usize,BoundaryIo>>,
        observe_return: bool,
    ) -> Self {
        // Host-written values on resume must not retain uniform/frame facts.
        let boundary_writes = boundary
            .map(|map| map.iter().map(|(&pc, io)| (pc, io.writes.vgprs().collect())).collect())
            .unwrap_or_default();
        // ReadFirstLane selects an EXEC-active lane. Its ordinary liveness is
        // tracked at the terminator; it does not observe every inactive value.
        let active_reads: BTreeSet<_>=program.blocks.values().filter_map(|b|match &b.term {
            Terminator::Yield {resume,action} if action.op==super::ir::typed::effect::EffectOp::Wave(super::ir::typed::effect::WaveOp::ReadFirstLane)=>Some(*resume),
            _=>None,
        }).collect();
        let boundary_reads: Vec<u32> = boundary
            .map(|map| map.iter().filter(|(pc,_)|!active_reads.contains(pc)).flat_map(|(_,io)| io.reads.vgprs()).collect())
            .unwrap_or_default();
        let exit_reads = if boundary.is_some() && observe_return { (0..256).collect::<Vec<_>>() } else { boundary_reads };

        let mut lifted = BTreeMap::new();
        let mut blocks: BTreeMap<_, _> = program.blocks.iter().map(|(&pc, block)| {
            let mut instructions = Vec::with_capacity(block.body.len());
            let mut semantics = Vec::with_capacity(block.body.len());
            for inst in &block.body {
                let lowering = super::lift::instruction_with_registry(inst, &registry);
                semantics.push(lowering);
                instructions.push(InstructionPlan {
                    elide_predicate: false,
                    entry_exec_unchanged: false,
                    nonempty_exec: false,
                    sqrt: None,
                    global_load: GlobalLoad::Gather,
                    action: InstructionAction::Emit,
                });
            }
            let specialize = false;
            let fresh = [0; 2];
            // Canonical pairs have no live i32 slots to synchronize on edges.
            let stale = [0; 2];
            lifted.insert(pc, semantics);
            (pc, BlockPlan { instructions, fresh, stale, specialize })
        }).collect();
        let lowerings = lifted.iter().map(|(&pc, block)| (pc, block.iter().collect())).collect();
        let mut lifted_function = super::lift::function::Function::lift(registry, program, &lowerings);
        for (&pc, block) in &mut blocks {
            let (normal, sqrt) = sqrt_idiom::analyze(&lifted_function.state.sites[&pc]);
            for (index, instruction) in block.instructions.iter_mut().enumerate() { instruction.sqrt = sqrt[index].clone(); }
            lifted_function.fold_normal_scales(pc, &normal);
        }
        let fresh_in = super::analysis::state::native_pairs(&lifted_function.ir, &lifted_function.state, false);
        let f64_pairs = super::analysis::state::f64_cells(&lifted_function.state);
        let ssa_elide = super::analysis::state::predication(&lifted_function.ir, &lifted_function.state, &exit_reads);
        let ssa_nonempty = super::analysis::state::nonempty(&lifted_function.ir, &lifted_function.state, width, boundary.is_none());
        let varying = super::analysis::state::varying(&lifted_function.ir, &lifted_function.state, &boundary_writes);
        let frame_values = super::analysis::state::frames(&lifted_function.ir, &lifted_function.state, &boundary_writes);
        for (&pc, block) in &mut blocks {
            block.fresh = fresh_in[&pc];
            block.stale = [block.fresh[0] & !f64_pairs[0], block.fresh[1] & !f64_pairs[1]];
            let sites = &lifted_function.state.sites[&pc];
            let memory = &lifted_function.blocks[&pc].memory;
            let mut cluster_rest = 0;
            let mut entry_exec_unchanged = true;
            for (index, instruction) in block.instructions.iter_mut().enumerate() {
                instruction.elide_predicate = ssa_elide[&pc][index];
                instruction.nonempty_exec = ssa_nonempty[&pc][index];
                instruction.entry_exec_unchanged = entry_exec_unchanged;
                entry_exec_unchanged &= !sites[index].exec.writes;
                let load = global_load_ssa(memory.get(&index).map(|p| &p.memory), &sites[index].address, &varying, &frame_values);
                instruction.global_load = load;
                let action = if cluster_rest > 0 {
                    cluster_rest -= 1; InstructionAction::ClusterMember
                } else if matches!(instruction.sqrt, Some(SqrtCollapse::Rescale { .. })) {
                    InstructionAction::Emit
                } else if let Some(cluster) = load_cluster::analyze(memory, sites, index, width, &varying, &frame_values) {
                    cluster_rest = cluster.len - 1; InstructionAction::Cluster(cluster)
                } else { InstructionAction::Emit };
                instruction.action = action;
            }
            block.specialize = block.instructions.iter().enumerate().any(|(index, instruction)|
                instruction.entry_exec_unchanged && !instruction.elide_predicate && !sites[index].writes.is_empty());
        }
        let preparation=super::lift::function::Preparation::Packet {
            inactive:blocks.iter().flat_map(|(&pc,b)|b.instructions.iter().enumerate()
                .filter_map(move |(index,i)|i.elide_predicate.then_some((pc,index)))).collect(),
            observe_return,
        };
        let mask_region = select_mask_region(&lifted_function, boundary.is_some());
        let mut function = lifted_function.prepare(preparation);
        for plan in function.blocks.values_mut().flat_map(|block| block.memory.values_mut()) {
            use super::ir::typed::effect::{MemoryOp, Space};
            // One ISA atomic instruction has no intervening per-lane effects.
            // If no old value is observed, equal-address additions can be
            // serialized consecutively and replaced by their wrapping sum.
            // Volatile accesses and result-returning atomics retain every event.
            plan.group_atomics = width >= 4 && plan.memory.space() == Space::Global
                && plan.memory.op == MemoryOp::AtomicAdd && !plan.memory.returns
                && !plan.memory.semantics.volatile;
        }
        Self { width, boundary, blocks, f64_pairs, mask_region, function }
    }
}

fn select_mask_region(function: &super::lift::function::LiftedFunction, cooperative: bool) -> Option<MaskRegion> {
    // Retain the existing single leaf-loop selection and its mask ownership
    // proof. Other blocks continue to use packed SGPR masks.
    let region = (!cooperative).then(|| super::structured::analyze(function))?
        .loops.into_iter().find(|region| {
            region.children.is_empty()
                && !region.mask_stack.local_scopes.is_empty()
                && region.mask_stack.boundary_live_saved.is_empty()
                && region.mask_stack.unrestored_saved.is_empty()
                && region.control.branch_conditions.iter().all(|cond| matches!(cond, Cond::ExecZ | Cond::ExecNz | Cond::Scc0 | Cond::Scc1))
        })?;
    let body: BTreeSet<_> = region.body.into_iter().collect();
    let exits = body.iter().copied().flat_map(|from| {
        let successors: Vec<_> = function.ir.blocks[&super::ir::typed::cfg::BlockId(from)]
            .term.edges().into_iter().map(|e| e.dst.0).collect();
        let body = &body;
        successors.into_iter().filter(move |to| !body.contains(to)).map(move |to| (from, to))
    }).collect();
    Some(MaskRegion { header: region.header, body, registers: region.mask_stack.mask_sgprs, exits })
}

pub(super) fn cooperative_vgpr_count(
    program: &ScalarProgram,
    declared_vgprs: usize,
    boundary: &BTreeMap<usize, BoundaryIo>,
) -> usize {
    // Some gfx1200 callers still decode the descriptor with the older 4-VGPR
    // granularity, so retain every register the IR or a lifted boundary can
    // observe. Unlike the former 256-register floor, this avoids copying 8 KiB
    // of dead packet state on every coroutine yield in small kernels.
    let required_vgprs = program
        .blocks
        .values()
        .flat_map(|block| block.body.iter())
        .flat_map(|inst| {
            let mut regs = super::lift::access::vgpr_reads(inst);
            regs.extend(super::lift::access::vgpr_writes(inst));
            regs
        })
        .chain(boundary.values().flat_map(|io| io.writes.vgprs()))
        .max()
        .map_or(1, |reg| reg as usize + 1);
    declared_vgprs.max(required_vgprs)
}

#[cfg(test)]
mod cooperative_vgpr_tests {
    use super::*;
    #[cfg(test)]
use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, VOP1};
    use super::super::ir::ScalarBlock;

    #[test]
    fn sizes_packet_state_from_ir_and_boundary_registers() {
        let program = ScalarProgram {
            entry_pc: 1,
            blocks: BTreeMap::from([(
                1,
                ScalarBlock {
                    pc: 1,
                    body: vec![InstFormat::VOP1(VOP1 {
                        src0: SourceOperand::VectorRegister(26),
                        op: I::V_MOV_B32,
                        vdst: 25,
                    })],
                    term: Terminator::Return,
                },
            )]),
        };

        assert_eq!(cooperative_vgpr_count(&program, 16, &BTreeMap::new()), 28);
        let mut boundary = BoundaryIo::default();
        boundary.writes.add_vgpr(31);
        assert_eq!(
            cooperative_vgpr_count(&program, 16, &BTreeMap::from([(2, boundary)])),
            32
        );
        assert_eq!(cooperative_vgpr_count(&program, 64, &BTreeMap::new()), 64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_instructions::{SourceOperand, VOP1, VOP3SD};

    fn mov(dst: u8, src: SourceOperand) -> InstFormat {
        InstFormat::VOP1(VOP1 { src0: src, op: I::V_MOV_B32, vdst: dst })
    }

    fn load(dst: u8, offset: i32) -> InstFormat {
        InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_LOAD_B128, vaddr: 10, vsrc: 0, vdst: dst,
            scope: 0, th: 0, ioffset: offset as u32 & 0x00ff_ffff,
            saddr: 124, sve: 0,
        })
    }

    fn frame() -> InstFormat {
        InstFormat::VOP3SD(VOP3SD {
            vdst: 10, sdst: 106, cm: 0, op: I::V_MAD_CO_U64_U32,
            src0: SourceOperand::VectorRegister(0),
            src1: SourceOperand::IntegerConstant(32),
            src2: SourceOperand::ScalarRegister(0), omod: 0, neg: 0,
        })
    }

    fn program(body: Vec<InstFormat>) -> ScalarProgram {
        ScalarProgram {
            entry_pc: 1,
            blocks: BTreeMap::from([(1, ScalarBlock { pc: 1, body, term: Terminator::Return })]),
        }
    }

    #[test]
    fn transfers_cluster_members_before_selecting_later_loads() {
        let program = program(vec![
            mov(10, SourceOperand::VectorRegister(0)),
            load(20, 0), load(24, 16), load(10, 32),
            load(28, 48),
            mov(10, SourceOperand::IntegerConstant(0)),
            mov(11, SourceOperand::IntegerConstant(0)),
            load(28, 0),
        ]);
        for width in [1, 2, 4, 8, 16] {
            let plan = PacketPlan::new(&program, width, None);
            let instructions = &plan.blocks[&1].instructions;
            assert!(matches!(&instructions[1].action,
                InstructionAction::Cluster(c) if c.len == 3 && c.span == 6));
            assert!(matches!(instructions[2].action, InstructionAction::ClusterMember));
            assert!(matches!(instructions[3].action, InstructionAction::ClusterMember));
            assert!(matches!(instructions[4].action, InstructionAction::Emit));
            assert_eq!(instructions[4].global_load, GlobalLoad::Gather);
            assert_eq!(instructions[7].global_load, GlobalLoad::Broadcast);
        }
    }

    #[test]
    fn frame_selection_checks_bounds_alignment_and_redefinition() {
        let program = program(vec![
            frame(), load(20, 16), load(20, 20), load(20, -4), load(20, 2),
            mov(11, SourceOperand::IntegerConstant(0)), load(20, 0),
        ]);
        let plan = PacketPlan::new(&program, 16, None);
        let instructions = &plan.blocks[&1].instructions;
        assert_eq!(instructions[1].global_load, GlobalLoad::Frame { stride_words: 8, offset_words: 4 });
        for index in [2, 3, 4, 6] {
            assert_eq!(instructions[index].global_load, GlobalLoad::Gather);
        }
    }

    #[test]
    fn resume_write_invalidates_frame_and_uniform_address_choices() {
        let mut program = program(vec![frame()]);
        program.blocks.get_mut(&1).unwrap().term = Terminator::Barrier { resume: 2 };
        program.blocks.insert(2, ScalarBlock { pc: 2, body: vec![load(20, 0)], term: Terminator::Return });
        let mut io = BoundaryIo::default();
        io.writes.add_vgpr(11);
        let boundary = BTreeMap::from([(2, io)]);
        let no_writes = BTreeMap::new();
        let before = PacketPlan::new(&program, 16, Some(&no_writes));
        assert!(matches!(before.blocks[&2].instructions[0].global_load, GlobalLoad::Frame { .. }));
        let after = PacketPlan::new(&program, 16, Some(&boundary));
        assert_eq!(after.blocks[&2].instructions[0].global_load, GlobalLoad::Gather);
        drop((before, after));

        program.blocks.get_mut(&1).unwrap().body = vec![
            mov(10, SourceOperand::IntegerConstant(0)),
            mov(11, SourceOperand::IntegerConstant(0)),
        ];
        let before = PacketPlan::new(&program, 16, Some(&no_writes));
        assert_eq!(before.blocks[&2].instructions[0].global_load, GlobalLoad::Broadcast);
        let after = PacketPlan::new(&program, 16, Some(&boundary));
        assert_eq!(after.blocks[&2].instructions[0].global_load, GlobalLoad::Gather);
    }

    #[test]
    fn boundary_observers_keep_predication_and_select_full_exec_clone() {
        let program = program(vec![mov(20, SourceOperand::IntegerConstant(1))]);
        let no_boundary = PacketPlan::new(&program, 16, None);
        assert!(no_boundary.blocks[&1].instructions[0].elide_predicate);
        assert!(!no_boundary.blocks[&1].specialize);
        let mut io = BoundaryIo::default();
        io.reads.add_vgpr(20);
        let boundary = BTreeMap::from([(2, io)]);
        let observed = PacketPlan::new(&program, 16, Some(&boundary));
        assert!(!observed.blocks[&1].instructions[0].elide_predicate);
        assert!(observed.blocks[&1].specialize);
    }
}

fn global_load_ssa(
    memory: Option<&super::lift::memory::Memory>,
    address: &[super::ir::typed::ValueId], varying: &[bool],
    frames: &BTreeMap<(super::ir::typed::ValueId, super::ir::typed::ValueId), u32>,
) -> GlobalLoad {
    use super::lift::memory::Address;
    use super::ir::typed::effect::{MemoryOp, MemSize};
    let Some(memory) = memory else { return GlobalLoad::Gather; };
    let Address::Global { scalar, offset, .. } = memory.address else { return GlobalLoad::Gather; };
    if memory.op != MemoryOp::Load(MemSize::B32) || !(1..=4).contains(&memory.words) { return GlobalLoad::Gather; }
    if address.iter().all(|v| !varying[v.0]) { return GlobalLoad::Broadcast; }
    if scalar.is_none() {
        if let Some(&stride) = frames.get(&(address[0], address[1])) {
            let words = stride / 4;
            let offset_word = offset / 4;
            if words >= 1 && offset % 4 == 0 && offset_word >= 0 && offset_word as u32 + memory.words <= words {
                return GlobalLoad::Frame { stride_words: words, offset_words: offset_word as u32 };
            }
        }
    }
    GlobalLoad::Gather
}
