//! Prepare the existing packet optimizations before LLVM emission.
//!
//! Plans borrow the exact program they describe, preventing mutation while its
//! facts are consumed. Flow facts are transferred once per source instruction,
//! including members of a clustered load. Only the selected lowering is kept;
//! per-instruction copies of the scratch-frame maps are unnecessary.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::instructions::I;
use crate::rdna_instructions::{sext_ioffset, InstFormat, VGLOBAL};

use super::boundary::BoundaryIo;
use super::ir::{Cond, ScalarBlock, ScalarProgram, Terminator};
use super::load_cluster::{self, VgCluster};
use super::regtype::RegSet;
use super::sqrt_idiom::{self, SqrtCollapse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlobalLoad {
    Gather,
    Broadcast,
    Frame { stride_words: u32, offset_words: u32 },
}

pub(super) enum InstructionAction {
    Emit,
    Cluster(VgCluster),
    ClusterMember,
}

pub(super) struct InstructionPlan<'a> {
    pub lowering: super::lift::Lowering<'a>,
    pub elide_predicate: bool,
    pub entry_exec_unchanged: bool,
    pub nonempty_exec: bool,
    pub normal_ldexp: bool,
    pub sqrt: Option<SqrtCollapse>,
    pub global_load: GlobalLoad,
    pub action: InstructionAction,
}

pub(super) struct BlockPlan<'a> {
    pub block: &'a ScalarBlock,
    pub instructions: Vec<InstructionPlan<'a>>,
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
    pub program: &'a ScalarProgram,
    pub width: u32,
    pub function: super::lift::function::Function,
    pub boundary: Option<&'a BTreeMap<usize, BoundaryIo>>,
    pub blocks: BTreeMap<usize, BlockPlan<'a>>,
    pub f64_pairs: RegSet,
    pub fresh_in: BTreeMap<usize, RegSet>,
    pub mask_region: Option<MaskRegion>,
}

impl<'a> PacketPlan<'a> {
    pub fn new(
        program: &'a ScalarProgram,
        width: u32,
        boundary: Option<&'a BTreeMap<usize, BoundaryIo>>,
    ) -> Self {
        // Host-written values on resume must not retain uniform/frame facts.
        let boundary_writes = boundary
            .map(|map| map.iter().map(|(&pc, io)| (pc, io.writes.vgprs().collect())).collect())
            .unwrap_or_default();
        // Boundary operations can read every lane, even an EXEC-inactive lane.
        let boundary_reads: Vec<u32> = boundary
            .map(|map| map.values().flat_map(|io| io.reads.vgprs()).collect())
            .unwrap_or_default();
        let exit_reads = if boundary.is_some() { (0..256).collect::<Vec<_>>() } else { boundary_reads };
        let elide = super::vec_live::analyze_with_exit_live(program, &exit_reads);
        let nonempty = nonempty_exec(program,width,boundary.is_none());
        let fresh_in = super::freshness::analyze(program);
        let f64_pairs = super::regtype::f64_read_pairs(program);
        let div_in = if boundary.is_some() {
            super::vec_live::divergent_entry_with_seed_and_boundary_writes(
                program, [1, 0], &boundary_writes,
            )
        } else {
            super::vec_live::divergent_entry(program)
        };
        let frame_in = if boundary.is_some() {
            super::vec_live::frame_entry_with_boundary_writes(program, &boundary_writes)
        } else {
            super::vec_live::frame_entry(program)
        };
        let blocks: BTreeMap<_, _> = program.blocks.iter().map(|(&pc, block)| {
            let flags = &elide[&pc];
            let normal = sqrt_idiom::normal_sqrt_ldexp_indices(&block.body);
            let sqrt = sqrt_idiom::sqrt_collapse_sites(&block.body);
            let mut divergent = div_in[&pc];
            let mut frames = frame_in[&pc].clone();
            let mut cluster_rest = 0;
            let mut instructions = Vec::with_capacity(block.body.len());
            let mut entry_exec_unchanged = true;
            for (idx, inst) in block.body.iter().enumerate() {
                let action = if cluster_rest > 0 {
                    cluster_rest -= 1;
                    InstructionAction::ClusterMember
                } else if matches!(sqrt[idx], Some(SqrtCollapse::Rescale { .. })) {
                    // Rescale substitution has priority over load clustering.
                    InstructionAction::Emit
                } else if let Some(cluster) = load_cluster::analyze(
                    &block.body[idx..], width, divergent, &frames,
                ) {
                    cluster_rest = cluster.len - 1;
                    InstructionAction::Cluster(cluster)
                } else {
                    InstructionAction::Emit
                };
                instructions.push(InstructionPlan {
                    lowering: super::lift::instruction(inst),
                    elide_predicate: flags[idx],
                    entry_exec_unchanged,
                    nonempty_exec: nonempty[&pc][idx],
                    normal_ldexp: normal[idx],
                    sqrt: sqrt[idx].clone(),
                    global_load: match inst {
                        InstFormat::VGLOBAL(g) => global_load(g, divergent, &frames),
                        _ => GlobalLoad::Gather,
                    },
                    action,
                });
                entry_exec_unchanged &= !super::active::writes_exec(inst);
                super::vec_live::div_transfer(inst, &mut divergent);
                super::vec_live::frame_transfer(inst, &mut frames);
            }
            // A full-EXEC clone avoids preserving the old value of a predicated
            // destination. Keep the existing profitability condition exactly.
            let specialize = block.body.iter().enumerate().any(|(idx, inst)| {
                instructions[idx].entry_exec_unchanged && !flags[idx]
                    && !super::freshness::vgpr_writes(inst).is_empty()
            });
            let fresh = fresh_in[&pc];
            // Canonical pairs have no live i32 slots to synchronize on edges.
            let stale = [fresh[0] & !f64_pairs[0], fresh[1] & !f64_pairs[1]];
            (pc, BlockPlan { block, instructions, fresh, stale, specialize })
        }).collect();
        let lowerings = blocks.iter().map(|(&pc, block)| (pc, block.instructions.iter().map(|i| &i.lowering).collect())).collect();
        let function = super::lift::function::Function::new(program, &lowerings, boundary);
        let mask_region = select_mask_region(program, boundary.is_some());
        Self { program, width, boundary, blocks, f64_pairs, fresh_in, mask_region, function }
    }
}

fn global_load(g: &VGLOBAL, divergent: [u128; 2], frames: &HashMap<u32, u32>) -> GlobalLoad {
    let words = match g.op {
        I::GLOBAL_LOAD_B32 => 1,
        I::GLOBAL_LOAD_B64 => 2,
        I::GLOBAL_LOAD_B96 => 3,
        I::GLOBAL_LOAD_B128 => 4,
        _ => return GlobalLoad::Gather,
    };
    let uniform = |r: u32| (divergent[(r >> 7) as usize] >> (r & 127)) & 1 == 0;
    let uniform_addr = uniform(g.vaddr as u32)
        && (g.saddr != 124 || uniform(g.vaddr as u32 + 1));
    if uniform_addr {
        return GlobalLoad::Broadcast;
    }
    if g.saddr == 124 {
        if let Some(&stride_bytes) = frames.get(&(g.vaddr as u32)) {
            let sp4 = stride_bytes / 4;
            let ioffset = sext_ioffset(g.ioffset) as i64 as u64;
            let ioff_w = (ioffset as i64) / 4;
            if sp4 >= 1 && ioffset % 4 == 0 && ioff_w >= 0 && (ioff_w as u32 + words) <= sp4 {
                return GlobalLoad::Frame { stride_words: sp4, offset_words: ioff_w as u32 };
            }
        }
    }
    GlobalLoad::Gather
}

fn select_mask_region(program: &ScalarProgram, cooperative: bool) -> Option<MaskRegion> {
    // Retain the existing single leaf-loop selection and its mask ownership
    // proof. Other blocks continue to use packed SGPR masks.
    let region = (!cooperative).then(|| super::analyze_structured(program))?
        .loops.into_iter().find(|region| {
            region.children.is_empty()
                && !region.mask_stack.local_scopes.is_empty()
                && region.mask_stack.boundary_live_saved.is_empty()
                && region.mask_stack.unrestored_saved.is_empty()
                && region.control.branch_conditions.iter().all(|cond| matches!(cond, Cond::ExecZ | Cond::ExecNz | Cond::Scc0 | Cond::Scc1))
        })?;
    let body: BTreeSet<_> = region.body.into_iter().collect();
    let exits = body.iter().copied().flat_map(|from| {
        let successors = match program.blocks[&from].term {
            Terminator::Return => vec![],
            Terminator::Jump(target) => vec![target],
            Terminator::Branch { taken, fallthrough, .. } => vec![taken, fallthrough],
            Terminator::Barrier { resume } | Terminator::Yield { resume, .. } => vec![resume],
        };
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
            let mut regs = super::vec_live::vgpr_reads(inst);
            regs.extend(super::freshness::vgpr_writes(inst));
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

/// Only establishes existence of an active packet lane, never that every lane
/// is active. This suffices for a uniform-address load: one active lane proves
/// the shared pointer must be valid. All EXEC writes invalidate the fact unless
/// a constant assignment or an OR preserving the old EXEC establishes it.
fn nonempty_exec(program: &ScalarProgram, width: u32, initial: bool) -> BTreeMap<usize, Vec<bool>> {
    use crate::rdna_instructions::SourceOperand;
    let transfer = |inst: &InstFormat, old: bool| -> bool {
        let wave_writes =
            super::lift::wave::instruction(inst).map_or(false, |a| a.io().writes.has_sgpr(126));
        if !super::active::writes_exec(inst) && !wave_writes {
            return old;
        }
        match inst {
            InstFormat::SOP1(i) if i.sdst == 126 && matches!(i.op, I::S_MOV_B32 | I::S_MOV_B64) => {
                match i.ssrc0 {
                    SourceOperand::IntegerConstant(v) => initial && v & ((1u64 << width) - 1) != 0,
                    SourceOperand::LiteralConstant(v) => initial && v & ((1u32 << width) - 1) != 0,
                    _ => false,
                }
            }
            InstFormat::SOP2(i) if i.sdst == 126 && matches!(i.op, I::S_OR_B32 | I::S_OR_B64) => {
                old && (matches!(i.ssrc0, SourceOperand::ScalarRegister(126))
                    || matches!(i.ssrc1, SourceOperand::ScalarRegister(126)))
            }
            _ => false,
        }
    };
    let mut entries: BTreeMap<_, _> = program.blocks.keys().map(|&pc| (pc, true)).collect();
    loop {
        let mut incoming = BTreeMap::from([(program.entry_pc, initial)]);
        for (&pc, b) in &program.blocks {
            let exit = b
                .body
                .iter()
                .fold(entries[&pc], |st, inst| transfer(inst, st));
            let edges = match &b.term {
                Terminator::Return => vec![],
                Terminator::Jump(p) => vec![(*p, exit)],
                Terminator::Barrier { resume } => vec![(*resume, false)],
                Terminator::Yield { resume, action } => {
                    vec![(*resume, exit && !action.io().writes.has_sgpr(126))]
                }
                Terminator::Branch {
                    cond,
                    taken,
                    fallthrough,
                } => match cond {
                    Cond::ExecZ => vec![(*taken, false), (*fallthrough, true)],
                    Cond::ExecNz => vec![(*taken, true), (*fallthrough, false)],
                    _ => vec![(*taken, exit), (*fallthrough, exit)],
                },
            };
            for (to, st) in edges {
                incoming.entry(to).and_modify(|v| *v &= st).or_insert(st);
            }
        }
        let mut changed = false;
        for (pc, st) in incoming {
            if entries[&pc] != st {
                entries.insert(pc, st);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    program
        .blocks
        .iter()
        .map(|(&pc, b)| {
            let mut st = entries[&pc];
            let values = b
                .body
                .iter()
                .map(|inst| {
                    let before = st;
                    st = transfer(inst, st);
                    before
                })
                .collect();
            (pc, values)
        })
        .collect()
}
