//! Reading a lane program off a wave program.
//!
//! A wave program runs its lanes in the EXEC register's lockstep: a branch
//! asks the wave whether any lane takes it, and a lane that does not sits
//! out the arm. The lane program describes one lane, whose branches are its
//! own, so the lockstep lowering can rebuild the masks from its control flow
//! and skip what no lane needs.
//!
//! Every query of the wave -- `any`, a ballot, a test of a ballot word, a
//! read of the first lane -- is first answered from the lane's own bit. The
//! [`proof`] checks that every store then happens, and writes, what the wave
//! stores for that lane. Where a store depends on how another lane answered,
//! the proof names the queries at fault and they are *kept*: the lane program
//! asks them of the lanes that are at them, which is what the wave answers
//! when the lanes that are not at them hold no bit of the question. An
//! operation that exchanges values between lanes, a barrier and a fence are
//! kept as they are; the lowering refuses a program where lanes may be
//! elsewhere at one of them, and the program then runs as a wave program.
//!
//! [`super::lockstep`] lowers the lane program into packets, with the kept
//! operations answered over the wave -- inside the packet where it holds the
//! wave, and through the cooperative scheduler otherwise.

mod fold;
mod logic;
mod proof;
mod rewrite;

use crate::rdna_spmd::analysis::facts;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::LiftedFunction;
use crate::rdna_spmd::refusal::Refusal;
use logic::Kept;
use std::collections::BTreeSet;

/// A lane program and the provenances of the kept queries the lowering must
/// run with every lane of the wave at them, since lanes that are not at them
/// may hold the bit they ask about.
pub(crate) struct Lane {
    pub function: LiftedFunction,
    pub everyone: BTreeSet<u64>,
}

impl From<LiftedFunction> for Lane {
    fn from(function: LiftedFunction) -> Self {
        Self {
            function,
            everyone: BTreeSet::new(),
        }
    }
}

fn supported(f: &Func) -> Result<(), Refusal> {
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            let refuse = |reason| Err(Refusal::at(id, Some(index), reason));
            match inst {
                Inst::Packet { .. } => return refuse("a packet query in a wave program"),
                Inst::Core {
                    op: Op::Env(Env::OutsideLanes | Env::PacketLaneId),
                    ..
                } => return refuse("a packet environment value in a wave program"),
                _ => {}
            }
        }
    }
    Ok(())
}

pub(crate) fn decompile(function: &LiftedFunction) -> Result<Lane, Refusal> {
    supported(&function.ir)?;
    let f = &function.ir;
    let exec = function.registry.registers().exec;
    let exec_index = function.parameter_inputs.iter().position(
        |p| matches!(p.source, crate::rdna_spmd::program::ParameterSource::MaskBit(r) if r == exec),
    );
    if let Some(path) = std::env::var_os("AMDGPU_SIM_DUMP_WAVE") {
        std::fs::write(
            path,
            crate::rdna_spmd::ir::print::func(&function.registry, f),
        )
        .unwrap();
    }
    // Every query is answered from the lane's own bit until the proof names
    // one whose answer the wave must give; each round keeps more, so the
    // rounds end.
    let mut kept = Kept::default();
    let (facts, everyone) = loop {
        let facts = facts::Facts::new(f, &function.parameter_inputs, &kept.words);
        let mut logic = logic::Logic::new(f, &facts, &kept.queries);
        match proof::prove(
            f,
            &facts,
            &mut logic,
            &function.parameter_inputs,
            exec_index,
            &kept,
        ) {
            Ok(everyone) => break (facts, everyone),
            Err(refusal) if !refusal.keep.is_empty() => {
                let before = kept.clone();
                for v in refusal.keep {
                    match facts.inst(f, v) {
                        Some(Inst::Effect {
                            op: EffectOp::Wave(WaveOp::Any),
                            ..
                        }) => {
                            kept.queries.insert(v);
                        }
                        Some(Inst::Core {
                            op: Op::Cmp(_, a, b),
                            ..
                        }) => {
                            let word = logic::lane_test(f, &facts, *a, *b)
                                .expect("a marked test is a test of a lane word");
                            kept.words.insert(word);
                        }
                        other => unreachable!("a marker names a query, not {:?}", other),
                    }
                }
                assert!(kept != before, "the proof named queries it already keeps");
            }
            Err(refusal) => return Err(refusal),
        }
    };
    if std::env::var_os("AMDGPU_SIM_PRINT_IR").is_some() {
        eprintln!(
            "; the lane program keeps {} queries and {} words",
            kept.queries.len(),
            kept.words.len()
        );
    }
    let mut lane = rewrite::lane_program(f, &facts, &kept);
    fold::fold(&mut lane, &function.parameter_inputs, exec_index);
    lane.compact();
    if let Err(e) = lane.check(&function.registry) {
        if cfg!(test) {
            panic!(
                "the lane program read off a proven wave program is invalid: {}",
                e
            );
        }
        return Err(Refusal::at(
            f.entry,
            None,
            "the lane program did not verify",
        ));
    }
    Ok(Lane {
        function: LiftedFunction {
            registry: function.registry.clone(),
            ir: lane,
            parameter_inputs: function.parameter_inputs.clone(),
            revision: function.revision + 1,
        },
        everyone,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, SOP1, SOP2, VGLOBAL, VOP2, VOPC};
    use crate::rdna_spmd::compiler::{compile_lockstep, compile_scalar};
    use crate::rdna_spmd::engine::kernel::Code;
    use crate::rdna_spmd::program::{Parameter, ParameterSource, Program};
    use crate::rdna_spmd::targets::rdna4::decode::{Cond, ScalarBlock, ScalarProgram, Terminator};
    use crate::rdna_spmd::targets::rdna4::lift::wave::{Destination, Operand, YieldAction};
    use crate::rdna_spmd::CompilationInput;

    const EXEC: u8 = 126;
    const VCC: u8 = 106;

    fn k(value: u64) -> SourceOperand {
        SourceOperand::IntegerConstant(value)
    }

    fn mov(sdst: u8, source: u8) -> InstFormat {
        InstFormat::SOP1(SOP1 {
            op: I::S_MOV_B32,
            ssrc0: SourceOperand::ScalarRegister(source),
            sdst,
        })
    }

    fn and(sdst: u8, a: u8, b: u8) -> InstFormat {
        InstFormat::SOP2(SOP2 {
            op: I::S_AND_B32,
            ssrc0: SourceOperand::ScalarRegister(a),
            ssrc1: SourceOperand::ScalarRegister(b),
            sdst,
        })
    }

    fn vop(op: I, src0: SourceOperand, vsrc1: u8, vdst: u8) -> InstFormat {
        InstFormat::VOP2(VOP2 {
            op,
            src0,
            vsrc1,
            vdst,
            literal_constant: None,
        })
    }

    fn nonzero(r: u8) -> InstFormat {
        InstFormat::VOPC(VOPC {
            op: I::V_CMP_NE_U32,
            src0: k(0),
            vsrc1: r,
        })
    }

    fn store(r: u8) -> InstFormat {
        store_at(0, r)
    }

    fn store_at(vaddr: u8, r: u8) -> InstFormat {
        InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_STORE_B32,
            vaddr,
            vsrc: r,
            vdst: 0,
            scope: 0,
            th: 0,
            ioffset: 0,
            saddr: 0,
            sve: 0,
        })
    }

    fn lifted(blocks: Vec<(Vec<InstFormat>, Terminator)>) -> LiftedFunction {
        let blocks = blocks
            .into_iter()
            .enumerate()
            .map(|(pc, (body, term))| (pc, ScalarBlock { pc, body, term }))
            .collect();
        ScalarProgram {
            entry_pc: 0,
            blocks,
        }
        .to_ssa()
        .function
    }

    #[test]
    fn a_loop_each_lane_leaves_after_its_own_trips_runs_them_at_every_width() {
        let wave = lifted(vec![
            (
                vec![
                    mov(20, EXEC),
                    vop(I::V_LSHRREV_B32, k(2), 0, 3),
                    vop(I::V_ADD_NC_U32, k(1), 3, 3),
                ],
                Terminator::Jump(1),
            ),
            (
                vec![
                    vop(I::V_ADD_NC_U32, k(7), 8, 8),
                    vop(I::V_SUBREV_NC_U32, k(1), 3, 3),
                    nonzero(3),
                    mov(EXEC, VCC),
                ],
                Terminator::Branch {
                    cond: Cond::ExecNz,
                    taken: 1,
                    fallthrough: 2,
                },
            ),
            (vec![mov(EXEC, 20), store(8)], Terminator::Return),
        ]);
        let lane = decompile(&wave).expect("a lane's trips depend on that lane alone");
        for block in lane.function.ir.blocks.values() {
            for inst in &block.insts {
                assert!(
                    !matches!(
                        inst,
                        Inst::Effect {
                            op: EffectOp::Wave(_),
                            ..
                        } | Inst::Core {
                            op: Op::Env(Env::ValidLane),
                            ..
                        }
                    ),
                    "the lane program keeps {:?}",
                    inst
                );
            }
        }
        for width in [0u32, 1, 2, 4, 8, 16, 32] {
            let lanes = width.max(1) as usize;
            let mut output = vec![0u32; lanes];
            let mut sgprs = [0u32; 128];
            let address = output.as_mut_ptr() as u64;
            sgprs[0] = address as u32;
            sgprs[1] = (address >> 32) as u32;
            let mut vgprs = vec![0u32; 256 * lanes];
            for l in 0..lanes {
                vgprs[l] = l as u32 * 4;
            }
            unsafe {
                if width == 0 {
                    compile_scalar(
                        Program {
                            function: lane.function.clone(),
                        },
                        256,
                    )
                    .run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0);
                } else {
                    let (Code::Packet(kernel), _) = compile_lockstep(&lane, 256, width, None)
                        .expect("every lane is at each operation over the wave")
                    else {
                        panic!("a lane program that keeps no wave operation runs alone")
                    };
                    kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0, u32::MAX, 0);
                }
            }
            let expected: Vec<u32> = (0..lanes as u32).map(|l| 7 * (l + 1)).collect();
            assert_eq!(output, expected, "width {}", width);
        }
    }

    fn kept_wave_ops(lane: &LiftedFunction) -> Vec<WaveOp> {
        lane.ir
            .blocks
            .values()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match inst {
                Inst::Effect {
                    op: EffectOp::Wave(op),
                    ..
                } => Some(*op),
                _ => None,
            })
            .collect()
    }

    /// Every lane stores its index when any lane's index is `wanted`: the
    /// lanes that hold it narrow EXEC, the branch asks the wave whether any
    /// lane is left, and the store runs under the EXEC saved before.
    fn store_when_any_lane_is(wanted: u64) -> LiftedFunction {
        lifted(vec![
            (
                vec![
                    mov(20, EXEC),
                    InstFormat::VOPC(VOPC {
                        op: I::V_CMP_EQ_U32,
                        src0: k(wanted),
                        vsrc1: 0,
                    }),
                    and(EXEC, VCC, EXEC),
                ],
                Terminator::Branch {
                    cond: Cond::ExecNz,
                    taken: 1,
                    fallthrough: 2,
                },
            ),
            (
                vec![
                    mov(EXEC, 20),
                    vop(I::V_LSHLREV_B32, k(2), 0, 3),
                    store_at(3, 0),
                ],
                Terminator::Return,
            ),
            (vec![], Terminator::Return),
        ])
    }

    #[test]
    fn a_wave_program_whose_query_is_kept_runs_at_every_width() {
        use crate::rdna_spmd::{compile, dispatch, CompileOptions, GridDims};
        let mut kd = crate::processor::decode_kernel_desc(&[0; 64]);
        kd.enable_sgpr_kernarg_segment_ptr = true;
        let count = 32u32;
        let dims = GridDims {
            num_wg_x: 1,
            num_wg_y: 1,
            num_wg_z: 1,
            wg_x: count,
            wg_y: 1,
            wg_z: 1,
        };
        for (wanted, stores) in [(5, true), (40, false)] {
            let program = Program {
                function: store_when_any_lane_is(wanted),
            };
            for width in [0u32, 1, 2, 4, 8, 16, 32] {
                let kernel = compile(
                    &program,
                    CompileOptions {
                        width,
                        num_vgprs: 256,
                        workgroup_x: Some(count),
                    },
                );
                let mut output = vec![u32::MAX; count as usize];
                dispatch(&kernel, &kd, output.as_mut_ptr() as u64, 0, dims, 0, 0, 1);
                let wanted: Vec<u32> = (0..count)
                    .map(|l| if stores { l } else { u32::MAX })
                    .collect();
                assert_eq!(output, wanted, "lane {wanted:?} at width {width}");
            }
        }
    }

    #[test]
    fn a_store_a_lane_reaches_only_behind_other_lanes_keeps_the_query() {
        let wave = lifted(vec![
            (
                vec![mov(20, EXEC), nonzero(0), and(EXEC, VCC, EXEC)],
                Terminator::Branch {
                    cond: Cond::ExecNz,
                    taken: 1,
                    fallthrough: 2,
                },
            ),
            (vec![mov(EXEC, 20), store(0)], Terminator::Return),
            (vec![], Terminator::Return),
        ]);
        let lane = decompile(&wave).expect("the query the store follows is kept");
        assert_eq!(
            kept_wave_ops(&lane.function),
            vec![WaveOp::Any],
            "lane 0 stores behind other lanes, so the lane program asks the wave"
        );
    }

    fn parsed(text: &str) -> LiftedFunction {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        let ir = crate::rdna_spmd::ir::parse::func(&registry, text).unwrap();
        let input = |source, ty| Parameter { source, ty };
        LiftedFunction {
            registry: std::sync::Arc::new(registry),
            ir,
            parameter_inputs: vec![
                input(ParameterSource::Vgpr(0), Ty::I32),
                input(ParameterSource::Sgpr(0), Ty::I32),
                input(ParameterSource::Sgpr(1), Ty::I32),
                input(ParameterSource::MaskBit(126), Ty::I1),
            ],
            revision: 0,
        }
    }

    /// A store every active lane makes when any lane holds a bit: the test
    /// of the ballot is the wave's, so the word stays a ballot. Over a bit
    /// only active lanes hold, a ballot over the lanes at it is the word the
    /// wave computes; over a raw compare, the wave's word holds bits of lanes
    /// that are not at it, which the lane program cannot read.
    fn store_when_any_lane_holds(masked: bool) -> Result<Lane, Refusal> {
        let balloted = if masked { "v9" } else { "v5" };
        decompile(&parsed(&format!(
            "func entry b0
             b0(v0: i32, v1: i32, v2: i32, v3: i1):
               v4: i32 = const i32 0x0
               v5: i1 = cmp ne v0, v4
               v9: i1 = int and v5, v3
               v6: i32 = effect !p0 wave ballot ({balloted})
               v7: i1 = cmp ne v6, v4
               v8: i1 = int and v7, v3
               v10: i64 = pack64 v1, v2
               v11: i64 = convert zext i64 v0
               v12: i64 = int add v10, v11
               v13: i32 = const i32 0x7
               effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v12, v13, v8)
               ret"
        )))
    }

    #[test]
    fn a_store_when_any_lane_holds_a_bit_keeps_the_word() {
        let lane = store_when_any_lane_holds(true).expect("the word the test reads is kept");
        assert_eq!(kept_wave_ops(&lane.function), vec![WaveOp::Ballot]);
    }

    #[test]
    fn a_store_when_any_lane_holds_a_bit_lanes_exec_leaves_out_may_hold_is_refused() {
        let refusal = store_when_any_lane_holds(false)
            .err()
            .expect("the ballot holds bits of lanes that are not at it");
        assert_eq!(
            refusal.reason,
            "whether a store happens depends on the other lanes"
        );
    }

    #[test]
    fn a_store_under_a_query_of_what_a_query_left_out_keeps_the_words() {
        let lane = decompile(&parsed(
            "func entry b0
             b0(v0: i32, v1: i32, v2: i32, v3: i1):
               v4: i32 = const i32 0x0
               v5: i1 = cmp ne v0, v4
               v19: i1 = int and v5, v3
               v6: i32 = effect !p0 wave ballot (v19)
               v7: i1 = cmp ne v6, v4
               v8: i1 = const i1 0x1
               v9: i1 = int xor v5, v8
               v10: i1 = int and v7, v9
               v20: i1 = int and v10, v3
               v11: i32 = effect !p1 wave ballot (v20)
               v12: i1 = cmp ne v11, v4
               v13: i1 = int and v12, v5
               v14: i1 = int and v13, v3
               v15: i64 = pack64 v1, v2
               v16: i64 = convert zext i64 v0
               v17: i64 = int add v15, v16
               v18: i32 = const i32 0x7
               effect !p2 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v17, v18, v14)
               ret",
        ))
        .expect("the words the store's test reads are kept");
        assert_eq!(
            kept_wave_ops(&lane.function),
            vec![WaveOp::Ballot, WaveOp::Ballot],
            "the tests of the two words are the wave's, so both words stay ballots"
        );
    }

    #[test]
    fn a_write_into_another_lane_is_kept() {
        let write = YieldAction::new(
            EffectOp::Wave(WaveOp::WriteLane),
            vec![
                Operand::Source(k(0)),
                Operand::Source(k(0)),
                Operand::Source(SourceOperand::VectorRegister(2)),
                Operand::Source(k(2)),
            ],
            vec![Destination::Vgpr(2)],
        );
        let wave = lifted(vec![
            (
                vec![],
                Terminator::Yield {
                    resume: 1,
                    action: Box::new(write),
                },
            ),
            (vec![], Terminator::Return),
        ]);
        let lane = decompile(&wave).expect("every lane is at the write");
        assert_eq!(kept_wave_ops(&lane.function), vec![WaveOp::WriteLane]);
    }
}
