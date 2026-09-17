mod bdd;
mod facts;
mod fold;
mod lockstep;
mod logic;
mod loops;
mod proof;
mod rewrite;

pub(crate) use proof::Refusal;
pub(crate) use lockstep::{lockstep, Packing};

use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::LiftedFunction;

fn supported(f: &Func) -> Result<(), Refusal> {
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            let refuse = |reason| {
                Err(Refusal {
                    block: id,
                    index: Some(index),
                    reason,
                })
            };
            match inst {
                Inst::Packet { .. } => return refuse("a packet query in a wave program"),
                Inst::Core {
                    op: Op::Env(Env::OutsideLanes | Env::PacketLaneId),
                    ..
                } => return refuse("a packet environment value in a wave program"),
                Inst::Effect { op, .. } => match op {
                    EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane) => {}
                    EffectOp::Wave(_) => {
                        return refuse("an operation that exchanges values between lanes")
                    }
                    EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                        return refuse("a workgroup barrier")
                    }
                    EffectOp::Memory {
                        space, op, semantics, ..
                    } => {
                        if *space == Space::Lds {
                            return refuse("workgroup-shared memory");
                        }
                        if *op == MemoryOp::Fence {
                            return refuse("a memory fence");
                        }
                        if semantics.ordering != Ordering::Relaxed {
                            return refuse("memory whose order across lanes is specified");
                        }
                    }
                },
                _ => {}
            }
        }
    }
    Ok(())
}

pub(crate) fn decompile(function: &LiftedFunction) -> Result<LiftedFunction, Refusal> {
    supported(&function.ir)?;
    let f = &function.ir;
    let facts = facts::Facts::new(f, &function.parameter_inputs);
    let mut logic = logic::Logic::new(f, &facts);
    let exec = function.registry.registers().exec;
    let exec_index = function.parameter_inputs.iter().position(|p| {
        matches!(p.source, crate::rdna_spmd::program::ParameterSource::MaskBit(r) if r == exec)
    });
    let exec = exec_index.map(|index| f.blocks[&f.entry].params[index].0);
    proof::prove(f, &facts, &mut logic, exec)?;
    let mut lane = rewrite::lane_program(f, &facts);
    fold::fold(&mut lane, &function.parameter_inputs, exec_index);
    lane.compact();
    if let Err(e) = lane.check(&function.registry) {
        if cfg!(test) {
            panic!("the lane program read off a proven wave program is invalid: {}", e);
        }
        return Err(Refusal {
            block: f.entry,
            index: None,
            reason: "the lane program did not verify",
        });
    }
    Ok(LiftedFunction {
        registry: function.registry.clone(),
        ir: lane,
        parameter_inputs: function.parameter_inputs.clone(),
        revision: function.revision + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, SOP1, VGLOBAL, VOP2, VOPC};
    use crate::rdna_spmd::compiler::{compile_lockstep, compile_scalar};
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
        InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_STORE_B32,
            vaddr: 0,
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
        for block in lane.ir.blocks.values() {
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
                    compile_scalar(Program { function: lane.clone() }, 256).run(
                        sgprs.as_mut_ptr(),
                        vgprs.as_mut_ptr(),
                        0,
                        0,
                    );
                } else {
                    compile_lockstep(&lane, 256, width, None).run(
                        sgprs.as_mut_ptr(),
                        vgprs.as_mut_ptr(),
                        0,
                        0,
                        u32::MAX,
                        0,
                    );
                }
            }
            let expected: Vec<u32> = (0..lanes as u32).map(|l| 7 * (l + 1)).collect();
            assert_eq!(output, expected, "width {}", width);
        }
    }

    #[test]
    fn a_store_a_lane_reaches_only_behind_other_lanes_is_refused() {
        let wave = lifted(vec![
            (
                vec![mov(20, EXEC), nonzero(0), mov(EXEC, VCC)],
                Terminator::Branch {
                    cond: Cond::ExecNz,
                    taken: 1,
                    fallthrough: 2,
                },
            ),
            (vec![mov(EXEC, 20), store(0)], Terminator::Return),
            (vec![], Terminator::Return),
        ]);
        let refusal = decompile(&wave).err().expect("lane 0 stores behind other lanes");
        assert_eq!(
            refusal.reason,
            "the wave program may store while the programs are apart"
        );
    }

    #[test]
    fn a_store_under_a_query_of_what_a_query_left_out_is_refused() {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        let ir = crate::rdna_spmd::ir::parse::func(
            &registry,
            "func entry b0
             b0(v0: i32, v1: i32, v2: i32, v3: i1):
               v4: i32 = const i32 0x0
               v5: i1 = cmp ne v0, v4
               v6: i32 = effect !p0 wave ballot (v5)
               v7: i1 = cmp ne v6, v4
               v8: i1 = const i1 0x1
               v9: i1 = int xor v5, v8
               v10: i1 = int and v7, v9
               v11: i32 = effect !p1 wave ballot (v10)
               v12: i1 = cmp ne v11, v4
               v13: i1 = int and v12, v5
               v14: i1 = int and v13, v3
               v15: i64 = pack64 v1, v2
               v16: i64 = convert zext i64 v0
               v17: i64 = int add v15, v16
               v18: i32 = const i32 0x7
               effect !p2 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v17, v18, v14)
               ret",
        )
        .unwrap();
        let input = |source, ty| Parameter { source, ty };
        let wave = LiftedFunction {
            registry: std::sync::Arc::new(registry),
            ir,
            parameter_inputs: vec![
                input(ParameterSource::Vgpr(0), Ty::I32),
                input(ParameterSource::Sgpr(0), Ty::I32),
                input(ParameterSource::Sgpr(1), Ty::I32),
                input(ParameterSource::MaskBit(126), Ty::I1),
            ],
            revision: 0,
        };
        let refusal = decompile(&wave)
            .err()
            .expect("the store follows the other lanes");
        assert_eq!(
            refusal.reason,
            "whether a store happens depends on the other lanes"
        );
    }

    #[test]
    fn a_write_into_another_lane_is_refused() {
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
        let refusal = decompile(&wave).err().expect("the write reaches lane 0");
        assert_eq!(
            refusal.reason,
            "an operation that exchanges values between lanes"
        );
    }
}
