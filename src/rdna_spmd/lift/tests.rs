use super::*;
use crate::rdna_instructions::{SourceOperand, SOP1, VGLOBAL, VOP2};
use crate::rdna_spmd::{Compiler, Cond, ScalarBlock, ScalarProgram, Terminator};
use std::collections::BTreeMap;

fn alu(op: I, src0: SourceOperand, dst: u8) -> InstFormat {
    InstFormat::VOP2(VOP2 {
        op,
        src0,
        vsrc1: 3,
        vdst: dst,
        literal_constant: None,
    })
}

#[test]
fn lift_keeps_reverse_operand_order_and_typed_minimum() {
    let rev = alu(I::V_SUBREV_NC_U32, SourceOperand::VectorRegister(2), 8);
    let min = alu(I::V_MIN_U32, SourceOperand::VectorRegister(2), 8);
    for (inst, expected, result) in [
        (
            &rev,
            vec![(Ty::I32, Op::Int(IntOp::Sub, ValueId(1), ValueId(0)))],
            ValueId(2),
        ),
        (
            &min,
            vec![
                (Ty::I1, Op::Cmp(IntPred::Ult, ValueId(0), ValueId(1))),
                (Ty::I32, Op::Select(ValueId(2), ValueId(0), ValueId(1))),
            ],
            ValueId(3),
        ),
    ] {
        match instruction(inst) {
            Lowering::TypedAlu { expr, .. } => assert_eq!(
                expr.expr(),
                &Expr {
                    params: vec![Ty::I32, Ty::I32],
                    insts: expected,
                    result,
                }
            ),
            _ => panic!("expected typed ALU"),
        }
    }
}

#[test]
fn carry_operations_remain_explicitly_legacy() {
    for op in [I::V_ADD_CO_CI_U32, I::V_SUB_CO_CI_U32] {
        let inst = alu(op, SourceOperand::VectorRegister(2), 8);
        assert!(matches!(instruction(&inst), Lowering::Legacy(i) if std::ptr::eq(i, &inst)));
    }
}

// Codegen integration test, not an ISA harness: expectations are the integer
// rules exercised here, including wraparound and shifts modulo 32. Every lane
// is observed after EXEC is restored, so inactive-destination preservation is
// checked too. Mixed legacy instructions consume the typed results in place.
#[test]
fn integer_lift_executes_edge_cases_and_predication_in_scalar_and_all_packet_widths() {
    let ops = [
        I::V_ADD_NC_U32,
        I::V_SUB_NC_U32,
        I::V_SUBREV_NC_U32,
        I::V_AND_B32,
        I::V_OR_B32,
        I::V_XOR_B32,
        I::V_LSHLREV_B32,
        I::V_LSHRREV_B32,
        I::V_MIN_U32,
        I::V_MAX_U32,
    ];
    let exec = |src| {
        InstFormat::SOP1(SOP1 {
            ssrc0: src,
            op: I::S_MOV_B32,
            sdst: 126,
        })
    };
    let mut body = vec![];
    for (index, op) in ops.iter().enumerate() {
        body.push(alu(*op, SourceOperand::VectorRegister(2), 8 + index as u8));
    }
    body.push(alu(
        I::V_SUBREV_NC_U32,
        SourceOperand::LiteralConstant(0xffff_ffff),
        18,
    ));
    body.push(alu(I::V_XOR_B32, SourceOperand::ScalarRegister(6), 19));
    // Destination aliases the first input; the other results already used it.
    body.push(alu(I::V_ADD_NC_U32, SourceOperand::VectorRegister(2), 2));
    // A typed result feeds another typed expression and then a legacy store.
    body.push(alu(I::V_XOR_B32, SourceOperand::VectorRegister(2), 20));
    let mut observe = vec![exec(SourceOperand::IntegerConstant(u32::MAX as u64))];
    let regs: Vec<u8> = (8..20).chain([2, 20]).collect();
    for (index, &reg) in regs.iter().enumerate() {
        observe.push(InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_STORE_B32,
            vaddr: 0,
            vsrc: reg,
            vdst: 0,
            scope: 0,
            th: 0,
            ioffset: index as u32 * 4,
            saddr: 0,
            sve: 0,
        }));
    }
    // Exercise reconvergence inside the same block: an EXEC write invalidates
    // the entry-full specialization, and the later widening observes old lanes.
    let mut single = vec![exec(SourceOperand::ScalarRegister(4))];
    single.extend(body.clone());
    single.extend(observe.clone());
    let single = ScalarProgram {
        entry_pc: 0,
        blocks: BTreeMap::from([(
            0,
            ScalarBlock {
                pc: 0,
                body: single,
                term: Terminator::Return,
            },
        )]),
    };
    let cfg = ScalarProgram {
        entry_pc: 0,
        blocks: BTreeMap::from([
            (
                0,
                ScalarBlock {
                    pc: 0,
                    body: vec![exec(SourceOperand::ScalarRegister(4))],
                    term: Terminator::Jump(1),
                },
            ),
            (
                1,
                ScalarBlock {
                    pc: 1,
                    body,
                    term: Terminator::Branch {
                        cond: Cond::Scc0,
                        taken: 2,
                        fallthrough: 3,
                    },
                },
            ),
            (
                2,
                ScalarBlock {
                    pc: 2,
                    body: vec![],
                    term: Terminator::Jump(4),
                },
            ),
            (
                3,
                ScalarBlock {
                    pc: 3,
                    body: vec![],
                    term: Terminator::Jump(4),
                },
            ),
            (
                4,
                ScalarBlock {
                    pc: 4,
                    body: observe,
                    term: Terminator::Return,
                },
            ),
        ]),
    };
    let inputs = [
        (0u32, 1u32),
        (1, u32::MAX),
        (31, 0x8000_0001),
        (32, 0x8000_0001),
        (33, 0x7654_3210),
        (63, 0xffff_fffe),
        (u32::MAX, 0x8000_0000),
        (0x8000_0000, 0),
    ];
    for program in [&single, &cfg] {
        for width in [0, 1, 2, 4, 8, 16] {
            let scalar = (width == 0).then(|| Compiler.compile_program(&program, 256));
            let packet = (width != 0).then(|| Compiler.compile_program_vec(&program, 256, width));
            let w = width.max(1) as usize;
            for mask in [0, u32::MAX, 0xaaaa_aaaa] {
                let mut output = vec![0u32; 32 * regs.len()];
                for base in (0..32).step_by(w) {
                    let mut sgprs = [0u32; 128];
                    let addr = output.as_mut_ptr() as u64;
                    sgprs[0] = addr as u32;
                    sgprs[1] = (addr >> 32) as u32;
                    sgprs[4] = mask >> base;
                    sgprs[6] = 0x1234_5678;
                    let mut vgprs = vec![0xdead_beef; 256 * w];
                    for lane in 0..w {
                        let (a, b) = inputs[(base + lane) % inputs.len()];
                        vgprs[lane] = ((base + lane) * regs.len() * 4) as u32;
                        vgprs[2 * w + lane] = a;
                        vgprs[3 * w + lane] = b;
                    }
                    unsafe {
                        if let Some(kernel) = &scalar {
                            kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0);
                        }
                        if let Some(kernel) = &packet {
                            kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0);
                        }
                    }
                }
                for lane in 0..32 {
                    let (a, b) = inputs[lane % inputs.len()];
                    let mut expected = [
                        a.wrapping_add(b),
                        a.wrapping_sub(b),
                        b.wrapping_sub(a),
                        a & b,
                        a | b,
                        a ^ b,
                        b.wrapping_shl(a),
                        b.wrapping_shr(a),
                        a.min(b),
                        a.max(b),
                        b.wrapping_sub(u32::MAX),
                        0x1234_5678 ^ b,
                        a.wrapping_add(b),
                        a.wrapping_add(b) ^ b,
                    ];
                    if (mask >> lane) & 1 == 0 {
                        expected.fill(0xdead_beef);
                        expected[12] = a;
                    }
                    assert_eq!(
                        &output[lane * regs.len()..(lane + 1) * regs.len()],
                        &expected,
                        "width={width} mask={mask:08x} lane={lane}"
                    );
                }
            }
        }
    }
}

#[test]
fn floating_sequences_conversions_comparisons_and_masks() {
    use crate::rdna_instructions::{VOP1, VOP3, VOPC};
    let exec = |s| {
        InstFormat::SOP1(SOP1 {
            ssrc0: s,
            op: I::S_MOV_B32,
            sdst: 126,
        })
    };
    let unary = |op, src, dst| {
        InstFormat::VOP1(VOP1 {
            op,
            src0: SourceOperand::VectorRegister(src),
            vdst: dst,
        })
    };
    let binary = |op, a, b, dst| {
        InstFormat::VOP2(VOP2 {
            op,
            src0: SourceOperand::VectorRegister(a),
            vsrc1: b,
            vdst: dst,
            literal_constant: None,
        })
    };
    let mut body = vec![
        exec(SourceOperand::ScalarRegister(4)),
        binary(I::V_MUL_F32, 2, 3, 32),
        binary(I::V_ADD_F32, 32, 3, 33),
        unary(I::V_CVT_I32_F32, 2, 34),
        unary(I::V_CVT_U32_F32, 2, 35),
        unary(I::V_CVT_F64_I32, 34, 36),
        binary(I::V_ADD_F64, 20, 22, 38),
        binary(I::V_MUL_F64, 38, 22, 40),
        InstFormat::VOPC(VOPC {
            op: I::V_CMP_NLT_F32,
            src0: SourceOperand::VectorRegister(2),
            vsrc1: 3,
        }),
        binary(I::V_CNDMASK_B32, 2, 3, 42),
        InstFormat::VOP3(VOP3 {
            op: I::V_FMA_F32,
            src0: SourceOperand::VectorRegister(2),
            src1: SourceOperand::VectorRegister(3),
            src2: SourceOperand::VectorRegister(33),
            vdst: 43,
            abs: 1,
            neg: 2,
            cm: 0,
            omod: 0,
            opsel: 0,
        }),
        unary(I::V_CVT_I32_F64, 20, 44),
        // A word write overlaps an f64 SSA view. The following add must reload
        // the changed pair instead of reusing the previously cached f64 value.
        unary(I::V_MOV_B32, 3, 38),
        binary(I::V_ADD_F64, 38, 22, 46),
        exec(SourceOperand::IntegerConstant(u32::MAX as u64)),
    ];
    let regs: Vec<u8> = (32..45).chain([46, 47]).collect();
    for (index, &r) in regs.iter().enumerate() {
        body.push(InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_STORE_B32,
            vaddr: 0,
            vsrc: r,
            vdst: 0,
            scope: 0,
            th: 0,
            ioffset: index as u32 * 4,
            saddr: 0,
            sve: 0,
        }));
    }
    let program = ScalarProgram {
        entry_pc: 0,
        blocks: BTreeMap::from([(
            0,
            ScalarBlock {
                pc: 0,
                body,
                term: Terminator::Return,
            },
        )]),
    };
    let cases = [
        (1.25f32, -2.5f32),
        (-0.0, 3.0),
        (f32::NAN, 1.0),
        (f32::INFINITY, 2.0),
        (-f32::INFINITY, 2.0),
        (2147483648.0, 0.5),
        (-2147483904.0, 0.5),
        (f32::from_bits(1), 2.0),
    ];
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler.compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler.compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for mask in [0u32, u32::MAX, 0xaaaa_aaaa] {
            let mut output = vec![0u32; 32 * regs.len()];
            for base in (0..32).step_by(w) {
                let mut sgprs = [0u32; 128];
                let ptr = output.as_mut_ptr() as u64;
                sgprs[0] = ptr as u32;
                sgprs[1] = (ptr >> 32) as u32;
                sgprs[4] = mask >> base;
                let mut vgprs = vec![0xdead_beef; 256 * w];
                for lane in 0..w {
                    let (a, b) = cases[(base + lane) % cases.len()];
                    vgprs[lane] = ((base + lane) * regs.len() * 4) as u32;
                    vgprs[2 * w + lane] = a.to_bits();
                    vgprs[3 * w + lane] = b.to_bits();
                    for (r, x) in [(20, a as f64), (22, b as f64)] {
                        vgprs[r * w + lane] = x.to_bits() as u32;
                        vgprs[(r + 1) * w + lane] = (x.to_bits() >> 32) as u32;
                    }
                }
                unsafe {
                    if let Some(k) = &scalar {
                        k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0);
                    }
                    if let Some(k) = &packet {
                        k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0);
                    }
                }
            }
            for lane in 0..32 {
                let row = &output[lane * regs.len()..(lane + 1) * regs.len()];
                if mask >> lane & 1 == 0 {
                    assert!(row.iter().all(|&x| x == 0xdead_beef));
                    continue;
                }
                let (a, b) = cases[lane % cases.len()];
                let f32eq = |got: u32, want: f32| {
                    if want.is_nan() {
                        assert!(f32::from_bits(got).is_nan());
                    } else {
                        assert_eq!(got, want.to_bits());
                    }
                };
                let f64eq = |lo: u32, hi: u32, want: f64| {
                    let bits = lo as u64 | ((hi as u64) << 32);
                    if want.is_nan() {
                        assert!(f64::from_bits(bits).is_nan());
                    } else {
                        assert_eq!(bits, want.to_bits());
                    }
                };
                f32eq(row[0], a * b);
                f32eq(row[1], a * b + b);
                assert_eq!(row[2], (a as i32) as u32);
                assert_eq!(row[3], a as u32);
                f64eq(row[4], row[5], (a as i32) as f64);
                let sum = a as f64 + b as f64;
                assert_eq!(row[6], b.to_bits());
                if sum.is_nan() {
                    assert!((row[7] & 0x7ff00000) == 0x7ff00000);
                } else {
                    assert_eq!(row[7], (sum.to_bits() >> 32) as u32);
                }
                f64eq(row[8], row[9], sum * b as f64);
                f32eq(row[10], if !(a < b) { b } else { a });
                f32eq(row[11], a.abs().mul_add(-b, a * b + b));
                assert_eq!(row[12], (a as f64 as i32) as u32);
                let changed =
                    f64::from_bits((sum.to_bits() & 0xffff_ffff_0000_0000) | b.to_bits() as u64);
                f64eq(row[13], row[14], changed + b as f64);
            }
        }
    }
}

#[test]
fn typed_function_loop_carries_values_through_block_arguments() {
    use crate::rdna_instructions::{SOP2, SOPC};
    let program = ScalarProgram {
        entry_pc: 0,
        blocks: BTreeMap::from([
            (
                0,
                ScalarBlock {
                    pc: 0,
                    body: vec![],
                    term: Terminator::Jump(1),
                },
            ),
            (
                1,
                ScalarBlock {
                    pc: 1,
                    body: vec![
                        alu(I::V_ADD_NC_U32, SourceOperand::VectorRegister(8), 8),
                        InstFormat::SOP2(SOP2 {
                            op: I::S_SUB_CO_I32,
                            ssrc0: SourceOperand::ScalarRegister(4),
                            ssrc1: SourceOperand::IntegerConstant(1),
                            sdst: 4,
                        }),
                        InstFormat::SOPC(SOPC {
                            op: I::S_CMP_LG_U32,
                            ssrc0: SourceOperand::ScalarRegister(4),
                            ssrc1: SourceOperand::IntegerConstant(0),
                        }),
                    ],
                    term: Terminator::Branch {
                        cond: Cond::Scc1,
                        taken: 1,
                        fallthrough: 2,
                    },
                },
            ),
            (
                2,
                ScalarBlock {
                    pc: 2,
                    body: vec![InstFormat::VGLOBAL(VGLOBAL {
                        op: I::GLOBAL_STORE_B32,
                        vaddr: 0,
                        vsrc: 8,
                        vdst: 0,
                        scope: 0,
                        th: 0,
                        ioffset: 0,
                        saddr: 0,
                        sve: 0,
                    })],
                    term: Terminator::Return,
                },
            ),
        ]),
    };
    let plan = super::super::scalar_plan::ScalarPlan::new(
        &program,
        super::super::scalar_plan::ScalarMode::Whole,
    );
    let f = plan.function.ir.func();
    let block = &f.blocks[&super::super::ir::typed::cfg::BlockId(1)];
    assert!(!block.params.is_empty());
    assert!(block
        .term
        .edges()
        .iter()
        .any(|e| e.dst.0 == 1 && e.args.iter().zip(&block.params).any(|(a, p)| *a != p.0)));
    for width in [0, 1, 2, 4, 8, 16] {
        let w = width.max(1) as usize;
        let mut output = vec![0u32; w];
        let ptr = output.as_mut_ptr() as u64;
        let mut sgprs = [0u32; 128];
        sgprs[0] = ptr as u32;
        sgprs[1] = (ptr >> 32) as u32;
        sgprs[4] = 7;
        let mut vgprs = vec![0u32; 256 * w];
        for lane in 0..w {
            vgprs[lane] = (lane * 4) as u32;
            vgprs[3 * w + lane] = (lane + 1) as u32;
            vgprs[8 * w + lane] = 9;
        }
        unsafe {
            if width == 0 {
                Compiler.compile_program(&program, 256).run(
                    sgprs.as_mut_ptr(),
                    vgprs.as_mut_ptr(),
                    0,
                );
            } else {
                Compiler.compile_program_vec(&program, 256, width).run(
                    sgprs.as_mut_ptr(),
                    vgprs.as_mut_ptr(),
                    0,
                    0,
                );
            }
        }
        assert_eq!(
            output,
            (0..w).map(|l| 9 + 7 * (l as u32 + 1)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn typed_lds_load_redefines_a_typed_input() {
    use crate::rdna_instructions::DS;
    let program = ScalarProgram {
        entry_pc: 0,
        blocks: BTreeMap::from([(
            0,
            ScalarBlock {
                pc: 0,
                body: vec![
                    alu(I::V_ADD_NC_U32, SourceOperand::VectorRegister(2), 8),
                    InstFormat::DS(DS {
                        op: I::DS_LOAD_U8,
                        addr: 0,
                        data0: 0,
                        data1: 0,
                        vdst: 2,
                        offset0: 0,
                        offset1: 0,
                    }),
                    alu(I::V_ADD_NC_U32, SourceOperand::VectorRegister(2), 9),
                ],
                term: Terminator::Return,
            },
        )]),
    };
    let kernel = Compiler.compile_cooperative(&program, 16);
    let mut sgprs = [0u32; 129];
    sgprs[126] = 1;
    let mut vgprs = [0u32; 16];
    vgprs[2] = 10;
    vgprs[3] = 2;
    let mut lds = [41u8];
    let mut spill = [0u32; 256];
    unsafe {
        assert_eq!(
            kernel.run(
                sgprs.as_mut_ptr(),
                vgprs.as_mut_ptr(),
                0,
                lds.as_mut_ptr() as u64,
                spill.as_mut_ptr(),
                0
            ),
            u64::MAX
        );
    }
    assert_eq!((vgprs[8], vgprs[9]), (12, 43));
}

#[test]
fn typed_lds_barrier_rounds_and_first_wave_execute_at_all_widths() {
    use crate::rdna_instructions::{DS, SOP2, SOPP};
    use crate::rdna_spmd::{
        dispatch_cooperative, dispatch_cooperative_vec, split_at_barriers, GridDims,
    };
    for count in [40u32, 64] {
        let mut body = vec![];
        let binary = |op, a, b, d| {
            InstFormat::VOP2(VOP2 {
                op,
                src0: a,
                vsrc1: b,
                vdst: d,
                literal_constant: None,
            })
        };
        body.push(binary(
            I::V_LSHLREV_B32,
            SourceOperand::IntegerConstant(2),
            0,
            2,
        ));
        body.push(binary(
            I::V_SUB_NC_U32,
            SourceOperand::IntegerConstant((count - 1) as u64),
            0,
            3,
        ));
        body.push(binary(
            I::V_LSHLREV_B32,
            SourceOperand::IntegerConstant(2),
            3,
            3,
        ));
        // Two rounds reuse the same ID, and every lane reads another wave's LDS.
        for add in [1u32, 101] {
            body.push(binary(
                I::V_ADD_NC_U32,
                SourceOperand::IntegerConstant(add as u64),
                0,
                4,
            ));
            body.push(InstFormat::DS(DS {
                op: I::DS_STORE_B32,
                offset0: 0,
                offset1: 0,
                addr: 2,
                data0: 4,
                data1: 0,
                vdst: 0,
            }));
            body.push(InstFormat::SOP1(SOP1 {
                op: I::S_BARRIER_SIGNAL_ISFIRST,
                ssrc0: SourceOperand::IntegerConstant(u64::MAX),
                sdst: 124,
            }));
            body.push(InstFormat::SOPP(SOPP {
                op: I::S_BARRIER_WAIT,
                simm16: u16::MAX,
            }));
            body.push(InstFormat::DS(DS {
                op: I::DS_LOAD_B32,
                offset0: 0,
                offset1: 0,
                addr: 3,
                data0: 0,
                data1: 0,
                vdst: 5,
            }));
            // Preserve the first-wave witness and store it alongside LDS data.
            body.push(InstFormat::SOP2(SOP2 {
                op: I::S_CSELECT_B32,
                ssrc0: SourceOperand::IntegerConstant(1),
                ssrc1: SourceOperand::IntegerConstant(0),
                sdst: 8,
            }));
            body.push(InstFormat::VOP1(crate::rdna_instructions::VOP1 {
                op: I::V_MOV_B32,
                src0: SourceOperand::ScalarRegister(8),
                vdst: 6,
            }));
            for (reg, offset) in [(5, 0), (6, count * 4)] {
                body.push(InstFormat::VGLOBAL(VGLOBAL {
                    op: I::GLOBAL_STORE_B32,
                    saddr: 0,
                    vaddr: 2,
                    vsrc: reg,
                    vdst: 0,
                    scope: 0,
                    th: 0,
                    ioffset: offset,
                    sve: 0,
                }));
            }
            // Wait after the reads before overwriting LDS in the next round.
            body.push(InstFormat::SOP1(SOP1 {
                op: I::S_BARRIER_SIGNAL,
                ssrc0: SourceOperand::IntegerConstant(7),
                sdst: 124,
            }));
            body.push(InstFormat::SOPP(SOPP {
                op: I::S_BARRIER_WAIT,
                simm16: 7,
            }));
        }
        let program = split_at_barriers(&ScalarProgram {
            entry_pc: 0,
            blocks: BTreeMap::from([(
                0,
                ScalarBlock {
                    pc: 0,
                    body,
                    term: Terminator::Return,
                },
            )]),
        });
        let mut kd = crate::processor::decode_kernel_desc(&[0; 64]);
        kd.enable_sgpr_kernarg_segment_ptr = true;
        let dims = GridDims {
            num_wg_x: 1,
            num_wg_y: 1,
            num_wg_z: 1,
            wg_x: count,
            wg_y: 1,
            wg_z: 1,
        };
        for width in [0, 1, 2, 4, 8, 16] {
            for threads in [1, 3] {
                let mut output = vec![u32::MAX; count as usize * 2];
                if width == 0 {
                    let kernel = Compiler.compile_cooperative(&program, 16);
                    dispatch_cooperative(
                        &kernel,
                        &kd,
                        output.as_mut_ptr() as u64,
                        0,
                        dims,
                        0,
                        256,
                        threads,
                    );
                } else {
                    let kernel = Compiler.compile_cooperative_vec(&program, 16, width);
                    dispatch_cooperative_vec(
                        &kernel,
                        &kd,
                        output.as_mut_ptr() as u64,
                        0,
                        dims,
                        0,
                        256,
                        threads,
                    );
                }
                for lane in 0..count as usize {
                    assert_eq!(
                        output[lane],
                        count - 1 - lane as u32 + 101,
                        "width={} lane={}",
                        width,
                        lane
                    );
                    assert_eq!(
                        output[count as usize + lane],
                        (lane < 32) as u32,
                        "first-wave width={} lane={}",
                        width,
                        lane
                    );
                }
            }
        }
    }
}

#[test]
fn masked_memory_never_dereferences_inactive_addresses_and_keeps_old_values() {
    use crate::rdna_spmd::fiber::{Fiber, KernelArgs, FIBER_DONE};
    for width in [0, 1, 2, 4, 8, 16] {
        let lanes = width.max(1) as usize;
        let mut body = vec![InstFormat::SOP1(SOP1 {
            op: I::S_MOV_B32,
            sdst: 126,
            ssrc0: SourceOperand::ScalarRegister(8),
        })];
        // Three adjacent pair loads exercise the transpose cluster, including
        // a negative immediate offset and an entirely empty EXEC mask.
        for (dst, offset) in [(4, -16i32), (8, 0), (12, 16)] {
            body.push(InstFormat::VGLOBAL(VGLOBAL {
                op: I::GLOBAL_LOAD_B128,
                saddr: 124,
                vaddr: 0,
                vsrc: 0,
                vdst: dst,
                scope: 0,
                th: 0,
                ioffset: offset as u32 & 0xffffff,
                sve: 0,
            }));
        }
        let program = ScalarProgram {
            entry_pc: 0,
            blocks: BTreeMap::from([(
                0,
                ScalarBlock {
                    pc: 0,
                    body,
                    term: Terminator::Return,
                },
            )]),
        };
        let scalar = (width == 0).then(|| Compiler.compile_writeback(&program, 32));
        let packet = (width != 0).then(|| Compiler.compile_cooperative_vec(&program, 32, width));
        let data: Vec<u32> = (0..lanes * 12).map(|x| x as u32 * 17 + 3).collect();
        for mask in [0u32, 0xaaaa_aaaa, u32::MAX] {
            let mut sgprs = [0u32; crate::rdna_spmd::emit::COOP_SGPR_BUF];
            sgprs[8] = mask;
            sgprs[126] = 1;
            let mut vgprs = vec![0xdead_beefu32; 256 * lanes];
            for lane in 0..lanes {
                let addr = if mask >> lane & 1 != 0 {
                    (unsafe { data.as_ptr().add(lane * 12 + 4) }) as u64
                } else {
                    0
                };
                vgprs[lane] = addr as u32;
                vgprs[lanes + lane] = (addr >> 32) as u32;
            }
            if let Some(kernel) = &scalar {
                unsafe {
                    kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0);
                }
            }
            if let Some(kernel) = &packet {
                let mut spill = [0u32; crate::rdna_spmd::emit::COOP_SPILL_SLOTS];
                let mut fiber = Fiber::new(32 << 10);
                fiber.start(KernelArgs {
                    valid_mask: u32::MAX,
                    entry: kernel.addr(),
                    sgprs: sgprs.as_mut_ptr(),
                    vgprs: vgprs.as_mut_ptr(),
                    spill: spill.as_mut_ptr(),
                    scratch_base: 0,
                    scratch_stride: 0,
                    lane_base: 0,
                    lds_base: 0,
                });
                assert_eq!(fiber.resume(), FIBER_DONE);
            }
            for lane in 0..lanes {
                for k in 0..12 {
                    assert_eq!(
                        vgprs[(4 + k) * lanes + lane],
                        if mask >> lane & 1 != 0 {
                            data[lane * 12 + k]
                        } else {
                            0xdead_beef
                        },
                        "width={} lane={} word={}",
                        width,
                        lane,
                        k
                    );
                }
            }
        }
    }
}

/// Exercise the register adapter as well as the emitted accesses. The scalar
/// and packet paths both expose their final register state in cooperative mode.
fn run_memory_case(
    program: &ScalarProgram,
    width: u32,
    sgprs: &mut [u32],
    vgprs: &mut [u32],
    scratch: u64,
    stride: u64,
    lds: u64,
) {
    use crate::rdna_spmd::fiber::{Fiber, KernelArgs, FIBER_DONE};
    let mut spill = [0u32; crate::rdna_spmd::emit::COOP_SPILL_SLOTS];
    if width == 0 {
        let kernel = Compiler.compile_cooperative(program, 32);
        assert_eq!(
            unsafe {
                kernel.run(
                    sgprs.as_mut_ptr(),
                    vgprs.as_mut_ptr(),
                    scratch,
                    lds,
                    spill.as_mut_ptr(),
                    0,
                )
            },
            u64::MAX
        );
    } else {
        let kernel = Compiler.compile_cooperative_vec(program, 32, width);
        let mut fiber = Fiber::new(32 << 10);
        fiber.start(KernelArgs {
            valid_mask: u32::MAX,
            entry: kernel.addr(),
            sgprs: sgprs.as_mut_ptr(),
            vgprs: vgprs.as_mut_ptr(),
            spill: spill.as_mut_ptr(),
            scratch_base: scratch,
            scratch_stride: stride,
            lane_base: 0,
            lds_base: lds,
        });
        assert_eq!(fiber.resume(), FIBER_DONE);
    }
}

#[test]
fn typed_memory_subwords_flat_aperture_and_atomic_returns() {
    use crate::rdna_instructions::{SMEM, VFLAT, VSCRATCH};
    let make = |body| ScalarProgram {
        entry_pc: 0,
        blocks: BTreeMap::from([(
            0,
            ScalarBlock {
                pc: 0,
                body,
                term: Terminator::Return,
            },
        )]),
    };
    let set_exec = || {
        InstFormat::SOP1(SOP1 {
            op: I::S_MOV_B32,
            sdst: 126,
            ssrc0: SourceOperand::ScalarRegister(8),
        })
    };
    for width in [0, 1, 2, 4, 8, 16] {
        let w = width.max(1) as usize;
        let mut sgprs = [0u32; crate::rdna_spmd::emit::COOP_SGPR_BUF];
        sgprs[126] = 1;
        sgprs[8] = u32::MAX;
        let mut vgprs = vec![0u32; 256 * w];
        let mut private = vec![0u32; 32 * w];
        let base = private.as_mut_ptr() as u64;
        let global = vec![0xface_80f1u32; w];
        for lane in 0..w {
            private[lane * 32 + 2] = 0xbeef_91e2;
            // Even lanes use the flat private aperture, odd lanes global memory.
            let address = if lane % 2 == 0 {
                base + 8
            } else {
                (unsafe { global.as_ptr().add(lane) }) as u64
            };
            vgprs[lane] = address as u32;
            vgprs[w + lane] = (address >> 32) as u32;
            vgprs[6 * w + lane] = 0x1234_5678;
        }
        let flat = |op, dst, offset| {
            InstFormat::VFLAT(VFLAT {
                op,
                saddr: 124,
                vaddr: 0,
                vsrc: 6,
                vdst: dst,
                scope: 0,
                th: 0,
                ioffset: offset,
                sve: 0,
            })
        };
        let scratch = |op, dst, offset| {
            InstFormat::VSCRATCH(VSCRATCH {
                op,
                saddr: 124,
                vaddr: 0,
                vsrc: 6,
                vdst: dst,
                scope: 0,
                th: 0,
                ioffset: offset,
                sve: 0,
            })
        };
        let program = make(vec![
            set_exec(),
            flat(I::FLAT_LOAD_I8, 2, 0),
            flat(I::FLAT_LOAD_I16, 3, 0),
            scratch(I::SCRATCH_STORE_B16, 0, 12),
            scratch(I::SCRATCH_LOAD_U16, 4, 12),
            scratch(I::SCRATCH_LOAD_I16, 5, 8),
        ]);
        run_memory_case(&program, width, &mut sgprs, &mut vgprs, base, 128, 0);
        for lane in 0..w {
            let value = if lane % 2 == 0 {
                0xbeef_91e2u32
            } else {
                global[lane]
            };
            assert_eq!(vgprs[2 * w + lane], value as u8 as i8 as i32 as u32);
            assert_eq!(vgprs[3 * w + lane], value as u16 as i16 as i32 as u32);
            assert_eq!(vgprs[4 * w + lane], 0x5678);
            assert_eq!(vgprs[5 * w + lane], 0xffff_91e2);
            assert_eq!(private[lane * 32 + 3], 0x5678);
        }
        // SOFFSET is included before the signed 24-bit immediate is applied.
        let words = [31u32, 47, 83, 101];
        let addr = words.as_ptr() as u64;
        sgprs[0] = addr as u32;
        sgprs[1] = (addr >> 32) as u32;
        sgprs[9] = 8;
        let program = make(vec![InstFormat::SMEM(SMEM {
            op: I::S_LOAD_B32,
            sbase: 0,
            sdata: 10,
            soffset: 9,
            ioffset: 0xfffffc,
            scope: 0,
            th: 0,
        })]);
        run_memory_case(&program, width, &mut sgprs, &mut vgprs, base, 128, 0);
        assert_eq!(sgprs[10], 47);
        for mask in [0u32, 0xaaaa_aaaa, u32::MAX] {
            let mut total = 19u32;
            let addr = &mut total as *mut u32 as u64;
            sgprs[0] = addr as u32;
            sgprs[1] = (addr >> 32) as u32;
            sgprs[8] = mask;
            for lane in 0..w {
                vgprs[lane] = 0;
                vgprs[2 * w + lane] = 1;
                vgprs[3 * w + lane] = 0xdead_beef;
            }
            let program = make(vec![
                set_exec(),
                InstFormat::VGLOBAL(VGLOBAL {
                    op: I::GLOBAL_ATOMIC_ADD_U32,
                    saddr: 0,
                    vaddr: 0,
                    vsrc: 2,
                    vdst: 3,
                    scope: 3,
                    th: 1,
                    ioffset: 0,
                    sve: 0,
                }),
            ]);
            run_memory_case(&program, width, &mut sgprs, &mut vgprs, base, 128, 0);
            let active = (0..w).filter(|&l| mask >> l & 1 != 0).count();
            assert_eq!(total, 19 + active as u32);
            let mut returns = vec![];
            for lane in 0..w {
                if mask >> lane & 1 != 0 {
                    returns.push(vgprs[3 * w + lane]);
                } else {
                    assert_eq!(vgprs[3 * w + lane], 0xdead_beef);
                }
            }
            returns.sort();
            assert_eq!(returns, (19..19 + active as u32).collect::<Vec<_>>());
        }
    }
}

#[test]
fn mixed_wave_memory_yields_preserve_full_wave_values_and_partial_waves() {
    use crate::rdna_instructions::{DS, VOP1, VOP3, VOPC};
    use crate::rdna_spmd::{dispatch_cooperative, dispatch_cooperative_vec, GridDims};
    let mov = |src0, vdst| {
        InstFormat::VOP1(VOP1 {
            op: I::V_MOV_B32,
            src0,
            vdst,
        })
    };
    let binary = |op, src0, vsrc1, vdst| {
        InstFormat::VOP2(VOP2 {
            op,
            src0,
            vsrc1,
            vdst,
            literal_constant: None,
        })
    };
    let lane = |op, src0, src1, vdst| {
        InstFormat::VOP3(VOP3 {
            op,
            src0,
            src1,
            src2: SourceOperand::IntegerConstant(0),
            vdst,
            abs: 0,
            neg: 0,
            cm: 0,
            omod: 0,
            opsel: 0,
        })
    };
    let count = 40u32;
    let mut body = vec![
        binary(I::V_ADD_NC_U32, SourceOperand::IntegerConstant(100), 0, 1),
        binary(I::V_AND_B32, SourceOperand::IntegerConstant(31), 0, 6),
        InstFormat::VOPC(VOPC {
            op: I::V_CMPX_EQ_U32,
            src0: SourceOperand::IntegerConstant(5),
            vsrc1: 6,
        }),
        InstFormat::VOP1(VOP1 {
            op: I::V_READFIRSTLANE_B32,
            src0: SourceOperand::VectorRegister(1),
            vdst: 10,
        }),
        InstFormat::SOP1(SOP1 {
            op: I::S_MOV_B32,
            ssrc0: SourceOperand::IntegerConstant(u64::MAX),
            sdst: 126,
        }),
        mov(SourceOperand::ScalarRegister(10), 2),
        lane(
            I::V_WRITELANE_B32,
            SourceOperand::IntegerConstant(777),
            SourceOperand::IntegerConstant(21),
            1,
        ),
        lane(
            I::V_READLANE_B32,
            SourceOperand::VectorRegister(1),
            SourceOperand::IntegerConstant(21),
            11,
        ),
        mov(SourceOperand::ScalarRegister(11), 3),
        binary(I::V_SUB_NC_U32, SourceOperand::IntegerConstant(31), 6, 7),
        binary(I::V_LSHLREV_B32, SourceOperand::IntegerConstant(2), 7, 7),
        InstFormat::DS(DS {
            op: I::DS_BPERMUTE_FI_B32,
            addr: 7,
            data0: 1,
            data1: 0,
            vdst: 4,
            offset0: 0,
            offset1: 0,
        }),
        binary(I::V_LSHLREV_B32, SourceOperand::IntegerConstant(2), 0, 6),
    ];
    for (k, reg) in [1u8, 2, 3, 4].iter().enumerate() {
        body.push(InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_STORE_B32,
            saddr: 0,
            vaddr: 6,
            vsrc: *reg,
            vdst: 0,
            scope: 0,
            th: 0,
            ioffset: k as u32 * count * 4,
            sve: 0,
        }));
    }
    let program = super::wave::split(
        &ScalarProgram {
            entry_pc: 0,
            blocks: BTreeMap::from([(
                0,
                ScalarBlock {
                    pc: 0,
                    body,
                    term: Terminator::Return,
                },
            )]),
        },
        |_| true,
    )
    .0;
    let mut kd = crate::processor::decode_kernel_desc(&[0; 64]);
    kd.enable_sgpr_kernarg_segment_ptr = true;
    let dims = GridDims {
        num_wg_x: 1,
        num_wg_y: 1,
        num_wg_z: 1,
        wg_x: count,
        wg_y: 1,
        wg_z: 1,
    };
    for width in [0, 1, 2, 4, 8, 16] {
        let mut output = vec![u32::MAX; count as usize * 4];
        if width == 0 {
            let kernel = Compiler.compile_cooperative(&program, 16);
            dispatch_cooperative(&kernel, &kd, output.as_mut_ptr() as u64, 0, dims, 0, 0, 2);
        } else {
            let kernel = Compiler.compile_cooperative_vec(&program, 16, width);
            dispatch_cooperative_vec(&kernel, &kd, output.as_mut_ptr() as u64, 0, dims, 0, 0, 2);
        }
        for id in 0..count as usize {
            let wave = id / 32;
            let src = wave * 32 + 31 - id % 32;
            for (k, expected) in [
                if id == 21 { 777 } else { 100 + id as u32 },
                105 + wave as u32 * 32,
                if wave == 0 { 777 } else { 0 },
                if src >= count as usize {
                    0
                } else if src == 21 {
                    777
                } else {
                    100 + src as u32
                },
            ]
            .iter()
            .enumerate()
            {
                assert_eq!(
                    output[k * count as usize + id],
                    *expected,
                    "width={} lane={} result={}",
                    width,
                    id,
                    k
                );
            }
        }
    }
}

#[test]
fn static_private_loads_reserve_their_cells_and_dynamic_inactive_loads_are_masked(){
    use crate::rdna_instructions::VSCRATCH;
    use crate::rdna_spmd::{dispatch_cooperative_vec,GridDims};
    let scratch=|scalar,offset|InstFormat::VSCRATCH(VSCRATCH{op:I::SCRATCH_LOAD_B128,saddr:scalar,vaddr:0,vsrc:0,vdst:4,scope:0,th:0,ioffset:offset,sve:0});
    let make=|body|ScalarProgram{entry_pc:0,blocks:BTreeMap::from([(0,ScalarBlock{pc:0,body,term:Terminator::Return})])};
    let mut kd=crate::processor::decode_kernel_desc(&[0;64]);kd.enable_sgpr_kernarg_segment_ptr=true;
    let dims=GridDims{num_wg_x:1,num_wg_y:1,num_wg_z:1,wg_x:40,wg_y:1,wg_z:1};
    let program=make(vec![scratch(124,4096),InstFormat::VOP2(VOP2{op:I::V_LSHLREV_B32,src0:SourceOperand::IntegerConstant(2),vsrc1:0,vdst:2,literal_constant:None}),InstFormat::VGLOBAL(VGLOBAL{op:I::GLOBAL_STORE_B32,saddr:0,vaddr:2,vsrc:4,vdst:0,scope:0,th:0,ioffset:0,sve:0})]);
    for width in [1,2,4,8,16] {
        let kernel=Compiler.compile_cooperative_vec(&program,16,width);assert_eq!(kernel.min_private_bytes,4112);
        let mut output=[u32::MAX;40];
        // The IR's static frame requirement is honored even when the caller's
        // descriptor reports no private segment; padding lanes are allocated too.
        dispatch_cooperative_vec(&kernel,&kd,output.as_mut_ptr()as u64,0,dims,0,0,1);assert_eq!(output,[0;40]);
        for (scalar,offset) in [(8,0),(124,0xfffff0)] {
            let program=make(vec![InstFormat::SOP1(SOP1{op:I::S_MOV_B32,sdst:126,ssrc0:SourceOperand::IntegerConstant(0)}),scratch(scalar,offset)]);
            let kernel=Compiler.compile_cooperative_vec(&program,16,width);assert_eq!(kernel.min_private_bytes,0);
            let mut sgprs=[0u32;crate::rdna_spmd::emit::COOP_SGPR_BUF];sgprs[126]=u32::MAX;sgprs[8]=0x7fff_ffff;
            let mut vgprs=vec![0xdead_beefu32;256*width as usize];
            run_memory_case(&program,width,&mut sgprs,&mut vgprs,0,0,0);
            assert!(vgprs[4*width as usize..8*width as usize].iter().all(|&v|v==0xdead_beef));
        }
    }
}
