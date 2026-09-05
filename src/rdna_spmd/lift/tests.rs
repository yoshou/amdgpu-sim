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
            Lowering::Legacy(_) => panic!("expected typed ALU"),
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
fn legacy_lds_load_redefines_a_typed_input() {
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
