use std::collections::BTreeMap;
use super::*;
use crate::rdna_instructions::{SourceOperand, SOP1, VGLOBAL};
use crate::rdna_spmd::{Compiler, Cond, ScalarBlock, ScalarProgram, Terminator};

fn alu(op: I, src0: SourceOperand, dst: u8) -> InstFormat {
    InstFormat::VOP2(VOP2 { op, src0, vsrc1: 3, vdst: dst, literal_constant: None })
}

#[test]
fn lift_keeps_reverse_operand_order_and_typed_minimum() {
    let rev = alu(I::V_SUBREV_NC_U32, SourceOperand::VectorRegister(2), 8);
    let min = alu(I::V_MIN_U32, SourceOperand::VectorRegister(2), 8);
    for (inst, expected, result) in [
        (&rev, vec![(Ty::I32, Op::Int(IntOp::Sub, ValueId(1), ValueId(0)))], ValueId(2)),
        (&min, vec![
            (Ty::I1, Op::Cmp(IntPred::Ult, ValueId(0), ValueId(1))),
            (Ty::I32, Op::Select(ValueId(2), ValueId(0), ValueId(1))),
        ], ValueId(3)),
    ] {
        match instruction(inst) {
            Lowering::TypedAlu { expr, .. } => assert_eq!(expr.expr(), &Expr {
                params: vec![Ty::I32, Ty::I32], insts: expected, result,
            }),
            Lowering::Legacy(_) => panic!("expected typed ALU"),
        }
    }
}

#[test]
fn carry_mask_and_float_operations_remain_explicitly_legacy() {
    for op in [I::V_ADD_CO_CI_U32, I::V_CNDMASK_B32, I::V_MUL_F64] {
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
    let ops = [I::V_ADD_NC_U32, I::V_SUB_NC_U32, I::V_SUBREV_NC_U32,
        I::V_AND_B32, I::V_OR_B32, I::V_XOR_B32,
        I::V_LSHLREV_B32, I::V_LSHRREV_B32, I::V_MIN_U32, I::V_MAX_U32];
    let exec = |src| InstFormat::SOP1(SOP1 { ssrc0: src, op: I::S_MOV_B32, sdst: 126 });
    let mut body = vec![];
    for (index, op) in ops.iter().enumerate() {
        body.push(alu(*op, SourceOperand::VectorRegister(2), 8 + index as u8));
    }
    body.push(alu(I::V_SUBREV_NC_U32, SourceOperand::LiteralConstant(0xffff_ffff), 18));
    body.push(alu(I::V_XOR_B32, SourceOperand::ScalarRegister(6), 19));
    // Destination aliases the first input; the other results already used it.
    body.push(alu(I::V_ADD_NC_U32, SourceOperand::VectorRegister(2), 2));
    // A typed result feeds another typed expression and then a legacy store.
    body.push(alu(I::V_XOR_B32, SourceOperand::VectorRegister(2), 20));
    let mut observe = vec![exec(SourceOperand::IntegerConstant(u32::MAX as u64))];
    let regs: Vec<u8> = (8..20).chain([2, 20]).collect();
    for (index, &reg) in regs.iter().enumerate() {
        observe.push(InstFormat::VGLOBAL(VGLOBAL {
            op: I::GLOBAL_STORE_B32, vaddr: 0, vsrc: reg, vdst: 0,
            scope: 0, th: 0, ioffset: index as u32 * 4, saddr: 0, sve: 0,
        }));
    }
    // The legacy full-EXEC block specialization requires EXEC changes to
    // precede the arithmetic block; reconvergence makes old destinations live.
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body: vec![exec(SourceOperand::ScalarRegister(4))],
            term: Terminator::Jump(1) }),
        (1, ScalarBlock { pc: 1, body, term: Terminator::Branch {
            cond: Cond::Scc0, taken: 2, fallthrough: 3,
        } }),
        (2, ScalarBlock { pc: 2, body: vec![], term: Terminator::Jump(4) }),
        (3, ScalarBlock { pc: 3, body: vec![], term: Terminator::Jump(4) }),
        (4, ScalarBlock { pc: 4, body: observe, term: Terminator::Return }),
    ]) };
    let inputs = [(0u32, 1u32), (1, u32::MAX), (31, 0x8000_0001),
        (32, 0x8000_0001), (33, 0x7654_3210), (63, 0xffff_fffe),
        (u32::MAX, 0x8000_0000), (0x8000_0000, 0)];
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
                    if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                    if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                }
            }
            for lane in 0..32 {
                let (a, b) = inputs[lane % inputs.len()];
                let mut expected = [a.wrapping_add(b), a.wrapping_sub(b), b.wrapping_sub(a),
                    a & b, a | b, a ^ b, b.wrapping_shl(a), b.wrapping_shr(a),
                    a.min(b), a.max(b), b.wrapping_sub(u32::MAX), 0x1234_5678 ^ b,
                    a.wrapping_add(b), a.wrapping_add(b) ^ b];
                if (mask >> lane) & 1 == 0 {
                    expected.fill(0xdead_beef);
                    expected[12] = a;
                }
                assert_eq!(&output[lane * regs.len()..(lane + 1) * regs.len()], &expected,
                    "width={width} mask={mask:08x} lane={lane}");
            }
        }
    }
}
