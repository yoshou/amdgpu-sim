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
                    insts: expected.into_iter().map(ExprInst::from).collect(),
                    results: vec![result],
                }
            ),
            _ => panic!("expected typed ALU"),
        }
    }
}

#[test]
fn carry_operations_have_value_and_flag_ssa_results() {
    for op in [I::V_ADD_CO_CI_U32, I::V_SUB_CO_CI_U32] {
        let inst = alu(op, SourceOperand::VectorRegister(2), 8);
        let Lowering::TypedAlu { outputs, expr, scalar, .. } = instruction(&inst) else {
            panic!("carry must be lifted");
        };
        assert!(!scalar);
        assert!(matches!(outputs.as_slice(), [Output::Vgpr(8, Ty::I32), Output::Mask(106)]));
        assert_eq!(expr.expr().results.len(), 2);
    }
}

#[test]
fn normal_readfirstlane_mask_destinations_follow_ssa_updates() {
    use crate::rdna_instructions::VOP1;
    let read=|dst|InstFormat::VOP1(VOP1 {op:I::V_READFIRSTLANE_B32,
        src0:SourceOperand::VectorRegister(2),vdst:dst});
    let copy=|dst,src0|InstFormat::VOP1(VOP1 {op:I::V_MOV_B32,src0,vdst:dst});
    let store=|vsrc,ioffset|InstFormat::VGLOBAL(VGLOBAL {op:I::GLOBAL_STORE_B32,
        vaddr:0,vsrc,vdst:0,scope:0,th:0,ioffset,saddr:0,sve:0});
    let program=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([(0,ScalarBlock {pc:0,
        body:vec![read(106),copy(4,SourceOperand::ScalarRegister(106)),read(126),
            copy(3,SourceOperand::IntegerConstant(77)),
            InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst:126,ssrc0:SourceOperand::LiteralConstant(u32::MAX)}),
            store(3,0),store(4,4)],term:Terminator::Return})])};
    for width in [0,1,2,4,8,16] {
        let scalar=(width==0).then(||Compiler::default().compile_program(&program,256));
        let packet=(width!=0).then(||Compiler::default().compile_program_vec(&program,256,width));
        let w=width.max(1) as usize;let bits=(1u32<<w)-1;
        for mask in [0u32,0xa5,u32::MAX,1<<21] {
            let mut output=vec![0u32;w*2];let ptr=output.as_mut_ptr() as u64;
            let mut s=[0u32;128];s[0]=ptr as u32;s[1]=(ptr>>32) as u32;
            let mut v=vec![0u32;w*256];
            for lane in 0..w {v[lane]=(lane*8) as u32;v[2*w+lane]=mask;v[3*w+lane]=99;}
            unsafe {
                if let Some(k)=&scalar {k.run(s.as_mut_ptr(),v.as_mut_ptr(),0);}
                if let Some(k)=&packet {k.run(s.as_mut_ptr(),v.as_mut_ptr(),0,0);}
            }
            for lane in 0..w {
                assert_eq!(output[lane*2],if mask>>lane&1!=0 {77} else {99},"EXEC width={width} lane={lane}");
                assert_eq!(output[lane*2+1],mask&bits,"VCC width={width} lane={lane}");
            }
        }
    }
}

#[test]
fn scalar_comparison_drives_ssa_branch_and_merge_in_all_widths() {
    use crate::rdna_instructions::{SOPC, VOP1};
    let value = |v| InstFormat::VOP1(VOP1 { op: I::V_MOV_B32,
        src0: SourceOperand::IntegerConstant(v), vdst: 2 });
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body: vec![InstFormat::SOPC(SOPC { op: I::S_CMP_LT_U32,
            ssrc0: SourceOperand::ScalarRegister(4), ssrc1: SourceOperand::ScalarRegister(5) })],
            term: Terminator::Branch { cond: Cond::Scc1, taken: 1, fallthrough: 2 } }),
        (1, ScalarBlock { pc: 1, body: vec![value(11)], term: Terminator::Jump(3) }),
        (2, ScalarBlock { pc: 2, body: vec![value(22)], term: Terminator::Jump(3) }),
        (3, ScalarBlock { pc: 3, body: vec![InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: 2, vdst: 0, scope: 0, th: 0, ioffset: 0, saddr: 0, sve: 0 })],
            term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for (a, b, expected) in [(0u32, 1u32, 11), (u32::MAX, 0, 22), (1, 1, 22)] {
            let mut output = vec![0u32; w];
            let addr = output.as_mut_ptr() as u64;
            let mut sgprs = [0u32; 128];
            sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
            sgprs[4] = a; sgprs[5] = b;
            let mut vgprs = vec![0u32; 256 * w];
            for lane in 0..w { vgprs[lane] = lane as u32 * 4; }
            unsafe {
                if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
            }
            assert_eq!(output, vec![expected; w], "width={width} a={a} b={b}");
        }
    }
}

#[test]
fn scalar_words_survive_pair_overlap_shape_changes_and_loop_edges() {
    use crate::rdna_instructions::{SMEM, SOP2, SOPC, SOPK, VOP1};
    let sr = |r| SourceOperand::ScalarRegister(r);
    let k = |v| SourceOperand::IntegerConstant(v);
    let sop = |op, sdst, ssrc0, ssrc1| InstFormat::SOP2(SOP2 { op, sdst, ssrc0, ssrc1 });
    let mov = |sdst, ssrc0| InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, sdst, ssrc0 });
    let copy = |vdst, r| InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, vdst, src0: sr(r) });
    let mut body = vec![
        InstFormat::SMEM(SMEM { op: I::S_LOAD_B64, sbase: 2, sdata: 8, soffset: 124, ioffset: 0, scope: 0, th: 0 }),
        sop(I::S_ADD_NC_U64, 10, sr(8), k(0)),
        copy(8, 10), copy(9, 11),
        mov(9, k(0x1357_2468)),
        sop(I::S_ADD_NC_U64, 12, sr(8), k(1)),
        copy(10, 12), copy(11, 13),
        sop(I::S_ADD_U32, 14, sr(8), k(7)),
        alu(I::V_ADD_NC_U32, sr(8), 12),
        InstFormat::SOPK(SOPK { op: I::S_ADDK_I32, sdst: 8, simm16: 0xffff }),
        sop(I::S_ADD_U32, 15, sr(8), k(7)),
        alu(I::V_ADD_NC_U32, sr(8), 13),
        copy(14, 14), copy(15, 15),
        mov(20, k(0)),
    ];
    // Both halves of a scalar load and subsequent arithmetic have real SSA
    // dependencies, even though their physical LLVM views have different widths.
    for i in &body[..body.len() - 1] { let _ = instruction(i); }
    let exit = (0..9).map(|index| InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
        vaddr: 0, vsrc: 8 + index, vdst: 0, scope: 0, th: 0,
        ioffset: index as u32 * 4, saddr: 0, sve: 0 })).collect();
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body: std::mem::take(&mut body), term: Terminator::Jump(1) }),
        (1, ScalarBlock { pc: 1, body: vec![
            sop(I::S_ADD_U32, 20, sr(20), k(1)), copy(16, 20),
            InstFormat::SOPC(SOPC { op: I::S_CMP_LT_U32, ssrc0: sr(20), ssrc1: sr(6) })],
            term: Terminator::Branch { cond: Cond::Scc1, taken: 1, fallthrough: 2 } }),
        (2, ScalarBlock { pc: 2, body: exit, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for low in [0u32, 1, u32::MAX] {
            for trips in [1, 5] {
                let input = [low, 0x8765_4321u32];
                let mut output = vec![0u32; w * 9];
                let mut sgprs = [0u32; 128];
                for (r, addr) in [(0, output.as_mut_ptr() as u64), (4, input.as_ptr() as u64)] {
                    sgprs[r] = addr as u32; sgprs[r + 1] = (addr >> 32) as u32;
                }
                sgprs[6] = trips;
                let mut vgprs = vec![0u32; 256 * w];
                for lane in 0..w { vgprs[lane] = lane as u32 * 36; vgprs[3 * w + lane] = 100 + lane as u32; }
                unsafe {
                    if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                    if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                }
                for lane in 0..w {
                    let pair = (((0x1357_2468u64) << 32) | low as u64).wrapping_add(1);
                    let expected = [low, input[1], pair as u32, (pair >> 32) as u32,
                        low.wrapping_add(100 + lane as u32), low.wrapping_sub(1).wrapping_add(100 + lane as u32),
                        low.wrapping_add(7), low.wrapping_sub(1).wrapping_add(7), trips];
                    assert_eq!(&output[lane * 9..lane * 9 + 9], &expected,
                        "width={width} low={low:x} trips={trips} lane={lane}");
                }
            }
        }
    }
}

#[test]
fn word_ssa_float_views_and_scalar_packet_uses_share_current_definitions() {
    use crate::rdna_instructions::{SOP2, VOP1};
    let sr = |r| SourceOperand::ScalarRegister(r);
    let unary = |op, src0, vdst| InstFormat::VOP1(VOP1 { op, src0, vdst });
    let mut body = vec![
        unary(I::V_CVT_I32_F32, sr(7), 8),
        InstFormat::SOP2(SOP2 { op: I::S_MUL_I32, sdst: 9, ssrc0: sr(7), ssrc1: SourceOperand::IntegerConstant(2) }),
        unary(I::V_MOV_B32, sr(9), 9),
        unary(I::V_CVT_I32_F32, sr(7), 10),
        // A new scalar definition must invalidate every typed view of s7.
        InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, sdst: 7, ssrc0: SourceOperand::FloatConstant(-2.5) }),
        unary(I::V_CVT_I32_F32, sr(7), 11),
    ];
    body.extend((0..4).map(|index| InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
        vaddr: 0, vsrc: 8 + index, vdst: 0, scope: 0, th: 0, ioffset: index as u32 * 4, saddr: 0, sve: 0 })));
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for input in [1.5f32, -17.75, f32::INFINITY, f32::NAN] {
            let mut output = vec![0u32; 4 * w]; let mut sgprs = [0u32; 128];
            let address = output.as_mut_ptr() as u64; sgprs[0] = address as u32; sgprs[1] = (address >> 32) as u32;
            sgprs[7] = input.to_bits(); let mut vgprs = vec![0u32; 256 * w];
            for lane in 0..w { vgprs[lane] = lane as u32 * 16; }
            unsafe {
                if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
            }
            let expected = [input as i32 as u32, input.to_bits().wrapping_mul(2), input as i32 as u32, (-2i32) as u32];
            for lane in 0..w { assert_eq!(&output[lane * 4..lane * 4 + 4], &expected, "width={width} lane={lane}"); }
        }
    }
}

#[test]
fn null_words_are_discarded_and_mask_high_words_are_ordinary_state() {
    use crate::rdna_instructions::VOP1;
    let mov = |op, sdst, ssrc0| InstFormat::SOP1(SOP1 { op, sdst, ssrc0 });
    let sr = |r| SourceOperand::ScalarRegister(r);
    let k = |v| SourceOperand::IntegerConstant(v);
    let mut exit = vec![mov(I::S_MOV_B64, 30, sr(107)), mov(I::S_MOV_B64, 32, sr(123)),
        mov(I::S_MOV_B64, 34, sr(124))];
    for (index, reg) in [30, 31, 32, 33, 34, 35, 125, 127].iter().enumerate() {
        exit.push(InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, src0: sr(*reg), vdst: 2 }));
        exit.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: 2, vdst: 0, scope: 0, th: 0, ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body: vec![
            mov(I::S_MOV_B64, 107, k(0x1357_2468_aaaa_bbbb)),
            mov(I::S_MOV_B64, 123, k(0x5555_6666_7777_8888)),
            mov(I::S_MOV_B64, 124, k(0x1122_3344_ffff_ffff)),
            mov(I::S_MOV_B32, 127, k(0xdead_beef))], term: Terminator::Jump(1) }),
        (1, ScalarBlock { pc: 1, body: vec![mov(I::S_MOV_B32, 107, k(0xcafe_babe)),
            mov(I::S_MOV_B32, 124, k(0x1234_5678))], term: Terminator::Jump(2) }),
        (2, ScalarBlock { pc: 2, body: exit, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        let mut output = vec![0u32; 8 * w];
        let addr = output.as_mut_ptr() as u64;
        let mut sgprs = [0u32; 128];
        sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32; sgprs[124] = 0xbad0_bad0;
        let mut vgprs = vec![0u32; 256 * w];
        for lane in 0..w { vgprs[lane] = lane as u32 * 32; }
        unsafe {
            if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
            if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
        }
        for lane in 0..w {
            assert_eq!(&output[lane * 8..lane * 8 + 8],
                &[0xcafe_babe, 0x1357_2468, 0x7777_8888, 0, 0, 0x1122_3344, 0x1122_3344, 0xdead_beef],
                "width={width} lane={lane}");
        }
    }
}

#[test]
fn wide_scalar_alu_preserves_full_words_flags_and_signed_literals() {
    use crate::rdna_instructions::{SOP2, SOPC, VOP1};
    let sr = |r| SourceOperand::ScalarRegister(r);
    for op in [I::S_MUL_U64, I::S_AND_B64, I::S_OR_B64, I::S_XOR_B64,
        I::S_LSHL_B64, I::S_LSHR_B64, I::S_ASHR_I64, I::S_CSELECT_B64] {
        for literal in [false, true] {
            let mut body = vec![
                InstFormat::SOPC(SOPC { op: I::S_CMP_EQ_U32, ssrc0: sr(12), ssrc1: SourceOperand::IntegerConstant(1) }),
                InstFormat::SOP2(SOP2 { op, sdst: 4, ssrc0: if literal { SourceOperand::LiteralConstant(0xdead_beef) } else { sr(4) }, ssrc1: sr(6) }),
                InstFormat::SOP2(SOP2 { op: I::S_CSELECT_B32, sdst: 8, ssrc0: SourceOperand::IntegerConstant(1), ssrc1: SourceOperand::IntegerConstant(0) }),
            ];
            for (index, reg) in [4, 5, 8].iter().enumerate() {
                body.push(InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, src0: sr(*reg), vdst: 2 }));
                body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
                    vaddr: 0, vsrc: 2, vdst: 0, scope: 0, th: 0, ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
            }
            let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
                (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
            for width in [0, 1, 2, 4, 8, 16] {
                let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
                let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
                let w = width.max(1) as usize;
                for (a, b) in [(0u64, 0u64), (u64::MAX, 65), (1 << 63, 1), (0x7654_3210_fedc_ba98, 32),
                    (0x0123_4567_89ab_cdef, 0x8000_0000_0000_003f)] {
                    for flag in [0, 1] {
                        let mut output = vec![0u32; w * 3];
                        let mut sgprs = [0u32; 128];
                        for (r, bits) in [(0, output.as_mut_ptr() as u64), (4, a), (6, b)] {
                            sgprs[r] = bits as u32; sgprs[r + 1] = (bits >> 32) as u32;
                        }
                        sgprs[12] = flag;
                        let mut vgprs = vec![0u32; 256 * w];
                        for lane in 0..w { vgprs[lane] = lane as u32 * 12; }
                        unsafe {
                            if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                            if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                        }
                        let a = if literal { if matches!(op, I::S_ASHR_I64) { 0xffff_ffff_dead_beef } else { 0xdead_beef } } else { a };
                        let value = match op {
                            I::S_MUL_U64 => a.wrapping_mul(b), I::S_AND_B64 => a & b,
                            I::S_OR_B64 => a | b, I::S_XOR_B64 => a ^ b,
                            I::S_LSHL_B64 => a.wrapping_shl(b as u32), I::S_LSHR_B64 => a.wrapping_shr(b as u32),
                            I::S_ASHR_I64 => (a as i64).wrapping_shr(b as u32) as u64,
                            I::S_CSELECT_B64 => if flag != 0 { a } else { b }, _ => unreachable!(),
                        };
                        let flag = if matches!(op, I::S_MUL_U64 | I::S_CSELECT_B64) { flag } else { (value != 0) as u32 };
                        for lane in 0..w { assert_eq!(&output[lane * 3..lane * 3 + 3], &[value as u32, (value >> 32) as u32, flag],
                            "op={op:?} width={width} a={a:x} b={b:x} literal={literal}"); }
                    }
                }
            }
        }
    }
}

#[test]
fn vector_bit_fields_keep_word_order_and_mask_shift_counts() {
    use crate::rdna_instructions::{VOP1, VOP3};
    let mut body = vec![];
    for (index, op) in [I::V_ALIGNBIT_B32, I::V_LSHLREV_B16, I::V_LSHRREV_B16].iter().enumerate() {
        body.push(InstFormat::VOP3(VOP3 { op: *op, vdst: 8 + index as u8,
            src0: SourceOperand::VectorRegister(2), src1: SourceOperand::VectorRegister(3),
            src2: SourceOperand::VectorRegister(4), abs: 0, neg: 0, cm: 0, omod: 0, opsel: 0 }));
    }
    body.push(InstFormat::VOP1(VOP1 { op: I::V_CLZ_I32_U32, src0: SourceOperand::VectorRegister(2), vdst: 11 }));
    body.push(alu(I::V_MUL_U32_U24, SourceOperand::VectorRegister(2), 12));
    body.push(alu(I::V_MUL_I32_I24, SourceOperand::VectorRegister(2), 13));
    for index in 0..6 {
        body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: 8 + index, vdst: 0, scope: 0, th: 0,
            ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for (a, b, amount) in [(0u32, 0xffff_ffffu32, 0u32), (1, 0x7654_3210, 1),
            (0x8000_0000, 0x89ab_cdef, 31), (u32::MAX, 0x1234_5678, 32), (33, 0x8765_4321, 63), (0x0080_0001, 0x00ff_fffe, 8)] {
            let mut output = vec![0u32; w * 6];
            let addr = output.as_mut_ptr() as u64;
            let mut sgprs = [0u32; 128];
            sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
            let mut vgprs = vec![0u32; 256 * w];
            for lane in 0..w {
                vgprs[lane] = lane as u32 * 24;
                vgprs[2 * w + lane] = a; vgprs[3 * w + lane] = b; vgprs[4 * w + lane] = amount;
            }
            unsafe {
                if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
            }
            let expected = [((((a as u64) << 32) | b as u64) >> (amount & 31)) as u32,
                ((b & 0xffff) << (a & 15)) & 0xffff, (b & 0xffff) >> (a & 15),
                if a == 0 { u32::MAX } else { a.leading_zeros() },
                (a & 0x00ff_ffff).wrapping_mul(b & 0x00ff_ffff),
                (((a << 8) as i32 >> 8).wrapping_mul((b << 8) as i32 >> 8)) as u32];
            for lane in 0..w { assert_eq!(&output[lane * 6..lane * 6 + 6], &expected, "width={width} amount={amount}"); }
        }
    }
}

#[test]
fn dual_issue_reads_both_old_destinations_before_predicated_writes() {
    use crate::rdna_instructions::VOPD;
    let exec = |source| InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, ssrc0: source, sdst: 126 });
    let mut body = vec![exec(SourceOperand::ScalarRegister(4)),
        InstFormat::VOPD(VOPD { opx: I::V_DUAL_ADD_NC_U32, opy: I::V_DUAL_ADD_NC_U32,
            src0x: SourceOperand::VectorRegister(4), src0y: SourceOperand::VectorRegister(2),
            vsrc1x: 1, vsrc1y: 4, vdstx: 2, vdsty: 0, literal_constant: None }),
        exec(SourceOperand::IntegerConstant(u32::MAX as u64))];
    for (index, &reg) in [2, 1].iter().enumerate() {
        body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: reg, vdst: 0, scope: 0, th: 0,
            ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for mask in [0u32, 0xaaaa_aaaa, u32::MAX] {
            let mut output = [0u32; 64];
            for base in (0..32).step_by(w) {
                let addr = output.as_mut_ptr() as u64;
                let mut sgprs = [0u32; 128];
                sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32; sgprs[4] = mask >> base;
                let mut vgprs = vec![0u32; 256 * w];
                for lane in 0..w {
                    let id = (base + lane) as u32;
                    vgprs[lane] = id * 8;
                    vgprs[w + lane] = 0xffff_fff0 + id % 16;
                    vgprs[2 * w + lane] = id * 3;
                    vgprs[4 * w + lane] = id * 5;
                }
                unsafe {
                    if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                    if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                }
            }
            for lane in 0..32 {
                let id = lane as u32;
                let (a, b, c) = (id * 3, 0xffff_fff0 + id % 16, id * 5);
                let expected = if mask >> lane & 1 != 0 { [b.wrapping_add(c), a.wrapping_add(c)] } else { [a, b] };
                assert_eq!(&output[lane * 2..lane * 2 + 2], &expected, "width={width} mask={mask:x} lane={lane}");
            }
        }
    }
}

#[test]
fn scalar_conversions_and_bit_count_preserve_scc_in_all_widths() {
    use crate::rdna_instructions::{SOP2, SOPC, VOP1};
    for op in [I::S_CTZ_I32_B32, I::S_CVT_F32_I32, I::S_CVT_F32_U32,
        I::S_CVT_I32_F32, I::S_CVT_U32_F32, I::S_MOV_B64] {
        let mut body = vec![
            InstFormat::SOPC(SOPC { op: I::S_CMP_EQ_U32,
                ssrc0: SourceOperand::ScalarRegister(4), ssrc1: SourceOperand::ScalarRegister(4) }),
            InstFormat::SOP1(SOP1 { op, ssrc0: SourceOperand::ScalarRegister(4), sdst: 6 }),
            InstFormat::SOP2(SOP2 { op: I::S_CSELECT_B32, ssrc0: SourceOperand::IntegerConstant(1),
                ssrc1: SourceOperand::IntegerConstant(0), sdst: 8 }),
        ];
        for (index, &reg) in [6, 7, 8].iter().enumerate() {
            body.push(InstFormat::VOP1(VOP1 { op: I::V_MOV_B32,
                src0: SourceOperand::ScalarRegister(reg), vdst: 2 }));
            body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
                vaddr: 0, vsrc: 2, vdst: 0, scope: 0, th: 0,
                ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
        }
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
            (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
        for width in [0, 1, 2, 4, 8, 16] {
            let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
            let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
            let w = width.max(1) as usize;
            for a in [0u32, 1, 16, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff,
                0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x3fc0_0000, 0xbfc0_0000] {
                let mut output = vec![0u32; 3 * w];
                let addr = output.as_mut_ptr() as u64;
                let mut sgprs = [0u32; 128];
                sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
                sgprs[4] = a; sgprs[5] = 0x1234_5678; sgprs[7] = 0xabcd_ef01;
                let mut vgprs = vec![0u32; 256 * w];
                for lane in 0..w { vgprs[lane] = lane as u32 * 12; }
                unsafe {
                    if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                    if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                }
                let value = match op {
                    I::S_CTZ_I32_B32 => if a == 0 { u32::MAX } else { a.trailing_zeros() },
                    I::S_CVT_F32_I32 => (a as i32 as f32).to_bits(),
                    I::S_CVT_F32_U32 => (a as f32).to_bits(),
                    I::S_CVT_I32_F32 => f32::from_bits(a) as i32 as u32,
                    I::S_CVT_U32_F32 => f32::from_bits(a) as u32,
                    I::S_MOV_B64 => a,
                    _ => unreachable!(),
                };
                let high = if matches!(op, I::S_MOV_B64) { 0x1234_5678 } else { 0xabcd_ef01 };
                for lane in 0..w { assert_eq!(&output[lane * 3..lane * 3 + 3], &[value, high, 1],
                    "{op:?} width={width} a={a:x}"); }
            }
        }
    }
}

#[test]
fn arithmetic_flags_execute_wraparound_aliasing_and_signed_overflow() {
    use crate::rdna_instructions::{SOP2, SOPC, VOP1};
    let cases = [(0u32, 0u32), (u32::MAX, 0), (u32::MAX, 1),
        (0x7fff_ffff, 1), (0x8000_0000, 1), (0x8000_0000, u32::MAX),
        (0xffff_ffff, 32 << 16), (0x8765_4321, (31 << 16) | 2)];
    for op in [I::V_ADD_CO_CI_U32, I::V_SUB_CO_CI_U32, I::V_SUBREV_CO_CI_U32,
        I::S_ADD_U32, I::S_ADD_CO_U32, I::S_ADD_CO_CI_U32, I::S_SUB_CO_U32, I::S_SUB_CO_CI_U32, I::S_ADD_I32, I::S_SUB_CO_I32, I::S_LSHL_B32, I::S_LSHR_B32,
        I::S_BFM_B32, I::S_BFE_U32, I::S_MAX_U32] {
        let scalar_op = !matches!(op, I::V_ADD_CO_CI_U32 | I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32);
        let mut body = if scalar_op {
            vec![InstFormat::SOP2(SOP2 { op, ssrc0: SourceOperand::ScalarRegister(4),
                ssrc1: SourceOperand::ScalarRegister(5), sdst: 4 }),
                InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, src0: SourceOperand::ScalarRegister(4), vdst: 2 }),
                InstFormat::SOP2(SOP2 { op: I::S_CSELECT_B32, ssrc0: SourceOperand::IntegerConstant(1),
                    ssrc1: SourceOperand::IntegerConstant(0), sdst: 6 }),
                InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, src0: SourceOperand::ScalarRegister(6), vdst: 8 })]
        } else {
            vec![alu(op, SourceOperand::VectorRegister(2), 2),
                alu(I::V_CNDMASK_B32, SourceOperand::IntegerConstant(0), 8)]
        };
        if matches!(op, I::S_ADD_CO_CI_U32 | I::S_SUB_CO_CI_U32) {
            body.insert(0, InstFormat::SOPC(SOPC { op: I::S_CMP_EQ_U32,
                ssrc0: SourceOperand::ScalarRegister(7), ssrc1: SourceOperand::IntegerConstant(1) }));
        }
        // V_CNDMASK uses v3 as its true input; replace it only after arithmetic.
        if !scalar_op {
            body.insert(1, InstFormat::VOP1(VOP1 { op: I::V_MOV_B32,
                src0: SourceOperand::IntegerConstant(1), vdst: 3 }));
        }
        for (index, &reg) in [2, 8].iter().enumerate() {
            body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
                vaddr: 0, vsrc: reg, vdst: 0, scope: 0, th: 0,
                ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
        }
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
            (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
        for width in [0, 1, 2, 4, 8, 16] {
            let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
            let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
            let w = width.max(1) as usize;
            for &(a, b) in &cases {
                for cin in [0u32, 1] {
                    let mut output = vec![0u32; 2 * w];
                    let addr = output.as_mut_ptr() as u64;
                    let mut sgprs = [0u32; 128];
                    sgprs[0] = addr as u32;
                    sgprs[1] = (addr >> 32) as u32;
                    sgprs[4] = a;
                    sgprs[5] = b;
                    sgprs[7] = cin;
                    sgprs[106] = if cin == 0 { 0 } else { u32::MAX };
                    let mut vgprs = vec![0u32; 256 * w];
                    for lane in 0..w {
                        vgprs[lane] = lane as u32 * 8;
                        vgprs[2 * w + lane] = a;
                        vgprs[3 * w + lane] = b;
                    }
                    unsafe {
                        if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                        if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                    }
                    let (value, flag) = match op {
                        I::S_ADD_U32 | I::S_ADD_CO_U32 => a.overflowing_add(b),
                        I::S_SUB_CO_U32 => a.overflowing_sub(b),
                        I::S_ADD_I32 => { let (v, c) = (a as i32).overflowing_add(b as i32); (v as u32, c) }
                        I::S_SUB_CO_I32 => { let (v, c) = (a as i32).overflowing_sub(b as i32); (v as u32, c) }
                        I::S_LSHL_B32 => { let v = a.wrapping_shl(b); (v, v != 0) }
                        I::S_LSHR_B32 => { let v = a.wrapping_shr(b); (v, v != 0) }
                        I::S_BFM_B32 => ((1u32.wrapping_shl(a).wrapping_sub(1)).wrapping_shl(b), false),
                        I::S_BFE_U32 => {
                            let width = (b >> 16) & 127;
                            let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
                            let v = a.wrapping_shr(b) & mask; (v, v != 0)
                        }
                        I::S_MAX_U32 => (a.max(b), a > b),
                        I::V_ADD_CO_CI_U32 | I::S_ADD_CO_CI_U32 => {
                            let v = a as u64 + b as u64 + cin as u64; (v as u32, v > u32::MAX as u64)
                        }
                        _ => {
                            let (a, b) = if matches!(op, I::V_SUBREV_CO_CI_U32) { (b, a) } else { (a, b) };
                            (a.wrapping_sub(b).wrapping_sub(cin), (a as u64) < b as u64 + cin as u64)
                        }
                    };
                    for lane in 0..w {
                        assert_eq!(&output[lane * 2..lane * 2 + 2], &[value, flag as u32],
                            "{op:?} width={width} a={a:x} b={b:x} cin={cin} lane={lane}");
                    }
                }
            }
        }
    }
}

// Codegen integration test, not an ISA harness: expectations are the integer
// rules exercised here, including wraparound and shifts modulo 32. Every lane
// is observed after EXEC is restored, so inactive-destination preservation is
// checked too. Memory instructions consume the typed results in place.
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
    // A typed result feeds another typed expression and then a typed store.
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
            let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
            let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
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
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
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
                Compiler::default().compile_program(&program, 256).run(
                    sgprs.as_mut_ptr(),
                    vgprs.as_mut_ptr(),
                    0,
                );
            } else {
                Compiler::default().compile_program_vec(&program, 256, width).run(
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
    let kernel = Compiler::default().compile_cooperative(&program, 16);
    let mut sgprs = [0u32; 129];
    sgprs[126] = 1;
    let mut vgprs = [0u32; 16];
    vgprs[2] = 10;
    vgprs[3] = 2;
    let mut lds = [41u8];
    let mut spill = [0u32; 256];
    unsafe {
        assert_eq!(
            run_scalar_fiber(&kernel, sgprs.as_mut_ptr(), vgprs.as_mut_ptr(),
                0, lds.as_mut_ptr() as u64, spill.as_mut_ptr()),
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
                    let kernel = Compiler::default().compile_cooperative(&program, 16);
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
                    let kernel = Compiler::default().compile_cooperative_vec(&program, 16, width);
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
        let scalar = (width == 0).then(|| Compiler::default().compile_writeback(&program, 32));
        let packet = (width != 0).then(|| Compiler::default().compile_cooperative_vec(&program, 32, width));
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

/// Exercise native register storage as well as the emitted accesses. The scalar
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
        let kernel = Compiler::default().compile_cooperative(program, 32);
        assert_eq!(
            unsafe {
                run_scalar_fiber(&kernel, sgprs.as_mut_ptr(), vgprs.as_mut_ptr(),
                    scratch, lds, spill.as_mut_ptr())
            },
            u64::MAX
        );
    } else {
        let kernel = Compiler::default().compile_cooperative_vec(program, 32, width);
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
fn inactive_image_samples_suppress_invalid_descriptors_and_preserve_destinations() {
    use crate::rdna_instructions::VSAMPLE;
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
        pc: 0, body: vec![
            InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, sdst: 126, ssrc0: SourceOperand::ScalarRegister(20) }),
            InstFormat::VSAMPLE(VSAMPLE { op: I::IMAGE_SAMPLE_LZ, dim: 1, tfe: 0, r128: 0, d16: 0,
                a16: 0, unrm: 1, dmask: 15, vdata: 8, lwe: 0, rsrc: 0, scope: 0, th: 0,
                samp: 8, vaddr0: 0, vaddr1: 1, vaddr2: 0, vaddr3: 0 }),
        ], term: Terminator::Return,
    })]) };
    let mut storage = vec![0u8; 128 + 255];
    let offset = (256 - storage.as_ptr() as usize % 256) % 256;
    let texels = &mut storage[offset..offset + 128];
    texels[0] = 173;
    let address = texels.as_ptr() as u64;
    for width in [0, 1, 2, 4, 8, 16] {
        let lanes = width.max(1) as usize;
        for mask in [0u32, 0xaaaa_aaaa, u32::MAX] {
            let mut sgprs = [0u32; crate::rdna_spmd::emit::COOP_SGPR_BUF];
            if mask == 0 {
                // Invalid address, unsupported format, selector, and filter
                // are unobservable when every lane has EXEC clear.
                sgprs[..12].fill(u32::MAX);
            } else {
                sgprs[0] = (address >> 8) as u32;
                sgprs[1] = (address >> 40) as u32 | (5 << 17);
                sgprs[3] = 4 | (4 << 3) | (4 << 6) | (4 << 9);
            }
            sgprs[20] = mask;
            let mut vgprs = vec![0xdead_beefu32; lanes * 256];
            for lane in 0..lanes {
                let coordinate = if mask >> lane & 1 != 0 { 0 } else { f32::NAN.to_bits() };
                vgprs[lane] = coordinate; vgprs[lanes + lane] = coordinate;
            }
            run_memory_case(&program, width, &mut sgprs, &mut vgprs, 0, 0, 0);
            for lane in 0..lanes { for component in 0..4 {
                assert_eq!(vgprs[(8 + component) * lanes + lane],
                    if mask >> lane & 1 != 0 { 173 } else { 0xdead_beef },
                    "width={width} mask={mask:x} lane={lane} component={component}");
            } }
        }
    }
}

#[test]
fn memory_word_ssa_preserves_address_overlap_and_inactive_destinations() {
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
        pc: 0, body: vec![
            InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, sdst: 126, ssrc0: SourceOperand::ScalarRegister(8) }),
            // The load overwrites both words of its own address. The following
            // ALU and store must consume the loaded words, preserving old words
            // for inactive lanes without dereferencing their invalid pointers.
            InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_LOAD_B64, saddr: 124, vaddr: 0,
                vsrc: 0, vdst: 0, scope: 0, th: 0, ioffset: 0, sve: 0 }),
            InstFormat::VOP2(VOP2 { op: I::V_ADD_NC_U32, src0: SourceOperand::VectorRegister(0),
                vsrc1: 1, vdst: 4, literal_constant: None }),
            InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32, saddr: 0, vaddr: 2,
                vsrc: 4, vdst: 0, scope: 0, th: 0, ioffset: 0, sve: 0 }),
        ], term: Terminator::Return,
    })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let lanes = width.max(1) as usize;
        let data: Vec<u32> = (0..lanes * 2).map(|i| 0xffff_ffe0u32.wrapping_add(i as u32 * 3)).collect();
        for mask in [0u32, 0xaaaa_aaaa, u32::MAX] {
            let mut output = vec![0xdead_beefu32; lanes];
            let mut sgprs = [0u32; crate::rdna_spmd::emit::COOP_SGPR_BUF];
            let base = output.as_mut_ptr() as u64;
            sgprs[0] = base as u32; sgprs[1] = (base >> 32) as u32; sgprs[8] = mask;
            let mut vgprs = vec![0x0123_4567u32; lanes * 256];
            for lane in 0..lanes {
                let address = if mask >> lane & 1 != 0 { (unsafe { data.as_ptr().add(lane * 2) }) as u64 } else { 0 };
                vgprs[lane] = address as u32; vgprs[lanes + lane] = (address >> 32) as u32;
                vgprs[2 * lanes + lane] = lane as u32 * 4;
            }
            run_memory_case(&program, width, &mut sgprs, &mut vgprs, 0, 0, 0);
            for lane in 0..lanes {
                let active = mask >> lane & 1 != 0;
                assert_eq!(vgprs[lane], if active { data[lane * 2] } else { 0 }, "low width={width} lane={lane}");
                assert_eq!(vgprs[lanes + lane], if active { data[lane * 2 + 1] } else { 0 }, "high width={width} lane={lane}");
                assert_eq!(output[lane], if active { data[lane * 2].wrapping_add(data[lane * 2 + 1]) } else { 0xdead_beef }, "store width={width} lane={lane}");
            }
        }
    }
}

#[test]
fn atomic_groups_preserve_wraparound_predication_and_observed_old_values() {
    for width in [0, 1, 2, 4, 8, 16] { for returns in [false, true] {
        let mut body = vec![InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, sdst: 126, ssrc0: SourceOperand::ScalarRegister(6) }),
            InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_ATOMIC_ADD_U32, vaddr: 2, vsrc: 4, vdst: 5,
                saddr: 0, sve: 0, ioffset: 0, scope: 0, th: returns as u8 })];
        if returns { body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32, vaddr: 0, vsrc: 5, vdst: 0,
            saddr: 2, sve: 0, ioffset: 0, scope: 0, th: 0 })); }
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        for buckets in [1, 2, 7, 16] { for mask in [0u32, 1, 0xaaaa, u32::MAX] {
            let mut bins = vec![u32::MAX - 3; buckets]; let mut expected = bins.clone();
            let mut old_values = vec![0x1234_5678u32; w]; let mut expected_old = old_values.clone();
            let mut sgprs = [0u32; 128]; let mut vgprs = vec![0u32; w * 256];
            for (r, address) in [(0, bins.as_mut_ptr() as u64), (2, old_values.as_mut_ptr() as u64)] {
                sgprs[r] = address as u32; sgprs[r + 1] = (address >> 32) as u32;
            }
            sgprs[6] = mask;
            for lane in 0..w {
                let bucket = (lane * 5 + 1) % buckets;
                let value = [0, 1, u32::MAX, 17][lane % 4];
                vgprs[lane] = lane as u32 * 4;
                vgprs[2 * w + lane] = if mask >> lane & 1 != 0 { bucket as u32 * 4 } else { 0x7000_0000 };
                vgprs[4 * w + lane] = value;
                if mask >> lane & 1 != 0 {
                    expected_old[lane] = expected[bucket];
                    expected[bucket] = expected[bucket].wrapping_add(value);
                }
            }
            unsafe {
                if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
            }
            assert_eq!(bins, expected, "width={width} returns={returns} buckets={buckets} mask={mask:x}");
            if returns { assert_eq!(old_values, expected_old, "width={width} buckets={buckets} mask={mask:x}"); }
        } }
    } }
}

#[test]
fn lds_wide_and_two_address_accesses_keep_offsets_data_and_predication() {
    use crate::rdna_instructions::DS;
    for width in [0, 1, 2, 4, 8, 16] {
        let w = width.max(1) as usize;
        for (store, load, n, stride) in [
            (I::DS_STORE_B96, I::DS_LOAD_B96, 3, 0),
            (I::DS_STORE_B128, I::DS_LOAD_B128, 4, 0),
            (I::DS_STORE_2ADDR_B32, I::DS_LOAD_2ADDR_B32, 1, 4),
            (I::DS_STORE_2ADDR_B64, I::DS_LOAD_2ADDR_B64, 2, 8),
            (I::DS_STORE_2ADDR_STRIDE64_B32, I::DS_LOAD_2ADDR_STRIDE64_B32, 1, 256),
            (I::DS_STORE_2ADDR_STRIDE64_B64, I::DS_LOAD_2ADDR_STRIDE64_B64, 2, 512),
        ] {
            let count = if stride == 0 { n } else { n * 2 };
            let ds = |op, vdst| InstFormat::DS(DS { op, vdst, addr: 0, data0: 2, data1: 9,
                offset0: 3, offset1: if stride == 0 { 0 } else { 1 } });
            let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
                (0, ScalarBlock { pc: 0, body: vec![InstFormat::SOP1(SOP1 {
                    op: I::S_MOV_B32, sdst: 126, ssrc0: SourceOperand::ScalarRegister(4) }),
                    ds(store, 0), ds(load, 16)], term: Terminator::Return })]) };
            for mask in [0u32, 0xaaaa_aaaa, u32::MAX] {
                let mut sgprs = [0u32; 129]; sgprs[4] = mask & ((1u32 << w) - 1); sgprs[126] = (1u32 << w) - 1;
                let mut vgprs = vec![0u32; 256 * w];
                let mut lds = vec![0x5a5a_5a5au32; 1024 * w];
                for lane in 0..w {
                    vgprs[lane] = lane as u32 * 4096;
                    for k in 0..4 {
                        vgprs[(2 + k) * w + lane] = 0x1122_3300 + k as u32 + lane as u32 * 16;
                        vgprs[(9 + k) * w + lane] = 0x8899_aa00 + k as u32 + lane as u32 * 16;
                        vgprs[(16 + k) * w + lane] = 0xfeed_0000 + k as u32;
                    }
                }
                let before = lds.clone();
                run_memory_case(&program, width, &mut sgprs, &mut vgprs, 0, 0, lds.as_mut_ptr() as u64);
                let mut expected = before;
                for lane in 0..w {
                    for k in 0..count {
                        let active = mask >> lane & 1 != 0;
                        let data = if stride != 0 && k >= n { 0x8899_aa00 + k - n } else { 0x1122_3300 + k } + lane as u32 * 16;
                        let offset = if stride == 0 { 3 + k * 4 }
                            else { if k < n { 3 * stride + k * 4 } else { stride + (k - n) * 4 } };
                        if active {
                            let bytes = unsafe { std::slice::from_raw_parts_mut(expected.as_mut_ptr().cast::<u8>(), expected.len() * 4) };
                            let address = lane * 4096 + offset as usize;
                            bytes[address..address + 4].copy_from_slice(&data.to_ne_bytes());
                        }
                        assert_eq!(vgprs[(16 + k as usize) * w + lane], if active { data } else { 0xfeed_0000 + k },
                            "width={width} store={store:?} mask={mask:x} lane={lane} word={k}");
                    }
                }
                assert_eq!(lds, expected, "width={width} store={store:?} mask={mask:x}");
            }
        }
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
    let program = crate::rdna_spmd::CompilationInput::to_ssa(
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
    ).split(|_| true).0;
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
            let kernel = Compiler::default().compile_cooperative(&program, 16);
            dispatch_cooperative(&kernel, &kd, output.as_mut_ptr() as u64, 0, dims, 0, 0, 2);
        } else {
            let kernel = Compiler::default().compile_cooperative_vec(&program, 16, width);
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
        let kernel=Compiler::default().compile_cooperative_vec(&program,16,width);assert_eq!(kernel.min_private_bytes,4112);
        let mut output=[u32::MAX;40];
        // The IR's static frame requirement is honored even when the caller's
        // descriptor reports no private segment; padding lanes are allocated too.
        dispatch_cooperative_vec(&kernel,&kd,output.as_mut_ptr()as u64,0,dims,0,0,1);assert_eq!(output,[0;40]);
        for (scalar,offset) in [(8,0),(124,0xfffff0)] {
            let program=make(vec![InstFormat::SOP1(SOP1{op:I::S_MOV_B32,sdst:126,ssrc0:SourceOperand::IntegerConstant(0)}),scratch(scalar,offset)]);
            let kernel=Compiler::default().compile_cooperative_vec(&program,16,width);assert_eq!(kernel.min_private_bytes,0);
            let mut sgprs=[0u32;crate::rdna_spmd::emit::COOP_SGPR_BUF];sgprs[126]=u32::MAX;sgprs[8]=0x7fff_ffff;
            let mut vgprs=vec![0xdead_beefu32;256*width as usize];
            run_memory_case(&program,width,&mut sgprs,&mut vgprs,0,0,0);
            assert!(vgprs[4*width as usize..8*width as usize].iter().all(|&v|v==0xdead_beef));
        }
    }
}

#[test]
fn comparison_classes_accept_dynamic_lane_selectors_at_all_widths() {
    use crate::rdna_instructions::VOP3;
    // One exact bit-pattern witness for each of the ten ISA classes. Include
    // both signs of NaNs separately: sign does not affect their class.
    for (bits, op, patterns) in [
        (16, I::V_CMP_CLASS_F16, vec![0x7c01u64,0x7e00,0xfc00,0xbc00,0x8001,0x8000,0,1,0x3c00,0x7c00,0xfc01,0xfe00]),
        (32, I::V_CMP_CLASS_F32, vec![0x7f800001,0x7fc00000,0xff800000,0xbf800000,0x80000001,0x80000000,0,1,0x3f800000,0x7f800000,0xff800001,0xffc00000]),
        (64, I::V_CMP_CLASS_F64, vec![0x7ff0000000000001,0x7ff8000000000000,0xfff0000000000000,0xbff0000000000000,0x8000000000000001,0x8000000000000000,0,1,0x3ff0000000000000,0x7ff0000000000000,0xfff0000000000001,0xfff8000000000000]),
    ] {
        let body = vec![
            InstFormat::VOP3(VOP3 { op, vdst: 106, src0: SourceOperand::VectorRegister(4),
                src1: SourceOperand::VectorRegister(6), src2: SourceOperand::IntegerConstant(0),
                abs: 0, neg: 0, cm: 0, omod: 0, opsel: 0 }),
            alu(I::V_CNDMASK_B32, SourceOperand::IntegerConstant(0), 8),
            InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
                vaddr: 0, vsrc: 8, vdst: 0, scope: 0, th: 0, ioffset: 0, saddr: 0, sve: 0 }),
        ];
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
            (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
        for width in [0, 1, 2, 4, 8, 16] {
            let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
            let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
            let w = width.max(1) as usize;
            for offset in 0..patterns.len() {
                for mask in [0u32, 1, 0x155, 0x2aa, 0x3ff, 0xffff_fc00] {
                    let mut output = vec![u32::MAX; w];
                    let mut sgprs = [0u32; 128];
                    let addr = output.as_mut_ptr() as u64;
                    sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
                    let mut vgprs = vec![0u32; 256 * w];
                    let mut expected = vec![];
                    for lane in 0..w {
                        let index = (offset + lane) % patterns.len();
                        let pattern = patterns[index];
                        let selector = mask.rotate_left(lane as u32);
                        vgprs[lane] = lane as u32 * 4; vgprs[3 * w + lane] = 1;
                        vgprs[4 * w + lane] = pattern as u32;
                        vgprs[5 * w + lane] = (pattern >> 32) as u32;
                        vgprs[6 * w + lane] = selector;
                        expected.push((selector >> (index % 10)) & 1);
                        if bits != 16 {
                            assert_eq!(crate::rdna_spmd::dialect::rdna4::reference_class(
                                if bits == 32 { Ty::F32 } else { Ty::F64 }, pattern, selector) as u32,
                                expected[lane]);
                        }
                    }
                    unsafe {
                        if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                        if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                    }
                    assert_eq!(output, expected, "bits={bits} width={width} offset={offset} mask={mask:x}");
                }
            }
        }
    }
}

#[test]
fn comparisons_distinguish_unordered_predicates_and_signed_word_widths() {
    use crate::instructions::{OP8, OP16};
    use crate::rdna_instructions::VOP3;
    let cases = [
        (I::V_CMP_O_F32, 0x7fc00000u64, 0x3f800000u64, false),
        (I::V_CMP_U_F32, 0x7fc00000, 0x3f800000, true),
        (I::V_CMP_NLG_F32, 0x7fc00000, 0x3f800000, true),
        (I::V_CMP_LG_F64, 0x7ff8000000000000, 0, false),
        (I::V_CMP_NEQ_F64, 0x7ff8000000000000, 0, true),
        (I::V_CMP_NGE_F64, 0x7ff8000000000000, 0, true),
        (I::V_CMP_EQ_I32, u32::MAX as u64, u32::MAX as u64, true),
        (I::V_CMP_GE_I32, u32::MAX as u64, 0, false),
        (I::V_CMP_LE_I32, u32::MAX as u64, 0, true),
        (I::V_CMP_NE_I32, u32::MAX as u64, 0, true),
        (I::V_CMP_LT_I16, 0x1234ffff, 0xffff0000, true),
        (I::V_CMP_LT_U16, 0x1234ffff, 0xffff0000, false),
        (I::V_CMP_LE_U16, 0x12340001, 0xffff0001, true),
        (I::V_CMP_NE_U16, 0x12340001, 0xffff0001, false),
        (I::V_CMP_U32(OP8::F), 0, 0, false),
        (I::V_CMP_U64(OP8::TRU), 0, 0, true),
        (I::V_CMP_F32(OP16::F), 0x7fc00000, 0, false),
        (I::V_CMP_F64(OP16::TRU), 0x7ff8000000000000, 0, true),
    ];
    let mut body = vec![];
    for (index, &(op, _, _, _)) in cases.iter().enumerate() {
        // Use source registers for arbitrary 64-bit patterns and keep each
        // comparison's result as a VGPR before the next overwrites VCC.
        body.push(InstFormat::VOP3(VOP3 { op, vdst: 106,
            src0: SourceOperand::VectorRegister(16 + index as u8 * 4),
            src1: SourceOperand::VectorRegister(18 + index as u8 * 4),
            src2: SourceOperand::IntegerConstant(0), abs: 0, neg: 0, cm: 0, omod: 0, opsel: 0 }));
        body.push(alu(I::V_CNDMASK_B32, SourceOperand::IntegerConstant(0), 8));
        body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: 8, vdst: 0, scope: 0, th: 0, ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        let mut output = vec![u32::MAX; w * cases.len()];
        let mut sgprs = [0u32; 128];
        let addr = output.as_mut_ptr() as u64;
        sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
        let mut vgprs = vec![0u32; 256 * w];
        for lane in 0..w {
            vgprs[lane] = (lane * cases.len() * 4) as u32; vgprs[3 * w + lane] = 1;
            for (index, &(_, a, b, _)) in cases.iter().enumerate() {
                let reg = 16 + index * 4;
                vgprs[reg * w + lane] = a as u32; vgprs[(reg + 1) * w + lane] = (a >> 32) as u32;
                vgprs[(reg + 2) * w + lane] = b as u32; vgprs[(reg + 3) * w + lane] = (b >> 32) as u32;
            }
        }
        unsafe {
            if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
            if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
        }
        for lane in 0..w {
            for (index, &(op, _, _, expected)) in cases.iter().enumerate() {
                assert_eq!(output[lane * cases.len() + index], expected as u32, "op={op:?} width={width} lane={lane}");
            }
        }
    }
}

#[test]
fn captured_float_edges_and_integer_clamps_execute_at_all_widths() {
    use crate::rdna_instructions::VOP3;
    // Float witnesses from tests/isa/vop1.rs and tests/isa/vop3/binary.rs
    // retain the hardware bit patterns. Integer cases check saturation edges.
    let cases = [
        (I::V_RCP_F32, 0x807fffffu32, 0u32, 0, 0, 0xff800000u32),
        (I::V_RCP_F32, 0x7f7fffff, 0, 0, 0, 0),
        (I::V_RSQ_F32, 0x807fffff, 0, 0, 0, 0xff800000),
        (I::V_SQRT_F32, 1, 0, 0, 0, 0),
        (I::V_SQRT_F32, 0x807fffff, 0, 0, 0, 0x80000000),
        (I::V_SUB_F32, 0x3f800000, 0x7fc00000, 0, 0, 0xffc00000),
        (I::V_SUB_F32, 0x3f800000, 0x7fa00000, 0, 0, 0xffe00000),
        (I::V_SUBREV_F32, 0x7fc00000, 0x3f800000, 0, 0, 0xffc00000),
        (I::V_SUBREV_NC_U32, 0xdeadbeef, 3, 1, 0, 0),
        (I::V_SUB_NC_U32, 0, 1, 1, 0, 0),
        (I::V_ADD_NC_U32, u32::MAX, 1, 1, 0, u32::MAX),
        (I::V_ADD_NC_U16, 0xffff, 1, 1, 0, 0xffff),
        (I::V_MUL_F32, 0x80000000, 0x3f800000, 0, 1, 0), // -0 with OMOD -> +0
        (I::V_MUL_F32, 0x00800000, 0x3f000000, 0, 3, 0), // output subnormal -> +0
        (I::V_MUL_F32, 0x80800000, 0x3f000000, 0, 2, 0x81000000), // scaled result is normal
        (I::V_MUL_F32, 0x00000001, 0x3f800000, 0, 0, 1), // no OMOD preserves subnormal
    ];
    let mut body = vec![];
    for (index, &(op, _, _, cm, omod, _)) in cases.iter().enumerate() {
        body.push(InstFormat::VOP3(VOP3 { op, vdst: 8,
            src0: SourceOperand::VectorRegister(16 + index as u8 * 2),
            src1: SourceOperand::VectorRegister(17 + index as u8 * 2),
            src2: SourceOperand::IntegerConstant(0), abs: 0, neg: 0, cm, omod, opsel: 0 }));
        body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: 8, vdst: 0, scope: 0, th: 0, ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let scalar = (width == 0).then(|| Compiler::default().compile_program(&program, 256));
        let packet = (width != 0).then(|| Compiler::default().compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        let mut output = vec![u32::MAX; w * cases.len()];
        let mut sgprs = [0u32; 128];
        let addr = output.as_mut_ptr() as u64;
        sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
        let mut vgprs = vec![0u32; 256 * w];
        for lane in 0..w {
            vgprs[lane] = (lane * cases.len() * 4) as u32;
            for (index, &(_, a, b, _, _, _)) in cases.iter().enumerate() {
                vgprs[(16 + index * 2) * w + lane] = a;
                vgprs[(17 + index * 2) * w + lane] = b;
            }
        }
        unsafe {
            if let Some(kernel) = &scalar { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
            if let Some(kernel) = &packet { kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
        }
        for lane in 0..w {
            for (index, &(op, _, _, _, _, expected)) in cases.iter().enumerate() {
                assert_eq!(output[lane * cases.len() + index], expected, "op={op:?} width={width} lane={lane}");
            }
        }
    }
}

#[test]
fn target_math_matches_reference_for_runtime_values_in_every_lane() {
    use crate::rdna_instructions::VOP1;
    use crate::rdna_spmd::dialect::rdna4::reference;
    for (ops, bits, size) in [
        ([I::V_RCP_F32, I::V_RSQ_F32, I::V_SQRT_F32],
            vec![0, 0x80000000, 1, 0x807fffff, 0x00800000, 0x3f800000,
                0x40800000, 0xbf800000, 0x7f000000, 0x7f7fffff, 0x7f800000,
                0xff800000, 0x7fa12345, 0xffc12345], 4),
        ([I::V_RCP_F64, I::V_RSQ_F64, I::V_SQRT_F64],
            vec![0, 0x8000000000000000, 1, 0x800fffffffffffff, 0x0010000000000000,
                0x3ff0000000000000, 0x4010000000000000, 0xbff0000000000000,
                0x7fe0000000000000, 0x7fefffffffffffff, 0x7ff0000000000000,
                0xfff0000000000000, 0x7ff1234512341234, 0xfff8234512341234], 8),
    ] {
        let body = ops.iter().enumerate().flat_map(|(i, &op)| [
            InstFormat::VOP1(VOP1 { op, src0: SourceOperand::VectorRegister(4), vdst: 8 }),
            InstFormat::VGLOBAL(VGLOBAL { op: if size == 4 { I::GLOBAL_STORE_B32 } else { I::GLOBAL_STORE_B64 },
                vaddr: 0, vsrc: 8, vdst: 0, scope: 0, th: 0, ioffset: i as u32 * size, saddr: 0, sve: 0 }),
        ]).collect();
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
            (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
        for width in [0, 1, 2, 4, 8, 16] {
            let compiler = Compiler::default();
            let scalar = (width == 0).then(|| compiler.compile_program(&program, 256));
            let packet = (width != 0).then(|| compiler.compile_program_vec(&program, 256, width));
            let w = width.max(1) as usize;
            let words = size as usize / 4;
            let mut output = vec![0u32; w * 3 * words];
            let mut sgprs = [0u32; 128];
            let addr = output.as_mut_ptr() as u64;
            sgprs[0] = addr as u32; sgprs[1] = (addr >> 32) as u32;
            let mut vgprs = vec![0u32; w * 256];
            for start in (0..bits.len()).step_by(w) {
                for lane in 0..w {
                    let x = bits[(start + lane) % bits.len()];
                    vgprs[lane] = (lane * 3 * size as usize) as u32;
                    vgprs[4 * w + lane] = x as u32;
                    vgprs[5 * w + lane] = (x >> 32) as u32;
                }
                unsafe {
                    if let Some(k) = &scalar { k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
                    if let Some(k) = &packet { k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
                }
                for lane in 0..w {
                    for (i, &op) in ops.iter().enumerate() {
                        let offset = (lane * 3 + i) * words;
                        let got = output[offset] as u64 | if words == 2 { (output[offset + 1] as u64) << 32 } else { 0 };
                        let expected = reference(op, bits[(start + lane) % bits.len()]);
                        let nan = |x| if size == 4 { f32::from_bits(x as u32).is_nan() } else { f64::from_bits(x).is_nan() };
                        // Rust's reference does not specify NaN payloads. The
                        // captured hardware tests separately enforce those bits.
                        assert!(got == expected || nan(got) && nan(expected), "op={:?} width={} lane={} got={:x} expected={:x}", op, width, lane, got, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn half_conversions_select_encoded_halves_and_preserve_lane_values() {
    use crate::rdna_instructions::{VOP1, VOP3};
    let v1 = |op, src0, vdst| InstFormat::VOP1(VOP1 { op, src0, vdst });
    let v3 = |op, src0, vdst, opsel| InstFormat::VOP3(VOP3 {
        op, src0, vdst, opsel, src1: SourceOperand::IntegerConstant(0),
        src2: SourceOperand::IntegerConstant(0), abs: 0, neg: 0, cm: 0, omod: 0 });
    let cases = [
        (v1(I::V_CVT_F32_F16, SourceOperand::VectorRegister(4), 8), 8),
        (v1(I::V_CVT_F32_F16, SourceOperand::VectorRegister(132), 9), 9),
        (v3(I::V_CVT_F32_F16, SourceOperand::VectorRegister(132), 10, 1), 10),
        (v1(I::V_CVT_F16_F32, SourceOperand::FloatConstant(1.5), 11), 11),
        (v1(I::V_CVT_F16_F32, SourceOperand::FloatConstant(1.5), 140), 12),
        (v3(I::V_CVT_F16_F32, SourceOperand::FloatConstant(1.5), 141, 8), 141),
        (v1(I::V_CVT_F32_F16, SourceOperand::FloatConstant(1.5), 14), 14),
        (v3(I::V_CVT_F32_F16, SourceOperand::FloatConstant(1.5), 15, 1), 15),
    ];
    let mut body = Vec::new();
    for (index, (inst, dst)) in cases.iter().enumerate() {
        body.push(inst.clone());
        body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: *dst, vdst: 0, scope: 0, th: 0,
            ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let compiler = Compiler::default();
        let scalar = (width == 0).then(|| compiler.compile_program(&program, 256));
        let packet = (width != 0).then(|| compiler.compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        let mut output = vec![0u32; w * cases.len()];
        let mut sgprs = [0u32; 128]; let address = output.as_mut_ptr() as u64;
        sgprs[0] = address as u32; sgprs[1] = (address >> 32) as u32;
        let mut vgprs = vec![0u32; w * 256];
        for lane in 0..w {
            vgprs[lane] = (lane * cases.len() * 4) as u32;
            vgprs[4 * w + lane] = 0x4000_3c00; // high=2, low=1
            vgprs[132 * w + lane] = 0x4200_4400; // high=3, low=4
            for &(_, dst) in &cases { vgprs[dst as usize * w + lane] = 0x1234_ab00 + lane as u32; }
        }
        unsafe {
            if let Some(k) = &scalar { k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
            if let Some(k) = &packet { k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
        }
        for lane in 0..w {
            let expected = [1f32.to_bits(), 2f32.to_bits(), 3f32.to_bits(),
                0x1234_3e00, 0x3e00_ab00 + lane as u32, 0x3e00_ab00 + lane as u32,
                1.5f32.to_bits(), 0];
            assert_eq!(&output[lane * cases.len()..(lane + 1) * cases.len()], &expected,
                "width={} lane={}", width, lane);
        }
    }
}

#[test]
fn packed_rounding_cancellation_and_uniform_partial_writes() {
    use crate::rdna_instructions::VOP3P;
    let packed = |op, src0, src1, src2, vdst, opsel_hi, opsel_hi2| InstFormat::VOP3P(VOP3P {
        op, src0, src1, src2, vdst, opsel_hi, opsel_hi2, opsel: 0, neg: 0, neg_hi: 0, cm: 0,
    });
    let v = SourceOperand::VectorRegister;
    let c = SourceOperand::FloatConstant;
    let cases = vec![
        (packed(I::V_PK_FMA_F16, v(4), v(5), v(6), 10, 3, 1), 10),
        (packed(I::V_DOT2_F32_BF16, v(7), v(8), v(9), 11, 3, 1), 11),
        (packed(I::V_FMA_MIXLO_F16, c(1.5), c(2.), c(0.), 12, 0, 0), 12),
        (packed(I::V_FMA_MIXHI_F16, c(1.5), c(2.), c(0.), 13, 0, 0), 13),
    ];
    let mut body = Vec::new();
    for (index, (inst, dst)) in cases.iter().enumerate() {
        body.push(inst.clone());
        body.push(InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32,
            vaddr: 0, vsrc: *dst, vdst: 0, scope: 0, th: 0,
            ioffset: index as u32 * 4, saddr: 0, sve: 0 }));
    }
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
    for width in [0, 1, 2, 4, 8, 16] {
        let compiler = Compiler::default();
        let scalar = (width == 0).then(|| compiler.compile_program(&program, 256));
        let packet = (width != 0).then(|| compiler.compile_program_vec(&program, 256, width));
        let w = width.max(1) as usize;
        let mut output = vec![0u32; w * cases.len()];
        let mut sgprs = [0u32; 128]; let address = output.as_mut_ptr() as u64;
        sgprs[0] = address as u32; sgprs[1] = (address >> 32) as u32;
        let mut vgprs = vec![0u32; w * 256];
        for lane in 0..w {
            vgprs[lane] = (lane * cases.len() * 4) as u32;
            // 1.5*(1+2^-10)-2^-24 is just below an F16 midpoint. Rounding
            // through ordinary F32 first would incorrectly produce 0x3e02.
            vgprs[4*w+lane] = 0x3e00_3e00;
            vgprs[5*w+lane] = 0x3c01_3c01;
            vgprs[6*w+lane] = 0x8001_8001;
            // +2^100 and -2^100 cancel, leaving the lane's F32 accumulator.
            vgprs[7*w+lane] = 0xd880_5880;
            vgprs[8*w+lane] = 0x5880_5880;
            vgprs[9*w+lane] = (lane as f32 + 1.).to_bits();
            vgprs[12*w+lane] = 0x1234_ab00 + lane as u32;
            vgprs[13*w+lane] = 0x1234_ab00 + lane as u32;
        }
        unsafe {
            if let Some(k) = &scalar { k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0); }
            if let Some(k) = &packet { k.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0); }
        }
        for lane in 0..w {
            assert_eq!(&output[lane*cases.len()..(lane+1)*cases.len()], &[
                0x3e01_3e01, (lane as f32 + 1.).to_bits(), 0x1234_4200, 0x4200_ab00 + lane as u32,
            ], "width={width} lane={lane}");
        }
    }
}


unsafe fn run_scalar_fiber(kernel: &crate::rdna_spmd::emit::CoopKernel,
    sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64, lds_base: u64, spill: *mut u32,
) -> u64 {
    use crate::rdna_spmd::fiber::{Fiber, KernelArgs};
    let mut fiber = Fiber::new(32 << 10);
    fiber.start(KernelArgs { entry: kernel.addr(), sgprs, vgprs, scratch_base,
        scratch_stride: 0, lds_base, spill, lane_base: 0, valid_mask: 1 });
    fiber.resume()
}

#[test]
fn cooperative_ssa_values_survive_yield_and_accept_only_explicit_results() {
    use crate::rdna_spmd::fiber::{Fiber, KernelArgs, FIBER_DONE};
    use crate::rdna_spmd::lift::wave::{YieldAction, Operand, Destination};
    use crate::rdna_spmd::ir::typed::effect::{EffectOp, WaveOp};
    let add = |dst, a, b| InstFormat::VOP2(VOP2 { op: I::V_ADD_NC_U32,
        src0: SourceOperand::VectorRegister(a), vsrc1: b, vdst: dst, literal_constant: None });
    let action = YieldAction::new(EffectOp::Wave(WaveOp::ReadLane),
        vec![Operand::Source(SourceOperand::VectorRegister(31)), Operand::Source(SourceOperand::IntegerConstant(0))],
        vec![Destination::Vgpr(31)]);
    let p = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
        (0, ScalarBlock { pc: 0, body: vec![add(8,31,30)], term: Terminator::Yield { resume: 1, action: Box::new(action) } }),
        (1, ScalarBlock { pc: 1, body: vec![add(9,8,31)], term: Terminator::Return }),
    ]) };
    for width in [0u32,1,2,4,8,16] {
        let kernel = if width == 0 { Compiler::default().compile_cooperative(&p,256) }
            else { Compiler::default().compile_cooperative_vec(&p,256,width) };
        let w=width.max(1) as usize;
        let mut s=[0u32;129]; s[126]=(1<<w)-1; s[16]=0x13572468; s[124]=0xfeedbeef;
        let mut v=vec![0u32;256*w]; let mut spill=vec![0;256];
        for lane in 0..w { v[30*w+lane]=1; v[31*w+lane]=100+lane as u32; }
        let mut fiber=Fiber::new(32<<10);
        fiber.start(KernelArgs { entry:kernel.addr(), sgprs:s.as_mut_ptr(), vgprs:v.as_mut_ptr(),
            spill:spill.as_mut_ptr(), scratch_base:0, scratch_stride:0, lds_base:0, lane_base:0, valid_mask:s[126] });
        assert_eq!(fiber.resume(),1);
        for lane in 0..w { unsafe {
            assert_eq!(*fiber.yield_values().add(lane),100+lane as u32);
            *fiber.yield_values().add(lane)=200;
        } }
        assert_eq!(fiber.resume(),FIBER_DONE);
        for lane in 0..w {
            assert_eq!((v[8*w+lane],v[9*w+lane],v[31*w+lane]),(101+lane as u32,301+lane as u32,200));
        }
        assert_eq!((s[16],s[124]),(0x13572468,0xfeedbeef));
    }
}

mod remaining;
