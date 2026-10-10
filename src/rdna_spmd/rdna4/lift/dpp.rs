use super::wave::{Destination, Operand, YieldAction};
use super::*;
use crate::rdna_instructions::DPP;
use crate::rdna_spmd::ir::{Env, WaveOp};

pub const INDEX: u32 = 253;
pub const VALUE: u32 = 254;
pub const ENABLE: u32 = 255;
pub const TEMPORARIES: [u32; 3] = [INDEX, VALUE, ENABLE];

fn source_operand(inner: &InstFormat) -> &SourceOperand {
    match inner {
        InstFormat::VOP1(i) => &i.src0,
        InstFormat::VOP2(i) => &i.src0,
        InstFormat::VOPC(i) => &i.src0,
        InstFormat::VOP3(i) => &i.src0,
        InstFormat::VOP3SD(i) => &i.src0,
        other => panic!("DPP wraps a format it cannot modify: {:?}", other),
    }
}

fn source_register(inner: &InstFormat) -> u32 {
    match source_operand(inner) {
        SourceOperand::VectorRegister(r) => *r as u32,
        other => panic!("DPP reads a source other than a VGPR: {:?}", other),
    }
}

fn with_source(inner: &InstFormat, register: u32, abs: u8, neg: u8) -> InstFormat {
    let replaced = SourceOperand::VectorRegister(register as u8);
    let modified = abs | neg != 0;
    match inner.clone() {
        InstFormat::VOP1(i) if modified => InstFormat::VOP3(crate::rdna_instructions::VOP3 {
            vdst: i.vdst,
            abs,
            opsel: 0,
            cm: 0,
            op: i.op,
            src0: replaced,
            src1: SourceOperand::IntegerConstant(0),
            src2: SourceOperand::IntegerConstant(0),
            omod: 0,
            neg,
        }),
        InstFormat::VOP2(i) if modified => InstFormat::VOP3(crate::rdna_instructions::VOP3 {
            vdst: i.vdst,
            abs,
            opsel: 0,
            cm: 0,
            op: i.op,
            src0: replaced,
            src1: SourceOperand::VectorRegister(i.vsrc1),
            src2: SourceOperand::IntegerConstant(0),
            omod: 0,
            neg,
        }),
        InstFormat::VOP1(mut i) => {
            i.src0 = replaced;
            InstFormat::VOP1(i)
        }
        InstFormat::VOP2(mut i) => {
            i.src0 = replaced;
            InstFormat::VOP2(i)
        }
        InstFormat::VOPC(mut i) => {
            i.src0 = replaced;
            InstFormat::VOPC(i)
        }
        InstFormat::VOP3(mut i) => {
            i.src0 = replaced;
            i.abs |= abs;
            i.neg |= neg;
            InstFormat::VOP3(i)
        }
        InstFormat::VOP3SD(mut i) => {
            i.src0 = replaced;
            i.neg |= neg;
            InstFormat::VOP3SD(i)
        }
        other => panic!("DPP wraps a format it cannot modify: {:?}", other),
    }
}

pub fn gated(lowering: Lowering, enable: u32, registry: &DialectRegistry) -> Lowering {
    use crate::rdna_spmd::ir::ExprInst;
    let Lowering::TypedAlu {
        inputs,
        outputs,
        scalar,
        expr,
    } = lowering
    else {
        panic!("DPP gates a lowering that is not an ALU expression")
    };
    assert!(!scalar, "DPP gates a scalar result");
    let expr = expr.expr().clone();
    let n = inputs.len();
    let mut all = inputs;
    let mut olds = Vec::new();
    for output in &outputs {
        match *output {
            Output::Vgpr(r, ty) => {
                assert_eq!(ty.bits(), 32, "DPP gates a result wider than a word");
                all.push(input(SourceOperand::VectorRegister(r as u8), Ty::I32));
            }
            Output::Mask(r) | Output::Compare(r) => all.push(Input {
                source: InputSource::MaskBit(r),
                ty: Ty::I1,
            }),
            other => panic!("DPP gates a result it cannot keep: {:?}", other),
        }
        olds.push(ValueId(all.len() - 1));
    }
    all.push(input(SourceOperand::VectorRegister(enable as u8), Ty::I32));
    let en = ValueId(all.len() - 1);
    let shift = all.len() - n;
    let remap = |v: ValueId| if v.0 < n { v } else { ValueId(v.0 + shift) };
    let mut q = Builder::new(registry, all);
    for inst in expr.insts {
        match inst {
            ExprInst::Core(ty, op) => {
                q.push(ty, op.map(remap));
            }
            ExprInst::Target { op, args, .. } => {
                q.target(op, args.map(remap));
            }
        }
    }
    let zero = q.k(Ty::I32, 0);
    let on = q.push(Ty::I1, Op::Cmp(IntPred::Ne, en, zero));
    let mut results = Vec::new();
    for ((output, result), old) in outputs.iter().zip(expr.results).zip(olds) {
        let ty = output.ty();
        let old = if matches!(ty, Ty::I32 | Ty::I1) {
            old
        } else {
            q.push(ty, Op::Convert(Cvt::Bitcast, ty, old))
        };
        let kept = q.push(ty, Op::Select(on, remap(result), old));
        results.push((*output, kept));
    }
    q.finish_many(false, results)
}

fn selection(q: &mut Builder, d: &DPP, lanes: u32) -> (ValueId, ValueId) {
    let lane = q.push(Ty::I32, Op::Env(Env::LaneId));
    let k = |q: &mut Builder, v: u64| q.k(Ty::I32, v);
    let fifteen = k(q, 15);
    let column = q.int(IntOp::And, lane, fifteen);
    let row_base = q.int(IntOp::Xor, lane, column);
    let truth = q.k(Ty::I1, 1);
    let within = |q: &mut Builder, value: ValueId, bound: u64| {
        let bound = k(q, bound);
        q.push(Ty::I1, Op::Cmp(IntPred::Ult, value, bound))
    };
    let at_least = |q: &mut Builder, value: ValueId, bound: u64| {
        let bound = k(q, bound);
        q.push(Ty::I1, Op::Cmp(IntPred::Uge, value, bound))
    };
    if let Some(sel) = d.sel {
        let packed = sel.iter().enumerate().fold(0u64, |p, (i, &s)| p | (s as u64) << (3 * i));
        let packed = k(q, packed);
        let seven = k(q, 7);
        let low = q.int(IntOp::And, lane, seven);
        let three = k(q, 3);
        let shift = q.int(IntOp::Mul, low, three);
        let chosen = q.int(IntOp::LShr, packed, shift);
        let chosen = q.int(IntOp::And, chosen, seven);
        let group = q.int(IntOp::Xor, lane, low);
        return (q.int(IntOp::Or, group, chosen), truth);
    }
    let ctrl = d.ctrl as u64;
    match ctrl {
        0x000..=0x0ff => {
            let three = k(q, 3);
            let quad = q.int(IntOp::And, lane, three);
            let one = k(q, 1);
            let shift = q.int(IntOp::Shl, quad, one);
            let table = k(q, ctrl);
            let chosen = q.int(IntOp::LShr, table, shift);
            let chosen = q.int(IntOp::And, chosen, three);
            let group = q.int(IntOp::Xor, lane, quad);
            (q.int(IntOp::Or, group, chosen), truth)
        }
        0x101..=0x10f => {
            let n = k(q, ctrl - 0x100);
            let moved = q.int(IntOp::Add, column, n);
            let valid = within(q, moved, 16);
            (q.int(IntOp::Add, lane, n), valid)
        }
        0x111..=0x11f => {
            let n = ctrl - 0x110;
            let valid = at_least(q, column, n);
            let n = k(q, n);
            (q.int(IntOp::Sub, lane, n), valid)
        }
        0x121..=0x12f => {
            let n = k(q, ctrl - 0x120);
            let moved = q.int(IntOp::Sub, column, n);
            let moved = q.int(IntOp::And, moved, fifteen);
            (q.int(IntOp::Or, row_base, moved), truth)
        }
        0x130 => {
            let one = k(q, 1);
            let moved = q.int(IntOp::Add, lane, one);
            let valid = within(q, moved, lanes as u64);
            (moved, valid)
        }
        0x134 => {
            let one = k(q, 1);
            let moved = q.int(IntOp::Add, lane, one);
            let last = k(q, lanes as u64 - 1);
            (q.int(IntOp::And, moved, last), truth)
        }
        0x138 => {
            let valid = at_least(q, lane, 1);
            let one = k(q, 1);
            (q.int(IntOp::Sub, lane, one), valid)
        }
        0x13c => {
            let one = k(q, 1);
            let moved = q.int(IntOp::Sub, lane, one);
            let last = k(q, lanes as u64 - 1);
            (q.int(IntOp::And, moved, last), truth)
        }
        0x140 => {
            let mirrored = q.int(IntOp::Sub, fifteen, column);
            (q.int(IntOp::Or, row_base, mirrored), truth)
        }
        0x141 => {
            let seven = k(q, 7);
            let low = q.int(IntOp::And, lane, seven);
            let mirrored = q.int(IntOp::Sub, seven, low);
            let group = q.int(IntOp::Xor, lane, low);
            (q.int(IntOp::Or, group, mirrored), truth)
        }
        0x142 => {
            let valid = at_least(q, lane, 16);
            let one = k(q, 1);
            (q.int(IntOp::Sub, row_base, one), valid)
        }
        0x143 => {
            let valid = at_least(q, lane, 32);
            (k(q, 31), valid)
        }
        0x150..=0x15f => {
            let n = k(q, ctrl - 0x150);
            (q.int(IntOp::Or, row_base, n), truth)
        }
        0x160..=0x16f => {
            let n = k(q, ctrl - 0x160);
            let flipped = q.int(IntOp::Xor, column, n);
            (q.int(IntOp::Or, row_base, flipped), truth)
        }
        _ => panic!("DPP control {:#x} has no lowering", ctrl),
    }
}

fn plan(d: &DPP, registry: &DialectRegistry, lanes: u32) -> Lowering {
    let mut q = Builder::new(registry, vec![]);
    let (source, in_range) = selection(&mut q, d, lanes);
    let lane = q.push(Ty::I32, Op::Env(Env::LaneId));
    let four = q.k(Ty::I32, 4);
    let row = q.int(IntOp::LShr, lane, four);
    let rows = q.k(Ty::I32, d.row_mask as u64);
    let row_ok = q.int(IntOp::LShr, rows, row);
    let two = q.k(Ty::I32, 2);
    let three = q.k(Ty::I32, 3);
    let bank = q.int(IntOp::LShr, lane, two);
    let bank = q.int(IntOp::And, bank, three);
    let banks = q.k(Ty::I32, d.bank_mask as u64);
    let bank_ok = q.int(IntOp::LShr, banks, bank);
    let one = q.k(Ty::I32, 1);
    let enabled = q.int(IntOp::And, row_ok, bank_ok);
    let enabled = q.int(IntOp::And, enabled, one);
    let enabled = q.int(IntOp::Shl, enabled, one);
    let in_range = q.push(Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, in_range));
    let flags = q.int(IntOp::Or, enabled, in_range);
    let address = q.int(IntOp::Shl, source, two);
    q.finish_many(
        false,
        vec![
            (Output::Vgpr(INDEX, Ty::I32), address),
            (Output::Vgpr(ENABLE, Ty::I32), flags),
        ],
    )
}

fn fetch(fi: bool, source: u32) -> Lowering {
    let v = |r: u32| Operand::Source(SourceOperand::VectorRegister(r as u8));
    Lowering::Wave(YieldAction::new(
        EffectOp::Wave(if fi { WaveOp::BpermuteFi } else { WaveOp::Bpermute }),
        vec![v(INDEX), v(source), Operand::Exec],
        vec![Destination::Vgpr(VALUE)],
    ))
}

fn presence() -> Lowering {
    let v = |r: u32| Operand::Source(SourceOperand::VectorRegister(r as u8));
    Lowering::Wave(YieldAction::new(
        EffectOp::Wave(WaveOp::Bpermute),
        vec![v(INDEX), Operand::Source(SourceOperand::IntegerConstant(1)), Operand::Exec],
        vec![Destination::Vgpr(INDEX)],
    ))
}

fn gate(fi: bool, bound_ctrl: bool, registry: &DialectRegistry) -> Lowering {
    let mut inputs = vec![
        input(SourceOperand::VectorRegister(VALUE as u8), Ty::I32),
        input(SourceOperand::VectorRegister(ENABLE as u8), Ty::I32),
    ];
    if !fi {
        inputs.push(input(SourceOperand::VectorRegister(INDEX as u8), Ty::I32));
    }
    let mut q = Builder::new(registry, inputs);
    let one = q.k(Ty::I32, 1);
    let in_range = q.int(IntOp::And, ValueId(1), one);
    let valid = if fi {
        in_range
    } else {
        q.int(IntOp::And, in_range, ValueId(2))
    };
    let rows = q.int(IntOp::LShr, ValueId(1), one);
    let allowed = if bound_ctrl {
        one
    } else {
        valid
    };
    let enabled = q.int(IntOp::And, rows, allowed);
    let zero = q.k(Ty::I32, 0);
    let present = q.push(Ty::I1, Op::Cmp(IntPred::Ne, valid, zero));
    let value = q.push(Ty::I32, Op::Select(present, ValueId(0), zero));
    q.finish_many(
        false,
        vec![
            (Output::Vgpr(VALUE, Ty::I32), value),
            (Output::Vgpr(ENABLE, Ty::I32), enabled),
        ],
    )
}

pub fn lowerings(d: &DPP, registry: &DialectRegistry, lanes: u32) -> Vec<Lowering> {
    let source = source_register(&d.inner);
    let mut out = vec![plan(d, registry, lanes), fetch(d.fi, source)];
    if !d.fi {
        out.push(presence());
    }
    out.push(gate(d.fi, d.bound_ctrl, registry));
    let inner = with_source(&d.inner, VALUE, d.abs, d.neg);
    let lowering = super::instruction_with_registry(&inner, registry, lanes);
    out.push(gated(lowering, ENABLE, registry));
    out
}

fn permlane_plan(i: &crate::rdna_instructions::VOP3, registry: &DialectRegistry) -> Lowering {
    let mut q = Builder::new(
        registry,
        vec![input(i.src1.clone(), Ty::I32), input(i.src2.clone(), Ty::I32)],
    );
    let lane = q.push(Ty::I32, Op::Env(Env::LaneId));
    let eight = q.k(Ty::I32, 8);
    let upper = q.int(IntOp::And, lane, eight);
    let zero = q.k(Ty::I32, 0);
    let is_upper = q.push(Ty::I1, Op::Cmp(IntPred::Ne, upper, zero));
    let selects = q.push(Ty::I32, Op::Select(is_upper, ValueId(1), ValueId(0)));
    let seven = q.k(Ty::I32, 7);
    let low = q.int(IntOp::And, lane, seven);
    let two = q.k(Ty::I32, 2);
    let shift = q.int(IntOp::Shl, low, two);
    let chosen = q.int(IntOp::LShr, selects, shift);
    let fifteen = q.k(Ty::I32, 15);
    let chosen = q.int(IntOp::And, chosen, fifteen);
    let column = q.int(IntOp::And, lane, fifteen);
    let row_base = q.int(IntOp::Xor, lane, column);
    let row_base = if matches!(i.op, I::V_PERMLANEX16_B32) {
        let sixteen = q.k(Ty::I32, 16);
        q.int(IntOp::Xor, row_base, sixteen)
    } else {
        row_base
    };
    let source = q.int(IntOp::Or, row_base, chosen);
    let address = q.int(IntOp::Shl, source, two);
    let flags = q.k(Ty::I32, 3);
    q.finish_many(
        false,
        vec![
            (Output::Vgpr(INDEX, Ty::I32), address),
            (Output::Vgpr(ENABLE, Ty::I32), flags),
        ],
    )
}

pub fn permlane_lowerings(
    i: &crate::rdna_instructions::VOP3,
    registry: &DialectRegistry,
    lanes: u32,
) -> Vec<Lowering> {
    let source = match i.src0 {
        SourceOperand::VectorRegister(r) => r as u32,
        ref other => panic!("permlane reads a source other than a VGPR: {:?}", other),
    };
    let fi = i.opsel & 1 != 0;
    let bound_ctrl = i.opsel & 2 != 0;
    let mut out = vec![permlane_plan(i, registry), fetch(fi, source)];
    if !fi {
        out.push(presence());
    }
    out.push(gate(fi, bound_ctrl, registry));
    let moved = InstFormat::VOP1(crate::rdna_instructions::VOP1 {
        src0: SourceOperand::VectorRegister(VALUE as u8),
        op: I::V_MOV_B32,
        vdst: i.vdst,
    });
    let lowering = super::instruction_with_registry(&moved, registry, lanes);
    out.push(gated(lowering, ENABLE, registry));
    out
}
