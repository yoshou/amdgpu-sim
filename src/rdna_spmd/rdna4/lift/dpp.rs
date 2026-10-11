use super::wave::{Destination, Operand, YieldAction};
use super::*;
use crate::rdna_instructions::DPP;
use crate::rdna_spmd::ir::{Env, WaveOp};

pub const INDEX: u32 = 252;
pub const VALUE: u32 = 253;
pub const ENABLE: u32 = 255;
pub const TEMPORARIES: [u32; 4] = [INDEX, VALUE, VALUE + 1, ENABLE];

fn source_operand(inner: &InstFormat) -> Result<&SourceOperand, String> {
    match inner {
        InstFormat::VOP1(i) => Ok(&i.src0),
        InstFormat::VOP2(i) => Ok(&i.src0),
        InstFormat::VOPC(i) => Ok(&i.src0),
        InstFormat::VOP3(i) => Ok(&i.src0),
        InstFormat::VOP3SD(i) => Ok(&i.src0),
        other => Err(format!("DPP wraps a format it cannot modify: {}", super::opcode(other))),
    }
}

fn source_register(inner: &InstFormat) -> Result<u32, String> {
    match source_operand(inner)? {
        SourceOperand::VectorRegister(r) => Ok(*r as u32),
        other => Err(format!("DPP reads a source other than a VGPR: {:?}", other)),
    }
}

fn with_source(inner: &InstFormat, register: u32, abs: u8, neg: u8) -> Result<InstFormat, String> {
    let replaced = SourceOperand::VectorRegister(register as u8);
    let modified = abs | neg != 0;
    Ok(match inner.clone() {
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
        other => return Err(format!("DPP wraps a format it cannot modify: {}", super::opcode(&other))),
    })
}

pub fn gated(lowering: Lowering, enable: u32, registry: &DialectRegistry) -> Result<Lowering, String> {
    use crate::rdna_spmd::ir::ExprInst;
    let Lowering::TypedAlu {
        inputs,
        outputs,
        scalar,
        expr,
    } = lowering
    else {
        return Err("DPP gates a lowering that is not an ALU expression".to_string());
    };
    if scalar {
        return Err("DPP gates a scalar result".to_string());
    }
    let expr = expr.expr().clone();
    let n = inputs.len();
    let mut all = inputs;
    let mut olds = Vec::new();
    for output in &outputs {
        match *output {
            Output::Vgpr(r, ty) => {
                let old = if ty.bits() == 64 { Ty::I64 } else { Ty::I32 };
                all.push(input(SourceOperand::VectorRegister(r as u8), old));
            }
            Output::Mask(r) | Output::Compare(r) => {
                all.push(input(SourceOperand::ScalarRegister(r as u8), Ty::I1));
            }
            other => return Err(format!("DPP gates a result it cannot keep: {:?}", other)),
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
        let old = if matches!(ty, Ty::I32 | Ty::I64 | Ty::I1) {
            old
        } else {
            q.push(ty, Op::Convert(Cvt::Bitcast, ty, old))
        };
        let kept = q.push(ty, Op::Select(on, remap(result), old));
        results.push((*output, kept));
    }
    Ok(q.finish_many(false, results))
}

fn selection(q: &mut Builder, d: &DPP, lanes: u32) -> Result<(ValueId, ValueId), String> {
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
        return Ok((q.int(IntOp::Or, group, chosen), truth));
    }
    let ctrl = d.ctrl as u64;
    Ok(match ctrl {
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
        _ => return Err(format!("DPP control {:#x} has no lowering", ctrl)),
    })
}

fn plan(d: &DPP, registry: &DialectRegistry, lanes: u32) -> Result<Lowering, String> {
    let mut q = Builder::new(registry, vec![]);
    let (source, in_range) = selection(&mut q, d, lanes)?;
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
    Ok(q.finish_many(
        false,
        vec![
            (Output::Vgpr(INDEX, Ty::I32), address),
            (Output::Vgpr(ENABLE, Ty::I32), flags),
        ],
    ))
}

fn fetch(fi: bool, source: u32, destination: u32) -> Lowering {
    let v = |r: u32| Operand::Source(SourceOperand::VectorRegister(r as u8));
    Lowering::Wave(YieldAction::new(
        EffectOp::Wave(if fi { WaveOp::BpermuteFi } else { WaveOp::Bpermute }),
        vec![v(INDEX), v(source), Operand::Exec],
        vec![Destination::Vgpr(destination)],
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

fn gate(fi: bool, bound_ctrl: bool, wide: bool, registry: &DialectRegistry) -> Lowering {
    let mut inputs = vec![
        input(SourceOperand::VectorRegister(VALUE as u8), Ty::I32),
        input(SourceOperand::VectorRegister(ENABLE as u8), Ty::I32),
        input(SourceOperand::VectorRegister(INDEX as u8), Ty::I32),
    ];
    if wide {
        inputs.push(input(SourceOperand::VectorRegister(VALUE as u8 + 1), Ty::I32));
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
    let mut outputs = vec![
        (Output::Vgpr(VALUE, Ty::I32), value),
        (Output::Vgpr(ENABLE, Ty::I32), enabled),
    ];
    if wide {
        let high = q.push(Ty::I32, Op::Select(present, ValueId(3), zero));
        outputs.push((Output::Vgpr(VALUE + 1, Ty::I32), high));
    }
    q.finish_many(false, outputs)
}

fn moved(lowering: &Lowering) -> Result<bool, String> {
    match lowering {
        Lowering::TypedAlu { inputs, .. } => Ok(inputs.first().is_some_and(|i| i.ty.bits() == 64)),
        _ => Err("DPP gates a lowering that is not an ALU expression".to_string()),
    }
}

pub fn lowerings(d: &DPP, registry: &DialectRegistry, lanes: u32) -> Result<Vec<Lowering>, String> {
    let source = source_register(&d.inner)?;
    let inner = with_source(&d.inner, VALUE, d.abs, d.neg)?;
    let lowering = super::instruction_with_registry(&inner, registry, lanes)?;
    let wide = moved(&lowering)?;
    let mut out = vec![plan(d, registry, lanes)?, fetch(d.fi, source, VALUE)];
    if wide {
        out.push(fetch(d.fi, source + 1, VALUE + 1));
    }
    if !d.fi {
        out.push(presence());
    }
    out.push(gate(d.fi, d.bound_ctrl, wide, registry));
    out.push(gated(lowering, ENABLE, registry)?);
    Ok(out)
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
) -> Result<Vec<Lowering>, String> {
    let source = match i.src0 {
        SourceOperand::VectorRegister(r) => r as u32,
        ref other => return Err(format!("permlane reads a source other than a VGPR: {:?}", other)),
    };
    let fi = i.opsel & 1 != 0;
    let bound_ctrl = i.opsel & 2 != 0;
    let mut out = vec![permlane_plan(i, registry), fetch(fi, source, VALUE)];
    if !fi {
        out.push(presence());
    }
    out.push(gate(fi, bound_ctrl, false, registry));
    let moved = InstFormat::VOP1(crate::rdna_instructions::VOP1 {
        src0: SourceOperand::VectorRegister(VALUE as u8),
        op: I::V_MOV_B32,
        vdst: i.vdst,
    });
    let lowering = super::instruction_with_registry(&moved, registry, lanes)?;
    out.push(gated(lowering, ENABLE, registry)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_instructions::{VOP1, VOP3};

    fn wrapped(inner: InstFormat, ctrl: u16) -> DPP {
        DPP {
            inner: Box::new(inner),
            ctrl,
            sel: None,
            row_mask: 0xf,
            bank_mask: 0xf,
            bound_ctrl: false,
            fi: false,
            neg: 0,
            abs: 0,
        }
    }

    fn moved_word(src0: SourceOperand) -> InstFormat {
        InstFormat::VOP1(VOP1 {
            src0,
            op: I::V_MOV_B32,
            vdst: 6,
        })
    }

    #[test]
    fn a_dpp_source_outside_the_vgprs_or_an_unknown_control_is_refused() {
        let registry = crate::rdna_spmd::rdna4::dialect().registry;
        let scalar = wrapped(moved_word(SourceOperand::ScalarRegister(3)), 0x111);
        assert_eq!(
            lowerings(&scalar, &registry, 32).err().as_deref(),
            Some("DPP reads a source other than a VGPR: ScalarRegister(3)")
        );
        let unknown = wrapped(moved_word(SourceOperand::VectorRegister(1)), 0x1ff);
        assert_eq!(lowerings(&unknown, &registry, 32).err().as_deref(), Some("DPP control 0x1ff has no lowering"));
    }

    #[test]
    fn a_dpp_operand_of_two_words_fetches_both_halves_and_keeps_a_wide_result() {
        let registry = crate::rdna_spmd::rdna4::dialect().registry;
        let add = InstFormat::VOP3(VOP3 {
            vdst: 6,
            abs: 0,
            opsel: 0,
            cm: 0,
            op: I::V_ADD_F64,
            src0: SourceOperand::VectorRegister(0),
            src1: SourceOperand::VectorRegister(2),
            src2: SourceOperand::IntegerConstant(0),
            omod: 0,
            neg: 0,
        });
        let wide = lowerings(&wrapped(add, 0x153), &registry, 32).unwrap();
        assert_eq!(wide.len(), 6, "plan, two fetches, presence, gate and the gated add");
        let fetched: Vec<u32> = wide[1..3]
            .iter()
            .map(|l| match l {
                Lowering::Wave(action) => match action.outputs[0] {
                    Destination::Vgpr(r) => r,
                    ref other => panic!("{:?}", other),
                },
                _ => panic!("a fetch is a wave action"),
            })
            .collect();
        assert_eq!(fetched, vec![VALUE, VALUE + 1]);
        let Lowering::TypedAlu { outputs, inputs, .. } = &wide[5] else {
            panic!("the gated add is an ALU expression")
        };
        assert!(matches!(outputs[..], [Output::Vgpr(6, Ty::F64)]), "{:?}", outputs);
        assert!(inputs.iter().any(|i| i.ty == Ty::I64), "the old pair is an input");
        let narrow = lowerings(&wrapped(moved_word(SourceOperand::VectorRegister(1)), 0x111), &registry, 32).unwrap();
        assert_eq!(narrow.len(), 5, "a one-word source needs one fetch");
    }
}
