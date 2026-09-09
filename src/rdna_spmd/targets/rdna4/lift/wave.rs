//! Lift wave and synchronization operations once, including their boundary values.
use super::regs::BoundaryIo;
use crate::rdna_spmd::ir::{*, Ty};
use crate::{
    instructions::I,
    rdna_instructions::{InstFormat, SourceOperand},
};

#[derive(Clone, Debug)]
pub(crate) enum Operand {
    Source(SourceOperand),
    Add(Box<Operand>, u32),
    Exec,
}
impl Operand {
    #[cfg(test)]
    pub fn eval(
        &self,
        lane: usize,
        read: &impl Fn(usize, &SourceOperand) -> u32,
        exec: &impl Fn(usize) -> bool,
    ) -> u32 {
        match self {
            Self::Source(s) => read(lane, s),
            Self::Add(s, k) => s.eval(lane, read, exec).wrapping_add(*k),
            Self::Exec => exec(lane) as u32,
        }
    }
    pub(in crate::rdna_spmd) fn registers(&self, io: &mut BoundaryIo) {
        match self {
            Self::Source(SourceOperand::ScalarRegister(r)) => io.reads.add_sgpr(*r as u32),
            Self::Source(SourceOperand::VectorRegister(r)) => io.reads.add_vgpr(*r as u32),
            Self::Add(a, _) => a.registers(io),
            Self::Exec => io.reads.add_sgpr(126),
            _ => {}
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Destination {
    Sgpr(u32),
    Vgpr(u32),
    Scc,
}
/// A verified effect with its ISA operand and destination bindings. The scheduler consumes
/// the same signature that function SSA and boundary IO use.
#[derive(Clone, Debug)]
pub struct YieldAction {
    pub(crate) op: EffectOp,
    pub(crate) inputs: Vec<Operand>,
    pub(crate) outputs: Vec<Destination>,
}
impl YieldAction {
    pub(crate) fn new(op: EffectOp, inputs: Vec<Operand>, outputs: Vec<Destination>) -> Self {
        let (a, r) = op.signature();
        assert_eq!(a.len(), inputs.len());
        assert_eq!(r.len(), outputs.len());
        assert!(a.len() <= 24 && r.len() <= 8, "yield operand/result limit");
        if op == EffectOp::Wave(WaveOp::WriteLane) {
            for value in &inputs[..2] {
                assert!(
                    matches!(value,Operand::Source(s) if !matches!(s,SourceOperand::VectorRegister(_) | SourceOperand::PrivateBase)),
                    "writelane requires uniform value and lane"
                );
            }
        }
        Self {
            op,
            inputs,
            outputs,
        }
    }
    pub(in crate::rdna_spmd) fn io(&self) -> BoundaryIo {
        let mut io = BoundaryIo::default();
        for operand in &self.inputs {
            operand.registers(&mut io);
        }
        for dest in &self.outputs {
            match dest {
                Destination::Sgpr(r) => io.writes.add_sgpr(*r),
                Destination::Vgpr(r) => io.writes.add_vgpr(*r),
                Destination::Scc => io.writes.scc = true,
            }
        }
        io
    }
    #[cfg(test)]
    pub(crate) fn evaluate(
        &self,
        valid: u32,
        read: impl Fn(usize, &SourceOperand) -> u32,
        exec: impl Fn(usize) -> bool,
    ) -> [[u32; 32]; 1] {
        assert!(self.inputs.len() <= 4 && self.outputs.len() == 1);
        let op = match self.op {
            EffectOp::Wave(op) => op,
            _ => panic!("workgroup barrier requires round state"),
        };
        [crate::rdna_spmd::engine::yields::evaluate(op, valid, |index, lane| {
            self.inputs[index].eval(lane, &read, &exec)
        })]
    }
}
fn src(s: SourceOperand) -> Operand {
    Operand::Source(s)
}
fn v(r: u32) -> Operand {
    src(SourceOperand::VectorRegister(r as u8))
}
pub(crate) fn instruction(inst: &InstFormat) -> Option<YieldAction> {
    use Destination::*;
    Some(match inst {
        InstFormat::VOP1(i) if matches!(i.op, I::V_READFIRSTLANE_B32) => YieldAction::new(
            EffectOp::Wave(WaveOp::ReadFirstLane),
            vec![src(i.src0.clone()), Operand::Exec],
            vec![Sgpr(i.vdst as u32)],
        ),
        InstFormat::VOP3(i) if matches!(i.op, I::V_READLANE_B32) => YieldAction::new(
            EffectOp::Wave(WaveOp::ReadLane),
            vec![src(i.src0.clone()), src(i.src1.clone()), src(SourceOperand::IntegerConstant(match i.src0 { SourceOperand::VectorRegister(r) => r as u64, _ => u64::MAX }))],
            vec![Sgpr(i.vdst as u32)],
        ),
        InstFormat::VOP3(i) if matches!(i.op, I::V_WRITELANE_B32) => YieldAction::new(
            EffectOp::Wave(WaveOp::WriteLane),
            vec![src(i.src0.clone()), src(i.src1.clone()), v(i.vdst as u32), src(SourceOperand::IntegerConstant(i.vdst as u64))],
            vec![Vgpr(i.vdst as u32)],
        ),
        InstFormat::DS(i) if matches!(i.op, I::DS_BPERMUTE_B32 | I::DS_BPERMUTE_FI_B32) => {
            YieldAction::new(
                EffectOp::Wave(if matches!(i.op, I::DS_BPERMUTE_B32) {
                    WaveOp::Bpermute
                } else {
                    WaveOp::BpermuteFi
                }),
                vec![
                    Operand::Add(Box::new(v(i.addr as u32)), i.offset0 as u32),
                    v(i.data0 as u32),
                    Operand::Exec,
                ],
                vec![Vgpr(i.vdst as u32)],
            )
        }
        InstFormat::VOP3P(i) if matches!(i.op, I::V_WMMA_F32_16X16X16_F16) => {
            let reg = |s: &SourceOperand| match s {
                SourceOperand::VectorRegister(r) => *r as u32,
                _ => panic!("WMMA source must be a VGPR"),
            };
            let mut args = vec![];
            for (s, n) in [(&i.src0, 4), (&i.src1, 4), (&i.src2, 8)] {
                args.extend((0..n).map(|k| v(reg(s) + k)));
            }
            YieldAction::new(
                EffectOp::Wave(WaveOp::Wmma),
                args,
                (0..8).map(|k| Vgpr(i.vdst as u32 + k)).collect(),
            )
        }
        InstFormat::SOP1(i)
            if matches!(i.op, I::S_BARRIER_SIGNAL | I::S_BARRIER_SIGNAL_ISFIRST) =>
        {
            let first = matches!(i.op, I::S_BARRIER_SIGNAL_ISFIRST);
            YieldAction::new(
                EffectOp::BarrierSignal { is_first: first },
                vec![src(i.ssrc0.clone())],
                if first { vec![Scc] } else { vec![] },
            )
        }
        InstFormat::SOPP(i) if matches!(i.op, I::S_BARRIER_WAIT) => YieldAction::new(
            EffectOp::BarrierWait,
            vec![src(SourceOperand::LiteralConstant(i.simm16 as u32))],
            vec![],
        ),
        _ => return None,
    })
}


pub(in crate::rdna_spmd) fn mark_scheduled(insts: &mut [crate::rdna_spmd::ir::Inst]) {
    for inst in insts {
        if let crate::rdna_spmd::ir::Inst::Effect { provenance, op, .. } = inst {
            if matches!(op, EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait) && *provenance & (1 << 63) == 0 { *provenance |= crate::rdna_spmd::ir::SCHEDULED; }
        }
    }
}

pub(in crate::rdna_spmd) struct Plan {
    pub core: std::ops::Range<usize>,
    pub end: usize,
    pub definitions: Vec<(Destination, super::ValueId)>,
}

impl YieldAction {
    pub(super) fn lift(
        &self,
        f: &mut crate::rdna_spmd::ir::Func,
        block: &mut crate::rdna_spmd::ir::Block,
        words: &mut super::regs::Words,
        provenance: &mut u64,
    ) -> Plan {
        use crate::rdna_spmd::ir::{Inst, IntOp, Op, ValueId};
        fn operand(
            arg: &Operand,
            ty: Ty,
            f: &mut crate::rdna_spmd::ir::Func,
            block: &mut crate::rdna_spmd::ir::Block,
            words: &super::regs::Words,
        ) -> ValueId {
            if let Operand::Add(a, k) = arg {
                let a = operand(a, ty, f, block, words);
                let b = f.value(ty);
                block.insts.push(Inst::Core {
                    value: b,
                    ty,
                    op: Op::Const(ty, *k as u64),
                });
                let value = f.value(ty);
                block.insts.push(Inst::Core {
                    value,
                    ty,
                    op: Op::Int(IntOp::Add, a, b),
                });
                return value;
            }
            let source = match arg {
                Operand::Source(source) => source.clone(),
                Operand::Exec => SourceOperand::ScalarRegister(126),
                Operand::Add(..) => unreachable!(),
            };
            let mut operands = super::regs::Operands::default();
            let value = operands.read(&super::input(source, ty), false, None,
                f, block, words, &mut super::regs::Views::new());
            block.insts.extend(operands.core);
            value
        }
        let start = block.insts.len();
        let (args, results) = self.op.signature();
        let inputs: Vec<_> = self
            .inputs
            .iter()
            .zip(args)
            .map(|(a, t)| operand(a, t, f, block, words))
            .collect();
        let outputs: Vec<_> = results.into_iter().map(|t| (f.value(t), t)).collect();
        let predicate = matches!(self.op, EffectOp::Wave(WaveOp::Bpermute | WaveOp::BpermuteFi))
            .then(|| inputs[2]);
        let end = block.insts.len();
        block.insts.push(Inst::Effect {
            provenance: *provenance << 8,
            op: self.op,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        });
        *provenance += 1;
        let mut definitions = Vec::new();
        for (&(v, t), dest) in outputs.iter().zip(&self.outputs) {
            let word = match *dest {
                Destination::Vgpr(r) => Some(super::regs::Word::Vgpr(r)),
                Destination::Sgpr(r) => super::regs::Word::scalar(r),
                Destination::Scc => None,
            };
            let mut stored = v;
            if let Some(mask) = predicate {
                let old = words[&word.expect("predicated wave destination must be a word")];
                stored = f.value(t);
                block.insts.push(Inst::Core { value: stored, ty: t, op: Op::Select(mask, v, old) });
            }
            if matches!(dest, Destination::Sgpr(_)) && t == Ty::I1 {
                let value = f.value(Ty::I32);
                block.insts.push(Inst::Core { value, ty: Ty::I32,
                    op: Op::Convert(crate::rdna_spmd::ir::Cvt::ZExt, Ty::I32, stored) });
                stored = value;
            }
            if t == Ty::F32 && word.is_some() {
                let value = f.value(Ty::I32);
                block.insts.push(Inst::Core { value, ty: Ty::I32,
                    op: Op::Convert(crate::rdna_spmd::ir::Cvt::Bitcast, Ty::I32, stored) });
                stored = value;
            }
            if let Some(word) = word {
                if matches!(word,super::regs::Word::Mask(_)) {
                    stored = super::regs::project(f,&mut block.insts,stored);
                }
                if word==super::regs::Word::Mask(126) {
                    stored=super::regs::valid_exec(f,&mut block.insts,stored);
                }
                words.insert(word, stored);
            } else if matches!(dest, Destination::Sgpr(r) if *r != 124) {
                panic!("invalid scalar wave destination: {:?}", dest);
            }
            definitions.push((*dest, stored));
        }
        Plan { core: start..end, end: block.insts.len(), definitions }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wave_operations_observe_all_32_lanes_and_validity() {
        let value = |lane: usize, _: &SourceOperand| lane as u32 + 100;
        for valid in [u32::MAX, 0x007f_ffff] {
            for mask in [0u32, 1 << 21, 0xaaaa_aaaa, u32::MAX] {
                let exec = |lane: usize| mask >> lane & 1 != 0;
                for op in [WaveOp::Any, WaveOp::Ballot] {
                    let action = YieldAction::new(
                        EffectOp::Wave(op),
                        vec![Operand::Exec],
                        vec![Destination::Sgpr(0)],
                    );
                    let expected = if op == WaveOp::Any {
                        ((mask & valid) != 0) as u32
                    } else {
                        mask & valid
                    };
                    assert_eq!(action.evaluate(valid, value, exec)[0], [expected; 32]);
                }
                let action = YieldAction::new(
                    EffectOp::Wave(WaveOp::ReadFirstLane),
                    vec![v(0), Operand::Exec],
                    vec![Destination::Sgpr(0)],
                );
                let lane = if mask & valid == 0 {
                    0
                } else {
                    (mask & valid).trailing_zeros()
                };
                assert_eq!(action.evaluate(valid, value, exec)[0], [lane + 100; 32]);
            }
        }
        let args = vec![v(0), v(1), Operand::Exec];
        let read = |lane: usize, s: &SourceOperand| match s {
            SourceOperand::VectorRegister(0) => ((31 - lane) as u32 * 4) + 3,
            _ => 100 + lane as u32,
        };
        for op in [WaveOp::Bpermute, WaveOp::BpermuteFi] {
            let action =
                YieldAction::new(EffectOp::Wave(op), args.clone(), vec![Destination::Vgpr(2)]);
            let result = action.evaluate(u32::MAX, read, |lane| lane % 2 == 0);
            for lane in 0..32 {
                assert_eq!(
                    result[0][lane],
                    if op == WaveOp::BpermuteFi || (31 - lane) % 2 == 0 {
                        131 - lane as u32
                    } else {
                        0
                    }
                );
            }
        }
        let action = YieldAction::new(
            EffectOp::Wave(WaveOp::ReadLane),
            vec![v(0), v(1), src(SourceOperand::IntegerConstant(0))],
            vec![Destination::Vgpr(2)],
        );
        let result = action.evaluate(
            u32::MAX,
            |lane, s| {
                if matches!(s, SourceOperand::VectorRegister(0)) {
                    lane as u32 + 100
                } else {
                    (lane as u32 + 41) % 64
                }
            },
            |_| false,
        );
        for lane in 0..32 {
            assert_eq!(result[0][lane], ((lane as u32 + 41) & 31) + 100);
        }
    }
    #[test]
    fn writelane_preserves_the_other_lanes_and_rejects_varying_arguments() {
        let action = YieldAction::new(
            EffectOp::Wave(WaveOp::WriteLane),
            vec![
                src(SourceOperand::IntegerConstant(99)),
                src(SourceOperand::IntegerConstant(53)),
                v(0),
                src(SourceOperand::IntegerConstant(0)),
            ],
            vec![Destination::Vgpr(0)],
        );
        let result = action.evaluate(
            u32::MAX,
            |lane, s| match s {
                SourceOperand::IntegerConstant(v) => *v as u32,
                _ => lane as u32,
            },
            |_| false,
        );
        for lane in 0..32 {
            assert_eq!(result[0][lane], if lane == 21 { 99 } else { lane as u32 });
        }
        assert!(std::panic::catch_unwind(|| YieldAction::new(
            EffectOp::Wave(WaveOp::WriteLane),
            vec![v(0), src(SourceOperand::IntegerConstant(0)), v(1), src(SourceOperand::IntegerConstant(1))],
            vec![Destination::Vgpr(1)]
        ))
        .is_err());
    }
}
