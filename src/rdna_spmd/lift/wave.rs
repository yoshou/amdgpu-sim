//! Lift wave and synchronization operations once, including their boundary values.
use super::super::{
    boundary::BoundaryIo,
    ir::typed::{effect::*, Ty},
};
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
    #[inline]
    pub(crate) fn is_uniform(&self) -> bool {
        match self {
            Self::Source(s) => !matches!(s, SourceOperand::VectorRegister(_) | SourceOperand::PrivateBase),
            Self::Add(a, _) => a.is_uniform(),
            Self::Exec => false,
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
    #[inline]
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
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Destination {
    Sgpr(u32),
    Vgpr(u32),
    Scc,
}
/// A verified effect with its register adapter bindings. The scheduler consumes
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
    pub(crate) fn is_wave(&self) -> bool {
        matches!(self.op, EffectOp::Wave(_))
    }
    /// Register layout selected from the typed operands, outside the lane loop.
    pub(crate) fn bpermute_registers(&self) -> Option<(u32, u32, u32, u32, bool)> {
        let fi = match self.op {
            EffectOp::Wave(WaveOp::Bpermute) => false,
            EffectOp::Wave(WaveOp::BpermuteFi) => true,
            _ => return None,
        };
        let (address, offset) = match &self.inputs[0] {
            Operand::Add(a, k) => (a.as_ref(), *k),
            a => (a, 0),
        };
        match (address, &self.inputs[1], &self.inputs[2], self.outputs[0]) {
            (Operand::Source(SourceOperand::VectorRegister(a)),
             Operand::Source(SourceOperand::VectorRegister(v)), Operand::Exec, Destination::Vgpr(d)) =>
                Some((*a as u32, *v as u32, d, offset, fi)),
            _ => None,
        }
    }
    /// Scalar lane selectors are wave-uniform in the register adapter. Capture
    /// the selected value before writing anything, and leave untouched lanes in
    /// their existing cells instead of materializing an entire wave result.
    #[inline]
    pub(crate) fn uniform_lane_access(
        &self,
        valid: u32,
        read: impl Fn(usize, &SourceOperand) -> u32,
    ) -> Option<(Destination, usize, u32)> {
        let first = valid.trailing_zeros() as usize;
        match self.op {
            EffectOp::Wave(WaveOp::ReadLane)
                if self.inputs[1].is_uniform() && matches!(self.outputs[0], Destination::Sgpr(_)) => {
                let lane = (self.inputs[1].eval(first, &read, &|_| false) & 31) as usize;
                Some((self.outputs[0], lane, if valid >> lane & 1 != 0 { self.inputs[0].eval(lane, &read, &|_| false) } else { 0 }))
            }
            EffectOp::Wave(WaveOp::WriteLane) => {
                let lane = (self.inputs[1].eval(first, &read, &|_| false) & 31) as usize;
                Some((self.outputs[0], lane, self.inputs[0].eval(first, &read, &|_| false)))
            }
            _ => None,
        }
    }
    #[inline]
    pub(crate) fn wmma_registers(&self) -> Option<(u32, u32, u32, u32)> {
        if self.op != EffectOp::Wave(WaveOp::Wmma) {
            return None;
        }
        let reg = |k| match self.inputs[k] {
            Operand::Source(SourceOperand::VectorRegister(r)) => r as u32,
            _ => unreachable!(),
        };
        let dst = match self.outputs[0] {
            Destination::Vgpr(r) => r,
            _ => unreachable!(),
        };
        Some((dst, reg(0), reg(4), reg(8)))
    }
    /// Semantic wave application. Operands are captured before any result is
    /// written, which also covers source/destination overlap and inactive reads.
    pub(crate) fn evaluate(
        &self,
        valid: u32,
        read: impl Fn(usize, &SourceOperand) -> u32,
        exec: impl Fn(usize) -> bool,
    ) -> [[u32; 32]; 1] {
        assert!(self.inputs.len() <= 3 && self.outputs.len() == 1);
        let arg = |index: usize, lane: usize| {
            if valid >> lane & 1 != 0 {
                self.inputs[index].eval(lane, &read, &exec)
            } else {
                0
            }
        };
        let op = match self.op {
            EffectOp::Wave(op) => op,
            _ => panic!("workgroup barrier requires round state"),
        };
        // Capture just the values used by this effect. The result is complete
        // before the caller writes any destination, including overlapping ones.
        let mut out = [[0; 32]; 1];
        match op {
            WaveOp::Any => out[0].fill((0..32).any(|lane| arg(0, lane) != 0) as u32),
            WaveOp::Ballot => {
                let mask = (0..32).fold(0, |mask, lane| mask | ((arg(0, lane) != 0) as u32) << lane);
                out[0].fill(mask);
            }
            WaveOp::ReadFirstLane => {
                let lane = (0..32).find(|&lane| arg(1, lane) != 0).unwrap_or(0);
                out[0].fill(arg(0, lane));
            }
            WaveOp::ReadLane => {
                out[0] = std::array::from_fn(|lane| arg(0, (arg(1, lane) & 31) as usize));
            }
            WaveOp::WriteLane => {
                let first = valid.trailing_zeros() as usize;
                assert!(first < 32, "empty wave");
                let value = arg(0, first);
                let selector = arg(1, first);
                out[0] = std::array::from_fn(|lane| {
                    if valid >> lane & 1 != 0 {
                        assert_eq!(arg(0, lane), value, "nonuniform writelane value");
                        assert_eq!(arg(1, lane), selector, "nonuniform writelane lane");
                    }
                    arg(2, lane)
                });
                out[0][(selector & 31) as usize] = value;
            }
            WaveOp::Bpermute | WaveOp::BpermuteFi => {
                out[0] = std::array::from_fn(|lane| {
                    let src = ((arg(0, lane) >> 2) & 31) as usize;
                    if op == WaveOp::BpermuteFi || arg(2, src) != 0 { arg(1, src) } else { 0 }
                });
            }
            WaveOp::Wmma => panic!("WMMA uses the existing fragment lowering"),
        }
        out
    }
    pub(crate) fn writes_lane(&self, lane: usize, valid: u32, exec: bool) -> bool {
        valid >> lane & 1 != 0
            && (!matches!(
                self.op,
                EffectOp::Wave(WaveOp::Bpermute | WaveOp::BpermuteFi)
            ) || exec)
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
            vec![src(i.src0.clone()), src(i.src1.clone())],
            vec![Sgpr(i.vdst as u32)],
        ),
        InstFormat::VOP3(i) if matches!(i.op, I::V_WRITELANE_B32) => YieldAction::new(
            EffectOp::Wave(WaveOp::WriteLane),
            vec![src(i.src0.clone()), src(i.src1.clone()), v(i.vdst as u32)],
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

pub(crate) fn split(
    program: &super::super::ir::ScalarProgram,
    accept: impl Fn(&YieldAction) -> bool,
) -> (
    super::super::ir::ScalarProgram,
    std::collections::BTreeMap<usize, YieldAction>,
) {
    use super::super::ir::{ScalarBlock, ScalarProgram, Terminator};
    let mut next = program.blocks.keys().max().copied().unwrap_or(0) + 1;
    let mut blocks = std::collections::BTreeMap::new();
    let mut yields = std::collections::BTreeMap::new();
    for block in program.blocks.values() {
        let mut pc = block.pc;
        let mut body = vec![];
        for inst in &block.body {
            let ops = if matches!(inst,InstFormat::SOPP(i) if matches!(i.op,I::S_BARRIER)) {
                vec![
                    YieldAction::new(
                        EffectOp::BarrierSignal { is_first: false },
                        vec![src(SourceOperand::LiteralConstant(u32::MAX))],
                        vec![],
                    ),
                    YieldAction::new(
                        EffectOp::BarrierWait,
                        vec![src(SourceOperand::LiteralConstant(u32::MAX))],
                        vec![],
                    ),
                ]
            } else {
                instruction(inst).into_iter().collect()
            };
            if !ops.is_empty() && ops.iter().all(&accept) {
                for action in ops {
                    let resume = next;
                    next += 1;
                    blocks.insert(
                        pc,
                        ScalarBlock {
                            pc,
                            body: std::mem::take(&mut body),
                            term: Terminator::Yield {
                                resume,
                                action: Box::new(action.clone()),
                            },
                        },
                    );
                    yields.insert(resume, action);
                    pc = resume;
                }
            } else {
                body.push(inst.clone());
            }
        }
        blocks.insert(
            pc,
            ScalarBlock {
                pc,
                body,
                term: block.term.clone(),
            },
        );
    }
    (
        ScalarProgram {
            entry_pc: program.entry_pc,
            blocks,
        },
        yields,
    )
}

impl YieldAction {
    pub(crate) fn lift(
        &self,
        f: &mut super::super::ir::typed::cfg::Func,
        block: &mut super::super::ir::typed::cfg::Block,
        words: &mut std::collections::BTreeMap<u32, super::super::ir::typed::ValueId>,
        provenance: &mut u64,
    ) {
        use super::super::ir::typed::{cfg::Inst, IntOp, Op, ValueId};
        fn operand(
            arg: &Operand,
            ty: Ty,
            f: &mut super::super::ir::typed::cfg::Func,
            block: &mut super::super::ir::typed::cfg::Block,
            words: &std::collections::BTreeMap<u32, ValueId>,
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
            let deps = match arg {
                Operand::Source(SourceOperand::VectorRegister(r)) => {
                    words.get(&(*r as u32)).copied().into_iter().collect()
                }
                _ => vec![],
            };
            let value = f.value(ty);
            block.insts.push(Inst::Boundary {
                inputs: deps,
                outputs: vec![(value, ty)],
            });
            value
        }
        let (args, results) = self.op.signature();
        let inputs = self
            .inputs
            .iter()
            .zip(args)
            .map(|(a, t)| operand(a, t, f, block, words))
            .collect();
        let outputs: Vec<_> = results.into_iter().map(|t| (f.value(t), t)).collect();
        block.insts.push(Inst::Effect {
            provenance: *provenance,
            op: self.op,
            inputs,
            outputs: outputs.clone(),
        });
        *provenance += 1;
        for (&(v, t), dest) in outputs.iter().zip(&self.outputs) {
            let stored = f.value(t);
            let mut deps = vec![v];
            if let Destination::Vgpr(r) = dest {
                if let Some(old) = words.get(r) {
                    deps.push(*old);
                }
                words.insert(*r, stored);
            }
            block.insts.push(Inst::Boundary {
                inputs: deps,
                outputs: vec![(stored, t)],
            });
        }
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
            vec![v(0), v(1)],
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
            vec![v(0), src(SourceOperand::IntegerConstant(0)), v(1)],
            vec![Destination::Vgpr(1)]
        ))
        .is_err());
    }
}
