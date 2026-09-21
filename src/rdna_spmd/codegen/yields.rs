use super::super::ir::{EffectOp, Func, Inst, Ty, WaveOp};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(crate) enum Argument {
    Lane,
    Uniform,
    Constant(u32),
}

#[derive(Clone)]
pub(crate) struct YieldValues {
    pub op: EffectOp,
    pub inputs: Vec<Ty>,
    pub outputs: Vec<Ty>,
    pub output_base: usize,
    pub base: usize,
    pub uniform_selector: bool,
    pub arguments: Vec<Argument>,
}
impl YieldValues {
    pub fn new(op: EffectOp) -> Self {
        let (inputs, outputs) = op.signature();
        assert!(inputs.len() <= 24 && outputs.len() <= 8);
        assert!(inputs.iter().chain(&outputs).all(|ty| ty.bits() <= 32));

        let output_base = match op {
            EffectOp::Wave(WaveOp::WriteLane) => 2,
            EffectOp::Wave(WaveOp::Bpermute | WaveOp::BpermuteFi) => 1,
            EffectOp::Wave(WaveOp::Wmma) => 8,
            _ => 0,
        };
        let arguments = vec![Argument::Lane; inputs.len()];
        Self {
            op,
            inputs,
            outputs,
            output_base,
            base: 0,
            uniform_selector: false,
            arguments,
        }
    }
    pub fn cells(&self) -> usize {
        self.inputs.len().max(self.output_base + self.outputs.len())
    }
    pub fn uniform_result(&self) -> bool {
        match self.op {
            EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane) => true,
            EffectOp::Wave(WaveOp::ReadLane) => self.uniform_selector,
            EffectOp::BarrierSignal { is_first: true } => true,
            _ => false,
        }
    }
    pub fn is_wave(&self) -> bool {
        matches!(self.op, EffectOp::Wave(_))
    }
}

pub(super) fn layouts(
    ir: &Func,
    uniform: &[bool],
    constants: &[Option<u64>],
) -> (
    BTreeMap<u64, YieldValues>,
    Vec<Vec<u64>>,
) {
    let mut out: BTreeMap<u64, YieldValues> = BTreeMap::new();
    let mut groups: Vec<Vec<u64>> = Vec::new();
    for block in ir.blocks.values() {
        let mut open: Option<(usize, std::collections::BTreeSet<usize>)> = None;
        for inst in &block.insts {
            let Inst::Effect {
                provenance,
                op,
                inputs,
                outputs,
            } = inst
            else {
                open = None;
                continue;
            };
            let scheduled = matches!(
                op,
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait
            );
            if !scheduled {
                open = None;
                continue;
            }
            let mut layout = YieldValues::new(*op);
            layout.uniform_selector =
                *op == EffectOp::Wave(WaveOp::ReadLane) && uniform[inputs[1].0];
            for (index, &input) in inputs.iter().enumerate() {
                if *op == EffectOp::Wave(WaveOp::Wmma)
                    || *op == EffectOp::Wave(WaveOp::WriteLane) && index == 2
                {
                    continue;
                }
                layout.arguments[index] = if let Some(k) = constants[input.0] {
                    Argument::Constant(k as u32)
                } else if uniform[input.0] {
                    Argument::Uniform
                } else {
                    Argument::Lane
                };
            }
            let joinable = matches!(op, EffectOp::Wave(w) if *w != WaveOp::Wmma);
            let joins = match (&open, joinable) {
                (Some((_, produced)), true) => !inputs.iter().any(|v| produced.contains(&v.0)),
                _ => false,
            };
            if joins {
                let (group, produced) = open.as_mut().unwrap();
                let last = *groups[*group].last().unwrap();
                layout.base = out[&last].base + out[&last].cells();
                for (v, _) in outputs {
                    produced.insert(v.0);
                }
                groups[*group].push(*provenance);
            } else {
                groups.push(vec![*provenance]);
                open = joinable
                    .then(|| (groups.len() - 1, outputs.iter().map(|(v, _)| v.0).collect()));
            }
            out.insert(*provenance, layout);
        }
    }
    (out, groups)
}
