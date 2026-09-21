use super::super::analysis::Analyses;
use super::super::ir::*;
use super::super::program::{Parameter, ParameterSource};
use super::Pass;

struct Facts<'a> {
    f: &'a Func,
    defs: Vec<Option<Op>>,
    parameter: Vec<Option<(BlockId, usize)>>,
    effect: Vec<bool>,
    target: Vec<Option<(bool, Vec<ValueId>)>>,
    inputs: &'a [Parameter],
    state: Vec<Option<bool>>,
}

impl Facts<'_> {
    fn raw(&self, mut value: ValueId) -> ValueId {
        loop {
            match self.defs[value.0] {
                Some(Op::Convert(Cvt::Bitcast, to, a)) if self.f.types[a.0] == to => value = a,
                _ => return value,
            }
        }
    }

    fn uniform(&mut self, value: ValueId) -> bool {
        let value = self.raw(value);
        if let Some(known) = self.state[value.0] {
            return known;
        }
        self.state[value.0] = Some(false);
        let known = self.judge(value);
        self.state[value.0] = Some(known);
        known
    }

    fn judge(&mut self, value: ValueId) -> bool {
        if let Some((block, index)) = self.parameter[value.0] {
            if block == self.f.entry {
                return match self.inputs.get(index).map(|p| p.source) {
                    Some(ParameterSource::Vgpr(0)) => false,
                    Some(ParameterSource::MaskBit(_)) => false,
                    Some(_) => true,
                    None => false,
                };
            }
            let mut args = self
                .f
                .blocks
                .values()
                .flat_map(|b| b.term.edges())
                .filter(|e| e.dst == block)
                .map(|e| e.args[index]);
            let Some(first) = args.next() else {
                return false;
            };
            if !args.all(|arg| arg == first) {
                return false;
            }
            return self.uniform(first);
        }
        if self.effect[value.0] {
            return false;
        }
        if let Some((pure, args)) = self.target[value.0].clone() {
            return pure && args.into_iter().all(|arg| self.uniform(arg));
        }
        let Some(op) = self.defs[value.0] else {
            return false;
        };
        match op {
            Op::Env(Env::LaneId | Env::PacketLaneId | Env::ValidLane) => false,
            Op::Env(_) => true,
            Op::Const(..) => true,
            _ => {
                let mut operands = Vec::new();
                op.map(|v| {
                    operands.push(v);
                    v
                });
                operands.into_iter().all(|v| self.uniform(v))
            }
        }
    }
}

pub(crate) fn run(f: &mut Func, inputs: &[Parameter]) -> usize {
    let mut defs = vec![None; f.types.len()];
    let mut parameter = vec![None; f.types.len()];
    let mut effect = vec![false; f.types.len()];
    let mut target: Vec<Option<(bool, Vec<ValueId>)>> = vec![None; f.types.len()];
    let mut queries: Vec<(ValueId, ValueId)> = Vec::new();
    for (&id, block) in &f.blocks {
        for (index, &(value, _)) in block.params.iter().enumerate() {
            parameter[value.0] = Some((id, index));
        }
        for inst in &block.insts {
            match inst {
                Inst::Core { value, op, .. } => defs[value.0] = Some(*op),
                Inst::Packet { output, .. } => effect[output.0] = true,
                Inst::Target {
                    provenance,
                    args,
                    outputs,
                    ..
                } => {
                    for &(v, _) in outputs {
                        target[v.0] = Some((provenance.is_none(), args.values().to_vec()));
                    }
                }
                Inst::Effect {
                    op,
                    inputs,
                    outputs,
                    ..
                } => {
                    let answered = matches!(
                        op,
                        EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane)
                    );
                    for &(v, _) in outputs {
                        effect[v.0] = !answered;
                    }
                    if matches!(op, EffectOp::Wave(WaveOp::Any | WaveOp::ReadFirstLane)) {
                        queries.push((outputs[0].0, inputs[0]));
                    }
                }
            }
        }
    }
    if queries.is_empty() {
        return 0;
    }
    let mut facts = Facts {
        f,
        defs,
        parameter,
        effect,
        target,
        inputs,
        state: vec![None; f.types.len()],
    };
    let mut renames = std::collections::BTreeMap::new();
    for (result, source) in queries {
        let answer = facts.raw(source);
        if facts.uniform(answer) {
            renames.insert(result, answer);
        }
    }
    if renames.is_empty() {
        return 0;
    }
    for block in f.blocks.values_mut() {
        block.insts.retain(|inst| !matches!(inst, Inst::Effect { outputs, .. } if outputs.iter().any(|(v, _)| renames.contains_key(v))));
    }
    super::simplify::rename(f, &renames);
    let count = renames.len();
    f.compact();
    count
}

pub(crate) struct UniformQueries;
impl Pass for UniformQueries {
    fn name(&self) -> &str {
        "uniform_queries"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        run(f, analyses.context().inputs) > 0
    }
}
