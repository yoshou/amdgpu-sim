use super::super::analysis::Analyses;
use super::super::ir::*;
use super::Pass;

struct Facts<'a> {
    f: &'a Func,
    defs: Vec<Option<Op>>,
    parameter: Vec<Option<(BlockId, usize)>>,
    effect: Vec<bool>,
    target: Vec<Option<(bool, Vec<ValueId>)>>,
    answered: Vec<bool>,
    inputs: &'a [Parameter],
    layout: &'a crate::rdna_spmd::engine::EntryLayout,
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
                    Some(ParameterSource::Vgpr(r)) if self.layout.workitem_register(r) => false,
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
        if self.answered[value.0] {
            return true;
        }
        if let Some((pure, args)) = self.target[value.0].clone() {
            return pure && args.into_iter().all(|arg| self.uniform(arg));
        }
        let Some(op) = self.defs[value.0] else {
            return false;
        };
        match op {
            Op::Env(Env::LaneId | Env::ValidLane) => false,
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

fn run(f: &mut Func, inputs: &[Parameter], layout: &crate::rdna_spmd::engine::EntryLayout) -> usize {
    let mut defs = vec![None; f.types.len()];
    let mut parameter = vec![None; f.types.len()];
    let mut effect = vec![false; f.types.len()];
    let mut target: Vec<Option<(bool, Vec<ValueId>)>> = vec![None; f.types.len()];
    let mut answered = vec![false; f.types.len()];
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
                    let query = matches!(
                        op,
                        EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. } | WaveOp::ReadFirstLane)
                    );
                    for &(v, _) in outputs {
                        effect[v.0] = !query;
                        answered[v.0] = query;
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
        answered,
        inputs,
        layout,
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
    f.rename(&renames);
    let count = renames.len();
    f.compact();
    count
}

pub struct UniformQueries;
impl Pass for UniformQueries {
    fn name(&self) -> &str {
        "uniform_queries"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        run(f, analyses.context().inputs, &analyses.context().entry) > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::engine::{EntryLayout, Field};

    fn answered(layout: EntryLayout) -> usize {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let sources = [ParameterSource::Vgpr(0), ParameterSource::Vgpr(1), ParameterSource::MaskBit(126)];
        let types = [Ty::I32, Ty::I32, Ty::I1];
        let params: Vec<_> = types.iter().map(|&t| (f.value(t), t)).collect();
        let first = f.value(Ty::I32);
        f.blocks.insert(
            BlockId(0),
            Block {
                params: params.clone(),
                insts: vec![Inst::Effect {
                    provenance: 0,
                    op: EffectOp::Wave(WaveOp::ReadFirstLane),
                    inputs: vec![params[1].0, params[2].0],
                    outputs: vec![(first, Ty::I32)],
                }],
                term: Term::Ret(vec![first]),
            },
        );
        let inputs: Vec<Parameter> = sources.iter().zip(types).map(|(&source, ty)| Parameter { source, ty }).collect();
        run(&mut f, &inputs, &layout)
    }

    #[test]
    fn a_query_of_a_work_item_id_stays_a_query() {
        let field = |register| Some(Field { register, shift: 0 });
        let separate = EntryLayout {
            workitem_ids: [field(0), field(1), None],
            ..EntryLayout::default()
        };
        assert_eq!(answered(separate), 0, "v1 holds each lane's y");
        let packed = EntryLayout {
            workitem_ids: EntryLayout::PACKED,
            ..EntryLayout::default()
        };
        assert_eq!(answered(packed), 1, "v1 holds no id, so every lane reads the same entry value");
    }
}
