use super::super::analysis::masks::Predication;
use super::super::ir::*;
use super::super::program::{Parameter, ParameterSource};
use super::{Analyses, Pass};

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
        if let Some(known) = self.state[value.0] { return known; }
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
            let mut args = self.f.blocks.values().flat_map(|b| b.term.edges())
                .filter(|e| e.dst == block).map(|e| e.args[index]);
            let Some(first) = args.next() else { return false };
            if !args.all(|arg| arg == first) { return false; }
            return self.uniform(first);
        }
        if self.effect[value.0] { return false; }
        if let Some((pure, args)) = self.target[value.0].clone() {
            return pure && args.into_iter().all(|arg| self.uniform(arg));
        }
        let Some(op) = self.defs[value.0] else { return false };
        match op {
            Op::Env(Env::LaneId | Env::PacketLaneId | Env::ValidLane) => false,
            Op::Env(_) => true,
            Op::Const(..) => true,
            _ => {
                let mut operands = Vec::new();
                op.map(|v| { operands.push(v); v });
                operands.into_iter().all(|v| self.uniform(v))
            }
        }
    }
}

pub(crate) fn run(f: &mut Func, _predication: &Predication, inputs: &[Parameter]) -> usize {
    let mut defs = vec![None; f.types.len()];
    let mut parameter = vec![None; f.types.len()];
    let mut effect = vec![false; f.types.len()];
    let mut target: Vec<Option<(bool, Vec<ValueId>)>> = vec![None; f.types.len()];
    let mut queries: Vec<(ValueId, ValueId)> = Vec::new();
    for (&id, block) in &f.blocks {
        for (index, &(value, _)) in block.params.iter().enumerate() { parameter[value.0] = Some((id, index)); }
        for inst in &block.insts {
            match inst {
                Inst::Core { value, op, .. } => defs[value.0] = Some(*op),
                Inst::Packet { output, .. } => effect[output.0] = true,
                Inst::Target { provenance, args, outputs, .. } => for &(v, _) in outputs {
                    target[v.0] = Some((provenance.is_none(), args.values().to_vec()));
                },
                Inst::Effect { op, inputs, outputs, .. } => {
                    let answered = matches!(op, EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane));
                    for &(v, _) in outputs { effect[v.0] = !answered; }
                    if matches!(op, EffectOp::Wave(WaveOp::Any | WaveOp::ReadFirstLane)) {
                        queries.push((outputs[0].0, inputs[0]));
                    }
                }
            }
        }
    }
    if queries.is_empty() { return 0; }
    let mut facts = Facts { f, defs, parameter, effect, target, inputs, state: vec![None; f.types.len()] };
    let mut renames = std::collections::BTreeMap::new();
    for (result, source) in queries {
        let answer = facts.raw(source);
        if facts.uniform(answer) { renames.insert(result, answer); }
    }
    if renames.is_empty() { return 0; }
    for block in f.blocks.values_mut() {
        block.insts.retain(|inst| !matches!(inst, Inst::Effect { outputs, .. } if outputs.iter().any(|(v, _)| renames.contains_key(v))));
    }
    super::simplify::rename(f, &renames);
    let count = renames.len();
    super::compact(f);
    count
}

pub(crate) struct UniformQueries<'a> { pub inputs: &'a [Parameter] }
impl Pass for UniformQueries<'_> {
    fn name(&self) -> &str { "uniform_queries" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        run(f, analyses.predication(f), self.inputs) > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::ir::{BlockId, Block, Ty, Term, IntOp, Env};
    use super::super::super::program::{Parameter, ParameterSource};
    use std::collections::BTreeMap;

    fn predication(f: &Func) -> Predication {
        let constants = super::super::super::analysis::constants(f);
        super::super::super::analysis::masks::predication(f, 0, &constants)
    }

    fn folded(shape: u8) -> Option<Op> {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let exec = f.value(Ty::I1);
        let old = f.value(Ty::I32);
        let upper = f.value(Ty::I1);
        let uniform = f.value(Ty::I32);
        let mut insts = Vec::new();
        let core = |f: &mut Func, insts: &mut Vec<Inst>, ty, op| { let v = f.value(ty); insts.push(Inst::Core { value: v, ty, op }); v };
        let valid = core(&mut f, &mut insts, Ty::I1, Op::Env(Env::ValidLane));
        let mask = match shape {
            3 => core(&mut f, &mut insts, Ty::I1, Op::Const(Ty::I1, 0)),
            _ => upper,
        };
        let narrow = core(&mut f, &mut insts, Ty::I1, Op::Int(IntOp::And, mask, valid));
        let seven = core(&mut f, &mut insts, Ty::I32, Op::Const(Ty::I32, 7));
        let written = core(&mut f, &mut insts, Ty::I32, Op::Select(narrow, seven, old));
        let reading = if shape == 2 { core(&mut f, &mut insts, Ty::I1, Op::Int(IntOp::And, exec, valid)) } else { narrow };
        let source = if shape == 0 { uniform } else { written };
        let answer = f.value(Ty::I32);
        insts.push(Inst::Effect {
            provenance: 1, op: EffectOp::Wave(WaveOp::ReadFirstLane),
            inputs: vec![source, reading], outputs: vec![(answer, Ty::I32)],
        });
        f.blocks.insert(BlockId(0), Block {
            params: vec![(exec, Ty::I1), (old, Ty::I32), (upper, Ty::I1), (uniform, Ty::I32)],
            insts, term: Term::Ret(vec![answer]) });
        let p = predication(&f);
        let inputs = [
            Parameter { source: ParameterSource::MaskBit(126), ty: Ty::I1 },
            Parameter { source: ParameterSource::Vgpr(1), ty: Ty::I32 },
            Parameter { source: ParameterSource::MaskBit(106), ty: Ty::I1 },
            Parameter { source: ParameterSource::Sgpr(4), ty: Ty::I32 },
        ];
        if run(&mut f, &p, &inputs) == 0 { return None; }
        let Term::Ret(returned) = &f.blocks[&BlockId(0)].term else { panic!("expected a return") };
        let kept = returned[0];
        f.blocks[&BlockId(0)].insts.iter().find_map(|inst| match inst {
            Inst::Core { value, op, .. } if *value == kept => Some(*op),
            _ => None,
        }).or(Some(Op::Env(Env::ValidLane)))
    }

    #[test]
    fn a_readfirstlane_folds_only_when_its_source_is_uniform_without_a_predicate() {
        assert!(folded(0).is_some(), "a wave-uniform dispatch input is the answer for every lane");
        for (shape, why) in [
            (1u8, "the first active lane is the one the write reached, but nothing says so here"),
            (2, "lanes reactivated after the write, so the first active lane may hold the old value"),
            (3, "the mask is empty, so the write never happened and lane 0 holds the old value"),
        ] {
            assert!(folded(shape).is_none(), "{}", why);
        }
    }
}
