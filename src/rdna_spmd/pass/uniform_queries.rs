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
}

impl<'a> Facts<'a> {
    fn new(
        f: &'a Func,
        inputs: &'a [Parameter],
        layout: &'a crate::rdna_spmd::engine::EntryLayout,
    ) -> (Self, Vec<(ValueId, ValueId)>) {
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
        let facts = Facts {
            f,
            defs,
            parameter,
            effect,
            target,
            answered,
            inputs,
            layout,
        };
        (facts, queries)
    }

    fn raw(&self, mut value: ValueId) -> ValueId {
        loop {
            match self.defs[value.0] {
                Some(Op::Convert(Cvt::Bitcast, to, a)) if self.f.types[a.0] == to => value = a,
                _ => return value,
            }
        }
    }

    fn incoming(&self) -> Vec<Vec<ValueId>> {
        let mut incoming: Vec<Vec<ValueId>> = vec![Vec::new(); self.f.types.len()];
        for block in self.f.blocks.values() {
            for edge in block.term.edges() {
                let params = &self.f.blocks[&edge.dst].params;
                for (&(param, _), &arg) in params.iter().zip(&edge.args) {
                    incoming[param.0].push(arg);
                }
            }
        }
        incoming
    }

    fn uniform(&self) -> Vec<bool> {
        let n = self.f.types.len();
        let incoming = self.incoming();
        let mut users: Vec<Vec<ValueId>> = vec![Vec::new(); n];
        for (param, args) in incoming.iter().enumerate() {
            for arg in args {
                users[arg.0].push(ValueId(param));
            }
        }
        for v in 0..n {
            let operands = match (&self.target[v], self.defs[v]) {
                (Some((_, args)), _) => args.clone(),
                (None, Some(op)) => {
                    let mut operands = Vec::new();
                    op.map(|x| {
                        operands.push(x);
                        x
                    });
                    operands
                }
                (None, None) => Vec::new(),
            };
            for x in operands {
                users[x.0].push(ValueId(v));
            }
        }
        let mut uniform = vec![true; n];
        let mut pending: Vec<ValueId> = (0..n).map(ValueId).collect();
        while let Some(v) = pending.pop() {
            if uniform[v.0] && !self.judge(v, &uniform, &incoming) {
                uniform[v.0] = false;
                pending.extend(users[v.0].iter().copied());
            }
        }
        uniform
    }

    fn judge(&self, value: ValueId, uniform: &[bool], incoming: &[Vec<ValueId>]) -> bool {
        if let Some((block, index)) = self.parameter[value.0] {
            if block == self.f.entry {
                return match self.inputs.get(index).map(|p| p.source) {
                    Some(ParameterSource::Vgpr(r)) if self.layout.workitem_register(r) => false,
                    Some(ParameterSource::MaskBit(_)) => false,
                    Some(_) => true,
                    None => false,
                };
            }
            let args = &incoming[value.0];
            return !args.is_empty() && args.iter().all(|a| uniform[a.0]);
        }
        if self.effect[value.0] {
            return false;
        }
        if self.answered[value.0] {
            return true;
        }
        if let Some((pure, args)) = &self.target[value.0] {
            return *pure && args.iter().all(|a| uniform[a.0]);
        }
        let Some(op) = self.defs[value.0] else {
            return false;
        };
        match op {
            Op::Env(Env::LaneId | Env::ValidLane) => false,
            Op::Env(_) => true,
            Op::Const(..) => true,
            _ => {
                let mut all = true;
                op.map(|v| {
                    all &= uniform[v.0];
                    v
                });
                all
            }
        }
    }
}

fn run(f: &mut Func, inputs: &[Parameter], layout: &crate::rdna_spmd::engine::EntryLayout) -> usize {
    let (facts, queries) = Facts::new(f, inputs, layout);
    if queries.is_empty() {
        return 0;
    }
    let uniform = facts.uniform();
    let mut renames = std::collections::BTreeMap::new();
    for (result, source) in queries {
        let answer = facts.raw(source);
        if uniform[answer.0] {
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

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    fn inputs() -> Vec<Parameter> {
        [
            (ParameterSource::Sgpr(0), Ty::I32),
            (ParameterSource::Vgpr(0), Ty::I32),
            (ParameterSource::MaskBit(126), Ty::I1),
        ]
        .iter()
        .map(|&(source, ty)| Parameter { source, ty })
        .collect()
    }

    fn layout() -> EntryLayout {
        EntryLayout {
            workitem_ids: [Some(Field { register: 0, shift: 0 }), None, None],
            ..EntryLayout::default()
        }
    }

    fn core(f: &mut Func, insts: &mut Vec<Inst>, ty: Ty, op: Op) -> ValueId {
        let value = f.value(ty);
        insts.push(Inst::Core { value, ty, op });
        value
    }

    fn any(f: &mut Func, insts: &mut Vec<Inst>, c: ValueId) -> ValueId {
        let q = f.value(Ty::I1);
        insts.push(Inst::Effect {
            provenance: 0,
            op: EffectOp::Wave(WaveOp::Any),
            inputs: vec![c],
            outputs: vec![(q, Ty::I1)],
        });
        q
    }

    fn looped(build: impl FnOnce(&mut Func, &mut Vec<Inst>, [ValueId; 4], ValueId, &[ValueId]) -> Vec<ValueId>, carried: usize) -> Func {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (entry, header, exit) = (BlockId(0), BlockId(1), BlockId(2));
        let s = f.value(Ty::I32);
        let x = f.value(Ty::I32);
        let exec = f.value(Ty::I1);
        let mut start = Vec::new();
        let zero = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 0));
        let lane = core(&mut f, &mut start, Ty::I32, Op::Env(Env::LaneId));
        let counter = f.value(Ty::I32);
        let params: Vec<ValueId> = (0..carried).map(|_| f.value(Ty::I32)).collect();
        let mut body = Vec::new();
        let mut back = build(&mut f, &mut body, [zero, s, x, lane], counter, &params);
        let one = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 1));
        let next = core(&mut f, &mut body, Ty::I32, Op::Int(IntOp::Add, counter, one));
        let limit = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 4));
        let more = core(&mut f, &mut body, Ty::I1, Op::Cmp(IntPred::Ult, next, limit));
        let first: Vec<ValueId> = back.drain(carried..).collect();
        let starts: Vec<ValueId> = std::iter::once(zero).chain(first).collect();
        let backs: Vec<ValueId> = std::iter::once(next).chain(back).collect();
        let header_params = std::iter::once(counter).chain(params).map(|v| (v, Ty::I32)).collect();
        f.blocks.insert(
            entry,
            Block {
                params: vec![(s, Ty::I32), (x, Ty::I32), (exec, Ty::I1)],
                insts: start,
                term: Term::Br(Edge { dst: header, args: starts }),
            },
        );
        f.blocks.insert(
            header,
            Block {
                params: header_params,
                insts: body,
                term: Term::CondBr {
                    cond: more,
                    yes: Edge { dst: header, args: backs },
                    no: Edge { dst: exit, args: Vec::new() },
                },
            },
        );
        f.blocks.insert(
            exit,
            Block {
                params: Vec::new(),
                insts: Vec::new(),
                term: Term::Ret(Vec::new()),
            },
        );
        f
    }

    fn random_loop(r: &mut Rng) -> Func {
        let carried = 1 + r.below(3);
        let choices: Vec<usize> = (0..64).map(|_| r.below(1 << 16)).collect();
        looped(
            |f, body, outside, counter, params| {
                let mut pick = choices.into_iter();
                let mut pool: Vec<ValueId> = outside.iter().copied().chain(std::iter::once(counter)).chain(params.iter().copied()).collect();
                for _ in 0..8 {
                    let a = pool[pick.next().unwrap() % pool.len()];
                    let b = pool[pick.next().unwrap() % pool.len()];
                    let v = match pick.next().unwrap() % 5 {
                        0 => core(f, body, Ty::I32, Op::Int(IntOp::Add, a, b)),
                        1 => core(f, body, Ty::I32, Op::Int(IntOp::And, a, b)),
                        2 => core(f, body, Ty::I32, Op::Int(IntOp::Xor, a, b)),
                        3 => {
                            let c = core(f, body, Ty::I1, Op::Cmp(IntPred::Ult, a, b));
                            let q = any(f, body, c);
                            core(f, body, Ty::I32, Op::Select(q, a, b))
                        }
                        _ => {
                            let c = core(f, body, Ty::I1, Op::Cmp(IntPred::Ult, a, b));
                            core(f, body, Ty::I32, Op::Select(c, a, b))
                        }
                    };
                    pool.push(v);
                }
                let mut ends: Vec<ValueId> = (0..params.len()).map(|_| pool[pick.next().unwrap() % pool.len()]).collect();
                ends.extend((0..params.len()).map(|_| outside[pick.next().unwrap() % outside.len()]));
                ends
            },
            carried,
        )
    }

    fn naive(facts: &Facts) -> Vec<bool> {
        let incoming = facts.incoming();
        let mut uniform = vec![true; facts.f.types.len()];
        loop {
            let mut changed = false;
            for v in 0..uniform.len() {
                if uniform[v] && !facts.judge(ValueId(v), &uniform, &incoming) {
                    uniform[v] = false;
                    changed = true;
                }
            }
            if !changed {
                return uniform;
            }
        }
    }

    fn claimed(uniform: &[bool], values: &[Vec<u64>], v: ValueId) {
        if uniform[v.0] {
            assert!(values[v.0].iter().all(|&x| x == values[v.0][0]), "v{} is judged uniform but its lanes differ", v.0);
        }
    }

    fn simulate(f: &Func, uniform: &[bool], lanes: usize) {
        let mut values: Vec<Vec<u64>> = vec![Vec::new(); f.types.len()];
        let entry = &f.blocks[&f.entry];
        values[entry.params[0].0 .0] = vec![7; lanes];
        values[entry.params[1].0 .0] = (0..lanes as u64).map(|l| 3 * l + 1).collect();
        values[entry.params[2].0 .0] = vec![1; lanes];
        for &(p, _) in &entry.params {
            claimed(uniform, &values, p);
        }
        let mut block = f.entry;
        for _ in 0..64 {
            let b = &f.blocks[&block];
            for inst in &b.insts {
                let out = match inst {
                    Inst::Core { value, ty, op } => {
                        let mask = if *ty == Ty::I1 { 1 } else { 0xffff_ffff };
                        let lane = |l: usize| {
                            let g = |v: ValueId| values[v.0][l];
                            let x = match *op {
                                Op::Const(_, k) => k,
                                Op::Env(Env::LaneId) => l as u64,
                                Op::Int(IntOp::Add, a, b) => g(a).wrapping_add(g(b)),
                                Op::Int(IntOp::And, a, b) => g(a) & g(b),
                                Op::Int(IntOp::Xor, a, b) => g(a) ^ g(b),
                                Op::Cmp(IntPred::Ult, a, b) => (g(a) < g(b)) as u64,
                                Op::Select(c, a, b) => if g(c) != 0 { g(a) } else { g(b) },
                                other => panic!("an operation the simulation lacks: {:?}", other),
                            };
                            x & mask
                        };
                        (*value, (0..lanes).map(lane).collect())
                    }
                    Inst::Effect {
                        op: EffectOp::Wave(WaveOp::Any),
                        inputs,
                        outputs,
                        ..
                    } => {
                        let set = values[inputs[0].0].iter().any(|&x| x != 0) as u64;
                        (outputs[0].0, vec![set; lanes])
                    }
                    other => panic!("an instruction the simulation lacks: {:?}", other),
                };
                values[out.0 .0] = out.1;
                claimed(uniform, &values, out.0);
            }
            let edge = match &b.term {
                Term::Ret(_) => return,
                Term::Br(e) => e,
                Term::CondBr { cond, yes, no } => {
                    let c = &values[cond.0];
                    assert!(c.iter().all(|&x| x == c[0]), "the wave branches on a bit every lane shares");
                    if c[0] != 0 {
                        yes
                    } else {
                        no
                    }
                }
            };
            let args: Vec<Vec<u64>> = edge.args.iter().map(|a| values[a.0].clone()).collect();
            for (&(p, _), arg) in f.blocks[&edge.dst].params.iter().zip(args) {
                values[p.0] = arg;
                claimed(uniform, &values, p);
            }
            block = edge.dst;
        }
        panic!("the loop does not end");
    }

    #[test]
    fn values_judged_uniform_agree_in_every_lane_of_random_loops() {
        let (inputs, layout) = (inputs(), layout());
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut answered, mut kept) = (0, 0);
        for _ in 0..500 {
            let f = random_loop(&mut r);
            let (facts, queries) = Facts::new(&f, &inputs, &layout);
            let uniform = facts.uniform();
            assert_eq!(uniform, naive(&facts), "the worklist stops at the greatest fixpoint of the rules");
            simulate(&f, &uniform, 64);
            for (_, source) in queries {
                if uniform[facts.raw(source).0] {
                    answered += 1;
                } else {
                    kept += 1;
                }
            }
        }
        assert!(answered > 0 && kept > 0, "the loops answer some queries and keep others: {} and {}", answered, kept);
    }

    #[test]
    fn a_query_of_a_counter_carried_around_a_loop_is_answered() {
        let mut f = looped(
            |f, body, [_, s, _, _], counter, _| {
                let c = core(f, body, Ty::I1, Op::Cmp(IntPred::Ult, counter, s));
                any(f, body, c);
                Vec::new()
            },
            0,
        );
        assert_eq!(run(&mut f, &inputs(), &layout()), 1, "the counter starts at zero and adds one, so every lane holds it");
    }

    #[test]
    fn a_query_of_a_value_a_lane_feeds_around_a_loop_stays_a_query() {
        let mut f = looped(
            |f, body, [zero, s, _, lane], _, params| {
                let c = core(f, body, Ty::I1, Op::Cmp(IntPred::Ult, params[0], s));
                any(f, body, c);
                vec![params[1], lane, zero, zero]
            },
            2,
        );
        assert_eq!(run(&mut f, &inputs(), &layout()), 0, "the lane id reaches the first parameter through the second one trip later");
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
