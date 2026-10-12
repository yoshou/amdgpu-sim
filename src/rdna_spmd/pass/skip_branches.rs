use super::super::analysis::Analyses;
use super::super::hash::HashMap;
use super::super::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub struct SkipBranches;
impl super::Pass for SkipBranches {
    fn name(&self) -> &str {
        "skip_branches"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        run(f, analyses.context().exec_index) > 0
    }
}

struct Skip {
    head: BlockId,
    body: BlockId,
    join: BlockId,
    join_on_true: bool,
    exec: ValueId,
}

fn root(defs: &[Option<Op>], types: &[Ty], mut v: ValueId) -> (ValueId, bool) {
    let constant = |x: ValueId| match defs[x.0] {
        Some(Op::Const(_, k)) => Some(k),
        _ => None,
    };
    let mut flipped = false;
    loop {
        match defs[v.0] {
            Some(Op::Convert(Cvt::Bitcast, Ty::I1, x)) if types[x.0] == Ty::I1 => v = x,
            Some(Op::Cmp(pred @ (IntPred::Eq | IntPred::Ne), x, y)) if types[x.0] == Ty::I1 => {
                let (x, k) = match (constant(x), constant(y)) {
                    (_, Some(k)) => (x, k),
                    (Some(k), _) => (y, k),
                    _ => return (v, flipped),
                };
                flipped ^= (pred == IntPred::Eq) == (k == 0);
                v = x;
            }
            Some(Op::Int(IntOp::Xor, x, y)) if types[v.0] == Ty::I1 => {
                let x = match (constant(x), constant(y)) {
                    (_, Some(1)) => x,
                    (Some(1), _) => y,
                    _ => return (v, flipped),
                };
                flipped = !flipped;
                v = x;
            }
            _ => return (v, flipped),
        }
    }
}

fn exec_position(f: &Func, block: &Block, exec_index: usize) -> usize {
    let k = f.blocks[&f.entry].params[..exec_index]
        .iter()
        .filter(|p| p.1 == Ty::I1)
        .count();
    block
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.1 == Ty::I1)
        .nth(k)
        .expect("block lacks its EXEC parameter")
        .0
}

fn zero_test(defs: &[Option<Op>], types: &[Ty], cond: ValueId) -> Option<(ValueId, bool)> {
    let (base, flipped) = root(defs, types, cond);
    let Some(Op::Cmp(pred @ (IntPred::Eq | IntPred::Ne), a, b)) = defs[base.0] else {
        return None;
    };
    let zero = |x: ValueId| matches!(defs[x.0], Some(Op::Const(_, 0)));
    let word = if zero(b) {
        a
    } else if zero(a) {
        b
    } else {
        return None;
    };
    (types[word.0] != Ty::I1).then_some((word, (pred == IntPred::Eq) != flipped))
}

fn lane_bit(defs: &[Option<Op>], v: ValueId) -> Option<ValueId> {
    let valid = |x: ValueId| matches!(defs[x.0], Some(Op::Env(Env::ValidLane)));
    let Some(Op::Int(IntOp::And, a, b)) = defs[v.0] else {
        return None;
    };
    let bit = if valid(b) {
        a
    } else if valid(a) {
        b
    } else {
        return None;
    };
    let Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) = defs[bit.0] else {
        return None;
    };
    let Some(Op::Int(IntOp::LShr, word, lane)) = defs[shifted.0] else {
        return None;
    };
    let lane = match defs[lane.0] {
        Some(Op::Convert(Cvt::ZExt, Ty::I64, x)) => x,
        _ => lane,
    };
    matches!(defs[lane.0], Some(Op::Env(Env::LaneId))).then_some(word)
}

fn speculable(defs: &[Option<Op>], inst: &Inst, exec: ValueId) -> bool {
    let masked = |m: ValueId| m == exec || matches!(defs[m.0], Some(Op::Int(IntOp::And, a, b)) if a == exec || b == exec);
    match inst {
        Inst::Core { .. } => true,
        Inst::Effect {
            op: EffectOp::Memory { op, .. },
            inputs,
            ..
        } => *op != MemoryOp::Fence && masked(inputs[op.mask_input()]),
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. } | WaveOp::ReadFirstLane),
            ..
        } => true,
        _ => false,
    }
}

fn find(f: &Func, exec_index: usize) -> Option<Skip> {
    let defs = f.definitions();
    let mut preds: BTreeMap<BlockId, usize> = BTreeMap::new();
    for block in f.blocks.values() {
        for e in block.term.edges() {
            *preds.entry(e.dst).or_default() += 1;
        }
    }
    for (&head, block) in &f.blocks {
        let Term::CondBr { cond, yes, no } = &block.term else {
            continue;
        };
        let Some((word, join_on_true)) = zero_test(&defs, &f.types, *cond) else {
            continue;
        };
        let (taken, skipped) = if join_on_true { (no, yes) } else { (yes, no) };
        let (body, join) = (taken.dst, skipped.dst);
        if body == join || body == head || join == head || body == f.entry || f.regions.contains_key(&body) || preds[&body] != 1 {
            continue;
        }
        let b = &f.blocks[&body];
        if !matches!(&b.term, Term::Br(exit) if exit.dst == join) {
            continue;
        }
        let k = exec_position(f, b, exec_index);
        let exec = b.params[k].0;
        if lane_bit(&defs, taken.args[k]) != Some(word) || !b.insts.iter().all(|inst| speculable(&defs, inst, exec)) {
            continue;
        }
        return Some(Skip {
            head,
            body,
            join,
            join_on_true,
            exec,
        });
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Leaf(ValueId),
    Ballot(bool, usize),
    Op(Ty, Op),
}

struct Zeroed<'a> {
    defs: &'a [Option<Op>],
    ballots: &'a BTreeMap<ValueId, (bool, ValueId)>,
    local: &'a BTreeSet<ValueId>,
    params: &'a BTreeMap<ValueId, ValueId>,
    types: &'a [Ty],
    exec: ValueId,
    nodes: Vec<Node>,
    ids: HashMap<Node, usize>,
    memo: BTreeMap<ValueId, usize>,
}

impl Zeroed<'_> {
    fn of(&mut self, v: ValueId) -> usize {
        if let Some(&n) = self.memo.get(&v) {
            return n;
        }
        let node = self.node(v);
        let n = match self.ids.get(&node) {
            Some(&n) => n,
            None => {
                self.nodes.push(node);
                self.ids.insert(node, self.nodes.len() - 1);
                self.nodes.len() - 1
            }
        };
        self.memo.insert(v, n);
        n
    }

    fn constant(&self, n: ValueId) -> Option<u64> {
        match self.nodes[n.0] {
            Node::Op(_, Op::Const(_, k)) => Some(k),
            _ => None,
        }
    }

    fn node(&mut self, v: ValueId) -> Node {
        let ty = self.types[v.0];
        let zero = Node::Op(ty, Op::Const(ty, 0));
        if v == self.exec {
            return zero;
        }
        if let Some(&arg) = self.params.get(&v) {
            let n = self.of(arg);
            return self.nodes[n];
        }
        if !self.local.contains(&v) {
            return Node::Leaf(v);
        }
        if let Some(&(high, x)) = self.ballots.get(&v) {
            let x = self.of(x);
            return if self.constant(ValueId(x)) == Some(0) { zero } else { Node::Ballot(high, x) };
        }
        let Some(op) = self.defs[v.0] else {
            return Node::Leaf(v);
        };
        let op = op.map(|x| ValueId(self.of(x)));
        let k = |x: ValueId| self.constant(x);
        let same = |x: ValueId| self.nodes[x.0];
        match op {
            Op::Int(IntOp::And, a, b) if k(a) == Some(0) || k(b) == Some(0) => zero,
            Op::Int(IntOp::Or | IntOp::Xor | IntOp::Add, a, b) if k(a) == Some(0) => same(b),
            Op::Int(IntOp::Or | IntOp::Xor | IntOp::Add | IntOp::Sub | IntOp::Shl | IntOp::LShr | IntOp::AShr, a, b)
                if k(b) == Some(0) =>
            {
                same(a)
            }
            Op::Int(IntOp::Shl | IntOp::LShr | IntOp::AShr, a, _) if k(a) == Some(0) => zero,
            Op::Convert(Cvt::Trunc | Cvt::ZExt | Cvt::SExt | Cvt::Bitcast, _, a)
            | Op::PopulationCount(a)
            | Op::UnpackLo(a)
            | Op::UnpackHi(a)
                if k(a) == Some(0) =>
            {
                zero
            }
            Op::Pack64(a, b) if k(a) == Some(0) && k(b) == Some(0) => zero,
            Op::Select(c, a, b) => match k(c) {
                Some(0) => same(b),
                Some(_) => same(a),
                None if a == b => same(a),
                None => Node::Op(ty, op),
            },
            Op::Cmp(pred @ (IntPred::Eq | IntPred::Ne), a, b) => match (k(a), k(b)) {
                (Some(x), Some(y)) => Node::Op(ty, Op::Const(ty, ((x == y) == (pred == IntPred::Eq)) as u64)),
                _ => Node::Op(ty, op),
            },
            Op::Int(o @ (IntOp::And | IntOp::Or | IntOp::Xor | IntOp::Add | IntOp::Mul), a, b) if a.0 > b.0 => {
                Node::Op(ty, Op::Int(o, b, a))
            }
            _ => Node::Op(ty, op),
        }
    }
}

fn merge(f: &mut Func, s: Skip) {
    let defs = f.definitions();
    let Term::CondBr { cond, yes, no } = f.blocks[&s.head].term.clone() else {
        unreachable!()
    };
    let (taken, skipped) = if s.join_on_true { (no, yes) } else { (yes, no) };
    let body = f.blocks.remove(&s.body).unwrap();
    let Term::Br(exit) = body.term else {
        unreachable!()
    };
    let params: BTreeMap<ValueId, ValueId> = body.params.iter().map(|p| p.0).zip(taken.args).collect();
    let mut local = BTreeSet::new();
    let mut ballots = BTreeMap::new();
    for inst in f.blocks[&s.head].insts.iter().chain(&body.insts) {
        inst.for_each_output(|v| {
            local.insert(v);
        });
        if let Inst::Effect {
            op: EffectOp::Wave(WaveOp::Ballot { high }),
            inputs,
            outputs,
            ..
        } = inst
        {
            ballots.insert(outputs[0].0, (*high, inputs[0]));
        }
    }
    let agree: Vec<bool> = {
        let mut zeroed = Zeroed {
            defs: &defs,
            ballots: &ballots,
            local: &local,
            params: &params,
            types: &f.types,
            exec: s.exec,
            nodes: Vec::new(),
            ids: HashMap::default(),
            memo: BTreeMap::new(),
        };
        skipped
            .args
            .iter()
            .zip(&exit.args)
            .map(|(&skip, &through)| zeroed.of(skip) == zeroed.of(through))
            .collect()
    };
    let mut insts = body.insts;
    let mut args = Vec::with_capacity(exit.args.len());
    for ((&skip, &through), agree) in skipped.args.iter().zip(&exit.args).zip(agree) {
        let through = params.get(&through).copied().unwrap_or(through);
        if skip == through || agree {
            args.push(through);
            continue;
        }
        let ty = f.types[skip.0];
        let value = f.value(ty);
        let (a, b) = if s.join_on_true { (skip, through) } else { (through, skip) };
        insts.push(Inst::Core {
            value,
            ty,
            op: Op::Select(cond, a, b),
        });
        args.push(value);
    }
    let head = f.blocks.get_mut(&s.head).unwrap();
    head.insts.extend(insts);
    head.term = Term::Br(Edge { dst: s.join, args });
    f.rename(&params);
}

fn run(f: &mut Func, exec_index: usize) -> usize {
    let mut merged = 0;
    while let Some(skip) = find(f, exec_index) {
        merge(f, skip);
        merged += 1;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    fn scratch(op: MemoryOp) -> EffectOp {
        EffectOp::Memory {
            space: Space::Scratch,
            op,
            semantics: MemorySemantics {
                scope: Scope::WorkItem,
                ordering: Ordering::Relaxed,
                cache_policy: CachePolicy::Temporal,
                volatile: false,
                deferred_scope: false,
            },
        }
    }

    struct Builder {
        f: Func,
        next: u64,
        lane: ValueId,
        valid: ValueId,
        shift: ValueId,
        none: ValueId,
    }

    impl Builder {
        fn new(lanes: u32, start: &mut Vec<Inst>) -> Self {
            let mut f = Func::new(BlockId(0), Presence::Wave, lanes);
            let lane = core(&mut f, start, Ty::I32, Op::Env(Env::LaneId));
            let valid = core(&mut f, start, Ty::I1, Op::Env(Env::ValidLane));
            let (shift, wide) = if lanes == 32 {
                (lane, Ty::I32)
            } else {
                (core(&mut f, start, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane)), Ty::I64)
            };
            let none = core(&mut f, start, wide, Op::Const(wide, 0));
            Self {
                f,
                next: 1,
                lane,
                valid,
                shift,
                none,
            }
        }

        fn effect(&mut self, insts: &mut Vec<Inst>, op: EffectOp, inputs: Vec<ValueId>, outputs: Vec<(ValueId, Ty)>) {
            insts.push(Inst::Effect {
                provenance: self.next << 8,
                op,
                inputs,
                outputs,
            });
            self.next += 1;
        }

        fn store(&mut self, insts: &mut Vec<Inst>, address: ValueId, data: ValueId, mask: ValueId) {
            self.effect(insts, scratch(MemoryOp::Store(MemSize::B32)), vec![address, data, mask], Vec::new());
        }

        fn load(&mut self, insts: &mut Vec<Inst>, address: ValueId, mask: ValueId) -> ValueId {
            let v = self.f.value(Ty::I32);
            self.effect(insts, scratch(MemoryOp::Load(MemSize::B32)), vec![address, mask], vec![(v, Ty::I32)]);
            v
        }

        fn half(&mut self, insts: &mut Vec<Inst>, bit: ValueId, high: bool) -> ValueId {
            let v = self.f.value(Ty::I32);
            self.effect(insts, EffectOp::Wave(WaveOp::Ballot { high }), vec![bit], vec![(v, Ty::I32)]);
            v
        }

        fn ballot(&mut self, insts: &mut Vec<Inst>, bit: ValueId) -> ValueId {
            let low = self.half(insts, bit, false);
            if self.f.lanes == 32 {
                return low;
            }
            let high = self.half(insts, bit, true);
            core(&mut self.f, insts, Ty::I64, Op::Pack64(low, high))
        }

        fn own(&mut self, insts: &mut Vec<Inst>, word: ValueId) -> ValueId {
            let ty = self.f.types[word.0];
            let shifted = core(&mut self.f, insts, ty, Op::Int(IntOp::LShr, word, self.shift));
            let bit = core(&mut self.f, insts, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
            core(&mut self.f, insts, Ty::I1, Op::Int(IntOp::And, bit, self.valid))
        }
    }

    struct Diamond {
        head: Vec<Inst>,
        cond: ValueId,
        body_first: bool,
        word: ValueId,
        saved: ValueId,
    }

    fn test_word(b: &mut Builder, head: &mut Vec<Inst>, values: &[ValueId], exec: ValueId, r: &mut Rng) -> Diamond {
        let limit = core(&mut b.f, head, Ty::I32, Op::Const(Ty::I32, r.below(64) as u64));
        let pred = [IntPred::Ult, IntPred::Ugt, IntPred::Eq, IntPred::Ne][r.below(4)];
        let p = core(&mut b.f, head, Ty::I1, Op::Cmp(pred, values[r.below(values.len())], limit));
        let t = core(&mut b.f, head, Ty::I1, Op::Int(IntOp::And, p, exec));
        let word = b.ballot(head, t);
        let saved = b.ballot(head, exec);
        let nonzero = core(&mut b.f, head, Ty::I1, Op::Cmp(IntPred::Ne, word, b.none));
        let (cond, body_first) = match r.below(3) {
            0 => {
                let off = core(&mut b.f, head, Ty::I1, Op::Const(Ty::I1, 0));
                (core(&mut b.f, head, Ty::I1, Op::Cmp(IntPred::Eq, nonzero, off)), false)
            }
            1 => (nonzero, true),
            _ => {
                let on = core(&mut b.f, head, Ty::I1, Op::Const(Ty::I1, 1));
                (core(&mut b.f, head, Ty::I1, Op::Int(IntOp::Xor, nonzero, on)), false)
            }
        };
        Diamond {
            head: std::mem::take(head),
            cond,
            body_first,
            word,
            saved,
        }
    }

    fn edges(d: &Diamond, body: Edge, join: Edge) -> Term {
        let (yes, no) = if d.body_first { (body, join) } else { (join, body) };
        Term::CondBr { cond: d.cond, yes, no }
    }

    fn random_program(r: &mut Rng, lanes: u32, diamonds: usize) -> Func {
        let mut start = Vec::new();
        let mut b = Builder::new(lanes, &mut start);
        let x = b.f.value(Ty::I32);
        let exec = b.f.value(Ty::I1);
        let slots: Vec<ValueId> = (0..3)
            .map(|k| core(&mut b.f, &mut start, Ty::I32, Op::Const(Ty::I32, 4 * k)))
            .collect();
        let on = core(&mut b.f, &mut start, Ty::I1, Op::Const(Ty::I1, 1));
        let first = vec![x, b.lane, x, exec];
        b.f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(x, Ty::I32), (exec, Ty::I1)],
                insts: start,
                term: Term::Br(Edge { dst: BlockId(1), args: first }),
            },
        );
        let mut id = 1;
        for _ in 0..diamonds {
            let (head, body, join) = (BlockId(id), BlockId(id + 1), BlockId(id + 2));
            id += 2;
            let carried: Vec<ValueId> = (0..3).map(|_| b.f.value(Ty::I32)).collect();
            let e = b.f.value(Ty::I1);
            let mut insts = Vec::new();
            let mut d = test_word(&mut b, &mut insts, &carried, e, r);
            let bit = b.own(&mut d.head, d.word);
            let restored_skip = if r.below(2) == 0 { e } else { b.own(&mut d.head, d.saved) };
            let inner: Vec<ValueId> = (0..3).map(|_| b.f.value(Ty::I32)).collect();
            let eb = b.f.value(Ty::I1);
            let mut code = Vec::new();
            let mut pool = inner.clone();
            for _ in 0..1 + r.below(6) {
                let (u, w) = (pool[r.below(pool.len())], pool[r.below(pool.len())]);
                match r.below(7) {
                    6 => {
                        let c = core(&mut b.f, &mut code, Ty::I1, Op::Cmp(IntPred::Ult, u, w));
                        pool.push(core(&mut b.f, &mut code, Ty::I32, Op::Select(c, u, w)));
                    }
                    0 => {
                        let op = [IntOp::Add, IntOp::Xor, IntOp::Sub, IntOp::And, IntOp::Or][r.below(5)];
                        pool.push(core(&mut b.f, &mut code, Ty::I32, Op::Int(op, u, w)));
                    }
                    5 => {
                        let k = core(&mut b.f, &mut code, Ty::I32, Op::Const(Ty::I32, r.below(3) as u64));
                        let op = [IntOp::Shl, IntOp::LShr][r.below(2)];
                        pool.push(core(&mut b.f, &mut code, Ty::I32, Op::Int(op, u, k)));
                    }
                    1 => pool.push(core(&mut b.f, &mut code, Ty::I32, Op::Select(eb, u, w))),
                    2 => b.store(&mut code, slots[r.below(3)], u, eb),
                    3 => {
                        let v = b.load(&mut code, slots[r.below(3)], eb);
                        pool.push(core(&mut b.f, &mut code, Ty::I32, Op::Select(eb, v, u)));
                    }
                    _ => {
                        let c = core(&mut b.f, &mut code, Ty::I1, Op::Cmp(IntPred::Ult, u, w));
                        let m = core(&mut b.f, &mut code, Ty::I1, Op::Int(IntOp::And, c, eb));
                        pool.push(b.half(&mut code, m, false));
                    }
                }
            }
            let mask = b.ballot(&mut code, eb);
            let wide = b.f.types[mask.0];
            let both = core(&mut b.f, &mut code, wide, Op::Int(IntOp::Or, mask, d.saved));
            let restored = b.own(&mut code, both);
            let mut through: Vec<ValueId> = (0..3).map(|_| pool[r.below(pool.len())]).collect();
            through.push(restored);
            let mut taken = carried.clone();
            taken.push(bit);
            let nil = core(&mut b.f, &mut d.head, Ty::I32, Op::Const(Ty::I32, 0));
            let mut skipped: Vec<ValueId> = (0..3)
                .map(|i| match r.below(4) {
                    0 => nil,
                    1 => carried[r.below(3)],
                    _ => carried[i],
                })
                .collect();
            skipped.push(restored_skip);
            let term = edges(&d, Edge { dst: body, args: taken }, Edge { dst: join, args: skipped });
            let mut head_params: Vec<(ValueId, Ty)> = carried.iter().map(|&v| (v, Ty::I32)).collect();
            head_params.push((e, Ty::I1));
            b.f.blocks.insert(
                head,
                Block {
                    params: head_params,
                    insts: d.head,
                    term,
                },
            );
            let mut body_params: Vec<(ValueId, Ty)> = inner.iter().map(|&v| (v, Ty::I32)).collect();
            body_params.push((eb, Ty::I1));
            b.f.blocks.insert(
                body,
                Block {
                    params: body_params,
                    insts: code,
                    term: Term::Br(Edge { dst: join, args: through }),
                },
            );
        }
        let carried: Vec<ValueId> = (0..3).map(|_| b.f.value(Ty::I32)).collect();
        let e = b.f.value(Ty::I1);
        let mut last = Vec::new();
        let mut out: Vec<ValueId> = carried.clone();
        out.push(e);
        for &slot in &slots {
            out.push(b.load(&mut last, slot, on));
        }
        let mut params: Vec<(ValueId, Ty)> = carried.iter().map(|&v| (v, Ty::I32)).collect();
        params.push((e, Ty::I1));
        b.f.blocks.insert(
            BlockId(id),
            Block {
                params,
                insts: last,
                term: Term::Ret(out),
            },
        );
        b.f
    }

    fn inputs(r: &mut Rng, lanes: u32) -> Vec<Vec<u64>> {
        let lanes = lanes as usize;
        let x = (0..lanes).map(|_| r.below(80) as u64).collect();
        let exec = match r.below(4) {
            0 => vec![0; lanes],
            1 => vec![1; lanes],
            _ => (0..lanes).map(|_| r.below(2) as u64).collect(),
        };
        vec![x, exec]
    }

    #[test]
    fn branches_that_skip_when_no_lane_is_active_become_straight_code_with_the_same_results() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        for lanes in [32, 64] {
            for _ in 0..200 {
                let diamonds = 1 + r.below(4);
                let before = random_program(&mut r, lanes, diamonds);
                let mut after = before.clone();
                assert_eq!(run(&mut after, 1), diamonds);
                assert!(after.blocks.values().all(|b| !matches!(b.term, Term::CondBr { .. })));
                for _ in 0..8 {
                    let entry = inputs(&mut r, lanes);
                    let want = simulate(&before, &entry, &mut |_, _| {});
                    let got = simulate(&after, &entry, &mut |_, _| {});
                    assert_eq!(got, want);
                }
            }
        }
    }

    fn single(
        lanes: u32,
        mask_of_store: impl Fn(ValueId, ValueId) -> ValueId,
        exec_of_body: impl Fn(&mut Builder, &mut Vec<Inst>, &Diamond) -> ValueId,
    ) -> Func {
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let mut start = Vec::new();
        let mut b = Builder::new(lanes, &mut start);
        let x = b.f.value(Ty::I32);
        let exec = b.f.value(Ty::I1);
        let slot = core(&mut b.f, &mut start, Ty::I32, Op::Const(Ty::I32, 0));
        let on = core(&mut b.f, &mut start, Ty::I1, Op::Const(Ty::I1, 1));
        b.f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(x, Ty::I32), (exec, Ty::I1)],
                insts: start,
                term: Term::Br(Edge {
                    dst: BlockId(1),
                    args: vec![x, exec],
                }),
            },
        );
        let carried = b.f.value(Ty::I32);
        let e = b.f.value(Ty::I1);
        let mut insts = Vec::new();
        let mut d = test_word(&mut b, &mut insts, &[carried], e, &mut r);
        let bit = exec_of_body(&mut b, &mut insts, &d);
        d.head.extend(insts);
        let restored_skip = b.own(&mut d.head, d.saved);
        let inner = b.f.value(Ty::I32);
        let eb = b.f.value(Ty::I1);
        let mut code = Vec::new();
        let one = core(&mut b.f, &mut code, Ty::I32, Op::Const(Ty::I32, 1));
        let bumped = core(&mut b.f, &mut code, Ty::I32, Op::Int(IntOp::Add, inner, one));
        let kept = core(&mut b.f, &mut code, Ty::I32, Op::Select(eb, bumped, inner));
        b.store(&mut code, slot, kept, mask_of_store(eb, on));
        let mask = b.ballot(&mut code, eb);
        let wide = b.f.types[mask.0];
        let both = core(&mut b.f, &mut code, wide, Op::Int(IntOp::Or, mask, d.saved));
        let restored = b.own(&mut code, both);
        let term = edges(
            &d,
            Edge {
                dst: BlockId(2),
                args: vec![carried, bit],
            },
            Edge {
                dst: BlockId(3),
                args: vec![carried, carried, restored_skip],
            },
        );
        b.f.blocks.insert(
            BlockId(1),
            Block {
                params: vec![(carried, Ty::I32), (e, Ty::I1)],
                insts: d.head,
                term,
            },
        );
        b.f.blocks.insert(
            BlockId(2),
            Block {
                params: vec![(inner, Ty::I32), (eb, Ty::I1)],
                insts: code,
                term: Term::Br(Edge {
                    dst: BlockId(3),
                    args: vec![kept, bumped, restored],
                }),
            },
        );
        let (out, raw, last_exec) = (b.f.value(Ty::I32), b.f.value(Ty::I32), b.f.value(Ty::I1));
        let mut last = Vec::new();
        let read = b.load(&mut last, slot, on);
        b.f.blocks.insert(
            BlockId(3),
            Block {
                params: vec![(out, Ty::I32), (raw, Ty::I32), (last_exec, Ty::I1)],
                insts: last,
                term: Term::Ret(vec![out, raw, last_exec, read]),
            },
        );
        b.f
    }

    #[test]
    fn a_join_selects_only_the_values_an_empty_mask_would_change() {
        let mut r = Rng(0x0123_4567_89ab_cdef);
        for lanes in [32, 64] {
            let before = single(lanes, |eb, _| eb, |b, insts, d| b.own(insts, d.word));
            let Term::CondBr { cond, .. } = &before.blocks[&BlockId(1)].term else {
                panic!("the program starts with a skip branch")
            };
            let mut f = before.clone();
            assert_eq!(run(&mut f, 1), 1);
            let Term::Br(edge) = &f.blocks[&BlockId(1)].term else {
                panic!("the skip branch stays")
            };
            let defs = f.definitions();
            let chosen = |v: ValueId| matches!(defs[v.0], Some(Op::Select(c, ..)) if c == *cond);
            assert!(!chosen(edge.args[0]), "a write masked by the body needs no select");
            assert!(chosen(edge.args[1]), "a value the body computes whatever its mask keeps its select");
            assert!(!chosen(edge.args[2]), "the restored mask needs no select");
            for _ in 0..16 {
                let entry = inputs(&mut r, lanes);
                assert_eq!(simulate(&f, &entry, &mut |_, _| {}), simulate(&before, &entry, &mut |_, _| {}));
            }
        }
    }

    #[test]
    fn a_body_that_writes_whatever_the_mask_keeps_its_branch() {
        for lanes in [32, 64] {
            let mut f = single(lanes, |_, on| on, |b, insts, d| b.own(insts, d.word));
            assert_eq!(run(&mut f, 1), 0);
        }
    }

    #[test]
    fn a_body_whose_mask_is_not_the_tested_word_keeps_its_branch() {
        for lanes in [32, 64] {
            let mut f = single(lanes, |eb, _| eb, |b, insts, d| b.own(insts, d.saved));
            assert_eq!(run(&mut f, 1), 0);
        }
    }
}
