use super::super::analysis::Analyses;
use super::super::hash::HashMap;
use super::super::ir::*;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

pub struct Halves;
impl super::Pass for Halves {
    fn name(&self) -> &str {
        "halves"
    }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        run(f) > 0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Claim {
    Equal(ValueId, ValueId),
    Symmetric(ValueId),
}

fn equal(a: ValueId, b: ValueId) -> Claim {
    Claim::Equal(a.min(b), a.max(b))
}

struct Program<'a> {
    f: &'a Func,
    defs: Vec<Option<Op>>,
    param: Vec<Option<(BlockId, usize)>>,
    incoming: BTreeMap<BlockId, Vec<&'a Edge>>,
}

impl<'a> Program<'a> {
    fn new(f: &'a Func) -> Self {
        let mut param = vec![None; f.types.len()];
        let mut incoming: BTreeMap<BlockId, Vec<&Edge>> = BTreeMap::new();
        for (&id, block) in &f.blocks {
            for (index, &(v, _)) in block.params.iter().enumerate() {
                param[v.0] = Some((id, index));
            }
            for edge in block.term.edges() {
                incoming.entry(edge.dst).or_default().push(edge);
            }
        }
        Self {
            f,
            defs: f.definitions(),
            param,
            incoming,
        }
    }

    fn edges(&self, block: BlockId) -> &[&'a Edge] {
        self.incoming.get(&block).map_or(&[], |e| e.as_slice())
    }

    fn premises(&self, claim: Claim) -> Option<Vec<Claim>> {
        match claim {
            Claim::Equal(a, b) => self.equal_premises(a, b),
            Claim::Symmetric(x) => self.symmetric_premises(x),
        }
    }

    fn equal_premises(&self, a: ValueId, b: ValueId) -> Option<Vec<Claim>> {
        if a == b {
            return Some(Vec::new());
        }
        if self.f.types[a.0] != self.f.types[b.0] {
            return None;
        }
        if let (Some((ba, ia)), Some((bb, ib))) = (self.param[a.0], self.param[b.0]) {
            if ba != bb || ba == self.f.entry {
                return None;
            }
            return Some(
                self.edges(ba)
                    .iter()
                    .filter(|e| e.args[ia] != e.args[ib])
                    .map(|e| equal(e.args[ia], e.args[ib]))
                    .collect(),
            );
        }
        let (Some(x), Some(y)) = (self.defs[a.0], self.defs[b.0]) else {
            return None;
        };
        let pairs = |claims: &[(ValueId, ValueId)]| -> Option<Vec<Claim>> {
            Some(claims.iter().filter(|(p, q)| p != q).map(|&(p, q)| equal(p, q)).collect())
        };
        match (x, y) {
            (Op::Const(t, k), Op::Const(u, l)) => (t == u && k == l).then(Vec::new),
            (Op::UnpackLo(p), Op::UnpackHi(q)) | (Op::UnpackHi(q), Op::UnpackLo(p)) if p == q => {
                Some(vec![Claim::Symmetric(p)])
            }
            (Op::Int(k, p1, p2), Op::Int(l, q1, q2)) if k == l => pairs(&[(p1, q1), (p2, q2)]),
            (Op::Select(c, p1, p2), Op::Select(d, q1, q2)) if c == d => pairs(&[(p1, q1), (p2, q2)]),
            (Op::Convert(k, t, p), Op::Convert(l, u, q)) if k == l && t == u => pairs(&[(p, q)]),
            _ => None,
        }
    }

    fn symmetric_premises(&self, x: ValueId) -> Option<Vec<Claim>> {
        if self.f.types[x.0] != Ty::I64 {
            return None;
        }
        if let Some((block, index)) = self.param[x.0] {
            if block == self.f.entry {
                return None;
            }
            return Some(self.edges(block).iter().map(|e| Claim::Symmetric(e.args[index])).collect());
        }
        match self.defs[x.0]? {
            Op::Const(_, k) => (k & 0xffff_ffff == k >> 32).then(Vec::new),
            Op::Pack64(lo, hi) => Some(if lo == hi { Vec::new() } else { vec![equal(lo, hi)] }),
            Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, p, q) => Some(vec![Claim::Symmetric(p), Claim::Symmetric(q)]),
            Op::Select(_, p, q) => Some(vec![Claim::Symmetric(p), Claim::Symmetric(q)]),
            _ => None,
        }
    }
}

fn proven(program: &Program, roots: Vec<Claim>) -> BTreeSet<Claim> {
    let mut premises: HashMap<Claim, Option<Vec<Claim>>> = HashMap::default();
    let mut pending = roots;
    while let Some(claim) = pending.pop() {
        if premises.contains_key(&claim) {
            continue;
        }
        let needs = program.premises(claim);
        if let Some(needs) = &needs {
            pending.extend(needs.iter().copied());
        }
        premises.insert(claim, needs);
    }
    greatest(&premises)
}

fn greatest<C: Copy + Ord + Hash>(premises: &HashMap<C, Option<Vec<C>>>) -> BTreeSet<C> {
    let mut users: HashMap<C, Vec<C>> = HashMap::default();
    let mut failed = Vec::new();
    for (&c, needs) in premises {
        match needs {
            Some(needs) => {
                for &n in needs {
                    users.entry(n).or_default().push(c);
                }
            }
            None => failed.push(c),
        }
    }
    let mut holds: HashMap<C, bool> = premises.iter().map(|(&c, p)| (c, p.is_some())).collect();
    while let Some(c) = failed.pop() {
        for &u in users.get(&c).into_iter().flatten() {
            if std::mem::replace(holds.get_mut(&u).unwrap(), false) {
                failed.push(u);
            }
        }
    }
    holds.into_iter().filter(|&(_, held)| held).map(|(c, _)| c).collect()
}

fn run(f: &mut Func) -> usize {
    let program = Program::new(f);
    let mut roots = Vec::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core {
                op: Op::Pack64(lo, hi),
                ..
            } = *inst
            {
                if lo != hi {
                    roots.push(equal(lo, hi));
                }
            }
        }
    }
    if roots.is_empty() {
        return 0;
    }
    let holds = proven(&program, roots);
    if !holds.iter().any(|c| matches!(c, Claim::Equal(..))) {
        return 0;
    }
    let mut position: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    for (&id, block) in &f.blocks {
        for (index, &(v, _)) in block.params.iter().enumerate() {
            position[v.0] = Some((id, index));
        }
        for (index, inst) in block.insts.iter().enumerate() {
            if let Inst::Core { value, .. } = inst {
                position[value.0] = Some((id, block.params.len() + index));
            }
        }
    }
    let mut parent: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    fn root(parent: &BTreeMap<ValueId, ValueId>, mut v: ValueId) -> ValueId {
        while let Some(&p) = parent.get(&v) {
            v = p;
        }
        v
    }
    for claim in &holds {
        let Claim::Equal(a, b) = *claim else {
            continue;
        };
        let (ra, rb) = (root(&parent, a), root(&parent, b));
        if ra == rb {
            continue;
        }
        let (keep, drop) = if position[ra.0].unwrap() <= position[rb.0].unwrap() { (ra, rb) } else { (rb, ra) };
        parent.insert(drop, keep);
    }
    if parent.is_empty() {
        return 0;
    }
    let map: BTreeMap<ValueId, ValueId> = parent.keys().map(|&v| (v, root(&parent, v))).collect();
    let count = map.len();
    f.rename(&map);
    count
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    struct Looped {
        f: Func,
        pairs: Vec<(ValueId, ValueId)>,
    }

    fn looped(r: &mut Rng) -> Looped {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (entry, header, exit) = (BlockId(0), BlockId(1), BlockId(2));
        let x = f.value(Ty::I32);
        let s = f.value(Ty::I32);
        let mut start = Vec::new();
        let ones = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 0xffff_ffff));
        let zero = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 0));
        let three = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 3));
        let c = core(&mut f, &mut start, Ty::I1, Op::Cmp(IntPred::Ult, x, three));
        let first = core(&mut f, &mut start, Ty::I32, Op::Select(c, ones, zero));
        let again = core(&mut f, &mut start, Ty::I32, Op::Select(c, ones, zero));
        let pairs = 2;
        let counter = f.value(Ty::I32);
        let carried: Vec<(ValueId, ValueId)> = (0..pairs).map(|_| (f.value(Ty::I32), f.value(Ty::I32))).collect();
        let (xh, sh) = (f.value(Ty::I32), f.value(Ty::I32));
        let entry_pair = |r: &mut Rng| match r.below(4) {
            0 => (first, again),
            1 => (first, first),
            2 => (x, s),
            _ => (first, zero),
        };
        let mut starts = vec![zero, x, s];
        for _ in 0..pairs {
            let (lo, hi) = entry_pair(r);
            starts.push(lo);
            starts.push(hi);
        }
        let mut body = Vec::new();
        let mut made: Vec<(ValueId, ValueId)> = carried.clone();
        let full = core(&mut f, &mut body, Ty::I64, Op::Const(Ty::I64, u64::MAX));
        let skew = core(&mut f, &mut body, Ty::I64, Op::Const(Ty::I64, 0xffff_ffff));
        let lane = core(&mut f, &mut body, Ty::I32, Op::Env(Env::LaneId));
        let few = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 3));
        for _ in 0..6 {
            let (lo, hi) = made[r.below(made.len())];
            let (plo, phi) = made[r.below(made.len())];
            let next = match r.below(6) {
                0 | 1 => {
                    let wide = core(&mut f, &mut body, Ty::I64, Op::Pack64(lo, hi));
                    let k = if r.below(3) == 0 { skew } else { full };
                    let op = [IntOp::Xor, IntOp::And, IntOp::Or][r.below(3)];
                    let w = core(&mut f, &mut body, Ty::I64, Op::Int(op, wide, k));
                    let nlo = core(&mut f, &mut body, Ty::I32, Op::UnpackLo(w));
                    let nhi = core(&mut f, &mut body, Ty::I32, Op::UnpackHi(w));
                    (nlo, nhi)
                }
                2 => {
                    let a = core(&mut f, &mut body, Ty::I32, Op::Int(IntOp::Xor, lo, plo));
                    let b = core(&mut f, &mut body, Ty::I32, Op::Int(IntOp::Xor, hi, phi));
                    (a, b)
                }
                3 => {
                    let t = core(&mut f, &mut body, Ty::I1, Op::Cmp(IntPred::Ult, lane, few));
                    let a = core(&mut f, &mut body, Ty::I32, Op::Select(t, lo, plo));
                    let b = core(&mut f, &mut body, Ty::I32, Op::Select(t, hi, phi));
                    (a, b)
                }
                4 => (lo, phi),
                _ => (xh, sh),
            };
            made.push(next);
        }
        let one = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 1));
        let next = core(&mut f, &mut body, Ty::I32, Op::Int(IntOp::Add, counter, one));
        let limit = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 4));
        let more = core(&mut f, &mut body, Ty::I1, Op::Cmp(IntPred::Ult, next, limit));
        let mut backs = vec![next, xh, sh];
        for _ in 0..pairs {
            let (lo, hi) = made[r.below(made.len())];
            backs.push(lo);
            backs.push(hi);
        }
        let mut exits = Vec::new();
        let mut packs = Vec::new();
        for &(lo, hi) in &made {
            packs.push(core(&mut f, &mut body, Ty::I64, Op::Pack64(lo, hi)));
        }
        let out: Vec<ValueId> = packs.iter().map(|_| f.value(Ty::I64)).collect();
        exits.extend(packs.iter().copied());
        let mut header_params = vec![(counter, Ty::I32), (xh, Ty::I32), (sh, Ty::I32)];
        for &(lo, hi) in &carried {
            header_params.push((lo, Ty::I32));
            header_params.push((hi, Ty::I32));
        }
        f.blocks.insert(
            entry,
            Block {
                params: vec![(x, Ty::I32), (s, Ty::I32)],
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
                    no: Edge { dst: exit, args: exits },
                },
            },
        );
        f.blocks.insert(
            exit,
            Block {
                params: out.iter().map(|&v| (v, Ty::I64)).collect(),
                insts: Vec::new(),
                term: Term::Ret(out),
            },
        );
        Looped { f, pairs: made }
    }

    fn inputs(r: &mut Rng) -> Vec<Vec<u64>> {
        let s = r.below(16) as u64;
        vec![(0..64).map(|_| r.below(6) as u64).collect(), vec![s; 64]]
    }

    fn defined(f: &Func, block: BlockId) -> BTreeSet<ValueId> {
        let b = &f.blocks[&block];
        b.params.iter().map(|&(v, _)| v).chain(b.insts.iter().filter_map(|i| match i {
            Inst::Core { value, .. } => Some(*value),
            _ => None,
        })).collect()
    }

    #[test]
    fn halves_proven_equal_hold_the_same_word_in_every_lane_and_trip() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let registry = registry();
        let (mut proved, mut merged) = (0, 0);
        for trial in 0..300 {
            let Looped { f, pairs } = looped(&mut r);
            f.check(&registry).unwrap_or_else(|e| panic!("trial {}: the generated program: {}", trial, e));
            let program = Program::new(&f);
            let roots: Vec<Claim> = pairs.iter().filter(|(a, b)| a != b).map(|&(a, b)| equal(a, b)).collect();
            let holds = proven(&program, roots);
            proved += holds.iter().filter(|c| matches!(c, Claim::Equal(..))).count();
            let blocks: Vec<(BlockId, BTreeSet<ValueId>)> = f.blocks.keys().map(|&b| (b, defined(&f, b))).collect();
            for _ in 0..3 {
                let entry = inputs(&mut r);
                let want = simulate(&f, &entry, &mut |block, values| {
                    let here = &blocks.iter().find(|(b, _)| *b == block).unwrap().1;
                    for claim in &holds {
                        match *claim {
                            Claim::Equal(a, b) if here.contains(&a) && here.contains(&b) => {
                                assert_eq!(values[a.0], values[b.0], "trial {}: v{} and v{} differ", trial, a.0, b.0);
                            }
                            Claim::Symmetric(w) if here.contains(&w) => {
                                for &x in &values[w.0] {
                                    assert_eq!(x & 0xffff_ffff, x >> 32, "trial {}: v{} has unequal halves", trial, w.0);
                                }
                            }
                            _ => {}
                        }
                    }
                });
                let mut renamed = f.clone();
                merged += run(&mut renamed);
                renamed.check(&registry).unwrap_or_else(|e| panic!("trial {}: the rewritten program: {}", trial, e));
                let got = simulate(&renamed, &entry, &mut |_, _| {});
                assert_eq!(want, got, "trial {}: the rewritten program returns other words", trial);
            }
        }
        assert!(proved > 0 && merged > 0, "some halves are proven equal and merged: {} {}", proved, merged);
    }

    fn mask_loop(second: impl FnOnce(&mut Func, &mut Vec<Inst>, ValueId, ValueId, ValueId) -> ValueId) -> (Func, ValueId) {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (entry, header, exit) = (BlockId(0), BlockId(1), BlockId(2));
        let x = f.value(Ty::I32);
        let mut start = Vec::new();
        let ones = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 0xffff_ffff));
        let zero = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 0));
        let three = core(&mut f, &mut start, Ty::I32, Op::Const(Ty::I32, 3));
        let c = core(&mut f, &mut start, Ty::I1, Op::Cmp(IntPred::Ult, x, three));
        let lo = core(&mut f, &mut start, Ty::I32, Op::Select(c, ones, zero));
        let hi = second(&mut f, &mut start, c, ones, zero);
        let (counter, plo, phi) = (f.value(Ty::I32), f.value(Ty::I32), f.value(Ty::I32));
        let mut body = Vec::new();
        let wide = core(&mut f, &mut body, Ty::I64, Op::Pack64(plo, phi));
        let flip = core(&mut f, &mut body, Ty::I64, Op::Const(Ty::I64, u64::MAX));
        let w = core(&mut f, &mut body, Ty::I64, Op::Int(IntOp::Xor, wide, flip));
        let nlo = core(&mut f, &mut body, Ty::I32, Op::UnpackLo(w));
        let nhi = core(&mut f, &mut body, Ty::I32, Op::UnpackHi(w));
        let one = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 1));
        let next = core(&mut f, &mut body, Ty::I32, Op::Int(IntOp::Add, counter, one));
        let limit = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 4));
        let more = core(&mut f, &mut body, Ty::I1, Op::Cmp(IntPred::Ult, next, limit));
        let out = f.value(Ty::I64);
        f.blocks.insert(entry, Block { params: vec![(x, Ty::I32)], insts: start, term: Term::Br(Edge { dst: header, args: vec![zero, lo, hi] }) });
        f.blocks.insert(
            header,
            Block {
                params: vec![(counter, Ty::I32), (plo, Ty::I32), (phi, Ty::I32)],
                insts: body,
                term: Term::CondBr {
                    cond: more,
                    yes: Edge { dst: header, args: vec![next, nlo, nhi] },
                    no: Edge { dst: exit, args: vec![wide] },
                },
            },
        );
        f.blocks.insert(exit, Block { params: vec![(out, Ty::I64)], insts: Vec::new(), term: Term::Ret(vec![out]) });
        (f, wide)
    }

    fn halves_of(f: &Func, wide: ValueId) -> (ValueId, ValueId) {
        match f.definitions()[wide.0] {
            Some(Op::Pack64(lo, hi)) => (lo, hi),
            other => panic!("not a pair: {:?}", other),
        }
    }

    #[test]
    fn a_boolean_mask_flipped_around_a_loop_keeps_equal_halves() {
        let (mut f, wide) = mask_loop(|f, insts, c, ones, zero| core(f, insts, Ty::I32, Op::Select(c, ones, zero)));
        assert!(run(&mut f) > 0);
        let (lo, hi) = halves_of(&f, wide);
        assert_eq!(lo, hi, "both halves select the same bit of the wave's flag");
    }

    #[test]
    fn the_greatest_fixpoint_keeps_exactly_the_claims_that_reach_no_failure() {
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let mut kept = 0;
        for _ in 0..2000 {
            let n = 1 + r.below(10);
            let premises: HashMap<usize, Option<Vec<usize>>> = (0..n)
                .map(|c| {
                    let needs = if r.below(6) == 0 {
                        None
                    } else {
                        Some((0..r.below(4)).map(|_| r.below(n)).collect())
                    };
                    (c, needs)
                })
                .collect();
            let fails = |start: usize| {
                let mut seen = vec![false; n];
                let mut stack = vec![start];
                while let Some(c) = stack.pop() {
                    if std::mem::replace(&mut seen[c], true) {
                        continue;
                    }
                    match &premises[&c] {
                        None => return true,
                        Some(needs) => stack.extend(needs),
                    }
                }
                false
            };
            let want: BTreeSet<usize> = (0..n).filter(|&c| !fails(c)).collect();
            kept += want.len();
            assert_eq!(greatest(&premises), want, "premises {:?}", premises);
        }
        assert!(kept > 1000, "the random premises let too few claims hold to test anything");
    }

    #[test]
    fn halves_that_start_apart_stay_apart() {
        let (mut f, wide) = mask_loop(|_, _, _, _, zero| zero);
        let before = f.clone();
        run(&mut f);
        let (lo, hi) = halves_of(&f, wide);
        assert_ne!(lo, hi, "the high half starts at zero while the low one starts at the flag");
        assert_eq!(f, before);
    }
}
