use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum Form {
    Value(ValueId),
    Core(Ty, Op),
    Target(TargetOp, Vec<usize>, usize),
    Load(Space, MemSize, usize),
    Hazard(BlockId, usize, usize),
    Opaque(usize, ValueId),
    Linear(Ty, Vec<(usize, u64)>, u64),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Leaf {
    Term(usize),
    Value(ValueId),
}

pub(super) struct Forms<'a> {
    f: &'a Func,
    facts: &'a Facts,
    terms: Vec<Form>,
    types: Vec<Ty>,
    index: HashMap<Form, usize>,
}

impl<'a> Forms<'a> {
    pub(super) fn new(f: &'a Func, facts: &'a Facts) -> Self {
        Self {
            f,
            facts,
            terms: Vec::new(),
            types: Vec::new(),
            index: HashMap::default(),
        }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.terms.len()
    }

    #[inline]
    pub(super) fn term(&self, t: usize) -> &Form {
        &self.terms[t]
    }

    #[inline]
    pub(super) fn ty(&self, t: usize) -> Ty {
        self.types[t]
    }

    pub(super) fn intern(&mut self, ty: Ty, form: Form) -> usize {
        let form = match form {
            Form::Core(t, Op::Int(k @ (IntOp::Add | IntOp::Mul | IntOp::And | IntOp::Or | IntOp::Xor), x, y)) if x.0 > y.0 => {
                Form::Core(t, Op::Int(k, y, x))
            }
            Form::Core(t, Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), x, y)) if x.0 > y.0 => Form::Core(t, Op::Cmp(p, y, x)),
            other => other,
        };
        match &form {
            Form::Core(_, Op::Convert(Cvt::Bitcast, _, a)) => {
                let a = a.0;
                if self.types[a] == ty {
                    return a;
                }
                if let Form::Core(_, Op::Convert(Cvt::Bitcast, _, b)) = self.terms[a] {
                    if self.types[b.0] == ty {
                        return b.0;
                    }
                }
            }
            Form::Core(_, Op::UnpackLo(p) | Op::UnpackHi(p)) => {
                let low = matches!(form, Form::Core(_, Op::UnpackLo(_)));
                let half = |e: &mut Self, x: ValueId| {
                    let op = if low {
                        Op::UnpackLo(x)
                    } else {
                        Op::UnpackHi(x)
                    };
                    e.intern(Ty::I32, Form::Core(Ty::I32, op))
                };
                match self.terms[p.0].clone() {
                    Form::Core(_, Op::Pack64(lo, hi)) => return if low { lo.0 } else { hi.0 },
                    Form::Core(_, Op::Const(_, k)) => {
                        let word = if low { k & 0xffff_ffff } else { k >> 32 };
                        return self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, word)));
                    }
                    Form::Core(_, Op::Convert(Cvt::ZExt, _, a)) if self.types[a.0] == Ty::I32 => {
                        return if low {
                            a.0
                        } else {
                            self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, 0)))
                        };
                    }
                    Form::Core(_, Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y)) => {
                        let (x, y) = (half(self, x), half(self, y));
                        return self.intern(
                            Ty::I32,
                            Form::Core(Ty::I32, Op::Int(k, ValueId(x), ValueId(y))),
                        );
                    }
                    Form::Core(_, Op::Int(k @ (IntOp::LShr | IntOp::Shl), x, s))
                        if self.constant_of(s.0) == Some(32) =>
                    {
                        let other = match (k, low) {
                            (IntOp::LShr, true) => Some(Op::UnpackHi(x)),
                            (IntOp::Shl, false) => Some(Op::UnpackLo(x)),
                            _ => None,
                        };
                        return match other {
                            Some(op) => self.intern(Ty::I32, Form::Core(Ty::I32, op)),
                            None => {
                                self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, 0)))
                            }
                        };
                    }
                    _ => {}
                }
            }
            Form::Core(_, Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y)) => {
                return self.gathered(ty, *k, x.0, y.0);
            }
            Form::Core(_, Op::Pack64(lo, hi)) => {
                if let (Form::Core(_, Op::UnpackLo(a)), Form::Core(_, Op::UnpackHi(b))) =
                    (&self.terms[lo.0], &self.terms[hi.0])
                {
                    if a == b {
                        return a.0;
                    }
                }
            }
            Form::Core(_, Op::Int(k @ (IntOp::Add | IntOp::Sub | IntOp::Mul | IntOp::Shl), x, y))
                if matches!(ty, Ty::I32 | Ty::I64) =>
            {
                if let Some((terms, constant)) = self.combine(ty, *k, x.0, y.0) {
                    return self.intern_linear(ty, terms, constant);
                }
                if *k == IntOp::Mul {
                    if let Some(t) = self.multiplied(ty, x.0, y.0) {
                        return t;
                    }
                }
            }
            _ => {}
        }
        self.insert(ty, form)
    }

    fn insert(&mut self, ty: Ty, form: Form) -> usize {
        if let Some(&t) = self.index.get(&form) {
            return t;
        }
        let t = self.terms.len();
        self.terms.push(form.clone());
        self.types.push(ty);
        self.index.insert(form, t);
        t
    }

    fn linear_parts(&self, t: usize) -> (Vec<(usize, u64)>, u64) {
        match &self.terms[t] {
            Form::Linear(_, terms, constant) => (terms.clone(), *constant),
            Form::Core(_, Op::Const(_, k)) => (Vec::new(), *k),
            _ => (vec![(t, 1)], 0),
        }
    }

    fn combine(&self, ty: Ty, k: IntOp, x: usize, y: usize) -> Option<(Vec<(usize, u64)>, u64)> {
        let (mut xs, xk) = self.linear_parts(x);
        let (ys, yk) = self.linear_parts(y);
        let scale = |terms: &[(usize, u64)], constant: u64, by: u64| -> (Vec<(usize, u64)>, u64) {
            (terms.iter().map(|&(t, c)| (t, c.wrapping_mul(by))).collect(), constant.wrapping_mul(by))
        };
        Some(match k {
            IntOp::Add => {
                xs.extend(ys);
                (xs, xk.wrapping_add(yk))
            }
            IntOp::Sub => {
                xs.extend(ys.iter().map(|&(t, c)| (t, c.wrapping_neg())));
                (xs, xk.wrapping_sub(yk))
            }
            IntOp::Mul if ys.is_empty() => scale(&xs, xk, yk),
            IntOp::Mul if xs.is_empty() => scale(&ys, yk, xk),
            IntOp::Shl if ys.is_empty() && yk < 32 => scale(&xs, xk, 1u64 << yk),
            _ => return None,
        })
        .filter(|_| matches!(ty, Ty::I32 | Ty::I64))
    }

    fn intern_linear(&mut self, ty: Ty, terms: Vec<(usize, u64)>, constant: u64) -> usize {
        let mask = if ty == Ty::I64 { u64::MAX } else { (1u64 << ty.bits()) - 1 };
        let mut merged: BTreeMap<usize, u64> = BTreeMap::new();
        for (t, c) in terms {
            let e = merged.entry(t).or_insert(0);
            *e = e.wrapping_add(c) & mask;
        }
        let terms: Vec<(usize, u64)> = merged.into_iter().filter(|&(_, c)| c != 0).collect();
        let constant = constant & mask;
        match terms.as_slice() {
            [] => self.intern(ty, Form::Core(ty, Op::Const(ty, constant))),
            [(t, 1)] if constant == 0 => *t,
            _ => self.intern(ty, Form::Linear(ty, terms, constant)),
        }
    }

    fn constant_of(&self, t: usize) -> Option<u64> {
        match self.terms[t] {
            Form::Core(_, Op::Const(_, k)) => Some(k),
            _ => None,
        }
    }

    fn multiplied(&mut self, ty: Ty, x: usize, y: usize) -> Option<usize> {
        let (xs, xk) = self.linear_parts(x);
        let (ys, yk) = self.linear_parts(y);
        if xs.len() * ys.len() > 16 {
            return None;
        }
        let mut terms: Vec<(usize, u64)> = Vec::new();
        terms.extend(xs.iter().map(|&(t, c)| (t, c.wrapping_mul(yk))));
        terms.extend(ys.iter().map(|&(t, c)| (t, c.wrapping_mul(xk))));
        for &(a, c) in &xs {
            for &(b, d) in &ys {
                let mut factors = self.operands_of(IntOp::Mul, a);
                factors.extend(self.operands_of(IntOp::Mul, b));
                if factors.len() > 8 {
                    return None;
                }
                factors.sort_unstable();
                let m = self.chained(ty, IntOp::Mul, &factors);
                terms.push((m, c.wrapping_mul(d)));
            }
        }
        Some(self.intern_linear(ty, terms, xk.wrapping_mul(yk)))
    }

    fn operands_of(&self, k: IntOp, t: usize) -> Vec<usize> {
        match self.terms[t] {
            Form::Core(_, Op::Int(op, a, b)) if op == k => {
                let mut out = self.operands_of(k, a.0);
                out.extend(self.operands_of(k, b.0));
                out
            }
            _ => vec![t],
        }
    }

    fn chained(&mut self, ty: Ty, k: IntOp, parts: &[usize]) -> usize {
        let mut t = parts[0];
        for &p in &parts[1..] {
            t = self.insert(ty, Form::Core(ty, Op::Int(k, ValueId(t), ValueId(p))));
        }
        t
    }

    fn gathered(&mut self, ty: Ty, k: IntOp, x: usize, y: usize) -> usize {
        let ones = if ty == Ty::I64 { u64::MAX } else { (1u64 << ty.bits()) - 1 };
        let mut parts = self.operands_of(k, x);
        parts.extend(self.operands_of(k, y));
        let mut constant: Option<u64> = None;
        let mut rest: Vec<usize> = Vec::new();
        for p in parts {
            match self.constant_of(p) {
                Some(c) => {
                    constant = Some(match (k, constant) {
                        (_, None) => c & ones,
                        (IntOp::And, Some(a)) => a & c,
                        (IntOp::Or, Some(a)) => a | c,
                        (_, Some(a)) => (a ^ c) & ones,
                    })
                }
                None => rest.push(p),
            }
        }
        rest.sort_unstable();
        if k == IntOp::Xor {
            let mut kept: Vec<usize> = Vec::new();
            for p in rest {
                if kept.last() == Some(&p) {
                    kept.pop();
                } else {
                    kept.push(p);
                }
            }
            rest = kept;
        } else {
            rest.dedup();
        }
        let identity = if k == IntOp::And { ones } else { 0 };
        let absorbing = match k {
            IntOp::And => Some(0),
            IntOp::Or => Some(ones),
            _ => None,
        };
        if let Some(c) = constant.filter(|&c| Some(c) == absorbing) {
            return self.intern(ty, Form::Core(ty, Op::Const(ty, c)));
        }
        if let Some(c) = constant.filter(|&c| c != identity) {
            let t = self.intern(ty, Form::Core(ty, Op::Const(ty, c)));
            rest.push(t);
        }
        match rest.as_slice() {
            [] => self.intern(ty, Form::Core(ty, Op::Const(ty, identity))),
            _ => self.chained(ty, k, &rest),
        }
    }

    pub(super) fn difference(&self, x: usize, y: usize) -> (BTreeMap<Leaf, u64>, u64) {
        let mask = if self.types[x].bits() >= 64 { u64::MAX } else { (1u64 << self.types[x].bits()) - 1 };
        let (mut terms, mut constant) = (BTreeMap::new(), 0u64);
        self.expand(x, 1, &mut terms, &mut constant, 0);
        self.expand(y, u64::MAX, &mut terms, &mut constant, 0);
        terms.retain(|_, c| *c & mask != 0);
        (terms.into_iter().map(|(l, c)| (l, c & mask)).collect(), constant & mask)
    }

    fn expand(&self, t: usize, scale: u64, terms: &mut BTreeMap<Leaf, u64>, constant: &mut u64, depth: usize) {
        match &self.terms[t] {
            Form::Linear(_, parts, k) => {
                *constant = constant.wrapping_add(k.wrapping_mul(scale));
                for &(u, c) in parts {
                    self.expand(u, scale.wrapping_mul(c), terms, constant, depth);
                }
            }
            Form::Core(_, Op::Const(_, k)) => *constant = constant.wrapping_add(k.wrapping_mul(scale)),
            Form::Value(v) => self.expand_value(*v, scale, terms, constant, depth),
            _ => {
                let c = terms.entry(Leaf::Term(t)).or_insert(0);
                *c = c.wrapping_add(scale);
            }
        }
    }

    fn expand_value(&self, v: ValueId, scale: u64, terms: &mut BTreeMap<Leaf, u64>, constant: &mut u64, depth: usize) {
        let (f, facts) = (self.f, self.facts);
        if let Some(k) = facts.constant(f, v) {
            *constant = constant.wrapping_add(k.wrapping_mul(scale));
            return;
        }
        let wide = |x: ValueId| f.types[x.0] == f.types[v.0];
        let op = if depth > 16 || !matches!(f.types[v.0], Ty::I32 | Ty::I64) { None } else { facts.op(f, v) };
        match op {
            Some(Op::Int(IntOp::Add, a, b)) if wide(a) && wide(b) => {
                self.expand_value(a, scale, terms, constant, depth + 1);
                self.expand_value(b, scale, terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Sub, a, b)) if wide(a) && wide(b) => {
                self.expand_value(a, scale, terms, constant, depth + 1);
                self.expand_value(b, scale.wrapping_neg(), terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Mul, a, k)) if wide(a) && facts.constant(f, k).is_some() => {
                self.expand_value(a, scale.wrapping_mul(facts.constant(f, k).unwrap()), terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Mul, k, a)) if wide(a) && facts.constant(f, k).is_some() => {
                self.expand_value(a, scale.wrapping_mul(facts.constant(f, k).unwrap()), terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Shl, a, k)) if wide(a) && facts.constant(f, k).is_some_and(|k| k < f.types[v.0].bits() as u64) => {
                self.expand_value(a, scale.wrapping_mul(1u64 << facts.constant(f, k).unwrap()), terms, constant, depth + 1);
            }
            _ => {
                let c = terms.entry(Leaf::Value(v)).or_insert(0);
                *c = c.wrapping_add(scale);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::super::testing::*;
    use super::*;
    use std::collections::BTreeSet;

    fn with_forms(test: impl FnOnce(&mut Forms)) {
        let (b, _) = Build::kernel();
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut forms = Forms::new(&b.f, &facts);
        test(&mut forms);
    }

    fn computed(e: &Forms, t: usize, leaves: &[u64]) -> u64 {
        let mask = u32::MAX as u64;
        match e.term(t) {
            Form::Opaque(i, _) => leaves[*i - 1000],
            Form::Core(_, Op::Const(_, k)) => *k & mask,
            Form::Core(_, Op::Int(op, a, b)) => {
                let (x, y) = (computed(e, a.0, leaves), computed(e, b.0, leaves));
                (match op {
                    IntOp::Add => x.wrapping_add(y),
                    IntOp::Sub => x.wrapping_sub(y),
                    IntOp::Mul => x.wrapping_mul(y),
                    IntOp::And => x & y,
                    IntOp::Or => x | y,
                    IntOp::Xor => x ^ y,
                    _ => panic!("no {:?} in these terms", op),
                }) & mask
            }
            Form::Linear(_, parts, k) => parts.iter().fold(*k, |acc, &(p, c)| acc.wrapping_add(c.wrapping_mul(computed(e, p, leaves)))) & mask,
            other => panic!("no {:?} in these terms", std::mem::discriminant(other)),
        }
    }

    fn random_term(e: &mut Forms, r: &mut Random, depth: usize, leaves: &[usize]) -> (usize, Box<dyn Fn(&[u64]) -> u64>) {
        if depth == 0 || r.below(4) == 0 {
            if r.below(3) == 0 {
                let k = [0u64, 1, 2, 3, 5, 0xff, 0xffff_ffff, 0x8000_0000][r.below(8) as usize];
                return (e.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, k))), Box::new(move |_| k));
            }
            let i = r.below(leaves.len() as u64) as usize;
            return (leaves[i], Box::new(move |v| v[i]));
        }
        let ops = [IntOp::Add, IntOp::Sub, IntOp::Mul, IntOp::And, IntOp::Or, IntOp::Xor];
        let op = ops[r.below(ops.len() as u64) as usize];
        let (a, fa) = random_term(e, r, depth - 1, leaves);
        let (b, fb) = random_term(e, r, depth - 1, leaves);
        let t = e.intern(Ty::I32, Form::Core(Ty::I32, Op::Int(op, ValueId(a), ValueId(b))));
        let mask = u32::MAX as u64;
        (
            t,
            Box::new(move |v| {
                let (x, y) = (fa(v), fb(v));
                (match op {
                    IntOp::Add => x.wrapping_add(y),
                    IntOp::Sub => x.wrapping_sub(y),
                    IntOp::Mul => x.wrapping_mul(y),
                    IntOp::And => x & y,
                    IntOp::Or => x | y,
                    _ => x ^ y,
                }) & mask
            }),
        )
    }

    #[test]
    fn interned_words_compute_what_their_operations_compute() {
        with_forms(|e| {
            let leaves: Vec<usize> = (0..3).map(|i| e.intern(Ty::I32, Form::Opaque(1000 + i, ValueId(0)))).collect();
            let mut r = Random::new(113);
            let mut wrong = Vec::new();
            for trial in 0..400 {
                let (t, truth) = random_term(e, &mut r, 4, &leaves);
                for _ in 0..8 {
                    let values: Vec<u64> = (0..3)
                        .map(|_| match r.below(3) {
                            0 => r.below(8),
                            1 => (r.next() as u32) as u64,
                            _ => [0xffff_ffffu64, 0x8000_0000, 0x7fff_ffff][r.below(3) as usize],
                        })
                        .collect();
                    if computed(e, t, &values) != truth(&values) {
                        wrong.push(format!("trial {} at {:?}: {:#x} not {:#x}", trial, values, computed(e, t, &values), truth(&values)));
                    }
                }
            }
            assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn interned_words_are_one_term_for_regrouped_distributed_and_reordered_operations() {
        with_forms(|e| {
            let [x, y, z]: [usize; 3] = std::array::from_fn(|i| e.intern(Ty::I32, Form::Opaque(1000 + i, ValueId(0))));
            let k = |e: &mut Forms, c: u64| e.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, c)));
            let op = |e: &mut Forms, o: IntOp, a: usize, b: usize| e.intern(Ty::I32, Form::Core(Ty::I32, Op::Int(o, ValueId(a), ValueId(b))));
            use IntOp::*;
            let mut loose = Vec::new();
            let mut same = |name: &str, a: usize, b: usize| {
                if a != b {
                    loose.push(name.to_string());
                }
            };
            let (xy, yz, zx) = (op(e, Mul, x, y), op(e, Mul, y, z), op(e, Mul, z, x));
            let (a, b, c) = (op(e, Mul, xy, z), op(e, Mul, x, yz), op(e, Mul, zx, y));
            same("(x y) z and x (y z)", a, b);
            same("(x y) z and (z x) y", a, c);
            let sum = op(e, Add, x, y);
            let (xz, yz2) = (op(e, Mul, x, z), op(e, Mul, y, z));
            let (a, b) = (op(e, Mul, sum, z), op(e, Add, xz, yz2));
            same("(x + y) z and x z + y z", a, b);
            let one = k(e, 1);
            let (up, down) = (op(e, Add, x, one), op(e, Sub, x, one));
            let xx = op(e, Mul, x, x);
            let (a, b) = (op(e, Mul, up, down), op(e, Sub, xx, one));
            same("(x + 1)(x - 1) and x x - 1", a, b);
            let (xy, yz) = (op(e, And, x, y), op(e, And, y, z));
            let (a, b) = (op(e, And, xy, z), op(e, And, x, yz));
            same("(x & y) & z and x & (y & z)", a, b);
            let zx = op(e, And, z, x);
            let c = op(e, And, zx, y);
            same("(x & y) & z and (z & x) & y", a, c);
            let (xo, xa) = (op(e, Or, x, x), op(e, And, x, x));
            same("x | x and x", xo, x);
            same("x & x and x", xa, x);
            let zero = k(e, 0);
            let xx = op(e, Xor, x, x);
            same("x ^ x and 0", xx, zero);
            let xy = op(e, Xor, x, y);
            let xyx = op(e, Xor, xy, x);
            same("(x ^ y) ^ x and y", xyx, y);
            let ones = k(e, 0xffff_ffff);
            let a = op(e, And, x, ones);
            same("x & ~0 and x", a, x);
            let (three, five) = (k(e, 3), k(e, 5));
            let x3 = op(e, And, x, three);
            let a = op(e, And, x3, five);
            let b = op(e, And, x, one);
            same("(x & 3) & 5 and x & 1", a, b);
            let x3 = op(e, Or, x, three);
            let a = op(e, Or, x3, five);
            let seven = k(e, 7);
            let b = op(e, Or, seven, x);
            same("(x | 3) | 5 and 7 | x", a, b);
            let a = op(e, Or, x, ones);
            same("x | ~0 and ~0", a, ones);
            assert!(loose.is_empty(), "each pair computes the same word: {:?}", loose);
        });
    }
}
