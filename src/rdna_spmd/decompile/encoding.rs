use super::address::{Classes, Form, Unknown, UnknownInfo, Wide};
use super::linear::{Clause, Domain, Linear, Problem, Var};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::IntPred;
use std::collections::BTreeSet;

const WORD: i128 = 1 << 32;
const SHARED: usize = 2;

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

type Shape = (i128, Vec<(i128, Domain)>, Option<(usize, usize)>);

#[derive(Default)]
pub(super) struct Shapes {
    sets: HashMap<Shape, Option<std::rc::Rc<Values>>>,
}

const ENUMERATED: i128 = 1 << 16;
const SPAN: i128 = 1 << 20;

enum Values {
    Sorted(Vec<i128>),
    Line { offset: i128, bits: Vec<bool> },
    Cycle { modulus: i128, bits: Vec<bool> },
}

impl Values {
    fn contains(&self, v: i128, modulus: i128) -> bool {
        match self {
            Values::Sorted(values) => values.binary_search(&(if modulus > 0 { v.rem_euclid(modulus) } else { v })).is_ok(),
            Values::Line { offset, bits } => {
                let end = offset + bits.len() as i128;
                if modulus == 0 {
                    return v >= *offset && v < end && bits[(v - offset) as usize];
                }
                let mut x = offset + (v - offset).rem_euclid(modulus);
                while x < end {
                    if bits[(x - offset) as usize] {
                        return true;
                    }
                    x += modulus;
                }
                false
            }
            Values::Cycle { modulus, bits } => bits[v.rem_euclid(*modulus) as usize],
        }
    }
}

pub(super) struct Encoding<'a> {
    unknowns: &'a [UnknownInfo],
    variant: &'a dyn Fn(&UnknownInfo) -> bool,
    problem: Problem,
    vars: HashMap<(usize, Unknown), Var>,
    words: Vec<((usize, Form), Var)>,
    difference: Linear,
    window: (i128, i128),
    modular: bool,
    signs: HashMap<Var, Var>,
}

impl<'a> Encoding<'a> {
    pub(super) fn new(unknowns: &'a [UnknownInfo], variant: &'a dyn Fn(&UnknownInfo) -> bool) -> Self {
        Encoding {
            unknowns,
            variant,
            problem: Problem::default(),
            vars: HashMap::default(),
            words: Vec::new(),
            difference: Linear::default(),
            window: (0, 0),
            modular: false,
            signs: HashMap::default(),
        }
    }

    fn side_of(&self, side: usize, u: Unknown) -> usize {
        if (self.variant)(&self.unknowns[u as usize]) {
            side
        } else {
            SHARED
        }
    }

    fn unknown(&mut self, side: usize, u: Unknown) -> Var {
        let key = (self.side_of(side, u), u);
        if let Some(&v) = self.vars.get(&key) {
            return v;
        }
        let info = &self.unknowns[u as usize];
        let (lo, hi) = info.range.unwrap_or((0, u32::MAX));
        let mut domain = Domain::between(lo as i128, hi as i128);
        if let Some(values) = &info.values {
            domain = domain.meet(&Domain::of_values(values.iter().map(|&k| k as i128)));
        }
        let v = self.problem.var(domain);
        self.vars.insert(key, v);
        v
    }

    pub(super) fn form(&mut self, side: usize, f: &Form) -> Linear {
        let mut e = Linear::constant(f.constant as i32 as i128);
        for &(u, c) in &f.terms {
            let v = self.unknown(side, u);
            e.add(v, c as i32 as i128);
        }
        e
    }

    fn wrapped(&mut self, e: Linear) -> Linear {
        let m = self.problem.free();
        e.term(m, -WORD)
    }

    fn word(&mut self, side: usize, f: &Form) -> Var {
        let key = (
            if f.terms.iter().any(|&(u, _)| self.side_of(side, u) == side) { side } else { SHARED },
            f.clone(),
        );
        if let Some((_, v)) = self.words.iter().find(|(k, _)| *k == key) {
            return *v;
        }
        let e = self.form(side, f);
        let w = self.problem.between(0, WORD - 1);
        let e = self.wrapped(e);
        self.problem.equal(e.term(w, -1));
        self.words.push((key, w));
        w
    }

    pub(super) fn wide(&mut self, side: usize, w: &Wide) -> Linear {
        let mut e = Linear::constant(w.constant);
        for &(u, c) in &w.terms {
            let v = self.unknown(side, u);
            e.add(v, c);
        }
        for (f, c) in &w.words {
            let v = self.word(side, f);
            e.add(v, *c);
        }
        e
    }

    pub(super) fn limits(&mut self, classes: [&Classes; 2]) -> usize {
        let mut done = [BTreeSet::new(), BTreeSet::new()];
        loop {
            let present: BTreeSet<Unknown> = self.vars.keys().map(|&(_, u)| u).collect();
            let mut added = false;
            for side in 0..2 {
                for (i, (form, pieces)) in classes[side].iter().enumerate() {
                    if done[side].contains(&i) || !form.terms.iter().any(|&(u, _)| present.contains(&u)) {
                        continue;
                    }
                    done[side].insert(i);
                    added = true;
                    let domain = Domain::Within(pieces.iter().map(|&(lo, hi)| (lo as i128, hi as i128)).collect());
                    let v = match form.terms.as_slice() {
                        &[(u, 1)] if form.constant == 0 => self.unknown(side, u),
                        _ => self.word(side, form),
                    };
                    self.problem.domains[v] = self.problem.domains[v].meet(&domain);
                }
            }
            if !added {
                break;
            }
        }
        done[0].len() + done[1].len()
    }

    pub(super) fn window(&mut self, difference: Linear, modular: bool, x_bytes: u32, y_bytes: u32) {
        self.window = (-(x_bytes as i128) + 1, y_bytes as i128 - 1);
        self.difference = difference.clone();
        self.modular = modular;
        let difference = if modular { self.wrapped(difference) } else { difference };
        self.problem.at_least(difference.clone().offset(-self.window.0));
        self.problem.at_most(difference, self.window.1);
    }

    pub(super) fn surely_apart(&self) -> bool {
        let (lo, hi) = self.window;
        let g = self.difference.terms.iter().fold(0i128, |g, &(_, c)| gcd(g, c));
        let g = if self.modular { gcd(g, WORD) } else { g };
        if g > 1 && !(lo..=hi).any(|t| (t - self.difference.constant).rem_euclid(g) == 0) {
            return true;
        }
        let (low, high) = self.problem.bounds(&self.difference);
        let (Some(low), Some(high)) = (low, high) else {
            return false;
        };
        if low > high {
            return true;
        }
        if !self.modular {
            return high < lo || low > hi;
        }
        let span = high - low;
        span + 1 < WORD && !(lo..=hi).any(|t| (t - low).rem_euclid(WORD) <= span)
    }

    fn shape(&self, copies: Option<(Var, Var)>) -> Option<Shape> {
        if !self.problem.equal.is_empty() || !self.problem.either.is_empty() {
            return None;
        }
        let mut modulus = if self.modular { WORD } else { 0 };
        let mut terms: Vec<(i128, Domain, Var)> = Vec::with_capacity(self.difference.terms.len());
        for &(v, c) in &self.difference.terms {
            if c == 0 {
                continue;
            }
            let Domain::Within(pieces) = &self.problem.domains[v] else {
                return None;
            };
            if pieces.is_empty() {
                return None;
            }
            let whole = self.modular && pieces.len() == 1 && pieces[0].0 <= 0 && pieces[0].1 >= WORD - 1;
            if whole && copies.is_none_or(|(p, q)| v != p && v != q) {
                modulus = gcd(modulus, c);
                continue;
            }
            terms.push((c, self.problem.domains[v].clone(), v));
        }
        terms.sort_by(|a, b| (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2)));
        let differ = copies.and_then(|(p, q)| {
            let i = terms.iter().position(|t| t.2 == p)?;
            let j = terms.iter().position(|t| t.2 == q)?;
            Some((i, j))
        });
        if copies.is_some() && differ.is_none() {
            return None;
        }
        Some((modulus, terms.into_iter().map(|(c, d, _)| (c, d)).collect(), differ))
    }

    fn values(shape: &Shape) -> Option<Values> {
        let (modulus, terms, differ) = shape;
        if differ.is_some() {
            return Self::enumerate(shape).map(Values::Sorted);
        }
        let mut span: i128 = 0;
        let mut offset: i128 = 0;
        for (c, d) in terms {
            let (Some(lo), Some(hi)) = d.hull()? else {
                return None;
            };
            if *c >= 0 {
                offset += c * lo;
                span += c * (hi - lo);
            } else {
                offset += c * hi;
                span += -c * (hi - lo);
            }
        }
        if *modulus > 0 && *modulus <= SPAN {
            return Some(Self::cycle(*modulus, terms));
        }
        if span < SPAN {
            return Some(Self::line(offset, span, terms));
        }
        Self::enumerate(shape).map(Values::Sorted)
    }

    fn line(offset: i128, span: i128, terms: &[(i128, Domain)]) -> Values {
        let n = span as usize + 1;
        let mut bits = vec![false; n];
        bits[0] = true;
        for (c, d) in terms {
            let Domain::Within(pieces) = d else { unreachable!() };
            let (lo, hi) = (pieces[0].0, pieces[pieces.len() - 1].1);
            let s = c.unsigned_abs() as usize;
            if s == 0 {
                continue;
            }
            let steps: Vec<(usize, usize)> = pieces
                .iter()
                .map(|&(a, b)| if *c >= 0 { ((a - lo) as usize, (b - lo) as usize) } else { ((hi - b) as usize, (hi - a) as usize) })
                .collect();
            let mut count = vec![0u32; n];
            for v in 0..n {
                count[v] = bits[v] as u32 + if v >= s { count[v - s] } else { 0 };
            }
            let at = |v: isize| if v < 0 { 0 } else { count[v as usize] };
            let mut next = vec![false; n];
            for v in 0..n {
                next[v] = steps.iter().any(|&(a, b)| {
                    let high = v as isize - (s * a) as isize;
                    high >= 0 && at(high) - at(v as isize - (s * (b + 1)) as isize) > 0
                });
            }
            bits = next;
        }
        Values::Line { offset, bits }
    }

    fn cycle(modulus: i128, terms: &[(i128, Domain)]) -> Values {
        let m = modulus as usize;
        let mut bits = vec![false; m];
        bits[0] = true;
        for (c, d) in terms {
            let Domain::Within(pieces) = d else { unreachable!() };
            let s = c.rem_euclid(modulus) as usize;
            if s == 0 {
                continue;
            }
            let g = gcd(s as i128, modulus) as usize;
            let cycle = m / g;
            let mut next = vec![false; m];
            let mut seq = vec![0u32; cycle];
            let mut prefix = vec![0u32; 2 * cycle + 1];
            for r in 0..g {
                let mut p = r;
                for i in 0..cycle {
                    seq[i] = bits[p] as u32;
                    p = (p + s) % m;
                }
                for i in 0..2 * cycle {
                    prefix[i + 1] = prefix[i] + seq[i % cycle];
                }
                let total = prefix[cycle];
                let mut p = r;
                for i in 0..cycle {
                    let reachable = pieces.iter().any(|&(a, b)| {
                        let width = (b - a + 1) as usize;
                        if width >= cycle {
                            return total > 0;
                        }
                        let start = (i as i128 - b).rem_euclid(cycle as i128) as usize;
                        prefix[start + width] - prefix[start] > 0
                    });
                    next[p] = reachable;
                    p = (p + s) % m;
                }
            }
            bits = next;
        }
        Values::Cycle { modulus, bits }
    }

    fn enumerate(shape: &Shape) -> Option<Vec<i128>> {
        let (modulus, terms, differ) = shape;
        let mut count: i128 = 1;
        for (_, d) in terms {
            let Domain::Within(pieces) = d else {
                return None;
            };
            let width: i128 = pieces.iter().map(|&(lo, hi)| hi - lo + 1).sum();
            count = count.checked_mul(width)?;
            if count > ENUMERATED {
                return None;
            }
        }
        let choices: Vec<Vec<i128>> = terms
            .iter()
            .map(|(_, d)| match d {
                Domain::Within(pieces) => pieces.iter().flat_map(|&(lo, hi)| lo..=hi).collect(),
                Domain::Free => Vec::new(),
            })
            .collect();
        let mut out = Vec::new();
        let mut index = vec![0usize; terms.len()];
        if choices.iter().any(|c| c.is_empty()) {
            return Some(out);
        }
        loop {
            let allowed = differ.is_none_or(|(i, j)| choices[i][index[i]] != choices[j][index[j]]);
            if allowed {
                let sum = terms.iter().zip(&index).zip(&choices).fold(0i128, |acc, (((c, _), &k), values)| acc + c * values[k]);
                out.push(if *modulus > 0 { sum.rem_euclid(*modulus) } else { sum });
            }
            let mut k = 0;
            loop {
                if k == terms.len() {
                    out.sort_unstable();
                    out.dedup();
                    return Some(out);
                }
                index[k] += 1;
                if index[k] < choices[k].len() {
                    break;
                }
                index[k] = 0;
                k += 1;
            }
        }
    }

    fn hits(&self, modulus: i128, values: &Values) -> bool {
        let (lo, hi) = self.window;
        (lo..=hi).any(|t| values.contains(t - self.difference.constant, modulus))
    }

    pub(super) fn feasible(&self, differ: Option<Unknown>, cache: &mut Shapes) -> Option<bool> {
        if self.surely_apart() {
            return Some(false);
        }
        let copies = differ.and_then(|u| Some((*self.vars.get(&(0, u))?, *self.vars.get(&(1, u))?)));
        if let Some(shape) = self.shape(copies) {
            let modulus = shape.0;
            let values = cache
                .sets
                .entry(shape)
                .or_insert_with_key(|shape| Self::values(shape).map(std::rc::Rc::new))
                .clone();
            if let Some(values) = values {
                return Some(self.hits(modulus, &values));
            }
        }
        let Some((p, q)) = copies else {
            return self.problem.feasible();
        };
        let mut unknown = false;
        for (a, b) in [(p, q), (q, p)] {
            let mut problem = self.problem.clone();
            problem.at_least(Linear::constant(-1).term(a, 1).term(b, -1));
            match problem.feasible() {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }
}

pub(super) fn quickly_apart(
    unknowns: &[UnknownInfo],
    x: &Form,
    y: &Form,
    x_bytes: u32,
    y_bytes: u32,
    variant: &dyn Fn(&UnknownInfo) -> bool,
) -> bool {
    let (lo, hi) = (-(x_bytes as i128) + 1, y_bytes as i128 - 1);
    let constant = (x.constant as i32 as i128) - (y.constant as i32 as i128);
    let (mut g, mut low, mut high) = (WORD, constant, constant);
    let mut add = |u: Unknown, c: i128| {
        if c == 0 {
            return;
        }
        g = gcd(g, c);
        let (a, b) = unknowns[u as usize].range.unwrap_or((0, u32::MAX));
        let (a, b) = if c > 0 { (a as i128 * c, b as i128 * c) } else { (b as i128 * c, a as i128 * c) };
        low += a;
        high += b;
    };
    let (mut i, mut j) = (0, 0);
    while i < x.terms.len() || j < y.terms.len() {
        match (x.terms.get(i), y.terms.get(j)) {
            (Some(&(u, cx)), Some(&(v, cy))) if u == v => {
                i += 1;
                j += 1;
                let (cx, cy) = (cx as i32 as i128, cy as i32 as i128);
                if variant(&unknowns[u as usize]) {
                    add(u, cx);
                    add(u, -cy);
                } else {
                    add(u, cx - cy);
                }
            }
            (Some(&(u, cx)), Some(&(v, _))) if u < v => {
                i += 1;
                add(u, cx as i32 as i128);
            }
            (Some(&(u, cx)), None) => {
                i += 1;
                add(u, cx as i32 as i128);
            }
            (_, Some(&(v, cy))) => {
                j += 1;
                add(v, -(cy as i32 as i128));
            }
            (None, None) => break,
        }
    }
    if g > 1 && !(lo..=hi).any(|t| (t - constant).rem_euclid(g) == 0) {
        return true;
    }
    let span = high - low;
    span + 1 < WORD && !(lo..=hi).any(|t| (t - low).rem_euclid(WORD) <= span)
}


pub(super) fn outcomes(pred: IntPred) -> u8 {
    let (equal, ll, lg, gl, gg) = (1u8, 2u8, 4u8, 8u8, 16u8);
    match pred {
        IntPred::Eq => equal,
        IntPred::Ne => ll | lg | gl | gg,
        IntPred::Ult => ll | lg,
        IntPred::Ule => ll | lg | equal,
        IntPred::Ugt => gl | gg,
        IntPred::Uge => gl | gg | equal,
        IntPred::Slt => ll | gl,
        IntPred::Sle => ll | gl | equal,
        IntPred::Sgt => lg | gg,
        IntPred::Sge => lg | gg | equal,
    }
}

pub(super) fn mirrored(mask: u8) -> u8 {
    (mask & 1) | (mask & 2) << 3 | (mask & 16) >> 3 | (mask & 4) << 1 | (mask & 8) >> 1
}

fn negated(pred: IntPred) -> IntPred {
    match pred {
        IntPred::Eq => IntPred::Ne,
        IntPred::Ne => IntPred::Eq,
        IntPred::Ult => IntPred::Uge,
        IntPred::Uge => IntPred::Ult,
        IntPred::Ule => IntPred::Ugt,
        IntPred::Ugt => IntPred::Ule,
        IntPred::Slt => IntPred::Sge,
        IntPred::Sge => IntPred::Slt,
        IntPred::Sle => IntPred::Sgt,
        IntPred::Sgt => IntPred::Sle,
    }
}

const HALF: i128 = 1 << 31;

impl<'a> Encoding<'a> {
    fn signed(&mut self, w: Var) -> Var {
        if let Some(&s) = self.signs.get(&w) {
            return s;
        }
        let s = self.problem.between(-HALF, HALF - 1);
        let h = self.problem.between(0, 1);
        self.problem.equal(Linear::constant(0).term(w, 1).term(h, -WORD).term(s, -1));
        self.signs.insert(w, s);
        s
    }

    fn outcome(&mut self, bit: u8, (wa, wb): (Var, Var)) -> Clause {
        let unsigned = Linear::constant(0).term(wa, 1).term(wb, -1);
        let mut clause = Clause::default();
        if bit == 1 {
            clause.equal.push(unsigned);
            return clause;
        }
        let (sa, sb) = (self.signed(wa), self.signed(wb));
        let signed = Linear::constant(0).term(sa, 1).term(sb, -1);
        let below = |e: &Linear| Linear::constant(-1).plus(e, -1);
        let above = |e: &Linear| e.clone().offset(-1);
        let (ub, ua) = (below(&unsigned), above(&unsigned));
        let (sb_, sa_) = (below(&signed), above(&signed));
        match bit {
            2 => clause.at_least.extend([ub, sb_]),
            4 => clause.at_least.extend([ub, sa_]),
            8 => clause.at_least.extend([ua, sb_]),
            _ => clause.at_least.extend([ua, sa_]),
        }
        clause
    }

    fn options(&mut self, mask: u8, (wa, wb): (Var, Var)) -> Vec<Clause> {
        let unsigned = Linear::constant(0).term(wa, 1).term(wb, -1);
        let one = |at_least: Vec<Linear>, equal: Vec<Linear>| vec![Clause { equal, at_least }];
        let below = |e: &Linear| Linear::constant(-1).plus(e, -1);
        let above = |e: &Linear| e.clone().offset(-1);
        let at_most = |e: &Linear| Linear::constant(0).plus(e, -1);
        let at_least = |e: &Linear| e.clone();
        let plain = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge];
        for pred in plain {
            if outcomes(pred) != mask {
                continue;
            }
            let e = if matches!(pred, IntPred::Ult | IntPred::Ule | IntPred::Ugt | IntPred::Uge) {
                unsigned
            } else {
                let (sa, sb) = (self.signed(wa), self.signed(wb));
                Linear::constant(0).term(sa, 1).term(sb, -1)
            };
            let bound = match pred {
                IntPred::Ult | IntPred::Slt => below(&e),
                IntPred::Ule | IntPred::Sle => at_most(&e),
                IntPred::Ugt | IntPred::Sgt => above(&e),
                _ => at_least(&e),
            };
            return one(vec![bound], Vec::new());
        }
        if mask == outcomes(IntPred::Eq) {
            return one(Vec::new(), vec![unsigned]);
        }
        if mask == outcomes(IntPred::Ne) {
            return vec![
                Clause { equal: Vec::new(), at_least: vec![below(&unsigned)] },
                Clause { equal: Vec::new(), at_least: vec![above(&unsigned)] },
            ];
        }
        (0..5).filter(|k| mask & (1 << k) != 0).map(|k| self.outcome(1 << k, (wa, wb))).collect()
    }

    pub(super) fn orders(&mut self, side: usize, orders: &[(Form, Form, u8)]) -> usize {
        let mut done = BTreeSet::new();
        loop {
            let present: BTreeSet<Unknown> = self.vars.keys().map(|&(_, u)| u).collect();
            let mut added = false;
            for (i, (a, b, mask)) in orders.iter().enumerate() {
                let mentions = |f: &Form| f.terms.iter().any(|&(u, _)| present.contains(&u));
                if done.contains(&i) || !(mentions(a) || mentions(b)) {
                    continue;
                }
                done.insert(i);
                added = true;
                let (wa, wb) = (self.word(side, a), self.word(side, b));
                let options = self.options(*mask, (wa, wb));
                self.problem.either.push(options);
            }
            if !added {
                break;
            }
        }
        done.len()
    }

    pub(super) fn decide(&mut self, pred: IntPred, x: &Form, y: &Form, classes: &Classes, orders: &[(Form, Form, u8)]) -> Option<bool> {
        let (wx, wy) = (self.word(0, x), self.word(0, y));
        let none = Classes::new();
        let constrained = self.limits([classes, &none]) + self.orders(0, orders) > 0
            || self.vars.keys().any(|&(_, u)| self.unknowns[u as usize].values.is_some());
        if !constrained {
            return None;
        }
        let mut answers = [None; 2];
        for (k, p) in [pred, negated(pred)].iter().copied().enumerate() {
            let options = self.options(outcomes(p), (wx, wy));
            let mut problem = self.problem.clone();
            problem.either.push(options);
            answers[k] = problem.feasible();
        }
        match answers {
            [Some(false), Some(true)] => Some(false),
            [Some(true), Some(false)] => Some(true),
            _ => None,
        }
    }
}
