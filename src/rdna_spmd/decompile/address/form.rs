use crate::rdna_spmd::ir::*;

pub const LANES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Region {
    Allocation(u64),
    Exposed,
    Kernarg,
    Dispatch,
    Lds,
    Private,
}

pub type Unknown = u32;

#[derive(Clone, Debug)]
pub struct UnknownInfo {
    pub shared: bool,
    pub block: BlockId,
    pub rank: usize,
    pub range: Option<(u32, u32)>,
    pub through: Vec<BlockId>,
    pub values: Option<std::rc::Rc<[u32]>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wide {
    pub constant: i128,
    pub terms: Vec<(Unknown, i128)>,
    pub words: Vec<(Form, i128)>,
}

impl Wide {
    pub(super) fn plus(&self, other: &Wide, sign: i128) -> Wide {
        let mut terms = self.terms.clone();
        for &(u, c) in &other.terms {
            match terms.iter_mut().find(|(v, _)| *v == u) {
                Some((_, old)) => *old += sign * c,
                None => terms.push((u, sign * c)),
            }
        }
        terms.retain(|&(_, c)| c != 0);
        terms.sort_unstable();
        let mut words = self.words.clone();
        for (f, c) in &other.words {
            match words.iter_mut().find(|(g, _)| g == f) {
                Some((_, old)) => *old += sign * c,
                None => words.push((f.clone(), sign * c)),
            }
        }
        words.retain(|(_, c)| *c != 0);
        Wide {
            constant: self.constant + sign * other.constant,
            terms,
            words,
        }
    }

    pub(super) fn times(&self, k: i128) -> Wide {
        Wide {
            constant: self.constant * k,
            terms: self.terms.iter().filter(|_| k != 0).map(|&(u, c)| (u, c * k)).collect(),
            words: self.words.iter().filter(|_| k != 0).map(|(f, c)| (f.clone(), c * k)).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Form {
    pub constant: u32,
    pub terms: Vec<(Unknown, u32)>,
}

impl Form {
    pub fn constant(k: u32) -> Self {
        Self {
            constant: k,
            terms: Vec::new(),
        }
    }

    pub(super) fn unknown(u: Unknown) -> Self {
        Self {
            constant: 0,
            terms: vec![(u, 1)],
        }
    }

    pub fn as_constant(&self) -> Option<u32> {
        self.terms.is_empty().then_some(self.constant)
    }

    pub(super) fn combine(&self, other: &Form, sign: u32) -> Form {
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        let (mut i, mut j) = (0, 0);
        while i < self.terms.len() || j < other.terms.len() {
            let take_self = j == other.terms.len()
                || (i < self.terms.len() && self.terms[i].0 < other.terms[j].0);
            let take_other = i == self.terms.len()
                || (j < other.terms.len() && other.terms[j].0 < self.terms[i].0);
            if take_self {
                terms.push(self.terms[i]);
                i += 1;
            } else if take_other {
                let (u, c) = other.terms[j];
                terms.push((u, c.wrapping_mul(sign)));
                j += 1;
            } else {
                let (u, c) = self.terms[i];
                let sum = c.wrapping_add(other.terms[j].1.wrapping_mul(sign));
                if sum != 0 {
                    terms.push((u, sum));
                }
                i += 1;
                j += 1;
            }
        }
        Form {
            constant: self.constant.wrapping_add(other.constant.wrapping_mul(sign)),
            terms,
        }
    }

    pub fn add(&self, other: &Form) -> Form {
        self.combine(other, 1)
    }

    pub fn sub(&self, other: &Form) -> Form {
        self.combine(other, u32::MAX)
    }

    pub(super) fn scale(&self, k: u32) -> Form {
        if k == 0 {
            return Form::constant(0);
        }
        Form {
            constant: self.constant.wrapping_mul(k),
            terms: self
                .terms
                .iter()
                .filter_map(|&(u, c)| {
                    let c = c.wrapping_mul(k);
                    (c != 0).then_some((u, c))
                })
                .collect(),
        }
    }

    pub(super) fn alignment(&self) -> u32 {
        self.terms
            .iter()
            .map(|&(_, c)| c.trailing_zeros())
            .min()
            .unwrap_or(32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub form: Form,
    pub region: Option<Region>,
}

impl Value {
    pub(super) fn of(form: Form) -> Self {
        Self { form, region: None }
    }

    pub(super) fn constant(k: u32) -> Self {
        Self::of(Form::constant(k))
    }
}

pub(super) type ValueKey = (ValueId, u8, Option<ValueId>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reliance {
    pub(super) used: bool,
    pub(super) open: bool,
    pub(super) lane: bool,
}

impl std::ops::BitOrAssign for Reliance {
    fn bitor_assign(&mut self, other: Self) {
        self.used |= other.used;
        self.open |= other.open;
        self.lane |= other.lane;
    }
}

pub(super) const ALL: u8 = u8::MAX;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Regions {
    pub(super) any: bool,
    pub(super) list: Vec<Option<Region>>,
}

impl Regions {
    #[inline]
    pub fn one(r: Option<Region>) -> Self {
        Self {
            any: false,
            list: vec![r],
        }
    }

    #[inline]
    pub(super) fn any() -> Self {
        Self {
            any: true,
            list: Vec::new(),
        }
    }

    #[inline]
    pub fn add(&mut self, r: Option<Region>) {
        if let Err(i) = self.list.binary_search(&r) {
            self.list.insert(i, r);
        }
    }

    #[inline]
    pub fn union(&mut self, other: &Regions) {
        self.any |= other.any;
        for &r in &other.list {
            self.add(r);
        }
    }

    #[inline]
    pub(super) fn joined(mut self, other: &Regions) -> Self {
        self.union(other);
        self
    }

    pub(super) fn combine(&self, other: &Regions, f: impl Fn(Option<Region>, Option<Region>) -> Option<Region>) -> Self {
        if self.any || other.any {
            return Self::any();
        }
        let mut out = Self::default();
        for &a in &self.list {
            for &b in &other.list {
                out.add(f(a, b));
            }
        }
        out
    }

    pub(super) fn single(&self) -> Option<Option<Region>> {
        match self.list.as_slice() {
            [r] if !self.any => Some(*r),
            _ => None,
        }
    }

    pub fn single_region(&self) -> bool {
        self.single().is_some()
    }

    #[inline]
    pub(super) fn within(&self, other: &Regions) -> bool {
        other.any || (!self.any && self.list.iter().all(|r| other.list.contains(r)))
    }

    pub fn overlaps(&self, other: &Regions, f: impl Fn(Option<Region>, Option<Region>) -> bool) -> bool {
        self.any || other.any || self.list.iter().any(|&a| other.list.iter().any(|&b| f(a, b)))
    }

    #[inline]
    pub fn reaches(&self, r: Option<Region>, f: impl Fn(Option<Region>, Option<Region>) -> bool) -> bool {
        self.any || self.list.iter().any(|&a| f(a, r))
    }

    #[inline]
    pub fn lost(&self) -> bool {
        self.any || self.list.contains(&None)
    }

}

pub(super) fn in_lane<T>(x: T) -> Assumed<T> {
    (
        x,
        Reliance {
            lane: true,
            ..Reliance::default()
        },
    )
}

pub(super) type Assumed<T> = (T, Reliance);

pub(super) fn unassumed<T>(x: T) -> Assumed<T> {
    (x, Reliance::default())
}

pub fn aligns(m: u32) -> bool {
    (!m).wrapping_add(1).is_power_of_two()
}

pub(super) fn low_bits(form: &Form, c: u32, op: IntOp) -> Option<Form> {
    let j = form.alignment();
    if form.terms.is_empty() || j >= 32 || c >= 1 << j {
        return None;
    }
    let low = form.constant & ((1 << j) - 1);
    let result = match op {
        IntOp::And => return Some(Form::constant(low & c)),
        IntOp::Or => low | c,
        IntOp::Xor => low ^ c,
        _ => return None,
    };
    Some(form.sub(&Form::constant(low)).add(&Form::constant(result)))
}

#[cfg(test)]
mod tests {
    use super::super::super::testing::*;
    use super::*;

    #[test]
    fn low_bits_matches_the_operation_for_every_value_of_the_unknowns() {
        let mut r = Random::new(19);
        for _ in 0..20000 {
            let form = Form {
                constant: interesting(&mut r),
                terms: vec![(0, (interesting(&mut r) | 1) << r.below(8)), (1, (interesting(&mut r) | 1) << r.below(8))],
            };
            let c = r.below(512) as u32;
            for op in [IntOp::And, IntOp::Or, IntOp::Xor] {
                let Some(result) = low_bits(&form, c, op) else { continue };
                for _ in 0..16 {
                    let values = [r.next() as u32, r.next() as u32];
                    let value = |f: &Form| {
                        f.terms
                            .iter()
                            .fold(f.constant, |a, &(u, k)| a.wrapping_add(k.wrapping_mul(values[u as usize])))
                    };
                    let v = value(&form);
                    let expected = match op {
                        IntOp::And => v & c,
                        IntOp::Or => v | c,
                        _ => v ^ c,
                    };
                    assert_eq!(value(&result), expected, "{:?} {:?} {:#x} at {:?}", form, op, c, values);
                }
            }
        }
    }

    #[test]
    fn forms_add_subtract_and_scale_as_words() {
        let mut r = Random::new(23);
        for _ in 0..20000 {
            let form = |r: &mut Random| {
                let mut terms: Vec<(Unknown, u32)> = Vec::new();
                for u in 0..4 {
                    let c = interesting(r);
                    if r.below(2) == 0 && c != 0 {
                        terms.push((u, c));
                    }
                }
                terms.sort();
                Form {
                    constant: interesting(r),
                    terms,
                }
            };
            let (a, b) = (form(&mut r), form(&mut r));
            let k = interesting(&mut r);
            let values: Vec<u32> = (0..4).map(|_| r.next() as u32).collect();
            let value = |f: &Form| {
                f.terms
                    .iter()
                    .fold(f.constant, |acc, &(u, c)| acc.wrapping_add(c.wrapping_mul(values[u as usize])))
            };
            let canonical = |f: &Form| {
                f.terms.windows(2).all(|w| w[0].0 < w[1].0) && f.terms.iter().all(|&(_, c)| c != 0)
            };
            for (result, expected) in [
                (a.add(&b), value(&a).wrapping_add(value(&b))),
                (a.sub(&b), value(&a).wrapping_sub(value(&b))),
                (a.scale(k), value(&a).wrapping_mul(k)),
            ] {
                assert_eq!(value(&result), expected);
                assert!(canonical(&result), "{:?}", result);
            }
            assert_eq!(a.sub(&a).as_constant(), Some(0));
        }
    }
}
