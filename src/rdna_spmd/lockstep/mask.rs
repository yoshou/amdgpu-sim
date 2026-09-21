use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::ir::ValueId;

pub(super) struct Atom {

    pub value: ValueId,

    pub uniform: bool,

    pub origin: Option<ValueId>,
}

pub(super) struct Masks {
    pub m: Manager,
    atoms: Vec<Atom>,
    invariants: Bdd,
    nonempty: Vec<Bdd>,
}

#[derive(Clone)]
pub(super) struct Knowledge {
    invariants: Bdd,
    nonempty: usize,
}

impl Masks {
    pub fn new() -> Self {
        Self {
            m: Manager::new(),
            atoms: Vec::new(),
            invariants: Bdd::TRUE,
            nonempty: Vec::new(),
        }
    }

    pub fn atom(&mut self, value: ValueId, uniform: bool, origin: Option<ValueId>) -> Bdd {
        let var = self.atoms.len() as u32;
        self.atoms.push(Atom {
            value,
            uniform,
            origin,
        });
        self.m.var(var)
    }

    pub fn at(&self, var: u32) -> &Atom {
        &self.atoms[var as usize]
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    pub fn and(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.m.and(a, b)
    }
    pub fn or(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.m.or(a, b)
    }
    pub fn not(&mut self, a: Bdd) -> Bdd {
        self.m.not(a)
    }
    pub fn ite(&mut self, c: Bdd, a: Bdd, b: Bdd) -> Bdd {
        self.m.ite(c, a, b)
    }

    pub fn uniform(&mut self, f: Bdd) -> bool {
        let support = self.m.support(f);
        support.into_iter().all(|var| self.at(var).uniform)
    }

    pub fn exact(&mut self, f: Bdd) -> Bdd {
        let care = self.invariants;
        self.simplest(f, care)
    }

    pub fn choice(&mut self, f: Bdd, care: Bdd) -> Bdd {
        let care = self.m.and(care, self.invariants);
        self.simplest(f, care)
    }

    fn simplest(&mut self, f: Bdd, care: Bdd) -> Bdd {
        let mut r = self.m.restrict(f, care);
        let agreed = self.m.and(r, care);
        loop {
            let mut dropped = false;
            for var in self.m.support(r) {
                for value in [true, false] {
                    let g = self.m.cofactor(r, var, value);
                    if self.m.and(g, care) == agreed {
                        r = g;
                        dropped = true;
                        break;
                    }
                }
                if dropped {
                    break;
                }
            }
            if !dropped {
                return r;
            }
        }
    }

    pub fn holds_a_lane(&mut self, f: Bdd) -> bool {
        if f == Bdd::TRUE {
            return true;
        }
        if f == Bdd::FALSE {
            return false;
        }
        let known = self.nonempty.clone();
        known.into_iter().any(|n| {
            let care = self.invariants;
            let implied = self.m.and(n, care);
            self.m.implies(implied, f)
        })
    }

    pub fn coexist(&mut self, a: Bdd, b: Bdd) -> bool {
        let care = self.invariants;
        let lane = self.m.and(a, care);
        let free: Vec<u32> = self
            .m
            .support(b)
            .into_iter()
            .chain(self.m.support(care))
            .filter(|&var| !self.at(var).uniform)
            .collect();
        let past = self.atoms.len() as u32;
        let renamed: Vec<(u32, Bdd)> = free
            .into_iter()
            .map(|var| (var, self.m.var(past + var)))
            .collect();
        let rename = |var: u32| {
            renamed
                .iter()
                .find(|&&(from, _)| from == var)
                .map(|&(_, to)| to)
        };
        let other = self.m.and(b, care);
        let other = self.m.compose(other, &rename);
        self.m.and(lane, other) != Bdd::FALSE
    }

    pub fn knowledge(&self) -> Knowledge {
        Knowledge {
            invariants: self.invariants,
            nonempty: self.nonempty.len(),
        }
    }

    pub fn restore(&mut self, k: Knowledge) {
        self.invariants = k.invariants;
        self.nonempty.truncate(k.nonempty);
    }

    pub fn assume(&mut self, f: Bdd) {
        self.invariants = self.m.and(self.invariants, f);
    }

    pub fn assume_a_lane(&mut self, f: Bdd) {
        if f != Bdd::TRUE {
            self.nonempty.push(f);
        }
    }

    pub fn assume_within(&mut self, inner: Bdd, outer: Bdd) {
        let ni = self.m.not(inner);
        let implication = self.m.or(ni, outer);
        self.assume(implication);
    }
}
