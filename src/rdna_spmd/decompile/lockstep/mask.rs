//! The algebra of lane masks.
//!
//! A mask is a boolean function of *atoms*: one-bit packet values -- a
//! comparison a lane made, the mask a loop carries, a bit a merge decided --
//! read per lane. Holding masks as functions rather than as emitted code is
//! what lets the lowering see that the two arms of a branch differ only in its
//! condition, that a loop whose exit every lane agrees on leaves no lane
//! behind, and that a query over lanes whose atoms are all uniform is a scalar
//! test rather than a reduction.
//!
//! Two kinds of knowledge sharpen it, both true of *every* lane where they are
//! recorded, so a mask simplified under them is still exact:
//!
//! - `invariants`: relations that hold wherever the lowering is, such as a
//!   loop's mask lying inside the mask that entered it, or a uniform condition
//!   a scalar branch has already decided.
//! - `nonempty`: masks at least one lane is known to be in, which is what
//!   makes a query over them answer yes without a reduction.

use super::super::bdd::{Bdd, Manager};
use crate::rdna_spmd::ir::ValueId;

pub(super) struct Atom {
    /// The packet value whose lane bits the atom stands for.
    pub value: ValueId,
    /// Whether every lane reads the same bit.
    pub uniform: bool,
    /// The lane-program value the atom came from, when it came from one.
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

    /// Whether every lane agrees on the mask's value.
    pub fn uniform(&mut self, f: Bdd) -> bool {
        let support = self.m.support(f);
        support.into_iter().all(|var| self.at(var).uniform)
    }

    /// The simplest function that agrees with `f` on every lane the knowledge
    /// admits. Sound for masks, which must be exact everywhere, because the
    /// invariants hold for every lane.
    pub fn exact(&mut self, f: Bdd) -> Bdd {
        let care = self.invariants;
        self.simplest(f, care)
    }

    /// The simplest function that agrees with `f` wherever `care` holds; for
    /// choosing between values, whose lanes outside `care` are never read.
    pub fn choice(&mut self, f: Bdd, care: Bdd) -> Bdd {
        let care = self.m.and(care, self.invariants);
        self.simplest(f, care)
    }

    /// Restricts `f` to `care`, then drops every test the care set still
    /// leaves redundant: restriction only drops a test where the care set
    /// decides it before `f` reads it, so a test `f` reads first -- a mask
    /// made earlier than the one that lies inside it -- survives it.
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

    /// Whether some lane is known to be in `f`.
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

    /// Whether one lane can be in `a` while another is in `b`. Uniform atoms
    /// read the same in both lanes; every other atom is free, so the second
    /// lane reads it through a variable of its own, numbered past every atom.
    /// The renamed functions only live for this test.
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

    /// Records a relation that holds for every lane from here on.
    pub fn assume(&mut self, f: Bdd) {
        self.invariants = self.m.and(self.invariants, f);
    }

    /// Records that `f` holds a lane from here on.
    pub fn assume_a_lane(&mut self, f: Bdd) {
        if f != Bdd::TRUE {
            self.nonempty.push(f);
        }
    }

    /// Records that every lane in `inner` is in `outer`.
    pub fn assume_within(&mut self, inner: Bdd, outer: Bdd) {
        let ni = self.m.not(inner);
        let implication = self.m.or(ni, outer);
        self.assume(implication);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atoms(m: &mut Masks, kinds: &[bool]) -> Vec<Bdd> {
        kinds
            .iter()
            .enumerate()
            .map(|(k, &uniform)| m.atom(ValueId(k), uniform, None))
            .collect()
    }

    #[test]
    fn a_loop_every_lane_leaves_together_lets_no_lane_exit_early() {
        let mut m = Masks::new();
        let vars = atoms(&mut m, &[false, true, false]);
        let (mask, trips, hit) = (vars[0], vars[1], vars[2]);
        m.assume_a_lane(mask);
        let carry = m.and(mask, trips);
        let leave = m.not(trips);
        let leave = m.and(mask, leave);
        assert!(
            !m.coexist(leave, carry),
            "a uniform trip count leaves no lane behind"
        );
        let varying = m.and(mask, hit);
        let nhit = m.not(hit);
        let staying = m.and(mask, nhit);
        assert!(
            m.coexist(varying, staying),
            "a lane can leave on its own hit while another stays"
        );
    }

    #[test]
    fn a_query_over_uniform_atoms_needs_no_lane_reduction() {
        let mut m = Masks::new();
        let vars = atoms(&mut m, &[false, true]);
        let (mask, cond) = (vars[0], vars[1]);
        let taken = m.and(mask, cond);
        assert!(!m.uniform(taken));
        assert!(m.uniform(cond));
        m.assume_a_lane(mask);
        assert!(m.holds_a_lane(mask));
        assert!(!m.holds_a_lane(taken));
        assert_eq!(
            m.choice(taken, mask),
            cond,
            "the arm differs only in its condition"
        );
    }

    #[test]
    fn knowledge_returns_to_what_it_was_outside_the_scope() {
        let mut m = Masks::new();
        let vars = atoms(&mut m, &[false, false]);
        let (outer, inner) = (vars[0], vars[1]);
        let both = m.and(inner, outer);
        let saved = m.knowledge();
        m.assume_a_lane(inner);
        m.assume_within(inner, outer);
        assert!(m.holds_a_lane(inner));
        assert_eq!(
            m.exact(both),
            inner,
            "inside the scope the inner mask is within the outer"
        );
        m.restore(saved);
        assert!(!m.holds_a_lane(inner));
        assert_eq!(m.exact(both), both);
    }
}
