use super::answers::Answers;
use super::atoms::{self, Atom, Atoms, Choice};
use super::cells::{self, Cells};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;
use std::rc::Rc;

pub(super) trait Queries {
    fn m(&mut self) -> &mut Manager;
    fn atoms(&self) -> &Atoms;
    fn atom(&mut self, atom: Atom) -> Bdd;
    fn support(&mut self, f: Bdd) -> Rc<Vec<u32>>;
    fn local(&mut self, c: Choice) -> Bdd;
    fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd;
    fn carried(&self, param: ValueId) -> bool;
    fn lane_values(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[u32; 32]>;
    fn cells(&self) -> &Cells;
    fn cells_mut(&mut self) -> &mut Cells;
    fn answers(&self) -> &Answers;
    fn answers_mut(&mut self) -> &mut Answers;
    fn bit(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Bdd;
    fn view(&mut self, f: &Func, facts: &Facts, w: ValueId) -> Bdd;

    #[inline]
    fn exists(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        atoms::exists(self.m(), vars, f)
    }

    #[inline]
    fn uniform_atom(&self, facts: &Facts, var: u32) -> bool {
        cells::uniform_atom(self.atoms(), self.cells(), facts, var)
    }

    #[inline]
    fn scope(&self, facts: &Facts, var: u32) -> Option<BlockId> {
        atoms::scope(self.atoms().of(var), facts)
    }

    fn scoped(&mut self, facts: &Facts, f: Bdd, block: BlockId) -> Vec<u32> {
        self.support(f)
            .iter()
            .copied()
            .filter(|&v| self.scope(facts, v) == Some(block))
            .collect()
    }

    fn lanes(&mut self, holds: impl Fn(u32) -> bool) -> Bdd {
        let bits: Vec<Bdd> = (0..5).map(|i| self.atom(Atom::Lane(i))).collect();
        let mut any = Bdd::FALSE;
        for l in (0..32u32).filter(|&l| holds(l)) {
            let mut one = Bdd::TRUE;
            for (i, &bit) in bits.iter().enumerate() {
                let literal = if l >> i & 1 == 1 { bit } else { self.m().not(bit) };
                one = self.m().and(one, literal);
            }
            any = self.m().or(any, one);
        }
        any
    }

    #[inline]
    fn word(&mut self, k: u32) -> Bdd {
        self.lanes(|l| k >> l & 1 == 1)
    }

    fn lane_dependent(&mut self, f: Bdd) -> bool {
        let support = self.support(f);
        support.iter().any(|&v| matches!(self.atoms().get(v), Some(Atom::Lane(_))))
    }

    fn at_lane(&mut self, f: Bdd, lane: u32) -> Bdd {
        let mut f = f;
        for i in 0..5u8 {
            if let Some(var) = self.atoms().var(Atom::Lane(i)) {
                f = self.m().cofactor(f, var, lane >> i & 1 == 1);
            }
        }
        f
    }

    fn word_is(&mut self, w: ValueId, value: u32) -> Bdd {
        let mut g = Bdd::TRUE;
        for i in 0..5u8 {
            let bit = self.atom(Atom::WordBit(w, i));
            let literal = if value >> i & 1 == 1 { bit } else { self.m().not(bit) };
            g = self.m().and(g, literal);
        }
        g
    }
}
