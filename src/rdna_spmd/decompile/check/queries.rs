use super::super::logic::Logic;
use super::program::Program;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::ir::*;

pub(super) trait Queries<'a> {
    fn program(&self) -> &Program<'a>;
    fn logic(&mut self) -> &mut Logic;
    fn safe(&self) -> Bdd;
    fn h(&self, v: ValueId) -> Bdd;
    fn word(&self, v: ValueId) -> Bdd;
    fn reachable(&self, b: BlockId) -> Bdd;
    fn loaded(&self, at: (BlockId, usize)) -> Option<Bdd>;
    fn reordered(&self, at: (BlockId, usize)) -> Option<Bdd>;
    fn masked(&self, v: ValueId) -> bool;
    fn faithful(&self, v: ValueId) -> bool;
    fn settles(&mut self, v: ValueId, op: Op) -> bool;
    fn require(&mut self, block: BlockId, index: usize, reason: &'static str, difference: Bdd);
    fn stopped(&self) -> bool;
    fn demand(&mut self, provenance: u64, condition: Bdd);

    #[inline]
    fn whole(&self, v: ValueId) -> Bdd {
        if self.program().facts.lane_word[v.0] {
            self.word(v)
        } else {
            self.h(v)
        }
    }

    #[inline]
    fn bit(&mut self, v: ValueId) -> Bdd {
        let (f, facts) = (self.program().f, self.program().facts);
        self.logic().bit(f, facts, v)
    }

    #[inline]
    fn view(&mut self, v: ValueId) -> Bdd {
        let (f, facts) = (self.program().f, self.program().facts);
        self.logic().view(f, facts, v)
    }

    #[inline]
    fn and(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.logic().m.and(a, b)
    }

    #[inline]
    fn or(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.logic().m.or(a, b)
    }

    #[inline]
    fn not(&mut self, a: Bdd) -> Bdd {
        self.logic().m.not(a)
    }
}
