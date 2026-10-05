use super::form::*;
use super::limits::Limits;
use super::program::{Conditions, Program};
use super::symbols::Symbols;
use crate::rdna_spmd::ir::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Target {
    Param(usize),
    Slot(u32, u32),
}

pub(super) trait Queries<'a> {
    fn program(&self) -> &Program<'a>;
    fn symbols(&self) -> &Symbols<'a>;
    fn symbols_mut(&mut self) -> &mut Symbols<'a>;
    fn reason<T>(&mut self, f: impl FnOnce(&mut Conditions, &Program<'a>) -> T) -> T;
    fn value(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value>;
    fn bit(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>>;
    fn limits(&mut self, b: BlockId) -> std::rc::Rc<Limits>;
    fn high(&mut self, v: ValueId, lane: usize) -> Form;
    fn word_bit(&mut self, w: ValueId, bit: usize) -> Option<bool>;
    fn slot_entry(&mut self, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value>;
    fn decision(&mut self, b: BlockId) -> Option<bool>;
    fn reached(&mut self, b: BlockId) -> bool;
    fn loop_bit(&mut self, header: BlockId, index: usize, lane: usize) -> Option<bool>;
    fn recur(&mut self, header: BlockId, lane: usize, target: Target) -> Option<Value>;
    fn sequence(&mut self, v: ValueId, header: BlockId, index: usize, lane: usize) -> Option<Value>;

    fn operand(&mut self, x: ValueId, at: BlockId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        let (value, reliance) = self.value(x, lane, assume);
        (self.symbols_mut().leave(value, at, lane), reliance)
    }
}
