pub(super) mod constant;
pub(super) mod dataflow;
pub(in crate::rdna_spmd) mod masks;
pub(super) mod memory;
pub(in crate::rdna_spmd) mod uniformity;

use super::dialect::DialectRegistry;
use super::ir::Func;
use super::program::Parameter;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) use constant::{Constants, DispatchConstants};
pub(crate) use masks::{Exec, Masks, Predication};
pub(crate) use memory::{Access, Accesses};
pub(crate) use uniformity::Uniformity;

#[derive(Clone, Copy)]
pub(crate) struct Packet { pub aligned: bool }

#[derive(Clone, Copy)]
pub(crate) struct Context<'r> {
    pub registry: &'r DialectRegistry,
    pub inputs: &'r [Parameter],
    pub exec_index: usize,
    pub lanes: u32,
    pub entry_full: bool,
    pub exec_initial: bool,
    pub packet: Option<Packet>,
}

impl<'r> Context<'r> {
    pub fn new(registry: &'r DialectRegistry, inputs: &'r [Parameter], exec_index: usize, lanes: u32) -> Self {
        Self { registry, inputs, exec_index, lanes, entry_full: false, exec_initial: false, packet: None }
    }
}

pub(crate) trait Analysis: 'static {
    type Result: PartialEq + 'static;
    const NAME: &'static str;
    fn compute(f: &Func, analyses: &Analyses) -> Self::Result;
}

pub(crate) struct Preserved(Vec<TypeId>);
impl Preserved {
    pub fn none() -> Self { Self(Vec::new()) }
    pub fn of<A: Analysis>() -> Self { Self::none().and::<A>() }
    pub fn and<A: Analysis>(mut self) -> Self { self.0.push(TypeId::of::<A>()); self }
    fn contains(&self, id: TypeId) -> bool { self.0.contains(&id) }
}

struct Slot {
    id: TypeId,
    name: &'static str,
    result: Rc<dyn Any>,
    dependencies: Vec<TypeId>,
    recompute: fn(&Func, &Analyses) -> Rc<dyn Any>,
    equals: fn(&dyn Any, &dyn Any) -> bool,
}

fn recompute<A: Analysis>(f: &Func, analyses: &Analyses) -> Rc<dyn Any> { Rc::new(A::compute(f, analyses)) }
fn equals<A: Analysis>(a: &dyn Any, b: &dyn Any) -> bool { a.downcast_ref::<A::Result>() == b.downcast_ref::<A::Result>() }

pub(crate) struct Analyses<'r> {
    ctx: Context<'r>,
    slots: RefCell<Vec<Slot>>,
    computing: RefCell<Vec<(TypeId, &'static str, Vec<TypeId>)>>,
}

impl<'r> Analyses<'r> {
    pub fn new(ctx: Context<'r>) -> Self {
        Self { ctx, slots: RefCell::new(Vec::new()), computing: RefCell::new(Vec::new()) }
    }
    pub fn context(&self) -> &Context<'r> { &self.ctx }
    pub fn get<A: Analysis>(&self, f: &Func) -> Rc<A::Result> {
        let id = TypeId::of::<A>();
        if let Some((_, _, dependencies)) = self.computing.borrow_mut().last_mut() {
            if !dependencies.contains(&id) { dependencies.push(id); }
        }
        let cached = self.slots.borrow().iter().find(|slot| slot.id == id).map(|slot| Rc::clone(&slot.result));
        if let Some(result) = cached {
            return result.downcast::<A::Result>().ok().expect("analysis result of another type");
        }
        {
            let mut computing = self.computing.borrow_mut();
            assert!(computing.iter().all(|(other, ..)| *other != id), "analysis {} requires itself", A::NAME);
            computing.push((id, A::NAME, Vec::new()));
        }
        let result = Rc::new(A::compute(f, self));
        let (_, _, dependencies) = self.computing.borrow_mut().pop().expect("analysis computation frame");
        let erased: Rc<dyn Any> = result.clone();
        self.slots.borrow_mut().push(Slot { id, name: A::NAME, result: erased, dependencies, recompute: recompute::<A>, equals: equals::<A> });
        result
    }
    pub fn invalidate(&mut self, preserved: &Preserved) {
        let mut kept: Vec<TypeId> = Vec::new();
        self.slots.get_mut().retain(|slot| {
            let keep = preserved.contains(slot.id) && slot.dependencies.iter().all(|d| kept.contains(d));
            if keep { kept.push(slot.id); }
            keep
        });
    }
    pub fn audit(&self, f: &Func, pass: &str) {
        let fresh = Analyses::new(self.ctx);
        for slot in self.slots.borrow().iter() {
            let again = (slot.recompute)(f, &fresh);
            assert!((slot.equals)(&*slot.result, &*again), "{pass}: claims to preserve {} but changed it", slot.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::{Block, BlockId, Inst, IntOp, Op, Term, Ty, ValueId};
    use std::collections::BTreeMap;

    fn func() -> (Func, ValueId) {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let exec = f.value(Ty::I1); let two = f.value(Ty::I32); let four = f.value(Ty::I32);
        f.blocks.insert(BlockId(0), Block { params: vec![(exec, Ty::I1)], insts: vec![
            Inst::Core { value: two, ty: Ty::I32, op: Op::Const(Ty::I32, 2) },
            Inst::Core { value: four, ty: Ty::I32, op: Op::Int(IntOp::Add, two, two) },
        ], term: Term::Ret(vec![four]) });
        (f, four)
    }

    #[test]
    fn results_are_cached_and_dependencies_fall_with_what_they_depend_on() {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        let (f, four) = func();
        let mut an = Analyses::new(Context::new(&registry, &[], 0, 16));
        let masks = an.get::<Masks>(&f);
        assert_eq!(an.get::<Constants>(&f)[four.0], Some(4));
        assert!(Rc::ptr_eq(&masks, &an.get::<Masks>(&f)));
        let order: Vec<&str> = an.slots.borrow().iter().map(|s| s.name).collect();
        assert_eq!(order, vec![Constants::NAME, Predication::NAME, Masks::NAME]);
        an.invalidate(&Preserved::of::<Masks>().and::<Predication>());
        assert!(an.slots.borrow().is_empty(), "masks and predication depend on the constants that were not preserved");
        an.get::<Masks>(&f);
        an.invalidate(&Preserved::of::<Constants>().and::<Masks>());
        let kept: Vec<&str> = an.slots.borrow().iter().map(|s| s.name).collect();
        assert_eq!(kept, vec![Constants::NAME]);
        an.audit(&f, "identity");
    }

    #[test]
    #[should_panic(expected = "claims to preserve constants")]
    fn an_audit_rejects_a_false_preservation_claim() {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        let (mut f, _) = func();
        let mut an = Analyses::new(Context::new(&registry, &[], 0, 16));
        an.get::<Constants>(&f);
        if let Inst::Core { op, .. } = &mut f.blocks.get_mut(&BlockId(0)).unwrap().insts[0] { *op = Op::Const(Ty::I32, 3); }
        an.invalidate(&Preserved::of::<Constants>());
        an.audit(&f, "edit");
    }
}
