use super::super::analysis::{Analyses, Masks};
use super::super::ir::{*, Cvt, Op, ValueId};
use std::collections::BTreeMap;

pub(crate) struct Narrow;
impl super::Pass for Narrow {
    fn name(&self) -> &str { "narrow" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let masks = analyses.get::<Masks>(f);
        let (narrowed, rewrites) = apply(f, &masks);
        rewrites > 0 || narrowed.iter().any(|&b| b)
    }
}

#[cfg(test)]
pub(crate) fn run(f: &mut Func, masks: &Masks) -> Vec<bool> { apply(f, masks).0 }

fn apply(f: &mut Func, masks: &Masks) -> (Vec<bool>, usize) {
    let mut narrowed = vec![false; f.types.len()];
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            if let Inst::Core { value, ty, op } = inst {
                if let Op::Select(_, new, _) = *op {
                    if masks.predicated[value.0].is_some_and(|(_, exec)| masks.exposed[value.0] == 0 || masks.full[exec.0]) {
                        narrowed[value.0] = true;
                        *op = Op::Convert(Cvt::Bitcast, *ty, new);
                    }
                }
            }
        }
    }
    let rewrites = forward(f, masks, &narrowed) + distribute(f, masks, &narrowed);
    narrowed.resize(f.types.len(), false);
    (narrowed, rewrites)
}

fn distribute(f: &mut Func, masks: &Masks, narrowed: &[bool]) -> usize {
    let mut count = 0;
    let mut defs: Vec<Option<Op>> = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { defs[value.0] = Some(*op); } }
    }
    let exposed = |v: ValueId| masks.predicated[v.0].is_some() && !narrowed[v.0];
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut insts = Vec::with_capacity(old.len());
        for inst in old {
            if let Inst::Core { value, ty, op: Op::Pack64(a, b) } = inst {
                if let (Some(Op::Select(c, x, y)), Some(Op::Select(d, z, w))) = (defs[a.0], defs[b.0]) {
                    if c == d && exposed(a) && exposed(b) {
                        let lo = f.value(ty);
                        let hi = f.value(ty);
                        insts.push(Inst::Core { value: lo, ty, op: Op::Pack64(x, z) });
                        insts.push(Inst::Core { value: hi, ty, op: Op::Pack64(y, w) });
                        insts.push(Inst::Core { value, ty, op: Op::Select(c, lo, hi) });
                        count += 1;
                        continue;
                    }
                }
            }
            insts.push(inst);
        }
        f.blocks.get_mut(&id).unwrap().insts = insts;
    }
    count
}

fn forward(f: &mut Func, masks: &Masks, narrowed: &[bool]) -> usize {
    let mut count = 0;
    let mut defs: Vec<Option<Op>> = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { defs[value.0] = Some(*op); } }
    }
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let mut insts = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut out: Vec<Inst> = Vec::with_capacity(insts.len());
        let mut memo: BTreeMap<(ValueId, ValueId), Option<ValueId>> = BTreeMap::new();
        for (index, inst) in insts.iter_mut().enumerate() {
            let current = masks.at(id, index);
            let mut fresh: Vec<Inst> = Vec::new();
            let mut substitute = |v: ValueId, f: &mut Func, fresh: &mut Vec<Inst>| {
                if let Some(&hit) = memo.get(&(v, current)) { return hit.unwrap_or(v); }
                let resolved = resolve(v, current, masks, narrowed, &defs, f, fresh, 3);
                memo.insert((v, current), resolved);
                resolved.unwrap_or(v)
            };
            match inst {
                Inst::Core { value, op, .. } => {
                    if masks.exposed.get(value.0).copied().unwrap_or(0) == 0 {
                        let before = *op;
                        match *op {
                            Op::Select(c, new, old) if masks.predicated.get(value.0).is_some_and(|p| p.is_some()) => *op = Op::Select(c, substitute(new, f, &mut fresh), old),
                            _ => *op = op.map(|v| substitute(v, f, &mut fresh)),
                        }
                        if *op != before { count += 1; }
                    }
                }
                Inst::Target { provenance: None, args, outputs, .. } => {
                    if outputs.iter().all(|(v, _)| masks.exposed.get(v.0).copied().unwrap_or(0) == 0) {
                        let before = *args;
                        *args = args.map(|v| substitute(v, f, &mut fresh));
                        if *args != before { count += 1; }
                    }
                }
                _ => {}
            }
            count += fresh.len();
            out.extend(fresh);
            out.push(inst.clone());
        }
        f.blocks.get_mut(&id).unwrap().insts = out;
    }
    count
}

fn resolve(v: ValueId, current: ValueId, masks: &Masks, narrowed: &[bool], _defs: &[Option<Op>], _f: &mut Func, _fresh: &mut Vec<Inst>, _depth: usize) -> Option<ValueId> {
    let (new, exec) = masks.predicated.get(v.0).copied().flatten()?;
    (!narrowed.get(v.0).copied().unwrap_or(false) && exec == current).then_some(new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::ir::{CachePolicy, EffectOp, MemSize, MemoryOp, MemorySemantics, Ordering, Scope, Space, Env, IntOp, IntPred, Ty};

    struct Builder { f: Func, insts: Vec<Inst> }
    impl Builder {
        fn core(&mut self, ty: Ty, op: Op) -> ValueId { let v = self.f.value(ty); self.insts.push(Inst::Core { value: v, ty, op }); v }
        fn ballot(&mut self, bit: ValueId) -> ValueId { let v = self.f.value(Ty::I32); self.insts.push(Inst::Packet { op: PacketOp::Ballot, input: bit, output: v }); v }
        fn store(&mut self, address: ValueId, data: ValueId, mask: ValueId) {
            let semantics = MemorySemantics { scope: Scope::WorkItem, ordering: Ordering::Relaxed, cache_policy: CachePolicy::Temporal, volatile: false, deferred_scope: false };
            self.insts.push(Inst::Effect { provenance: 0, op: EffectOp::Memory { space: Space::Lds, op: MemoryOp::Store(MemSize::B32), semantics }, inputs: vec![address, data, mask], outputs: vec![] });
        }
    }
    fn masks(f: &Func) -> std::rc::Rc<Masks> {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        Analyses::new(super::super::super::analysis::Context::new(&registry, &[], 0, 16)).get::<Masks>(f)
    }

    #[test]
    fn unobserved_predicated_writes_lose_their_select_and_exposed_ones_keep_it() {
        let mut b = Builder { f: Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] }, insts: vec![] };
        let exec = b.f.value(Ty::I1); let x = b.f.value(Ty::I32); let old = b.f.value(Ty::I32); let address = b.f.value(Ty::I32);
        let pred = b.core(Ty::I1, Op::Convert(Cvt::Bitcast, Ty::I1, exec));
        let stored = b.core(Ty::I32, Op::Select(pred, x, old));
        b.store(address, stored, pred);
        let observed = b.core(Ty::I32, Op::Select(pred, x, old));
        let zero = b.core(Ty::I32, Op::Const(Ty::I32, 0));
        let bit = b.core(Ty::I1, Op::Cmp(IntPred::Ne, observed, zero));
        let word = b.ballot(bit);
        b.store(address, word, pred);
        let Builder { mut f, insts } = b;
        f.blocks.insert(BlockId(0), Block { params: vec![(exec, Ty::I1), (x, Ty::I32), (old, Ty::I32), (address, Ty::I32)], insts, term: Term::Ret(vec![]) });
        let masks = masks(&f);
        let narrowed = run(&mut f, &masks);
        assert!(narrowed[stored.0]);
        assert!(!narrowed[observed.0]);
        assert!(f.blocks[&BlockId(0)].insts.iter().any(|inst| matches!(inst, Inst::Core { value, op: Op::Convert(Cvt::Bitcast, _, a), .. } if *value == stored && *a == x)));
        assert!(f.blocks[&BlockId(0)].insts.iter().any(|inst| matches!(inst, Inst::Core { value, op: Op::Select(..), .. } if *value == observed)));
    }

    #[test]
    fn masked_compare_words_do_not_expose_their_operands() {
        let mut b = Builder { f: Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] }, insts: vec![] };
        let exec = b.f.value(Ty::I1); let x = b.f.value(Ty::I32); let old = b.f.value(Ty::I32); let address = b.f.value(Ty::I32);
        let valid = b.core(Ty::I1, Op::Env(Env::ValidLane));
        let current = b.core(Ty::I1, Op::Int(IntOp::And, exec, valid));
        let written = b.core(Ty::I32, Op::Select(current, x, old));
        let limit = b.core(Ty::I32, Op::Const(Ty::I32, 7));
        let compare = b.core(Ty::I1, Op::Cmp(IntPred::Ult, written, limit));
        let masked = b.core(Ty::I1, Op::Int(IntOp::And, compare, current));
        let word = b.ballot(masked);
        b.store(address, word, current);
        let Builder { mut f, insts } = b;
        f.blocks.insert(BlockId(0), Block { params: vec![(exec, Ty::I1), (x, Ty::I32), (old, Ty::I32), (address, Ty::I32)], insts, term: Term::Ret(vec![]) });
        let masks = masks(&f);
        assert!(masks.masked[masked.0]);
        assert_eq!(masks.exposed[written.0], 0);
        let narrowed = run(&mut f, &masks);
        assert!(narrowed[written.0]);
    }
}
