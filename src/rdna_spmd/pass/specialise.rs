use super::super::analysis::{Analyses, Masks};
use super::super::ir::{*, EffectOp, Env, IntOp, Op, Ty, ValueId};
use std::collections::BTreeMap;

pub(crate) struct Specialise;
impl super::Pass for Specialise {
    fn name(&self) -> &str { "specialise" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let exec_index = analyses.context().exec_index;
        let masks = analyses.get::<Masks>(f);
        let blocks = candidates(f, &masks, exec_index);
        !blocks.is_empty() && run(f, &blocks, exec_index) > 0
    }
}

pub(crate) fn candidates(f: &Func, masks: &Masks, exec_index: usize) -> Vec<BlockId> {
    let mut out = Vec::new();
    for (&id, block) in &f.blocks {
        if id == f.entry { continue; }
        let entry = &f.blocks[&f.entry];
        let k = entry.params[..exec_index].iter().filter(|p| p.1 == Ty::I1).count();
        let Some(&(exec, _)) = block.params.iter().filter(|p| p.1 == Ty::I1).nth(k) else { continue; };
        if masks.full[exec.0] { continue; }
        if block.insts.iter().any(|inst| matches!(inst, Inst::Effect { provenance, op, .. }
            if *provenance & crate::rdna_spmd::ir::SCHEDULED != 0 || matches!(op, EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait))) { continue; }
        let wanted = block.insts.iter().filter(|inst| match inst {
            Inst::Core { value, op: Op::Select(..), .. } => masks.predicated[value.0].is_some_and(|(_, e)| e == exec),
            _ => false,
        }).count();
        if wanted != 0 { out.push(id); }
    }
    out
}

pub(crate) fn run(f: &mut Func, blocks: &[BlockId], exec_index: usize) -> usize {
    let mut next_id = f.blocks.keys().map(|b| b.0).max().unwrap_or(0) + 1;
    let entry = f.entry;
    let mut count = 0;
    for &id in blocks {
        let original = f.blocks[&id].clone();
        let masked = BlockId(next_id); next_id += 1;
        let all = BlockId(next_id); next_id += 1;
        let mut map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
        let fresh = |f: &mut Func, v: ValueId, map: &mut BTreeMap<ValueId, ValueId>| { let n = f.value(f.types[v.0]); map.insert(v, n); n };
        let mut clone = Block { params: Vec::new(), insts: Vec::new(), term: original.term.clone() };
        for &(p, ty) in &original.params { let n = fresh(f, p, &mut map); clone.params.push((n, ty)); }
        for inst in &original.insts {
            let cloned = match inst {
                Inst::Core { value, ty, op } => { let n = fresh(f, *value, &mut map); Inst::Core { value: n, ty: *ty, op: *op } }
                Inst::Packet { op, input, output } => { let n = fresh(f, *output, &mut map); Inst::Packet { op: *op, input: *input, output: n } }
                Inst::Target { provenance, op, args, outputs } => {
                    let outputs = outputs.iter().map(|&(v, ty)| (fresh(f, v, &mut map), ty)).collect();
                    Inst::Target { provenance: provenance.map(|p| p + (1 << 40)), op: *op, args: *args, outputs }
                }
                Inst::Effect { provenance, op, inputs, outputs } => {
                    let outputs = outputs.iter().map(|&(v, ty)| (fresh(f, v, &mut map), ty)).collect();
                    Inst::Effect { provenance: provenance + (1 << 40), op: *op, inputs: inputs.clone(), outputs }
                }
            };
            clone.insts.push(cloned);
        }
        let m = |v: ValueId| map.get(&v).copied().unwrap_or(v);
        for inst in &mut clone.insts {
            match inst {
                Inst::Core { op, .. } => *op = op.map(m),
                Inst::Packet { input, .. } => *input = m(*input),
                Inst::Target { args, .. } => *args = args.map(m),
                Inst::Effect { inputs, .. } => for v in inputs { *v = m(*v); },
            }
        }
        match &mut clone.term {
            Term::Br(e) => for v in &mut e.args { *v = m(*v); },
            Term::CondBr { cond, yes, no } => { *cond = m(*cond); for v in yes.args.iter_mut().chain(&mut no.args) { *v = m(*v); } }
            Term::Ret(args) => for v in args { *v = m(*v); },
        }
        let mut guard = Block { params: Vec::new(), insts: Vec::new(), term: Term::Ret(vec![]) };
        let mut guard_map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
        for &(p, ty) in &original.params { let n = f.value(ty); guard_map.insert(p, n); guard.params.push((n, ty)); }
        let exec_original = original_exec(&original, f, entry, exec_index);
        let exec = guard_map[&exec_original];
        let one = f.value(Ty::I1);
        let flipped = f.value(Ty::I1);
        let valid = f.value(Ty::I1);
        let inactive = f.value(Ty::I1);
        let any = f.value(Ty::I1);
        guard.insts.push(Inst::Core { value: one, ty: Ty::I1, op: Op::Const(Ty::I1, 1) });
        guard.insts.push(Inst::Core { value: flipped, ty: Ty::I1, op: Op::Int(IntOp::Xor, exec, one) });
        guard.insts.push(Inst::Core { value: valid, ty: Ty::I1, op: Op::Env(Env::ValidLane) });
        guard.insts.push(Inst::Core { value: inactive, ty: Ty::I1, op: Op::Int(IntOp::And, flipped, valid) });
        guard.insts.push(Inst::Packet { op: PacketOp::Any, input: inactive, output: any });
        let args: Vec<ValueId> = guard.params.iter().map(|p| p.0).collect();
        guard.term = Term::CondBr { cond: any, yes: Edge { dst: masked, args: args.clone() }, no: Edge { dst: all, args } };
        f.blocks.insert(masked, original);
        f.blocks.insert(all, clone);
        f.blocks.insert(id, guard);
        count += 1;
    }
    let _ = entry;
    count
}

fn original_exec(block: &Block, f: &Func, entry: BlockId, exec_index: usize) -> ValueId {
    let entry_block = &f.blocks[&entry];
    let k = entry_block.params[..exec_index].iter().filter(|p| p.1 == Ty::I1).count();
    block.params.iter().filter(|p| p.1 == Ty::I1).nth(k).expect("block lacks its EXEC parameter").0
}
