//! Width-independent facts derived from SSA definitions and CFG edges.
//! Architectural register classes do not supply facts to this analysis.

use super::super::ir::{*, Cvt, IntOp, IntPred, Op, Ty, ValueId};
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fact { Pending, Constant(u64), Dynamic }
impl Fact {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Pending, a) | (a, Self::Pending) => a,
            (Self::Constant(a), Self::Constant(b)) if a == b => self,
            _ => Self::Dynamic,
        }
    }
}
enum Definition { External, Phi, Core(Ty, Op) }

struct Lists { starts: Vec<u32>, items: Vec<u32> }
impl Lists {
    fn build(entries: usize, count: impl Fn(&mut dyn FnMut(usize)), fill: impl Fn(&mut dyn FnMut(usize, u32))) -> Self {
        let mut starts = vec![0u32; entries + 1];
        count(&mut |key| starts[key + 1] += 1);
        for i in 0..entries { starts[i + 1] += starts[i]; }
        let mut cursor = starts.clone();
        let mut items = vec![0u32; starts[entries] as usize];
        fill(&mut |key, value| { items[cursor[key] as usize] = value; cursor[key] += 1; });
        Self { starts, items }
    }
    fn get(&self, key: usize) -> &[u32] { &self.items[self.starts[key] as usize..self.starts[key + 1] as usize] }
}

/// A conservative, monotone constant analysis. All incoming CFG edges take
/// part, including loop backedges; an unseeded cycle proves no constant.
pub(crate) fn constants(function: &Func) -> Vec<Option<u64>> {
    constant_facts(function,&[],false)
}
/// Facts valid on allocated work items. Use these only to fold queries which
/// explicitly ignore padding; they must not replace ordinary EXEC/memory masks.
pub(crate) fn valid_predicate_constants(f:&Func,entry_true:&[ValueId])->Vec<Option<u64>> {
    constant_facts(f,entry_true,true)
}
fn constant_facts(f:&Func,entry_true:&[ValueId],valid_queries:bool)->Vec<Option<u64>> {
    let mut definitions: Vec<_> = (0..f.types.len()).map(|_| Definition::External).collect();
    for (&id, block) in &f.blocks {
        if id != f.entry {
            for &(value, _) in &block.params { definitions[value.0] = Definition::Phi; }
        }
        for inst in &block.insts {
            if let Inst::Core { value, ty, op } = *inst {
                let op=if valid_queries&&matches!(op,Op::Env(crate::rdna_spmd::ir::Env::ValidLane)) {Op::Const(Ty::I1,1)} else {op};
                definitions[value.0] = Definition::Core(ty, op);
            } else if valid_queries {
                let query=match inst {
                    Inst::Packet {op:PacketOp::Any,input,output}=>Some((*input,*output)),
                    Inst::Effect {op:super::super::ir::EffectOp::Wave(super::super::ir::WaveOp::Any),inputs,outputs,..}=>Some((inputs[0],outputs[0].0)),
                    _=>None,
                };
                if let Some((input,output))=query {
                    definitions[output.0]=Definition::Core(Ty::I1,Op::Int(IntOp::Or,input,input));
                }
            }
        }
    }
    let arguments = {
        let each = |visit: &mut dyn FnMut(usize, u32)| {
            for block in f.blocks.values() {
                for edge in block.term.edges() {
                    let params = &f.blocks[&edge.dst].params;
                    for (&arg, &(value, _)) in edge.args.iter().zip(params) {
                        if matches!(definitions[value.0], Definition::Phi) { visit(value.0, arg.0 as u32); }
                    }
                }
            }
        };
        Lists::build(definitions.len(), |count| each(&mut |key, _| count(key)), |fill| each(fill))
    };
    for &id in entry_true {definitions[id.0]=Definition::Core(Ty::I1,Op::Const(Ty::I1,1));}
    let users = {
        let each = |visit: &mut dyn FnMut(usize, u32)| {
            for (id, definition) in definitions.iter().enumerate() {
                match definition {
                    Definition::Core(_, op) => { op.map(|value| { visit(value.0, id as u32); value }); }
                    Definition::Phi => { for &value in arguments.get(id) { visit(value as usize, id as u32); } }
                    Definition::External => {}
                }
            }
        };
        Lists::build(definitions.len(), |count| each(&mut |key, _| count(key)), |fill| each(fill))
    };
    let mut facts = vec![Fact::Pending; definitions.len()];
    let mut queue: VecDeque<_> = (0..definitions.len()).collect();
    let mut queued = vec![true; definitions.len()];
    while let Some(id) = queue.pop_front() {
        queued[id] = false;
        let next = match &definitions[id] {
            Definition::External => Fact::Dynamic,
            Definition::Phi => arguments.get(id).iter().fold(Fact::Pending, |a, &b| a.join(facts[b as usize])),
            Definition::Core(ty, op) => evaluate(*ty, *op, &f.types, &facts),
        };
        let next = facts[id].join(next);
        if next != facts[id] {
            facts[id] = next;
            for &user in users.get(id) {
                let user = user as usize;
                if !queued[user] { queued[user] = true; queue.push_back(user); }
            }
        }
    }
    facts.into_iter().map(|fact| match fact { Fact::Constant(v) => Some(v), _ => None }).collect()
}

fn mask(ty: Ty) -> u64 { u64::MAX >> (64 - ty.bits()) }
fn signed(value: u64, ty: Ty) -> i64 { ((value << (64 - ty.bits())) as i64) >> (64 - ty.bits()) }
fn evaluate(ty: Ty, op: Op, types: &[Ty], facts: &[Fact]) -> Fact {
    use Fact::*;
    if let Op::Const(_, value) = op { return Constant(value & mask(ty)); }
    if let Op::Select(p, a, b) = op {
        return match facts[p.0] {
            Constant(0) => facts[b.0], Constant(_) => facts[a.0],
            _ if facts[a.0] == facts[b.0] => facts[a.0],
            Pending => Pending,
            _ => facts[a.0].join(facts[b.0]),
        };
    }
    let mut pending = false; let mut dynamic = false;
    op.map(|v| { pending |= facts[v.0] == Pending; dynamic |= facts[v.0] == Dynamic; v });
    if dynamic { return Dynamic; }
    if pending { return Pending; }
    let value = |v: ValueId| match facts[v.0] { Constant(x) => x, _ => unreachable!() };
    let result = match op {
        Op::Int(kind, a, b) => {
            let (a, b) = (value(a), value(b));
            match kind {
                IntOp::Add => a.wrapping_add(b), IntOp::Sub => a.wrapping_sub(b), IntOp::Mul => a.wrapping_mul(b),
                IntOp::And => a & b, IntOp::Or => a | b, IntOp::Xor => a ^ b,
                // An oversized shift is not evidence of a defined constant.
                IntOp::Shl | IntOp::LShr | IntOp::AShr if b >= ty.bits() as u64 => return Dynamic,
                IntOp::Shl => a << b, IntOp::LShr => a >> b, IntOp::AShr => (signed(a, ty) >> b) as u64,
            }
        }
        Op::Convert(Cvt::Bitcast | Cvt::ZExt | Cvt::Trunc, _, a) => value(a),
        Op::Convert(Cvt::SExt, _, a) => signed(value(a), types[a.0]) as u64,
        Op::Pack64(lo, hi) => value(lo) | (value(hi) << 32),
        Op::UnpackLo(a) => value(a) & 0xffff_ffff, Op::UnpackHi(a) => value(a) >> 32,
        Op::TrailingZeros(a) => (value(a).trailing_zeros()).min(ty.bits()) as u64,
        Op::LeadingZeros(a) => (value(a).leading_zeros() - (64 - ty.bits())) as u64,
        Op::PopulationCount(a) => value(a).count_ones() as u64,
        Op::ReverseBits(a) => value(a).reverse_bits() >> (64 - ty.bits()),
        Op::Cmp(kind, a, b) => {
            let t = types[a.0]; let (a, b) = (value(a), value(b));
            (match kind {
                IntPred::Eq => a == b, IntPred::Ne => a != b,
                IntPred::Ult => a < b, IntPred::Ugt => a > b, IntPred::Ule => a <= b, IntPred::Uge => a >= b,
                IntPred::Slt => signed(a,t) < signed(b,t), IntPred::Sgt => signed(a,t) > signed(b,t),
                IntPred::Sle => signed(a,t) <= signed(b,t), IntPred::Sge => signed(a,t) >= signed(b,t),
            }) as u64
        }
        _ => return Dynamic,
    };
    Constant(result & mask(ty))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn loop_backedge_can_disprove_a_constant_without_losing_invariants() {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let initial = f.value(Ty::I32); let a = f.value(Ty::I32); let invariant = f.value(Ty::I32);
        let one = f.value(Ty::I32); let next = f.value(Ty::I32); let condition = f.value(Ty::I1);
        let result = f.value(Ty::I32);
        f.blocks.insert(BlockId(0), Block { params: vec![], insts: vec![Inst::Core {
            value: initial, ty: Ty::I32, op: Op::Const(Ty::I32,7),
        }], term: Term::Br(Edge { dst: BlockId(1), args: vec![initial, initial] }) });
        f.blocks.insert(BlockId(1), Block { params: vec![(a,Ty::I32),(invariant,Ty::I32)], insts: vec![
            Inst::Core { value: one, ty: Ty::I32, op: Op::Const(Ty::I32,1) },
            Inst::Core { value: next, ty: Ty::I32, op: Op::Int(IntOp::Add,a,one) },
            Inst::Core { value: condition, ty: Ty::I1, op: Op::Env(crate::rdna_spmd::ir::Env::ValidLane) },
        ], term: Term::CondBr { cond: condition,
            yes: Edge { dst: BlockId(1), args: vec![next,invariant] },
            no: Edge { dst: BlockId(2), args: vec![invariant] },
        } });
        f.blocks.insert(BlockId(2), Block { params: vec![(result,Ty::I32)], insts: vec![], term: Term::Ret(vec![]) });
        let facts = constants(f.verify().unwrap().func());
        assert_eq!(facts[a.0],None);
        assert_eq!(facts[next.0],None);
        assert_eq!(facts[invariant.0],Some(7));
        assert_eq!(facts[result.0],Some(7));
    }

    #[test]
    fn constant_words_keep_width_and_do_not_fold_undefined_shifts() {
        let facts = [Fact::Constant(0x8000_0001),Fact::Constant(32),Fact::Constant(1)];
        let types = [Ty::I32;3]; let a = ValueId(0); let b = ValueId(1); let one = ValueId(2);
        assert!(evaluate(Ty::I32,Op::Int(IntOp::Shl,a,b),&types,&facts)==Fact::Dynamic);
        assert!(evaluate(Ty::I32,Op::Int(IntOp::AShr,a,one),&types,&facts)==Fact::Constant(0xc000_0000));
        assert!(evaluate(Ty::I64,Op::Convert(Cvt::SExt,Ty::I64,a),&types,&facts)==Fact::Constant(0xffff_ffff_8000_0001));
        assert!(evaluate(Ty::I32,Op::Int(IntOp::Add,a,a),&types,&facts)==Fact::Constant(2));
    }
}
