//! Width-independent facts derived from SSA definitions and CFG edges.
//! Architectural register classes do not supply facts to this analysis.
pub(in crate::rdna_spmd) mod state;

use super::ir::typed::{cfg::*, Cvt, IntOp, IntPred, Op, Ty, ValueId};
use std::collections::{BTreeMap,VecDeque};

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
enum Definition { External, Phi(Vec<ValueId>), Core(Ty, Op) }

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
            for &(value, _) in &block.params { definitions[value.0] = Definition::Phi(vec![]); }
        }
        for inst in &block.insts {
            if let Inst::Core { value, ty, op } = *inst {
                let op=if valid_queries&&matches!(op,Op::Env(crate::rdna_spmd::ir::typed::Env::ValidLane)) {Op::Const(Ty::I1,1)} else {op};
                definitions[value.0] = Definition::Core(ty, op);
            } else if valid_queries {
                let query=match inst {
                    Inst::Packet {op:PacketOp::Any,input,output}=>Some((*input,*output)),
                    Inst::Effect {op:super::ir::typed::effect::EffectOp::Wave(super::ir::typed::effect::WaveOp::Any),inputs,outputs,..}=>Some((inputs[0],outputs[0].0)),
                    _=>None,
                };
                if let Some((input,output))=query {
                    definitions[output.0]=Definition::Core(Ty::I1,Op::Int(IntOp::Or,input,input));
                }
            }
        }
    }
    for block in f.blocks.values() {
        for edge in block.term.edges() {
            for (&arg, &(value, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                if let Definition::Phi(inputs) = &mut definitions[value.0] { inputs.push(arg); }
            }
        }
    }
    for &id in entry_true {definitions[id.0]=Definition::Core(Ty::I1,Op::Const(Ty::I1,1));}
    let mut users = vec![vec![]; definitions.len()];
    for (id, definition) in definitions.iter().enumerate() {
        let mut add = |value: ValueId| { users[value.0].push(id); value };
        match definition {
            Definition::Core(_, op) => { op.map(&mut add); }
            Definition::Phi(inputs) => { for &value in inputs { add(value); } }
            Definition::External => {}
        }
    }
    let mut facts = vec![Fact::Pending; definitions.len()];
    let mut queue: VecDeque<_> = (0..definitions.len()).collect();
    let mut queued = vec![true; definitions.len()];
    while let Some(id) = queue.pop_front() {
        queued[id] = false;
        let next = match &definitions[id] {
            Definition::External => Fact::Dynamic,
            Definition::Phi(inputs) => inputs.iter().fold(Fact::Pending, |a, b| a.join(facts[b.0])),
            Definition::Core(ty, op) => evaluate(*ty, *op, &f.types, &facts),
        };
        let next = facts[id].join(next);
        if next != facts[id] {
            facts[id] = next;
            for &user in &users[id] {
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
            Inst::Core { value: condition, ty: Ty::I1, op: Op::Env(crate::rdna_spmd::ir::typed::Env::ValidLane) },
        ], term: Term::CondBr { cond: condition,
            yes: Edge { dst: BlockId(1), args: vec![next,invariant] },
            no: Edge { dst: BlockId(2), args: vec![invariant] },
        } });
        f.blocks.insert(BlockId(2), Block { params: vec![(result,Ty::I32)], insts: vec![], term: Term::Ret });
        let facts = constants(f.verify().unwrap().func());
        assert_eq!(facts[a.0],None);
        assert_eq!(facts[next.0],None);
        assert_eq!(facts[invariant.0],Some(7));
        assert_eq!(facts[result.0],Some(7));
    }

    #[test]
    fn packet_divergence_does_not_make_distinct_uniform_inputs_a_uniform_phi() {
        let mut f=Func {entry:BlockId(0),blocks:BTreeMap::new(),types:vec![]};
        let cond=f.value(Ty::I1); let a=f.value(Ty::I32); let b=f.value(Ty::I32);
        let result=f.value(Ty::I32); let unchanged=f.value(Ty::I32);
        f.blocks.insert(BlockId(0),Block {params:vec![(cond,Ty::I1)],insts:vec![
            Inst::Core {value:a,ty:Ty::I32,op:Op::Const(Ty::I32,0)},
            Inst::Core {value:b,ty:Ty::I32,op:Op::Const(Ty::I32,1)},
        ],term:Term::CondBr {cond,yes:Edge {dst:BlockId(3),args:vec![a,a]},no:Edge {dst:BlockId(3),args:vec![b,a]}}});
        f.blocks.insert(BlockId(3),Block {params:vec![(result,Ty::I32),(unchanged,Ty::I32)],insts:vec![],term:Term::Ret});
        let verified=f.verify().unwrap();
        let facts=uniformity(verified.func(),&[]);
        assert!(!facts[result.0]);
        assert!(facts[unchanged.0]);
        assert!(uniformity(verified.func(),&[cond])[result.0]);
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

/// Wave uniformity from explicit entry bindings, SSA operands, and CFG edges.
/// Entry facts describe the invocation ABI; register numbers are not queried.
pub(crate) fn uniformity(f: &Func, uniform_entry: &[ValueId]) -> Vec<bool> {
    use super::ir::typed::effect::{EffectOp,WaveOp,MemoryOp};
    // The CFG is immutable during this analysis. Index its edges once rather
    // than scanning and allocating every terminator for every phi parameter.
    let successors:BTreeMap<_,_>=f.blocks.iter().map(|(&pc,b)|(pc,b.term.edges())).collect();
    let mut predecessors:BTreeMap<BlockId,Vec<&Edge>>=f.blocks.keys().map(|&pc|(pc,Vec::new())).collect();
    for edges in successors.values() {for &edge in edges {predecessors.get_mut(&edge.dst).unwrap().push(edge);}}
    let mut uniform=vec![true;f.types.len()];
    for &(id,_) in &f.blocks[&f.entry].params {uniform[id.0]=uniform_entry.contains(&id);}
    loop {
        let mut changed=false;
        // Packet-local branches can select different incoming constants in
        // different packets. Uniform operands alone do not prove a uniform phi.
        let mut divergent=std::collections::BTreeSet::new();
        let mut pending=Vec::new();
        for (&pc,block) in &f.blocks {
            if let Term::CondBr {cond,..}=&block.term {
                if !uniform[cond.0] {pending.extend(successors[&pc].iter().map(|e|e.dst));}
            }
        }
        while let Some(pc)=pending.pop() {
            if divergent.insert(pc) {pending.extend(successors[&pc].iter().map(|e|e.dst));}
        }
        for (&pc,block) in &f.blocks {
            if pc!=f.entry {
                for (index,&(id,_)) in block.params.iter().enumerate() {
                    let incoming=&predecessors[&pc];
                    let same=incoming.first().is_some_and(|first|incoming.iter().all(|e|e.args[index]==first.args[index]));
                    let proof=(!divergent.contains(&pc)||same)&&incoming.iter().all(|e|uniform[e.args[index].0]);
                    if uniform[id.0] && !proof {uniform[id.0]=false;changed=true;}
                }
            }
            for inst in &block.insts {
                let (outputs,proof) = match inst {
                    Inst::Packet {output,..}=>(vec![*output],false),
                    Inst::Core {value,op,..} => {
                        let mut proof=!matches!(op,Op::Env(_));
                        op.map(|v|{proof &= uniform[v.0];v});
                        (vec![*value],proof)
                    }
                    Inst::Target {args,outputs,..} => (outputs.iter().map(|v|v.0).collect(),args.values().iter().all(|v|uniform[v.0])),
                    Inst::Effect {op,inputs,outputs,..} => {
                        let proof=match op {
                            EffectOp::Wave(WaveOp::Any|WaveOp::Ballot|WaveOp::ReadFirstLane) => true,
                            EffectOp::Wave(WaveOp::ReadLane) => uniform[inputs[1].0],
                            EffectOp::Memory {op:MemoryOp::Load(_),..} => inputs.iter().all(|v|uniform[v.0]),
                            EffectOp::BarrierSignal {..} => true,
                            _ => false,
                        };
                        (outputs.iter().map(|v|v.0).collect(),proof)
                    }
                };
                if !proof { for id in outputs {if uniform[id.0] {uniform[id.0]=false;changed=true;}} }
            }
        }
        if !changed {return uniform;}
    }
}

/// Backward SSA reachability, including block arguments and loop backedges.
/// Non-query effects retain their inputs regardless of whether results escape.
pub(crate) fn live_values(f: &Func, roots: impl IntoIterator<Item=ValueId>) -> Vec<bool> {
    use super::ir::typed::effect::{EffectOp,WaveOp};
    let mut deps=vec![Vec::new();f.types.len()];
    let mut pending:Vec<_>=roots.into_iter().collect();
    for block in f.blocks.values() {
        for edge in block.term.edges() {
            for (&arg,&(param,_)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {deps[param.0].push(arg);}
        }
        if let super::ir::typed::cfg::Term::CondBr {cond,..}=block.term {pending.push(cond);}
        for inst in &block.insts {
            match inst {
                Inst::Packet {input,output,..}=>{deps[output.0].push(*input);},
                Inst::Core {value,op,..} => {op.map(|v|{deps[value.0].push(v);v});}
                Inst::Effect {op,inputs,outputs,..} => {
                    for &(id,_) in outputs {deps[id.0].extend(inputs);}
                    if !matches!(op,EffectOp::Wave(WaveOp::Any|WaveOp::Ballot)) {pending.extend(inputs);}
                }
                Inst::Target {provenance,args,outputs,..} => {
                    for &(id,_) in outputs {deps[id.0].extend(args.values());}
                    if provenance.is_some() {pending.extend(args.values());}
                }

            }
        }
    }
    let mut live=vec![false;f.types.len()];
    while let Some(id)=pending.pop() {
        if !live[id.0] {live[id.0]=true;pending.extend(&deps[id.0]);}
    }
    live
}

pub(in crate::rdna_spmd) mod rewrite;

pub(crate) struct Analysis<T> { revision: u64, value: T }
impl<T> Analysis<T> {
    pub fn get(&self, f: &super::lift::function::LiftedFunction) -> &T {
        assert_eq!(self.revision, f.revision, "analysis result predates an SSA edit");
        &self.value
    }
}
impl super::lift::function::LiftedFunction {
    pub(crate) fn analyze<T>(&self, compute: impl FnOnce(&Self) -> T) -> Analysis<T> {
        Analysis { revision: self.revision, value: compute(self) }
    }
}

pub(crate) fn retained(f: &Func, blocks: &BTreeMap<usize, super::lift::function::BlockPlan>) -> Vec<bool> {
    let mut retained=vec![false;f.types.len()];
    let mut local=vec![0usize;f.types.len()];
    let mut bound=vec![0usize;f.types.len()];
    let mut generation=0;
    for (&pc,plan) in blocks {for alu in plan.instructions.iter().flatten() {
        generation+=1;
        let insts=&f.blocks[&BlockId(pc)].insts[alu.core.clone()];
        for inst in insts {match inst {
            Inst::Core {value,..}|Inst::Packet {output:value,..}=>{local[value.0]=generation;},
            Inst::Target {outputs,..}|Inst::Effect {outputs,..}=>{for &(id,_) in outputs {local[id.0]=generation;}},
        }}
        for (_,id) in &alu.inputs {local[id.0]=generation;bound[id.0]=generation;}
        for &(id,_) in &alu.pairs {bound[id.0]=generation;}
        for inst in insts {if let Inst::Core {value,op,..}=inst {
            if bound[value.0]==generation {continue;}
            op.map(|id|{if local[id.0]!=generation {retained[id.0]=true;} id});
        }}
    }}
    retained
}

pub(crate) fn native_live(f: &Func, blocks: &BTreeMap<usize, super::lift::function::BlockPlan>, live: &[bool]) -> Vec<bool> {
    live_values(f, live.iter().enumerate().filter_map(|(id,&live)|live.then_some(ValueId(id)))
        .chain(blocks.values().flat_map(|block|block.instructions.iter().flatten())
            .flat_map(|alu|alu.outputs.iter().map(|&(_,id)|id))))
}
