use super::super::ir::{Cvt, IntOp, IntPred, Op, Ty, ValueId, *};
use super::{Analyses, Analysis};
use std::collections::VecDeque;

pub struct Constants;
impl Analysis for Constants {
    type Result = Vec<Option<u64>>;
    const NAME: &'static str = "constants";
    fn compute(f: &Func, _: &Analyses) -> Self::Result {
        constant_facts(f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fact {
    Pending,
    Constant(u64),
    Dynamic,
}
impl Fact {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Pending, a) | (a, Self::Pending) => a,
            (Self::Constant(a), Self::Constant(b)) if a == b => self,
            _ => Self::Dynamic,
        }
    }
}
enum Definition {
    External,
    Phi,
    Core(Ty, Op),
}

struct Lists {
    starts: Vec<u32>,
    items: Vec<u32>,
}
impl Lists {
    fn build(
        entries: usize,
        count: impl Fn(&mut dyn FnMut(usize)),
        fill: impl Fn(&mut dyn FnMut(usize, u32)),
    ) -> Self {
        let mut starts = vec![0u32; entries + 1];
        count(&mut |key| starts[key + 1] += 1);
        for i in 0..entries {
            starts[i + 1] += starts[i];
        }
        let mut cursor = starts.clone();
        let mut items = vec![0u32; starts[entries] as usize];
        fill(&mut |key, value| {
            items[cursor[key] as usize] = value;
            cursor[key] += 1;
        });
        Self { starts, items }
    }
    fn get(&self, key: usize) -> &[u32] {
        &self.items[self.starts[key] as usize..self.starts[key + 1] as usize]
    }
}

fn constant_facts(f: &Func) -> Vec<Option<u64>> {
    let mut definitions: Vec<_> = (0..f.types.len()).map(|_| Definition::External).collect();
    for (&id, block) in &f.blocks {
        if id != f.entry {
            for &(value, _) in &block.params {
                definitions[value.0] = Definition::Phi;
            }
        }
        for inst in &block.insts {
            if let Inst::Core { value, ty, op } = *inst {
                definitions[value.0] = Definition::Core(ty, op);
            }
        }
    }
    let arguments = {
        let each = |visit: &mut dyn FnMut(usize, u32)| {
            for block in f.blocks.values() {
                for edge in block.term.edges() {
                    let params = &f.blocks[&edge.dst].params;
                    for (&arg, &(value, _)) in edge.args.iter().zip(params) {
                        if matches!(definitions[value.0], Definition::Phi) {
                            visit(value.0, arg.0 as u32);
                        }
                    }
                }
            }
        };
        Lists::build(
            definitions.len(),
            |count| each(&mut |key, _| count(key)),
            |fill| each(fill),
        )
    };
    let users = {
        let each = |visit: &mut dyn FnMut(usize, u32)| {
            for (id, definition) in definitions.iter().enumerate() {
                match definition {
                    Definition::Core(_, op) => {
                        op.map(|value| {
                            visit(value.0, id as u32);
                            value
                        });
                    }
                    Definition::Phi => {
                        for &value in arguments.get(id) {
                            visit(value as usize, id as u32);
                        }
                    }
                    Definition::External => {}
                }
            }
        };
        Lists::build(
            definitions.len(),
            |count| each(&mut |key, _| count(key)),
            |fill| each(fill),
        )
    };
    let mut facts = vec![Fact::Pending; definitions.len()];
    let mut queue: VecDeque<_> = (0..definitions.len()).collect();
    let mut queued = vec![true; definitions.len()];
    while let Some(id) = queue.pop_front() {
        queued[id] = false;
        let next = match &definitions[id] {
            Definition::External => Fact::Dynamic,
            Definition::Phi => arguments
                .get(id)
                .iter()
                .fold(Fact::Pending, |a, &b| a.join(facts[b as usize])),
            Definition::Core(ty, op) => evaluate(*ty, *op, &f.types, &facts),
        };
        let next = facts[id].join(next);
        if next != facts[id] {
            facts[id] = next;
            for &user in users.get(id) {
                let user = user as usize;
                if !queued[user] {
                    queued[user] = true;
                    queue.push_back(user);
                }
            }
        }
    }
    facts
        .into_iter()
        .map(|fact| match fact {
            Fact::Constant(v) => Some(v),
            _ => None,
        })
        .collect()
}

fn mask(ty: Ty) -> u64 {
    u64::MAX >> (64 - ty.bits())
}
fn signed(value: u64, ty: Ty) -> i64 {
    ((value << (64 - ty.bits())) as i64) >> (64 - ty.bits())
}
fn evaluate(ty: Ty, op: Op, types: &[Ty], facts: &[Fact]) -> Fact {
    use Fact::*;
    if let Op::Const(_, value) = op {
        return Constant(value & mask(ty));
    }
    if let Op::Select(p, a, b) = op {
        return match facts[p.0] {
            Constant(0) => facts[b.0],
            Constant(_) => facts[a.0],
            _ if facts[a.0] == facts[b.0] => facts[a.0],
            Pending => Pending,
            _ => facts[a.0].join(facts[b.0]),
        };
    }
    let mut pending = false;
    let mut dynamic = false;
    op.map(|v| {
        pending |= facts[v.0] == Pending;
        dynamic |= facts[v.0] == Dynamic;
        v
    });
    if dynamic {
        return Dynamic;
    }
    if pending {
        return Pending;
    }
    let value = |v: ValueId| match facts[v.0] {
        Constant(x) => x,
        _ => unreachable!(),
    };
    let result = match op {
        Op::Int(kind, a, b) => {
            let (a, b) = (value(a), value(b));
            match kind {
                IntOp::Add => a.wrapping_add(b),
                IntOp::Sub => a.wrapping_sub(b),
                IntOp::Mul => a.wrapping_mul(b),
                IntOp::And => a & b,
                IntOp::Or => a | b,
                IntOp::Xor => a ^ b,

                IntOp::Shl | IntOp::LShr | IntOp::AShr if b >= ty.bits() as u64 => return Dynamic,
                IntOp::Shl => a << b,
                IntOp::LShr => a >> b,
                IntOp::AShr => (signed(a, ty) >> b) as u64,
            }
        }
        Op::Convert(Cvt::Bitcast | Cvt::ZExt | Cvt::Trunc, _, a) => value(a),
        Op::Convert(Cvt::SExt, _, a) => signed(value(a), types[a.0]) as u64,
        Op::Pack64(lo, hi) => value(lo) | (value(hi) << 32),
        Op::UnpackLo(a) => value(a) & 0xffff_ffff,
        Op::UnpackHi(a) => value(a) >> 32,
        Op::TrailingZeros(a) => (value(a).trailing_zeros()).min(ty.bits()) as u64,
        Op::LeadingZeros(a) => (value(a).leading_zeros() - (64 - ty.bits())) as u64,
        Op::PopulationCount(a) => value(a).count_ones() as u64,
        Op::ReverseBits(a) => value(a).reverse_bits() >> (64 - ty.bits()),
        Op::Cmp(kind, a, b) => {
            let t = types[a.0];
            let (a, b) = (value(a), value(b));
            (match kind {
                IntPred::Eq => a == b,
                IntPred::Ne => a != b,
                IntPred::Ult => a < b,
                IntPred::Ugt => a > b,
                IntPred::Ule => a <= b,
                IntPred::Uge => a >= b,
                IntPred::Slt => signed(a, t) < signed(b, t),
                IntPred::Sgt => signed(a, t) > signed(b, t),
                IntPred::Sle => signed(a, t) <= signed(b, t),
                IntPred::Sge => signed(a, t) >= signed(b, t),
            }) as u64
        }
        _ => return Dynamic,
    };
    Constant(result & mask(ty))
}
