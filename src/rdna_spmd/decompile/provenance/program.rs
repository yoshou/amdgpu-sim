use super::super::address::{condition, literals, Copies, LANES};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub(super) struct Program<'a> {
    pub(super) f: &'a Func,
    pub(super) facts: &'a Facts,
    pub(super) copies: &'a Copies,
    pub(super) inputs: &'a [Parameter],
    pub(super) entry: &'a EntryLayout,
    pub(super) registry: &'a DialectRegistry,
    pub(super) bindings: HashMap<u32, u64>,
    pub(super) folded: Vec<Option<u32>>,
    pub(super) rank: HashMap<BlockId, usize>,
    pub(super) incoming: Vec<Vec<(&'a [ValueId], bool)>>,
    pub(super) carried: Vec<(usize, usize, ValueId)>,
    workgroup: usize,
}

impl<'a> Program<'a> {
    pub(super) fn new(
        f: &'a Func,
        facts: &'a Facts,
        copies: &'a Copies,
        inputs: &'a [Parameter],
        entry: &'a EntryLayout,
        env: &Environment,
        headers: &BTreeSet<BlockId>,
        registry: &'a DialectRegistry,
    ) -> Self {
        let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let incoming: Vec<Vec<(&[ValueId], bool)>> = facts
            .order
            .iter()
            .map(|b| {
                facts.incoming[b]
                    .iter()
                    .map(|&(pred, slot)| {
                        let edge = f.blocks[&pred].term.edges().nth(slot).unwrap();
                        (&edge.args[..], headers.contains(b) && rank[&pred] >= rank[b])
                    })
                    .collect()
            })
            .collect();
        let mut carried = Vec::new();
        for (r, &block) in facts.order.iter().enumerate() {
            for (index, &(v, _)) in f.blocks[&block].params.iter().enumerate() {
                if incoming[r].iter().any(|&(args, back)| back && args[index] != v) {
                    carried.push((r, index, v));
                }
            }
        }
        let mut folded = vec![None; f.types.len()];
        for b in &facts.order {
            for inst in &f.blocks[b].insts {
                if let Inst::Core { value, op, .. } = inst {
                    folded[value.0] = match *op {
                        Op::Const(_, k) => Some(k as u32),
                        Op::Int(IntOp::Add, x, y) => folded[x.0].zip(folded[y.0]).map(|(x, y): (u32, u32)| x.wrapping_add(y)),
                        Op::Int(IntOp::Sub, x, y) => folded[x.0].zip(folded[y.0]).map(|(x, y): (u32, u32)| x.wrapping_sub(y)),
                        _ => None,
                    };
                }
            }
        }
        Self {
            f,
            facts,
            copies,
            inputs,
            entry,
            registry,
            bindings: env.bindings.iter().map(|b| (b.offset, b.allocation)).collect(),
            folded,
            rank,
            incoming,
            carried,
            workgroup: env.workgroup_size() as usize,
        }
    }

    pub(super) fn partial(&self) -> bool {
        self.workgroup % LANES != 0
    }

    pub(super) fn lacks(&self, lane: usize) -> bool {
        self.partial() && lane >= self.workgroup % LANES
    }

    pub(super) fn words(&self, address: ValueId, bytes: u32) -> Option<(u32, u32)> {
        self.folded[address.0].map(|t| (t / 4, (t as u64 + bytes as u64).div_ceil(4) as u32))
    }

    pub(super) fn constant(&self, x: ValueId) -> Option<u32> {
        match self.facts.op(self.f, x) {
            Some(Op::Const(_, k)) => Some(k as u32),
            _ => None,
        }
    }

    pub(super) fn offset(&self, mut x: ValueId) -> Option<u32> {
        let mut at = 0u32;
        loop {
            x = self.copies.get(&x).copied().unwrap_or(x);
            match self.facts.site[x.0] {
                Site::Param { block, index } if block == self.f.entry => {
                    return match self.inputs[index].source {
                        ParameterSource::Sgpr(n) if Some(n) == self.entry.kernarg_ptr => Some(at),
                        _ => None,
                    };
                }
                _ => match self.facts.op(self.f, x)? {
                    Op::Pack64(lo, _) => x = lo,
                    Op::Int(IntOp::Add, a, b) => match (self.folded[a.0], self.folded[b.0]) {
                        (None, Some(k)) => (x, at) = (a, at.wrapping_add(k)),
                        (Some(k), None) => (x, at) = (b, at.wrapping_add(k)),
                        _ => return None,
                    },
                    _ => return None,
                },
            }
        }
    }

    pub(super) fn conjuncts(&self, c: ValueId) -> Vec<ValueId> {
        let mut list = Vec::new();
        literals(&condition(self.f, self.facts, self.copies, c, true), &mut list);
        list.into_iter().filter(|l| l.1).map(|l| l.0).collect()
    }

    pub(super) fn pure(&self, op: TargetOp) -> bool {
        self.registry.operation(op).map_or(true, |spec| spec.effect == Effect::Pure)
    }
}
