use super::super::hazard::{Hazards, Known};
use super::super::logic::lane_test;
use crate::rdna_spmd::analysis::facts::{Facts, Site, Use};
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

pub(super) const LOWER: u8 = 1;
pub(super) const UPPER: u8 = 2;

pub(super) struct Program<'a> {
    pub(super) f: &'a Func,
    pub(super) facts: &'a Facts,
    pub(super) inputs: &'a [Parameter],
    pub(super) exec_index: Option<usize>,
    pub(super) loops: &'a Loops,
    pub(super) hazards: &'a Hazards,
    pub(super) known: Option<&'a Known>,
    pub(super) meetings: BTreeMap<(BlockId, usize), usize>,
    pub(super) rank: BTreeMap<BlockId, usize>,
    pub(super) access: BTreeMap<(BlockId, usize), usize>,
    pub(super) halves: Vec<u8>,
}

impl<'a> Program<'a> {
    pub(super) fn new(
        f: &'a Func,
        facts: &'a Facts,
        inputs: &'a [Parameter],
        exec_index: Option<usize>,
        loops: &'a Loops,
        hazards: &'a Hazards,
        known: Option<&'a Known>,
    ) -> Self {
        Self {
            f,
            facts,
            inputs,
            exec_index,
            loops,
            hazards,
            known,
            meetings: hazards
                .meetings
                .iter()
                .enumerate()
                .map(|(i, &position)| (position, i))
                .collect(),
            rank: facts
                .order
                .iter()
                .enumerate()
                .map(|(r, &b)| (b, r))
                .collect(),
            access: hazards
                .accesses
                .iter()
                .enumerate()
                .map(|(i, a)| ((a.block, a.index), i))
                .collect(),
            halves: halves(f, facts),
        }
    }

    pub(super) fn partnered(&self, at: (BlockId, usize)) -> bool {
        let hazards = self.hazards;
        hazards
            .accesses
            .iter()
            .position(|a| (a.block, a.index) == at)
            .is_some_and(|i| hazards.together.iter().chain(&hazards.apart).any(|&(p, q)| p == i || q == i))
    }
}

/// The halves of a wave of 64 lanes, `LOWER` and `UPPER`, in whose lanes some use reads the difference
/// of each lane word. A word op, a select and a block parameter read their words where their own
/// differences are read, a pack reads its low word in the lower half and its high word in the upper
/// half, and an unpack reads its word in its own half. A lane shift, a truncation and a test against
/// zero of a word that is not materialized read every lane. Any other use reads the whole difference of
/// a materialized word instead, and the lane program returns nothing. Every value but a lane word, and
/// every value of a narrower wave, counts as read in every lane.
fn halves(f: &Func, facts: &Facts) -> Vec<u8> {
    let n = f.types.len();
    if f.lanes != 64 {
        return vec![LOWER | UPPER; n];
    }
    let mut halves: Vec<u8> = (0..n).map(|v| if facts.lane_word[v] { 0 } else { LOWER | UPPER }).collect();
    let mut pending: Vec<ValueId> = (0..n).filter(|&v| facts.lane_word[v]).map(ValueId).collect();
    while let Some(v) = pending.pop() {
        let read = facts.uses[v.0].iter().fold(halves[v.0], |read, &u| read | reads(f, facts, &halves, v, u));
        if read == halves[v.0] {
            continue;
        }
        halves[v.0] = read;
        let sources: Vec<ValueId> = match facts.site[v.0] {
            Site::Param { block, index } if block != f.entry => facts.arguments(f, block, index).collect(),
            Site::Inst { block, index } => f.blocks[&block].insts[index].operands(),
            _ => Vec::new(),
        };
        pending.extend(sources.into_iter().filter(|s| facts.lane_word[s.0]));
    }
    halves
}

fn reads(f: &Func, facts: &Facts, halves: &[u8], v: ValueId, u: Use) -> u8 {
    match u {
        Use::Arg { block, edge, index } => {
            let dst = f.blocks[&block].term.edges().nth(edge).unwrap().dst;
            halves[f.blocks[&dst].params[index].0 .0]
        }
        Use::Inst { block, index } => match f.blocks[&block].insts[index] {
            Inst::Core { value, op, .. } if facts.lane_word[value.0] => {
                let read = halves[value.0];
                match op {
                    Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, ..)
                    | Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, _) => read,
                    Op::Select(c, ..) if c != v => read,
                    Op::Pack64(low, high) => {
                        (if low == v { read & LOWER } else { 0 }) | (if high == v { read & UPPER } else { 0 })
                    }
                    Op::UnpackLo(_) => read & LOWER,
                    Op::UnpackHi(_) => read & UPPER,
                    _ => LOWER | UPPER,
                }
            }
            Inst::Core { op: Op::Int(IntOp::LShr, w, s), .. } if w == v && facts.is_lane_shift(f, w, s) => {
                LOWER | UPPER
            }
            Inst::Core { op: Op::Convert(Cvt::Trunc, Ty::I1, _), .. } => LOWER | UPPER,
            Inst::Core { op: Op::Cmp(IntPred::Eq | IntPred::Ne, a, b), .. }
                if lane_test(f, facts, a, b) == Some(v) && !facts.materialized[v.0] =>
            {
                LOWER | UPPER
            }
            Inst::Core { op: Op::Select(c, ..), .. } if c == v => LOWER | UPPER,
            _ => 0,
        },
        Use::Cond(_) => LOWER | UPPER,
        Use::Ret(_) => 0,
    }
}
