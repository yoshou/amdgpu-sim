mod demand;
mod exposure;
mod layout;
mod program;
mod sets;

use super::address::{Copies, Region, Regions};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use layout::{merge, points, Layout};
use program::Program;
use sets::Sets;
use std::collections::BTreeSet;

pub(super) struct Bounds {
    pub(super) known: Vec<Option<Option<Region>>>,
    pub(super) allocating: Vec<bool>,
    pub(super) plain: Vec<bool>,
    pub(super) carried: HashMap<ValueId, Vec<Regions>>,
    pub(super) loaded: HashMap<ValueId, Regions>,
    pub(super) spills: Vec<Spill>,
    pub(super) exposing: Vec<Exposure>,
    pub(super) written: HashMap<(BlockId, usize), Regions>,
}

pub(super) struct Spill {
    pub(super) at: (BlockId, usize),
    pub(super) words: Option<(u32, u32)>,
    pub(super) data: Vec<ValueId>,
    pub(super) mask: ValueId,
    pub(super) parts: Vec<Regions>,
}

impl Spill {
    pub(super) fn part(&self, lane: usize) -> &Regions {
        &self.parts[if self.parts.len() == 1 { 0 } else { lane }]
    }
}

#[derive(Clone)]
pub(super) struct Exposure {
    pub(super) operands: Vec<ValueId>,
    pub(super) assume: Option<ValueId>,
    pub(super) candidates: Vec<u64>,
}

pub(super) fn bounds(
    f: &Func,
    facts: &Facts,
    copies: &Copies,
    users: &HashMap<ValueId, Vec<ValueId>>,
    inputs: &[Parameter],
    entry: &EntryLayout,
    env: &Environment,
    headers: &BTreeSet<BlockId>,
    registry: &DialectRegistry,
) -> Bounds {
    let program = Program::new(f, facts, copies, inputs, entry, env, headers, registry);
    let layout = Layout::new(env);
    let sets = Sets::solve(&program, &layout, users);
    let mut bounds = HashMap::default();
    let plain = carried(&sets, &mut bounds);
    let known = known(&sets);
    let allocating = (0..f.types.len()).map(|x| layout.allocates(sets.of(ValueId(x)))).collect();
    let (exposing, spills) = exposure::exposures(&sets, env);
    let loaded = loaded(&sets, &spills, &mut bounds);
    let written = written(&sets);
    Bounds {
        known,
        allocating,
        plain,
        carried: bounds,
        loaded,
        spills,
        exposing,
        written,
    }
}

fn carried(sets: &Sets, bounds: &mut HashMap<ValueId, Vec<Regions>>) -> Vec<bool> {
    let program = sets.program();
    let mut plain = vec![false; program.f.types.len()];
    let mut acc = vec![0; sets.words()];
    for &(r, index, v) in &program.carried {
        if program.copies.contains_key(&v) {
            continue;
        }
        let back: Vec<ValueId> = program.incoming[r]
            .iter()
            .filter(|&&(args, back)| back && args[index] != v)
            .map(|&(args, _)| args[index])
            .collect();
        acc.fill(0);
        for &a in &back {
            merge(&mut acc, sets.of(a));
        }
        if !points(&acc) {
            plain[v.0] = acc[0] & 1 != 0;
            continue;
        }
        let lanes = if back.iter().any(|&a| sets.has_parts(a)) { program.lanes() } else { 1 };
        let mut parts = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            acc.fill(0);
            for &a in &back {
                merge(&mut acc, sets.part(a, lane));
            }
            parts.push(sets.layout().regions(&acc));
        }
        bounds.insert(v, parts);
    }
    plain
}

fn known(sets: &Sets) -> Vec<Option<Option<Region>>> {
    (0..sets.program().f.types.len())
        .map(|x| {
            let set = sets.of(ValueId(x));
            (set.iter().map(|w| w.count_ones()).sum::<u32>() == 1).then(|| {
                let (word, bits) = set.iter().enumerate().find(|&(_, &w)| w != 0).unwrap();
                sets.layout().region(word * 64 + bits.trailing_zeros() as usize)
            })
        })
        .collect()
}

fn loaded(sets: &Sets, spills: &[Spill], bounds: &mut HashMap<ValueId, Vec<Regions>>) -> HashMap<ValueId, Regions> {
    let program = sets.program();
    let (f, facts) = (program.f, program.facts);
    let mut loaded = HashMap::default();
    for b in &facts.order {
        for inst in &f.blocks[b].insts {
            let Inst::Effect {
                op: EffectOp::Memory {
                    op: MemoryOp::Load(size),
                    space,
                    ..
                },
                inputs,
                outputs,
                ..
            } = inst
            else {
                continue;
            };
            let v = outputs[0].0;
            if *space != Space::Scratch {
                if points(sets.of(v)) {
                    let mut set = sets.layout().regions(sets.of(v));
                    set.add(None);
                    loaded.insert(v, set);
                }
                continue;
            }
            let words = program.words(inputs[0], size.bytes());
            let from: Vec<&Spill> = spills
                .iter()
                .filter(|sp| match (words, sp.words) {
                    (Some((a, b)), Some((c, d))) => a < d && c < b,
                    _ => true,
                })
                .collect();
            let lanes = if from.iter().any(|sp| sp.parts.len() > 1) { program.lanes() } else { 1 };
            let mut parts = Vec::with_capacity(lanes);
            let mut pointing = false;
            for lane in 0..lanes {
                let mut set = Regions::one(None);
                for sp in &from {
                    set.union(sp.part(lane));
                }
                pointing |= set != Regions::one(None);
                parts.push(set);
            }
            if pointing {
                bounds.insert(v, parts);
            }
        }
    }
    loaded
}

fn written(sets: &Sets) -> HashMap<(BlockId, usize), Regions> {
    let program = sets.program();
    let (f, facts) = (program.f, program.facts);
    let mut written = HashMap::default();
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            if let Inst::Effect {
                op:
                    EffectOp::Memory {
                        op,
                        space: Space::Global,
                        ..
                    },
                inputs,
                ..
            } = inst
            {
                if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) {
                    written.insert((b, index), sets.layout().regions(sets.of(inputs[0])));
                }
            }
        }
    }
    written
}
