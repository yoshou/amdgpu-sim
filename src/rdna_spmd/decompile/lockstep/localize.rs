//! From the dominance form the lowering builds in to the block-local form the
//! IR requires.
//!
//! The lowering lets a block read any value a dominating block defined. Two
//! steps take it from there: a parameter every edge passes the same value
//! (or itself) is that value, and every value a block reads from elsewhere
//! becomes a parameter of that block, passed along every edge into it.
//! Constants are defined again where they are read instead, so an operand a
//! target operation requires to be constant stays one.

use super::super::facts::{operands, outputs};
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

/// Folds every parameter that only forwards one value, returning what each
/// folded parameter became.
pub(super) fn fold_forwarding(f: &mut Func) -> BTreeMap<ValueId, ValueId> {
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let find = |renames: &BTreeMap<ValueId, ValueId>, mut v: ValueId| {
        while let Some(&next) = renames.get(&v) {
            v = next;
        }
        v
    };
    loop {
        let mut incoming: BTreeMap<BlockId, Vec<Vec<ValueId>>> = BTreeMap::new();
        for block in f.blocks.values() {
            for edge in block.term.edges() {
                incoming
                    .entry(edge.dst)
                    .or_default()
                    .push(edge.args.clone());
            }
        }
        let mut folded: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
        for (&id, block) in &f.blocks {
            if id == f.entry {
                continue;
            }
            let Some(edges) = incoming.get(&id) else {
                continue;
            };
            for (index, &(param, _)) in block.params.iter().enumerate() {
                let mut unique = None;
                let mut forwards = true;
                for args in edges {
                    let arg = find(&renames, args[index]);
                    if arg == param {
                        continue;
                    }
                    match unique {
                        None => unique = Some(arg),
                        Some(u) if u == arg => {}
                        Some(_) => forwards = false,
                    }
                }
                if let (true, Some(value)) = (forwards, unique) {
                    renames.insert(param, value);
                    folded.entry(id).or_default().push(index);
                }
            }
        }
        if folded.is_empty() {
            break;
        }
        for (id, indices) in &folded {
            let block = f.blocks.get_mut(id).unwrap();
            for &index in indices.iter().rev() {
                block.params.remove(index);
            }
        }
        for block in f.blocks.values_mut() {
            for edge in block.term.edges_mut() {
                if let Some(indices) = folded.get(&edge.dst) {
                    for &index in indices.iter().rev() {
                        edge.args.remove(index);
                    }
                }
            }
        }
    }
    let resolved: BTreeMap<ValueId, ValueId> =
        renames.keys().map(|&v| (v, find(&renames, v))).collect();
    rename(f, &resolved);
    resolved
}

fn rename(f: &mut Func, map: &BTreeMap<ValueId, ValueId>) {
    if map.is_empty() {
        return;
    }
    let m = |v: ValueId| map.get(&v).copied().unwrap_or(v);
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            rename_operands(inst, &m);
        }
        rename_term(&mut block.term, &m);
    }
}

fn rename_operands(inst: &mut Inst, m: &dyn Fn(ValueId) -> ValueId) {
    match inst {
        Inst::Core { op, .. } => *op = op.map(m),
        Inst::Packet { input, .. } => *input = m(*input),
        Inst::Target { args, .. } => *args = args.map(m),
        Inst::Effect { inputs, .. } => {
            for v in inputs {
                *v = m(*v);
            }
        }
    }
}

fn rename_term(term: &mut Term, m: &dyn Fn(ValueId) -> ValueId) {
    match term {
        Term::CondBr { cond, .. } => *cond = m(*cond),
        Term::Ret(args) => {
            for v in args {
                *v = m(*v);
            }
        }
        Term::Br(_) => {}
    }
    for edge in term.edges_mut() {
        for v in &mut edge.args {
            *v = m(*v);
        }
    }
}

/// Passes every value a block reads from another block in as a parameter.
pub(super) fn localize(f: &mut Func) {
    let mut home: Vec<Option<BlockId>> = vec![None; f.types.len()];
    let mut constant: Vec<Option<(Ty, u64)>> = vec![None; f.types.len()];
    let mut defs: BTreeMap<BlockId, BTreeSet<ValueId>> = BTreeMap::new();
    let mut reads: BTreeMap<BlockId, BTreeSet<ValueId>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        let mut local: BTreeSet<ValueId> = block.params.iter().map(|p| p.0).collect();
        let mut read = BTreeSet::new();
        for inst in &block.insts {
            for v in operands(inst) {
                if !local.contains(&v) {
                    read.insert(v);
                }
            }
            for v in outputs(inst) {
                local.insert(v);
            }
            if let Inst::Core {
                value,
                ty,
                op: Op::Const(_, bits),
            } = inst
            {
                constant[value.0] = Some((*ty, *bits));
            }
        }
        let mut term_reads: Vec<ValueId> = Vec::new();
        match &block.term {
            Term::CondBr { cond, .. } => term_reads.push(*cond),
            Term::Ret(args) => term_reads.extend(args.iter().copied()),
            Term::Br(_) => {}
        }
        for edge in block.term.edges() {
            term_reads.extend(edge.args.iter().copied());
        }
        for v in term_reads {
            if !local.contains(&v) {
                read.insert(v);
            }
        }
        for &v in &local {
            home[v.0] = Some(id);
        }
        defs.insert(id, local);
        reads.insert(id, read);
    }
    let mut live: BTreeMap<BlockId, BTreeSet<ValueId>> = reads
        .iter()
        .map(|(&id, read)| {
            (
                id,
                read.iter()
                    .copied()
                    .filter(|v| constant[v.0].is_none())
                    .collect(),
            )
        })
        .collect();
    let successors: BTreeMap<BlockId, Vec<BlockId>> = f
        .blocks
        .iter()
        .map(|(&id, block)| (id, block.term.edges().map(|e| e.dst).collect()))
        .collect();
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &id in ids.iter().rev() {
            let mut add: Vec<ValueId> = Vec::new();
            for dst in &successors[&id] {
                for &v in &live[dst] {
                    if !defs[&id].contains(&v) && !live[&id].contains(&v) {
                        add.push(v);
                    }
                }
            }
            if !add.is_empty() {
                live.get_mut(&id).unwrap().extend(add);
                changed = true;
            }
        }
    }
    assert!(
        live[&f.entry].is_empty(),
        "the packet program reads a value no block defines"
    );
    let mut param_of: BTreeMap<(BlockId, ValueId), ValueId> = BTreeMap::new();
    let mut added: BTreeMap<BlockId, Vec<ValueId>> = BTreeMap::new();
    for &id in &ids {
        let ins: Vec<ValueId> = live[&id].iter().copied().collect();
        for &v in &ins {
            let ty = f.types[v.0];
            let p = f.value(ty);
            param_of.insert((id, v), p);
            f.blocks.get_mut(&id).unwrap().params.push((p, ty));
        }
        added.insert(id, ins);
    }
    for &id in &ids {
        let mut clones: BTreeMap<ValueId, ValueId> = BTreeMap::new();
        let mut needed: Vec<ValueId> = Vec::new();
        for &v in &reads[&id] {
            if let Some((ty, _)) = constant[v.0].filter(|_| home[v.0] != Some(id)) {
                let copy = f.value(ty);
                clones.insert(v, copy);
                needed.push(v);
            }
        }
        let local = |v: ValueId| -> ValueId {
            if home[v.0] == Some(id) {
                v
            } else if let Some(&copy) = clones.get(&v) {
                copy
            } else {
                param_of[&(id, v)]
            }
        };
        let block = f.blocks.get_mut(&id).unwrap();
        for inst in &mut block.insts {
            rename_operands(inst, &local);
        }
        rename_term(&mut block.term, &local);
        for edge in block.term.edges_mut() {
            for &v in &added[&edge.dst] {
                let arg = if let Some(&copy) = clones.get(&v) {
                    copy
                } else if home[v.0] == Some(id) {
                    v
                } else {
                    param_of[&(id, v)]
                };
                edge.args.push(arg);
            }
        }
        let prefix: Vec<Inst> = needed
            .iter()
            .map(|v| {
                let (ty, bits) = constant[v.0].unwrap();
                Inst::Core {
                    value: clones[v],
                    ty,
                    op: Op::Const(ty, bits),
                }
            })
            .collect();
        block.insts.splice(0..0, prefix);
    }
}
