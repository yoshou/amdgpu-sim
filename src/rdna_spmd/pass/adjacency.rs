use super::super::analysis::{Analyses, Constants, Preserved};
use super::super::ir::*;
use super::Pass;
use std::collections::BTreeSet;

pub struct Adjacency;
impl Pass for Adjacency {
    fn name(&self) -> &str {
        "adjacency"
    }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        run(f) > 0
    }
    fn preserves(&self) -> Preserved {
        Preserved::of::<Constants>()
    }
}

fn member(inst: &Inst) -> bool {
    matches!(inst, Inst::Effect { op: EffectOp::Wave(w), .. } if *w != WaveOp::Wmma)
}

fn barrier(inst: &Inst) -> bool {
    matches!(inst, Inst::Effect { .. } | Inst::Target { .. })
}

fn movable(inst: &Inst) -> bool {
    matches!(inst, Inst::Core { .. } | Inst::Packet { .. })
}

fn run(f: &mut Func) -> usize {
    let mut moved = 0;
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut out: Vec<Inst> = Vec::with_capacity(old.len());
        let mut index = 0;
        while index < old.len() {
            if !member(&old[index]) {
                out.push(old[index].clone());
                index += 1;
                continue;
            }
            let mut produced: BTreeSet<usize> = old[index].outputs().iter().map(|v| v.0).collect();
            let mut members = vec![index];
            let mut scan = index + 1;
            let mut end = index + 1;
            while scan < old.len() {
                if member(&old[scan]) {
                    let Inst::Effect { inputs, .. } = &old[scan] else {
                        unreachable!()
                    };
                    let mut needs = BTreeSet::new();
                    let mut pending: Vec<ValueId> = inputs.to_vec();
                    let span: Vec<&Inst> = old[index..scan].iter().collect();
                    while let Some(v) = pending.pop() {
                        if !needs.insert(v.0) {
                            continue;
                        }
                        if let Some(def) = span.iter().find(|i| i.outputs().contains(&v)) {
                            if !movable(def) {
                                needs.insert(usize::MAX);
                            }
                            def.for_each_operand(|o| pending.push(o));
                        }
                    }
                    if needs.contains(&usize::MAX) || inputs.iter().any(|v| produced.contains(&v.0))
                    {
                        break;
                    }
                    for v in old[scan].outputs() {
                        produced.insert(v.0);
                    }
                    members.push(scan);
                    end = scan + 1;
                } else if barrier(&old[scan]) {
                    break;
                }
                scan += 1;
            }
            if members.len() == 1 {
                out.push(old[index].clone());
                index += 1;
                continue;
            }
            let mut needed: BTreeSet<usize> = BTreeSet::new();
            let mut pending: Vec<ValueId> = members
                .iter()
                .flat_map(|&m| match &old[m] {
                    Inst::Effect { inputs, .. } => inputs.clone(),
                    _ => Vec::new(),
                })
                .collect();
            while let Some(v) = pending.pop() {
                if !needed.insert(v.0) {
                    continue;
                }
                if let Some(def) = old[index..end]
                    .iter()
                    .filter(|i| movable(i))
                    .find(|i| i.outputs().contains(&v))
                {
                    def.for_each_operand(|o| pending.push(o));
                }
            }
            let member: BTreeSet<usize> = members.iter().copied().collect();
            let mut after = Vec::new();
            for slot in index..end {
                if member.contains(&slot) {
                    continue;
                }
                if movable(&old[slot]) && old[slot].outputs().iter().any(|v| needed.contains(&v.0))
                {
                    out.push(old[slot].clone());
                } else {
                    after.push(old[slot].clone());
                }
            }
            for &slot in &members {
                out.push(old[slot].clone());
            }
            moved += members.len() - 1;
            out.extend(after);
            index = end;
        }
        f.blocks.get_mut(&id).unwrap().insts = out;
    }
    moved
}
