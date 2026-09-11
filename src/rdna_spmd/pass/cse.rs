//! Reuse wave answers with identical SSA operands in the same block.
use super::super::ir::*;
use super::{Analyses, Pass};

pub(crate) struct Cse;
impl Pass for Cse {
    fn name(&self) -> &str { "cse" }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool { run(f) > 0 }
}

fn run(f: &mut Func) -> usize {
    let mut renames = std::collections::BTreeMap::new();
    for block in f.blocks.values_mut() {
        let mut seen: Vec<(WaveOp, Vec<ValueId>, ValueId)> = Vec::new();
        block.insts.retain(|inst| {
            let Inst::Effect { op: EffectOp::Wave(op), inputs, outputs, .. } = inst else { return true };
            if !matches!(op, WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane) { return true; }
            // These queries observe only their captured operands and the
            // wave's immutable valid lanes. In particular, compare the full
            // ReadFirstLane source AND predicate, without stripping selects.
            if let Some((_, _, previous)) = seen.iter().find(|(kind, args, _)| kind == op && args == inputs) {
                renames.insert(outputs[0].0, *previous);
                false
            } else { seen.push((*op, inputs.clone(), outputs[0].0)); true }
        });
    }
    // Resolve uses, including edge arguments, before mask projection. Leaving
    // an identity cast here can hide a ballot's lane predicate from that pass
    // and unnecessarily retain a wave rendezvous.
    if !renames.is_empty() {
        super::simplify::rename(f, &renames);
        super::compact(f);
    }
    renames.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn reuses_answers_without_conflating_sources_masks_or_blocks() {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let params: Vec<_> = [Ty::I1, Ty::I1, Ty::I32, Ty::I32].iter().copied()
            .map(|ty| (f.value(ty), ty)).collect();
        let (exec, restored, source, old) = (params[0].0, params[1].0, params[2].0, params[3].0);
        let written = f.value(Ty::I32);
        let mut insts = vec![Inst::Core { value: written, ty: Ty::I32, op: Op::Select(exec, source, old) }];
        let mut answers = Vec::new();
        for (op, inputs) in [
            (WaveOp::Any, vec![exec]), (WaveOp::Any, vec![exec]),
            (WaveOp::Ballot, vec![exec]), (WaveOp::Ballot, vec![exec]),
            (WaveOp::ReadFirstLane, vec![written, exec]),
            (WaveOp::ReadFirstLane, vec![written, exec]),
            (WaveOp::ReadFirstLane, vec![written, restored]),
            (WaveOp::ReadFirstLane, vec![source, exec]),
            (WaveOp::Any, vec![restored]),
        ] {
            let ty = EffectOp::Wave(op).signature().1[0];
            let value = f.value(ty);
            insts.push(Inst::Effect { provenance: answers.len() as u64, op: EffectOp::Wave(op), inputs, outputs: vec![(value, ty)] });
            answers.push(value);
        }
        let next = f.value(Ty::I32);
        let next_source = f.value(Ty::I32); let next_exec = f.value(Ty::I1);
        let mut next_params = vec![(next_source, Ty::I32), (next_exec, Ty::I1)];
        for &answer in &answers {
            let ty = f.types[answer.0];
            next_params.push((f.value(ty), ty));
        }
        let next_answers = next_params[2..].iter().map(|p| p.0).collect();
        let mut args = vec![written, exec]; args.extend(&answers);
        f.blocks.insert(BlockId(0), Block { params, insts,
            term: Term::Br(Edge { dst: BlockId(1), args }) });
        f.blocks.insert(BlockId(1), Block { params: next_params, insts: vec![Inst::Effect {
            provenance: 20, op: EffectOp::Wave(WaveOp::ReadFirstLane),
            inputs: vec![next_source, next_exec], outputs: vec![(next, Ty::I32)],
        }], term: Term::Ret(next_answers) });
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        f.check(&registry).unwrap();
        assert_eq!(run(&mut f), 3);
        super::super::simplify::run(&mut f);
        f.check(&registry).unwrap();
        let kept: BTreeMap<_, _> = f.blocks[&BlockId(0)].insts.iter().filter_map(|inst| match inst {
            Inst::Effect { provenance, outputs, .. } => Some((*provenance, outputs[0].0)), _ => None,
        }).collect();
        assert_eq!(kept.keys().copied().collect::<Vec<_>>(), vec![0, 2, 4, 6, 7, 8]);
        let Term::Br(edge) = &f.blocks[&BlockId(0)].term else { unreachable!() };
        assert_eq!(&edge.args[2..], &[kept[&0], kept[&0], kept[&2], kept[&2], kept[&4], kept[&4], kept[&6], kept[&7], kept[&8]]);
        assert!(matches!(f.blocks[&BlockId(1)].insts[0], Inst::Effect { .. }));
        assert_eq!(run(&mut f), 0);
    }

    #[test]
    fn a_reused_ballot_crossing_a_block_edge_still_projects_to_its_lane_bit() {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let predicate = f.value(Ty::I1);
        let first = f.value(Ty::I32); let second = f.value(Ty::I32);
        let zero = f.value(Ty::I32); let nonzero = f.value(Ty::I1);
        let carried = f.value(Ty::I32); let previous = f.value(Ty::I1);
        let next_zero = f.value(Ty::I32); let next_nonzero = f.value(Ty::I1);
        f.blocks.insert(BlockId(0), Block { params: vec![(predicate, Ty::I1)], insts: vec![
            Inst::Effect { provenance: 0, op: EffectOp::Wave(WaveOp::Ballot), inputs: vec![predicate], outputs: vec![(first, Ty::I32)] },
            Inst::Effect { provenance: 1, op: EffectOp::Wave(WaveOp::Ballot), inputs: vec![predicate], outputs: vec![(second, Ty::I32)] },
            Inst::Core { value: zero, ty: Ty::I32, op: Op::Const(Ty::I32, 0) },
            Inst::Core { value: nonzero, ty: Ty::I1, op: Op::Cmp(IntPred::Ne, first, zero) },
        ], term: Term::Br(Edge { dst: BlockId(1), args: vec![second, nonzero] }) });
        f.blocks.insert(BlockId(1), Block { params: vec![(carried, Ty::I32), (previous, Ty::I1)], insts: vec![
            Inst::Core { value: next_zero, ty: Ty::I32, op: Op::Const(Ty::I32, 0) },
            Inst::Core { value: next_nonzero, ty: Ty::I1, op: Op::Cmp(IntPred::Ne, carried, next_zero) },
        ], term: Term::Ret(vec![previous, next_nonzero]) });
        assert_eq!(run(&mut f), 1);
        for _ in 0..4 {
            super::super::mask_projection::run(&mut f);
            super::super::simplify::run(&mut f);
            super::super::dce::run(&mut f);
            super::super::dce::dead_params(&mut f);
        }
        f.check(&crate::rdna_spmd::targets::rdna4::registry()).unwrap();
        assert!(!f.blocks.values().flat_map(|b| &b.insts).any(|inst| matches!(inst,
            Inst::Effect { op: EffectOp::Wave(WaveOp::Ballot), .. })));
    }
}
