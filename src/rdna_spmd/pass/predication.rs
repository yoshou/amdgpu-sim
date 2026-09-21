use super::super::ir::{Cvt, EffectOp, IntOp, Op, Ty, ValueId, WaveOp, *};
use std::collections::BTreeMap;

fn exec_param(block: &Block, exec_index: usize, f: &Func) -> ValueId {
    let entry = &f.blocks[&f.entry];
    let k = entry.params[..exec_index]
        .iter()
        .filter(|p| p.1 == Ty::I1)
        .count();
    block
        .params
        .iter()
        .filter(|p| p.1 == Ty::I1)
        .nth(k)
        .expect("block lacks its EXEC parameter")
        .0
}

fn same_bit(value: ValueId, exec: ValueId, defs: &[Option<Op>]) -> bool {
    let mut value = value;
    loop {
        if value == exec {
            return true;
        }
        match defs[value.0].as_ref() {
            Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => value = *inner,
            _ => return false,
        }
    }
}

pub struct Predication {
    pub predicated: Vec<Option<(ValueId, ValueId)>>,
    pub masked: Vec<bool>,
    pub masked_result: Vec<Option<ValueId>>,
    pub chain: BTreeMap<BlockId, Vec<(usize, ValueId)>>,
}

pub fn predication(f: &Func, exec_index: usize) -> Predication {
    let defs = &f.definitions();
    let mut predicated: Vec<Option<(ValueId, ValueId)>> = vec![None; f.types.len()];
    let mut masked = vec![false; f.types.len()];
    let mut masked_result = vec![None; f.types.len()];
    let mut chain: BTreeMap<BlockId, Vec<(usize, ValueId)>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        let mut current = exec_param(block, exec_index, f);
        let mut entries = vec![(0usize, current)];
        let query = match &block.term {
            Term::CondBr { cond, .. } => block
                .insts
                .iter()
                .find_map(|inst| match inst {
                    Inst::Packet {
                        op: PacketOp::Any,
                        input,
                        output,
                    } if output == cond => Some(*input),
                    Inst::Effect {
                        op: EffectOp::Wave(WaveOp::Any),
                        inputs,
                        outputs,
                        ..
                    } if outputs[0].0 == *cond => Some(inputs[0]),
                    _ => None,
                })
                .filter(|input| !block.term.edges().any(|e| e.args.contains(input))),
            _ => None,
        };
        for (index, inst) in block.insts.iter().enumerate() {
            let Inst::Core { value, ty, op } = inst else {
                continue;
            };
            if valid_masked(defs, *value).is_some() && query != Some(*value) {
                current = *value;
                entries.push((index + 1, current));
                continue;
            }
            match *op {
                Op::Select(c, new, _) if same_bit(c, current, defs) => {
                    predicated[value.0] = Some((new, current))
                }
                Op::Int(IntOp::And, a, b)
                    if *ty == Ty::I1
                        && (same_bit(a, current, defs) || same_bit(b, current, defs)) =>
                {
                    masked[value.0] = true;
                    masked_result[value.0] = Some(if same_bit(b, current, defs) { a } else { b });
                }
                _ => {}
            }
        }
        chain.insert(id, entries);
    }
    Predication {
        predicated,
        masked,
        masked_result,
        chain,
    }
}

fn valid_masked(defs: &[Option<Op>], value: ValueId) -> Option<ValueId> {
    let valid = |v: ValueId| matches!(defs[v.0], Some(Op::Env(Env::ValidLane)));
    match defs[value.0] {
        Some(Op::Int(IntOp::And, a, b)) if valid(b) => Some(a),
        Some(Op::Int(IntOp::And, a, b)) if valid(a) => Some(b),
        _ => None,
    }
}
