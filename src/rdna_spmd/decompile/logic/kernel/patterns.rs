use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;

pub fn constant_choices(f: &Func, facts: &Facts, v: ValueId) -> Option<Vec<u64>> {
    let mut found = Vec::new();
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match facts.op(f, v)? {
            Op::Const(_, k) => {
                if !found.contains(&k) {
                    found.push(k);
                }
            }
            Op::Select(_, a, b) => stack.extend([a, b]),
            _ => return None,
        }
        if found.len() + stack.len() > 8 {
            return None;
        }
    }
    Some(found)
}

pub fn lane_test(f: &Func, facts: &Facts, a: ValueId, b: ValueId) -> Option<ValueId> {
    if facts.lane_word[a.0] && facts.constant(f, b) == Some(0) {
        Some(a)
    } else if facts.lane_word[b.0] && facts.constant(f, a) == Some(0) {
        Some(b)
    } else {
        None
    }
}

pub fn projected_word(f: &Func, facts: &Facts, s: ValueId) -> Option<ValueId> {
    match facts.op(f, s) {
        Some(Op::Int(IntOp::LShr, w, lane)) if facts.is_lane_id(f, lane) => Some(w),
        _ => None,
    }
}
