use super::super::ir::{Cvt, IntOp, Op, Ty, *};
use super::dataflow::{Cfg, Sparse};
use super::masks::any_of;
use super::{Access, Analyses};

pub(crate) fn holds_a_lane(f: &Func, analyses: &Analyses, accesses: &[Access]) -> Vec<bool> {
    let ctx = analyses.context();
    let nonempty = nonempty(f, ctx.exec_index, ctx.exec_initial);
    accesses.iter().map(|a| nonempty[a.mask.0]).collect()
}

fn nonempty(f: &Func, exec_index: usize, initial: bool) -> Vec<bool> {
    let cfg = Cfg::new(f);
    let defs = f.definitions();
    let queries = any_of(f);
    let entry_exec = f.blocks[&f.entry].params[exec_index].0;
    let boundary = |p: ValueId| p == entry_exec && initial;
    let edge = |edge: &Edge, src: usize, position: usize, facts: &[bool]| {
        let arg = edge.args[position];
        if facts[arg.0] {
            return true;
        }
        match &cfg.blocks[src].term {
            Term::CondBr { cond, yes, .. } => {
                std::ptr::eq(edge, yes) && queries[cond.0] == Some(arg)
            }
            _ => false,
        }
    };
    let transfer = |_: &Inst, v: ValueId, facts: &[bool]| match defs[v.0] {
        Some(Op::Const(Ty::I1, bits)) => bits & 1 != 0,
        Some(Op::Int(IntOp::Or, a, b)) => facts[a.0] || facts[b.0],
        Some(Op::Convert(Cvt::Bitcast, Ty::I1, a)) => facts[a.0],
        _ => false,
    };
    Sparse {
        cfg: &cfg,
        start: true,
        boundary: &boundary,
        edge: &edge,
        transfer: &transfer,
    }
    .solve(f.types.len())
}
