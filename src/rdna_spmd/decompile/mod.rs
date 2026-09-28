mod address;
mod check;
mod direct;
mod fold;
mod hazard;
mod logic;
mod provenance;
mod rewrite;
mod search;
#[cfg(test)]
mod testing;

use crate::rdna_spmd::analysis::facts;
use crate::rdna_spmd::ir::{BlockId, Func, Term, Ty, ValueId};
use crate::rdna_spmd::program::Program;
use std::collections::BTreeSet;

pub struct Lane {
    pub function: Program,
    pub everyone: BTreeSet<u64>,
}

impl From<Program> for Lane {
    fn from(function: Program) -> Self {
        Self {
            function,
            everyone: BTreeSet::new(),
        }
    }
}

pub use hazard::Hazards;

const MEETING: u64 = 0xC000_0000_0000_0000;

pub fn decompile(function: &Program, hazards: &Hazards) -> Lane {
    let f = &function.ir;
    assert!(
        !f.reads_the_packet(),
        "a wave program holds no packet operation"
    );
    let exec = function.registry.registers().exec;
    let exec_index = function.parameter_inputs.iter().position(
        |p| matches!(p.source, crate::rdna_spmd::ir::ParameterSource::MaskBit(r) if r == exec),
    );
    let inputs = &function.parameter_inputs;
    assert_premises(f, &facts::Facts::new(f, inputs, &BTreeSet::new()), exec_index);
    let meetings = &hazards.meetings;
    let (kept, everyone) = match std::env::var("AMDGPU_SIM_PROOF").as_deref() {
        Ok("direct") => direct::prove(f, inputs, exec_index, hazards),
        Ok("search") | Err(_) => search::prove(f, inputs, exec_index, hazards),
        Ok(other) => panic!("AMDGPU_SIM_PROOF={}: expected search or direct", other),
    };
    let facts = facts::Facts::new(f, &function.parameter_inputs, &kept.words);
    if std::env::var_os("AMDGPU_SIM_PRINT_IR").is_some() {
        eprintln!(
            "; the lane program keeps {} queries, {} words and {} of {} meetings",
            kept.queries.len(),
            kept.words.len(),
            kept.meets.len(),
            meetings.len()
        );
    }
    if std::env::var_os("AMDGPU_SIM_PRINT_MEETINGS").is_some() {
        eprintln!(
            "; b{}: {} accesses, {} pairs whose order may matter, keeps {} of {} meetings",
            f.entry.0,
            hazards.accesses.len(),
            hazards.conflicts().len(),
            kept.meets.len(),
            meetings.len()
        );
        for &(p, q) in &hazards.conflicts() {
            let (a, b) = (&hazards.accesses[p], &hazards.accesses[q]);
            eprintln!(
                ";   {}{} {} {:?} {:?} b{}:{} ({:?} if {:?}) and {} {:?} b{}:{} ({:?} if {:?})",
                if hazards.together.contains(&(p, q)) { "T" } else { "-" },
                if hazards.apart.contains(&(p, q)) { "A" } else { "-" },
                p, a.space, a.kind, a.block.0, a.index, a.address, a.predicate, q, b.kind, b.block.0, b.index, b.address, b.predicate
            );
        }
        for &m in &kept.meets {
            let (b, i) = meetings[m];
            eprintln!(";   meeting before b{}:{}", b.0, i);
        }
    }
    let kept_meetings: std::collections::BTreeMap<(crate::rdna_spmd::ir::BlockId, usize), u64> = kept
        .meets
        .iter()
        .map(|&m| (meetings[m], MEETING | m as u64))
        .collect();
    let mut lane = rewrite::lane_program(f, &facts, &kept, &kept_meetings);
    fold::fold(&mut lane, &function.parameter_inputs, &kept.words, exec_index);
    lane.compact();
    if let Err(e) = lane.check(&function.registry) {
        panic!(
            "the lane program read off a proven wave program is invalid: {}",
            e
        );
    }
    Lane {
        function: Program {
            registry: function.registry.clone(),
            ir: lane,
            parameter_inputs: function.parameter_inputs.clone(),
            entry: function.entry,
        },
        everyone,
    }
}

fn assert_exec_position(f: &Func, facts: &facts::Facts, exec_index: Option<usize>) {
    let Some(i) = exec_index else {
        return;
    };
    for &b in &facts.order {
        assert!(
            f.blocks[&b].params.get(i).is_some_and(|&(_, ty)| ty == Ty::I1),
            "b{}: the exec mask is not parameter {} of every block",
            b.0,
            i
        );
    }
}

fn assert_premises(f: &Func, facts: &facts::Facts, exec_index: Option<usize>) {
    assert_exec_position(f, facts, exec_index);
    for &b in &facts.order {
        let block = &f.blocks[&b];
        let mut own: BTreeSet<ValueId> = block.params.iter().map(|p| p.0).collect();
        let local = |own: &BTreeSet<ValueId>, v: ValueId, b: BlockId| {
            assert!(own.contains(&v), "b{}: uses v{}, which another block defines", b.0, v.0);
        };
        for inst in &block.insts {
            for v in inst.operands() {
                local(&own, v, b);
            }
            own.extend(inst.outputs());
        }
        for edge in block.term.edges() {
            for &v in &edge.args {
                local(&own, v, b);
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => local(&own, *cond, b),
            Term::Ret(args) => {
                for &v in args {
                    local(&own, v, b);
                }
            }
            Term::Br(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::rdna_spmd::ir::{Env, Op, ParameterSource};

    fn premises(b: &Build) {
        let facts = facts::Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        assert_premises(&b.f, &facts, Some(0));
    }

    #[test]
    fn premises_hold_for_blocks_that_pass_every_value_they_use() {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let (next, n) = b.block(&[Ty::I1, Ty::I32]);
        b.br(e, next, vec![p[0], lane]);
        b.f.blocks.get_mut(&next).unwrap().term = Term::Ret(vec![n[1]]);
        premises(&b);
    }

    #[test]
    #[should_panic(expected = "the exec mask is not parameter 0 of every block")]
    fn premises_reject_a_block_whose_exec_parameter_is_missing() {
        let (mut b, _) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let (next, _) = b.block(&[Ty::I32]);
        b.br(e, next, vec![lane]);
        premises(&b);
    }

    #[test]
    #[should_panic(expected = "which another block defines")]
    fn premises_reject_a_block_that_uses_a_value_of_another_block() {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let (next, _) = b.block(&[Ty::I1]);
        b.br(e, next, vec![p[0]]);
        b.f.blocks.get_mut(&next).unwrap().term = Term::Ret(vec![lane]);
        premises(&b);
    }

    #[test]
    #[should_panic(expected = "the exec mask is not parameter 0 of every block")]
    fn hazards_reject_a_block_whose_exec_parameter_is_not_a_bit() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let (next, n) = b.block(&[Ty::I32, Ty::I1, Ty::I64]);
        b.br(e, next, vec![lane, k.exec, buf]);
        let zero = b.constant(next, Ty::I32, 0);
        b.store(next, crate::rdna_spmd::ir::Space::Global, crate::rdna_spmd::ir::MemSize::B32, n[2], zero, n[1]);
        Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
    }
}
