mod address;
mod check;
mod direct;
mod fold;
mod hazard;
mod logic;
mod rewrite;
mod search;

use crate::rdna_spmd::analysis::facts;
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
