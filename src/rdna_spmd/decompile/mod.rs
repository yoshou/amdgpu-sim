mod check;
mod direct;
mod fold;
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

pub fn decompile(function: &Program) -> Lane {
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
    let (kept, everyone) = match std::env::var("AMDGPU_SIM_PROOF").as_deref() {
        Ok("direct") => direct::prove(f, inputs, exec_index),
        Ok("search") | Err(_) => search::prove(f, inputs, exec_index),
        Ok(other) => panic!("AMDGPU_SIM_PROOF={}: expected search or direct", other),
    };
    let facts = facts::Facts::new(f, &function.parameter_inputs, &kept.words);
    if std::env::var_os("AMDGPU_SIM_PRINT_IR").is_some() {
        eprintln!(
            "; the lane program keeps {} queries and {} words",
            kept.queries.len(),
            kept.words.len()
        );
    }
    let mut lane = rewrite::lane_program(f, &facts, &kept);
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
        },
        everyone,
    }
}
