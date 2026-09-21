use super::super::analysis::dataflow::{Cfg, Sparse};
use super::super::analysis::uniformity::Fact;
use super::super::analysis::{Analyses, Constants, Context, Packet, Uniformity};
use super::super::ir::{Cvt, IntOp, Op, Ty, *};
use super::access::{accesses, Access};
use super::super::pass::{
    Adjacency, Dce, DeadParams, Driver, PacketState, Pairs, Simplify, WideMemory,
};
use super::super::program::Program;
use super::Prepared;
use std::collections::BTreeMap;

pub fn prepare(
    f: Program,
    packing: super::super::lockstep::Packing,
    num_vgprs: usize,
) -> Prepared {
    let Program {
        mut ir,
        parameter_inputs: inputs,
        registry,
    } = f;
    ir.lowered_to_packets();
    let driver = Driver::new();
    let mut an = Analyses::new(Context {
        exec_initial: true,
        packet: Some(Packet {
            aligned: packing.aligned,
        }),
        ..Context::of(&registry, &inputs, packing.lanes)
    });
    let limit = 1 + ir.types.len();
    driver
        .fixpoint(&mut ir, &mut an, "simplify", limit, &[&Simplify, &Dce])
        .unwrap();
    driver.pipeline(&mut ir, &mut an, &[&PacketState]).unwrap();
    driver
        .pipeline(&mut ir, &mut an, &[&Simplify, &Dce])
        .unwrap();
    driver.pipeline(&mut ir, &mut an, &[&Pairs]).unwrap();
    let limit = 1 + ir.types.len();
    driver
        .fixpoint(
            &mut ir,
            &mut an,
            "simplify",
            limit,
            &[&Simplify, &Dce, &DeadParams, &WideMemory],
        )
        .unwrap();
    driver.pipeline(&mut ir, &mut an, &[&Adjacency]).unwrap();
    let constants = an.get::<Constants>(&ir);
    let uniformity = an.get::<Uniformity>(&ir);
    let accesses = accesses(&ir, &constants, &uniformity.uniform());
    let holds_a_lane = holds_a_lane(&ir, &an, &accesses);
    let uniform = uniformity.uniform();
    let affine: BTreeMap<ValueId, u32> = uniformity
        .facts
        .iter()
        .enumerate()
        .filter_map(|(v, fact)| match *fact {
            Fact::Affine { stride, .. }
                if stride > 0 && stride <= 256 && ir.types[v] == Ty::I64 =>
            {
                Some((ValueId(v), stride as u32))
            }
            _ => None,
        })
        .collect();
    let (yields, groups) = super::yields::layouts(&ir, &uniform, &constants);
    let shapes = accesses
        .iter()
        .map(|a| {
            super::memory::shape(
                a,
                packing.lanes,
                super::memory::global_load(a, &uniform, &affine),
                &constants,
            )
        })
        .collect();
    let clusters =
        super::memory::clusters(&ir, &accesses, packing.lanes, &uniform, &affine);
    let min_private_bytes = accesses
        .iter()
        .filter_map(|a| a.static_scratch_end(&constants))
        .max()
        .unwrap_or(0) as usize;
    let ir = ir
        .verify_with(&registry)
        .expect("invalid prepared function SSA");
    Prepared {
        registry,
        ir,
        inputs,
        width: packing.lanes,
        uniform,
        holds_a_lane,
        accesses,
        shapes,
        clusters,
        yields,
        groups,
        min_private_bytes,
        num_vgprs,
    }
}

fn holds_a_lane(f: &Func, analyses: &Analyses, accesses: &[Access]) -> Vec<bool> {
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

fn any_of(f: &Func) -> Vec<Option<ValueId>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet {
                    op: PacketOp::Any,
                    input,
                    output,
                } => out[output.0] = Some(*input),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    inputs,
                    outputs,
                    ..
                } => out[outputs[0].0 .0] = Some(inputs[0]),
                _ => {}
            }
        }
    }
    out
}
