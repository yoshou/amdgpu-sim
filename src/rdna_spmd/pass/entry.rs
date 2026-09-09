use super::super::ir::*;
use super::super::lift::Input;
use super::super::dialect::DialectRegistry;

pub(crate) fn assume_dispatch_exec(parameter_inputs: &[Input], registry: &DialectRegistry, f: &mut Func) {
    let index = super::super::compiler::exec_index(parameter_inputs, registry);
    let exec = f.blocks[&f.entry].params[index].0;
    super::constant_queries(f, &[exec]);
}

pub(crate) fn packet_state(f: &mut Func) {
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            if let Inst::Effect { provenance, op, inputs, outputs } = inst {
                if *provenance & (1u64 << 63) != 0 {
                    let op = match *op {
                        EffectOp::Wave(WaveOp::Any) => PacketOp::Any,
                        EffectOp::Wave(WaveOp::Ballot) => PacketOp::Ballot,
                        _ => continue,
                    };
                    *inst = Inst::Packet { op, input: inputs[0], output: outputs[0].0 };
                }
            }
        }
    }
}
