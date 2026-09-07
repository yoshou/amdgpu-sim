//! Semantic rewrites of typed SSA, independent of ISA operands and width.
use super::ir::typed::{cfg::*, effect::*, Cvt, Env, IntOp, IntPred, Op, Ty, ValueId};

/// WriteLane requires wave-uniform value and selector. Consequently lane i's
/// result is exactly `i == (selector & 31) ? value : old[i]`: no value needs
/// to cross a packet boundary. This removes the rendezvous, retaining the
/// operation's disregard for EXEC and its selected-lane update.
pub(crate) fn local_write_lane(f: &mut Func, block: BlockId, index: usize) -> Option<usize> {
    let Inst::Effect { op: EffectOp::Wave(WaveOp::WriteLane), inputs, outputs, .. }
        = &f.blocks[&block].insts[index] else { return None; };
    let args = inputs.clone(); let (result, ty) = outputs[0];
    assert_eq!(ty,Ty::I32);
    let lane = f.value(Ty::I32); let mask = f.value(Ty::I32);
    let selector = f.value(Ty::I32); let selected = f.value(Ty::I1);
    let replacement = [
        Inst::Core { value: lane, ty: Ty::I32, op: Op::Env(Env::LaneId) },
        Inst::Core { value: mask, ty: Ty::I32, op: Op::Const(Ty::I32,31) },
        Inst::Core { value: selector, ty: Ty::I32, op: Op::Int(IntOp::And,args[1],mask) },
        Inst::Core { value: selected, ty: Ty::I1, op: Op::Cmp(IntPred::Eq,lane,selector) },
        Inst::Core { value: result, ty: Ty::I32, op: Op::Select(selected,args[0],args[2]) },
    ];
    let end = index + replacement.len();
    f.blocks.get_mut(&block).unwrap().insts.splice(index..index+1,replacement);
    Some(end)
}

/// Only a wave-uniform predicate permits elimination of a wave-wide Any.
/// Replacing the effect in place preserves the instruction-range bindings.
pub(crate) fn uniform_queries(f: &mut Func, uniform_entry: &[super::ir::typed::ValueId]) {
    let facts=super::analysis::uniformity(f,uniform_entry);
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            if let Inst::Effect {op:EffectOp::Wave(WaveOp::Any),inputs,outputs,..}=inst {
                if facts[inputs[0].0] {
                    *inst=Inst::Core {value:outputs[0].0,ty:Ty::I1,op:Op::Convert(Cvt::Bitcast,Ty::I1,inputs[0])};
                }
            }
        }
    }
}

/// Any ignores padding and each launched wave contains a valid work item.
/// A constant predicate on valid lanes therefore has a constant wave result.
pub(crate) fn constant_queries(f:&mut Func,entry_true:&[ValueId]) {
    let facts=super::analysis::valid_predicate_constants(f,entry_true);
    for block in f.blocks.values_mut() {for inst in &mut block.insts {
        let query=match inst {
            Inst::Packet {op:PacketOp::Any,input,output}=>Some((*input,*output)),
            Inst::Effect {op:EffectOp::Wave(WaveOp::Any),inputs,outputs,..}=>Some((inputs[0],outputs[0].0)),
            _=>None,
        };
        if let Some((input,output))=query {
            if let Some(bits)=facts[input.0] {
                *inst=Inst::Core {value:output,ty:Ty::I1,op:Op::Const(Ty::I1,bits)};
            }
        }
    }}
}
