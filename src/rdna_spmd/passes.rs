//! Semantic rewrites of typed SSA, independent of ISA operands and width.
use super::ir::typed::{cfg::*, effect::*, Env, IntOp, IntPred, Op, Ty};

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
