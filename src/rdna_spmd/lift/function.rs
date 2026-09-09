//! Whole-function SSA construction from lifted instruction semantics.
//! Register words become block parameters and explicit definitions; effects
//! define their results before outgoing edges are constructed.
use super::*;
use crate::rdna_spmd::decode::{ScalarProgram, Terminator};
use std::collections::{BTreeMap, BTreeSet};
use super::regs::{Word, Words, footprint, words as register_words};

/// Typed SSA before native representation and predication decisions are applied.
#[derive(Clone)]
pub(in crate::rdna_spmd) struct LiftedFunction {
    pub registry: std::sync::Arc<DialectRegistry>,
    pub ir: Func,
    pub parameter_inputs: Vec<Input>,
    pub revision: u64,
}
pub(in crate::rdna_spmd) fn lift(
        registry: std::sync::Arc<DialectRegistry>,
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering>>,
    ) -> LiftedFunction {
        let mut regs = BTreeSet::new();
        for block in lowerings.values() {
            for lowering in block {
                let io = footprint(lowering);
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
            }
        }
        for block in program.blocks.values() {
            if let Terminator::Yield { action, .. } = &block.term {
                let io = action.io();
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
            }
        }
        // Preserve the source use envelope while expressing each observation
        // as the current SSA definition, including conservative adjacent words.
        for block in program.blocks.values() {
            for inst in &block.body {
                let rewrite = super::rewrite::effects_of(inst);
                regs.extend(rewrite.reads.into_iter().chain(rewrite.kills).filter_map(super::rewrite::word));
                regs.extend(super::access::math_reads(inst).into_iter().map(Word::Vgpr));
                regs.extend(super::control::mask_scalar_reads(inst).into_iter().filter_map(Word::scalar));
                regs.extend(super::access::f64_pairs(inst).into_iter().flat_map(|r| [Word::Vgpr(r), Word::Vgpr(r+1)]));
                regs.extend(super::access::sgpr_u64_defs(inst).into_iter().flat_map(|r| [r, r+1]).filter_map(Word::scalar));
                regs.extend(super::access::vgpr_reads(inst).into_iter().chain(super::access::div_reads(inst)).map(Word::Vgpr));
            }
        }
        regs.extend([Word::Mask(106), Word::Mask(126)]);
        let regs: Vec<_> = regs.into_iter().collect();
        let mut f = Func {
            entry: BlockId(program.entry_pc),
            blocks: BTreeMap::new(),
            types: vec![],
        };
        // Block parameters make all incoming definitions, including backedges,
        // explicit before any block body is lifted.
        for &pc in program.blocks.keys() {
            let mut params: Vec<_> = regs.iter().map(|r| (f.value(r.ty()), r.ty())).collect();
            params.push((f.value(Ty::I1), Ty::I1));
            f.blocks.insert(
                BlockId(pc),
                Block {
                    params,
                    insts: vec![],
                    term: Term::Ret(vec![]),
                },
            );
        }
        let mut provenance = 0;
        for (&pc, source) in &program.blocks {
            let mut block = f.blocks.remove(&BlockId(pc)).unwrap();
            let mut words: Words = regs
                .iter()
                .copied()
                .zip(block.params.iter().map(|p| p.0))
                .collect();
            let mut views: BTreeMap<(Word, Ty, bool), ValueId> = BTreeMap::new();
            let mut scc = block.params.last().unwrap().0;
            for lowering in lowerings[&pc].iter() {
                let io = footprint(lowering);
                let writes: Vec<_> = register_words(&io.writes).collect();
                match lowering {
                    Lowering::Wave(action) => {
                        let plan=action.lift(&mut f,&mut block,&mut words,&mut provenance);
                        for &(dest, value) in &plan.definitions {
                            if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                        }
                        invalidate(&mut views, &writes);
                    }
                    Lowering::Memory(m) => {
                        m.lift(&mut f, &mut block, &mut words, &mut views, &mut provenance);
                        invalidate(&mut views, &writes);
                    }
                    Lowering::TypedAlu {
                        inputs,
                        outputs,
                        scalar,
                        expr,
                    } => {
                        alu(&registry, &mut f, &mut block, &mut words, &mut views, &mut scc,
                            &mut provenance, inputs, outputs, scalar, expr, &writes);
                    }
                }
            }
            if let Terminator::Yield { ref action, .. } = source.term {
                let plan = action.lift(&mut f, &mut block, &mut words, &mut provenance);
                super::wave::mark_scheduled(&mut block.insts[plan.core.start..plan.end]);
                for &(dest, value) in &plan.definitions {
                    if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                }
            }

            let edge = |pc| Edge {
                dst: BlockId(pc),
                args: regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect(),
            };
            block.term = match source.term {
                Terminator::Return => Term::Ret(regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect()),
                Terminator::Jump(pc) | Terminator::Barrier { resume: pc } | Terminator::Yield { resume: pc, .. } => Term::Br(edge(pc)),
                Terminator::Branch {
                    cond, taken, fallthrough,
                } => {
                    let result = super::regs::branch(&mut f, &mut block.insts, &words, scc, cond);
                    Term::CondBr {
                        cond: result,
                        yes: edge(taken),
                        no: edge(fallthrough),
                    }
                }
            };
            f.blocks.insert(BlockId(pc), block);
        }
        let parameter_inputs:Vec<_>=regs.iter().map(|word|Input {ty:word.ty(),source:match *word {
            Word::Vgpr(r)=>InputSource::Operand(SourceOperand::VectorRegister(r as u8)),
            Word::Sgpr(r)=>InputSource::Operand(SourceOperand::ScalarRegister(r as u8)),
            Word::Mask(r)=>InputSource::MaskBit(r),
        }}).chain(std::iter::once(Input {ty:Ty::I1,source:InputSource::Scc})).collect();
        LiftedFunction { registry, ir: f, parameter_inputs, revision: 0 }
}
pub(in crate::rdna_spmd) fn assume_dispatch_exec(parameter_inputs:&[Input],f:&mut Func) {
    let index=parameter_inputs.iter().position(|p|matches!(p.source,InputSource::MaskBit(126))).unwrap();
    let exec=f.blocks[&f.entry].params[index].0;
    crate::rdna_spmd::pass::constant_queries(f,&[exec]);
}
/// The whole-program ABI keeps mask words within one packet. Explicit
/// cross-lane instructions retain their separately selected scope.
pub(in crate::rdna_spmd) fn packet_state(f:&mut Func) {
        for block in f.blocks.values_mut() {for inst in &mut block.insts {
            if let Inst::Effect {provenance,op,inputs,outputs}=inst {
                if *provenance & (1u64<<63)!=0 {
                    let op=match *op {
                        effect::EffectOp::Wave(effect::WaveOp::Any)=>PacketOp::Any,
                        effect::EffectOp::Wave(effect::WaveOp::Ballot)=>PacketOp::Ballot,
                        _=>continue,
                    };
                    *inst=Inst::Packet {op,input:inputs[0],output:outputs[0].0};
                }
            }
        }}
    }
fn invalidate(views: &mut BTreeMap<(Word, Ty, bool), ValueId>, writes: &[Word]) {
    views.retain(|&(r, t, _), _| {
        !(0..t.bits().div_ceil(32)).filter_map(|k| r.offset(k)).any(|r| writes.contains(&r))
    });
}

/// Instantiate a replacement ALU expression in an existing SSA value namespace.
/// Used by both initial lifting and typed instruction rewrites.
pub(super) fn alu(
    registry: &DialectRegistry, f: &mut Func, block: &mut Block,
    words: &mut Words, views: &mut super::regs::Views, scc: &mut ValueId,
    provenance: &mut u64, inputs: &[Input], outputs: &[Output], scalar: &bool,
    expr: &VerifiedExpr, writes: &[Word],
) {
    let first_value = f.types.len();
                        let mut operands = super::regs::Operands::default();
                        let args: Vec<_> = inputs.iter().map(|input| operands.read(
                            input, *scalar, Some(*scc), f, block, words, views,
                        )).collect();
                        // A destination's old value is an update dependency,
                        // not a source view available to later instructions.
                        let mut previous_views=views.clone();
                        let previous:Vec<_>=outputs.iter().map(|output|match *output {
                            Output::Vgpr(reg,ty)=>{
                                let input=Input {source:InputSource::Operand(SourceOperand::VectorRegister(reg as u8)),ty};
                                let value=operands.read(&input,*scalar,Some(*scc),f,block,words,&mut previous_views);
                                Some(value)
                            },
                            _=>None,
                        }).collect();
                        // Predicated destinations and mask writes read the previous
                        // architectural definitions before any result is assigned.
                        for &word in writes.iter().chain(std::iter::once(&Word::Mask(126))) {
                            let source = match word {
                                Word::Vgpr(r) => InputSource::Operand(SourceOperand::VectorRegister(r as u8)),
                                Word::Sgpr(r) => InputSource::Operand(SourceOperand::ScalarRegister(r as u8)),
                                Word::Mask(r) => InputSource::MaskBit(r),
                            };
                            operands.bindings.push((Input { source, ty: word.ty() },words[&word]));
                        }
                        let mut values = args;
                        block.insts.extend(operands.core);
                        for inst in &expr.expr().insts {
                            match inst {
                                ExprInst::Core(ty, op) => {
                                    let op = op.map(|v| values[v.0]);
                                    let value = f.value(*ty);
                                    block.insts.push(Inst::Core { value, ty: *ty, op });
                                    values.push(value);
                                }
                                ExprInst::Target { op, args, outputs } => {
                                    let args = args.map(|v| values[v.0]);
                                    let outputs = outputs.iter().map(|&ty| (f.value(ty), ty)).collect::<Vec<_>>();
                                    values.extend(outputs.iter().map(|&(id, _)| id));
                                    let effect = registry.operation(*op).unwrap().effect;
                                    let id = if effect == crate::rdna_spmd::dialect::Effect::Pure { None }
                                        else { let id = *provenance; *provenance += 1; Some(id << 8) };
                                    block.insts.push(Inst::Target { provenance: id, op: *op, args, outputs });
                                }
                            }
                        }
                        let results: Vec<_> = outputs.iter().copied()
                            .zip(expr.expr().results.iter().map(|id| values[id.0])).collect();
                        let defined=super::regs::define(f,&mut block.insts,words,&results,&previous,*scalar,scc,first_value,writes);
                        let super::regs::Definitions {stored,..}=defined;
                        invalidate(views, &writes);
                        for (&output, &(stored, _)) in outputs.iter().zip(&stored) {
                            match output {
                                Output::Vgpr(reg, ty) => { views.insert((Word::Vgpr(reg), ty, false), stored); }
                                Output::Scalar(reg, ty) => if let Some(reg) = Word::scalar(reg).filter(|r| r.ordinary_span(ty)) {
                                    views.insert((reg, ty, true), stored);
                                },
                                _ => {}
                            }
                        }
}
