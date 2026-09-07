//! Whole-function SSA construction over the mixed migration stream.
//!
//! Register words are SSA variables at the adapter boundary. Each CFG edge
//! passes their current definitions to destination block parameters. A legacy
//! operation consumes the previous state and defines only its written words;
//! typed ALU operations share the same value namespace. The backend lowers
//! boundary words/block arguments through explicit native SSA definitions, so
//! this adds no runtime register-file traffic or lane helper calls.
use super::*;
use crate::rdna_spmd::boundary::BoundaryIo;
use crate::rdna_spmd::ir::typed::cfg::*;
use crate::rdna_spmd::ir::{ScalarProgram, Terminator};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use super::state::{Word, Words, footprint, words as register_words};

pub(in crate::rdna_spmd) struct Alu {
    pub inputs: Vec<(Input, ValueId)>,
    pub source_inputs: usize,
    pub previous_core: Range<usize>,
    pub updates_start: usize,
    pub core: Range<usize>,
    pub outputs: Vec<(Output, ValueId)>,
    pub scalar: bool,
    pub mask_logic: bool,
    pub predicated: Vec<(ValueId,ValueId)>,
    pub packet_elidable: Vec<ValueId>,
    pub scalar_unmasked: Vec<ValueId>,
    pub mask_updates: Vec<(ValueId,u32)>,
    pub vector_words: Vec<ValueId>,
    /// A native canonical pair can satisfy its explicit Pack64 and float view
    /// together. This is representation coalescing, not another SSA definition.
    pub pairs: Vec<(ValueId, ValueId)>,
}
pub(in crate::rdna_spmd) struct BlockPlan {
    pub instructions: Vec<Option<Alu>>,
    pub memory: BTreeMap<usize, super::memory::Plan>,
    pub wave: BTreeMap<usize,(super::wave::YieldAction,super::wave::Plan)>,
    pub outgoing: Vec<ValueId>,
    pub yield_values: Option<super::wave::Plan>,
    pub condition: Option<Condition>,
    yielding: bool,
}
pub(in crate::rdna_spmd) struct Condition {
    pub input: Input,
    pub input_value: ValueId,
    pub core: Range<usize>,
}
pub(in crate::rdna_spmd) enum Control {
    Return,
    Jump(usize),
    Branch { cond: ValueId, taken: usize, fallthrough: usize },
    Yield { resume: usize },
}
pub(in crate::rdna_spmd) enum Preparation {
    Packet { inactive: Vec<(usize,usize)>, observe_return: bool },
    Scalar { active: Vec<(usize,usize)>, dispatch: bool },
    #[cfg(test)]
    Inspect,
}
pub(in crate::rdna_spmd) struct Function {
    pub registry: std::sync::Arc<DialectRegistry>,
    /// Architectural words assigned by this compiled function. Unassigned
    /// words remain in the scheduler's private state buffer across a call.
    pub written: crate::rdna_spmd::boundary::RegSet,
    pub ir: VerifiedFunc,
    pub blocks: BTreeMap<usize, BlockPlan>,
    pub live: Vec<bool>,
    pub observable_return: bool,
}
impl Function {
    pub fn new<'a>(
        registry: std::sync::Arc<DialectRegistry>,
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering<'a>>>,
        boundary: Option<&BTreeMap<usize, BoundaryIo>>,
        preparation: Preparation,
    ) -> Self {
        let mut regs = BTreeSet::new();
        let mut written = crate::rdna_spmd::boundary::RegSet::default();
        for block in lowerings.values() {
            for lowering in block {
                let io = footprint(lowering);
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
                for r in io.writes.sgprs() { if r != 124 { written.add_sgpr(r); } }
                for r in io.writes.vgprs() { written.add_vgpr(r); }
                written.scc |= io.writes.scc;
            }
        }
        for block in program.blocks.values() {
            if let Terminator::Yield { action, .. } = &block.term {
                let io = action.io();
                for r in io.writes.sgprs() { if r != 124 { written.add_sgpr(r); } }
                for r in io.writes.vgprs() { written.add_vgpr(r); }
                written.scc |= io.writes.scc;
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
            }
        }
        if let Some(boundary) = boundary {
            for io in boundary.values() {
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
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
                    term: Term::Ret,
                },
            );
        }
        let mut plans = BTreeMap::new();
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
            let mut instructions = Vec::with_capacity(source.body.len());
            let mut memory = BTreeMap::new();
            let mut wave = BTreeMap::new();
            for lowering in &lowerings[&pc] {
                let io = footprint(lowering);
                let writes: Vec<_> = register_words(&io.writes).collect();
                let plan = match lowering {
                    Lowering::Wave(action) => {
                        let plan=action.lift(&mut f,&mut block,&mut words,&mut provenance);
                        for &(dest, value) in &plan.definitions {
                            if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                        }
                        invalidate(&mut views, &writes);
                        wave.insert(instructions.len(),(action.clone(),plan));
                        None
                    }
                    Lowering::Memory(m) => {
                        let plan=m.lift(&mut f, &mut block, &mut words, &mut views, &mut provenance);
                        memory.insert(instructions.len(),plan);
                        invalidate(&mut views, &writes);
                        None
                    }
                    Lowering::TypedAlu {
                        inputs,
                        outputs,
                        scalar,
                        expr,
                    } => {
                        let mut operands = super::state::Operands::default();
                        let args: Vec<_> = inputs.iter().map(|input| operands.read(
                            input, *scalar, Some(scc), &mut f, &mut block, &words, &mut views,
                        )).collect();
                        let source_inputs=operands.bindings.len();
                        let previous_start=operands.core.len();
                        // A destination's old value is an update dependency,
                        // not a source view available to later instructions.
                        let mut previous_views=views.clone();
                        let previous:Vec<_>=outputs.iter().map(|output|match *output {
                            Output::Vgpr(reg,ty)=>{
                                let input=Input {source:InputSource::Operand(SourceOperand::VectorRegister(reg as u8)),ty};
                                let value=operands.read(&input,*scalar,Some(scc),&mut f,&mut block,&words,&mut previous_views);
                                if !operands.bindings.iter().any(|(_,id)|*id==value) {operands.bindings.push((input,value));}
                                Some(value)
                            },
                            _=>None,
                        }).collect();
                        let previous_end=operands.core.len();
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
                        let start = block.insts.len();
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
                                        else { let id = provenance; provenance += 1; Some(id) };
                                    block.insts.push(Inst::Target { provenance: id, op: *op, args, outputs });
                                }
                            }
                        }
                        let updates_start=block.insts.len();
                        let results: Vec<_> = outputs.iter().copied()
                            .zip(expr.expr().results.iter().map(|id| values[id.0])).collect();
                        for &(output, value) in &results {
                            if matches!(output, Output::Scc) { scc = if *scalar { value } else {
                                super::state::query(&mut f,&mut block.insts,effect::WaveOp::Any,value)
                            }; }
                        }
                        let exec=super::state::core(&mut f,&mut block.insts,Ty::I1,
                            Op::Convert(Cvt::Bitcast,Ty::I1,words[&Word::Mask(126)]));
                        operands.bindings.push((Input {source:InputSource::ExecPredicate,ty:Ty::I1},exec));
                        let exec=Some(exec);
                        let mut stored = Vec::new();
                        let mut predicated=Vec::new();
                        let mut packet_elidable=Vec::new();
                        let mut scalar_unmasked=Vec::new();
                        let mut mask_updates=Vec::new();
                        let mut word_defs = BTreeMap::new();
                        for (output_index,&(output, result)) in results.iter().enumerate() {
                            let ty = output.ty();
                            let value = match output {
                                Output::Vgpr(_, _) => {
                                    let old=previous[output_index].unwrap();
                                    let value = f.value(ty);
                                    block.insts.push(Inst::Core { value, ty,
                                        op: Op::Select(exec.unwrap(), result, old) });
                                    value
                                }
                                Output::Scalar(..) | Output::Scc => result,
                                Output::Compare(_) | Output::Mask(_) => {
                                    super::state::core(&mut f,&mut block.insts,Ty::I1,
                                        Op::Int(IntOp::And,result,exec.unwrap()))
                                }
                                Output::MaskBit(_) => result,
                            };
                            if matches!(output,Output::Vgpr(..)|Output::Compare(_)|Output::Mask(_)) {
                                predicated.push((value,result));
                            }
                            if matches!(output,Output::Vgpr(..)) || matches!(output,Output::Compare(r) if r!=126) {
                                packet_elidable.push(value);
                            }
                            if matches!(output,Output::Mask(_)) {scalar_unmasked.push(value);}
                            if let Output::Mask(r)|Output::Compare(r)|Output::MaskBit(r)=output {mask_updates.push((value,r));}
                            stored.push((value, ty));
                            if let Output::Compare(reg) | Output::Mask(reg) | Output::MaskBit(reg) = output {
                                if let Some(word) = Word::scalar(reg) {
                                    let raw = if matches!(word,Word::Mask(_)) { value }
                                        else { super::state::query(&mut f,&mut block.insts,effect::WaveOp::Ballot,value) };
                                    let raw=if word==Word::Mask(126) {super::state::valid_exec(&mut f,&mut block.insts,raw)} else {raw};
                                    if matches!(word,Word::Mask(_)) && raw!=value {mask_updates.push((raw,reg));}
                                    word_defs.insert(word,raw);
                                }
                                continue;
                            }
                            let (reg, scalar) = match output {
                                Output::Vgpr(r, _) => (r,false),
                                Output::Scalar(r, _) => (r,true),
                                Output::Scc => continue,
                                _ => unreachable!(),
                            };
                            let bits = if matches!(ty,Ty::F32|Ty::F64) {
                                let int_ty = if ty == Ty::F32 { Ty::I32 } else { Ty::I64 };
                                super::state::core(&mut f,&mut block.insts,int_ty,Op::Convert(Cvt::Bitcast,int_ty,value))
                            } else { value };
                            for k in 0..ty.bits().div_ceil(32) {
                                let word = if scalar { Word::scalar(reg+k) } else { Some(Word::Vgpr(reg+k)) };
                                if let Some(word) = word {
                                    let raw = if ty.bits() == 32 { bits } else {
                                        super::state::core(&mut f,&mut block.insts,Ty::I32,
                                            if k==0 { Op::UnpackLo(bits) } else { Op::UnpackHi(bits) })
                                    };
                                    let raw = if matches!(word,Word::Mask(_)) {
                                        super::state::project(&mut f,&mut block.insts,raw)
                                    } else { raw };
                                    let raw=if word==Word::Mask(126) {super::state::valid_exec(&mut f,&mut block.insts,raw)} else {raw};
                                    if let Word::Mask(reg)=word {mask_updates.push((raw,reg));}
                                    word_defs.insert(word,raw);
                                }
                            }
                        }
                        for &r in &writes {
                            words.insert(r,*word_defs.get(&r).expect("typed output lacks its architectural SSA definition"));
                        }
                        let end = block.insts.len();
                        let mut emitted = Vec::new();
                        for (output_index,&(output,result)) in results.iter().enumerate() {
                            match output {
                                Output::Compare(r) | Output::Mask(r) | Output::MaskBit(r) => {
                                    if let Some(word) = Word::scalar(r) {
                                        emitted.push((if matches!(word,Word::Mask(_)) { Output::MaskBit(r) }
                                            else { Output::Scalar(r,Ty::I32) },word_defs[&word]));
                                    }
                                }
                                Output::Scalar(r,ty) if !Word::scalar(r).is_some_and(|w| w.ordinary_span(ty)) => {
                                    for k in 0..ty.bits().div_ceil(32) {
                                        if let Some(word) = Word::scalar(r+k) {
                                            emitted.push((if matches!(word,Word::Mask(_)) { Output::MaskBit(r+k) }
                                                else { Output::Scalar(r+k,Ty::I32) },word_defs[&word]));
                                        }
                                    }
                                }
                                Output::Scc => emitted.push((output,scc)),
                                Output::Vgpr(..) => emitted.push((output,stored[output_index].0)),
                                _ => emitted.push((output,result)),
                            }
                        }
                        invalidate(&mut views, &writes);
                        for (&output, &(stored, _)) in outputs.iter().zip(&stored) {
                            match output {
                                Output::Vgpr(reg, ty) => { views.insert((Word::Vgpr(reg), ty, false), stored); }
                                Output::Scalar(reg, ty) => if let Some(reg) = Word::scalar(reg).filter(|r| r.ordinary_span(ty)) {
                                    views.insert((reg, ty, true), stored);
                                },
                                _ => {}
                            }
                        }
                        Some(Alu {
                            inputs: operands.bindings,
                            source_inputs,
                            previous_core:(start+previous_start)..(start+previous_end),
                            updates_start,
                            core: start..end,
                            outputs: emitted,
                            scalar: *scalar,
                            mask_logic:*scalar&&inputs.iter().all(|i|i.ty==Ty::I32)
                                &&outputs.iter().all(|o|matches!(o,Output::Scalar(_,Ty::I32)|Output::Scc))
                                &&expr.expr().insts.iter().all(|i|matches!(i,ExprInst::Core(_,Op::Const(..)|Op::Int(IntOp::And|IntOp::Or|IntOp::Xor,_,_)|Op::Cmp(IntPred::Ne,_,_)))),
                            predicated,
                            packet_elidable,
                            scalar_unmasked,
                            mask_updates,
                            vector_words: word_defs.iter().filter_map(|(word,&id)|matches!(word,Word::Vgpr(_)).then_some(id)).collect(),
                            pairs: operands.pairs,
                        })
                    }
                    Lowering::Legacy(_) => {
                        // Remaining register/control and target-specific adapters
                        // retain their original order during state migration.
                        let mut deps: Vec<_> = register_words(&io.reads)
                            .filter_map(|r| words.get(&r).copied()).collect();
                        deps.extend(writes.iter().filter_map(|r| words.get(r).copied()));
                        let mut outputs = vec![];
                        for &r in &writes {
                            let v = f.value(r.ty());
                            words.insert(r, v);
                            outputs.push((v, r.ty()));
                        }
                        // The remaining mask SALU defines SCC; ordinary moves
                        // and vector operations preserve its previous definition.
                        if io.writes.scc {
                            deps.push(scc);
                            scc = f.value(Ty::I1);
                            outputs.push((scc, Ty::I1));
                        }
                        block.insts.push(Inst::Boundary {
                            inputs: deps,
                            outputs,
                        });
                        invalidate(&mut views, &writes);
                        None
                    }
                };
                instructions.push(plan);
            }
            let yield_values = if let Terminator::Yield { ref action, .. } = source.term {
                let plan = action.lift(&mut f, &mut block, &mut words, &mut provenance);
                for &(dest, value) in &plan.definitions {
                    if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                }
                Some(plan)
            } else { None };
            if let Terminator::Barrier { resume } = source.term {
                if let Some(io) = boundary.and_then(|map| map.get(&resume)) {
                    // A host-side wave operation runs between yield and resume.
                    // Its register outputs are new definitions on this edge,
                    // not the values written back before yielding.
                    let mut inputs: Vec<_> = register_words(&io.reads)
                        .chain(register_words(&io.writes))
                        .map(|r| words[&r])
                        .collect();
                    let mut outputs: Vec<_> = register_words(&io.writes)
                        .map(|r| {
                            let value = f.value(r.ty());
                            words.insert(r, value);
                            (value, r.ty())
                        })
                        .collect();
                    if io.reads.scc || io.writes.scc { inputs.push(scc); }
                    if io.writes.scc {
                        scc = f.value(Ty::I1);
                        outputs.push((scc, Ty::I1));
                    }
                    block.insts.push(Inst::Boundary { inputs, outputs });
                }
            }
            let edge = |pc| Edge {
                dst: BlockId(pc),
                args: regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect(),
            };
            let mut condition = None;
            block.term = match source.term {
                Terminator::Return => Term::Ret,
                Terminator::Jump(pc) | Terminator::Barrier { resume: pc } | Terminator::Yield { resume: pc, .. } => Term::Br(edge(pc)),
                Terminator::Branch {
                    cond, taken, fallthrough,
                } => {
                    use crate::rdna_spmd::ir::Cond;
                    let (input_source, input_value) = match cond {
                        Cond::Scc0 | Cond::Scc1 => (InputSource::Scc,scc),
                        Cond::ExecZ | Cond::ExecNz => (InputSource::MaskBit(126),words[&Word::Mask(126)]),
                        Cond::VccZ | Cond::VccNz => (InputSource::MaskBit(106),words[&Word::Mask(106)]),
                    };
                    let start = block.insts.len();
                    let query=if matches!(cond,Cond::Scc0|Cond::Scc1) {input_value} else {
                        let query=f.value(Ty::I1);
                        block.insts.push(Inst::Packet {op:PacketOp::Any,input:input_value,output:query});
                        query
                    };
                    let result = if matches!(cond, Cond::Scc0 | Cond::ExecZ | Cond::VccZ) {
                        let zero = f.value(Ty::I1);
                        block.insts.push(Inst::Core { value: zero, ty: Ty::I1, op: Op::Const(Ty::I1, 0) });
                        let result = f.value(Ty::I1);
                        block.insts.push(Inst::Core { value: result, ty: Ty::I1, op: Op::Cmp(IntPred::Eq, query, zero) });
                        result
                    } else { query };
                    condition = Some(Condition { input: Input { source: input_source, ty: Ty::I1 },
                        input_value, core: start..block.insts.len() });
                    Term::CondBr {
                        cond: result,
                        yes: edge(taken),
                        no: edge(fallthrough),
                    }
                }
            };
            f.blocks.insert(BlockId(pc), block);
            plans.insert(
                pc,
                BlockPlan {
                    instructions,
                    memory,
                    wave,
                    yield_values,
                    condition,
                    yielding: matches!(source.term, Terminator::Yield { .. } | Terminator::Barrier { .. }),
                    outgoing: regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect(),
                },
            );
        }
        for (&pc, block) in &mut plans {
            if let Some(plan) = &mut block.yield_values {
                if let Some(end) = crate::rdna_spmd::passes::local_write_lane(&mut f,BlockId(pc),plan.core.end) {
                    plan.core.end = end;
                    plan.local = true;
                }
            }
        }
        // The invocation ABI supplies one scalar register bank per wave;
        // per-lane VGPR and mask bindings carry no uniformity assumption.
        let entry=&f.blocks[&f.entry];
        let uniform_entry: Vec<_>=regs.iter().zip(&entry.params)
            .filter_map(|(r,p)|matches!(r,Word::Sgpr(_)).then_some(p.0))
            .chain(std::iter::once(entry.params[regs.len()].0)).collect();
        crate::rdna_spmd::passes::constant_queries(&mut f,&[]);
        crate::rdna_spmd::passes::uniform_queries(&mut f,&uniform_entry);
        let constants = crate::rdna_spmd::analysis::constants(&f);
        for block in plans.values_mut() {
            if let Some(plan) = &mut block.yield_values {
                for (index, id) in plan.arguments.iter().enumerate() {
                    use crate::rdna_spmd::ir::typed::effect::{EffectOp, WaveOp};
                    if plan.layout.op == EffectOp::Wave(WaveOp::Wmma)
                        || plan.layout.op == EffectOp::Wave(WaveOp::WriteLane) && index == 2 { continue; }
                    if let Some(bits) = constants[id.0] {
                        plan.layout.arguments[index] = crate::rdna_spmd::yield_values::Argument::Constant(bits as u32);
                    }
                }
            }
        }
        let parameter_inputs:Vec<_>=regs.iter().map(|word|Input {ty:word.ty(),source:match *word {
            Word::Vgpr(r)=>InputSource::Operand(SourceOperand::VectorRegister(r as u8)),
            Word::Sgpr(r)=>InputSource::Operand(SourceOperand::ScalarRegister(r as u8)),
            Word::Mask(r)=>InputSource::MaskBit(r),
        }}).chain(std::iter::once(Input {ty:Ty::I1,source:InputSource::Scc})).collect();
        // Finish all plan rewrites while the graph is owned by construction.
        // Only the final graph crosses the verified code-generation boundary.
        let mut scalar_live=None;
        let observable_return=match preparation {
            Preparation::Packet {inactive,observe_return}=>{
                Self::packet_state(&mut f);
                if !observe_return {
                    Self::elide_inactive_updates(&plans,&mut f,inactive);
                    Self::assume_dispatch_exec(&parameter_inputs,&mut f);
                }
                observe_return
            },
            Preparation::Scalar {active,dispatch}=>{
                Self::packet_state(&mut f);
                Self::scalar_mask_selects(&plans,&mut f);
                if !dispatch {
                    scalar_live=Some(crate::rdna_spmd::analysis::live_values(&f,plans.values().flat_map(|b|&b.outgoing).copied()));
                }
                Self::elide_active_updates(&plans,&mut f,active);
                if dispatch {Self::assume_dispatch_exec(&parameter_inputs,&mut f);}
                !dispatch
            },
            #[cfg(test)]
            Preparation::Inspect=>true,
        };
        let live=scalar_live.unwrap_or_else(||{
            let mut roots=vec![];
            for (&pc,plan) in &plans {
                if let Some(yielding)=&plan.yield_values {roots.extend(yielding.results.iter().map(|p|p.0));}
                if observable_return&&matches!(f.blocks[&BlockId(pc)].term,Term::Ret) {roots.extend(&plan.outgoing);}
            }
            crate::rdna_spmd::analysis::live_values(&f,roots)
        });
        let ir=f.verify_with(&registry).expect("invalid prepared function SSA");
        Self {
            ir,
            live,
            observable_return,
            registry,
            written,
            blocks: plans,
        }
    }
    /// Apply the plan's proof that these temporary destinations cannot be
    /// observed by a reactivated lane or an unpredicated wave consumer.
    fn elide_inactive_updates(blocks:&BTreeMap<usize,BlockPlan>,f:&mut Func,sites:impl IntoIterator<Item=(usize,usize)>) {
        for (pc,index) in sites {
            let Some(alu)=&blocks[&pc].instructions[index] else {continue};
            for (output,id) in &alu.outputs {
                if !matches!(output,Output::Vgpr(..)) {continue;}
                for inst in &mut f.blocks.get_mut(&BlockId(pc)).unwrap().insts[alu.core.clone()] {
                    if let Inst::Core {value,ty,op}=inst {
                        if value==id {
                            if let Op::Select(_,yes,_)=*op {*op=Op::Convert(Cvt::Bitcast,*ty,yes);}
                        }
                    }
                }
            }
        }
    }
    /// Transfer the existing scalar plan's active-lane facts to the explicit
    /// SSA updates, including comparison masks. The native emitter must not
    /// silently change architectural SSA values when dropping predication.
    fn elide_active_updates(blocks:&BTreeMap<usize,BlockPlan>,f:&mut Func,sites:impl IntoIterator<Item=(usize,usize)>) {
        let active:std::collections::BTreeSet<_>=sites.into_iter().collect();
        for (&pc,plan) in blocks {for (index,alu) in plan.instructions.iter().enumerate() {
            let Some(alu)=alu else {continue};
            for inst in &mut f.blocks.get_mut(&BlockId(pc)).unwrap().insts[alu.core.clone()] {
                if let Inst::Core {value,ty,op}=inst {
                    if let Some(&(_,raw))=alu.predicated.iter().find(|(id,_)|id==value&&(active.contains(&(pc,index))||alu.scalar_unmasked.contains(value))) {
                        *op=match *op {
                            Op::Select(..)|Op::Int(IntOp::And,_,_)|Op::Convert(Cvt::Bitcast,_,_)=>Op::Convert(Cvt::Bitcast,*ty,raw),
                            _=>unreachable!("unknown predicated SSA update"),
                        };
                    }
                }
            }
        }}
    }
    /// The ordinary dispatch/run ABI seeds EXEC from immutable lane validity.
    /// Cooperative callers may supply arbitrary EXEC and do not use this proof.
    fn assume_dispatch_exec(parameter_inputs:&[Input],f:&mut Func) {
        let index=parameter_inputs.iter().position(|p|matches!(p.source,InputSource::MaskBit(126))).unwrap();
        let exec=f.blocks[&f.entry].params[index].0;
        crate::rdna_spmd::passes::constant_queries(f,&[exec]);
    }
    /// Ordinary dispatch exposes memory effects, not the final register bank.
    #[cfg(test)]
    pub fn discard_return_state(&mut self) {
        self.observable_return = false;
        let roots = self.blocks.values().filter_map(|b| b.yield_values.as_ref())
            .flat_map(|p| p.results.iter().map(|r| r.0));
        self.live = crate::rdna_spmd::analysis::live_values(self.ir.func(), roots);
    }
    pub fn min_private_bytes(&self) -> usize {
        self.blocks.values().flat_map(|b|b.memory.values())
            .filter_map(|p|p.memory.private_load_end()).max().unwrap_or(0) as usize
    }
    /// The existing whole-program ABI keeps mask words within one packet.
    /// Explicit cross-lane instructions retain their separately selected scope.
    fn packet_state(f:&mut Func) {
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
    /// Preserve the scalar backend's existing de-SIMT mask blend rule. Its
    /// two masked SGPR definitions represent whole scalar values, and their
    /// final OR selects a value using the single simulated lane's mask bit.
    /// This is specific to the ordinary scalar ABI, not a wave-word identity.
    fn scalar_mask_selects(blocks:&BTreeMap<usize,BlockPlan>,f:&mut Func) {
        for (&pc,plan) in blocks {
            let block=f.blocks.get_mut(&BlockId(pc)).unwrap();
            let mut definitions=BTreeMap::new();
            let mut ballots=BTreeMap::new();
            let mut blends:BTreeMap<ValueId,(bool,ValueId,ValueId)>=BTreeMap::new();
            for alu in plan.instructions.iter().flatten() {
                for inst in &block.insts[alu.core.clone()] {
                    match inst {
                        Inst::Core {value,op,..}=>{definitions.insert(*value,Some(*op));},
                        Inst::Packet {op:PacketOp::Ballot,input,output}=>{
                            ballots.insert(*output,*input);
                        },
                        _=>{},
                    }
                }
                for &(output,id) in &alu.outputs {
                    if matches!(output,Output::MaskBit(_)) {blends.clear();}
                    if !matches!(output,Output::Scalar(r,Ty::I32) if !matches!(r,106|126|124)) {continue;}
                    match definitions.get(&id).copied().flatten() {
                        Some(Op::Int(IntOp::And,a,b))=>{
                            let observed=|v|ballots.get(&v).copied().filter(|bit|alu.inputs.iter().any(|(input,id)|
                                id==bit&&matches!(input.source,InputSource::MaskBit(106|126))));
                            let mask=|v|if let Some(bit)=observed(v) {Some((false,bit))} else {match definitions.get(&v).copied().flatten() {
                                Some(Op::Int(IntOp::Xor,word,all)) if matches!(definitions.get(&all),Some(Some(Op::Const(Ty::I32,0xffff_ffff))))=>{
                                    observed(word).map(|bit|(true,bit))
                                },
                                _=>None,
                            }};
                            if let Some((inverted,bit))=mask(b) {blends.insert(id,(inverted,a,bit));}
                            else if let Some((false,bit))=mask(a) {blends.insert(id,(false,b,bit));}
                        },
                        Some(Op::Int(IntOp::Or,a,b))=>{
                            if let (Some(&(ia,va,ma)),Some(&(ib,vb,mb)))=(blends.get(&a),blends.get(&b)) {
                                if ia!=ib&&ma==mb {
                                    let op=if ia {Op::Select(ma,vb,va)} else {Op::Select(ma,va,vb)};
                                    for inst in &mut block.insts[alu.core.clone()] {
                                        if let Inst::Core {value,op:old,..}=inst {if *value==id {*old=op;}}
                                    }
                                    definitions.insert(id,Some(op));
                                }
                            }
                        },
                        _=>{},
                    }
                }
            }
        }
    }
    pub fn value_yields(&self) -> BTreeMap<usize, crate::rdna_spmd::yield_values::YieldValues> {
        self.blocks.iter().filter_map(|(&pc, block)| {
            let plan = block.yield_values.as_ref()?;
            if plan.local { return None; }
            let Term::Br(edge) = &self.ir.func().blocks[&BlockId(pc)].term else { panic!("yield lacks resume edge") };
            Some((edge.dst.0, plan.layout.clone()))
        }).collect()
    }
    /// Preserve the ABI's cooperative yield while taking targets from typed CFG.
    pub fn terminator(&self, pc: usize) -> Control {
        let term = &self.ir.func().blocks[&BlockId(pc)].term;
        // Coalesce each parallel edge copy with the register-boundary slot:
        // every destination parameter and its incoming definition occupy the
        // same word slot. Check the mapping rather than silently ignoring SSA
        // arguments; an IR rewrite requiring moves must lower those explicitly.
        for edge in term.edges() {
            assert_eq!(
                edge.args[..self.blocks[&pc].outgoing.len()], self.blocks[&pc].outgoing,
                "unlowered SSA edge copies"
            );
        }
        match term {
            Term::Ret => Control::Return,
            Term::Br(e) if self.blocks[&pc].yielding => Control::Yield { resume: e.dst.0 },
            Term::Br(e) => Control::Jump(e.dst.0),
            Term::CondBr { cond, yes, no } => Control::Branch {
                cond: *cond,
                taken: yes.dst.0,
                fallthrough: no.dst.0,
            },
        }
    }
}
fn invalidate(views: &mut BTreeMap<(Word, Ty, bool), ValueId>, writes: &[Word]) {
    views.retain(|&(r, t, _), _| {
        !(0..t.bits().div_ceil(32)).filter_map(|k| r.offset(k)).any(|r| writes.contains(&r))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::ir::ScalarBlock;
    #[test]
    fn ordinary_alu_memory_and_wave_values_have_no_opaque_word_boundaries() {
        use crate::rdna_instructions::{VGLOBAL, VOP1};
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
            pc: 0, body: vec![
                InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, src0: SourceOperand::IntegerConstant(17), vdst: 2 }),
                InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_LOAD_B64, saddr: 0, vaddr: 2,
                    vsrc: 0, vdst: 4, scope: 0, th: 0, ioffset: 0, sve: 0 }),
                InstFormat::VOP1(VOP1 { op: I::V_READFIRSTLANE_B32, src0: SourceOperand::VectorRegister(4), vdst: 6 }),
                InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32, saddr: 0, vaddr: 2,
                    vsrc: 5, vdst: 0, scope: 0, th: 0, ioffset: 0, sve: 0 }),
            ], term: Terminator::Return,
        })]) };
        let lowerings: Vec<_> = program.blocks[&0].body.iter().map(super::super::instruction).collect();
        let f = Function::new(std::sync::Arc::new(DialectRegistry::rdna4()), &program,
            &BTreeMap::from([(0, lowerings.iter().collect())]), None,Preparation::Inspect);
        let block = &f.ir.func().blocks[&BlockId(0)];
        for inst in &block.insts {
            if let Inst::Boundary { outputs, .. } = inst {
                assert!(outputs.iter().all(|(_, ty)| *ty == Ty::I1), "opaque ordinary word: {inst:?}");
            }
        }
        let load = f.blocks[&0].memory[&1].effects[0];
        let Inst::Effect { outputs, .. } = &block.insts[load] else { unreachable!() };
        let loaded_word = outputs[0].0;
        let selected = block.insts.iter().find_map(|inst| match inst {
            Inst::Core { value, op: Op::Select(_, loaded, _), .. } if *loaded == loaded_word => Some(*value),
            _ => None,
        }).expect("loaded value must explicitly preserve inactive lanes");
        assert!(block.insts.iter().any(|inst| matches!(inst,
            Inst::Effect { op: effect::EffectOp::Wave(effect::WaveOp::ReadFirstLane), inputs, .. }
            if inputs[0] == selected)));
    }

    #[test]
    fn bvh_effect_keeps_third_components_and_exec_live_without_return_state() {
        use crate::rdna_instructions::{VIMAGE,VOP1,SOP1};
        let mut body:Vec<_>=[22,32,42,43].iter().map(|&reg|InstFormat::VOP1(VOP1 {
            op:I::V_MOV_B32,src0:SourceOperand::IntegerConstant(17),vdst:reg,
        })).collect();
        body.push(InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,ssrc0:SourceOperand::IntegerConstant(0),sdst:126}));
        body.push(InstFormat::VIMAGE(VIMAGE {op:I::IMAGE_BVH64_INTERSECT_RAY,
            dim:0,r128:0,d16:0,a16:0,dmask:15,vdata:50,rsrc:0,scope:0,th:0,tfe:0,
            vaddr0:10,vaddr1:15,vaddr2:20,vaddr3:30,vaddr4:40}));
        let program=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([(0,ScalarBlock {pc:0,body,term:Terminator::Return})])};
        let lowerings:Vec<_>=program.blocks[&0].body.iter().map(super::super::instruction).collect();
        let mut f=Function::new(std::sync::Arc::new(DialectRegistry::rdna4()),&program,
            &BTreeMap::from([(0,lowerings.iter().collect())]),None,Preparation::Inspect);
        f.discard_return_state();
        for index in [0usize,1,2,4] {
            assert!(f.blocks[&0].instructions[index].as_ref().unwrap().outputs.iter().all(|p|f.live[p.1.0]));
        }
        assert!(f.blocks[&0].instructions[3].as_ref().unwrap().outputs.iter().all(|p|!f.live[p.1.0]));
    }

    #[test]
    fn host_written_resume_values_are_new_edge_definitions() {
        let program = ScalarProgram {
            entry_pc: 0,
            blocks: BTreeMap::from([
                (
                    0,
                    ScalarBlock {
                        pc: 0,
                        body: vec![],
                        term: Terminator::Barrier { resume: 1 },
                    },
                ),
                (
                    1,
                    ScalarBlock {
                        pc: 1,
                        body: vec![],
                        term: Terminator::Return,
                    },
                ),
            ]),
        };
        let mut io = BoundaryIo::default();
        io.reads.add_vgpr(7);
        io.writes.add_vgpr(23);
        io.reads.add_sgpr(7);
        io.writes.add_sgpr(23);
        io.writes.scc = true;
        let boundary = BTreeMap::from([(1, io)]);
        let f = Function::new(
            std::sync::Arc::new(DialectRegistry::rdna4()),
            &program,
            &BTreeMap::from([(0, vec![]), (1, vec![])]),
            Some(&boundary),
            Preparation::Inspect,
        );
        let block = &f.ir.func().blocks[&BlockId(0)];
        let Inst::Boundary { inputs, outputs } = &block.insts[0] else {
            panic!("missing host boundary")
        };
        assert_eq!(inputs.len(), 5);
        assert_eq!(outputs.len(), 3);
        let edge = block.term.edges()[0];
        assert_eq!(edge.args[0], block.params[0].0);
        assert_eq!(edge.args[1], outputs[0].0);
        assert_ne!(edge.args[1], block.params[1].0);
        assert_eq!(edge.args[2], block.params[2].0);
        assert_eq!(edge.args[3], outputs[1].0);
        assert_ne!(edge.args[3], block.params[3].0);
        assert_eq!(*edge.args.last().unwrap(), outputs[2].0);
        assert_ne!(*edge.args.last().unwrap(), block.params.last().unwrap().0);
        assert_eq!(outputs[2].1, Ty::I1);
    }
}
