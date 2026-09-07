//! Whole-function SSA construction from lifted instruction semantics.
//! Register words become block parameters and explicit definitions; effects
//! define their results before outgoing edges are constructed.
use super::*;
use crate::rdna_spmd::ir::typed::cfg::*;
use crate::rdna_spmd::ir::{ScalarProgram, Terminator};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use super::state::{Word, Words, footprint, words as register_words};

#[derive(Clone)]
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
#[derive(Clone)]
pub(in crate::rdna_spmd) struct BlockPlan {
    pub instructions: Vec<Option<Alu>>,
    pub memory: BTreeMap<usize, super::memory::Plan>,
    pub wave: BTreeMap<usize,(super::wave::YieldAction,super::wave::Plan)>,
    pub outgoing: Vec<ValueId>,
    pub yield_values: Option<super::wave::Plan>,
    pub condition: Option<Condition>,
    pub(in crate::rdna_spmd) yielding: bool,
    pub(in crate::rdna_spmd) yield_action: Option<super::wave::YieldAction>,
}
#[derive(Clone)]
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
    pub native_live: Vec<bool>,
    pub retained: Vec<bool>,
    pub observable_return: bool,
}
/// Typed SSA before native representation and predication decisions are applied.
#[derive(Clone)]
pub(in crate::rdna_spmd) struct LiftedFunction {
    pub registry: std::sync::Arc<DialectRegistry>,
    pub written: crate::rdna_spmd::boundary::RegSet,
    pub ir: Func,
    pub blocks: BTreeMap<usize, BlockPlan>,
    pub parameter_inputs: Vec<Input>,
    pub state: crate::rdna_spmd::analysis::state::StateGraph,
    pub revision: u64,
}
impl Function {
    #[cfg(test)]
    pub fn new(
        registry: std::sync::Arc<DialectRegistry>,
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering>>,
        preparation: Preparation,
    ) -> Self {
        Self::lift(registry, program, lowerings).prepare(preparation)
    }
    pub fn lift(
        registry: std::sync::Arc<DialectRegistry>,
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering>>,
    ) -> LiftedFunction {
        let mut lifted = Self::lift_raw(registry, program, lowerings);
        lifted.prepare_queries();
        lifted
    }
    /// Input rewrites need the original SSA definitions only. Native query and
    /// scheduler-layout preparation is performed once by `lift` for compilation.
    pub fn lift_raw(
        registry: std::sync::Arc<DialectRegistry>,
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering>>,
    ) -> LiftedFunction {
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
        let mut state = crate::rdna_spmd::analysis::state::StateGraph::default();
        state.vector_parameters = regs.iter().enumerate().filter_map(|(index, word)|
            if let Word::Vgpr(slot) = word { Some((*slot, index)) } else { None }).collect();
        state.scalar_parameters = regs.iter().enumerate().filter_map(|(index, word)|
            match word { Word::Sgpr(r) | Word::Mask(r) => Some((*r, index)), _ => None }).collect();
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
            let mut sites = Vec::new();
            for (index, lowering) in lowerings[&pc].iter().enumerate() {
                let source_inst = &source.body[index];
                let previous_words = words.clone();
                let reads = super::access::vgpr_reads(source_inst).iter()
                    .map(|&r| words[&Word::Vgpr(r)]).collect();
                let varying_inputs = super::access::div_reads(source_inst).iter()
                    .map(|&r| words[&Word::Vgpr(r)]).collect();
                let address = if let Lowering::Memory(memory) = lowering {
                    match memory.address {
                        super::memory::Address::Global { vector, scalar, .. } => {
                            let mut values = vec![words[&Word::Vgpr(vector)]];
                            if scalar.is_none() { values.push(words[&Word::Vgpr(vector+1)]); }
                            values
                        },
                        _ => Vec::new(),
                    }
                } else { Vec::new() };
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
                        Some(alu(&registry, &mut f, &mut block, &mut words, &mut views, &mut scc,
                            &mut provenance, inputs, outputs, scalar, expr, &writes))
                    }

                };
                sites.push(crate::rdna_spmd::analysis::state::Site {
                    rewrite: super::rewrite::observation(source_inst, &previous_words, &words),
                    math_reads: super::access::math_reads(source_inst).into_iter().map(|r| (r, previous_words[&Word::Vgpr(r)])).collect(),
                    sqrt: super::access::sqrt_policy(source_inst, &previous_words, &words),
                    masks: super::control::mask_policy(source_inst, &previous_words, &words),
                    f64_definitions: super::access::f64_defs(source_inst).into_iter().filter_map(|r|
                        Some((*words.get(&Word::Vgpr(r))?, *words.get(&Word::Vgpr(r+1))?))).collect(),
                    u64_definitions: super::access::sgpr_u64_defs(source_inst).into_iter().filter_map(|r|
                        Some((*words.get(&Word::scalar(r)?)?, *words.get(&Word::scalar(r+1)?)?))).collect(),
                    f64_uses: super::access::f64_pairs(source_inst).into_iter().map(|r|
                        (r, words[&Word::Vgpr(r)], words[&Word::Vgpr(r+1)])).collect(),
                    exec: super::control::policy(source_inst, &previous_words, &words),
                    reads,
                    varying_inputs,
                    intrinsically_varying: super::access::uses_private(source_inst),
                    address,
                    frame: super::access::frame_def(source_inst).map(|(r, stride)|
                        (r, words[&Word::Vgpr(r)], words[&Word::Vgpr(r+1)], stride)),
                    writes: writes.iter().filter_map(|word| if let Word::Vgpr(r) = word { Some((*r, words[word])) } else { None }).collect(),
                    reactivation: if super::control::may_enable_lanes(source_inst) {
                        words.iter().filter_map(|(word, &value)| if let Word::Vgpr(r) = word {Some((*r, value))} else {None}).collect()
                    } else { Vec::new() },
                });
                instructions.push(plan);
            }
            if matches!(source.term, Terminator::Barrier { .. }) { state.barriers.insert(pc); }
            state.sites.insert(pc, sites);
            state.scalar_outgoing.insert(pc, words.iter().filter_map(|(word, &value)|
                match word { Word::Sgpr(r) | Word::Mask(r) => Some((*r, value)), _ => None }).collect());
            if let Terminator::Branch { cond, taken, fallthrough } = source.term {
                state.conditions.insert(pc, cond);
                use crate::rdna_spmd::ir::Cond;
                if matches!(cond, Cond::ExecZ | Cond::ExecNz) {
                    state.exec_edges.insert((pc, taken), cond == Cond::ExecNz);
                    state.exec_edges.insert((pc, fallthrough), cond == Cond::ExecZ);
                }
            }
            state.outgoing.insert(pc, words.iter().filter_map(|(word, &value)|
                if let Word::Vgpr(r) = word { Some((*r, value)) } else { None }).collect());
            if matches!(source.term, Terminator::Yield { .. } | Terminator::Barrier { .. }) { state.yielding.insert(pc); }
            let yield_values = if let Terminator::Yield { ref action, .. } = source.term {
                let io = action.io();
                if io.writes.has_sgpr(126) { state.yield_exec_writes.insert(pc); }
                state.wave_reads.extend(io.reads.vgprs().map(|r| words[&Word::Vgpr(r)]));
                let previous: Vec<_> = io.writes.vgprs().map(|r| (r, words[&Word::Vgpr(r)])).collect();
                let plan = action.lift(&mut f, &mut block, &mut words, &mut provenance);
                for (r, old) in previous { state.resume_observations.push((words[&Word::Vgpr(r)], old)); }
                for &(dest, value) in &plan.definitions {
                    if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                }
                Some(plan)
            } else { None };

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
                    state.conditions.insert(pc, cond);
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
                    yield_action: if let Terminator::Yield { action, .. } = &source.term { Some(*action.clone()) } else { None },
                    yielding: matches!(source.term, Terminator::Yield { .. } | Terminator::Barrier { .. }),
                    outgoing: regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect(),
                },
            );
        }
        let parameter_inputs:Vec<_>=regs.iter().map(|word|Input {ty:word.ty(),source:match *word {
            Word::Vgpr(r)=>InputSource::Operand(SourceOperand::VectorRegister(r as u8)),
            Word::Sgpr(r)=>InputSource::Operand(SourceOperand::ScalarRegister(r as u8)),
            Word::Mask(r)=>InputSource::MaskBit(r),
        }}).chain(std::iter::once(Input {ty:Ty::I1,source:InputSource::Scc})).collect();
        LiftedFunction { registry, written, ir: f, blocks: plans, parameter_inputs, state, revision: 0 }
    }
    /// Apply the plan's proof that these temporary destinations cannot be
    /// observed by a reactivated lane or an unpredicated wave consumer.
    pub(in crate::rdna_spmd) fn elide_inactive_updates(blocks:&BTreeMap<usize,BlockPlan>,f:&mut Func,sites:impl IntoIterator<Item=(usize,usize)>) {
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
    pub(in crate::rdna_spmd) fn elide_active_updates(blocks:&BTreeMap<usize,BlockPlan>,f:&mut Func,sites:impl IntoIterator<Item=(usize,usize)>) {
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
    pub(in crate::rdna_spmd) fn assume_dispatch_exec(parameter_inputs:&[Input],f:&mut Func) {
        let index=parameter_inputs.iter().position(|p|matches!(p.source,InputSource::MaskBit(126))).unwrap();
        let exec=f.blocks[&f.entry].params[index].0;
        crate::rdna_spmd::passes::constant_queries(f,&[exec]);
    }
    /// Ordinary dispatch exposes memory effects, not the final register bank.
    #[cfg(test)]
    pub fn discard_return_state(&mut self) -> Vec<bool> {
        self.observable_return = false;
        let roots = self.blocks.values().filter_map(|b| b.yield_values.as_ref())
            .flat_map(|p| p.results.iter().map(|r| r.0));
        let live = crate::rdna_spmd::analysis::live_values(self.ir.func(), roots);
        self.native_live = crate::rdna_spmd::analysis::native_live(self.ir.func(), &self.blocks, &live);
        live
    }
    pub fn min_private_bytes(&self) -> usize {
        self.blocks.values().flat_map(|b|b.memory.values())
            .filter_map(|p|p.memory.private_load_end()).max().unwrap_or(0) as usize
    }
    /// The existing whole-program ABI keeps mask words within one packet.
    /// Explicit cross-lane instructions retain their separately selected scope.
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
    /// Preserve the scalar backend's existing de-SIMT mask blend rule. Its
    /// two masked SGPR definitions represent whole scalar values, and their
    /// final OR selects a value using the single simulated lane's mask bit.
    /// This is specific to the ordinary scalar ABI, not a wave-word identity.
    pub(in crate::rdna_spmd) fn scalar_mask_selects(blocks:&BTreeMap<usize,BlockPlan>,f:&mut Func) {
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
            &BTreeMap::from([(0, lowerings.iter().collect())]), Preparation::Inspect);
        let block = &f.ir.func().blocks[&BlockId(0)];
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
            &BTreeMap::from([(0,lowerings.iter().collect())]),Preparation::Inspect);
        let live=f.discard_return_state();
        for index in [0usize,1,2,4] {
            assert!(f.blocks[&0].instructions[index].as_ref().unwrap().outputs.iter().all(|p|live[p.1.0]));
        }
        assert!(f.blocks[&0].instructions[3].as_ref().unwrap().outputs.iter().all(|p|!live[p.1.0]));
    }


    #[test]
    fn yield_results_are_new_edge_definitions() {
        use super::super::wave::{YieldAction, Operand, Destination};
        use effect::{EffectOp, WaveOp};
        for destination in [Destination::Sgpr(23), Destination::Vgpr(23), Destination::Scc] {
            let action = if matches!(destination, Destination::Scc) {
                YieldAction::new(EffectOp::BarrierSignal { is_first: true },
                    vec![Operand::Source(SourceOperand::ScalarRegister(7))], vec![destination])
            } else {
                YieldAction::new(EffectOp::Wave(WaveOp::ReadLane),
                    vec![Operand::Source(SourceOperand::VectorRegister(7)),
                        Operand::Source(SourceOperand::ScalarRegister(7))], vec![destination])
            };
            let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([
                (0, ScalarBlock { pc: 0, body: vec![], term: Terminator::Yield { resume: 1, action: Box::new(action) } }),
                (1, ScalarBlock { pc: 1, body: vec![], term: Terminator::Return }),
            ]) };
            let f = Function::new(std::sync::Arc::new(DialectRegistry::rdna4()), &program,
                &BTreeMap::from([(0, vec![]), (1, vec![])]), Preparation::Inspect);
            let block = &f.ir.func().blocks[&BlockId(0)];
            let plan = f.blocks[&0].yield_values.as_ref().unwrap();
            let definition = plan.definitions[0].1;
            let edge = block.term.edges()[0];
            let slot = edge.args.iter().position(|&value| value == definition).expect("yield definition must reach resume");
            assert_ne!(edge.args[slot], block.params[slot].0);
            assert!(block.insts.iter().any(|inst| matches!(inst, Inst::Effect { outputs, .. } if outputs == &plan.results)));
        }
    }
}

impl crate::rdna_spmd::passes::Program for LiftedFunction {
    type Snapshot = (Func, Vec<usize>);
    fn snapshot(&self) -> Self::Snapshot { (self.ir.clone(), self.state.sites.values().map(|s| s.len()).collect()) }
    fn ir(&self) -> &Func { &self.ir }
    fn registry(&self) -> &DialectRegistry { &self.registry }
    fn touch(&mut self) { self.revision += 1; }
}
impl LiftedFunction {
    pub(in crate::rdna_spmd) fn prepare_queries(&mut self) {
        // Removed sites no longer assign scheduler state. Derive the writeback
        // footprint from the surviving SSA definitions and effect results.
        let mut written=crate::rdna_spmd::boundary::RegSet::default();
        for site in self.state.sites.values().flatten() {
            for &(slot,_) in &site.rewrite.defined_words {
                if slot>=512 {written.add_vgpr(slot-512);}else if slot!=124 {written.add_sgpr(slot);}
            }
        }
        for b in self.blocks.values() {
            written.scc|=b.instructions.iter().flatten().any(|a|a.outputs.iter().any(|p|matches!(p.0,Output::Scc)));
            for action in b.wave.values().map(|p|&p.0).chain(b.yield_action.iter()) {
                let io=action.io();
                for r in io.writes.sgprs() {if r!=124 {written.add_sgpr(r);}}
                for r in io.writes.vgprs() {written.add_vgpr(r);}
                written.scc|=io.writes.scc;
            }
        }
        self.written=written;
        crate::rdna_spmd::compiler::query_passes(self);
    }

    /// Expand proven normal scales in place, preserving every SSA result ID.
    /// Instruction intervals are remapped together with their native bindings.
    pub fn fold_normal_scales(&mut self, pc: usize, normal: &[bool]) {
        let block = self.ir.blocks.get_mut(&BlockId(pc)).unwrap();
        let mut selected = vec![false; block.insts.len()];
        for (index, alu) in self.blocks[&pc].instructions.iter().enumerate() {
            if normal[index] && matches!(self.state.sites[&pc][index].sqrt.shape,
                crate::rdna_spmd::sqrt_idiom::Shape::Scale { exponent_unmodified: true, .. }) {
                if let Some(alu) = alu { selected[alu.core.clone()].fill(true); }
            }
        }
        if !selected.iter().any(|&yes| yes) { return; }
        let old = std::mem::take(&mut block.insts);
        let mut positions = Vec::with_capacity(old.len()+1);
        let mut insts = Vec::new();
        for (index, inst) in old.into_iter().enumerate() {
            positions.push(insts.len());
            if selected[index] { insts.extend(crate::rdna_spmd::dialect::rdna4::fold_normal(&mut self.ir, inst, &self.registry)); }
            else { insts.push(inst); }
        }
        positions.push(insts.len());
        self.ir.blocks.get_mut(&BlockId(pc)).unwrap().insts = insts;
        self.remap_positions(pc, &positions);
    }

    pub(in crate::rdna_spmd) fn remap_positions(&mut self, pc: usize, positions: &[usize]) {
        self.revision += 1;
        let range = |r: &mut Range<usize>| { *r = positions[r.start]..positions[r.end]; };
        let block = self.blocks.get_mut(&pc).unwrap();
        for alu in block.instructions.iter_mut().flatten() {
            range(&mut alu.core); range(&mut alu.previous_core); alu.updates_start = positions[alu.updates_start];
        }
        for memory in block.memory.values_mut() {
            range(&mut memory.core); memory.end = positions[memory.end];
            for effect in &mut memory.effects { *effect = positions[*effect]; }
            if let Some((ref mut r, ..)) = memory.flat { range(r); }
        }
        for (_, wave) in block.wave.values_mut() { range(&mut wave.core); wave.end = positions[wave.end]; }
        if let Some(wave) = &mut block.yield_values { range(&mut wave.core); wave.end = positions[wave.end]; }
        if let Some(condition) = &mut block.condition { range(&mut condition.core); }
    }

    pub fn prepare(mut self, preparation: Preparation) -> Function {
        let (observable_return,scalar_live)=crate::rdna_spmd::compiler::preparation_passes(&mut self,preparation);
        let Self { registry, written, ir: f, blocks: plans, parameter_inputs: _, state: _, revision: _ } = self;
        let live=scalar_live.unwrap_or_else(||{
            let mut roots=vec![];
            for (&pc,plan) in &plans {
                if let Some(yielding)=&plan.yield_values {roots.extend(yielding.results.iter().map(|p|p.0));}
                if observable_return&&matches!(f.blocks[&BlockId(pc)].term,Term::Ret) {roots.extend(&plan.outgoing);}
            }
            crate::rdna_spmd::analysis::live_values(&f,roots)
        });
        let ir=f.verify_with(&registry).expect("invalid prepared function SSA");
        let native_live=crate::rdna_spmd::analysis::native_live(ir.func(),&plans,&live);
        let retained=crate::rdna_spmd::analysis::retained(ir.func(),&plans);
        Function {
            ir,
            native_live,
            retained,
            observable_return,
            registry,
            written,
            blocks: plans,
        }
    }
}

/// Instantiate a replacement ALU expression in an existing SSA value namespace.
/// Used by both initial lifting and typed instruction rewrites.
pub(super) fn alu(
    registry: &DialectRegistry, f: &mut Func, block: &mut Block,
    words: &mut Words, views: &mut super::state::Views, scc: &mut ValueId,
    provenance: &mut u64, inputs: &[Input], outputs: &[Output], scalar: &bool,
    expr: &VerifiedExpr, writes: &[Word],
) -> Alu {
    let first_value = f.types.len();
                        let mut operands = super::state::Operands::default();
                        let args: Vec<_> = inputs.iter().map(|input| operands.read(
                            input, *scalar, Some(*scc), f, block, words, views,
                        )).collect();
                        let source_inputs=operands.bindings.len();
                        let previous_start=operands.core.len();
                        // A destination's old value is an update dependency,
                        // not a source view available to later instructions.
                        let mut previous_views=views.clone();
                        let previous:Vec<_>=outputs.iter().map(|output|match *output {
                            Output::Vgpr(reg,ty)=>{
                                let input=Input {source:InputSource::Operand(SourceOperand::VectorRegister(reg as u8)),ty};
                                let value=operands.read(&input,*scalar,Some(*scc),f,block,words,&mut previous_views);
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
                                        else { let id = *provenance; *provenance += 1; Some(id) };
                                    block.insts.push(Inst::Target { provenance: id, op: *op, args, outputs });
                                }
                            }
                        }
                        let updates_start=block.insts.len();
                        let results: Vec<_> = outputs.iter().copied()
                            .zip(expr.expr().results.iter().map(|id| values[id.0])).collect();
                        for &(output, value) in &results {
                            if matches!(output, Output::Scc) { *scc = if *scalar { value } else {
                                super::state::query(f,&mut block.insts,effect::WaveOp::Any,value)
                            }; }
                        }
                        let exec=super::state::core(f,&mut block.insts,Ty::I1,
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
                                    super::state::core(f,&mut block.insts,Ty::I1,
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
                                        else { super::state::query(f,&mut block.insts,effect::WaveOp::Ballot,value) };
                                    let raw=if word==Word::Mask(126) {super::state::valid_exec(f,&mut block.insts,raw)} else {raw};
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
                                super::state::core(f,&mut block.insts,int_ty,Op::Convert(Cvt::Bitcast,int_ty,value))
                            } else { value };
                            for k in 0..ty.bits().div_ceil(32) {
                                let word = if scalar { Word::scalar(reg+k) } else { Some(Word::Vgpr(reg+k)) };
                                if let Some(word) = word {
                                    let raw = if ty.bits() == 32 { bits } else {
                                        super::state::core(f,&mut block.insts,Ty::I32,
                                            if k==0 { Op::UnpackLo(bits) } else { Op::UnpackHi(bits) })
                                    };
                                    let raw = if matches!(word,Word::Mask(_)) {
                                        super::state::project(f,&mut block.insts,raw)
                                    } else { raw };
                                    let raw=if word==Word::Mask(126) {super::state::valid_exec(f,&mut block.insts,raw)} else {raw};
                                    if let Word::Mask(reg)=word {mask_updates.push((raw,reg));}
                                    word_defs.insert(word,raw);
                                }
                            }
                        }
                        // Preserve a distinct architectural definition for a
                        // scalar copy. Native lowering coalesces the identity;
                        // definition-scoped proof restrictions remain explicit.
                        for (word, raw) in &mut word_defs {
                            if matches!(word, Word::Sgpr(_)) && raw.0 < first_value {
                                *raw = super::state::core(f, &mut block.insts, Ty::I32,
                                    Op::Convert(Cvt::Bitcast, Ty::I32, *raw));
                            }
                        }
                        for &r in writes {
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
                                Output::Scc => emitted.push((output,*scc)),
                                Output::Vgpr(..) => emitted.push((output,stored[output_index].0)),
                                _ => emitted.push((output,result)),
                            }
                        }
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
                        Alu {
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
                        }
}
