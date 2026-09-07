//! Owned typed SSA between decode/lift, preparation, CFG edits and codegen.
use std::collections::BTreeMap;
use std::sync::Arc;
use super::lift::function::{Function, LiftedFunction};
use super::ir::ScalarProgram;
use super::dialect::DialectRegistry;

/// Typed SSA prepared for reuse across execution widths and CFG fragments.
/// It owns no decoded instruction stream or register-CFG rewrite result.
#[derive(Clone)]
pub struct Program { pub(super) function: LiftedFunction }

/// Inputs accepted by the compiler. Decoded fixtures are lifted at this
/// boundary; an already prepared program retains its SSA graph.
pub trait CompilationInput {
    fn to_ssa(&self) -> Program;
}
impl<T: CompilationInput + ?Sized> CompilationInput for &T {
    fn to_ssa(&self) -> Program { (**self).to_ssa() }
}
impl CompilationInput for Program {
    fn to_ssa(&self) -> Program { self.clone() }
}
impl CompilationInput for ScalarProgram {
    fn to_ssa(&self) -> Program { Program::lift(self, Arc::new(DialectRegistry::rdna4())) }
}
impl Program {
    pub(super) fn lift(source: &ScalarProgram, registry: Arc<DialectRegistry>) -> Self {
        use super::lift::{Lowering,wave::YieldAction};
        use super::ir::typed::effect::EffectOp;
        use crate::{instructions::I,rdna_instructions::{InstFormat,SourceOperand}};
        // Resolve the combined barrier while decoding, before SSA exists. Its
        // signal/wait both retain the original source proof barrier and ID.
        let mut normalized=source.clone();
        let lowered:BTreeMap<_,Vec<_>>=normalized.blocks.iter_mut().map(|(&pc,b)| {
            let mut body=Vec::new();let mut lowerings=Vec::new();
            for inst in &b.body {
                if matches!(inst,InstFormat::SOPP(i) if matches!(i.op,I::S_BARRIER)) {
                    for op in [EffectOp::BarrierSignal {is_first:false},EffectOp::BarrierWait] {
                        body.push(inst.clone());
                        lowerings.push(Lowering::Wave(YieldAction::new(op,
                            vec![super::lift::wave::Operand::Source(SourceOperand::LiteralConstant(u32::MAX))],vec![])));
                    }
                }else{body.push(inst.clone());lowerings.push(super::lift::instruction_with_registry(inst,&registry));}
            }
            b.body=body;(pc,lowerings)
        }).collect();
        let refs=lowered.iter().map(|(&pc,b)|(pc,b.iter().collect())).collect();
        Self {function:Function::lift_raw(registry,&normalized,&refs)}
    }

    pub(super) fn decoded(source: &crate::rdna_translator::RDNAProgram, registry: Arc<DialectRegistry>) -> Self {
        let normalized = ScalarProgram { entry_pc:source.entry_pc(),blocks:source.blocks().iter().map(|(&pc,b)|
            (pc,super::ir::lower_block(pc,b.insts(),b.next_pcs()))).collect() };
        let mut program = Self::lift(&normalized,registry);
        let positions = source.blocks().iter().map(|(&pc,b)| {
            let mut index=0;
            let mut order=Vec::new();
            for (position,inst) in b.insts().iter().enumerate() {
                let last_term = position+1==b.insts().len() && matches!(inst,crate::rdna_instructions::InstFormat::SOPP(i)
                    if matches!(i.op,crate::instructions::I::S_ENDPGM|crate::instructions::I::S_BRANCH
                        |crate::instructions::I::S_CBRANCH_EXECZ|crate::instructions::I::S_CBRANCH_EXECNZ
                        |crate::instructions::I::S_CBRANCH_VCCZ|crate::instructions::I::S_CBRANCH_VCCNZ
                        |crate::instructions::I::S_CBRANCH_SCC0|crate::instructions::I::S_CBRANCH_SCC1));
                let combined_barrier=matches!(inst,crate::rdna_instructions::InstFormat::SOPP(i) if matches!(i.op,crate::instructions::I::S_BARRIER));
                if super::ir::is_noop(inst)||last_term||combined_barrier {
                    let words = program.function.words_before(pc,index);
                    order.push(super::lift::optimize::Position::Control(super::lift::rewrite::observation(inst,&words,&words)));
                    if combined_barrier {index+=2;}
                } else {
                    order.push(super::lift::optimize::Position::Instruction(index));index+=1;
                }
            }
            (pc,order)
        }).collect();
        program.function.optimize(positions);
        program
    }
}

impl Program {
    pub(super) fn boundary_io(&self) -> BTreeMap<usize,super::boundary::BoundaryIo> {
        self.function.blocks.iter().filter_map(|(&pc,b)| {
            let action=b.yield_action.as_ref()?;
            Some((self.function.ir.blocks[&super::ir::typed::cfg::BlockId(pc)].term.edges()[0].dst.0,action.io()))
        }).collect()
    }
    pub(super) fn vgpr_count(&self, declared: usize, boundary: &BTreeMap<usize,super::boundary::BoundaryIo>) -> usize {
        let needed=self.function.state.sites.values().flatten().flat_map(|s|s.writes.iter().map(|&(r,_)|r)
            .chain(s.math_reads.iter().map(|&(r,_)|r)))
            .chain(self.function.state.vector_parameters.iter().map(|&(r,_)|r))
            .chain(boundary.values().flat_map(|io|io.writes.vgprs())).max().map_or(1,|r|r as usize+1);
        declared.max(needed)
    }
}

impl Program {
    pub(super) fn fragment(&self, ranges: &BTreeMap<usize,(std::ops::Range<usize>,bool)>, entry:usize) -> Self {
        Self { function:self.function.fragment(ranges,entry) }
    }
    pub(super) fn rename_block(&mut self, old:usize, new:usize) {
        use super::ir::typed::cfg::BlockId;
        let f=&mut self.function;
        let block=f.ir.blocks.remove(&BlockId(old)).unwrap();f.ir.blocks.insert(BlockId(new),block);
        let plan=f.blocks.remove(&old).unwrap();f.blocks.insert(new,plan);
        if f.ir.entry.0==old {f.ir.entry=BlockId(new);}
        let g=&mut f.state;
        if let Some(v)=g.sites.remove(&old) {g.sites.insert(new,v);}
        if let Some(v)=g.outgoing.remove(&old) {g.outgoing.insert(new,v);}
        if let Some(v)=g.scalar_outgoing.remove(&old) {g.scalar_outgoing.insert(new,v);}
        if let Some(v)=g.conditions.remove(&old) {g.conditions.insert(new,v);}
        for set in [&mut g.yielding,&mut g.barriers,&mut g.yield_exec_writes] {if set.remove(&old) {set.insert(new);}}
        g.exec_edges=std::mem::take(&mut g.exec_edges).into_iter().map(|((a,b),v)|((if a==old{new}else{a},b),v)).collect();
    }
    /// Preserve the segmented post-fragment's existing EXEC guard.
    pub(super) fn guard_fragment(&mut self) {
        use super::ir::typed::{cfg::*,*};
        use super::lift::{Input,InputSource,function::{BlockPlan,Condition}};
        let old=self.function.ir.entry.0;
        self.rename_block(old,1);
        let f=&mut self.function;
        for pc in [0,2] {
            let params:Vec<_>=f.parameter_inputs.clone().iter().map(|i|(f.ir.value(i.ty),i.ty)).collect();
            let outgoing=params.iter().map(|p|p.0).collect::<Vec<_>>();
            f.state.sites.insert(pc,vec![]);
            f.state.outgoing.insert(pc,f.state.vector_parameters.iter().map(|&(r,i)|(r,outgoing[i])).collect());
            f.state.scalar_outgoing.insert(pc,f.state.scalar_parameters.iter().map(|&(r,i)|(r,outgoing[i])).collect());
            let mut block=Block {params,insts:vec![],term:Term::Ret};
            let condition=if pc==0 {
                let input_value=outgoing[f.state.scalar_parameters.iter().find(|p|p.0==126).unwrap().1];
                let query=f.ir.value(Ty::I1);let zero=f.ir.value(Ty::I1);let cond=f.ir.value(Ty::I1);
                block.insts=vec![Inst::Packet {op:PacketOp::Any,input:input_value,output:query},
                    Inst::Core {value:zero,ty:Ty::I1,op:Op::Const(Ty::I1,0)},
                    Inst::Core {value:cond,ty:Ty::I1,op:Op::Cmp(IntPred::Eq,query,zero)}];
                block.term=Term::CondBr {cond,yes:Edge {dst:BlockId(2),args:outgoing.clone()},no:Edge {dst:BlockId(1),args:outgoing.clone()}};
                Some(Condition {input:Input {source:InputSource::MaskBit(126),ty:Ty::I1},input_value,core:0..3})
            }else{None};
            f.ir.blocks.insert(BlockId(pc),block);
            f.blocks.insert(pc,BlockPlan {instructions:vec![],memory:Default::default(),wave:Default::default(),outgoing,
                yield_values:None,yield_action:None,yielding:false,condition});
        }
        f.ir.entry=BlockId(0);
        f.state.conditions.insert(0,super::ir::Cond::ExecZ);
        f.state.exec_edges.insert((0,2),false);f.state.exec_edges.insert((0,1),true);
    }
}

impl Program {
    pub(super) fn split(&self, accept:impl Fn(&super::lift::wave::YieldAction)->bool) -> (Self,BTreeMap<usize,super::lift::wave::YieldAction>) {
        use super::ir::typed::cfg::{BlockId,Term,Edge};
        let mut next=self.function.blocks.keys().max().copied().unwrap_or(0)+1;
        let mut fragments=Vec::new();let mut yields=BTreeMap::new();
        for (&original,b) in &self.function.blocks {
            let mut pc=original;let mut start=0;
            for (&index,(action,_)) in &b.wave {
                if !accept(action) {continue;}
                let resume=next;next+=1;
                let mut fragment=self.fragment(&BTreeMap::from([(original,(start..index+1,false))]),original);
                if pc!=original {fragment.rename_block(original,pc);}
                let f=&mut fragment.function;
                let last=f.blocks[&pc].instructions.len()-1;
                let before=f.words_before(pc,last);
                let b=f.blocks.get_mut(&pc).unwrap();
                let (action,plan)=b.wave.remove(&last).unwrap();
                let io=action.io();
                f.state.wave_reads.extend(io.reads.vgprs().map(|r|before[&super::lift::state::Word::Vgpr(r)]));
                for &(dest,value) in &plan.definitions {
                    if let super::lift::wave::Destination::Vgpr(r)=dest {
                        f.state.resume_observations.push((value,before[&super::lift::state::Word::Vgpr(r)]));
                    }
                }
                f.state.scalar_outgoing.insert(pc,f.state.scalar_parameters.iter().map(|&(r,_)|(r,before[&super::lift::state::Word::scalar(r).unwrap()])).collect());
                f.state.outgoing.insert(pc,f.state.vector_parameters.iter().map(|&(r,_)|(r,before[&super::lift::state::Word::Vgpr(r)])).collect());
                b.instructions.pop();f.state.sites.get_mut(&pc).unwrap().pop();
                b.yield_values=Some(plan);b.yield_action=Some(action.clone());b.yielding=true;
                f.state.yielding.insert(pc);if io.writes.has_sgpr(126) {f.state.yield_exec_writes.insert(pc);}
                f.ir.blocks.get_mut(&BlockId(pc)).unwrap().term=Term::Br(Edge {dst:BlockId(resume),args:b.outgoing.clone()});
                yields.insert(resume,action);fragments.push(fragment);
                pc=resume;start=index+1;
            }
            let mut tail=self.fragment(&BTreeMap::from([(original,(start..b.instructions.len(),true))]),original);
            if pc!=original {tail.rename_block(original,pc);}
            fragments.push(tail);
        }
        let mut iter=fragments.into_iter();let mut program=iter.next().unwrap();
        for mut next in iter {
            let offset=program.function.ir.types.len();
            next.function.map_values(|v|super::ir::typed::ValueId(v.0+offset),true);
            let a=&mut program.function;let b=next.function;
            a.ir.types.extend(b.ir.types);a.ir.blocks.extend(b.ir.blocks);a.blocks.extend(b.blocks);
            a.state.sites.extend(b.state.sites);a.state.outgoing.extend(b.state.outgoing);a.state.scalar_outgoing.extend(b.state.scalar_outgoing);
            a.state.conditions.extend(b.state.conditions);a.state.exec_edges.extend(b.state.exec_edges);
            a.state.yielding.extend(b.state.yielding);a.state.barriers.extend(b.state.barriers);a.state.yield_exec_writes.extend(b.state.yield_exec_writes);
            a.state.wave_reads.extend(b.state.wave_reads);a.state.resume_observations.extend(b.state.resume_observations);
        }
        program.function.ir.entry=self.function.ir.entry;
        (program,yields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instructions::I,rdna_instructions::{InstFormat,SOP1,SOPP,SOPC,SourceOperand}};
    use crate::rdna_spmd::ir::{ScalarBlock,Terminator,Cond};
    use crate::rdna_spmd::ir::typed::{cfg::*,effect::EffectOp};
    fn program(body:Vec<InstFormat>,term:Terminator)->ScalarProgram {
        ScalarProgram {entry_pc:1,blocks:BTreeMap::from([(1,ScalarBlock {pc:1,body,term})])}
    }
    fn verify(p:&Program) {p.function.ir.clone().verify_with(&p.function.registry).unwrap();}
    #[test]
    fn combined_barrier_splits_into_signal_wait_and_preserves_loop_backedge() {
        let source=program(vec![InstFormat::SOPP(SOPP {op:I::S_BARRIER,simm16:0})],Terminator::Jump(1));
        let (p,actions)=source.to_ssa().split(|_|true);
        assert_eq!(source.blocks[&1].body.len(),1);
        assert_eq!(p.function.blocks.len(),3);
        assert!(matches!(actions[&2].op,EffectOp::BarrierSignal {is_first:false}));
        assert_eq!(actions[&3].op,EffectOp::BarrierWait);
        for action in actions.values() {assert!(matches!(&action.inputs[0],super::super::lift::wave::Operand::Source(SourceOperand::LiteralConstant(u32::MAX))));}
        assert!(matches!(&p.function.ir.blocks[&BlockId(3)].term,Term::Br(e) if e.dst==BlockId(1)));
        verify(&p);
    }
    #[test]
    fn continuation_passes_scc_after_its_compare_operand_is_overwritten() {
        let mut source=program(vec![
            InstFormat::SOPC(SOPC {op:I::S_CMP_LG_U32,ssrc0:SourceOperand::ScalarRegister(2),ssrc1:SourceOperand::IntegerConstant(0)}),
            InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst:2,ssrc0:SourceOperand::IntegerConstant(0)}),
            InstFormat::SOPP(SOPP {op:I::S_BARRIER_WAIT,simm16:3}),
        ],Terminator::Branch {cond:Cond::Scc1,taken:2,fallthrough:3});
        for pc in [2,3] {source.blocks.insert(pc,ScalarBlock {pc,body:vec![],term:Terminator::Return});}
        let p=source.to_ssa().split(|_|true).0;
        let resume=p.function.ir.blocks[&BlockId(1)].term.edges()[0].dst;
        let b=&p.function.ir.blocks[&resume];
        assert_eq!(p.function.blocks[&resume.0].condition.as_ref().unwrap().input_value,b.params.last().unwrap().0);
        verify(&p);
    }
    #[test]
    fn splitting_again_preserves_existing_yield_values_and_memory_effects() {
        use crate::rdna_instructions::{VSCRATCH,VOP3};
        let load=|dst|InstFormat::VSCRATCH(VSCRATCH {op:I::SCRATCH_LOAD_B32,vaddr:0,vsrc:0,vdst:dst,scope:0,th:0,ioffset:0,saddr:124,sve:0});
        let source=program(vec![load(2),InstFormat::SOPP(SOPP {op:I::S_BARRIER_WAIT,simm16:0}),
            InstFormat::VOP3(VOP3 {op:I::V_WRITELANE_B32,vdst:2,src0:SourceOperand::IntegerConstant(7),src1:SourceOperand::IntegerConstant(0),
                src2:SourceOperand::IntegerConstant(0),neg:0,abs:0,cm:0,omod:0,opsel:0}),load(3)],Terminator::Return);
        let barrier=source.to_ssa().split(|a|!a.is_wave()).0;
        let p=barrier.split(|a|a.is_wave()).0;
        assert_eq!(p.function.blocks.values().filter(|b|b.yield_values.is_some()).count(),2);
        let effects:Vec<_>=p.function.ir.blocks.values().flat_map(|b|&b.insts).filter_map(|i|if let Inst::Effect {provenance,op,..}=i {Some((*provenance,*op))}else{None}).collect();
        assert_eq!(effects.iter().filter(|(_,op)|matches!(op,EffectOp::Memory {..})).count(),2);
        verify(&p);
    }
}
