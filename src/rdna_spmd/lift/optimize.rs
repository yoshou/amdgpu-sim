//! Existing preparation passes applied to the owned SSA graph.
use super::*;
use super::function::LiftedFunction;
use super::state::{Word, Words};
use crate::rdna_spmd::ir::typed::cfg::*;
use crate::rdna_spmd::analysis::rewrite::Observation;
use std::collections::BTreeMap;

impl LiftedFunction {
    pub(in crate::rdna_spmd) fn words_before(&self, pc: usize, index: usize) -> Words {
        let mut words: Words = self.state.vector_parameters.iter().map(|&(r,i)| (Word::Vgpr(r),self.ir.blocks[&BlockId(pc)].params[i].0))
            .chain(self.state.scalar_parameters.iter().map(|&(r,i)| (Word::scalar(r).unwrap(),self.ir.blocks[&BlockId(pc)].params[i].0))).collect();
        for s in &self.state.sites[&pc][..index] {
            for &(slot,v) in &s.rewrite.defined_words { if let Some(w) = super::rewrite::word(slot) { words.insert(w,v); } }
        }
        words
    }
    pub(super) fn substitute(&mut self, replacements: &BTreeMap<ValueId,ValueId>) {
        self.map_values(|mut v| { while let Some(&next) = replacements.get(&v) { assert_ne!(v,next); v = next; } v }, false);
    }
    /// Bypass architectural definitions of proven-dead sites, including the
    /// predicated pair views used by subsequent native consumers.
    pub(super) fn remove_sites(&mut self, pc: usize, remove: &[bool]) {
        let mut replacements = BTreeMap::new();
        let mut words = self.words_before(pc,0);
        for (index,site) in self.state.sites[&pc].iter().enumerate() {
            if remove[index] {
                for &(slot,value) in &site.rewrite.defined_words {
                    if let Some(w) = super::rewrite::word(slot) { if words[&w] != value { replacements.insert(value,words[&w]); } }
                }
                if let Some(a) = &self.blocks[&pc].instructions[index] {
                    for &(value,_) in &a.predicated {
                        if let Some(old) = self.ir.blocks[&BlockId(pc)].insts[a.core.clone()].iter().find_map(|i|
                            match i { Inst::Core { value: v, op: Op::Select(_,_,old), .. } if *v == value => Some(*old), _=>None }) {
                            if value != old { replacements.insert(value,old); }
                        }
                    }
                }
            }
            for &(slot,v) in &site.rewrite.defined_words { if let Some(w) = super::rewrite::word(slot) { words.insert(w,v); } }
        }
        self.substitute(&replacements);
        let mut next = 0;
        let indices: Vec<_> = remove.iter().map(|&r| if r { None } else { let i=next;next+=1;Some(i) }).collect();
        let retain = |i: &mut usize| { let keep = !remove[*i]; *i += 1; keep };
        let mut i=0; self.state.sites.get_mut(&pc).unwrap().retain(|_|retain(&mut i));
        let b = self.blocks.get_mut(&pc).unwrap();
        let mut i=0; b.instructions.retain(|_|retain(&mut i));
        b.memory = std::mem::take(&mut b.memory).into_iter().filter_map(|(i,p)|indices[i].map(|i|(i,p))).collect();
        b.wave = std::mem::take(&mut b.wave).into_iter().filter_map(|(i,p)|indices[i].map(|i|(i,p))).collect();
    }
    pub(super) fn replace_sqrt(&mut self, pc: usize, index: usize, destination: u32, input: u32) {
        let before = self.words_before(pc,index);
        let mut words = before.clone();
        let source = super::input(SourceOperand::VectorRegister(input as u8),Ty::F64);
        let target = crate::rdna_spmd::dialect::rdna4::unary(&self.registry,I::V_SQRT_F64).unwrap();
        let expr = Expr { params:vec![Ty::F64], insts:vec![ExprInst::Target {
            op:target,args:Arguments::Unary(ValueId(0)),outputs:vec![Ty::F64],
        }], results:vec![ValueId(1)] }.verify_with(&self.registry).unwrap();
        let mut temporary = Block { params:vec![],insts:vec![],term:Term::Ret };
        let mut scc = self.ir.blocks[&BlockId(pc)].params.last().unwrap().0;
        let mut provenance = 0;
        let mut plan = super::function::alu(&self.registry,&mut self.ir,&mut temporary,&mut words,&mut BTreeMap::new(),
            &mut scc,&mut provenance,&[source],&[Output::Vgpr(destination,Ty::F64)],&false,&expr,
            &[Word::Vgpr(destination),Word::Vgpr(destination+1)]);
        let old = self.blocks[&pc].instructions[index].as_ref().unwrap();
        let position = old.core.start;
        let mut replacements = BTreeMap::new();
        for (&(_,old),&(_,new)) in old.outputs.iter().zip(&plan.outputs) { replacements.insert(old,new); }
        for &(slot,value) in &self.state.sites[&pc][index].rewrite.defined_words {
            if let Some(w) = super::rewrite::word(slot) { replacements.insert(value,words[&w]); }
        }
        let count = temporary.insts.len();
        let positions: Vec<_> = (0..=self.ir.blocks[&BlockId(pc)].insts.len()).map(|i|if i<position{i}else{i+count}).collect();
        self.ir.blocks.get_mut(&BlockId(pc)).unwrap().insts.splice(position..position,temporary.insts);
        self.remap_positions(pc,&positions);
        plan.core = position+plan.core.start..position+plan.core.end;
        plan.previous_core = position+plan.previous_core.start..position+plan.previous_core.end;
        plan.updates_start += position;
        self.blocks.get_mut(&pc).unwrap().instructions[index] = Some(plan);
        self.substitute(&replacements);
        let pair = |r| [words[&Word::Vgpr(r)],words[&Word::Vgpr(r+1)]];
        let reads = vec![before[&Word::Vgpr(input)],before[&Word::Vgpr(input+1)]];
        let writes: Vec<_> = [destination,destination+1].map(|r|(r,words[&Word::Vgpr(r)])).into();
        let site = &mut self.state.sites.get_mut(&pc).unwrap()[index];
        site.rewrite = Observation { reads:reads.clone(),definitions:writes.iter().map(|&(r,v)|(r+512,v)).collect(),
            defined_words:writes.iter().map(|&(r,v)|(r+512,v)).collect(), replaced:[destination,destination+1].map(|r|before[&Word::Vgpr(r)]).into(),
            known:true,removable:true,math:None };
        site.math_reads = [input,input+1].map(|r|(r,before[&Word::Vgpr(r)])).into();
        site.reads = reads.clone(); site.varying_inputs=reads;
        site.f64_definitions=vec![(pair(destination)[0],pair(destination)[1])];
        site.f64_uses=vec![(input,pair(input)[0],pair(input)[1]),(destination,pair(destination)[0],pair(destination)[1])];
        site.writes=writes;
        site.sqrt = crate::rdna_spmd::sqrt_idiom::Policy { shape:crate::rdna_spmd::sqrt_idiom::Shape::Sqrt {
            input:[before[&Word::Vgpr(input)],before[&Word::Vgpr(input+1)]],output:pair(destination),
        },steppable:true,replaced:site.rewrite.replaced.clone() };
    }
}

/// Positions retained only until the historical local pass has run. Scheduling
/// hints and the terminating source position constrain its proof window.
#[derive(Clone)]
pub(in crate::rdna_spmd) enum Position { Instruction(usize), Control(Observation) }
impl LiftedFunction {
    pub(in crate::rdna_spmd) fn observations(&self, pc: usize, order: &[Position]) -> Vec<Observation> {
        order.iter().map(|p|match p {
            Position::Instruction(i) => self.state.sites[&pc][*i].rewrite.clone(),
            Position::Control(o) => o.clone(),
        }).collect()
    }
    pub(in crate::rdna_spmd) fn local_square_roots(&mut self, pc: usize, order: &mut Vec<Position>) {
        let (remove, edits) = crate::rdna_spmd::combine::square_roots(&self.observations(pc,order));
        for (at,dst,input) in edits {
            let Position::Instruction(index) = order[at] else { unreachable!() };
            self.replace_sqrt(pc,index,dst,input);
        }
        self.remove_positions(pc,order,&remove);
    }
    pub(in crate::rdna_spmd) fn local_dead(&mut self, pc: usize, order: &mut Vec<Position>) {
        let remove = crate::rdna_spmd::combine::dead(&self.observations(pc,order));
        self.remove_positions(pc,order,&remove);
    }
    pub(in crate::rdna_spmd) fn local_divisions(&mut self, pc: usize) {
        let observations = self.state.sites[&pc].iter().map(|s|s.rewrite.clone()).collect::<Vec<_>>();
        let remove = crate::rdna_spmd::combine::divisions(&observations);
        self.remove_sites(pc,&remove);
    }
    pub(in crate::rdna_spmd) fn cross_block_square_roots(&mut self) {
        let edits = crate::rdna_spmd::mathcombine::analyze(self);
        for (pc,(removed,rewrites)) in edits {
            for (index,dst,input) in rewrites { self.replace_sqrt(pc,index,dst,input); }
            self.remove_sites(pc,&(0..self.state.sites[&pc].len()).map(|i|removed.contains(&i)).collect::<Vec<_>>());
        }
    }
    fn remove_positions(&mut self, pc: usize, order: &mut Vec<Position>, remove: &[bool]) {
        let mut body_remove = vec![false;self.state.sites[&pc].len()];
        for (position,&removed) in order.iter().zip(remove) {
            if let Position::Instruction(index) = position { body_remove[*index] = removed; }
            else { assert!(!removed); }
        }
        self.remove_sites(pc,&body_remove);
        let mut next=0;
        let indices:Vec<_>=body_remove.iter().map(|&remove|{let index=next;next+=(!remove) as usize;index}).collect();
        let mut flags=remove.iter();
        order.retain_mut(|p| {
            if *flags.next().unwrap() {return false;}
            if let Position::Instruction(i) = p { *i=indices[*i]; }
            true
        });
    }
}
