//! Fragment extraction from SSA. Cut points bind architectural values to the
//! continuation parameters and rematerialize only their pure operand views.
use super::*;
use super::function::LiftedFunction;
use super::state::Word;
use crate::rdna_spmd::ir::typed::cfg::*;
use std::collections::{BTreeMap,BTreeSet};
use std::ops::Range;

impl LiftedFunction {
    pub(in crate::rdna_spmd) fn fragment(&self, ranges: &BTreeMap<usize,(Range<usize>,bool)>, entry: usize) -> Self {
        let mut out=self.clone();
        out.ir.entry=BlockId(entry);
        out.ir.blocks.retain(|pc,_|ranges.contains_key(&pc.0));
        out.blocks.retain(|pc,_|ranges.contains_key(pc));
        out.state.sites.retain(|pc,_|ranges.contains_key(pc));
        out.state.outgoing.retain(|pc,_|ranges.contains_key(pc));
        out.state.scalar_outgoing.retain(|pc,_|ranges.contains_key(pc));
        out.state.conditions.retain(|pc,_|ranges.get(pc).is_some_and(|r|r.1));
        out.state.exec_edges.retain(|(pc,_),_|ranges.get(pc).is_some_and(|r|r.1));
        out.state.yielding.retain(|pc|ranges.get(pc).is_some_and(|r|r.1));
        out.state.barriers.retain(|pc|ranges.get(pc).is_some_and(|r|r.1));
        out.state.yield_exec_writes.retain(|pc|ranges.get(pc).is_some_and(|r|r.1));
        out.state.wave_reads.clear();out.state.resume_observations.clear();
        for (&pc,(range,keep_term)) in ranges {
            let original=&self.blocks[&pc];
            let mut keep=vec![false;self.ir.blocks[&BlockId(pc)].insts.len()];
            for index in range.clone() {
                if let Some(a)=&original.instructions[index] {keep[a.core.clone()].fill(true);}
                if let Some(p)=original.memory.get(&index) {keep[p.core.start..p.end].fill(true);}
                if let Some((_,p))=original.wave.get(&index) {keep[p.core.start..p.end].fill(true);}
            }
            if *keep_term {
                if let Some(c)=&original.condition {keep[c.core.clone()].fill(true);}
                if let Some(p)=&original.yield_values {keep[p.core.start..p.end].fill(true);}
            }
            let b=out.blocks.get_mut(&pc).unwrap();
            b.instructions=b.instructions[range.clone()].to_vec();
            b.memory=std::mem::take(&mut b.memory).into_iter().filter_map(|(i,p)|range.contains(&i).then_some((i.saturating_sub(range.start),p))).collect();
            b.wave=std::mem::take(&mut b.wave).into_iter().filter_map(|(i,p)|range.contains(&i).then_some((i.saturating_sub(range.start),p))).collect();
            if !keep_term {b.condition=None;b.yield_values=None;b.yield_action=None;b.yielding=false;out.ir.blocks.get_mut(&BlockId(pc)).unwrap().term=Term::Ret;}
            out.state.sites.insert(pc,self.state.sites[&pc][range.clone()].to_vec());
            let after=self.words_before(pc,range.end);
            if !keep_term {
                for &(r,i) in &out.state.vector_parameters {b.outgoing[i]=after[&Word::Vgpr(r)];}
                for &(r,i) in &out.state.scalar_parameters {b.outgoing[i]=after[&Word::scalar(r).unwrap()];}
                let scc=self.scc_before(pc,range.end);
                *b.outgoing.last_mut().unwrap()=scc;
                out.state.outgoing.insert(pc,out.state.vector_parameters.iter().map(|&(r,i)|(r,b.outgoing[i])).collect());
                out.state.scalar_outgoing.insert(pc,out.state.scalar_parameters.iter().map(|&(r,i)|(r,b.outgoing[i])).collect());
            }
            let mut positions=vec![0];let mut next=0;
            let mut i=0;
            out.ir.blocks.get_mut(&BlockId(pc)).unwrap().insts.retain(|_|{let yes=keep[i];i+=1;next+=yes as usize;positions.push(next);yes});
            out.remap_positions(pc,&positions);
            out.rebind_fragment(self,pc,range.start);
            if let (Some(action),Some(plan))=(&out.blocks[&pc].yield_action,&out.blocks[&pc].yield_values) {
                let io=action.io();
                let reads:Vec<_>=io.reads.vgprs().map(|r|out.state.outgoing[&pc][&r]).collect();
                out.state.wave_reads.extend(reads);
                for &(dest,value) in &plan.definitions {
                    if let super::wave::Destination::Vgpr(r)=dest {out.state.resume_observations.push((value,out.state.outgoing[&pc][&r]));}
                }
            }
        }
        out.compact();
        out
    }

    fn scc_before(&self,pc:usize,index:usize)->ValueId {
        let mut value=self.ir.blocks[&BlockId(pc)].params.last().unwrap().0;
        let b=&self.blocks[&pc];
        for i in 0..index {
            if let Some(a)=&b.instructions[i] {for &(out,v) in &a.outputs {if matches!(out,Output::Scc) {value=v;}}}
            if let Some((_,p))=b.wave.get(&i) {for &(dest,v) in &p.definitions {if matches!(dest,super::wave::Destination::Scc) {value=v;}}}
        }
        value
    }

    fn rebind_fragment(&mut self, original: &Self, pc: usize, start: usize) {
        let entry_words=self.words_before(pc,0);
        let before=original.words_before(pc,start);
        let mut replacements:BTreeMap<_,_>=before.iter().filter_map(|(w,&v)|(v!=entry_words[w]).then_some((v,entry_words[w]))).collect();
        let before_scc=original.scc_before(pc,start);
        let entry_scc=self.ir.blocks[&BlockId(pc)].params.last().unwrap().0;
        if before_scc!=entry_scc {replacements.insert(before_scc,entry_scc);}
        let block=&self.ir.blocks[&BlockId(pc)];
        let local:BTreeSet<_>=block.params.iter().map(|p|p.0).chain(block.insts.iter().flat_map(definitions)).collect();
        let old_types=original.ir.types.len();
        let mut prefix=Block {params:vec![],insts:vec![],term:Term::Ret};
        let mut views=BTreeMap::new();
        let mut bindings=Vec::new();
        let b=&self.blocks[&pc];
        for a in b.instructions.iter().flatten() {bindings.extend(a.inputs.iter().cloned());}
        for p in b.memory.values() {bindings.extend(p.parameters.iter().filter_map(|(p,v)|match p {
            super::memory::Parameter::Register(input)=>Some((input.clone(),*v)),_=>None,
        }));}
        for (_,p) in b.wave.values() {bindings.extend(p.parameters.iter().cloned());}
        if let Some(p)=&b.yield_values {bindings.extend(p.parameters.iter().cloned());}
        if let Some(c)=&b.condition {bindings.push((c.input.clone(),c.input_value));}
        let scc=self.ir.blocks[&BlockId(pc)].params.last().unwrap().0;
        for (input,id) in bindings {
            if local.contains(&id)||replacements.contains_key(&id) {continue;}
            let scalar=matches!(input.source,InputSource::Operand(SourceOperand::ScalarRegister(_))|InputSource::Scc);
            let mut operands=super::state::Operands::default();
            let value=operands.read(&input,scalar,Some(scc),&mut self.ir,&mut prefix,&entry_words,&mut views);
            prefix.insts.extend(operands.core);replacements.insert(id,value);
        }
        // Resolve retained pure views originally materialized by a removed site.
        // All architectural words at the cut are already explicit parameters.
        let definitions:BTreeMap<_,_>=original.ir.blocks[&BlockId(pc)].insts.iter().flat_map(|i|definitions(i).into_iter().map(move |v|(v,i.clone()))).collect();
        fn materialize(v:ValueId,local:&BTreeSet<ValueId>,old_types:usize,defs:&BTreeMap<ValueId,Inst>,map:&mut BTreeMap<ValueId,ValueId>,
            f:&mut Func,prefix:&mut Vec<Inst>) -> ValueId {
            if let Some(&next)=map.get(&v) {return next;}
            if local.contains(&v)||v.0>=old_types {return v;}
            let i=defs.get(&v).expect("fragment input has no SSA definition");
            let new=f.value(f.types[v.0]);
            let i=match i {
                Inst::Core {ty,op,..}=>Inst::Core {value:new,ty:*ty,op:op.map(|v|materialize(v,local,old_types,defs,map,f,prefix))},
                Inst::Packet {op,input,..}=>Inst::Packet {op:*op,input:materialize(*input,local,old_types,defs,map,f,prefix),output:new},
                Inst::Effect {op:effect::EffectOp::Wave(op @ (effect::WaveOp::Any|effect::WaveOp::Ballot)),inputs,..}=>{
                    let input=materialize(inputs[0],local,old_types,defs,map,f,prefix);
                    Inst::Effect {provenance:(1u64<<63)|new.0 as u64,op:effect::EffectOp::Wave(*op),inputs:vec![input],outputs:vec![(new,f.types[new.0])]}
                },
                _=>panic!("fragment must bind an earlier effect/target result through its architectural value: {:?} {:?}",v,i),
            };
            prefix.push(i);map.insert(v,new);new
        }
        use std::cell::RefCell;
        // Collect references of just this block, including metadata.
        let mut one=self.clone();
        one.ir.blocks.retain(|id,_|id.0==pc);one.blocks.retain(|&id,_|id==pc);
        one.state.sites.retain(|&id,_|id==pc);one.state.outgoing.retain(|&id,_|id==pc);one.state.scalar_outgoing.retain(|&id,_|id==pc);
        one.state.wave_reads.clear();one.state.resume_observations.clear();
        let references=RefCell::new(Vec::new());one.map_values(|v|{references.borrow_mut().push(v);v},false);
        for v in references.into_inner() {materialize(v,&local,old_types,&definitions,&mut replacements,&mut self.ir,&mut prefix.insts);}
        one.substitute(&replacements);
        let count=prefix.insts.len();
        let positions:Vec<_>=(0..=one.ir.blocks[&BlockId(pc)].insts.len()).map(|i|i+count).collect();
        one.remap_positions(pc,&positions);
        prefix.insts.extend(std::mem::take(&mut one.ir.blocks.get_mut(&BlockId(pc)).unwrap().insts));
        one.ir.blocks.get_mut(&BlockId(pc)).unwrap().insts=prefix.insts;
        self.ir.blocks.insert(BlockId(pc),one.ir.blocks.remove(&BlockId(pc)).unwrap());
        self.blocks.insert(pc,one.blocks.remove(&pc).unwrap());self.state.sites.insert(pc,one.state.sites.remove(&pc).unwrap());
        self.state.outgoing.insert(pc,one.state.outgoing.remove(&pc).unwrap());self.state.scalar_outgoing.insert(pc,one.state.scalar_outgoing.remove(&pc).unwrap());
    }
}
fn definitions(i:&Inst)->Vec<ValueId> {match i {
    Inst::Core {value,..}|Inst::Packet {output:value,..}=>vec![*value],
    Inst::Target {outputs,..}|Inst::Effect {outputs,..}=>outputs.iter().map(|p|p.0).collect(),
}}
