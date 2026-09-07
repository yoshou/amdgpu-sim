//! SSA graph edits keep semantic uses, native bindings and analysis observations
//! in the same value namespace. No edit reconstructs decoded instructions.
use super::*;
use super::function::LiftedFunction;
use crate::rdna_spmd::ir::typed::cfg::*;

fn wave(p: &mut super::wave::Plan, m: &impl Fn(ValueId) -> ValueId) {
    for (_, v) in &mut p.parameters { *v = m(*v); }
    for v in &mut p.arguments { *v = m(*v); }
    for (v, _) in &mut p.results { *v = m(*v); }
    for (_, v) in &mut p.definitions { *v = m(*v); }
}
impl LiftedFunction {
    pub(in crate::rdna_spmd) fn map_values(&mut self, m: impl Fn(ValueId) -> ValueId, definitions: bool) {
        for b in self.ir.blocks.values_mut() {
            if definitions { for (v, _) in &mut b.params { *v = m(*v); } }
            for i in &mut b.insts {
                match i {
                    Inst::Core { value, op, .. } => { *op = op.map(&m); if definitions { *value = m(*value); } },
                    Inst::Packet { input, output, .. } => { *input = m(*input); if definitions { *output = m(*output); } },
                    Inst::Target { args, outputs, .. } => { *args = args.map(&m); if definitions { for (v, _) in outputs { *v = m(*v); } } },
                    Inst::Effect { inputs, outputs, .. } => { for v in inputs { *v = m(*v); } if definitions { for (v, _) in outputs { *v = m(*v); } } },
                }
            }
            match &mut b.term {
                Term::Br(e) => { for v in &mut e.args { *v = m(*v); } },
                Term::CondBr { cond, yes, no } => { *cond = m(*cond); for v in yes.args.iter_mut().chain(&mut no.args) { *v = m(*v); } },
                Term::Ret => {},
            }
        }
        for b in self.blocks.values_mut() {
            for a in b.instructions.iter_mut().flatten() {
                for (_, v) in &mut a.inputs { *v = m(*v); }
                for (_, v) in &mut a.outputs { *v = m(*v); }
                for (a,b) in a.predicated.iter_mut().chain(&mut a.pairs) { *a = m(*a); *b = m(*b); }
                for v in a.packet_elidable.iter_mut().chain(&mut a.scalar_unmasked).chain(&mut a.vector_words) { *v = m(*v); }
                for (v, _) in &mut a.mask_updates { *v = m(*v); }
            }
            for p in b.memory.values_mut() {
                for (_, v) in &mut p.parameters { *v = m(*v); }
                for (a,b) in p.pairs.iter_mut().chain(&mut p.scalar_results) { *a = m(*a); *b = m(*b); }
                p.base = m(p.base); p.address = m(p.address); p.mask = m(p.mask);
                if let Some((_, a,b,c)) = &mut p.flat { *a = m(*a); *b = m(*b); *c = m(*c); }
            }
            for (_, p) in b.wave.values_mut() { wave(p, &m); }
            if let Some(p) = &mut b.yield_values { wave(p, &m); }
            if let Some(c) = &mut b.condition { c.input_value = m(c.input_value); }
            for v in &mut b.outgoing { *v = m(*v); }
        }
        let g = &mut self.state;
        for v in g.scalar_outgoing.values_mut().chain(g.outgoing.values_mut()).flat_map(|b| b.values_mut()) { *v = m(*v); }
        for v in &mut g.wave_reads { *v = m(*v); }
        for (a,b) in &mut g.resume_observations { *a = m(*a); *b = m(*b); }
        for s in g.sites.values_mut().flatten() {
            let r = &mut s.rewrite;
            for v in r.reads.iter_mut().chain(&mut r.replaced) { *v = m(*v); }
            for (_, v) in r.definitions.iter_mut().chain(&mut r.defined_words) { *v = m(*v); }
            if let Some(math) = &mut r.math { for i in &mut math.inputs { for v in i.words.iter_mut().flatten() { *v = m(*v); } } }
            for (_, v) in s.math_reads.iter_mut().chain(&mut s.writes).chain(&mut s.reactivation) { *v = m(*v); }
            for (a,b) in s.f64_definitions.iter_mut().chain(&mut s.u64_definitions) { *a = m(*a); *b = m(*b); }
            for (_,a,b) in &mut s.f64_uses { *a = m(*a); *b = m(*b); }
            for v in s.reads.iter_mut().chain(&mut s.varying_inputs).chain(&mut s.address) { *v = m(*v); }
            if let Some((_,a,b,_)) = &mut s.frame { *a = m(*a); *b = m(*b); }
            let e = &mut s.exec;
            e.before = m(e.before); e.after = m(e.after);
            if let Some(v) = &mut e.saved { *v = m(*v); }
            for v in &mut e.killed { *v = m(*v); }
            use crate::rdna_spmd::analysis::state::Activation;
            match &mut e.update {
                Activation::Copy(v) => *v = m(*v),
                Activation::Or { exec, saved } => { for v in exec.iter_mut().chain(saved) { *v = m(*v); } },
                _ => {},
            }
            for (v, sources, _) in &mut s.masks.definitions { *v = m(*v); for v in sources { *v = m(*v); } }
            for (_, v) in &mut s.masks.reads { *v = m(*v); }
            for v in &mut s.sqrt.replaced { *v = m(*v); }
            use crate::rdna_spmd::sqrt_idiom::Shape;
            match &mut s.sqrt.shape {
                Shape::Other => {},
                Shape::Exponent(v) => *v = m(*v),
                Shape::Scale { input, exponent, output, .. } => {
                    for v in input.iter_mut().flatten().chain(exponent).chain(output) { *v = m(*v); }
                },
                Shape::Sqrt { input, output } => { for v in input.iter_mut().chain(output) { *v = m(*v); } },
                Shape::Class(input) => { for v in input { *v = m(*v); } },
            }
        }
    }
}

impl LiftedFunction {
    /// Drop unreferenced pure remnants of edits and densely renumber values.
    /// Native instruction ranges are roots until their lowering plan is removed.
    pub(in crate::rdna_spmd) fn compact(&mut self) {
        use std::cell::RefCell;
        let ir = std::mem::replace(&mut self.ir,Func {entry:BlockId(0),blocks:Default::default(),types:vec![]});
        let roots = RefCell::new(Vec::new());
        self.map_values(|v| { roots.borrow_mut().push(v);v },false);
        self.ir=ir;
        let mut roots=roots.into_inner();
        for (&pc,b) in &self.blocks {
            let insts=&self.ir.blocks[&BlockId(pc)].insts;
            for a in b.instructions.iter().flatten() {
                for i in &insts[a.core.clone()] { roots.extend(outputs(i)); }
            }
            for p in b.memory.values() { for i in &insts[p.core.start..p.end] {roots.extend(outputs(i));} }
            for (_,p) in b.wave.values() {for i in &insts[p.core.start..p.end] {roots.extend(outputs(i));}}
            if let Some(c)=&b.condition {for i in &insts[c.core.clone()] {roots.extend(outputs(i));}}
            if let Some(p)=&b.yield_values {for i in &insts[p.core.start..p.end] {roots.extend(outputs(i));}}
        }
        // Also preserve block arguments on retained edges; their destination
        // may be outside a fragment until the split CFG is assembled.
        for b in self.ir.blocks.values() {
            roots.extend(b.params.iter().map(|p|p.0));
            for e in b.term.edges() {roots.extend(&e.args);}
            if let Term::CondBr {cond,..}=b.term {roots.push(cond);}
        }
        let mut dependencies=vec![Vec::new();self.ir.types.len()];
        for b in self.ir.blocks.values() {for i in &b.insts {
            let mut uses=Vec::new();
            match i {
                Inst::Core {op,..}=>{op.map(|v|{uses.push(v);v});},
                Inst::Packet {input,..}=>uses.push(*input),
                Inst::Target {args,..}=>uses.extend(args.values()),
                Inst::Effect {inputs,..}=>uses.extend(inputs),
            }
            let defs=outputs(i);
            for v in &defs {dependencies[v.0].extend(&uses);dependencies[v.0].extend(&defs);}
            if defs.is_empty() {roots.extend(uses);}
        }}
        let mut live=vec![false;self.ir.types.len()];
        while let Some(v)=roots.pop() {if !live[v.0] {live[v.0]=true;roots.extend(&dependencies[v.0]);}}
        for pc in self.blocks.keys().copied().collect::<Vec<_>>() {
            let b=self.ir.blocks.get_mut(&BlockId(pc)).unwrap();
            let mut positions=Vec::new();let mut next=0;
            b.insts.retain(|i| {positions.push(next);let defs=outputs(i);let keep=defs.is_empty()||defs.iter().any(|v|live[v.0]);next+=keep as usize;keep});
            positions.push(next);self.remap_positions(pc,&positions);
        }
        let mut ids=vec![None;self.ir.types.len()];let mut types=Vec::new();
        for b in self.ir.blocks.values() {
            for v in b.params.iter().map(|p|p.0).chain(b.insts.iter().flat_map(outputs)) {
                assert!(ids[v.0].is_none(),"duplicate SSA definition during compaction");
                ids[v.0]=Some(ValueId(types.len()));types.push(self.ir.types[v.0]);
            }
        }
        self.map_values(|v|ids[v.0].expect("SSA edit left a dangling reference"),true);
        self.ir.types=types;
    }
}
fn outputs(i: &Inst) -> Vec<ValueId> {
    match i {Inst::Core {value,..}|Inst::Packet {output:value,..}=>vec![*value],
        Inst::Target {outputs,..}|Inst::Effect {outputs,..}=>outputs.iter().map(|p|p.0).collect()}
}
