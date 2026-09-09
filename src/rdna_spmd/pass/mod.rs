//! Semantic rewrites of typed SSA, independent of ISA operands and width.
pub(crate) mod narrow;
pub(crate) mod specialise;
pub(crate) mod active;
pub(crate) mod pairs;
pub(crate) mod simplify;
pub(crate) mod dce;
pub(crate) mod entry;
pub(crate) mod idioms;
use super::ir::{*, Op, Ty, ValueId};

use super::analysis::masks::{Exec, Masks};
use super::analysis::uniformity::{Fact, Uniformity};
use super::dialect::DialectRegistry;
use std::cell::OnceCell;

pub(crate) struct Context<'r> {
    pub registry: &'r DialectRegistry,
    pub exec_index: usize,
    pub lanes: u32,
    pub entry_full: bool,
    pub exec_initial: bool,
    pub exec_packed: bool,
    pub packet: Option<(&'r [super::program::Parameter], bool)>,
}

pub(crate) struct Analyses<'r> {
    ctx: Context<'r>,
    constants: OnceCell<Vec<Option<u64>>>,
    masks: OnceCell<Masks>,
    exec: OnceCell<Exec>,
    uniformity: OnceCell<Option<Uniformity>>,
    uniform: OnceCell<Vec<bool>>,
}

impl<'r> Analyses<'r> {
    pub fn new(ctx: Context<'r>) -> Self {
        Self { ctx, constants: OnceCell::new(), masks: OnceCell::new(), exec: OnceCell::new(), uniformity: OnceCell::new(), uniform: OnceCell::new() }
    }
    pub fn context(&self) -> &Context<'r> { &self.ctx }
    pub fn invalidate(&mut self) {
        self.constants.take();
        self.masks.take();
        self.exec.take();
        self.uniformity.take();
        self.uniform.take();
    }
    pub fn constants(&self, f: &Func) -> &[Option<u64>] {
        self.constants.get_or_init(|| super::analysis::constants(f))
    }
    pub fn masks(&self, f: &Func) -> &Masks {
        self.masks.get_or_init(|| super::analysis::masks::analyze(self.ctx.registry, f, self.ctx.exec_index, self.constants(f), self.ctx.lanes, self.ctx.entry_full))
    }
    pub fn exec(&self, f: &Func) -> &Exec {
        self.exec.get_or_init(|| super::analysis::masks::exec(f, self.ctx.exec_index, self.constants(f), self.ctx.lanes, self.ctx.exec_initial, self.ctx.exec_packed))
    }
    pub fn uniformity(&self, f: &Func) -> Option<&Uniformity> {
        self.uniformity.get_or_init(|| self.ctx.packet.map(|(inputs, aligned)| {
            let entry = entry::packet_entry(f, inputs, aligned);
            super::analysis::uniformity::packet(f, &entry, self.constants(f), &self.masks(f).guarded)
        })).as_ref()
    }
    pub fn uniform(&self, f: &Func) -> &[bool] {
        self.uniform.get_or_init(|| match self.uniformity(f) {
            Some(facts) => facts.facts.iter().map(|&fact| fact == Fact::Uniform).collect(),
            None => vec![true; f.types.len()],
        })
    }
}

pub(crate) trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool;
}

pub(super) trait Program {
    type Snapshot: PartialEq;
    #[cfg_attr(not(test), allow(dead_code))]
    fn snapshot(&self) -> Self::Snapshot;
    fn ir(&self) -> &Func;
    fn ir_mut(&mut self) -> &mut Func;
    fn registry(&self) -> &DialectRegistry;
    fn touch(&mut self);
}
pub(super) struct Driver { trace: bool }
pub(super) struct FuncProgram<'r> { pub ir: Func, pub registry: &'r DialectRegistry }
impl Program for FuncProgram<'_> {
    type Snapshot = Func;
    fn snapshot(&self) -> Func { self.ir.clone() }
    fn ir(&self) -> &Func { &self.ir }
    fn ir_mut(&mut self) -> &mut Func { &mut self.ir }
    fn registry(&self) -> &DialectRegistry { self.registry }
    fn touch(&mut self) {}
}
pub(crate) fn compact(f: &mut Func) {
    let mut map: Vec<Option<ValueId>> = vec![None; f.types.len()];
    let mut types = Vec::new();
    let assign = |v: ValueId, map: &mut Vec<Option<ValueId>>, types: &mut Vec<Ty>| {
        let next = ValueId(types.len());
        types.push(f.types[v.0]);
        map[v.0] = Some(next);
    };
    let mut definitions: Vec<ValueId> = Vec::new();
    for block in f.blocks.values() {
        for &(p, _) in &block.params { definitions.push(p); }
        for inst in &block.insts {
            match inst {
                Inst::Core { value, .. } | Inst::Packet { output: value, .. } => definitions.push(*value),
                Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => definitions.extend(outputs.iter().map(|o| o.0)),
            }
        }
    }
    for v in definitions { assign(v, &mut map, &mut types); }
    let m = |v: ValueId| map[v.0].expect("use of a removed SSA value");
    for block in f.blocks.values_mut() {
        for (p, _) in &mut block.params { *p = m(*p); }
        for inst in &mut block.insts {
            match inst {
                Inst::Core { value, op, .. } => { *op = op.map(m); *value = m(*value); }
                Inst::Packet { input, output, .. } => { *input = m(*input); *output = m(*output); }
                Inst::Target { args, outputs, .. } => { *args = args.map(m); for (v, _) in outputs { *v = m(*v); } }
                Inst::Effect { inputs, outputs, .. } => { for v in inputs { *v = m(*v); } for (v, _) in outputs { *v = m(*v); } }
            }
        }
        match &mut block.term {
            Term::Br(e) => for v in &mut e.args { *v = m(*v); },
            Term::CondBr { cond, yes, no } => { *cond = m(*cond); for v in yes.args.iter_mut().chain(&mut no.args) { *v = m(*v); } }
            Term::Ret(args) => for v in args { *v = m(*v); },
        }
    }
    f.types = types;
}

impl Driver {
    pub fn new() -> Self { Self { trace: std::env::var_os("AMDGPU_SIM_PRINT_IR").is_some() } }
    pub fn pipeline<P: Program>(&self, program: &mut P, analyses: &mut Analyses, passes: &[&dyn Pass]) -> Result<bool, String> {
        let mut any = false;
        for pass in passes {
            #[cfg(test)]
            let before = program.snapshot();
            let changed = pass.run(program.ir_mut(), analyses);
            #[cfg(test)]
            assert!(changed || program.snapshot() == before, "{}: changed the function without reporting it", pass.name());
            if changed {
                program.touch();
                analyses.invalidate();
                any = true;
            }
            self.check(program, pass.name())?;
        }
        Ok(any)
    }
    pub fn fixpoint<P: Program>(&self, program: &mut P, analyses: &mut Analyses, name: &str, limit: usize, passes: &[&dyn Pass]) -> Result<(), String> {
        if limit == 0 { return Err(format!("{name}: fixpoint group requires a nonzero iteration limit")); }
        for _ in 0..limit {
            if !self.pipeline(program, analyses, passes)? { return Ok(()); }
        }
        Err(format!("{name}: no fixed point within {limit} iterations"))
    }
    fn check<P: Program>(&self, program: &P, name: &str) -> Result<(), String> {
        if self.trace { eprintln!("; after {name}\n{}", super::ir::print::func(program.registry(), program.ir())); }
        program.ir().clone().verify_with(program.registry()).map_err(|e| format!("{name}: {e}"))?;
        Ok(())
    }
}

pub(crate) struct LocalWriteLanes;
impl Pass for LocalWriteLanes {
    fn name(&self) -> &str { "local_write_lanes" }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool { local_write_lanes(f) > 0 }
}

/// WriteLane requires wave-uniform value and selector. Lane i's result is
/// exactly `i == (selector & 31) ? value : old[i]`, so no value crosses a
/// packet boundary and the rendezvous can be removed.
pub(crate) fn local_write_lanes(f: &mut Func) -> usize {
    let scheduled_reads = f.blocks.values().flat_map(|b| &b.insts).any(|inst| matches!(inst,
        Inst::Effect { op: EffectOp::Wave(WaveOp::ReadLane), provenance, .. } if provenance & crate::rdna_spmd::ir::SCHEDULED != 0));
    if !scheduled_reads { return 0; }
    let mut count = 0;
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut insts = Vec::with_capacity(old.len());
        for inst in old {
            match inst {
                Inst::Effect { op: EffectOp::Wave(WaveOp::WriteLane), inputs, outputs, provenance } if provenance & (crate::rdna_spmd::ir::SCHEDULED | (1 << 63)) == 0 => {
                    let (result, ty) = outputs[0];
                    assert_eq!(ty, Ty::I32);
                    let lane = f.value(Ty::I32); let mask = f.value(Ty::I32);
                    let selector = f.value(Ty::I32); let selected = f.value(Ty::I1);
                    insts.push(Inst::Core { value: lane, ty: Ty::I32, op: Op::Env(super::ir::Env::LaneId) });
                    insts.push(Inst::Core { value: mask, ty: Ty::I32, op: Op::Const(Ty::I32, 31) });
                    insts.push(Inst::Core { value: selector, ty: Ty::I32, op: Op::Int(super::ir::IntOp::And, inputs[1], mask) });
                    insts.push(Inst::Core { value: selected, ty: Ty::I1, op: Op::Cmp(super::ir::IntPred::Eq, lane, selector) });
                    insts.push(Inst::Core { value: result, ty: Ty::I32, op: Op::Select(selected, inputs[0], inputs[2]) });
                    count += 1;
                }
                other => insts.push(other),
            }
        }
        f.blocks.get_mut(&id).unwrap().insts = insts;
    }
    count
}

/// Any ignores padding and each launched wave contains a valid work item.
/// A constant predicate on valid lanes therefore has a constant wave result.
pub(crate) fn constant_queries(f:&mut Func,entry_true:&[ValueId]) -> usize {
    let facts=super::analysis::valid_predicate_constants(f,entry_true);
    let mut count=0;
    for block in f.blocks.values_mut() {for inst in &mut block.insts {
        let query=match inst {
            Inst::Packet {op:PacketOp::Any,input,output}=>Some((*input,*output)),
            Inst::Effect {op:EffectOp::Wave(WaveOp::Any),inputs,outputs,..}=>Some((inputs[0],outputs[0].0)),
            _=>None,
        };
        if let Some((input,output))=query {
            if let Some(bits)=facts[input.0] {
                *inst=Inst::Core {value:output,ty:Ty::I1,op:Op::Const(Ty::I1,bits)};
                count+=1;
            }
        }
    }}
    count
}
