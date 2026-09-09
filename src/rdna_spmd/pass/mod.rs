//! Semantic rewrites of typed SSA, independent of ISA operands and width.
pub(crate) mod narrow;
pub(crate) mod specialise;
pub(crate) mod idioms;
pub(crate) mod active;
pub(crate) mod pairs;
pub(crate) mod simplify;
pub(crate) mod dce;
pub(crate) mod entry;
use super::ir::{*, Op, Ty, ValueId};

pub(super) trait Program {
    type Snapshot: PartialEq;
    fn snapshot(&self) -> Self::Snapshot;
    fn ir(&self) -> &Func;
    fn registry(&self) -> &super::dialect::DialectRegistry;
    fn touch(&mut self);
}

pub(super) struct Driver { trace: bool }

pub(super) struct FuncProgram<'r> { pub ir: Func, pub registry: &'r super::dialect::DialectRegistry }
impl Program for FuncProgram<'_> {
    type Snapshot = Func;
    fn snapshot(&self) -> Func { self.ir.clone() }
    fn ir(&self) -> &Func { &self.ir }
    fn registry(&self) -> &super::dialect::DialectRegistry { self.registry }
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
    pub fn run<P: Program>(&self, program: &mut P, name: &str, pass: impl FnOnce(&mut P)) -> Result<(), String> {
        pass(program);
        program.touch();
        self.check(program, name)
    }
    pub fn fixpoint<P: Program>(&self, program: &mut P, name: &str, limit: usize, mut pass: impl FnMut(&mut P)) -> Result<(), String> {
        if limit == 0 { return Err(format!("{name}: fixpoint group requires a nonzero iteration limit")); }
        for _ in 0..limit {
            let before = program.snapshot();
            pass(program);
            program.touch();
            self.check(program, name)?;
            if program.snapshot() == before { return Ok(()); }
        }
        Err(format!("{name}: no fixed point within {limit} iterations"))
    }
    fn check<P: Program>(&self, program: &P, name: &str) -> Result<(), String> {
        if self.trace { eprintln!("; after {name}\n{}", super::ir::print::func(program.registry(), program.ir())); }
        program.ir().clone().verify_with(program.registry()).map_err(|e| format!("{name}: {e}"))?;
        Ok(())
    }
}

/// WriteLane requires wave-uniform value and selector. Lane i's result is
/// exactly `i == (selector & 31) ? value : old[i]`, so no value crosses a
/// packet boundary and the rendezvous can be removed.
pub(crate) fn local_write_lanes(f: &mut Func) -> usize {
    let scheduled_reads = f.blocks.values().flat_map(|b| &b.insts).any(|inst| matches!(inst,
        Inst::Effect { op: EffectOp::Wave(WaveOp::ReadLane), provenance, .. } if provenance & super::lift::wave::SCHEDULED != 0));
    if !scheduled_reads { return 0; }
    let mut count = 0;
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut insts = Vec::with_capacity(old.len());
        for inst in old {
            match inst {
                Inst::Effect { op: EffectOp::Wave(WaveOp::WriteLane), inputs, outputs, provenance } if provenance & (super::lift::wave::SCHEDULED | (1 << 63)) == 0 => {
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
