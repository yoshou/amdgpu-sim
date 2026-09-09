use crate::rdna_spmd::analysis::masks::Masks;
use crate::rdna_spmd::dialect::{Arguments, DialectRegistry, TargetOp};
use crate::rdna_spmd::ir::{*, FloatOp, FloatUnary, Op, Ty, ValueId};
use crate::rdna_spmd::pass::idioms::Idiom;
use std::collections::BTreeMap;

pub(in crate::rdna_spmd) struct SqrtIdioms { sqrt: TargetOp, rsq: TargetOp, ldexp: TargetOp }
impl SqrtIdioms {
    pub(in crate::rdna_spmd) fn new(registry: &DialectRegistry) -> Self {
        let op = |name: &str| registry.lookup(super::ID, name).expect("missing RDNA4 provider");
        Self { sqrt: op("sqrt.f64"), rsq: op("rsq.f64"), ldexp: op("ldexp.f64") }
    }
}
impl Idiom for SqrtIdioms {
    fn rewrite(&self, f: &mut Func, masks: &Masks, constants: &[Option<u64>]) -> usize { run(f, masks, constants, self) }
}

struct View<'a> {
    f: &'a Func,
    defs: Vec<Option<Op>>,
    targets: Vec<Option<(TargetOp, Vec<ValueId>)>>,
    masks: &'a Masks,
    constants: &'a [Option<u64>],
}

impl<'a> View<'a> {
    fn new(f: &'a Func, masks: &'a Masks, constants: &'a [Option<u64>]) -> Self {
        let mut defs = vec![None; f.types.len()];
        let mut targets = vec![None; f.types.len()];
        for block in f.blocks.values() {
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, op, .. } => defs[value.0] = Some(*op),
                    Inst::Target { op, args, outputs, .. } if outputs.len() == 1 => targets[outputs[0].0 .0] = Some((*op, args.values().to_vec())),
                    _ => {}
                }
            }
        }
        Self { f, defs, targets, masks, constants }
    }
    fn raw(&self, mut v: ValueId) -> ValueId {
        loop {
            match self.masks.predicated[v.0] {
                Some((new, _)) => v = new,
                None => match self.defs[v.0] {
                    Some(Op::Convert(Cvt::Bitcast, to, a)) if self.f.types[a.0] == to => v = a,
                    _ => return v,
                },
            }
        }
    }
    fn target(&self, v: ValueId, wanted: TargetOp) -> Option<&[ValueId]> {
        let (op, args) = self.targets[self.raw(v).0].as_ref()?;
        (wanted == *op).then_some(args.as_slice())
    }
    fn neg(&self, v: ValueId) -> Option<ValueId> {
        match self.defs[self.raw(v).0] { Some(Op::Unary(FloatUnary::Neg, a)) => Some(self.raw(a)), _ => None }
    }
    fn constant_f64(&self, v: ValueId, expected: f64) -> bool {
        self.constants[self.raw(v).0] == Some(expected.to_bits())
    }
    fn mul(&self, v: ValueId) -> Option<(ValueId, ValueId)> {
        match self.defs[self.raw(v).0] { Some(Op::Float(FloatOp::Mul, a, b)) => Some((self.raw(a), self.raw(b))), _ => None }
    }
    fn fma(&self, v: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
        match self.defs[self.raw(v).0] { Some(Op::MulAdd(a, b, c)) | Some(Op::Fma(a, b, c)) => Some((a, b, c)), _ => None }
    }
    fn same(&self, a: ValueId, b: ValueId) -> bool { self.raw(a) == self.raw(b) }
}

fn either(view: &View, pair: (ValueId, ValueId), x: ValueId, y: ValueId) -> bool {
    (view.same(pair.0, x) && view.same(pair.1, y)) || (view.same(pair.0, y) && view.same(pair.1, x))
}

fn sqrt_chain(view: &View, result: ValueId, ops: &SqrtIdioms) -> Option<ValueId> {
    let (b3, r3, a3) = view.fma(result)?;
    let (na3, a3b, x) = view.fma(b3)?;
    let a3n = view.neg(na3)?;
    if !view.same(a3n, a3) || !view.same(a3b, a3) { return None; }
    let (b2, r3b, a2) = view.fma(a3)?;
    if !view.same(r3b, r3) { return None; }
    let (na2, a2b, xb) = view.fma(b2)?;
    let a2n = view.neg(na2)?;
    if !view.same(a2n, a2) || !view.same(a2b, a2) || !view.same(xb, x) { return None; }
    let (r2, b, r2b) = view.fma(r3)?;
    if !view.same(r2b, r2) { return None; }
    let (a, bb, ab) = view.fma(a2)?;
    if !view.same(bb, b) || !view.same(ab, a) { return None; }
    let (nr2, ac, half) = view.fma(b)?;
    let r2n = view.neg(nr2)?;
    if !view.same(r2n, r2) || !view.same(ac, a) || !view.constant_f64(half, 0.5) { return None; }
    let (h, r) = view.mul(r2)?;
    let (h, r) = if view.constant_f64(h, 0.5) { (h, r) } else if view.constant_f64(r, 0.5) { (r, h) } else { return None; };
    let _ = h;
    if !either(view, view.mul(a)?, x, r) { return None; }
    let rsq = view.target(r, ops.rsq)?;
    if !view.same(rsq[0], x) { return None; }
    Some(view.raw(x))
}

fn exponent_select(view: &View, v: ValueId) -> Option<(ValueId, i32, i32)> {
    let raw = view.raw(v);
    let Some(Op::Select(c, a, b)) = view.defs[raw.0] else {
        let k = view.constants[raw.0]? as u32 as i32;
        return Some((ValueId(usize::MAX), k, k));
    };
    let (ka, kb) = (view.constants[view.raw(a).0]? as u32 as i32, view.constants[view.raw(b).0]? as u32 as i32);
    Some((view.raw(c), ka, kb))
}

fn scale(view: &View, v: ValueId, ops: &SqrtIdioms) -> Option<(ValueId, (ValueId, i32, i32))> {
    if let Some(args) = view.target(v, ops.ldexp) {
        return Some((view.raw(args[0]), exponent_select(view, args[1])?));
    }
    let Some(Op::Select(c, a, b)) = view.defs[view.raw(v).0] else { return None; };
    for (scaled, plain, flipped) in [(a, b, false), (b, a, true)] {
        let Some(args) = view.target(scaled, ops.ldexp) else { continue; };
        if !view.same(args[0], plain) { continue; }
        let k = view.constants[view.raw(args[1]).0]? as u32 as i32;
        return Some((view.raw(plain), (view.raw(c), if flipped { 0 } else { k }, if flipped { k } else { 0 })));
    }
    None
}

fn scaled_sqrt(view: &View, out: ValueId, ops: &SqrtIdioms) -> Option<ValueId> {
    let (root, (c2, a2, b2)) = scale(view, out, ops)?;
    let sqrt = view.target(root, ops.sqrt)?;
    let (x, (c1, a1, b1)) = scale(view, sqrt[0], ops)?;
    if c1 != c2 { return None; }
    let halves = |e: i32, e2: i32| e % 2 == 0 && e2 == -(e / 2) && e.abs() <= 1022;
    if !(halves(a1, a2) && halves(b1, b2)) { return None; }
    Some(view.raw(x))
}

fn run(f: &mut Func, masks: &Masks, constants: &[Option<u64>], ops: &SqrtIdioms) -> usize {
    let mut rewrites: Vec<(BlockId, usize, ValueId, ValueId)> = Vec::new();
    {
        let view = View::new(f, masks, constants);
        for (&id, block) in &f.blocks {
            for (index, inst) in block.insts.iter().enumerate() {
                let value = match inst {
                    Inst::Core { value, ty: Ty::F64, .. } => *value,
                    Inst::Target { outputs, .. } if outputs.len() == 1 && outputs[0].1 == Ty::F64 => outputs[0].0,
                    _ => continue,
                };
                if view.raw(value) != value { continue; }
                let replacement = sqrt_chain(&view, value, ops).filter(|&x| scale(&view, x, ops).is_some())
                    .or_else(|| scaled_sqrt(&view, value, ops));
                if let Some(x) = replacement { rewrites.push((id, index, value, x)); }
            }
        }
    }
    if rewrites.is_empty() { return 0; }
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut inserted: BTreeMap<BlockId, Vec<(usize, Inst)>> = BTreeMap::new();
    for &(block, index, value, x) in &rewrites {
        let root = f.value(Ty::F64);
        inserted.entry(block).or_default().push((index, Inst::Target { provenance: None, op: ops.sqrt, args: Arguments::Unary(x), outputs: vec![(root, Ty::F64)] }));
        renames.insert(value, root);
    }
    for (block, mut insts) in inserted {
        insts.sort_by_key(|(index, _)| std::cmp::Reverse(*index));
        let b = f.blocks.get_mut(&block).unwrap();
        for (index, inst) in insts { b.insts.insert(index + 1, inst); }
    }
    crate::rdna_spmd::pass::simplify::rename(f, &renames);
    rewrites.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, VOP1, VOP2, VOP3};
    use crate::rdna_spmd::{CompilationInput, ScalarBlock, ScalarProgram, Terminator};

    fn v(r: u8) -> SourceOperand { SourceOperand::VectorRegister(r) }
    fn fma(dst: u8, a: SourceOperand, b: SourceOperand, c: SourceOperand, neg: u8) -> InstFormat {
        InstFormat::VOP3(VOP3 { op: I::V_FMA_F64, vdst: dst, src0: a, src1: b, src2: c, neg, abs: 0, cm: 0, omod: 0, opsel: 0 })
    }
    fn mul(dst: u8, a: SourceOperand, b: u8) -> InstFormat {
        InstFormat::VOP2(VOP2 { op: I::V_MUL_F64, vdst: dst, src0: a, vsrc1: b, literal_constant: None })
    }

    #[test]
    fn rsq_newton_chain_inside_the_scale_guard_folds_to_sqrt() {
        let half = SourceOperand::FloatConstant(0.5);
        let body = vec![
            InstFormat::VOP3(VOP3 { op: I::V_CNDMASK_B32, vdst: 20, src0: SourceOperand::IntegerConstant(0), src1: SourceOperand::LiteralConstant(768), src2: SourceOperand::ScalarRegister(106), neg: 0, abs: 0, cm: 0, omod: 0, opsel: 0 }),
            InstFormat::VOP3(VOP3 { op: I::V_LDEXP_F64, vdst: 2, src0: v(0), src1: v(20), src2: SourceOperand::IntegerConstant(0), neg: 0, abs: 0, cm: 0, omod: 0, opsel: 0 }),
            InstFormat::VOP1(VOP1 { op: I::V_RSQ_F64, vdst: 4, src0: v(2) }),
            mul(6, v(2), 4),
            mul(4, half.clone(), 4),
            fma(8, v(4), v(6), half.clone(), 1),
            fma(6, v(6), v(8), v(6), 0),
            fma(4, v(4), v(8), v(4), 0),
            fma(8, v(6), v(6), v(2), 1),
            fma(6, v(8), v(4), v(6), 0),
            fma(8, v(6), v(6), v(2), 1),
            fma(4, v(8), v(4), v(6), 0),
            InstFormat::VOP3(VOP3 { op: I::V_CNDMASK_B32, vdst: 21, src0: SourceOperand::IntegerConstant(0), src1: SourceOperand::LiteralConstant((-384i32) as u32), src2: SourceOperand::ScalarRegister(106), neg: 0, abs: 0, cm: 0, omod: 0, opsel: 0 }),
            InstFormat::VOP3(VOP3 { op: I::V_LDEXP_F64, vdst: 10, src0: v(4), src1: v(21), src2: SourceOperand::IntegerConstant(0), neg: 0, abs: 0, cm: 0, omod: 0, opsel: 0 }),
        ];
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock { pc: 0, body, term: Terminator::Return })]) };
        let f = program.to_ssa().function;
        let registry = f.registry.clone();
        let mut ir = f.ir.clone();
        let keep: Vec<usize> = f.parameter_inputs.iter().enumerate().filter(|(_, p)| matches!(p.source, crate::rdna_spmd::program::ParameterSource::Vgpr(10 | 11))).map(|(i, _)| i).collect();
        for block in ir.blocks.values_mut() { if let Term::Ret(args) = &mut block.term { *args = keep.iter().map(|&i| args[i]).collect(); } }
        let exec_index = crate::rdna_spmd::compiler::exec_index(&f.parameter_inputs, &registry);
        let count = |ir: &Func, name: &str| ir.blocks.values().flat_map(|b| &b.insts).filter(|i| matches!(i, Inst::Target { op, .. } if registry.lookup(crate::rdna_spmd::targets::rdna4::dialect::ID, name).ok() == Some(*op))).count();
        assert_eq!((count(&ir, "rsq.f64"), count(&ir, "sqrt.f64"), count(&ir, "ldexp.f64")), (1, 0, 2));
        for _ in 0..3 {
            let constants = crate::rdna_spmd::analysis::constants(&ir);
            let masks = crate::rdna_spmd::analysis::masks::analyze(&registry, &ir, exec_index, &constants, 32, false);
            run(&mut ir, &masks, &constants, &SqrtIdioms::new(&registry));
            let masks = crate::rdna_spmd::analysis::masks::analyze(&registry, &ir, exec_index, &constants, 32, false);
            crate::rdna_spmd::pass::dce::dead_writes(&mut ir, &masks, exec_index);
            crate::rdna_spmd::pass::simplify::run(&mut ir);
            crate::rdna_spmd::pass::dce::run(&mut ir);
        }
        ir.clone().verify_with(&registry).unwrap();
        assert_eq!((count(&ir, "rsq.f64"), count(&ir, "sqrt.f64"), count(&ir, "ldexp.f64")), (0, 1, 0));
    }
}
