use crate::rdna_spmd::analysis::masks::Predication;
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
    fn rewrite(&self, f: &mut Func, masks: &Predication, constants: &[Option<u64>]) -> usize { run(f, masks, constants, self) }
}

struct View<'a> {
    f: &'a Func,
    defs: Vec<Option<Op>>,
    targets: Vec<Option<(TargetOp, usize, Vec<ValueId>)>>,
    ballots: Vec<Option<ValueId>>,
    masks: &'a Predication,
    constants: &'a [Option<u64>],
}

impl<'a> View<'a> {
    fn new(f: &'a Func, masks: &'a Predication, constants: &'a [Option<u64>]) -> Self {
        let mut defs = vec![None; f.types.len()];
        let mut targets = vec![None; f.types.len()];
        let mut ballots = vec![None; f.types.len()];
        for block in f.blocks.values() {
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, op, .. } => defs[value.0] = Some(*op),
                    Inst::Effect { op: EffectOp::Wave(WaveOp::Ballot), inputs, outputs, .. } => {
                        ballots[outputs[0].0.0] = Some(inputs[0]);
                    }
                    Inst::Target { op, args, outputs, .. } => for (index, out) in outputs.iter().enumerate() {
                        targets[out.0 .0] = Some((*op, index, args.values().to_vec()));
                    },
                    _ => {}
                }
            }
        }
        Self { f, defs, targets, ballots, masks, constants }
    }
    fn alias(&self, mut v: ValueId) -> ValueId {
        while let Some(Op::Convert(Cvt::Bitcast, to, a)) = self.defs[v.0] {
            if self.f.types[a.0] != to { break; }
            v = a;
        }
        v
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
        self.result(v, wanted, 0)
    }
    fn result(&self, v: ValueId, wanted: TargetOp, index: usize) -> Option<&[ValueId]> {
        let (op, produced, args) = self.targets[self.raw(v).0].as_ref()?;
        (wanted == *op && *produced == index).then_some(args.as_slice())
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
    fn exec_at(&self, block: BlockId, index: usize) -> Option<ValueId> {
        let entries = self.masks.chain.get(&block)?;
        entries.iter().rev().find(|(at, _)| *at <= index).map(|&(_, v)| v)
    }
    fn same(&self, a: ValueId, b: ValueId) -> bool {
        let (a, b) = (self.raw(a), self.raw(b));
        a == b || (self.f.types[a.0] == self.f.types[b.0]
            && self.constants[a.0].is_some() && self.constants[a.0] == self.constants[b.0])
    }
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

pub(in crate::rdna_spmd) struct DivisionIdioms { fixup: TargetOp, fmas: TargetOp, scale: TargetOp, rcp: TargetOp }
impl DivisionIdioms {
    pub(in crate::rdna_spmd) fn new(registry: &DialectRegistry) -> Self {
        let op = |name: &str| registry.lookup(super::ID, name).expect("missing RDNA4 provider");
        Self { fixup: op("div_fixup.f64"), fmas: op("div_fmas.f64"), scale: op("div_scale.f64"), rcp: op("rcp.f64") }
    }
}
impl Idiom for DivisionIdioms {
    fn rewrite(&self, f: &mut Func, masks: &Predication, constants: &[Option<u64>]) -> usize {
        let count = divisions(f, masks, constants, self);
        count + quotient_fixups(f, self.fixup)
    }
}

/// A quotient formed from these exact operands already has the right sign
/// and handles zero/infinity. Keep the ISA-specific NaN and tiny-result rules
/// as ordinary SSA operations; code generation needs no special case.
pub(super) fn quotient_fixups(f: &mut Func, fixup: TargetOp) -> usize {
    let defs = crate::rdna_spmd::analysis::masks::definitions(f);
    let alias = |mut value: ValueId| {
        while let Some(Op::Convert(Cvt::Bitcast, to, source)) = defs[value.0] {
            if to != f.types[source.0] { break; }
            value = source;
        }
        value
    };
    let mut sites: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            let Inst::Target { op, args, .. } = inst else { continue };
            if *op != fixup { continue; }
            let a = args.values();
            // Do not strip predicated selects: an inactive lane may still
            // hold an old quotient or different denominator/numerator.
            if let Some(Op::Float(FloatOp::Div, numerator, denominator)) = defs[alias(a[0]).0] {
                if alias(numerator) == alias(a[2]) && alias(denominator) == alias(a[1]) {
                    sites.entry(id).or_default().push(index);
                }
            }
        }
    }
    let count = sites.values().map(Vec::len).sum();
    for (id, indices) in sites {
        for index in indices.into_iter().rev() {
            let Inst::Target { args, outputs, .. } = &f.blocks[&id].insts[index] else { unreachable!() };
            let a = args.values();
            let (quotient, denominator, numerator, result) = (a[0], a[1], a[2], outputs[0].0);
            let insts = fixup_division(f, quotient, denominator, numerator, result);
            f.blocks.get_mut(&id).unwrap().insts.splice(index..=index, insts);
        }
    }
    count
}

fn fixup_division(f: &mut Func, quotient: ValueId, denominator: ValueId, numerator: ValueId, result: ValueId) -> Vec<Inst> {
    let mut insts = Vec::new();
    let mut push = |ty, op| {
        let value = f.value(ty);
        insts.push(Inst::Core { value, ty, op });
        value
    };
    let zero = push(Ty::I64, Op::Const(Ty::I64, 0));
    let sign = push(Ty::I64, Op::Const(Ty::I64, 0x8000_0000_0000_0000));
    let magnitude = push(Ty::I64, Op::Const(Ty::I64, 0x7FFF_FFFF_FFFF_FFFF));
    let infinity = push(Ty::I64, Op::Const(Ty::I64, 0x7FF0_0000_0000_0000));
    let quiet = push(Ty::I64, Op::Const(Ty::I64, 0x0008_0000_0000_0000));
    let invalid = push(Ty::I64, Op::Const(Ty::I64, 0xFFF8_0000_0000_0000));
    let fraction = push(Ty::I64, Op::Const(Ty::I64, 52));
    let exponent_mask = push(Ty::I64, Op::Const(Ty::I64, 0x7FF));
    let tiny_limit = push(Ty::I64, Op::Const(Ty::I64, (-1075i64) as u64));
    let den = push(Ty::I64, Op::Convert(Cvt::Bitcast, Ty::I64, denominator));
    let num = push(Ty::I64, Op::Convert(Cvt::Bitcast, Ty::I64, numerator));
    let abs_den = push(Ty::I64, Op::Int(IntOp::And, den, magnitude));
    let abs_num = push(Ty::I64, Op::Int(IntOp::And, num, magnitude));
    let den_nan = push(Ty::I1, Op::Cmp(IntPred::Ugt, abs_den, infinity));
    let num_nan = push(Ty::I1, Op::Cmp(IntPred::Ugt, abs_num, infinity));
    let den_zero = push(Ty::I1, Op::Cmp(IntPred::Eq, abs_den, zero));
    let num_zero = push(Ty::I1, Op::Cmp(IntPred::Eq, abs_num, zero));
    let both_zero = push(Ty::I1, Op::Int(IntOp::And, den_zero, num_zero));
    let den_inf = push(Ty::I1, Op::Cmp(IntPred::Eq, abs_den, infinity));
    let num_inf = push(Ty::I1, Op::Cmp(IntPred::Eq, abs_num, infinity));
    let both_inf = push(Ty::I1, Op::Int(IntOp::And, den_inf, num_inf));
    let den_exp = push(Ty::I64, Op::Int(IntOp::LShr, den, fraction));
    let den_exp = push(Ty::I64, Op::Int(IntOp::And, den_exp, exponent_mask));
    let num_exp = push(Ty::I64, Op::Int(IntOp::LShr, num, fraction));
    let num_exp = push(Ty::I64, Op::Int(IntOp::And, num_exp, exponent_mask));
    let delta = push(Ty::I64, Op::Int(IntOp::Sub, num_exp, den_exp));
    let tiny = push(Ty::I1, Op::Cmp(IntPred::Slt, delta, tiny_limit));
    let sign_out = push(Ty::I64, Op::Int(IntOp::Xor, den, num));
    let signed_zero = push(Ty::I64, Op::Int(IntOp::And, sign_out, sign));
    let invalid_operands = push(Ty::I1, Op::Int(IntOp::Or, both_zero, both_inf));
    let fixed = push(Ty::I64, Op::Select(invalid_operands, invalid, signed_zero));
    let quiet_den = push(Ty::I64, Op::Int(IntOp::Or, den, quiet));
    let quiet_num = push(Ty::I64, Op::Int(IntOp::Or, num, quiet));
    // Reverse ISA priority: numerator NaN wins over denominator NaN, then
    // invalid operand pairs, then the exponent-based signed-zero correction.
    let fixed = push(Ty::I64, Op::Select(den_nan, quiet_den, fixed));
    let fixed = push(Ty::I64, Op::Select(num_nan, quiet_num, fixed));
    let fix = push(Ty::I1, Op::Int(IntOp::Or, tiny, invalid_operands));
    let nan = push(Ty::I1, Op::Int(IntOp::Or, den_nan, num_nan));
    let fix = push(Ty::I1, Op::Int(IntOp::Or, fix, nan));
    let quotient = push(Ty::I64, Op::Convert(Cvt::Bitcast, Ty::I64, quotient));
    let bits = push(Ty::I64, Op::Select(fix, fixed, quotient));
    insts.push(Inst::Core { value: result, ty: Ty::F64, op: Op::Convert(Cvt::Bitcast, Ty::F64, bits) });
    insts
}

fn reciprocal(view: &View, r: ValueId, d: ValueId, ops: &DivisionIdioms) -> bool {
    if let Some(args) = view.target(r, ops.rcp) { return view.same(args[0], d); }
    let Some((previous, error, added)) = view.fma(r) else { return false };
    if !view.same(previous, added) { return false; }
    let Some((negated, refined, one)) = view.fma(error) else { return false };
    let Some(positive) = view.neg(negated) else { return false };
    view.same(positive, d) && view.same(refined, previous) && view.constant_f64(one, 1.0)
        && reciprocal(view, previous, d, ops)
}

fn scale_flag(view: &View, v: ValueId, denominator: ValueId, numerator: ValueId, exec: ValueId, ops: &DivisionIdioms) -> bool {
    if let Some(args) = view.result(v, ops.scale, 1) {
        return (view.same(args[0], denominator) || view.same(args[0], numerator))
            && view.same(args[1], denominator) && view.same(args[2], numerator);
    }
    let raw = view.raw(v);
    // Packing a lane flag into its wave word and projecting that same lane
    // preserves the flag. A packet ballot or a different lane is not enough.
    // Only strip bitcast aliases here: a selected word may contain old bits.
    if let Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) = view.defs[raw.0] {
        if let Some(Op::Int(IntOp::LShr, word, lane)) = view.defs[view.alias(shifted).0] {
            if matches!(view.defs[view.alias(lane).0], Some(Op::Env(Env::LaneId))) {
                if let Some(bit) = view.ballots[view.alias(word).0] {
                    return scale_flag(view, bit, denominator, numerator, exec, ops);
                }
            }
        }
    }
    let Some(bit) = view.masks.masked[raw.0].then(|| view.masks.masked_result[raw.0]).flatten() else { return false };
    let Some(Op::Int(IntOp::And, a, b)) = view.defs[raw.0] else { return false };
    let mask = if view.same(a, bit) { b } else { a };
    view.same(mask, exec) && scale_flag(view, bit, denominator, numerator, exec, ops)
}

fn division_macro(view: &View, quotient: ValueId, denominator: ValueId, numerator: ValueId, exec: ValueId, ops: &DivisionIdioms) -> bool {
    let scaled = |v: ValueId, first: ValueId| match view.target(v, ops.scale) {
        Some(args) => view.same(args[0], first) && view.same(args[1], denominator) && view.same(args[2], numerator),
        None => false,
    };
    let Some(fmas) = view.target(quotient, ops.fmas) else { return false };
    let (error, refined, approximate) = (fmas[0], fmas[1], fmas[2]);
    let scaling = scale_flag(view, fmas[3], denominator, numerator, exec, ops);
    if !scaling { return false; }
    let Some(product) = view.mul(approximate) else { return false };
    let Some((negated, multiplied, scaled_numerator)) = view.fma(error) else { return false };
    let Some(scaled_denominator) = view.neg(negated) else { return false };
    either(view, product, scaled_numerator, refined)
        && view.same(multiplied, approximate)
        && scaled(scaled_denominator, denominator)
        && scaled(scaled_numerator, numerator)
        && reciprocal(view, refined, scaled_denominator, ops)
}

fn divisions(f: &mut Func, masks: &Predication, constants: &[Option<u64>], ops: &DivisionIdioms) -> usize {
    let mut sites: Vec<(BlockId, usize, ValueId, ValueId)> = Vec::new();
    {
        let view = View::new(f, masks, constants);
        for (&id, block) in &f.blocks {
            for (index, inst) in block.insts.iter().enumerate() {
                let Inst::Target { op, args, .. } = inst else { continue };
                if *op != ops.fixup { continue; }
                let args = args.values();
                let (quotient, denominator, numerator) = (args[0], args[1], args[2]);
                let Some(exec) = view.exec_at(id, index) else { continue };
                if division_macro(&view, quotient, denominator, numerator, exec, ops) {
                    sites.push((id, index, denominator, numerator));
                }
            }
        }
    }
    if sites.is_empty() { return 0; }
    let mut by_block: BTreeMap<BlockId, Vec<(usize, ValueId, ValueId, ValueId)>> = BTreeMap::new();
    for &(block, index, denominator, numerator) in &sites {
        let direct = f.value(Ty::F64);
        by_block.entry(block).or_default().push((index, direct, denominator, numerator));
    }
    for (block, mut sites) in by_block {
        sites.sort_by_key(|(index, ..)| std::cmp::Reverse(*index));
        let b = f.blocks.get_mut(&block).unwrap();
        for (index, direct, denominator, numerator) in sites {
            if let Inst::Target { args, .. } = &mut b.insts[index] {
                let mut values: Vec<ValueId> = args.values().to_vec();
                values[0] = direct;
                *args = Arguments::Ternary([values[0], values[1], values[2]]);
            }
            b.insts.insert(index, Inst::Core { value: direct, ty: Ty::F64, op: Op::Float(FloatOp::Div, numerator, denominator) });
        }
    }
    sites.len()
}

fn run(f: &mut Func, masks: &Predication, constants: &[Option<u64>], ops: &SqrtIdioms) -> usize {
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
    use crate::rdna_instructions::{InstFormat, SOP1, SOP2, SourceOperand, VOP1, VOP2, VOP3, VOP3SD, VOPC};
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
            let masks = crate::rdna_spmd::analysis::masks::predication(&ir, exec_index, &constants);
            run(&mut ir, &masks, &constants, &SqrtIdioms::new(&registry));
            let masks = crate::rdna_spmd::analysis::masks::analyze(&registry, &ir, exec_index, &constants, 32, false);
            crate::rdna_spmd::pass::dce::dead_writes(&mut ir, &masks, exec_index);
            crate::rdna_spmd::pass::simplify::run(&mut ir);
            crate::rdna_spmd::pass::dce::run(&mut ir);
        }
        ir.clone().verify_with(&registry).unwrap();
        assert_eq!((count(&ir, "rsq.f64"), count(&ir, "sqrt.f64"), count(&ir, "ldexp.f64")), (0, 1, 0));
    }

    fn division_body(flag: u8) -> Vec<InstFormat> {
        let scale = |dst: u8, a: SourceOperand, b: u8, c: SourceOperand| InstFormat::VOP3SD(VOP3SD {
            op: I::V_DIV_SCALE_F64, vdst: dst, sdst: 106, src0: a, src1: SourceOperand::VectorRegister(b), src2: c,
            neg: 0, cm: 0, omod: 0 });
        let one = SourceOperand::FloatConstant(1.0);
        let mut body = vec![
            scale(4, v(2), 2, v(0)),
            scale(6, v(0), 2, v(0)),
            InstFormat::VOP1(VOP1 { op: I::V_RCP_F64, vdst: 8, src0: v(4) }),
            fma(10, v(4), v(8), one.clone(), 1),
            fma(8, v(8), v(10), v(8), 0),
            fma(10, v(4), v(8), one, 1),
            fma(8, v(8), v(10), v(8), 0),
            mul(12, v(6), 8),
            fma(14, v(4), v(12), v(6), 1),
        ];
        if flag == 1 {
            body.push(InstFormat::VOPC(VOPC { op: I::V_CMP_EQ_U32, src0: SourceOperand::IntegerConstant(0), vsrc1: 0 }));
        }
        if flag == 3 {
            body.insert(0, InstFormat::SOP1(SOP1 { op: I::S_AND_SAVEEXEC_B32, sdst: 2, ssrc0: SourceOperand::ScalarRegister(3) }));
            body.push(InstFormat::SOP1(SOP1 { op: I::S_MOV_B32, sdst: 126, ssrc0: SourceOperand::IntegerConstant(u32::MAX as u64) }));
        }
        if flag == 2 {
            body.push(InstFormat::SOP2(SOP2 { op: I::S_AND_B32, sdst: 106,
                ssrc0: SourceOperand::ScalarRegister(106), ssrc1: SourceOperand::ScalarRegister(20) }));
        }
        body.push(InstFormat::VOP3(VOP3 { op: I::V_DIV_FMAS_F64, vdst: 16, src0: v(14), src1: v(8), src2: v(12),
            neg: 0, abs: 0, cm: 0, omod: 0, opsel: 0 }));
        body.push(InstFormat::VOP3(VOP3 { op: I::V_DIV_FIXUP_F64, vdst: 18, src0: v(16), src1: v(2), src2: v(0),
            neg: 0, abs: 0, cm: 0, omod: 0, opsel: 0 }));
        body
    }

    fn division_fixups(flag: u8, projection: u8) -> usize {
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock { pc: 0, body: division_body(flag), term: Terminator::Return })]) };
        let f = program.to_ssa().function;
        let registry = f.registry.clone();
        let mut ir = f.ir.clone();
        if projection != 0 {
            // Exercise the SSA word round trip used when VCC is observed as
            // an SGPR before FMAS consumes its lane bit.
            let fmas = registry.lookup(super::super::ID, "div_fmas.f64").unwrap();
            let (index, inputs) = ir.blocks[&BlockId(0)].insts.iter().enumerate().find_map(|(index, inst)| {
                match inst { Inst::Target { op, args, .. } if *op == fmas => Some((index, args.values().to_vec())), _ => None }
            }).unwrap();
            let mut inserted = Vec::new();
            let word = ir.value(Ty::I32);
            inserted.push(if projection == 4 {
                Inst::Packet { op: PacketOp::Ballot, input: inputs[3], output: word }
            } else {
                Inst::Effect { provenance: 999, op: EffectOp::Wave(WaveOp::Ballot), inputs: vec![inputs[3]], outputs: vec![(word, Ty::I32)] }
            });
            let lane = ir.value(Ty::I32);
            inserted.push(Inst::Core { value: lane, ty: Ty::I32, op: if projection == 2 { Op::Const(Ty::I32, 0) } else { Op::Env(Env::LaneId) } });
            let word = if projection == 3 {
                let mask = ir.value(Ty::I32); let narrowed = ir.value(Ty::I32);
                inserted.push(Inst::Core { value: mask, ty: Ty::I32, op: Op::Const(Ty::I32, 0x55555555) });
                inserted.push(Inst::Core { value: narrowed, ty: Ty::I32, op: Op::Int(IntOp::And, word, mask) });
                narrowed
            } else { word };
            let shifted = ir.value(Ty::I32); let bit = ir.value(Ty::I1);
            inserted.push(Inst::Core { value: shifted, ty: Ty::I32, op: Op::Int(IntOp::LShr, word, lane) });
            inserted.push(Inst::Core { value: bit, ty: Ty::I1, op: Op::Convert(Cvt::Trunc, Ty::I1, shifted) });
            let block = ir.blocks.get_mut(&BlockId(0)).unwrap();
            if let Inst::Target { args, .. } = &mut block.insts[index] {
                *args = Arguments::Quaternary([inputs[0], inputs[1], inputs[2], bit]);
            }
            block.insts.splice(index..index, inserted);
        }
        let exec_index = crate::rdna_spmd::compiler::exec_index(&f.parameter_inputs, &registry);
        let constants = crate::rdna_spmd::analysis::constants(&ir);
        let masks = crate::rdna_spmd::analysis::masks::predication(&ir, exec_index, &constants);
        let collapsed = divisions(&mut ir, &masks, &constants, &DivisionIdioms::new(&registry));
        let fixup = registry.lookup(super::super::ID, "div_fixup.f64").unwrap();
        assert_eq!(quotient_fixups(&mut ir, fixup), collapsed, "each recognized macro gets its SSA correction");
        ir.clone().verify_with(&registry).unwrap();
        collapsed
    }

    #[test]
    fn a_division_macro_collapses_only_when_its_own_scale_sets_the_fmas_flag() {
        for projection in [0, 1] {
            assert_eq!(division_fixups(0, projection), 1, "the macro's own scale sets the flag, so it is a division");
            assert_eq!(division_fixups(1, projection), 0, "a foreign flag scales the result by 2^64, so it is not");
            assert_eq!(division_fixups(2, projection), 0, "a flag narrowed by anything but the lane mask changes the scaling too");
            assert_eq!(division_fixups(3, projection), 0, "lanes reactivated after the flag was made, so its cleared bits still count");
        }
        assert_eq!(division_fixups(0, 2), 0, "another lane's bit cannot prove this lane's scaling");
        assert_eq!(division_fixups(0, 3), 0, "masking the wave word changes the scaling");
        assert_eq!(division_fixups(0, 4), 0, "a packet ballot is not a wave word");
    }
}
