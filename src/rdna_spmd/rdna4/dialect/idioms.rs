use crate::rdna_spmd::pass::Predication;
use crate::rdna_spmd::dialect::{Arguments, DialectRegistry, TargetOp};
use crate::rdna_spmd::ir::{FloatOp, FloatUnary, Op, Ty, ValueId, *};
use crate::rdna_spmd::pass::Idiom;
use std::collections::BTreeMap;

pub struct SqrtIdioms {
    sqrt: TargetOp,
    rsq: TargetOp,
    ldexp: TargetOp,
    class: TargetOp,
}
impl SqrtIdioms {
    pub fn new(registry: &DialectRegistry) -> Self {
        let op = |name: &str| {
            registry
                .lookup(super::ID, name)
                .expect("missing RDNA4 provider")
        };
        Self {
            sqrt: op("sqrt.f64"),
            rsq: op("rsq.f64"),
            ldexp: op("ldexp.f64"),
            class: op("cmp_class.f64"),
        }
    }
}
impl Idiom for SqrtIdioms {
    fn rewrite(&self, f: &mut Func, masks: &Predication, constants: &[Option<u64>]) -> usize {
        run(f, masks, constants, self)
    }
}

struct View<'a> {
    f: &'a Func,
    defs: Vec<Option<Op>>,
    targets: Vec<Option<(TargetOp, usize, Vec<ValueId>)>>,
    ballots: Vec<Option<ValueId>>,
    anys: Vec<Option<ValueId>>,
    masks: &'a Predication,
    constants: &'a [Option<u64>],
}

impl<'a> View<'a> {
    fn new(f: &'a Func, masks: &'a Predication, constants: &'a [Option<u64>]) -> Self {
        let mut defs = vec![None; f.types.len()];
        let mut targets = vec![None; f.types.len()];
        let mut ballots = vec![None; f.types.len()];
        let mut anys = vec![None; f.types.len()];
        for block in f.blocks.values() {
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, op, .. } => defs[value.0] = Some(*op),
                    Inst::Effect {
                        op: EffectOp::Wave(query @ (WaveOp::Ballot | WaveOp::Any)),
                        inputs,
                        outputs,
                        ..
                    } => {
                        let asked = if *query == WaveOp::Ballot {
                            &mut ballots
                        } else {
                            &mut anys
                        };
                        asked[outputs[0].0 .0] = Some(inputs[0]);
                    }
                    Inst::Target {
                        op, args, outputs, ..
                    } => {
                        for (index, out) in outputs.iter().enumerate() {
                            targets[out.0 .0] = Some((*op, index, args.values().to_vec()));
                        }
                    }
                    _ => {}
                }
            }
        }
        Self {
            f,
            defs,
            targets,
            ballots,
            anys,
            masks,
            constants,
        }
    }
    fn alias(&self, mut v: ValueId) -> ValueId {
        while let Some(Op::Convert(Cvt::Bitcast, to, a)) = self.defs[v.0] {
            if self.f.types[a.0] != to {
                break;
            }
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
        match self.defs[self.raw(v).0] {
            Some(Op::Unary(FloatUnary::Neg, a)) => Some(self.raw(a)),
            _ => None,
        }
    }
    fn constant_f64(&self, v: ValueId, expected: f64) -> bool {
        self.constants[self.raw(v).0] == Some(expected.to_bits())
    }
    fn mul(&self, v: ValueId) -> Option<(ValueId, ValueId)> {
        match self.defs[self.raw(v).0] {
            Some(Op::Float(FloatOp::Mul, a, b)) => Some((self.raw(a), self.raw(b))),
            _ => None,
        }
    }
    fn fma(&self, v: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
        match self.defs[self.raw(v).0] {
            Some(Op::MulAdd(a, b, c)) | Some(Op::Fma(a, b, c)) => Some((a, b, c)),
            _ => None,
        }
    }
    fn exec_at(&self, block: BlockId, index: usize) -> Option<ValueId> {
        let entries = self.masks.chain.get(&block)?;
        entries
            .iter()
            .rev()
            .find(|(at, _)| *at <= index)
            .map(|&(_, v)| v)
    }
    fn same(&self, a: ValueId, b: ValueId) -> bool {
        let (a, b) = (self.raw(a), self.raw(b));
        a == b
            || (self.f.types[a.0] == self.f.types[b.0]
                && self.constants[a.0].is_some()
                && self.constants[a.0] == self.constants[b.0])
    }
}

fn either(view: &View, pair: (ValueId, ValueId), x: ValueId, y: ValueId) -> bool {
    (view.same(pair.0, x) && view.same(pair.1, y)) || (view.same(pair.0, y) && view.same(pair.1, x))
}

fn sqrt_chain(view: &View, result: ValueId, ops: &SqrtIdioms) -> Option<ValueId> {
    let (b3, r3, a3) = view.fma(result)?;
    let (na3, a3b, x) = view.fma(b3)?;
    let a3n = view.neg(na3)?;
    if !view.same(a3n, a3) || !view.same(a3b, a3) {
        return None;
    }
    let (b2, r3b, a2) = view.fma(a3)?;
    if !view.same(r3b, r3) {
        return None;
    }
    let (na2, a2b, xb) = view.fma(b2)?;
    let a2n = view.neg(na2)?;
    if !view.same(a2n, a2) || !view.same(a2b, a2) || !view.same(xb, x) {
        return None;
    }
    let (r2, b, r2b) = view.fma(r3)?;
    if !view.same(r2b, r2) {
        return None;
    }
    let (a, bb, ab) = view.fma(a2)?;
    if !view.same(bb, b) || !view.same(ab, a) {
        return None;
    }
    let (nr2, ac, half) = view.fma(b)?;
    let r2n = view.neg(nr2)?;
    if !view.same(r2n, r2) || !view.same(ac, a) || !view.constant_f64(half, 0.5) {
        return None;
    }
    let (h, r) = view.mul(r2)?;
    let (h, r) = if view.constant_f64(h, 0.5) {
        (h, r)
    } else if view.constant_f64(r, 0.5) {
        (r, h)
    } else {
        return None;
    };
    let _ = h;
    if !either(view, view.mul(a)?, x, r) {
        return None;
    }
    let rsq = view.target(r, ops.rsq)?;
    if !view.same(rsq[0], x) {
        return None;
    }
    Some(view.raw(x))
}

fn exponent_select(view: &View, v: ValueId) -> Option<(ValueId, i32, i32)> {
    let raw = view.raw(v);
    let Some(Op::Select(c, a, b)) = view.defs[raw.0] else {
        let k = view.constants[raw.0]? as u32 as i32;
        return Some((ValueId(usize::MAX), k, k));
    };
    let (ka, kb) = (
        view.constants[view.raw(a).0]? as u32 as i32,
        view.constants[view.raw(b).0]? as u32 as i32,
    );
    Some((view.raw(c), ka, kb))
}

fn scale(view: &View, v: ValueId, ops: &SqrtIdioms) -> Option<(ValueId, (ValueId, i32, i32))> {
    if let Some(args) = view.target(v, ops.ldexp) {
        return Some((view.raw(args[0]), exponent_select(view, args[1])?));
    }
    let Some(Op::Select(c, a, b)) = view.defs[view.raw(v).0] else {
        return None;
    };
    for (scaled, plain, flipped) in [(a, b, false), (b, a, true)] {
        let Some(args) = view.target(scaled, ops.ldexp) else {
            continue;
        };
        if !view.same(args[0], plain) {
            continue;
        }
        let k = view.constants[view.raw(args[1]).0]? as u32 as i32;
        return Some((
            view.raw(plain),
            (
                view.raw(c),
                if flipped { 0 } else { k },
                if flipped { k } else { 0 },
            ),
        ));
    }
    None
}

fn scaled_sqrt(view: &View, out: ValueId, ops: &SqrtIdioms) -> Option<ValueId> {
    let (root, (c2, a2, b2)) = scale(view, out, ops)?;
    let sqrt = view.target(root, ops.sqrt)?;
    let (x, (c1, a1, b1)) = scale(view, sqrt[0], ops)?;
    if c1 != c2 {
        return None;
    }
    let halves = |e: i32, e2: i32| e % 2 == 0 && e2 == -(e / 2) && e.abs() <= 1022;
    if !(halves(a1, a2) && halves(b1, b2)) {
        return None;
    }
    Some(view.raw(x))
}

const ROOT_FIXED_CLASSES: u64 = (1 << 1) | (1 << 5) | (1 << 6) | (1 << 9);

fn bits_of(view: &View, v: ValueId) -> ValueId {
    let mut v = view.raw(v);
    while let Some(Op::Convert(Cvt::Bitcast, _, a)) = view.defs[v.0] {
        v = view.raw(a);
    }
    v
}

fn same_question(view: &View, a: ValueId, b: ValueId) -> bool {
    if a == b {
        return true;
    }
    if a.0 >= view.f.types.len() || b.0 >= view.f.types.len() {
        return false;
    }
    match (view.anys[view.raw(a).0], view.anys[view.raw(b).0]) {
        (Some(x), Some(y)) => view.same(x, y),
        _ => false,
    }
}

fn guarded_sqrt(view: &View, out: ValueId, ops: &SqrtIdioms) -> Option<ValueId> {
    let Some(Op::Select(c, fixed, rescaled)) = view.defs[view.raw(out).0] else {
        return None;
    };
    let c = view.masks.masked_result[view.raw(c).0].unwrap_or(c);
    let class = view.target(c, ops.class)?;
    let selector = view.constants[view.raw(class[1]).0]?;
    if selector & !ROOT_FIXED_CLASSES != 0 {
        return None;
    }
    let scaled = bits_of(view, class[0]);
    if bits_of(view, fixed) != scaled {
        return None;
    }
    let (root, (down, d1, d2)) = scale(view, bits_of(view, rescaled), ops)?;
    let radicand = view.target(root, ops.sqrt)?;
    if bits_of(view, radicand[0]) != scaled {
        return None;
    }
    let (x, (up, u1, u2)) = scale(view, scaled, ops)?;
    if !same_question(view, up, down) {
        return None;
    }
    let halves = |e: i32, e2: i32| e % 2 == 0 && e2 == -(e / 2) && e.abs() <= 1022;
    if !(halves(u1, d1) && halves(u2, d2)) {
        return None;
    }
    Some(view.raw(x))
}

pub struct DivisionIdioms {
    fixup: TargetOp,
    fmas: TargetOp,
    scale: TargetOp,
    rcp: TargetOp,
}
impl DivisionIdioms {
    pub fn new(registry: &DialectRegistry) -> Self {
        let op = |name: &str| {
            registry
                .lookup(super::ID, name)
                .expect("missing RDNA4 provider")
        };
        Self {
            fixup: op("div_fixup.f64"),
            fmas: op("div_fmas.f64"),
            scale: op("div_scale.f64"),
            rcp: op("rcp.f64"),
        }
    }
}
impl Idiom for DivisionIdioms {
    fn rewrite(&self, f: &mut Func, masks: &Predication, constants: &[Option<u64>]) -> usize {
        let count = divisions(f, masks, constants, self);
        count + quotient_fixups(f, self.fixup)
    }
}

fn quotient_fixups(f: &mut Func, fixup: TargetOp) -> usize {
    let defs = f.definitions();
    let alias = |mut value: ValueId| {
        while let Some(Op::Convert(Cvt::Bitcast, to, source)) = defs[value.0] {
            if to != f.types[source.0] {
                break;
            }
            value = source;
        }
        value
    };
    let mut sites: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            let Inst::Target { op, args, .. } = inst else {
                continue;
            };
            if *op != fixup {
                continue;
            }
            let a = args.values();

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
            let Inst::Target { args, outputs, .. } = &f.blocks[&id].insts[index] else {
                unreachable!()
            };
            let a = args.values();
            let (quotient, denominator, numerator, result) = (a[0], a[1], a[2], outputs[0].0);
            let insts = fixup_division(f, quotient, denominator, numerator, result);
            f.blocks
                .get_mut(&id)
                .unwrap()
                .insts
                .splice(index..=index, insts);
        }
    }
    count
}

fn fixup_division(
    f: &mut Func,
    quotient: ValueId,
    denominator: ValueId,
    numerator: ValueId,
    result: ValueId,
) -> Vec<Inst> {
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

    let fixed = push(Ty::I64, Op::Select(den_nan, quiet_den, fixed));
    let fixed = push(Ty::I64, Op::Select(num_nan, quiet_num, fixed));
    let fix = push(Ty::I1, Op::Int(IntOp::Or, tiny, invalid_operands));
    let nan = push(Ty::I1, Op::Int(IntOp::Or, den_nan, num_nan));
    let fix = push(Ty::I1, Op::Int(IntOp::Or, fix, nan));
    let quotient = push(Ty::I64, Op::Convert(Cvt::Bitcast, Ty::I64, quotient));
    let bits = push(Ty::I64, Op::Select(fix, fixed, quotient));
    insts.push(Inst::Core {
        value: result,
        ty: Ty::F64,
        op: Op::Convert(Cvt::Bitcast, Ty::F64, bits),
    });
    insts
}

fn reciprocal(view: &View, r: ValueId, d: ValueId, ops: &DivisionIdioms) -> bool {
    if let Some(args) = view.target(r, ops.rcp) {
        return view.same(args[0], d);
    }
    let Some((previous, error, added)) = view.fma(r) else {
        return false;
    };
    if !view.same(previous, added) {
        return false;
    }
    let Some((negated, refined, one)) = view.fma(error) else {
        return false;
    };
    let Some(positive) = view.neg(negated) else {
        return false;
    };
    view.same(positive, d)
        && view.same(refined, previous)
        && view.constant_f64(one, 1.0)
        && reciprocal(view, previous, d, ops)
}

fn scale_flag(
    view: &View,
    v: ValueId,
    denominator: ValueId,
    numerator: ValueId,
    exec: ValueId,
    ops: &DivisionIdioms,
) -> bool {

    if let Some((new, mask)) = view.masks.predicated[view.alias(v).0] {
        return view.same(mask, exec)
            && scale_flag(view, new, denominator, numerator, exec, ops);
    }
    if let Some(args) = view.result(v, ops.scale, 1) {
        return (view.same(args[0], denominator) || view.same(args[0], numerator))
            && view.same(args[1], denominator)
            && view.same(args[2], numerator);
    }
    let raw = view.raw(v);

    if let Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) = view.defs[raw.0] {
        if let Some(Op::Int(IntOp::LShr, word, lane)) = view.defs[view.alias(shifted).0] {
            if matches!(view.defs[view.alias(lane).0], Some(Op::Env(Env::LaneId))) {
                if let Some(bit) = view.ballots[view.alias(word).0] {
                    return scale_flag(view, bit, denominator, numerator, exec, ops);
                }
            }
        }
    }
    let Some(bit) = view.masks.masked[raw.0]
        .then(|| view.masks.masked_result[raw.0])
        .flatten()
    else {
        return false;
    };
    let Some(Op::Int(IntOp::And, a, b)) = view.defs[raw.0] else {
        return false;
    };
    let mask = if view.same(a, bit) { b } else { a };
    view.same(mask, exec) && scale_flag(view, bit, denominator, numerator, exec, ops)
}

fn division_macro(
    view: &View,
    quotient: ValueId,
    denominator: ValueId,
    numerator: ValueId,
    exec: ValueId,
    ops: &DivisionIdioms,
) -> bool {
    let scaled = |v: ValueId, first: ValueId| match view.target(v, ops.scale) {
        Some(args) => {
            view.same(args[0], first)
                && view.same(args[1], denominator)
                && view.same(args[2], numerator)
        }
        None => false,
    };
    let Some(fmas) = view.target(quotient, ops.fmas) else {
        return false;
    };
    let (error, refined, approximate) = (fmas[0], fmas[1], fmas[2]);
    let scaling = scale_flag(view, fmas[3], denominator, numerator, exec, ops);
    if !scaling {
        return false;
    }
    let Some(product) = view.mul(approximate) else {
        return false;
    };
    let Some((negated, multiplied, scaled_numerator)) = view.fma(error) else {
        return false;
    };
    let Some(scaled_denominator) = view.neg(negated) else {
        return false;
    };
    either(view, product, scaled_numerator, refined)
        && view.same(multiplied, approximate)
        && scaled(scaled_denominator, denominator)
        && scaled(scaled_numerator, numerator)
        && reciprocal(view, refined, scaled_denominator, ops)
}

fn divisions(
    f: &mut Func,
    masks: &Predication,
    constants: &[Option<u64>],
    ops: &DivisionIdioms,
) -> usize {
    let mut sites: Vec<(BlockId, usize, ValueId, ValueId)> = Vec::new();
    {
        let view = View::new(f, masks, constants);
        for (&id, block) in &f.blocks {
            for (index, inst) in block.insts.iter().enumerate() {
                let Inst::Target { op, args, .. } = inst else {
                    continue;
                };
                if *op != ops.fixup {
                    continue;
                }
                let args = args.values();
                let (quotient, denominator, numerator) = (args[0], args[1], args[2]);
                let Some(exec) = view.exec_at(id, index) else {
                    continue;
                };
                if division_macro(&view, quotient, denominator, numerator, exec, ops) {
                    sites.push((id, index, denominator, numerator));
                }
            }
        }
    }
    if sites.is_empty() {
        return 0;
    }
    let mut by_block: BTreeMap<BlockId, Vec<(usize, ValueId, ValueId, ValueId)>> = BTreeMap::new();
    for &(block, index, denominator, numerator) in &sites {
        let direct = f.value(Ty::F64);
        by_block
            .entry(block)
            .or_default()
            .push((index, direct, denominator, numerator));
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
            b.insts.insert(
                index,
                Inst::Core {
                    value: direct,
                    ty: Ty::F64,
                    op: Op::Float(FloatOp::Div, numerator, denominator),
                },
            );
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
                    Inst::Core {
                        value, ty: Ty::F64, ..
                    } => *value,

                    Inst::Core {
                        value,
                        ty: Ty::I64,
                        op: Op::Select(..),
                    } => *value,
                    Inst::Target { outputs, .. }
                        if outputs.len() == 1 && outputs[0].1 == Ty::F64 =>
                    {
                        outputs[0].0
                    }
                    _ => continue,
                };
                if view.raw(value) != value {
                    continue;
                }
                let replacement = sqrt_chain(&view, value, ops)
                    .filter(|&x| scale(&view, x, ops).is_some())
                    .or_else(|| scaled_sqrt(&view, value, ops))
                    .or_else(|| guarded_sqrt(&view, value, ops));
                if let Some(x) = replacement {
                    rewrites.push((id, index, value, x));
                }
            }
        }
    }
    if rewrites.is_empty() {
        return 0;
    }
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut inserted: BTreeMap<BlockId, Vec<(usize, Inst)>> = BTreeMap::new();
    for &(block, index, value, x) in &rewrites {
        let root = f.value(Ty::F64);
        let mut replacement = root;

        if f.types[value.0] == Ty::I64 {
            let bits = f.value(Ty::I64);
            inserted.entry(block).or_default().push((
                index,
                Inst::Core {
                    value: bits,
                    ty: Ty::I64,
                    op: Op::Convert(Cvt::Bitcast, Ty::I64, root),
                },
            ));
            replacement = bits;
        }
        inserted.entry(block).or_default().push((
            index,
            Inst::Target {
                provenance: None,
                op: ops.sqrt,
                args: Arguments::Unary(x),
                outputs: vec![(root, Ty::F64)],
            },
        ));
        renames.insert(value, replacement);
    }
    for (block, mut insts) in inserted {
        insts.sort_by_key(|(index, _)| std::cmp::Reverse(*index));
        let b = f.blocks.get_mut(&block).unwrap();
        for (index, inst) in insts {
            b.insts.insert(index + 1, inst);
        }
    }
    f.rename(&renames);
    rewrites.len()
}
