use super::control::*;
use super::fields::*;
use super::form::*;
use super::memory::load;
use super::queries::*;
use super::symbols::Key;
use super::wide::known_high;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::ir::*;

pub(super) fn compute<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
    let (f, facts) = (q.program().f, q.program().facts);
    if let Some(&root) = q.program().copies.get(&v) {
        if q.program().narrowable[v.0] {
            if let Some(form) = narrowed_everywhere(q, v, lane) {
                return unassumed(Value::of(form));
            }
        }
        return q.value(root, lane, assume);
    }
    match facts.site[v.0] {
        Site::Param { block, index } if block == f.entry => unassumed(q.symbols_mut().input(v, index, lane)),
        Site::Param { block, index } => join(q, v, block, index, lane),
        Site::Inst { block, index } => match &f.blocks[&block].insts[index] {
            Inst::Core { op, .. } => core(q, v, *op, lane, assume),
            Inst::Effect {
                op, inputs, outputs, ..
            } => effect(q, v, *op, inputs, outputs, lane, assume),
            _ => unassumed(q.symbols_mut().opaque(v, lane, None)),
        },
        Site::Unreached => unassumed(q.symbols_mut().opaque(v, lane, None)),
    }
}

fn join<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, block: BlockId, index: usize, lane: usize) -> Assumed<Value> {
    let Some(arguments) = incoming_edges(q, v, block, index) else {
        if let Some((value, masked)) = stepped(q, v, block, index, lane) {
            if !masked || q.reason(|conditions, program| conditions.step_safe(program, v, block)) {
                return unassumed(value);
            }
        }
        if let Some(value) = q.recur(block, lane, Target::Param(index)) {
            return unassumed(value);
        }
        if let Some(value) = q.sequence(v, block, index, lane) {
            return unassumed(value);
        }
        let range = induction(q, v, block, index, lane);
        return unassumed(q.symbols_mut().opaque(v, lane, range));
    };
    let mut joined: Option<Value> = None;
    let mut region: Option<Option<Region>> = None;
    let mut agreed = true;
    let mut forms: Vec<Form> = Vec::new();
    let narrowable = q.program().narrowable[v.0] && !q.program().headers.contains(&block);
    for &(edge, a) in &arguments {
        let narrowed = if !narrowable {
            None
        } else if q.program().f.types[v.0] == Ty::I32 {
            narrowed_form(q, edge, a, block, lane)
        } else {
            q.reason(|conditions, program| conditions.narrowing(program, edge, a)).map(Form::constant)
        };
        let value = match narrowed {
            Some(form) => Value::of(form),
            None => q.operand(a, block, lane, None).0,
        };
        region = Some(match region {
            None => value.region,
            Some(r) if r == value.region => r,
            Some(_) => None,
        });
        forms.push(value.form.clone());
        match &joined {
            None => joined = Some(value),
            Some(old) if *old == value => {}
            Some(_) => agreed = false,
        }
    }
    if !agreed {
        let edges: Vec<(BlockId, usize)> = arguments.iter().map(|&(e, _)| e).collect();
        if let Some(form) = q.symbols_mut().selected(v, block, &edges, &forms, lane) {
            return unassumed(Value {
                form,
                region: region.flatten(),
            });
        }
        let shared = q.program().facts.uniform[v.0];
        let key = Key::Spread(v, if shared { 0 } else { lane as u8 }, forms.clone());
        let spread = q.symbols_mut().spread(key, shared, block, &forms);
        return unassumed(Value {
            region: region.flatten(),
            ..spread.map(Value::of).unwrap_or_else(|| q.symbols_mut().opaque(v, lane, None))
        });
    }
    unassumed(joined.unwrap_or_else(|| q.symbols_mut().opaque(v, lane, None)))
}

fn narrowed_form<'a, Q: Queries<'a>>(q: &mut Q, (pred, slot): (BlockId, usize), arg: ValueId, at: BlockId, lane: usize) -> Option<Form> {
    if !q.program().narrowing_edges.contains(&(pred, slot)) {
        return None;
    }
    if let Some(k) = q.reason(|conditions, program| conditions.narrowing(program, (pred, slot), arg)) {
        return Some(Form::constant(k));
    }
    let (cond, taken) = q.program().edge_condition(pred, slot)?;
    let fixed = q.reason(|conditions, program| conditions.fixed_words(program, cond, taken));
    let mut form = q.operand(arg, at, lane, None).0.form;
    let mut changed = false;
    for (x, k) in fixed {
        if q.program().f.types[x.0] != Ty::I32 {
            continue;
        }
        let fx = q.operand(x, at, lane, None).0.form;
        let &[(t, 1)] = fx.terms.as_slice() else {
            continue;
        };
        let Some(&(_, c)) = form.terms.iter().find(|&&(u, _)| u == t) else {
            continue;
        };
        let value = k.wrapping_sub(fx.constant);
        form = form.sub(&Form::unknown(t).scale(c)).add(&Form::constant(value.wrapping_mul(c)));
        changed = true;
    }
    changed.then_some(form)
}

fn narrowed_everywhere<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, lane: usize) -> Option<Form> {
    let Site::Param { block, index } = q.program().facts.site[v.0] else {
        return None;
    };
    if block == q.program().f.entry || !matches!(q.program().f.types[v.0], Ty::I32 | Ty::I64) {
        return None;
    }
    let facts = q.program().facts;
    let own = q.program().rank[&block];
    let root = |this: &Q, x: ValueId| this.program().copies.get(&x).copied().unwrap_or(x);
    let mut narrowed: Vec<(BlockId, usize, Option<Form>)> = Vec::new();
    for &(pred, slot) in &facts.incoming[&block] {
        let arg = q.program().edge_arg((pred, slot), index);
        if q.program().rank[&pred] >= own {
            if arg == v || root(q, arg) == root(q, v) {
                continue;
            }
            return None;
        }
        let form = if q.program().f.types[v.0] == Ty::I32 {
            narrowed_form(q, (pred, slot), arg, block, lane)
        } else {
            q.reason(|conditions, program| conditions.narrowing(program, (pred, slot), arg)).map(Form::constant)
        };
        narrowed.push((pred, slot, form));
    }
    if narrowed.iter().all(|(_, _, k)| k.is_none()) {
        return None;
    }
    let mut found: Option<Form> = None;
    for (pred, slot, k) in narrowed {
        if k.is_some() && k == found {
            continue;
        }
        if !can_take(q, pred, slot, block) {
            continue;
        }
        let k = k?;
        if found.is_some() {
            return None;
        }
        found = Some(k);
    }
    found
}

fn core<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, op: Op, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
    let ty = q.program().f.types[v.0];
    let wide = ty == Ty::I64;
    let at = q.program().block_of(v);
    let mut used = Reliance::default();
    macro_rules! get {
        ($this:expr, $x:expr) => {{
            let (value, u) = $this.operand($x, at, lane, assume);
            used |= u;
            value
        }};
    }
    let value = match op {
        Op::Const(_, k) => Value::constant(k as u32),
        Op::Env(Env::LaneId) => Value::constant(lane as u32),
        Op::Env(Env::ScratchBase) => q.symbols_mut().base(Region::Private),
        Op::Int(IntOp::Add, a, b) => {
            let (a, b) = (get!(q, a), get!(q, b));
            let region = match (a.region, b.region) {
                (Some(r), None) | (None, Some(r)) => Some(r),
                _ => None,
            };
            Value {
                form: a.form.add(&b.form),
                region,
            }
        }
        Op::Int(IntOp::Sub, a, b) => {
            let (a, b) = (get!(q, a), get!(q, b));
            let region = match (a.region, b.region) {
                (Some(r), None) => Some(r),
                _ => None,
            };
            Value {
                form: a.form.sub(&b.form),
                region,
            }
        }
        Op::Int(IntOp::Mul, a, b) => {
            let (a, b) = (get!(q, a), get!(q, b));
            match (a.form.as_constant(), b.form.as_constant()) {
                (Some(k), _) => Value::of(b.form.scale(k)),
                (_, Some(k)) => Value::of(a.form.scale(k)),
                _ => q.symbols_mut().product(v, &a.form, &b.form, lane),
            }
        }
        Op::Int(IntOp::Shl, a, s) => {
            let (a, s) = (get!(q, a), get!(q, s));
            let s = Value::of(match s.form.as_constant() {
                Some(k) => Form::constant(k & (ty.bits() - 1)),
                None => s.form,
            });
            match s.form.as_constant() {
                Some(k) if k < 32 => Value::of(a.form.scale(1 << k)),
                Some(k) if wide && k < 64 => Value::constant(0),
                None if !wide || q.symbols().bounds(&s.form).is_some_and(|(_, high)| high < 32) => {
                    let power = q.symbols_mut().power(v, &s.form, lane);
                    match a.form.as_constant() {
                        Some(k) => Value::of(power.scale(k)),
                        None => q.symbols_mut().product(v, &a.form, &power, lane),
                    }
                }
                None => Value::of(q.symbols_mut().shifted_by(v, IntOp::Shl, &a.form, &s.form, lane)),
                _ => q.symbols_mut().opaque(v, lane, None),
            }
        }
        Op::Int(IntOp::LShr, a, s) if !wide => {
            let (a, s) = (get!(q, a), get!(q, s));
            match s.form.as_constant().map(|k| k & 31) {
                Some(0) => a,
                Some(k) => shift(q, v, &a.form, k, lane),
                None => Value::of(q.symbols_mut().shifted_by(v, IntOp::LShr, &a.form, &s.form, lane)),
            }
        }
        Op::Int(IntOp::AShr, a, s) if !wide => {
            let (a, s) = (get!(q, a), get!(q, s));
            match (a.form.as_constant(), s.form.as_constant().map(|k| k & 31)) {
                (Some(x), Some(k)) => Value::constant(((x as i32) >> k) as u32),
                (None, Some(k)) if q.symbols().bounds(&a.form).is_some_and(|(_, high)| high < 1 << 31) => {
                    shift(q, v, &a.form, k, lane)
                }
                (_, None) => Value::of(q.symbols_mut().shifted_by(v, IntOp::AShr, &a.form, &s.form, lane)),
                _ => q.symbols_mut().opaque(v, lane, None),
            }
        }
        Op::Int(kind @ (IntOp::LShr | IntOp::AShr), x, s) => {
            let (a, s) = (get!(q, x), get!(q, s));
            let Some(high) = known_high(q, x, at, lane) else {
                return (q.symbols_mut().opaque(v, lane, None), used);
            };
            let signed = kind == IntOp::AShr;
            let non_negative = q.symbols().bounds(&high).is_some_and(|(_, top)| top < 1 << 31);
            match s.form.as_constant().map(|k| k & 63) {
                Some(0) => a,
                Some(k) if k >= 32 && (!signed || non_negative) => match shifted_part(q, v, &high, k - 32, lane) {
                    Some(form) => Value::of(form),
                    None => q.symbols_mut().opaque(v, lane, None),
                },
                Some(k) if k < 32 => match shifted_part(q, v, &a.form, k, lane) {
                    Some(form) => Value::of(form.add(&high.scale(1u32 << (32 - k)))),
                    None => q.symbols_mut().opaque(v, lane, None),
                },
                _ => q.symbols_mut().opaque(v, lane, None),
            }
        }
        Op::Int(IntOp::And, a, b) => {
            let (a, b) = (get!(q, a), get!(q, b));
            let region = match (a.form.as_constant(), b.form.as_constant()) {
                (_, Some(m)) if b.region.is_none() && aligns(m) => a.region,
                (Some(m), _) if a.region.is_none() && aligns(m) => b.region,
                _ => None,
            };
            let masked = match (a.form.as_constant(), b.form.as_constant()) {
                (Some(x), Some(y)) => Value::constant(x & y),
                (Some(m), None) | (None, Some(m)) => {
                    let form = if a.form.as_constant().is_some() { &b.form } else { &a.form };
                    match low_bits(form, m, IntOp::And) {
                        Some(result) => Value::of(result),
                        None => mask(q, v, form, m, lane),
                    }
                }
                _ => Value::of(q.symbols_mut().both(v, &a.form, &b.form, lane)),
            };
            Value { region, ..masked }
        }
        Op::Int(k @ (IntOp::Or | IntOp::Xor), a, b) => {
            let (a, b) = (get!(q, a), get!(q, b));
            match (a.form.as_constant(), b.form.as_constant()) {
                (Some(x), Some(y)) => Value::constant(if k == IntOp::Or { x | y } else { x ^ y }),
                (_, Some(u32::MAX)) if k == IntOp::Xor && !wide => {
                    Value::of(Form::constant(u32::MAX).sub(&a.form))
                }
                (Some(u32::MAX), _) if k == IntOp::Xor && !wide => {
                    Value::of(Form::constant(u32::MAX).sub(&b.form))
                }
                _ if q.symbols().disjoint_bits(&a.form, &b.form) => Value {
                    form: a.form.add(&b.form),
                    region: a.region.or(b.region),
                },
                (None, Some(c)) if low_bits(&a.form, c, k).is_some() => Value {
                    form: low_bits(&a.form, c, k).unwrap(),
                    region: a.region,
                },
                (Some(c), None) if low_bits(&b.form, c, k).is_some() => Value {
                    form: low_bits(&b.form, c, k).unwrap(),
                    region: b.region,
                },
                (None, Some(c)) if c != 0 => match split_low_bits(q, v, &a.form, c, k, lane) {
                    Some(x) => Value { region: a.region, ..x },
                    None => q.symbols_mut().opaque(v, lane, None),
                },
                (Some(c), None) if c != 0 => match split_low_bits(q, v, &b.form, c, k, lane) {
                    Some(x) => Value { region: b.region, ..x },
                    None => q.symbols_mut().opaque(v, lane, None),
                },
                (None, None) => {
                    let both = q.symbols_mut().both(v, &a.form, &b.form, lane);
                    let common = if k == IntOp::Or { both } else { both.scale(2) };
                    Value::of(a.form.add(&b.form).sub(&common))
                }
                _ => q.symbols_mut().opaque(v, lane, None),
            }
        }
        Op::Select(c, a, b) => {
            let (bit, u) = q.bit(c, lane, assume);
            used |= u;
            match bit {
                Some(true) => get!(q, a),
                Some(false) => get!(q, b),
                None => {
                    let (x, y) = (get!(q, a), get!(q, b));
                    if x == y {
                        x
                    } else if let Some(form) = q.symbols_mut().chosen(v, c, &x.form, &y.form, lane) {
                        let region = if x.region == y.region { x.region } else { None };
                        Value { form, region }
                    } else {
                        let region = if x.region == y.region { x.region } else { None };
                        let shared = q.program().facts.uniform[v.0];
                        let forms = vec![x.form.clone(), y.form.clone()];
                        let key = Key::Spread(v, if shared { 0 } else { lane as u8 }, forms.clone());
                        let block = q.program().block_of(v);
                        match q.symbols_mut().spread(key, shared, block, &forms) {
                            Some(form) => Value { form, region },
                            None => Value {
                                region,
                                ..q.symbols_mut().opaque(v, lane, None)
                            },
                        }
                    }
                }
            }
        }
        Op::Convert(k @ (Cvt::ZExt | Cvt::SExt), _, a) if q.program().f.types[a.0] == Ty::I1 => {
            let (bit, u) = q.bit(a, lane, assume);
            used |= u;
            let ones = if k == Cvt::SExt { u32::MAX } else { 1 };
            match bit {
                Some(b) => Value::constant(if b { ones } else { 0 }),
                None => Value::of(q.symbols_mut().opaque(v, lane, Some((0, 1))).form.scale(ones)),
            }
        }
        Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a)
            if to.bits() >= 32 && q.program().f.types[a.0].bits() >= 32 =>
        {
            get!(q, a)
        }
        Op::Convert(k @ (Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz), to, a) => {
            match q.program().converted(k, to, a) {
                Some(bits) => Value::constant(bits as u32),
                None => q.symbols_mut().opaque(v, lane, None),
            }
        }
        Op::Pack64(lo, _) | Op::UnpackLo(lo) => get!(q, lo),
        Op::UnpackHi(x) => match q.program().facts.op(q.program().f, x) {
            Some(Op::Pack64(_, hi)) => get!(q, hi),
            _ => match known_high(q, x, at, lane) {
                Some(high) => Value::of(high),
                None => q.symbols_mut().opaque(v, lane, None),
            },
        },
        Op::TrailingZeros(a) | Op::LeadingZeros(a) | Op::PopulationCount(a) | Op::ReverseBits(a)
            if !wide =>
        {
            let x = get!(q, a).form;
            match x.as_constant() {
                Some(x) => Value::constant(match op {
                    Op::TrailingZeros(_) => x.trailing_zeros(),
                    Op::LeadingZeros(_) => x.leading_zeros(),
                    Op::PopulationCount(_) => x.count_ones(),
                    _ => x.reverse_bits(),
                }),
                None => {
                    let (low, high) = q.symbols().bounds(&x).map_or((0, u32::MAX), |(l, h)| (l as u32, h as u32));
                    let range = match op {
                        Op::PopulationCount(_) => Some(((low != 0) as u32, 32 - high.leading_zeros())),
                        Op::LeadingZeros(_) => Some((high.leading_zeros(), low.leading_zeros())),
                        Op::TrailingZeros(_) if low != 0 => Some((0, 31 - high.leading_zeros())),
                        Op::TrailingZeros(_) => Some((0, 32)),
                        _ => None,
                    };
                    q.symbols_mut().opaque(v, lane, range)
                }
            }
        }
        _ => q.symbols_mut().opaque(v, lane, None),
    };
    (value, used)
}

fn effect<'a, Q: Queries<'a>>(
    q: &mut Q,
    v: ValueId,
    op: EffectOp,
    inputs: &[ValueId],
    outputs: &[(ValueId, Ty)],
    lane: usize,
    assume: Option<ValueId>,
) -> Assumed<Value> {
    let _ = outputs;
    match op {
        EffectOp::Memory {
            op: MemoryOp::Load(size),
            ..
        } => {
            let at = q.program().block_of(v);
            let (address, used) = q.operand(inputs[0], at, lane, assume);
            if let Some(value) = read_back(q, v, lane) {
                return (value, used);
            }
            (load(q, v, &address, size, lane), used)
        }
        EffectOp::Wave(WaveOp::ReadFirstLane) => {
            let mut first = Some(0);
            let lanes: Vec<usize> = (0..LANES).filter(|&l| q.symbols().valid(l)).collect();
            for l in lanes {
                match q.bit(inputs[1], l, None).0 {
                    Some(true) => {
                        first = Some(l);
                        break;
                    }
                    Some(false) => {}
                    None => {
                        first = None;
                        break;
                    }
                }
            }
            unassumed(uniform_read(q, v, inputs[0], first))
        }
        EffectOp::Wave(WaveOp::ReadLane) => {
            let (selector, _) = q.value(inputs[1], lane, None);
            match selector.form.as_constant().map(|k| (k & 31) as usize) {
                Some(chosen) if !q.symbols().valid(chosen) => unassumed(Value::constant(0)),
                Some(chosen) => unassumed(uniform_read(q, v, inputs[0], Some(chosen))),
                None => unassumed(any_lane_read(q, v, inputs[0], inputs[1], lane)),
            }
        }
        EffectOp::Wave(WaveOp::WriteLane) => {
            let (selector, _) = q.value(inputs[1], lane, None);
            match selector.form.as_constant() {
                Some(k) if (k & 31) as usize == lane => {
                    let at = q.program().block_of(v);
                    unassumed(q.operand(inputs[0], at, lane, None).0)
                }
                Some(_) => {
                    let at = q.program().block_of(v);
                    unassumed(q.operand(inputs[2], at, lane, None).0)
                }
                None => unassumed(q.symbols_mut().opaque(v, lane, None)),
            }
        }
        EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)) => {
            let (index, _) = q.value(inputs[0], lane, None);
            let Some(byte) = index.form.as_constant() else {
                return unassumed(q.symbols_mut().opaque(v, lane, None));
            };
            let source = ((byte >> 2) & 31) as usize;
            if !q.symbols().valid(source) {
                return unassumed(Value::constant(0));
            }
            let taken = match op {
                WaveOp::Bpermute => q.bit(inputs[2], source, None).0,
                _ => Some(true),
            };
            let at = q.program().block_of(v);
            match taken {
                Some(true) => unassumed(q.operand(inputs[1], at, source, None).0),
                Some(false) => unassumed(Value::constant(0)),
                None => unassumed(q.symbols_mut().opaque(v, lane, None)),
            }
        }
        EffectOp::Wave(WaveOp::Ballot) => {
            let mut mask = 0u32;
            for l in 0..LANES {
                if !q.symbols().valid(l) {
                    continue;
                }
                match q.bit(inputs[0], l, None).0 {
                    Some(true) => mask |= 1 << l,
                    Some(false) => {}
                    None => return unassumed(q.symbols_mut().opaque(v, lane, None)),
                }
            }
            unassumed(Value::constant(mask))
        }
        _ => unassumed(q.symbols_mut().opaque(v, lane, None)),
    }
}

fn any_lane_read<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, x: ValueId, selector: ValueId, lane: usize) -> Value {
    let at = q.program().block_of(v);
    let mut agreed: Option<Value> = None;
    let mut differ = false;
    let lanes: Vec<usize> = (0..LANES).filter(|&l| q.symbols().valid(l)).collect();
    for l in lanes {
        let (value, _) = q.operand(x, at, l, None);
        differ |= agreed.as_ref().is_some_and(|a| *a != value);
        agreed = Some(value);
    }
    if !(0..LANES).all(|l| q.symbols().valid(l)) && agreed.as_ref().is_some_and(|a| a.form.as_constant() != Some(0)) {
        differ = true;
    }
    match agreed {
        Some(value) if !differ => value,
        _ if q.program().facts.uniform[selector.0] => q.symbols_mut().uniform(v),
        _ => q.symbols_mut().opaque(v, lane, None),
    }
}

fn uniform_read<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, x: ValueId, from: Option<usize>) -> Value {
    let lanes: Vec<usize> = match from {
        Some(l) => vec![l],
        None => (0..LANES).filter(|&l| q.symbols().valid(l)).collect(),
    };
    let mut agreed: Option<Value> = None;
    let at = q.program().block_of(v);
    for l in lanes {
        let (value, _) = q.operand(x, at, l, None);
        if agreed.as_ref().is_some_and(|a| *a != value) {
            return q.symbols_mut().uniform(v);
        }
        agreed = Some(value);
    }
    agreed.unwrap_or_else(|| q.symbols_mut().uniform(v))
}

fn read_back<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, lane: usize) -> Option<Value> {
    let Site::Inst { block, index } = q.program().facts.site[v.0] else {
        return None;
    };
    let (data, base, target) = q.reason(|conditions, program| conditions.written_back(program, block, index))?;
    if let Some(base) = base {
        let start = q.operand(base, block, lane, None).0;
        let wide = q.program().f.types[base.0] == Ty::I64;
        if start.form.as_constant().is_none() || (wide && q.high(base, lane).as_constant().is_none()) {
            return None;
        }
    }
    if let Some(target) = target {
        let Inst::Effect { inputs, .. } = &q.program().f.blocks[&block].insts[index] else {
            return None;
        };
        if q.operand(inputs[0], block, lane, None).0.region != Some(Region::Allocation(target)) {
            return None;
        }
    }
    Some(q.operand(data, block, lane, None).0)
}

pub(super) fn concrete<'a, Q: Queries<'a>>(q: &mut Q, x: ValueId, param: ValueId, value: u32, header: BlockId, lane: usize, depth: usize) -> Option<u64> {
    if depth > 64 {
        return None;
    }
    let x = q.program().copies.get(&x).copied().unwrap_or(x);
    if x == param {
        return Some(value as u64);
    }
    let ty = q.program().f.types[x.0];
    let bits = ty.bits() as u64;
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let inside = q.program().loops.get(&q.program().block_of(x)).is_some_and(|l| l.contains(&header));
    if !inside {
        return match ty {
            Ty::I1 => q.bit(x, lane, None).0.map(u64::from),
            Ty::I32 => q.value(x, lane, None).0.form.as_constant().map(u64::from),
            _ => q.program().facts.constant(q.program().f, x),
        };
    }
    let signed = |a: u64, bits: u64| ((a << (64 - bits)) as i64) >> (64 - bits);
    let get = |this: &mut Q, a: ValueId| concrete(this, a, param, value, header, lane, depth + 1);
    let result = match q.program().facts.op(q.program().f, x)? {
        Op::Const(_, k) => k,
        Op::Env(Env::LaneId) => lane as u64,
        Op::Int(k, a, b) => {
            let (a, b) = (get(q, a)?, get(q, b)?);
            let amount = b & (bits - 1);
            match k {
                IntOp::Add => a.wrapping_add(b),
                IntOp::Sub => a.wrapping_sub(b),
                IntOp::Mul => a.wrapping_mul(b),
                IntOp::And => a & b,
                IntOp::Or => a | b,
                IntOp::Xor => a ^ b,
                IntOp::Shl => a << amount,
                IntOp::LShr => a >> amount,
                IntOp::AShr => (signed(a, bits) >> amount) as u64,
            }
        }
        Op::Cmp(p, a, b) => {
            let width = q.program().f.types[a.0].bits() as u64;
            let (a, b) = (get(q, a)?, get(q, b)?);
            let (sa, sb) = (signed(a, width), signed(b, width));
            (match p {
                IntPred::Eq => a == b,
                IntPred::Ne => a != b,
                IntPred::Ult => a < b,
                IntPred::Ugt => a > b,
                IntPred::Ule => a <= b,
                IntPred::Uge => a >= b,
                IntPred::Slt => sa < sb,
                IntPred::Sgt => sa > sb,
                IntPred::Sle => sa <= sb,
                IntPred::Sge => sa >= sb,
            }) as u64
        }
        Op::Select(c, a, b) => {
            if get(q, c)? != 0 {
                get(q, a)?
            } else {
                get(q, b)?
            }
        }
        Op::Convert(Cvt::ZExt | Cvt::Trunc, _, a) => get(q, a)?,
        Op::Convert(Cvt::SExt, _, a) => {
            let from = q.program().f.types[a.0].bits() as u64;
            signed(get(q, a)?, from) as u64
        }
        _ => return None,
    };
    Some(result & mask)
}

fn stepped<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, block: BlockId, index: usize, lane: usize) -> Option<(Value, bool)> {
    let exec_index = q.program().exec_index?;
    let own = q.program().rank[&block];
    let resolve = |this: &Q, x: ValueId| this.program().copies.get(&x).copied().unwrap_or(x);
    let mut step: Option<u32> = None;
    let mut masked = false;
    let mut entering: Option<Value> = None;
    for &(pred, slot) in &q.program().facts.incoming[&block].clone() {
        if !can_take(q, pred, slot, block) {
            continue;
        }
        let edge = q.program().f.blocks[&pred].term.edges().nth(slot).unwrap();
        let (arg, mask) = (edge.args[index], edge.args[exec_index]);
        if q.program().rank[&pred] < own {
            let (value, _) = q.operand(arg, block, lane, None);
            match &entering {
                None => entering = Some(value),
                Some(old) if *old == value => {}
                Some(_) => return None,
            }
            continue;
        }
        let arg = resolve(q, arg);
        let edge = q.program().edge_condition(pred, slot);
        let added = match q.program().facts.op(q.program().f, arg) {
            Some(Op::Select(c, x, y)) if q.program().source(y, lane) == (v, lane) && q.reason(|conditions, program| conditions.implies(program, mask, c, edge)) => {
                masked = true;
                x
            }
            _ => arg,
        };
        let k = match q.program().source(added, lane) {
            (x, l) if x == v && l == lane => 0,
            (x, l) if l == lane => {
                let other = q.program().increment(x, v, lane)?;
                q.value(other, lane, None).0.form.as_constant()?
            }
            _ => return None,
        };
        match step {
            None => step = Some(k),
            Some(old) if old == k => {}
            Some(_) => return None,
        }
    }
    let (step, entering) = (step?, entering?);
    if step == 0 {
        return Some((entering, false));
    }
    let trips = q.symbols_mut().trips(block);
    let value = Value {
        form: entering.form.add(&Form::unknown(trips).scale(step)),
        region: entering.region,
    };
    Some((value, masked))
}

fn induction<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, block: BlockId, index: usize, lane: usize) -> Option<(u32, u32)> {
    let own = q.program().rank[&block];
    let mut high = 0u64;
    for &(pred, slot) in &q.program().facts.incoming[&block].clone() {
        if !can_take(q, pred, slot, block) {
            continue;
        }
        let arg = q.program().f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
        if q.program().rank[&pred] >= own {
            let arg = q.program().copies.get(&arg).copied().unwrap_or(arg);
            let Some(Op::Int(IntOp::LShr, x, k)) = q.program().facts.op(q.program().f, arg) else {
                return None;
            };
            let x = q.program().copies.get(&x).copied().unwrap_or(x);
            if x != v || !matches!(q.program().facts.constant(q.program().f, k), Some(k) if k >= 1) {
                return None;
            }
        } else {
            let (value, _) = q.value(arg, lane, None);
            high = high.max(q.symbols().bounds(&value.form)?.1);
        }
    }
    Some((0, high as u32))
}
