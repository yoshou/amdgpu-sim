use super::super::address::compare;
use super::super::logic::{constant_choices, lane_test, projected_word, Atom, Choice};
use super::super::terms::Terms;
use super::lanes::*;
use super::queries::Queries;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;

pub(super) fn inst<'a, Q: Queries<'a>>(q: &mut Q, b: BlockId, index: usize, inst: &Inst) -> Bdd {
    match inst {
        Inst::Core { value, ty, op } => core(q, inst, *value, *ty, *op),
        Inst::Target { args, .. } => {
            let operands = any_of(q, args.values());
            let loaded = q.loaded((b, index)).unwrap_or(Bdd::FALSE);
            q.or(operands, loaded)
        }
        Inst::Packet { .. } => unreachable!("a packet query in a wave program"),
        Inst::Effect {
            provenance,
            op,
            inputs,
            outputs,
        } => match op {
            EffectOp::Wave(op) => wave(q, b, *provenance, *op, inputs, outputs),
            EffectOp::Memory {
                op: MemoryOp::Load(_),
                ..
            } => {
                let fp = q.bit(inputs[1]);
                let absent = q.not(fp);
                let operands = any_of(q, &inputs[..2]);
                let differs = q.or(operands, absent);
                let loaded = q.loaded((b, index)).unwrap_or(Bdd::FALSE);
                q.or(differs, loaded)
            }
            EffectOp::Memory {
                op: MemoryOp::Fence,
                ..
            } => Bdd::FALSE,
            EffectOp::Memory { op: memory, .. } => store(q, b, index, *memory, inputs),
            EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => any_of(q, inputs),
        },
    }
}

fn core<'a, Q: Queries<'a>>(q: &mut Q, inst: &Inst, value: ValueId, ty: Ty, op: Op) -> Bdd {
    let (f, facts) = (q.program().f, q.program().facts);
    match op {
        Op::Const(..) | Op::Env(_) => Bdd::FALSE,
        Op::Select(c, x, y) => {
            let (hx, hy) = if facts.lane_word[value.0] {
                (q.h(x), q.h(y))
            } else {
                (q.whole(x), q.whole(y))
            };
            let same = x == y || facts.constant(f, x).is_some_and(|k| facts.constant(f, y) == Some(k));
            if same {
                return q.or(hx, hy);
            }
            let hc = q.h(c);
            let arms = if hx == Bdd::FALSE && hy == Bdd::FALSE {
                Bdd::FALSE
            } else {
                let fc = q.bit(c);
                let taken = q.and(fc, hx);
                let nfc = q.not(fc);
                let other = q.and(nfc, hy);
                q.or(taken, other)
            };
            q.or(hc, arms)
        }
        Op::Int(k @ (IntOp::And | IntOp::Or), x, y) if ty == Ty::I1 || facts.lane_word[value.0] => {
            let (hx, hy) = (q.h(x), q.h(y));
            if hx == Bdd::FALSE && hy == Bdd::FALSE {
                return Bdd::FALSE;
            }
            let (fx, fy) = if ty == Ty::I1 {
                (q.bit(x), q.bit(y))
            } else {
                (q.view(x), q.view(y))
            };
            let conjunction = k == IntOp::And;
            let zx = absorbs(q.logic(), fx, hx, conjunction);
            let zy = absorbs(q.logic(), fy, hy, conjunction);
            let either = q.or(hx, hy);
            let open_x = q.not(zx);
            let open_y = q.not(zy);
            let open = q.and(open_x, open_y);
            q.and(either, open)
        }
        Op::Int(IntOp::Xor, x, y) if facts.lane_word[value.0] => q.or(q.h(x), q.h(y)),
        Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Mul), x, y)
            if [x, y].iter().any(|&v| {
                let ones = if ty == Ty::I64 { u64::MAX } else { u32::MAX as u64 };
                facts.constant(f, v) == Some(if k == IntOp::Or { ones } else { 0 })
            }) =>
        {
            Bdd::FALSE
        }
        Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, x) if facts.lane_word[value.0] => q.h(x),
        Op::Pack64(x, y) if facts.lane_word[value.0] => {
            let low = q.h(x);
            if f.lanes == 32 {
                return low;
            }
            let high = q.h(y);
            let upper = q.logic().atom(Atom::Lane(5));
            q.logic().m.ite(upper, high, low)
        }
        Op::UnpackLo(x) if facts.lane_word[value.0] => {
            let own = q.h(x);
            q.logic().half(false, own, Bdd::TRUE)
        }
        Op::UnpackHi(x) if facts.lane_word[value.0] => {
            let own = q.h(x);
            q.logic().half(true, own, Bdd::TRUE)
        }
        Op::Convert(Cvt::Trunc, Ty::I1, s) => match projected_word(f, facts, s) {
            Some(w) if facts.lane_word[w.0] => q.h(w),
            _ => q.h(s),
        },
        Op::Int(IntOp::LShr, w, lane) if facts.lane_word[w.0] && facts.is_lane_shift(f, w, lane) => q.h(w),
        Op::Cmp(IntPred::Eq | IntPred::Ne, x, y) if lane_test(f, facts, x, y).is_some() => {
            let w = lane_test(f, facts, x, y).unwrap();
            let mode = q.logic().materialized(facts, w);
            if mode == Bdd::TRUE {
                return q.whole(w);
            }
            let fw = q.view(w);
            let hw = q.h(w);
            let tag = q.logic().tag(Choice::Word(w));
            let local = query(q.logic(), facts, fw, hw, tag);
            let whole = q.whole(w);
            q.logic().m.ite(mode, whole, local)
        }
        other if q.settles(value, other) => Bdd::FALSE,
        _ => any_of(q, &inst.operands()),
    }
}

fn wave<'a, Q: Queries<'a>>(
    q: &mut Q,
    b: BlockId,
    provenance: u64,
    op: WaveOp,
    inputs: &[ValueId],
    outputs: &[(ValueId, Ty)],
) -> Bdd {
    let (f, facts) = (q.program().f, q.program().facts);
    match op {
        WaveOp::Any => {
            let (out, x) = (outputs[0].0, inputs[0]);
            let local = q.logic().local(Choice::Query(out));
            if !q.masked(x) {
                let kept = q.not(local);
                q.demand(provenance, kept);
            }
            let hx = q.h(x);
            if local == Bdd::FALSE {
                return some_lane(q.logic(), facts, hx);
            }
            let fx = q.bit(x);
            let tag = q.logic().tag(Choice::Query(out));
            let unchanged = q.not(hx);
            let certain = q.and(fx, unchanged);
            let possible = q.or(fx, hx);
            let low = q.logic().answer(f, facts, out, certain);
            let high = q.logic().answer(f, facts, out, possible);
            let differs_low = q.logic().m.xor(fx, low);
            let differs_high = q.logic().m.xor(fx, high);
            let differs = q.or(differs_low, differs_high);
            let answered = q.and(tag, differs);
            if local == Bdd::TRUE {
                return answered;
            }
            let gathered = some_lane(q.logic(), facts, hx);
            q.logic().m.ite(local, answered, gathered)
        }
        WaveOp::Ballot { high } => {
            let own = q.h(inputs[0]);
            q.logic().half(high, own, Bdd::TRUE)
        }
        WaveOp::ReadFirstLane => {
            let (x, mask) = (inputs[0], inputs[1]);
            if facts.uniform[x.0] {
                return q.whole(x);
            }
            if !q.masked(mask) {
                q.demand(provenance, Bdd::TRUE);
            }
            let first = masked_read(q, mask, x);
            let hx = q.whole(x);
            if hx == Bdd::FALSE {
                return first;
            }
            let fm = q.bit(mask);
            let clear = q.not(fm);
            let unset = q.and(clear, hx);
            let zero = some_lane(q.logic(), facts, unset);
            let zero = q.and(clear, zero);
            q.or(first, zero)
        }
        WaveOp::ReadLane => {
            let (x, selector) = (inputs[0], inputs[1]);
            let own = q.whole(selector);
            let hx = q.whole(x);
            let last = q.logic().lane_count() - 1;
            let read = match (q.logic().lane_function(f, facts, selector), constant_choices(f, facts, selector)) {
                (Some(sources), _) => read_from(q.logic(), facts, hx, |l| sources[l] & last),
                (None, Some(lanes)) => {
                    let mut any = Bdd::FALSE;
                    for lane in lanes {
                        let there = at_lane(q.logic(), hx, lane as u32 & last);
                        let there = some_lane(q.logic(), facts, there);
                        any = q.or(any, there);
                    }
                    any
                }
                (None, None) => some_lane(q.logic(), facts, hx),
            };
            q.or(own, read)
        }
        WaveOp::WriteLane => {
            let written = any_of(q, &inputs[..2]);
            let written = at_lane(q.logic(), written, 0);
            let written = some_lane(q.logic(), facts, written);
            let old = any_of(q, &inputs[2..]);
            let last = q.logic().lane_count() as u64 - 1;
            match facts.constant(f, inputs[1]) {
                Some(k) => {
                    let target = q.logic().lanes(|l| l as u64 == k & last);
                    q.logic().m.ite(target, written, old)
                }
                None => match q.logic().lane_is(f, facts, b, inputs[1]) {
                    Some(target) => q.logic().m.ite(target, written, old),
                    None => q.or(old, written),
                },
            }
        }
        WaveOp::Bpermute | WaveOp::BpermuteFi => {
            let (index, x, mask) = (inputs[0], inputs[1], inputs[2]);
            let count = q.logic().lane_count();
            let last = count - 1;
            let read = match q.logic().lane_function(f, facts, index) {
                Some(indices) => {
                    let hx = q.whole(x);
                    let h = gated(q, op, mask, hx);
                    read_from(q.logic(), facts, h, |l| (indices[l] >> 2) & last)
                }
                None => match q.logic().lane_bits(f, facts, index) {
                    Some(bits) if bits[..count as usize].iter().any(|&(m, _)| (m >> 2) & last != 0) => {
                        let hx = q.whole(x);
                        let h = gated(q, op, mask, hx);
                        read_from_any(q.logic(), facts, h, |l| {
                            let (m, v) = bits[l];
                            let (m, v) = ((m >> 2) & last, (v >> 2) & last);
                            (0..count).filter(|s| s & m == v & m).fold(0u64, |set, s| set | 1 << s)
                        })
                    }
                    _ if op == WaveOp::Bpermute => masked_read(q, mask, x),
                    _ => {
                        let hx = q.whole(x);
                        some_lane(q.logic(), facts, hx)
                    }
                },
            };
            let own = q.whole(index);
            q.or(own, read)
        }
        WaveOp::Wmma => {
            let mut any = Bdd::FALSE;
            for m in 0..8 {
                let output = wmma_output(q, inputs, m);
                any = q.or(any, output);
            }
            any
        }
        WaveOp::Meet => any_of(q, inputs),
    }
}

fn gated<'a, Q: Queries<'a>>(q: &mut Q, op: WaveOp, mask: ValueId, hx: Bdd) -> Bdd {
    if op != WaveOp::Bpermute {
        return hx;
    }
    let hm = q.h(mask);
    let fm = q.bit(mask);
    let set = q.and(fm, hx);
    q.or(hm, set)
}

fn store<'a, Q: Queries<'a>>(q: &mut Q, b: BlockId, index: usize, memory: MemoryOp, inputs: &[ValueId]) -> Bdd {
    let m = memory.mask_input();
    let pred = inputs[m];
    let reachable = q.reachable(b);
    let hp = q.h(pred);
    if hp != Bdd::FALSE {
        let happens = q.and(hp, reachable);
        q.require(
            b,
            index,
            "whether a store happens depends on the other lanes",
            happens,
        );
        if q.stopped() {
            return Bdd::FALSE;
        }
    }
    let fp = q.bit(pred);
    let operands = any_of(q, &inputs[..m]);
    if operands != Bdd::FALSE {
        let performed = q.and(fp, reachable);
        let writes = q.and(performed, operands);
        q.require(
            b,
            index,
            "what a store writes depends on the other lanes",
            writes,
        );
        if q.stopped() {
            return Bdd::FALSE;
        }
    }
    if let Some(order) = q.reordered((b, index)) {
        let performed = q.and(fp, reachable);
        let reordered = q.and(performed, order);
        q.require(
            b,
            index,
            "another lane may write these bytes on the other side of the store",
            reordered,
        );
        if q.stopped() {
            return Bdd::FALSE;
        }
    }
    let absent = q.not(fp);
    let differs = q.or(operands, absent);
    let loaded = q.loaded((b, index)).unwrap_or(Bdd::FALSE);
    q.or(differs, loaded)
}

pub(super) fn word_difference<'a, Q: Queries<'a>>(q: &mut Q, inst: &Inst, value: ValueId) -> Bdd {
    let facts = q.program().facts;
    match inst {
        Inst::Effect {
            provenance,
            op: EffectOp::Wave(WaveOp::Ballot { high }),
            inputs,
            ..
        } => {
            if !q.faithful(value) {
                let whole = q.logic().materialized(facts, value);
                q.demand(*provenance, whole);
            }
            let x = q.h(inputs[0]);
            let high = *high;
            let half = q.logic().lanes(|l| (l >= 32) == high);
            let x = q.and(x, half);
            some_lane(q.logic(), facts, x)
        }
        Inst::Core {
            op: Op::Select(c, a, b),
            ..
        } => {
            let (wa, wb) = (q.whole(*a), q.whole(*b));
            let arms = if wa == Bdd::FALSE && wb == Bdd::FALSE {
                Bdd::FALSE
            } else {
                let fc = q.bit(*c);
                q.logic().m.ite(fc, wa, wb)
            };
            q.or(q.h(*c), arms)
        }
        _ => any_of(q, &inst.operands()),
    }
}

pub(super) fn wmma_output<'a, Q: Queries<'a>>(q: &mut Q, inputs: &[ValueId], m: usize) -> Bdd {
    let facts = q.program().facts;
    let fragments = any_of(q, &inputs[..8]);
    let fragments = some_lane(q.logic(), facts, fragments);
    let accumulator = q.whole(inputs[8 + m]);
    q.or(accumulator, fragments)
}

fn any_of<'a, Q: Queries<'a>>(q: &mut Q, values: &[ValueId]) -> Bdd {
    let mut r = Bdd::FALSE;
    for &v in values {
        let x = q.whole(v);
        r = q.or(r, x);
    }
    r
}

fn masked_read<'a, Q: Queries<'a>>(q: &mut Q, mask: ValueId, x: ValueId) -> Bdd {
    let hm = q.h(mask);
    let hx = q.whole(x);
    if hm == Bdd::FALSE && hx == Bdd::FALSE {
        return Bdd::FALSE;
    }
    let facts = q.program().facts;
    let fm = q.bit(mask);
    let set = q.and(fm, hx);
    let read = q.or(hm, set);
    some_lane(q.logic(), facts, read)
}

pub(super) fn settled(f: &Func, facts: &Facts, op: Op) -> bool {
    let (Op::Cmp(_, x, y) | Op::Int(_, x, y)) = op else {
        return false;
    };
    if x == y && matches!(op, Op::Int(IntOp::Sub | IntOp::Xor, ..) | Op::Cmp(..)) {
        return true;
    }
    if f.types[x.0] != Ty::I32 {
        return false;
    }
    if let Op::Cmp(p, ..) = op {
        if Terms::new(f, facts).decided(p, x, y).is_some() {
            return true;
        }
    }
    let (Some(xs), Some(ys)) = (constant_choices(f, facts, x), constant_choices(f, facts, y)) else {
        return false;
    };
    let mut answers = Vec::new();
    for &a in &xs {
        for &b in &ys {
            let (a, b) = (a as u32, b as u32);
            answers.push(match op {
                Op::Cmp(p, ..) => compare(p, a, b) as u32,
                Op::Int(k, ..) => match k {
                    IntOp::Add => a.wrapping_add(b),
                    IntOp::Sub => a.wrapping_sub(b),
                    IntOp::Mul => a.wrapping_mul(b),
                    IntOp::And => a & b,
                    IntOp::Or => a | b,
                    IntOp::Xor => a ^ b,
                    IntOp::Shl => a << (b & 31),
                    IntOp::LShr => a >> (b & 31),
                    IntOp::AShr => ((a as i32) >> (b & 31)) as u32,
                },
                _ => return false,
            });
        }
    }
    answers.iter().all(|&t| t == answers[0])
}
