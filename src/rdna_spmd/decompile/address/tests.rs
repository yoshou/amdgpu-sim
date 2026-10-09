use super::super::encoding::{mirrored, outcomes};
use super::super::testing::*;
use super::form::*;
use super::limits::*;
use super::*;
use crate::rdna_spmd::analysis::loops::Loops;

pub(super) fn addresses<T>(b: &Build, env: &Environment, f: impl FnOnce(&mut Addresses) -> T) -> T {
    let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
    let loops = Loops::new(&b.f, &facts).expect("a reducible test program");
    let headers: BTreeSet<BlockId> = (0..loops.count()).map(|l| facts.order[loops.header(l)]).collect();
    let mut a = Addresses::new(&b.f, &facts, &b.inputs, EXEC, b.entry, env, headers, &b.registry);
    a.enter(0);
    f(&mut a)
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Int(IntOp, bool),
    Cmp(IntPred),
    Select,
    Count(u8),
    Extend(Cvt, Ty),
    Wide(IntOp),
    Pack,
    Bits(IntOp),
    Truncate,
}

fn leaf(b: &mut Build, e: BlockId, lane: ValueId, m: u32, c: u32) -> ValueId {
    let km = b.constant(e, Ty::I32, m as u64);
    let kc = b.constant(e, Ty::I32, c as u64);
    let scaled = b.int(e, IntOp::Mul, lane, km);
    b.int(e, IntOp::Add, scaled, kc)
}

fn compare(pred: IntPred, x: u32, y: u32) -> bool {
    super::compare(pred, x, y)
}

enum Truth {
    Word(Vec<u32>),
    Bit(Vec<bool>),
}

fn cases(seed: u64, count: usize) -> Vec<(Build, ValueId, Truth, String)> {
    use IntOp::*;
    let mut r = Random::new(seed);
    let ops = [Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr];
    let preds = [
        IntPred::Eq,
        IntPred::Ne,
        IntPred::Ult,
        IntPred::Ugt,
        IntPred::Ule,
        IntPred::Uge,
        IntPred::Slt,
        IntPred::Sgt,
        IntPred::Sle,
        IntPred::Sge,
    ];
    let pick = |r: &mut Random| -> u32 {
        match r.below(4) {
            0 => r.below(8) as u32,
            1 => (r.below(8) as u32).wrapping_neg(),
            2 => 0x8000_0000u32.wrapping_add(r.below(4) as u32),
            _ => r.next() as u32,
        }
    };
    let mut out = Vec::new();
    for _ in 0..count {
        let (mut b, _) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let (m1, c1, m2, c2) = (pick(&mut r), pick(&mut r), pick(&mut r), pick(&mut r));
        let x = leaf(&mut b, e, lane, m1, c1);
        let y = leaf(&mut b, e, lane, m2, c2);
        let tx: Vec<u32> = (0..32u32).map(|l| l.wrapping_mul(m1).wrapping_add(c1)).collect();
        let ty: Vec<u32> = (0..32u32).map(|l| l.wrapping_mul(m2).wrapping_add(c2)).collect();
        let case = match r.below(9) {
            0 => Case::Int(ops[r.below(9) as usize], false),
            1 => Case::Int(ops[r.below(9) as usize], true),
            2 => Case::Cmp(preds[r.below(10) as usize]),
            3 => Case::Select,
            4 => Case::Count(r.below(4) as u8),
            5 => Case::Extend(if r.below(2) == 0 { Cvt::ZExt } else { Cvt::SExt }, if r.below(2) == 0 { Ty::I32 } else { Ty::I64 }),
            6 => Case::Wide([Add, Sub, Mul, And, Or, Xor, Shl][r.below(7) as usize]),
            7 => Case::Pack,
            _ => {
                if r.below(2) == 0 {
                    Case::Bits([And, Or, Xor][r.below(3) as usize])
                } else {
                    Case::Truncate
                }
            }
        };
        let (v, truth) = match case {
            Case::Int(op, constant_shift) => {
                let amount = r.below(32) as u32;
                let (y, ty) = if matches!(op, Shl | LShr | AShr) || constant_shift {
                    if matches!(op, Shl | LShr | AShr) {
                        (b.constant(e, Ty::I32, amount as u64), vec![amount; 32])
                    } else {
                        let k = pick(&mut r);
                        (b.constant(e, Ty::I32, k as u64), vec![k; 32])
                    }
                } else {
                    (y, ty.clone())
                };
                let v = b.int(e, op, x, y);
                let t = (0..32)
                    .map(|l| {
                        let (a, s) = (tx[l], ty[l]);
                        match op {
                            Add => a.wrapping_add(s),
                            Sub => a.wrapping_sub(s),
                            Mul => a.wrapping_mul(s),
                            And => a & s,
                            Or => a | s,
                            Xor => a ^ s,
                            Shl => a << s,
                            LShr => a >> s,
                            _ => ((a as i32) >> s) as u32,
                        }
                    })
                    .collect();
                (v, Truth::Word(t))
            }
            Case::Cmp(pred) => {
                let v = b.cmp(e, pred, x, y);
                (v, Truth::Bit((0..32).map(|l| compare(pred, tx[l], ty[l])).collect()))
            }
            Case::Select => {
                let pred = preds[r.below(10) as usize];
                let c = b.cmp(e, pred, x, y);
                let v = b.core(e, Ty::I32, Op::Select(c, x, y));
                (v, Truth::Word((0..32).map(|l| if compare(pred, tx[l], ty[l]) { tx[l] } else { ty[l] }).collect()))
            }
            Case::Count(k) => {
                let op = [Op::PopulationCount(x), Op::TrailingZeros(x), Op::LeadingZeros(x), Op::ReverseBits(x)][k as usize];
                let v = b.core(e, Ty::I32, op);
                let t = (0..32)
                    .map(|l| {
                        let a = tx[l];
                        match k {
                            0 => a.count_ones(),
                            1 => a.trailing_zeros(),
                            2 => a.leading_zeros(),
                            _ => a.reverse_bits(),
                        }
                    })
                    .collect();
                (v, Truth::Word(t))
            }
            Case::Extend(cvt, to) => {
                let pred = preds[r.below(10) as usize];
                let c = b.cmp(e, pred, x, y);
                let v = b.core(e, to, Op::Convert(cvt, to, c));
                let t = (0..32)
                    .map(|l| match (compare(pred, tx[l], ty[l]), cvt) {
                        (false, _) => 0,
                        (true, Cvt::ZExt) => 1,
                        (true, _) => u32::MAX,
                    })
                    .collect();
                (v, Truth::Word(t))
            }
            Case::Wide(op) => {
                let wx = b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, x));
                let wy = b.core(e, Ty::I64, Op::Pack64(y, x));
                let amount = r.below(64);
                let wy = if op == Shl { b.constant(e, Ty::I64, amount) } else { wy };
                let v = b.int(e, op, wx, wy);
                let t = (0..32)
                    .map(|l| {
                        let a = tx[l] as i32 as i64 as u64;
                        let s = if op == Shl { amount } else { ty[l] as u64 | (tx[l] as u64) << 32 };
                        let r = match op {
                            Add => a.wrapping_add(s),
                            Sub => a.wrapping_sub(s),
                            Mul => a.wrapping_mul(s),
                            And => a & s,
                            Or => a | s,
                            Xor => a ^ s,
                            _ => a << s,
                        };
                        r as u32
                    })
                    .collect();
                (v, Truth::Word(t))
            }
            Case::Pack => {
                let p = b.core(e, Ty::I64, Op::Pack64(x, y));
                let high = r.below(2) == 0;
                let v = b.core(e, Ty::I32, if high { Op::UnpackHi(p) } else { Op::UnpackLo(p) });
                (v, Truth::Word(if high { ty.clone() } else { tx.clone() }))
            }
            Case::Bits(op) => {
                let p1 = preds[r.below(10) as usize];
                let p2 = preds[r.below(10) as usize];
                let c1 = b.cmp(e, p1, x, y);
                let c2 = b.cmp(e, p2, y, x);
                let v = b.int(e, op, c1, c2);
                let t = (0..32)
                    .map(|l| {
                        let (a, s) = (compare(p1, tx[l], ty[l]), compare(p2, ty[l], tx[l]));
                        match op {
                            And => a && s,
                            Or => a || s,
                            _ => a != s,
                        }
                    })
                    .collect();
                (v, Truth::Bit(t))
            }
            Case::Truncate => {
                let k = r.below(32);
                let amount = b.constant(e, Ty::I32, k);
                let shifted = b.int(e, LShr, x, amount);
                let v = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
                (v, Truth::Bit((0..32).map(|l| tx[l] >> k & 1 != 0).collect()))
            }
        };
        out.push((b, v, truth, format!("{:?}", case)));
    }
    out
}

fn decided(seed: u64, count: usize) -> (Vec<String>, Vec<String>) {
    let env = environment(32, &[(0, 1, 0x1000)]);
    let mut wrong = Vec::new();
    let mut undecided = Vec::new();
    for (b, v, truth, name) in cases(seed, count) {
        addresses(&b, &env, |a| {
            for lane in 0..32 {
                match &truth {
                    Truth::Word(t) => match a.values.value(v, lane, None).0.form.as_constant() {
                        Some(k) if k != t[lane] => wrong.push(format!("{} lane {}: {:#x}, not {:#x}", name, lane, k, t[lane])),
                        Some(_) => {}
                        None => undecided.push(format!("{} lane {}", name, lane)),
                    },
                    Truth::Bit(t) => match a.bit(v, lane, None).0 {
                        Some(k) if k != t[lane] => wrong.push(format!("{} lane {}: {}, not {}", name, lane, k, t[lane])),
                        Some(_) => {}
                        None => undecided.push(format!("{} lane {}", name, lane)),
                    },
                }
            }
        });
    }
    (wrong, undecided)
}

struct Unknowns {
    b: Build,
    words: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)>,
    bits: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> bool>)>,
}

fn unknowns() -> Unknowns {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let u = b.load(e, Space::Global, MemSize::U16, table, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
    let mut words: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)> = Vec::new();
    let mut bits: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> bool>)> = Vec::new();
    let four = c(&mut b, 4);
    let lane4 = b.int(e, IntOp::Mul, lane, four);
    let v = b.int(e, IntOp::Add, u, lane4);
    words.push(("u + 4 lane", v, Box::new(|u, _, l| u.wrapping_add(4 * l))));
    let eight = c(&mut b, 8);
    let u4 = b.int(e, IntOp::Mul, u, four);
    let v = b.int(e, IntOp::Add, u4, eight);
    words.push(("4u + 8", v, Box::new(|u, _, _| u.wrapping_mul(4).wrapping_add(8))));
    let one = c(&mut b, 1);
    let v = b.int(e, IntOp::LShr, u, one);
    words.push(("u >> 1", v, Box::new(|u, _, _| u >> 1)));
    let two = c(&mut b, 2);
    let u2 = b.int(e, IntOp::Add, u, two);
    let v = b.int(e, IntOp::LShr, u2, one);
    words.push(("(u + 2) >> 1", v, Box::new(|u, _, _| (u + 2) >> 1)));
    let ff = c(&mut b, 0xff);
    let v = b.int(e, IntOp::And, u, ff);
    words.push(("u & 0xff", v, Box::new(|u, _, _| u & 0xff)));
    let fffc = c(&mut b, 0xfffc);
    let v = b.int(e, IntOp::And, u, fffc);
    words.push(("u & 0xfffc", v, Box::new(|u, _, _| u & 0xfffc)));
    let fourteen = c(&mut b, 14);
    let quarter = b.int(e, IntOp::LShr, u, fourteen);
    let seven = c(&mut b, 7);
    for base in [8u32, 6] {
        let k = c(&mut b, base as u64);
        let shifted = b.int(e, IntOp::Add, quarter, k);
        let v = b.int(e, IntOp::And, shifted, seven);
        let name: &'static str = if base == 8 { "((u >> 14) + 8) & 7" } else { "((u >> 14) + 6) & 7" };
        words.push((name, v, Box::new(move |u, _, _| ((u >> 14) + base) & 7)));
    }
    let high = c(&mut b, 0x1_0000);
    let v = b.int(e, IntOp::Or, u, high);
    words.push(("u | 0x10000", v, Box::new(|u, _, _| u | 0x1_0000)));
    let ones = c(&mut b, 0xffff_ffff);
    let v = b.int(e, IntOp::Xor, u, ones);
    words.push(("u ^ ~0", v, Box::new(|u, _, _| !u)));
    let u_shl = b.int(e, IntOp::Shl, u, two);
    let v = b.int(e, IntOp::Add, u_shl, lane);
    words.push(("(u << 2) + lane", v, Box::new(|u, _, l| (u << 2).wrapping_add(l))));
    let seventeen = c(&mut b, 17);
    let v = b.int(e, IntOp::LShr, u, seventeen);
    words.push(("u >> 17", v, Box::new(|u, _, _| u >> 17)));
    let v = b.int(e, IntOp::Add, w, lane);
    words.push(("w + lane", v, Box::new(|_, w, l| w.wrapping_add(l))));
    let three = c(&mut b, 3);
    let w3 = b.int(e, IntOp::And, w, three);
    let v = b.int(e, IntOp::Mul, w3, four);
    words.push(("(w & 3) * 4", v, Box::new(|_, w, _| (w & 3) * 4)));
    let hundred = c(&mut b, 100);
    let small = b.cmp(e, IntPred::Ult, u, hundred);
    let v = b.core(e, Ty::I32, Op::Select(small, u, hundred));
    words.push(("min(u, 100)", v, Box::new(|u, _, _| u.min(100))));
    let next = b.int(e, IntOp::Add, u, one);
    let past = c(&mut b, 101);
    let v = b.core(e, Ty::I32, Op::Select(small, next, past));
    words.push(("select(u < 100, u + 1, 101)", v, Box::new(|u, _, _| if u < 100 { u + 1 } else { 101 })));
    let wide_small = b.cmp(e, IntPred::Ult, w, hundred);
    let v = b.core(e, Ty::I32, Op::Select(wide_small, u, w));
    words.push(("select(w < 100, u, w)", v, Box::new(|u, w, _| if w < 100 { u } else { w })));
    let v = b.core(e, Ty::I32, Op::Select(wide_small, next, w));
    words.push(("select(w < 100, u + 1, w)", v, Box::new(|u, w, _| if w < 100 { u + 1 } else { w })));
    let v = b.core(e, Ty::I32, Op::Select(wide_small, u, hundred));
    words.push(("select(w < 100, u, 100)", v, Box::new(|u, w, _| if w < 100 { u } else { 100 })));
    let thousand = c(&mut b, 1000);
    let below = b.cmp(e, IntPred::Ult, u, thousand);
    let v = b.core(e, Ty::I32, Op::Select(below, u, hundred));
    words.push(("select(u < 1000, u, 100)", v, Box::new(|u, _, _| if u < 1000 { u } else { 100 })));
    let low_byte = b.int(e, IntOp::And, w, ff);
    let more = b.int(e, IntOp::Add, u, low_byte);
    let v = b.core(e, Ty::I32, Op::Select(small, more, u));
    words.push(("select(u < 100, u + (w & 0xff), u)", v, Box::new(|u, w, _| if u < 100 { u + (w & 0xff) } else { u })));
    let v = b.core(e, Ty::I32, Op::Select(small, u, more));
    words.push(("select(u < 100, u, u + (w & 0xff))", v, Box::new(|u, w, _| if u < 100 { u } else { u + (w & 0xff) })));
    let twice = b.int(e, IntOp::Add, u, u);
    let v = b.core(e, Ty::I32, Op::Select(small, u, twice));
    words.push(("select(u < 100, u, 2u)", v, Box::new(|u, _, _| if u < 100 { u } else { 2 * u })));
    let v = b.int(e, IntOp::Mul, u, u);
    words.push(("u * u", v, Box::new(|u, _, _| u.wrapping_mul(u))));
    let sixteen = c(&mut b, 16);
    let up = b.int(e, IntOp::Shl, u, sixteen);
    let v = b.int(e, IntOp::LShr, up, sixteen);
    words.push(("(u << 16) >> 16", v, Box::new(|u, _, _| (u << 16) >> 16)));
    let v = b.int(e, IntOp::Sub, u, one);
    words.push(("u - 1", v, Box::new(|u, _, _| u.wrapping_sub(1))));
    let v = b.int(e, IntOp::Sub, u2, u);
    words.push(("(u + 2) - u", v, Box::new(|_, _, _| 2)));
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
    let hi = b.core(e, Ty::I32, Op::UnpackHi(wide));
    words.push(("hi(zext u)", hi, Box::new(|_, _, _| 0)));
    let both = b.int(e, IntOp::And, u, w);
    words.push(("u & w", both, Box::new(|u, w, _| u & w)));
    let either = b.int(e, IntOp::Or, w, u);
    words.push(("w | u", either, Box::new(|u, w, _| w | u)));
    let v = b.int(e, IntOp::Xor, u, w);
    words.push(("u ^ w", v, Box::new(|u, w, _| u ^ w)));
    let v = b.int(e, IntOp::Add, both, either);
    words.push(("(u & w) + (w | u)", v, Box::new(|u, w, _| (u & w).wrapping_add(w | u))));
    let v = b.int(e, IntOp::Add, u, w);
    words.push(("u + w", v, Box::new(|u, w, _| u.wrapping_add(w))));
    let v = b.int(e, IntOp::Mul, u, w);
    words.push(("u * w", v, Box::new(|u, w, _| u.wrapping_mul(w))));
    let v = b.int(e, IntOp::Mul, w, u);
    words.push(("w * u", v, Box::new(|u, w, _| w.wrapping_mul(u))));
    let v = b.int(e, IntOp::Mul, u2, w);
    words.push(("(u + 2) * w", v, Box::new(|u, w, _| (u + 2).wrapping_mul(w))));
    let thirty_one = c(&mut b, 31);
    let s = b.int(e, IntOp::And, w, thirty_one);
    let v = b.int(e, IntOp::Shl, u, s);
    words.push(("u << (w & 31)", v, Box::new(|u, w, _| u << (w & 31))));
    let power = b.int(e, IntOp::Shl, one, s);
    let v = b.int(e, IntOp::Mul, power, u);
    words.push(("(1 << (w & 31)) * u", v, Box::new(|u, w, _| (1u32 << (w & 31)).wrapping_mul(u))));
    let v = b.int(e, IntOp::Shl, w, u);
    words.push(("w << u", v, Box::new(|u, w, _| w << (u & 31))));
    let v = b.int(e, IntOp::LShr, u, s);
    words.push(("u >> (w & 31)", v, Box::new(|u, w, _| u >> (w & 31))));
    let v = b.int(e, IntOp::AShr, u, s);
    words.push(("u >>> (w & 31)", v, Box::new(|u, w, _| ((u as i32) >> (w & 31)) as u32)));
    let low_half = c(&mut b, 0xffff);
    let half = b.int(e, IntOp::And, u, low_half);
    let v = b.int(e, IntOp::LShr, half, s);
    words.push(("(u & 0xffff) >> (w & 31)", v, Box::new(|u, w, _| (u & 0xffff) >> (w & 31))));
    let v = b.int(e, IntOp::AShr, half, s);
    words.push(("(u & 0xffff) >>> (w & 31)", v, Box::new(|u, w, _| (u & 0xffff) >> (w & 31))));
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
    let wide_s = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, s));
    let shifted = b.int(e, IntOp::Shl, wide, wide_s);
    let v = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, shifted));
    words.push(("low(zext(u) << (w & 31))", v, Box::new(|u, w, _| ((u as u64) << (w & 31)) as u32)));
    let sixty_three = c(&mut b, 63);
    let far = b.int(e, IntOp::And, w, sixty_three);
    let wide_far = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, far));
    let shifted = b.int(e, IntOp::Shl, wide, wide_far);
    let v = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, shifted));
    words.push(("low(zext(u) << (w & 63))", v, Box::new(|u, w, _| ((u as u64) << (w & 63)) as u32)));
    let wide_w = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, w));
    let sum = b.int(e, IntOp::Add, wide, wide_w);
    let again = b.int(e, IntOp::Add, sum, wide);
    let v = b.core(e, Ty::I32, Op::UnpackHi(again));
    words.push(("hi((zext(u) + zext(w)) + zext(u))", v, Box::new(|u, w, _| ((u as u64 + w as u64 + u as u64) >> 32) as u32)));
    let back = b.int(e, IntOp::Sub, again, wide_w);
    let v = b.core(e, Ty::I32, Op::UnpackHi(back));
    words.push(("hi((zext(u) + zext(w)) + zext(u) - zext(w))", v, Box::new(|u, _, _| ((2 * u as u64) >> 32) as u32)));
    for (name, op) in [("popcount(u)", 0u8), ("leading zeros of u", 1), ("trailing zeros of u", 2), ("popcount(w)", 3), ("trailing zeros of w", 4)] {
        let v = match op {
            0 => b.core(e, Ty::I32, Op::PopulationCount(u)),
            1 => b.core(e, Ty::I32, Op::LeadingZeros(u)),
            2 => b.core(e, Ty::I32, Op::TrailingZeros(u)),
            3 => b.core(e, Ty::I32, Op::PopulationCount(w)),
            _ => b.core(e, Ty::I32, Op::TrailingZeros(w)),
        };
        words.push((
            name,
            v,
            Box::new(move |u, w, _| match op {
                0 => u.count_ones(),
                1 => u.leading_zeros(),
                2 => u.trailing_zeros(),
                3 => w.count_ones(),
                _ => w.trailing_zeros(),
            }),
        ));
    }
    let big = c(&mut b, 70000);
    let v = b.cmp(e, IntPred::Ult, u, big);
    bits.push(("u < 70000", v, Box::new(|u, _, _| u < 70000)));
    let v = b.cmp(e, IntPred::Ult, u, hundred);
    bits.push(("u < 100", v, Box::new(|u, _, _| u < 100)));
    let top = b.int(e, IntOp::LShr, u, sixteen);
    let zero = c(&mut b, 0);
    let v = b.cmp(e, IntPred::Eq, top, zero);
    bits.push(("u >> 16 == 0", v, Box::new(|u, _, _| u >> 16 == 0)));
    let v = b.cmp(e, IntPred::Ne, u2, u);
    bits.push(("u + 2 != u", v, Box::new(|_, _, _| true)));
    let v = b.cmp(e, IntPred::Ult, w, w);
    bits.push(("w < w", v, Box::new(|_, _, _| false)));
    let v = b.cmp(e, IntPred::Slt, u, zero);
    bits.push(("u < 0 signed", v, Box::new(|u, _, _| (u as i32) < 0)));
    let v = b.cmp(e, IntPred::Ugt, u2, u);
    bits.push(("u + 2 > u", v, Box::new(|u, _, _| u.wrapping_add(2) > u)));
    Unknowns { b, words, bits }
}

fn samples() -> Vec<(u32, Vec<u32>)> {
    let mut r = Random::new(43);
    let mut values = vec![0u32, 1, 2, 3, 99, 100, 101, 255, 256, 32767, 32768, 65534, 65535];
    for _ in 0..8 {
        values.push(r.below(65536) as u32);
    }
    values.into_iter().map(|u| (u, (0..32).map(|_| r.next() as u32).collect())).collect()
}

fn representable(unknowns: &[UnknownInfo], form: &Form, truth: u32) -> bool {
    super::super::hazard::may_overlap_for_tests(unknowns, form, &Form::constant(truth))
}

fn exactly_representable(unknowns: &[UnknownInfo], form: &Form, truth: u32) -> Option<bool> {
    let choices: Vec<Vec<u32>> = form
        .terms
        .iter()
        .map(|&(u, _)| match (&unknowns[u as usize].values, unknowns[u as usize].range) {
            (Some(set), _) => Some(set.to_vec()),
            (None, Some((lo, hi))) if hi - lo < 4096 => Some((lo..=hi).collect()),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let mut index: Vec<usize> = vec![0; choices.len()];
    loop {
        let value = form
            .terms
            .iter()
            .zip(&index)
            .zip(&choices)
            .fold(form.constant, |acc, ((&(_, c), &i), set)| acc.wrapping_add(c.wrapping_mul(set[i])));
        if value == truth {
            return Some(true);
        }
        let mut k = 0;
        loop {
            if k == index.len() {
                return Some(false);
            }
            if index[k] + 1 < choices[k].len() {
                index[k] += 1;
                break;
            }
            index[k] = 0;
            k += 1;
        }
    }
}

#[test]
fn shifts_of_wrapping_words_decide_only_what_holds() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let z = b.load(e, Space::Global, MemSize::B32, table, yes);
    let two = b.constant(e, Ty::I32, 2);
    let scaled = b.int(e, IntOp::Shl, z, two);
    let back = b.int(e, IntOp::LShr, scaled, two);
    let same = b.cmp(e, IntPred::Eq, back, z);
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let decided = addresses(&b, &env, |a| a.bit(same, 0, None).0);
    assert_eq!(decided, None, "(z << 2) >> 2 == z fails for z >= 2^30, so it cannot be decided true");
}

#[test]
fn arithmetic_shifts_and_field_masks_keep_what_the_operations_determine() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let u = b.load(e, Space::Global, MemSize::U16, table, yes);
    let one = b.constant(e, Ty::I32, 1);
    let halved = b.int(e, IntOp::AShr, u, one);
    let field = b.constant(e, Ty::I32, 0x0ff0);
    let masked = b.int(e, IntOp::And, u, field);
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut loose = Vec::new();
    addresses(&b, &env, |a| {
        let f = a.values.value(halved, 0, None).0.form;
        if a.bounds(&f) != Some((0, 32767)) {
            loose.push(format!("u ashr 1: {:?} over {:?}, not within [0, 32767]", f, a.bounds(&f)));
        }
        let f = a.values.value(masked, 0, None).0.form;
        for (value, holds) in [(0u32, true), (0x10, true), (0x0ff0, true), (0x0ff1, false), (0x18, false)] {
            if exactly_representable(a.unknowns(), &f, value) != Some(holds) {
                loose.push(format!("u & 0xff0: {:?} at {:#x}: not {}", f, value, holds));
            }
        }
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

fn joined_values() -> (Build, Vec<(ValueId, Vec<Box<dyn Fn(u32) -> u32>>)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let x = b.load(e, Space::Global, MemSize::B32, table, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let first = b.cmp(e, IntPred::Ne, x, zero);
    let one = b.constant(e, Ty::I32, 1);
    let second = b.cmp(e, IntPred::Ugt, x, one);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let thirty_one = b.constant(e, Ty::I32, 31);
    let mirrored = b.int(e, IntOp::Sub, thirty_one, lane);
    let eight = b.constant(e, Ty::I32, 8);
    let past = b.int(e, IntOp::Add, lane, eight);
    let (five, nine, thirteen) = (b.constant(e, Ty::I32, 5), b.constant(e, Ty::I32, 9), b.constant(e, Ty::I32, 13));
    let (below, hundred) = (b.constant(e, Ty::I32, 0xffff_fff0), b.constant(e, Ty::I32, 100));
    let (middle, _) = b.block(&[Ty::I1]);
    let (join, j) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32, Ty::I32]);
    b.cond_br(e, first, (middle, vec![k.exec]), (join, vec![k.exec, lane, five, lane, five]));
    let exec = b.f.blocks[&middle].params[0].0;
    b.cond_br(
        middle,
        second,
        (join, vec![exec, mirrored, nine, past, below]),
        (join, vec![exec, mirrored, thirteen, lane, hundred]),
    );
    let values: Vec<(ValueId, Vec<Box<dyn Fn(u32) -> u32>>)> = vec![
        (j[1], vec![Box::new(|l| l), Box::new(|l| 31 - l)]),
        (j[2], vec![Box::new(|_| 5), Box::new(|_| 9), Box::new(|_| 13)]),
        (j[3], vec![Box::new(|l| l), Box::new(|l| l + 8)]),
        (j[4], vec![Box::new(|_| 5), Box::new(|_| 0xffff_fff0), Box::new(|_| 100)]),
    ];
    (b, values)
}

#[test]
fn joins_hold_every_value_an_incoming_edge_brings() {
    let (b, values) = joined_values();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut wrong = Vec::new();
    addresses(&b, &env, |a| {
        for (v, truths) in &values {
            for l in 0..32 {
                let f = a.values.value(*v, l, None).0.form;
                for truth in truths {
                    if !representable(a.unknowns(), &f, truth(l as u32)) {
                        wrong.push(format!("{:?} lane {}: {:?} cannot be {}", v, l, f, truth(l as u32)));
                    }
                }
            }
        }
    });
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn joins_hold_only_the_values_on_the_steps_between_them() {
    let (b, values) = joined_values();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut loose = Vec::new();
    addresses(&b, &env, |a| {
        let f = a.values.value(values[1].0, 3, None).0.form;
        for (value, holds) in [(5u32, true), (7, false), (9, true), (11, false), (13, true), (17, false)] {
            if exactly_representable(a.unknowns(), &f, value) != Some(holds) {
                loose.push(format!("{:?} at {}: not {}", f, value, holds));
            }
        }
        let f = a.values.value(values[3].0, 3, None).0.form;
        for (value, holds) in [(5u32, true), (0xffff_fff0, true), (100, true), (0, false), (52, false), (0xffff_fff1, false), (6, false)] {
            if exactly_representable(a.unknowns(), &f, value) != Some(holds) {
                loose.push(format!("{:?} at {:#x}: not {}", f, value, holds));
            }
        }
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

fn low_bit_masks() -> (Build, Vec<(&'static str, ValueId, Box<dyn Fn(u32) -> u32>)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let z = b.load(e, Space::Global, MemSize::B32, table, yes);
    let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
    let mut words: Vec<(&'static str, ValueId, Box<dyn Fn(u32) -> u32>)> = Vec::new();
    let one = c(&mut b, 1);
    let v = b.int(e, IntOp::Or, z, one);
    words.push(("z | 1", v, Box::new(|z| z | 1)));
    let seven = c(&mut b, 7);
    let v = b.int(e, IntOp::Or, seven, z);
    words.push(("7 | z", v, Box::new(|z| z | 7)));
    let clear = c(&mut b, !7u32 as u64);
    let v = b.int(e, IntOp::And, z, clear);
    words.push(("z & ~7", v, Box::new(|z| z & !7)));
    let two = c(&mut b, 2);
    let v = b.int(e, IntOp::Or, z, two);
    words.push(("z | 2", v, Box::new(|z| z | 2)));
    let three = c(&mut b, 3);
    let bumped = b.int(e, IntOp::Add, z, three);
    let v = b.int(e, IntOp::Or, bumped, one);
    words.push(("(z + 3) | 1", v, Box::new(|z| z.wrapping_add(3) | 1)));
    let five = c(&mut b, 5);
    let v = b.int(e, IntOp::Or, z, five);
    words.push(("z | 5", v, Box::new(|z| z | 5)));
    let v = b.int(e, IntOp::Xor, z, three);
    words.push(("z ^ 3", v, Box::new(|z| z ^ 3)));
    let top = c(&mut b, 0x8000_0001);
    let v = b.int(e, IntOp::Xor, z, top);
    words.push(("z ^ 0x80000001", v, Box::new(|z| z ^ 0x8000_0001)));
    (b, words)
}

#[test]
fn low_bit_masks_hold_every_value_of_a_uniform_word() {
    let (b, words) = low_bit_masks();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut wrong = Vec::new();
    addresses(&b, &env, |a| {
        let forms: Vec<Form> = words.iter().map(|&(_, v, _)| a.values.value(v, 0, None).0.form).collect();
        let mut r = Random::new(71);
        let mut values = vec![0u32, 1, 2, 6, 7, 8, 0x7fff_ffff, 0x8000_0000, u32::MAX - 2, u32::MAX];
        values.extend((0..40).map(|_| r.next() as u32));
        for z in values {
            for (i, (name, _, truth)) in words.iter().enumerate() {
                if !representable(a.unknowns(), &forms[i], truth(z)) {
                    wrong.push(format!("{} at z = {:#x}: {:?} cannot be {:#x}", name, z, forms[i], truth(z)));
                }
            }
        }
    });
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn low_bit_masks_keep_the_high_part_of_a_uniform_word() {
    let (b, words) = low_bit_masks();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut loose = Vec::new();
    addresses(&b, &env, |a| {
        let &(_, odd, _) = words.iter().find(|w| w.0 == "(z + 3) | 1").unwrap();
        let f = a.values.value(odd, 0, None).0.form;
        if f.terms.is_empty() || f.terms.iter().any(|&(_, c)| c % 2 != 0) || f.constant % 2 != 1 {
            loose.push(format!("(z + 3) | 1: {:?} can be even", f));
        }
        for (name, scale, constant) in [("z | 1", 2u32, 1u32), ("7 | z", 8, 7), ("z & ~7", 8, 0)] {
            let &(_, v, _) = words.iter().find(|w| w.0 == name).unwrap();
            let f = a.values.value(v, 0, None).0.form;
            let shaped = f.constant == constant && f.terms.len() == 1 && f.terms[0].1 == scale;
            if !shaped {
                loose.push(format!("{}: {:?}, not {} times one unknown plus {}", name, f, scale, constant));
            }
        }
        for (name, scale, low) in [("z | 2", 4u32, (2u32, 3u32)), ("z | 5", 8, (5, 7)), ("z ^ 3", 4, (0, 3))] {
            let &(_, v, _) = words.iter().find(|w| w.0 == name).unwrap();
            let f = a.values.value(v, 0, None).0.form;
            let ranges: Vec<Option<(u32, u32)>> = f.terms.iter().map(|&(u, _)| a.unknowns()[u as usize].range).collect();
            let shaped = f.constant == 0
                && f.terms.len() == 2
                && f.terms.iter().any(|&(_, c)| c == scale)
                && f.terms.iter().zip(&ranges).any(|(&(_, c), &r)| c == 1 && r == Some(low));
            if !shaped {
                loose.push(format!("{}: {:?} over {:?}, not {} times the high part plus a low part in {:?}", name, f, ranges, scale, low));
            }
        }
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

#[test]
fn value_forms_hold_every_value_the_loaded_words_can_give() {
    let Unknowns { b, words, bits } = unknowns();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut wrong = Vec::new();
    addresses(&b, &env, |a| {
        let forms: Vec<Vec<Form>> = words.iter().map(|&(_, v, _)| (0..32).map(|l| a.values.value(v, l, None).0.form).collect()).collect();
        let decided: Vec<Vec<Option<bool>>> = bits.iter().map(|&(_, v, _)| (0..32).map(|l| a.bit(v, l, None).0).collect()).collect();
        for (u, w) in samples() {
            for (i, (name, _, truth)) in words.iter().enumerate() {
                for l in 0..32 {
                    let t = truth(u, w[l], l as u32);
                    if !representable(a.unknowns(), &forms[i][l], t) {
                        wrong.push(format!("{} lane {} at u = {}: {:?} cannot be {:#x}", name, l, u, forms[i][l], t));
                    }
                }
            }
            for (i, (name1, _, t1)) in words.iter().enumerate() {
                for (j, (name2, _, t2)) in words.iter().enumerate() {
                    for (l1, l2) in [(0usize, 0usize), (0, 5), (3, 17)] {
                        let (f1, f2) = (&forms[i][l1], &forms[j][l2]);
                        if f1.terms.is_empty() || f1.terms != f2.terms {
                            continue;
                        }
                        let d = f1.constant.wrapping_sub(f2.constant);
                        let t = t1(u, w[l1], l1 as u32).wrapping_sub(t2(u, w[l2], l2 as u32));
                        if d != t {
                            wrong.push(format!("{} lane {} minus {} lane {} at u = {}: {} not {}", name1, l1, name2, l2, u, d as i32, t as i32));
                        }
                    }
                }
            }
            for (i, (name, _, truth)) in bits.iter().enumerate() {
                for l in 0..32 {
                    if let Some(k) = decided[i][l] {
                        if k != truth(u, w[l], l as u32) {
                            wrong.push(format!("{} lane {} at u = {}: decided {}", name, l, u, k));
                        }
                    }
                }
            }
        }
    });
    wrong.sort();
    wrong.dedup();
    assert!(wrong.is_empty(), "{} wrong, first {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
}

#[test]
fn value_forms_keep_what_the_operations_determine() {
    let Unknowns { b, words, bits } = unknowns();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut loose = Vec::new();
    addresses(&b, &env, |a| {
        let form = |a: &mut Addresses, name: &str| {
            let &(_, v, _) = words.iter().find(|w| w.0 == name).unwrap();
            a.values.value(v, 3, None).0.form
        };
        let base = form(a, "u + 4 lane");
        let bounds = |a: &Addresses, f: &Form| a.bounds(f);
        for (name, expected) in [
            ("4u + 8", Some((4u32, 8u32))),
            ("u | 0x10000", Some((1, 0x1_0000))),
            ("u ^ ~0", Some((u32::MAX, u32::MAX))),
            ("(u << 16) >> 16", Some((1, 0))),
            ("u - 1", Some((1, u32::MAX))),
        ] {
            let f = form(a, name);
            let (scale, constant) = expected.unwrap();
            let want = Form {
                constant,
                terms: base.terms.iter().map(|&(t, c)| (t, c.wrapping_mul(scale))).collect(),
            };
            if f != want {
                loose.push(format!("{}: {:?}, not {:?}", name, f, want));
            }
        }
        for (name, range) in [("u >> 1", (0u64, 32767u64)), ("(u + 2) >> 1", (1, 32768)), ("u & 0xff", (0, 255)), ("((u >> 14) + 8) & 7", (0, 3)), ("u >> 17", (0, 0)), ("(u + 2) - u", (2, 2)), ("hi(zext u)", (0, 0))] {
            let f = form(a, name);
            match bounds(a, &f) {
                Some(found) if found == range => {}
                other => loose.push(format!("{}: bounds {:?}, not {:?}", name, other, range)),
            }
        }
        for (name, want) in [("u < 70000", true), ("u >> 16 == 0", true), ("u + 2 != u", true), ("w < w", false), ("u < 0 signed", false)] {
            let &(_, v, _) = bits.iter().find(|x| x.0 == name).unwrap();
            if a.bit(v, 3, None).0 != Some(want) {
                loose.push(format!("{}: undecided", name));
            }
        }
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

struct Placed {
    b: Build,
    cases: Vec<(&'static str, ValueId, Box<dyn Fn(usize) -> Option<Region>>)>,
}

fn placed() -> Placed {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let yes = b.constant(e, Ty::I1, 1);
    let v = b.load(e, Space::Global, MemSize::B32, second, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let unknown = b.cmp(e, IntPred::Ne, v, zero);
    let five = b.constant(e, Ty::I32, 5);
    let low = b.cmp(e, IntPred::Ult, lane, five);
    let alloc = |id| Some(Region::Allocation(id));
    let mut cases: Vec<(&'static str, ValueId, Box<dyn Fn(usize) -> Option<Region>>)> = Vec::new();
    cases.push(("buf", first, Box::new(move |_| alloc(1))));
    let own = byte_offset(&mut b, e, first, lane, 4);
    cases.push(("buf + 4 lane", own, Box::new(move |_| alloc(1))));
    let lanes = b.core(e, Ty::I64, Op::Select(low, first, second));
    cases.push(("select(lane < 5, first, second)", lanes, Box::new(move |l| alloc(if l < 5 { 1 } else { 2 }))));
    let lo = b.core(e, Ty::I32, Op::UnpackLo(first));
    let hi = b.core(e, Ty::I32, Op::UnpackHi(first));
    let eight = b.constant(e, Ty::I32, 8);
    let lo8 = b.int(e, IntOp::Add, lo, eight);
    let rebuilt = b.core(e, Ty::I64, Op::Pack64(lo8, hi));
    cases.push(("pack(lo + 8, hi)", rebuilt, Box::new(move |_| alloc(1))));
    let base = b.core(e, Ty::I64, Op::Env(Env::ScratchBase));
    let sixteen = b.constant(e, Ty::I64, 16);
    let private = b.int(e, IntOp::Add, base, sixteen);
    cases.push(("scratch base + 16", private, Box::new(|_| Some(Region::Private))));
    let kernarg = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
    cases.push(("kernarg pointer", kernarg, Box::new(|_| Some(Region::Kernarg))));
    let zero64 = b.constant(e, Ty::I64, 0);
    let or = b.int(e, IntOp::Or, first, zero64);
    cases.push(("buf | 0", or, Box::new(move |_| alloc(1))));
    let three = b.constant(e, Ty::I32, 3);
    let read = b.wave(e, WaveOp::ReadLane, vec![lo, three, zero]);
    let back = b.core(e, Ty::I64, Op::Pack64(read, hi));
    cases.push(("pack(readlane(lo, 3), hi)", back, Box::new(move |_| alloc(1))));
    let slot = b.constant(e, Ty::I32, 16);
    b.store(e, Space::Scratch, MemSize::B64, slot, first, k.exec);
    let spilled = b.load(e, Space::Scratch, MemSize::B64, slot, k.exec);
    cases.push(("reloaded spill of buf", spilled, Box::new(move |_| alloc(1))));
    let either = b.core(e, Ty::I64, Op::Select(unknown, first, second));
    cases.push(("select(loaded bit, first, second)", either, Box::new(move |_| None)));
    let integer = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
    cases.push(("zext lane", integer, Box::new(|_| None)));
    Placed { b, cases }
}

fn settle(a: &mut Addresses, cases: &[(&'static str, ValueId, Box<dyn Fn(usize) -> Option<Region>>)]) {
    loop {
        for (_, v, _) in cases {
            for l in 0..32 {
                for refine in [false, true] {
                    a.regions(*v, l, None, refine);
                }
            }
        }
        if !a.settle_loops() {
            return;
        }
    }
}

#[test]
fn regions_hold_the_region_every_address_points_into() {
    let Placed { b, cases } = placed();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut wrong = Vec::new();
    addresses(&b, &env, |a| {
        settle(a, &cases);
        for (name, v, truth) in &cases {
            for l in [0usize, 3, 7, 31] {
                let truths: Vec<Option<Region>> = match truth(l) {
                    None if *name == "select(loaded bit, first, second)" => vec![Some(Region::Allocation(1)), Some(Region::Allocation(2))],
                    t => vec![t],
                };
                for refine in [false, true] {
                    let set = a.regions(*v, l, None, refine);
                    for t in &truths {
                        let meets = |x: Option<Region>, y: Option<Region>| x == y;
                        if !set.reaches(*t, meets) && !(t.is_none() && set.lost()) {
                            wrong.push(format!("{} lane {} refine {}: {:?} misses {:?}", name, l, refine, set, t));
                        }
                    }
                }
                let value = a.values.value(*v, l, None).0;
                if let Some(r) = value.region {
                    if !truths.contains(&Some(r)) {
                        wrong.push(format!("{} lane {}: value says {:?}, truth {:?}", name, l, r, truths));
                    }
                }
            }
        }
    });
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn regions_name_only_the_region_an_address_points_into_when_it_is_known() {
    let Placed { b, cases } = placed();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let mut loose = Vec::new();
    addresses(&b, &env, |a| {
        settle(a, &cases);
        for (name, v, truth) in &cases {
            if matches!(*name, "select(loaded bit, first, second)" | "zext lane") {
                continue;
            }
            for l in [0usize, 7] {
                let set = a.regions(*v, l, None, true);
                if set != Regions::one(truth(l)) {
                    loose.push(format!("{} lane {}: {:?}", name, l, set));
                }
            }
        }
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

struct Branches {
    b: Build,
    blocks: Vec<(&'static str, BlockId, bool)>,
}

fn branches(lanes: u32) -> Branches {
    let (mut b, k, extra) = Build::kernel_with(&[(ParameterSource::Sgpr(crate::rdna_spmd::engine::WORKGROUP_ID_X), Ty::I32)]);
    b.entry.workgroup_ids[0] = Some(crate::rdna_spmd::engine::Field { register: crate::rdna_spmd::engine::WORKGROUP_ID_X, shift: 0 });
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let u = b.load(e, Space::Global, MemSize::U16, table, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let mut blocks = Vec::new();
    let mut at = e;
    let mut exec = k.exec;
    let mut conditions: Vec<(&'static str, &'static str, Box<dyn Fn(&mut Build, BlockId) -> ValueId>, bool, bool)> = Vec::new();
    conditions.push((
        "u < 70000 taken",
        "u < 70000 not taken",
        Box::new(move |b, x| {
            let k = b.constant(x, Ty::I32, 70000);
            b.cmp(x, IntPred::Ult, u, k)
        }),
        true,
        false,
    ));
    conditions.push((
        "u < 100 taken",
        "u < 100 not taken",
        Box::new(move |b, x| {
            let k = b.constant(x, Ty::I32, 100);
            b.cmp(x, IntPred::Ult, u, k)
        }),
        true,
        true,
    ));
    conditions.push((
        "any(lane == 3) taken",
        "any(lane == 3) not taken",
        Box::new(move |b, x| {
            let three = b.constant(x, Ty::I32, 3);
            let is = b.cmp(x, IntPred::Eq, lane, three);
            b.wave(x, WaveOp::Any, vec![is])
        }),
        lanes > 3,
        lanes <= 3,
    ));
    let wgid = extra[0];
    conditions.push((
        "workgroup < 4 taken",
        "workgroup < 4 not taken",
        Box::new(move |b, x| {
            let four = b.constant(x, Ty::I32, 4);
            b.cmp(x, IntPred::Ult, wgid, four)
        }),
        true,
        false,
    ));
    let mut reached = true;
    for (yes_name, no_name, condition, taken, other) in conditions {
        let c = condition(&mut b, at);
        let (then, t) = b.block(&[Ty::I1]);
        let (skip, _) = b.block(&[Ty::I1]);
        b.cond_br(at, c, (then, vec![exec]), (skip, vec![exec]));
        blocks.push((yes_name, then, reached && taken));
        blocks.push((no_name, skip, reached && other));
        reached = reached && taken;
        at = then;
        exec = t[0];
    }
    Branches { b, blocks }
}

fn reaching(lanes: u32) -> (Vec<String>, Vec<String>) {
    let Branches { b, blocks } = branches(lanes);
    let mut env = environment(lanes, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    env.grid = [4, 1, 1];
    let (mut wrong, mut loose) = (Vec::new(), Vec::new());
    addresses(&b, &env, |a| {
        for (name, block, truth) in &blocks {
            match (a.reaches_block(*block), *truth) {
                (false, true) => wrong.push(format!("{}: said unreachable", name)),
                (true, false) => loose.push(format!("{}: said reachable", name)),
                _ => {}
            }
        }
    });
    (wrong, loose)
}

#[test]
fn reaches_block_drops_only_blocks_no_execution_reaches() {
    let mut wrong = reaching(32).0;
    wrong.extend(reaching(2).0);
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn reaches_block_drops_every_block_whose_branch_is_decided_against_it() {
    let mut loose = reaching(32).1;
    loose.extend(reaching(2).1);
    assert!(loose.is_empty(), "{:?}", loose);
}

struct Counted {
    b: Build,
    index: ValueId,
    truth: Vec<u32>,
}

fn counted(start: u32, step: u32, pred: IntPred, limit: u32) -> Counted {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let s = b.constant(e, Ty::I32, start as u64);
    let (body, p) = b.block(&[Ty::I1, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, s]);
    let d = b.constant(body, Ty::I32, step as u64);
    let next = b.int(body, IntOp::Add, p[1], d);
    let l = b.constant(body, Ty::I32, limit as u64);
    let again = b.cmp(body, pred, next, l);
    b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
    let mut truth = vec![start];
    let mut i = start;
    loop {
        let next = i.wrapping_add(step);
        if !super::compare(pred, next, limit) || truth.len() > 1000 {
            break;
        }
        truth.push(next);
        i = next;
    }
    Counted { b, index: p[1], truth }
}

fn loops() -> Vec<(&'static str, Counted)> {
    vec![
        ("0, 1, ... while < 4", counted(0, 1, IntPred::Ult, 4)),
        ("0, 2, ... while != 8", counted(0, 2, IntPred::Ne, 8)),
        ("10, 7, ... while > 0 signed", counted(10, (-3i32) as u32, IntPred::Sgt, 0)),
        ("0, 5, ... while <= 20", counted(0, 5, IntPred::Ule, 20)),
        ("3, 4, ... while < 3", counted(3, 1, IntPred::Ult, 3)),
    ]
}

#[test]
fn loop_values_hold_every_value_an_iteration_gives() {
    let env = environment(32, &[(0, 1, 0x1000)]);
    let mut wrong = Vec::new();
    for (name, c) in loops() {
        addresses(&c.b, &env, |a| {
            let form = a.values.value(c.index, 0, None).0.form;
            for &t in &c.truth {
                if !representable(a.unknowns(), &form, t) {
                    wrong.push(format!("{}: {:?} cannot be {}", name, form, t));
                }
            }
        });
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn loop_values_hold_only_the_values_the_iterations_give() {
    let env = environment(32, &[(0, 1, 0x1000)]);
    let mut loose = Vec::new();
    for (name, c) in loops() {
        addresses(&c.b, &env, |a| {
            let form = a.values.value(c.index, 0, None).0.form;
            let extra: Vec<u32> = vec![c.truth[0].wrapping_sub(1), c.truth.last().unwrap().wrapping_add(1), 0x7fff_ffff]
                .into_iter()
                .filter(|x| !c.truth.contains(x))
                .filter(|&x| exactly_representable(a.unknowns(), &form, x) != Some(false))
                .collect();
            if !extra.is_empty() {
                loose.push(format!("{}: {:?} also holds {:?}", name, form, extra));
            }
        });
    }
    assert!(loose.is_empty(), "{:?}", loose);
}

fn carried_param(alternate: bool) -> (Build, ValueId, [u32; 2]) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, first, second, zero]);
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[3], one);
    let four = b.constant(body, Ty::I32, 4);
    let again = b.cmp(body, IntPred::Ult, next, four);
    let (carried, spare) = if alternate { (p[2], p[1]) } else { (p[1], p[2]) };
    b.cond_br(body, again, (body, vec![p[0], carried, spare, next]), (exit, vec![p[0]]));
    (b, p[1], [0x1000, 0x2000])
}

#[test]
fn a_parameter_the_loop_passes_back_unchanged_keeps_its_entering_value() {
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let (b, p, _) = carried_param(false);
    let form = addresses(&b, &env, |a| a.values.value(p, 0, None).0.form);
    assert_eq!(form, Form::constant(0x1000));
}

#[test]
fn a_parameter_the_loop_swaps_holds_both_of_its_values() {
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let (b, p, truth) = carried_param(true);
    let missed: Vec<u32> = addresses(&b, &env, |a| {
        let form = a.values.value(p, 0, None).0.form;
        truth.iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the parameter is the first pointer in even iterations and the second in odd ones: {:?}", missed);
}

#[test]
fn value_and_bit_are_right_whenever_they_decide_an_operation_of_lane_known_words() {
    let (wrong, _) = decided(41, 3000);
    assert!(wrong.is_empty(), "{} wrong, first {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
}

#[test]
fn value_and_bit_decide_every_operation_of_lane_known_words() {
    let (_, undecided) = decided(41, 3000);
    let mut kinds: Vec<String> = undecided.iter().map(|s| s.split(" lane").next().unwrap().to_string()).collect();
    kinds.sort();
    kinds.dedup();
    assert!(undecided.is_empty(), "{} undecided, kinds {:?}", undecided.len(), kinds);
}

fn scratch(b: &mut Build, e: BlockId, offset: u64) -> (ValueId, ValueId) {
    let base = b.core(e, Ty::I64, Op::Env(Env::ScratchBase));
    let k = b.constant(e, Ty::I64, offset);
    (base, b.int(e, IntOp::Add, base, k))
}

fn aperture(bound: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> Option<bool> {
    let (mut b, _) = Build::kernel();
    let e = BlockId(0);
    let (base, p) = scratch(&mut b, e, 8);
    let end = bound(&mut b, e, base);
    let above = b.cmp(e, IntPred::Uge, p, base);
    let below = b.cmp(e, IntPred::Ult, p, end);
    let inside = b.int(e, IntOp::And, above, below);
    addresses(&b, &environment(32, &[]), |a| a.bit(inside, 0, None).0)
}

#[test]
fn aperture_tests_find_a_scratch_pointer_below_the_scratch_size() {
    let found = aperture(|b, e, base| {
        let size = b.core(e, Ty::I64, Op::Env(Env::ScratchSize));
        b.int(e, IntOp::Add, base, size)
    });
    assert_eq!(found, Some(true));
}

#[test]
fn aperture_tests_decide_only_what_their_bound_gives() {
    let found = aperture(|b, e, base| {
        let four = b.constant(e, Ty::I64, 4);
        b.int(e, IntOp::Add, base, four)
    });
    assert_ne!(found, Some(true), "base + 8 is not below base + 4");
}

#[test]
fn comparisons_of_offsets_into_one_region_decide_what_holds_for_every_base() {
    let (mut b, _) = Build::kernel();
    let e = BlockId(0);
    let (_, far) = scratch(&mut b, e, 16);
    let (_, near) = scratch(&mut b, e, 8);
    let far_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, far));
    let near_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, near));
    let cases = [
        ("base + 16 != base + 8", b.cmp(e, IntPred::Ne, far, near)),
        ("low(base + 16) != low(base + 8)", b.cmp(e, IntPred::Ne, far_low, near_low)),
        ("low(base + 16) == low(base + 16)", b.cmp(e, IntPred::Eq, far_low, far_low)),
    ];
    let undecided: Vec<&str> = addresses(&b, &environment(32, &[]), |a| {
        cases.iter().filter(|c| a.bit(c.1, 0, None).0 != Some(true)).map(|c| c.0).collect()
    });
    assert!(undecided.is_empty(), "{:?}", undecided);
}

#[test]
fn comparisons_of_offsets_into_a_region_decide_nothing_that_depends_on_its_base() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let (_, p) = scratch(&mut b, e, 16);
    let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
    let sixteen = b.constant(e, Ty::I32, 16);
    let hundred = b.constant(e, Ty::I32, 100);
    let zero = b.constant(e, Ty::I32, 0);
    let address = b.constant(e, Ty::I64, 0x1010);
    let cases = [
        ("low(base + 16) == 16", b.cmp(e, IntPred::Eq, low, sixteen)),
        ("low(base + 16) < 100", b.cmp(e, IntPred::Ult, low, hundred)),
        ("low(kernarg) == 0", b.cmp(e, IntPred::Eq, k.kernarg.0, zero)),
        ("base + 16 == 0x1010", b.cmp(e, IntPred::Eq, p, address)),
    ];
    let decided: Vec<&str> = addresses(&b, &environment(32, &[]), |a| {
        cases.iter().filter(|c| a.bit(c.1, 0, None).0.is_some()).map(|c| c.0).collect()
    });
    assert!(decided.is_empty(), "each holds for some bases and not for others: {:?}", decided);
}

#[test]
fn values_computed_from_offsets_into_a_region_fix_nothing_that_depends_on_its_base() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let (_, p) = scratch(&mut b, e, 16);
    let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
    let four = b.constant(e, Ty::I32, 4);
    let two = b.constant(e, Ty::I32, 2);
    let three = b.constant(e, Ty::I64, 3);
    let kernarg = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
    let cases = [
        ("low(base + 16) >> 4", b.int(e, IntOp::LShr, low, four)),
        ("low(base + 16) * 2", b.int(e, IntOp::Mul, low, two)),
        ("(base + 16) | 3", b.int(e, IntOp::Or, p, three)),
        ("(base + 16) - kernarg", b.int(e, IntOp::Sub, p, kernarg)),
        ("low(kernarg) >> 4", b.int(e, IntOp::LShr, k.kernarg.0, four)),
    ];
    let fixed: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
        cases
            .iter()
            .filter_map(|c| {
                let value = a.values.value(c.1, 0, None).0;
                (value.region.is_none() && value.form.as_constant().is_some())
                    .then(|| format!("{} = {:#x}", c.0, value.form.constant))
            })
            .collect()
    });
    assert!(fixed.is_empty(), "each depends on where the regions start: {:?}", fixed);
}

#[test]
fn values_computed_from_offsets_into_one_region_keep_what_holds_for_every_base() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let (base, far) = scratch(&mut b, e, 16);
    let (_, near) = scratch(&mut b, e, 8);
    let far_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, far));
    let near_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, near));
    let high = b.constant(e, Ty::I64, 0xffff_ffff_0000_0000);
    let aperture = b.int(e, IntOp::And, base, high);
    let kernarg = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
    let twelve = b.constant(e, Ty::I64, 12);
    let past = b.int(e, IntOp::Add, kernarg, twelve);
    let cases = [
        ("(base + 16) - base", b.int(e, IntOp::Sub, far, base), 16),
        ("low(base + 16) - low(base + 8)", b.int(e, IntOp::Sub, far_low, near_low), 8),
        ("base & 0xffffffff00000000", aperture, 0),
        ("(kernarg + 12) - kernarg", b.int(e, IntOp::Sub, past, kernarg), 12),
    ];
    let loose: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
        cases
            .iter()
            .filter_map(|c| {
                let value = a.values.value(c.1, 0, None).0;
                (value.form.as_constant() != Some(c.2)).then(|| format!("{}: {:?}", c.0, value.form))
            })
            .collect()
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

fn lane_read(selector: impl Fn(&mut Build, &Kernel, BlockId, ValueId) -> ValueId) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let four = b.constant(e, Ty::I32, 4);
    let eight = b.constant(e, Ty::I32, 8);
    let scaled = b.int(e, IntOp::Mul, lane, four);
    let x = b.int(e, IntOp::Add, scaled, eight);
    let s = selector(&mut b, &k, e, lane);
    let register = b.constant(e, Ty::I32, 0);
    let read = b.wave(e, WaveOp::ReadLane, vec![x, s, register]);
    (b, read)
}

#[test]
fn lane_reads_of_lanes_the_wave_lacks_fix_no_value_but_zero() {
    let (b, read) = lane_read(|b, _, e, _| b.constant(e, Ty::I32, 20));
    let form = addresses(&b, &environment(16, &[]), |a| a.values.value(read, 3, None).0.form);
    assert!(form.as_constant().is_none_or(|k| k == 0), "lane 20 is not in a wave of 16 lanes, so the read gives 0, not {:?}", form);
}

#[test]
fn lane_reads_of_lanes_the_wave_lacks_give_zero() {
    let (b, read) = lane_read(|b, _, e, _| b.constant(e, Ty::I32, 20));
    let form = addresses(&b, &environment(16, &[]), |a| a.values.value(read, 3, None).0.form);
    assert_eq!(form, Form::constant(0));
}

#[test]
fn lane_reads_of_lanes_the_last_wave_lacks_point_into_no_region() {
    let (mut b, _) = Build::kernel();
    let e = BlockId(0);
    let (_, p) = scratch(&mut b, e, 16);
    let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
    let high = b.core(e, Ty::I32, Op::UnpackHi(p));
    let last = b.constant(e, Ty::I32, 31);
    let register = b.constant(e, Ty::I32, 0);
    let read = b.wave(e, WaveOp::ReadLane, vec![low, last, register]);
    let pointer = b.core(e, Ty::I64, Op::Pack64(read, high));
    let regions: Vec<Regions> = addresses(&b, &environment(48, &[]), |a| {
        (0..2)
            .map(|wave| {
                a.enter(wave);
                a.regions(pointer, 0, None, true)
            })
            .collect()
    });
    assert_eq!(
        regions,
        vec![Regions::one(Some(Region::Private)), Regions::one(None)],
        "the second wave of 48 lanes lacks lane 31, so its read gives 0 and points into no region"
    );
}

#[test]
fn lane_reads_of_one_word_in_a_partial_wave_fix_no_value_but_zero() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let seven = b.constant(e, Ty::I32, 7);
    let table = k.buffer(&mut b, e, 0);
    let yes = b.constant(e, Ty::I1, 1);
    let selector = b.load(e, Space::Global, MemSize::B32, table, yes);
    let register = b.constant(e, Ty::I32, 0);
    let read = b.wave(e, WaveOp::ReadLane, vec![seven, selector, register]);
    let form = addresses(&b, &environment(16, &[(0, 1, 0x1000)]), |a| a.values.value(read, 3, None).0.form);
    assert!(form.as_constant().is_none_or(|k| k == 0), "the selector may name a lane the wave of 16 lacks: {:?}", form);
}

#[test]
fn lane_reads_with_a_uniform_selector_give_every_lane_one_value() {
    let (b, read) = lane_read(|b, k, e, _| {
        let table = k.buffer(b, e, 0);
        let yes = b.constant(e, Ty::I1, 1);
        b.load(e, Space::Global, MemSize::B32, table, yes)
    });
    let env = environment(32, &[(0, 1, 0x1000)]);
    let (third, fifth) = addresses(&b, &env, |a| (a.values.value(read, 3, None).0.form, a.values.value(read, 5, None).0.form));
    assert_eq!(third, fifth, "every lane reads the lane the one selector names");
}

#[test]
fn lane_reads_with_a_varying_selector_read_each_lane_own_source() {
    let (b, read) = lane_read(|_, _, _, lane| lane);
    let form = addresses(&b, &environment(32, &[]), |a| a.values.value(read, 3, None).0.form);
    assert!(form.as_constant().is_none_or(|k| k == 20), "lane 3 reads its own 3 * 4 + 8, not {:?}", form);
}

#[test]
fn lane_reads_with_a_selector_each_lane_loads_leave_the_lanes_apart() {
    let (b, read) = lane_read(|b, k, e, lane| {
        let buf = k.buffer(b, e, 0);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        let address = b.int(e, IntOp::Add, buf, wide);
        let yes = b.constant(e, Ty::I1, 1);
        b.load(e, Space::Global, MemSize::U8, address, yes)
    });
    let env = environment(32, &[(0, 1, 0x1000)]);
    let (third, fifth) = addresses(&b, &env, |a| (a.values.value(read, 3, None).0.form, a.values.value(read, 5, None).0.form));
    assert!(third != fifth || third.as_constant().is_some(), "lanes 3 and 5 may read different lanes: {:?}", third);
}

#[test]
fn scratch_pointers_read_from_another_lane_equal_the_lane_own() {
    let (mut b, _) = Build::kernel();
    let e = BlockId(0);
    let (_, p) = scratch(&mut b, e, 16);
    let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
    let zero = b.constant(e, Ty::I32, 0);
    let register = b.constant(e, Ty::I32, 0);
    let read = b.wave(e, WaveOp::ReadLane, vec![low, zero, register]);
    let same = b.cmp(e, IntPred::Eq, read, low);
    let found = addresses(&b, &environment(32, &[]), |a| a.bit(same, 3, None).0);
    assert_eq!(found, Some(true), "every lane has the same scratch base");
}

#[test]
fn spills_at_the_top_of_the_scratch_offsets_reload_the_pointer_they_hold() {
    for slot in [0xFFFF_FFF8u64, 0xFFFF_FFFE] {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I32, slot);
        b.store(e, Space::Scratch, MemSize::B64, at, first, k.exec);
        let spilled = b.load(e, Space::Scratch, MemSize::B64, at, k.exec);
        let set = addresses(&b, &environment(32, &[(0, 1, 0x1000)]), |a| a.regions(spilled, 0, None, true));
        assert_eq!(set, Regions::one(Some(Region::Allocation(1))), "the spill at {:#x} holds the buffer pointer", slot);
    }
}

#[test]
fn float_conversions_to_a_bit_decide_only_what_the_saturating_conversion_gives() {
    let (mut b, _) = Build::kernel();
    let e = BlockId(0);
    let cases: Vec<(&str, Cvt, f32, bool)> = vec![
        ("signed -1.0", Cvt::FloatToSignedSatRtz, -1.0, true),
        ("signed 1.0000001", Cvt::FloatToSignedSatRtz, f32::from_bits(0x3f80_0001), false),
        ("signed -2.5", Cvt::FloatToSignedSatRtz, -2.5, true),
        ("unsigned 1.0", Cvt::FloatToUnsignedSatRtz, 1.0, true),
        ("unsigned 3.0", Cvt::FloatToUnsignedSatRtz, 3.0, true),
        ("unsigned 0.5", Cvt::FloatToUnsignedSatRtz, 0.5, false),
    ];
    let bits: Vec<(&str, ValueId, bool)> = cases
        .iter()
        .map(|&(name, cvt, x, truth)| {
            let k = b.constant(e, Ty::F32, x.to_bits() as u64);
            (name, b.core(e, Ty::I1, Op::Convert(cvt, Ty::I1, k)), truth)
        })
        .collect();
    let wrong: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
        bits.iter()
            .filter_map(|&(name, v, truth)| {
                let found = a.bit(v, 0, None).0;
                found.is_some_and(|x| x != truth).then(|| format!("{}: {:?}", name, found))
            })
            .collect()
    });
    assert!(wrong.is_empty(), "{:?}", wrong);
}

struct Wide {
    b: Build,
    cases: Vec<(&'static str, ValueId, Vec<u64>)>,
}

fn wide_values() -> Wide {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let c64 = |b: &mut Build, k: u64| b.constant(e, Ty::I64, k);
    let c32 = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let u = b.load(e, Space::Global, MemSize::U8, table, yes);
    let wide_u = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
    let mut cases = Vec::new();
    let lo = c32(&mut b, 0x89ab_cdef);
    let hi = c32(&mut b, 0x0123_4567);
    let packed = b.core(e, Ty::I64, Op::Pack64(lo, hi));
    cases.push(("pack", packed, vec![0x0123_4567_89ab_cdef]));
    let near = c64(&mut b, 0xffff_fff0);
    let twenty = c64(&mut b, 0x20);
    let carried = b.int(e, IntOp::Add, near, twenty);
    cases.push(("0xfffffff0 + 0x20", carried, vec![0x1_0000_0010]));
    let sixteen = c64(&mut b, 0x10);
    let small = b.int(e, IntOp::Sub, sixteen, twenty);
    cases.push(("0x10 - 0x20", small, vec![0xffff_ffff_ffff_fff0]));
    let edge = c64(&mut b, 0xffff_ff80);
    let maybe = b.int(e, IntOp::Add, edge, wide_u);
    cases.push(("0xffffff80 + u", maybe, (0..256u64).map(|x| 0xffff_ff80 + x).collect()));
    let four = c64(&mut b, 36);
    let lifted = b.int(e, IntOp::Shl, wide_u, four);
    cases.push(("u << 36", lifted, (0..256u64).map(|x| x << 36).collect()));
    let eight = c64(&mut b, 8);
    let dropped = b.int(e, IntOp::LShr, packed, eight);
    cases.push(("pack >> 8", dropped, vec![0x0123_4567_89ab_cdef >> 8]));
    let far = c64(&mut b, 40);
    let top = b.int(e, IntOp::LShr, packed, far);
    cases.push(("pack >> 40", top, vec![0x0123_4567_89ab_cdef >> 40]));
    let minus = c32(&mut b, 0xffff_fff0);
    let extended = b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, minus));
    cases.push(("sext -16", extended, vec![0xffff_ffff_ffff_fff0]));
    let masked_bits = c64(&mut b, 0xff00_0000_ffff_0000);
    let masked = b.int(e, IntOp::And, packed, masked_bits);
    cases.push(("pack & mask", masked, vec![0x0123_4567_89ab_cdef & 0xff00_0000_ffff_0000]));
    let shifted_up = b.int(e, IntOp::Shl, packed, eight);
    cases.push(("pack << 8", shifted_up, vec![0x0123_4567_89ab_cdef << 8]));
    Wide { b, cases }
}

#[test]
fn high_words_hold_the_high_half_of_every_wide_value() {
    let Wide { b, cases } = wide_values();
    let missed: Vec<String> = addresses(&b, &environment(32, &[(8, 1, 0x1000)]), |a| {
        let mut missed = Vec::new();
        for (name, v, truths) in &cases {
            let (low, high) = (a.values.value(*v, 0, None).0.form, a.high(*v, 0));
            for &t in truths {
                if !representable(a.unknowns(), &low, t as u32) || !representable(a.unknowns(), &high, (t >> 32) as u32) {
                    missed.push(format!("{}: {:#x} with {:?} and {:?}", name, t, low, high));
                    break;
                }
            }
        }
        missed
    });
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn high_words_are_exact_where_the_operations_determine_them() {
    let Wide { b, cases } = wide_values();
    let loose: Vec<String> = addresses(&b, &environment(32, &[(8, 1, 0x1000)]), |a| {
        cases
            .iter()
            .filter(|(_, _, truths)| truths.len() == 1)
            .filter_map(|(name, v, truths)| {
                let (low, high) = (a.values.value(*v, 0, None).0.form, a.high(*v, 0));
                let t = truths[0];
                (low.as_constant() != Some(t as u32) || high.as_constant() != Some((t >> 32) as u32))
                    .then(|| format!("{}: {:?} and {:?}", name, low, high))
            })
            .collect()
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

fn affine_loop(trips: u64) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let one = b.constant(e, Ty::I32, 1);
    let zero = b.constant(e, Ty::I32, 0);
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, one, zero]);
    let three = b.constant(body, Ty::I32, 3);
    let tripled = b.int(body, IntOp::Mul, p[1], three);
    let one = b.constant(body, Ty::I32, 1);
    let next_value = b.int(body, IntOp::Add, tripled, one);
    let next = b.int(body, IntOp::Add, p[2], one);
    let limit = b.constant(body, Ty::I32, trips);
    let again = b.cmp(body, IntPred::Ult, next, limit);
    b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
    (b, p[1])
}

fn carried_values(b: &Build, v: ValueId) -> Option<Vec<u32>> {
    addresses(b, &environment(32, &[]), |a| {
        let form = a.values.value(v, 0, None).0.form;
        match form.terms.as_slice() {
            [(u, 1)] if form.constant == 0 => a.unknowns()[*u as usize].values.as_ref().map(|s| s.to_vec()),
            _ => None,
        }
    })
}

#[test]
fn loop_values_hold_every_value_an_affine_step_carries() {
    let (b, v) = affine_loop(5);
    let truth = [1u32, 4, 13, 40, 121];
    let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
        let form = a.values.value(v, 0, None).0.form;
        truth.iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the parameter runs 1, 4, 13, 40, 121: {:?}", missed);
}

#[test]
fn loop_values_are_exactly_those_an_affine_step_carries() {
    let (b, v) = affine_loop(5);
    assert_eq!(carried_values(&b, v), Some(vec![1, 4, 13, 40, 121]));
}

fn affine_loop_from_either() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let zero = b.constant(e, Ty::I32, 0);
    let is_zero = b.cmp(e, IntPred::Eq, u, zero);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let start = b.core(e, Ty::I32, Op::Select(is_zero, one, two));
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, start, zero]);
    let three = b.constant(body, Ty::I32, 3);
    let tripled = b.int(body, IntOp::Mul, p[1], three);
    let one = b.constant(body, Ty::I32, 1);
    let next_value = b.int(body, IntOp::Add, tripled, one);
    let next = b.int(body, IntOp::Add, p[2], one);
    let limit = b.constant(body, Ty::I32, 5);
    let again = b.cmp(body, IntPred::Ult, next, limit);
    b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
    (b, p[1])
}

#[test]
fn loop_values_hold_every_value_an_affine_step_carries_from_either_start() {
    let (b, v) = affine_loop_from_either();
    let truth = [1u32, 2, 4, 7, 13, 22, 40, 67, 121, 202];
    let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
        let form = a.values.value(v, 0, None).0.form;
        truth.iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the parameter runs 1, 4, 13, 40, 121 or 2, 7, 22, 67, 202: {:?}", missed);
}

#[test]
fn loop_values_are_exactly_those_an_affine_step_carries_from_either_start() {
    let (b, v) = affine_loop_from_either();
    assert_eq!(carried_values(&b, v), Some(vec![1, 2, 4, 7, 13, 22, 40, 67, 121, 202]));
}

fn hundred_affine_values() -> Vec<u32> {
    let mut values: Vec<u32> = std::iter::successors(Some(1u32), |x| Some(x.wrapping_mul(3).wrapping_add(1))).take(100).collect();
    values.sort_unstable();
    values
}

#[test]
fn loop_values_hold_every_value_an_affine_step_carries_over_a_hundred_iterations() {
    let (b, v) = affine_loop(100);
    let truth = hundred_affine_values();
    let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
        let form = a.values.value(v, 0, None).0.form;
        truth.iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the parameter runs 1, 4, 13 and on for 100 iterations: {:?}", missed);
}

#[test]
fn loop_values_are_exactly_those_an_affine_step_carries_over_a_hundred_iterations() {
    let (b, v) = affine_loop(100);
    let carried = carried_values(&b, v).map(|mut s| {
        s.sort_unstable();
        s
    });
    assert_eq!(carried, Some(hundred_affine_values()));
}

#[test]
fn float_conversions_to_integers_give_what_the_saturating_conversion_gives() {
    let (mut b, _) = Build::kernel();
    let e = BlockId(0);
    let cases: Vec<(&str, Cvt, Ty, f32, u32)> = vec![
        ("signed bit of -1.0", Cvt::FloatToSignedSatRtz, Ty::I1, -1.0, 1),
        ("signed bit of -0.5", Cvt::FloatToSignedSatRtz, Ty::I1, -0.5, 0),
        ("signed bit of 1.0", Cvt::FloatToSignedSatRtz, Ty::I1, 1.0, 0),
        ("unsigned bit of 1.0", Cvt::FloatToUnsignedSatRtz, Ty::I1, 1.0, 1),
        ("unsigned bit of 0.99", Cvt::FloatToUnsignedSatRtz, Ty::I1, 0.99, 0),
        ("unsigned bit of NaN", Cvt::FloatToUnsignedSatRtz, Ty::I1, f32::NAN, 0),
        ("signed word of -2.5", Cvt::FloatToSignedSatRtz, Ty::I32, -2.5, (-2i32) as u32),
        ("signed word of 3e9", Cvt::FloatToSignedSatRtz, Ty::I32, 3e9, i32::MAX as u32),
        ("unsigned word of -7.0", Cvt::FloatToUnsignedSatRtz, Ty::I32, -7.0, 0),
        ("unsigned word of 5e9", Cvt::FloatToUnsignedSatRtz, Ty::I32, 5e9, u32::MAX),
        ("signed low word of -1.0", Cvt::FloatToSignedSatRtz, Ty::I64, -1.0, u32::MAX),
        ("unsigned low word of 2^33 + 2^10", Cvt::FloatToUnsignedSatRtz, Ty::I64, 8589935616.0, 1024),
    ];
    let converted: Vec<(&str, ValueId, Ty, u32)> = cases
        .iter()
        .map(|&(name, cvt, to, x, truth)| {
            let k = b.constant(e, Ty::F32, x.to_bits() as u64);
            (name, b.core(e, to, Op::Convert(cvt, to, k)), to, truth)
        })
        .collect();
    let loose: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
        converted
            .iter()
            .filter_map(|&(name, v, to, truth)| {
                let found = if to == Ty::I1 {
                    a.bit(v, 0, None).0.map(|x| x as u32)
                } else {
                    a.values.value(v, 0, None).0.form.as_constant()
                };
                (found != Some(truth)).then(|| format!("{}: {:?}", name, found))
            })
            .collect()
    });
    assert!(loose.is_empty(), "{:?}", loose);
}

fn loaded_word(b: &mut Build, k: &Kernel, e: BlockId, offset: u64, size: MemSize) -> ValueId {
    let table = k.buffer(b, e, 8);
    let at = b.constant(e, Ty::I64, offset);
    let address = b.int(e, IntOp::Add, table, at);
    let yes = b.constant(e, Ty::I1, 1);
    b.load(e, Space::Global, size, address, yes)
}

fn two_words() -> Environment {
    environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)])
}

fn identities() -> (Build, Vec<(&'static str, ValueId)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
    let byte = loaded_word(&mut b, &k, e, 8, MemSize::U8);
    let thirty_one = b.constant(e, Ty::I32, 31);
    let s = b.int(e, IntOp::And, byte, thirty_one);
    let uv = b.int(e, IntOp::Mul, u, v);
    let vu = b.int(e, IntOp::Mul, v, u);
    let products = b.int(e, IntOp::Sub, uv, vu);
    let both = b.int(e, IntOp::And, u, v);
    let either = b.int(e, IntOp::Or, u, v);
    let sum = b.int(e, IntOp::Add, u, v);
    let parts = b.int(e, IntOp::Add, both, either);
    let bits = b.int(e, IntOp::Sub, parts, sum);
    let shifted = b.int(e, IntOp::Shl, u, s);
    let one = b.constant(e, Ty::I32, 1);
    let power = b.int(e, IntOp::Shl, one, s);
    let scaled = b.int(e, IntOp::Mul, u, power);
    let shifts = b.int(e, IntOp::Sub, shifted, scaled);
    (b, vec![("u * v - v * u", products), ("(u & v) + (u | v) - (u + v)", bits), ("(u << s) - u * (1 << s)", shifts)])
}

#[test]
fn identities_of_nonlinear_operations_hold_zero() {
    let (b, cases) = identities();
    let missed: Vec<&str> = addresses(&b, &two_words(), |a| {
        cases
            .iter()
            .filter(|c| {
                let form = a.values.value(c.1, 3, None).0.form;
                !representable(a.unknowns(), &form, 0)
            })
            .map(|c| c.0)
            .collect()
    });
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn identities_of_nonlinear_operations_give_zero() {
    let (b, cases) = identities();
    let loose: Vec<&str> = addresses(&b, &two_words(), |a| {
        cases.iter().filter(|c| a.values.value(c.1, 3, None).0.form.as_constant() != Some(0)).map(|c| c.0).collect()
    });
    assert!(loose.is_empty(), "each is 0 for every u, v and s: {:?}", loose);
}

fn counted_bits() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let count = b.core(e, Ty::I32, Op::PopulationCount(u));
    let limit = b.constant(e, Ty::I32, 32);
    let within = b.cmp(e, IntPred::Ule, count, limit);
    (b, within)
}

#[test]
fn bit_counts_decide_nothing_false() {
    let (b, within) = counted_bits();
    let found = addresses(&b, &two_words(), |a| a.bit(within, 0, None).0);
    assert_ne!(found, Some(false), "a word has at most 32 bits set");
}

#[test]
fn bit_counts_stay_within_the_width() {
    let (b, within) = counted_bits();
    let found = addresses(&b, &two_words(), |a| a.bit(within, 0, None).0);
    assert_eq!(found, Some(true), "a word has at most 32 bits set");
}

fn wide_sums() -> (Build, Vec<(&'static str, ValueId, u32)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let step = b.constant(e, Ty::I64, 256);
    let past = b.int(e, IntOp::Add, buf, step);
    let eight = b.constant(e, Ty::I64, 8);
    let units = b.int(e, IntOp::LShr, past, eight);
    let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, units));
    let far = b.constant(e, Ty::I64, 0x1_0000_0000);
    let above = b.int(e, IntOp::Add, buf, far);
    let high = b.core(e, Ty::I32, Op::UnpackHi(above));
    (b, vec![("low((buf + 256) >> 8)", low, 0x11), ("high(buf + 2^32)", high, 1)])
}

#[test]
fn wide_shifts_and_high_halves_of_sums_hold_their_values() {
    let (b, cases) = wide_sums();
    let missed: Vec<&str> = addresses(&b, &two_words(), |a| {
        cases
            .iter()
            .filter(|c| {
                let form = a.values.value(c.1, 0, None).0.form;
                !representable(a.unknowns(), &form, c.2)
            })
            .map(|c| c.0)
            .collect()
    });
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn wide_shifts_and_high_halves_of_sums_give_their_values() {
    let (b, cases) = wide_sums();
    let loose: Vec<String> = addresses(&b, &two_words(), |a| {
        cases
            .iter()
            .filter_map(|c| {
                let form = a.values.value(c.1, 0, None).0.form;
                (form.as_constant() != Some(c.2)).then(|| format!("{}: {:?}", c.0, form))
            })
            .collect()
    });
    assert!(loose.is_empty(), "buf is 0x1000: {:?}", loose);
}

fn shared_word() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let four = b.constant(e, Ty::I32, 4);
    let own = b.int(e, IntOp::Mul, lane, four);
    b.store(e, Space::Lds, MemSize::B32, own, lane, k.exec);
    let back = b.load(e, Space::Lds, MemSize::B32, own, k.exec);
    (b, back)
}

#[test]
fn lds_words_read_back_hold_the_word_the_lane_stored() {
    let (b, back) = shared_word();
    let held = addresses(&b, &two_words(), |a| {
        let form = a.values.value(back, 3, None).0.form;
        representable(a.unknowns(), &form, 3)
    });
    assert!(held, "lane 3 reads back the 3 it stored");
}

#[test]
fn lds_words_read_back_give_the_word_the_lane_stored() {
    let (b, back) = shared_word();
    let form = addresses(&b, &two_words(), |a| a.values.value(back, 3, None).0.form);
    assert_eq!(form, Form::constant(3), "word 3 of the LDS only ever holds 3");
}

fn squared_loop() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let one = b.constant(e, Ty::I32, 1);
    let zero = b.constant(e, Ty::I32, 0);
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, one, zero]);
    let squared = b.int(body, IntOp::Mul, p[1], p[1]);
    let one = b.constant(body, Ty::I32, 1);
    let next_value = b.int(body, IntOp::Add, squared, one);
    let next = b.int(body, IntOp::Add, p[2], one);
    let three = b.constant(body, Ty::I32, 3);
    let again = b.cmp(body, IntPred::Ult, next, three);
    b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
    (b, p[1])
}

#[test]
fn loop_values_hold_every_value_a_square_step_carries() {
    let (b, v) = squared_loop();
    let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
        let form = a.values.value(v, 0, None).0.form;
        [1u32, 2, 5].iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the parameter runs 1, 2, 5: {:?}", missed);
}

#[test]
fn loop_values_are_exactly_those_a_square_step_carries() {
    let (b, v) = squared_loop();
    assert_eq!(carried_values(&b, v), Some(vec![1, 2, 5]));
}

fn halves_in_two_blocks() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let one = b.constant(e, Ty::I32, 1);
    let x = b.int(e, IntOp::LShr, u, one);
    let (next, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    b.br(e, next, vec![k.exec, u, x]);
    let one = b.constant(next, Ty::I32, 1);
    let y = b.int(next, IntOp::LShr, p[1], one);
    let difference = b.int(next, IntOp::Sub, p[2], y);
    (b, difference)
}

#[test]
fn halves_of_one_word_in_two_blocks_hold_their_difference() {
    let (b, difference) = halves_in_two_blocks();
    let held = addresses(&b, &two_words(), |a| {
        let form = a.values.value(difference, 0, None).0.form;
        representable(a.unknowns(), &form, 0)
    });
    assert!(held);
}

#[test]
fn halves_of_one_word_in_two_blocks_are_equal() {
    let (b, difference) = halves_in_two_blocks();
    let form = addresses(&b, &two_words(), |a| a.values.value(difference, 0, None).0.form);
    assert_eq!(form, Form::constant(0), "both are u >> 1 of the same u");
}

fn chosen_after_branch(entered: bool) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let zero = b.constant(e, Ty::I32, 0);
    let is_zero = b.cmp(e, IntPred::Eq, u, zero);
    let (then, t) = b.block(&[Ty::I1, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I32]);
    b.cond_br(e, is_zero, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
    let (block, p) = if entered { (then, t) } else { (other, o) };
    let zero = b.constant(block, Ty::I32, 0);
    let test = b.cmp(block, IntPred::Eq, p[1], zero);
    let seven = b.constant(block, Ty::I32, 7);
    let nine = b.constant(block, Ty::I32, 9);
    let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
    (b, chosen)
}

#[test]
fn selects_after_a_branch_hold_the_arm_the_branch_leaves() {
    let missed: Vec<bool> = [true, false]
        .iter()
        .copied()
        .filter(|&entered| {
            let (b, chosen) = chosen_after_branch(entered);
            let truth = if entered { 7 } else { 9 };
            !addresses(&b, &two_words(), |a| {
                let form = a.values.value(chosen, 0, None).0.form;
                representable(a.unknowns(), &form, truth)
            })
        })
        .collect();
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn selects_after_a_branch_take_the_arm_the_branch_decides() {
    let (b, chosen) = chosen_after_branch(true);
    let form = addresses(&b, &two_words(), |a| a.values.value(chosen, 0, None).0.form);
    assert_eq!(form, Form::constant(7), "the block runs only when u is 0");
}

fn zero_cases(b: &Build, cases: &[(&'static str, ValueId)], exact: bool) -> Vec<String> {
    addresses(b, &two_words(), |a| {
        cases
            .iter()
            .filter_map(|&(name, v)| {
                let form = a.values.value(v, 3, None).0.form;
                let wrong = if exact { form.as_constant() != Some(0) } else { !representable(a.unknowns(), &form, 0) };
                wrong.then(|| format!("{}: {:?}", name, form))
            })
            .collect()
    })
}

fn distributed_products() -> (Build, Vec<(&'static str, ValueId)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
    let w = loaded_word(&mut b, &k, e, 8, MemSize::B32);
    let one = b.constant(e, Ty::I32, 1);
    let next = b.int(e, IntOp::Add, v, one);
    let product = b.int(e, IntOp::Mul, u, next);
    let uv = b.int(e, IntOp::Mul, u, v);
    let sum = b.int(e, IntOp::Add, uv, u);
    let distributed = b.int(e, IntOp::Sub, product, sum);
    let vw = b.int(e, IntOp::Mul, v, w);
    let left = b.int(e, IntOp::Mul, u, vw);
    let uv = b.int(e, IntOp::Mul, u, v);
    let right = b.int(e, IntOp::Mul, uv, w);
    let associated = b.int(e, IntOp::Sub, left, right);
    (b, vec![("u * (v + 1) - (u * v + u)", distributed), ("u * (v * w) - (u * v) * w", associated)])
}

#[test]
fn distributed_and_regrouped_products_hold_zero() {
    let (b, cases) = distributed_products();
    let missed = zero_cases(&b, &cases, false);
    assert!(missed.is_empty(), "{:?}", missed);
}

fn polynomials() -> (Build, [ValueId; 3], Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::U16);
    let w = loaded_word(&mut b, &k, e, 8, MemSize::U8);
    let c = |b: &mut Build, k: u32| b.constant(e, Ty::I32, k as u64);
    let mut out: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)> = Vec::new();
    let (three, five, seven) = (c(&mut b, 3), c(&mut b, 5), c(&mut b, 7));
    let u3 = b.int(e, IntOp::Add, u, three);
    let v5 = b.int(e, IntOp::Add, v, five);
    let x = b.int(e, IntOp::Mul, u3, v5);
    out.push(("(u + 3)(v + 5)", x, Box::new(|u, v, _| u.wrapping_add(3).wrapping_mul(v + 5))));
    let difference = b.int(e, IntOp::Sub, u, v);
    let sum = b.int(e, IntOp::Add, u, v);
    let x = b.int(e, IntOp::Mul, difference, sum);
    out.push(("(u - v)(u + v)", x, Box::new(|u, v, _| u.wrapping_sub(v).wrapping_mul(u.wrapping_add(v)))));
    let vw = b.int(e, IntOp::Mul, v, w);
    let x = b.int(e, IntOp::Mul, u, vw);
    out.push(("u (v w)", x, Box::new(|u, v, w| u.wrapping_mul(v * w))));
    let uv = b.int(e, IntOp::Mul, u, v);
    let uv7 = b.int(e, IntOp::Add, uv, seven);
    let two = c(&mut b, 2);
    let w2 = b.int(e, IntOp::Sub, w, two);
    let x = b.int(e, IntOp::Mul, uv7, w2);
    out.push(("(u v + 7)(w - 2)", x, Box::new(|u, v, w| u.wrapping_mul(v).wrapping_add(7).wrapping_mul(w.wrapping_sub(2)))));
    let (four, one) = (c(&mut b, 4), c(&mut b, 1));
    let u2 = b.int(e, IntOp::Mul, u, two);
    let v3 = b.int(e, IntOp::Mul, v, three);
    let left = b.int(e, IntOp::Add, u2, v3);
    let w4 = b.int(e, IntOp::Mul, w, four);
    let right = b.int(e, IntOp::Add, w4, one);
    let lr = b.int(e, IntOp::Mul, left, right);
    let x = b.int(e, IntOp::Mul, lr, u);
    out.push(("(2u + 3v)(4w + 1) u", x, Box::new(|u, v, w| u.wrapping_mul(2).wrapping_add(3 * v).wrapping_mul(4 * w + 1).wrapping_mul(u))));
    let u1 = b.int(e, IntOp::Add, u, one);
    let square = b.int(e, IntOp::Mul, u1, u1);
    let x = b.int(e, IntOp::Mul, square, u1);
    out.push(("(u + 1)^3", x, Box::new(|u, _, _| u.wrapping_add(1).wrapping_mul(u.wrapping_add(1)).wrapping_mul(u.wrapping_add(1)))));
    let vv = b.int(e, IntOp::Mul, v, v);
    let x = b.int(e, IntOp::Mul, vv, vv);
    out.push(("v^4", x, Box::new(|_, v, _| v.wrapping_mul(v).wrapping_mul(v).wrapping_mul(v))));
    let sixteen = c(&mut b, 16);
    let shifted = b.int(e, IntOp::Shl, v, sixteen);
    let x = b.int(e, IntOp::Mul, shifted, w);
    out.push(("(v << 16) w", x, Box::new(|_, v, w| (v << 16).wrapping_mul(w))));
    (b, [u, v, w], out)
}

#[test]
fn products_expand_into_monomials_whose_factors_multiply_to_the_true_value() {
    let (b, words, cases) = polynomials();
    let mut wrong = Vec::new();
    addresses(&b, &two_words(), |a| {
        let bases: Vec<Form> = words.iter().map(|&x| a.values.value(x, 0, None).0.form).collect();
        let forms: Vec<Form> = cases.iter().map(|&(_, x, _)| a.values.value(x, 0, None).0.form).collect();
        let mut r = Random::new(109);
        let mut samples: Vec<[u32; 3]> = vec![[0, 0, 0], [1, 1, 1], [u32::MAX, 65535, 255], [0x8000_0000, 0x8000, 0x80], [12345, 678, 9]];
        samples.extend((0..40).map(|_| [r.next() as u32, r.below(65536) as u32, r.below(256) as u32]));
        for sample in samples {
            let value = |u: Unknown| -> Option<u32> {
                let base = |u: Unknown| bases.iter().position(|f| *f == Form::unknown(u)).map(|i| sample[i]);
                match a.values.symbols().monomials.get(&u) {
                    Some(factors) => factors.iter().try_fold(1u32, |p, &f| Some(p.wrapping_mul(base(f)?))),
                    None => base(u),
                }
            };
            for (i, (name, _, truth)) in cases.iter().enumerate() {
                let evaluated = forms[i].terms.iter().try_fold(forms[i].constant, |acc, &(u, c)| Some(acc.wrapping_add(c.wrapping_mul(value(u)?))));
                let t = truth(sample[0], sample[1], sample[2]);
                if evaluated != Some(t) {
                    wrong.push(format!("{} at {:?}: {:?} gives {:?}, not {:#x}", name, sample, forms[i], evaluated, t));
                }
            }
            for (u, factors) in &a.values.symbols().monomials {
                let Some(p) = value(*u) else {
                    continue;
                };
                if let Some((low, high)) = a.unknowns()[*u as usize].range {
                    if p < low || p > high {
                        wrong.push(format!("monomial {:?} at {:?} is {:#x}, outside {:?}", factors, sample, p, (low, high)));
                    }
                }
            }
        }
    });
    wrong.sort();
    wrong.dedup();
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
}

#[test]
fn wide_values_are_the_addresses_as_integers_whenever_they_are_given() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let s = loaded_word(&mut b, &k, e, 4, MemSize::U8);
    let t = loaded_word(&mut b, &k, e, 8, MemSize::U16);
    let zext = |b: &mut Build, x: ValueId| b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, x));
    let wide = |b: &mut Build, k: u64| b.constant(e, Ty::I64, k);
    let (zu, zs, zt) = (zext(&mut b, u), zext(&mut b, s), zext(&mut b, t));
    let mut cases: Vec<(&'static str, ValueId, bool, Box<dyn Fn(u64, u64, u64) -> u64>)> = Vec::new();
    let four = wide(&mut b, 4);
    let scaled = b.int(e, IntOp::Mul, zu, four);
    let x = b.int(e, IntOp::Add, buf, scaled);
    cases.push(("buf + 4u", x, true, Box::new(|u, _, _| 0x1000 + 4 * u)));
    let eight = wide(&mut b, 8);
    let y = b.int(e, IntOp::Sub, x, eight);
    cases.push(("buf + 4u - 8", y, true, Box::new(|u, _, _| 0x1000 + 4 * u - 8)));
    let three = wide(&mut b, 3);
    let shifted = b.int(e, IntOp::Shl, zu, three);
    let two = wide(&mut b, 2);
    let doubled = b.int(e, IntOp::Mul, two, zs);
    let x = b.int(e, IntOp::Add, shifted, doubled);
    cases.push(("(u << 3) + 2s", x, true, Box::new(|u, s, _| (u << 3) + 2 * s)));
    let row = wide(&mut b, 256);
    let rows = b.int(e, IntOp::Mul, zt, row);
    let x = b.int(e, IntOp::Add, zs, rows);
    cases.push(("s + 256t", x, true, Box::new(|_, s, t| s + 256 * t)));
    let big = wide(&mut b, 1 << 33);
    let far = b.int(e, IntOp::Mul, zu, big);
    let x = b.int(e, IntOp::Add, buf, far);
    cases.push(("buf + 2^33 u", x, false, Box::new(|u, _, _| 0x1000u64.wrapping_add(u.wrapping_mul(1 << 33)))));
    let five = b.constant(e, Ty::I32, 5);
    let bumped = b.int(e, IntOp::Add, u, five);
    let x = zext(&mut b, bumped);
    cases.push(("zext(u + 5)", x, true, Box::new(|u, _, _| (u as u32).wrapping_add(5) as u64)));
    let bumped = b.int(e, IntOp::Add, t, five);
    let x = zext(&mut b, bumped);
    cases.push(("zext(t + 5)", x, true, Box::new(|_, _, t| t + 5)));
    let x = b.int(e, IntOp::Sub, zs, zt);
    cases.push(("s - t", x, false, Box::new(|_, s, t| s.wrapping_sub(t))));
    let mut wrong = Vec::new();
    addresses(&b, &two_words(), |a| {
        let bases: Vec<Form> = [u, s, t].iter().map(|&x| a.values.value(x, 0, None).0.form).collect();
        let forms: Vec<Option<super::Wide>> = cases.iter().map(|&(_, x, _, _)| a.wide_value(x, e, 0)).collect();
        let mut r = Random::new(127);
        let mut samples: Vec<[u64; 3]> = vec![[0, 0, 0], [u32::MAX as u64, 255, 65535], [0x8000_0000, 128, 32768], [1, 1, 1]];
        samples.extend((0..40).map(|_| [r.next() as u32 as u64, r.below(256), r.below(65536)]));
        for (i, (name, _, given, truth)) in cases.iter().enumerate() {
            match &forms[i] {
                None if *given => wrong.push(format!("{}: no wide form", name)),
                None => {}
                Some(w) => {
                    for sample in &samples {
                        let at = |x: Unknown| bases.iter().position(|f| *f == Form::unknown(x)).map(|i| sample[i]);
                        let terms = w.terms.iter().try_fold(w.constant, |acc, &(x, c)| Some(acc + c * at(x)? as i128));
                        let value = terms.and_then(|terms| w.words.iter().try_fold(terms, |acc, (f, c)| {
                            let word = f.terms.iter().try_fold(f.constant, |acc, &(x, k)| Some(acc.wrapping_add(k.wrapping_mul(at(x)? as u32))))?;
                            Some(acc + c * word as i128)
                        }));
                        let t = truth(sample[0], sample[1], sample[2]) as i128;
                        if value != Some(t) {
                            wrong.push(format!("{} at {:?}: {:?} gives {:?}, not {:#x}", name, sample, w, value, t));
                        }
                    }
                }
            }
        }
    });
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

#[test]
fn distributed_and_regrouped_products_give_zero() {
    let (b, cases) = distributed_products();
    let loose = zero_cases(&b, &cases, true);
    assert!(loose.is_empty(), "each is 0 for every u, v and w: {:?}", loose);
}

fn repeated_shifts() -> (Build, Vec<(&'static str, ValueId)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
    let byte = loaded_word(&mut b, &k, e, 8, MemSize::U8);
    let thirty_one = b.constant(e, Ty::I32, 31);
    let s = b.int(e, IntOp::And, byte, thirty_one);
    let mut cases = Vec::new();
    for (name, op) in [("(u >> s) - (u >> s)", IntOp::LShr), ("(u >>> s) - (u >>> s)", IntOp::AShr)] {
        let first = b.int(e, op, u, s);
        let second = b.int(e, op, u, s);
        cases.push((name, b.int(e, IntOp::Sub, first, second)));
    }
    let wide_u = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
    let wide_v = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, v));
    let wide_s = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, s));
    let uv = b.int(e, IntOp::Mul, wide_u, wide_v);
    let vu = b.int(e, IntOp::Mul, wide_v, wide_u);
    let products = b.int(e, IntOp::Sub, uv, vu);
    cases.push(("low(zext(u) * zext(v) - zext(v) * zext(u))", b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, products))));
    let first = b.int(e, IntOp::Shl, wide_u, wide_s);
    let second = b.int(e, IntOp::Shl, wide_u, wide_s);
    let shifts = b.int(e, IntOp::Sub, first, second);
    cases.push(("low((zext(u) << s) - (zext(u) << s))", b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, shifts))));
    (b, cases)
}

#[test]
fn repeated_variable_and_wide_shifts_and_products_hold_zero_differences() {
    let (b, cases) = repeated_shifts();
    let missed = zero_cases(&b, &cases, false);
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn repeated_variable_and_wide_shifts_and_products_give_zero_differences() {
    let (b, cases) = repeated_shifts();
    let loose = zero_cases(&b, &cases, true);
    assert!(loose.is_empty(), "each is the same value twice: {:?}", loose);
}

fn products_after_two_loops() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let (first, f) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    let (middle, m) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    let (second, g) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
    let (after, a) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
    let (last, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
    b.br(e, first, vec![k.exec, table, zero]);
    let yes = b.constant(first, Ty::I1, 1);
    let x = b.load(first, Space::Global, MemSize::B32, f[1], yes);
    let one = b.constant(first, Ty::I32, 1);
    let next = b.int(first, IntOp::Add, f[2], one);
    let two = b.constant(first, Ty::I32, 2);
    let again = b.cmp(first, IntPred::Ult, next, two);
    b.cond_br(first, again, (first, vec![f[0], f[1], next]), (middle, vec![f[0], f[1], x]));
    let zero = b.constant(middle, Ty::I32, 0);
    b.br(middle, second, vec![m[0], m[1], m[2], zero]);
    let four = b.constant(second, Ty::I64, 4);
    let at = b.int(second, IntOp::Add, g[1], four);
    let yes = b.constant(second, Ty::I1, 1);
    let y = b.load(second, Space::Global, MemSize::B32, at, yes);
    let one = b.constant(second, Ty::I32, 1);
    let next = b.int(second, IntOp::Add, g[3], one);
    let two = b.constant(second, Ty::I32, 2);
    let again = b.cmp(second, IntPred::Ult, next, two);
    b.cond_br(second, again, (second, vec![g[0], g[1], g[2], next]), (after, vec![g[0], g[1], g[2], y]));
    let product = b.int(after, IntOp::Mul, a[2], a[3]);
    b.br(after, last, vec![a[0], a[2], a[3], product]);
    let again = b.int(last, IntOp::Mul, l[1], l[2]);
    let difference = b.int(last, IntOp::Sub, l[3], again);
    (b, difference)
}

#[test]
fn products_of_words_from_two_loops_in_two_blocks_hold_their_difference() {
    let (b, difference) = products_after_two_loops();
    let held = addresses(&b, &two_words(), |a| {
        let form = a.values.value(difference, 0, None).0.form;
        representable(a.unknowns(), &form, 0)
    });
    assert!(held);
}

#[test]
fn products_of_words_from_two_loops_in_two_blocks_are_equal() {
    let (b, difference) = products_after_two_loops();
    let form = addresses(&b, &two_words(), |a| a.values.value(difference, 0, None).0.form);
    assert_eq!(form, Form::constant(0), "both are x * y of the same x from the first loop and y from the second");
}

fn chosen_after_bound(entered: bool) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let four = b.constant(e, Ty::I32, 4);
    let below = b.cmp(e, IntPred::Ult, u, four);
    let (then, t) = b.block(&[Ty::I1, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I32]);
    b.cond_br(e, below, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
    let (block, p) = if entered { (then, t) } else { (other, o) };
    let eight = b.constant(block, Ty::I32, 8);
    let test = b.cmp(block, IntPred::Ult, p[1], eight);
    let seven = b.constant(block, Ty::I32, 7);
    let nine = b.constant(block, Ty::I32, 9);
    let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
    (b, chosen)
}

#[test]
fn selects_after_a_bound_hold_the_arms_the_bound_leaves() {
    let missed: Vec<(bool, u32)> = [(true, 7u32), (false, 7), (false, 9)]
        .iter()
        .copied()
        .filter(|&(entered, truth)| {
            let (b, chosen) = chosen_after_bound(entered);
            !addresses(&b, &two_words(), |a| {
                let form = a.values.value(chosen, 0, None).0.form;
                representable(a.unknowns(), &form, truth)
            })
        })
        .collect();
    assert!(missed.is_empty(), "u below 4 gives 7; u of 4 or more gives 7 below 8 and 9 above: {:?}", missed);
}

#[test]
fn selects_after_a_bound_take_the_arm_the_bound_decides() {
    let (b, chosen) = chosen_after_bound(true);
    let form = addresses(&b, &two_words(), |a| a.values.value(chosen, 0, None).0.form);
    assert_eq!(form, Form::constant(7), "the block runs only when u is below 4, so u is below 8");
}

type WordCases = Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32) -> u32>)>;

fn operations_after_a_bound(entered: bool) -> (Build, WordCases, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let sixteen = b.constant(e, Ty::I32, 16);
    let small = b.cmp(e, IntPred::Ult, u, sixteen);
    let (then, t) = b.block(&[Ty::I1, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I32]);
    b.cond_br(e, small, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
    let (block, p) = if entered { (then, t) } else { (other, o) };
    let mut cases: WordCases = Vec::new();
    let ff = b.constant(block, Ty::I32, 0xff);
    let x = b.int(block, IntOp::And, p[1], ff);
    cases.push(("u & 0xff", x, Box::new(|u, _| u & 0xff)));
    let eight = b.constant(block, Ty::I32, 8);
    let x = b.int(block, IntOp::LShr, p[1], eight);
    cases.push(("u >> 8", x, Box::new(|u, _| u >> 8)));
    let wide = b.core(block, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, p[1]));
    let near = b.constant(block, Ty::I64, 0xffff_fff0);
    let sum = b.int(block, IntOp::Add, wide, near);
    let x = b.core(block, Ty::I32, Op::UnpackHi(sum));
    cases.push(("high(u + 0xfffffff0)", x, Box::new(|u, _| ((u as u64 + 0xffff_fff0) >> 32) as u32)));
    (b, cases, p[1])
}

#[test]
fn operations_after_a_bound_hold_every_value_the_bound_leaves() {
    let mut wrong = Vec::new();
    for entered in [true, false] {
        let (b, cases, _) = operations_after_a_bound(entered);
        addresses(&b, &two_words(), |a| {
            let forms: Vec<Form> = cases.iter().map(|&(_, x, _)| a.values.value(x, 0, None).0.form).collect();
            for u in [0u32, 1, 15, 16, 17, 255, 256, 0x1234, u32::MAX - 16, u32::MAX] {
                if (u < 16) != entered {
                    continue;
                }
                for (i, (name, _, truth)) in cases.iter().enumerate() {
                    if !representable(a.unknowns(), &forms[i], truth(u, 0)) {
                        wrong.push(format!("{} in {} at {}: {:?}", name, entered, u, forms[i]));
                    }
                }
            }
        });
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn operations_after_a_bound_settle_what_the_bound_decides() {
    let (b, cases, u) = operations_after_a_bound(true);
    let loose = addresses(&b, &two_words(), |a| {
        let word = a.values.value(u, 0, None).0.form;
        let expected = [word, Form::constant(0), Form::constant(0)];
        cases
            .iter()
            .zip(expected)
            .filter_map(|(&(name, x, _), want)| {
                let form = a.values.value(x, 0, None).0.form;
                (form != want).then(|| format!("{}: {:?}, not {:?}", name, form, want))
            })
            .collect::<Vec<_>>()
    });
    assert!(loose.is_empty(), "the block runs only when u is below 16: {:?}", loose);
}

fn selects_after_an_order(entered: bool) -> (Build, WordCases) {
    use IntPred::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
    let below = b.cmp(e, Ult, u, v);
    let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    b.cond_br(e, below, (then, vec![k.exec, u, v]), (other, vec![k.exec, u, v]));
    let (block, p) = if entered { (then, t) } else { (other, o) };
    let (x, y) = (p[1], p[2]);
    let seven = b.constant(block, Ty::I32, 7);
    let nine = b.constant(block, Ty::I32, 9);
    let mut cases: WordCases = Vec::new();
    for (name, pred, flipped) in [
        ("v > u", Ugt, true),
        ("u >= v", Uge, false),
        ("u != v", Ne, false),
        ("v <= u", Ule, true),
        ("u <= v", Ule, false),
        ("u == v", Eq, false),
        ("u < v signed", Slt, false),
    ] {
        let (first, second) = if flipped { (y, x) } else { (x, y) };
        let test = b.cmp(block, pred, first, second);
        let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
        let holds = move |u: u32, v: u32| {
            let (a, c) = if flipped { (v, u) } else { (u, v) };
            if compare(pred, a, c) { 7 } else { 9 }
        };
        cases.push((name, chosen, Box::new(holds)));
    }
    (b, cases)
}

#[test]
fn selects_after_an_order_hold_the_arms_the_order_leaves() {
    let mut wrong = Vec::new();
    for entered in [true, false] {
        let (b, cases) = selects_after_an_order(entered);
        addresses(&b, &two_words(), |a| {
            let forms: Vec<Form> = cases.iter().map(|&(_, x, _)| a.values.value(x, 0, None).0.form).collect();
            let samples = [(0u32, 1u32), (1, 0), (5, 5), (0, 0x8000_0000), (0x8000_0000, 0), (u32::MAX, 3), (3, u32::MAX), (7, 9)];
            for (u, v) in samples {
                if (u < v) != entered {
                    continue;
                }
                for (i, (name, _, truth)) in cases.iter().enumerate() {
                    if !representable(a.unknowns(), &forms[i], truth(u, v)) {
                        wrong.push(format!("{} in {} at ({}, {}): {:?}", name, entered, u, v, forms[i]));
                    }
                }
            }
        });
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}

fn outcome(x: u32, y: u32) -> u8 {
    if x == y {
        return 1;
    }
    match (x < y, (x as i32) < (y as i32)) {
        (true, true) => 2,
        (true, false) => 4,
        (false, true) => 8,
        (false, false) => 16,
    }
}

const OUTCOMES: [(u32, u32); 5] = [(5, 5), (1, 2), (1, 0x8000_0000), (0x8000_0000, 1), (2, 1)];

#[test]
fn order_outcomes_hold_exactly_the_pairs_each_predicate_accepts() {
    let mut r = Random::new(137);
    let mut samples: Vec<(u32, u32)> = OUTCOMES.to_vec();
    let points = [0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, u32::MAX - 1, u32::MAX];
    for &x in &points {
        for &y in &points {
            samples.push((x, y));
        }
    }
    samples.extend((0..200).map(|_| (r.next() as u32, r.next() as u32)));
    let mut wrong = Vec::new();
    for pred in PREDICATES {
        for &(x, y) in &samples {
            if (outcomes(pred) & outcome(x, y) != 0) != compare(pred, x, y) {
                wrong.push(format!("{:?} at ({:#x}, {:#x})", pred, x, y));
            }
            if (mirrored(outcomes(pred)) & outcome(y, x) != 0) != compare(pred, x, y) {
                wrong.push(format!("{:?} swapped at ({:#x}, {:#x})", pred, x, y));
            }
        }
    }
    let seen: u8 = samples.iter().fold(0, |m, &(x, y)| m | outcome(x, y));
    assert_eq!(seen, 31, "every outcome occurs");
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

#[test]
fn orders_conjoin_exactly_and_disjoin_exactly_on_one_pair() {
    let words = [Form::unknown(0), Form::unknown(1), Form::unknown(2)];
    let holds = |orders: &[(Form, Form, u8)], values: [u32; 3]| {
        orders.iter().all(|(x, y, mask)| {
            let at = |f: &Form| values[words.iter().position(|w| w == f).unwrap()];
            mask & outcome(at(x), at(y)) != 0
        })
    };
    let mut r = Random::new(139);
    let mut wrong = Vec::new();
    for _ in 0..400 {
        let build = |r: &mut Random| {
            let mut orders = Vec::new();
            for _ in 0..r.below(3) {
                let (i, j) = (r.below(3) as usize, r.below(3) as usize);
                let pred = PREDICATES[r.below(10) as usize];
                orders = orders_conjoined(orders, order_limit(&words[i], &words[j], pred));
            }
            orders
        };
        let (a, b) = (build(&mut r), build(&mut r));
        let both = orders_conjoined(a.clone(), b.clone());
        let either = orders_disjoined(a.clone(), b.clone());
        let one_pair = a.len() == 1 && b.len() == 1 && (&a[0].0, &a[0].1) == (&b[0].0, &b[0].1);
        for x in OUTCOMES.iter().flat_map(|&(p, q)| [p, q]) {
            for y in OUTCOMES.iter().flat_map(|&(p, q)| [p, q]) {
                for z in [0u32, 1, 2, 5, 0x8000_0000] {
                    let values = [x, y, z];
                    let (ha, hb) = (holds(&a, values), holds(&b, values));
                    if holds(&both, values) != (ha && hb) {
                        wrong.push(format!("{:?} and {:?} at {:?}", a, b, values));
                    }
                    if (ha || hb) && !holds(&either, values) {
                        wrong.push(format!("{:?} or {:?} misses {:?}", a, b, values));
                    }
                    if one_pair && holds(&either, values) != (ha || hb) {
                        wrong.push(format!("{:?} or {:?} loose at {:?}", a, b, values));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(6)]);
}

fn branches_on_two_words() -> (Build, Vec<(&'static str, Box<dyn Fn(u32, u32) -> bool>, Vec<(IntPred, bool, ValueId)>)>) {
    use IntPred::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
    let fork = |b: &mut Build, at: BlockId, cond: ValueId, exec: ValueId| {
        let (yes, y) = b.block(&[Ty::I1]);
        let (no, n) = b.block(&[Ty::I1]);
        b.cond_br(at, cond, (yes, vec![exec]), (no, vec![exec]));
        ((yes, y[0]), (no, n[0]))
    };
    let c1 = b.cmp(e, Ult, u, v);
    let ((b1, e1), (x1, f1)) = fork(&mut b, e, c1, k.exec);
    let c2 = b.cmp(b1, Sgt, v, u);
    let ((b2, e2), (x2, f2)) = fork(&mut b, b1, c2, e1);
    let equal = b.cmp(x1, Eq, u, v);
    let above = b.cmp(x1, Sgt, u, v);
    let c3 = b.int(x1, IntOp::Or, equal, above);
    let ((b3, _), (x3, _)) = fork(&mut b, x1, c3, f1);
    let (j, _) = b.block(&[Ty::I1]);
    let (z, _) = b.block(&[Ty::I1]);
    let c4 = b.cmp(b2, Ne, u, v);
    b.cond_br(b2, c4, (j, vec![e2]), (z, vec![e2]));
    let c5 = b.cmp(x2, Uge, v, u);
    b.cond_br(x2, c5, (j, vec![f2]), (z, vec![f2]));
    let c1 = |u: u32, v: u32| u < v;
    let c2 = |u: u32, v: u32| (v as i32) > (u as i32);
    let c3 = |u: u32, v: u32| u == v || (u as i32) > (v as i32);
    let c4 = |u: u32, v: u32| u != v;
    let c5 = |u: u32, v: u32| v >= u;
    let reach: Vec<(&'static str, BlockId, Box<dyn Fn(u32, u32) -> bool>)> = vec![
        ("b1", b1, Box::new(move |u, v| c1(u, v))),
        ("x1", x1, Box::new(move |u, v| !c1(u, v))),
        ("b2", b2, Box::new(move |u, v| c1(u, v) && c2(u, v))),
        ("x2", x2, Box::new(move |u, v| c1(u, v) && !c2(u, v))),
        ("b3", b3, Box::new(move |u, v| !c1(u, v) && c3(u, v))),
        ("x3", x3, Box::new(move |u, v| !c1(u, v) && !c3(u, v))),
        ("j", j, Box::new(move |u, v| c1(u, v) && if c2(u, v) { c4(u, v) } else { c5(u, v) })),
        ("z", z, Box::new(move |u, v| c1(u, v) && if c2(u, v) { !c4(u, v) } else { !c5(u, v) })),
    ];
    let mut blocks = Vec::new();
    for (name, block, reaches) in reach {
        let mut queries = Vec::new();
        for pred in PREDICATES {
            queries.push((pred, false, b.cmp(block, pred, u, v)));
            queries.push((pred, true, b.cmp(block, pred, v, u)));
        }
        blocks.push((name, reaches, queries));
    }
    (b, blocks)
}

#[test]
fn comparisons_under_order_branches_decide_exactly_what_the_orders_settle() {
    let (b, blocks) = branches_on_two_words();
    let mut wrong = Vec::new();
    addresses(&b, &two_words(), |a| {
        for (name, reaches, queries) in &blocks {
            for &(pred, flipped, q) in queries {
                let mut seen = [false; 2];
                for &(u, v) in &OUTCOMES {
                    if reaches(u, v) {
                        let holds = if flipped { compare(pred, v, u) } else { compare(pred, u, v) };
                        seen[holds as usize] = true;
                    }
                }
                let truth = match seen {
                    [false, false] => continue,
                    [true, false] => Some(false),
                    [false, true] => Some(true),
                    _ => None,
                };
                let got = a.bit(q, 0, None).0;
                if got != truth {
                    wrong.push(format!("{}: {:?} flipped {}: {:?}, truth {:?}", name, pred, flipped, got, truth));
                }
            }
        }
    });
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

#[test]
fn selects_after_an_order_take_the_arm_the_order_decides() {
    let (b, cases) = selects_after_an_order(true);
    let loose = addresses(&b, &two_words(), |a| {
        cases
            .iter()
            .filter(|&&(name, _, _)| name != "u < v signed")
            .filter_map(|&(name, x, ref truth)| {
                let form = a.values.value(x, 0, None).0.form;
                let want = Form::constant(truth(0, 1));
                (form != want).then(|| format!("{}: {:?}, not {:?}", name, form, want))
            })
            .collect::<Vec<_>>()
    });
    assert!(loose.is_empty(), "the block runs only when u < v: {:?}", loose);
}

const PREDICATES: [IntPred; 10] = [
    IntPred::Eq,
    IntPred::Ne,
    IntPred::Ult,
    IntPred::Ule,
    IntPred::Ugt,
    IntPred::Uge,
    IntPred::Slt,
    IntPred::Sle,
    IntPred::Sgt,
    IntPred::Sge,
];

fn within(set: &[(u64, u64)], x: u32) -> bool {
    set.iter().any(|&(low, high)| low <= x as u64 && x as u64 <= high)
}

fn ends_of(set: &[(u64, u64)]) -> Vec<u32> {
    set.iter()
        .flat_map(|&(low, high)| [low as u32, (low as u32).wrapping_sub(1), high as u32, (high as u32).wrapping_add(1)])
        .collect()
}

fn well_formed(set: &[(u64, u64)]) -> bool {
    set.iter().all(|&(low, high)| low <= high && high <= u32::MAX as u64) && set.windows(2).all(|w| w[0].1 + 1 < w[1].0)
}

fn random_pieces(r: &mut Random) -> Vec<(u64, u64)> {
    let points = [0u32, 1, 5, 100, 0x7fff_fff0, 0x7fff_ffff, 0x8000_0000, 0x8000_0010, u32::MAX - 3, u32::MAX];
    let mut set = Vec::new();
    for _ in 0..r.below(3) {
        let mut pick = || if r.below(2) == 0 { points[r.below(points.len() as u64) as usize] } else { r.next() as u32 };
        let (x, y) = (pick(), pick());
        set.push((x.min(y) as u64, x.max(y) as u64));
    }
    set
}

#[test]
fn satisfying_holds_exactly_the_words_each_predicate_accepts() {
    let mut r = Random::new(97);
    let mut constants = vec![0u32, 1, 2, 7, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, u32::MAX - 1, u32::MAX];
    constants.extend((0..20).map(|_| r.next() as u32));
    let mut wrong = Vec::new();
    for pred in PREDICATES {
        for &k in &constants {
            let set = satisfying(pred, k);
            if !set.iter().all(|&(low, high)| low <= high && high <= u32::MAX as u64) || !set.windows(2).all(|w| w[0].1 < w[1].0) {
                wrong.push(format!("{:?} {:#x}: {:?}", pred, k, set));
            }
            let mut words = constants.clone();
            words.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
            words.extend(ends_of(&set));
            for x in words {
                if within(&set, x) != compare(pred, x, k) {
                    wrong.push(format!("{:?} {:#x} at {:#x}: {:?}", pred, k, x, set));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn piece_operations_hold_exactly_the_words_their_pieces_hold() {
    let mut r = Random::new(101);
    let mut wrong = Vec::new();
    for _ in 0..600 {
        let (a, b) = (random_pieces(&mut r), random_pieces(&mut r));
        let k = match r.below(3) {
            0 => r.below(8) as u32,
            1 => (r.below(8) as u32).wrapping_neg(),
            _ => r.next() as u32,
        };
        let shifted = shifted_pieces(&a, k);
        let common = intersected(&a, &b);
        let merged = normalized([a.clone(), b.clone()].concat());
        for set in [&shifted, &common, &merged] {
            if !well_formed(set) {
                wrong.push(format!("{:?} and {:?}: not normal {:?}", a, b, set));
            }
        }
        let mut words: Vec<u32> = [ends_of(&a), ends_of(&b), ends_of(&common), ends_of(&merged)].concat();
        words.extend(ends_of(&shifted).iter().map(|y| y.wrapping_sub(k)));
        words.extend([0, 0x8000_0000, u32::MAX]);
        for x in words {
            if within(&shifted, x.wrapping_add(k)) != within(&a, x) {
                wrong.push(format!("{:?} + {:#x} at {:#x}: {:?}", a, k, x, shifted));
            }
            if within(&common, x) != (within(&a, x) && within(&b, x)) {
                wrong.push(format!("{:?} and {:?} at {:#x}: {:?}", a, b, x, common));
            }
            if within(&merged, x) != (within(&a, x) || within(&b, x)) {
                wrong.push(format!("{:?} or {:?} at {:#x}: {:?}", a, b, x, merged));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn limits_of_word_classes_conjoin_exactly_and_disjoin_exactly_within_one_class() {
    let (u, v) = (Form::unknown(0), Form::unknown(1));
    let holds = |limits: &[(Form, Vec<(u64, u64)>)], x: u32, y: u32| {
        limits.iter().all(|(c, set)| within(set, if *c == u { x } else { y }))
    };
    let mut r = Random::new(103);
    let mut wrong = Vec::new();
    for _ in 0..400 {
        let mut build = |r: &mut Random| {
            let mut limits = Vec::new();
            for class in [&u, &v] {
                if r.below(3) != 0 {
                    let c = if r.below(2) == 0 { r.below(8) as u32 } else { r.next() as u32 };
                    let values = random_pieces(r);
                    let limit = class_limit(&class.add(&Form::constant(c)), &values);
                    if limit.0 != *class || !well_formed(&limit.1) {
                        wrong.push(format!("{:?} + {:#x} in {:?}: {:?}", class, c, values, limit));
                    }
                    for x in [ends_of(&values).iter().map(|y| y.wrapping_sub(c)).collect(), ends_of(&limit.1)].concat() {
                        if within(&limit.1, x) != within(&values, x.wrapping_add(c)) {
                            wrong.push(format!("{:?} + {:#x} in {:?} at {:#x}: {:?}", class, c, values, x, limit));
                        }
                    }
                    limits.push(limit);
                }
            }
            limits
        };
        let (a, b) = (build(&mut r), build(&mut r));
        let both = conjoined(a.clone(), b.clone());
        let either = disjoined(a.clone(), b.clone());
        let one_class = a.len() == 1 && b.len() == 1 && a[0].0 == b[0].0;
        let words = |class: &Form| -> Vec<u32> {
            let mut out = vec![0, 0x8000_0000, u32::MAX];
            for limits in [&a, &b, &both, &either] {
                for (c, set) in limits.iter() {
                    if c == class {
                        out.extend(ends_of(set));
                    }
                }
            }
            out
        };
        let (xs, ys) = (words(&u), words(&v));
        for &x in &xs {
            for &y in &ys {
                let (ha, hb) = (holds(&a, x, y), holds(&b, x, y));
                if holds(&both, x, y) != (ha && hb) {
                    wrong.push(format!("{:?} and {:?} at ({:#x}, {:#x}): {:?}", a, b, x, y, both));
                }
                if (ha || hb) && !holds(&either, x, y) {
                    wrong.push(format!("{:?} or {:?} at ({:#x}, {:#x}) misses: {:?}", a, b, x, y, either));
                }
                if one_class && holds(&either, x, y) != (ha || hb) {
                    wrong.push(format!("{:?} or {:?} at ({:#x}, {:#x}) loose: {:?}", a, b, x, y, either));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(10)]);
}

struct BranchesOnOneWord {
    b: Build,
    conditions: Vec<(IntPred, u32, u32)>,
    blocks: Vec<(&'static str, Box<dyn Fn(u32) -> bool>, Vec<(IntPred, bool, u32, u32, ValueId)>)>,
}

fn branches_on_one_word() -> BranchesOnOneWord {
    use IntPred::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let against = |b: &mut Build, at: BlockId, pred: IntPred, x: ValueId, c: u32| {
        let c = b.constant(at, Ty::I32, c as u64);
        b.cmp(at, pred, x, c)
    };
    let fork = |b: &mut Build, at: BlockId, cond: ValueId, exec: ValueId| {
        let (yes, y) = b.block(&[Ty::I1]);
        let (no, n) = b.block(&[Ty::I1]);
        b.cond_br(at, cond, (yes, vec![exec]), (no, vec![exec]));
        ((yes, y[0]), (no, n[0]))
    };
    let c1 = against(&mut b, e, Ult, u, 1000);
    let ((b1, e1), (x1, f1)) = fork(&mut b, e, c1, k.exec);
    let five = b.constant(b1, Ty::I32, 5);
    let u5 = b.int(b1, IntOp::Add, u, five);
    let above = against(&mut b, b1, Ugt, u5, 20);
    let big = against(&mut b, b1, Uge, u, 500);
    let one = b.constant(b1, Ty::I1, 1);
    let small = b.int(b1, IntOp::Xor, big, one);
    let c2 = b.int(b1, IntOp::And, above, small);
    let ((b2, e2), (x2, _)) = fork(&mut b, b1, c2, e1);
    let three_hundred = b.constant(b2, Ty::I32, 300);
    let m = b.int(b2, IntOp::Sub, u, three_hundred);
    let c3 = against(&mut b, b2, Sge, m, 0);
    let ((b3, e3), (x3, _)) = fork(&mut b, b2, c3, e2);
    let c4 = against(&mut b, b3, Ne, u, 400);
    let ((b4, e4), (x4, _)) = fork(&mut b, b3, c4, e3);
    let low = against(&mut b, b4, Ult, u, 320);
    let high = against(&mut b, b4, Ugt, u, 450);
    let c5 = b.int(b4, IntOp::Or, low, high);
    let ((b5, e5), (x5, f5)) = fork(&mut b, b4, c5, e4);
    let (j5, _) = b.block(&[Ty::I1]);
    let (z5, _) = b.block(&[Ty::I1]);
    let (y5, _) = b.block(&[Ty::I1]);
    let c8 = against(&mut b, b5, Ult, u, 310);
    b.cond_br(b5, c8, (j5, vec![e5]), (z5, vec![e5]));
    let c7 = against(&mut b, x5, Uge, u, 440);
    b.cond_br(x5, c7, (j5, vec![f5]), (y5, vec![f5]));
    let bound = b.constant(x1, Ty::I32, 0x9000_0000);
    let c6 = b.cmp(x1, Ugt, bound, u);
    let ((b6, _), (x6, _)) = fork(&mut b, x1, c6, f1);
    let conditions = vec![
        (Ult, 0, 1000),
        (Ugt, 5, 20),
        (Uge, 0, 500),
        (Sge, 300u32.wrapping_neg(), 0),
        (Ne, 0, 400),
        (Ult, 0, 320),
        (Ugt, 0, 450),
        (Ult, 0, 0x9000_0000),
        (Ult, 0, 310),
        (Uge, 0, 440),
    ];
    let c1 = |u: u32| u < 1000;
    let c2 = |u: u32| u.wrapping_add(5) > 20 && u < 500;
    let c3 = |u: u32| u.wrapping_sub(300) as i32 >= 0;
    let c4 = |u: u32| u != 400;
    let c5 = |u: u32| !(320..=450).contains(&u);
    let c6 = |u: u32| u < 0x9000_0000;
    let c7 = |u: u32| u >= 440;
    let c8 = |u: u32| u < 310;
    let reach: Vec<(&'static str, BlockId, Box<dyn Fn(u32) -> bool>)> = vec![
        ("b1", b1, Box::new(move |u| c1(u))),
        ("x1", x1, Box::new(move |u| !c1(u))),
        ("b2", b2, Box::new(move |u| c1(u) && c2(u))),
        ("x2", x2, Box::new(move |u| c1(u) && !c2(u))),
        ("b3", b3, Box::new(move |u| c1(u) && c2(u) && c3(u))),
        ("x3", x3, Box::new(move |u| c1(u) && c2(u) && !c3(u))),
        ("b4", b4, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u))),
        ("x4", x4, Box::new(move |u| c1(u) && c2(u) && c3(u) && !c4(u))),
        ("b5", b5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && c5(u))),
        ("x5", x5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && !c5(u))),
        ("j5", j5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && if c5(u) { c8(u) } else { c7(u) })),
        ("z5", z5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && c5(u) && !c8(u))),
        ("y5", y5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && !c5(u) && !c7(u))),
        ("b6", b6, Box::new(move |u| !c1(u) && c6(u))),
        ("x6", x6, Box::new(move |u| !c1(u) && !c6(u))),
    ];
    let offsets = [0u32, 1, u32::MAX, 0x8000_0000, 700];
    let constants = [0u32, 15, 16, 300, 319, 400, 401, 451, 499, 500, 999, 1000, 0x7fff_ffff, 0x8000_0000, 0x9000_0000, u32::MAX];
    let mut blocks = Vec::new();
    for (name, block, reaches) in reach {
        let mut queries = Vec::new();
        for &offset in &offsets {
            let x = if offset == 0 {
                u
            } else {
                let c = b.constant(block, Ty::I32, offset as u64);
                b.int(block, IntOp::Add, u, c)
            };
            for &c in &constants {
                let kc = b.constant(block, Ty::I32, c as u64);
                for pred in PREDICATES {
                    queries.push((pred, false, offset, c, b.cmp(block, pred, x, kc)));
                    queries.push((pred, true, offset, c, b.cmp(block, pred, kc, x)));
                }
            }
        }
        blocks.push((name, reaches, queries));
    }
    BranchesOnOneWord { b, conditions, blocks }
}

fn settled_by_branches(branches: &BranchesOnOneWord, pred: IntPred, flipped: bool, offset: u32, c: u32, reaches: &dyn Fn(u32) -> bool) -> Option<Option<bool>> {
    let holds = |u: u32| {
        let x = u.wrapping_add(offset);
        if flipped {
            compare(pred, c, x)
        } else {
            compare(pred, x, c)
        }
    };
    let mut starts = vec![0u32];
    for &(_, shift, k) in branches.conditions.iter().chain([(pred, offset, c)].iter()) {
        starts.extend([k.wrapping_sub(shift), k.wrapping_sub(shift).wrapping_add(1), shift.wrapping_neg(), 0x8000_0000u32.wrapping_sub(shift)]);
    }
    let mut seen = [false; 2];
    for u in starts {
        if reaches(u) {
            seen[holds(u) as usize] = true;
        }
    }
    match seen {
        [false, false] => None,
        [true, false] => Some(Some(false)),
        [false, true] => Some(Some(true)),
        _ => Some(None),
    }
}

#[test]
fn comparisons_under_branches_decide_only_what_every_reaching_word_gives() {
    let branches = branches_on_one_word();
    let mut wrong = Vec::new();
    addresses(&branches.b, &two_words(), |a| {
        for (name, reaches, queries) in &branches.blocks {
            for &(pred, flipped, offset, c, q) in queries {
                let Some(truth) = settled_by_branches(&branches, pred, flipped, offset, c, reaches.as_ref()) else {
                    continue;
                };
                let got = a.bit(q, 0, None).0;
                if got.is_some() && got != truth {
                    wrong.push(format!("{}: {:?} flipped {} (u + {:#x}) against {:#x}: {:?}, truth {:?}", name, pred, flipped, offset, c, got, truth));
                }
            }
        }
    });
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

#[test]
fn comparisons_under_branches_decide_what_the_conditions_on_one_word_settle() {
    let branches = branches_on_one_word();
    let mut loose = Vec::new();
    addresses(&branches.b, &two_words(), |a| {
        for (name, reaches, queries) in &branches.blocks {
            for &(pred, flipped, offset, c, q) in queries {
                let Some(truth) = settled_by_branches(&branches, pred, flipped, offset, c, reaches.as_ref()) else {
                    continue;
                };
                let got = a.bit(q, 0, None).0;
                if truth.is_some() && got != truth {
                    loose.push(format!("{}: {:?} flipped {} (u + {:#x}) against {:#x}: {:?}, truth {:?}", name, pred, flipped, offset, c, got, truth));
                }
            }
        }
    });
    assert!(loose.is_empty(), "{} loose: {:?}", loose.len(), &loose[..loose.len().min(10)]);
}

fn chosen_after_branch_on_a_source(entered: bool) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let zero = b.constant(e, Ty::I32, 0);
    let is_zero = b.cmp(e, IntPred::Eq, u, zero);
    let one = b.constant(e, Ty::I32, 1);
    let w = b.int(e, IntOp::Add, u, one);
    let (then, t) = b.block(&[Ty::I1, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I32]);
    b.cond_br(e, is_zero, (then, vec![k.exec, w]), (other, vec![k.exec, w]));
    let (block, p) = if entered { (then, t) } else { (other, o) };
    let one = b.constant(block, Ty::I32, 1);
    let test = b.cmp(block, IntPred::Eq, p[1], one);
    let seven = b.constant(block, Ty::I32, 7);
    let nine = b.constant(block, Ty::I32, 9);
    let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
    (b, chosen)
}

#[test]
fn selects_on_a_word_derived_from_the_branch_word_hold_the_arm_the_branch_leaves() {
    let missed: Vec<bool> = [true, false]
        .iter()
        .copied()
        .filter(|&entered| {
            let (b, chosen) = chosen_after_branch_on_a_source(entered);
            let truth = if entered { 7 } else { 9 };
            !addresses(&b, &two_words(), |a| {
                let form = a.values.value(chosen, 0, None).0.form;
                representable(a.unknowns(), &form, truth)
            })
        })
        .collect();
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn selects_on_a_word_derived_from_the_branch_word_take_the_arm_the_branch_decides() {
    let (b, chosen) = chosen_after_branch_on_a_source(true);
    let form = addresses(&b, &two_words(), |a| a.values.value(chosen, 0, None).0.form);
    assert_eq!(form, Form::constant(7), "the block runs only when u is 0, so u + 1 is 1");
}

fn chosen_in_a_loop_entered_on_zero() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let zero = b.constant(e, Ty::I32, 0);
    let is_zero = b.cmp(e, IntPred::Eq, u, zero);
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.cond_br(e, is_zero, (body, vec![k.exec, u, zero]), (exit, vec![k.exec]));
    let zero = b.constant(body, Ty::I32, 0);
    let test = b.cmp(body, IntPred::Eq, p[1], zero);
    let seven = b.constant(body, Ty::I32, 7);
    let nine = b.constant(body, Ty::I32, 9);
    let chosen = b.core(body, Ty::I32, Op::Select(test, seven, nine));
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[2], one);
    let three = b.constant(body, Ty::I32, 3);
    let again = b.cmp(body, IntPred::Ult, next, three);
    b.cond_br(body, again, (body, vec![p[0], p[1], next]), (exit, vec![p[0]]));
    (b, chosen)
}

#[test]
fn selects_in_a_loop_entered_on_zero_hold_the_arm_zero_takes() {
    let (b, chosen) = chosen_in_a_loop_entered_on_zero();
    let held = addresses(&b, &two_words(), |a| {
        let form = a.values.value(chosen, 0, None).0.form;
        representable(a.unknowns(), &form, 7)
    });
    assert!(held);
}

#[test]
fn selects_in_a_loop_entered_on_zero_take_the_arm_zero_takes() {
    let (b, chosen) = chosen_in_a_loop_entered_on_zero();
    let form = addresses(&b, &two_words(), |a| a.values.value(chosen, 0, None).0.form);
    assert_eq!(form, Form::constant(7), "the loop runs only when u is 0 and carries u unchanged");
}

fn affine_values(start: u32, trips: usize) -> Vec<u32> {
    std::iter::successors(Some(start), |x| Some(x.wrapping_mul(3).wrapping_add(1))).take(trips).collect()
}

#[test]
fn loop_values_hold_every_value_an_affine_step_carries_over_five_thousand_iterations() {
    let (b, v) = affine_loop(5000);
    let truth = affine_values(1, 5000);
    let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
        let form = a.values.value(v, 0, None).0.form;
        truth.iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "{} values missed", missed.len());
}

#[test]
fn loop_values_are_exactly_those_an_affine_step_carries_over_five_thousand_iterations() {
    let (b, v) = affine_loop(5000);
    let mut truth = affine_values(1, 5000);
    truth.sort_unstable();
    truth.dedup();
    let carried = carried_values(&b, v).map(|mut s| {
        s.sort_unstable();
        s
    });
    assert!(carried.as_ref() == Some(&truth), "the parameter runs 1, 4, 13 and on for 5000 iterations; found {:?} values", carried.map(|s| s.len()));
}

fn affine_loop_from_a_byte() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let mask = b.constant(e, Ty::I32, 127);
    let start = b.int(e, IntOp::And, u, mask);
    let zero = b.constant(e, Ty::I32, 0);
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, start, zero]);
    let three = b.constant(body, Ty::I32, 3);
    let tripled = b.int(body, IntOp::Mul, p[1], three);
    let one = b.constant(body, Ty::I32, 1);
    let next_value = b.int(body, IntOp::Add, tripled, one);
    let next = b.int(body, IntOp::Add, p[2], one);
    let limit = b.constant(body, Ty::I32, 3);
    let again = b.cmp(body, IntPred::Ult, next, limit);
    b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
    (b, p[1])
}

fn affine_values_from_a_byte() -> Vec<u32> {
    let mut values: Vec<u32> = (0..128).flat_map(|s| affine_values(s, 3)).collect();
    values.sort_unstable();
    values.dedup();
    values
}

#[test]
fn loop_values_hold_every_value_an_affine_step_carries_from_a_hundred_and_twenty_eight_starts() {
    let (b, v) = affine_loop_from_a_byte();
    let truth = affine_values_from_a_byte();
    let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
        let form = a.values.value(v, 0, None).0.form;
        truth.iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn loop_values_are_exactly_those_an_affine_step_carries_from_a_hundred_and_twenty_eight_starts() {
    let (b, v) = affine_loop_from_a_byte();
    let carried = addresses(&b, &two_words(), |a| {
        let form = a.values.value(v, 0, None).0.form;
        match form.terms.as_slice() {
            [(u, 1)] if form.constant == 0 => a.unknowns()[*u as usize].values.as_ref().map(|s| {
                let mut s = s.to_vec();
                s.sort_unstable();
                s
            }),
            _ => None,
        }
    });
    assert!(carried == Some(affine_values_from_a_byte()), "the start is below 128 and the loop runs three times; found {:?} values", carried.map(|s| s.len()));
}

fn squared_loop_with_two_back_edges() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let one = b.constant(e, Ty::I32, 1);
    let zero = b.constant(e, Ty::I32, 0);
    let (head, h) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (left, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (right, r) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, head, vec![k.exec, one, zero]);
    let squared = b.int(head, IntOp::Mul, h[1], h[1]);
    let one = b.constant(head, Ty::I32, 1);
    let next_value = b.int(head, IntOp::Add, squared, one);
    let next = b.int(head, IntOp::Add, h[2], one);
    let low = b.int(head, IntOp::And, next, one);
    let odd = b.cmp(head, IntPred::Eq, low, one);
    b.cond_br(head, odd, (left, vec![h[0], next_value, next]), (right, vec![h[0], next_value, next]));
    for (block, p) in [(left, &l), (right, &r)] {
        let three = b.constant(block, Ty::I32, 3);
        let again = b.cmp(block, IntPred::Ult, p[2], three);
        b.cond_br(block, again, (head, vec![p[0], p[1], p[2]]), (exit, vec![p[0]]));
    }
    (b, h[1])
}

#[test]
fn loop_values_hold_every_value_a_square_step_carries_along_two_back_edges() {
    let (b, v) = squared_loop_with_two_back_edges();
    let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
        let form = a.values.value(v, 0, None).0.form;
        [1u32, 2, 5].iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the parameter runs 1, 2, 5: {:?}", missed);
}

#[test]
fn loop_values_are_exactly_those_a_square_step_carries_along_two_back_edges() {
    let (b, v) = squared_loop_with_two_back_edges();
    assert_eq!(carried_values(&b, v), Some(vec![1, 2, 5]));
}

fn sums_of_sums() -> (Build, Vec<(&'static str, ValueId, u32)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let far = b.constant(e, Ty::I64, 0x1_0000_0000);
    let above = b.int(e, IntOp::Add, buf, far);
    let twice = b.int(e, IntOp::Add, above, far);
    let high = b.core(e, Ty::I32, Op::UnpackHi(twice));
    let back = b.int(e, IntOp::Sub, twice, far);
    let back_high = b.core(e, Ty::I32, Op::UnpackHi(back));
    (b, vec![("high((buf + 2^32) + 2^32)", high, 2), ("high(((buf + 2^32) + 2^32) - 2^32)", back_high, 1)])
}

#[test]
fn high_halves_of_sums_of_sums_hold_their_values() {
    let (b, cases) = sums_of_sums();
    let missed: Vec<&str> = addresses(&b, &two_words(), |a| {
        cases
            .iter()
            .filter(|c| {
                let form = a.values.value(c.1, 0, None).0.form;
                !representable(a.unknowns(), &form, c.2)
            })
            .map(|c| c.0)
            .collect()
    });
    assert!(missed.is_empty(), "{:?}", missed);
}

#[test]
fn high_halves_of_sums_of_sums_give_their_values() {
    let (b, cases) = sums_of_sums();
    let loose: Vec<String> = addresses(&b, &two_words(), |a| {
        cases
            .iter()
            .filter_map(|c| {
                let form = a.values.value(c.1, 0, None).0.form;
                (form.as_constant() != Some(c.2)).then(|| format!("{}: {:?}", c.0, form))
            })
            .collect()
    });
    assert!(loose.is_empty(), "buf is 0x1000: {:?}", loose);
}

fn shared_word_by_item() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let four = b.constant(e, Ty::I32, 4);
    let own = b.int(e, IntOp::Mul, k.item, four);
    b.store(e, Space::Lds, MemSize::B32, own, k.item, k.exec);
    let back = b.load(e, Space::Lds, MemSize::B32, own, k.exec);
    (b, back)
}

#[test]
fn lds_words_read_back_by_work_item_hold_the_word_the_item_stored() {
    let (b, back) = shared_word_by_item();
    let held = addresses(&b, &two_words(), |a| {
        let form = a.values.value(back, 3, None).0.form;
        representable(a.unknowns(), &form, 3)
    });
    assert!(held, "item 3 reads back the 3 it stored");
}

#[test]
fn lds_words_read_back_by_work_item_give_the_word_the_item_stored() {
    let (b, back) = shared_word_by_item();
    let form = addresses(&b, &two_words(), |a| a.values.value(back, 3, None).0.form);
    assert_eq!(form, Form::constant(3), "word 3 of the LDS only ever holds 3, which only item 3 writes");
}

fn squared_loop_with_two_bounds() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let one = b.constant(e, Ty::I32, 1);
    let zero = b.constant(e, Ty::I32, 0);
    let (head, h) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (even, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (odd, r) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, head, vec![k.exec, one, zero]);
    let squared = b.int(head, IntOp::Mul, h[1], h[1]);
    let one = b.constant(head, Ty::I32, 1);
    let next_value = b.int(head, IntOp::Add, squared, one);
    let next = b.int(head, IntOp::Add, h[2], one);
    let low = b.int(head, IntOp::And, next, one);
    let is_odd = b.cmp(head, IntPred::Eq, low, one);
    b.cond_br(head, is_odd, (odd, vec![h[0], next_value, next]), (even, vec![h[0], next_value, next]));
    for (block, p, bound) in [(even, &l, 3u64), (odd, &r, 5)] {
        let limit = b.constant(block, Ty::I32, bound);
        let again = b.cmp(block, IntPred::Ult, p[2], limit);
        b.cond_br(block, again, (head, vec![p[0], p[1], p[2]]), (exit, vec![p[0]]));
    }
    (b, h[1])
}

#[test]
fn loop_values_hold_every_value_a_square_step_carries_along_two_back_edges_with_different_bounds() {
    let (b, v) = squared_loop_with_two_bounds();
    let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
        let form = a.values.value(v, 0, None).0.form;
        [1u32, 2, 5, 26].iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the odd back edge goes on below 5 and the even one below 3, so the loop runs four times and carries 1, 2, 5, 26: {:?}", missed);
}

fn chosen_after_offset_branch(entered: bool) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let three = b.constant(e, Ty::I32, 3);
    let x = b.int(e, IntOp::Add, u, three);
    let five = b.constant(e, Ty::I32, 5);
    let hit = b.cmp(e, IntPred::Eq, x, five);
    let two = b.constant(e, Ty::I32, 2);
    let doubled = b.int(e, IntOp::Mul, u, two);
    let one = b.constant(e, Ty::I32, 1);
    let w = b.int(e, IntOp::Add, doubled, one);
    let (then, t) = b.block(&[Ty::I1, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I32]);
    b.cond_br(e, hit, (then, vec![k.exec, w]), (other, vec![k.exec, w]));
    let (block, p) = if entered { (then, t) } else { (other, o) };
    let five = b.constant(block, Ty::I32, 5);
    let test = b.cmp(block, IntPred::Eq, p[1], five);
    let seven = b.constant(block, Ty::I32, 7);
    let nine = b.constant(block, Ty::I32, 9);
    let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
    (b, chosen)
}

#[test]
fn selects_on_a_word_derived_from_an_offset_branch_word_hold_the_arms_the_branch_leaves() {
    let missed: Vec<(bool, u32)> = [(true, 7u32), (false, 7), (false, 9)]
        .iter()
        .copied()
        .filter(|&(entered, truth)| {
            let (b, chosen) = chosen_after_offset_branch(entered);
            !addresses(&b, &two_words(), |a| {
                let form = a.values.value(chosen, 0, None).0.form;
                representable(a.unknowns(), &form, truth)
            })
        })
        .collect();
    assert!(missed.is_empty(), "u + 3 = 5 gives u = 2 and 2u + 1 = 5; otherwise u = 2^31 + 2 still gives 5: {:?}", missed);
}

#[test]
fn selects_on_a_word_derived_from_an_offset_branch_word_take_the_arm_the_branch_decides() {
    let (b, chosen) = chosen_after_offset_branch(true);
    let form = addresses(&b, &two_words(), |a| a.values.value(chosen, 0, None).0.form);
    assert_eq!(form, Form::constant(7), "the block runs only when u + 3 is 5, so 2u + 1 is 5");
}

fn chosen_in_a_loop_entered_on_zero_that_counts_up() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let zero = b.constant(e, Ty::I32, 0);
    let is_zero = b.cmp(e, IntPred::Eq, u, zero);
    let (body, p) = b.block(&[Ty::I1, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.cond_br(e, is_zero, (body, vec![k.exec, u]), (exit, vec![k.exec]));
    let zero = b.constant(body, Ty::I32, 0);
    let test = b.cmp(body, IntPred::Eq, p[1], zero);
    let seven = b.constant(body, Ty::I32, 7);
    let nine = b.constant(body, Ty::I32, 9);
    let chosen = b.core(body, Ty::I32, Op::Select(test, seven, nine));
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[1], one);
    let three = b.constant(body, Ty::I32, 3);
    let again = b.cmp(body, IntPred::Ult, next, three);
    b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
    (b, chosen)
}

#[test]
fn selects_in_a_loop_entered_on_zero_that_counts_up_hold_both_arms() {
    let (b, chosen) = chosen_in_a_loop_entered_on_zero_that_counts_up();
    let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
        let form = a.values.value(chosen, 0, None).0.form;
        [7u32, 9].iter().copied().filter(|&t| !representable(a.unknowns(), &form, t)).collect()
    });
    assert!(missed.is_empty(), "the first iteration sees 0 and gives 7, the next ones 1 and 2 and give 9: {:?}", missed);
}

fn branches_on_chained_orders() -> (Build, Vec<(&'static str, Box<dyn Fn(u32, u32, u32) -> bool>, Vec<(IntPred, ValueId)>)>) {
    use IntPred::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
    let w = loaded_word(&mut b, &k, e, 8, MemSize::B32);
    let fork = |b: &mut Build, at: BlockId, cond: ValueId, exec: ValueId| {
        let (yes, y) = b.block(&[Ty::I1]);
        let (no, n) = b.block(&[Ty::I1]);
        b.cond_br(at, cond, (yes, vec![exec]), (no, vec![exec]));
        ((yes, y[0]), (no, n[0]))
    };
    let c1 = b.cmp(e, Ult, u, v);
    let ((b1, e1), (x1, _)) = fork(&mut b, e, c1, k.exec);
    let c2 = b.cmp(b1, Ult, v, w);
    let ((b2, _), (x2, _)) = fork(&mut b, b1, c2, e1);
    let c1 = |u: u32, v: u32, _: u32| u < v;
    let c2 = |_: u32, v: u32, w: u32| v < w;
    let reach: Vec<(&'static str, BlockId, Box<dyn Fn(u32, u32, u32) -> bool>)> = vec![
        ("b1", b1, Box::new(move |u, v, w| c1(u, v, w))),
        ("x1", x1, Box::new(move |u, v, w| !c1(u, v, w))),
        ("b2", b2, Box::new(move |u, v, w| c1(u, v, w) && c2(u, v, w))),
        ("x2", x2, Box::new(move |u, v, w| c1(u, v, w) && !c2(u, v, w))),
    ];
    let mut blocks = Vec::new();
    for (name, block, reaches) in reach {
        let queries = PREDICATES.iter().map(|&pred| (pred, b.cmp(block, pred, u, w))).collect();
        blocks.push((name, reaches, queries));
    }
    (b, blocks)
}

fn chained_order_answers(exact: bool) -> Vec<String> {
    let (b, blocks) = branches_on_chained_orders();
    let points = [0u32, 1, 2, 3, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000, u32::MAX - 1, u32::MAX];
    let mut wrong = Vec::new();
    addresses(&b, &two_words(), |a| {
        for (name, reaches, queries) in &blocks {
            for &(pred, q) in queries {
                let mut seen = [false; 2];
                for &u in &points {
                    for &v in &points {
                        for &w in &points {
                            if reaches(u, v, w) {
                                seen[compare(pred, u, w) as usize] = true;
                            }
                        }
                    }
                }
                let truth = match seen {
                    [false, false] => continue,
                    [true, false] => Some(false),
                    [false, true] => Some(true),
                    _ => None,
                };
                let got = a.bit(q, 0, None).0;
                let ok = if exact { got == truth } else { got.is_none() || got == truth };
                if !ok {
                    wrong.push(format!("{}: u {:?} w: {:?}, truth {:?}", name, pred, got, truth));
                }
            }
        }
    });
    wrong
}

#[test]
fn comparisons_under_chained_orders_decide_exactly_what_the_orders_settle_together() {
    let wrong = chained_order_answers(true);
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

#[test]
fn comparisons_under_chained_orders_hold_every_value_the_orders_leave() {
    let wrong = chained_order_answers(false);
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

fn branches_on_an_order_over_bytes() -> (Build, BlockId, Vec<(IntPred, u32, ValueId)>) {
    use IntPred::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let u = loaded_word(&mut b, &k, e, 0, MemSize::U8);
    let v = loaded_word(&mut b, &k, e, 4, MemSize::U8);
    let ten = b.constant(e, Ty::I32, 10);
    let shifted = b.int(e, IntOp::Add, u, ten);
    let c = b.cmp(e, Ugt, v, shifted);
    let (yes, y) = b.block(&[Ty::I1]);
    let (no, n) = b.block(&[Ty::I1]);
    b.cond_br(e, c, (yes, vec![k.exec]), (no, vec![k.exec]));
    let _ = (y, n);
    let mut queries = Vec::new();
    for &bound in &[9u32, 10, 11, 12, 254, 255] {
        let k = b.constant(yes, Ty::I32, bound as u64);
        for pred in PREDICATES {
            queries.push((pred, bound, b.cmp(yes, pred, v, k)));
        }
    }
    (b, yes, queries)
}

fn byte_order_answers(exact: bool) -> Vec<String> {
    let (b, _, queries) = branches_on_an_order_over_bytes();
    let mut wrong = Vec::new();
    addresses(&b, &two_words(), |a| {
        for &(pred, bound, q) in &queries {
            let mut seen = [false; 2];
            for u in 0u32..256 {
                for v in 0u32..256 {
                    if v > u + 10 {
                        seen[compare(pred, v, bound) as usize] = true;
                    }
                }
            }
            let truth = match seen {
                [false, false] => continue,
                [true, false] => Some(false),
                [false, true] => Some(true),
                _ => None,
            };
            let got = a.bit(q, 0, None).0;
            let ok = if exact { got == truth } else { got.is_none() || got == truth };
            if !ok {
                wrong.push(format!("v {:?} {}: {:?}, truth {:?}", pred, bound, got, truth));
            }
        }
    });
    wrong
}

#[test]
fn comparisons_with_constants_under_an_order_over_bounded_words_decide_exactly_what_the_order_settles() {
    let wrong = byte_order_answers(true);
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

#[test]
fn comparisons_with_constants_under_an_order_over_bounded_words_hold_every_value_the_order_leaves() {
    let wrong = byte_order_answers(false);
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

fn read_after_a_restore() -> (Build, ValueId, ValueId, ValueId) {
    let (mut b, k, _) = Build::kernel_in(&[], 64);
    let e = BlockId(0);
    let near = k.buffer(&mut b, e, 0);
    let far = k.buffer(&mut b, e, 8);
    let flags = k.buffer(&mut b, e, 16);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, near, lane, 4);
    let at = byte_offset(&mut b, e, flags, lane, 4);
    let flag = b.load(e, Space::Global, MemSize::B32, at, k.exec);
    let zero = b.constant(e, Ty::I32, 0);
    let hit = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, hit, k.exec);
    let pointer = b.core(e, Ty::I64, Op::Select(c, own, far));
    let (region, r) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    b.br(e, region, vec![c, pointer, k.exec]);
    let low = b.wave(region, WaveOp::Ballot { high: false }, vec![r[0]]);
    let high = b.wave(region, WaveOp::Ballot { high: true }, vec![r[0]]);
    let lane = b.core(region, Ty::I32, Op::Env(Env::LaneId));
    let five = b.constant(region, Ty::I32, 5);
    let few = b.cmp(region, IntPred::Ult, lane, five);
    let inner = b.int(region, IntOp::And, few, r[0]);
    let yes = b.constant(region, Ty::I1, 1);
    let rest = b.int(region, IntOp::Xor, few, yes);
    let other = b.int(region, IntOp::And, rest, r[0]);
    let any = b.wave(region, WaveOp::Any, vec![inner]);
    let shape = [Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I1];
    let (then, t) = b.block(&shape);
    let (otherwise, o) = b.block(&shape);
    b.cond_br(region, any, (then, vec![inner, r[1], low, high, r[2]]), (otherwise, vec![other, r[1], low, high, r[2]]));
    let (join, j) = b.block(&shape);
    b.br(then, join, t.clone());
    b.br(otherwise, join, o.clone());
    let lo = b.wave(join, WaveOp::Ballot { high: false }, vec![j[0]]);
    let hi = b.wave(join, WaveOp::Ballot { high: true }, vec![j[0]]);
    let now = b.core(join, Ty::I64, Op::Pack64(lo, hi));
    let saved = b.core(join, Ty::I64, Op::Pack64(j[2], j[3]));
    let word = b.int(join, IntOp::Or, now, saved);
    let lane = b.core(join, Ty::I32, Op::Env(Env::LaneId));
    let wide = b.core(join, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
    let shifted = b.int(join, IntOp::LShr, word, wide);
    let bit = b.core(join, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let valid = b.core(join, Ty::I1, Op::Env(Env::ValidLane));
    let restored = b.int(join, IntOp::And, bit, valid);
    let (read, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    b.br(join, read, vec![restored, j[1], j[4]]);
    (b, x[1], x[0], x[2])
}

#[test]
fn a_pointer_read_after_a_restore_points_where_the_reading_lanes_set_it() {
    let (b, pointer, restored, everyone) = read_after_a_restore();
    let env = environment(64, &[(0, 1, 0x1000), (8, 2, 0x2000), (16, 3, 0x3000)]);
    let near = Regions::one(Some(Region::Allocation(1)));
    for refine in [false, true] {
        let (inside, outside) = addresses(&b, &env, |a| {
            (a.regions(pointer, 0, Some(restored), refine), a.regions(pointer, 0, Some(everyone), refine))
        });
        assert_eq!(inside, near, "refine {}: a lane in the restored mask was in the region when it set its pointer", refine);
        assert!(
            outside.list.contains(&Some(Region::Allocation(2))),
            "refine {}: a lane outside the region keeps the second buffer: {:?}",
            refine,
            outside
        );
    }
}

fn colliding_lds_word() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let four = b.constant(e, Ty::I32, 4);
    let own = b.int(e, IntOp::Mul, lane, four);
    let top = b.constant(e, Ty::I32, 128);
    let base = b.int(e, IntOp::Sub, top, own);
    let address = b.int(e, IntOp::Add, base, own);
    b.store(e, Space::Lds, MemSize::B32, address, lane, k.exec);
    let back = b.load(e, Space::Lds, MemSize::B32, address, k.exec);
    (b, back)
}

#[test]
fn lds_read_back_with_a_lane_varying_base_does_not_claim_the_lane_own_word() {
    let (b, back) = colliding_lds_word();
    let form = addresses(&b, &two_words(), |a| a.values.value(back, 3, None).0.form);
    assert_ne!(
        form,
        Form::constant(3),
        "every lane stores its lane id to LDS byte 128 ((128 - 4 lane) + 4 lane), so lane 3 reads back the winning lane's id, not necessarily 3"
    );
}

fn colliding_global_word() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let narrow_four = b.constant(e, Ty::I32, 4);
    let narrow_own = b.int(e, IntOp::Mul, lane, narrow_four);
    let top = b.constant(e, Ty::I32, 124);
    let rest = b.int(e, IntOp::Sub, top, narrow_own);
    let wide_rest = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, rest));
    let base = b.int(e, IntOp::Add, buf, wide_rest);
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
    let four = b.constant(e, Ty::I64, 4);
    let own = b.int(e, IntOp::Mul, wide, four);
    let address = b.int(e, IntOp::Add, base, own);
    b.store(e, Space::Global, MemSize::B32, address, lane, k.exec);
    let back = b.load(e, Space::Global, MemSize::B32, address, k.exec);
    (b, back)
}

#[test]
fn global_read_back_with_a_lane_varying_base_does_not_claim_the_lane_own_word() {
    let (b, back) = colliding_global_word();
    let form = addresses(&b, &environment(32, &[(0, 1, 0x1000)]), |a| a.values.value(back, 3, None).0.form);
    assert_ne!(
        form,
        Form::constant(3),
        "every lane stores its lane id to buf + 124 ((buf + zext(124 - 4 lane)) + 4 zext(lane)), so lane 3 reads back the winning lane's id, not necessarily 3"
    );
}

fn lds_word_per_wave_base() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let four = b.constant(e, Ty::I32, 4);
    let own = b.int(e, IntOp::Mul, lane, four);
    let high = b.constant(e, Ty::I32, 0xffff_ffe0);
    let first = b.int(e, IntOp::And, k.item, high);
    let base = b.int(e, IntOp::Add, first, first);
    let address = b.int(e, IntOp::Add, base, own);
    b.store(e, Space::Lds, MemSize::B32, address, lane, k.exec);
    let back = b.load(e, Space::Lds, MemSize::B32, address, k.exec);
    (b, back)
}

#[test]
fn lds_read_back_with_a_base_that_differs_between_waves_does_not_claim_the_lane_own_word() {
    let (b, back) = lds_word_per_wave_base();
    let env = environment(64, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let form = addresses(&b, &env, |a| a.values.value(back, 16, None).0.form);
    assert_ne!(
        form,
        Form::constant(16),
        "wave 0 lane 16 stores 16 at LDS byte 2 * 0 + 64, wave 1 lane 0 stores 0 at 2 * 32 + 0 = 64 too, so the word read back is not fixed"
    );
}

fn wrapping_lds_word() -> (Build, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let mask = b.constant(e, Ty::I32, 0x3333_3333);
    let x = b.int(e, IntOp::And, w, mask);
    let five = b.constant(e, Ty::I32, 5);
    let scaled = b.int(e, IntOp::Mul, x, five);
    let four = b.constant(e, Ty::I32, 4);
    let address = b.int(e, IntOp::Add, scaled, four);
    b.store(e, Space::Lds, MemSize::B32, address, x, k.exec);
    let back = b.load(e, Space::Lds, MemSize::B32, address, k.exec);
    let difference = b.int(e, IntOp::Sub, back, x);
    (b, back, difference)
}

#[test]
fn lds_read_back_does_not_claim_words_whose_addresses_overlap_after_wrapping() {
    let (b, _, difference) = wrapping_lds_word();
    let form = addresses(&b, &two_words(), |a| a.values.value(difference, 0, None).0.form);
    assert_ne!(
        form.as_constant(),
        Some(0),
        "x = w & 0x33333333 is 0 in one lane and 0x33333333 in another: 4 + 5 * 0x33333333 wraps to 3, so the two lanes' words overlap and neither need read back its own x"
    );
}

fn power_minus_both() -> (Build, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let four = b.constant(e, Ty::I32, 4);
    let at = b.constant(e, Ty::I64, 4);
    let next = b.int(e, IntOp::Add, own, at);
    let x = b.load(e, Space::Global, MemSize::B32, next, k.exec);
    let both = b.int(e, IntOp::And, w, x);
    let thirty_one = b.constant(e, Ty::I32, 31);
    let s = b.int(e, IntOp::And, w, thirty_one);
    let one = b.constant(e, Ty::I32, 1);
    let power = b.int(e, IntOp::Shl, one, s);
    let scaled = b.int(e, IntOp::Mul, power, four);
    let v = b.int(e, IntOp::Sub, scaled, both);
    let three = b.constant(e, Ty::I32, 3);
    let small = b.cmp(e, IntPred::Ult, v, three);
    (b, v, small)
}

#[test]
fn bounds_of_a_form_whose_coefficient_products_pass_two_to_the_sixty_four_hold_its_values() {
    let (b, v, _) = power_minus_both();
    let (form, bounds) = addresses(&b, &two_words(), |a| {
        let form = a.values.value(v, 0, None).0.form;
        let bounds = a.bounds(&form);
        (form, bounds)
    });
    if let Some((low, high)) = bounds {
        assert!(low <= 4 && 4 <= high, "4 * (1 << (w & 31)) - (w & x) is 4 when w = 0, yet bounds of {:?} are {:?}", form, (low, high));
    }
}

#[test]
fn comparisons_of_a_form_whose_coefficient_products_pass_two_to_the_sixty_four_decide_nothing_false() {
    let (b, _, small) = power_minus_both();
    let found = addresses(&b, &two_words(), |a| a.bit(small, 0, None).0);
    assert_eq!(found, None, "4 * (1 << (w & 31)) - (w & x) < 3 is false for w = 0 (4) and true for w = x = 0xffffffff (4 * 2^31 - 0xffffffff = 1 mod 2^32)");
}

#[test]
fn a_loop_whose_header_is_the_entry_block_is_analysed_in_finite_time() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(0), Ty::I32)]);
        let e = BlockId(0);
        let one = b.constant(e, Ty::I32, 1);
        let next = b.int(e, IntOp::Add, p[1], one);
        b.br(e, e, vec![p[0], next]);
        let reached = addresses(&b, &two_words(), |a| a.reaches_block(e));
        let _ = tx.send(reached);
    });
    let outcome = rx.recv_timeout(std::time::Duration::from_secs(20));
    assert!(!matches!(outcome, Err(std::sync::mpsc::RecvTimeoutError::Timeout)), "the analysis did not finish within 20 s");
}

fn halving_wide_loop() -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let zero = b.constant(e, Ty::I32, 0);
    let one = b.constant(e, Ty::I32, 1);
    let start = b.core(e, Ty::I64, Op::Pack64(zero, one));
    let (body, p) = b.block(&[Ty::I1, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, start]);
    let shift = b.constant(body, Ty::I64, 1);
    let next = b.int(body, IntOp::LShr, p[1], shift);
    let none = b.constant(body, Ty::I64, 0);
    let again = b.cmp(body, IntPred::Ne, next, none);
    b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
    (b, p[1])
}

#[test]
fn low_words_of_a_wide_value_halved_each_iteration_hold_every_value_it_takes() {
    let (b, v) = halving_wide_loop();
    let env = environment(32, &[(0, 1, 0x1000)]);
    let (form, held) = addresses(&b, &env, |a| {
        let form = a.values.value(v, 0, None).0.form;
        let held = representable(a.unknowns(), &form, 0x8000_0000);
        let ranges: Vec<_> = form.terms.iter().map(|&(u, _)| a.unknowns()[u as usize].range).collect();
        (format!("{:?} with ranges {:?}", form, ranges), held)
    });
    assert!(held, "v takes 2^32 then 2^31 = 0x80000000, whose low word is 0x80000000, but its low word form is {:?}", form);
}


static ORACLE_CHECKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ORACLE_TIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn oracle_holds(unknowns: &[UnknownInfo], form: &Form, truth: u32) -> bool {
    let mut enumerated: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut free_shift = 32u32;
    for &(u, c) in &form.terms {
        let info = &unknowns[u as usize];
        let set: Option<Vec<u32>> = match (&info.values, info.range) {
            (Some(set), _) if set.len() <= 4096 => Some(set.to_vec()),
            (_, Some((lo, hi))) if hi - lo < 4096 => Some((lo..=hi).collect()),
            _ => None,
        };
        match set {
            Some(s) => enumerated.push((c, s)),
            None => free_shift = free_shift.min(c.trailing_zeros()),
        }
    }
    let mask = if free_shift >= 32 { u32::MAX } else { (1u32 << free_shift) - 1 };
    ORACLE_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if free_shift >= 32 {
        ORACLE_TIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let mut reachable: std::collections::HashSet<u32> = std::collections::HashSet::new();
    reachable.insert(form.constant & mask);
    for (c, set) in &enumerated {
        let mut next = std::collections::HashSet::new();
        for &x in &reachable {
            for &v in set {
                next.insert(x.wrapping_add(c.wrapping_mul(v)) & mask);
            }
            if next.len() > 1 << 16 {
                return true;
            }
        }
        reachable = next;
    }
    reachable.contains(&(truth & mask))
}

#[derive(Clone, Copy, Debug)]
enum Update {
    Add(u32),
    Affine(u32, u32),
    Xor(u32),
    And(u32),
    Or(u32),
    AddOther,
    Other,
    Choose(IntPred, u32, u32, u32),
    ShlOrCounter,
    Shr(u32),
    AddLane,
    AddCounter,
    SubFrom(u32),
    MulOther,
}

fn random_update(r: &mut Random) -> Update {
    use Update::*;
    match r.below(14) {
        0 => Add(interesting(r)),
        1 => Affine(interesting(r), interesting(r)),
        2 => Xor(interesting(r)),
        3 => And(interesting(r)),
        4 => Or(interesting(r)),
        5 => AddOther,
        6 => Other,
        7 => Choose(PREDICATES[r.below(10) as usize], interesting(r), interesting(r), interesting(r)),
        8 => ShlOrCounter,
        9 => Shr(r.below(33) as u32),
        10 => AddLane,
        11 => AddCounter,
        12 => SubFrom(interesting(r)),
        _ => MulOther,
    }
}

fn build_update(b: &mut Build, h: BlockId, u: Update, own: ValueId, other: ValueId, i: ValueId, lane: ValueId) -> ValueId {
    use Update::*;
    let c = |b: &mut Build, k: u32| b.constant(h, Ty::I32, k as u64);
    match u {
        Add(k) => {
            let k = c(b, k);
            b.int(h, IntOp::Add, own, k)
        }
        Affine(m, k) => {
            let (m, k) = (c(b, m), c(b, k));
            let x = b.int(h, IntOp::Mul, own, m);
            b.int(h, IntOp::Add, x, k)
        }
        Xor(k) => {
            let k = c(b, k);
            b.int(h, IntOp::Xor, own, k)
        }
        And(k) => {
            let k = c(b, k);
            b.int(h, IntOp::And, own, k)
        }
        Or(k) => {
            let k = c(b, k);
            b.int(h, IntOp::Or, own, k)
        }
        AddOther => b.int(h, IntOp::Add, own, other),
        Other => {
            let zero = c(b, 0);
            b.int(h, IntOp::Add, other, zero)
        }
        Choose(p, k, x, y) => {
            let (k, x, y) = (c(b, k), c(b, x), c(b, y));
            let t = b.cmp(h, p, own, k);
            let ox = b.int(h, IntOp::Add, own, x);
            let oy = b.int(h, IntOp::Add, own, y);
            b.core(h, Ty::I32, Op::Select(t, ox, oy))
        }
        ShlOrCounter => {
            let (one, k1) = (c(b, 1), c(b, 1));
            let s = b.int(h, IntOp::Shl, own, one);
            let low = b.int(h, IntOp::And, i, k1);
            b.int(h, IntOp::Or, s, low)
        }
        Shr(k) => {
            let k = c(b, k);
            b.int(h, IntOp::LShr, own, k)
        }
        AddLane => b.int(h, IntOp::Add, own, lane),
        AddCounter => b.int(h, IntOp::Add, own, i),
        SubFrom(k) => {
            let k = c(b, k);
            b.int(h, IntOp::Sub, k, own)
        }
        MulOther => b.int(h, IntOp::Mul, own, other),
    }
}

fn run_update(u: Update, own: u32, other: u32, i: u32, lane: u32) -> u32 {
    use Update::*;
    match u {
        Add(k) => own.wrapping_add(k),
        Affine(m, k) => own.wrapping_mul(m).wrapping_add(k),
        Xor(k) => own ^ k,
        And(k) => own & k,
        Or(k) => own | k,
        AddOther => own.wrapping_add(other),
        Other => other,
        Choose(p, k, x, y) => {
            if compare(p, own, k) {
                own.wrapping_add(x)
            } else {
                own.wrapping_add(y)
            }
        }
        ShlOrCounter => (own << 1) | (i & 1),
        Shr(k) => own >> (k & 31),
        AddLane => own.wrapping_add(lane),
        AddCounter => own.wrapping_add(i),
        SubFrom(k) => k.wrapping_sub(own),
        MulOther => own.wrapping_mul(other),
    }
}

struct Loop {
    b: Build,
    header: BlockId,
    params: [ValueId; 3],
    nexts: [ValueId; 3],
    exits: [ValueId; 3],
    again: ValueId,
    visits: Vec<Vec<[u32; 3]>>,
    steps: Vec<Vec<([u32; 3], bool)>>,
    finals: Vec<Option<[u32; 3]>>,
    name: String,
}

const ORACLE_CAP: usize = 300;

fn oracle_loop(r: &mut Random) -> Loop {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let (ma, ca) = (if r.below(2) == 0 { 0 } else { interesting(r) }, interesting(r));
    let (mb, cb) = (if r.below(2) == 0 { 0 } else { interesting(r) }, interesting(r));
    let i0 = if r.below(2) == 0 { r.below(8) as u32 } else { interesting(r) };
    let si = [1u32, 2, 3, u32::MAX, u32::MAX - 1, 0x8000_0000, 4, interesting(r)][r.below(8) as usize];
    let pred = PREDICATES[r.below(10) as usize];
    let limit = if r.below(2) == 0 { i0.wrapping_add(si.wrapping_mul(r.below(40) as u32)) } else { interesting(r) };
    let (ua, ub) = (random_update(r), random_update(r));
    let lane_free = |u: Update| !matches!(u, Update::AddLane);
    let uniform = ma == 0 && mb == 0 && lane_free(ua) && lane_free(ub);
    let on_a = uniform && r.below(3) == 0;
    let init = |b: &mut Build, m: u32, c: u32| {
        let km = b.constant(e, Ty::I32, m as u64);
        let kc = b.constant(e, Ty::I32, c as u64);
        let x = b.int(e, IntOp::Mul, lane, km);
        b.int(e, IntOp::Add, x, kc)
    };
    let a0 = init(&mut b, ma, ca);
    let b0 = init(&mut b, mb, cb);
    let start = b.constant(e, Ty::I32, i0 as u64);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
    let (x, xp) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
    b.br(e, h, vec![k.exec, start, a0, b0]);
    let hl = b.core(h, Ty::I32, Op::Env(Env::LaneId));
    let ks = b.constant(h, Ty::I32, si as u64);
    let ni = b.int(h, IntOp::Add, p[1], ks);
    let na = build_update(&mut b, h, ua, p[2], p[3], p[1], hl);
    let nb = build_update(&mut b, h, ub, p[3], p[2], p[1], hl);
    let kl = b.constant(h, Ty::I32, limit as u64);
    let again = b.cmp(h, pred, if on_a { na } else { ni }, kl);
    b.cond_br(h, again, (h, vec![p[0], ni, na, nb]), (x, vec![p[0], ni, na, nb]));
    let mut visits = Vec::new();
    let mut steps = Vec::new();
    let mut finals = Vec::new();
    for l in 0..32u32 {
        let (mut i, mut a, mut c) = (i0, l.wrapping_mul(ma).wrapping_add(ca), l.wrapping_mul(mb).wrapping_add(cb));
        let mut seen = Vec::new();
        let mut stepped = Vec::new();
        let mut last = None;
        for _ in 0..ORACLE_CAP {
            seen.push([i, a, c]);
            let ni = i.wrapping_add(si);
            let na = run_update(ua, a, c, i, l);
            let nb = run_update(ub, c, a, i, l);
            let go = compare(pred, if on_a { na } else { ni }, limit);
            stepped.push(([ni, na, nb], go));
            if !go {
                last = Some([ni, na, nb]);
                break;
            }
            (i, a, c) = (ni, na, nb);
        }
        visits.push(seen);
        steps.push(stepped);
        finals.push(last);
    }
    let name = format!(
        "i0 {:#x} step {:#x} {:?} {:#x} on_a {} a0 = lane*{:#x}+{:#x} b0 = lane*{:#x}+{:#x} a' = {:?} b' = {:?}",
        i0, si, pred, limit, on_a, ma, ca, mb, cb, ua, ub
    );
    Loop {
        b,
        header: h,
        params: [p[1], p[2], p[3]],
        nexts: [ni, na, nb],
        exits: [xp[1], xp[2], xp[3]],
        again,
        visits,
        steps,
        finals,
        name,
    }
}

fn loop_failures(seed: u64, count: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    let only: Option<usize> = std::env::var("ORACLE_CASE").ok().and_then(|s| s.parse().ok());
    for case in 0..count {
        let c = oracle_loop(&mut r);
        if only.is_some_and(|o| o != case) {
            continue;
        }
        if std::env::var("ORACLE_TRACE").is_ok() {
            eprintln!("case {} [{}]", case, c.name);
        }
        let env = environment(32, &[]);
        addresses(&c.b, &env, |a| {
            for &lane in &[0usize, 5, 31] {
                let names = ["i", "a", "b"];
                for k in 0..3 {
                    let form = a.values.value(c.params[k], lane, None).0.form;
                    if let Some(t) = c.visits[lane].iter().map(|v| v[k]).find(|&t| !oracle_holds(a.unknowns(), &form, t)) {
                        wrong.push(format!("case {} [{}] lane {}: param {} = {:?} misses {:#x}", case, c.name, lane, names[k], form, t));
                    }
                    let form = a.values.value(c.nexts[k], lane, None).0.form;
                    if let Some(t) = c.steps[lane].iter().map(|v| v.0[k]).find(|&t| !oracle_holds(a.unknowns(), &form, t)) {
                        wrong.push(format!("case {} [{}] lane {}: next {} = {:?} misses {:#x}", case, c.name, lane, names[k], form, t));
                    }
                    if let Some(f) = c.finals[lane] {
                        let form = a.values.value(c.exits[k], lane, None).0.form;
                        if !oracle_holds(a.unknowns(), &form, f[k]) {
                            wrong.push(format!("case {} [{}] lane {}: exit {} = {:?} misses {:#x}", case, c.name, lane, names[k], form, f[k]));
                        }
                    }
                }
                if let Some(bit) = a.bit(c.again, lane, None).0 {
                    if c.steps[lane].iter().any(|s| s.1 != bit) {
                        wrong.push(format!("case {} [{}] lane {}: the back-edge condition said always {}", case, c.name, lane, bit));
                    }
                }
            }
            if let Some(u) = a.trip(c.header) {
                if let Some((_, last)) = a.unknowns()[u as usize].range {
                    let visits = c.visits[0].len();
                    if visits as u64 > last as u64 + 1 {
                        wrong.push(format!("case {} [{}]: trips said at most {}, the loop visits its header {} times", case, c.name, last, visits));
                    }
                }
            }
        });
    }
    wrong
}

#[test]
fn loop_values_hold_every_value_a_random_loop_gives() {
    let seed = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(101);
    let count = std::env::var("ORACLE_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let wrong = loop_failures(seed, count);
    if std::env::var("ORACLE_STATS").is_ok() {
        eprintln!(
            "checks {} tight {}",
            ORACLE_CHECKS.load(std::sync::atomic::Ordering::Relaxed),
            ORACLE_TIGHT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    assert!(wrong.is_empty(), "{} wrong, first: {:#?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

fn offset_by_second_vgpr() -> (Build, Kernel, ValueId, ValueId) {
    offset_by_input(ParameterSource::Vgpr(1))
}

fn offset_by_input(source: ParameterSource) -> (Build, Kernel, ValueId, ValueId) {
    let (mut b, k, extra) = Build::kernel_with(&[(source, Ty::I32)]);
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, extra[0]));
    let address = b.int(e, IntOp::Add, buf, wide);
    (b, k, buf, address)
}

#[test]
fn an_address_offset_by_a_second_vgpr_input_points_into_the_buffer() {
    let (b, _, _, address) = offset_by_second_vgpr();
    let env = environment(32, &[(0, 1, 0x1000)]);
    let (coarse, refined) = addresses(&b, &env, |a| (a.regions(address, 3, None, false), a.regions(address, 3, None, true)));
    for set in [coarse, refined] {
        assert!(
            set.reaches(Some(Region::Allocation(1)), |x, y| x == y),
            "buf + v1 points into allocation 1, but its regions are {:?}",
            set
        );
    }
}

fn stores_through_input_meet(source: ParameterSource) -> bool {
    let (mut b, k, buf, through) = offset_by_input(source);
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let zero = b.constant(e, Ty::I32, 0);
    b.store(e, Space::Global, MemSize::B32, through, zero, k.exec);
    b.store(e, Space::Global, MemSize::B32, own, zero, k.exec);
    let h = super::super::hazard::Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
    !h.conflicts().is_empty()
}

#[test]
fn hazards_report_stores_through_an_unknown_scalar_input_that_meet_another_lane() {
    assert!(stores_through_input_meet(ParameterSource::Sgpr(20)), "control: an unknown scalar offset may be 0");
}

#[test]
fn hazards_report_stores_through_a_second_vgpr_input_that_meet_another_lane() {
    assert!(
        stores_through_input_meet(ParameterSource::Vgpr(1)),
        "lanes 1..31 store to buf[0] through v1 while lane 0 stores to buf[0] through its lane id"
    );
}

fn nested_affine_loops(outer: u64, inner: u64) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let one = b.constant(e, Ty::I32, 1);
    let zero = b.constant(e, Ty::I32, 0);
    let (h1, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (h2, q) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32, Ty::I32]);
    let (latch, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, h1, vec![k.exec, one, zero]);
    let z = b.constant(h1, Ty::I32, 0);
    b.br(h1, h2, vec![p[0], p[1], p[2], p[1], z]);
    let five = b.constant(h2, Ty::I32, 5);
    let seven = b.constant(h2, Ty::I32, 7);
    let scaled = b.int(h2, IntOp::Mul, q[3], five);
    let next_j = b.int(h2, IntOp::Add, scaled, seven);
    let one2 = b.constant(h2, Ty::I32, 1);
    let next_m = b.int(h2, IntOp::Add, q[4], one2);
    let limit2 = b.constant(h2, Ty::I32, inner);
    let again2 = b.cmp(h2, IntPred::Ult, next_m, limit2);
    b.cond_br(h2, again2, (h2, vec![q[0], q[1], q[2], next_j, next_m]), (latch, vec![q[0], q[1], q[2]]));
    let three = b.constant(latch, Ty::I32, 3);
    let one3 = b.constant(latch, Ty::I32, 1);
    let tripled = b.int(latch, IntOp::Mul, l[1], three);
    let next_x = b.int(latch, IntOp::Add, tripled, one3);
    let next_n = b.int(latch, IntOp::Add, l[2], one3);
    let limit1 = b.constant(latch, Ty::I32, outer);
    let again1 = b.cmp(latch, IntPred::Ult, next_n, limit1);
    b.cond_br(latch, again1, (h1, vec![l[0], next_x, next_n]), (exit, vec![l[0]]));
    (b, q[3])
}

#[test]
fn loop_value_sets_stay_within_the_sequence_limit() {
    let (b, j) = nested_affine_loops(300, 300);
    let largest = addresses(&b, &environment(32, &[]), |a| {
        let form = a.values.value(j, 0, None).0.form;
        form.terms.iter().filter_map(|&(u, _)| a.unknowns()[u as usize].values.as_ref().map(|s| s.len())).max()
    });
    assert!(largest.is_none_or(|n| n <= 1 << 16), "a sequence unknown holds {:?} words", largest);
}

#[derive(Clone, Copy, Debug)]
enum Pointer {
    Same,
    Other,
    Offset,
    ByCounter,
    ByLane,
    First,
    Second,
    Reloaded,
    Integer,
}

fn random_pointer(r: &mut Random) -> Pointer {
    use Pointer::*;
    [Same, Other, Offset, ByCounter, ByLane, First, Second, Reloaded, Integer][r.below(9) as usize]
}

struct PointerLoop {
    b: Build,
    watched: Vec<(&'static str, ValueId, Vec<Vec<Option<u64>>>)>,
    name: String,
}

fn pointer_loop(r: &mut Random) -> PointerLoop {
    use Pointer::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let slot = b.constant(e, Ty::I32, 32);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let entry_spill = r.below(2) == 0;
    if entry_spill {
        b.store(e, Space::Scratch, MemSize::B64, slot, first, k.exec);
    }
    let zero = b.constant(e, Ty::I32, 0);
    let (fp, fq) = (r.below(2) == 0, r.below(2) == 0);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
    let (x, _) = b.block(&[Ty::I1]);
    b.br(e, h, vec![k.exec, zero, if fp { first } else { second }, if fq { first } else { second }]);
    let (up, uq) = (random_pointer(r), random_pointer(r));
    let trips = 1 + r.below(6) as u32;
    let respill = r.below(2) == 0;
    let reloaded = b.load(h, Space::Scratch, MemSize::B64, slot, p[0]);
    let one = b.constant(h, Ty::I32, 1);
    let parity = b.int(h, IntOp::And, p[1], one);
    let even = b.cmp(h, IntPred::Eq, parity, zero);
    let sixteen = b.constant(h, Ty::I32, 16);
    let low = b.cmp(h, IntPred::Ult, lane, sixteen);
    let eight = b.constant(h, Ty::I64, 8);
    let build = |b: &mut Build, u: Pointer, own: ValueId, other: ValueId| match u {
        Same => own,
        Other => other,
        Offset => b.int(h, IntOp::Add, own, eight),
        ByCounter => b.core(h, Ty::I64, Op::Select(even, own, other)),
        ByLane => b.core(h, Ty::I64, Op::Select(low, own, other)),
        First => first,
        Second => second,
        Reloaded => reloaded,
        Integer => {
            let w = b.core(h, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
            b.int(h, IntOp::Add, w, eight)
        }
    };
    let np = build(&mut b, up, p[2], p[3]);
    let nq = build(&mut b, uq, p[3], p[2]);
    if respill {
        b.store(h, Space::Scratch, MemSize::B64, slot, nq, p[0]);
    }
    let ni = b.int(h, IntOp::Add, p[1], one);
    let limit = b.constant(h, Ty::I32, trips as u64);
    let again = b.cmp(h, IntPred::Ult, ni, limit);
    b.cond_br(h, again, (h, vec![p[0], ni, np, nq]), (x, vec![p[0]]));
    let tag = |f: bool| Some(if f { 1u64 } else { 2 });
    let mut tp: Vec<Vec<Option<u64>>> = vec![Vec::new(); 32];
    let mut tq = tp.clone();
    let mut tr = tp.clone();
    let mut tnp = tp.clone();
    let mut tnq = tp.clone();
    for l in 0..32usize {
        let (mut cp, mut cq) = (tag(fp), tag(fq));
        let mut held = if entry_spill { Some(1u64) } else { None };
        for t in 0..trips {
            let rl = held;
            let step = |u: Pointer, own: Option<u64>, other: Option<u64>| match u {
                Same | Offset => own,
                Other => other,
                ByCounter => {
                    if t & 1 == 0 {
                        own
                    } else {
                        other
                    }
                }
                ByLane => {
                    if l < 16 {
                        own
                    } else {
                        other
                    }
                }
                First => Some(1),
                Second => Some(2),
                Reloaded => rl,
                Integer => None,
            };
            let (np, nq) = (step(up, cp, cq), step(uq, cq, cp));
            tp[l].push(cp);
            tq[l].push(cq);
            tr[l].push(rl);
            tnp[l].push(np);
            tnq[l].push(nq);
            if respill {
                held = nq;
            }
            (cp, cq) = (np, nq);
        }
    }
    let name = format!("p0 {} q0 {} spill {} respill {} p' {:?} q' {:?} trips {}", tag(fp).unwrap(), tag(fq).unwrap(), entry_spill, respill, up, uq, trips);
    PointerLoop {
        b,
        watched: vec![("p", p[2], tp), ("q", p[3], tq), ("reloaded", reloaded, tr), ("p'", np, tnp), ("q'", nq, tnq)],
        name,
    }
}

fn modes() -> Vec<bool> {
    match std::env::var("ORACLE_MODES").as_deref() {
        Ok("refined") => vec![true],
        Ok("coarse") => vec![false],
        _ => vec![false, true],
    }
}

fn pointer_failures(seed: u64, count: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for case in 0..count {
        let c = pointer_loop(&mut r);
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        addresses(&c.b, &env, |a| {
            loop {
                for (_, v, _) in &c.watched {
                    for l in 0..32 {
                        for refine in modes() {
                            a.regions(*v, l, None, refine);
                        }
                    }
                }
                if !a.settle_loops() {
                    break;
                }
            }
            for (name, v, truth) in &c.watched {
                for l in [0usize, 20] {
                    for refine in modes() {
                        let set = a.regions(*v, l, None, refine);
                        for t in &truth[l] {
                            let hit = match t {
                                Some(id) => set.reaches(Some(Region::Allocation(*id)), |x, y| x == y),
                                None => set.lost(),
                            };
                            if !hit {
                                wrong.push(format!("case {} [{}] {} lane {} refine {}: {:?} misses {:?}", case, c.name, name, l, refine, set, t));
                                break;
                            }
                        }
                    }
                    let value = a.values.value(*v, l, None).0;
                    if let Some(region) = value.region {
                        if truth[l].iter().any(|t| t.map(Region::Allocation) != Some(region)) {
                            wrong.push(format!("case {} [{}] {} lane {}: value says {:?}, truth {:?}", case, c.name, name, l, region, truth[l]));
                        }
                    }
                }
            }
        });
    }
    wrong
}

#[test]
fn regions_hold_every_allocation_a_random_pointer_loop_carries() {
    let seed = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let count = std::env::var("ORACLE_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let wrong = pointer_failures(seed, count);
    assert!(wrong.is_empty(), "{} wrong, first: {:#?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

fn spilled_choice() -> (Build, Kernel, ValueId, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let u = loaded_word(&mut b, &k, e, 16, MemSize::B32);
    let zero = b.constant(e, Ty::I32, 0);
    let c = b.cmp(e, IntPred::Ne, u, zero);
    let either = b.core(e, Ty::I64, Op::Select(c, first, second));
    let slot = b.constant(e, Ty::I32, 16);
    b.store(e, Space::Scratch, MemSize::B64, slot, either, k.exec);
    let reloaded = b.load(e, Space::Scratch, MemSize::B64, slot, k.exec);
    (b, k, either, reloaded, second)
}

#[test]
fn a_reloaded_spill_of_a_choice_between_two_buffers_points_into_both() {
    let (b, _, either, reloaded, _) = spilled_choice();
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000), (16, 3, 0x3000)]);
    let (before, after) = addresses(&b, &env, |a| {
        let mut sets = Vec::new();
        for v in [either, reloaded] {
            loop {
                for refine in [false, true] {
                    a.regions(v, 0, None, refine);
                }
                if !a.settle_loops() {
                    break;
                }
            }
            sets.push(a.regions(v, 0, None, true));
        }
        (sets[0].clone(), sets[1].clone())
    });
    for id in [1u64, 2] {
        assert!(before.reaches(Some(Region::Allocation(id)), |x, y| x == y), "the choice: {:?}", before);
        assert!(
            after.reaches(Some(Region::Allocation(id)), |x, y| x == y),
            "the spill reloads the choice between allocations 1 and 2, but its regions are {:?} (before the spill {:?})",
            after,
            before
        );
    }
}

fn stores_through_choice_meet(spilled: bool) -> bool {
    let (mut b, k, either, reloaded, _) = spilled_choice();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, first, lane, 4);
    let zero = b.constant(e, Ty::I32, 0);
    b.store(e, Space::Global, MemSize::B32, if spilled { reloaded } else { either }, zero, k.exec);
    b.store(e, Space::Global, MemSize::B32, own, zero, k.exec);
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000), (16, 3, 0x3000)]);
    let h = super::super::hazard::Hazards::find(&b.program(), &env);
    !h.conflicts().is_empty()
}

#[test]
fn hazards_report_stores_through_a_choice_between_two_buffers() {
    assert!(stores_through_choice_meet(false), "control: the choice may be first");
}

#[test]
fn hazards_report_stores_through_a_reloaded_spill_of_a_choice_between_two_buffers() {
    assert!(stores_through_choice_meet(true), "the reloaded choice may be first, so lanes 1..31 meet lane 0 at first[0]");
}

struct Shaped {
    b: Build,
    header: BlockId,
    watched: Vec<(&'static str, ValueId, Vec<Vec<u32>>)>,
    guard: Option<(ValueId, Vec<Vec<bool>>)>,
    name: String,
}

fn shaped_loop(r: &mut Random) -> Shaped {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let (ma, ca) = (if r.below(2) == 0 { 0 } else { interesting(r) }, interesting(r));
    let (mb, cb) = (if r.below(2) == 0 { 0 } else { interesting(r) }, interesting(r));
    let s0 = interesting(r);
    let i0 = r.below(8) as u32;
    let si = [1u32, 2, 3, u32::MAX, 4][r.below(5) as usize];
    let pred = PREDICATES[r.below(10) as usize];
    let limit = if r.below(3) != 0 { i0.wrapping_add(si.wrapping_mul(r.below(30) as u32)) } else { interesting(r) };
    let (ua, ub, us) = (random_update(r), random_update(r), random_update(r));
    let branch_kind = r.below(3);
    let branch_k = r.below(6) as u32;
    let split = r.below(2) == 0;
    let store_in_t = r.below(2) == 0;
    let init = |b: &mut Build, m: u32, c: u32| {
        let km = b.constant(e, Ty::I32, m as u64);
        let kc = b.constant(e, Ty::I32, c as u64);
        let x = b.int(e, IntOp::Mul, lane, km);
        b.int(e, IntOp::Add, x, kc)
    };
    let a0 = init(&mut b, ma, ca);
    let b0 = init(&mut b, mb, cb);
    let slot = b.constant(e, Ty::I32, 64);
    let ks0 = b.constant(e, Ty::I32, s0 as u64);
    b.store(e, Space::Scratch, MemSize::B32, slot, ks0, k.exec);
    let start = b.constant(e, Ty::I32, i0 as u64);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
    let (t, tp) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32, Ty::I32]);
    let (f, fp) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32, Ty::I32]);
    let (j, jp) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
    let (x, xp) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
    b.br(e, h, vec![k.exec, start, a0, b0]);
    let hslot = b.constant(h, Ty::I32, 64);
    let s = b.load(h, Space::Scratch, MemSize::B32, hslot, p[0]);
    let kb = b.constant(h, Ty::I32, branch_k as u64);
    let cond_in = match branch_kind {
        0 => {
            let one = b.constant(h, Ty::I32, 1);
            let parity = b.int(h, IntOp::And, p[1], one);
            let zero = b.constant(h, Ty::I32, 0);
            b.cmp(h, IntPred::Eq, parity, zero)
        }
        1 => b.cmp(h, IntPred::Ult, p[1], kb),
        _ => b.cmp(h, IntPred::Ne, p[1], kb),
    };
    let taken_in = |i: u32| match branch_kind {
        0 => i & 1 == 0,
        1 => i < branch_k,
        _ => i != branch_k,
    };
    b.cond_br(h, cond_in, (t, vec![p[0], p[1], p[2], p[3], s]), (f, vec![p[0], p[1], p[2], p[3], s]));
    let tl = b.core(t, Ty::I32, Op::Env(Env::LaneId));
    let na = build_update(&mut b, t, ua, tp[2], tp[3], tp[1], tl);
    let ns = build_update(&mut b, t, us, tp[4], tp[2], tp[1], tl);
    if store_in_t {
        let tslot = b.constant(t, Ty::I32, 64);
        b.store(t, Space::Scratch, MemSize::B32, tslot, ns, tp[0]);
    }
    let fl = b.core(f, Ty::I32, Op::Env(Env::LaneId));
    let nb = build_update(&mut b, f, ub, fp[3], fp[2], fp[1], fl);
    let (ka, kf) = (b.constant(t, Ty::I32, si as u64), b.constant(f, Ty::I32, si as u64));
    let (la, lf) = (b.constant(t, Ty::I32, limit as u64), b.constant(f, Ty::I32, limit as u64));
    let (kj, lj) = (b.constant(j, Ty::I32, si as u64), b.constant(j, Ty::I32, limit as u64));
    let guard;
    if split {
        let ti = b.int(t, IntOp::Add, tp[1], ka);
        let tc = b.cmp(t, pred, ti, la);
        b.cond_br(t, tc, (h, vec![tp[0], ti, na, tp[3]]), (x, vec![tp[0], ti, na, tp[3]]));
        let fi = b.int(f, IntOp::Add, fp[1], kf);
        let fc = b.cmp(f, pred, fi, lf);
        b.cond_br(f, fc, (h, vec![fp[0], fi, fp[2], nb]), (x, vec![fp[0], fi, fp[2], nb]));
        let _ = (kj, lj, jp);
        b.br(j, x, vec![k.exec, p[1], p[2], p[3]]);
        guard = None;
    } else {
        b.br(t, j, vec![tp[0], tp[1], na, tp[3]]);
        b.br(f, j, vec![fp[0], fp[1], fp[2], nb]);
        let ji = b.int(j, IntOp::Add, jp[1], kj);
        let jc = b.cmp(j, pred, ji, lj);
        b.cond_br(j, jc, (h, vec![jp[0], ji, jp[2], jp[3]]), (x, vec![jp[0], ji, jp[2], jp[3]]));
        guard = Some(jc);
    }
    let mut ti = vec![Vec::new(); 32];
    let (mut ta, mut tb, mut ts, mut tna, mut tnb, mut tns, mut tx) = (ti.clone(), ti.clone(), ti.clone(), ti.clone(), ti.clone(), ti.clone(), ti.clone());
    let mut tg: Vec<Vec<bool>> = vec![Vec::new(); 32];
    for l in 0..32u32 {
        let (mut i, mut a, mut c, mut held) = (i0, l.wrapping_mul(ma).wrapping_add(ca), l.wrapping_mul(mb).wrapping_add(cb), s0);
        for _ in 0..ORACLE_CAP {
            let li = l as usize;
            ti[li].push(i);
            ta[li].push(a);
            tb[li].push(c);
            ts[li].push(held);
            let ni = i.wrapping_add(si);
            if taken_in(i) {
                let na = run_update(ua, a, c, i, l);
                let ns = run_update(us, held, a, i, l);
                tna[li].push(na);
                tns[li].push(ns);
                if store_in_t {
                    held = ns;
                }
                a = na;
            } else {
                let nb = run_update(ub, c, a, i, l);
                tnb[li].push(nb);
                c = nb;
            }
            let go = compare(pred, ni, limit);
            tg[li].push(go);
            if !go {
                tx[li].push(a);
                break;
            }
            i = ni;
        }
    }
    let name = format!(
        "i0 {} step {:#x} {:?} {:#x} branch {} {} split {} store {} a0 = lane*{:#x}+{:#x} b0 = lane*{:#x}+{:#x} s0 {:#x} a' {:?} b' {:?} s' {:?}",
        i0, si, pred, limit, branch_kind, branch_k, split, store_in_t, ma, ca, mb, cb, s0, ua, ub, us
    );
    let exit_a = xp[2];
    Shaped {
        b,
        header: h,
        watched: vec![
            ("i", p[1], ti),
            ("a", p[2], ta),
            ("b", p[3], tb),
            ("slot", s, ts),
            ("a'", na, tna),
            ("b'", nb, tnb),
            ("s'", ns, tns),
            ("exit a", exit_a, tx),
        ],
        guard: guard.map(|g| (g, tg)),
        name,
    }
}

fn shaped_failures(seed: u64, count: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    let only: Option<usize> = std::env::var("ORACLE_CASE").ok().and_then(|s| s.parse().ok());
    for case in 0..count {
        let c = shaped_loop(&mut r);
        if only.is_some_and(|o| o != case) {
            continue;
        }
        let env = environment(32, &[]);
        addresses(&c.b, &env, |a| {
            for &lane in &[0usize, 5, 31] {
                for (name, v, truth) in &c.watched {
                    let form = a.values.value(*v, lane, None).0.form;
                    if let Some(t) = truth[lane].iter().find(|&&t| !oracle_holds(a.unknowns(), &form, t)) {
                        wrong.push(format!("case {} [{}] lane {}: {} = {:?} misses {:#x}", case, c.name, lane, name, form, t));
                    }
                }
                if let Some((g, truth)) = &c.guard {
                    if let Some(bit) = a.bit(*g, lane, None).0 {
                        if truth[lane].iter().any(|&t| t != bit) {
                            wrong.push(format!("case {} [{}] lane {}: the guard said always {}", case, c.name, lane, bit));
                        }
                    }
                }
            }
            if let Some(u) = a.trip(c.header) {
                if let Some((_, last)) = a.unknowns()[u as usize].range {
                    let visits = c.watched[0].2[0].len();
                    if visits as u64 > last as u64 + 1 {
                        wrong.push(format!("case {} [{}]: trips said at most {}, the loop visits its header {} times", case, c.name, last, visits));
                    }
                }
            }
        });
    }
    wrong
}

#[test]
fn values_hold_every_value_a_random_branching_loop_gives() {
    let seed = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(11);
    let count = std::env::var("ORACLE_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let wrong = shaped_failures(seed, count);
    assert!(wrong.is_empty(), "{} wrong, first: {:#?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

fn flag_loop(r: &mut Random) -> Shaped {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let (ma, ca) = (if r.below(2) == 0 { 0 } else { interesting(r) }, interesting(r));
    let f0_kind = r.below(3);
    let i0 = r.below(8) as u32;
    let si = [1u32, 2, 3, u32::MAX, 4][r.below(5) as usize];
    let pred = PREDICATES[r.below(10) as usize];
    let limit = if r.below(3) != 0 { i0.wrapping_add(si.wrapping_mul(r.below(30) as u32)) } else { interesting(r) };
    let x = interesting(r);
    let ua = random_update(r);
    let uf = r.below(7);
    let fk = r.below(8) as u32;
    let nested = r.below(2) == 0;
    let inner = 1 + r.below(4) as u32;
    let m = interesting(r);
    let on_flag = f0_kind != 2 && !matches!(uf, 5) && r.below(4) == 0;
    let km = b.constant(e, Ty::I32, ma as u64);
    let kc = b.constant(e, Ty::I32, ca as u64);
    let scaled = b.int(e, IntOp::Mul, lane, km);
    let a0 = b.int(e, IntOp::Add, scaled, kc);
    let f0 = match f0_kind {
        0 => b.constant(e, Ty::I1, 1),
        1 => b.constant(e, Ty::I1, 0),
        _ => {
            let eight = b.constant(e, Ty::I32, 8);
            b.cmp(e, IntPred::Ult, lane, eight)
        }
    };
    let start = b.constant(e, Ty::I32, i0 as u64);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I1]);
    let (x_block, _) = b.block(&[Ty::I1]);
    b.br(e, h, vec![k.exec, start, a0, f0]);
    let hl = b.core(h, Ty::I32, Op::Env(Env::LaneId));
    let kx = b.constant(h, Ty::I32, x as u64);
    let plus = b.int(h, IntOp::Add, p[2], kx);
    let other = build_update(&mut b, h, ua, p[2], p[2], p[1], hl);
    let chosen = b.core(h, Ty::I32, Op::Select(p[3], plus, other));
    let kf = b.constant(h, Ty::I32, fk as u64);
    let one1 = b.constant(h, Ty::I1, 1);
    let cmp_i = b.cmp(h, IntPred::Ult, p[1], kf);
    let nf = match uf {
        0 => p[3],
        1 => b.int(h, IntOp::Xor, p[3], one1),
        2 => cmp_i,
        3 => b.int(h, IntOp::And, p[3], cmp_i),
        4 => b.int(h, IntOp::Or, p[3], cmp_i),
        5 => {
            let low = b.cmp(h, IntPred::Ult, hl, kf);
            b.int(h, IntOp::And, p[3], low)
        }
        _ => {
            let z = b.constant(h, Ty::I32, 0);
            let c = b.cmp(h, IntPred::Ne, chosen, z);
            b.int(h, IntOp::And, p[3], c)
        }
    };
    let ks = b.constant(h, Ty::I32, si as u64);
    let ni = b.int(h, IntOp::Add, p[1], ks);
    let kl = b.constant(h, Ty::I32, limit as u64);
    let (body_end, na) = if nested {
        let (ih, ip) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (after, ap) = b.block(&[Ty::I1, Ty::I32]);
        let zero = b.constant(h, Ty::I32, 0);
        b.br(h, ih, vec![p[0], chosen, zero]);
        let kmm = b.constant(ih, Ty::I32, m as u64);
        let added = b.int(ih, IntOp::Add, ip[1], kmm);
        let one = b.constant(ih, Ty::I32, 1);
        let nj = b.int(ih, IntOp::Add, ip[2], one);
        let ki = b.constant(ih, Ty::I32, inner as u64);
        let more = b.cmp(ih, IntPred::Ult, nj, ki);
        b.cond_br(ih, more, (ih, vec![ip[0], added, nj]), (after, vec![ip[0], added]));
        (after, ap[1])
    } else {
        (h, chosen)
    };
    let again = if on_flag { nf } else { b.cmp(body_end, pred, ni, kl) };
    b.cond_br(body_end, again, (h, vec![p[0], ni, na, nf]), (x_block, vec![p[0]]));
    let mut ti = vec![Vec::new(); 32];
    let (mut ta, mut tna, mut tf) = (ti.clone(), ti.clone(), ti.clone());
    let mut tg: Vec<Vec<bool>> = vec![Vec::new(); 32];
    for l in 0..32u32 {
        let li = l as usize;
        let (mut i, mut a) = (i0, l.wrapping_mul(ma).wrapping_add(ca));
        let mut f = match f0_kind {
            0 => true,
            1 => false,
            _ => l < 8,
        };
        for _ in 0..ORACLE_CAP {
            ti[li].push(i);
            ta[li].push(a);
            tf[li].push(f as u32);
            let chosen = if f { a.wrapping_add(x) } else { run_update(ua, a, a, i, l) };
            let nf = match uf {
                0 => f,
                1 => !f,
                2 => i < fk,
                3 => f && i < fk,
                4 => f || i < fk,
                5 => f && l < fk,
                _ => f && chosen != 0,
            };
            let na = if nested { chosen.wrapping_add(m.wrapping_mul(inner)) } else { chosen };
            tna[li].push(na);
            let ni = i.wrapping_add(si);
            let go = if on_flag { nf } else { compare(pred, ni, limit) };
            tg[li].push(go);
            if !go {
                break;
            }
            (i, a, f) = (ni, na, nf);
        }
    }
    let name = format!(
        "i0 {} step {:#x} {:?} {:#x} on_flag {} a0 = lane*{:#x}+{:#x} f0 {} x {:#x} a' {:?} f' {} k {} nested {} inner {} m {:#x}",
        i0, si, pred, limit, on_flag, ma, ca, f0_kind, x, ua, uf, fk, nested, inner, m
    );
    Shaped {
        b,
        header: h,
        watched: vec![("i", p[1], ti), ("a", p[2], ta), ("a'", na, tna), ("f", p[3], tf)],
        guard: Some((again, tg)),
        name,
    }
}

fn flag_failures(seed: u64, count: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    let only: Option<usize> = std::env::var("ORACLE_CASE").ok().and_then(|s| s.parse().ok());
    for case in 0..count {
        let c = flag_loop(&mut r);
        if only.is_some_and(|o| o != case) {
            continue;
        }
        let env = environment(32, &[]);
        addresses(&c.b, &env, |a| {
            for &lane in &[0usize, 5, 31] {
                for (name, v, truth) in &c.watched {
                    if *name == "f" {
                        if let Some(bit) = a.bit(*v, lane, None).0 {
                            if truth[lane].iter().any(|&t| (t != 0) != bit) {
                                wrong.push(format!("case {} [{}] lane {}: flag said always {}, truth {:?}", case, c.name, lane, bit, &truth[lane][..truth[lane].len().min(8)]));
                            }
                        }
                        continue;
                    }
                    let form = a.values.value(*v, lane, None).0.form;
                    if let Some(t) = truth[lane].iter().find(|&&t| !oracle_holds(a.unknowns(), &form, t)) {
                        wrong.push(format!("case {} [{}] lane {}: {} = {:?} misses {:#x}", case, c.name, lane, name, form, t));
                    }
                }
                if let Some((g, truth)) = &c.guard {
                    if let Some(bit) = a.bit(*g, lane, None).0 {
                        if truth[lane].iter().any(|&t| t != bit) {
                            wrong.push(format!("case {} [{}] lane {}: the guard said always {}", case, c.name, lane, bit));
                        }
                    }
                }
            }
            if let Some(u) = a.trip(c.header) {
                if let Some((_, last)) = a.unknowns()[u as usize].range {
                    let visits = c.watched[0].2[0].len();
                    if visits as u64 > last as u64 + 1 {
                        wrong.push(format!("case {} [{}]: trips said at most {}, the loop visits its header {} times", case, c.name, last, visits));
                    }
                }
            }
        });
    }
    wrong
}

#[test]
fn values_hold_every_value_a_random_loop_with_a_carried_flag_gives() {
    let seed = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(13);
    let count = std::env::var("ORACLE_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let wrong = flag_failures(seed, count);
    assert!(wrong.is_empty(), "{} wrong, first: {:#?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

#[test]
fn first_failure_names_the_first_exit_even_far_out() {
    let mut r = Random::new(77);
    let mut checked = 0;
    let mut wrong = Vec::new();
    for _ in 0..2_000_000 {
        let pred = PREDICATES[r.below(10) as usize];
        let taken = r.below(2) == 0;
        let x = (interesting(&mut r), if r.below(2) == 0 { 0 } else { interesting(&mut r) });
        let y = (interesting(&mut r), if r.below(3) == 0 { interesting(&mut r) } else { 0 });
        let Some(last) = first_failure(pred, taken, x, y) else { continue };
        if !(4096..=1 << 22).contains(&last) {
            continue;
        }
        checked += 1;
        let at = |v: (u32, u32), t: u32| v.0.wrapping_add(v.1.wrapping_mul(t));
        if let Some(t) = (0..last).find(|&t| compare(pred, at(x, t), at(y, t)) != taken) {
            wrong.push(format!("{:?} taken={} {:?} {:?}: leaves at {}, said {}", pred, taken, x, y, t, last));
        }
        if checked > 300 {
            break;
        }
    }
    assert!(wrong.is_empty(), "checked {}: {:?}", checked, wrong);
}

fn derived_store_meets(
    env: &Environment,
    derive: impl Fn(&mut Build, BlockId, ValueId, ValueId) -> ValueId,
    second_is_target: bool,
) -> bool {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let derived = derive(&mut b, e, first, second);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, if second_is_target { second } else { first }, lane, 4);
    let zero = b.constant(e, Ty::I32, 0);
    b.store(e, Space::Global, MemSize::B32, derived, zero, k.exec);
    b.store(e, Space::Global, MemSize::B32, own, zero, k.exec);
    let h = super::super::hazard::Hazards::find(&b.program(), env);
    !h.conflicts().is_empty()
}

#[test]
fn hazards_report_stores_through_a_pointer_negated_twice() {
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let meets = derived_store_meets(
        &env,
        |b, e, first, _| {
            let zero = b.constant(e, Ty::I64, 0);
            let negated = b.int(e, IntOp::Sub, zero, first);
            b.int(e, IntOp::Sub, zero, negated)
        },
        false,
    );
    assert!(meets, "0 - (0 - first) is first, so every lane meets lane 0 at first[0]");
}

#[test]
fn hazards_report_stores_through_a_pointer_aligned_by_shifting() {
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let meets = derived_store_meets(
        &env,
        |b, e, first, _| {
            let four = b.constant(e, Ty::I64, 4);
            let down = b.int(e, IntOp::LShr, first, four);
            b.int(e, IntOp::Shl, down, four)
        },
        false,
    );
    assert!(meets, "(first >> 4) << 4 is first, so every lane meets lane 0 at first[0]");
}

fn pointer_slots(r: &mut Random) -> PointerLoop {
    use Pointer::*;
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let init = [r.below(3), r.below(3)];
    let slots_init: Vec<ValueId> = (0..2u64).map(|s| b.constant(e, Ty::I32, 32 + 8 * s)).collect();
    for (s, &which) in init.iter().enumerate() {
        match which {
            0 => b.store(e, Space::Scratch, MemSize::B64, slots_init[s], first, k.exec),
            1 => b.store(e, Space::Scratch, MemSize::B64, slots_init[s], second, k.exec),
            _ => {}
        }
    }
    let zero = b.constant(e, Ty::I32, 0);
    let (fp, fq) = (r.below(2) == 0, r.below(2) == 0);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
    let (x, _) = b.block(&[Ty::I1]);
    b.br(e, h, vec![k.exec, zero, if fp { first } else { second }, if fq { first } else { second }]);
    let d = r.below(2) as u32;
    let stored = r.below(3);
    let (up, uq) = (random_pointer(r), random_pointer(r));
    let nested = r.below(2) == 0;
    let inner = 1 + r.below(3) as u32;
    let trips = 1 + r.below(6) as u32;
    let one = b.constant(h, Ty::I32, 1);
    let eight = b.constant(h, Ty::I32, 8);
    let base = b.constant(h, Ty::I32, 32);
    let kd = b.constant(h, Ty::I32, d as u64);
    let shifted = b.int(h, IntOp::Add, p[1], kd);
    let rbit = b.int(h, IntOp::And, shifted, one);
    let roff = b.int(h, IntOp::Mul, rbit, eight);
    let raddr = b.int(h, IntOp::Add, base, roff);
    let reloaded = b.load(h, Space::Scratch, MemSize::B64, raddr, p[0]);
    let wbit = b.int(h, IntOp::And, p[1], one);
    let woff = b.int(h, IntOp::Mul, wbit, eight);
    let waddr = b.int(h, IntOp::Add, base, woff);
    let data = [p[2], p[3], reloaded][stored as usize];
    b.store(h, Space::Scratch, MemSize::B64, waddr, data, p[0]);
    let even = b.cmp(h, IntPred::Eq, wbit, zero);
    let sixteen = b.constant(h, Ty::I32, 16);
    let low = b.cmp(h, IntPred::Ult, lane, sixteen);
    let (body, bp) = if nested {
        let (ih, ip) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I32]);
        let (after, ap) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        b.br(h, ih, vec![p[0], p[2], p[3], zero]);
        let one2 = b.constant(ih, Ty::I32, 1);
        let nj = b.int(ih, IntOp::Add, ip[3], one2);
        let ki = b.constant(ih, Ty::I32, inner as u64);
        let more = b.cmp(ih, IntPred::Ult, nj, ki);
        b.cond_br(ih, more, (ih, vec![ip[0], ip[2], ip[1], nj]), (after, vec![ip[0], ip[2], ip[1]]));
        (after, [ap[1], ap[2]])
    } else {
        (h, [p[2], p[3]])
    };
    let eight64 = b.constant(body, Ty::I64, 8);
    let build = |b: &mut Build, u: Pointer, own: ValueId, other: ValueId| match u {
        Same => own,
        Other => other,
        Offset => b.int(body, IntOp::Add, own, eight64),
        ByCounter => b.core(body, Ty::I64, Op::Select(even, own, other)),
        ByLane => b.core(body, Ty::I64, Op::Select(low, own, other)),
        First => first,
        Second => second,
        Reloaded => reloaded,
        Integer => {
            let w = b.core(body, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
            b.int(body, IntOp::Add, w, eight64)
        }
    };
    let np = build(&mut b, up, bp[0], bp[1]);
    let nq = build(&mut b, uq, bp[1], bp[0]);
    let one3 = b.constant(body, Ty::I32, 1);
    let ni = b.int(body, IntOp::Add, p[1], one3);
    let limit = b.constant(body, Ty::I32, trips as u64);
    let again = b.cmp(body, IntPred::Ult, ni, limit);
    b.cond_br(body, again, (h, vec![p[0], ni, np, nq]), (x, vec![p[0]]));
    let tag = |f: bool| Some(if f { 1u64 } else { 2 });
    let mut tp: Vec<Vec<Option<u64>>> = vec![Vec::new(); 32];
    let (mut tq, mut tr, mut tnp, mut tnq) = (tp.clone(), tp.clone(), tp.clone(), tp.clone());
    for l in 0..32usize {
        let (mut cp, mut cq) = (tag(fp), tag(fq));
        let mut slot: [Option<u64>; 2] = [None, None];
        for s in 0..2 {
            slot[s] = match init[s] {
                0 => Some(1),
                1 => Some(2),
                _ => None,
            };
        }
        for t in 0..trips {
            let rl = slot[((t + d) & 1) as usize];
            let data = [cp, cq, rl][stored as usize];
            slot[(t & 1) as usize] = data;
            let (bp, bq) = if nested && inner % 2 == 1 { (cq, cp) } else { (cp, cq) };
            let step = |u: Pointer, own: Option<u64>, other: Option<u64>| match u {
                Same | Offset => own,
                Other => other,
                ByCounter => {
                    if t & 1 == 0 {
                        own
                    } else {
                        other
                    }
                }
                ByLane => {
                    if l < 16 {
                        own
                    } else {
                        other
                    }
                }
                First => Some(1),
                Second => Some(2),
                Reloaded => rl,
                Integer => None,
            };
            let (np, nq) = (step(up, bp, bq), step(uq, bq, bp));
            tp[l].push(cp);
            tq[l].push(cq);
            tr[l].push(rl);
            tnp[l].push(np);
            tnq[l].push(nq);
            (cp, cq) = (np, nq);
        }
    }
    let name = format!(
        "p0 {} q0 {} init {:?} d {} stored {} p' {:?} q' {:?} nested {} inner {} trips {}",
        tag(fp).unwrap(),
        tag(fq).unwrap(),
        init,
        d,
        stored,
        up,
        uq,
        nested,
        inner,
        trips
    );
    PointerLoop {
        b,
        watched: vec![("p", p[2], tp), ("q", p[3], tq), ("reloaded", reloaded, tr), ("p'", np, tnp), ("q'", nq, tnq)],
        name,
    }
}

#[test]
fn regions_hold_every_allocation_random_pointer_slots_carry() {
    let seed = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(17);
    let count: usize = std::env::var("ORACLE_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for case in 0..count {
        let c = pointer_slots(&mut r);
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        addresses(&c.b, &env, |a| {
            loop {
                for (_, v, _) in &c.watched {
                    for l in 0..32 {
                        for refine in modes() {
                            a.regions(*v, l, None, refine);
                        }
                    }
                }
                if !a.settle_loops() {
                    break;
                }
            }
            for (name, v, truth) in &c.watched {
                for l in [0usize, 20] {
                    for refine in modes() {
                        let set = a.regions(*v, l, None, refine);
                        for t in &truth[l] {
                            let hit = match t {
                                Some(id) => set.reaches(Some(Region::Allocation(*id)), |x, y| x == y),
                                None => set.lost(),
                            };
                            if !hit {
                                wrong.push(format!("case {} [{}] {} lane {} refine {}: {:?} misses {:?}", case, c.name, name, l, refine, set, t));
                                break;
                            }
                        }
                    }
                    let value = a.values.value(*v, l, None).0;
                    if let Some(region) = value.region {
                        if truth[l].iter().any(|t| t.map(Region::Allocation) != Some(region)) {
                            wrong.push(format!("case {} [{}] {} lane {}: value says {:?}, truth {:?}", case, c.name, name, l, region, truth[l]));
                        }
                    }
                }
            }
        });
    }
    assert!(wrong.is_empty(), "{} wrong, first: {:#?}", wrong.len(), &wrong[..wrong.len().min(10)]);
}

#[test]
fn nested_affine_loop_values_hold_every_value_the_inner_parameter_takes() {
    let mut wrong = Vec::new();
    for (outer, inner) in [(1u64, 1u64), (2, 3), (5, 7), (7, 5), (40, 3), (3, 40)] {
        let (b, j) = nested_affine_loops(outer, inner);
        let mut truth = Vec::new();
        let mut x = 1u32;
        for _ in 0..outer {
            let mut v = x;
            for _ in 0..inner {
                truth.push(v);
                v = v.wrapping_mul(5).wrapping_add(7);
            }
            x = x.wrapping_mul(3).wrapping_add(1);
        }
        addresses(&b, &environment(32, &[]), |a| {
            let form = a.values.value(j, 0, None).0.form;
            for &t in &truth {
                if !oracle_holds(a.unknowns(), &form, t) {
                    wrong.push(format!("outer {} inner {}: {:?} misses {:#x}", outer, inner, form, t));
                    break;
                }
            }
        });
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn a_reloaded_spill_of_a_pointer_a_loop_swaps_points_into_both_buffers() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = k.buffer(&mut b, e, 0);
    let second = k.buffer(&mut b, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
    let (x, _) = b.block(&[Ty::I1]);
    b.br(e, h, vec![k.exec, zero, first, second]);
    let slot = b.constant(h, Ty::I32, 16);
    b.store(h, Space::Scratch, MemSize::B64, slot, p[2], p[0]);
    let reloaded = b.load(h, Space::Scratch, MemSize::B64, slot, p[0]);
    let one = b.constant(h, Ty::I32, 1);
    let ni = b.int(h, IntOp::Add, p[1], one);
    let four = b.constant(h, Ty::I32, 4);
    let again = b.cmp(h, IntPred::Ult, ni, four);
    b.cond_br(h, again, (h, vec![p[0], ni, p[3], p[2]]), (x, vec![p[0]]));
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let (param, spilled) = addresses(&b, &env, |a| {
        loop {
            for v in [p[2], reloaded] {
                for refine in [false, true] {
                    a.regions(v, 0, None, refine);
                }
            }
            if !a.settle_loops() {
                break;
            }
        }
        (a.regions(p[2], 0, None, true), a.regions(reloaded, 0, None, true))
    });
    for id in [1u64, 2] {
        assert!(param.reaches(Some(Region::Allocation(id)), |x, y| x == y), "p: {:?}", param);
        assert!(
            spilled.reaches(Some(Region::Allocation(id)), |x, y| x == y),
            "the reload gives p back, which points into allocations 1 and 2, but its regions are {:?} (p: {:?})",
            spilled,
            param
        );
    }
}

#[test]
fn copies_name_a_root_every_incoming_edge_brings_or_copies() {
    let mut r = Random::new(29);
    let mut wrong = Vec::new();
    for case in 0..3000 {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let consts: Vec<ValueId> = (0..3).map(|i| b.constant(e, Ty::I32, i)).collect();
        let n = 2 + r.below(5) as usize;
        let mut blocks = Vec::new();
        for _ in 0..n {
            let count = 1 + r.below(3) as usize;
            let mut types = vec![Ty::I1];
            types.extend(std::iter::repeat_n(Ty::I32, count));
            blocks.push(b.block(&types));
        }
        let pick = |r: &mut Random, from: &[ValueId]| -> ValueId {
            let all: Vec<ValueId> = consts.iter().copied().chain(from.iter().copied()).collect();
            all[r.below(all.len() as u64) as usize]
        };
        let args_for = |r: &mut Random, dst: usize, own: &[ValueId], exec: ValueId| -> Vec<ValueId> {
            let mut args = vec![exec];
            for _ in 1..blocks[dst].1.len() {
                args.push(pick(r, own));
            }
            args
        };
        let first_args = args_for(&mut r, 0, &[], k.exec);
        b.br(e, blocks[0].0, first_args);
        for i in 0..n {
            let (id, ref params) = blocks[i];
            let own: Vec<ValueId> = params[1..].to_vec();
            let next = (i + 1).min(n - 1);
            let back = r.below(i as u64 + 1) as usize;
            if i + 1 == n {
                if r.below(2) == 0 {
                    let c = b.constant(id, Ty::I1, 0);
                    let args = args_for(&mut r, back, &own, params[0]);
                    let (exit, _) = b.block(&[Ty::I1]);
                    b.cond_br(id, c, (blocks[back].0, args), (exit, vec![params[0]]));
                }
                continue;
            }
            let c = b.constant(id, Ty::I1, 0);
            let target = if r.below(2) == 0 { back } else { (i + 1 + r.below((n - i - 1) as u64) as usize).min(n - 1) };
            let a1 = args_for(&mut r, next, &own, params[0]);
            let a2 = args_for(&mut r, target, &own, params[0]);
            b.cond_br(id, c, (blocks[next].0, a1), (blocks[target].0, a2));
        }
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let copies = super::graph::copies(&b.f, &facts);
        let params: BTreeSet<ValueId> = blocks.iter().flat_map(|(_, p)| p[1..].iter().copied()).collect();
        for &p in &params {
            let Some(&root) = copies.get(&p) else { continue };
            if facts.site[p.0] == crate::rdna_spmd::analysis::facts::Site::Unreached {
                continue;
            }
            let crate::rdna_spmd::analysis::facts::Site::Param { block, index } = facts.site[p.0] else { continue };
            for &(pred, slot) in &facts.incoming[&block] {
                let a = b.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
                let ra = copies.get(&a).copied().unwrap_or(a);
                if ra != root {
                    wrong.push(format!("case {}: v{} called a copy of v{}, but an edge brings v{} (a copy of v{})", case, p.0, root.0, a.0, ra.0));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(5)]);
}

#[test]
fn a_loop_parameter_fed_through_a_second_vgpr_input_points_into_the_buffer() {
    let (mut b, k, extra) = Build::kernel_with(&[(ParameterSource::Vgpr(1), Ty::I32)]);
    let e = BlockId(0);
    let second = k.buffer(&mut b, e, 8);
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, extra[0]));
    let zero = b.constant(e, Ty::I32, 0);
    let null = b.constant(e, Ty::I64, 0);
    let (h, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64]);
    let (x, _) = b.block(&[Ty::I1]);
    b.br(e, h, vec![k.exec, zero, null]);
    let moved = b.int(h, IntOp::Add, second, wide);
    let one = b.constant(h, Ty::I32, 1);
    let ni = b.int(h, IntOp::Add, p[1], one);
    let four = b.constant(h, Ty::I32, 4);
    let again = b.cmp(h, IntPred::Ult, ni, four);
    b.cond_br(h, again, (h, vec![p[0], ni, moved]), (x, vec![p[0]]));
    let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
    let set = addresses(&b, &env, |a| {
        loop {
            for refine in [false, true] {
                a.regions(p[2], 0, None, refine);
            }
            if !a.settle_loops() {
                break;
            }
        }
        a.regions(p[2], 0, None, true)
    });
    assert!(set.reaches(Some(Region::Allocation(2)), |x, y| x == y), "p is second from the second trip on, but its regions are {:?}", set);
}


fn halving_wide_loop_counted(bounded: bool) -> (Build, ValueId, ValueId, Vec<u32>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let start = b.constant(e, Ty::I64, 0x1_0000_0000);
    let zero = b.constant(e, Ty::I32, 0);
    let limit = if bounded { b.constant(e, Ty::I32, 4) } else { loaded_word(&mut b, &k, e, 0, MemSize::B32) };
    let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, start, zero]);
    let low = b.core(body, Ty::I32, Op::UnpackLo(p[1]));
    let nothing = b.constant(body, Ty::I32, 0);
    let low_is_zero = b.cmp(body, IntPred::Eq, low, nothing);
    let one64 = b.constant(body, Ty::I64, 1);
    let halved = b.int(body, IntOp::LShr, p[1], one64);
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[2], one);
    let again = b.cmp(body, IntPred::Ult, next, limit);
    b.cond_br(body, again, (body, vec![p[0], halved, next]), (exit, vec![p[0]]));
    (b, low, low_is_zero, vec![0, 0x8000_0000, 0x4000_0000, 0x2000_0000])
}

fn halving_wide_answers(bounded: bool) -> Vec<String> {
    let (b, low, low_is_zero, truth) = halving_wide_loop_counted(bounded);
    let mut wrong = Vec::new();
    addresses(&b, &two_words(), |a| {
        let form = a.values.value(low, 0, None).0.form;
        for &t in &truth {
            if !representable(a.unknowns(), &form, t) {
                let info: Vec<_> = form.terms.iter().map(|&(u, _)| (a.unknowns()[u as usize].range, a.unknowns()[u as usize].values.clone())).collect();
                wrong.push(format!("low word {:?} (ranges {:?}) cannot be {:#x}", form, info, t));
            }
        }
        if let Some(decided) = a.bit(low_is_zero, 0, None).0 {
            wrong.push(format!("low word == 0 decided {}, but it holds only in the first iteration", decided));
        }
    });
    wrong
}

#[test]
fn halving_wide_loop_parameters_hold_every_low_word_over_a_known_trip_count() {
    let wrong = halving_wide_answers(true);
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn halving_wide_loop_parameters_hold_every_low_word_over_an_unknown_trip_count() {
    let wrong = halving_wide_answers(false);
    assert!(wrong.is_empty(), "{:?}", wrong);
}

type Eval = std::rc::Rc<dyn Fn(&[u32; 4]) -> u32>;

fn oracle_constant(r: &mut Random) -> u32 {
    match r.below(4) {
        0 => r.below(40) as u32,
        1 => (1u32 << r.below(32)).wrapping_sub(r.below(2) as u32),
        2 => !((1u32 << r.below(32)).wrapping_sub(1)),
        _ => interesting(r),
    }
}

fn oracle_word(b: &mut Build, e: BlockId, r: &mut Random, leaves: &[ValueId; 4], depth: u32) -> (ValueId, String, Eval) {
    use IntOp::*;
    if depth == 0 || r.below(5) == 0 {
        let i = r.below(5) as usize;
        if i < 4 {
            let name = ["u", "z", "w", "lane"][i].to_string();
            return (leaves[i], name, std::rc::Rc::new(move |x: &[u32; 4]| x[i]));
        }
        let k = oracle_constant(r);
        return (b.constant(e, Ty::I32, k as u64), format!("{:#x}", k), std::rc::Rc::new(move |_: &[u32; 4]| k));
    }
    let ops = [Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr];
    match r.below(10) {
        0..=5 => {
            let op = ops[r.below(9) as usize];
            let (x, nx, fx) = oracle_word(b, e, r, leaves, depth - 1);
            let (y, ny, fy) = if r.below(2) == 0 {
                let k = if matches!(op, Shl | LShr | AShr) { r.below(40) as u32 } else { oracle_constant(r) };
                let f: Eval = std::rc::Rc::new(move |_: &[u32; 4]| k);
                (b.constant(e, Ty::I32, k as u64), format!("{:#x}", k), f)
            } else {
                oracle_word(b, e, r, leaves, depth - 1)
            };
            let v = b.int(e, op, x, y);
            let f: Eval = std::rc::Rc::new(move |i: &[u32; 4]| {
                let (a, s) = (fx(i), fy(i));
                match op {
                    Add => a.wrapping_add(s),
                    Sub => a.wrapping_sub(s),
                    Mul => a.wrapping_mul(s),
                    And => a & s,
                    Or => a | s,
                    Xor => a ^ s,
                    Shl => a << (s & 31),
                    LShr => a >> (s & 31),
                    AShr => ((a as i32) >> (s & 31)) as u32,
                }
            });
            (v, format!("({} {:?} {})", nx, op, ny), f)
        }
        6 | 7 => {
            let preds = [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Ugt, IntPred::Ule, IntPred::Uge, IntPred::Slt, IntPred::Sgt, IntPred::Sle, IntPred::Sge];
            let pred = preds[r.below(10) as usize];
            let (x, nx, fx) = oracle_word(b, e, r, leaves, depth - 1);
            let (y, ny, fy) = oracle_word(b, e, r, leaves, depth - 1);
            let (p, np, fp) = oracle_word(b, e, r, leaves, depth - 1);
            let (q, nq, fq) = oracle_word(b, e, r, leaves, depth - 1);
            let c = b.cmp(e, pred, x, y);
            let v = b.core(e, Ty::I32, Op::Select(c, p, q));
            let f: Eval = std::rc::Rc::new(move |i: &[u32; 4]| if super::compare(pred, fx(i), fy(i)) { fp(i) } else { fq(i) });
            (v, format!("({} {:?} {} ? {} : {})", nx, pred, ny, np, nq), f)
        }
        8 => {
            let k = r.below(4) as u8;
            let (x, nx, fx) = oracle_word(b, e, r, leaves, depth - 1);
            let op = [Op::PopulationCount(x), Op::TrailingZeros(x), Op::LeadingZeros(x), Op::ReverseBits(x)][k as usize];
            let v = b.core(e, Ty::I32, op);
            let f: Eval = std::rc::Rc::new(move |i: &[u32; 4]| {
                let a = fx(i);
                match k {
                    0 => a.count_ones(),
                    1 => a.trailing_zeros(),
                    2 => a.leading_zeros(),
                    _ => a.reverse_bits(),
                }
            });
            (v, format!("{:?}({})", ["popcount", "tz", "lz", "rev"][k as usize], nx), f)
        }
        _ => {
            let (x, nx, fx) = oracle_word(b, e, r, leaves, depth - 1);
            let preds = [IntPred::Ult, IntPred::Slt, IntPred::Eq];
            let pred = preds[r.below(3) as usize];
            let k = oracle_constant(r);
            let kk = b.constant(e, Ty::I32, k as u64);
            let c = b.cmp(e, pred, x, kk);
            let sext = r.below(2) == 0;
            let v = b.core(e, Ty::I32, Op::Convert(if sext { Cvt::SExt } else { Cvt::ZExt }, Ty::I32, c));
            let f: Eval = std::rc::Rc::new(move |i: &[u32; 4]| match (super::compare(pred, fx(i), k), sext) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => u32::MAX,
            });
            (v, format!("ext({} {:?} {:#x})", nx, pred, k), f)
        }
    }
}

fn oracle_samples(r: &mut Random) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for &u in &[0u32, 1, 2, 3, 7, 8, 0x7fff, 0x8000, 0xfffe, 0xffff] {
        for _ in 0..3 {
            out.push([u, interesting(r), interesting(r)]);
        }
    }
    for _ in 0..30 {
        out.push([r.below(65536) as u32, interesting(r), r.next() as u32]);
    }
    out
}

fn random_words(seed: u64, programs: usize, per: usize, depth: u32) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for _ in 0..programs {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let four = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, four);
        let z = b.load(e, Space::Global, MemSize::B32, second, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let leaves = [u, z, w, lane];
        let mut words = Vec::new();
        let mut bits = Vec::new();
        for _ in 0..per {
            let (v, name, f) = oracle_word(&mut b, e, &mut r, &leaves, depth);
            let (v2, name2, f2) = oracle_word(&mut b, e, &mut r, &leaves, depth - 1);
            let preds = [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Ugt, IntPred::Ule, IntPred::Uge, IntPred::Slt, IntPred::Sgt, IntPred::Sle, IntPred::Sge];
            let pred = preds[r.below(10) as usize];
            let c = b.cmp(e, pred, v, v2);
            bits.push((c, format!("{} {:?} {}", name, pred, name2), f.clone(), f2.clone(), pred));
            words.push((v, name, f));
            words.push((v2, name2, f2));
        }
        let samples = oracle_samples(&mut r);
        let ws: Vec<Vec<u32>> = (0..samples.len()).map(|_| (0..32).map(|_| interesting(&mut r)).collect()).collect();
        addresses(&b, &two_words(), |a| {
            for lane in [0usize, 5, 31] {
                let forms: Vec<Form> = words.iter().map(|(v, _, _)| a.values.value(*v, lane, None).0.form).collect();
                let decided: Vec<Option<bool>> = bits.iter().map(|(c, ..)| a.bit(*c, lane, None).0).collect();
                for (s, sample) in samples.iter().enumerate() {
                    let input = [sample[0], sample[1], ws[s][lane], lane as u32];
                    for (i, (_, name, f)) in words.iter().enumerate() {
                        let t = f(&input);
                        if !core_holds(a.unknowns(), &forms[i], t) {
                            wrong.push(format!("{} lane {} at {:x?}: {:?} cannot be {:#x}", name, lane, input, forms[i], t));
                        }
                    }
                    for i in 0..words.len() {
                        for j in 0..i {
                            if forms[i].terms.is_empty() || forms[i].terms != forms[j].terms {
                                continue;
                            }
                            let d = forms[i].constant.wrapping_sub(forms[j].constant);
                            let t = (words[i].2)(&input).wrapping_sub((words[j].2)(&input));
                            if d != t {
                                wrong.push(format!("{} minus {} lane {} at {:x?}: forms differ by {:#x}, values by {:#x}", words[i].1, words[j].1, lane, input, d, t));
                            }
                        }
                    }
                    for (i, (_, name, f1, f2, pred)) in bits.iter().enumerate() {
                        if let Some(k) = decided[i] {
                            if k != super::compare(*pred, f1(&input), f2(&input)) {
                                wrong.push(format!("{} lane {} at {:x?}: decided {}", name, lane, input, k));
                            }
                        }
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_word_expressions_hold_every_value_and_decide_only_what_holds() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let depth: u32 = std::env::var("ORACLE_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let wrong = random_words(seed, n, 4, depth);
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

fn solves(lo: u32, hi: u32, c: u32, need: u32) -> bool {
    let z = c.trailing_zeros();
    if z >= 32 {
        return need == 0;
    }
    if need & ((1u64 << z) - 1) as u32 != 0 {
        return false;
    }
    let modulus = 1u64 << (32 - z);
    let odd = (c >> z) as u64;
    let mut inverse = 1u64;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(inverse))) % modulus;
    }
    let first = ((need >> z) as u64 * inverse) % modulus;
    let start = if (lo as u64) <= first { first } else { first + (lo as u64 - first).div_ceil(modulus) * modulus };
    start <= hi as u64
}

fn core_holds(unknowns: &[UnknownInfo], form: &Form, truth: u32) -> bool {
    let choices = |u: Unknown| -> Option<Vec<u32>> {
        match (&unknowns[u as usize].values, unknowns[u as usize].range) {
            (Some(set), _) => Some(set.to_vec()),
            (None, Some((lo, hi))) if hi - lo < 1 << 16 => Some((lo..=hi).collect()),
            _ => None,
        }
    };
    let size = |u: Unknown| -> u64 {
        match (&unknowns[u as usize].values, unknowns[u as usize].range) {
            (Some(set), _) => set.len() as u64,
            (None, Some((lo, hi))) => hi as u64 - lo as u64 + 1,
            _ => 1 << 32,
        }
    };
    if form.terms.is_empty() {
        return form.constant == truth;
    }
    let solved = (0..form.terms.len())
        .filter(|&i| unknowns[form.terms[i].0 as usize].values.is_none())
        .max_by_key(|&i| size(form.terms[i].0));
    let rest: Vec<usize> = (0..form.terms.len()).filter(|&i| Some(i) != solved).collect();
    let sets: Option<Vec<Vec<u32>>> = rest.iter().map(|&i| choices(form.terms[i].0)).collect();
    let Some(sets) = sets else {
        return if std::env::var("ORACLE_SOLVE").is_ok() { representable(unknowns, form, truth) } else { true };
    };
    if sets.iter().try_fold(1u64, |acc, s| acc.checked_mul(s.len() as u64)).is_none_or(|n| n > 1 << 13) {
        return if std::env::var("ORACLE_SOLVE").is_ok() { representable(unknowns, form, truth) } else { true };
    }
    let mut index = vec![0usize; sets.len()];
    loop {
        let partial = rest
            .iter()
            .zip(&index)
            .zip(&sets)
            .fold(form.constant, |acc, ((&t, &i), set)| acc.wrapping_add(form.terms[t].1.wrapping_mul(set[i])));
        let hit = match solved {
            Some(t) => {
                let (u, c) = form.terms[t];
                let (lo, hi) = unknowns[u as usize].range.unwrap_or((0, u32::MAX));
                solves(lo, hi, c, truth.wrapping_sub(partial))
            }
            None => partial == truth,
        };
        if hit {
            return true;
        }
        let mut k = 0;
        loop {
            if k == index.len() {
                return false;
            }
            if index[k] + 1 < sets[k].len() {
                index[k] += 1;
                break;
            }
            index[k] = 0;
            k += 1;
        }
    }
}

fn fields(leaf_size: MemSize) -> Vec<String> {
    use IntOp::*;
    let scales = [1u32, 2, 3, 4, 6, 8, 0x10, 0x10000, 0x8000_0000, 0xffff_fffc, 0xffff_ffff];
    let offsets = [0u32, 1, 3, 4, 7, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0xffff_fffc, 0x1234_5678];
    let mut operations: Vec<(IntOp, u32)> = Vec::new();
    for k in [1u32, 2, 3, 4, 16, 31] {
        operations.push((LShr, k));
        operations.push((AShr, k));
    }
    for m in [0u32, 7, 0xff, 0x3c, 0xff00, 0xfff0, 0xffff_fff0, 0x8000_0000, 0x7fff_ffff, 0xf_ffff, 0xffff_0000] {
        operations.push((And, m));
    }
    for c in [1u32, 2, 3, 5, 0x10, 0x3c, 0x7fff_ffff] {
        operations.push((Or, c));
        operations.push((Xor, c));
    }
    let mut wrong = Vec::new();
    for chunk in scales.chunks(3) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let x = b.load(e, Space::Global, leaf_size, table, yes);
        let mut cases = Vec::new();
        for &m in chunk {
            let km = b.constant(e, Ty::I32, m as u64);
            let scaled = b.int(e, Mul, x, km);
            for &c in &offsets {
                let kc = b.constant(e, Ty::I32, c as u64);
                let sum = b.int(e, Add, scaled, kc);
                for &(op, n) in &operations {
                    let kn = b.constant(e, Ty::I32, n as u64);
                    let v = b.int(e, op, sum, kn);
                    cases.push((v, m, c, op, n));
                }
            }
        }
        let mut r = Random::new(99);
        let mut values: Vec<u32> = vec![0, 1, 2, 3, 0x7fff, 0x8000, 0xffff, 0x3fff_ffff, 0x4000_0000, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0xffff_fffe];
        for _ in 0..24 {
            values.push(r.next() as u32);
        }
        if leaf_size == MemSize::U16 {
            values.iter_mut().for_each(|v| *v &= 0xffff);
        }
        addresses(&b, &two_words(), |a| {
            let forms: Vec<Form> = cases.iter().map(|c| a.values.value(c.0, 0, None).0.form).collect();
            for &x in &values {
                let truths: Vec<u32> = cases
                    .iter()
                    .map(|&(_, m, c, op, n)| {
                        let s = x.wrapping_mul(m).wrapping_add(c);
                        match op {
                            LShr => s >> n,
                            AShr => ((s as i32) >> n) as u32,
                            And => s & n,
                            Or => s | n,
                            _ => s ^ n,
                        }
                    })
                    .collect();
                for (i, &(_, m, c, op, n)) in cases.iter().enumerate() {
                    if !core_holds(a.unknowns(), &forms[i], truths[i]) {
                        wrong.push(format!("({:#x} x + {:#x}) {:?} {:#x} at x = {:#x}: {:?} cannot be {:#x}", m, c, op, n, x, forms[i], truths[i]));
                    }
                }
                for i in 0..cases.len() {
                    for j in 0..i {
                        if forms[i].terms.is_empty() || forms[i].terms != forms[j].terms {
                            continue;
                        }
                        let d = forms[i].constant.wrapping_sub(forms[j].constant);
                        let t = truths[i].wrapping_sub(truths[j]);
                        if d != t {
                            let (ci, cj) = (cases[i], cases[j]);
                            wrong.push(format!("({:#x} x + {:#x}) {:?} {:#x} minus ({:#x} x + {:#x}) {:?} {:#x} at x = {:#x}: forms differ by {:#x}, values by {:#x}", ci.1, ci.2, ci.3, ci.4, cj.1, cj.2, cj.3, cj.4, x, d, t));
                        }
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn fields_of_scaled_unbounded_words_hold_every_value() {
    let wrong = fields(MemSize::B32);
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

#[test]
fn fields_of_scaled_halfwords_hold_every_value() {
    let wrong = fields(MemSize::U16);
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

type WideEval = std::rc::Rc<dyn Fn(&[u32; 3]) -> u64>;

fn wide_constant(r: &mut Random) -> u64 {
    match r.below(5) {
        0 => r.below(70),
        1 => 1u64 << r.below(64),
        2 => (r.below(16)).wrapping_neg(),
        3 => (interesting(r) as u64) | (interesting(r) as u64) << 32,
        _ => r.next(),
    }
}

fn oracle_wide(b: &mut Build, e: BlockId, r: &mut Random, leaves: &[ValueId; 3], depth: u32) -> (ValueId, String, WideEval) {
    use IntOp::*;
    if depth == 0 || r.below(4) == 0 {
        return match r.below(6) {
            0 => (b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, leaves[0])), "zext u".into(), std::rc::Rc::new(|x: &[u32; 3]| x[0] as u64)),
            1 => {
                let k = interesting(r);
                let kk = b.constant(e, Ty::I32, k as u64);
                let s = b.int(e, Sub, leaves[0], kk);
                (b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, s)), format!("sext(u - {:#x})", k), std::rc::Rc::new(move |x: &[u32; 3]| x[0].wrapping_sub(k) as i32 as i64 as u64))
            }
            2 => (b.core(e, Ty::I64, Op::Pack64(leaves[0], leaves[1])), "pack(u, z)".into(), std::rc::Rc::new(|x: &[u32; 3]| x[0] as u64 | (x[1] as u64) << 32)),
            3 => (b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, leaves[1])), "zext z".into(), std::rc::Rc::new(|x: &[u32; 3]| x[1] as u64)),
            4 => (b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, leaves[2])), "zext lane".into(), std::rc::Rc::new(|x: &[u32; 3]| x[2] as u64)),
            _ => {
                let k = wide_constant(r);
                (b.constant(e, Ty::I64, k), format!("{:#x}", k), std::rc::Rc::new(move |_: &[u32; 3]| k))
            }
        };
    }
    let ops = [Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr];
    match r.below(8) {
        0..=6 => {
            let op = ops[r.below(9) as usize];
            let (x, nx, fx) = oracle_wide(b, e, r, leaves, depth - 1);
            let constant = matches!(op, Mul | And | Or | Xor | Shl | LShr | AShr) || r.below(2) == 0;
            let (y, ny, fy) = if constant {
                let k = if matches!(op, Shl | LShr | AShr) { r.below(70) } else { wide_constant(r) };
                let f: WideEval = std::rc::Rc::new(move |_: &[u32; 3]| k);
                (b.constant(e, Ty::I64, k), format!("{:#x}", k), f)
            } else {
                oracle_wide(b, e, r, leaves, depth - 1)
            };
            let v = b.int(e, op, x, y);
            let f: WideEval = std::rc::Rc::new(move |i: &[u32; 3]| {
                let (a, s) = (fx(i), fy(i));
                match op {
                    Add => a.wrapping_add(s),
                    Sub => a.wrapping_sub(s),
                    Mul => a.wrapping_mul(s),
                    And => a & s,
                    Or => a | s,
                    Xor => a ^ s,
                    Shl => a << (s & 63),
                    LShr => a >> (s & 63),
                    AShr => ((a as i64) >> (s & 63)) as u64,
                }
            });
            (v, format!("({} {:?} {})", nx, op, ny), f)
        }
        _ => {
            let (x, nx, fx) = oracle_wide(b, e, r, leaves, depth - 1);
            let (y, ny, fy) = oracle_wide(b, e, r, leaves, depth - 1);
            let k = r.below(70000) as u32;
            let kk = b.constant(e, Ty::I32, k as u64);
            let c = b.cmp(e, IntPred::Ult, leaves[0], kk);
            let v = b.core(e, Ty::I64, Op::Select(c, x, y));
            let f: WideEval = std::rc::Rc::new(move |i: &[u32; 3]| if i[0] < k { fx(i) } else { fy(i) });
            (v, format!("(u < {} ? {} : {})", k, nx, ny), f)
        }
    }
}

fn random_wides(seed: u64, programs: usize, per: usize, depth: u32) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for _ in 0..programs {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let four = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, four);
        let z = b.load(e, Space::Global, MemSize::B32, second, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let leaves = [u, z, lane];
        let mut cases = Vec::new();
        for _ in 0..per {
            let (v, name, f) = oracle_wide(&mut b, e, &mut r, &leaves, depth);
            let hi = b.core(e, Ty::I32, Op::UnpackHi(v));
            let (w, wname, g) = oracle_wide(&mut b, e, &mut r, &leaves, depth - 1);
            let preds = [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Slt];
            let pred = preds[r.below(4) as usize];
            let c = b.cmp(e, pred, v, w);
            cases.push((v, hi, c, name, f, wname, g, pred));
        }
        let mut samples: Vec<[u32; 2]> = Vec::new();
        for &uu in &[0u32, 1, 2, 3, 0x7fff, 0x8000, 0xfffe, 0xffff] {
            for _ in 0..3 {
                samples.push([uu, interesting(&mut r)]);
            }
        }
        for _ in 0..16 {
            samples.push([r.below(65536) as u32, r.next() as u32]);
        }
        addresses(&b, &two_words(), |a| {
            let uu = a.values.value(u, 0, None).0.form;
            let zz = a.values.value(z, 0, None).0.form;
            let (Some(&(ui, 1)), Some(&(zi, 1))) = (uu.terms.first(), zz.terms.first()) else {
                wrong.push(format!("leaves are not plain unknowns: {:?} {:?}", uu, zz));
                return;
            };
            for lane in [0usize, 7] {
                for (v, hi, c, name, f, wname, g, pred) in &cases {
                    let low = a.values.value(*v, lane, None).0.form;
                    let high = a.high(*v, lane);
                    let unpacked = a.values.value(*hi, lane, None).0.form;
                    let decided = a.bit(*c, lane, None).0;
                    let wide = a.wide_value(*v, e, lane);
                    let exact = wide.as_ref().filter(|w| w.words.is_empty() && w.terms.iter().all(|&(t, _)| t == ui || t == zi));
                    for s in &samples {
                        let input = [s[0], s[1], lane as u32];
                        let t = f(&input);
                        if !core_holds(a.unknowns(), &low, t as u32) {
                            wrong.push(format!("low of {} lane {} at {:x?}: {:?} cannot be {:#x}", name, lane, input, low, t as u32));
                        }
                        if !core_holds(a.unknowns(), &high, (t >> 32) as u32) {
                            wrong.push(format!("high of {} lane {} at {:x?}: {:?} cannot be {:#x}", name, lane, input, high, (t >> 32) as u32));
                        }
                        if !core_holds(a.unknowns(), &unpacked, (t >> 32) as u32) {
                            wrong.push(format!("unpacked high of {} lane {} at {:x?}: {:?} cannot be {:#x}", name, lane, input, unpacked, (t >> 32) as u32));
                        }
                        if let Some(k) = decided {
                            let (x, y) = (t, g(&input));
                            let truth = match pred {
                                IntPred::Eq => x == y,
                                IntPred::Ne => x != y,
                                IntPred::Ult => x < y,
                                _ => (x as i64) < (y as i64),
                            };
                            if k != truth {
                                wrong.push(format!("{} {:?} {} lane {} at {:x?}: decided {}", name, pred, wname, lane, input, k));
                            }
                        }
                        if let Some(w) = exact {
                            let value = w.terms.iter().fold(w.constant, |acc, &(t, c)| acc + c * if t == ui { s[0] as i128 } else { s[1] as i128 });
                            if value != t as i128 {
                                wrong.push(format!("wide value of {} lane {} at {:x?}: {:?} gives {:#x}, not {:#x}", name, lane, input, w, value, t));
                            }
                        }
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_wide_expressions_hold_every_value_and_decide_only_what_holds() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(11);
    let depth: u32 = std::env::var("ORACLE_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let wrong = random_wides(seed, n, 4, depth);
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

fn or_of_select_and_bit(sext: bool) -> (Build, ValueId, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let yes = b.constant(e, Ty::I1, 1);
    let u = b.load(e, Space::Global, MemSize::U16, table, yes);
    let four = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, four);
    let z = b.load(e, Space::Global, MemSize::B32, second, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let c = b.cmp(e, IntPred::Sge, u, w);
    let small = b.constant(e, Ty::I32, 4);
    let big = b.constant(e, Ty::I32, 0x8000_0005);
    let chosen = b.core(e, Ty::I32, Op::Select(c, small, big));
    let limit = b.constant(e, Ty::I32, 0xfffc_0000);
    let below = b.cmp(e, IntPred::Ult, z, limit);
    let cvt = if sext { Cvt::SExt } else { Cvt::ZExt };
    let bit = b.core(e, Ty::I32, Op::Convert(cvt, Ty::I32, below));
    let either = b.int(e, IntOp::Or, chosen, bit);
    let less = b.cmp(e, IntPred::Ult, either, u);
    (b, either, u, less)
}

#[test]
fn or_of_a_select_and_an_extended_bit_decides_only_what_holds() {
    let mut wrong = Vec::new();
    for sext in [false, true] {
        let (b, either, u, less) = or_of_select_and_bit(sext);
        addresses(&b, &two_words(), |a| {
            let x = a.values.value(either, 0, None).0.form;
            let y = a.values.value(u, 0, None).0.form;
            let info: Vec<_> = x.terms.iter().chain(&y.terms).map(|&(t, _)| (t, a.unknowns()[t as usize].range, a.unknowns()[t as usize].values.clone())).collect();
            let decided = a.bit(less, 0, None).0;
            if let Some((low, high)) = a.bounds(&x) {
                if low > high || !core_holds(a.unknowns(), &x, 4) || 4 < low || 4 > high {
                    wrong.push(format!("sext {}: bounds of {:?} over {:?} are {:?}, which leave out 4 (u = 0x8000, w = 5, z = ~0)", sext, x, info, (low, high)));
                }
            }
            if decided == Some(false) {
                wrong.push(format!("sext {}: ({:?}) < ({:?}) over {:?} decided false, but u = 0x8000, w = 5, z = ~0 gives 4 < 0x8000", sext, x, y, info));
            }
        });
    }
    assert!(wrong.is_empty(), "{:#?}", wrong);
}

type LoopEval = std::rc::Rc<dyn Fn(u64, u64, u32) -> u64>;

fn oracle_step(b: &mut Build, e: BlockId, r: &mut Random, ty: Ty, v: ValueId, i: ValueId, depth: u32) -> (ValueId, String, LoopEval) {
    use IntOp::*;
    let bits = ty.bits();
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let leaf = |b: &mut Build, r: &mut Random| -> (ValueId, String, LoopEval) {
        match r.below(4) {
            0 | 2 => (v, "v".into(), std::rc::Rc::new(|v: u64, _: u64, _: u32| v)),
            1 => {
                let x = if ty == Ty::I64 { b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, i)) } else { i };
                (x, "i".into(), std::rc::Rc::new(|_: u64, i: u64, _: u32| i))
            }
            _ => {
                let k = if r.below(2) == 0 { r.below(9) } else { wide_constant(r) & mask };
                (b.constant(e, ty, k), format!("{:#x}", k), std::rc::Rc::new(move |_: u64, _: u64, _: u32| k))
            }
        }
    };
    if depth == 0 {
        return leaf(b, r);
    }
    let ops = [Add, Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr];
    let op = ops[r.below(10) as usize];
    let (x, nx, fx) = if r.below(3) == 0 { leaf(b, r) } else { oracle_step(b, e, r, ty, v, i, depth - 1) };
    let (y, ny, fy) = if matches!(op, Shl | LShr | AShr) {
        let k = r.below(bits as u64 + 4);
        let f: LoopEval = std::rc::Rc::new(move |_: u64, _: u64, _: u32| k);
        (b.constant(e, ty, k), format!("{}", k), f)
    } else if r.below(2) == 0 {
        leaf(b, r)
    } else {
        oracle_step(b, e, r, ty, v, i, depth - 1)
    };
    let value = b.int(e, op, x, y);
    let f: LoopEval = std::rc::Rc::new(move |v: u64, i: u64, u: u32| {
        let (a, s) = (fx(v, i, u), fy(v, i, u));
        let out = match op {
            Add => a.wrapping_add(s),
            Sub => a.wrapping_sub(s),
            Mul => a.wrapping_mul(s),
            And => a & s,
            Or => a | s,
            Xor => a ^ s,
            Shl => a << (s & (bits as u64 - 1)),
            LShr => (a & mask) >> (s & (bits as u64 - 1)),
            AShr => (((a << (64 - bits)) as i64 >> (64 - bits)) >> (s & (bits as u64 - 1))) as u64,
        };
        out & mask
    });
    (value, format!("({} {:?} {})", nx, op, ny), f)
}

fn random_loops(seed: u64, programs: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for _ in 0..programs {
        let ty = if r.below(3) == 0 && std::env::var("ORACLE_TY").map_or(true, |t| t == "64") { Ty::I64 } else { Ty::I32 };
        let mask = if ty == Ty::I64 { u64::MAX } else { u32::MAX as u64 };
        let bounded = r.below(2) == 0;
        let trips = 1 + r.below(6) as u32;
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U8, table, yes);
        let from_u = r.below(2) == 0;
        let start_k = wide_constant(&mut r) & mask;
        let start = if from_u {
            let kk = b.constant(e, Ty::I32, start_k & 0xffff_ffff);
            let s = b.int(e, IntOp::Add, u, kk);
            if ty == Ty::I64 { b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, s)) } else { s }
        } else {
            b.constant(e, ty, start_k)
        };
        let first = move |uu: u32| if from_u { (uu as u64).wrapping_add(start_k & 0xffff_ffff) & 0xffff_ffff } else { start_k };
        let limit = if bounded { b.constant(e, Ty::I32, trips as u64) } else { loaded_word(&mut b, &k, e, 4, MemSize::B32) };
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, ty, Ty::I32]);
        let (exit, x) = b.block(&[Ty::I1, ty]);
        b.br(e, body, vec![k.exec, start, zero]);
        let (next, name, f) = oracle_step(&mut b, body, &mut r, ty, p[1], p[2], 2);
        let one = b.constant(body, Ty::I32, 1);
        let ni = b.int(body, IntOp::Add, p[2], one);
        let again = b.cmp(body, IntPred::Ult, ni, limit);
        b.cond_br(body, again, (body, vec![p[0], next, ni]), (exit, vec![p[0], next]));
        let runs = if bounded { trips } else { 7 };
        let described = if bounded { trips.to_string() } else { "unknown".into() };
        addresses(&b, &two_words(), |a| {
            let inside = a.values.value(p[1], 0, None).0.form;
            let after = a.values.value(x[1], 0, None).0.form;
            for uu in [0u32, 1, 2, 77, 128, 254, 255] {
                let mut v = first(uu);
                for t in 0..runs {
                    if !core_holds(a.unknowns(), &inside, v as u32) {
                        wrong.push(format!("{:?} v' = {} from {:#x} (u = {}), {} trips: iteration {} value {:#x} not in {:?}", ty, name, first(uu), uu, described, t, v, inside));
                    }
                    v = f(v, t as u64, uu);
                    if (bounded && t + 1 == runs || !bounded) && !core_holds(a.unknowns(), &after, v as u32) {
                        wrong.push(format!("{:?} v' = {} from {:#x} (u = {}), {} trips: exit after {} value {:#x} not in {:?}", ty, name, first(uu), uu, described, t + 1, v, after));
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_loops_hold_every_value_an_iteration_gives() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(13);
    let wrong = random_loops(seed, n);
    if std::env::var("ORACLE_ALL").is_ok() {
        for w in &wrong {
            eprintln!("{}", w);
        }
    }
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

fn random_branches(seed: u64, programs: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    let preds = [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Ugt, IntPred::Ule, IntPred::Uge, IntPred::Slt, IntPred::Sgt, IntPred::Sle, IntPred::Sge];
    for _ in 0..programs {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let four = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, four);
        let z = b.load(e, Space::Global, MemSize::B32, second, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let leaves = [u, z, w, lane];
        let on_u = r.below(2) == 0;
        let s = if on_u { u } else { z };
        let pred = preds[r.below(10) as usize];
        let bound = if on_u { [0u32, 1, 7, 0x8000, 0xffff, 100][r.below(6) as usize] } else { [0u32, 1, 7, 0x8000_0000, 0xffff_ffff, 0x7fff_ffff, 100][r.below(7) as usize] };
        let kb = b.constant(e, Ty::I32, bound as u64);
        let c = b.cmp(e, pred, s, kb);
        let direct = r.below(2) == 0;
        let (merge, m) = b.block(&[Ty::I1, Ty::I32]);
        let (arm_yes, yv, yname, fy) = if direct {
            let (v, n, f) = oracle_word(&mut b, e, &mut r, &leaves, 2);
            (e, v, n, f)
        } else {
            let (blk, _) = b.block(&[Ty::I1]);
            let (v, n, f) = oracle_word(&mut b, blk, &mut r, &leaves, 2);
            b.br(blk, merge, vec![k.exec, v]);
            (blk, v, n, f)
        };
        let (arm_no, _) = b.block(&[Ty::I1]);
        let (nv, nname, fnn) = oracle_word(&mut b, arm_no, &mut r, &leaves, 2);
        b.br(arm_no, merge, vec![k.exec, nv]);
        let yes_target = if direct { (merge, vec![k.exec, yv]) } else { (arm_yes, vec![k.exec]) };
        b.cond_br(e, c, yes_target, (arm_no, vec![k.exec]));
        let (after, an, fa) = oracle_word(&mut b, merge, &mut r, &[m[1], z, w, lane], 1);
        let cmp_pred = preds[r.below(10) as usize];
        let probe = b.cmp(merge, cmp_pred, m[1], u);
        let mut samples: Vec<[u32; 3]> = Vec::new();
        for _ in 0..30 {
            let mut uu = [0u32, 1, 7, 0x8000, 0xffff, 100, 99, 101][r.below(8) as usize];
            let mut zz = interesting(&mut r);
            if r.below(3) == 0 {
                if on_u {
                    uu = bound & 0xffff;
                } else {
                    zz = bound;
                }
            }
            if r.below(4) == 0 {
                uu = r.below(65536) as u32;
            }
            samples.push([uu, zz, interesting(&mut r)]);
        }
        addresses(&b, &two_words(), |a| {
            for lane in [0usize, 9] {
                let joined = a.values.value(m[1], lane, None).0.form;
                let later = a.values.value(after, lane, None).0.form;
                let arm_y = a.values.value(yv, lane, None).0.form;
                let arm_n = a.values.value(nv, lane, None).0.form;
                let decided = a.bit(probe, lane, None).0;
                let _ = arm_yes;
                for sm in &samples {
                    let input = [sm[0], sm[1], sm[2], lane as u32];
                    let taken = super::compare(pred, if on_u { sm[0] } else { sm[1] }, bound);
                    let t = if taken { fy(&input) } else { fnn(&input) };
                    let what = format!("if {} {:?} {:#x} then {} else {} (direct {}) lane {} at {:x?}", if on_u { "u" } else { "z" }, pred, bound, yname, nname, direct, lane, input);
                    if !core_holds(a.unknowns(), &joined, t) {
                        wrong.push(format!("{}: joined {:?} cannot be {:#x}", what, joined, t));
                    }
                    let ta = fa(&[t, sm[1], sm[2], lane as u32]);
                    if !core_holds(a.unknowns(), &later, ta) {
                        wrong.push(format!("{}: {} after the join {:?} cannot be {:#x}", what, an, later, ta));
                    }
                    if taken && !direct && !core_holds(a.unknowns(), &arm_y, t) {
                        wrong.push(format!("{}: yes arm {:?} cannot be {:#x}", what, arm_y, t));
                    }
                    if !taken && !core_holds(a.unknowns(), &arm_n, t) {
                        wrong.push(format!("{}: no arm {:?} cannot be {:#x}", what, arm_n, t));
                    }
                    if let Some(d) = decided {
                        if d != super::compare(cmp_pred, t, sm[0]) {
                            wrong.push(format!("{}: joined {:?} u decided {}", what, cmp_pred, d));
                        }
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_branches_hold_every_value_the_taken_edge_brings() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(17);
    let wrong = random_branches(seed, n);
    if std::env::var("ORACLE_ALL").is_ok() {
        for w in &wrong {
            eprintln!("{}", w);
        }
    }
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

#[derive(Clone, Copy, Debug)]
enum Data {
    Constant(u32),
    U(u32),
    Lane,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum OracleWhen {
    Always,
    Never,
    Below(u32),
    Exec,
}

fn random_scratch(seed: u64, programs: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for _ in 0..programs {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let no = b.constant(e, Ty::I1, 0);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let four = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, four);
        let z = b.load(e, Space::Global, MemSize::B32, second, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let symbolic = r.below(2) == 0;
        let base = if symbolic {
            let k64 = b.constant(e, Ty::I32, 64);
            b.int(e, IntOp::Mul, u, k64)
        } else {
            b.constant(e, Ty::I32, 0x100)
        };
        let diamond = r.below(2) == 0;
        let split = b.constant(e, Ty::I32, 0x8000_0000);
        let branch = b.cmp(e, IntPred::Ult, z, split);
        let (then, _) = b.block(&[Ty::I1]);
        let (other, _) = b.block(&[Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        let count = 1 + r.below(5) as usize;
        let mut plan = Vec::new();
        for _ in 0..count {
            let place = if diamond { r.below(4) as usize } else { 0 };
            let off = [0u32, 4, 8, 2, 1, 6, 0xffff_fffc][r.below(7) as usize];
            let size = [MemSize::B32, MemSize::B32, MemSize::B32, MemSize::U8, MemSize::U16][r.below(5) as usize];
            let data = match r.below(3) {
                0 => Data::Constant(interesting(&mut r)),
                1 => Data::U(interesting(&mut r)),
                _ => Data::Lane,
            };
            let when = match r.below(5) {
                0 | 1 => OracleWhen::Always,
                2 => OracleWhen::Never,
                3 => OracleWhen::Below([0x10u32, 0x8000_0000, 0xffff_ffff][r.below(3) as usize]),
                _ => OracleWhen::Exec,
            };
            plan.push((place, off, size, data, when));
        }
        plan.sort_by_key(|p| p.0);
        let load_off = [0u32, 4, 8, 0xffff_fffc, 2][r.below(5) as usize];
        let block_of = |place: usize| if !diamond { e } else { [e, then, other, join][place] };
        for &(place, off, size, data, when) in &plan {
            let at = block_of(place);
            let ko = b.constant(at, Ty::I32, off as u64);
            let address = b.int(at, IntOp::Add, base, ko);
            let value = match data {
                Data::Constant(c) => b.constant(at, Ty::I32, c as u64),
                Data::U(c) => {
                    let kc = b.constant(at, Ty::I32, c as u64);
                    b.int(at, IntOp::Add, u, kc)
                }
                Data::Lane => lane,
            };
            let predicate = match when {
                OracleWhen::Always => yes,
                OracleWhen::Never => no,
                OracleWhen::Below(limit) => {
                    let kl = b.constant(at, Ty::I32, limit as u64);
                    b.cmp(at, IntPred::Ult, z, kl)
                }
                OracleWhen::Exec => k.exec,
            };
            b.store(at, Space::Scratch, size, address, value, predicate);
        }
        if diamond {
            b.cond_br(e, branch, (then, vec![k.exec]), (other, vec![k.exec]));
            b.br(then, join, vec![k.exec]);
            b.br(other, join, vec![k.exec]);
        }
        let last = if diamond { join } else { e };
        let kl = b.constant(last, Ty::I32, load_off as u64);
        let address = b.int(last, IntOp::Add, base, kl);
        let loaded = b.load(last, Space::Scratch, MemSize::B32, address, k.exec);
        let samples: Vec<(u32, u32)> = (0..24).map(|_| ([0u32, 1, 2, 0x7fff, 0xffff][r.below(5) as usize], interesting(&mut r))).collect();
        addresses(&b, &two_words(), |a| {
            for l in [0usize, 3] {
                let form = a.values.value(loaded, l, None).0.form;
                for &(uu, zz) in &samples {
                    let mut memory: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
                    for &(place, off, size, data, when) in &plan {
                        let taken = zz < 0x8000_0000;
                        if diamond && ((place == 1 && !taken) || (place == 2 && taken)) {
                            continue;
                        }
                        let runs = match when {
                            OracleWhen::Always | OracleWhen::Exec => true,
                            OracleWhen::Never => false,
                            OracleWhen::Below(limit) => zz < limit,
                        };
                        if !runs {
                            continue;
                        }
                        let value = match data {
                            Data::Constant(c) => c,
                            Data::U(c) => uu.wrapping_add(c),
                            Data::Lane => l as u32,
                        };
                        for i in 0..size.bytes() {
                            memory.insert(off.wrapping_add(i), (value >> (8 * i)) as u8);
                        }
                    }
                    let bytes: Option<Vec<u8>> = (0..4).map(|i| memory.get(&load_off.wrapping_add(i)).copied()).collect();
                    let Some(bytes) = bytes else { continue };
                    let t = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    if !core_holds(a.unknowns(), &form, t) {
                        wrong.push(format!("symbolic {} diamond {} stores {:x?} load at {:#x} lane {} u {:#x} z {:#x}: {:?} cannot be {:#x}", symbolic, diamond, plan, load_off, l, uu, zz, form, t));
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_private_stores_read_back_every_word_they_leave() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(19);
    let wrong = random_scratch(seed, n);
    if std::env::var("ORACLE_ALL").is_ok() {
        for w in &wrong {
            eprintln!("{}", w);
        }
    }
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

fn random_scratch_loops(seed: u64, programs: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for _ in 0..programs {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let symbolic = r.below(2) == 0;
        let base = if symbolic {
            let k64 = b.constant(e, Ty::I32, 64);
            b.int(e, IntOp::Mul, u, k64)
        } else {
            b.constant(e, Ty::I32, 0x100)
        };
        let offs = [0u32, 4, 8, 2];
        let init: Vec<(u32, u32)> = offs.iter().map(|&o| (o, interesting(&mut r))).collect();
        for &(o, v) in &init {
            let ko = b.constant(e, Ty::I32, o as u64);
            let address = b.int(e, IntOp::Add, base, ko);
            let kv = b.constant(e, Ty::I32, v as u64);
            if o != 2 {
                b.store(e, Space::Scratch, MemSize::B32, address, kv, yes);
            }
        }
        let trips = 1 + r.below(5) as u32;
        let bounded = r.below(2) == 0;
        let limit = if bounded { b.constant(e, Ty::I32, trips as u64) } else { loaded_word(&mut b, &k, e, 4, MemSize::B32) };
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero]);
        let load_off = offs[r.below(3) as usize];
        let kl = b.constant(body, Ty::I32, load_off as u64);
        let at = b.int(body, IntOp::Add, base, kl);
        let inside = b.load(body, Space::Scratch, MemSize::B32, at, k.exec);
        let store_off = offs[r.below(4) as usize];
        let size = [MemSize::B32, MemSize::B32, MemSize::U8, MemSize::U16][r.below(4) as usize];
        let ks = b.constant(body, Ty::I32, store_off as u64);
        let sat = b.int(body, IntOp::Add, base, ks);
        let mode = r.below(3);
        let scale = interesting(&mut r);
        let kscale = b.constant(body, Ty::I32, scale as u64);
        let data = match mode {
            0 => b.int(body, IntOp::Mul, p[1], kscale),
            1 => b.int(body, IntOp::Add, inside, kscale),
            _ => kscale,
        };
        let guarded = r.below(2) == 0;
        let predicate = if guarded {
            let two = b.constant(body, Ty::I32, 2);
            b.cmp(body, IntPred::Ult, p[1], two)
        } else {
            yes
        };
        b.store(body, Space::Scratch, size, sat, data, predicate);
        let one = b.constant(body, Ty::I32, 1);
        let ni = b.int(body, IntOp::Add, p[1], one);
        let again = b.cmp(body, IntPred::Ult, ni, limit);
        b.cond_br(body, again, (body, vec![p[0], ni]), (exit, vec![p[0]]));
        let kx = b.constant(exit, Ty::I32, load_off as u64);
        let xat = b.int(exit, IntOp::Add, base, kx);
        let after = b.load(exit, Space::Scratch, MemSize::B32, xat, k.exec);
        let runs = if bounded { trips } else { 6 };
        addresses(&b, &two_words(), |a| {
            let fin = a.values.value(inside, 0, None).0.form;
            let fout = a.values.value(after, 0, None).0.form;
            let mut memory: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
            for &(o, v) in &init {
                if o != 2 {
                    for i in 0..4 {
                        memory.insert(o + i, (v >> (8 * i)) as u8);
                    }
                }
            }
            let read = |m: &std::collections::HashMap<u32, u8>, o: u32| u32::from_le_bytes([0, 1, 2, 3].map(|i| m[&(o + i)]));
            for t in 0..runs {
                let x = read(&memory, load_off);
                let what = format!("symbolic {} bounded {} trips {} load {} store {} {:?} mode {} scale {:#x} guarded {}", symbolic, bounded, runs, load_off, store_off, size, mode, scale, guarded);
                if !core_holds(a.unknowns(), &fin, x) {
                    wrong.push(format!("{}: iteration {} reload {:?} cannot be {:#x}", what, t, fin, x));
                }
                let value = match mode {
                    0 => t.wrapping_mul(scale),
                    1 => x.wrapping_add(scale),
                    _ => scale,
                };
                if !guarded || t < 2 {
                    for i in 0..size.bytes() {
                        memory.insert(store_off + i, (value >> (8 * i)) as u8);
                    }
                }
                if bounded && t + 1 == runs || !bounded {
                    let y = read(&memory, load_off);
                    if !core_holds(a.unknowns(), &fout, y) {
                        wrong.push(format!("{}: after {} trips {:?} cannot be {:#x}", what, t + 1, fout, y));
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_private_words_a_loop_stores_read_back_every_value() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(23);
    let wrong = random_scratch_loops(seed, n);
    if std::env::var("ORACLE_ALL").is_ok() {
        for w in &wrong {
            eprintln!("{}", w);
        }
    }
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}

type OracleBallot = std::rc::Rc<dyn Fn(u32, &[u32; 32], usize, u32) -> u32>;

fn ballot_word(b: &mut Build, e: BlockId, r: &mut Random, k: &Kernel, leaves: &[ValueId; 3], depth: u32) -> (ValueId, String, OracleBallot) {
    let preds = [IntPred::Ult, IntPred::Ugt, IntPred::Eq, IntPred::Ne, IntPred::Slt];
    if depth == 0 || r.below(3) == 0 {
        if r.below(5) == 0 {
            let c = interesting(r);
            return (b.constant(e, Ty::I32, c as u64), format!("{:#x}", c), std::rc::Rc::new(move |_: u32, _: &[u32; 32], _: usize, _: u32| c));
        }
        let pred = preds[r.below(5) as usize];
        let which = r.below(3) as usize;
        let bound = match which {
            0 => r.below(33) as u32,
            1 => [0u32, 1, 100, 0x8000_0000][r.below(4) as usize],
            _ => [0u32, 3, 0x8000, 0xffff][r.below(4) as usize],
        };
        let kb = b.constant(e, Ty::I32, bound as u64);
        let c = b.cmp(e, pred, leaves[which], kb);
        let masked = b.int(e, IntOp::And, c, k.exec);
        let word = b.wave(e, WaveOp::Ballot { high: false }, vec![masked]);
        let f: OracleBallot = std::rc::Rc::new(move |u: u32, w: &[u32; 32], _: usize, valid: u32| {
            (0..32).filter(|&l| valid >> l & 1 != 0).fold(0u32, |m, l| {
                let x = [l as u32, w[l], u][which];
                m | (super::compare(pred, x, bound) as u32) << l
            })
        });
        return (word, format!("ballot({} {:?} {:#x})", ["lane", "w", "u"][which], pred, bound), f);
    }
    let (x, nx, fx) = ballot_word(b, e, r, k, leaves, depth - 1);
    let (y, ny, fy) = ballot_word(b, e, r, k, leaves, depth - 1);
    match r.below(4) {
        0 => {
            let v = b.int(e, IntOp::And, x, y);
            (v, format!("({} & {})", nx, ny), std::rc::Rc::new(move |u: u32, w: &[u32; 32], l: usize, valid: u32| fx(u, w, l, valid) & fy(u, w, l, valid)))
        }
        1 => {
            let v = b.int(e, IntOp::Or, x, y);
            (v, format!("({} | {})", nx, ny), std::rc::Rc::new(move |u: u32, w: &[u32; 32], l: usize, valid: u32| fx(u, w, l, valid) | fy(u, w, l, valid)))
        }
        2 => {
            let v = b.int(e, IntOp::Xor, x, y);
            (v, format!("({} ^ {})", nx, ny), std::rc::Rc::new(move |u: u32, w: &[u32; 32], l: usize, valid: u32| fx(u, w, l, valid) ^ fy(u, w, l, valid)))
        }
        _ => {
            let kb = [0u32, 0x8000][r.below(2) as usize];
            let kk = b.constant(e, Ty::I32, kb as u64);
            let c = b.cmp(e, IntPred::Ult, leaves[2], kk);
            let v = b.core(e, Ty::I32, Op::Select(c, x, y));
            (v, format!("(u < {:#x} ? {} : {})", kb, nx, ny), std::rc::Rc::new(move |u: u32, w: &[u32; 32], l: usize, valid: u32| if u < kb { fx(u, w, l, valid) } else { fy(u, w, l, valid) }))
        }
    }
}

fn random_ballots(seed: u64, programs: usize) -> Vec<String> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for _ in 0..programs {
        let size = [32u32, 20, 1, 31][r.below(4) as usize];
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let leaves = [lane, w, u];
        let (word, name, f) = ballot_word(&mut b, e, &mut r, &k, &leaves, 2);
        let by_lane = r.below(2) == 0;
        let fixed = r.below(32) as u32;
        let amount = if by_lane { lane } else { b.constant(e, Ty::I32, fixed as u64) };
        let shifted = b.int(e, IntOp::LShr, word, amount);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let valid: u32 = if size == 32 { u32::MAX } else { (1u32 << size) - 1 };
        let samples: Vec<(u32, [u32; 32])> = (0..12)
            .map(|_| ([0u32, 3, 0x7fff, 0x8000, 0xffff][r.below(5) as usize], std::array::from_fn(|_| [0u32, 1, 100, 0x8000_0000, 99][r.below(5) as usize])))
            .collect();
        addresses(&b, &environment(size, &[(0, 1, 0x1000), (8, 2, 0x2000)]), |a| {
            for l in 0..size as usize {
                let decided = a.bit(bit, l, None).0;
                let form = a.values.value(word, l, None).0.form;
                for (uu, ws) in &samples {
                    let t = f(*uu, ws, l, valid);
                    let s = if by_lane { l as u32 } else { fixed };
                    let truth = t >> s & 1 != 0;
                    if decided.is_some_and(|d| d != truth) {
                        wrong.push(format!("{} >> {} in a wave of {} lane {} u {:#x}: decided {:?}, truth {}", name, if by_lane { "lane".to_string() } else { fixed.to_string() }, size, l, uu, decided, truth));
                    }
                    if !core_holds(a.unknowns(), &form, t) {
                        wrong.push(format!("{} in a wave of {} lane {} u {:#x}: {:?} cannot be {:#x}", name, size, l, uu, form, t));
                    }
                }
            }
        });
    }
    wrong.sort();
    wrong.dedup();
    wrong
}

#[test]
fn random_ballot_bits_decide_only_what_holds() {
    let n: usize = std::env::var("ORACLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = std::env::var("ORACLE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(29);
    let wrong = random_ballots(seed, n);
    if std::env::var("ORACLE_ALL").is_ok() {
        for w in &wrong {
            eprintln!("{}", w);
        }
    }
    assert!(wrong.is_empty(), "{} wrong, first {:#?}", wrong.len(), &wrong[..wrong.len().min(12)]);
}
