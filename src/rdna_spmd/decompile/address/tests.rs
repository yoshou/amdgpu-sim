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
    b.entry.workgroup_id_x = true;
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
        let mut build = |r: &mut Random| {
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
