use super::super::address::compare;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

#[derive(Default)]
pub(super) struct LaneValues(HashMap<ValueId, Option<[u32; 32]>>);

impl LaneValues {
    pub(super) fn of(&mut self, f: &Func, facts: &Facts, v: ValueId, depth: u32) -> Option<[u32; 32]> {
        if let Some(&r) = self.0.get(&v) {
            return r;
        }
        if depth > 64 {
            return None;
        }
        let r = self.compute(f, facts, v, depth);
        self.0.insert(v, r);
        r
    }

    fn compute(&mut self, f: &Func, facts: &Facts, v: ValueId, depth: u32) -> Option<[u32; 32]> {
        let bits = match f.types[v.0] {
            Ty::I1 => 1,
            Ty::I32 => u32::MAX,
            _ => return None,
        };
        let mut out = [0u32; 32];
        match facts.op(f, v)? {
            Op::Const(_, k) => out = [k as u32; 32],
            Op::Env(Env::LaneId) => out = std::array::from_fn(|l| l as u32),
            Op::Int(k, a, b) => {
                let a = self.of(f, facts, a, depth + 1)?;
                let b = self.of(f, facts, b, depth + 1)?;
                for l in 0..32 {
                    let (x, y) = (a[l], b[l]);
                    out[l] = match k {
                        IntOp::Add => x.wrapping_add(y),
                        IntOp::Sub => x.wrapping_sub(y),
                        IntOp::Mul => x.wrapping_mul(y),
                        IntOp::And => x & y,
                        IntOp::Or => x | y,
                        IntOp::Xor => x ^ y,
                        IntOp::Shl | IntOp::LShr | IntOp::AShr if y >= 32 => return None,
                        IntOp::Shl => x << y,
                        IntOp::LShr => x >> y,
                        IntOp::AShr => ((x as i32) >> y) as u32,
                    };
                }
            }
            Op::Cmp(p, a, b) if f.types[a.0] == Ty::I32 => {
                let a = self.of(f, facts, a, depth + 1)?;
                let b = self.of(f, facts, b, depth + 1)?;
                out = std::array::from_fn(|l| compare(p, a[l], b[l]) as u32);
            }
            Op::Select(c, a, b) => {
                let c = self.of(f, facts, c, depth + 1)?;
                let a = self.of(f, facts, a, depth + 1)?;
                let b = self.of(f, facts, b, depth + 1)?;
                out = std::array::from_fn(|l| if c[l] & 1 == 1 { a[l] } else { b[l] });
            }
            _ => return None,
        }
        Some(out.map(|x| x & bits))
    }

    pub(super) fn known_bits(&mut self, f: &Func, facts: &Facts, v: ValueId, depth: u32) -> [(u32, u32); 32] {
        if let Some(values) = self.of(f, facts, v, 0) {
            return std::array::from_fn(|l| (u32::MAX, values[l]));
        }
        if depth > 16 || f.types[v.0] != Ty::I32 {
            return [(0, 0); 32];
        }
        let constant = |x: ValueId| facts.constant(f, x).map(|k| k as u32);
        match facts.op(f, v) {
            Some(Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b)) => {
                let (x, y) = (self.known_bits(f, facts, a, depth + 1), self.known_bits(f, facts, b, depth + 1));
                std::array::from_fn(|l| {
                    let ((mx, vx), (my, vy)) = (x[l], y[l]);
                    let mask = match k {
                        IntOp::And => (mx & my) | (mx & !vx) | (my & !vy),
                        IntOp::Or => (mx & my) | (mx & vx) | (my & vy),
                        _ => mx & my,
                    };
                    let value = match k {
                        IntOp::And => vx & vy,
                        IntOp::Or => vx | vy,
                        _ => vx ^ vy,
                    };
                    (mask, value & mask)
                })
            }
            Some(Op::Int(k @ (IntOp::Add | IntOp::Sub), a, b)) => {
                let (x, y) = (self.known_bits(f, facts, a, depth + 1), self.known_bits(f, facts, b, depth + 1));
                std::array::from_fn(|l| {
                    let ((mx, vx), (my, vy)) = (x[l], y[l]);
                    let (zero_x, one_x) = (mx & !vx, vx);
                    let (zero_y, one_y, carry) = match k {
                        IntOp::Add => (my & !vy, vy, 0u32),
                        _ => (vy, my & !vy, 1u32),
                    };
                    let sum_zero = (!zero_x).wrapping_add(!zero_y).wrapping_add(carry);
                    let sum_one = one_x.wrapping_add(one_y).wrapping_add(carry);
                    let carry_zero = !(sum_zero ^ zero_x ^ zero_y);
                    let carry_one = sum_one ^ one_x ^ one_y;
                    let known = (zero_x | one_x) & (zero_y | one_y) & (carry_zero | carry_one);
                    (known, sum_one & known)
                })
            }
            Some(Op::Int(IntOp::Shl, a, s)) if constant(s).is_some() => {
                let k = constant(s).unwrap() & 31;
                let x = self.known_bits(f, facts, a, depth + 1);
                std::array::from_fn(|l| ((x[l].0 << k) | ((1u32 << k) - 1), x[l].1 << k))
            }
            Some(Op::Int(IntOp::Mul, a, s)) if constant(s).is_some_and(|k| k.is_power_of_two()) => {
                let k = constant(s).unwrap().trailing_zeros();
                let x = self.known_bits(f, facts, a, depth + 1);
                std::array::from_fn(|l| ((x[l].0 << k) | ((1u32 << k) - 1), x[l].1 << k))
            }
            Some(Op::Int(IntOp::LShr, a, s)) if constant(s).is_some() => {
                let k = constant(s).unwrap() & 31;
                let x = self.known_bits(f, facts, a, depth + 1);
                std::array::from_fn(|l| ((x[l].0 >> k) | !(u32::MAX >> k), x[l].1 >> k))
            }
            _ => [(0, 0); 32],
        }
    }
}
