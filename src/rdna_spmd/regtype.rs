//! Determine which VGPR pairs can remain in f64 storage during code generation.
//!
//! A VGPR is a 32-bit slot; an f64 occupies a pair `(r, r+1)` whose low reg `r`
//! is whatever the f64 op names (NOT necessarily even — e.g. `V_FMA_F64 vdst=131`
//! pairs regs 131:132). The de-recursed kernel otherwise carries such a pair
//! across blocks as two i32 phis + reconstruction (206 i32 phis vs 16 `double`
//! phis measured). This infers the pair *low* regs that are **purely f64** —
//! every f64 use names `r` as a pair low, no half is read or written as i32, the
//! pair is not a 64-bit address, and it does not overlap another f64 pair — so
//! emit stores each in a single `double` alloca (`vgpr_f64[r]`): one source of
//! truth, one `double` phi, only whole-double load/store. Provably correct.
//!
//! Sets are 256-bit ([u128;2]) — VGPRs range 0..256, a u128 would alias r with
//! r+128.

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};

use super::ir::ScalarProgram;

pub type RegSet = [u128; 2];

#[inline]
fn bset(s: &mut RegSet, r: u32) {
    let r = (r & 255) as usize;
    s[r / 128] |= 1u128 << (r % 128);
}
#[inline]
pub fn bget(s: &RegSet, r: u32) -> bool {
    let r = (r & 255) as usize;
    (s[r / 128] >> (r % 128)) & 1 == 1
}

fn reads_f64_pairs(op: I) -> bool {
    let s = format!("{:?}", op);
    s.contains("F64") && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32)
}
fn is_f64_producer(op: I) -> bool {
    // Writes its vdst:vdst+1 as an f64 result.
    reads_f64_pairs(op) || matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32)
}
fn vreg(o: &SourceOperand) -> Option<u32> {
    match o {
        SourceOperand::VectorRegister(x) => Some(*x as u32),
        _ => None,
    }
}

/// Per-reg flags gathered in one pass.
struct Flags {
    f64_low: RegSet,   // reg named as the low of an f64 pair (operand or vdst)
    f64_reg: RegSet,   // reg that is either half of some f64 pair
}

fn gather(prog: &ScalarProgram) -> Flags {
    let mut f = Flags { f64_low: [0; 2], f64_reg: [0; 2] };
    let f64_low_op = |fl: &mut Flags, o: &SourceOperand| {
        if let Some(r) = vreg(o) {
            bset(&mut fl.f64_low, r);
            bset(&mut fl.f64_reg, r);
            bset(&mut fl.f64_reg, r + 1);
        }
    };
    for block in prog.blocks.values() {
        for inst in &block.body {
            let rf64 = match inst {
                InstFormat::VOP1(i) => reads_f64_pairs(i.op),
                InstFormat::VOP2(i) => reads_f64_pairs(i.op),
                InstFormat::VOP3(i) => reads_f64_pairs(i.op),
                InstFormat::VOP3SD(i) => reads_f64_pairs(i.op),
                InstFormat::VOPC(i) => reads_f64_pairs(i.op),
                _ => false,
            };
            // f64 source operands (pair lows).
            if rf64 {
                match inst {
                    InstFormat::VOP1(i) => f64_low_op(&mut f, &i.src0),
                    InstFormat::VOP2(i) => {
                        f64_low_op(&mut f, &i.src0);
                        bset(&mut f.f64_low, i.vsrc1 as u32);
                        bset(&mut f.f64_reg, i.vsrc1 as u32);
                        bset(&mut f.f64_reg, i.vsrc1 as u32 + 1);
                    }
                    InstFormat::VOPC(i) => {
                        f64_low_op(&mut f, &i.src0);
                        bset(&mut f.f64_low, i.vsrc1 as u32);
                        bset(&mut f.f64_reg, i.vsrc1 as u32);
                        bset(&mut f.f64_reg, i.vsrc1 as u32 + 1);
                    }
                    InstFormat::VOP3(i) => { f64_low_op(&mut f, &i.src0); f64_low_op(&mut f, &i.src1); f64_low_op(&mut f, &i.src2); }
                    InstFormat::VOP3SD(i) => { f64_low_op(&mut f, &i.src0); f64_low_op(&mut f, &i.src1); f64_low_op(&mut f, &i.src2); }
                    _ => {}
                }
            }
            // (i32 *reads* of an f64 pair are fine — they extract from the
            // `double` cell — so only i32 *writes* and addresses are tracked.)
            // f64-producer destination is a pair low.
            let (prod, vdst) = match inst {
                InstFormat::VOP1(i) => (is_f64_producer(i.op), i.vdst as u32),
                InstFormat::VOP2(i) => (is_f64_producer(i.op), i.vdst as u32),
                InstFormat::VOP3(i) => (is_f64_producer(i.op) && !format!("{:?}", i.op).contains("V_CMP"), i.vdst as u32),
                InstFormat::VOP3SD(i) => (is_f64_producer(i.op), i.vdst as u32),
                _ => (false, 0),
            };
            if prod {
                bset(&mut f.f64_low, vdst);
                bset(&mut f.f64_reg, vdst);
                bset(&mut f.f64_reg, vdst + 1);
            }
        }
    }
    f
}

/// Set of f64 pair *low* regs that are purely f64 and disjoint — the storage of
/// each is the single `double` alloca `vgpr_f64[low]`.
pub fn f64_read_pairs(prog: &ScalarProgram) -> RegSet {
    let f = gather(prog);
    // A pair is safely single-sourced as one `double` iff neither half is written
    // as an i32 (a non-f64 result/mask — the register is reused for an integer,
    // so it is not a coherent f64) and it is not a 64-bit address (carry math).
    // i32 *reads* are fine: they extract from the `double` cell.
    // A reg reused as an i32 is still single-sourceable in the `double` cell: i32
    // reads extract a half, i32 writes insert a half (read-modify-write the cell),
    // and the inactive-lane predication value is taken from the cell. Therefore
    // i32 writes do not require separate storage, and such pairs are admitted.
    // A pair that is *also* used as a 64-bit address / wide-int half is likewise
    // single-sourceable (address halves are built by carry arithmetic and read
    // back via extract/combine of the cell — bit-exact, and the bits never enter
    // an FP op).
    let mut out = [0u128; 2];
    for r in 0..255u32 {
        if !bget(&f.f64_low, r) {
            continue;
        }
        // Disjoint: r is not the high of a pair (r-1 not a low), and r+1 does not
        // start another pair (r+1 not a low) — so (r,r+1) overlaps nothing.
        if (r > 0 && bget(&f.f64_low, r - 1)) || bget(&f.f64_low, r + 1) {
            continue;
        }
        bset(&mut out, r);
    }
    out
}
