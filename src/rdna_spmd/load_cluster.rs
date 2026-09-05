//! Select consecutive global loads for record loading and transposition.
//!
//! The result describes the instruction prefix and its memory span. Selection
//! uses the existing divergence and scratch-frame facts at that instruction;
//! it neither changes those facts nor emits LLVM instructions.

use std::collections::HashMap;

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, VGLOBAL};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct VgCluster {
    pub(super) len: usize, // number of member instructions (≥ 3)
    pub(super) vaddr: u32, // shared per-lane pointer pair
    pub(super) lo: i64,    // lowest sign-extended ioffset (span start, bytes)
    pub(super) span: u32,  // span length in f64 fields
}

pub(super) fn analyze(
    body: &[InstFormat],
    width: u32,
    divergent: [u128; 2],
    frames: &HashMap<u32, u32>,
) -> Option<VgCluster> {
    // Transpose tiles require power-of-two packet widths.
    if !width.is_power_of_two() {
        return None;
    }
    vg_cluster(body).filter(|c| {
        let uniform = |r: u32| (divergent[(r >> 7) as usize] >> (r & 127)) & 1 == 0;
        // Uniform addresses use scalar-load + broadcast. Scratch-frame
        // addresses use the existing coalesced/register-resident paths.
        !(uniform(c.vaddr) && uniform(c.vaddr + 1)) && !frames.contains_key(&c.vaddr)
    })
}

pub(super) fn vg_sext_ioff(io: u32) -> i64 {
    (((io << 8) as i32) >> 8) as i64
}

// A run of consecutive VGLOBAL f64-shaped loads (B64/B128) off the same
// per-lane pointer (saddr=124), covering a contiguous, pairwise 8-aligned byte
// span, is one per-lane record read. Consecutive-only prevents intervening
// changes to EXEC, memory or address VGPRs. Scheduling no-ops have already
// been filtered out of `body` by ir::is_noop.
fn vg_cluster(body: &[InstFormat]) -> Option<VgCluster> {
    fn f64_words(g: &VGLOBAL) -> Option<u32> {
        match g.op {
            I::GLOBAL_LOAD_B64 => Some(2),
            I::GLOBAL_LOAD_B128 => Some(4),
            _ => None,
        }
    }
    let first = match body.first()? {
        InstFormat::VGLOBAL(g) if g.saddr == 124 && f64_words(g).is_some() => g,
        _ => return None,
    };
    let vaddr = first.vaddr;
    let mut members: Vec<&VGLOBAL> = Vec::new();
    for inst in body {
        let g = match inst {
            InstFormat::VGLOBAL(g) => g,
            _ => break,
        };
        let Some(w) = f64_words(g) else { break };
        if g.saddr != 124 || g.vaddr != vaddr {
            break;
        }
        members.push(g);
        // This load overwrites the pointer pair: later loads would read the
        // NEW address — stop extending (this member itself is still fine).
        let dst = g.vdst as u32..g.vdst as u32 + w;
        if dst.contains(&(vaddr as u32)) || dst.contains(&(vaddr as u32 + 1)) {
            break;
        }
    }
    if members.len() < 3 {
        return None;
    }
    // Contiguous coverage: every byte in [lo, hi) is read by some member, so
    // for an ACTIVE lane the whole span is guest-dereferenced (fault-safe).
    let mut ranges: Vec<(i64, i64)> = members
        .iter()
        .map(|g| {
            let s = vg_sext_ioff(g.ioffset);
            (s, s + 4 * f64_words(g).unwrap() as i64)
        })
        .collect();
    ranges.sort();
    let lo = ranges[0].0;
    let mut hi = ranges[0].1;
    for &(a, b) in &ranges[1..] {
        if a > hi {
            return None;
        }
        hi = hi.max(b);
    }
    // Members must decompose into f64 columns of the span.
    if members.iter().any(|g| (vg_sext_ioff(g.ioffset) - lo) % 8 != 0) || (hi - lo) % 8 != 0 {
        return None;
    }
    let span = ((hi - lo) / 8) as u32;
    if span < 4 {
        return None;
    }
    Some(VgCluster { len: members.len(), vaddr: vaddr as u32, lo, span })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(op: I, vdst: u8, offset: i32) -> InstFormat {
        InstFormat::VGLOBAL(VGLOBAL {
            op, vaddr: 10, vsrc: 0, vdst, scope: 0, th: 0,
            ioffset: offset as u32 & 0x00ff_ffff, saddr: 124, sve: 0,
        })
    }

    fn signed_record() -> Vec<InstFormat> {
        vec![
            load(I::GLOBAL_LOAD_B128, 20, 0),
            load(I::GLOBAL_LOAD_B64, 24, -8),
            load(I::GLOBAL_LOAD_B64, 26, 16),
        ]
    }

    #[test]
    fn selects_contiguous_signed_span_at_every_packet_width() {
        for width in [1, 2, 4, 8, 16] {
            assert_eq!(
                analyze(&signed_record(), width, [1 << 10, 0], &HashMap::new()),
                Some(VgCluster { len: 3, vaddr: 10, lo: -8, span: 4 }),
            );
        }
        for width in [0, 3] {
            assert!(analyze(&signed_record(), width, [1 << 10, 0], &HashMap::new()).is_none());
        }
    }

    #[test]
    fn keeps_uniform_and_scratch_frame_addresses_on_existing_paths() {
        let body = signed_record();
        assert!(analyze(&body, 16, [0; 2], &HashMap::new()).is_none());
        // Either half of the pointer being divergent makes the address divergent.
        for divergent in [[1 << 10, 0], [1 << 11, 0]] {
            assert!(analyze(&body, 16, divergent, &HashMap::new()).is_some());
            let frames = HashMap::from([(10, 0)]);
            assert!(analyze(&body, 16, divergent, &frames).is_none());
        }
    }

    #[test]
    fn rejects_gaps_and_misaligned_columns() {
        for offset in [24, 12] {
            let mut body = signed_record();
            body[2] = load(I::GLOBAL_LOAD_B64, 26, offset);
            assert!(analyze(&body, 16, [1 << 10, 0], &HashMap::new()).is_none());
        }
    }

    #[test]
    fn includes_pointer_overwrite_but_not_later_loads() {
        let mut body = vec![
            load(I::GLOBAL_LOAD_B128, 20, 0),
            load(I::GLOBAL_LOAD_B128, 24, 16),
            load(I::GLOBAL_LOAD_B128, 10, 32),
            load(I::GLOBAL_LOAD_B128, 28, 48),
        ];
        assert_eq!(
            analyze(&body, 16, [1 << 10, 0], &HashMap::new()),
            Some(VgCluster { len: 3, vaddr: 10, lo: 0, span: 6 }),
        );
        body[1] = load(I::GLOBAL_LOAD_B128, 9, 16);
        assert!(analyze(&body, 16, [1 << 10, 0], &HashMap::new()).is_none());
    }

    #[test]
    fn does_not_cross_an_intervening_store() {
        let mut body = signed_record();
        body.insert(1, load(I::GLOBAL_STORE_B64, 0, 0));
        assert!(analyze(&body, 16, [1 << 10, 0], &HashMap::new()).is_none());
    }
}
