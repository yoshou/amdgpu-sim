//! Select consecutive global loads for record loading and transposition.
//!
//! The result describes the instruction prefix and its memory span. Selection
//! uses the existing divergence and scratch-frame facts at that instruction;
//! it neither changes those facts nor emits LLVM instructions.

#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
use crate::instructions::I;
#[cfg(test)]
use crate::rdna_instructions::{InstFormat, VGLOBAL};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct VgCluster {
    pub(super) len: usize, // number of member instructions (≥ 3)
    pub(super) vaddr: u32, // shared per-lane pointer pair
    pub(super) lo: i64,    // lowest sign-extended ioffset (span start, bytes)
    pub(super) span: u32,  // span length in f64 fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select(body: &[InstFormat], width: u32, divergent: [u128; 2], frames: &HashMap<u32, u32>) -> Option<VgCluster> {
        use std::collections::BTreeMap;
        use crate::rdna_spmd::{ir::{ScalarProgram, ScalarBlock, Terminator}, lift};
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
            pc: 0, body: body.to_vec(), term: Terminator::Return,
        })]) };
        let semantics: Vec<_> = body.iter().map(lift::instruction).collect();
        let refs = BTreeMap::from([(0, semantics.iter().collect())]);
        let f = lift::function::Function::lift(std::sync::Arc::new(crate::rdna_spmd::dialect::DialectRegistry::rdna4()), &program, &refs);
        let mut varying = vec![false; f.ir.types.len()];
        let mut frame_values = BTreeMap::new();
        for (&index, plan) in &f.blocks[&0].memory {
            if let lift::memory::Address::Global { vector, .. } = plan.memory.address {
                let address = &f.state.sites[&0][index].address;
                for (k, value) in address.iter().enumerate() {
                    let r = vector + k as u32;
                    varying[value.0] = divergent[(r / 128) as usize] & (1 << (r % 128)) != 0;
                }
                if address.len() == 2 {
                    if let Some(&stride) = frames.get(&vector) { frame_values.insert((address[0], address[1]), stride); }
                }
            }
        }
        analyze(&f.blocks[&0].memory, &f.state.sites[&0], 0, width, &varying, &frame_values)
    }

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
                select(&signed_record(), width, [1 << 10, 0], &HashMap::new()),
                Some(VgCluster { len: 3, vaddr: 10, lo: -8, span: 4 }),
            );
        }
        for width in [0, 3] {
            assert!(select(&signed_record(), width, [1 << 10, 0], &HashMap::new()).is_none());
        }
    }

    #[test]
    fn keeps_uniform_and_scratch_frame_addresses_on_existing_paths() {
        let body = signed_record();
        assert!(select(&body, 16, [0; 2], &HashMap::new()).is_none());
        // Either half of the pointer being divergent makes the address divergent.
        for divergent in [[1 << 10, 0], [1 << 11, 0]] {
            assert!(select(&body, 16, divergent, &HashMap::new()).is_some());
            let frames = HashMap::from([(10, 0)]);
            assert!(select(&body, 16, divergent, &frames).is_none());
        }
    }

    #[test]
    fn rejects_gaps_and_misaligned_columns() {
        for offset in [24, 12] {
            let mut body = signed_record();
            body[2] = load(I::GLOBAL_LOAD_B64, 26, offset);
            assert!(select(&body, 16, [1 << 10, 0], &HashMap::new()).is_none());
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
            select(&body, 16, [1 << 10, 0], &HashMap::new()),
            Some(VgCluster { len: 3, vaddr: 10, lo: 0, span: 6 }),
        );
        body[1] = load(I::GLOBAL_LOAD_B128, 9, 16);
        assert!(select(&body, 16, [1 << 10, 0], &HashMap::new()).is_none());
    }

    #[test]
    fn does_not_cross_an_intervening_store() {
        let mut body = signed_record();
        body.insert(1, load(I::GLOBAL_STORE_B64, 0, 0));
        assert!(select(&body, 16, [1 << 10, 0], &HashMap::new()).is_none());
    }
}

/// Select the existing record-load shape from typed memory effects and SSA
/// addresses. Consecutive positions retain the original scheduling restriction.
pub(super) fn analyze(
    memory: &std::collections::BTreeMap<usize, super::lift::memory::Plan>,
    sites: &[super::analysis::state::Site],
    start: usize,
    width: u32,
    varying: &[bool],
    frames: &std::collections::BTreeMap<(super::ir::typed::ValueId, super::ir::typed::ValueId), u32>,
) -> Option<VgCluster> {
    use super::lift::memory::Address;
    use super::ir::typed::effect::{MemoryOp, MemSize};
    if !width.is_power_of_two() { return None; }
    let address = &sites.get(start)?.address;
    if address.len() != 2 || address.iter().all(|v| !varying[v.0])
        || frames.contains_key(&(address[0], address[1])) { return None; }
    let mut ranges = Vec::new();
    let mut vaddr = 0;
    for index in start..sites.len() {
        let Some(plan) = memory.get(&index) else { break; };
        let m = &plan.memory;
        let Address::Global { scalar: None, vector, offset } = m.address else { break; };
        if m.op != MemoryOp::Load(MemSize::B32) || !matches!(m.words, 2 | 4)
            || sites[index].address != *address { break; }
        if index == start { vaddr = vector; }
        ranges.push((offset, offset + 4 * m.words as i64));
    }
    if ranges.len() < 3 { return None; }
    let len = ranges.len();
    ranges.sort();
    let lo = ranges[0].0;
    let mut hi = ranges[0].1;
    for &(a, b) in &ranges[1..] {
        if a > hi { return None; }
        hi = hi.max(b);
    }
    if ranges.iter().any(|&(a, _)| (a-lo) % 8 != 0) || (hi-lo) % 8 != 0 { return None; }
    let span = ((hi-lo) / 8) as u32;
    (span >= 4).then_some(VgCluster { len, vaddr, lo, span })
}
