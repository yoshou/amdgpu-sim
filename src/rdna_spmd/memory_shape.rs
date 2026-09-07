use super::ir::typed::effect::{MemSize, MemoryOp, Space};
use super::lift::memory::{Address, Memory};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlobalLoad {
    Gather,
    Broadcast,
    Frame { stride_words: u32, offset_words: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lanes {
    Gather,
    Broadcast,
    Lds,
    Affine { allocated: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StoreShape {
    Narrow,
    Lds,
    Affine,
    Scatter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PacketShape {
    Fence,
    ScalarWords,
    AtomicAdd { grouped: bool },
    Store(StoreShape),
    NarrowLoad,
    PrivateTile { tile: u32 },
    Frame { stride_words: u32, offset_words: u32, group: u32 },
    Words { lanes: Lanes, pairs: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarShape {
    Fence,
    AtomicAdd,
    Store,
    Words { pairs: bool },
}

pub(super) fn packet(m: &Memory, width: u32, load: GlobalLoad) -> PacketShape {
    if m.op == MemoryOp::Fence { return PacketShape::Fence; }
    if m.scalar() { return PacketShape::ScalarWords; }
    if m.op == MemoryOp::AtomicAdd {
        let grouped = width >= 4 && m.space() == Space::Global && !m.returns && !m.semantics.volatile;
        return PacketShape::AtomicAdd { grouped };
    }
    let affine = matches!(m.address, Address::Scratch { vector: None, .. });
    if m.stores() {
        return PacketShape::Store(if m.size() != MemSize::B32 { StoreShape::Narrow }
            else if m.space() == Space::Lds { StoreShape::Lds }
            else if affine { StoreShape::Affine }
            else { StoreShape::Scatter });
    }
    if m.size() != MemSize::B32 { return PacketShape::NarrowLoad; }
    let allocated = m.private_load_end().is_some();
    if allocated && m.words >= 2 {
        return PacketShape::PrivateTile { tile: if width % 4 == 0 { 4 } else if width % 2 == 0 { 2 } else { 1 } };
    }
    let global = matches!(m.address, Address::Global { .. });
    if let (true, GlobalLoad::Frame { stride_words, offset_words }) = (global, load) {
        return PacketShape::Frame { stride_words, offset_words, group: width.min(8) };
    }
    let lanes = if global && load == GlobalLoad::Broadcast { Lanes::Broadcast }
        else if m.space() == Space::Lds { Lanes::Lds }
        else if affine { Lanes::Affine { allocated } }
        else { Lanes::Gather };
    PacketShape::Words { lanes, pairs: global }
}

pub(super) fn scalar(m: &Memory) -> ScalarShape {
    if m.op == MemoryOp::Fence { return ScalarShape::Fence; }
    if m.op == MemoryOp::AtomicAdd { return ScalarShape::AtomicAdd; }
    if m.stores() { return ScalarShape::Store; }
    ScalarShape::Words { pairs: matches!(m.address, Address::Global { .. }) && m.size() == MemSize::B32 && !m.semantics.volatile }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, DS, SMEM, VFLAT, VGLOBAL, VSCRATCH};
    use super::super::lift::memory::instruction;

    fn global(op: I, saddr: u8, th: u8) -> Memory {
        instruction(&InstFormat::VGLOBAL(VGLOBAL { op, vaddr: 10, vsrc: 2, vdst: 20, scope: 0, th, ioffset: 0, saddr, sve: 0 })).unwrap()
    }
    fn scratch(op: I, sve: u8, offset: i32) -> Memory {
        instruction(&InstFormat::VSCRATCH(VSCRATCH { op, vaddr: 4, vsrc: 2, vdst: 20, scope: 0, th: 0, ioffset: offset as u32 & 0x00ff_ffff, saddr: 124, sve })).unwrap()
    }
    fn lds(op: I) -> Memory {
        instruction(&InstFormat::DS(DS { offset0: 0, offset1: 0, op, addr: 4, data0: 2, data1: 3, vdst: 20 })).unwrap()
    }
    fn flat(op: I) -> Memory {
        instruction(&InstFormat::VFLAT(VFLAT { op, vaddr: 10, vsrc: 2, vdst: 20, scope: 0, th: 0, ioffset: 0, saddr: 124, sve: 0 })).unwrap()
    }

    #[test]
    fn global_loads_keep_pairs_and_follow_the_uniformity_and_frame_facts() {
        let m = global(I::GLOBAL_LOAD_B64, 124, 0);
        assert_eq!(packet(&m, 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Gather, pairs: true });
        assert_eq!(packet(&m, 16, GlobalLoad::Broadcast), PacketShape::Words { lanes: Lanes::Broadcast, pairs: true });
        assert_eq!(packet(&m, 16, GlobalLoad::Frame { stride_words: 8, offset_words: 4 }),
            PacketShape::Frame { stride_words: 8, offset_words: 4, group: 8 });
        assert_eq!(packet(&m, 4, GlobalLoad::Frame { stride_words: 8, offset_words: 4 }),
            PacketShape::Frame { stride_words: 8, offset_words: 4, group: 4 });
        assert_eq!(packet(&global(I::GLOBAL_LOAD_B32, 0, 0), 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Gather, pairs: true });
        assert_eq!(packet(&flat(I::FLAT_LOAD_B64), 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Gather, pairs: false });
    }

    #[test]
    fn scratch_loads_select_static_tiles_or_affine_lanes() {
        assert_eq!(packet(&scratch(I::SCRATCH_LOAD_B64, 0, 8), 16, GlobalLoad::Gather), PacketShape::PrivateTile { tile: 4 });
        assert_eq!(packet(&scratch(I::SCRATCH_LOAD_B64, 0, 8), 2, GlobalLoad::Gather), PacketShape::PrivateTile { tile: 2 });
        assert_eq!(packet(&scratch(I::SCRATCH_LOAD_B64, 0, 8), 1, GlobalLoad::Gather), PacketShape::PrivateTile { tile: 1 });
        assert_eq!(packet(&scratch(I::SCRATCH_LOAD_B32, 0, 8), 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Affine { allocated: true }, pairs: false });
        assert_eq!(packet(&scratch(I::SCRATCH_LOAD_B32, 0, -4), 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Affine { allocated: false }, pairs: false });
        assert_eq!(packet(&scratch(I::SCRATCH_LOAD_B64, 1, 8), 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Gather, pairs: false });
        assert_eq!(packet(&lds(I::DS_LOAD_B64), 16, GlobalLoad::Gather), PacketShape::Words { lanes: Lanes::Lds, pairs: false });
    }

    #[test]
    fn narrow_accesses_stores_atomics_scalars_and_fences_have_their_own_shapes() {
        assert_eq!(packet(&global(I::GLOBAL_LOAD_U8, 124, 0), 16, GlobalLoad::Broadcast), PacketShape::NarrowLoad);
        assert_eq!(packet(&global(I::GLOBAL_STORE_B8, 124, 0), 16, GlobalLoad::Gather), PacketShape::Store(StoreShape::Narrow));
        assert_eq!(packet(&global(I::GLOBAL_STORE_B64, 124, 0), 16, GlobalLoad::Gather), PacketShape::Store(StoreShape::Scatter));
        assert_eq!(packet(&lds(I::DS_STORE_B32), 16, GlobalLoad::Gather), PacketShape::Store(StoreShape::Lds));
        assert_eq!(packet(&scratch(I::SCRATCH_STORE_B32, 0, 0), 16, GlobalLoad::Gather), PacketShape::Store(StoreShape::Affine));
        assert_eq!(packet(&scratch(I::SCRATCH_STORE_B32, 1, 0), 16, GlobalLoad::Gather), PacketShape::Store(StoreShape::Scatter));
        assert_eq!(packet(&global(I::GLOBAL_ATOMIC_ADD_U32, 124, 0), 16, GlobalLoad::Gather), PacketShape::AtomicAdd { grouped: true });
        assert_eq!(packet(&global(I::GLOBAL_ATOMIC_ADD_U32, 124, 0), 2, GlobalLoad::Gather), PacketShape::AtomicAdd { grouped: false });
        assert_eq!(packet(&global(I::GLOBAL_ATOMIC_ADD_U32, 124, 1), 16, GlobalLoad::Gather), PacketShape::AtomicAdd { grouped: false });
        assert_eq!(packet(&lds(I::DS_ADD_U32), 16, GlobalLoad::Gather), PacketShape::AtomicAdd { grouped: false });
        let smem = instruction(&InstFormat::SMEM(SMEM { op: I::S_LOAD_B64, sdata: 4, sbase: 0, ioffset: 0, soffset: 124, scope: 0, th: 0 })).unwrap();
        assert_eq!(packet(&smem, 16, GlobalLoad::Gather), PacketShape::ScalarWords);
        assert_eq!(packet(&global(I::GLOBAL_WB, 124, 0), 16, GlobalLoad::Gather), PacketShape::Fence);
    }

    #[test]
    fn scalar_pairs_require_global_full_word_loads() {
        assert_eq!(scalar(&global(I::GLOBAL_LOAD_B64, 124, 0)), ScalarShape::Words { pairs: true });
        assert_eq!(scalar(&global(I::GLOBAL_LOAD_U16, 124, 0)), ScalarShape::Words { pairs: false });
        assert_eq!(scalar(&flat(I::FLAT_LOAD_B64)), ScalarShape::Words { pairs: false });
        assert_eq!(scalar(&scratch(I::SCRATCH_LOAD_B64, 0, 0)), ScalarShape::Words { pairs: false });
        assert_eq!(scalar(&lds(I::DS_LOAD_B64)), ScalarShape::Words { pairs: false });
        assert_eq!(scalar(&global(I::GLOBAL_STORE_B32, 124, 0)), ScalarShape::Store);
        assert_eq!(scalar(&global(I::GLOBAL_ATOMIC_ADD_U32, 124, 0)), ScalarShape::AtomicAdd);
        assert_eq!(scalar(&global(I::GLOBAL_INV, 124, 0)), ScalarShape::Fence);
    }
}
