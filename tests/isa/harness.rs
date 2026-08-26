//! Loading, patching and running the conformance harness kernels.
//!
//! `tests/data/harness_gfx1200.o` holds one kernel per instruction format. Each
//! places the sources where its format expects them, executes a patch slot, and
//! reads back everything that format can write. This module finds the slot,
//! writes the instruction under test into it, and runs one wave on a chosen
//! engine. It carries no knowledge of what any instruction means.

use crate::encoding::{slot_marker, SLOT_BYTES, S_NOP};
use amdgpu_sim::buffer::*;
use amdgpu_sim::processor::*;
use amdgpu_sim::rdna_processor::*;
use object::*;
use std::fs::File;
use std::io::Read;

pub(crate) const LANES: usize = 32;

/// The pattern the memory harness puts in its buffer: word `k` holds this, so a
/// loaded value says which address it came from.
pub(crate) const fn data_word(k: u32) -> u32 {
    0xA000_0000 | k.wrapping_mul(0x0101_0101)
}

/// One instruction format's harness kernel: where it puts the sources, what it
/// reads back, and the slot it executes in between.
pub(crate) struct Harness {
    pub(crate) mem: Vec<u8>,
    pub(crate) slot: usize,
    pub(crate) kernel_addr: usize,
    pub(crate) kernarg_size: usize,
    /// Scratch bytes per work-item, as the kernel declares them.
    pub(crate) private_segment_size: usize,
    /// Source dwords the kernel reads per lane.
    pub(crate) src_stride: usize,
    /// Result dwords the kernel writes per lane.
    pub(crate) out_stride: usize,
}

#[derive(serde::Deserialize)]
pub(crate) struct KernelMeta {
    #[serde(alias = ".name")]
    name: String,
    #[serde(alias = ".kernarg_segment_size")]
    kernarg_segment_size: i64,
    #[serde(alias = ".private_segment_fixed_size")]
    private_segment_fixed_size: i64,
}

#[derive(serde::Deserialize)]
pub(crate) struct Meta {
    #[serde(alias = "amdhsa.kernels")]
    amdhsa_kernels: Vec<KernelMeta>,
}

pub(crate) fn align(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

/// The kernarg and private segment sizes the named kernel declares. Both come
/// from the object rather than from a constant here, since the compiler decides
/// them.
pub(crate) fn segment_sizes(note: &[u8], kernel: &str) -> (usize, usize) {
    let mut pos = 0;
    while pos < note.len() {
        let name_size = get_u32(note, pos) as usize;
        let data_size = get_u32(note, pos + 4) as usize;
        let note_type = get_u32(note, pos + 8) as usize;
        pos = align(pos + 12 + name_size, 4);
        let data = get_bytes(note, pos, data_size);
        pos = align(pos + data_size, 4);
        if note_type == 32 {
            let map: Meta = rmp_serde::from_slice(&data).unwrap();
            let meta = map
                .amdhsa_kernels
                .iter()
                .find(|k| k.name == kernel)
                .unwrap_or_else(|| panic!("no metadata for {}", kernel));
            return (
                meta.kernarg_segment_size as usize,
                meta.private_segment_fixed_size as usize,
            );
        }
    }
    panic!("no MessagePack metadata note in the harness object");
}

impl Harness {
    pub(crate) fn vop1() -> Self {
        Self::load("harness_vop1.kd", 0x1111_1111, 2, 2)
    }

    pub(crate) fn vop2() -> Self {
        Self::load("harness_vop2.kd", 0x2222_2222, 4, 4)
    }

    pub(crate) fn vop3() -> Self {
        Self::load("harness_vop3.kd", 0x3333_3333, 6, 4)
    }

    pub(crate) fn vopc() -> Self {
        Self::load("harness_vopc.kd", 0x4444_4444, 4, 4)
    }

    pub(crate) fn vopd() -> Self {
        Self::load("harness_vopd.kd", 0x9999_9999, 6, 4)
    }

    pub(crate) fn salu() -> Self {
        Self::load("harness_salu.kd", 0x5555_5555, 4, 4)
    }

    pub(crate) fn mem() -> Self {
        Self::load("harness_mem.kd", 0x6666_6666, 2, 16)
    }

    pub(crate) fn lds() -> Self {
        Self::load("harness_lds.kd", 0x7777_7777, 4, 16)
    }

    pub(crate) fn scratch() -> Self {
        Self::load("harness_scratch.kd", 0x8888_8888, 4, 16)
    }

    pub(crate) fn load(
        kernel: &str,
        marker_literal: u32,
        src_stride: usize,
        out_stride: usize,
    ) -> Self {
        let marker = slot_marker(marker_literal);
        let mut data = vec![];
        File::open("tests/data/harness_gfx1200.o")
            .expect("tests/data/harness_gfx1200.o")
            .read_to_end(&mut data)
            .unwrap();
        let elf = ElfFile::parse(&data).expect("failed to parse the harness object");

        let note = elf
            .sections()
            .find(|s| s.name() == Some(".note"))
            .expect("no .note section");
        let (kernarg_size, private_segment_size) =
            segment_sizes(note.data(), kernel.trim_end_matches(".kd"));

        let mut mem = Vec::<u8>::new();
        for segment in elf.segments() {
            let offset = segment.address() as usize;
            let size = segment.size() as usize;
            let new_size = mem.len().max(offset + size);
            mem.resize(new_size, 0);
            mem[offset..(offset + size.min(segment.data().len()))].copy_from_slice(segment.data());
        }

        let slot = mem
            .windows(marker.len())
            .position(|w| w == marker)
            .unwrap_or_else(|| panic!("patch slot marker for {} not found", kernel));
        // The marker alone is not proof we found the right place; the padding
        // behind it must be there too, or the harness and this file disagree.
        for i in (marker.len()..SLOT_BYTES).step_by(4) {
            assert_eq!(
                get_u32(&mem, slot + i),
                S_NOP,
                "harness slot is shorter than SLOT_BYTES at +{}",
                i
            );
        }

        let kernel_addr = elf
            .symbols()
            .find(|sym| sym.name() == Some(kernel))
            .unwrap_or_else(|| panic!("{} not found", kernel))
            .address() as usize;

        Harness {
            mem,
            slot,
            kernel_addr,
            kernarg_size,
            private_segment_size,
            src_stride,
            out_stride,
        }
    }

    /// Patch `words` into the slot, pad the rest with S_NOP, and run one wave.
    /// `src` holds `src_stride` dwords per lane and `uni` the values the kernel
    /// puts in SGPRs; the result is `out_stride` dwords per lane.
    pub(crate) fn run(&self, engine: Engine, words: &[u32], src: &[u32], uni: &[u32]) -> Vec<u32> {
        assert_eq!(src.len(), LANES * self.src_stride);
        assert!(
            words.len() * 4 <= SLOT_BYTES,
            "instruction sequence exceeds the slot"
        );
        let mut mem = self.mem.clone();
        for i in 0..(SLOT_BYTES / 4) {
            let word = words.get(i).copied().unwrap_or(S_NOP);
            mem[self.slot + i * 4..self.slot + i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut out = vec![0u32; LANES * self.out_stride];
        // The memory harness takes a fourth argument: the buffer it loads from
        // and stores into. Word k holds a value that identifies k, so a loaded
        // value says which address it came from. The zeroed words on either side
        // stand in for the memory around the hardware's allocation, so a case
        // whose offset reaches outside the buffer reads the same zeros the
        // hardware read rather than whatever happens to be next to this
        // process's allocation.
        const GUARD: usize = 64;
        let mut data: Vec<u32> = vec![0; GUARD];
        data.extend((0..256u32).map(data_word));
        data.extend(std::iter::repeat(0).take(GUARD));
        let mut arg_buffer = vec![0u8; self.kernarg_size];
        set_u64(&mut arg_buffer, 0, out.as_mut_ptr() as u64);
        set_u64(&mut arg_buffer, 8, src.as_ptr() as u64);
        set_u64(&mut arg_buffer, 16, uni.as_ptr() as u64);
        if self.kernarg_size >= 32 {
            set_u64(&mut arg_buffer, 24, unsafe { data.as_mut_ptr().add(GUARD) }
                as u64);
        }

        let aql = HsaKernelDispatchPacket {
            header: 0,
            setup: 0,
            workgroup_size_x: LANES as u16,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            grid_size_x: 1,
            grid_size_y: 1,
            grid_size_z: 1,
            private_segment_size: self.private_segment_size as u32,
            group_segment_size: 0,
            kernel_object: Pointer::new(&mem, self.kernel_addr),
            kernarg_address: Pointer::new(&arg_buffer, 0),
        };

        let mut processor = RDNAProcessor::with_engine(&aql, 32, 32, &mem, engine);
        processor.execute();
        out
    }
}

pub(crate) fn engine_name(engine: Engine) -> &'static str {
    match engine {
        Engine::Interpreter => "interpreter",
        Engine::LlvmJit => "LLVM JIT",
    }
}
