//! The memory formats: SMEM, DS, and the FLAT / GLOBAL / SCRATCH family.
//!
//! A load cannot be checked by its address -- the simulator and the hardware
//! never place a buffer at the same place. So the harness fills a buffer with a
//! pattern that identifies each word, and the tests compare the value that came
//! back. Stores are checked the same way round: the test states what the word in
//! memory became.
//!
//! Every case also states what the *other* destinations hold, so an instruction
//! that writes the wrong register or the wrong address fails rather than passing
//! on the one value the test happened to look at.

use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// Where the destination and the memory windows sit in the harness output.
const VDST: usize = 0;
const SDST: usize = 4;
const DATA: usize = 12;

pub(crate) struct MemLoad {
    pub(crate) ioffset: i32,
    /// SADDR: `SADDR_NULL` for a 64-bit vector address, or an SGPR pair.
    pub(crate) saddr: u32,
    /// The four dwords of v[6:9] after the load.
    pub(crate) expected: [u32; 4],
}

pub(crate) struct MemStore {
    pub(crate) store_value: u64,
    pub(crate) ioffset: i32,
    pub(crate) saddr: u32,
    /// The lane's word in the buffer after the store.
    pub(crate) expected_data: u32,
}

pub(crate) struct SmemLoad {
    pub(crate) ioffset: i32,
    /// The eight dwords of s[16:23] after the load.
    pub(crate) expected: [u32; 8],
}

fn read_all(harness: &Harness, engine: Engine, words: &[u32], store: u64) -> Vec<u32> {
    let mut src = vec![0u32; LANES * harness.src_stride];
    for lane in 0..LANES {
        src[lane * harness.src_stride] = store as u32;
        src[lane * harness.src_stride + 1] = (store >> 32) as u32;
    }
    let uni = vec![0u32; 8];
    harness.run(engine, words, &src, &uni)
}

fn report(failures: Vec<String>, total: usize) {
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(),
        total,
        failures.join("\n"),
    );
}

/// A FLAT, GLOBAL or SCRATCH load. Lane `n` addresses word `n` of the buffer,
/// so the value that comes back says which address was used.
pub(crate) fn check_vmem_load(enc: u32, op: u32, cases: &[MemLoad]) {
    let harness = Harness::mem();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let words = vmem(enc, op, 6, 0, 0, case.saddr, case.ioffset).to_vec();
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = read_all(&harness, engine, &words, 0);
            let got = [out[VDST], out[VDST + 1], out[VDST + 2], out[VDST + 3]];
            let data = out[DATA];
            if got == case.expected && data == data_word(0) {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} ioffset={} saddr=0x{:02X} hardware=({:08X?}, buffer 0x{:08X}) simulator=({:08X?}, buffer 0x{:08X})",
                engine_name(engine), i, case.ioffset, case.saddr,
                case.expected, data_word(0), got, data,
            ));
        }
    }
    report(failures, cases.len() * 2);
}

/// A FLAT, GLOBAL or SCRATCH store. The destination register must stay clear:
/// a store has no destination, so a write there is a decoding mistake.
pub(crate) fn check_vmem_store(enc: u32, op: u32, cases: &[MemStore]) {
    let harness = Harness::mem();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let words = vmem(enc, op, 0, 2, 0, case.saddr, case.ioffset).to_vec();
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = read_all(&harness, engine, &words, case.store_value);
            let (data, vdst) = (out[DATA], out[VDST]);
            if data == case.expected_data && vdst == 0 {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} value=0x{:016X} ioffset={} hardware=(buffer 0x{:08X}, vdst 0) simulator=(buffer 0x{:08X}, vdst 0x{:08X})",
                engine_name(engine), i, case.store_value, case.ioffset,
                case.expected_data, data, vdst,
            ));
        }
    }
    report(failures, cases.len() * 2);
}

/// An SMEM load, which reads through a wave-uniform base in s[10:11] and writes
/// SGPRs.
pub(crate) fn check_smem_load(op: u32, cases: &[SmemLoad]) {
    let harness = Harness::mem();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let words = smem(op, 16, 5, case.ioffset).to_vec();
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = read_all(&harness, engine, &words, 0);
            let mut got = [0u32; 8];
            got.copy_from_slice(&out[SDST..SDST + 8]);
            if got == case.expected {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} ioffset={} hardware={:08X?} simulator={:08X?}",
                engine_name(engine),
                i,
                case.ioffset,
                case.expected,
                got,
            ));
        }
    }
    report(failures, cases.len() * 2);
}

#[test]
fn global_load_b128_load() {
    // GLOBAL_LOAD_B128.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        23,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0xA303_0303] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0xB313_1313] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0xA000_0000, 0xA101_0101, 0xA202_0202] },
        ],
    );
}

#[test]
fn global_load_b32_load() {
    // GLOBAL_LOAD_B32.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        20,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_load_b64_load() {
    // GLOBAL_LOAD_B64.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        21,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0xA101_0101, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0xA202_0202, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0xA303_0303, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0xB111_1111, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0xA000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_load_b96_load() {
    // GLOBAL_LOAD_B96.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        22,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0xA000_0000, 0xA101_0101, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_load_i16_load() {
    // GLOBAL_LOAD_I16.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        19,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_load_i8_load() {
    // GLOBAL_LOAD_I8.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        17,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0002, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_load_u16_load() {
    // GLOBAL_LOAD_U16.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        18,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_load_u8_load() {
    // GLOBAL_LOAD_U8.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VGLOBAL,
        16,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0002, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn global_store_b128_store() {
    // GLOBAL_STORE_B128.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VGLOBAL,
        29,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn global_store_b16_store() {
    // GLOBAL_STORE_B16.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VGLOBAL,
        25,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn global_store_b32_store() {
    // GLOBAL_STORE_B32.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VGLOBAL,
        26,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn global_store_b64_store() {
    // GLOBAL_STORE_B64.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VGLOBAL,
        27,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn global_store_b8_store() {
    // GLOBAL_STORE_B8.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VGLOBAL,
        24,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_00EF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_0078 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn global_store_b96_store() {
    // GLOBAL_STORE_B96.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VGLOBAL,
        28,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn flat_load_b128_load() {
    // FLAT_LOAD_B128.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        23,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0xA303_0303] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0xB313_1313] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0xA000_0000, 0xA101_0101, 0xA202_0202] },
        ],
    );
}

#[test]
fn flat_load_b32_load() {
    // FLAT_LOAD_B32.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        20,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_load_b64_load() {
    // FLAT_LOAD_B64.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        21,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0xA101_0101, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0xA202_0202, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0xA303_0303, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0xB111_1111, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0xA000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_load_b96_load() {
    // FLAT_LOAD_B96.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        22,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0xA000_0000, 0xA101_0101, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_load_i16_load() {
    // FLAT_LOAD_I16.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        19,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_load_i8_load() {
    // FLAT_LOAD_I8.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        17,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0002, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_load_u16_load() {
    // FLAT_LOAD_U16.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        18,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_load_u8_load() {
    // FLAT_LOAD_U8.
    // Lane n addresses word n of the buffer, so the value that comes back
    // identifies the address the instruction used. IOFFSET is varied, including
    // a negative one.
    check_vmem_load(
        VFLAT,
        16,
        &[
            MemLoad { ioffset: 0, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 4, saddr: SADDR_NULL, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 8, saddr: SADDR_NULL, expected: [0x0000_0002, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: 64, saddr: SADDR_NULL, expected: [0x0000_0010, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            MemLoad { ioffset: -4, saddr: SADDR_NULL, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn flat_store_b128_store() {
    // FLAT_STORE_B128.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VFLAT,
        29,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn flat_store_b16_store() {
    // FLAT_STORE_B16.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VFLAT,
        25,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn flat_store_b32_store() {
    // FLAT_STORE_B32.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VFLAT,
        26,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn flat_store_b64_store() {
    // FLAT_STORE_B64.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VFLAT,
        27,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn flat_store_b8_store() {
    // FLAT_STORE_B8.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VFLAT,
        24,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_00EF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xA000_0078 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn flat_store_b96_store() {
    // FLAT_STORE_B96.
    // The word in the buffer after the store is what is checked; the test
    // also requires the destination register to be untouched, since a store
    // has none.
    check_vmem_store(
        VFLAT,
        28,
        &[
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x0000_0000 },
            MemStore { store_value: 0x0000_0000_0000_0000, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 0, saddr: SADDR_NULL, expected_data: 0xDEAD_BEEF },
            MemStore { store_value: 0x0000_0000_DEAD_BEEF, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 0, saddr: SADDR_NULL, expected_data: 0x1234_5678 },
            MemStore { store_value: 0xCAFE_BABE_1234_5678, ioffset: 4, saddr: SADDR_NULL, expected_data: 0xA000_0000 },
        ],
    );
}

#[test]
fn s_load_b128_smem() {
    // S_LOAD_B128.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        2,
        &[
            SmemLoad { ioffset: 0, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0xA303_0303, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 8, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 64, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0xB313_1313, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_b256_smem() {
    // S_LOAD_B256.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        3,
        &[
            SmemLoad { ioffset: 0, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0xA606_0606, 0xA707_0707] },
            SmemLoad { ioffset: 4, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0xA606_0606, 0xA707_0707, 0xA808_0808] },
            SmemLoad { ioffset: 8, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0xA606_0606, 0xA707_0707, 0xA808_0808, 0xA909_0909] },
            SmemLoad { ioffset: 64, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0xB313_1313, 0xB414_1414, 0xB515_1515, 0xB616_1616, 0xB717_1717] },
        ],
    );
}

#[test]
fn s_load_b32_smem() {
    // S_LOAD_B32.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        0,
        &[
            SmemLoad { ioffset: 0, expected: [0xA000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0xA101_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 8, expected: [0xA202_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 64, expected: [0xB010_1010, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_b512_smem() {
    // S_LOAD_B512.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        4,
        &[
            SmemLoad { ioffset: 0, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0xA606_0606, 0xA707_0707] },
            SmemLoad { ioffset: 4, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0xA606_0606, 0xA707_0707, 0xA808_0808] },
            SmemLoad { ioffset: 8, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0xA505_0505, 0xA606_0606, 0xA707_0707, 0xA808_0808, 0xA909_0909] },
            SmemLoad { ioffset: 64, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0xB313_1313, 0xB414_1414, 0xB515_1515, 0xB616_1616, 0xB717_1717] },
        ],
    );
}

#[test]
fn s_load_b64_smem() {
    // S_LOAD_B64.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        1,
        &[
            SmemLoad { ioffset: 0, expected: [0xA000_0000, 0xA101_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0xA101_0101, 0xA202_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 8, expected: [0xA202_0202, 0xA303_0303, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 64, expected: [0xB010_1010, 0xB111_1111, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_b96_smem() {
    // S_LOAD_B96.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        5,
        &[
            SmemLoad { ioffset: 0, expected: [0xA000_0000, 0xA101_0101, 0xA202_0202, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0xA101_0101, 0xA202_0202, 0xA303_0303, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 8, expected: [0xA202_0202, 0xA303_0303, 0xA404_0404, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 64, expected: [0xB010_1010, 0xB111_1111, 0xB212_1212, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_i16_smem() {
    // S_LOAD_I16.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        10,
        &[
            SmemLoad { ioffset: 0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 2, expected: [0xFFFF_A000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0x0000_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 6, expected: [0xFFFF_A101, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_i8_smem() {
    // S_LOAD_I8.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        8,
        &[
            SmemLoad { ioffset: 0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 1, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 2, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 3, expected: [0xFFFF_FFA0, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 7, expected: [0xFFFF_FFA1, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_u16_smem() {
    // S_LOAD_U16.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        11,
        &[
            SmemLoad { ioffset: 0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 2, expected: [0x0000_A000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0x0000_0101, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 6, expected: [0x0000_A101, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}

#[test]
fn s_load_u8_smem() {
    // S_LOAD_U8.
    // SMEM reads through the wave-uniform base in s[10:11]. The eight SGPRs
    // the harness reads back cover up to a 256-bit load.
    check_smem_load(
        9,
        &[
            SmemLoad { ioffset: 0, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 1, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 2, expected: [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 3, expected: [0x0000_00A0, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 4, expected: [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
            SmemLoad { ioffset: 7, expected: [0x0000_00A1, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000] },
        ],
    );
}
