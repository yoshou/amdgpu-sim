//! Cooperative *wavefront* dispatch for kernels whose only cross-lane traffic is
//! a wave-wide op (`v_wmma_*`, `v_readlane`/`v_writelane`) — including one that
//! sits **inside a loop**.
//!
//! This reuses the same coroutine machinery as the workgroup-barrier scheduler
//! ([`super::cooperative`]): each lane is a resumable, single-lane (W=0) scalar
//! coroutine ([`CoopKernel`]) that yields at a boundary and returns its resume
//! pc. Where [`super::segmented`] handles a cross-lane op *outside* loops by
//! splitting the kernel into `[pre, boundary, post]` fragments, that model cannot
//! express a boundary reached many times by a back-edge. Modelling the boundary
//! as a coroutine *yield* removes that restriction: the loop's back-edge simply
//! flows back to the post-yield block, so the same boundary yields once per
//! iteration and the host driver runs the wave-level op each time.
//!
//! Only the cross-lane ops are lifted; every other instruction runs per-lane on
//! the scalar backend. For a *uniform* boundary (the same op reached the same
//! number of times by all lanes — true of `rocwmma`'s K-loop) all 32 lanes yield
//! at the same pc in lockstep, so the driver applies one wave-level op per pass.

use std::collections::BTreeMap;
use std::thread;

use half::{f16, slice::HalfFloatSliceExt};

use crate::instructions::I;
use crate::processor::KernelDescriptor;
use crate::rdna_instructions::{InstFormat, SourceOperand};

use super::dispatch::{setup_sgprs, GridDims};
use super::emit::{CoopKernel, COOP_DONE, COOP_SGPR_BUF, COOP_SPILL_SLOTS};
use super::emit_vec::CoopVecKernel;
use super::ir::{ScalarBlock, ScalarProgram, Terminator};

const WAVE: usize = 32;
const EXEC: usize = 126;
const SCC: usize = 128; // persisted-SCC slot (see emit.rs COOP_SGPR_BUF)

/// A wave-level cross-lane op lifted out of the per-lane instruction stream.
#[derive(Clone)]
pub enum XlaneOp {
    /// `sdst(uniform) = lane[src1] of vgpr src0`.
    ReadLane { vdst: u8, src0: SourceOperand, src1: SourceOperand },
    /// `vgpr[vdst] lane[src1] = scalar src0` (other lanes preserved).
    WriteLane { vdst: u8, src0: SourceOperand, src1: SourceOperand },
    /// `D = A·B + C` 16×16×16 f16 matrix multiply-accumulate across the wave.
    Wmma { vdst: u8, a: u8, b: u8, c: u8 },
}

/// The cross-lane op an instruction represents, if any.
fn xlane_of(inst: &InstFormat) -> Option<XlaneOp> {
    match inst {
        InstFormat::VOP3(i) if matches!(i.op, I::V_READLANE_B32) => Some(XlaneOp::ReadLane {
            vdst: i.vdst,
            src0: i.src0.clone(),
            src1: i.src1.clone(),
        }),
        InstFormat::VOP3(i) if matches!(i.op, I::V_WRITELANE_B32) => Some(XlaneOp::WriteLane {
            vdst: i.vdst,
            src0: i.src0.clone(),
            src1: i.src1.clone(),
        }),
        InstFormat::VOP3P(i) if matches!(i.op, I::V_WMMA_F32_16X16X16_F16) => {
            let reg = |o: &SourceOperand| match o {
                SourceOperand::VectorRegister(r) => *r,
                _ => panic!("WMMA source must be a VGPR, got {:?}", o),
            };
            Some(XlaneOp::Wmma {
                vdst: i.vdst,
                a: reg(&i.src0),
                b: reg(&i.src1),
                c: reg(&i.src2),
            })
        }
        _ => None,
    }
}

/// Split every block at each cross-lane op so the op becomes a coroutine yield: a
/// [`Terminator::Barrier`] whose resume pc keys the lifted [`XlaneOp`]. The op
/// instruction itself is dropped from the body (the wave-level effect is applied
/// by the driver at the yield). Blocks without a cross-lane op — and every
/// existing branch target — are preserved, so the K-loop's back-edge still points
/// at the post-yield block and re-reaches the boundary each iteration.
pub fn split_at_xlane(program: &ScalarProgram) -> (ScalarProgram, BTreeMap<usize, XlaneOp>) {
    let mut next_pc = program.blocks.keys().copied().max().unwrap_or(0) + 1;
    let mut blocks: BTreeMap<usize, ScalarBlock> = BTreeMap::new();
    let mut ops: BTreeMap<usize, XlaneOp> = BTreeMap::new();

    for block in program.blocks.values() {
        if !block.body.iter().any(|i| xlane_of(i).is_some()) {
            blocks.insert(block.pc, block.clone());
            continue;
        }
        // Cut the body at each cross-lane op. Segment i (bodies[i]) is terminated
        // by the op seg_ops[i] (None for the final segment).
        let mut bodies: Vec<Vec<InstFormat>> = Vec::new();
        let mut seg_ops: Vec<Option<XlaneOp>> = Vec::new();
        let mut cur: Vec<InstFormat> = Vec::new();
        for inst in &block.body {
            if let Some(op) = xlane_of(inst) {
                bodies.push(std::mem::take(&mut cur));
                seg_ops.push(Some(op));
            } else {
                cur.push(inst.clone());
            }
        }
        bodies.push(cur);
        seg_ops.push(None);

        let n = bodies.len();
        let mut pcs = Vec::with_capacity(n);
        pcs.push(block.pc);
        for _ in 1..n {
            pcs.push(next_pc);
            next_pc += 1;
        }
        for i in 0..n {
            let term = if i + 1 < n {
                ops.insert(pcs[i + 1], seg_ops[i].clone().unwrap());
                Terminator::Barrier { resume: pcs[i + 1] }
            } else {
                block.term.clone()
            };
            blocks.insert(pcs[i], ScalarBlock { pc: pcs[i], body: std::mem::take(&mut bodies[i]), term });
        }
    }

    (ScalarProgram { entry_pc: program.entry_pc, blocks }, ops)
}

fn xlane_boundary_writes(xlane: &BTreeMap<usize, XlaneOp>) -> BTreeMap<usize, Vec<u32>> {
    xlane
        .iter()
        .filter_map(|(&resume, op)| {
            let writes = match op {
                XlaneOp::ReadLane { .. } => Vec::new(),
                XlaneOp::WriteLane { vdst, .. } => vec![*vdst as u32],
                XlaneOp::Wmma { vdst, .. } => {
                    (0..8).map(|offset| *vdst as u32 + offset).collect()
                }
            };
            (!writes.is_empty()).then_some((resume, writes))
        })
        .collect()
}

/// Compile a split cross-lane program with the host-applied VGPR side effects
/// attached to its Barrier→resume edges. The vector divergence/frame analyses
/// need these effects to distinguish genuinely uniform addresses from values
/// produced by writelane or WMMA across a coroutine yield.
pub fn compile_xlane_vec(
    program: &ScalarProgram,
    xlane: &BTreeMap<usize, XlaneOp>,
    num_vgprs: usize,
    width: u32,
) -> CoopVecKernel {
    let boundary_writes = xlane_boundary_writes(xlane);
    super::emit_vec::compile_cooperative_with_boundary_writes(
        program,
        num_vgprs,
        width,
        &boundary_writes,
    )
}

// ---- wave-level op application (on the 32 lanes' persisted register state) ----

fn write_sgpr(sgprs: &mut [u32; COOP_SGPR_BUF], idx: usize, value: u32) {
    if idx == 124 || idx == 125 {
        return; // null / m0 conventions — never a real destination
    }
    sgprs[idx] = value;
}

/// Uniform scalar operand, read from lane 0 (SSRC values are wave-uniform).
fn eval_scalar(sgprs: &[[u32; COOP_SGPR_BUF]], op: &SourceOperand) -> u32 {
    match op {
        SourceOperand::LiteralConstant(v) => *v,
        SourceOperand::IntegerConstant(v) => *v as u32,
        SourceOperand::FloatConstant(v) => (*v as f32).to_bits(),
        SourceOperand::ScalarRegister(r) => sgprs[0][*r as usize],
        SourceOperand::VectorRegister(r) => panic!("scalar boundary operand reads VGPR {}", r),
        SourceOperand::PrivateBase => panic!("scalar boundary operand reads private base"),
    }
}

fn eval_vector(sgprs: &[[u32; COOP_SGPR_BUF]], vgprs: &[Vec<u32>], lane: usize, op: &SourceOperand) -> u32 {
    match op {
        SourceOperand::LiteralConstant(v) => *v,
        SourceOperand::IntegerConstant(v) => *v as u32,
        SourceOperand::FloatConstant(v) => (*v as f32).to_bits(),
        SourceOperand::ScalarRegister(r) => sgprs[lane][*r as usize],
        SourceOperand::VectorRegister(r) => vgprs[lane][*r as usize],
        SourceOperand::PrivateBase => panic!("vector boundary operand reads private base"),
    }
}

/// `v_wmma_f32_16x16x16_f16`: bit-layout port of the masked interpreter's
/// `RDNAProcessor::v_wmma_f32_16x16x16_f16`, over the 32 lanes' VGPR state. The
/// interpreter ignores EXEC here, so all lanes are read and written.
fn wmma_apply(vdst: usize, a: usize, b: usize, c: usize, vgprs: &mut [Vec<u32>]) {
    let lanes = vgprs.len().min(WAVE);
    let frag_f16 = |lane: usize, base: usize, m: usize| -> f32 {
        let word = vgprs[lane][base + m / 2];
        let bits = if m % 2 == 0 { (word & 0xffff) as u16 } else { (word >> 16) as u16 };
        f16::from_bits(bits).to_f32()
    };

    let mut mat_a = [0f32; 256];
    let mut mat_b = [0f32; 256];
    let mut mat_c = [0f32; 256];
    for e in 0..lanes {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let elem = k + j * 2 + i * 4;
                    let col = (k + j * 2 + i * 8) + (e / 16) * 4;
                    let row = e % 16;
                    mat_a[row * 16 + col] = frag_f16(e, a, elem);
                    let row_b = (k + j * 2 + i * 8) + (e / 16) * 4;
                    let col_b = e % 16;
                    mat_b[row_b * 16 + col_b] = frag_f16(e, b, elem);
                }
            }
        }
        for m in 0..8 {
            let row = m + (e / 16) * 8;
            let col = e % 16;
            mat_c[row * 16 + col] = f32::from_bits(vgprs[e][c + m]);
        }
    }

    let mut mat_d = [0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = mat_c[i * 16 + j];
            for k in 0..16 {
                acc += mat_a[i * 16 + k] * mat_b[k * 16 + j];
            }
            mat_d[i * 16 + j] = acc;
        }
    }

    for e in 0..lanes {
        for m in 0..8 {
            let row = m + (e / 16) * 8;
            let col = e % 16;
            vgprs[e][vdst + m] = mat_d[row * 16 + col].to_bits();
        }
    }
}

fn apply_xlane(op: &XlaneOp, sgprs: &mut [[u32; COOP_SGPR_BUF]], vgprs: &mut [Vec<u32>]) {
    match op {
        XlaneOp::ReadLane { vdst, src0, src1 } => {
            let lane = (eval_scalar(sgprs, src1) & 0x1f) as usize;
            let value = eval_vector(sgprs, vgprs, lane, src0);
            for s in sgprs.iter_mut() {
                write_sgpr(s, *vdst as usize, value);
            }
        }
        XlaneOp::WriteLane { vdst, src0, src1 } => {
            let value = eval_scalar(sgprs, src0);
            let lane = (eval_scalar(sgprs, src1) & 0x1f) as usize;
            if lane < vgprs.len() {
                vgprs[lane][*vdst as usize] = value;
            }
        }
        XlaneOp::Wmma { vdst, a, b, c } => {
            wmma_apply(*vdst as usize, *a as usize, *b as usize, *c as usize, vgprs);
        }
    }
}

fn packet_vgpr(vgprs: &[Vec<u32>], width: usize, lane: usize, reg: usize) -> u32 {
    let packet = lane / width;
    let packet_lane = lane % width;
    vgprs[packet][reg * width + packet_lane]
}

fn set_packet_vgpr(
    vgprs: &mut [Vec<u32>],
    width: usize,
    lane: usize,
    reg: usize,
    value: u32,
) {
    let packet = lane / width;
    let packet_lane = lane % width;
    vgprs[packet][reg * width + packet_lane] = value;
}

trait PacketLayout: Copy {
    fn get(self, vgprs: &[Vec<u32>], lane: usize, reg: usize) -> u32;
    fn set(self, vgprs: &mut [Vec<u32>], lane: usize, reg: usize, value: u32);
}

#[derive(Clone, Copy)]
struct FixedPacketLayout<const WIDTH: usize>;

impl<const WIDTH: usize> PacketLayout for FixedPacketLayout<WIDTH> {
    #[inline(always)]
    fn get(self, vgprs: &[Vec<u32>], lane: usize, reg: usize) -> u32 {
        vgprs[lane / WIDTH][reg * WIDTH + lane % WIDTH]
    }

    #[inline(always)]
    fn set(self, vgprs: &mut [Vec<u32>], lane: usize, reg: usize, value: u32) {
        vgprs[lane / WIDTH][reg * WIDTH + lane % WIDTH] = value;
    }
}

#[derive(Clone, Copy)]
struct DynamicPacketLayout(usize);

impl PacketLayout for DynamicPacketLayout {
    #[inline(always)]
    fn get(self, vgprs: &[Vec<u32>], lane: usize, reg: usize) -> u32 {
        packet_vgpr(vgprs, self.0, lane, reg)
    }

    #[inline(always)]
    fn set(self, vgprs: &mut [Vec<u32>], lane: usize, reg: usize, value: u32) {
        set_packet_vgpr(vgprs, self.0, lane, reg, value);
    }
}

#[inline(always)]
fn unpack_f16_fragments(
    mut read_a: impl FnMut(usize) -> u32,
    mut read_b: impl FnMut(usize) -> u32,
) -> [f32; 16] {
    let mut halves = [f16::from_bits(0); 16];
    for word_index in 0..4 {
        let word_a = read_a(word_index);
        let word_b = read_b(word_index);
        halves[word_index * 2] = f16::from_bits(word_a as u16);
        halves[word_index * 2 + 1] = f16::from_bits((word_a >> 16) as u16);
        halves[8 + word_index * 2] = f16::from_bits(word_b as u16);
        halves[8 + word_index * 2 + 1] = f16::from_bits((word_b >> 16) as u16);
    }

    let mut values = [0.0; 16];
    halves.convert_to_f32_slice(&mut values);
    values
}

fn eval_vector_packets(
    sgprs: &[[u32; COOP_SGPR_BUF]],
    vgprs: &[Vec<u32>],
    width: usize,
    lane: usize,
    op: &SourceOperand,
) -> u32 {
    match op {
        SourceOperand::LiteralConstant(v) => *v,
        SourceOperand::IntegerConstant(v) => *v as u32,
        SourceOperand::FloatConstant(v) => (*v as f32).to_bits(),
        SourceOperand::ScalarRegister(r) => sgprs[lane / width][*r as usize],
        SourceOperand::VectorRegister(r) => packet_vgpr(vgprs, width, lane, *r as usize),
        SourceOperand::PrivateBase => panic!("vector boundary operand reads private base"),
    }
}

fn wmma_apply_packets(
    vdst: usize,
    a: usize,
    b: usize,
    c: usize,
    width: usize,
    vgprs: &mut [Vec<u32>],
) {
    match width {
        1 => wmma_apply_packets_layout(FixedPacketLayout::<1>, vdst, a, b, c, vgprs),
        2 => wmma_apply_packets_layout(FixedPacketLayout::<2>, vdst, a, b, c, vgprs),
        4 => wmma_apply_packets_layout(FixedPacketLayout::<4>, vdst, a, b, c, vgprs),
        8 => wmma_apply_packets_layout(FixedPacketLayout::<8>, vdst, a, b, c, vgprs),
        16 => wmma_apply_packets_layout(DynamicPacketLayout(width), vdst, a, b, c, vgprs),
        _ => unreachable!("unsupported packet width {}", width),
    }
}

fn wmma_apply_packets_layout(
    layout: impl PacketLayout,
    vdst: usize,
    a: usize,
    b: usize,
    c: usize,
    vgprs: &mut [Vec<u32>],
) {
    let mut mat_a = [0f32; 256];
    let mut mat_b = [0f32; 256];
    let mut mat_c = [0f32; 256];

    for lane in 0..WAVE {
        let fragments = unpack_f16_fragments(
            |word| layout.get(vgprs, lane, a + word),
            |word| layout.get(vgprs, lane, b + word),
        );
        for elem in 0..8 {
            let matrix_index = elem + (elem / 4) * 4 + (lane / 16) * 4;
            mat_a[(lane % 16) * 16 + matrix_index] = fragments[elem];
            mat_b[matrix_index * 16 + lane % 16] = fragments[8 + elem];
        }
        for m in 0..8 {
            let row = m + (lane / 16) * 8;
            let col = lane % 16;
            mat_c[row * 16 + col] = f32::from_bits(layout.get(vgprs, lane, c + m));
        }
    }

    let mut mat_d = [0f32; 256];
    for row in 0..16 {
        for col in 0..16 {
            let mut acc = mat_c[row * 16 + col];
            for k in 0..16 {
                acc += mat_a[row * 16 + k] * mat_b[k * 16 + col];
            }
            mat_d[row * 16 + col] = acc;
        }
    }

    for lane in 0..WAVE {
        for m in 0..8 {
            let row = m + (lane / 16) * 8;
            let col = lane % 16;
            layout.set(vgprs, lane, vdst + m, mat_d[row * 16 + col].to_bits());
        }
    }
}

fn apply_xlane_packets(
    op: &XlaneOp,
    width: usize,
    sgprs: &mut [[u32; COOP_SGPR_BUF]],
    vgprs: &mut [Vec<u32>],
) {
    match op {
        XlaneOp::ReadLane { vdst, src0, src1 } => {
            let lane = (eval_scalar(sgprs, src1) & 0x1f) as usize;
            let value = eval_vector_packets(sgprs, vgprs, width, lane, src0);
            for packet_sgprs in sgprs.iter_mut() {
                write_sgpr(packet_sgprs, *vdst as usize, value);
            }
        }
        XlaneOp::WriteLane { vdst, src0, src1 } => {
            let value = eval_scalar(sgprs, src0);
            let lane = (eval_scalar(sgprs, src1) & 0x1f) as usize;
            set_packet_vgpr(vgprs, width, lane, *vdst as usize, value);
        }
        XlaneOp::Wmma { vdst, a, b, c } => {
            wmma_apply_packets(
                *vdst as usize,
                *a as usize,
                *b as usize,
                *c as usize,
                width,
                vgprs,
            );
        }
    }
}

/// Run a cross-lane cooperative kernel over the whole grid, one 32-lane wavefront
/// at a time. `xlane` maps each yield's resume pc to the wave-level op to apply
/// there (built by [`split_at_xlane`]).
pub fn dispatch_xlane(
    kernel: &CoopKernel,
    xlane: &BTreeMap<usize, XlaneOp>,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    let num_threads = num_threads.max(1);

    let wg_size = dims.workgroup_size() as usize;
    let waves_per_wg = (wg_size + WAVE - 1) / WAVE;
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let total_waves = num_wg * waves_per_wg as u64;
    let num_vgprs = kernel.num_vgprs.max(1);
    let scratch_u64 = (private_segment_size as usize / 8) + 2;
    let entry_pc = kernel.entry_pc as u64;

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let kernel = &kernel;
            let kd = &kd;
            let dims = dims;
            scope.spawn(move || {
                let mut sgprs: Vec<[u32; COOP_SGPR_BUF]> = vec![[0u32; COOP_SGPR_BUF]; WAVE];
                let mut vgprs: Vec<Vec<u32>> = vec![vec![0u32; num_vgprs]; WAVE];
                let mut scratch: Vec<Vec<u64>> = vec![vec![0u64; scratch_u64]; WAVE];
                let mut spill: Vec<[u32; COOP_SPILL_SLOTS]> = vec![[0u32; COOP_SPILL_SLOTS]; WAVE];
                let mut resume: Vec<u64> = vec![0; WAVE];
                let mut done: Vec<bool> = vec![true; WAVE];
                // This backend targets kernels without LDS; a zeroed dummy buffer
                // satisfies the CoopKernel signature's `lds` parameter.
                let lds = [0u8; 16];
                let lds_base = lds.as_ptr() as u64;

                let mut wave = tid as u64;
                while wave < total_waves {
                    let wg = wave / waves_per_wg as u64;
                    let wave_in_wg = (wave % waves_per_wg as u64) as usize;
                    let local_base = wave_in_wg * WAVE;
                    let wg_id = (
                        (wg % dims.num_wg_x as u64) as u32,
                        ((wg / dims.num_wg_x as u64) % dims.num_wg_y as u64) as u32,
                        ((wg / (dims.num_wg_x as u64 * dims.num_wg_y as u64)) % dims.num_wg_z as u64) as u32,
                    );

                    for lane in 0..WAVE {
                        for v in vgprs[lane].iter_mut() {
                            *v = 0;
                        }
                        for s in scratch[lane].iter_mut() {
                            *s = 0;
                        }
                        spill[lane] = [0u32; COOP_SPILL_SLOTS];
                        let local = local_base + lane;
                        if local >= wg_size {
                            done[lane] = true; // inactive tail lane of a partial wave
                            continue;
                        }
                        let scratch_base = scratch[lane].as_ptr() as u64;
                        let s = setup_sgprs(kd, kernarg_ptr, aql_packet_addr, scratch_base, private_segment_size, wg_id);
                        sgprs[lane][..128].copy_from_slice(&s);
                        sgprs[lane][EXEC] = 1;
                        sgprs[lane][SCC] = 0;
                        let lx = (local as u32) % dims.wg_x;
                        let ly = ((local as u32) / dims.wg_x) % dims.wg_y;
                        let lz = (local as u32) / (dims.wg_x * dims.wg_y);
                        vgprs[lane][0] = lx | (ly << 10) | (lz << 20);
                        resume[lane] = entry_pc;
                        done[lane] = false;
                    }

                    // Round-robin: each pass advances every live lane to its next
                    // yield (cross-lane boundary) or to s_endpgm; then the wave-level
                    // op is applied before the next pass resumes them past it.
                    let mut pass = 0u64;
                    loop {
                        let mut any = false;
                        for lane in 0..WAVE {
                            if done[lane] {
                                continue;
                            }
                            any = true;
                            let scratch_base = scratch[lane].as_ptr() as u64;
                            let r = unsafe {
                                kernel.run(
                                    sgprs[lane].as_mut_ptr(),
                                    vgprs[lane].as_mut_ptr(),
                                    scratch_base,
                                    lds_base,
                                    spill[lane].as_mut_ptr(),
                                    resume[lane],
                                )
                            };
                            if r == COOP_DONE {
                                done[lane] = true;
                            } else {
                                resume[lane] = r;
                            }
                        }
                        if !any {
                            break;
                        }
                        // All lanes still live yielded at a boundary this pass. For a
                        // uniform boundary they share one resume pc; apply its op once.
                        let mut yield_pc: Option<u64> = None;
                        for lane in 0..WAVE {
                            if done[lane] {
                                continue;
                            }
                            match yield_pc {
                                None => yield_pc = Some(resume[lane]),
                                Some(p) if p != resume[lane] => panic!(
                                    "dispatch_xlane: non-uniform cross-lane boundary (lane {} at {:#x}, others at {:#x})",
                                    lane, resume[lane], p
                                ),
                                _ => {}
                            }
                        }
                        if let Some(p) = yield_pc {
                            let op = xlane.get(&(p as usize)).unwrap_or_else(|| {
                                panic!("dispatch_xlane: yield at {:#x} has no cross-lane op", p)
                            });
                            apply_xlane(op, &mut sgprs, &mut vgprs);
                        }
                        pass += 1;
                        if pass > 1_000_000 {
                            panic!("dispatch_xlane: wave {} did not converge ({} passes)", wave, pass);
                        }
                    }

                    wave += num_threads as u64;
                }
            });
        }
    });
}

/// Packed counterpart of [`dispatch_xlane`]. Each CPU worker owns complete
/// 32-lane waves; it advances all `32 / W` packets to a lifted cross-lane
/// boundary, applies the operation once to their persisted SoA register state,
/// and resumes the packets. No packet of a wave is scheduled on another thread.
pub fn dispatch_xlane_vec(
    kernel: &CoopVecKernel,
    xlane: &BTreeMap<usize, XlaneOp>,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    let num_threads = num_threads.max(1);
    let width = kernel.width as usize;
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
    assert_eq!(WAVE % width, 0);
    let packets_per_wave = WAVE / width;

    let wg_size = dims.workgroup_size() as usize;
    let waves_per_wg = (wg_size + WAVE - 1) / WAVE;
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let total_waves = num_wg * waves_per_wg as u64;
    let scratch_u64 = (private_segment_size as usize / 8) + 2;
    let scratch_stride = (scratch_u64 * 8) as u64;

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let kernel = &kernel;
            let kd = &kd;
            let dims = dims;
            scope.spawn(move || {
                let mut sgprs = vec![[0u32; COOP_SGPR_BUF]; packets_per_wave];
                let mut vgprs = (0..packets_per_wave)
                    .map(|_| vec![0u32; kernel.num_vgprs * width])
                    .collect::<Vec<_>>();
                let mut spill = (0..packets_per_wave)
                    .map(|_| vec![0u32; COOP_SPILL_SLOTS])
                    .collect::<Vec<_>>();
                let mut resume = vec![0u64; packets_per_wave];
                let mut done = vec![true; packets_per_wave];
                let mut scratch: aligned_vec::AVec<
                    u8,
                    aligned_vec::ConstAlign<0x1_0000_0000>,
                > = aligned_vec::AVec::new(0x1_0000_0000);
                scratch.resize(scratch_u64 * WAVE * 8, 0u8);

                let mut wave = tid as u64;
                while wave < total_waves {
                    let wg = wave / waves_per_wg as u64;
                    let wave_in_wg = (wave % waves_per_wg as u64) as usize;
                    let local_base = wave_in_wg * WAVE;
                    let wg_id = (
                        (wg % dims.num_wg_x as u64) as u32,
                        ((wg / dims.num_wg_x as u64) % dims.num_wg_y as u64) as u32,
                        ((wg / (dims.num_wg_x as u64 * dims.num_wg_y as u64))
                            % dims.num_wg_z as u64) as u32,
                    );
                    scratch.fill(0);
                    let scratch_base = scratch.as_ptr() as u64;
                    let initial_sgprs = setup_sgprs(
                        kd,
                        kernarg_ptr,
                        aql_packet_addr,
                        scratch_base,
                        private_segment_size,
                        wg_id,
                    );

                    for packet in 0..packets_per_wave {
                        sgprs[packet] = [0u32; COOP_SGPR_BUF];
                        sgprs[packet][..128].copy_from_slice(&initial_sgprs);
                        sgprs[packet][SCC] = 0;
                        vgprs[packet].fill(0);
                        spill[packet].fill(0);

                        let packet_base = local_base + packet * width;
                        let valid_lanes = wg_size.saturating_sub(packet_base).min(width);
                        if valid_lanes == 0 {
                            done[packet] = true;
                            resume[packet] = COOP_DONE;
                            continue;
                        }
                        let exec = if valid_lanes == 32 {
                            u32::MAX
                        } else {
                            ((1u64 << valid_lanes) - 1) as u32
                        };
                        sgprs[packet][EXEC] = exec;
                        for packet_lane in 0..valid_lanes {
                            let local = packet_base + packet_lane;
                            let lx = (local as u32) % dims.wg_x;
                            let ly = ((local as u32) / dims.wg_x) % dims.wg_y;
                            let lz = (local as u32) / (dims.wg_x * dims.wg_y);
                            vgprs[packet][packet_lane] = lx | (ly << 10) | (lz << 20);
                        }
                        resume[packet] = kernel.entry_pc as u64;
                        done[packet] = false;
                    }

                    let mut pass = 0u64;
                    loop {
                        let mut any = false;
                        for packet in 0..packets_per_wave {
                            if done[packet] {
                                continue;
                            }
                            any = true;
                            let r = unsafe {
                                kernel.run(
                                    sgprs[packet].as_mut_ptr(),
                                    vgprs[packet].as_mut_ptr(),
                                    scratch_base,
                                    scratch_stride,
                                    spill[packet].as_mut_ptr(),
                                    resume[packet],
                                    (packet * width) as u64,
                                )
                            };
                            if r == COOP_DONE {
                                done[packet] = true;
                            } else {
                                resume[packet] = r;
                            }
                        }
                        if !any {
                            break;
                        }

                        let mut yield_pc: Option<u64> = None;
                        for packet in 0..packets_per_wave {
                            if done[packet] {
                                continue;
                            }
                            match yield_pc {
                                None => yield_pc = Some(resume[packet]),
                                Some(pc) if pc != resume[packet] => panic!(
                                    "dispatch_xlane_vec: non-uniform boundary (packet {} at {:#x}, others at {:#x})",
                                    packet,
                                    resume[packet],
                                    pc
                                ),
                                _ => {}
                            }
                        }
                        if let Some(pc) = yield_pc {
                            let op = xlane.get(&(pc as usize)).unwrap_or_else(|| {
                                panic!(
                                    "dispatch_xlane_vec: yield at {:#x} has no cross-lane op",
                                    pc
                                )
                            });
                            apply_xlane_packets(op, width, &mut sgprs, &mut vgprs);
                        }

                        pass += 1;
                        if pass > 1_000_000 {
                            panic!(
                                "dispatch_xlane_vec: wave {} did not converge ({} passes)",
                                wave,
                                pass
                            );
                        }
                    }

                    wave += num_threads as u64;
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_wmma_matches_scalar_lane_layout_at_every_supported_width() {
        const REGS: usize = 64;
        const VDST: usize = 32;
        const A: usize = 0;
        const B: usize = 4;
        const C: usize = 16;

        let mut state = vec![vec![0u32; REGS]; WAVE];
        let mut seed = 0x9e37_79b9u32;
        for lane in &mut state {
            for reg in lane {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *reg = seed;
            }
        }

        let mut scalar = state.clone();
        wmma_apply(VDST, A, B, C, &mut scalar);
        for width in [1usize, 2, 4, 8, 16] {
            let mut packets = vec![vec![0u32; REGS * width]; WAVE / width];
            for lane in 0..WAVE {
                for reg in 0..REGS {
                    set_packet_vgpr(&mut packets, width, lane, reg, state[lane][reg]);
                }
            }
            wmma_apply_packets(VDST, A, B, C, width, &mut packets);
            for lane in 0..WAVE {
                for reg in VDST..VDST + 8 {
                    assert_eq!(
                        packet_vgpr(&packets, width, lane, reg),
                        scalar[lane][reg],
                        "width={width}, lane={lane}, reg={reg}"
                    );
                }
            }
        }
    }

    #[test]
    fn lifted_cross_lane_writes_are_attached_to_resume_edges() {
        let ops = BTreeMap::from([
            (
                10,
                XlaneOp::WriteLane {
                    vdst: 7,
                    src0: SourceOperand::ScalarRegister(0),
                    src1: SourceOperand::IntegerConstant(3),
                },
            ),
            (20, XlaneOp::Wmma { vdst: 32, a: 0, b: 4, c: 8 }),
            (
                30,
                XlaneOp::ReadLane {
                    vdst: 0,
                    src0: SourceOperand::VectorRegister(1),
                    src1: SourceOperand::IntegerConstant(0),
                },
            ),
        ]);
        let writes = xlane_boundary_writes(&ops);
        assert_eq!(writes[&10], vec![7]);
        assert_eq!(writes[&20], (32..40).collect::<Vec<_>>());
        assert!(!writes.contains_key(&30));
    }
}
