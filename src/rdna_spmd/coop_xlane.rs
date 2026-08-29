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

use half::f16;

use crate::instructions::I;
use crate::processor::KernelDescriptor;
use crate::rdna_instructions::{InstFormat, SourceOperand};

use super::dispatch::{setup_sgprs, GridDims};
use super::emit::{CoopKernel, COOP_DONE, COOP_SGPR_BUF, COOP_SPILL_SLOTS};
use super::emit_vec::{BoundaryIo, CoopVecKernel, RegSet};
use super::fiber::{Fiber, KernelArgs, FIBER_DONE};
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

/// What each lifted op touches, keyed by the resume pc of its boundary. The
/// compiled kernel hands the host the `reads` before yielding and takes back
/// the `writes` afterwards; the divergence and frame analyses use the writes
/// to see which values the host produced.
fn xlane_boundary_io(xlane: &BTreeMap<usize, XlaneOp>) -> BTreeMap<usize, BoundaryIo> {
    let scalar_src = |set: &mut RegSet, op: &SourceOperand| {
        if let SourceOperand::ScalarRegister(reg) = op {
            set.add_sgpr(*reg as u32);
        }
    };
    xlane
        .iter()
        .map(|(&resume, op)| {
            let mut io = BoundaryIo::default();
            match op {
                XlaneOp::ReadLane { vdst, src0, src1 } => {
                    scalar_src(&mut io.reads, src0);
                    scalar_src(&mut io.reads, src1);
                    if let SourceOperand::VectorRegister(reg) = src0 {
                        io.reads.add_vgpr(*reg as u32);
                    }
                    io.writes.add_sgpr(*vdst as u32);
                }
                XlaneOp::WriteLane { vdst, src0, src1 } => {
                    scalar_src(&mut io.reads, src0);
                    scalar_src(&mut io.reads, src1);
                    io.reads.add_vgpr(*vdst as u32); // writes a single lane
                    io.writes.add_vgpr(*vdst as u32);
                }
                XlaneOp::Wmma { vdst, a, b, c } => {
                    for k in 0..4 {
                        io.reads.add_vgpr(*a as u32 + k);
                        io.reads.add_vgpr(*b as u32 + k);
                    }
                    for k in 0..8 {
                        io.reads.add_vgpr(*c as u32 + k);
                        io.writes.add_vgpr(*vdst as u32 + k); // whole register
                    }
                }
            }
            (resume, io)
        })
        .collect()
}

/// Compile a split cross-lane program into a width-W packet kernel, passing
/// along what the host does at each boundary (see [`xlane_boundary_io`]).
pub fn compile_xlane_vec(
    program: &ScalarProgram,
    xlane: &BTreeMap<usize, XlaneOp>,
    num_vgprs: usize,
    width: u32,
) -> CoopVecKernel {
    if xlane.values().any(|op| matches!(op, XlaneOp::Wmma { .. })) {
        // Build the boundary apply here, not on the dispatch's first yield.
        super::wmma::warm(width as usize);
    }
    super::emit_vec::compile_cooperative(program, num_vgprs, width, &xlane_boundary_io(xlane))
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
    // One fused JIT pass over the packet arrays (see [`super::wmma`]). It
    // performs the same float operations in the same order as `wmma_apply`
    // above, which the layout test below checks bit for bit.
    super::wmma::apply(vdst as u32, a as u32, b as u32, c as u32, width, vgprs);
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
    let width = kernel.width as usize;
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
    let wg_size = dims.workgroup_size() as usize;
    let scratch_u64 = (private_segment_size as usize / 8) + 2;
    let dispatch = VecDispatch {
        kernel,
        xlane,
        kd,
        kernarg_ptr,
        aql_packet_addr,
        dims,
        private_segment_size,
        width,
        packets_per_wave: WAVE / width,
        wg_size,
        waves_per_wg: (wg_size + WAVE - 1) / WAVE,
        scratch_bytes: scratch_u64 * WAVE * 8,
        scratch_stride: (scratch_u64 * 8) as u64,
    };
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let total_waves = num_wg * dispatch.waves_per_wg as u64;
    let num_threads = num_threads.max(1);

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let dispatch = &dispatch;
            scope.spawn(move || {
                let mut bufs = dispatch.new_bufs();
                let mut wave = tid as u64;
                while wave < total_waves {
                    dispatch.run_wave(wave, &mut bufs);
                    wave += num_threads as u64;
                }
            });
        }
    });
}

/// The wave-invariant half of [`dispatch_xlane_vec`].
struct VecDispatch<'a> {
    kernel: &'a CoopVecKernel,
    xlane: &'a BTreeMap<usize, XlaneOp>,
    kd: &'a KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    width: usize,
    packets_per_wave: usize,
    wg_size: usize,
    waves_per_wg: usize,
    scratch_bytes: usize,
    scratch_stride: u64,
}

/// Per-worker state, reused for every wave the worker runs.
struct WaveBufs {
    sgprs: Vec<[u32; COOP_SGPR_BUF]>,
    vgprs: Vec<Vec<u32>>,
    spill: Vec<Vec<u32>>,
    fibers: Vec<Fiber>,
    resume: Vec<u64>,
    done: Vec<bool>,
    /// 4 GiB-aligned so its low 32 bits are zero: kernels using flat-scratch
    /// addressing take the pointer's high word from SRC_PRIVATE_BASE and add a
    /// per-lane low offset, so a nonzero low word would corrupt every private
    /// pointer.
    scratch: aligned_vec::AVec<u8, aligned_vec::ConstAlign<0x1_0000_0000>>,
}

/// Stack per packet fiber. A wave allocates `32 / W` of them, so the size is
/// kept modest; [`Fiber`] guards the deepest bytes, so a kernel whose frame
/// does not fit fails loudly rather than silently.
const FIBER_STACK_BYTES: usize = 32 << 10;

impl VecDispatch<'_> {
    fn new_bufs(&self) -> WaveBufs {
        let packets = self.packets_per_wave;
        let mut scratch = aligned_vec::AVec::new(0x1_0000_0000);
        scratch.resize(self.scratch_bytes, 0u8);
        WaveBufs {
            sgprs: vec![[0u32; COOP_SGPR_BUF]; packets],
            vgprs: (0..packets).map(|_| vec![0u32; self.kernel.num_vgprs * self.width]).collect(),
            spill: (0..packets).map(|_| vec![0u32; COOP_SPILL_SLOTS]).collect(),
            fibers: (0..packets).map(|_| Fiber::new(FIBER_STACK_BYTES)).collect(),
            resume: vec![0; packets],
            done: vec![true; packets],
            scratch,
        }
    }

    /// Reset every packet of `wave` to its entry state and arm its fiber.
    fn start_wave(&self, wave: u64, bufs: &mut WaveBufs) {
        let wg = wave / self.waves_per_wg as u64;
        let local_base = (wave % self.waves_per_wg as u64) as usize * WAVE;
        let wg_id = (
            (wg % self.dims.num_wg_x as u64) as u32,
            ((wg / self.dims.num_wg_x as u64) % self.dims.num_wg_y as u64) as u32,
            ((wg / (self.dims.num_wg_x as u64 * self.dims.num_wg_y as u64))
                % self.dims.num_wg_z as u64) as u32,
        );

        bufs.scratch.fill(0);
        let scratch_base = bufs.scratch.as_ptr() as u64;
        let initial_sgprs = setup_sgprs(
            self.kd,
            self.kernarg_ptr,
            self.aql_packet_addr,
            scratch_base,
            self.private_segment_size,
            wg_id,
        );

        for packet in 0..self.packets_per_wave {
            bufs.sgprs[packet] = [0u32; COOP_SGPR_BUF];
            bufs.sgprs[packet][..128].copy_from_slice(&initial_sgprs);
            bufs.vgprs[packet].fill(0);
            bufs.spill[packet].fill(0);

            let packet_base = local_base + packet * self.width;
            let valid_lanes = self.wg_size.saturating_sub(packet_base).min(self.width);
            bufs.done[packet] = valid_lanes == 0; // tail packet of a partial wave
            if bufs.done[packet] {
                continue;
            }
            bufs.sgprs[packet][EXEC] = if valid_lanes == 32 {
                u32::MAX
            } else {
                ((1u64 << valid_lanes) - 1) as u32
            };
            for lane in 0..valid_lanes {
                let local = (packet_base + lane) as u32;
                let x = local % self.dims.wg_x;
                let y = (local / self.dims.wg_x) % self.dims.wg_y;
                let z = local / (self.dims.wg_x * self.dims.wg_y);
                bufs.vgprs[packet][lane] = x | (y << 10) | (z << 20);
            }
            bufs.fibers[packet].start(KernelArgs {
                entry: self.kernel.addr(),
                sgprs: bufs.sgprs[packet].as_mut_ptr(),
                vgprs: bufs.vgprs[packet].as_mut_ptr(),
                spill: bufs.spill[packet].as_mut_ptr(),
                scratch_base,
                scratch_stride: self.scratch_stride,
                lane_base: (packet * self.width) as u64,
            });
        }
    }

    /// Run one 32-lane wave to completion: advance every live packet to its
    /// next boundary, apply the wave-level op once there, repeat.
    fn run_wave(&self, wave: u64, bufs: &mut WaveBufs) {
        self.start_wave(wave, bufs);
        loop {
            let mut live = false;
            for packet in 0..self.packets_per_wave {
                if bufs.done[packet] {
                    continue;
                }
                live = true;
                let pc = bufs.fibers[packet].resume();
                bufs.done[packet] = pc == FIBER_DONE;
                bufs.resume[packet] = pc;
            }
            if !live {
                return;
            }
            if let Some(pc) = self.boundary_pc(wave, bufs) {
                let op = self.xlane.get(&(pc as usize)).unwrap_or_else(|| {
                    panic!("dispatch_xlane_vec: yield at {:#x} has no cross-lane op", pc)
                });
                apply_xlane_packets(op, self.width, &mut bufs.sgprs, &mut bufs.vgprs);
            }
        }
    }

    /// The boundary every live packet stopped at, or `None` once they all
    /// finished. A wave-level op is one operation over all 32 lanes, so the
    /// live packets have to agree on which boundary they reached.
    fn boundary_pc(&self, wave: u64, bufs: &WaveBufs) -> Option<u64> {
        let mut boundary = None;
        for packet in 0..self.packets_per_wave {
            if bufs.done[packet] {
                continue;
            }
            match boundary {
                None => boundary = Some(bufs.resume[packet]),
                Some(pc) if pc != bufs.resume[packet] => panic!(
                    "dispatch_xlane_vec: wave {} reached a non-uniform boundary \
                     (packet {} at {:#x}, others at {:#x})",
                    wave, packet, bufs.resume[packet], pc
                ),
                _ => {}
            }
        }
        boundary
    }
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

        // Two input sets: arbitrary bit patterns — whose f16 fragments are
        // mostly Inf/NaN/subnormal — and the same patterns forced finite by
        // clearing one exponent bit of every f16 half and f32 accumulator.
        for finite in [false, true] {
            let mut state = vec![vec![0u32; REGS]; WAVE];
            let mut seed = 0x9e37_79b9u32;
            for lane in &mut state {
                for (reg, value) in lane.iter_mut().enumerate() {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    *value = seed;
                    if finite {
                        if (A..A + 4).contains(&reg) || (B..B + 4).contains(&reg) {
                            *value &= !0x4000_4000; // f16 halves
                        } else if (C..C + 8).contains(&reg) {
                            *value &= !0x4000_0000; // f32 accumulator
                        }
                    }
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
                        let actual = packet_vgpr(&packets, width, lane, reg);
                        let expected = scalar[lane][reg];
                        if actual == expected {
                            continue;
                        }
                        // Which NaN payload survives `acc + a * b` depends on
                        // the operand order the target picks for a commutative
                        // add, so a produced NaN is only required to *be* a
                        // NaN. Every other result must match bit for bit.
                        assert!(
                            !finite
                                && f32::from_bits(actual).is_nan()
                                && f32::from_bits(expected).is_nan(),
                            "width={}, lane={}, reg={}: \
                             {:#010x} != {:#010x} (finite inputs: {})",
                            width, lane, reg, actual, expected, finite
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_io_describes_each_lifted_op() {
        let ops = BTreeMap::from([
            (
                10,
                XlaneOp::WriteLane {
                    vdst: 7,
                    src0: SourceOperand::ScalarRegister(5),
                    src1: SourceOperand::IntegerConstant(3),
                },
            ),
            (20, XlaneOp::Wmma { vdst: 32, a: 0, b: 4, c: 8 }),
            (
                30,
                XlaneOp::ReadLane {
                    vdst: 2,
                    src0: SourceOperand::VectorRegister(1),
                    src1: SourceOperand::IntegerConstant(0),
                },
            ),
        ]);
        let io = xlane_boundary_io(&ops);
        let vgprs = |set: &RegSet| set.vgprs().collect::<Vec<_>>();

        // writelane: a scalar value in, one lane of vdst out — and vdst is
        // read back too, so the lanes it does not touch survive the yield.
        assert_eq!(vgprs(&io[&10].reads), vec![7]);
        assert_eq!(vgprs(&io[&10].writes), vec![7]);
        assert!(io[&10].reads.has_sgpr(5));

        // WMMA: 4 + 4 f16 fragment registers and the 8-register accumulator
        // in, the 8-register result out.
        assert_eq!(vgprs(&io[&20].reads), (0..16).collect::<Vec<_>>());
        assert_eq!(vgprs(&io[&20].writes), (32..40).collect::<Vec<_>>());

        // readlane: one VGPR in, a uniform SGPR out.
        assert_eq!(vgprs(&io[&30].reads), vec![1]);
        assert!(vgprs(&io[&30].writes).is_empty());
        assert!(io[&30].writes.has_sgpr(2));
    }
}
