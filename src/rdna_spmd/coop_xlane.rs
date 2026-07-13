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
