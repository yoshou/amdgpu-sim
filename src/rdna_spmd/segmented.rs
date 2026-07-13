use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::instructions::I;
use crate::processor::KernelDescriptor;
use crate::rdna_instructions::{InstFormat, SourceOperand, DS, VOP3};
use crate::rdna_translator::RDNAProgram;

use super::dispatch::{setup_sgprs, GridDims};
use super::emit::{compile_program_writeback, ScalarKernel};
use super::ir::{build_scalar_program, Cond, ScalarBlock, ScalarProgram, Terminator};

const WAVE_SIZE: usize = 32;

enum SegmentStep {
    Fragment(ScalarKernel),
    Boundary(BoundaryOp),
}

#[derive(Clone)]
enum BoundaryOp {
    ReadLane(VOP3),
    WriteLane(VOP3),
    Bpermute(DS),
}

pub struct SegmentedProgram {
    steps: Vec<SegmentStep>,
    num_vgprs: usize,
}

struct WaveState {
    sgprs: Vec<[u32; 128]>,
    vgprs: Vec<Vec<u32>>,
    scratch: Vec<Vec<u64>>,
    active_lanes: usize,
}

impl SegmentedProgram {
    pub fn compile(program: &RDNAProgram, num_vgprs: usize) -> Result<Self, String> {
        let scalar = build_scalar_program(program);

        // Validate every lane-local instruction up front.
        for block in scalar.blocks.values() {
            for inst in &block.body {
                if BoundaryOp::from_inst(inst).is_none() && !is_lane_local_supported(inst) {
                    return Err(format!("unsupported opcode: {:?}", inst));
                }
            }
        }

        // Locate cross-lane boundaries. Each fragment between boundaries keeps
        // its *real* per-thread control flow (multi-block CFG, JIT-compiled by
        // `compile_program_writeback`) — we never flatten branches into EXEC
        // predication, so a lane that skips a guarded region via `s_cbranch_execz`
        // genuinely does not execute that region's memory ops. Only the cross-lane
        // op itself is lifted to a wavefront-wide `BoundaryOp`.
        let mut boundaries: Vec<(usize, usize)> = Vec::new();
        for (&pc, block) in &scalar.blocks {
            for (idx, inst) in block.body.iter().enumerate() {
                if BoundaryOp::from_inst(inst).is_some() {
                    boundaries.push((pc, idx));
                }
            }
        }

        let single_block = scalar.blocks.len() == 1
            && matches!(scalar.blocks.values().next().unwrap().term, Terminator::Return);

        let steps = if single_block {
            // Straight-line kernel: no divergent guards, so every active lane runs
            // the whole body. Split the linear stream at each cross-lane op into
            // per-thread scalar fragments (any number of boundaries).
            let body = &scalar.blocks.values().next().unwrap().body;
            build_linear_steps(body, num_vgprs)
        } else {
            // Kernel with control flow: fragments must keep their real branches so
            // guard-skipped lanes never run guarded memory operations. The current
            // implementation handles one cross-lane operation in this case.
            match boundaries.len() {
                0 => {
                    let kernel = compile_program_writeback(&scalar, num_vgprs);
                    vec![SegmentStep::Fragment(kernel)]
                }
                1 => build_single_boundary_steps(&scalar, boundaries[0], num_vgprs)?,
                _ => {
                    return Err("the segmented backend supports only one cross-lane operation \
                                when the kernel contains control flow"
                        .to_string())
                }
            }
        };

        let num_vgprs = steps
            .iter()
            .filter_map(|s| match s {
                SegmentStep::Fragment(k) => Some(k.num_vgprs),
                SegmentStep::Boundary(_) => None,
            })
            .max()
            .unwrap_or(num_vgprs.max(256));

        Ok(SegmentedProgram { steps, num_vgprs })
    }

    fn run_wave(&self, state: &mut WaveState) -> Result<(), String> {
        for step in &self.steps {
            match step {
                SegmentStep::Fragment(kernel) => {
                    for lane in 0..state.active_lanes {
                        let scratch_base = state.scratch[lane].as_mut_ptr() as u64;
                        unsafe {
                            kernel.run(
                                state.sgprs[lane].as_mut_ptr(),
                                state.vgprs[lane].as_mut_ptr(),
                                scratch_base,
                            );
                        }
                    }
                }
                SegmentStep::Boundary(op) => op.apply(state)?,
            }
        }
        Ok(())
    }
}

/// Split a straight-line body into `[Fragment, Boundary, Fragment, ...]`, one
/// fragment per maximal run of lane-local instructions between cross-lane ops.
fn build_linear_steps(body: &[InstFormat], num_vgprs: usize) -> Vec<SegmentStep> {
    let mut steps = Vec::new();
    let mut run: Vec<InstFormat> = Vec::new();
    let flush = |run: &mut Vec<InstFormat>, steps: &mut Vec<SegmentStep>| {
        if run.is_empty() {
            return;
        }
        let mut blocks = BTreeMap::new();
        blocks.insert(0, ScalarBlock { pc: 0, body: std::mem::take(run), term: Terminator::Return });
        let program = ScalarProgram { entry_pc: 0, blocks };
        steps.push(SegmentStep::Fragment(compile_program_writeback(&program, num_vgprs)));
    };
    for inst in body {
        if let Some(boundary) = BoundaryOp::from_inst(inst) {
            flush(&mut run, &mut steps);
            steps.push(SegmentStep::Boundary(boundary));
        } else {
            run.push(inst.clone());
        }
    }
    flush(&mut run, &mut steps);
    steps
}

/// Successors of a block's terminator.
fn successors(term: &Terminator) -> Vec<usize> {
    match term {
        Terminator::Return => vec![],
        Terminator::Jump(t) => vec![*t],
        Terminator::Branch { taken, fallthrough, .. } => vec![*taken, *fallthrough],
        Terminator::Barrier { resume } => vec![*resume],
    }
}

/// Blocks reachable from `start` (inclusive) within `program`.
fn reachable_from(program: &ScalarProgram, start: usize) -> std::collections::BTreeSet<usize> {
    let mut seen = std::collections::BTreeSet::new();
    let mut stack = vec![start];
    while let Some(pc) = stack.pop() {
        if !seen.insert(pc) {
            continue;
        }
        if let Some(block) = program.blocks.get(&pc) {
            for s in successors(&block.term) {
                stack.push(s);
            }
        }
    }
    seen
}

/// Split a kernel with exactly one cross-lane op into
/// `[Fragment(pre), Boundary(op), Fragment(post)]`.
///
/// Both fragments are ordinary per-thread scalar programs that retain their real
/// branches; the fragment JIT executes each lane's own control flow. `pre` is the
/// original CFG with the boundary block truncated to the instructions preceding
/// the cross-lane op, so a lane that never reached that block (its EXEC bit was
/// cleared by a guard) never runs the loads there. `post` runs the boundary
/// block's tail, re-guarded by `s_cbranch_execz` so lanes inactive at the
/// boundary skip it — reproducing that they never re-entered the guarded region.
fn build_single_boundary_steps(
    scalar: &ScalarProgram,
    boundary: (usize, usize),
    num_vgprs: usize,
) -> Result<Vec<SegmentStep>, String> {
    let (bpc, k) = boundary;
    let bblock = &scalar.blocks[&bpc];
    let op = BoundaryOp::from_inst(&bblock.body[k]).expect("boundary op at located index");

    // The current implementation requires every successor of the cross-lane
    // operation to lead directly to a return. The boundary block's remaining
    // successors must carry no side effects, so the post-fragment can terminate
    // the tail with a `Return`.
    for succ in successors(&bblock.term) {
        for pc in reachable_from(scalar, succ) {
            let b = &scalar.blocks[&pc];
            if !b.body.is_empty() || !matches!(b.term, Terminator::Return) {
                return Err(format!(
                    "segmented backend: side-effecting block {:#x} follows the cross-lane operation",
                    pc
                ));
            }
        }
    }

    // Pre-fragment: whole CFG, boundary block cut to the instructions before the
    // cross-lane op and terminated with a return.
    let mut pre_blocks = scalar.blocks.clone();
    {
        let b = pre_blocks.get_mut(&bpc).unwrap();
        b.body.truncate(k);
        b.term = Terminator::Return;
    }
    let pre = ScalarProgram { entry_pc: scalar.entry_pc, blocks: pre_blocks };
    let pre_kernel = compile_program_writeback(&pre, num_vgprs);

    // Post-fragment: the boundary block's tail, entered only when EXEC != 0.
    let tail: Vec<InstFormat> = bblock.body[k + 1..].to_vec();
    let (p_entry, p_body, p_ret) = (0usize, 1usize, 2usize);
    let mut post_blocks = BTreeMap::new();
    post_blocks.insert(p_entry, ScalarBlock {
        pc: p_entry,
        body: vec![],
        term: Terminator::Branch { cond: Cond::ExecZ, taken: p_ret, fallthrough: p_body },
    });
    post_blocks.insert(p_body, ScalarBlock { pc: p_body, body: tail, term: Terminator::Return });
    post_blocks.insert(p_ret, ScalarBlock { pc: p_ret, body: vec![], term: Terminator::Return });
    let post = ScalarProgram { entry_pc: p_entry, blocks: post_blocks };
    let post_kernel = compile_program_writeback(&post, num_vgprs);

    Ok(vec![
        SegmentStep::Fragment(pre_kernel),
        SegmentStep::Boundary(op),
        SegmentStep::Fragment(post_kernel),
    ])
}

fn is_lane_local_supported(inst: &InstFormat) -> bool {
    match inst {
        InstFormat::SOP1(i) => matches!(
            i.op,
            I::S_MOV_B32
                | I::S_MOV_B64
                | I::S_AND_SAVEEXEC_B32
                | I::S_AND_NOT1_SAVEEXEC_B32
        ),
        InstFormat::SOP2(i) => matches!(
            i.op,
            I::S_ADD_CO_I32
                | I::S_ADD_U32
                | I::S_ADD_I32
                | I::S_SUB_CO_I32
                | I::S_ADD_NC_U64
                | I::S_AND_B32
                | I::S_OR_B32
                | I::S_XOR_B32
                | I::S_AND_NOT1_B32
                | I::S_OR_NOT1_B32
                | I::S_LSHR_B32
                | I::S_LSHL_B32
                | I::S_BFM_B32
                | I::S_CSELECT_B32
        ),
        InstFormat::SOPC(i) => matches!(
            i.op,
            I::S_CMP_EQ_U32
                | I::S_CMP_LG_U32
                | I::S_CMP_GT_U32
                | I::S_CMP_LT_U32
                | I::S_CMP_GE_U32
                | I::S_CMP_LE_U32
                | I::S_CMP_EQ_U64
                | I::S_CMP_LG_U64
        ),
        InstFormat::SMEM(i) => matches!(
            i.op,
            I::S_LOAD_B32 | I::S_LOAD_B64 | I::S_LOAD_B96 | I::S_LOAD_B128 | I::S_LOAD_U16
        ),
        InstFormat::VOP1(i) => matches!(
            i.op,
            I::V_MOV_B32
                | I::V_CVT_F64_I32
                | I::V_CVT_F64_U32
                | I::V_CVT_I32_F64
                | I::V_RCP_F64
                | I::V_RSQ_F64
                | I::V_SQRT_F64
                | I::V_FRACT_F64
                | I::V_RNDNE_F64
        ),
        InstFormat::VOP2(i) => matches!(
            i.op,
            I::V_ADD_F64
                | I::V_MUL_F64
                | I::V_MAX_NUM_F64
                | I::V_MIN_NUM_F64
                | I::V_ADD_NC_U32
                | I::V_AND_B32
                | I::V_XOR_B32
                | I::V_OR_B32
                | I::V_LSHLREV_B32
                | I::V_LSHRREV_B32
                | I::V_MAX_U32
                | I::V_MIN_U32
                | I::V_LSHLREV_B64
                | I::V_CNDMASK_B32
        ),
        InstFormat::VOP3(i) => matches!(
            i.op,
            I::V_ADD_NC_U32
                | I::V_AND_B32
                | I::V_XOR_B32
                | I::V_OR_B32
                | I::V_LSHLREV_B32
                | I::V_LSHRREV_B32
                | I::V_MUL_LO_U32
                | I::V_MAX_U32
                | I::V_MIN_U32
                | I::V_LSHLREV_B64
                | I::V_ADD3_U32
                | I::V_XOR3_B32
                | I::V_XAD_U32
                | I::V_BFE_U32
                | I::V_CNDMASK_B32
                | I::V_ADD_F64
                | I::V_MUL_F64
                | I::V_FMA_F64
                | I::V_MAX_NUM_F64
                | I::V_LDEXP_F64
                | I::V_DIV_SCALE_F64
                | I::V_DIV_FIXUP_F64
                | I::V_DIV_FMAS_F64
                | I::V_TRIG_PREOP_F64
                | I::V_CMP_CLASS_F64
                | I::V_CMP_EQ_U32
                | I::V_CMPX_GT_U32
        ),
        InstFormat::VOP3SD(i) => matches!(
            i.op,
            I::V_ADD_CO_U32 | I::V_ADD_CO_CI_U32 | I::V_MAD_CO_U64_U32 | I::V_DIV_SCALE_F64
        ),
        InstFormat::VOPC(i) => matches!(
            i.op,
            I::V_CMP_EQ_U32
                | I::V_CMPX_EQ_U32
                | I::V_CMP_NE_U32
                | I::V_CMPX_NE_U32
                | I::V_CMP_GT_U32
                | I::V_CMPX_GT_U32
                | I::V_CMP_LT_U32
                | I::V_CMPX_LT_U32
                | I::V_CMP_GE_U32
                | I::V_CMPX_GE_U32
                | I::V_CMP_LE_U32
                | I::V_CMPX_LE_U32
                | I::V_CMP_LT_I32
                | I::V_CMPX_LT_I32
                | I::V_CMP_GT_I32
                | I::V_CMPX_GT_I32
        ),
        InstFormat::VOPD(i) => matches!(
            i.opx,
            I::V_DUAL_MOV_B32
                | I::V_DUAL_AND_B32
                | I::V_DUAL_ADD_NC_U32
                | I::V_DUAL_LSHLREV_B32
                | I::V_DUAL_CNDMASK_B32
        ) && matches!(
            i.opy,
            I::V_DUAL_MOV_B32
                | I::V_DUAL_AND_B32
                | I::V_DUAL_ADD_NC_U32
                | I::V_DUAL_LSHLREV_B32
                | I::V_DUAL_CNDMASK_B32
        ),
        InstFormat::VGLOBAL(i) => matches!(
            i.op,
            I::GLOBAL_LOAD_B32
                | I::GLOBAL_LOAD_B64
                | I::GLOBAL_LOAD_B96
                | I::GLOBAL_LOAD_B128
                | I::GLOBAL_STORE_B32
                | I::GLOBAL_STORE_B64
                | I::GLOBAL_STORE_B96
                | I::GLOBAL_STORE_B128
        ),
        _ => false,
    }
}

impl BoundaryOp {
    fn from_inst(inst: &InstFormat) -> Option<Self> {
        match inst {
            InstFormat::VOP3(i) if matches!(i.op, I::V_READLANE_B32) => {
                Some(BoundaryOp::ReadLane(i.clone()))
            }
            InstFormat::VOP3(i) if matches!(i.op, I::V_WRITELANE_B32) => {
                Some(BoundaryOp::WriteLane(i.clone()))
            }
            InstFormat::DS(i) if matches!(i.op, I::DS_BPERMUTE_B32) => {
                Some(BoundaryOp::Bpermute(i.clone()))
            }
            _ => None,
        }
    }

    fn apply(&self, state: &mut WaveState) -> Result<(), String> {
        match self {
            BoundaryOp::ReadLane(i) => {
                let lane = (eval_scalar_u32(state, &i.src1)? & 0x1f) as usize;
                let value = eval_vector_u32(state, lane, &i.src0)?;
                for sgprs in &mut state.sgprs {
                    write_sgpr(sgprs, i.vdst as usize, value);
                }
            }
            BoundaryOp::WriteLane(i) => {
                let value = eval_scalar_u32(state, &i.src0)?;
                let lane = (eval_scalar_u32(state, &i.src1)? & 0x1f) as usize;
                if lane < state.vgprs.len() {
                    state.vgprs[lane][i.vdst as usize] = value;
                }
            }
            // DS_BPERMUTE_B32: per-lane gather within the wavefront. Lane `e`
            // pulls data0 from the lane addressed by `(v_addr[e] + offset0) >> 2`
            // (masked to the 32-lane wave); inactive source lanes read 0. A lane
            // participates iff its EXEC bit (sgpr126) is set, so lanes that a
            // guard branched away hold their EXEC=0 and neither contribute nor
            // receive. Mirrors `RDNAProcessor::ds_bpermute_b32`.
            BoundaryOp::Bpermute(i) => {
                let addr = i.addr as usize;
                let data0 = i.data0 as usize;
                let vdst = i.vdst as usize;
                let offset0 = i.offset0 as u32;
                let active: Vec<bool> =
                    (0..WAVE_SIZE).map(|e| state.sgprs[e][126] & 1 != 0).collect();
                let values: Vec<u32> =
                    (0..WAVE_SIZE).map(|e| state.vgprs[e][data0]).collect();
                let mut results = vec![0u32; WAVE_SIZE];
                for e in 0..WAVE_SIZE {
                    if !active[e] {
                        continue;
                    }
                    let lane =
                        ((state.vgprs[e][addr].wrapping_add(offset0) >> 2) & 31) as usize;
                    results[e] = if active[lane] { values[lane] } else { 0 };
                }
                for e in 0..WAVE_SIZE {
                    if active[e] {
                        state.vgprs[e][vdst] = results[e];
                    }
                }
            }
        }
        Ok(())
    }
}

fn write_sgpr(sgprs: &mut [u32; 128], idx: usize, value: u32) {
    if idx == 124 || idx == 125 {
        return;
    }
    sgprs[idx] = value;
}

fn eval_scalar_u32(state: &WaveState, op: &SourceOperand) -> Result<u32, String> {
    match op {
        SourceOperand::LiteralConstant(v) => Ok(*v),
        SourceOperand::IntegerConstant(v) => Ok(*v as u32),
        SourceOperand::FloatConstant(v) => Ok((*v as f32).to_bits()),
        SourceOperand::ScalarRegister(r) => Ok(state.sgprs[0][*r as usize]),
        SourceOperand::VectorRegister(r) => Err(format!(
            "unsupported vector source v{} in scalar boundary operand",
            r
        )),
        SourceOperand::PrivateBase => Err("unsupported private-base scalar boundary operand".to_string()),
    }
}

fn eval_vector_u32(state: &WaveState, lane: usize, op: &SourceOperand) -> Result<u32, String> {
    if lane >= state.vgprs.len() {
        return Ok(0);
    }
    match op {
        SourceOperand::LiteralConstant(v) => Ok(*v),
        SourceOperand::IntegerConstant(v) => Ok(*v as u32),
        SourceOperand::FloatConstant(v) => Ok((*v as f32).to_bits()),
        SourceOperand::ScalarRegister(r) => Ok(state.sgprs[lane][*r as usize]),
        SourceOperand::VectorRegister(r) => Ok(state.vgprs[lane][*r as usize]),
        SourceOperand::PrivateBase => Err("unsupported private-base vector boundary operand".to_string()),
    }
}

pub fn dispatch_segmented(
    program: &SegmentedProgram,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) -> Result<(), String> {
    let wg_size = dims.workgroup_size() as u64;
    let waves_per_wg = (wg_size + WAVE_SIZE as u64 - 1) / WAVE_SIZE as u64;
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let total_waves = num_wg * waves_per_wg;
    let num_threads = num_threads.max(1);
    let scratch_u64 = (private_segment_size as usize / 8) + 2;
    let first_error = Arc::new(Mutex::new(None::<String>));

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let first_error = Arc::clone(&first_error);
            scope.spawn(move || {
                let mut wave = tid as u64;
                while wave < total_waves {
                    if first_error.lock().unwrap().is_some() {
                        return;
                    }

                    let wg = wave / waves_per_wg;
                    let local_base = (wave % waves_per_wg) * WAVE_SIZE as u64;
                    let active_lanes = (wg_size.saturating_sub(local_base)).min(WAVE_SIZE as u64) as usize;
                    let wg_id = (
                        (wg % dims.num_wg_x as u64) as u32,
                        ((wg / dims.num_wg_x as u64) % dims.num_wg_y as u64) as u32,
                        ((wg / (dims.num_wg_x as u64 * dims.num_wg_y as u64)) % dims.num_wg_z as u64) as u32,
                    );

                    let mut state = WaveState {
                        sgprs: Vec::with_capacity(WAVE_SIZE),
                        vgprs: Vec::with_capacity(WAVE_SIZE),
                        scratch: Vec::with_capacity(WAVE_SIZE),
                        active_lanes,
                    };

                    for lane in 0..WAVE_SIZE {
                        let mut scratch = vec![0u64; scratch_u64];
                        let scratch_base = scratch.as_mut_ptr() as u64;
                        let mut sgprs = setup_sgprs(
                            kd,
                            kernarg_ptr,
                            aql_packet_addr,
                            scratch_base,
                            private_segment_size,
                            wg_id,
                        );
                        sgprs[126] = if lane < active_lanes { 1 } else { 0 };

                        let local = local_base + lane as u64;
                        let lx = (local % dims.wg_x as u64) as u32;
                        let ly = ((local / dims.wg_x as u64) % dims.wg_y as u64) as u32;
                        let lz = ((local / (dims.wg_x as u64 * dims.wg_y as u64)) % dims.wg_z as u64) as u32;
                        let mut vgprs = vec![0u32; program.num_vgprs];
                        vgprs[0] = lx | (ly << 10) | (lz << 20);

                        state.sgprs.push(sgprs);
                        state.vgprs.push(vgprs);
                        state.scratch.push(scratch);
                    }

                    if let Err(e) = program.run_wave(&mut state) {
                        *first_error.lock().unwrap() = Some(e);
                        return;
                    }

                    wave += num_threads as u64;
                }
            });
        }
    });

    let error = first_error.lock().unwrap().clone();
    if let Some(e) = error {
        Err(e)
    } else {
        Ok(())
    }
}
