use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::processor::KernelDescriptor;
use crate::rdna_instructions::SourceOperand;
use crate::rdna_translator::RDNAProgram;

use super::dispatch::{setup_sgprs, GridDims};
use super::compiler::{build_scalar_program, Compiler};
use super::emit::ScalarKernel;

const WAVE_SIZE: usize = 32;

enum SegmentStep {
    Fragment(ScalarKernel),
    Boundary(BoundaryOp),
}

#[derive(Clone)]
struct BoundaryOp(super::lift::wave::YieldAction, LaneAccess);

// The segmented scalar route still resolves uniform lane accesses while
// preparing its fragments. Packet fibers use SSA value frames instead.
#[derive(Clone)]
enum LaneAccess {
    ReadLane { dst: u32, value: SourceOperand, lane: SourceOperand },
    WriteLane { dst: u32, value: SourceOperand, lane: SourceOperand },
    General,
}
impl LaneAccess {
    fn lower(action: &super::lift::wave::YieldAction) -> Self {
        use super::lift::wave::{Operand, Destination};
        use super::ir::typed::effect::{EffectOp, WaveOp};
        if let [Operand::Source(value), Operand::Source(lane), ..] = action.inputs.as_slice() {
            match (action.op, action.outputs.first()) {
                (EffectOp::Wave(WaveOp::ReadLane), Some(Destination::Sgpr(dst))) if action.inputs[1].is_uniform() =>
                    return Self::ReadLane { dst: *dst, value: value.clone(), lane: lane.clone() },
                (EffectOp::Wave(WaveOp::WriteLane), Some(Destination::Vgpr(dst))) =>
                    return Self::WriteLane { dst: *dst, value: value.clone(), lane: lane.clone() },
                _ => {}
            }
        }
        Self::General
    }
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

        let boundaries:Vec<_>=scalar.function.blocks.iter().flat_map(|(&pc,b)|b.wave.iter()
            .filter(|(_, (action,_))|action.is_wave()&&action.wmma_registers().is_none()).map(move |(&index,_)|(pc,index))).collect();
        let single_block=scalar.function.blocks.len()==1 && matches!(scalar.function.ir.blocks.values().next().unwrap().term,super::ir::typed::cfg::Term::Ret);
        let steps=if single_block {
            build_linear_steps(&scalar,num_vgprs)
        }else{match boundaries.len() {
            0=>vec![SegmentStep::Fragment(Compiler::default().compile_writeback(&scalar,num_vgprs))],
            1=>build_single_boundary_steps(&scalar,boundaries[0],num_vgprs)?,
            _=>return Err("the segmented backend supports only one cross-lane operation when the kernel contains control flow".into()),
        }};

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
                    let count = state.active_lanes;
                    // Check the three frame lengths once at the boundary.
                    for ((sgprs, vgprs), scratch) in state.sgprs[..count].iter_mut()
                        .zip(&mut state.vgprs[..count]).zip(&mut state.scratch[..count]) {
                        let scratch_base = scratch.as_mut_ptr() as u64;
                        unsafe {
                            kernel.run(
                                sgprs.as_mut_ptr(),
                                vgprs.as_mut_ptr(),
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

/// Extract maximal lane-local SSA fragments around wave effects.
fn build_linear_steps(program:&super::Program,num_vgprs:usize)->Vec<SegmentStep> {
    let pc=program.function.ir.entry.0;
    let block=&program.function.blocks[&pc];
    let mut steps=Vec::new();let mut start=0;
    let compile=|start,end,steps:&mut Vec<SegmentStep>| {
        if start==end {return;}
        let mut fragment=program.fragment(&BTreeMap::from([(pc,(start..end,false))]),pc);
        if pc!=0 {fragment.rename_block(pc,0);}
        steps.push(SegmentStep::Fragment(Compiler::default().compile_writeback(&fragment,num_vgprs)));
    };
    for (&index,(action,_)) in &block.wave {
        if !action.is_wave()||action.wmma_registers().is_some() {continue;}
        compile(start,index,&mut steps);
        steps.push(SegmentStep::Boundary(BoundaryOp(action.clone(),LaneAccess::lower(action))));
        start=index+1;
    }
    compile(start,block.instructions.len(),&mut steps);
    steps
}
fn build_single_boundary_steps(program:&super::Program,boundary:(usize,usize),num_vgprs:usize)->Result<Vec<SegmentStep>,String> {
    use super::ir::typed::cfg::{BlockId,Term};
    let (pc,index)=boundary;
    let f=&program.function;
    let (action,_)=&f.blocks[&pc].wave[&index];
    let op=BoundaryOp(action.clone(),LaneAccess::lower(action));
    let mut pending:Vec<_>=f.ir.blocks[&BlockId(pc)].term.edges().iter().map(|e|e.dst.0).collect();
    let mut seen=std::collections::BTreeSet::new();
    while let Some(next)=pending.pop() {
        if !seen.insert(next) {continue;}
        if !f.blocks[&next].instructions.is_empty()||!matches!(f.ir.blocks[&BlockId(next)].term,Term::Ret) {
            return Err(format!("segmented backend: side-effecting block {:#x} follows the cross-lane operation",next));
        }
        pending.extend(f.ir.blocks[&BlockId(next)].term.edges().iter().map(|e|e.dst.0));
    }
    let ranges=f.blocks.iter().map(|(&at,b)|(at,if at==pc {(0..index,false)}else{(0..b.instructions.len(),true)})).collect();
    let pre=program.fragment(&ranges,f.ir.entry.0);
    let mut post=program.fragment(&BTreeMap::from([(pc,(index+1..f.blocks[&pc].instructions.len(),false))]),pc);
    post.guard_fragment();
    Ok(vec![SegmentStep::Fragment(Compiler::default().compile_writeback(&pre,num_vgprs)),SegmentStep::Boundary(op),
        SegmentStep::Fragment(Compiler::default().compile_writeback(&post,num_vgprs))])
}

impl BoundaryOp {
    fn apply(&self,state:&mut WaveState)->Result<(),String>{
        let valid=if state.active_lanes==32{u32::MAX}else{(1u32<<state.active_lanes)-1};
        match &self.1 {
            LaneAccess::ReadLane { dst, value, lane } => {
                let lane=(super::coop_xlane::eval_uniform(&state.sgprs,lane)&31) as usize;
                let value=if valid >> lane & 1 != 0 {eval_vector_u32(state,lane,value)?}else{0};
                for s in state.sgprs.iter_mut().take(state.active_lanes) {write_sgpr(s,*dst as usize,value);}
                return Ok(());
            }
            LaneAccess::WriteLane { dst, value, lane } => {
                let lane=(super::coop_xlane::eval_uniform(&state.sgprs,lane)&31) as usize;
                let value=super::coop_xlane::eval_uniform(&state.sgprs,value);
                if valid >> lane & 1 != 0 {state.vgprs[lane][*dst as usize]=value;}
                return Ok(());
            }
            _=>{}
        }
        if let Some((address, data, dest, offset, fi)) = self.0.bpermute_registers() {
            let active: [bool; 32] = std::array::from_fn(|lane| valid >> lane & 1 != 0 && state.sgprs[lane][126] & 1 != 0);
            let values: [u32; 32] = std::array::from_fn(|lane| if valid >> lane & 1 != 0 { state.vgprs[lane][data as usize] } else { 0 });
            let mut results = [0; 32];
            for lane in 0..32 {
                if active[lane] {
                    let source = ((state.vgprs[lane][address as usize].wrapping_add(offset) >> 2) & 31) as usize;
                    results[lane] = if fi || active[source] { values[source] } else { 0 };
                }
            }
            for lane in 0..32 { if active[lane] { state.vgprs[lane][dest as usize] = results[lane]; } }
            return Ok(());
        }
        let mut read_error=None;
        let results=self.0.evaluate(valid,|lane,source|eval_vector_u32(state,lane,source).expect("invalid wave operand"),|lane|state.sgprs[lane][126]&1!=0);
        for (dest,result) in self.0.outputs.iter().zip(results){
            for lane in 0..32{if self.0.writes_lane(lane,valid,state.sgprs[lane][126]&1!=0){match dest{
                super::lift::wave::Destination::Sgpr(r)=>write_sgpr(&mut state.sgprs[lane],*r as usize,result[lane]),
                super::lift::wave::Destination::Vgpr(r)=>state.vgprs[lane][*r as usize]=result[lane],
                super::lift::wave::Destination::Scc=>read_error=Some("segmented wave result cannot write SCC".to_string()),
            }}}
        }
        if let Some(err)=read_error{Err(err)}else{Ok(())}
    }
}

fn write_sgpr(sgprs: &mut [u32; 128], idx: usize, value: u32) {
    if idx == 124 || idx == 125 {
        return;
    }
    sgprs[idx] = value;
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
