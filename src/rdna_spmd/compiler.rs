use super::analysis::{Analyses, Context};
use super::decompile::{decompile, Hazards, Lane};
use super::engine::{Kernel, Region, Scheduler};
use super::environment::Environment;
use super::ir::{EffectOp, Func};
use super::pass::{BranchSelects, Dce, Driver, Halves, Idiom, Idioms, PrivateSlots, Simplify, UniformQueries};
use super::program::Program;

fn wave_passes(f: &mut Program, idioms: &[Box<dyn Idiom>]) {
    let driver = Driver::new();
    let limit = 1 + f.ir.types.len();
    let mut an = Analyses::new(Context::of(&f.registry, &f.parameter_inputs, f.ir.lanes, f.entry));
    driver
        .fixpoint(
            &mut f.ir,
            &mut an,
            "wave",
            limit,
            &[&PrivateSlots, &Halves, &Idioms(idioms), &UniformQueries, &BranchSelects, &Simplify, &Dce],
        )
        .unwrap();
}

fn aligned(workgroup_x: Option<u32>, width: u32) -> bool {
    workgroup_x.map_or(true, |x| x % width == 0)
}

fn schedule(shares: &Sharing) -> Scheduler {
    if shares.barrier || shares.group {
        Scheduler::Workgroup
    } else if shares.exchange {
        Scheduler::Wave
    } else {
        Scheduler::Independent
    }
}

fn compile_lockstep(
    lane: &super::decompile::Lane,
    num_vgprs: usize,
    width: u32,
    workgroup_x: Option<u32>,
) -> Kernel {
    let packing = super::lockstep::Packing {
        lanes: width,
        aligned: aligned(workgroup_x, width),
    };
    let packet = super::lockstep::lockstep(lane, packing);
    let p = super::codegen::prepare(packet, packing, num_vgprs.max(256));
    let scheduler = schedule(&sharing(p.func(), None));
    let yields = p.resume_layouts();
    if yields
        .iter()
        .flatten()
        .any(|l| l.op == EffectOp::Wave(super::ir::WaveOp::Wmma))
    {
        super::engine::warm_wmma(width as usize);
    }
    let runtime = [(super::codegen::YIELD, super::engine::yield_address())];
    let lowerings = std::sync::Arc::new(super::rdna4::dialect().lowerings);
    let compiled = super::codegen::compile_regions(&p, lowerings, "vec_kernel", &runtime);
    let regions = compiled
        .regions
        .into_iter()
        .map(|region| {
            let shares = sharing(p.func(), Some(&region.blocks));
            Region {
                address: region.address,
                scheduler: schedule(&shares),
                children: region.children,
            }
        })
        .collect();
    Kernel::new(
        compiled.code,
        regions,
        yields,
        p.registers(),
        compiled.frame_words,
        p.num_vgprs(),
        p.min_private_bytes(),
        workgroup_x,
        scheduler,
        width,
        lane.function.ir.lanes,
        lane.function.entry,
    )
}

pub fn decode_program(
    arch: &str,
    descriptor: &crate::processor::KernelDescriptor,
    entry_pc: usize,
    memory: &[u8],
    lanes: u32,
) -> Result<Program, String> {
    if !matches!(lanes, 32 | 64) {
        return Err(format!("no SPMD target runs waves of {lanes} lanes"));
    }
    let mut program = if super::rdna4::supports(arch) {
        let mut program = super::rdna4::decode(entry_pc, memory, lanes)?;
        program.entry = super::engine::EntryLayout::of(descriptor);
        program
    } else if super::gcn3::supports(arch) {
        let (mut program, layout) = super::gcn3::decode(descriptor, entry_pc, memory, lanes)?;
        program.entry = layout;
        program
    } else {
        return Err(format!("no SPMD target supports {arch}"));
    };
    wave_passes(&mut program, &super::rdna4::dialect().idioms);
    program.ir.detach_entry();
    Ok(program)
}


struct Sharing {
    barrier: bool,
    group: bool,
    exchange: bool,
}

fn sharing(
    f: &Func,
    blocks: Option<&std::collections::BTreeSet<super::ir::BlockId>>,
) -> Sharing {
    use super::ir::{EffectOp, Inst, Space};
    let mut out = Sharing {
        barrier: false,
        group: false,
        exchange: false,
    };
    for (id, block) in &f.blocks {
        if blocks.is_some_and(|blocks| !blocks.contains(id)) {
            continue;
        }
        for inst in &block.insts {
            let Inst::Effect { op, .. } = inst else {
                continue;
            };
            match op {
                EffectOp::Memory {
                    space: Space::Lds, ..
                } => out.group = true,
                EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => out.barrier = true,
                EffectOp::Wave(_) => out.exchange = true,
                _ => {}
            }
        }
    }
    out
}

pub struct Jit {
    program: Program,
    num_vgprs: usize,
    last: Option<(Environment, usize)>,
    lanes: Vec<(Hazards, std::sync::Arc<Lane>)>,
    kernels: Vec<((usize, u32, u32), std::sync::Arc<Kernel>)>,
}

impl Jit {
    pub fn new(program: Program, num_vgprs: usize) -> Self {
        Self {
            program,
            num_vgprs,
            last: None,
            lanes: Vec::new(),
            kernels: Vec::new(),
        }
    }

    pub fn kernel(&mut self, width: u32, environment: &Environment) -> std::sync::Arc<Kernel> {
        assert!(
            matches!(width, 1 | 2 | 4 | 8 | 16 | 32),
            "unsupported packet width {}",
            width
        );
        let index = match &self.last {
            Some((last, index)) if last == environment => *index,
            _ => {
                let hazards = Hazards::find(&self.program, environment);
                let index = match self.lanes.iter().position(|(h, _)| *h == hazards) {
                    Some(index) => index,
                    None => {
                        let lane = decompile(&self.program, &hazards);
                        self.lanes.push((hazards, std::sync::Arc::new(lane)));
                        self.lanes.len() - 1
                    }
                };
                self.last = Some((environment.clone(), index));
                index
            }
        };
        let workgroup_x = environment.block[0];
        let key = (index, width, workgroup_x);
        if let Some((_, kernel)) = self.kernels.iter().find(|(k, _)| *k == key) {
            return kernel.clone();
        }
        let lane = self.lanes[index].1.clone();
        let kernel = std::sync::Arc::new(compile_lockstep(
            &lane,
            self.num_vgprs,
            width,
            Some(workgroup_x),
        ));
        self.kernels.push((key, kernel.clone()));
        kernel
    }
}
