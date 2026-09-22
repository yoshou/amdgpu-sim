use super::analysis::{Analyses, Context};
use super::engine::{Kernel, Region, Scheduler};
use super::ir::{EffectOp, Func};
use super::pass::{BranchSelects, Dce, Driver, Idiom, Idioms, Simplify, UniformQueries};
use super::program::Program;

fn wave_passes(f: &mut Program, idioms: &[Box<dyn Idiom>]) {
    let driver = Driver::new();
    let limit = 1 + f.ir.types.len();
    let mut an = Analyses::new(Context::of(&f.registry, &f.parameter_inputs, 32));
    driver
        .fixpoint(
            &mut f.ir,
            &mut an,
            "wave",
            limit,
            &[&Idioms(idioms), &UniformQueries, &BranchSelects, &Simplify, &Dce],
        )
        .unwrap();
}

fn aligned(workgroup_x: Option<u32>, width: u32) -> bool {
    workgroup_x.map_or(true, |x| x % width == 0)
}

fn schedule(shares: &Sharing) -> Scheduler {
    if shares.barrier {
        Scheduler::Workgroup
    } else if shares.exchange {
        Scheduler::Wave
    } else if shares.group {
        Scheduler::Workgroup
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
    )
}

pub fn decode_program(arch: &str, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
    if !super::rdna4::supports(arch) {
        return Err(format!("no SPMD target supports {arch}"));
    }
    let mut program = super::rdna4::decode(entry_pc, memory)?;
    wave_passes(&mut program, &super::rdna4::dialect().idioms);
    Ok(program)
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    pub width: u32,
    pub num_vgprs: usize,
    pub workgroup_x: Option<u32>,
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

pub fn compile(program: &Program, options: CompileOptions) -> Kernel {
    assert!(
        matches!(options.width, 1 | 2 | 4 | 8 | 16 | 32),
        "unsupported packet width {}",
        options.width
    );
    let lane = super::decompile::decompile(program);
    compile_lockstep(&lane, options.num_vgprs, options.width, options.workgroup_x)
}
