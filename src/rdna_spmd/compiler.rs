use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use super::target::Target;

use super::engine::kernel::{
    Kernel, Region, Scheduler,
};

use super::analysis::uniformity::Fact;
use super::analysis::{
    Accesses, Analyses, Constants, Context, MaskValues, Masking, Packet, Uniformity,
};
use super::codegen::Prepared;
use super::dialect::DialectRegistry;
use super::ir::{EffectOp, Func, ValueId};
use super::pass::uniform_queries::UniformQueries;
use super::pass::{
    active::Active,
    adjacency::Adjacency,
    dce::{Dce, DeadParams},
    entry::{DiscardReturn, PacketState},
    idioms::Idioms,
    pairs::{Pairs, WideMemory},
    simplify::Simplify,
};
use super::pass::{Driver, Pass};
use super::program::{CompilationInput, Parameter, ParameterSource, Program};

pub struct Compiler {
    target: Arc<dyn Target>,
}
impl Compiler {
    pub fn for_arch(arch: &str) -> Result<Self, String> {
        super::targets::select(arch)
            .map(|target| Self { target })
            .ok_or_else(|| format!("no SPMD target supports {arch}"))
    }
}

pub(super) fn exec_index(inputs: &[Parameter], registry: &DialectRegistry) -> usize {
    let exec = registry.registers().exec;
    inputs
        .iter()
        .position(|p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec))
        .unwrap()
}

fn context<'r>(registry: &'r DialectRegistry, inputs: &'r [Parameter], lanes: u32) -> Context<'r> {
    Context::new(registry, inputs, exec_index(inputs, registry), lanes)
}

pub(super) fn wave_passes(f: &mut Program) {
    let driver = Driver::new();
    let limit = 1 + f.ir.types.len();
    let mut an = Analyses::new(context(&f.registry, &f.parameter_inputs, 32));
    driver
        .fixpoint(
            &mut f.ir,
            &mut an,
            "wave",
            limit,
            &[&Idioms, &UniformQueries, &Simplify, &Dce],
        )
        .unwrap();
    f.revision += 1;
}

fn yield_layouts_ir(
    ir: &Func,
    uniform: &[bool],
    constants: &[Option<u64>],
) -> (
    BTreeMap<u64, super::engine::yields::YieldValues>,
    Vec<Vec<u64>>,
) {
    use super::engine::yields::{Argument, YieldValues};
    use super::ir::{EffectOp, Inst, WaveOp};
    let mut out: BTreeMap<u64, YieldValues> = BTreeMap::new();
    let mut groups: Vec<Vec<u64>> = Vec::new();
    for block in ir.blocks.values() {
        let mut open: Option<(usize, std::collections::BTreeSet<usize>)> = None;
        for inst in &block.insts {
            let Inst::Effect {
                provenance,
                op,
                inputs,
                outputs,
            } = inst
            else {
                open = None;
                continue;
            };
            let scheduled = matches!(
                op,
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait
            );
            if !scheduled {
                open = None;
                continue;
            }
            let mut layout = YieldValues::new(*op);
            layout.uniform_selector =
                *op == EffectOp::Wave(WaveOp::ReadLane) && uniform[inputs[1].0];
            for (index, &input) in inputs.iter().enumerate() {
                if *op == EffectOp::Wave(WaveOp::Wmma)
                    || *op == EffectOp::Wave(WaveOp::WriteLane) && index == 2
                {
                    continue;
                }
                layout.arguments[index] = if let Some(k) = constants[input.0] {
                    Argument::Constant(k as u32)
                } else if uniform[input.0] {
                    Argument::Uniform
                } else {
                    Argument::Lane
                };
            }
            let joinable = matches!(op, EffectOp::Wave(w) if *w != WaveOp::Wmma);
            let joins = match (&open, joinable) {
                (Some((_, produced)), true) => !inputs.iter().any(|v| produced.contains(&v.0)),
                _ => false,
            };
            if joins {
                let (group, produced) = open.as_mut().unwrap();
                let last = *groups[*group].last().unwrap();
                layout.base = out[&last].base + out[&last].cells();
                for (v, _) in outputs {
                    produced.insert(v.0);
                }
                groups[*group].push(*provenance);
            } else {
                groups.push(vec![*provenance]);
                open = joinable
                    .then(|| (groups.len() - 1, outputs.iter().map(|(v, _)| v.0).collect()));
            }
            out.insert(*provenance, layout);
        }
    }
    (out, groups)
}

struct Bare {
    ir: Func,
    inputs: Vec<Parameter>,
    registry: Arc<DialectRegistry>,
}

fn bare(f: Program) -> Bare {
    Bare {
        ir: f.ir,
        inputs: f.parameter_inputs,
        registry: f.registry,
    }
}

fn prepared<M: Masking>(
    mut ir: Func,
    an: &mut Analyses,
    registry: &Arc<DialectRegistry>,
    width: Option<u32>,
    num_vgprs: usize,
) -> Prepared {
    let driver = Driver::new();
    let mut passes: Vec<&dyn Pass> = vec![&PacketState];
    if width.is_none() {
        passes.push(&Active);
    }
    driver.pipeline(&mut ir, an, &passes).unwrap();
    if std::env::var("AMDGPU_SIM_PAIRS").map_or(true, |v| v != "0") {
        driver.pipeline(&mut ir, an, &[&Simplify, &Dce]).unwrap();
        driver
            .pipeline(&mut ir, an, &[&Pairs::<M>(PhantomData)])
            .unwrap();
        let limit = 1 + ir.types.len();
        driver
            .fixpoint(
                &mut ir,
                an,
                "simplify",
                limit,
                &[&Simplify, &Dce, &DeadParams::<M>(PhantomData), &WideMemory],
            )
            .unwrap();
    }
    driver.pipeline(&mut ir, an, &[&Adjacency]).unwrap();
    let constants = an.get::<Constants>(&ir);
    let uniformity = an.get::<Uniformity<M>>(&ir);
    let accesses = an.get::<Accesses<M>>(&ir);
    let holds_a_lane = M::holds_a_lane(&ir, an, &accesses);
    let uniform = uniformity.uniform();
    let affine: BTreeMap<ValueId, u32> = uniformity
        .facts
        .iter()
        .enumerate()
        .filter_map(|(v, fact)| match *fact {
            Fact::Affine { stride, .. }
                if stride > 0 && stride <= 256 && ir.types[v] == super::ir::Ty::I64 =>
            {
                Some((ValueId(v), stride as u32))
            }
            _ => None,
        })
        .collect();
    let (yields, groups) = yield_layouts_ir(&ir, &uniform, &constants);
    let shapes = accesses
        .iter()
        .map(|a| {
            super::codegen::memory::shape(
                a,
                width,
                super::codegen::memory::global_load(a, &uniform, &affine),
                &constants,
            )
        })
        .collect();
    let clusters = super::codegen::memory::clusters(&ir, &accesses, width, &uniform, &affine);
    let min_private_bytes = accesses
        .iter()
        .filter_map(|a| a.static_scratch_end(&constants))
        .max()
        .unwrap_or(0) as usize;
    let inputs = an.context().inputs.to_vec();
    let ir = ir
        .verify_with(registry)
        .expect("invalid prepared function SSA");
    Prepared {
        registry: Arc::clone(registry),
        ir,
        inputs,
        width,
        uniform,
        holds_a_lane,
        accesses,
        shapes,
        clusters,
        yields,
        groups,
        min_private_bytes,
        num_vgprs,
    }
}

pub(super) fn prepare_lockstep(
    f: Program,
    packing: super::lockstep::Packing,
    num_vgprs: usize,
) -> Prepared {
    let Bare {
        mut ir,
        inputs,
        registry,
    } = bare(f);
    ir.lowered_to_packets();
    let driver = Driver::new();
    let mut an = Analyses::new(Context {
        exec_initial: true,
        packet: Some(Packet {
            aligned: packing.aligned,
        }),
        ..context(&registry, &inputs, packing.lanes)
    });
    driver
        .pipeline(&mut ir, &mut an, &[&DiscardReturn])
        .unwrap();
    let limit = 1 + ir.types.len();
    driver
        .fixpoint(&mut ir, &mut an, "simplify", limit, &[&Simplify, &Dce])
        .unwrap();
    prepared::<MaskValues>(
        ir,
        &mut an,
        &registry,
        Some(packing.lanes),
        num_vgprs,
    )
}

impl Compiler {
    pub fn decode_program(&self, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
        let mut program = self.target.decode(entry_pc, memory)?;
        wave_passes(&mut program);
        Ok(program)
    }
}

pub(crate) fn aligned(workgroup_x: Option<u32>, width: u32) -> bool {
    workgroup_x.map_or(true, |x| x % width == 0)
}

pub(crate) fn compile_lane(
    lane: &super::decompile::Lane,
    num_vgprs: usize,
    width: u32,
    workgroup_x: Option<u32>,
) -> Kernel {
    compile_lockstep(lane, num_vgprs, width, workgroup_x)
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

pub(crate) fn compile_lockstep(
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
    let p = prepare_lockstep(packet, packing, num_vgprs.max(256));
    let scheduler = schedule(&sharing(p.ir.func(), None));
    let yields = p.resume_layouts();
    if yields
        .iter()
        .flatten()
        .any(|l| l.op == EffectOp::Wave(super::ir::WaveOp::Wmma))
    {
        super::engine::wmma::warm(width as usize);
    }
    let compiled = super::codegen::compile_regions(&p, "vec_kernel");
    let regions = compiled
        .regions
        .into_iter()
        .map(|region| {
            let shares = sharing(p.ir.func(), Some(&region.blocks));
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
        p.registry.registers(),
        compiled.frame_words,
        p.num_vgprs,
        p.min_private_bytes,
        workgroup_x,
        scheduler,
        width,
    )
}

pub fn decode_program(arch: &str, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
    Compiler::for_arch(arch)?.decode_program(entry_pc, memory)
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

impl Compiler {
    pub fn compile(&self, program: &impl CompilationInput, options: CompileOptions) -> Kernel {
        compile(program, options)
    }
}

pub fn compile(program: &impl CompilationInput, options: CompileOptions) -> Kernel {
    assert!(
        matches!(options.width, 1 | 2 | 4 | 8 | 16 | 32),
        "unsupported packet width {}",
        options.width
    );
    let program = program.to_ssa();
    let lane = super::decompile::decompile(&program);
    compile_lane(&lane, options.num_vgprs, options.width, options.workgroup_x)
}
