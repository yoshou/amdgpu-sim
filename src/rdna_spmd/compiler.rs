//! Decode/lift once, prepare typed SSA, then emit LLVM from the IR alone.

use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(test)]
use crate::rdna_instructions::InstFormat;
use super::target::Target;

use super::engine::kernel::{Code, CoopKernel, CoopVecKernel, Kernel, ScalarKernel, Scheduler, VecKernel};
#[cfg(test)]
use crate::rdna_spmd::targets::rdna4::decode::{self as ir, ScalarBlock, ScalarProgram};
use super::program::{Program, CompilationInput};

use super::analysis::{Accesses, Analyses, Constants, Context, Exec, Packet, Uniformity};
use super::analysis::uniformity::Fact;
use super::codegen::{Abi, Prepared};
use super::dialect::DialectRegistry;
#[cfg(test)]
use super::ir::BlockId;
use super::ir::{Func, ValueId};
use super::program::{LiftedFunction, Parameter, ParameterSource};
use super::pass::{Driver, Pass};
use super::pass::uniform_queries::UniformQueries;
use super::pass::cse::Cse;
use super::pass::{active::Active, adjacency::Adjacency, dce::{Dce, DeadParams, DeadWrites}, entry::{AssumeDispatchExec, DiscardReturn, LocalWriteLanes, PacketState}, idioms::Idioms, mask_projection::MaskProjection, narrow::Narrow, pairs::Pairs, simplify::Simplify, specialise::Specialise};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarMode { Whole, Cooperative }

/// Coordinates the SPMD compilation stages.
pub struct Compiler { target: Arc<dyn Target> }
#[cfg(test)]
impl Default for Compiler {
    fn default() -> Self { Self::for_arch("gfx1200").unwrap() }
}
impl Compiler {
    pub fn for_arch(arch: &str) -> Result<Self, String> {
        super::targets::select(arch).map(|target| Self { target }).ok_or_else(|| format!("no SPMD target supports {arch}"))
    }
}

pub(super) fn exec_index(inputs: &[Parameter], registry: &DialectRegistry) -> usize {
    let exec = registry.registers().exec;
    inputs.iter().position(|p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec)).unwrap()
}

fn context<'r>(registry: &'r DialectRegistry, inputs: &'r [Parameter], lanes: u32) -> Context<'r> {
    Context::new(registry, inputs, exec_index(inputs, registry), lanes)
}

pub(super) fn input_passes_ir(f: &mut LiftedFunction) {
    let driver = Driver::new();
    let limit = 1 + f.ir.types.len();
    let mut an = Analyses::new(context(&f.registry, &f.parameter_inputs, 32));
    driver.fixpoint(&mut f.ir, &mut an, "input", limit, &[&Idioms, &Simplify, &Dce]).unwrap();
    f.revision += 1;
}

pub(super) fn dispatch_passes_ir(f: &mut LiftedFunction, fold_masks: bool) {
    let driver = Driver::new();
    let limit = 1 + f.ir.types.len();
    let mut an = Analyses::new(context(&f.registry, &f.parameter_inputs, 32));
    let mut passes: Vec<&dyn Pass> = Vec::new();
    if fold_masks { passes.extend([&DeadWrites as &dyn Pass, &Cse, &UniformQueries, &MaskProjection]); }
    passes.extend([&Simplify as &dyn Pass, &Dce, &DeadParams]);
    driver.pipeline(&mut f.ir, &mut an, &[&DiscardReturn]).unwrap();
    driver.fixpoint(&mut f.ir, &mut an, "mask_words", limit, &passes).unwrap();
    f.revision += 1;
}

fn dead_writes(f: &mut Func, an: &mut Analyses, driver: &Driver) {
    let limit = 1 + f.types.len();
    driver.fixpoint(f, an, "dead_writes", limit, &[&DeadWrites, &Simplify, &Dce]).unwrap();
}

fn yield_layouts_ir(ir: &Func, uniform: &[bool], constants: &[Option<u64>])
    -> (BTreeMap<u64, super::engine::yields::YieldValues>, Vec<Vec<u64>>) {
    use super::ir::{Inst, EffectOp, WaveOp};
    use super::engine::yields::{Argument, YieldValues};
    let mut out: BTreeMap<u64, YieldValues> = BTreeMap::new();
    let mut groups: Vec<Vec<u64>> = Vec::new();
    for block in ir.blocks.values() {
        let mut open: Option<(usize, std::collections::BTreeSet<usize>)> = None;
        for inst in &block.insts {
            let Inst::Effect { provenance, op, inputs, outputs } = inst else { open = None; continue };
            let scheduled = *provenance & crate::rdna_spmd::ir::SCHEDULED != 0
                || matches!(op, EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait);
            if !scheduled { open = None; continue; }
            let mut layout = YieldValues::new(*op);
            layout.uniform_selector = *op == EffectOp::Wave(WaveOp::ReadLane) && uniform[inputs[1].0];
            for (index, &input) in inputs.iter().enumerate() {
                if *op == EffectOp::Wave(WaveOp::Wmma) || *op == EffectOp::Wave(WaveOp::WriteLane) && index == 2 { continue; }
                layout.arguments[index] = if let Some(k) = constants[input.0] { Argument::Constant(k as u32) }
                    else if uniform[input.0] { Argument::Uniform } else { Argument::Lane };
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
                for (v, _) in outputs { produced.insert(v.0); }
                groups[*group].push(*provenance);
            } else {
                groups.push(vec![*provenance]);
                open = joinable.then(|| (groups.len() - 1, outputs.iter().map(|(v, _)| v.0).collect()));
            }
            out.insert(*provenance, layout);
        }
    }
    (out, groups)
}

struct Bare { ir: Func, inputs: Vec<Parameter>, registry: Arc<DialectRegistry> }

fn bare(f: LiftedFunction) -> Bare { Bare { ir: f.ir, inputs: f.parameter_inputs, registry: f.registry } }

fn prepared(mut ir: Func, an: &mut Analyses, registry: &Arc<DialectRegistry>, width: Option<u32>, abi: Abi, observable_return: bool, num_vgprs: usize) -> Prepared {
    let driver = Driver::new();
    let mut passes: Vec<&dyn Pass> = vec![&PacketState];
    if abi == Abi::Cooperative { passes.push(&LocalWriteLanes); }
    if width.is_none() { passes.push(&Active); }
    if !observable_return || (width.is_none() && abi == Abi::Whole) { passes.push(&AssumeDispatchExec); }
    driver.pipeline(&mut ir, an, &passes).unwrap();
    if std::env::var("AMDGPU_SIM_PAIRS").map_or(true, |v| v != "0") {
        driver.pipeline(&mut ir, an, &[&Simplify, &Dce]).unwrap();
        driver.pipeline(&mut ir, an, &[&Pairs]).unwrap();
        let limit = 1 + ir.types.len();
        driver.fixpoint(&mut ir, an, "simplify", limit, &[&Simplify, &Dce, &DeadParams]).unwrap();
    }
    driver.pipeline(&mut ir, an, &[&Adjacency]).unwrap();
    let constants = an.get::<Constants>(&ir);
    let exec = an.get::<Exec>(&ir);
    let uniformity = an.get::<Uniformity>(&ir);
    let accesses = an.get::<Accesses>(&ir);
    let uniform = uniformity.uniform();
    let affine: BTreeMap<ValueId, u32> = uniformity.facts.iter().enumerate().filter_map(|(v, fact)| match *fact {
        Fact::Affine { stride, .. } if stride > 0 && stride <= 256 && ir.types[v] == super::ir::Ty::I64 => Some((ValueId(v), stride as u32)),
        _ => None,
    }).collect();
    let (yields, groups) = yield_layouts_ir(&ir, &uniform, &constants);
    let shapes = accesses.iter().map(|a| super::codegen::memory::shape(a, width, super::codegen::memory::global_load(a, &uniform, &affine), &constants)).collect();
    let clusters = super::codegen::memory::clusters(&ir, &accesses, width, &uniform, &affine);
    let min_private_bytes = accesses.iter().filter_map(|a| a.static_scratch_end(&constants)).max().unwrap_or(0) as usize;
    let inputs = an.context().inputs.to_vec();
    let ir = ir.verify_with(registry).expect("invalid prepared function SSA");
    Prepared {
        registry: Arc::clone(registry), ir, inputs, width, abi, observable_return,
        uniform, exec, constants, accesses, shapes, clusters, yields, groups, min_private_bytes, num_vgprs,
    }
}

pub(super) fn prepare_packet(f: LiftedFunction, width: u32, cooperative: bool, observe_return: bool, num_vgprs: usize, aligned: bool) -> Prepared {
    let Bare { mut ir, inputs, registry } = bare(f);
    let driver = Driver::new();
    let base = context(&registry, &inputs, width);
    {
        let mut an = Analyses::new(base);
        if !observe_return { driver.pipeline(&mut ir, &mut an, &[&DiscardReturn]).unwrap(); }
        dead_writes(&mut ir, &mut an, &driver);
    }
    let mut an = Analyses::new(Context { entry_full: !observe_return, exec_initial: !cooperative, packet: Some(Packet { aligned }), ..base });
    let mut passes: Vec<&dyn Pass> = vec![&Narrow];
    if std::env::var("AMDGPU_SIM_SPECIALISE").map_or(true, |v| v != "0") { passes.push(&Specialise); }
    passes.push(&Narrow);
    driver.pipeline(&mut ir, &mut an, &passes).unwrap();
    let abi = if cooperative { Abi::Cooperative } else { Abi::Whole };
    prepared(ir, &mut an, &registry, Some(width), abi, observe_return, num_vgprs)
}

pub(super) fn prepare_scalar(f: LiftedFunction, mode: ScalarMode, num_vgprs: usize) -> Prepared {
    let Bare { mut ir, inputs, registry } = bare(f);
    let driver = Driver::new();
    let base = context(&registry, &inputs, 1);
    {
        let mut an = Analyses::new(base);
        if mode == ScalarMode::Whole { driver.pipeline(&mut ir, &mut an, &[&DiscardReturn]).unwrap(); }
        dead_writes(&mut ir, &mut an, &driver);
    }
    let abi = match mode { ScalarMode::Whole => Abi::Whole, ScalarMode::Cooperative => Abi::Cooperative };
    let mut an = Analyses::new(Context { entry_full: true, exec_initial: true, ..base });
    prepared(ir, &mut an, &registry, None, abi, mode != ScalarMode::Whole, num_vgprs)
}

impl Compiler {
    pub fn decode_program(&self, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
        let mut function = self.target.decode(entry_pc, memory)?;
        input_passes_ir(&mut function);
        Ok(Program { function })
    }

}

/// Compile lane-local execution. General 32-lane effects are scheduled
pub(crate) fn compile_scalar(program: Program, num_vgprs: usize) -> ScalarKernel {
    let p = prepare_scalar(program.function, ScalarMode::Whole, num_vgprs.max(256));
    let group = p.group();
    let code = unsafe { super::codegen::compile(&p, "scalar_kernel", super::jit::Mode::Scalar) };
    ScalarKernel::from_code(code, p.num_vgprs, group)
}


pub(crate) fn compile_packet(program: Program, num_vgprs: usize, width: u32, workgroup_x: Option<u32>) -> VecKernel {
    let aligned = workgroup_x.map_or(true, |x| x % width == 0);
    let p = prepare_packet(program.function, width, false, false, num_vgprs.max(256), aligned);
    let group = p.group();
    let code = unsafe { super::codegen::compile(&p, "vec_kernel", super::jit::Mode::Packet) };
    VecKernel::from_code(code, p.num_vgprs, width, p.min_private_bytes, workgroup_x, group)
}


pub(crate) fn compile_cooperative_packet(program: Program, num_vgprs: usize, width: u32, workgroup_x: Option<u32>) -> CoopVecKernel {
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16 | 32));
    let program = super::program::split_at_effects(program, width >= 32);
    let aligned = workgroup_x.map_or(true, |x| x % width == 0);
    let num_vgprs = program.vgpr_count(num_vgprs);
    let p = prepare_packet(program.function, width, true, true, num_vgprs, aligned);
    let code = unsafe { super::codegen::compile(&p, "vec_kernel", super::jit::Mode::Packet) };
    let yields = p.resume_layouts();
    if yields.iter().flatten().any(|l| l.op == super::ir::EffectOp::Wave(super::ir::WaveOp::Wmma)) { super::engine::wmma::warm(width as usize); }
    CoopVecKernel::from_code(code, yields, p.num_vgprs, width, p.min_private_bytes, workgroup_x, p.registry.registers())
}

/// Compile a program whose scheduled effects yield to the cooperative scheduler.
pub(crate) fn compile_cooperative_scalar(program: Program, num_vgprs: usize) -> CoopKernel {
    let program = super::program::split_at_effects(program, false);
    let num_vgprs = program.vgpr_count(num_vgprs);
    let p = prepare_scalar(program.function, ScalarMode::Cooperative, num_vgprs);
    let code = unsafe { super::codegen::compile(&p, "scalar_kernel", super::jit::Mode::Scalar) };
    let yields = p.resume_layouts();
    if yields.iter().flatten().any(|l| l.op == super::ir::EffectOp::Wave(super::ir::WaveOp::Wmma)) { super::engine::wmma::warm(1); }
    CoopKernel::from_code(code, yields, p.num_vgprs, 1, p.min_private_bytes, None, p.registry.registers())
}

pub fn decode_program(arch: &str, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
    Compiler::for_arch(arch)?.decode_program(entry_pc, memory)
}

#[cfg(test)]
impl Compiler {
    pub(crate) fn compile_program(&self, program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel { compile_scalar(program.to_ssa(), num_vgprs) }
    pub(crate) fn compile_program_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> VecKernel { compile_packet(program.to_ssa(), num_vgprs, width, None) }
    pub(crate) fn compile_cooperative_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> CoopVecKernel { compile_cooperative_packet(program.to_ssa(), num_vgprs, width, None) }
    pub(crate) fn compile_cooperative(&self, program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel { compile_cooperative_scalar(program.to_ssa(), num_vgprs) }
}

const WAVE: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    pub width: u32,
    pub num_vgprs: usize,
    pub workgroup_x: Option<u32>,
}

struct Sharing { barrier: bool, group: bool, exchange: bool }

fn sharing(program: &Program, whole_wave: bool) -> Sharing {
    use super::ir::{EffectOp, Inst, Space, WaveOp};
    let f = &program.function.ir;
    let constants = Analyses::new(context(&program.function.registry, &program.function.parameter_inputs, 32)).get::<Constants>(f);
    let mut out = Sharing { barrier: false, group: false, exchange: false };
    for block in f.blocks.values() {
        for inst in &block.insts {
            let Inst::Effect { op, inputs, .. } = inst else { continue; };
            match op {
                EffectOp::Memory { space: Space::Lds, .. } => out.group = true,
                EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => out.barrier = true,
                EffectOp::Wave(WaveOp::Any) => {}
                EffectOp::Wave(WaveOp::Ballot | WaveOp::ReadFirstLane) if whole_wave => {}
                _ => out.exchange |= super::program::exchange(op, inputs, &constants),
            }
        }
    }
    out
}

impl Compiler {
    pub fn compile(&self, program: &impl CompilationInput, options: CompileOptions) -> Kernel { compile(program, options) }
}

pub fn compile(program: &impl CompilationInput, options: CompileOptions) -> Kernel {
    assert!(matches!(options.width, 0 | 1 | 2 | 4 | 8 | 16 | 32), "unsupported packet width {}", options.width);
    let program = program.to_ssa();
    let mut folded = program.clone();
    dispatch_passes_ir(&mut folded.function, true);
    let sharing = sharing(&folded, false);
    let width = options.width;
    let program = if width >= WAVE || !(sharing.barrier || sharing.exchange) {
        let mut program = program;
        dispatch_passes_ir(&mut program.function, false);
        program
    } else { folded };
    let sharing = if width >= WAVE { self::sharing(&program, true) } else { sharing };
    let cooperative = |program: Program| if width == 0 {
        compile_cooperative_scalar(program, options.num_vgprs)
    } else {
        compile_cooperative_packet(program, options.num_vgprs, width, options.workgroup_x)
    };
    let alone = |program: Program| if width == 0 {
        Code::Scalar(compile_scalar(program, options.num_vgprs))
    } else {
        Code::Packet(compile_packet(program, options.num_vgprs, width, options.workgroup_x))
    };
    let (scheduler, code) = if sharing.barrier {
        (Scheduler::Workgroup, Code::Cooperative(cooperative(program)))
    } else if sharing.exchange {
        (Scheduler::Wave, Code::Cooperative(cooperative(program)))
    } else if sharing.group {
        (Scheduler::Workgroup, alone(program))
    } else {
        (Scheduler::Independent, alone(program))
    };
    Kernel::new(code, scheduler, width)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{SourceOperand, SOPP, VOP1};
    use crate::rdna_spmd::targets::rdna4::decode::Terminator;
    use super::super::ir::{Inst, Term, Op};

    fn mov(value: u32) -> InstFormat {
        InstFormat::VOP1(VOP1 {
            src0: SourceOperand::LiteralConstant(value), op: I::V_MOV_B32, vdst: 1,
        })
    }

    fn control(op: I) -> InstFormat {
        InstFormat::SOPP(SOPP { simm16: 0, op })
    }

    fn prepared(insts: Vec<InstFormat>, next: &[usize]) -> Program {
        let source = ir::lower_block(4, &insts, next);
        let mut blocks = BTreeMap::from([(4, source)]);
        for &pc in next { blocks.entry(pc).or_insert(ScalarBlock { pc, body: vec![], term: Terminator::Return }); }
        let source = ScalarProgram { entry_pc: 4, blocks };
        let mut program = source.to_ssa();
        input_passes_ir(&mut program.function);
        program.function.ir.clone().verify_with(&program.function.registry).unwrap();
        assert_eq!(source.blocks[&4].body.len(), insts.iter().filter(|i| !matches!(i, InstFormat::SOPP(_))).count());
        program
    }

    fn constants(f: &super::super::ir::Func, pc: usize) -> Vec<u64> {
        f.blocks[&BlockId(pc)].insts.iter().filter_map(|inst| match inst {
            Inst::Core { op: Op::Const(_, k), .. } => Some(*k),
            _ => None,
        }).collect()
    }

    #[test]
    fn normalization_preserves_writes_and_compiler_applies_dce() {
        let insts = vec![mov(1), mov(2), control(I::S_NOP), control(I::S_ENDPGM)];
        assert_eq!(ir::lower_block(4, &insts, &[]).body.len(), 2);
        let p = prepared(insts, &[]);
        assert!(constants(&p.function.ir, 4).contains(&2));
        assert!(matches!(p.function.ir.blocks[&BlockId(4)].term, Term::Ret(_)));
        assert!(constants(&p.function.ir, 4).contains(&1));
        let packet = prepare_packet(p.function, 16, false, false, 256, true);
        assert!(constants(packet.ir.func(), 4).is_empty());
    }

    #[test]
    fn preparation_preserves_fallthrough_instruction_and_branch_successors() {
        let p = prepared(vec![mov(1), mov(2)], &[8]);
        assert!(constants(&p.function.ir, 4).contains(&2));
        assert!(matches!(&p.function.ir.blocks[&BlockId(4)].term, Term::Br(e) if e.dst == BlockId(8)));
        let p = prepared(vec![mov(2), control(I::S_CBRANCH_EXECZ)], &[8, 12]);
        assert!(constants(&p.function.ir, 4).contains(&2));
        assert!(matches!(&p.function.ir.blocks[&BlockId(4)].term, Term::CondBr { yes, no, .. } if yes.dst == BlockId(12) && no.dst == BlockId(8)));
    }
}
