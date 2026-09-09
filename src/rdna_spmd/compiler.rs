//! Decode/lift once, prepare typed SSA, then emit LLVM from the IR alone.

use std::collections::BTreeMap;

#[cfg(test)]
use crate::rdna_instructions::InstFormat;
use super::target::Target;

use super::engine::kernel::{Code, CoopKernel, CoopVecKernel, Kernel, ScalarKernel, Scheduler, VecKernel};
#[cfg(test)]
use crate::rdna_spmd::targets::rdna4::decode::{self as ir, ScalarBlock, ScalarProgram};
use super::program::{Program, CompilationInput};

use super::program::{LiftedFunction, Parameter, ParameterSource};
use super::pass::{Analyses, Context, Driver, LocalWriteLanes, Pass};
use super::pass::{active::Active, dce::{Dce, DeadParams, DeadWrites}, entry::{AssumeDispatchExec, DiscardReturn, PacketState}, idioms::Idioms, narrow::Narrow, pairs::Pairs, simplify::Simplify, specialise::Specialise};
use super::ir::BlockId;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarMode { Whole, Cooperative }

/// Coordinates the SPMD compilation stages.
pub struct Compiler { target: std::sync::Arc<dyn Target> }
#[cfg(test)]
impl Default for Compiler {
    fn default() -> Self { Self::for_arch("gfx1200").unwrap() }
}
impl Compiler {
    pub fn for_arch(arch: &str) -> Result<Self, String> {
        super::targets::select(arch).map(|target| Self { target }).ok_or_else(|| format!("no SPMD target supports {arch}"))
    }
}

pub(super) fn exec_index(inputs: &[Parameter], registry: &super::dialect::DialectRegistry) -> usize {
    let exec = registry.registers().exec;
    inputs.iter().position(|p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec)).unwrap()
}

fn analyses<'r>(registry: &'r super::dialect::DialectRegistry, exec_index: usize, lanes: u32, entry_full: bool, exec_initial: bool, exec_packed: bool, packet: Option<(&'r [Parameter], bool)>) -> Analyses<'r> {
    Analyses::new(Context { registry, exec_index, lanes, entry_full, exec_initial, exec_packed, packet })
}

pub(super) fn input_passes_ir(f: &mut LiftedFunction) {
    let registry = f.registry.clone();
    let exec_index = exec_index(&f.parameter_inputs, &registry);
    let mut program = super::pass::FuncProgram { ir: std::mem::replace(&mut f.ir, super::ir::Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] }), registry: &registry };
    let driver = Driver::new();
    let limit = 1 + program.ir.types.len();
    let mut an = analyses(&registry, exec_index, 32, false, false, false, None);
    driver.fixpoint(&mut program, &mut an, "input", limit, &[&Idioms, &Simplify, &Dce]).unwrap();
    f.ir = program.ir;
    f.revision += 1;
}

#[cfg(test)]
pub(super) fn packet_uniformity(f: &LiftedFunction, width: u32, entry_full: bool, aligned: bool) -> super::analysis::uniformity::Uniformity {
    let constants = super::analysis::constants(&f.ir);
    let exec_index = exec_index(&f.parameter_inputs, &f.registry);
    let masks = super::analysis::masks::analyze(&f.registry, &f.ir, exec_index, &constants, width, entry_full);
    super::analysis::uniformity::packet(&f.ir, &super::pass::entry::packet_entry(&f.ir, &f.parameter_inputs, aligned), &constants, &masks.guarded)
}

fn dead_writes(program: &mut super::pass::FuncProgram, driver: &Driver, exec_index: usize, width: u32) {
    let limit = 1 + program.ir.types.len();
    let mut an = analyses(program.registry, exec_index, width, false, false, false, None);
    driver.fixpoint(program, &mut an, "dead_writes", limit, &[&DeadWrites, &Simplify, &Dce]).unwrap();
}

fn yield_layouts_ir(ir: &super::ir::Func, uniform: &[bool], constants: &[Option<u64>]) -> BTreeMap<u64, super::engine::yields::YieldValues> {
    use super::ir::{Inst, EffectOp, WaveOp};
    use super::engine::yields::{Argument, YieldValues};
    let mut out = BTreeMap::new();
    for block in ir.blocks.values() {
        for inst in &block.insts {
            let Inst::Effect { provenance, op, inputs, .. } = inst else { continue; };
            let scheduled = *provenance & crate::rdna_spmd::ir::SCHEDULED != 0 || matches!(op, EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait);
            if !scheduled || *provenance & (1 << 63) != 0 { continue; }
            let mut layout = YieldValues::new(*op);
            layout.uniform_selector = *op == EffectOp::Wave(WaveOp::ReadLane) && uniform[inputs[1].0];
            for (index, &input) in inputs.iter().enumerate() {
                if *op == EffectOp::Wave(WaveOp::Wmma) || *op == EffectOp::Wave(WaveOp::WriteLane) && index == 2 { continue; }
                layout.arguments[index] = if let Some(k) = constants[input.0] { Argument::Constant(k as u32) }
                    else if uniform[input.0] { Argument::Uniform } else { Argument::Lane };
            }
            out.insert(*provenance, layout);
        }
    }
    out
}

struct Bare { ir: super::ir::Func, inputs: Vec<Parameter>, registry: std::sync::Arc<super::dialect::DialectRegistry> }

fn bare(f: LiftedFunction) -> Bare { Bare { ir: f.ir, inputs: f.parameter_inputs, registry: f.registry } }

fn prepared(bare: Bare, width: Option<u32>, abi: super::codegen::Abi, observable_return: bool, entry_full: bool, initial_exec: bool, num_vgprs: usize, aligned: bool) -> super::codegen::Prepared {
    use super::analysis::uniformity::Fact;
    let Bare { ir, inputs, registry } = bare;
    let mut program = super::pass::FuncProgram { ir, registry: &registry };
    let driver = Driver::new();
    let exec_index = exec_index(&inputs, &registry);
    let lanes = width.unwrap_or(1);
    let mut an = analyses(&registry, exec_index, lanes, entry_full, initial_exec, lanes > 1, width.map(|_| (inputs.as_slice(), aligned)));
    let assume = AssumeDispatchExec { inputs: &inputs };
    let mut passes: Vec<&dyn Pass> = vec![&PacketState];
    if abi == super::codegen::Abi::Cooperative { passes.push(&LocalWriteLanes); }
    if width.is_none() { passes.push(&Active); }
    if !observable_return || (width.is_none() && abi == super::codegen::Abi::Whole) { passes.push(&assume); }
    driver.pipeline(&mut program, &mut an, &passes).unwrap();
    if std::env::var("AMDGPU_SIM_PAIRS").map_or(true, |v| v != "0") {
        driver.pipeline(&mut program, &mut an, &[&Simplify, &Dce]).unwrap();
        driver.pipeline(&mut program, &mut an, &[&Pairs]).unwrap();
        let limit = 1 + program.ir.types.len();
        driver.fixpoint(&mut program, &mut an, "simplify", limit, &[&Simplify, &Dce, &DeadParams]).unwrap();
    }
    let ir = program.ir;
    let constants = an.constants(&ir).to_vec();
    let exec = super::analysis::masks::exec(&ir, exec_index, &constants, lanes, initial_exec, lanes > 1);
    let uniform = an.uniform(&ir).to_vec();
    let affine: BTreeMap<super::ir::ValueId, u32> = match an.uniformity(&ir) {
        Some(facts) => facts.facts.iter().enumerate().filter_map(|(v, fact)| match *fact {
            Fact::Affine { stride, .. } if stride > 0 && stride <= 256 && ir.types[v] == super::ir::Ty::I64 => Some((super::ir::ValueId(v), stride as u32)),
            _ => None,
        }).collect(),
        None => BTreeMap::new(),
    };
    let yields = yield_layouts_ir(&ir, &uniform, &constants);
    let accesses = super::analysis::memory::accesses(&ir, &constants, &uniform);
    let shapes = accesses.iter().map(|a| super::codegen::memory::shape(a, width, super::codegen::memory::global_load(a, &uniform, &affine), &constants)).collect();
    let clusters = super::codegen::memory::clusters(&ir, &accesses, width, &uniform, &affine);
    let min_private_bytes = accesses.iter().filter_map(|a| a.static_scratch_end(&constants)).max().unwrap_or(0) as usize;
    let ir = ir.verify_with(&registry).expect("invalid prepared function SSA");
    super::codegen::Prepared {
        registry, ir, inputs, width, abi, observable_return,
        uniform, exec, constants, accesses, shapes, clusters, yields, min_private_bytes, num_vgprs,
    }
}

pub(super) fn prepare_packet(f: LiftedFunction, width: u32, cooperative: bool, observe_return: bool, num_vgprs: usize, aligned: bool) -> super::codegen::Prepared {
    let Bare { ir, inputs, registry } = bare(f);
    let mut program = super::pass::FuncProgram { ir, registry: &registry };
    let driver = Driver::new();
    let exec_index = exec_index(&inputs, &registry);
    if !observe_return {
        let mut an = analyses(&registry, exec_index, width, false, false, false, None);
        driver.pipeline(&mut program, &mut an, &[&DiscardReturn]).unwrap();
    }
    dead_writes(&mut program, &driver, exec_index, width);
    let mut an = analyses(&registry, exec_index, width, !observe_return, false, false, None);
    let mut passes: Vec<&dyn Pass> = vec![&Narrow];
    if std::env::var("AMDGPU_SIM_SPECIALISE").map_or(true, |v| v != "0") { passes.push(&Specialise); }
    passes.push(&Narrow);
    driver.pipeline(&mut program, &mut an, &passes).unwrap();
    let abi = if cooperative { super::codegen::Abi::Cooperative } else { super::codegen::Abi::Whole };
    prepared(Bare { ir: program.ir, inputs, registry }, Some(width), abi, observe_return, !observe_return, !cooperative, num_vgprs, aligned)
}

pub(super) fn prepare_scalar(f: LiftedFunction, mode: ScalarMode, num_vgprs: usize) -> super::codegen::Prepared {
    let Bare { ir, inputs, registry } = bare(f);
    let mut program = super::pass::FuncProgram { ir, registry: &registry };
    {
        let driver = Driver::new();
        let exec_index = exec_index(&inputs, &registry);
        if mode == ScalarMode::Whole {
            let mut an = analyses(&registry, exec_index, 1, false, false, false, None);
            driver.pipeline(&mut program, &mut an, &[&DiscardReturn]).unwrap();
        }
        dead_writes(&mut program, &driver, exec_index, 1);
    }
    let abi = match mode { ScalarMode::Whole => super::codegen::Abi::Whole, ScalarMode::Cooperative => super::codegen::Abi::Cooperative };
    prepared(Bare { ir: program.ir, inputs, registry }, None, abi, mode != ScalarMode::Whole, true, true, num_vgprs, true)
}

impl Compiler {
    pub fn decode_program(&self, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
        let mut function = self.target.decode(entry_pc, memory)?;
        input_passes_ir(&mut function);
        Ok(Program { function })
    }

}

/// Compile lane-local execution. General 32-lane effects are scheduled
/// with `split_at_xlane` and executed by a wave/cooperative dispatcher.
pub(crate) fn compile_scalar(program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel {
    let p = prepare_scalar(program.to_ssa().function, ScalarMode::Whole, num_vgprs.max(256));
    let code = unsafe { super::codegen::compile(&p, "scalar_kernel", super::jit::Mode::Scalar) };
    ScalarKernel::from_code(code, p.num_vgprs)
}


pub(crate) fn compile_packet(program: &impl CompilationInput, num_vgprs: usize, width: u32, workgroup_x: Option<u32>) -> VecKernel {
    let aligned = workgroup_x.map_or(true, |x| x % width == 0);
    let p = prepare_packet(program.to_ssa().function, width, false, false, num_vgprs.max(256), aligned);
    let code = unsafe { super::codegen::compile(&p, "vec_kernel", super::jit::Mode::Packet) };
    VecKernel::from_code(code, p.num_vgprs, width, p.min_private_bytes, workgroup_x)
}


pub(crate) fn compile_cooperative_packet(program: &impl CompilationInput, num_vgprs: usize, width: u32, workgroup_x: Option<u32>) -> CoopVecKernel {
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
    let aligned = workgroup_x.map_or(true, |x| x % width == 0);
    let program = program.to_ssa();
    let num_vgprs = program.vgpr_count(num_vgprs);
    let p = prepare_packet(program.function, width, true, true, num_vgprs, aligned);
    let code = unsafe { super::codegen::compile(&p, "vec_kernel", super::jit::Mode::Packet) };
    let yields = p.resume_layouts();
    if yields.iter().any(|l| l.op == super::ir::EffectOp::Wave(super::ir::WaveOp::Wmma)) { super::engine::wmma::warm(width as usize); }
    CoopVecKernel::from_code(code, yields, p.num_vgprs, width, p.min_private_bytes, workgroup_x, p.registry.registers())
}

/// Compile a program whose scheduled effects yield to the cooperative scheduler.
pub(crate) fn compile_cooperative_scalar(program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel {
    let program = program.to_ssa();
    let num_vgprs = program.vgpr_count(num_vgprs);
    let p = prepare_scalar(program.function, ScalarMode::Cooperative, num_vgprs);
    let code = unsafe { super::codegen::compile(&p, "scalar_kernel", super::jit::Mode::Scalar) };
    let yields = p.resume_layouts();
    if yields.iter().any(|l| l.op == super::ir::EffectOp::Wave(super::ir::WaveOp::Wmma)) { super::engine::wmma::warm(1); }
    CoopKernel::from_code(code, yields, p.num_vgprs, 1, p.min_private_bytes, None, p.registry.registers())
}

pub fn decode_program(arch: &str, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
    Compiler::for_arch(arch)?.decode_program(entry_pc, memory)
}

#[cfg(test)]
impl Compiler {
    pub(crate) fn compile_program(&self, program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel { compile_scalar(program, num_vgprs) }
    pub(crate) fn compile_program_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> VecKernel { compile_packet(program, num_vgprs, width, None) }
    pub(crate) fn compile_cooperative_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> CoopVecKernel { compile_cooperative_packet(program, num_vgprs, width, None) }
    pub(crate) fn compile_cooperative(&self, program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel { compile_cooperative_scalar(program, num_vgprs) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    pub width: u32,
    pub num_vgprs: usize,
    pub workgroup_x: Option<u32>,
}

fn scheduler_for(program: &Program) -> Scheduler {
    use super::ir::{EffectOp, Inst, WaveOp};
    let f = &program.function.ir;
    let constants = super::analysis::constants(f);
    let known = |v: super::ir::ValueId| constants[v.0].is_some();
    let mut barrier = false;
    let mut exchange = false;
    for block in f.blocks.values() {
        for inst in &block.insts {
            let Inst::Effect { provenance, op, inputs, .. } = inst else { continue; };
            if *provenance & (1 << 63) != 0 { continue; }
            match op {
                EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => barrier = true,
                EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane) => {}
                EffectOp::Wave(WaveOp::ReadLane) => exchange |= !(known(inputs[1]) && known(inputs[2])),
                EffectOp::Wave(WaveOp::WriteLane) => exchange |= !(known(inputs[1]) && known(inputs[3])),
                EffectOp::Wave(_) => exchange = true,
                _ => {}
            }
        }
    }
    if barrier { Scheduler::Workgroup } else if exchange { Scheduler::Wave } else { Scheduler::Independent }
}

impl Compiler {
    pub fn compile(&self, program: &impl CompilationInput, options: CompileOptions) -> Kernel { compile(program, options) }
}

pub fn compile(program: &impl CompilationInput, options: CompileOptions) -> Kernel {
    assert!(matches!(options.width, 0 | 1 | 2 | 4 | 8 | 16), "unsupported packet width {}", options.width);
    let program = program.to_ssa();
    let scheduler = scheduler_for(&program);
    let cooperative = |program: &Program| if options.width == 0 {
        compile_cooperative_scalar(program, options.num_vgprs)
    } else {
        compile_cooperative_packet(program, options.num_vgprs, options.width, options.workgroup_x)
    };
    let code = match scheduler {
        Scheduler::Independent if options.width == 0 => Code::Scalar(compile_scalar(&program, options.num_vgprs)),
        Scheduler::Independent => Code::Packet(compile_packet(&program, options.num_vgprs, options.width, options.workgroup_x)),
        Scheduler::Wave => Code::Cooperative(cooperative(&super::engine::xlane::split_at_xlane(&program).0)),
        Scheduler::Workgroup => Code::Cooperative(cooperative(&super::program::split_at_barriers(&program))),
    };
    Kernel::new(code, scheduler, options.width)
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
