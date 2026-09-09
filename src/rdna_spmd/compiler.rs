//! Decode/lift once, prepare typed SSA, then emit LLVM from the IR alone.

use std::collections::BTreeMap;

#[cfg(test)]
use crate::rdna_instructions::InstFormat;

use super::engine::kernel::{CoopKernel, CoopVecKernel, ScalarKernel, VecKernel};
#[cfg(test)]
use super::decode::{self as ir, ScalarBlock, ScalarProgram};
use super::program::{Program, CompilationInput};

use super::lift::function::{self, LiftedFunction};
use super::pass::Driver;
use super::ir::BlockId;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarMode { Whole, Cooperative }

/// Coordinates the SPMD compilation stages.
pub struct Compiler { registry: std::sync::Arc<super::dialect::DialectRegistry> }
impl Default for Compiler {
    fn default() -> Self { Self { registry: std::sync::Arc::new(super::dialect::DialectRegistry::rdna4()) } }
}

pub(super) fn input_passes_ir(f: &mut LiftedFunction) {
    use super::lift::InputSource;
    let registry = f.registry.clone();
    let exec_index = f.parameter_inputs.iter().position(|p| matches!(p.source, InputSource::MaskBit(126))).unwrap();
    let mut program = super::pass::FuncProgram { ir: std::mem::replace(&mut f.ir, super::ir::Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] }), registry: &registry };
    let driver = Driver::new();
    let limit = 1 + program.ir.types.len();
    driver.fixpoint(&mut program, "idioms", limit, |p| {
        let constants = super::analysis::constants(&p.ir);
        let masks = super::analysis::masks::analyze(&registry, &p.ir, exec_index, &constants, 32, false);
        super::pass::idioms::run(&mut p.ir, &masks, &constants, &registry);
        super::pass::simplify::run(&mut p.ir);
        super::pass::dce::run(&mut p.ir);
    }).unwrap();
    f.ir = program.ir;
    f.revision += 1;
}

#[cfg(test)]
pub(super) fn packet_uniformity(f: &LiftedFunction, width: u32, entry_full: bool, aligned: bool) -> super::analysis::uniformity::Uniformity {
    use super::lift::InputSource;
    let constants = super::analysis::constants(&f.ir);
    let exec_index = f.parameter_inputs.iter().position(|p| matches!(p.source, InputSource::MaskBit(126))).unwrap();
    let masks = super::analysis::masks::analyze(&f.registry, &f.ir, exec_index, &constants, width, entry_full);
    super::analysis::uniformity::packet(&f.ir, &packet_entry(&f.ir, &f.parameter_inputs, aligned), &constants, &masks.guarded)
}

fn packet_entry(ir: &super::ir::Func, inputs: &[super::lift::Input], aligned: bool) -> super::analysis::uniformity::Entry {
    use super::lift::InputSource;
    use crate::rdna_instructions::SourceOperand;
    let entry = &ir.blocks[&ir.entry];
    let mut uniform = Vec::new();
    let mut affine = Vec::new();
    let mut varying = Vec::new();
    for (input, &(id, _)) in inputs.iter().zip(&entry.params) {
        match input.source {
            InputSource::Operand(SourceOperand::VectorRegister(0)) => if aligned { affine.push((id, 1, Some((0, 10)))) } else { varying.push(id) },
            InputSource::Operand(SourceOperand::VectorRegister(_)) | InputSource::Operand(SourceOperand::ScalarRegister(_)) | InputSource::Scc => uniform.push(id),
            _ => {}
        }
    }
    super::analysis::uniformity::Entry { uniform, affine, varying }
}

fn dead_writes(program: &mut super::pass::FuncProgram, driver: &Driver, exec_index: usize, width: u32) {
    let limit = 1 + program.ir.types.len();
    driver.fixpoint(program, "dead_writes", limit, |p| {
        let constants = super::analysis::constants(&p.ir);
        let masks = super::analysis::masks::analyze(p.registry, &p.ir, exec_index, &constants, width, false);
        super::pass::dce::dead_writes(&mut p.ir, &masks, exec_index);
        super::pass::simplify::run(&mut p.ir);
        super::pass::dce::run(&mut p.ir);
    }).unwrap();
}

fn yield_layouts_ir(ir: &super::ir::Func, uniform: &[bool], constants: &[Option<u64>]) -> BTreeMap<u64, super::engine::yields::YieldValues> {
    use super::ir::{Inst, EffectOp, WaveOp};
    use super::engine::yields::{Argument, YieldValues};
    let mut out = BTreeMap::new();
    for block in ir.blocks.values() {
        for inst in &block.insts {
            let Inst::Effect { provenance, op, inputs, .. } = inst else { continue; };
            let scheduled = *provenance & super::lift::wave::SCHEDULED != 0 || matches!(op, EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait);
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

struct Bare { ir: super::ir::Func, inputs: Vec<super::lift::Input>, registry: std::sync::Arc<super::dialect::DialectRegistry> }

fn bare(f: LiftedFunction) -> Bare { Bare { ir: f.ir, inputs: f.parameter_inputs, registry: f.registry } }

fn prepared(bare: Bare, width: Option<u32>, abi: super::codegen::Abi, observable_return: bool, entry_full: bool, initial_exec: bool, num_vgprs: usize, aligned: bool) -> super::codegen::Prepared {
    use super::analysis::uniformity::Fact;
    use super::lift::InputSource;
    let Bare { ir, inputs, registry } = bare;
    let mut program = super::pass::FuncProgram { ir, registry: &registry };
    let driver = Driver::new();
    let exec_index = inputs.iter().position(|p| matches!(p.source, InputSource::MaskBit(126))).unwrap();
    let lanes = width.unwrap_or(1);
    driver.run(&mut program, "packet_state", |p| function::packet_state(&mut p.ir)).unwrap();
    if abi == super::codegen::Abi::Cooperative {
        driver.run(&mut program, "local_write_lanes", |p| { super::pass::local_write_lanes(&mut p.ir); }).unwrap();
    }
    if width.is_none() {
        let constants = super::analysis::constants(&program.ir);
        let masks = super::analysis::masks::analyze(&registry, &program.ir, exec_index, &constants, 1, true);
        let exec = super::analysis::masks::exec(&program.ir, exec_index, &constants, 1, true, false);
        driver.run(&mut program, "active", |p| { super::pass::active::run(&mut p.ir, &masks, &exec); }).unwrap();
    }
    if !observable_return || (width.is_none() && abi == super::codegen::Abi::Whole) {
        driver.run(&mut program, "assume_dispatch_exec", |p| function::assume_dispatch_exec(&inputs, &mut p.ir)).unwrap();
    }
    if std::env::var("AMDGPU_SIM_PAIRS").map_or(true, |v| v != "0") {
        let uniform: Vec<bool> = match width {
            Some(w) => {
                let constants = super::analysis::constants(&program.ir);
                let masks = super::analysis::masks::analyze(&registry, &program.ir, exec_index, &constants, w, entry_full);
                let facts = super::analysis::uniformity::packet(&program.ir, &packet_entry(&program.ir, &inputs, aligned), &constants, &masks.guarded);
                facts.facts.iter().map(|&fact| fact == Fact::Uniform).collect()
            }
            None => vec![true; program.ir.types.len()],
        };
        driver.run(&mut program, "simplify", |p| { super::pass::simplify::run(&mut p.ir); super::pass::dce::run(&mut p.ir); }).unwrap();
        driver.run(&mut program, "pairs", |p| { super::pass::pairs::run(&mut p.ir, &uniform); }).unwrap();
        let limit = 1 + program.ir.types.len();
        driver.fixpoint(&mut program, "simplify", limit, |p| { super::pass::simplify::run(&mut p.ir); super::pass::dce::run(&mut p.ir); super::pass::dce::dead_params(&mut p.ir); }).unwrap();
    }
    let ir = program.ir;
    let constants = super::analysis::constants(&ir);
    let exec = super::analysis::masks::exec(&ir, exec_index, &constants, lanes, initial_exec, lanes > 1);
    let (uniform, affine): (Vec<bool>, BTreeMap<super::ir::ValueId, u32>) = match width {
        Some(w) => {
            let masks = super::analysis::masks::analyze(&registry, &ir, exec_index, &constants, w, entry_full);
            let facts = super::analysis::uniformity::packet(&ir, &packet_entry(&ir, &inputs, aligned), &constants, &masks.guarded);
            let uniform = facts.facts.iter().map(|&fact| fact == Fact::Uniform).collect();
            let affine = facts.facts.iter().enumerate().filter_map(|(v, fact)| match *fact {
                Fact::Affine { stride, .. } if stride > 0 && stride <= 256 && ir.types[v] == super::ir::Ty::I64 => Some((super::ir::ValueId(v), stride as u32)),
                _ => None,
            }).collect();
            (uniform, affine)
        }
        None => (vec![true; ir.types.len()], BTreeMap::new()),
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
    use super::lift::InputSource;
    let Bare { ir, inputs, registry } = bare(f);
    let mut program = super::pass::FuncProgram { ir, registry: &registry };
    let driver = Driver::new();
    let exec_index = inputs.iter().position(|p| matches!(p.source, InputSource::MaskBit(126))).unwrap();
    if !observe_return {
        driver.run(&mut program, "discard_return", |p| for block in p.ir.blocks.values_mut() {
            if let super::ir::Term::Ret(args) = &mut block.term { args.clear(); }
        }).unwrap();
    }
    dead_writes(&mut program, &driver, exec_index, width);
    let narrow = |program: &mut super::pass::FuncProgram, driver: &Driver| {
        let constants = super::analysis::constants(&program.ir);
        let masks = super::analysis::masks::analyze(&registry, &program.ir, exec_index, &constants, width, !observe_return);
        driver.run(program, "narrow", |p| { super::pass::narrow::run(&mut p.ir, &masks); }).unwrap();
    };
    narrow(&mut program, &driver);
    if std::env::var("AMDGPU_SIM_SPECIALISE").map_or(true, |v| v != "0") {
        let constants = super::analysis::constants(&program.ir);
        let masks = super::analysis::masks::analyze(&registry, &program.ir, exec_index, &constants, width, !observe_return);
        let blocks = super::pass::specialise::candidates(&program.ir, &masks, exec_index);
        if !blocks.is_empty() {
            driver.run(&mut program, "specialise", |p| { super::pass::specialise::run(&mut p.ir, &blocks, exec_index); }).unwrap();
        }
    }
    narrow(&mut program, &driver);
    let abi = if cooperative { super::codegen::Abi::Cooperative } else { super::codegen::Abi::Whole };
    prepared(Bare { ir: program.ir, inputs, registry }, Some(width), abi, observe_return, !observe_return, !cooperative, num_vgprs, aligned)
}

pub(super) fn prepare_scalar(f: LiftedFunction, mode: ScalarMode, num_vgprs: usize) -> super::codegen::Prepared {
    let Bare { ir, inputs, registry } = bare(f);
    let mut program = super::pass::FuncProgram { ir, registry: &registry };
    {
        use super::lift::InputSource;
        let driver = Driver::new();
        let exec_index = inputs.iter().position(|p| matches!(p.source, InputSource::MaskBit(126))).unwrap();
        if mode == ScalarMode::Whole {
            driver.run(&mut program, "discard_return", |p| for block in p.ir.blocks.values_mut() {
                if let super::ir::Term::Ret(args) = &mut block.term { args.clear(); }
            }).unwrap();
        }
        dead_writes(&mut program, &driver, exec_index, 1);
    }
    let abi = match mode { ScalarMode::Whole => super::codegen::Abi::Whole, ScalarMode::Cooperative => super::codegen::Abi::Cooperative };
    prepared(Bare { ir: program.ir, inputs, registry }, None, abi, mode != ScalarMode::Whole, true, true, num_vgprs, true)
}

impl Compiler {
    pub fn decode_program(&self, entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
        Ok(Program::decoded(&super::decode::program(entry_pc, memory)?, self.registry.clone()))
    }

    /// Compile lane-local execution. General 32-lane effects are scheduled
    /// with `split_at_xlane` and executed by a wave/cooperative dispatcher.
    pub fn compile_program(&self, program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel {
        let p = prepare_scalar(program.to_ssa().function, ScalarMode::Whole, num_vgprs.max(256));
        let code = unsafe { super::codegen::compile(&p, "scalar_kernel", super::jit::Mode::Scalar) };
        ScalarKernel::from_code(code, p.num_vgprs)
    }

    pub fn compile_program_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> VecKernel {
        self.compile_program_vec_layout(program, num_vgprs, width, None)
    }

    pub fn compile_program_vec_layout(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32, workgroup_x: Option<u32>) -> VecKernel {
        let aligned = workgroup_x.map_or(true, |x| x % width == 0);
        let p = prepare_packet(program.to_ssa().function, width, false, false, num_vgprs.max(256), aligned);
        let code = unsafe { super::codegen::compile(&p, "vec_kernel", super::jit::Mode::Packet) };
        VecKernel::from_code(code, p.num_vgprs, width, p.min_private_bytes, workgroup_x)
    }

    pub fn compile_cooperative_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> CoopVecKernel {
        self.compile_cooperative_vec_layout(program, num_vgprs, width, None)
    }

    pub fn compile_cooperative_vec_layout(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32, workgroup_x: Option<u32>) -> CoopVecKernel {
        assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
        let aligned = workgroup_x.map_or(true, |x| x % width == 0);
        let program = program.to_ssa();
        let num_vgprs = program.vgpr_count(num_vgprs);
        let p = prepare_packet(program.function, width, true, true, num_vgprs, aligned);
        let code = unsafe { super::codegen::compile(&p, "vec_kernel", super::jit::Mode::Packet) };
        let yields = p.resume_layouts();
        if yields.iter().any(|l| l.op == super::ir::EffectOp::Wave(super::ir::WaveOp::Wmma)) { super::engine::wmma::warm(width as usize); }
        CoopVecKernel::from_code(code, yields, p.num_vgprs, width, p.min_private_bytes, workgroup_x)
    }

    /// Compile a program whose scheduled effects yield to the cooperative scheduler.
    pub fn compile_cooperative(&self, program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel {
        let program = program.to_ssa();
        let num_vgprs = program.vgpr_count(num_vgprs);
        let p = prepare_scalar(program.function, ScalarMode::Cooperative, num_vgprs);
        let code = unsafe { super::codegen::compile(&p, "scalar_kernel", super::jit::Mode::Scalar) };
        let yields = p.resume_layouts();
        if yields.iter().any(|l| l.op == super::ir::EffectOp::Wave(super::ir::WaveOp::Wmma)) { super::engine::wmma::warm(1); }
        CoopKernel::from_code(code, yields, p.num_vgprs, 1, p.min_private_bytes, None)
    }
}

pub fn decode_program(entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
    Compiler::default().decode_program(entry_pc, memory)
}

pub fn compile_program(program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel {
    Compiler::default().compile_program(program, num_vgprs)
}

pub fn compile_program_vec(program: &impl CompilationInput, num_vgprs: usize, width: u32) -> VecKernel {
    Compiler::default().compile_program_vec(program, num_vgprs, width)
}

pub fn compile_program_vec_layout(program: &impl CompilationInput, num_vgprs: usize, width: u32, workgroup_x: u32) -> VecKernel {
    Compiler::default().compile_program_vec_layout(program, num_vgprs, width, Some(workgroup_x))
}

pub fn compile_cooperative(program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel {
    Compiler::default().compile_cooperative(program, num_vgprs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{SourceOperand, SOPP, VOP1};
    use super::super::decode::Terminator;
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
