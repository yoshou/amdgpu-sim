//! Running a wave program as the lifter makes it, without reading the lane
//! program off it: every packet keeps the EXEC register's masks and answers
//! the wave's queries over its own lanes. The lifter's tests run the programs
//! they lift this way, so what they check is the lifter alone.

use super::{
    aligned, bare, compile_scalar, context, dead_writes, packet_kernel, prepared, Bare, Compiler,
};
use crate::rdna_spmd::analysis::{Analyses, Context, ExecRegister, Packet};
use crate::rdna_spmd::codegen::{Abi, Prepared};
use crate::rdna_spmd::engine::kernel::{CoopVecKernel, ScalarKernel, VecKernel};
use crate::rdna_spmd::pass::entry::DiscardReturn;
use crate::rdna_spmd::pass::narrow::Narrow;
use crate::rdna_spmd::pass::specialise::Specialise;
use crate::rdna_spmd::pass::{Driver, Pass};
use crate::rdna_spmd::program::{CompilationInput, LiftedFunction, Program};

struct PacketOptions {
    pub width: u32,
    pub cooperative: bool,
    pub observe_return: bool,
    pub num_vgprs: usize,
    pub aligned: bool,
}

fn prepare_packet(f: LiftedFunction, options: PacketOptions) -> Prepared {
    let PacketOptions {
        width,
        cooperative,
        observe_return,
        num_vgprs,
        aligned,
    } = options;
    let Bare {
        mut ir,
        inputs,
        registry,
    } = bare(f);
    ir.lowered_to_packets();
    let driver = Driver::new();
    let base = context(&registry, &inputs, width);
    {
        let mut an = Analyses::new(base);
        if !observe_return {
            driver
                .pipeline(&mut ir, &mut an, &[&DiscardReturn])
                .unwrap();
        }
        dead_writes(&mut ir, &mut an, &driver);
    }
    let mut an = Analyses::new(Context {
        entry_full: !observe_return,
        exec_initial: !cooperative,
        packet: Some(Packet { aligned }),
        ..base
    });
    let mut passes: Vec<&dyn Pass> = vec![&Narrow];
    if std::env::var("AMDGPU_SIM_SPECIALISE").map_or(true, |v| v != "0") {
        passes.push(&Specialise);
    }
    passes.push(&Narrow);
    driver.pipeline(&mut ir, &mut an, &passes).unwrap();
    let abi = if cooperative {
        Abi::Cooperative
    } else {
        Abi::Whole
    };
    prepared::<ExecRegister>(
        ir,
        &mut an,
        &registry,
        Some(width),
        abi,
        observe_return,
        num_vgprs,
    )
}

fn compile_packet(
    program: Program,
    num_vgprs: usize,
    width: u32,
    workgroup_x: Option<u32>,
) -> VecKernel {
    let p = prepare_packet(
        program.function,
        PacketOptions {
            width,
            cooperative: false,
            observe_return: false,
            num_vgprs: num_vgprs.max(256),
            aligned: aligned(workgroup_x, width),
        },
    );
    packet_kernel(p, width, workgroup_x)
}

fn compile_cooperative_packet(
    program: Program,
    num_vgprs: usize,
    width: u32,
    workgroup_x: Option<u32>,
) -> CoopVecKernel {
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16 | 32));
    let program = crate::rdna_spmd::program::split_at_effects(program, width >= 32);
    let aligned = aligned(workgroup_x, width);
    let num_vgprs = program.vgpr_count(num_vgprs);
    let p = prepare_packet(
        program.function,
        PacketOptions {
            width,
            cooperative: true,
            observe_return: true,
            num_vgprs,
            aligned,
        },
    );
    let code = crate::rdna_spmd::codegen::compile(
        &p,
        "vec_kernel",
        crate::rdna_spmd::native::jit::Mode::Packet,
    );
    let yields = p.resume_layouts();
    if yields
        .iter()
        .flatten()
        .any(|l| l.op == crate::rdna_spmd::ir::EffectOp::Wave(crate::rdna_spmd::ir::WaveOp::Wmma))
    {
        crate::rdna_spmd::engine::wmma::warm(width as usize);
    }
    CoopVecKernel::from_code(
        code,
        yields,
        p.num_vgprs,
        width,
        p.min_private_bytes,
        workgroup_x,
        p.registry.registers(),
    )
}

/// A lane run alone that yields at each scheduled effect, so what it returns
/// is observed and nothing is assumed of the EXEC register it resumes with.
fn prepare_cooperative_scalar(f: LiftedFunction, num_vgprs: usize) -> Prepared {
    let Bare {
        mut ir,
        inputs,
        registry,
    } = bare(f);
    let driver = Driver::new();
    let base = context(&registry, &inputs, 1);
    dead_writes(&mut ir, &mut Analyses::new(base), &driver);
    let mut an = Analyses::new(Context {
        entry_full: true,
        exec_initial: true,
        ..base
    });
    prepared::<ExecRegister>(
        ir,
        &mut an,
        &registry,
        None,
        Abi::Cooperative,
        true,
        num_vgprs,
    )
}

/// Compile a program whose scheduled effects yield to the cooperative scheduler.
fn compile_cooperative_scalar(program: Program, num_vgprs: usize) -> CoopVecKernel {
    let program = crate::rdna_spmd::program::split_at_effects(program, false);
    let num_vgprs = program.vgpr_count(num_vgprs);
    let p = prepare_cooperative_scalar(program.function, num_vgprs);
    let code = crate::rdna_spmd::codegen::compile(
        &p,
        "scalar_kernel",
        crate::rdna_spmd::native::jit::Mode::Scalar,
    );
    let yields = p.resume_layouts();
    if yields
        .iter()
        .flatten()
        .any(|l| l.op == crate::rdna_spmd::ir::EffectOp::Wave(crate::rdna_spmd::ir::WaveOp::Wmma))
    {
        crate::rdna_spmd::engine::wmma::warm(1);
    }
    CoopVecKernel::from_code(
        code,
        yields,
        p.num_vgprs,
        1,
        p.min_private_bytes,
        None,
        p.registry.registers(),
    )
}

impl Compiler {
    pub(crate) fn compile_program(
        &self,
        program: &impl CompilationInput,
        num_vgprs: usize,
    ) -> ScalarKernel {
        compile_scalar(program.to_ssa(), num_vgprs)
    }
    pub(crate) fn compile_program_vec(
        &self,
        program: &impl CompilationInput,
        num_vgprs: usize,
        width: u32,
    ) -> VecKernel {
        compile_packet(program.to_ssa(), num_vgprs, width, None)
    }
    pub(crate) fn compile_cooperative_vec(
        &self,
        program: &impl CompilationInput,
        num_vgprs: usize,
        width: u32,
    ) -> CoopVecKernel {
        compile_cooperative_packet(program.to_ssa(), num_vgprs, width, None)
    }
    pub(crate) fn compile_cooperative(
        &self,
        program: &impl CompilationInput,
        num_vgprs: usize,
    ) -> CoopVecKernel {
        compile_cooperative_scalar(program.to_ssa(), num_vgprs)
    }
}
