use crate::instructions::*;
use crate::rdna4_decoder::*;
use crate::rdna_instructions::*;
use crate::rdna_processor::Signals;

use llvm_sys as llvm;
use num::FromPrimitive;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::c_ulonglong;
use std::os::raw::c_void;

mod bvh;
mod combine;
mod emitter;
mod register_usage;

pub use bvh::*;
pub(crate) use combine::*;
pub(crate) use emitter::*;
pub(crate) use register_usage::*;

pub(crate) const USE_INSTRUCTION_COMBINE: bool = true;
pub(crate) const USE_VGPR_CACHE: bool = true;
pub(crate) const USE_SGPR_CACHE: bool = true;
pub(crate) const USE_SIMD: bool = true;
pub(crate) const SIMD_WIDTH: usize = 16;

pub fn is_terminator(inst: &InstFormat) -> bool {
    match inst {
        InstFormat::SOPP(inst) => match inst.op {
            I::S_CBRANCH_SCC0
            | I::S_CBRANCH_SCC1
            | I::S_CBRANCH_VCCZ
            | I::S_CBRANCH_VCCNZ
            | I::S_CBRANCH_EXECZ
            | I::S_CBRANCH_EXECNZ
            | I::S_BRANCH
            | I::S_BARRIER_WAIT
            | I::S_ENDPGM => true,
            _ => false,
        },
        _ => false,
    }
}

#[derive(Clone)]
pub struct InstBlock {
    context: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    addr: u64,
    reg_usage: RegisterUsage,
    pub call_count: u64,
    pub elapsed_time: u64,
    pub num_instructions: usize,
    pub instruction_usage: HashMap<String, u32>,
}

impl InstBlock {
    pub fn new() -> Self {
        InstBlock {
            context: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
            addr: 0,
            reg_usage: RegisterUsage::new(),
            call_count: 0,
            elapsed_time: 0,
            num_instructions: 0,
            instruction_usage: HashMap::new(),
        }
    }

    pub fn execute(
        &mut self,
        sgprs_ptr: *mut u32,
        vgprs_ptr: *mut u32,
        scc_ptr: *mut bool,
        pc: &mut u64,
        scratch_base: u64,
        lds_ptr: *mut u8,
    ) -> Signals {
        unsafe {
            let func = std::mem::transmute::<
                _,
                extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    c_ulonglong,
                    *mut c_void,
                ) -> u32,
            >(self.addr);

            use std::time::Instant;
            let start = Instant::now();

            let pc_ptr = &mut *pc as *mut u64;

            let signal = func(
                sgprs_ptr as *mut c_void,
                vgprs_ptr as *mut c_void,
                scc_ptr as *mut c_void,
                pc_ptr as *mut c_void,
                scratch_base as c_ulonglong,
                lds_ptr as *mut c_void,
            );

            let end = Instant::now();
            let duration = end.duration_since(start);
            let elapsed_ns = duration.as_nanos() as u64;

            self.call_count += 1;
            self.elapsed_time += elapsed_ns;

            FromPrimitive::from_u32(signal).unwrap()
        }
    }
}

#[derive(Clone)]
pub struct RDNATranslator {
    pub addresses: Vec<u64>,
    pub insts: Vec<InstFormat>,
    context: llvm::prelude::LLVMContextRef,
    pub insts_blocks: HashMap<u64, InstBlock>,
}

impl RDNATranslator {
    pub fn new() -> Self {
        RDNATranslator {
            addresses: Vec::new(),
            insts: Vec::new(),
            context: unsafe { llvm::core::LLVMContextCreate() },
            insts_blocks: HashMap::new(),
        }
    }

    pub fn add_inst(&mut self, addr: u64, inst: InstFormat) {
        self.addresses.push(addr);
        self.insts.push(inst);
    }

    fn analyze(&self) -> RegisterUsage {
        let mut reg_usage = RegisterUsage::new();

        reg_usage.incomming_sgprs.insert(106);
        reg_usage.incomming_sgprs.insert(126);

        reg_usage.use_sgprs.insert(106);
        reg_usage.use_sgprs.insert(126);

        reg_usage.def_sgprs.insert(106);
        reg_usage.def_sgprs.insert(126);

        for inst in &self.insts[..self.insts.len() - 1] {
            Self::analyze_instructions(inst, &mut reg_usage);
        }
        reg_usage
    }

    pub fn build_from_program(
        &mut self,
        program: &RDNAProgram,
        scratch_size: usize,
    ) -> &mut InstBlock {
        let mut inst_block = InstBlock::new();

        let mut entry_pcs = Vec::new();
        entry_pcs.push(program.entry_pc as u64);

        unsafe {
            llvm::target::LLVM_InitializeNativeTarget();
            llvm::target::LLVM_InitializeAllTargetMCs();
            llvm::target::LLVM_InitializeAllAsmParsers();
            llvm::target::LLVM_InitializeAllAsmPrinters();

            let context = self.context;
            let module = llvm::core::LLVMModuleCreateWithNameInContext(
                format!("kernel").as_ptr() as *const _,
                context,
            );

            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
            let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
            let mut param_ty = vec![ty_p0, ty_p0, ty_p0, ty_p0, ty_i64, ty_p0];
            let ty_function = llvm::core::LLVMFunctionType(
                ty_i32,
                param_ty.as_mut_ptr(),
                param_ty.len() as u32,
                0,
            );

            let func_name = std::ffi::CString::new("kernel").unwrap();
            let function = llvm::core::LLVMAddFunction(module, func_name.as_ptr(), ty_function);

            let entry_bb = llvm::core::LLVMAppendBasicBlockInContext(
                context,
                function,
                b"entry\0".as_ptr() as *const _,
            );

            let builder = llvm::core::LLVMCreateBuilderInContext(context);
            llvm::core::LLVMPositionBuilderAtEnd(builder, entry_bb);

            let sgprs_ptr = llvm::core::LLVMGetParam(function, 0);
            let vgprs_ptr = llvm::core::LLVMGetParam(function, 1);
            let scc_ptr = llvm::core::LLVMGetParam(function, 2);
            let pc_ptr = llvm::core::LLVMGetParam(function, 3);
            let scratch_base = llvm::core::LLVMGetParam(function, 4);
            let lds_ptr = llvm::core::LLVMGetParam(function, 5);

            let ret_value = llvm::core::LLVMConstInt(ty_i32, Signals::None as u64, 0);

            let mut emitter = IREmitter {
                context,
                module,
                function,
                builder,
                sgprs_ptr,
                vgprs_ptr,
                scc_ptr,
                scratch_base,
                scratch_size,
                pc_ptr,
                lds_ptr,
                ret_value,
                exec_value: std::ptr::null_mut(),
                local_scc_ptr: std::ptr::null_mut(),
                sgpr_ptr_map: HashMap::new(),
                vgpr_ptr_map: HashMap::new(),
                vgpr_reg_map: HashMap::new(),
                vgpr_incomming_reg_map: HashMap::new(),
                vgpr_reg_f64_map: HashMap::new(),
                use_vgpr_cache: USE_VGPR_CACHE,
                use_scc_cache: true,
                ray: RayIntersectionEmitter::new(),
                matrix: MatrixEmitter::new(),
            };

            let mut reg_usage = RegisterUsage::new();

            for (_, block) in &program.insts_blocks {
                for inst in &block.insts {
                    Self::analyze_instructions(inst, &mut reg_usage);
                }
            }

            reg_usage.incomming_sgprs.insert(106);
            reg_usage.incomming_sgprs.insert(126);

            reg_usage.use_sgprs.insert(106);
            reg_usage.use_sgprs.insert(126);

            reg_usage.def_sgprs.insert(106);
            reg_usage.def_sgprs.insert(126);

            for reg in 0..16 {
                if reg_usage.use_sgprs.contains(&reg) {
                    reg_usage.incomming_sgprs.insert(reg);
                }
            }

            for reg in 106..128 {
                if reg_usage.use_sgprs.contains(&reg) {
                    reg_usage.incomming_sgprs.insert(reg);
                }
            }

            reg_usage.incomming_vgprs.clear();

            for reg in 0..1 {
                if reg_usage.use_vgprs.contains(&reg) || reg_usage.def_vgprs.contains(&reg) {
                    reg_usage.incomming_vgprs.insert(reg);
                }
            }

            emitter.emit_alloc_registers(&reg_usage);
            emitter.emit_restore_stack(entry_bb, &reg_usage);

            let mut basic_blocks = HashMap::new();

            let mut addrs = program.insts_blocks.keys().collect::<Vec<_>>();
            addrs.sort_by_key(|addr| *addr);

            for addr in addrs {
                let basic_block = llvm::core::LLVMAppendBasicBlockInContext(
                    context,
                    function,
                    format!("block{:x}\0", addr).as_ptr() as *const _,
                );
                basic_blocks.insert(addr, basic_block);
            }

            for (addr, block) in &program.insts_blocks {
                let mut basic_block = *basic_blocks.get(addr).unwrap();

                llvm::core::LLVMPositionBuilderAtEnd(builder, basic_block);

                let mut block_reg_usage = RegisterUsage::new();
                block_reg_usage.incomming_sgprs.insert(106);
                block_reg_usage.incomming_sgprs.insert(126);

                block_reg_usage.use_sgprs.insert(106);
                block_reg_usage.use_sgprs.insert(126);

                block_reg_usage.def_sgprs.insert(106);
                block_reg_usage.def_sgprs.insert(126);

                for inst in &block.insts {
                    Self::analyze_instructions(inst, &mut block_reg_usage);
                }

                basic_block = emitter.emit_restore_registers(basic_block, &block_reg_usage);

                if is_terminator(block.insts.last().unwrap()) {
                    for inst in &block.insts[..block.insts.len() - 1] {
                        basic_block = emitter.emit_instruction(basic_block, inst);
                    }

                    basic_block = emitter.emit_save_registers(basic_block, &block_reg_usage);

                    let last_inst = block.insts.last().unwrap();
                    if let InstFormat::SOPP(inst) = last_inst {
                        match inst.op {
                            I::S_CBRANCH_EXECZ => {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let exec_value = emitter.emit_load_sgpr_u32(126);
                                let zero = llvm::core::LLVMConstInt(ty_i32, 0, 0);
                                let cond = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntEQ,
                                    exec_value,
                                    zero,
                                    empty_name.as_ptr() as *const _,
                                );
                                llvm::core::LLVMBuildCondBr(
                                    builder,
                                    cond,
                                    *basic_blocks.get(&block.next_pcs[1]).unwrap(),
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_CBRANCH_EXECNZ => {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let exec_value = emitter.emit_load_sgpr_u32(126);
                                let zero = llvm::core::LLVMConstInt(ty_i32, 0, 0);
                                let cond = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntNE,
                                    exec_value,
                                    zero,
                                    empty_name.as_ptr() as *const _,
                                );
                                llvm::core::LLVMBuildCondBr(
                                    builder,
                                    cond,
                                    *basic_blocks.get(&block.next_pcs[1]).unwrap(),
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_CBRANCH_VCCZ => {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let vcc_value = emitter.emit_load_sgpr_u32(106);
                                let zero = llvm::core::LLVMConstInt(ty_i32, 0, 0);
                                let cond = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntEQ,
                                    vcc_value,
                                    zero,
                                    empty_name.as_ptr() as *const _,
                                );
                                llvm::core::LLVMBuildCondBr(
                                    builder,
                                    cond,
                                    *basic_blocks.get(&block.next_pcs[1]).unwrap(),
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_CBRANCH_VCCNZ => {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let vcc_value = emitter.emit_load_sgpr_u32(106);
                                let zero = llvm::core::LLVMConstInt(ty_i32, 0, 0);
                                let cond = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntNE,
                                    vcc_value,
                                    zero,
                                    empty_name.as_ptr() as *const _,
                                );
                                llvm::core::LLVMBuildCondBr(
                                    builder,
                                    cond,
                                    *basic_blocks.get(&block.next_pcs[1]).unwrap(),
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_CBRANCH_SCC0 => {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let scc_value = emitter.emit_load_scc_u8();

                                let zero = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt8TypeInContext(context),
                                    0,
                                    0,
                                );
                                let cond = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntEQ,
                                    scc_value,
                                    zero,
                                    empty_name.as_ptr() as *const _,
                                );
                                llvm::core::LLVMBuildCondBr(
                                    builder,
                                    cond,
                                    *basic_blocks.get(&block.next_pcs[1]).unwrap(),
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_CBRANCH_SCC1 => {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let scc_value = emitter.emit_load_scc_u8();

                                let zero = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt8TypeInContext(context),
                                    0,
                                    0,
                                );
                                let cond = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntNE,
                                    scc_value,
                                    zero,
                                    empty_name.as_ptr() as *const _,
                                );
                                llvm::core::LLVMBuildCondBr(
                                    builder,
                                    cond,
                                    *basic_blocks.get(&block.next_pcs[1]).unwrap(),
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_BRANCH => {
                                llvm::core::LLVMBuildBr(
                                    builder,
                                    *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                                );
                            }
                            I::S_ENDPGM => {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let ret_value = llvm::core::LLVMConstInt(
                                    ty_i32,
                                    Signals::EndOfProgram as u64,
                                    0,
                                );
                                llvm::core::LLVMBuildRet(builder, ret_value);
                            }
                            I::S_BARRIER_WAIT => {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                                emitter.emit_save_stack(basic_block, &reg_usage);

                                let next_pc_value =
                                    llvm::core::LLVMConstInt(ty_i64, block.next_pcs[0] as u64, 0);

                                llvm::core::LLVMBuildStore(builder, next_pc_value, emitter.pc_ptr);

                                let ret_value =
                                    llvm::core::LLVMConstInt(ty_i32, Signals::Switch as u64, 0);
                                llvm::core::LLVMBuildRet(builder, ret_value);

                                entry_pcs.push(block.next_pcs[0] as u64);
                            }
                            _ => panic!("Unsupported terminator instruction: {:?}", inst),
                        }
                    } else {
                        panic!(
                            "Last instruction in block {} is not a terminator: {:?}",
                            addr, last_inst
                        );
                    }
                } else {
                    for inst in &block.insts {
                        basic_block = emitter.emit_instruction(basic_block, inst);
                    }

                    emitter.emit_save_registers(basic_block, &block_reg_usage);

                    if block.next_pcs.len() == 1 {
                        llvm::core::LLVMBuildBr(
                            builder,
                            *basic_blocks.get(&block.next_pcs[0]).unwrap(),
                        );
                    } else {
                        panic!("Block {} has multiple next PCs: {:?}", addr, block.next_pcs);
                    }
                }
            }

            llvm::core::LLVMPositionBuilderAtEnd(builder, entry_bb);

            let empty_name = std::ffi::CString::new("").unwrap();
            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

            let pc_value =
                llvm::core::LLVMBuildLoad2(builder, ty_i64, emitter.pc_ptr, empty_name.as_ptr());

            let switch = llvm::core::LLVMBuildSwitch(
                builder,
                pc_value,
                *basic_blocks.get(&program.entry_pc).unwrap(),
                entry_pcs.len() as u32,
            );

            for entry_pc in &entry_pcs {
                let case_value = llvm::core::LLVMConstInt(ty_i64, *entry_pc, 0);
                llvm::core::LLVMAddCase(
                    switch,
                    case_value,
                    *basic_blocks.get(&(*entry_pc as usize)).unwrap(),
                );
            }

            llvm::core::LLVMDisposeBuilder(builder);

            let mut err = std::ptr::null_mut();
            let is_err = llvm::analysis::LLVMVerifyModule(
                module,
                llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
                &mut err,
            );
            if is_err != 0 {
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                llvm::core::LLVMDisposeMessage(err.into_raw());
                panic!("Failed to verify main module: {}", err_.to_str().unwrap());
            }

            let triple = llvm::target_machine::LLVMGetDefaultTargetTriple();
            let mut target = std::ptr::null_mut();
            let mut err = std::ptr::null_mut();
            let result =
                llvm::target_machine::LLVMGetTargetFromTriple(triple, &mut target, &mut err);
            if result != 0 {
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to get target from triple: {}",
                    err_.to_str().unwrap()
                );
            }
            let cpu_name = llvm::target_machine::LLVMGetHostCPUName();
            let cpu_feature = llvm::target_machine::LLVMGetHostCPUFeatures();
            let tm = llvm::target_machine::LLVMCreateTargetMachine(
                target,
                triple,
                cpu_name,
                cpu_feature,
                llvm::target_machine::LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                llvm::target_machine::LLVMRelocMode::LLVMRelocDefault,
                llvm::target_machine::LLVMCodeModel::LLVMCodeModelJITDefault,
            );

            let pass_builder_options =
                llvm::transforms::pass_builder::LLVMCreatePassBuilderOptions();
            let err = llvm::transforms::pass_builder::LLVMRunPassesOnFunction(
                function,
                b"lcssa,adce,early-cse,instcombine<no-verify-fixpoint>,aggressive-instcombine,mem2reg,gvn,dse,instsimplify,load-store-vectorizer,loop-fusion,loop-reduce,sink,loop-load-elim,reassociate,function-simplification<O3>,loop-vectorize,simplifycfg,loop-unroll<O3>\0".as_ptr() as *const _,
                tm,
                pass_builder_options,
            );

            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to run passes on function: {}",
                    err_.to_str().unwrap()
                );
            }

            let mut instruction_count = 0;

            let mut bb = llvm::core::LLVMGetFirstBasicBlock(function);
            while bb != std::ptr::null_mut() {
                let mut inst = llvm::core::LLVMGetFirstInstruction(bb);
                while inst != std::ptr::null_mut() {
                    instruction_count += 1;
                    inst = llvm::core::LLVMGetNextInstruction(inst);
                }

                bb = llvm::core::LLVMGetNextBasicBlock(bb);
            }

            let jit_builder = llvm::orc2::lljit::LLVMOrcCreateLLJITBuilder();

            let jtmb = if false {
                let mut jtmb = std::ptr::null_mut();
                let err = llvm::orc2::LLVMOrcJITTargetMachineBuilderDetectHost(&mut jtmb);
                if !err.is_null() {
                    let err = llvm::error::LLVMGetErrorMessage(err);
                    let err = std::ffi::CString::from_raw(err);
                    let err_ = err.clone();
                    panic!(
                        "Failed to detect host JIT target machine: {}",
                        err_.to_str().unwrap()
                    );
                }
                jtmb
            } else {
                let jtmb = llvm::orc2::LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);
                jtmb
            };

            llvm::orc2::lljit::LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(jit_builder, jtmb);

            let mut jit = std::ptr::null_mut();

            let err = llvm::orc2::lljit::LLVMOrcCreateLLJIT(&mut jit, jit_builder);
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!("Failed to create LLJIT: {}", err_.to_str().unwrap());
            }

            let dylib = llvm::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(jit);

            let lljit_gl_prefix = llvm::orc2::lljit::LLVMOrcLLJITGetGlobalPrefix(jit);

            let mut dg = std::ptr::null_mut();
            let err = llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut dg,
                lljit_gl_prefix,
                None,
                std::ptr::null_mut(),
            );
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to create dynamic library search generator: {}",
                    err_.to_str().unwrap()
                );
            }

            llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg);

            let lib_path = if cfg!(debug_assertions) {
                "target/debug/libamdgpu_sim.so\0"
            } else {
                "target/release/libamdgpu_sim.so\0"
            };

            let mut dg = std::ptr::null_mut();
            let err = llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
                &mut dg,
                lib_path.as_ptr() as *const _,
                lljit_gl_prefix,
                None,
                std::ptr::null_mut(),
            );
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to create dynamic library search generator: {}",
                    err_.to_str().unwrap()
                );
            }

            llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg);

            let tsctx = llvm::orc2::LLVMOrcCreateNewThreadSafeContext();
            let tsm = llvm::orc2::LLVMOrcCreateNewThreadSafeModule(module, tsctx);

            let err = llvm::orc2::lljit::LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm);
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!("Failed to add LLVM IR module: {}", err_.to_str().unwrap());
            }

            let mut func = 0u64;
            let err = llvm::orc2::lljit::LLVMOrcLLJITLookup(jit, &mut func, func_name.as_ptr());
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!("Failed to lookup function: {}", err_.to_str().unwrap());
            }

            inst_block.context = context;
            inst_block.module = module;
            inst_block.addr = func;
            inst_block.reg_usage = reg_usage;
            inst_block.num_instructions = instruction_count;
        }

        for entry_pc in &entry_pcs {
            self.insts_blocks.insert(*entry_pc, inst_block.clone());
        }

        self.clear();

        let block_addr = program.entry_pc as u64;

        self.insts_blocks.get_mut(&block_addr).unwrap()
    }

    pub fn get_or_build(&mut self, scratch_size: usize) -> &mut InstBlock {
        if self.insts_blocks.contains_key(&self.get_address().unwrap()) {
            return self
                .insts_blocks
                .get_mut(&self.get_address().unwrap())
                .unwrap();
        }

        let mut inst_block = InstBlock::new();

        unsafe {
            llvm::target::LLVM_InitializeNativeTarget();
            llvm::target::LLVM_InitializeAllTargetMCs();
            llvm::target::LLVM_InitializeAllAsmParsers();
            llvm::target::LLVM_InitializeAllAsmPrinters();

            let context = self.context;
            let module = llvm::core::LLVMModuleCreateWithNameInContext(
                format!("block{}", self.get_address().unwrap()).as_ptr() as *const _,
                context,
            );

            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
            let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
            let mut param_ty = vec![ty_p0, ty_p0, ty_p0, ty_p0, ty_i64, ty_p0];
            let ty_function = llvm::core::LLVMFunctionType(
                ty_i32,
                param_ty.as_mut_ptr(),
                param_ty.len() as u32,
                0,
            );

            let func_name = format!("block{}\0", self.get_address().unwrap());
            let function =
                llvm::core::LLVMAddFunction(module, func_name.as_ptr() as *const _, ty_function);

            let mut bb = llvm::core::LLVMAppendBasicBlockInContext(
                context,
                function,
                b"entry\0".as_ptr() as *const _,
            );

            let builder = llvm::core::LLVMCreateBuilderInContext(context);
            llvm::core::LLVMPositionBuilderAtEnd(builder, bb);

            let sgprs_ptr = llvm::core::LLVMGetParam(function, 0);
            let vgprs_ptr = llvm::core::LLVMGetParam(function, 1);
            let scc_ptr = llvm::core::LLVMGetParam(function, 2);
            let pc_ptr = llvm::core::LLVMGetParam(function, 3);
            let scratch_base = llvm::core::LLVMGetParam(function, 4);
            let lds_ptr = llvm::core::LLVMGetParam(function, 5);
            let ret_value = llvm::core::LLVMConstInt(ty_i32, Signals::None as u64, 0);

            let mut emitter = IREmitter {
                context,
                module,
                function,
                builder,
                sgprs_ptr,
                vgprs_ptr,
                scc_ptr,
                pc_ptr,
                scratch_base,
                scratch_size,
                lds_ptr,
                ret_value,
                exec_value: std::ptr::null_mut(),
                local_scc_ptr: std::ptr::null_mut(),
                sgpr_ptr_map: HashMap::new(),
                vgpr_ptr_map: HashMap::new(),
                vgpr_reg_map: HashMap::new(),
                vgpr_incomming_reg_map: HashMap::new(),
                vgpr_reg_f64_map: HashMap::new(),
                use_vgpr_cache: false,
                use_scc_cache: false,
                ray: RayIntersectionEmitter::new(),
                matrix: MatrixEmitter::new(),
            };

            {
                let mut instruction_usage = HashMap::new();

                for inst in &self.insts {
                    match inst {
                        InstFormat::SOPP(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::SOP1(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::SOP2(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::SOPK(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::SOPC(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VOPC(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VOP1(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VOP2(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VOP3(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VOP3SD(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VOPD(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.opx))
                                .or_insert(0) += 1;
                            *instruction_usage
                                .entry(format!("{:?}", inst.opy))
                                .or_insert(0) += 1;
                        }
                        InstFormat::VGLOBAL(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        InstFormat::SMEM(inst) => {
                            *instruction_usage
                                .entry(format!("{:?}", inst.op))
                                .or_insert(0) += 1;
                        }
                        _ => {}
                    }
                }

                inst_block.instruction_usage = instruction_usage;
            }

            let reg_usage = self.analyze();

            emitter.emit_alloc_registers(&reg_usage);

            bb = emitter.emit_restore_stack(bb, &reg_usage);
            bb = emitter.emit_restore_registers(bb, &reg_usage);

            for inst in &self.insts[..self.insts.len() - 1] {
                bb = emitter.emit_instruction(bb, inst);
            }

            bb = emitter.emit_terminator(
                bb,
                self.insts.last().unwrap(),
                *self.addresses.last().unwrap(),
            );

            bb = emitter.emit_save_registers(bb, &reg_usage);
            emitter.emit_save_stack(bb, &reg_usage);

            llvm::core::LLVMBuildRet(builder, emitter.ret_value);

            llvm::core::LLVMDisposeBuilder(builder);

            let mut err = std::ptr::null_mut();
            let is_err = llvm::analysis::LLVMVerifyModule(
                module,
                llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
                &mut err,
            );
            if is_err != 0 {
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                llvm::core::LLVMDisposeMessage(err.into_raw());
                panic!("Failed to verify main module: {}", err_.to_str().unwrap());
            }

            let triple = llvm::target_machine::LLVMGetDefaultTargetTriple();
            let mut target = std::ptr::null_mut();
            let mut err = std::ptr::null_mut();
            let result =
                llvm::target_machine::LLVMGetTargetFromTriple(triple, &mut target, &mut err);
            if result != 0 {
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to get target from triple: {}",
                    err_.to_str().unwrap()
                );
            }
            let cpu_name = llvm::target_machine::LLVMGetHostCPUName();
            let cpu_feature = llvm::target_machine::LLVMGetHostCPUFeatures();
            let tm = llvm::target_machine::LLVMCreateTargetMachine(
                target,
                triple,
                cpu_name,
                cpu_feature,
                llvm::target_machine::LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                llvm::target_machine::LLVMRelocMode::LLVMRelocDefault,
                llvm::target_machine::LLVMCodeModel::LLVMCodeModelJITDefault,
            );

            let pass_builder_options =
                llvm::transforms::pass_builder::LLVMCreatePassBuilderOptions();
            let err = llvm::transforms::pass_builder::LLVMRunPassesOnFunction(
                function,
                b"early-cse,instcombine<no-verify-fixpoint>,aggressive-instcombine,mem2reg,gvn,dse,instsimplify,load-store-vectorizer,loop-fusion,loop-load-elim,reassociate,function-simplification<O3>,loop-vectorize,simplifycfg,loop-unroll<O3>\0".as_ptr() as *const _,
                tm,
                pass_builder_options,
            );

            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to run passes on function: {}",
                    err_.to_str().unwrap()
                );
            }

            let mut instruction_count = 0;

            let mut bb = llvm::core::LLVMGetFirstBasicBlock(function);
            while bb != std::ptr::null_mut() {
                let mut inst = llvm::core::LLVMGetFirstInstruction(bb);
                while inst != std::ptr::null_mut() {
                    instruction_count += 1;
                    inst = llvm::core::LLVMGetNextInstruction(inst);
                }

                bb = llvm::core::LLVMGetNextBasicBlock(bb);
            }

            let jit_builder = llvm::orc2::lljit::LLVMOrcCreateLLJITBuilder();

            let jtmb = if false {
                let mut jtmb = std::ptr::null_mut();
                let err = llvm::orc2::LLVMOrcJITTargetMachineBuilderDetectHost(&mut jtmb);
                if !err.is_null() {
                    let err = llvm::error::LLVMGetErrorMessage(err);
                    let err = std::ffi::CString::from_raw(err);
                    let err_ = err.clone();
                    panic!(
                        "Failed to detect host JIT target machine: {}",
                        err_.to_str().unwrap()
                    );
                }
                jtmb
            } else {
                let jtmb = llvm::orc2::LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);
                jtmb
            };

            llvm::orc2::lljit::LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(jit_builder, jtmb);

            let mut jit = std::ptr::null_mut();

            let err = llvm::orc2::lljit::LLVMOrcCreateLLJIT(&mut jit, jit_builder);
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!("Failed to create LLJIT: {}", err_.to_str().unwrap());
            }

            let dylib = llvm::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(jit);

            let lljit_gl_prefix = llvm::orc2::lljit::LLVMOrcLLJITGetGlobalPrefix(jit);

            let mut dg = std::ptr::null_mut();
            let err = llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut dg,
                lljit_gl_prefix,
                None,
                std::ptr::null_mut(),
            );
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to create dynamic library search generator: {}",
                    err_.to_str().unwrap()
                );
            }

            llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg);

            let lib_path = if cfg!(debug_assertions) {
                "target/debug/libamdgpu_sim.so\0"
            } else {
                "target/release/libamdgpu_sim.so\0"
            };

            let mut dg = std::ptr::null_mut();
            let err = llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
                &mut dg,
                lib_path.as_ptr() as *const _,
                lljit_gl_prefix,
                None,
                std::ptr::null_mut(),
            );
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!(
                    "Failed to create dynamic library search generator: {}",
                    err_.to_str().unwrap()
                );
            }

            llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg);

            let tsctx = llvm::orc2::LLVMOrcCreateNewThreadSafeContext();
            let tsm = llvm::orc2::LLVMOrcCreateNewThreadSafeModule(module, tsctx);

            let err = llvm::orc2::lljit::LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm);
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!("Failed to add LLVM IR module: {}", err_.to_str().unwrap());
            }

            let mut func = 0u64;
            let err = llvm::orc2::lljit::LLVMOrcLLJITLookup(
                jit,
                &mut func,
                func_name.as_ptr() as *const _,
            );
            if !err.is_null() {
                let err = llvm::error::LLVMGetErrorMessage(err);
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                panic!("Failed to lookup function: {}", err_.to_str().unwrap());
            }

            inst_block.context = context;
            inst_block.module = module;
            inst_block.addr = func;
            inst_block.reg_usage = reg_usage;
            inst_block.num_instructions = instruction_count;
        }

        let block_addr = self.get_address().unwrap();

        self.clear();

        self.insts_blocks.insert(block_addr, inst_block);

        self.insts_blocks.get_mut(&block_addr).unwrap()
    }

    pub fn get_address(&self) -> Option<u64> {
        self.addresses.first().copied()
    }

    pub fn clear(&mut self) {
        self.addresses.clear();
        self.insts.clear();
    }
}

pub struct RDNAProgram {
    entry_pc: usize,
    insts_blocks: HashMap<usize, BasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Range {
    start: usize,
    end: usize, // Exclusive end
}

struct BasicBlock {
    insts: Vec<InstFormat>,
    next_pcs: Vec<usize>,
}

impl RDNAProgram {
    fn get_next_pcs(pc: usize, inst: &InstFormat) -> Vec<usize> {
        let mut next_pscs = Vec::new();
        match inst {
            InstFormat::SOPP(inst) => match inst.op {
                I::S_CBRANCH_EXECZ
                | I::S_CBRANCH_EXECNZ
                | I::S_CBRANCH_VCCZ
                | I::S_CBRANCH_VCCNZ
                | I::S_CBRANCH_SCC0
                | I::S_CBRANCH_SCC1 => {
                    next_pscs.push(pc);
                    let next_pc = ((pc as i64) + ((inst.simm16 as i16 as i64) * 4)) as usize;
                    next_pscs.push(next_pc);
                }
                I::S_BRANCH => {
                    let next_pc = ((pc as i64) + ((inst.simm16 as i16 as i64) * 4)) as usize;
                    next_pscs.push(next_pc);
                }
                I::S_ENDPGM => {}
                I::S_BARRIER_WAIT => {
                    next_pscs.push(pc);
                }
                _ => {
                    next_pscs.push(pc);
                }
            },
            _ => {
                next_pscs.push(pc);
            }
        }
        next_pscs
    }

    fn search_instruction_ranges(
        pc: usize,
        inst_stream: &InstStream,
        visited: &mut HashSet<Range>,
    ) {
        let mut pc = pc;
        let start_pc = pc;
        let mut insts = Vec::new();
        let mut doing = true;
        while doing {
            let current_inst_stream = InstStream {
                insts: &inst_stream.insts[pc..],
            };
            let (inst, size) = decode_rdna4(current_inst_stream).unwrap();
            insts.push(inst.clone());
            pc += size;

            let mut reg_usage = RegisterUsage::new();
            RDNATranslator::analyze_instructions(&inst, &mut reg_usage);

            let change_exec = reg_usage.def_sgprs.contains(&126);

            if is_terminator(&inst)
                || visited
                    .iter()
                    .find(|range| pc >= range.start && pc < range.end)
                    .is_some()
                || change_exec
            {
                doing = false;
            }
        }

        let next_pcs = Self::get_next_pcs(pc, &insts.last().unwrap());

        visited.insert(Range {
            start: start_pc,
            end: pc,
        });

        for next_pc in next_pcs {
            if let Some(range) = visited
                .iter()
                .find(|range| next_pc >= range.start && next_pc < range.end)
                .cloned()
            {
                if range.start < next_pc {
                    visited.insert(Range {
                        start: range.start,
                        end: next_pc,
                    });
                    visited.insert(Range {
                        start: next_pc,
                        end: range.end,
                    });
                    visited.remove(&range);
                } else {
                }
            } else {
                Self::search_instruction_ranges(next_pc as usize, inst_stream, visited);
            }
        }
    }

    fn create_basic_block_from_range(range: &Range, inst_stream: &InstStream) -> BasicBlock {
        let mut insts = Vec::new();
        let mut pc = range.start;
        while pc < range.end {
            let current_inst_stream = InstStream {
                insts: &inst_stream.insts[pc..],
            };
            let (inst, size) = decode_rdna4(current_inst_stream).unwrap();
            insts.push(inst.clone());
            pc += size;
        }
        let next_pcs = Self::get_next_pcs(range.end, &insts.last().unwrap());
        BasicBlock { insts, next_pcs }
    }

    pub fn new(pc: usize, inst_stream: &[u8]) -> Self {
        let inst_stream = InstStream { insts: inst_stream };
        let mut visited = HashSet::new();
        Self::search_instruction_ranges(pc, &inst_stream, &mut visited);

        let mut ranges = visited.into_iter().collect::<Vec<_>>();
        ranges.sort_by_key(|r| r.start);

        for i in 0..ranges.len() - 1 {
            if ranges[i].end != ranges[i + 1].start {
                panic!(
                    "Ranges are not contiguous: {:?} and {:?}",
                    ranges[i],
                    ranges[i + 1]
                );
            }
        }

        let mut basic_blocks = HashMap::new();
        for range in ranges {
            let block = Self::create_basic_block_from_range(&range, &inst_stream);
            basic_blocks.insert(range.start, block);
        }

        if USE_INSTRUCTION_COMBINE {
            let mut removed = 0;
            for block in basic_blocks.values_mut() {
                removed += combine_block(&mut block.insts);
            }
            if removed > 0 {
                println!("Instruction combine removed {} instructions", removed);
            }
        }

        RDNAProgram {
            entry_pc: pc,
            insts_blocks: basic_blocks,
        }
    }
}
