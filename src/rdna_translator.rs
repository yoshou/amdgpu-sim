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

const USE_VGPR_CACHE: bool = true;
const USE_SGPR_CACHE: bool = true;
const USE_SIMD: bool = true;
const USE_MASKED_GATHER: bool = true;
const SIMD_WIDTH: usize = 16;

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

extern "C" fn llvm_obj_linking_layer_create(
    _: *mut c_void,
    es: llvm::orc2::LLVMOrcExecutionSessionRef,
    _: *const ::libc::c_char,
) -> llvm::orc2::LLVMOrcObjectLayerRef {
    unsafe { llvm::orc2::ee::LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(es) }
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

struct IREmitter {
    context: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    function: llvm::prelude::LLVMValueRef,
    builder: llvm::prelude::LLVMBuilderRef,
    sgprs_ptr: llvm::prelude::LLVMValueRef,
    vgprs_ptr: llvm::prelude::LLVMValueRef,
    scc_ptr: llvm::prelude::LLVMValueRef,
    scratch_base: llvm::prelude::LLVMValueRef,
    lds_ptr: llvm::prelude::LLVMValueRef,
    pc_ptr: llvm::prelude::LLVMValueRef,
    ret_value: llvm::prelude::LLVMValueRef,
    exec_value: llvm::prelude::LLVMValueRef,
    local_scc_ptr: llvm::prelude::LLVMValueRef,
    sgpr_ptr_map: HashMap<u32, llvm::prelude::LLVMValueRef>,
    vgpr_ptr_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    vgpr_reg_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    vgpr_incomming_reg_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    vgpr_reg_f64_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    use_vgpr_cache: bool,
    use_scc_cache: bool,
}

impl IREmitter {
    unsafe fn emit_exec_bit(
        &mut self,
        index: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let exec_value = self.emit_load_sgpr_u32(126);

        let index_mask = llvm::core::LLVMBuildShl(
            builder,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            index,
            empty_name.as_ptr(),
        );
        let masked = llvm::core::LLVMBuildAnd(builder, exec_value, index_mask, empty_name.as_ptr());
        llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntNE,
            masked,
            llvm::core::LLVMConstInt(ty_i32, 0, 0),
            empty_name.as_ptr(),
        )
    }

    unsafe fn emit_vcc_bit(
        &mut self,
        index: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let exec_value = self.emit_load_sgpr_u32(106);

        let index_mask = llvm::core::LLVMBuildShl(
            builder,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            index,
            empty_name.as_ptr(),
        );
        let masked = llvm::core::LLVMBuildAnd(builder, exec_value, index_mask, empty_name.as_ptr());
        llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntNE,
            masked,
            llvm::core::LLVMConstInt(ty_i32, 0, 0),
            empty_name.as_ptr(),
        )
    }

    unsafe fn emit_vop(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        predicate: impl Fn(
            &mut IREmitter,
            llvm::prelude::LLVMBasicBlockRef,
            llvm::prelude::LLVMValueRef,
        ) -> llvm::prelude::LLVMBasicBlockRef,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let function = self.function;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());
        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
        let index_i32 = llvm::core::LLVMBuildTrunc(builder, index, ty_i32, empty_name.as_ptr());
        let exec = self.emit_exec_bit(index_i32);

        let bb_loop_skip_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_cond =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_exit =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        llvm::core::LLVMBuildCondBr(builder, exec, bb_loop_body, bb_loop_skip_body);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_skip_body);

        let next_index1 = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i64, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let bb_loop_body = predicate(self, bb_loop_body, index);

        let next_index2 = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i64, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

        let next_index = llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
        let mut incoming_value = vec![next_index1, next_index2];
        let mut incoming_blocks = vec![bb_loop_skip_body, bb_loop_body];
        llvm::core::LLVMAddIncoming(
            next_index,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        let cmp = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            next_index,
            llvm::core::LLVMConstInt(ty_i64, 32, 0),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i64, 0, 0), next_index];
        let mut incoming_blocks = vec![bb, bb_loop_cond];
        llvm::core::LLVMAddIncoming(
            index,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_exit);

        bb_loop_exit
    }

    unsafe fn emit_vop_update_sgpr(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        sgpr_reg: u32,
        predicate: impl Fn(
            &mut IREmitter,
            llvm::prelude::LLVMBasicBlockRef,
            llvm::prelude::LLVMValueRef,
        ) -> (
            llvm::prelude::LLVMBasicBlockRef,
            llvm::prelude::LLVMValueRef,
        ),
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let function = self.function;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());
        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
        let vcc = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
        let index_i32 = llvm::core::LLVMBuildTrunc(builder, index, ty_i32, empty_name.as_ptr());
        let exec = self.emit_exec_bit(index_i32);

        let bb_loop_skip_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_cond =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_exit =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        llvm::core::LLVMBuildCondBr(builder, exec, bb_loop_body, bb_loop_skip_body);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_skip_body);

        let next_index1 = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i64, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let (bb_loop_body, cc) = predicate(self, bb_loop_body, index);

        let cc = llvm::core::LLVMBuildZExt(builder, cc, ty_i32, empty_name.as_ptr());
        let cc_shifted = llvm::core::LLVMBuildShl(builder, cc, index_i32, empty_name.as_ptr());

        let update_vcc = llvm::core::LLVMBuildOr(builder, cc_shifted, vcc, empty_name.as_ptr());

        let next_index2 = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i64, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

        let next_index = llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
        let new_vcc = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
        let mut incoming_value = vec![next_index1, next_index2];
        let mut incoming_blocks = vec![bb_loop_skip_body, bb_loop_body];
        llvm::core::LLVMAddIncoming(
            next_index,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );
        let mut incoming_value = vec![vcc, update_vcc];
        let mut incoming_blocks = vec![bb_loop_skip_body, bb_loop_body];
        llvm::core::LLVMAddIncoming(
            new_vcc,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        let cmp = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            next_index,
            llvm::core::LLVMConstInt(ty_i64, 32, 0),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i64, 0, 0), next_index];
        let mut incoming_blocks = vec![bb, bb_loop_cond];
        llvm::core::LLVMAddIncoming(
            index,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i32, 0, 0), new_vcc];
        let mut incoming_blocks = vec![bb, bb_loop_cond];
        llvm::core::LLVMAddIncoming(
            vcc,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_exit);

        self.emit_store_sgpr_u32(sgpr_reg, new_vcc);

        bb_loop_exit
    }

    unsafe fn emit_load_scc_u8(&mut self) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let scc_ptr = if self.use_scc_cache {
            self.local_scc_ptr
        } else {
            self.scc_ptr
        };

        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildLoad2(builder, ty_i8, scc_ptr, empty_name.as_ptr())
    }

    unsafe fn emit_store_scc_u8(&mut self, value: llvm::prelude::LLVMValueRef) {
        let builder = self.builder;
        let scc_ptr = if self.use_scc_cache {
            self.local_scc_ptr
        } else {
            self.scc_ptr
        };

        llvm::core::LLVMBuildStore(builder, value, scc_ptr);
    }

    unsafe fn emit_load_vgpr_u32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(elem) != ty_i64 {
            panic!("Type of elem is not i64");
        }

        if self.use_vgpr_cache {
            panic!("Not implemented");
        }

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(
                llvm::core::LLVMInt64TypeInContext(context),
                reg as u64 * 32,
                0,
            ),
            elem,
            empty_name.as_ptr(),
        );
        let mut indices = vec![index];
        let value_ptr = llvm::core::LLVMBuildGEP2(
            builder,
            ty_i32,
            vgprs_ptr,
            indices.as_mut_ptr(),
            indices.len() as u32,
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildLoad2(builder, ty_i32, value_ptr, empty_name.as_ptr())
    }

    unsafe fn emit_load_vgpr_f32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let value = self.emit_load_vgpr_u32(reg, elem);
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f32, empty_name.as_ptr())
    }

    unsafe fn emit_load_vgpr_u64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let value_lo = self.emit_load_vgpr_u32(reg, elem);
        let value_hi = self.emit_load_vgpr_u32(reg + 1, elem);

        let value_lo = llvm::core::LLVMBuildZExt(
            builder,
            value_lo,
            llvm::core::LLVMInt64TypeInContext(context),
            empty_name.as_ptr(),
        );
        let value_hi = llvm::core::LLVMBuildZExt(
            builder,
            value_hi,
            llvm::core::LLVMInt64TypeInContext(context),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildOr(
            builder,
            llvm::core::LLVMBuildShl(
                builder,
                value_hi,
                llvm::core::LLVMConstInt(llvm::core::LLVMInt64TypeInContext(context), 32, 0),
                empty_name.as_ptr(),
            ),
            value_lo,
            empty_name.as_ptr(),
        )
    }

    unsafe fn emit_load_vgpr_f64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let value = self.emit_load_vgpr_u64(reg, elem);
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64, empty_name.as_ptr())
    }

    unsafe fn emit_load_vgpr_u64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(self.context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

        let value_lo = self.emit_load_vgpr_u32xn::<N>(reg, elem, mask);
        let value_hi = self.emit_load_vgpr_u32xn::<N>(reg + 1, elem, mask);

        let mut shuffle_indices = Vec::new();
        for i in 0..N {
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, i as u64, 0));
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, (i + N) as u64, 0));
        }

        let value_lo_hi = llvm::core::LLVMBuildShuffleVector(
            builder,
            value_lo,
            value_hi,
            llvm::core::LLVMConstVector(shuffle_indices.as_mut_ptr(), N as u32 * 2),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBitCast(builder, value_lo_hi, ty_i64xn, empty_name.as_ptr())
    }

    unsafe fn emit_load_vgpr_f64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        if self.use_vgpr_cache {
            let value = self.vgpr_reg_f64_map.get(&reg).unwrap()[elem as usize / N];
            if value != std::ptr::null_mut() {
                return value;
            }
        }

        let value = self.emit_load_vgpr_u64xn::<N>(reg, elem, mask);
        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64xn, empty_name.as_ptr())
    }

    unsafe fn emit_load_stack_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
    ) -> llvm::prelude::LLVMValueRef {
        if reg == 124 {
            return llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstInt(llvm::core::LLVMInt32TypeInContext(self.context), 0, 0);
                    N]
                    .as_mut_ptr(),
                N as u32,
            );
        }
        self.vgpr_reg_map.get(&reg).unwrap()[elem as usize / N]
    }

    unsafe fn emit_load_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
        let ty_i1 = llvm::core::LLVMInt1TypeInContext(self.context);
        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
        let alignment = llvm::core::LLVMConstInt(ty_i32, 4, 0);

        if self.use_vgpr_cache {
            return self.emit_load_stack_vgpr_u32xn::<N>(reg, elem);
        }

        let elem =
            llvm::core::LLVMConstInt(llvm::core::LLVMInt64TypeInContext(context), elem as u64, 0);

        let mut param_tys = vec![ty_i32xn, ty_p0];
        let intrinsic_name = format!("llvm.masked.load.v{}i32\0", N);
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(
                llvm::core::LLVMInt64TypeInContext(context),
                reg as u64 * 32,
                0,
            ),
            elem,
            empty_name.as_ptr(),
        );
        let mut indices = vec![index];
        let value_ptr = llvm::core::LLVMBuildGEP2(
            builder,
            ty_i32,
            vgprs_ptr,
            indices.as_mut_ptr(),
            indices.len() as u32,
            empty_name.as_ptr(),
        );

        let mut param_tys = vec![ty_p0, ty_i32, ty_i1xn, ty_i32xn];
        let value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(
                ty_i32xn,
                param_tys.as_mut_ptr(),
                param_tys.len() as u32,
                0,
            ),
            intrinsic,
            [
                value_ptr,
                alignment,
                mask,
                llvm::core::LLVMGetPoison(ty_i32xn),
            ]
            .as_mut_ptr(),
            4,
            empty_name.as_ptr(),
        );
        value
    }

    unsafe fn emit_load_vgpr_f32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = self.emit_load_vgpr_u32xn::<N>(reg, elem, mask);

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f32xn, empty_name.as_ptr())
    }

    unsafe fn emit_store_stack_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        _mask: llvm::prelude::LLVMValueRef,
    ) {
        if reg == 124 {
            return;
        }

        self.vgpr_reg_f64_map.get_mut(&reg).unwrap()[elem as usize / N] = std::ptr::null_mut();

        self.vgpr_reg_map.get_mut(&reg).unwrap()[elem as usize / N] = value;
    }

    unsafe fn emit_store_vgpr_u32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(elem) != ty_i64 {
            panic!("Type of elem is not i64");
        }

        if self.use_vgpr_cache {
            panic!("Not implemented");
        }

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(
                llvm::core::LLVMInt64TypeInContext(context),
                reg as u64 * 32,
                0,
            ),
            elem,
            empty_name.as_ptr(),
        );
        let mut indices = vec![index];
        let value_ptr = llvm::core::LLVMBuildGEP2(
            builder,
            ty_i32,
            vgprs_ptr,
            indices.as_mut_ptr(),
            indices.len() as u32,
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildStore(builder, value, value_ptr);
    }

    unsafe fn emit_store_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
        let empty_name = std::ffi::CString::new("").unwrap();
        let alignment = llvm::core::LLVMConstInt(ty_i32, 4, 0);

        if self.use_vgpr_cache {
            return self.emit_store_stack_vgpr_u32xn::<N>(reg, elem, value, mask);
        }

        let elem =
            llvm::core::LLVMConstInt(llvm::core::LLVMInt64TypeInContext(context), elem as u64, 0);

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(
                llvm::core::LLVMInt64TypeInContext(context),
                reg as u64 * 32,
                0,
            ),
            elem,
            empty_name.as_ptr(),
        );
        let mut indices = vec![index];
        let value_ptr = llvm::core::LLVMBuildGEP2(
            builder,
            ty_i32,
            vgprs_ptr,
            indices.as_mut_ptr(),
            indices.len() as u32,
            empty_name.as_ptr(),
        );

        let mut param_tys = vec![ty_i32xn, ty_p0];
        let intrinsic_name = format!("llvm.masked.store.v{}i32\0", N);
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut param_tys = vec![ty_i32xn, ty_p0, ty_i32, ty_i1xn];
        llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(
                llvm::core::LLVMVoidTypeInContext(context),
                param_tys.as_mut_ptr(),
                param_tys.len() as u32,
                0,
            ),
            intrinsic,
            [value, value_ptr, alignment, mask].as_mut_ptr(),
            4,
            empty_name.as_ptr(),
        );
    }

    unsafe fn emit_store_vgpr_u64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(elem) != ty_i64 {
            panic!("Type of elem is not i64");
        }

        if llvm::core::LLVMTypeOf(value) != ty_i64 {
            panic!("Type of value is not i64");
        }

        let value_lo = llvm::core::LLVMBuildTrunc(builder, value, ty_i32, empty_name.as_ptr());
        let value_hi = llvm::core::LLVMBuildLShr(
            builder,
            value,
            llvm::core::LLVMConstInt(llvm::core::LLVMInt64TypeInContext(context), 32, 0),
            empty_name.as_ptr(),
        );
        let value_hi = llvm::core::LLVMBuildTrunc(builder, value_hi, ty_i32, empty_name.as_ptr());
        self.emit_store_vgpr_u32(reg, elem, value_lo);
        self.emit_store_vgpr_u32(reg + 1, elem, value_hi);
    }

    unsafe fn emit_store_vgpr_f32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(value) != ty_f32 {
            panic!("Type of value is not f32");
        }

        let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i32, empty_name.as_ptr());
        self.emit_store_vgpr_u32(reg, elem, value);
    }

    unsafe fn emit_store_vgpr_f64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i64, empty_name.as_ptr());
        self.emit_store_vgpr_u64(reg, elem, value);
    }

    unsafe fn emit_store_vgpr_u64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32x2n = llvm::core::LLVMVectorType(ty_i32, N as u32 * 2);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i32x2n, empty_name.as_ptr());

        let mut shuffle_indices = Vec::new();
        for i in 0..N {
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, (i * 2) as u64, 0));
        }

        let value_lo = llvm::core::LLVMBuildShuffleVector(
            builder,
            value,
            llvm::core::LLVMGetPoison(ty_i32x2n),
            llvm::core::LLVMConstVector(shuffle_indices.as_mut_ptr(), N as u32),
            empty_name.as_ptr(),
        );

        let mut shuffle_indices = Vec::new();
        for i in 0..N {
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, (i * 2 + 1) as u64, 0));
        }

        let value_hi = llvm::core::LLVMBuildShuffleVector(
            builder,
            value,
            llvm::core::LLVMGetPoison(ty_i32x2n),
            llvm::core::LLVMConstVector(shuffle_indices.as_mut_ptr(), N as u32),
            empty_name.as_ptr(),
        );

        self.emit_store_vgpr_u32xn::<N>(reg, elem, value_lo, mask);
        self.emit_store_vgpr_u32xn::<N>(reg + 1, elem, value_hi, mask);
    }

    unsafe fn emit_store_vgpr_f32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value =
            llvm::core::LLVMBuildBitCast(self.builder, value, ty_i32xn, empty_name.as_ptr());
        self.emit_store_vgpr_u32xn::<N>(reg, elem, value, mask);
    }

    unsafe fn emit_store_vgpr_f64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(self.context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value_u64xn =
            llvm::core::LLVMBuildBitCast(self.builder, value, ty_i64xn, empty_name.as_ptr());
        self.emit_store_vgpr_u64xn::<N>(reg, elem, value_u64xn, mask);

        if self.use_vgpr_cache {
            self.vgpr_reg_f64_map.get_mut(&reg).unwrap()[elem as usize / N] = value;
        }
    }

    unsafe fn emit_load_stack_sgpr_u32(&mut self, reg: u32) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if reg == 124 {
            llvm::core::LLVMConstInt(ty_i32, 0, 0)
        } else {
            let value_ptr = *self.sgpr_ptr_map.get(&reg).unwrap();
            llvm::core::LLVMBuildLoad2(builder, ty_i32, value_ptr, empty_name.as_ptr())
        }
    }

    unsafe fn emit_load_sgpr_u32(&mut self, reg: u32) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if USE_SGPR_CACHE {
            return self.emit_load_stack_sgpr_u32(reg);
        }

        if reg == 124 {
            llvm::core::LLVMConstInt(ty_i32, 0, 0)
        } else {
            let mut indices = vec![llvm::core::LLVMConstInt(
                llvm::core::LLVMInt64TypeInContext(context),
                reg as u64,
                0,
            )];
            let value_ptr = llvm::core::LLVMBuildGEP2(
                builder,
                ty_i32,
                sgprs_ptr,
                indices.as_mut_ptr(),
                indices.len() as u32,
                empty_name.as_ptr(),
            );
            llvm::core::LLVMBuildLoad2(builder, ty_i32, value_ptr, empty_name.as_ptr())
        }
    }

    unsafe fn emit_load_sgpr_u64(&mut self, reg: u32) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;

        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value_hi = self.emit_load_sgpr_u32(reg + 1);
        let value_lo = self.emit_load_sgpr_u32(reg);

        let value_hi = llvm::core::LLVMBuildZExt(builder, value_hi, ty_i64, empty_name.as_ptr());
        let value_lo = llvm::core::LLVMBuildZExt(builder, value_lo, ty_i64, empty_name.as_ptr());

        let value_hi_shifted = llvm::core::LLVMBuildShl(
            builder,
            value_hi,
            llvm::core::LLVMConstInt(ty_i64, 32, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildOr(builder, value_hi_shifted, value_lo, empty_name.as_ptr())
    }

    unsafe fn emit_store_stack_sgpr_u32(&mut self, reg: u32, value: llvm::prelude::LLVMValueRef) {
        let builder = self.builder;

        if reg != 124 {
            let value_ptr = *self.sgpr_ptr_map.get(&reg).unwrap();
            llvm::core::LLVMBuildStore(builder, value, value_ptr);
        }
    }

    unsafe fn emit_store_sgpr_u32(&mut self, reg: u32, value: llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if USE_SGPR_CACHE {
            return self.emit_store_stack_sgpr_u32(reg, value);
        }

        if reg != 124 {
            let mut indices = vec![llvm::core::LLVMConstInt(
                llvm::core::LLVMInt64TypeInContext(context),
                reg as u64,
                0,
            )];
            let value_ptr = llvm::core::LLVMBuildGEP2(
                builder,
                ty_i32,
                sgprs_ptr,
                indices.as_mut_ptr(),
                indices.len() as u32,
                empty_name.as_ptr(),
            );
            llvm::core::LLVMBuildStore(builder, value, value_ptr);
        }
    }

    unsafe fn emit_store_sgpr_u64(&mut self, reg: u32, value: llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value_lo = llvm::core::LLVMBuildTrunc(builder, value, ty_i32, empty_name.as_ptr());
        let value_hi = llvm::core::LLVMBuildLShr(
            builder,
            value,
            llvm::core::LLVMConstInt(ty_i64, 32, 0),
            empty_name.as_ptr(),
        );
        let value_hi = llvm::core::LLVMBuildTrunc(builder, value_hi, ty_i32, empty_name.as_ptr());

        self.emit_store_sgpr_u32(reg, value_lo);
        self.emit_store_sgpr_u32(reg + 1, value_hi);
    }

    unsafe fn emit_scalar_source_operand_u32(
        &mut self,
        operand: &SourceOperand,
    ) -> llvm::prelude::LLVMValueRef {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstInt(ty_i32, *value as u64, 0)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstInt(ty_i32, *value as u64, 0)
            }
            SourceOperand::ScalarRegister(value) => self.emit_load_sgpr_u32(*value as u32),
            _ => panic!("Unsupported source operand type"),
        }
    }

    unsafe fn emit_scalar_source_operand_u64(
        &mut self,
        operand: &SourceOperand,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstInt(ty_i64, *value as u64, 0)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstInt(ty_i64, *value as u64, 0)
            }
            SourceOperand::FloatConstant(value) => {
                llvm::core::LLVMConstInt(ty_i64, f64::to_bits(*value), 0)
            }
            SourceOperand::ScalarRegister(value) => self.emit_load_sgpr_u64(*value as u32),
            _ => panic!("Unsupported source operand type: {:?}", operand),
        }
    }

    unsafe fn emit_vector_source_operand_u32(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstInt(ty_i32, *value as u64, 0)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstInt(ty_i32, *value as u64, 0)
            }
            SourceOperand::ScalarRegister(value) => self.emit_load_sgpr_u32(*value as u32),
            SourceOperand::VectorRegister(value) => self.emit_load_vgpr_u32(*value as u32, elem),
            _ => panic!("Unsupported source operand type"),
        }
    }

    unsafe fn emit_vector_source_operand_f32(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value) as f64)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value as u32) as f64)
            }
            SourceOperand::FloatConstant(value) => llvm::core::LLVMConstReal(ty_f32, *value),
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);
                llvm::core::LLVMBuildBitCast(builder, value, ty_f32, empty_name.as_ptr())
            }
            SourceOperand::VectorRegister(value) => {
                let value = self.emit_load_vgpr_u32(*value as u32, elem);
                llvm::core::LLVMBuildBitCast(self.builder, value, ty_f32, empty_name.as_ptr())
            }
        }
    }

    unsafe fn emit_vector_source_operand_u64(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstInt(ty_i64, *value as u64, 0)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstInt(ty_i64, *value as u64, 0)
            }
            SourceOperand::FloatConstant(value) => {
                llvm::core::LLVMConstInt(ty_i64, f64::to_bits(*value), 0)
            }
            SourceOperand::ScalarRegister(value) => self.emit_load_sgpr_u64(*value as u32),
            SourceOperand::VectorRegister(value) => self.emit_load_vgpr_u64(*value as u32, elem),
        }
    }

    unsafe fn emit_vector_source_operand_f64(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);
        let empty_name = std::ffi::CString::new("").unwrap();

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstReal(ty_f64, f64::from_bits((*value as u64) << 32))
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstReal(ty_f64, f64::from_bits((*value as u64) << 32))
            }
            SourceOperand::FloatConstant(value) => llvm::core::LLVMConstReal(ty_f64, *value),
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u64(*value as u32);
                llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64, empty_name.as_ptr())
            }
            SourceOperand::VectorRegister(value) => {
                let value = self.emit_load_vgpr_u64(*value as u32, elem);
                llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64, empty_name.as_ptr())
            }
        }
    }

    unsafe fn emit_bit_mask_u32xn<const N: usize>(
        &mut self,
        elem: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let mut bit_mask_values = Vec::new();
        for i in 0..N {
            bit_mask_values.push(llvm::core::LLVMConstInt(ty_i32, (1 << i) << elem, 0));
        }

        let bit_mask =
            llvm::core::LLVMConstVector(bit_mask_values.as_mut_ptr(), bit_mask_values.len() as u32);

        bit_mask
    }

    unsafe fn emit_bits_to_mask_u32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        elem: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
        let ty_i1x32 = llvm::core::LLVMVectorType(ty_i1, 32);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let bit_vec = llvm::core::LLVMBuildBitCast(builder, value, ty_i1x32, empty_name.as_ptr());

        let poison = llvm::core::LLVMGetPoison(ty_i1x32);

        let mut indices_values = Vec::new();
        for i in 0..N {
            indices_values.push(llvm::core::LLVMConstInt(ty_i32, elem as u64 + i as u64, 0));
        }

        let mask = llvm::core::LLVMBuildShuffleVector(
            builder,
            bit_vec,
            poison,
            llvm::core::LLVMConstVector(indices_values.as_mut_ptr(), N as u32),
            empty_name.as_ptr(),
        );

        mask
    }

    unsafe fn emit_vector_source_operand_u64xn<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(self.context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_i64xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value = llvm::core::LLVMConstInt(ty_i64, *value as u64, 0);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstInt(ty_i64, *value as u64, 0);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstInt(ty_i64, f64::to_bits(*value), 0);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u64(*value as u32);

                let value = llvm::core::LLVMBuildInsertElement(
                    builder,
                    llvm::core::LLVMGetUndef(ty_i64xn),
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let value = llvm::core::LLVMBuildShuffleVector(
                    builder,
                    value,
                    poison,
                    zero_vec,
                    empty_name.as_ptr(),
                );
                value
            }
            SourceOperand::VectorRegister(value) => {
                self.emit_load_vgpr_u64xn::<N>(*value as u32, elem, mask)
            }
        }
    }

    unsafe fn emit_vector_source_operand_f64xn<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_f64xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value =
                    llvm::core::LLVMConstReal(ty_f64, f64::from_bits((*value as u64) << 32));
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::IntegerConstant(value) => {
                let value =
                    llvm::core::LLVMConstReal(ty_f64, f64::from_bits((*value as u64) << 32));
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f64, *value);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u64(*value as u32);
                let value =
                    llvm::core::LLVMBuildBitCast(builder, value, ty_f64, empty_name.as_ptr());

                let value = llvm::core::LLVMBuildInsertElement(
                    builder,
                    llvm::core::LLVMGetUndef(ty_f64xn),
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let value = llvm::core::LLVMBuildShuffleVector(
                    builder,
                    value,
                    poison,
                    zero_vec,
                    empty_name.as_ptr(),
                );
                value
            }
            SourceOperand::VectorRegister(value) => {
                self.emit_load_vgpr_f64xn::<N>(*value as u32, elem, mask)
            }
        }
    }

    unsafe fn emit_vector_source_operand_u32xn<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_i32xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value = llvm::core::LLVMConstInt(ty_i32, *value as u64, 0);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstInt(ty_i32, *value as u64, 0);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstInt(ty_i32, f32::to_bits(*value as f32) as u64, 0);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);

                let value = llvm::core::LLVMBuildInsertElement(
                    builder,
                    poison,
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let value = llvm::core::LLVMBuildShuffleVector(
                    builder,
                    value,
                    poison,
                    zero_vec,
                    empty_name.as_ptr(),
                );
                value
            }
            SourceOperand::VectorRegister(value) => {
                self.emit_load_vgpr_u32xn::<N>(*value as u32, elem, mask)
            }
        }
    }

    unsafe fn emit_vector_source_operand_f32xn<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(self.context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_f32xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value as u32) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f32, *value);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);
                let value =
                    llvm::core::LLVMBuildBitCast(builder, value, ty_f32, empty_name.as_ptr());

                let value = llvm::core::LLVMBuildInsertElement(
                    builder,
                    llvm::core::LLVMGetUndef(ty_f32xn),
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let value = llvm::core::LLVMBuildShuffleVector(
                    builder,
                    value,
                    poison,
                    zero_vec,
                    empty_name.as_ptr(),
                );
                value
            }
            SourceOperand::VectorRegister(value) => {
                self.emit_load_vgpr_f32xn::<N>(*value as u32, elem, mask)
            }
        }
    }

    unsafe fn emit_abs(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

        let mut param_tys = vec![ty_f64];
        let intrinsic_name = b"llvm.fabs.f64\0";
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut param_tys = vec![ty_f64];
        let abs_value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 1, 0),
            intrinsic,
            [value].as_mut_ptr(),
            1,
            empty_name.as_ptr(),
        );

        abs_value
    }

    unsafe fn emit_exp_f64(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let mut param_tys = vec![ty_f64, ty_i32];
        let intrinsic_name = b"llvm.frexp.f64\0";
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut return_tys = vec![ty_f64, ty_i32];
        let mut param_tys = vec![ty_f64];
        let frexp_value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(
                llvm::core::LLVMStructTypeInContext(
                    context,
                    return_tys.as_mut_ptr(),
                    return_tys.len() as u32,
                    0,
                ),
                param_tys.as_mut_ptr(),
                1,
                0,
            ),
            intrinsic,
            [value].as_mut_ptr(),
            1,
            empty_name.as_ptr(),
        );

        let exp_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 1, empty_name.as_ptr());
        exp_value
    }

    unsafe fn _emit_exp_f64xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

        let mut param_tys = vec![ty_f64xn, ty_i32xn];
        let intrinsic_name = format!("llvm.frexp.v{}f64\0", N);
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut return_tys = vec![ty_f64xn, ty_i32xn];
        let mut param_tys = vec![ty_f64xn];
        let frexp_value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(
                llvm::core::LLVMStructTypeInContext(
                    context,
                    return_tys.as_mut_ptr(),
                    return_tys.len() as u32,
                    0,
                ),
                param_tys.as_mut_ptr(),
                1,
                0,
            ),
            intrinsic,
            [value].as_mut_ptr(),
            1,
            empty_name.as_ptr(),
        );

        let exp_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 1, empty_name.as_ptr());
        exp_value
    }

    unsafe fn emit_abs_neg(
        &mut self,
        abs: u8,
        neg: u8,
        value: llvm::prelude::LLVMValueRef,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

        let value = if (abs >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f64);
            self.emit_abs(value)
        } else {
            value
        };

        let value = if (neg >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f64);
            llvm::core::LLVMBuildFNeg(builder, value, empty_name.as_ptr())
        } else {
            value
        };

        value
    }

    unsafe fn emit_abs_neg_f64xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        abs: u8,
        neg: u8,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

        let negative_zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i64, 0x8000000000000000, 0); N].as_mut_ptr(),
            N as u32,
        );

        let value = if (abs >> idx) & 1 != 0 {
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i64xn, empty_name.as_ptr());
            let value =
                llvm::core::LLVMBuildAnd(builder, value, negative_zero_vec, empty_name.as_ptr());
            let value =
                llvm::core::LLVMBuildXor(builder, value, negative_zero_vec, empty_name.as_ptr());
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_f64xn, empty_name.as_ptr());
            value
        } else {
            value
        };

        let value = if (neg >> idx) & 1 != 0 {
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i64xn, empty_name.as_ptr());
            let value =
                llvm::core::LLVMBuildXor(builder, value, negative_zero_vec, empty_name.as_ptr());
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_f64xn, empty_name.as_ptr());
            value
        } else {
            value
        };

        value
    }

    unsafe fn emit_omod_clamp(
        &mut self,
        omod: u8,
        clamp: u8,
        value: llvm::prelude::LLVMValueRef,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

        let value = if (omod >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f64);
            let two = llvm::core::LLVMConstReal(ty_f64, 2.0);
            let four = llvm::core::LLVMConstReal(ty_f64, 4.0);
            let half = llvm::core::LLVMConstReal(ty_f64, 0.5);

            match idx {
                0 => llvm::core::LLVMBuildFMul(builder, value, two, empty_name.as_ptr()),
                1 => llvm::core::LLVMBuildFMul(builder, value, four, empty_name.as_ptr()),
                2 => llvm::core::LLVMBuildFMul(builder, value, half, empty_name.as_ptr()),
                _ => value,
            }
        } else {
            value
        };

        let value = if (clamp >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f64);
            let zero = llvm::core::LLVMConstReal(ty_f64, 0.0);
            let one = llvm::core::LLVMConstReal(ty_f64, 1.0);

            let mut param_tys = vec![ty_f64];
            let intrinsic_name = b"llvm.minnum.f64\0";
            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                intrinsic_name.as_ptr() as *const _,
                intrinsic_name.len() as usize,
            );
            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                self.module,
                intrinsic_id,
                param_tys.as_mut_ptr(),
                param_tys.len() as usize,
            );
            let mut param_tys = vec![ty_f64, ty_f64];
            let min_value = llvm::core::LLVMBuildCall2(
                builder,
                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 2, 0),
                intrinsic,
                [value, one].as_mut_ptr(),
                2,
                empty_name.as_ptr(),
            );

            let mut param_tys = vec![ty_f64];
            let intrinsic_name = b"llvm.maxnum.f64\0";
            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                intrinsic_name.as_ptr() as *const _,
                intrinsic_name.len() as usize,
            );
            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                self.module,
                intrinsic_id,
                param_tys.as_mut_ptr(),
                param_tys.len() as usize,
            );
            let mut param_tys = vec![ty_f64, ty_f64];
            let max_value = llvm::core::LLVMBuildCall2(
                builder,
                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 2, 0),
                intrinsic,
                [min_value, zero].as_mut_ptr(),
                2,
                empty_name.as_ptr(),
            );

            max_value
        } else {
            value
        };

        value
    }

    unsafe fn emit_omod_clamp_f64xn<const N: usize>(
        &mut self,
        omod: u8,
        clamp: u8,
        value: llvm::prelude::LLVMValueRef,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        let value = if (omod >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f64);
            let two = llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstReal(ty_f64, 2.0); N].as_mut_ptr(),
                N as u32,
            );
            let four = llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstReal(ty_f64, 4.0); N].as_mut_ptr(),
                N as u32,
            );
            let half = llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstReal(ty_f64, 0.5); N].as_mut_ptr(),
                N as u32,
            );

            match idx {
                0 => llvm::core::LLVMBuildFMul(builder, value, two, empty_name.as_ptr()),
                1 => llvm::core::LLVMBuildFMul(builder, value, four, empty_name.as_ptr()),
                2 => llvm::core::LLVMBuildFMul(builder, value, half, empty_name.as_ptr()),
                _ => value,
            }
        } else {
            value
        };

        let value = if (clamp >> idx) & 1 != 0 {
            let zero = llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstReal(ty_f64, 0.0); N].as_mut_ptr(),
                N as u32,
            );
            let one = llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstReal(ty_f64, 1.0); N].as_mut_ptr(),
                N as u32,
            );

            let mut param_tys = vec![ty_f64xn];
            let intrinsic_name = format!("llvm.minnum.v{}f64\0", N);
            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                intrinsic_name.as_ptr() as *const _,
                intrinsic_name.len() as usize,
            );
            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                self.module,
                intrinsic_id,
                param_tys.as_mut_ptr(),
                param_tys.len() as usize,
            );
            let mut param_tys = vec![ty_f64xn, ty_f64xn];
            let min_value = llvm::core::LLVMBuildCall2(
                builder,
                llvm::core::LLVMFunctionType(ty_f64xn, param_tys.as_mut_ptr(), 2, 0),
                intrinsic,
                [value, one].as_mut_ptr(),
                2,
                empty_name.as_ptr(),
            );

            let mut param_tys = vec![ty_f64xn];
            let intrinsic_name = format!("llvm.maxnum.v{}f64\0", N);
            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                intrinsic_name.as_ptr() as *const _,
                intrinsic_name.len() as usize,
            );
            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                self.module,
                intrinsic_id,
                param_tys.as_mut_ptr(),
                param_tys.len() as usize,
            );
            let mut param_tys = vec![ty_f64xn, ty_f64xn];
            let max_value = llvm::core::LLVMBuildCall2(
                builder,
                llvm::core::LLVMFunctionType(ty_f64xn, param_tys.as_mut_ptr(), 2, 0),
                intrinsic,
                [min_value, zero].as_mut_ptr(),
                2,
                empty_name.as_ptr(),
            );

            max_value
        } else {
            value
        };

        value
    }

    unsafe fn emit_fma_f64xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
        value2: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        let mut param_tys = vec![ty_f64xn];
        let intrinsic_name = format!("llvm.fma.v{}f64\0", N);
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut param_tys = vec![ty_f64xn, ty_f64xn, ty_f64xn];
        let fma_value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(ty_f64xn, param_tys.as_mut_ptr(), 3, 0),
            intrinsic,
            [value0, value1, value2].as_mut_ptr(),
            3,
            empty_name.as_ptr(),
        );
        fma_value
    }

    unsafe fn emit_fadd(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let add_value = llvm::core::LLVMBuildFAdd(builder, value0, value1, empty_name.as_ptr());
        add_value
    }

    unsafe fn emit_u32_to_f64xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = llvm::core::LLVMBuildZExt(builder, value, ty_i64xn, empty_name.as_ptr());

        let value = llvm::core::LLVMBuildUIToFP(builder, value, ty_f64xn, empty_name.as_ptr());

        value
    }

    unsafe fn emit_i32_to_f64xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = llvm::core::LLVMBuildSIToFP(builder, value, ty_f64xn, empty_name.as_ptr());

        value
    }

    unsafe fn emit_exp2_f64xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        let mut param_tys = vec![ty_f64xn];
        let intrinsic_name = format!("llvm.exp2.v{}f64\0", N);
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut param_tys = vec![ty_f64xn];
        let exp2_value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(ty_f64xn, param_tys.as_mut_ptr(), 1, 0),
            intrinsic,
            [value0].as_mut_ptr(),
            1,
            empty_name.as_ptr(),
        );
        exp2_value
    }

    unsafe fn _emit_ldexp_f64xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

        let mut param_tys = vec![ty_f64xn, ty_i32xn];
        let intrinsic_name = format!("llvm.ldexp.v{}f64\0", N);
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
            intrinsic_name.as_ptr() as *const _,
            intrinsic_name.len() as usize,
        );
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_tys.as_mut_ptr(),
            param_tys.len() as usize,
        );
        let mut param_tys = vec![ty_f64xn, ty_i32xn];
        let ldexp_value = llvm::core::LLVMBuildCall2(
            builder,
            llvm::core::LLVMFunctionType(ty_f64xn, param_tys.as_mut_ptr(), 2, 0),
            intrinsic,
            [value0, value1].as_mut_ptr(),
            2,
            empty_name.as_ptr(),
        );
        ldexp_value
    }

    unsafe fn emit_fmul(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let mul_value = llvm::core::LLVMBuildFMul(builder, value0, value1, empty_name.as_ptr());
        mul_value
    }

    unsafe fn emit_concat_pair<const N: usize>(
        &mut self,
        values: &Vec<llvm::prelude::LLVMValueRef>,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let context = self.context;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let len = values.len() as u32;

        let mut index_values = Vec::new();
        for i in 0..(2 * N) {
            index_values.push(llvm::core::LLVMConstInt(ty_i32, i as u64, 0));
        }

        let indices =
            llvm::core::LLVMConstVector(index_values.as_mut_ptr(), index_values.len() as u32);

        let mut result = Vec::new();
        for i in (0..len).step_by(2) {
            let cmp_value = llvm::core::LLVMBuildShuffleVector(
                builder,
                values[i as usize],
                values[i as usize + 1],
                indices,
                empty_name.as_ptr(),
            );
            result.push(cmp_value);
        }
        result
    }

    unsafe fn emit_concat<const N: usize>(
        &mut self,
        values: &Vec<llvm::prelude::LLVMValueRef>,
    ) -> llvm::prelude::LLVMValueRef {
        let mut len = values.len() as u32;
        let mut values = values.clone();
        while len > 1 {
            let new_values = self.emit_concat_pair::<N>(&values);
            values = new_values;
            len = values.len() as u32;
        }
        values[0]
    }

    unsafe fn emit_alloc_registers(&mut self, reg_usage: &RegisterUsage) {
        if USE_SGPR_CACHE {
            let sgprs: Vec<u32> = reg_usage
                .use_sgprs
                .union(&reg_usage.def_sgprs)
                .cloned()
                .collect::<Vec<_>>();

            for sgpr in &sgprs {
                let sgpr_ptr = llvm::core::LLVMBuildAlloca(
                    self.builder,
                    llvm::core::LLVMInt32TypeInContext(self.context),
                    std::ffi::CString::new(format!("sgpr{}", sgpr))
                        .unwrap()
                        .as_ptr(),
                );
                self.sgpr_ptr_map.insert(*sgpr, sgpr_ptr);
            }
        }

        if self.use_vgpr_cache {
            let vgprs: Vec<u32> = reg_usage
                .use_vgprs
                .union(&reg_usage.def_vgprs)
                .cloned()
                .collect::<Vec<_>>();

            for vgpr in &vgprs {
                let mut vgpr_ptr = [std::ptr::null_mut(); 32 / SIMD_WIDTH];

                for i in 0..(32 / SIMD_WIDTH) {
                    vgpr_ptr[i] = llvm::core::LLVMBuildAlloca(
                        self.builder,
                        llvm::core::LLVMVectorType(
                            llvm::core::LLVMInt32TypeInContext(self.context),
                            SIMD_WIDTH as u32,
                        ),
                        std::ffi::CString::new(format!("vgpr{}.{}", vgpr, i))
                            .unwrap()
                            .as_ptr(),
                    );
                }
                self.vgpr_ptr_map.insert(*vgpr, vgpr_ptr);
            }
        }
    }

    unsafe fn emit_restore_stack(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        reg_usage: &RegisterUsage,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        if USE_SGPR_CACHE {
            for sgpr in &reg_usage.incomming_sgprs {
                let sgpr_ptr = *self.sgpr_ptr_map.get(sgpr).unwrap();
                let context = self.context;
                let builder = self.builder;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                let mut indices = vec![llvm::core::LLVMConstInt(
                    llvm::core::LLVMInt64TypeInContext(context),
                    *sgpr as u64,
                    0,
                )];
                let value_ptr = llvm::core::LLVMBuildGEP2(
                    builder,
                    ty_i32,
                    self.sgprs_ptr,
                    indices.as_mut_ptr(),
                    indices.len() as u32,
                    empty_name.as_ptr(),
                );

                llvm::core::LLVMBuildMemCpy(
                    builder,
                    sgpr_ptr,
                    4,
                    value_ptr,
                    4,
                    llvm::core::LLVMConstInt(ty_i32, 4, 0),
                );
            }
        }

        if self.use_vgpr_cache {
            for vgpr in &reg_usage.incomming_vgprs {
                let context = self.context;
                let builder = self.builder;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                const N: usize = SIMD_WIDTH;

                for i in (0..32).step_by(N) {
                    let mut indices: Vec<*mut llvm_sys::LLVMValue> =
                        vec![llvm::core::LLVMConstInt(
                            llvm::core::LLVMInt64TypeInContext(context),
                            *vgpr as u64 * 32 + i as u64,
                            0,
                        )];
                    let value_ptr = llvm::core::LLVMBuildGEP2(
                        builder,
                        ty_i32,
                        self.vgprs_ptr,
                        indices.as_mut_ptr(),
                        indices.len() as u32,
                        empty_name.as_ptr(),
                    );

                    llvm::core::LLVMBuildMemCpy(
                        builder,
                        self.vgpr_ptr_map.get(vgpr).unwrap()[i / N],
                        4,
                        value_ptr,
                        4,
                        llvm::core::LLVMConstInt(ty_i32, 4 * N as u64, 0),
                    );
                }
            }
        }

        if self.use_scc_cache {
            let context = self.context;
            let builder = self.builder;
            let empty_name = std::ffi::CString::new("").unwrap();
            let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);

            self.local_scc_ptr =
                llvm::core::LLVMBuildAlloca(self.builder, ty_i8, empty_name.as_ptr());

            llvm::core::LLVMBuildMemCpy(
                builder,
                self.local_scc_ptr,
                1,
                self.scc_ptr,
                1,
                llvm::core::LLVMConstInt(ty_i8, 1, 0),
            );
        }

        bb
    }

    unsafe fn emit_restore_registers(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        reg_usage: &RegisterUsage,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        if self.use_vgpr_cache {
            let vgprs: Vec<u32> = reg_usage
                .use_vgprs
                .union(&reg_usage.def_vgprs)
                .cloned()
                .collect::<Vec<_>>();

            self.vgpr_reg_map.clear();
            self.vgpr_reg_f64_map.clear();

            for vgpr in &vgprs {
                self.vgpr_incomming_reg_map
                    .insert(*vgpr, [std::ptr::null_mut(); 32 / SIMD_WIDTH]);
                self.vgpr_reg_map
                    .insert(*vgpr, [std::ptr::null_mut(); 32 / SIMD_WIDTH]);
                self.vgpr_reg_f64_map
                    .insert(*vgpr, [std::ptr::null_mut(); 32 / SIMD_WIDTH]);
            }
            for vgpr in &vgprs {
                let context = self.context;
                let builder = self.builder;
                let empty_name = std::ffi::CString::new("").unwrap();

                const N: usize = SIMD_WIDTH;

                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let value_ptr = self.vgpr_ptr_map.get_mut(vgpr).unwrap()[i / N];
                    let value = llvm::core::LLVMBuildLoad2(
                        builder,
                        ty_i32xn,
                        value_ptr,
                        empty_name.as_ptr(),
                    );
                    self.vgpr_reg_map.get_mut(vgpr).unwrap()[i / N] = value;
                    self.vgpr_incomming_reg_map.get_mut(vgpr).unwrap()[i / N] = value;
                }
            }
            self.exec_value = self.emit_load_sgpr_u32(126);
        }

        bb
    }

    unsafe fn emit_save_stack(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        reg_usage: &RegisterUsage,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        if USE_SGPR_CACHE {
            for sgpr in &reg_usage.def_sgprs {
                let sgpr_ptr = *self.sgpr_ptr_map.get(sgpr).unwrap();
                let mut indices = vec![llvm::core::LLVMConstInt(
                    llvm::core::LLVMInt64TypeInContext(context),
                    *sgpr as u64,
                    0,
                )];
                let value_ptr = llvm::core::LLVMBuildGEP2(
                    builder,
                    ty_i32,
                    self.sgprs_ptr,
                    indices.as_mut_ptr(),
                    indices.len() as u32,
                    empty_name.as_ptr(),
                );

                llvm::core::LLVMBuildMemCpy(
                    builder,
                    value_ptr,
                    4,
                    sgpr_ptr,
                    4,
                    llvm::core::LLVMConstInt(ty_i32, 4, 0),
                );
            }
        }
        bb
    }

    unsafe fn emit_save_registers(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        reg_usage: &RegisterUsage,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        if self.use_vgpr_cache {
            for vgpr in &reg_usage.def_vgprs {
                const N: usize = SIMD_WIDTH;

                for i in (0..32).step_by(N) {
                    let value_ptr = self.vgpr_ptr_map.get(vgpr).unwrap()[i / N];
                    let value = self.vgpr_reg_map.get(vgpr).unwrap()[i / N];

                    let incomming_value = self.vgpr_incomming_reg_map.get(vgpr).unwrap()[i / N];

                    let mask = self.emit_bits_to_mask_u32xn::<N>(self.exec_value, i as u32);

                    let value = llvm::core::LLVMBuildSelect(
                        builder,
                        mask,
                        value,
                        incomming_value,
                        empty_name.as_ptr(),
                    );

                    llvm::core::LLVMBuildStore(builder, value, value_ptr);
                }
            }
        }
        bb
    }

    unsafe fn emit_terminator(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &InstFormat,
        pc: u64,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        match inst {
            InstFormat::SOPP(inst) => match inst.op {
                I::S_CBRANCH_EXECZ => {
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let exec_value = self.emit_load_sgpr_u32(126);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntEQ,
                        exec_value,
                        llvm::core::LLVMConstInt(llvm::core::LLVMInt32TypeInContext(context), 0, 0),
                        empty_name.as_ptr(),
                    );

                    let pc_value_if_false =
                        llvm::core::LLVMConstInt(ty_i64, (pc as i64 + 4) as u64, 0);

                    let pc_value_if_true = llvm::core::LLVMConstInt(
                        ty_i64,
                        (pc as i64 + 4 + (inst.simm16 as i16 as i64 * 4)) as u64,
                        0,
                    );

                    let next_pc_value = llvm::core::LLVMBuildSelect(
                        builder,
                        cmp_value,
                        pc_value_if_true,
                        pc_value_if_false,
                        empty_name.as_ptr(),
                    );

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);
                }
                I::S_CBRANCH_VCCNZ => {
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let vcc_value = self.emit_load_sgpr_u32(106);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        vcc_value,
                        llvm::core::LLVMConstInt(llvm::core::LLVMInt32TypeInContext(context), 0, 0),
                        empty_name.as_ptr(),
                    );

                    let pc_value_if_false =
                        llvm::core::LLVMConstInt(ty_i64, (pc as i64 + 4) as u64, 0);

                    let pc_value_if_true = llvm::core::LLVMConstInt(
                        ty_i64,
                        (pc as i64 + 4 + (inst.simm16 as i16 as i64 * 4)) as u64,
                        0,
                    );

                    let next_pc_value = llvm::core::LLVMBuildSelect(
                        builder,
                        cmp_value,
                        pc_value_if_true,
                        pc_value_if_false,
                        empty_name.as_ptr(),
                    );

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);
                }
                I::S_CBRANCH_SCC0 => {
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let scc_value = self.emit_load_scc_u8();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntEQ,
                        scc_value,
                        llvm::core::LLVMConstInt(ty_i8, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let pc_value_if_false =
                        llvm::core::LLVMConstInt(ty_i64, (pc as i64 + 4) as u64, 0);

                    let pc_value_if_true = llvm::core::LLVMConstInt(
                        ty_i64,
                        (pc as i64 + 4 + (inst.simm16 as i16 as i64 * 4)) as u64,
                        0,
                    );

                    let next_pc_value = llvm::core::LLVMBuildSelect(
                        builder,
                        cmp_value,
                        pc_value_if_true,
                        pc_value_if_false,
                        empty_name.as_ptr(),
                    );

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);
                }
                I::S_CBRANCH_SCC1 => {
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let scc_value = self.emit_load_scc_u8();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        scc_value,
                        llvm::core::LLVMConstInt(ty_i8, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let pc_value_if_false =
                        llvm::core::LLVMConstInt(ty_i64, (pc as i64 + 4) as u64, 0);

                    let pc_value_if_true = llvm::core::LLVMConstInt(
                        ty_i64,
                        (pc as i64 + 4 + (inst.simm16 as i16 as i64 * 4)) as u64,
                        0,
                    );

                    let next_pc_value = llvm::core::LLVMBuildSelect(
                        builder,
                        cmp_value,
                        pc_value_if_true,
                        pc_value_if_false,
                        empty_name.as_ptr(),
                    );

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);
                }
                I::S_BRANCH => {
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let next_pc_value = llvm::core::LLVMConstInt(
                        ty_i64,
                        (pc as i64 + 4 + (inst.simm16 as i16 as i64 * 4)) as u64,
                        0,
                    );

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);
                }
                I::S_ENDPGM => {
                    self.ret_value = llvm::core::LLVMConstInt(
                        llvm::core::LLVMInt32TypeInContext(context),
                        Signals::EndOfProgram as u64,
                        0,
                    );
                }
                I::S_BARRIER_WAIT => {
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                    let next_pc_value = llvm::core::LLVMConstInt(ty_i64, (pc as i64 + 4) as u64, 0);

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);

                    self.ret_value = llvm::core::LLVMConstInt(
                        llvm::core::LLVMInt32TypeInContext(context),
                        Signals::Switch as u64,
                        0,
                    );
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }
        bb
    }

    unsafe fn emit_instruction(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &InstFormat,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst {
            InstFormat::SOPP(inst) => match inst.op {
                I::S_CLAUSE => {}
                I::S_WAIT_KMCNT => {}
                I::S_DELAY_ALU => {}
                I::S_WAIT_ALU => {}
                I::S_WAIT_LOADCNT => {}
                I::S_WAIT_BVHCNT => {}
                I::S_WAIT_SAMPLECNT => {}
                I::S_WAIT_STORECNT => {}
                I::S_WAIT_LOADCNT_DSCNT => {}
                I::S_WAIT_DSCNT => {}
                I::S_NOP => {}
                I::S_SENDMSG => {}
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOPC(inst) => match inst.op {
                I::V_CMP_GT_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_EQ_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_NE_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntNE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntNE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_GT_U64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_GT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_LT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_NLT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_NGT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMP_LE_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(106, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMPX_NGT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAnd(
                            builder,
                            d_value,
                            exec_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(126, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMPX_NGE_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAnd(
                            builder,
                            d_value,
                            exec_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(126, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMPX_LT_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAnd(
                            builder,
                            d_value,
                            exec_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(126, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMPX_EQ_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAnd(
                            builder,
                            d_value,
                            exec_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(126, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                I::V_CMPX_LT_I32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAnd(
                            builder,
                            d_value,
                            exec_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(126, d_value);
                    } else {
                        bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            (bb, cmp_value)
                        });
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP1(inst) => match inst.op {
                I::V_CVT_F64_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let d_value = emitter.emit_u32_to_f64xn::<N>(s0_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildUIToFP(
                                builder,
                                s0_value,
                                llvm::core::LLVMDoubleTypeInContext(context),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildBitCast(
                                builder,
                                d_value,
                                llvm::core::LLVMInt64TypeInContext(context),
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MOV_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let d_value = s0_value;

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let d_value = s0_value;

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_RCP_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstReal(ty_f64, 1.0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                llvm::core::LLVMConstReal(
                                    llvm::core::LLVMDoubleTypeInContext(context),
                                    1.0,
                                ),
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_RSQ_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let mut param_tys = vec![ty_f64xn];
                            let intrinsic_name = format!("llvm.sqrt.v{}f64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64xn];
                            let sqrt_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64xn,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstReal(ty_f64, 1.0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                sqrt_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.sqrt.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64];
                            let intrinsic_call = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                llvm::core::LLVMConstReal(ty_f64, 1.0),
                                intrinsic_call,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_RNDNE_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let mut param_tys = vec![ty_f64xn];
                            let intrinsic_name = format!("llvm.roundeven.v{}f64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64xn];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64xn,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.roundeven.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_FRACT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let mut param_tys = vec![ty_f64xn];
                            let intrinsic_name = format!("llvm.floor.v{}f64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64xn];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64xn,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildFSub(
                                builder,
                                s0_value,
                                d_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.floor.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildFSub(
                                builder,
                                s0_value,
                                d_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CVT_I32_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let d_value = llvm::core::LLVMBuildFPToSI(
                                builder,
                                s0_value,
                                ty_i32xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildFPToSI(
                                builder,
                                s0_value,
                                llvm::core::LLVMInt32TypeInContext(context),
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CVT_F64_I32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let d_value = llvm::core::LLVMBuildSIToFP(
                                builder,
                                s0_value,
                                ty_f64xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildSIToFP(
                                builder,
                                s0_value,
                                llvm::core::LLVMDoubleTypeInContext(context),
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CVT_U32_F32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                            let d_value = llvm::core::LLVMBuildFPToUI(
                                builder,
                                s0_value,
                                ty_i32xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildFPToUI(
                                builder,
                                s0_value,
                                llvm::core::LLVMInt32TypeInContext(context),
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CVT_F32_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let d_value = llvm::core::LLVMBuildUIToFP(
                                builder,
                                s0_value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildUIToFP(
                                builder,
                                s0_value,
                                ty_f32,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_RCP_IFLAG_F32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstReal(ty_f32, 1.0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                llvm::core::LLVMConstReal(ty_f32, 1.0),
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_READFIRSTLANE_B32 => {
                    let emitter = self;
                    let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let mut param_tys = vec![ty_i32];
                    let intrinsic_name = "llvm.cttz.i32\0";
                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                        intrinsic_name.as_ptr() as *const _,
                        intrinsic_name.len() as usize,
                    );
                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                        emitter.module,
                        intrinsic_id,
                        param_tys.as_mut_ptr(),
                        param_tys.len() as usize,
                    );
                    let mut param_tys = vec![ty_i32, ty_i1];
                    let elem = llvm::core::LLVMBuildCall2(
                        builder,
                        llvm::core::LLVMFunctionType(
                            ty_i32,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as u32,
                            0,
                        ),
                        intrinsic,
                        [exec_value, llvm::core::LLVMConstInt(ty_i1, 0, 0)].as_mut_ptr(),
                        2,
                        empty_name.as_ptr(),
                    );

                    let elem = llvm::core::LLVMBuildAnd(
                        builder,
                        elem,
                        llvm::core::LLVMConstInt(ty_i32, 31, 0),
                        empty_name.as_ptr(),
                    );

                    let elem =
                        llvm::core::LLVMBuildZExt(builder, elem, ty_i64, empty_name.as_ptr());

                    let d_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                    emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP2(inst) => match inst.op {
                I::V_ADD_NC_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_AND_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_XOR_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildXor(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let d_value = llvm::core::LLVMBuildXor(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_LSHLREV_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    31,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_LSHRREV_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildLShr(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    31,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildLShr(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CNDMASK_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let vcc_value = emitter.emit_load_sgpr_u32(106);
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                            let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cond,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);

                            let elem_i32 = llvm::core::LLVMBuildTrunc(
                                emitter.builder,
                                elem,
                                llvm::core::LLVMInt32TypeInContext(context),
                                empty_name.as_ptr(),
                            );

                            let vcc_value = emitter.emit_vcc_bit(elem_i32);

                            let d_value = llvm::core::LLVMBuildSelect(
                                emitter.builder,
                                vcc_value,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MUL_F32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = emitter.emit_fmul(s0_value, s1_value);

                            emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
                            let d_value = llvm::core::LLVMBuildFMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_ADD_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = emitter.emit_fadd(s0_value, s1_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let d_value = llvm::core::LLVMBuildFAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MUL_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let d_value = emitter.emit_fmul(s0_value, s1_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
                            let d_value = llvm::core::LLVMBuildFMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MAX_NUM_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

                            let mut param_tys = vec![ty_f64xn];
                            let intrinsic_name = format!("llvm.maxnum.v{}f64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64xn, ty_f64xn];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64xn,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value, s1_value].as_mut_ptr(),
                                2,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.maxnum.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64, ty_f64];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [s0_value, s1_value].as_mut_ptr(),
                                2,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3(inst) => match inst.op {
                I::V_ADD_NC_U16 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s0_value,
                                ty_i16xn,
                                empty_name.as_ptr(),
                            );

                            let s1_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s1_value,
                                ty_i16xn,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildZExt(
                                builder,
                                d_value,
                                ty_i32xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s0_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s0_value,
                                ty_i16,
                                empty_name.as_ptr(),
                            );

                            let s1_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s1_value,
                                ty_i16,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildZExt(
                                builder,
                                d_value,
                                ty_i32,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_BFE_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);

                            let s1_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s1_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        31,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let shifted = llvm::core::LLVMBuildLShr(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let s2_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s2_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        31,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let mask_value = llvm::core::LLVMBuildShl(
                                builder,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        1,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            let mask_value = llvm::core::LLVMBuildSub(
                                builder,
                                mask_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        1,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                shifted,
                                mask_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);

                            let s1_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s1_value,
                                llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                empty_name.as_ptr(),
                            );
                            let shifted = llvm::core::LLVMBuildLShr(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let s2_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s2_value,
                                llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                empty_name.as_ptr(),
                            );
                            let mask = llvm::core::LLVMBuildShl(
                                builder,
                                llvm::core::LLVMConstInt(ty_i32, 1, 0),
                                s2_value,
                                empty_name.as_ptr(),
                            );
                            let mask = llvm::core::LLVMBuildSub(
                                builder,
                                mask,
                                llvm::core::LLVMConstInt(ty_i32, 1, 0),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                shifted,
                                mask,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_LSHRREV_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildLShr(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildLShr(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_LSHLREV_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_ASHRREV_I32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAShr(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildAShr(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_LSHL_OR_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);

                            let s1_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s1_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        31,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let shifted = llvm::core::LLVMBuildShl(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildOr(
                                builder,
                                shifted,
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);

                            let s1_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s1_value,
                                llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                empty_name.as_ptr(),
                            );
                            let shifted = llvm::core::LLVMBuildShl(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildOr(
                                builder,
                                shifted,
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CNDMASK_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let vcc_value = emitter.emit_scalar_source_operand_u32(&inst.src2);
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                            let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cond,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s2_value = emitter.emit_scalar_source_operand_u32(&inst.src2);

                            let elem_i32 = llvm::core::LLVMBuildTrunc(
                                emitter.builder,
                                elem,
                                llvm::core::LLVMInt32TypeInContext(context),
                                empty_name.as_ptr(),
                            );
                            let elem_shifted = llvm::core::LLVMBuildShl(
                                emitter.builder,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    1,
                                    0,
                                ),
                                elem_i32,
                                empty_name.as_ptr(),
                            );

                            let elem_masked = llvm::core::LLVMBuildAnd(
                                emitter.builder,
                                s2_value,
                                elem_shifted,
                                empty_name.as_ptr(),
                            );

                            let cond = llvm::core::LLVMBuildICmp(
                                emitter.builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                elem_masked,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    0,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildSelect(
                                emitter.builder,
                                cond,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CMP_GT_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntUGT,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_EQ_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntEQ,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_NE_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntNE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntNE,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_EQ_U16 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s0_value,
                                ty_i16,
                                empty_name.as_ptr(),
                            );
                            let s1_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s1_value,
                                ty_i16,
                                empty_name.as_ptr(),
                            );

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let s0_value = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    s0_value,
                                    ty_i16,
                                    empty_name.as_ptr(),
                                );

                                let s1_value = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    s1_value,
                                    ty_i16,
                                    empty_name.as_ptr(),
                                );

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntEQ,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_GT_U16 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s0_value,
                                ty_i16,
                                empty_name.as_ptr(),
                            );
                            let s1_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                s1_value,
                                ty_i16,
                                empty_name.as_ptr(),
                            );

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let s0_value = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    s0_value,
                                    ty_i16,
                                    empty_name.as_ptr(),
                                );

                                let s1_value = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    s1_value,
                                    ty_i16,
                                    empty_name.as_ptr(),
                                );

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntUGT,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_LSHLREV_B64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u64xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, 0x3F, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u64(&inst.src1, elem);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(ty_i64, 0x3F, 0),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CMP_NLT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealOLT,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );
                                let cmp_value = llvm::core::LLVMBuildNot(
                                    builder,
                                    cmp_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_NGT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let cmp_value =
                                llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr());

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealOGT,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );
                                let cmp_value = llvm::core::LLVMBuildNot(
                                    builder,
                                    cmp_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_LT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealOLT,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_GT_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOGT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealOGT,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_LG_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealONE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealONE,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_LE_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealOLE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealOLE,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_CMP_NEQ_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let cmp_value = llvm::core::LLVMBuildFCmp(
                                builder,
                                llvm::LLVMRealPredicate::LLVMRealONE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s0_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                                let s1_value =
                                    emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                                let cmp_value = llvm::core::LLVMBuildFCmp(
                                    builder,
                                    llvm::LLVMRealPredicate::LLVMRealONE,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_ADD_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let d_value = emitter.emit_fadd(s0_value, s1_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                            let s0_value = emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                            let s1_value = emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                            let d_value = llvm::core::LLVMBuildFAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MUL_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let d_value = emitter.emit_fmul(s0_value, s1_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                            let s0_value = emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                            let s1_value = emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                            let d_value = llvm::core::LLVMBuildFMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_FMA_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src2, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);
                            let s2_value =
                                emitter.emit_abs_neg_f64xn::<N>(s2_value, inst.abs, inst.neg, 2);

                            let d_value = emitter.emit_fma_f64xn::<N>(s0_value, s1_value, s2_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_f64(&inst.src2, elem);

                            let s0_value = emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                            let s1_value = emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);
                            let s2_value = emitter.emit_abs_neg(inst.abs, inst.neg, s2_value, 2);

                            let mut param_tys = vec![ty_f64];

                            let intrinsic_name = b"llvm.fma.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );

                            let mut param_tys = vec![ty_f64, ty_f64, ty_f64];

                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 3, 0),
                                intrinsic,
                                [s0_value, s1_value, s2_value].as_mut_ptr(),
                                3,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_DIV_FMAS_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);
                        let vcc_value = emitter.emit_load_sgpr_u32(106);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                            let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src2, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);
                            let s2_value =
                                emitter.emit_abs_neg_f64xn::<N>(s2_value, inst.abs, inst.neg, 2);

                            let fma_result =
                                emitter.emit_fma_f64xn::<N>(s0_value, s1_value, s2_value);

                            let muled = llvm::core::LLVMBuildFMul(
                                builder,
                                fma_result,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstReal(
                                        llvm::core::LLVMDoubleTypeInContext(context),
                                        f64::from_bits(0x43F0000000000000),
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cond,
                                muled,
                                fma_result,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_f64(&inst.src2, elem);

                            let vcc_value = emitter.emit_load_sgpr_u32(106);

                            let elem_i32 = llvm::core::LLVMBuildTrunc(
                                emitter.builder,
                                elem,
                                llvm::core::LLVMInt32TypeInContext(context),
                                empty_name.as_ptr(),
                            );
                            let elem_shifted = llvm::core::LLVMBuildShl(
                                emitter.builder,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    1,
                                    0,
                                ),
                                elem_i32,
                                empty_name.as_ptr(),
                            );

                            let elem_masked = llvm::core::LLVMBuildAnd(
                                emitter.builder,
                                vcc_value,
                                elem_shifted,
                                empty_name.as_ptr(),
                            );

                            let cond = llvm::core::LLVMBuildICmp(
                                emitter.builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                elem_masked,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    0,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.fma.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64, ty_f64, ty_f64];
                            let fma_result = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 3, 0),
                                intrinsic,
                                [s0_value, s1_value, s2_value].as_mut_ptr(),
                                3,
                                empty_name.as_ptr(),
                            );

                            let muled = llvm::core::LLVMBuildFMul(
                                builder,
                                fma_result,
                                llvm::core::LLVMConstReal(
                                    ty_f64,
                                    f64::from_bits(0x43F0000000000000),
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cond,
                                fma_result,
                                muled,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_DIV_FIXUP_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
                            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                            let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src2, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);
                            let s2_value =
                                emitter.emit_abs_neg_f64xn::<N>(s2_value, inst.abs, inst.neg, 2);

                            let s1_value = llvm::core::LLVMBuildBitCast(
                                builder,
                                s1_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let s2_value = llvm::core::LLVMBuildBitCast(
                                builder,
                                s2_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let mut param_tys = vec![ty_f64xn];
                            let intrinsic_name = format!("llvm.fabs.v{}f64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64xn];
                            let abs_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64xn,
                                    param_tys.as_mut_ptr(),
                                    1,
                                    0,
                                ),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            let neg_value =
                                llvm::core::LLVMBuildFNeg(builder, abs_value, empty_name.as_ptr());
                            let sign_out = llvm::core::LLVMBuildXor(
                                builder,
                                s1_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );
                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    0,
                                    0,
                                ); N]
                                    .as_mut_ptr(),
                                N as u32,
                            );
                            let sign_out = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSLT,
                                sign_out,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                sign_out,
                                neg_value,
                                abs_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u64(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_u64(&inst.src2, elem);

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.fabs.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64];
                            let abs_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 1, 0),
                                intrinsic,
                                [s0_value].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            let neg_value =
                                llvm::core::LLVMBuildFNeg(builder, abs_value, empty_name.as_ptr());
                            let sign_out = llvm::core::LLVMBuildXor(
                                builder,
                                s1_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );
                            let sign_out = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSLT,
                                sign_out,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    0,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                sign_out,
                                neg_value,
                                abs_value,
                                empty_name.as_ptr(),
                            );
                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_LDEXP_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value = emitter.emit_i32_to_f64xn::<N>(s1_value);

                            let s1_value = emitter.emit_exp2_f64xn::<N>(s1_value);
                            let d_value = emitter.emit_fmul(s0_value, s1_value);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let mut param_tys = vec![ty_f64, ty_i32];
                            let intrinsic_name = b"llvm.ldexp.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64, ty_i32];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 2, 0),
                                intrinsic,
                                [s0_value, s1_value].as_mut_ptr(),
                                2,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_CMP_CLASS_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let context = emitter.context;

                        const N: usize = SIMD_WIDTH;

                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        let mut agg_value = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                            N as u32,
                        );

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let mut cmp_value = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i1, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );

                            for cls in 0..10 {
                                let mut param_tys = vec![ty_f64xn];
                                let intrinsic_name = format!("llvm.is.fpclass.v{}f64\0", N);
                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );
                                let mut param_tys = vec![ty_f64xn, ty_i32];
                                let class_value = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_i1xn,
                                        param_tys.as_mut_ptr(),
                                        2,
                                        0,
                                    ),
                                    intrinsic,
                                    [s0_value, llvm::core::LLVMConstInt(ty_i32, 1 << cls, 0)]
                                        .as_mut_ptr(),
                                    2,
                                    empty_name.as_ptr(),
                                );

                                let class_value = llvm::core::LLVMBuildAnd(
                                    builder,
                                    class_value,
                                    llvm::core::LLVMBuildICmp(
                                        builder,
                                        llvm::LLVMIntPredicate::LLVMIntNE,
                                        llvm::core::LLVMBuildAnd(
                                            builder,
                                            s1_value,
                                            llvm::core::LLVMConstInt(ty_i32, 1 << cls, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                                        empty_name.as_ptr(),
                                    ),
                                    empty_name.as_ptr(),
                                );

                                cmp_value = llvm::core::LLVMBuildOr(
                                    builder,
                                    cmp_value,
                                    class_value,
                                    empty_name.as_ptr(),
                                );
                            }

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );

                            let bit_flags = emitter.emit_bit_mask_u32xn::<N>(i);

                            let flag_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cmp_value,
                                bit_flags,
                                zero_vec,
                                empty_name.as_ptr(),
                            );

                            agg_value = llvm::core::LLVMBuildOr(
                                builder,
                                agg_value,
                                flag_value,
                                empty_name.as_ptr(),
                            );
                        }

                        let mut param_tys = vec![ty_i32xn];
                        let intrinsic_name = format!("llvm.vector.reduce.or.v{}i32\0", N);
                        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                            intrinsic_name.as_ptr() as *const _,
                            intrinsic_name.len() as usize,
                        );
                        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                            emitter.module,
                            intrinsic_id,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as usize,
                        );
                        let d_value = llvm::core::LLVMBuildCall2(
                            builder,
                            llvm::core::LLVMFunctionType(ty_i32, param_tys.as_mut_ptr(), 1, 0),
                            intrinsic,
                            [agg_value].as_mut_ptr(),
                            1,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let mut param_tys = vec![ty_f64];
                                let intrinsic_name = b"llvm.is.fpclass.f64\0";
                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );
                                let mut param_tys = vec![ty_f64, ty_i32];
                                let class_value = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        llvm::core::LLVMInt1TypeInContext(context),
                                        param_tys.as_mut_ptr(),
                                        2,
                                        0,
                                    ),
                                    intrinsic,
                                    [s0_value, s1_value].as_mut_ptr(),
                                    2,
                                    empty_name.as_ptr(),
                                );

                                (bb, class_value)
                            });
                    }
                }
                I::V_XAD_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildXor(
                                    builder,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                ),
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);

                            let xor_value = llvm::core::LLVMBuildXor(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                xor_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_XOR3_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);

                            let d_value = llvm::core::LLVMBuildXor(
                                builder,
                                llvm::core::LLVMBuildXor(
                                    builder,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                ),
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);

                            let xor_value = llvm::core::LLVMBuildXor(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildXor(
                                builder,
                                xor_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_ADD3_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);

                            let add_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                add_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);

                            let add_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                add_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MUL_LO_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let d_value = llvm::core::LLVMBuildMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let d_value = llvm::core::LLVMBuildMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_ADD_NC_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_AND_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_TRIG_PREOP_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let context = emitter.context;
                        let function = emitter.function;

                        const N: usize = SIMD_WIDTH;

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
                        let ty_i1201 = llvm::core::LLVMIntTypeInContext(context, 1201);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        let two_over_pi_fraction_value =
                            llvm::core::LLVMConstIntOfArbitraryPrecision(
                                ty_i1201,
                                19,
                                [
                                    0xBA10AC06608DF8F6,
                                    0x25D4D7F6BF623F1A,
                                    0xE2F67A0E73EF14A5,
                                    0xD45AEA4F758FD7CB,
                                    0x136E9E8C7ECD3CBF,
                                    0xDA3EDA6CFD9E4F96,
                                    0x301FDE5E2316B414,
                                    0x50763FF12FFFBC0B,
                                    0x73E93908BF177BF2,
                                    0xFC827323AC7306A6,
                                    0x8909D338E04D68BE,
                                    0x4E7DD1046BEA5D76,
                                    0x2439FC3BD6396253,
                                    0xA5C00C925DD413A3,
                                    0x8AC36E48DC74849B,
                                    0x2083FCA2C757BD77,
                                    0xBB81B6C52B327887,
                                    0x2A53F84EAFA3EA69,
                                    0x000145F306DC9C88,
                                ]
                                .as_mut_ptr(),
                            );

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);

                            let bb_loop_entry = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                function,
                                empty_name.as_ptr(),
                            );
                            llvm::core::LLVMBuildBr(builder, bb_loop_entry);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

                            let index =
                                llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
                            let d_value =
                                llvm::core::LLVMBuildPhi(builder, ty_f64xn, empty_name.as_ptr());
                            let index_i32 = llvm::core::LLVMBuildTrunc(
                                builder,
                                index,
                                ty_i32,
                                empty_name.as_ptr(),
                            );

                            let exec = llvm::core::LLVMBuildExtractElement(
                                builder,
                                mask,
                                index_i32,
                                empty_name.as_ptr(),
                            );

                            let bb_loop_skip_body = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                function,
                                empty_name.as_ptr(),
                            );

                            let bb_loop_body = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                function,
                                empty_name.as_ptr(),
                            );

                            let bb_loop_cond = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                function,
                                empty_name.as_ptr(),
                            );

                            let bb_loop_exit = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                function,
                                empty_name.as_ptr(),
                            );

                            llvm::core::LLVMBuildCondBr(
                                builder,
                                exec,
                                bb_loop_body,
                                bb_loop_skip_body,
                            );

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_skip_body);

                            let next_index1 = llvm::core::LLVMBuildAdd(
                                builder,
                                index,
                                llvm::core::LLVMConstInt(ty_i64, 1, 0),
                                empty_name.as_ptr(),
                            );

                            llvm::core::LLVMBuildBr(builder, bb_loop_cond);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

                            let update_d_value = {
                                let s0_value = llvm::core::LLVMBuildExtractElement(
                                    builder,
                                    s0_value,
                                    index_i32,
                                    empty_name.as_ptr(),
                                );
                                let s1_value = llvm::core::LLVMBuildExtractElement(
                                    builder,
                                    s1_value,
                                    index_i32,
                                    empty_name.as_ptr(),
                                );

                                let s0_exp_value = emitter.emit_exp_f64(s0_value);

                                let s1_value = llvm::core::LLVMBuildAnd(
                                    builder,
                                    s1_value,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        0x1F,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );

                                let shift = llvm::core::LLVMBuildMul(
                                    builder,
                                    s1_value,
                                    llvm::core::LLVMConstInt(ty_i32, 53, 0),
                                    empty_name.as_ptr(),
                                );

                                let cmp = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntSGT,
                                    s0_exp_value,
                                    llvm::core::LLVMConstInt(ty_i32, 1077, 0),
                                    empty_name.as_ptr(),
                                );

                                let sub = llvm::core::LLVMBuildSub(
                                    builder,
                                    s0_exp_value,
                                    llvm::core::LLVMConstInt(ty_i32, 1077, 0),
                                    empty_name.as_ptr(),
                                );

                                let shift = llvm::core::LLVMBuildSelect(
                                    builder,
                                    cmp,
                                    llvm::core::LLVMBuildAdd(
                                        builder,
                                        shift,
                                        sub,
                                        empty_name.as_ptr(),
                                    ),
                                    shift,
                                    empty_name.as_ptr(),
                                );

                                let bitpos = llvm::core::LLVMBuildSub(
                                    builder,
                                    llvm::core::LLVMConstInt(ty_i32, 1201 - 53, 0),
                                    shift,
                                    empty_name.as_ptr(),
                                );

                                let bitpos = llvm::core::LLVMBuildZExt(
                                    builder,
                                    bitpos,
                                    ty_i1201,
                                    empty_name.as_ptr(),
                                );

                                let shifted_fraction = llvm::core::LLVMBuildLShr(
                                    builder,
                                    two_over_pi_fraction_value,
                                    bitpos,
                                    empty_name.as_ptr(),
                                );

                                let trunc = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    shifted_fraction,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );

                                let result = llvm::core::LLVMBuildAnd(
                                    builder,
                                    trunc,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        (1u64 << 53) - 1,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );

                                let result = llvm::core::LLVMBuildUIToFP(
                                    builder,
                                    result,
                                    ty_f64,
                                    empty_name.as_ptr(),
                                );

                                let scale = llvm::core::LLVMBuildSub(
                                    builder,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        -53i64 as u64,
                                        0,
                                    ),
                                    shift,
                                    empty_name.as_ptr(),
                                );

                                let cmp = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntSGT,
                                    s0_exp_value,
                                    llvm::core::LLVMConstInt(ty_i32, 1968, 0),
                                    empty_name.as_ptr(),
                                );

                                let scale = llvm::core::LLVMBuildSelect(
                                    builder,
                                    cmp,
                                    llvm::core::LLVMBuildAdd(
                                        builder,
                                        scale,
                                        llvm::core::LLVMConstInt(ty_i32, 128, 0),
                                        empty_name.as_ptr(),
                                    ),
                                    scale,
                                    empty_name.as_ptr(),
                                );

                                let mut param_tys = vec![ty_f64];
                                let intrinsic_name = b"llvm.exp2.f64\0";
                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );
                                let mut param_tys = vec![ty_f64];
                                let exp2_scale = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_f64,
                                        param_tys.as_mut_ptr(),
                                        1,
                                        0,
                                    ),
                                    intrinsic,
                                    [llvm::core::LLVMBuildSIToFP(
                                        builder,
                                        scale,
                                        ty_f64,
                                        empty_name.as_ptr(),
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let result = llvm::core::LLVMBuildFMul(
                                    builder,
                                    result,
                                    exp2_scale,
                                    empty_name.as_ptr(),
                                );

                                llvm::core::LLVMBuildInsertElement(
                                    builder,
                                    d_value,
                                    result,
                                    index_i32,
                                    empty_name.as_ptr(),
                                )
                            };

                            let next_index2 = llvm::core::LLVMBuildAdd(
                                builder,
                                index,
                                llvm::core::LLVMConstInt(ty_i64, 1, 0),
                                empty_name.as_ptr(),
                            );

                            llvm::core::LLVMBuildBr(builder, bb_loop_cond);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

                            let next_index =
                                llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
                            let mut incoming_value = vec![next_index1, next_index2];
                            let mut incoming_blocks = vec![bb_loop_skip_body, bb_loop_body];
                            llvm::core::LLVMAddIncoming(
                                next_index,
                                incoming_value.as_mut_ptr(),
                                incoming_blocks.as_mut_ptr(),
                                incoming_value.len() as u32,
                            );

                            let next_d_value =
                                llvm::core::LLVMBuildPhi(builder, ty_f64xn, empty_name.as_ptr());
                            let mut incoming_value = vec![d_value, update_d_value];
                            let mut incoming_blocks = vec![bb_loop_skip_body, bb_loop_body];
                            llvm::core::LLVMAddIncoming(
                                next_d_value,
                                incoming_value.as_mut_ptr(),
                                incoming_blocks.as_mut_ptr(),
                                incoming_value.len() as u32,
                            );

                            let cmp = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                next_index,
                                llvm::core::LLVMConstInt(ty_i64, N as u64, 0),
                                empty_name.as_ptr(),
                            );
                            llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

                            let mut incoming_value =
                                vec![llvm::core::LLVMConstInt(ty_i64, 0, 0), next_index];
                            let mut incoming_blocks = vec![bb, bb_loop_cond];
                            llvm::core::LLVMAddIncoming(
                                index,
                                incoming_value.as_mut_ptr(),
                                incoming_blocks.as_mut_ptr(),
                                incoming_value.len() as u32,
                            );

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstReal(ty_f64, 0.0); N].as_mut_ptr(),
                                N as u32,
                            );

                            let mut incoming_value = vec![zero_vec, next_d_value];
                            let mut incoming_blocks = vec![bb, bb_loop_cond];
                            llvm::core::LLVMAddIncoming(
                                d_value,
                                incoming_value.as_mut_ptr(),
                                incoming_blocks.as_mut_ptr(),
                                incoming_value.len() as u32,
                            );

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_exit);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);

                            bb = bb_loop_exit
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                            let ty_i1201 = llvm::core::LLVMIntTypeInContext(context, 1201);

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                            let s0_value = emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);

                            let s0_exp_value = emitter.emit_exp_f64(s0_value);

                            let s1_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s1_value,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    0x1F,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );

                            let two_over_pi_fraction = llvm::core::LLVMGetNamedGlobal(
                                emitter.module,
                                b"v_trig_preop_f64.TWO_OVER_PI_FRACTION\0".as_ptr() as *const _,
                            );

                            let two_over_pi_fraction = if two_over_pi_fraction.is_null() {
                                let two_over_pi_fraction = llvm::core::LLVMAddGlobal(
                                    emitter.module,
                                    ty_i1201,
                                    b"v_trig_preop_f64.TWO_OVER_PI_FRACTION\0".as_ptr() as *const _,
                                );
                                llvm::core::LLVMSetInitializer(
                                    two_over_pi_fraction,
                                    llvm::core::LLVMConstIntOfArbitraryPrecision(
                                        ty_i1201,
                                        19,
                                        [
                                            0xBA10AC06608DF8F6,
                                            0x25D4D7F6BF623F1A,
                                            0xE2F67A0E73EF14A5,
                                            0xD45AEA4F758FD7CB,
                                            0x136E9E8C7ECD3CBF,
                                            0xDA3EDA6CFD9E4F96,
                                            0x301FDE5E2316B414,
                                            0x50763FF12FFFBC0B,
                                            0x73E93908BF177BF2,
                                            0xFC827323AC7306A6,
                                            0x8909D338E04D68BE,
                                            0x4E7DD1046BEA5D76,
                                            0x2439FC3BD6396253,
                                            0xA5C00C925DD413A3,
                                            0x8AC36E48DC74849B,
                                            0x2083FCA2C757BD77,
                                            0xBB81B6C52B327887,
                                            0x2A53F84EAFA3EA69,
                                            0x000145F306DC9C88,
                                        ]
                                        .as_mut_ptr(),
                                    ),
                                );
                                two_over_pi_fraction
                            } else {
                                two_over_pi_fraction
                            };

                            let two_over_pi_fraction_value = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i1201,
                                two_over_pi_fraction,
                                empty_name.as_ptr(),
                            );

                            let shift = llvm::core::LLVMBuildMul(
                                builder,
                                s1_value,
                                llvm::core::LLVMConstInt(ty_i32, 53, 0),
                                empty_name.as_ptr(),
                            );

                            let cmp = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSGT,
                                s0_exp_value,
                                llvm::core::LLVMConstInt(ty_i32, 1077, 0),
                                empty_name.as_ptr(),
                            );

                            let sub = llvm::core::LLVMBuildSub(
                                builder,
                                s0_exp_value,
                                llvm::core::LLVMConstInt(ty_i32, 1077, 0),
                                empty_name.as_ptr(),
                            );

                            let shift = llvm::core::LLVMBuildSelect(
                                builder,
                                cmp,
                                llvm::core::LLVMBuildAdd(builder, shift, sub, empty_name.as_ptr()),
                                shift,
                                empty_name.as_ptr(),
                            );

                            let bitpos = llvm::core::LLVMBuildSub(
                                builder,
                                llvm::core::LLVMConstInt(ty_i32, 1201 - 53, 0),
                                shift,
                                empty_name.as_ptr(),
                            );

                            let bitpos = llvm::core::LLVMBuildZExt(
                                builder,
                                bitpos,
                                ty_i1201,
                                empty_name.as_ptr(),
                            );

                            let shifted_fraction = llvm::core::LLVMBuildLShr(
                                builder,
                                two_over_pi_fraction_value,
                                bitpos,
                                empty_name.as_ptr(),
                            );

                            let trunc = llvm::core::LLVMBuildTrunc(
                                builder,
                                shifted_fraction,
                                llvm::core::LLVMInt64TypeInContext(context),
                                empty_name.as_ptr(),
                            );

                            let result = llvm::core::LLVMBuildAnd(
                                builder,
                                trunc,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    (1u64 << 53) - 1,
                                    0,
                                ),
                                empty_name.as_ptr(),
                            );

                            let result = llvm::core::LLVMBuildUIToFP(
                                builder,
                                result,
                                ty_f64,
                                empty_name.as_ptr(),
                            );

                            let scale = llvm::core::LLVMBuildSub(
                                builder,
                                llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    -53i64 as u64,
                                    0,
                                ),
                                shift,
                                empty_name.as_ptr(),
                            );

                            let cmp = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSGT,
                                s0_exp_value,
                                llvm::core::LLVMConstInt(ty_i32, 1968, 0),
                                empty_name.as_ptr(),
                            );

                            let scale = llvm::core::LLVMBuildSelect(
                                builder,
                                cmp,
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    scale,
                                    llvm::core::LLVMConstInt(ty_i32, 128, 0),
                                    empty_name.as_ptr(),
                                ),
                                scale,
                                empty_name.as_ptr(),
                            );

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.exp2.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64];
                            let exp2_scale = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 1, 0),
                                intrinsic,
                                [llvm::core::LLVMBuildSIToFP(
                                    builder,
                                    scale,
                                    ty_f64,
                                    empty_name.as_ptr(),
                                )]
                                .as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildFMul(
                                builder,
                                result,
                                exp2_scale,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_MAX_NUM_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);
                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);

                            let mut param_tys = vec![ty_f64xn];
                            let intrinsic_name = format!("llvm.maxnum.v{}f64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64xn, ty_f64xn];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_f64xn,
                                    param_tys.as_mut_ptr(),
                                    2,
                                    0,
                                ),
                                intrinsic,
                                [s0_value, s1_value].as_mut_ptr(),
                                2,
                                empty_name.as_ptr(),
                            );
                            let d_value =
                                emitter.emit_omod_clamp_f64xn::<N>(inst.omod, inst.cm, d_value, 0);

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    } else {
                        bb = self.emit_vop(bb, |emitter, bb, elem| {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                            let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                            let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                            let s0_value = emitter.emit_abs_neg(inst.abs, inst.neg, s0_value, 0);
                            let s1_value = emitter.emit_abs_neg(inst.abs, inst.neg, s1_value, 1);

                            let mut param_tys = vec![ty_f64];
                            let intrinsic_name = b"llvm.maxnum.f64\0";
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );
                            let mut param_tys = vec![ty_f64, ty_f64];
                            let d_value = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(ty_f64, param_tys.as_mut_ptr(), 2, 0),
                                intrinsic,
                                [s0_value, s1_value].as_mut_ptr(),
                                2,
                                empty_name.as_ptr(),
                            );
                            let d_value = emitter.emit_omod_clamp(inst.omod, inst.cm, d_value, 0);

                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                            bb
                        });
                    }
                }
                I::V_WRITELANE_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.src0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.src1);

                    let s1_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, 0x1F, 0),
                        empty_name.as_ptr(),
                    );

                    let s1_value =
                        llvm::core::LLVMBuildZExt(builder, s1_value, ty_i64, empty_name.as_ptr());

                    emitter.emit_store_vgpr_u32(inst.vdst as u32, s1_value, s0_value);
                }
                I::V_READLANE_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.src1);

                    let s1_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, 0x1F, 0),
                        empty_name.as_ptr(),
                    );

                    let s1_value =
                        llvm::core::LLVMBuildZExt(builder, s1_value, ty_i64, empty_name.as_ptr());

                    let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, s1_value);

                    emitter.emit_store_sgpr_u32(inst.vdst as u32, s0_value);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3SD(inst) => match inst.op {
                I::V_MAD_CO_U64_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let mut agg_value = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                            N as u32,
                        );

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_u64xn::<N>(&inst.src2, i, mask);

                            let s0_value = llvm::core::LLVMBuildZExt(
                                builder,
                                s0_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );
                            let s1_value = llvm::core::LLVMBuildZExt(
                                builder,
                                s1_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let muled = llvm::core::LLVMBuildMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );
                            let mut param_tys = vec![ty_i64xn];

                            let intrinsic_name = format!("llvm.uadd.with.overflow.v{}i64\0", N);
                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );

                            let mut return_tys = vec![ty_i64xn, ty_i1xn];
                            let mut param_tys = vec![ty_i64xn, ty_i64xn];

                            let add_overflow = llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    llvm::core::LLVMStructTypeInContext(
                                        context,
                                        return_tys.as_mut_ptr(),
                                        return_tys.len() as u32,
                                        0,
                                    ),
                                    param_tys.as_mut_ptr(),
                                    2,
                                    0,
                                ),
                                intrinsic,
                                [muled, s2_value].as_mut_ptr(),
                                2,
                                empty_name.as_ptr(),
                            );
                            let added = llvm::core::LLVMBuildExtractValue(
                                builder,
                                add_overflow,
                                0,
                                empty_name.as_ptr(),
                            );
                            let cmp_value = llvm::core::LLVMBuildExtractValue(
                                builder,
                                add_overflow,
                                1,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u64xn::<N>(inst.vdst as u32, i, added, mask);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );

                            let bit_flags = emitter.emit_bit_mask_u32xn::<N>(i);

                            let flag_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cmp_value,
                                bit_flags,
                                zero_vec,
                                empty_name.as_ptr(),
                            );

                            agg_value = llvm::core::LLVMBuildOr(
                                builder,
                                agg_value,
                                flag_value,
                                empty_name.as_ptr(),
                            );
                        }

                        let mut param_tys = vec![ty_i32xn];
                        let intrinsic_name = format!("llvm.vector.reduce.or.v{}i32\0", N);
                        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                            intrinsic_name.as_ptr() as *const _,
                            intrinsic_name.len() as usize,
                        );
                        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                            emitter.module,
                            intrinsic_id,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as usize,
                        );
                        let d_value = llvm::core::LLVMBuildCall2(
                            builder,
                            llvm::core::LLVMFunctionType(ty_i32, param_tys.as_mut_ptr(), 1, 0),
                            intrinsic,
                            [agg_value].as_mut_ptr(),
                            1,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let s2_value =
                                    emitter.emit_vector_source_operand_u64(&inst.src2, elem);

                                let s0_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    s0_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                let s1_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    s1_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                let muled = llvm::core::LLVMBuildMul(
                                    builder,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );
                                let mut param_tys = vec![ty_i64];

                                let intrinsic_name = b"llvm.uadd.with.overflow.i64\0";
                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );

                                let mut return_tys =
                                    vec![ty_i64, llvm::core::LLVMInt1TypeInContext(context)];
                                let mut param_tys = vec![ty_i64, ty_i64];

                                let add_overflow = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        llvm::core::LLVMStructTypeInContext(
                                            context,
                                            return_tys.as_mut_ptr(),
                                            return_tys.len() as u32,
                                            0,
                                        ),
                                        param_tys.as_mut_ptr(),
                                        2,
                                        0,
                                    ),
                                    intrinsic,
                                    [muled, s2_value].as_mut_ptr(),
                                    2,
                                    empty_name.as_ptr(),
                                );
                                let d0_value = llvm::core::LLVMBuildExtractValue(
                                    builder,
                                    add_overflow,
                                    0,
                                    empty_name.as_ptr(),
                                );
                                let d1_value = llvm::core::LLVMBuildExtractValue(
                                    builder,
                                    add_overflow,
                                    1,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u64(inst.vdst as u32, elem, d0_value);

                                (bb, d1_value)
                            });
                    }
                }
                I::V_DIV_SCALE_F64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                            let s2_value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src2, i, mask);

                            let s0_value =
                                emitter.emit_abs_neg_f64xn::<N>(s0_value, 0, inst.neg, 0);

                            let s1_value =
                                emitter.emit_abs_neg_f64xn::<N>(s1_value, 0, inst.neg, 1);

                            let s2_value =
                                emitter.emit_abs_neg_f64xn::<N>(s2_value, 0, inst.neg, 2);

                            let muled = llvm::core::LLVMBuildFMul(
                                builder,
                                s0_value,
                                s2_value,
                                empty_name.as_ptr(),
                            );
                            let d_value = llvm::core::LLVMBuildFDiv(
                                builder,
                                muled,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }

                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        emitter.emit_store_sgpr_u32(
                            inst.sdst as u32,
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        );
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                                let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                                let s1_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                                let s2_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src2, elem);

                                let muled = llvm::core::LLVMBuildFMul(
                                    builder,
                                    s0_value,
                                    s2_value,
                                    empty_name.as_ptr(),
                                );
                                let d0_value = llvm::core::LLVMBuildFDiv(
                                    builder,
                                    muled,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );
                                let d1_value = llvm::core::LLVMConstInt(ty_i1, 0, 0);

                                emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d0_value);

                                (bb, d1_value)
                            });
                    }
                }
                I::V_ADD_CO_CI_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);
                        let vcc_value = emitter.emit_scalar_source_operand_u32(&inst.src2);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                            let vcc_value = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildZExt(
                                builder,
                                s0_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );
                            let s1_value = llvm::core::LLVMBuildZExt(
                                builder,
                                s1_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );
                            let vcc_value = llvm::core::LLVMBuildZExt(
                                builder,
                                vcc_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let added = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                ),
                                vcc_value,
                                empty_name.as_ptr(),
                            );

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                added,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, 0x1_0000_0000, 0); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);

                            let d_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                added,
                                ty_i32xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let elem_i32 = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    elem,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                );

                                let s2_value = emitter.emit_scalar_source_operand_u32(&inst.src2);

                                let index_mask = llvm::core::LLVMBuildShl(
                                    builder,
                                    llvm::core::LLVMConstInt(ty_i32, 1, 0),
                                    elem_i32,
                                    empty_name.as_ptr(),
                                );
                                let masked = llvm::core::LLVMBuildAnd(
                                    builder,
                                    s2_value,
                                    index_mask,
                                    empty_name.as_ptr(),
                                );
                                let vcc_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntNE,
                                    masked,
                                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                                    empty_name.as_ptr(),
                                );

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let s0_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    s0_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );
                                let s1_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    s1_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );
                                let vcc_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vcc_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );

                                let added = llvm::core::LLVMBuildAdd(
                                    builder,
                                    llvm::core::LLVMBuildAdd(
                                        builder,
                                        s0_value,
                                        s1_value,
                                        empty_name.as_ptr(),
                                    ),
                                    vcc_value,
                                    empty_name.as_ptr(),
                                );

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntUGE,
                                    added,
                                    llvm::core::LLVMConstInt(ty_i64, 0x1_0000_0000, 0),
                                    empty_name.as_ptr(),
                                );

                                let d0_value = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    added,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d0_value);

                                (bb, cmp_value)
                            });
                    }
                }
                I::V_ADD_CO_U32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const N: usize = SIMD_WIDTH;

                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                        let mut cmp_values = Vec::new();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                            let s1_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);

                            let s0_value = llvm::core::LLVMBuildZExt(
                                builder,
                                s0_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );
                            let s1_value = llvm::core::LLVMBuildZExt(
                                builder,
                                s1_value,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let added = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            let cmp_value = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                added,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, 0x1_0000_0000, 0); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            cmp_values.push(cmp_value);

                            let d_value = llvm::core::LLVMBuildTrunc(
                                builder,
                                added,
                                ty_i32xn,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }

                        let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            cmp_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                    } else {
                        bb =
                            self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let s0_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    s0_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );
                                let s1_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    s1_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );

                                let added = llvm::core::LLVMBuildAdd(
                                    builder,
                                    s0_value,
                                    s1_value,
                                    empty_name.as_ptr(),
                                );

                                let cmp_value = llvm::core::LLVMBuildICmp(
                                    builder,
                                    llvm::LLVMIntPredicate::LLVMIntUGE,
                                    added,
                                    llvm::core::LLVMConstInt(ty_i64, 0x1_0000_0000, 0),
                                    empty_name.as_ptr(),
                                );

                                let d0_value = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    added,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d0_value);

                                (bb, cmp_value)
                            });
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOPD(inst) => {
                let emitter = self;
                let mut opx_results = Vec::new();
                let mut opy_results = Vec::new();
                let exec_value = emitter.emit_load_sgpr_u32(126);

                const N: usize = SIMD_WIDTH;

                match inst.opx {
                    I::V_DUAL_CNDMASK_B32 => {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let vcc_value = emitter.emit_load_sgpr_u32(106);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                            let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0x, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1x as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cond,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            opx_results.push(d_value);
                        }
                    }
                    I::V_DUAL_MOV_B32 => {
                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0x, i, mask);

                            let d_value = s0_value;

                            opx_results.push(d_value);
                        }
                    }
                    _ => {
                        panic!("Unsupported instruction: {:?}", inst);
                    }
                }
                match inst.opy {
                    I::V_DUAL_CNDMASK_B32 => {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let vcc_value = emitter.emit_load_sgpr_u32(106);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                            let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0y, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildSelect(
                                builder,
                                cond,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            opy_results.push(d_value);
                        }
                    }
                    I::V_DUAL_MOV_B32 => {
                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0y, i, mask);

                            let d_value = s0_value;

                            opy_results.push(d_value);
                        }
                    }
                    I::V_DUAL_ADD_NC_U32 => {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0y, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

                            let d_value = llvm::core::LLVMBuildAdd(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            );

                            opy_results.push(d_value);
                        }
                    }
                    I::V_DUAL_LSHLREV_B32 => {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                            let s0_value =
                                emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0y, i, mask);

                            let s1_value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

                            let s0_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 0x1F, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let d_value = llvm::core::LLVMBuildShl(
                                builder,
                                s1_value,
                                s0_value,
                                empty_name.as_ptr(),
                            );

                            opy_results.push(d_value);
                        }
                    }
                    _ => {
                        panic!("Unsupported instruction: {:?}", inst);
                    }
                }

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let vdstx = inst.vdstx as u32;
                    emitter.emit_store_vgpr_u32xn::<N>(vdstx, i, opx_results[i as usize / N], mask);

                    let vdsty = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as u32;
                    emitter.emit_store_vgpr_u32xn::<N>(vdsty, i, opy_results[i as usize / N], mask);
                }
            }
            InstFormat::SMEM(inst) => match inst.op {
                I::S_LOAD_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32 * 2);

                    {
                        let offset = llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                        let addr =
                            llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());

                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            addr,
                            ty_p0,
                            empty_name.as_ptr(),
                        );

                        let data =
                            llvm::core::LLVMBuildLoad2(builder, ty_i32, ptr, empty_name.as_ptr());

                        emitter.emit_store_sgpr_u32(inst.sdata as u32, data);
                    }
                }
                I::S_LOAD_B64 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32 * 2);

                    for i in 0..2 {
                        let offset =
                            llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                        let addr =
                            llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            addr,
                            ty_p0,
                            empty_name.as_ptr(),
                        );
                        let data =
                            llvm::core::LLVMBuildLoad2(builder, ty_i32, ptr, empty_name.as_ptr());

                        emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                    }
                }
                I::S_LOAD_B96 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32 * 2);

                    for i in 0..3 {
                        let offset =
                            llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                        let addr =
                            llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            addr,
                            ty_p0,
                            empty_name.as_ptr(),
                        );
                        let data =
                            llvm::core::LLVMBuildLoad2(builder, ty_i32, ptr, empty_name.as_ptr());

                        emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                    }
                }
                I::S_LOAD_B128 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32 * 2);

                    for i in 0..4 {
                        let offset =
                            llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                        let addr =
                            llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            addr,
                            ty_p0,
                            empty_name.as_ptr(),
                        );
                        let data =
                            llvm::core::LLVMBuildLoad2(builder, ty_i32, ptr, empty_name.as_ptr());

                        emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOP1(inst) => match inst.op {
                I::S_BARRIER_SIGNAL => {}
                I::S_AND_SAVEEXEC_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                    let s1_value = emitter.emit_load_sgpr_u32(126);

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, s1_value);

                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(126, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_OR_SAVEEXEC_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                    let s1_value = emitter.emit_load_sgpr_u32(126);

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, s1_value);

                    let d_value =
                        llvm::core::LLVMBuildOr(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(126, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_AND_NOT1_SAVEEXEC_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                    let s1_value = emitter.emit_load_sgpr_u32(126);

                    let not1_value = llvm::core::LLVMBuildXor(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, -1 as i64 as u64, 0),
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, s1_value);

                    let d_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s0_value,
                        not1_value,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(126, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_MOV_B32 => {
                    let emitter = self;
                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                    let d_value = s0_value;

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                }
                I::S_MOV_B64 => {
                    let emitter = self;
                    let s0_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc0);

                    let d_value = s0_value;

                    emitter.emit_store_sgpr_u64(inst.sdst as u32, d_value);
                }
                I::S_CTZ_I32_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                    let mut param_tys = vec![ty_i32];
                    let intrinsic_name = "llvm.cttz.i32\0";
                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                        intrinsic_name.as_ptr() as *const _,
                        intrinsic_name.len() as usize,
                    );
                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                        emitter.module,
                        intrinsic_id,
                        param_tys.as_mut_ptr(),
                        param_tys.len() as usize,
                    );
                    let mut param_tys = vec![ty_i32, ty_i1];
                    let d_value = llvm::core::LLVMBuildCall2(
                        builder,
                        llvm::core::LLVMFunctionType(
                            ty_i32,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as u32,
                            0,
                        ),
                        intrinsic,
                        [s0_value, llvm::core::LLVMConstInt(ty_i1, 0, 0)].as_mut_ptr(),
                        2,
                        empty_name.as_ptr(),
                    );

                    let d_value = llvm::core::LLVMBuildSelect(
                        builder,
                        llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntEQ,
                            s0_value,
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        ),
                        llvm::core::LLVMConstInt(ty_i32, -1i64 as u64, 0),
                        d_value,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOP2(inst) => match inst.op {
                I::S_AND_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_OR_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let d_value =
                        llvm::core::LLVMBuildOr(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_XOR_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let d_value =
                        llvm::core::LLVMBuildXor(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_AND_NOT1_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let not1_value = llvm::core::LLVMBuildXor(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, -1 as i64 as u64, 0),
                        empty_name.as_ptr(),
                    );

                    let d_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s0_value,
                        not1_value,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_OR_NOT1_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let not1_value = llvm::core::LLVMBuildXor(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, -1 as i64 as u64, 0),
                        empty_name.as_ptr(),
                    );

                    let d_value =
                        llvm::core::LLVMBuildOr(builder, s0_value, not1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_LSHR_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let s1_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, 31, 0),
                        empty_name.as_ptr(),
                    );
                    let d_value =
                        llvm::core::LLVMBuildLShr(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_LSHL_B32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let s1_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, 31, 0),
                        empty_name.as_ptr(),
                    );
                    let d_value =
                        llvm::core::LLVMBuildShl(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CSELECT_B32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let scc_value = emitter.emit_load_scc_u8();

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        scc_value,
                        llvm::core::LLVMConstInt(ty_i8, 0, 0),
                        empty_name.as_ptr(),
                    );
                    let d_value = llvm::core::LLVMBuildSelect(
                        builder,
                        cmp,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                }
                I::S_ADD_NC_U64 => {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc1);

                    let d_value =
                        llvm::core::LLVMBuildAdd(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u64(inst.sdst as u32, d_value);
                }
                I::S_ADD_CO_I32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let mut param_tys = vec![ty_i32];

                    let intrinsic_name = "llvm.sadd.with.overflow.i32\0";
                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                        intrinsic_name.as_ptr() as *const _,
                        intrinsic_name.len() as usize,
                    );
                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                        emitter.module,
                        intrinsic_id,
                        param_tys.as_mut_ptr(),
                        param_tys.len() as usize,
                    );

                    let mut return_tys = vec![ty_i32, ty_i1];
                    let mut param_tys = vec![ty_i32, ty_i32];

                    let add_overflow = llvm::core::LLVMBuildCall2(
                        builder,
                        llvm::core::LLVMFunctionType(
                            llvm::core::LLVMStructTypeInContext(
                                context,
                                return_tys.as_mut_ptr(),
                                return_tys.len() as u32,
                                0,
                            ),
                            param_tys.as_mut_ptr(),
                            2,
                            0,
                        ),
                        intrinsic,
                        [s0_value, s1_value].as_mut_ptr(),
                        2,
                        empty_name.as_ptr(),
                    );
                    let added = llvm::core::LLVMBuildExtractValue(
                        builder,
                        add_overflow,
                        0,
                        empty_name.as_ptr(),
                    );
                    let cmp = llvm::core::LLVMBuildExtractValue(
                        builder,
                        add_overflow,
                        1,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, added);

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_SUB_CO_I32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let mut param_tys = vec![ty_i32];

                    let intrinsic_name = "llvm.ssub.with.overflow.i32\0";
                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                        intrinsic_name.as_ptr() as *const _,
                        intrinsic_name.len() as usize,
                    );
                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                        emitter.module,
                        intrinsic_id,
                        param_tys.as_mut_ptr(),
                        param_tys.len() as usize,
                    );

                    let mut return_tys = vec![ty_i32, ty_i1];
                    let mut param_tys = vec![ty_i32, ty_i32];

                    let add_overflow = llvm::core::LLVMBuildCall2(
                        builder,
                        llvm::core::LLVMFunctionType(
                            llvm::core::LLVMStructTypeInContext(
                                context,
                                return_tys.as_mut_ptr(),
                                return_tys.len() as u32,
                                0,
                            ),
                            param_tys.as_mut_ptr(),
                            2,
                            0,
                        ),
                        intrinsic,
                        [s0_value, s1_value].as_mut_ptr(),
                        2,
                        empty_name.as_ptr(),
                    );
                    let added = llvm::core::LLVMBuildExtractValue(
                        builder,
                        add_overflow,
                        0,
                        empty_name.as_ptr(),
                    );
                    let cmp = llvm::core::LLVMBuildExtractValue(
                        builder,
                        add_overflow,
                        1,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, added);

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_MUL_I32 => {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let d_value =
                        llvm::core::LLVMBuildMul(builder, s0_value, s1_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                }
                I::S_MUL_HI_U32 => {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let s0_value =
                        llvm::core::LLVMBuildZExt(builder, s0_value, ty_i64, empty_name.as_ptr());
                    let s1_value =
                        llvm::core::LLVMBuildZExt(builder, s1_value, ty_i64, empty_name.as_ptr());

                    let d_value =
                        llvm::core::LLVMBuildMul(builder, s0_value, s1_value, empty_name.as_ptr());

                    let d_value = llvm::core::LLVMBuildLShr(
                        builder,
                        d_value,
                        llvm::core::LLVMConstInt(ty_i64, 32, 0),
                        empty_name.as_ptr(),
                    );
                    let d_value =
                        llvm::core::LLVMBuildTrunc(builder, d_value, ty_i32, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                }
                I::S_MAX_U32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntUGE,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let d_value = llvm::core::LLVMBuildSelect(
                        builder,
                        cmp,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOPC(inst) => match inst.op {
                I::S_CMP_LG_U32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CMP_EQ_U32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntEQ,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CMP_LT_U32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntULT,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CMP_GE_U32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntUGE,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CMP_LT_I32 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntSLT,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CMP_LG_U64 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                I::S_CMP_EQ_U64 => {
                    let emitter = self;
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let s0_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc0);
                    let s1_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc1);

                    let cmp = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntEQ,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );

                    let scc_value =
                        llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                    emitter.emit_store_scc_u8(scc_value);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VGLOBAL(inst) => match inst.op {
                I::GLOBAL_WB => {}
                I::GLOBAL_INV => {}
                I::GLOBAL_LOAD_U8 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                        let ty_i8xn = llvm::core::LLVMVectorType(ty_i8, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            let vaddr_value = if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask)
                            };

                            let ioffset = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(
                                    ty_i64,
                                    ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                    0,
                                ); N]
                                    .as_mut_ptr(),
                                N as u32,
                            );
                            let addr = llvm::core::LLVMBuildAdd(
                                builder,
                                vaddr_value,
                                ioffset,
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                addr,
                                ty_p0xn,
                                empty_name.as_ptr(),
                            );

                            {
                                let intrinsic_name =
                                    format!("llvm.masked.gather.v{}i8.v{}p0\0", N, N);
                                let mut param_tys = vec![ty_i8xn, ty_p0xn];

                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );

                                let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i8xn];
                                let data = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_i8xn,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as u32,
                                        0,
                                    ),
                                    intrinsic,
                                    [
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i8xn),
                                    ]
                                    .as_mut_ptr(),
                                    4,
                                    empty_name.as_ptr(),
                                );

                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, data, mask);
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i8,
                                    ptr,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, data);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_LOAD_U16 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            let vaddr_value = if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask)
                            };

                            let ioffset = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(
                                    ty_i64,
                                    ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                    0,
                                ); N]
                                    .as_mut_ptr(),
                                N as u32,
                            );
                            let addr = llvm::core::LLVMBuildAdd(
                                builder,
                                vaddr_value,
                                ioffset,
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                addr,
                                ty_p0xn,
                                empty_name.as_ptr(),
                            );

                            {
                                let intrinsic_name =
                                    format!("llvm.masked.gather.v{}i16.v{}p0\0", N, N);
                                let mut param_tys = vec![ty_i16xn, ty_p0xn];

                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );

                                let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i16xn];
                                let data = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_i16xn,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as u32,
                                        0,
                                    ),
                                    intrinsic,
                                    [
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i16xn),
                                    ]
                                    .as_mut_ptr(),
                                    4,
                                    empty_name.as_ptr(),
                                );

                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, data, mask);
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64,
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i16,
                                    ptr,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, data);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_LOAD_B32 => {
                    if USE_SIMD && USE_MASKED_GATHER {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 1;

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            let vaddr_value = if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask)
                            };

                            let ioffset = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(
                                    ty_i64,
                                    ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                    0,
                                ); N]
                                    .as_mut_ptr(),
                                N as u32,
                            );
                            let addr = llvm::core::LLVMBuildAdd(
                                builder,
                                vaddr_value,
                                ioffset,
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                addr,
                                ty_p0xn,
                                empty_name.as_ptr(),
                            );

                            for j in 0..NUM_WORDS {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let intrinsic_name =
                                    format!("llvm.masked.gather.v{}i32.v{}p0\0", N, N);
                                let mut param_tys = vec![ty_i32xn, ty_p0xn];

                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );

                                let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i32xn];
                                let data = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_i32xn,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as u32,
                                        0,
                                    ),
                                    intrinsic,
                                    [
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i32xn),
                                    ]
                                    .as_mut_ptr(),
                                    4,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32xn::<N>(
                                    inst.vdst as u32 + j as u32,
                                    i,
                                    data,
                                    mask,
                                );
                            }
                        }
                    } else if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 1;

                        for i in (0..32).step_by(N) {
                            let mut addr_values = [std::ptr::null_mut(); N];
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);

                                for j in 0..N {
                                    let elem = llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        j as u64,
                                        0,
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildExtractElement(
                                        builder,
                                        vaddr_value,
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildZExt(
                                        builder,
                                        vaddr_value,
                                        ty_i64,
                                        empty_name.as_ptr(),
                                    );
                                    let addr_value = llvm::core::LLVMBuildAdd(
                                        builder,
                                        saddr_value,
                                        vaddr_value,
                                        empty_name.as_ptr(),
                                    );
                                    addr_values[j] = addr_value;
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                for j in 0..N {
                                    let elem = llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        j as u64,
                                        0,
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildExtractElement(
                                        builder,
                                        vaddr_value,
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                    addr_values[j] = vaddr_value;
                                }
                            }

                            let mut values = [llvm::core::LLVMGetUndef(ty_i32xn); NUM_WORDS];
                            for j in 0..N {
                                let vec_elem = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    j as u64,
                                    0,
                                );

                                let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                    context,
                                    emitter.function,
                                    empty_name.as_ptr(),
                                );

                                let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                    context,
                                    emitter.function,
                                    empty_name.as_ptr(),
                                );

                                let exec = llvm::core::LLVMBuildExtractElement(
                                    builder,
                                    mask,
                                    vec_elem,
                                    empty_name.as_ptr(),
                                );

                                llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                                llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                                let offset = addr_values[j as usize];

                                let mut words = Vec::new();

                                for k in 0..NUM_WORDS {
                                    let addr = llvm::core::LLVMBuildAdd(
                                        builder,
                                        offset,
                                        llvm::core::LLVMConstInt(
                                            llvm::core::LLVMInt64TypeInContext(context),
                                            ((((inst.ioffset << 8) as i32) >> 8) as i64
                                                + (k as i64) * 4)
                                                as u64,
                                            0,
                                        ),
                                        empty_name.as_ptr(),
                                    );
                                    let ptr = llvm::core::LLVMBuildIntToPtr(
                                        builder,
                                        addr,
                                        ty_p0,
                                        empty_name.as_ptr(),
                                    );
                                    let data = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        ptr,
                                        empty_name.as_ptr(),
                                    );

                                    words.push(data);
                                }

                                llvm::core::LLVMBuildBr(builder, bb_cont);
                                llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);

                                for k in 0..NUM_WORDS {
                                    let phi_value = llvm::core::LLVMBuildPhi(
                                        builder,
                                        ty_i32,
                                        empty_name.as_ptr(),
                                    );

                                    llvm::core::LLVMAddIncoming(
                                        phi_value,
                                        [
                                            words[k],
                                            llvm::core::LLVMConstInt(
                                                llvm::core::LLVMInt32TypeInContext(context),
                                                0,
                                                0,
                                            ),
                                        ]
                                        .as_mut_ptr(),
                                        [bb_exec, bb].as_mut_ptr(),
                                        2,
                                    );

                                    values[k] = llvm::core::LLVMBuildInsertElement(
                                        builder,
                                        values[k],
                                        phi_value,
                                        vec_elem,
                                        empty_name.as_ptr(),
                                    );
                                }
                                bb = bb_cont;
                            }

                            for k in 0..NUM_WORDS {
                                emitter.emit_store_vgpr_u32xn::<N>(
                                    inst.vdst as u32 + k as u32,
                                    i as u32,
                                    values[k as usize],
                                    mask,
                                );
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            for j in 0..1 {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64 + j * 4) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(
                                    inst.vdst as u32 + j as u32,
                                    elem,
                                    data,
                                );
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_LOAD_B64 => {
                    if USE_SIMD && USE_MASKED_GATHER {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 2;

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                let vaddr_value = llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                );

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in 0..NUM_WORDS {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.gather.v{}i32.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i32xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i32xn];
                                    let data = llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_i32xn,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [
                                            ptr,
                                            llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                            mask,
                                            llvm::core::LLVMGetPoison(ty_i32xn),
                                        ]
                                        .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );

                                    emitter.emit_store_vgpr_u32xn::<N>(
                                        inst.vdst as u32 + j as u32,
                                        i,
                                        data,
                                        mask,
                                    );
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.gather.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i64xn];
                                    let data = llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_i64xn,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [
                                            ptr,
                                            llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                            mask,
                                            llvm::core::LLVMGetPoison(ty_i64xn),
                                        ]
                                        .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );

                                    emitter.emit_store_vgpr_u64xn::<N>(
                                        inst.vdst as u32 + j as u32,
                                        i,
                                        data,
                                        mask,
                                    );
                                }
                            }
                        }
                    } else if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 2;

                        for i in (0..32).step_by(N) {
                            let mut addr_values = [std::ptr::null_mut(); N];
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);

                                for j in 0..N {
                                    let elem = llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        j as u64,
                                        0,
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildExtractElement(
                                        builder,
                                        vaddr_value,
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildZExt(
                                        builder,
                                        vaddr_value,
                                        ty_i64,
                                        empty_name.as_ptr(),
                                    );
                                    let addr_value = llvm::core::LLVMBuildAdd(
                                        builder,
                                        saddr_value,
                                        vaddr_value,
                                        empty_name.as_ptr(),
                                    );
                                    addr_values[j] = addr_value;
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                for j in 0..N {
                                    let elem = llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        j as u64,
                                        0,
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildExtractElement(
                                        builder,
                                        vaddr_value,
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                    addr_values[j] = vaddr_value;
                                }
                            }

                            let mut values = [llvm::core::LLVMGetUndef(ty_i32xn); NUM_WORDS];
                            for j in 0..N {
                                let vec_elem = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    j as u64,
                                    0,
                                );

                                let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                    context,
                                    emitter.function,
                                    empty_name.as_ptr(),
                                );

                                let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                    context,
                                    emitter.function,
                                    empty_name.as_ptr(),
                                );

                                let exec = llvm::core::LLVMBuildExtractElement(
                                    builder,
                                    mask,
                                    vec_elem,
                                    empty_name.as_ptr(),
                                );

                                llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                                llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                                let offset = addr_values[j as usize];

                                let mut words = Vec::new();

                                for k in 0..NUM_WORDS {
                                    let addr = llvm::core::LLVMBuildAdd(
                                        builder,
                                        offset,
                                        llvm::core::LLVMConstInt(
                                            llvm::core::LLVMInt64TypeInContext(context),
                                            ((((inst.ioffset << 8) as i32) >> 8) as i64
                                                + (k as i64) * 4)
                                                as u64,
                                            0,
                                        ),
                                        empty_name.as_ptr(),
                                    );
                                    let ptr = llvm::core::LLVMBuildIntToPtr(
                                        builder,
                                        addr,
                                        ty_p0,
                                        empty_name.as_ptr(),
                                    );
                                    let data = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        ptr,
                                        empty_name.as_ptr(),
                                    );

                                    words.push(data);
                                }

                                llvm::core::LLVMBuildBr(builder, bb_cont);
                                llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);

                                for k in 0..NUM_WORDS {
                                    let phi_value = llvm::core::LLVMBuildPhi(
                                        builder,
                                        ty_i32,
                                        empty_name.as_ptr(),
                                    );

                                    llvm::core::LLVMAddIncoming(
                                        phi_value,
                                        [
                                            words[k],
                                            llvm::core::LLVMConstInt(
                                                llvm::core::LLVMInt32TypeInContext(context),
                                                0,
                                                0,
                                            ),
                                        ]
                                        .as_mut_ptr(),
                                        [bb_exec, bb].as_mut_ptr(),
                                        2,
                                    );

                                    words[k] = phi_value;
                                }

                                for k in 0..NUM_WORDS {
                                    values[k] = llvm::core::LLVMBuildInsertElement(
                                        builder,
                                        values[k],
                                        words[k],
                                        vec_elem,
                                        empty_name.as_ptr(),
                                    );
                                }
                                bb = bb_cont;
                            }

                            for k in 0..NUM_WORDS {
                                emitter.emit_store_vgpr_u32xn::<N>(
                                    inst.vdst as u32 + k as u32,
                                    i as u32,
                                    values[k as usize],
                                    mask,
                                );
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..2 {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64 + j * 4) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(
                                    inst.vdst as u32 + j as u32,
                                    elem,
                                    data,
                                );
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_LOAD_B128 => {
                    if USE_SIMD && USE_MASKED_GATHER {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 4;

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                let vaddr_value = llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                );

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.gather.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i64xn];
                                    let data = llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_i64xn,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [
                                            ptr,
                                            llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                            mask,
                                            llvm::core::LLVMGetPoison(ty_i64xn),
                                        ]
                                        .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );

                                    emitter.emit_store_vgpr_u64xn::<N>(
                                        inst.vdst as u32 + j as u32,
                                        i,
                                        data,
                                        mask,
                                    );
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.gather.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i64xn];
                                    let data = llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_i64xn,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [
                                            ptr,
                                            llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                            mask,
                                            llvm::core::LLVMGetPoison(ty_i64xn),
                                        ]
                                        .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );

                                    emitter.emit_store_vgpr_u64xn::<N>(
                                        inst.vdst as u32 + j as u32,
                                        i,
                                        data,
                                        mask,
                                    );
                                }
                            }
                        }
                    } else if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 4;

                        let saddr_value = if inst.saddr != 124 {
                            emitter.emit_load_sgpr_u64(inst.saddr as u32)
                        } else {
                            llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt64TypeInContext(context),
                                0,
                                0,
                            )
                        };

                        for i in (0..32).step_by(N) {
                            let mut addr_values = [std::ptr::null_mut(); N];
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);

                                for j in 0..N {
                                    let elem = llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        j as u64,
                                        0,
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildExtractElement(
                                        builder,
                                        vaddr_value,
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildZExt(
                                        builder,
                                        vaddr_value,
                                        ty_i64,
                                        empty_name.as_ptr(),
                                    );
                                    let addr_value = llvm::core::LLVMBuildAdd(
                                        builder,
                                        saddr_value,
                                        vaddr_value,
                                        empty_name.as_ptr(),
                                    );
                                    addr_values[j] = addr_value;
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                for j in 0..N {
                                    let elem = llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        j as u64,
                                        0,
                                    );
                                    let vaddr_value = llvm::core::LLVMBuildExtractElement(
                                        builder,
                                        vaddr_value,
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                    addr_values[j] = vaddr_value;
                                }
                            }

                            let mut values = [llvm::core::LLVMGetUndef(ty_i32xn); NUM_WORDS];
                            for j in 0..N {
                                let elem = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    j as u64,
                                    0,
                                );

                                let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                    context,
                                    emitter.function,
                                    empty_name.as_ptr(),
                                );

                                let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                    context,
                                    emitter.function,
                                    empty_name.as_ptr(),
                                );

                                let exec = llvm::core::LLVMBuildExtractElement(
                                    builder,
                                    mask,
                                    elem,
                                    empty_name.as_ptr(),
                                );

                                llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                                llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                                let offset = addr_values[j as usize];

                                let mut words = Vec::new();

                                for k in 0..NUM_WORDS {
                                    let addr = llvm::core::LLVMBuildAdd(
                                        builder,
                                        offset,
                                        llvm::core::LLVMConstInt(
                                            llvm::core::LLVMInt64TypeInContext(context),
                                            ((((inst.ioffset << 8) as i32) >> 8) as i64
                                                + (k as i64) * 4)
                                                as u64,
                                            0,
                                        ),
                                        empty_name.as_ptr(),
                                    );
                                    let ptr = llvm::core::LLVMBuildIntToPtr(
                                        builder,
                                        addr,
                                        ty_p0,
                                        empty_name.as_ptr(),
                                    );
                                    let data = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        llvm::core::LLVMInt32TypeInContext(context),
                                        ptr,
                                        empty_name.as_ptr(),
                                    );

                                    words.push(data);
                                }

                                llvm::core::LLVMBuildBr(builder, bb_cont);
                                llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);

                                for k in 0..NUM_WORDS {
                                    let phi_value = llvm::core::LLVMBuildPhi(
                                        builder,
                                        ty_i32,
                                        empty_name.as_ptr(),
                                    );

                                    llvm::core::LLVMAddIncoming(
                                        phi_value,
                                        [
                                            words[k],
                                            llvm::core::LLVMConstInt(
                                                llvm::core::LLVMInt32TypeInContext(context),
                                                0,
                                                0,
                                            ),
                                        ]
                                        .as_mut_ptr(),
                                        [bb_exec, bb].as_mut_ptr(),
                                        2,
                                    );

                                    words[k] = phi_value;
                                }

                                for k in 0..NUM_WORDS {
                                    values[k] = llvm::core::LLVMBuildInsertElement(
                                        builder,
                                        values[k],
                                        words[k],
                                        elem,
                                        empty_name.as_ptr(),
                                    );
                                }
                                bb = bb_cont;
                            }

                            for k in 0..NUM_WORDS {
                                emitter.emit_store_vgpr_u32xn::<N>(
                                    inst.vdst as u32 + k as u32,
                                    i as u32,
                                    values[k as usize],
                                    mask,
                                );
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..4 {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64 + j * 4) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(
                                    inst.vdst as u32 + j as u32,
                                    elem,
                                    data,
                                );
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_STORE_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 1;

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            let vaddr_value = if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask)
                            };

                            let ioffset = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(
                                    ty_i64,
                                    ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                    0,
                                ); N]
                                    .as_mut_ptr(),
                                N as u32,
                            );
                            let addr = llvm::core::LLVMBuildAdd(
                                builder,
                                vaddr_value,
                                ioffset,
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                addr,
                                ty_p0xn,
                                empty_name.as_ptr(),
                            );

                            for j in 0..NUM_WORDS {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let value = emitter.emit_load_vgpr_u32xn::<N>(
                                    inst.vsrc as u32 + j as u32,
                                    i as u32,
                                    mask,
                                );

                                let intrinsic_name =
                                    format!("llvm.masked.scatter.v{}i32.v{}p0\0", N, N);
                                let mut param_tys = vec![ty_i32xn, ty_p0xn];

                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );

                                let mut param_tys = vec![ty_i32xn, ty_p0xn, ty_i32, ty_i1xn];
                                llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_void,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as u32,
                                        0,
                                    ),
                                    intrinsic,
                                    [value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask]
                                        .as_mut_ptr(),
                                    4,
                                    empty_name.as_ptr(),
                                );
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..1 {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64 + j * 4) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );

                                let data =
                                    emitter.emit_load_vgpr_u32(inst.vsrc as u32 + j as u32, elem);

                                llvm::core::LLVMBuildStore(builder, data, ptr);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_STORE_B64 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 2;

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                let vaddr_value = llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                );

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let value = emitter.emit_load_vgpr_u64xn::<N>(
                                        inst.vsrc as u32 + j as u32,
                                        i as u32,
                                        mask,
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.scatter.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_i64xn, ty_p0xn, ty_i32, ty_i1xn];
                                    llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_void,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask]
                                            .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let value = emitter.emit_load_vgpr_u64xn::<N>(
                                        inst.vsrc as u32 + j as u32,
                                        i as u32,
                                        mask,
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.scatter.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_i64xn, ty_p0xn, ty_i32, ty_i1xn];
                                    llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_void,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask]
                                            .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );
                                }
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..2 {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64 + j * 4) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );

                                let data =
                                    emitter.emit_load_vgpr_u32(inst.vsrc as u32 + j as u32, elem);

                                llvm::core::LLVMBuildStore(builder, data, ptr);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::GLOBAL_STORE_B128 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 4;

                        let saddr_value = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);

                            let zero_vec = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                            let saddr_value = llvm::core::LLVMBuildInsertElement(
                                builder,
                                poison,
                                saddr_value,
                                llvm::core::LLVMConstInt(ty_i64, 0, 0),
                                empty_name.as_ptr(),
                            );
                            let saddr_value = llvm::core::LLVMBuildShuffleVector(
                                builder,
                                saddr_value,
                                poison,
                                zero_vec,
                                empty_name.as_ptr(),
                            );
                            saddr_value
                        } else {
                            std::ptr::null_mut()
                        };

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);
                            if inst.saddr != 124 {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i, mask);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    ty_i64xn,
                                    empty_name.as_ptr(),
                                );
                                let vaddr_value = llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                );

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let value = emitter.emit_load_vgpr_u64xn::<N>(
                                        inst.vsrc as u32 + j as u32,
                                        i as u32,
                                        mask,
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.scatter.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_i64xn, ty_p0xn, ty_i32, ty_i1xn];
                                    llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_void,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask]
                                            .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );
                                }
                            } else {
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask);

                                let ioffset = llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                                        0,
                                    ); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    vaddr_value,
                                    ioffset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0xn,
                                    empty_name.as_ptr(),
                                );

                                for j in (0..NUM_WORDS).step_by(2) {
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_i32,
                                        ptr,
                                        [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)]
                                            .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );

                                    let value = emitter.emit_load_vgpr_u64xn::<N>(
                                        inst.vsrc as u32 + j as u32,
                                        i as u32,
                                        mask,
                                    );

                                    let intrinsic_name =
                                        format!("llvm.masked.scatter.v{}i64.v{}p0\0", N, N);
                                    let mut param_tys = vec![ty_i64xn, ty_p0xn];

                                    let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                        intrinsic_name.as_ptr() as *const _,
                                        intrinsic_name.len() as usize,
                                    );
                                    let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                        emitter.module,
                                        intrinsic_id,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as usize,
                                    );

                                    let mut param_tys = vec![ty_i64xn, ty_p0xn, ty_i32, ty_i1xn];
                                    llvm::core::LLVMBuildCall2(
                                        builder,
                                        llvm::core::LLVMFunctionType(
                                            ty_void,
                                            param_tys.as_mut_ptr(),
                                            param_tys.len() as u32,
                                            0,
                                        ),
                                        intrinsic,
                                        [value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask]
                                            .as_mut_ptr(),
                                        4,
                                        empty_name.as_ptr(),
                                    );
                                }
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );
                            let offset = if inst.saddr != 124 {
                                let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                                let vaddr_value =
                                    emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
                                let vaddr_value = llvm::core::LLVMBuildZExt(
                                    builder,
                                    vaddr_value,
                                    llvm::core::LLVMInt64TypeInContext(context),
                                    empty_name.as_ptr(),
                                );
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    saddr_value,
                                    vaddr_value,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem)
                            };
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(
                                llvm::core::LLVMInt32TypeInContext(context),
                                i as u64,
                                0,
                            );

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..4 {
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    offset,
                                    llvm::core::LLVMConstInt(
                                        llvm::core::LLVMInt64TypeInContext(context),
                                        ((((inst.ioffset << 8) as i32) >> 8) as i64 + j * 4) as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );

                                let data =
                                    emitter.emit_load_vgpr_u32(inst.vsrc as u32 + j as u32, elem);

                                llvm::core::LLVMBuildStore(builder, data, ptr);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VSCRATCH(inst) => match inst.op {
                I::SCRATCH_STORE_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 1;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                            let ioffset_value =
                                llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                            let saddr_value = llvm::core::LLVMBuildZExt(
                                builder,
                                saddr_value,
                                ty_i64,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                saddr_value,
                                ioffset_value,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildMul(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, 32, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                emitter.scratch_base,
                                offset,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                offset,
                                ty_p0,
                                empty_name.as_ptr(),
                            );
                            for j in 0..NUM_WORDS {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let data = emitter.emit_load_vgpr_u32xn::<N>(
                                    inst.vsrc as u32 + j as u32,
                                    i,
                                    mask,
                                );

                                llvm::core::LLVMBuildStore(builder, data, ptr);
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                            let ioffset_value =
                                llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                            let saddr_value = llvm::core::LLVMBuildZExt(
                                builder,
                                saddr_value,
                                ty_i64,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                saddr_value,
                                ioffset_value,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildMul(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, 32, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                emitter.scratch_base,
                                offset,
                                empty_name.as_ptr(),
                            );
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..1 {
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    offset,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let data =
                                    emitter.emit_load_vgpr_u32(inst.vsrc as u32 + j as u32, elem);

                                llvm::core::LLVMBuildStore(builder, data, ptr);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::SCRATCH_LOAD_B32 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        const NUM_WORDS: usize = 1;

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                            let ioffset_value =
                                llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                            let saddr_value = llvm::core::LLVMBuildZExt(
                                builder,
                                saddr_value,
                                ty_i64,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                saddr_value,
                                ioffset_value,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildMul(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, 32, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                emitter.scratch_base,
                                offset,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                offset,
                                ty_p0,
                                empty_name.as_ptr(),
                            );
                            for j in 0..NUM_WORDS {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i32xn,
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32xn::<N>(
                                    inst.vdst as u32 + j as u32,
                                    i as u32,
                                    data,
                                    mask,
                                );
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                        let mut offsets = Vec::new();
                        for i in 0..32 {
                            let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                            let ioffset_value =
                                llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                            let saddr_value = llvm::core::LLVMBuildZExt(
                                builder,
                                saddr_value,
                                ty_i64,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                saddr_value,
                                ioffset_value,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildMul(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, 32, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                emitter.scratch_base,
                                offset,
                                empty_name.as_ptr(),
                            );
                            offsets.push(offset);
                        }

                        for i in 0..32 {
                            let empty_name = std::ffi::CString::new("").unwrap();
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            let offset = offsets[i];

                            for j in 0..1 {
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    offset,
                                    ty_p0,
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(
                                    inst.vdst as u32 + j as u32,
                                    elem,
                                    data,
                                );
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::DS(inst) => match inst.op {
                I::DS_STORE_B8 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                        let ty_i8xn = llvm::core::LLVMVectorType(ty_i8, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            let offset =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, i, mask);

                            let offset = llvm::core::LLVMBuildZExt(
                                builder,
                                offset,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, inst.offset0 as u64, 0); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i8,
                                emitter.lds_ptr,
                                [offset].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let value = emitter.emit_load_vgpr_u32xn::<N>(
                                inst.data0 as u32,
                                i as u32,
                                mask,
                            );

                            let value = llvm::core::LLVMBuildTrunc(
                                builder,
                                value,
                                ty_i8xn,
                                empty_name.as_ptr(),
                            );

                            let intrinsic_name = format!("llvm.masked.scatter.v{}i8.v{}p0\0", N, N);
                            let mut param_tys = vec![ty_i8xn, ty_p0xn];

                            let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                intrinsic_name.as_ptr() as *const _,
                                intrinsic_name.len() as usize,
                            );
                            let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                emitter.module,
                                intrinsic_id,
                                param_tys.as_mut_ptr(),
                                param_tys.len() as usize,
                            );

                            let mut param_tys = vec![ty_i8xn, ty_p0xn, ty_i32, ty_i1xn];
                            llvm::core::LLVMBuildCall2(
                                builder,
                                llvm::core::LLVMFunctionType(
                                    ty_void,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as u32,
                                    0,
                                ),
                                intrinsic,
                                [value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask]
                                    .as_mut_ptr(),
                                4,
                                empty_name.as_ptr(),
                            );
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);

                            let offset = emitter.emit_load_vgpr_u32(inst.addr as u32, elem);
                            let offset = llvm::core::LLVMBuildZExt(
                                builder,
                                offset,
                                ty_i64,
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, inst.offset0 as u64, 0),
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i8,
                                emitter.lds_ptr,
                                [offset].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            {
                                let data = emitter.emit_load_vgpr_u32(inst.data0 as u32, elem);

                                let data = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    data,
                                    ty_i8,
                                    empty_name.as_ptr(),
                                );

                                llvm::core::LLVMBuildStore(builder, data, ptr);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                I::DS_LOAD_U8 => {
                    if USE_SIMD {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();

                        const N: usize = SIMD_WIDTH;

                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                        let ty_i8xn = llvm::core::LLVMVectorType(ty_i8, N as u32);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);

                        let exec_value = emitter.emit_load_sgpr_u32(126);

                        for i in (0..32).step_by(N) {
                            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                            let offset =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, i, mask);

                            let offset = llvm::core::LLVMBuildZExt(
                                builder,
                                offset,
                                ty_i64xn,
                                empty_name.as_ptr(),
                            );

                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, inst.offset0 as u64, 0); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i8,
                                emitter.lds_ptr,
                                [offset].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            {
                                let intrinsic_name =
                                    format!("llvm.masked.gather.v{}i8.v{}p0\0", N, N);
                                let mut param_tys = vec![ty_i8xn, ty_p0xn];

                                let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(
                                    intrinsic_name.as_ptr() as *const _,
                                    intrinsic_name.len() as usize,
                                );
                                let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
                                    emitter.module,
                                    intrinsic_id,
                                    param_tys.as_mut_ptr(),
                                    param_tys.len() as usize,
                                );

                                let mut param_tys = vec![ty_p0xn, ty_i32, ty_i1xn, ty_i8xn];
                                let data = llvm::core::LLVMBuildCall2(
                                    builder,
                                    llvm::core::LLVMFunctionType(
                                        ty_i8xn,
                                        param_tys.as_mut_ptr(),
                                        param_tys.len() as u32,
                                        0,
                                    ),
                                    intrinsic,
                                    [
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i8xn),
                                    ]
                                    .as_mut_ptr(),
                                    4,
                                    empty_name.as_ptr(),
                                );

                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, data, mask);
                            }
                        }
                    } else {
                        let emitter = self;
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        for i in 0..32 {
                            let elem = llvm::core::LLVMConstInt(ty_i64, i as u64, 0);
                            let offset = emitter.emit_load_vgpr_u32(inst.addr as u32, elem);

                            let offset = llvm::core::LLVMBuildZExt(
                                builder,
                                offset,
                                ty_i64,
                                empty_name.as_ptr(),
                            );

                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(ty_i64, inst.offset0 as u64, 0),
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i8,
                                emitter.lds_ptr,
                                [offset].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let bb_exec = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let bb_cont = llvm::core::LLVMAppendBasicBlockInContext(
                                context,
                                emitter.function,
                                empty_name.as_ptr(),
                            );

                            let exec = emitter.emit_exec_bit(elem);

                            llvm::core::LLVMBuildCondBr(builder, exec, bb_exec, bb_cont);

                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);

                            {
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i8,
                                    ptr,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, data);
                            }

                            llvm::core::LLVMBuildBr(builder, bb_cont);
                            llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                            bb = bb_cont;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }
}

#[derive(Debug, Clone)]
struct RegisterUsage {
    use_sgprs: HashSet<u32>,
    use_vgprs: HashSet<u32>,
    def_sgprs: HashSet<u32>,
    def_vgprs: HashSet<u32>,
    incomming_sgprs: HashSet<u32>,
    incomming_vgprs: HashSet<u32>,
}

impl RegisterUsage {
    fn new() -> Self {
        RegisterUsage {
            use_sgprs: HashSet::new(),
            use_vgprs: HashSet::new(),
            def_sgprs: HashSet::new(),
            def_vgprs: HashSet::new(),
            incomming_sgprs: HashSet::new(),
            incomming_vgprs: HashSet::new(),
        }
    }

    fn use_sgpr_u32(&mut self, reg: u32) {
        if reg == 124 {
            return;
        }
        if !self.def_sgprs.contains(&reg) {
            self.incomming_sgprs.insert(reg);
        }
        self.use_sgprs.insert(reg);
    }

    fn use_sgpr_u64(&mut self, reg: u32) {
        self.use_sgpr_u32(reg);
        self.use_sgpr_u32(reg + 1);
    }

    fn _use_sgpr_f64(&mut self, reg: u32) {
        self.use_sgpr_u32(reg);
        self.use_sgpr_u32(reg + 1);
    }

    fn use_vgpr_u32(&mut self, reg: u32) {
        if !self.def_vgprs.contains(&reg) {
            self.incomming_vgprs.insert(reg);
        }
        self.use_vgprs.insert(reg);
    }

    fn use_vgpr_f32(&mut self, reg: u32) {
        if !self.def_vgprs.contains(&reg) {
            self.incomming_vgprs.insert(reg);
        }
        self.use_vgprs.insert(reg);
    }

    fn use_vgpr_u64(&mut self, reg: u32) {
        self.use_vgpr_u32(reg);
        self.use_vgpr_u32(reg + 1);
    }

    fn use_vgpr_f64(&mut self, reg: u32) {
        self.use_vgpr_u32(reg);
        self.use_vgpr_u32(reg + 1);
    }

    fn use_operand_u32(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => self.use_sgpr_u32(*reg as u32),
            SourceOperand::VectorRegister(reg) => self.use_vgpr_u32(*reg as u32),
            _ => {}
        };
    }

    fn use_operand_f32(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => self.use_sgpr_u32(*reg as u32),
            SourceOperand::VectorRegister(reg) => self.use_vgpr_u32(*reg as u32),
            _ => {}
        };
    }

    fn use_operand_u64(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => {
                self.use_sgpr_u32(*reg as u32);
                self.use_sgpr_u32((*reg + 1) as u32);
            }
            SourceOperand::VectorRegister(reg) => {
                self.use_vgpr_u32(*reg as u32);
                self.use_vgpr_u32((*reg + 1) as u32);
            }
            _ => {}
        };
    }

    fn use_operand_f64(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => {
                self.use_sgpr_u32(*reg as u32);
                self.use_sgpr_u32((*reg + 1) as u32);
            }
            SourceOperand::VectorRegister(reg) => {
                self.use_vgpr_u32(*reg as u32);
                self.use_vgpr_u32((*reg + 1) as u32);
            }
            _ => {}
        };
    }

    fn def_sgpr_u32(&mut self, reg: u32) {
        if reg == 124 {
            return;
        }
        self.def_sgprs.insert(reg);
    }

    fn def_sgpr_u64(&mut self, reg: u32) {
        self.def_sgpr_u32(reg);
        self.def_sgpr_u32(reg + 1);
    }

    fn _def_sgpr_f64(&mut self, reg: u32) {
        self.def_sgpr_u32(reg);
        self.def_sgpr_u32(reg + 1);
    }

    fn def_vgpr_u32(&mut self, reg: u32) {
        self.use_vgpr_u32(reg);
        self.def_vgprs.insert(reg);
    }

    fn def_vgpr_f32(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
    }

    fn def_vgpr_u64(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
        self.def_vgpr_u32(reg + 1);
    }

    fn def_vgpr_f64(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
        self.def_vgpr_u32(reg + 1);
    }
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

    fn analyze_instructions(inst: &InstFormat, reg_usage: &mut RegisterUsage) {
        match inst {
            InstFormat::SOPP(inst) => match inst.op {
                I::S_CLAUSE => {}
                I::S_WAIT_KMCNT => {}
                I::S_DELAY_ALU => {}
                I::S_WAIT_ALU => {}
                I::S_WAIT_LOADCNT => {}
                I::S_CBRANCH_SCC0 => {}
                I::S_CBRANCH_SCC1 => {}
                I::S_BRANCH => {}
                I::S_ENDPGM => {}
                I::S_WAIT_BVHCNT => {}
                I::S_WAIT_SAMPLECNT => {}
                I::S_WAIT_STORECNT => {}
                I::S_WAIT_LOADCNT_DSCNT => {}
                I::S_WAIT_DSCNT => {}
                I::S_BARRIER_WAIT => {}
                I::S_NOP => {}
                I::S_SENDMSG => {}
                I::S_CBRANCH_EXECZ => {
                    reg_usage.use_sgpr_u32(126);
                }
                I::S_CBRANCH_EXECNZ => {
                    reg_usage.use_sgpr_u32(126);
                }
                I::S_CBRANCH_VCCZ => {
                    reg_usage.use_sgpr_u32(106);
                }
                I::S_CBRANCH_VCCNZ => {
                    reg_usage.use_sgpr_u32(106);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOPC(inst) => match inst.op {
                I::V_CMP_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_GT_U64 => {
                    reg_usage.use_operand_u64(&inst.src0);
                    reg_usage.use_vgpr_u64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_GT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NLT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NGT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMPX_NGT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_NGE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_LT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_LT_I32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP1(inst) => match inst.op {
                I::V_CVT_F64_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MOV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_RCP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_RSQ_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_RNDNE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_FRACT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_CVT_I32_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CVT_F64_I32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_READFIRSTLANE_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CVT_F32_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_CVT_U32_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_RCP_IFLAG_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP2(inst) => match inst.op {
                I::V_ADD_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_AND_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_XOR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHRREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CNDMASK_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.use_sgpr_u32(106);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MUL_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MAX_NUM_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MUL_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3(inst) => match inst.op {
                I::V_BFE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CNDMASK_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NLT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NGT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LG_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NEQ_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MUL_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_FMA_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_DIV_FMAS_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_DIV_FIXUP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_LDEXP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_CMP_CLASS_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_XAD_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_XOR3_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD3_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MUL_LO_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_TRIG_PREOP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MAX_NUM_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_READLANE_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_WRITELANE_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_AND_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_EQ_U16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_U16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_LSHRREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHL_OR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ASHRREV_I32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_NC_U16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B64 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u64(&inst.src1);
                    reg_usage.def_vgpr_u64(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3SD(inst) => match inst.op {
                I::V_MAD_CO_U64_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u64(&inst.src2);
                    reg_usage.def_vgpr_u64(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_DIV_SCALE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_ADD_CO_CI_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_ADD_CO_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOPD(inst) => {
                let vdstx = inst.vdstx as u32;
                let vdsty = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as u32;
                match inst.opx {
                    I::V_DUAL_CNDMASK_B32 => {
                        reg_usage.use_operand_u32(&inst.src0x);
                        reg_usage.use_vgpr_u32(inst.vsrc1x as u32);
                        reg_usage.use_sgpr_u32(106);
                        reg_usage.def_vgpr_u32(vdstx);
                    }
                    I::V_DUAL_MOV_B32 => {
                        reg_usage.use_operand_u32(&inst.src0x);
                        reg_usage.def_vgpr_u32(vdstx);
                    }
                    _ => {
                        panic!("Unsupported instruction: {:?}", inst);
                    }
                }
                match inst.opy {
                    I::V_DUAL_CNDMASK_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.use_sgpr_u32(106);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_MOV_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_ADD_NC_U32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_LSHLREV_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    _ => {
                        panic!("Unsupported instruction: {:?}", inst);
                    }
                }
            }
            InstFormat::SMEM(inst) => match inst.op {
                I::S_LOAD_B32 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    reg_usage.def_sgpr_u32(inst.sdata as u32);
                }
                I::S_LOAD_B64 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..2 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                I::S_LOAD_B96 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..3 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                I::S_LOAD_B128 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..4 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOP1(inst) => match inst.op {
                I::S_BARRIER_SIGNAL => {}
                I::S_AND_NOT1_SAVEEXEC_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_sgpr_u32(126);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::S_OR_SAVEEXEC_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_sgpr_u32(126);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::S_MOV_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MOV_B64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.def_sgpr_u64(inst.sdst as u32);
                }
                I::S_CTZ_I32_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOP2(inst) => match inst.op {
                I::S_AND_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_OR_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_XOR_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_AND_NOT1_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_OR_NOT1_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_LSHR_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_CSELECT_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_ADD_NC_U64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u64(&inst.ssrc1);
                    reg_usage.def_sgpr_u64(inst.sdst as u32);
                }
                I::S_ADD_CO_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_SUB_CO_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MUL_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MUL_HI_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_LSHL_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MAX_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOPC(inst) => match inst.op {
                I::S_CMP_LG_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_LT_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_GE_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_LT_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_LG_U64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u64(&inst.ssrc1);
                }
                I::S_CMP_EQ_U64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u64(&inst.ssrc1);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VGLOBAL(inst) => match inst.op {
                I::GLOBAL_LOAD_U8 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_U16 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_B32 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_B64 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..2 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_B128 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..4 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_STORE_B32 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                I::GLOBAL_STORE_B64 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..2 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                I::GLOBAL_STORE_B128 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..4 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                I::GLOBAL_WB => {}
                I::GLOBAL_INV => {}
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VSCRATCH(inst) => match inst.op {
                I::SCRATCH_LOAD_B32 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::SCRATCH_LOAD_B64 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..2 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::SCRATCH_STORE_B32 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    reg_usage.use_vgpr_u32(inst.vsrc as u32);
                }
                I::SCRATCH_STORE_B64 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..2 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::DS(inst) => match inst.op {
                I::DS_LOAD_U8 => {
                    reg_usage.use_vgpr_u32(inst.addr as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::DS_STORE_B8 => {
                    reg_usage.use_vgpr_u32(inst.addr as u32);
                    reg_usage.use_vgpr_u32(inst.data0 as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }
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

    pub fn build_from_program(&mut self, program: &RDNAProgram) -> &mut InstBlock {
        let mut inst_block = InstBlock::new();

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

            let ret_value = llvm::core::LLVMConstInt(
                llvm::core::LLVMInt32TypeInContext(context),
                Signals::None as u64,
                0,
            );

            let mut emitter = IREmitter {
                context,
                module,
                function,
                builder,
                sgprs_ptr,
                vgprs_ptr,
                scc_ptr,
                scratch_base,
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

                let mut reg_usage = RegisterUsage::new();
                reg_usage.incomming_sgprs.insert(106);
                reg_usage.incomming_sgprs.insert(126);

                reg_usage.use_sgprs.insert(106);
                reg_usage.use_sgprs.insert(126);

                reg_usage.def_sgprs.insert(106);
                reg_usage.def_sgprs.insert(126);

                for inst in &block.insts {
                    Self::analyze_instructions(inst, &mut reg_usage);
                }

                basic_block = emitter.emit_restore_registers(basic_block, &reg_usage);

                if is_terminator(block.insts.last().unwrap()) {
                    for inst in &block.insts[..block.insts.len() - 1] {
                        basic_block = emitter.emit_instruction(basic_block, inst);
                    }

                    emitter.emit_save_registers(basic_block, &reg_usage);

                    let last_inst = block.insts.last().unwrap();
                    if let InstFormat::SOPP(inst) = last_inst {
                        match inst.op {
                            I::S_CBRANCH_EXECZ => {
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let exec_value = emitter.emit_load_sgpr_u32(126);
                                let zero = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    0,
                                    0,
                                );
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
                            I::S_CBRANCH_VCCNZ => {
                                let empty_name = std::ffi::CString::new("").unwrap();
                                let vcc_value = emitter.emit_load_sgpr_u32(106);
                                let zero = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    0,
                                    0,
                                );
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
                                let ret_value = llvm::core::LLVMConstInt(
                                    llvm::core::LLVMInt32TypeInContext(context),
                                    Signals::EndOfProgram as u64,
                                    0,
                                );
                                llvm::core::LLVMBuildRet(builder, ret_value);
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

                    emitter.emit_save_registers(basic_block, &reg_usage);

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

            llvm::core::LLVMBuildBr(builder, *basic_blocks.get(&program.entry_pc).unwrap());

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
                b"lcssa,adce,early-cse,instcombine,aggressive-instcombine,mem2reg,gvn,dse,instsimplify,load-store-vectorizer,loop-fusion,loop-reduce,sink,loop-load-elim,reassociate,function-simplification<O3>,loop-vectorize,simplifycfg,loop-unroll<O3>\0".as_ptr() as *const _,
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

            llvm::orc2::lljit::LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
                jit_builder,
                llvm_obj_linking_layer_create,
                std::ptr::null_mut(),
            );

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

        let block_addr = program.entry_pc as u64;

        self.clear();

        self.insts_blocks.insert(block_addr, inst_block);

        self.insts_blocks.get_mut(&block_addr).unwrap()
    }

    pub fn get_or_build(&mut self) -> &mut InstBlock {
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
            let ret_value = llvm::core::LLVMConstInt(
                llvm::core::LLVMInt32TypeInContext(context),
                Signals::None as u64,
                0,
            );

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

            llvm::orc2::lljit::LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
                jit_builder,
                llvm_obj_linking_layer_create,
                std::ptr::null_mut(),
            );

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

        RDNAProgram {
            entry_pc: pc,
            insts_blocks: basic_blocks,
        }
    }
}
