use core::panic;
use std::os::raw::c_void;

use crate::instructions::*;
use crate::rdna_instructions::*;
use llvm_sys as llvm;

pub fn is_terminator(inst: &InstFormat) -> bool {
    match inst {
        InstFormat::SOPP(inst) => match inst.op {
            I::S_CBRANCH_SCC0
            | I::S_CBRANCH_SCC1
            | I::S_CBRANCH_VCCZ
            | I::S_CBRANCH_VCCNZ
            | I::S_CBRANCH_EXECZ
            | I::S_CBRANCH_EXECNZ
            | I::S_BRANCH => true,
            _ => false,
        },
        _ => false,
    }
}

pub struct InstBlock {
    context: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    engine: llvm::execution_engine::LLVMExecutionEngineRef,
    func: llvm_sys::prelude::LLVMValueRef,
    pub next_pc: usize,
}

impl Drop for InstBlock {
    fn drop(&mut self) {
        unsafe {
            if !self.module.is_null() {
                llvm::core::LLVMDisposeModule(self.module);
            }
            if !self.engine.is_null() {
                llvm::execution_engine::LLVMDisposeExecutionEngine(self.engine);
            }
            llvm::core::LLVMContextDispose(self.context);
        }
    }
}

impl InstBlock {
    pub fn new() -> Self {
        InstBlock {
            context: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
            engine: std::ptr::null_mut(),
            func: std::ptr::null_mut(),
            next_pc: 0,
        }
    }

    pub fn dispose(&mut self) {
        unsafe {
            llvm::core::LLVMContextDispose(self.context);
            if !self.module.is_null() {
                llvm::core::LLVMDisposeModule(self.module);
            }
        }
    }

    pub fn execute(&self, sgprs_ptr: *mut u32, vgprs_ptr: *mut u32, scc_ptr: *mut bool) {
        unsafe {
            let mut length = 0 as usize;
            let length_ptr = &mut length as *mut usize;

            let addr = llvm::execution_engine::LLVMGetFunctionAddress(
                self.engine,
                llvm::core::LLVMGetValueName2(self.func, length_ptr as *mut _),
            );

            let func = std::mem::transmute::<_, extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>(
                addr,
            );

            func(
                sgprs_ptr as *mut c_void,
                vgprs_ptr as *mut c_void,
                scc_ptr as *mut c_void,
            );
        }
    }
}

pub struct RDNATranslator {
    addresses: Vec<u64>,
    insts: Vec<InstFormat>,
    context: llvm::prelude::LLVMContextRef,
}

struct IREmitter {
    context: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    function: llvm::prelude::LLVMValueRef,
    builder: llvm::prelude::LLVMBuilderRef,
    sgprs_ptr: llvm::prelude::LLVMValueRef,
    vgprs_ptr: llvm::prelude::LLVMValueRef,
    scc_ptr: llvm::prelude::LLVMValueRef,
}

impl IREmitter {
    unsafe fn emit_vop_execmask(
        &self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        predicate: impl Fn(
            &IREmitter,
            llvm::prelude::LLVMBasicBlockRef,
            llvm::prelude::LLVMValueRef,
        ) -> llvm::prelude::LLVMBasicBlockRef,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let function = self.function;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let mut indices = vec![llvm::core::LLVMConstInt(
            llvm::core::LLVMInt64TypeInContext(context),
            126,
            0,
        )];
        let exec_ptr = llvm::core::LLVMBuildGEP2(
            builder,
            ty_i32,
            sgprs_ptr,
            indices.as_mut_ptr(),
            indices.len() as u32,
            empty_name.as_ptr(),
        );

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());
        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());
        let exec_value = llvm::core::LLVMBuildLoad2(builder, ty_i32, exec_ptr, empty_name.as_ptr());

        let index_i32 = llvm::core::LLVMBuildTrunc(builder, index, ty_i32, empty_name.as_ptr());
        let shifted = llvm::core::LLVMBuildShl(
            builder,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            index_i32,
            empty_name.as_ptr(),
        );
        let anded = llvm::core::LLVMBuildAnd(builder, exec_value, shifted, empty_name.as_ptr());
        let cmp = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            anded,
            llvm::core::LLVMConstInt(ty_i32, 0, 0),
            empty_name.as_ptr(),
        );

        let bb_loop_skip_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_cond =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_exit =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_skip_body, bb_loop_body);

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

    unsafe fn emit_vop_execmask_update_sgpr(
        &self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        sgpr_reg: u32,
        predicate: impl Fn(
            &IREmitter,
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
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let mut indices = vec![llvm::core::LLVMConstInt(
            llvm::core::LLVMInt64TypeInContext(context),
            126,
            0,
        )];
        let exec_ptr = llvm::core::LLVMBuildGEP2(
            builder,
            ty_i32,
            sgprs_ptr,
            indices.as_mut_ptr(),
            indices.len() as u32,
            empty_name.as_ptr(),
        );

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());
        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i64, empty_name.as_ptr());

        let vcc = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());

        let exec_value = llvm::core::LLVMBuildLoad2(builder, ty_i32, exec_ptr, empty_name.as_ptr());

        let index_i32 = llvm::core::LLVMBuildTrunc(builder, index, ty_i32, empty_name.as_ptr());
        let shifted = llvm::core::LLVMBuildShl(
            builder,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            index_i32,
            empty_name.as_ptr(),
        );
        let anded = llvm::core::LLVMBuildAnd(builder, exec_value, shifted, empty_name.as_ptr());
        let cmp = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            anded,
            llvm::core::LLVMConstInt(ty_i32, 0, 0),
            empty_name.as_ptr(),
        );

        let bb_loop_skip_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_cond =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_exit =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_skip_body, bb_loop_body);

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

    unsafe fn emit_load_vgpr_u32(
        &self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

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

    unsafe fn emit_load_vgpr_u64(
        &self,
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
        &self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let value = self.emit_load_vgpr_u64(reg, elem);
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64, empty_name.as_ptr())
    }

    unsafe fn emit_store_vgpr_u32(
        &self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

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

    unsafe fn emit_store_vgpr_u64(
        &self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

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

    unsafe fn emit_store_vgpr_f64(
        &self,
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

    unsafe fn emit_load_sgpr_u32(&self, reg: u32) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

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

    unsafe fn emit_load_sgpr_u64(&self, reg: u32) -> llvm::prelude::LLVMValueRef {
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

    unsafe fn emit_store_sgpr_u32(&self, reg: u32, value: llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

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

    unsafe fn emit_store_sgpr_u64(&self, reg: u32, value: llvm::prelude::LLVMValueRef) {
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
        &self,
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
        &self,
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
        &self,
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

    unsafe fn emit_vector_source_operand_u64(
        &self,
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
            _ => panic!("Unsupported source operand type: {:?}", operand),
        }
    }

    unsafe fn emit_vector_source_operand_f64(
        &self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);

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
                llvm::core::LLVMBuildBitCast(
                    self.builder,
                    value,
                    ty_f64,
                    std::ffi::CString::new("").unwrap().as_ptr(),
                )
            }
            SourceOperand::VectorRegister(value) => {
                let value = self.emit_load_vgpr_u64(*value as u32, elem);
                llvm::core::LLVMBuildBitCast(
                    self.builder,
                    value,
                    ty_f64,
                    std::ffi::CString::new("").unwrap().as_ptr(),
                )
            }
        }
    }
}

impl RDNATranslator {
    pub fn new() -> Self {
        RDNATranslator {
            addresses: Vec::new(),
            insts: Vec::new(),
            context: unsafe { llvm::core::LLVMContextCreate() },
        }
    }

    pub fn add_inst(&mut self, addr: u64, inst: InstFormat) {
        self.addresses.push(addr);
        self.insts.push(inst);
    }

    pub fn build(&mut self) -> InstBlock {
        for inst in &self.insts {
            println!("{:?}", inst);
        }
        let mut inst_block = InstBlock::new();

        unsafe {
            let context = self.context;
            let module = llvm::core::LLVMModuleCreateWithNameInContext(
                format!("block{}", self.get_address().unwrap()).as_ptr() as *const _,
                context,
            );

            let ty_void = llvm::core::LLVMVoidTypeInContext(context);
            let mut param_ty = vec![
                llvm::core::LLVMPointerType(llvm::core::LLVMInt32TypeInContext(context), 0),
                llvm::core::LLVMPointerType(llvm::core::LLVMInt32TypeInContext(context), 0),
                llvm::core::LLVMPointerType(llvm::core::LLVMInt8TypeInContext(context), 0),
            ];
            let ty_function = llvm::core::LLVMFunctionType(
                ty_void,
                param_ty.as_mut_ptr(),
                param_ty.len() as u32,
                0,
            );
            let function = llvm::core::LLVMAddFunction(
                module,
                format!("block{}", self.get_address().unwrap()).as_ptr() as *const _,
                ty_function,
            );

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

            let emitter = IREmitter {
                context,
                module,
                function,
                builder,
                sgprs_ptr,
                vgprs_ptr,
                scc_ptr,
            };

            for inst in &self.insts {
                match inst {
                    InstFormat::SOPP(inst) => match inst.op {
                        I::S_CLAUSE => {}
                        I::S_WAIT_KMCNT => {}
                        I::S_DELAY_ALU => {}
                        I::S_WAIT_ALU => {}
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::VOPC(inst) => match inst.op {
                        I::V_CMP_GT_U32 => {
                            bb = emitter.emit_vop_execmask_update_sgpr(
                                bb,
                                106,
                                |emitter, bb, elem| {
                                    let empty_name = std::ffi::CString::new("").unwrap();

                                    let s0_value =
                                        emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                                    let s1_value =
                                        emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
                                    let cmp_value = llvm::core::LLVMBuildICmp(
                                        builder,
                                        llvm::LLVMIntPredicate::LLVMIntUGT,
                                        s0_value,
                                        s1_value,
                                        empty_name.as_ptr(),
                                    );

                                    (bb, cmp_value)
                                },
                            );
                        }
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::VOP1(inst) => match inst.op {
                        I::V_CVT_F64_U32 => {
                            bb = emitter.emit_vop_execmask(bb, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                                let d_value = llvm::core::LLVMBuildSIToFP(
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
                        I::V_MOV_B32 => {
                            bb = emitter.emit_vop_execmask(bb, |emitter, bb, elem| {
                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                                let d_value = s0_value;

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                                bb
                            });
                        }
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::VOP2(inst) => match inst.op {
                        I::V_AND_B32 => {
                            bb = emitter.emit_vop_execmask(bb, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);
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
                        I::V_MUL_F64 => {
                            bb = emitter.emit_vop_execmask(bb, |emitter, bb, elem| {
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                                let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
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
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::VOP3(inst) => match inst.op {
                        I::V_BFE_U32 => {
                            bb = emitter.emit_vop_execmask(bb, |emitter, bb, elem| {
                                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                                let empty_name = std::ffi::CString::new("").unwrap();

                                let s0_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                                let s1_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src1, elem);

                                let s2_value =
                                    emitter.emit_vector_source_operand_u32(&inst.src2, elem);

                                let s1_value = llvm::core::LLVMBuildAnd(
                                    builder,
                                    s1_value,
                                    llvm::core::LLVMConstInt(ty_i32, 31, 0),
                                    empty_name.as_ptr(),
                                );
                                let s0_value = llvm::core::LLVMBuildLShr(
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
                                let s2_value = llvm::core::LLVMBuildShl(
                                    builder,
                                    llvm::core::LLVMConstInt(ty_i32, -1 as i64 as u64, 0),
                                    s2_value,
                                    empty_name.as_ptr(),
                                );
                                let s2_value = llvm::core::LLVMBuildXor(
                                    builder,
                                    s2_value,
                                    llvm::core::LLVMConstInt(ty_i32, -1 as i64 as u64, 0),
                                    empty_name.as_ptr(),
                                );
                                let d_value = llvm::core::LLVMBuildAnd(
                                    builder,
                                    s0_value,
                                    s2_value,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                                bb
                            });
                        }
                        I::V_CMP_GT_U32 => {
                            bb = emitter.emit_vop_execmask_update_sgpr(
                                bb,
                                inst.vdst as u32,
                                |emitter, bb, elem| {
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
                                },
                            );
                        }
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::VOP3SD(inst) => match inst.op {
                        I::V_MAD_CO_U64_U32 => {
                            bb = emitter.emit_vop_execmask_update_sgpr(
                                bb,
                                inst.sdst as u32,
                                |emitter, bb, elem| {
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
                                },
                            );
                        }
                        I::V_DIV_SCALE_F64 => {
                            bb = emitter.emit_vop_execmask_update_sgpr(
                                bb,
                                inst.sdst as u32,
                                |emitter, bb, elem| {
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
                                },
                            );
                        }
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::SMEM(inst) => match inst.op {
                        I::S_LOAD_B32 => {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32);

                            {
                                let offset =
                                    llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    sbase,
                                    offset,
                                    empty_name.as_ptr(),
                                );

                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    llvm::core::LLVMPointerType(ty_i32, 0),
                                    empty_name.as_ptr(),
                                );

                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_sgpr_u32(inst.sdata as u32, data);
                            }
                        }
                        I::S_LOAD_B64 => {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32);

                            for i in 0..2 {
                                let offset = llvm::core::LLVMConstInt(
                                    ty_i64,
                                    (inst.ioffset + i * 4) as u64,
                                    0,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    sbase,
                                    offset,
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    llvm::core::LLVMPointerType(ty_i32, 0),
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                            }
                        }
                        I::S_LOAD_B96 => {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32);

                            for i in 0..3 {
                                let offset = llvm::core::LLVMConstInt(
                                    ty_i64,
                                    (inst.ioffset + i * 4) as u64,
                                    0,
                                );
                                let addr = llvm::core::LLVMBuildAdd(
                                    builder,
                                    sbase,
                                    offset,
                                    empty_name.as_ptr(),
                                );
                                let ptr = llvm::core::LLVMBuildIntToPtr(
                                    builder,
                                    addr,
                                    llvm::core::LLVMPointerType(ty_i32, 0),
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    empty_name.as_ptr(),
                                );

                                emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                            }
                        }
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::SOP1(inst) => match inst.op {
                        I::S_AND_SAVEEXEC_B32 => {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                            let s1_value = emitter.emit_load_sgpr_u32(126);

                            emitter.emit_store_sgpr_u32(inst.sdst as u32, s1_value);

                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                s1_value,
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

                            llvm::core::LLVMBuildStore(builder, scc_value, emitter.scc_ptr);
                        }
                        I::S_MOV_B32 => {
                            let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                            let d_value = s0_value;

                            emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                        }
                        I::S_MOV_B64 => {
                            let s0_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc0);

                            let d_value = s0_value;

                            emitter.emit_store_sgpr_u64(inst.sdst as u32, d_value);
                        }
                        _ => {
                            panic!("Unsupported instruction: {:?}", inst);
                        }
                    },
                    InstFormat::SOP2(inst) => match inst.op {
                        I::S_AND_B32 => {
                            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                            let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                            let empty_name = std::ffi::CString::new("").unwrap();

                            let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                            let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                            let d_value = llvm::core::LLVMBuildAnd(
                                builder,
                                s0_value,
                                s1_value,
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

                            llvm::core::LLVMBuildStore(builder, scc_value, emitter.scc_ptr);
                        }
                        I::S_LSHR_B32 => {
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
                            let d_value = llvm::core::LLVMBuildLShr(
                                builder,
                                s0_value,
                                s1_value,
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

                            llvm::core::LLVMBuildStore(builder, scc_value, scc_ptr);
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

            llvm::core::LLVMBuildRetVoid(builder);

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
                llvm_sys::core::LLVMDisposeMessage(err.into_raw());
                panic!("Failed to verify main module: {}", err_.to_str().unwrap());
            }

            let pass_manager = llvm::core::LLVMCreateFunctionPassManagerForModule(module);
            llvm_sys::transforms::util::LLVMAddPromoteMemoryToRegisterPass(pass_manager);
            llvm_sys::transforms::instcombine::LLVMAddInstructionCombiningPass(pass_manager);
            llvm::transforms::scalar::LLVMAddReassociatePass(pass_manager);
            llvm_sys::transforms::scalar::LLVMAddGVNPass(pass_manager);
            llvm_sys::transforms::scalar::LLVMAddCFGSimplificationPass(pass_manager);
            llvm::core::LLVMInitializeFunctionPassManager(pass_manager);

            llvm::core::LLVMRunFunctionPassManager(pass_manager, function);

            llvm::core::LLVMDisposePassManager(pass_manager);

            llvm::core::LLVMDumpModule(module);

            llvm::execution_engine::LLVMLinkInMCJIT();
            llvm::execution_engine::LLVMLinkInInterpreter();

            llvm::target::LLVM_InitializeNativeTarget();
            llvm::target::LLVM_InitializeAllTargetMCs();
            llvm::target::LLVM_InitializeAllAsmParsers();
            llvm::target::LLVM_InitializeAllAsmPrinters();

            let mut engine = std::ptr::null_mut();
            let mut err = std::ptr::null_mut();
            let is_err = llvm::execution_engine::LLVMCreateJITCompilerForModule(
                &mut engine,
                module,
                2,
                &mut err,
            );

            if is_err != 0 {
                let err = std::ffi::CString::from_raw(err);
                let err_ = err.clone();
                llvm_sys::core::LLVMDisposeMessage(err.into_raw());
                panic!("Failed to create JIT compiler: {}", err_.to_str().unwrap());
            }

            inst_block.context = context;
            inst_block.module = module;
            inst_block.engine = engine;
            inst_block.func = function;
        }

        inst_block
    }

    pub fn get_address(&self) -> Option<u64> {
        self.addresses.first().copied()
    }

    pub fn clear(&mut self) {
        self.addresses.clear();
        self.insts.clear();
    }
}
