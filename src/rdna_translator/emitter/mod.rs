use crate::rdna_processor::Signals;

use llvm_sys as llvm;
use std::collections::HashMap;

use super::*;

mod math;
mod mem;
mod operands;
mod regs;
mod salu;
mod valu;
mod vop3;

pub(crate) struct RayIntersectionEmitter {
    pub node_addr_ptr: llvm::prelude::LLVMValueRef,
    pub values_ptr: llvm::prelude::LLVMValueRef,
    pub results_ptr: llvm::prelude::LLVMValueRef,
}

impl RayIntersectionEmitter {
    pub(crate) fn new() -> Self {
        RayIntersectionEmitter {
            node_addr_ptr: std::ptr::null_mut(),
            values_ptr: std::ptr::null_mut(),
            results_ptr: std::ptr::null_mut(),
        }
    }

    pub(crate) fn emit_alloc(
        &mut self,
        context: llvm::prelude::LLVMContextRef,
        builder: llvm::prelude::LLVMBuilderRef,
    ) {
        unsafe {
            let empty_name = std::ffi::CString::new("").unwrap();
            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
            let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
            let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

            const N: usize = SIMD_WIDTH;
            let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
            let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
            let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

            self.node_addr_ptr = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_i64xn,
                llvm::core::LLVMConstInt(ty_i32, 4, 0),
                empty_name.as_ptr(),
            );
            self.values_ptr = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_f32xn,
                llvm::core::LLVMConstInt(ty_i32, 10 * 32 / N as u64, 0),
                empty_name.as_ptr(),
            );
            self.results_ptr = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_i32xn,
                llvm::core::LLVMConstInt(ty_i32, 10 * 4, 0),
                empty_name.as_ptr(),
            );
        }
    }
}

pub(crate) struct MatrixEmitter {
    pub matrix_a_ptr: llvm::prelude::LLVMValueRef,
    pub matrix_b_ptr: llvm::prelude::LLVMValueRef,
    pub matrix_c_ptr: llvm::prelude::LLVMValueRef,
    pub matrix_d_ptr: llvm::prelude::LLVMValueRef,
}

impl MatrixEmitter {
    pub(crate) fn new() -> Self {
        MatrixEmitter {
            matrix_a_ptr: std::ptr::null_mut(),
            matrix_b_ptr: std::ptr::null_mut(),
            matrix_c_ptr: std::ptr::null_mut(),
            matrix_d_ptr: std::ptr::null_mut(),
        }
    }

    pub(crate) fn emit_alloc(
        &mut self,
        context: llvm::prelude::LLVMContextRef,
        builder: llvm::prelude::LLVMBuilderRef,
    ) {
        unsafe {
            const N: usize = SIMD_WIDTH;

            let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
            let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
            let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
            let empty_name = std::ffi::CString::new("").unwrap();

            let matrix_a = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_f32xn,
                llvm::core::LLVMConstInt(ty_i32, 16 * 16 / N as u64, 0),
                empty_name.as_ptr(),
            ); // col major
            let matrix_b = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_f32xn,
                llvm::core::LLVMConstInt(ty_i32, 16 * 16 / N as u64, 0),
                empty_name.as_ptr(),
            ); // row major
            let matrix_c = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_f32xn,
                llvm::core::LLVMConstInt(ty_i32, 16 * 16 / N as u64, 0),
                empty_name.as_ptr(),
            ); // row major
            let matrix_d = llvm::core::LLVMBuildArrayAlloca(
                builder,
                ty_f32xn,
                llvm::core::LLVMConstInt(ty_i32, 16 * 16 / N as u64, 0),
                empty_name.as_ptr(),
            ); // row major

            self.matrix_a_ptr = matrix_a;
            self.matrix_b_ptr = matrix_b;
            self.matrix_c_ptr = matrix_c;
            self.matrix_d_ptr = matrix_d;
        }
    }
}

pub(crate) struct IREmitter {
    pub(crate) context: llvm::prelude::LLVMContextRef,
    pub(crate) module: llvm::prelude::LLVMModuleRef,
    pub(crate) function: llvm::prelude::LLVMValueRef,
    pub(crate) builder: llvm::prelude::LLVMBuilderRef,
    pub(crate) sgprs_ptr: llvm::prelude::LLVMValueRef,
    pub(crate) vgprs_ptr: llvm::prelude::LLVMValueRef,
    pub(crate) scc_ptr: llvm::prelude::LLVMValueRef,
    pub(crate) scratch_base: llvm::prelude::LLVMValueRef,
    pub(crate) scratch_size: usize,
    pub(crate) lds_ptr: llvm::prelude::LLVMValueRef,
    pub(crate) pc_ptr: llvm::prelude::LLVMValueRef,
    pub(crate) ret_value: llvm::prelude::LLVMValueRef,
    pub(crate) exec_value: llvm::prelude::LLVMValueRef,
    pub(crate) local_scc_ptr: llvm::prelude::LLVMValueRef,
    pub(crate) sgpr_ptr_map: HashMap<u32, llvm::prelude::LLVMValueRef>,
    pub(crate) vgpr_ptr_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    pub(crate) vgpr_reg_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    pub(crate) vgpr_incomming_reg_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    pub(crate) vgpr_reg_f64_map: HashMap<u32, [llvm::prelude::LLVMValueRef; 32 / SIMD_WIDTH]>,
    pub(crate) use_vgpr_cache: bool,
    pub(crate) use_scc_cache: bool,
    pub(crate) ray: RayIntersectionEmitter,
    pub(crate) matrix: MatrixEmitter,
}

pub(crate) struct IntrinsicDeclaration {
    pub(crate) builder: llvm::prelude::LLVMBuilderRef,
    pub(crate) declaration: llvm::prelude::LLVMValueRef,
}

impl IntrinsicDeclaration {
    pub(crate) unsafe fn emit_call(
        &self,
        return_type: llvm::prelude::LLVMTypeRef,
        args: &[llvm::prelude::LLVMValueRef],
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let param_types = args
            .iter()
            .map(|&arg| llvm::core::LLVMTypeOf(arg))
            .collect::<Vec<llvm::prelude::LLVMTypeRef>>();

        llvm::core::LLVMBuildCall2(
            self.builder,
            llvm::core::LLVMFunctionType(
                return_type,
                param_types.as_ptr() as *mut _,
                param_types.len() as u32,
                0,
            ),
            self.declaration,
            args.as_ptr() as *mut _,
            args.len() as u32,
            empty_name.as_ptr(),
        )
    }
}

impl IREmitter {
    pub(crate) unsafe fn get_intrinsic_declaration(
        &mut self,
        name: &str,
        param_types: &[llvm::prelude::LLVMTypeRef],
    ) -> IntrinsicDeclaration {
        let mut param_types = param_types.to_vec();
        let intrinsic_id = llvm::core::LLVMLookupIntrinsicID(name.as_ptr() as *const _, name.len());
        let intrinsic = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            intrinsic_id,
            param_types.as_mut_ptr(),
            param_types.len() as usize,
        );
        IntrinsicDeclaration {
            builder: self.builder,
            declaration: intrinsic,
        }
    }

    pub(crate) unsafe fn emit_exec_bit(
        &mut self,
        index: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(index) != ty_i32 {
            panic!("Type of index is not i32");
        }

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

    pub(crate) unsafe fn emit_vcc_bit(
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

    pub(crate) unsafe fn emit_loop(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        n: u64,
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
        let empty_name = std::ffi::CString::new("").unwrap();

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_cond =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_exit =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());

        llvm::core::LLVMBuildBr(builder, bb_loop_body);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let bb_loop_body = predicate(self, bb_loop_body, index);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

        let next_index = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            empty_name.as_ptr(),
        );

        let cmp = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            next_index,
            llvm::core::LLVMConstInt(ty_i32, n, 0),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i32, 0, 0), next_index];
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

    pub(crate) unsafe fn emit_loop_reduce(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        init_value: llvm::prelude::LLVMValueRef,
        n: u64,
        predicate: impl Fn(
            &mut IREmitter,
            llvm::prelude::LLVMBasicBlockRef,
            llvm::prelude::LLVMValueRef,
            llvm::prelude::LLVMValueRef,
        ) -> (
            llvm::prelude::LLVMBasicBlockRef,
            llvm::prelude::LLVMValueRef,
        ),
    ) -> (
        llvm::prelude::LLVMBasicBlockRef,
        llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let function = self.function;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_body =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        let bb_loop_exit =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());

        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let value = llvm::core::LLVMBuildPhi(
            builder,
            llvm::core::LLVMTypeOf(init_value),
            empty_name.as_ptr(),
        );
        let index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());

        llvm::core::LLVMBuildBr(builder, bb_loop_body);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let (bb_loop_body, next_value) = predicate(self, bb_loop_body, value, index);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let next_index = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            empty_name.as_ptr(),
        );

        let cmp = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            next_index,
            llvm::core::LLVMConstInt(ty_i32, n, 0),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i32, 0, 0), next_index];
        let mut incoming_blocks = vec![bb, bb_loop_body];
        llvm::core::LLVMAddIncoming(
            index,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        let mut incoming_value = vec![init_value, next_value];
        let mut incoming_blocks = vec![bb, bb_loop_body];
        llvm::core::LLVMAddIncoming(
            value,
            incoming_value.as_mut_ptr(),
            incoming_blocks.as_mut_ptr(),
            incoming_value.len() as u32,
        );

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_exit);

        (bb_loop_exit, next_value)
    }

    pub(crate) unsafe fn emit_vop(
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
        let empty_name = std::ffi::CString::new("").unwrap();

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());
        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
        let exec = self.emit_exec_bit(index);

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
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let bb_loop_body = predicate(self, bb_loop_body, index);

        let next_index2 = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

        let next_index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
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
            llvm::core::LLVMConstInt(ty_i32, 32, 0),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i32, 0, 0), next_index];
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

    pub(crate) unsafe fn emit_vop_update_sgpr(
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
        let empty_name = std::ffi::CString::new("").unwrap();

        let bb_loop_entry =
            llvm::core::LLVMAppendBasicBlockInContext(context, function, empty_name.as_ptr());
        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

        let index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
        let vcc = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
        let exec = self.emit_exec_bit(index);

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
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_body);

        let (bb_loop_body, cc) = predicate(self, bb_loop_body, index);

        let cc = llvm::core::LLVMBuildZExt(builder, cc, ty_i32, empty_name.as_ptr());
        let cc_shifted = llvm::core::LLVMBuildShl(builder, cc, index, empty_name.as_ptr());

        let update_vcc = llvm::core::LLVMBuildOr(builder, cc_shifted, vcc, empty_name.as_ptr());

        let next_index2 = llvm::core::LLVMBuildAdd(
            builder,
            index,
            llvm::core::LLVMConstInt(ty_i32, 1, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

        let next_index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
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
            llvm::core::LLVMConstInt(ty_i32, 32, 0),
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

        let mut incoming_value = vec![llvm::core::LLVMConstInt(ty_i32, 0, 0), next_index];
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

    pub(crate) unsafe fn emit_alloc_registers(&mut self, reg_usage: &RegisterUsage) {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        if USE_SGPR_CACHE {
            let sgprs: Vec<u32> = reg_usage
                .use_sgprs
                .union(&reg_usage.def_sgprs)
                .cloned()
                .collect::<Vec<_>>();

            for sgpr in &sgprs {
                let sgpr_ptr = llvm::core::LLVMBuildAlloca(
                    self.builder,
                    ty_i32,
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
                        llvm::core::LLVMVectorType(ty_i32, SIMD_WIDTH as u32),
                        std::ffi::CString::new(format!("vgpr{}.{}", vgpr, i))
                            .unwrap()
                            .as_ptr(),
                    );
                }
                self.vgpr_ptr_map.insert(*vgpr, vgpr_ptr);
            }
        }

        self.ray.emit_alloc(self.context, self.builder);

        self.matrix.emit_alloc(self.context, self.builder);

        if self.use_scc_cache {
            let context = self.context;
            let builder = self.builder;
            let empty_name = std::ffi::CString::new("").unwrap();
            let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);

            self.local_scc_ptr = llvm::core::LLVMBuildAlloca(builder, ty_i8, empty_name.as_ptr());
        }
    }

    pub(crate) unsafe fn emit_restore_stack(
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

                let mut indices = vec![llvm::core::LLVMConstInt(ty_i32, *sgpr as u64, 0)];
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
                            ty_i32,
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
            let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);

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

    pub(crate) unsafe fn emit_restore_registers(
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

    pub(crate) unsafe fn emit_save_stack(
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
                let mut indices = vec![llvm::core::LLVMConstInt(ty_i32, *sgpr as u64, 0)];
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

        if self.use_vgpr_cache {
            for vgpr in &reg_usage.def_vgprs {
                let context = self.context;
                let builder = self.builder;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                const N: usize = SIMD_WIDTH;

                for i in (0..32).step_by(N) {
                    let mut indices: Vec<*mut llvm_sys::LLVMValue> =
                        vec![llvm::core::LLVMConstInt(
                            ty_i32,
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
                        value_ptr,
                        4,
                        self.vgpr_ptr_map.get(vgpr).unwrap()[i / N],
                        4,
                        llvm::core::LLVMConstInt(ty_i32, 4 * N as u64, 0),
                    );
                }
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_save_registers(
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

    pub(crate) unsafe fn emit_terminator(
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
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let exec_value = self.emit_load_sgpr_u32(126);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntEQ,
                        exec_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
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
                I::S_CBRANCH_EXECNZ => {
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let exec_value = self.emit_load_sgpr_u32(126);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        exec_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
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
                I::S_CBRANCH_VCCZ => {
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let vcc_value = self.emit_load_sgpr_u32(106);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntEQ,
                        vcc_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
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
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let vcc_value = self.emit_load_sgpr_u32(106);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let cmp_value = llvm::core::LLVMBuildICmp(
                        builder,
                        llvm::LLVMIntPredicate::LLVMIntNE,
                        vcc_value,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
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
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    self.ret_value =
                        llvm::core::LLVMConstInt(ty_i32, Signals::EndOfProgram as u64, 0);
                }
                I::S_BARRIER_WAIT => {
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                    let next_pc_value = llvm::core::LLVMConstInt(ty_i64, (pc as i64 + 4) as u64, 0);

                    llvm::core::LLVMBuildStore(builder, next_pc_value, self.pc_ptr);

                    self.ret_value = llvm::core::LLVMConstInt(ty_i32, Signals::Switch as u64, 0);
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

    pub(crate) unsafe fn emit_instruction(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &InstFormat,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        match inst {
            InstFormat::SOPP(inst) => self.emit_sopp(bb, inst),
            InstFormat::VOPC(inst) => self.emit_vopc(bb, inst),
            InstFormat::VOP1(inst) => self.emit_vop1(bb, inst),
            InstFormat::VOP2(inst) => self.emit_vop2(bb, inst),
            InstFormat::VOP3(inst) => self.emit_vop3(bb, inst),
            InstFormat::VOP3SD(inst) => self.emit_vop3sd(bb, inst),
            InstFormat::VOP3P(inst) => self.emit_vop3p(bb, inst),
            InstFormat::VOPD(inst) => self.emit_vopd(bb, inst),
            InstFormat::SMEM(inst) => self.emit_smem(bb, inst),
            InstFormat::SOP1(inst) => self.emit_sop1(bb, inst),
            InstFormat::SOP2(inst) => self.emit_sop2(bb, inst),
            InstFormat::SOPC(inst) => self.emit_sopc(bb, inst),
            InstFormat::VFLAT(inst) => self.emit_vflat(bb, inst),
            InstFormat::VGLOBAL(inst) => self.emit_vglobal(bb, inst),
            InstFormat::VSCRATCH(inst) => self.emit_vscratch(bb, inst),
            InstFormat::DS(inst) => self.emit_ds(bb, inst),
            InstFormat::SOPK(inst) => self.emit_sopk(bb, inst),
            InstFormat::VIMAGE(inst) => self.emit_vimage(bb, inst),
            InstFormat::VSAMPLE(inst) => self.emit_vsample(bb, inst),
        }
    }
}
