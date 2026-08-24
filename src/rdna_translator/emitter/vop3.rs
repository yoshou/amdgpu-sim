use crate::rdna_translator::*;

use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_vop3(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOP3,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
            I::V_LSHLREV_B16 => {
                if USE_SIMD {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                    let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0xF, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

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

                        let d_value = llvm::core::LLVMBuildShl(
                            builder,
                            s1_value,
                            s0_value,
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
                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstInt(ty_i32, 0xF, 0),
                            empty_name.as_ptr(),
                        );
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
                        let d_value = llvm::core::LLVMBuildShl(
                            builder,
                            s1_value,
                            s0_value,
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s1_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s1_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
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
                                [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let mask_value = llvm::core::LLVMBuildShl(
                            builder,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 1, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            s2_value,
                            empty_name.as_ptr(),
                        );

                        let mask_value = llvm::core::LLVMBuildSub(
                            builder,
                            mask_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 1, 0); N].as_mut_ptr(),
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
                        let d_value =
                            llvm::core::LLVMBuildAnd(builder, shifted, mask, empty_name.as_ptr());

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s1_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s1_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
            I::V_AND_OR_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let anded = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value =
                            llvm::core::LLVMBuildOr(builder, anded, s2_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let anded = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value =
                            llvm::core::LLVMBuildOr(builder, anded, s2_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_ADD_LSHL_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let added = llvm::core::LLVMBuildAdd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let s2_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s2_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let d_value =
                            llvm::core::LLVMBuildShl(builder, added, s2_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let added = llvm::core::LLVMBuildAdd(
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

                        let d_value =
                            llvm::core::LLVMBuildShl(builder, added, s2_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_LSHL_ADD_U32 => {
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s1_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s1_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
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

                        let d_value = llvm::core::LLVMBuildAdd(
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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

                        let d_value = llvm::core::LLVMBuildAdd(
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
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            cond,
                            s1_value,
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s2_value = emitter.emit_scalar_source_operand_u32(&inst.src2);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

                        let elem_shifted = llvm::core::LLVMBuildShl(
                            emitter.builder,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            elem,
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
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildSelect(
                            emitter.builder,
                            cond,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CVT_F64_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = emitter.emit_u32_to_f64xn::<N>(s0_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildUIToFP(
                            builder,
                            s0_value,
                            ty_f64,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = s0_value;

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = s0_value;

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_NOT_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value =
                            llvm::core::LLVMBuildNot(builder, s0_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value =
                            llvm::core::LLVMBuildNot(builder, s0_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_TRUNC_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.trunc.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.trunc.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CEIL_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.ceil.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.ceil.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_EXP_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.exp2.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.exp2.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_LOG_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.log2.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.log2.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FLOOR_F64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_TRUNC_F64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.trunc.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.trunc.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_BFREV_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.bitreverse.", &[ty_i32xn]);
                        let d_value = intrinsic.emit_call(ty_i32xn, &[s0_value]);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.bitreverse.", &[ty_i32]);
                        let d_value = intrinsic.emit_call(ty_i32, &[s0_value]);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_RSQ_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f32xn]);
                        let sqrt_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let one = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(ty_f32, 1.0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            one,
                            sqrt_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f32]);
                        let sqrt_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstReal(ty_f32, 1.0),
                            sqrt_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_SIN_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        // The ISA operand is in revolutions: D = sin(S0 * 2 * PI).
                        // Subtracting the nearest integer is exact in binary and
                        // reduces the argument to [-0.5, 0.5] for any input.
                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.rint.", &[ty_f32xn]);
                        let rounded = intrinsic.emit_call(ty_f32xn, &[s0_value]);
                        let reduced = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            rounded,
                            empty_name.as_ptr(),
                        );

                        // A zero remainder keeps the sign of the operand, which
                        // sin() carries into its result.
                        let zero = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(ty_f32, 0.0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let is_zero = llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOEQ,
                            reduced,
                            zero,
                            empty_name.as_ptr(),
                        );
                        let signed_zero =
                            llvm::core::LLVMBuildFMul(builder, s0_value, zero, empty_name.as_ptr());
                        let reduced = llvm::core::LLVMBuildSelect(
                            builder,
                            is_zero,
                            signed_zero,
                            reduced,
                            empty_name.as_ptr(),
                        );

                        let reduced = llvm::core::LLVMBuildFPExt(
                            builder,
                            reduced,
                            ty_f64xn,
                            empty_name.as_ptr(),
                        );
                        let two_pi = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(ty_f64, std::f64::consts::TAU); N]
                                .as_mut_ptr(),
                            N as u32,
                        );
                        let theta = llvm::core::LLVMBuildFMul(
                            builder,
                            reduced,
                            two_pi,
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sin.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[theta]);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            d_value,
                            ty_f32xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.rint.", &[ty_f32]);
                        let rounded = intrinsic.emit_call(ty_f32, &[s0_value]);
                        let reduced = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            rounded,
                            empty_name.as_ptr(),
                        );

                        let zero = llvm::core::LLVMConstReal(ty_f32, 0.0);
                        let is_zero = llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOEQ,
                            reduced,
                            zero,
                            empty_name.as_ptr(),
                        );
                        let signed_zero =
                            llvm::core::LLVMBuildFMul(builder, s0_value, zero, empty_name.as_ptr());
                        let reduced = llvm::core::LLVMBuildSelect(
                            builder,
                            is_zero,
                            signed_zero,
                            reduced,
                            empty_name.as_ptr(),
                        );

                        let reduced = llvm::core::LLVMBuildFPExt(
                            builder,
                            reduced,
                            ty_f64,
                            empty_name.as_ptr(),
                        );
                        let theta = llvm::core::LLVMBuildFMul(
                            builder,
                            reduced,
                            llvm::core::LLVMConstReal(ty_f64, std::f64::consts::TAU),
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sin.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[theta]);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            d_value,
                            ty_f32,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_COS_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        // The ISA operand is in revolutions: D = cos(S0 * 2 * PI).
                        // Subtracting the nearest integer is exact in binary and
                        // reduces the argument to [-0.5, 0.5] for any input.
                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.rint.", &[ty_f32xn]);
                        let rounded = intrinsic.emit_call(ty_f32xn, &[s0_value]);
                        let reduced = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            rounded,
                            empty_name.as_ptr(),
                        );

                        // A zero remainder keeps the sign of the operand, which
                        // sin() carries into its result.
                        let zero = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(ty_f32, 0.0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let is_zero = llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOEQ,
                            reduced,
                            zero,
                            empty_name.as_ptr(),
                        );
                        let signed_zero =
                            llvm::core::LLVMBuildFMul(builder, s0_value, zero, empty_name.as_ptr());
                        let reduced = llvm::core::LLVMBuildSelect(
                            builder,
                            is_zero,
                            signed_zero,
                            reduced,
                            empty_name.as_ptr(),
                        );

                        let reduced = llvm::core::LLVMBuildFPExt(
                            builder,
                            reduced,
                            ty_f64xn,
                            empty_name.as_ptr(),
                        );
                        let two_pi = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(ty_f64, std::f64::consts::TAU); N]
                                .as_mut_ptr(),
                            N as u32,
                        );
                        let theta = llvm::core::LLVMBuildFMul(
                            builder,
                            reduced,
                            two_pi,
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.cos.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[theta]);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            d_value,
                            ty_f32xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.rint.", &[ty_f32]);
                        let rounded = intrinsic.emit_call(ty_f32, &[s0_value]);
                        let reduced = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            rounded,
                            empty_name.as_ptr(),
                        );

                        let zero = llvm::core::LLVMConstReal(ty_f32, 0.0);
                        let is_zero = llvm::core::LLVMBuildFCmp(
                            builder,
                            llvm::LLVMRealPredicate::LLVMRealOEQ,
                            reduced,
                            zero,
                            empty_name.as_ptr(),
                        );
                        let signed_zero =
                            llvm::core::LLVMBuildFMul(builder, s0_value, zero, empty_name.as_ptr());
                        let reduced = llvm::core::LLVMBuildSelect(
                            builder,
                            is_zero,
                            signed_zero,
                            reduced,
                            empty_name.as_ptr(),
                        );

                        let reduced = llvm::core::LLVMBuildFPExt(
                            builder,
                            reduced,
                            ty_f64,
                            empty_name.as_ptr(),
                        );
                        let theta = llvm::core::LLVMBuildFMul(
                            builder,
                            reduced,
                            llvm::core::LLVMConstReal(ty_f64, std::f64::consts::TAU),
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.cos.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[theta]);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            d_value,
                            ty_f32,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CVT_F16_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f16 = llvm::core::LLVMHalfTypeInContext(context);
                    let ty_f16xn = llvm::core::LLVMVectorType(ty_f16, N as u32);
                    let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                    let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            s0_value,
                            ty_f16xn,
                            empty_name.as_ptr(),
                        );

                        // The f16 result occupies the low half of the register.
                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            d_value,
                            ty_i16xn,
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
                        let ty_f16 = llvm::core::LLVMHalfTypeInContext(context);
                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            s0_value,
                            ty_f16,
                            empty_name.as_ptr(),
                        );
                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            d_value,
                            ty_i16,
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
            I::V_FREXP_EXP_I32_F64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = emitter._emit_exp_f64xn::<N>(s0_value);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = emitter._emit_exp_f64(s0_value);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_SQRT_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_RNDNE_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

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

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstReal(ty_f64, 1.0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstReal(
                                llvm::core::LLVMDoubleTypeInContext(context),
                                1.0,
                            ),
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            // Not a native RDNA single-instruction; synthesized by the
            // instruction combine pass to collapse the rsq+Newton-Raphson
            // sqrt expansion into one correctly-rounded sqrt.
            I::V_SQRT_F64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f64xn]);
                        let sqrt_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);
                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstReal(ty_f64, 1.0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            sqrt_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f64]);
                        let sqrt_value = intrinsic.emit_call(ty_f64, &[s0_value]);
                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstReal(ty_f64, 1.0),
                            sqrt_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

                        let d_value = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            d_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_fract_below_one(d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

                        let d_value = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            d_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_fract_below_one(d_value);
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

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = emitter.emit_fp_to_si_sat(s0_value, ty_i32xn);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = emitter.emit_fp_to_si_sat(s0_value, ty_i32);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildSIToFP(
                            builder,
                            s0_value,
                            ty_f64xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildSIToFP(
                            builder,
                            s0_value,
                            llvm::core::LLVMDoubleTypeInContext(context),
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = emitter.emit_fp_to_ui_sat(s0_value, ty_i32xn);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = emitter.emit_fp_to_ui_sat(s0_value, ty_i32);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CVT_F32_I32 => {
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildSIToFP(
                            builder,
                            s0_value,
                            ty_f32xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildSIToFP(
                            builder,
                            s0_value,
                            ty_f32,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CVT_I32_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = emitter.emit_fp_to_si_sat(s0_value, ty_i32xn);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = emitter.emit_fp_to_si_sat(s0_value, ty_i32);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

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

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstReal(ty_f32, 1.0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstReal(ty_f32, 1.0),
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FREXP_EXP_I32_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_raw, inst.abs, inst.neg, 0);

                        let d_value = emitter.emit_exp_f32xn::<N>(s0_value);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_raw, 0);

                        let d_value = emitter.emit_exp_f32(s0_value);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_MIN_NUM_F64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_raw, inst.abs, inst.neg, 0);
                        let s1_raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);
                        let s1_value =
                            emitter.emit_abs_neg_f64xn::<N>(s1_raw, inst.abs, inst.neg, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.minnum.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value, s1_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s0_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_raw, 0);
                        let s1_raw = emitter.emit_vector_source_operand_f64(&inst.src1, elem);
                        let s1_value =
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_raw, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.minnum.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value, s1_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }            I::V_CVT_F32_F64 => {
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
                            {
                            let raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                            emitter.emit_abs_neg_f64xn::<N>(raw, inst.abs, inst.neg, 0)
                        };

                        let d_value =
                            llvm::core::LLVMBuildFPTrunc(builder, s0_value, ty_f32xn, empty_name.as_ptr());

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = {
                            let raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, raw, 0)
                        };

                        let d_value =
                            llvm::core::LLVMBuildFPTrunc(builder, s0_value, ty_f32, empty_name.as_ptr());

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }            I::V_CVT_U32_F64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            {
                            let raw = emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                            emitter.emit_abs_neg_f64xn::<N>(raw, inst.abs, inst.neg, 0)
                        };

                        let d_value = emitter.emit_fp_to_ui_sat(s0_value, ty_i32xn);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = {
                            let raw = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, raw, 0)
                        };

                        let d_value = emitter.emit_fp_to_ui_sat(s0_value, ty_i32);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CLZ_I32_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.ctlz.", &[ty_i32xn]);
                        let d_value = intrinsic.emit_call(
                            ty_i32xn,
                            &[s0_value, llvm::core::LLVMConstInt(ty_i1, 0, 0)],
                        );

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            ),
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, -1i64 as u64, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            d_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.ctlz.", &[ty_i32]);
                        let d_value = intrinsic
                            .emit_call(ty_i32, &[s0_value, llvm::core::LLVMConstInt(ty_i1, 0, 0)]);

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

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CMP_GT_U32
            | I::V_CMP_LT_U32
            | I::V_CMP_GE_U32
            | I::V_CMP_LE_U32
            | I::V_CMP_EQ_U32
            | I::V_CMP_NE_U32
            | I::V_CMP_GT_I32
            | I::V_CMP_LT_I32
            | I::V_CMPX_EQ_U32
            | I::V_CMPX_GE_U32
            | I::V_CMPX_LE_U32
            | I::V_CMPX_LT_U32
            | I::V_CMPX_GT_I32
            | I::V_CMPX_LT_I32 => {
                let pred = match inst.op {
                    I::V_CMP_EQ_U32 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMP_NE_U32 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMP_LT_U32 => llvm::LLVMIntPredicate::LLVMIntULT,
                    I::V_CMP_LE_U32 => llvm::LLVMIntPredicate::LLVMIntULE,
                    I::V_CMP_GT_U32 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    I::V_CMP_GE_U32 => llvm::LLVMIntPredicate::LLVMIntUGE,
                    I::V_CMP_LT_I32 => llvm::LLVMIntPredicate::LLVMIntSLT,
                    I::V_CMP_GT_I32 => llvm::LLVMIntPredicate::LLVMIntSGT,
                    I::V_CMPX_EQ_U32 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMPX_LT_U32 => llvm::LLVMIntPredicate::LLVMIntULT,
                    I::V_CMPX_LE_U32 => llvm::LLVMIntPredicate::LLVMIntULE,
                    I::V_CMPX_GE_U32 => llvm::LLVMIntPredicate::LLVMIntUGE,
                    I::V_CMPX_LT_I32 => llvm::LLVMIntPredicate::LLVMIntSLT,
                    I::V_CMPX_GT_I32 => llvm::LLVMIntPredicate::LLVMIntSGT,
                    _ => unreachable!(),
                };
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
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

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    // The CMPX forms write EXEC rather than the usual destination.
                    let destination = match inst.op {
                        I::V_CMPX_EQ_U32 => 126,
                        I::V_CMPX_GE_U32 => 126,
                        I::V_CMPX_LE_U32 => 126,
                        I::V_CMPX_LT_U32 => 126,
                        I::V_CMPX_GT_I32 => 126,
                        I::V_CMPX_LT_I32 => 126,
                        _ => inst.vdst as u32,
                    };
                    emitter.emit_store_sgpr_u32(destination, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        (bb, cmp_value)
                    });
                }
            }
            I::V_CMPX_NE_U32 | I::V_CMPX_GT_U32 => {
                let pred = match inst.op {
                    I::V_CMPX_NE_U32 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMPX_GT_U32 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    _ => unreachable!(),
                };
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
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

                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, d_value, exec_value, empty_name.as_ptr());

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    emitter.emit_store_sgpr_u32(126, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        (bb, cmp_value)
                    });
                }
            }
            I::V_CMP_EQ_U16 | I::V_CMP_GT_U16 => {
                let pred = match inst.op {
                    I::V_CMP_EQ_U16 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMP_GT_U16 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    _ => unreachable!(),
                };
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                    let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                    let mut cmp_values = Vec::new();

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            s0_value,
                            ty_i16xn,
                            empty_name.as_ptr(),
                        );
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            s1_value,
                            ty_i16xn,
                            empty_name.as_ptr(),
                        );
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
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

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            s0_value,
                            ty_i16,
                            empty_name.as_ptr(),
                        );
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            s1_value,
                            ty_i16,
                            empty_name.as_ptr(),
                        );
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
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
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u64xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0x3F, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let s0_value = llvm::core::LLVMBuildZExt(
                            builder,
                            s0_value,
                            ty_i64xn,
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u64(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstInt(ty_i32, 0x3F, 0),
                            empty_name.as_ptr(),
                        );
                        let s0_value = llvm::core::LLVMBuildZExt(
                            builder,
                            s0_value,
                            ty_i64,
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
            I::V_LSHRREV_B64 => {
                if USE_SIMD {
                    let emitter = self;
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u64xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0x3F, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let s0_value = llvm::core::LLVMBuildZExt(
                            builder,
                            s0_value,
                            ty_i64xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildLShr(
                            builder,
                            s1_value,
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u64(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstInt(ty_i32, 0x3F, 0),
                            empty_name.as_ptr(),
                        );
                        let s0_value = llvm::core::LLVMBuildZExt(
                            builder,
                            s0_value,
                            ty_i64,
                            empty_name.as_ptr(),
                        );
                        let d_value = llvm::core::LLVMBuildLShr(
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
            I::V_CMP_GE_F32
            | I::V_CMP_LT_F32
            | I::V_CMP_GT_F32
            | I::V_CMP_LE_F32
            | I::V_CMP_LG_F32
            | I::V_CMP_NGT_F32
            | I::V_CMP_NGE_F32
            | I::V_CMP_NLE_F32
            | I::V_CMP_NLT_F32
            | I::V_CMP_O_F32
            | I::V_CMP_EQ_F32
            | I::V_CMP_NEQ_F32
            | I::V_CMPX_EQ_F32
            | I::V_CMPX_GE_F32
            | I::V_CMPX_GT_F32
            | I::V_CMPX_LE_F32
            | I::V_CMPX_LG_F32
            | I::V_CMPX_LT_F32
            | I::V_CMPX_NEQ_F32
            | I::V_CMPX_NGE_F32
            | I::V_CMPX_NLE_F32
            | I::V_CMPX_NLT_F32 => {
                let pred = match inst.op {
                    I::V_CMP_GE_F32 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMP_LT_F32 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMP_GT_F32 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_LE_F32 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMP_LG_F32 => llvm::LLVMRealPredicate::LLVMRealONE,
                    I::V_CMP_O_F32 => llvm::LLVMRealPredicate::LLVMRealORD,
                    // The N-prefixed compares negate the ordered compare, so
                    // unordered operands compare true.
                    I::V_CMP_NGT_F32 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_NGE_F32 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMP_NLE_F32 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMP_NLT_F32 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMP_EQ_F32 | I::V_CMP_NEQ_F32 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMPX_EQ_F32 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMPX_GE_F32 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMPX_GT_F32 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMPX_LE_F32 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMPX_LG_F32 => llvm::LLVMRealPredicate::LLVMRealONE,
                    I::V_CMPX_LT_F32 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMPX_NEQ_F32 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMPX_NGE_F32 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMPX_NLE_F32 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMPX_NLT_F32 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    _ => unreachable!(),
                };
                let not = matches!(
                    inst.op,
                    I::V_CMP_NGT_F32
                        | I::V_CMP_NGE_F32
                        | I::V_CMP_NLE_F32
                        | I::V_CMP_NLT_F32
                        | I::V_CMP_NEQ_F32
                        | I::V_CMPX_NEQ_F32
                        | I::V_CMPX_NGE_F32
                        | I::V_CMPX_NLE_F32
                        | I::V_CMPX_NLT_F32
                );
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
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildFCmp(
                            builder,
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let cmp_value = if not {
                            llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr())
                        } else {
                            cmp_value
                        };

                        cmp_values.push(cmp_value);
                    }

                    let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                    let d_value = llvm::core::LLVMBuildBitCast(
                        builder,
                        cmp_value,
                        ty_i32,
                        empty_name.as_ptr(),
                    );

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    // The CMPX forms write EXEC rather than the usual destination.
                    let destination = match inst.op {
                        I::V_CMPX_EQ_F32 => 126,
                        I::V_CMPX_GE_F32 => 126,
                        I::V_CMPX_GT_F32 => 126,
                        I::V_CMPX_LE_F32 => 126,
                        I::V_CMPX_LG_F32 => 126,
                        I::V_CMPX_LT_F32 => 126,
                        I::V_CMPX_NEQ_F32 => 126,
                        I::V_CMPX_NGE_F32 => 126,
                        I::V_CMPX_NLE_F32 => 126,
                        I::V_CMPX_NLT_F32 => 126,
                        _ => inst.vdst as u32,
                    };
                    emitter.emit_store_sgpr_u32(destination, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

                        let cmp_value = llvm::core::LLVMBuildFCmp(
                            builder,
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let cmp_value = if not {
                            llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr())
                        } else {
                            cmp_value
                        };

                        (bb, cmp_value)
                    });
                }
            }
            I::V_CMP_CLASS_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let context = emitter.context;

                    const N: usize = SIMD_WIDTH;

                    let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                    let ty_i1xn = llvm::core::LLVMVectorType(ty_i1, N as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let mut agg_value = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                        N as u32,
                    );

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let mut cmp_value = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i1, 0, 0); N].as_mut_ptr(),
                            N as u32,
                        );

                        for cls in 0..10 {
                            let class_bit = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 1 << cls, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let no_bit = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let intrinsic =
                                emitter.get_intrinsic_declaration("llvm.is.fpclass.", &[ty_f32xn]);
                            let class_value = intrinsic.emit_call(
                                ty_i1xn,
                                &[s0_value, llvm::core::LLVMConstInt(ty_i32, 1 << cls, 0)],
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
                                        class_bit,
                                        empty_name.as_ptr(),
                                    ),
                                    no_bit,
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

                    let intrinsic =
                        emitter.get_intrinsic_declaration("llvm.vector.reduce.or.", &[ty_i32xn]);
                    let d_value = intrinsic.emit_call(ty_i32, &[agg_value]);

                    emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.is.fpclass.", &[ty_f32]);
                        let class_value = intrinsic.emit_call(ty_i1, &[s0_value, s1_value]);

                        (bb, class_value)
                    });
                }
            }
            I::V_CMP_EQ_U64
            | I::V_CMP_NE_U64
            | I::V_CMP_LT_U64
            | I::V_CMP_LE_U64
            | I::V_CMP_GT_U64
            | I::V_CMP_GE_U64
            | I::V_CMP_EQ_I64
            | I::V_CMP_NE_I64
            | I::V_CMP_LT_I64
            | I::V_CMP_LE_I64
            | I::V_CMP_GT_I64
            | I::V_CMP_GE_I64
            | I::V_CMPX_EQ_U64
            | I::V_CMPX_NE_U64
            | I::V_CMPX_LT_U64
            | I::V_CMPX_LE_U64
            | I::V_CMPX_GT_U64
            | I::V_CMPX_GE_U64
            | I::V_CMPX_EQ_I64
            | I::V_CMPX_NE_I64
            | I::V_CMPX_LT_I64
            | I::V_CMPX_LE_I64
            | I::V_CMPX_GT_I64
            | I::V_CMPX_GE_I64 => {
                let pred = match inst.op {
                    I::V_CMP_EQ_U64 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMP_NE_U64 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMP_LT_U64 => llvm::LLVMIntPredicate::LLVMIntULT,
                    I::V_CMP_LE_U64 => llvm::LLVMIntPredicate::LLVMIntULE,
                    I::V_CMP_GT_U64 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    I::V_CMP_GE_U64 => llvm::LLVMIntPredicate::LLVMIntUGE,
                    I::V_CMP_EQ_I64 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMP_NE_I64 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMP_LT_I64 => llvm::LLVMIntPredicate::LLVMIntSLT,
                    I::V_CMP_LE_I64 => llvm::LLVMIntPredicate::LLVMIntSLE,
                    I::V_CMP_GT_I64 => llvm::LLVMIntPredicate::LLVMIntSGT,
                    I::V_CMP_GE_I64 => llvm::LLVMIntPredicate::LLVMIntSGE,
                    I::V_CMPX_EQ_U64 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMPX_NE_U64 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMPX_LT_U64 => llvm::LLVMIntPredicate::LLVMIntULT,
                    I::V_CMPX_LE_U64 => llvm::LLVMIntPredicate::LLVMIntULE,
                    I::V_CMPX_GT_U64 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    I::V_CMPX_GE_U64 => llvm::LLVMIntPredicate::LLVMIntUGE,
                    I::V_CMPX_EQ_I64 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMPX_NE_I64 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMPX_LT_I64 => llvm::LLVMIntPredicate::LLVMIntSLT,
                    I::V_CMPX_LE_I64 => llvm::LLVMIntPredicate::LLVMIntSLE,
                    I::V_CMPX_GT_I64 => llvm::LLVMIntPredicate::LLVMIntSGT,
                    I::V_CMPX_GE_I64 => llvm::LLVMIntPredicate::LLVMIntSGE,
                    _ => unreachable!(),
                };
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u64xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
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

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    // The CMPX forms write EXEC rather than the SGPR in VDST.
                    let destination = match inst.op {
                        I::V_CMPX_EQ_U64 => 126,
                        I::V_CMPX_NE_U64 => 126,
                        I::V_CMPX_LT_U64 => 126,
                        I::V_CMPX_LE_U64 => 126,
                        I::V_CMPX_GT_U64 => 126,
                        I::V_CMPX_GE_U64 => 126,
                        I::V_CMPX_EQ_I64 => 126,
                        I::V_CMPX_NE_I64 => 126,
                        I::V_CMPX_LT_I64 => 126,
                        I::V_CMPX_LE_I64 => 126,
                        I::V_CMPX_GT_I64 => 126,
                        I::V_CMPX_GE_I64 => 126,
                        _ => inst.vdst as u32,
                    };
                    emitter.emit_store_sgpr_u32(destination, d_value);
                } else {
                    let destination = match inst.op {
                        I::V_CMPX_EQ_U64 => 126,
                        I::V_CMPX_NE_U64 => 126,
                        I::V_CMPX_LT_U64 => 126,
                        I::V_CMPX_LE_U64 => 126,
                        I::V_CMPX_GT_U64 => 126,
                        I::V_CMPX_GE_U64 => 126,
                        I::V_CMPX_EQ_I64 => 126,
                        I::V_CMPX_NE_I64 => 126,
                        I::V_CMPX_LT_I64 => 126,
                        I::V_CMPX_LE_I64 => 126,
                        I::V_CMPX_GT_I64 => 126,
                        I::V_CMPX_GE_I64 => 126,
                        _ => inst.vdst as u32,
                    };
                    bb = self.emit_vop_update_sgpr(bb, destination, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u64(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u64(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        (bb, cmp_value)
                    });
                }
            }
            I::V_CMP_LT_F64
            | I::V_CMP_GT_F64
            | I::V_CMP_GE_F64
            | I::V_CMP_LG_F64
            | I::V_CMP_LE_F64
            | I::V_CMP_NLT_F64
            | I::V_CMP_NGT_F64
            | I::V_CMP_NGE_F64
            | I::V_CMP_NLE_F64
            | I::V_CMP_O_F64
            | I::V_CMP_NEQ_F64
            | I::V_CMP_EQ_F64
            | I::V_CMPX_EQ_F64
            | I::V_CMPX_GE_F64
            | I::V_CMPX_GT_F64
            | I::V_CMPX_LE_F64
            | I::V_CMPX_LG_F64
            | I::V_CMPX_LT_F64
            | I::V_CMPX_NEQ_F64
            | I::V_CMPX_NGE_F64
            | I::V_CMPX_NGT_F64
            | I::V_CMPX_NLE_F64
            | I::V_CMPX_NLT_F64 => {
                let pred = match inst.op {
                    I::V_CMP_EQ_F64 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMP_LT_F64 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMP_GT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_GE_F64 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMP_LG_F64 => llvm::LLVMRealPredicate::LLVMRealONE,
                    I::V_CMP_LE_F64 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMP_O_F64 => llvm::LLVMRealPredicate::LLVMRealORD,
                    I::V_CMP_NLT_F64 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMP_NGT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_NGE_F64 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMP_NLE_F64 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMP_NEQ_F64 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMPX_EQ_F64 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMPX_GE_F64 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMPX_GT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMPX_LE_F64 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMPX_LG_F64 => llvm::LLVMRealPredicate::LLVMRealONE,
                    I::V_CMPX_LT_F64 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMPX_NEQ_F64 => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    I::V_CMPX_NGE_F64 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMPX_NGT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMPX_NLE_F64 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMPX_NLT_F64 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    _ => unreachable!(),
                };
                let not = match inst.op {
                    I::V_CMP_NLT_F64
                    | I::V_CMP_NGT_F64
                    | I::V_CMP_NGE_F64
                    | I::V_CMP_NLE_F64
                    | I::V_CMP_NEQ_F64 => true,
                        I::V_CMPX_NEQ_F64 => true,
                        I::V_CMPX_NGE_F64 => true,
                        I::V_CMPX_NGT_F64 => true,
                        I::V_CMPX_NLE_F64 => true,
                        I::V_CMPX_NLT_F64 => true,
                    _ => false,
                };
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
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let cmp_value = if not {
                            llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr())
                        } else {
                            cmp_value
                        };

                        cmp_values.push(cmp_value);
                    }

                    let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                    let d_value = llvm::core::LLVMBuildBitCast(
                        builder,
                        cmp_value,
                        ty_i32,
                        empty_name.as_ptr(),
                    );

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    // The CMPX forms write EXEC rather than the usual destination.
                    let destination = match inst.op {
                        I::V_CMPX_EQ_F64 => 126,
                        I::V_CMPX_GE_F64 => 126,
                        I::V_CMPX_GT_F64 => 126,
                        I::V_CMPX_LE_F64 => 126,
                        I::V_CMPX_LG_F64 => 126,
                        I::V_CMPX_LT_F64 => 126,
                        I::V_CMPX_NEQ_F64 => 126,
                        I::V_CMPX_NGE_F64 => 126,
                        I::V_CMPX_NGT_F64 => 126,
                        I::V_CMPX_NLE_F64 => 126,
                        I::V_CMPX_NLT_F64 => 126,
                        _ => inst.vdst as u32,
                    };
                    emitter.emit_store_sgpr_u32(destination, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_value, 1);

                        let cmp_value = llvm::core::LLVMBuildFCmp(
                            builder,
                            pred,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let cmp_value = if not {
                            llvm::core::LLVMBuildNot(builder, cmp_value, empty_name.as_ptr())
                        } else {
                            cmp_value
                        };

                        (bb, cmp_value)
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
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let mut cmp_value = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i1, 0, 0); N].as_mut_ptr(),
                            N as u32,
                        );

                        for cls in 0..10 {
                            let class_bit = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 1 << cls, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let no_bit = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            let intrinsic =
                                emitter.get_intrinsic_declaration("llvm.is.fpclass.", &[ty_f64xn]);
                            let class_value = intrinsic.emit_call(
                                ty_i1xn,
                                &[s0_value, llvm::core::LLVMConstInt(ty_i32, 1 << cls, 0)],
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
                                        class_bit,
                                        empty_name.as_ptr(),
                                    ),
                                    no_bit,
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

                    let intrinsic =
                        emitter.get_intrinsic_declaration("llvm.vector.reduce.or.", &[ty_i32xn]);
                    let d_value = intrinsic.emit_call(ty_i32, &[agg_value]);

                    emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.vdst as u32, |emitter, bb, elem| {
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.is.fpclass.", &[ty_f64]);
                        let class_value = intrinsic.emit_call(ty_i1, &[s0_value, s1_value]);

                        (bb, class_value)
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

                        // A negated second operand makes this a subtract, and the
                        // sign of a NaN it propagates has to follow.
                        let d_value = if inst.neg & 2 != 0 {
                            emitter.emit_sub_nan_sign(d_value, s0_value, s1_value)
                        } else {
                            d_value
                        };

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_value, 1);

                        let d_value = llvm::core::LLVMBuildFAdd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let d_value = if inst.neg & 2 != 0 {
                            emitter.emit_sub_nan_sign(d_value, s0_value, s1_value)
                        } else {
                            d_value
                        };

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_value, 1);

                        let d_value = llvm::core::LLVMBuildFMul(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f64(&inst.src2, elem);

                        let s0_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_value, 1);
                        let s2_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s2_value, 2);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value, s1_value, s2_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                        let fma_result = emitter.emit_fma_f64xn::<N>(s0_value, s1_value, s2_value);

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

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f64(&inst.src2, elem);

                        let vcc_value = emitter.emit_load_sgpr_u32(106);

                        let elem_shifted = llvm::core::LLVMBuildShl(
                            emitter.builder,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            elem,
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
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f64]);
                        let fma_result =
                            intrinsic.emit_call(ty_f64, &[s0_value, s1_value, s2_value]);

                        let muled = llvm::core::LLVMBuildFMul(
                            builder,
                            fma_result,
                            llvm::core::LLVMConstReal(ty_f64, f64::from_bits(0x43F0000000000000)),
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            cond,
                            fma_result,
                            muled,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_DIV_FIXUP_F64 => {
                // S1 and S2 are the original denominator and numerator, so the
                // IEEE quotient can be computed here directly. This makes the
                // upstream div_scale/rcp/div_fmas refinement chain dead code,
                // which the instruction combine pass removes.
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s1_value =
                            emitter.emit_vector_source_operand_f64xn::<N>(&inst.src1, i, mask);

                        let s2_value =
                            emitter.emit_vector_source_operand_f64xn::<N>(&inst.src2, i, mask);

                        let s1_value =
                            emitter.emit_abs_neg_f64xn::<N>(s1_value, inst.abs, inst.neg, 1);
                        let s2_value =
                            emitter.emit_abs_neg_f64xn::<N>(s2_value, inst.abs, inst.neg, 2);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            s2_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f64(&inst.src2, elem);

                        let s1_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_value, 1);
                        let s2_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s2_value, 2);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            s2_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let d_value = emitter.emit_ldexp_f64xn::<N>(s0_value, s1_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.ldexp.", &[ty_f64, ty_i32]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value, s1_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
            I::V_OR3_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let d_value = llvm::core::LLVMBuildOr(
                            builder,
                            llvm::core::LLVMBuildOr(
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let d_value = llvm::core::LLVMBuildOr(
                            builder,
                            llvm::core::LLVMBuildOr(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            ),
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

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
            I::V_ADD_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

                        let d_value = emitter.emit_fadd(s0_value, s1_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

                        let d_value = llvm::core::LLVMBuildFAdd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FMAC_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

                        let d_old = emitter.emit_load_vgpr_f32xn::<N>(inst.vdst as u32, i, mask);

                        let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, d_old);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

                        let d_old = emitter.emit_load_vgpr_f32(inst.vdst as u32, elem);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value, d_old]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_RCP_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_raw = emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_ftz_f32(s0_raw);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstReal(ty_f32, 1.0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_raw = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value = emitter.emit_ftz_f32(s0_raw);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);

                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstReal(ty_f32, 1.0),
                            s0_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        let d_value = emitter.emit_ftz_f32(d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FLOOR_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_SUB_NC_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSub(
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSub(
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
            I::V_MUL_HI_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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

                        let prod = llvm::core::LLVMBuildMul(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let shift = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i64, 32, 0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let hi =
                            llvm::core::LLVMBuildLShr(builder, prod, shift, empty_name.as_ptr());

                        let d_value =
                            llvm::core::LLVMBuildTrunc(builder, hi, ty_i32xn, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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

                        let prod = llvm::core::LLVMBuildMul(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let hi = llvm::core::LLVMBuildLShr(
                            builder,
                            prod,
                            llvm::core::LLVMConstInt(ty_i64, 32, 0),
                            empty_name.as_ptr(),
                        );

                        let d_value =
                            llvm::core::LLVMBuildTrunc(builder, hi, ty_i32, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_S_RCP_F32 => {
                let emitter = self;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.src0);
                let s0_value =
                    llvm::core::LLVMBuildBitCast(builder, s0_value, ty_f32, empty_name.as_ptr());
                let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);

                let d_value = llvm::core::LLVMBuildFDiv(
                    builder,
                    llvm::core::LLVMConstReal(ty_f32, 1.0),
                    s0_value,
                    empty_name.as_ptr(),
                );

                emitter.emit_store_sgpr_f32(inst.vdst as u32, d_value);
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

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

                    let two_over_pi_fraction_value = llvm::core::LLVMConstIntOfArbitraryPrecision(
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
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value =
                            emitter.emit_abs_neg_f64xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let bb_loop_entry = llvm::core::LLVMAppendBasicBlockInContext(
                            context,
                            function,
                            empty_name.as_ptr(),
                        );
                        llvm::core::LLVMBuildBr(builder, bb_loop_entry);

                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_entry);

                        let index = llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
                        let d_value =
                            llvm::core::LLVMBuildPhi(builder, ty_f64xn, empty_name.as_ptr());

                        let exec = llvm::core::LLVMBuildExtractElement(
                            builder,
                            mask,
                            index,
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

                        let update_d_value = {
                            let s0_value = llvm::core::LLVMBuildExtractElement(
                                builder,
                                s0_value,
                                index,
                                empty_name.as_ptr(),
                            );
                            let s1_value = llvm::core::LLVMBuildExtractElement(
                                builder,
                                s1_value,
                                index,
                                empty_name.as_ptr(),
                            );

                            let s0_exp_value = emitter.emit_exponent_f64(s0_value);

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
                                ty_i64,
                                empty_name.as_ptr(),
                            );

                            let result = llvm::core::LLVMBuildAnd(
                                builder,
                                trunc,
                                llvm::core::LLVMConstInt(ty_i64, (1u64 << 53) - 1, 0),
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
                                llvm::core::LLVMConstInt(ty_i32, -53i64 as u64, 0),
                                shift,
                                empty_name.as_ptr(),
                            );

                            let cmp = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntSGE,
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

                            let result = emitter.emit_ldexp_f32(result, scale);

                            llvm::core::LLVMBuildInsertElement(
                                builder,
                                d_value,
                                result,
                                index,
                                empty_name.as_ptr(),
                            )
                        };

                        let next_index2 = llvm::core::LLVMBuildAdd(
                            builder,
                            index,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            empty_name.as_ptr(),
                        );

                        llvm::core::LLVMBuildBr(builder, bb_loop_cond);

                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_loop_cond);

                        let next_index =
                            llvm::core::LLVMBuildPhi(builder, ty_i32, empty_name.as_ptr());
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
                            llvm::core::LLVMConstInt(ty_i32, N as u64, 0),
                            empty_name.as_ptr(),
                        );
                        llvm::core::LLVMBuildCondBr(builder, cmp, bb_loop_exit, bb_loop_entry);

                        let mut incoming_value =
                            vec![llvm::core::LLVMConstInt(ty_i32, 0, 0), next_index];
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

                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, next_d_value, mask);

                        bb = bb_loop_exit
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_i1201 = llvm::core::LLVMIntTypeInContext(context, 1201);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_value, 0);

                        let s0_exp_value = emitter.emit_exponent_f64(s0_value);

                        let s1_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s1_value,
                            llvm::core::LLVMConstInt(ty_i32, 0x1F, 0),
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
                            ty_i64,
                            empty_name.as_ptr(),
                        );

                        let result = llvm::core::LLVMBuildAnd(
                            builder,
                            trunc,
                            llvm::core::LLVMConstInt(ty_i64, (1u64 << 53) - 1, 0),
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
                            llvm::core::LLVMConstInt(ty_i32, -53i64 as u64, 0),
                            shift,
                            empty_name.as_ptr(),
                        );

                        let cmp = llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntSGE,
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

                        let d_value = emitter.emit_ldexp_f32(result, scale);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
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

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value, s1_value]);
                        let d_value =
                            emitter.emit_omod_clamp_f64xn::<N>(inst.omod, inst.cm, d_value, 0);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f64(inst.abs, inst.neg, s1_value, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value, s1_value]);
                        let d_value = emitter.emit_omod_clamp(inst.omod, inst.cm, d_value, 0);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_WRITELANE_B32 => {
                let emitter = self;
                let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                const N: usize = SIMD_WIDTH;

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.src0);
                let s1_value = emitter.emit_scalar_source_operand_u32(&inst.src1);

                let s1_value = llvm::core::LLVMBuildAnd(
                    builder,
                    s1_value,
                    llvm::core::LLVMConstInt(ty_i32, 0x1F, 0),
                    empty_name.as_ptr(),
                );

                if USE_SIMD {
                    let mut values = Vec::new();
                    for i in (0..32).step_by(N) {
                        let value = emitter.emit_load_vgpr_u32xn::<N>(
                            inst.vdst as u32,
                            i as u32,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i1, 1, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                        );
                        values.push(value);
                    }
                    let value = emitter.emit_concat::<N>(&values);
                    let value = llvm::core::LLVMBuildInsertElement(
                        builder,
                        value,
                        s0_value,
                        s1_value,
                        empty_name.as_ptr(),
                    );
                    let values = emitter.emit_split::<N>(value);
                    for i in (0..32).step_by(N) {
                        emitter.emit_store_vgpr_u32xn::<N>(
                            inst.vdst as u32,
                            i as u32,
                            values[i / N],
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i1, 1, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                        );
                    }
                } else {
                    emitter.emit_store_vgpr_u32(inst.vdst as u32, s1_value, s0_value);
                };
            }
            I::V_READLANE_B32 => {
                let emitter = self;
                let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                const N: usize = SIMD_WIDTH;

                let s1_value = emitter.emit_scalar_source_operand_u32(&inst.src1);

                let s1_value = llvm::core::LLVMBuildAnd(
                    builder,
                    s1_value,
                    llvm::core::LLVMConstInt(ty_i32, 0x1F, 0),
                    empty_name.as_ptr(),
                );

                let s0_value = if USE_SIMD {
                    let mut values = Vec::new();
                    for i in (0..32).step_by(N) {
                        let value = emitter.emit_vector_source_operand_u32xn::<N>(
                            &inst.src0,
                            i as u32,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i1, 1, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                        );
                        values.push(value);
                    }
                    let value = emitter.emit_concat::<N>(&values);
                    llvm::core::LLVMBuildExtractElement(
                        builder,
                        value,
                        s1_value,
                        empty_name.as_ptr(),
                    )
                } else {
                    emitter.emit_vector_source_operand_u32(&inst.src0, s1_value)
                };

                emitter.emit_store_sgpr_u32(inst.vdst as u32, s0_value);
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildUIToFP(
                            builder,
                            s0_value,
                            ty_f32xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildUIToFP(
                            builder,
                            s0_value,
                            ty_f32,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CVT_F32_F16 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        // RDNA4 ISA: OPSEL[0] selects src0's f16 half (1=high, 0=low).
                        let s0_value = if inst.opsel & 1 != 0 {
                            emitter.emit_vector_source_operand_hi_f16xn::<N>(&inst.src0, i, mask)
                        } else {
                            emitter.emit_vector_source_operand_f16xn::<N>(&inst.src0, i, mask)
                        };

                        let d_value = llvm::core::LLVMBuildFPExt(
                            builder,
                            s0_value,
                            ty_f32xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = if inst.opsel & 1 != 0 {
                            emitter.emit_vector_source_operand_hi_f16(&inst.src0, elem)
                        } else {
                            emitter.emit_vector_source_operand_f16(&inst.src0, elem)
                        };

                        let d_value = llvm::core::LLVMBuildFPExt(
                            builder,
                            s0_value,
                            ty_f32,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

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
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

                        let d_value = emitter.emit_fmul(s0_value, s1_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

                        let d_value = llvm::core::LLVMBuildFMul(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FMA_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s2_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src2, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);
                        let s2_value =
                            emitter.emit_abs_neg_f32xn::<N>(s2_value, inst.abs, inst.neg, 2);

                        let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, s2_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f32(&inst.src2, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);
                        let s2_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s2_value, 2);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value, s2_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CMPX_NGT_F32 => {
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
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

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

                    // Lanes that were inactive stay inactive.
                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, d_value, exec_value, empty_name.as_ptr());

                    let d_value = emitter.emit_mask_with_exec(d_value, exec_value);
                    emitter.emit_store_sgpr_u32(126, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

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
            I::V_MAX_NUM_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value, s1_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s1_value, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_CVT_F64_F32 => {
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
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let d_value = llvm::core::LLVMBuildFPExt(
                            builder,
                            s0_value,
                            ty_f64xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_f32(inst.abs, inst.neg, s0_value, 0);

                        let d_value = llvm::core::LLVMBuildFPExt(
                            builder,
                            s0_value,
                            ty_f64,
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FREXP_MANT_F32 | I::V_FREXP_MANT_F64 => {
                let is_double = matches!(inst.op, I::V_FREXP_MANT_F64);
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let ty = if is_double { ty_f64xn } else { ty_f32xn };
                        let s0_value = if is_double {
                            let value =
                                emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);
                            emitter.emit_abs_neg_f64xn::<N>(value, inst.abs, inst.neg, 0)
                        } else {
                            let value =
                                emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);
                            emitter.emit_abs_neg_f32xn::<N>(value, inst.abs, inst.neg, 0)
                        };

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.frexp.", &[ty, ty_i32xn]);
                        let mut return_tys = vec![ty, ty_i32xn];
                        let frexp_value = intrinsic.emit_call(
                            llvm::core::LLVMStructTypeInContext(
                                context,
                                return_tys.as_mut_ptr(),
                                return_tys.len() as u32,
                                0,
                            ),
                            &[s0_value],
                        );
                        let d_value = llvm::core::LLVMBuildExtractValue(
                            builder,
                            frexp_value,
                            0,
                            empty_name.as_ptr(),
                        );

                        if is_double {
                            let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                            emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                        } else {
                            let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                            emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                        }
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let ty = if is_double { ty_f64 } else { ty_f32 };
                        let s0_value = if is_double {
                            let value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                            emitter.emit_abs_neg_f64(inst.abs, inst.neg, value, 0)
                        } else {
                            let value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                            emitter.emit_abs_neg_f32(inst.abs, inst.neg, value, 0)
                        };

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.frexp.", &[ty, ty_i32]);
                        let mut return_tys = vec![ty, ty_i32];
                        let frexp_value = intrinsic.emit_call(
                            llvm::core::LLVMStructTypeInContext(
                                context,
                                return_tys.as_mut_ptr(),
                                return_tys.len() as u32,
                                0,
                            ),
                            &[s0_value],
                        );
                        let d_value = llvm::core::LLVMBuildExtractValue(
                            builder,
                            frexp_value,
                            0,
                            empty_name.as_ptr(),
                        );

                        if is_double {
                            let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                            emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);
                        } else {
                            let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                            emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);
                        }

                        bb
                    });
                }
            }
            I::V_BFI_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);
                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        // D = (S0 & S1) | (~S0 & S2)
                        let masked = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let inverted =
                            llvm::core::LLVMBuildNot(builder, s0_value, empty_name.as_ptr());
                        let kept = llvm::core::LLVMBuildAnd(
                            builder,
                            inverted,
                            s2_value,
                            empty_name.as_ptr(),
                        );
                        let d_value =
                            llvm::core::LLVMBuildOr(builder, masked, kept, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);
                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let masked = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );
                        let inverted =
                            llvm::core::LLVMBuildNot(builder, s0_value, empty_name.as_ptr());
                        let kept = llvm::core::LLVMBuildAnd(
                            builder,
                            inverted,
                            s2_value,
                            empty_name.as_ptr(),
                        );
                        let d_value =
                            llvm::core::LLVMBuildOr(builder, masked, kept, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_BCNT_U32_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        // D = countbits(S0) + S1
                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.ctpop.", &[ty_i32xn]);
                        let count = intrinsic.emit_call(ty_i32xn, &[s0_value]);

                        let d_value =
                            llvm::core::LLVMBuildAdd(builder, count, s1_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.ctpop.", &[ty_i32]);
                        let count = intrinsic.emit_call(ty_i32, &[s0_value]);

                        let d_value =
                            llvm::core::LLVMBuildAdd(builder, count, s1_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_DIV_FMAS_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);
                    let vcc_value = emitter.emit_load_sgpr_u32(106);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                        let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s2_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src2, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);
                        let s2_value =
                            emitter.emit_abs_neg_f32xn::<N>(s2_value, inst.abs, inst.neg, 2);

                        let d_value = emitter.emit_div_fmas_f32(s0_value, s1_value, s2_value, cond);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f32(&inst.src2, elem);

                        let vcc_value = emitter.emit_load_sgpr_u32(106);

                        let elem_shifted = llvm::core::LLVMBuildShl(
                            emitter.builder,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            elem,
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
                            llvm::LLVMIntPredicate::LLVMIntNE,
                            elem_masked,
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        );

                        let d_value = emitter.emit_div_fmas_f32(s0_value, s1_value, s2_value, cond);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_DIV_FIXUP_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s2_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src2, i, mask);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.abs, inst.neg, 1);
                        let s2_value =
                            emitter.emit_abs_neg_f32xn::<N>(s2_value, inst.abs, inst.neg, 2);

                        let d_value = emitter.emit_div_fixup_f32(s0_value, s1_value, s2_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f32(&inst.src2, elem);

                        let d_value = emitter.emit_div_fixup_f32(s0_value, s1_value, s2_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_LDEXP_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = llvm::core::LLVMBuildSIToFP(
                            builder,
                            s1_value,
                            ty_f32xn,
                            empty_name.as_ptr(),
                        );

                        let s1_value = emitter.emit_exp2_f32xn::<N>(s1_value);
                        let d_value = emitter.emit_fmul(s0_value, s1_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.ldexp.", &[ty_f32, ty_i32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value]);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_ALIGNBIT_B32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s2_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s2_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 31, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

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

                        let s2_value = llvm::core::LLVMBuildZExt(
                            builder,
                            s2_value,
                            ty_i64xn,
                            empty_name.as_ptr(),
                        );

                        let encoded = llvm::core::LLVMBuildOr(
                            builder,
                            llvm::core::LLVMBuildShl(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, 32, 0); N].as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            ),
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildLShr(
                            builder,
                            encoded,
                            s2_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            d_value,
                            ty_i32xn,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s2_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s2_value,
                            llvm::core::LLVMConstInt(ty_i32, 31, 0),
                            empty_name.as_ptr(),
                        );

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

                        let s2_value = llvm::core::LLVMBuildZExt(
                            builder,
                            s2_value,
                            ty_i64,
                            empty_name.as_ptr(),
                        );

                        let encoded = llvm::core::LLVMBuildOr(
                            builder,
                            llvm::core::LLVMBuildShl(
                                builder,
                                s0_value,
                                llvm::core::LLVMConstInt(ty_i64, 32, 0),
                                empty_name.as_ptr(),
                            ),
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildLShr(
                            builder,
                            encoded,
                            s2_value,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildTrunc(
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
            I::V_MAD_U32_U24 => {
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0xFFFFFF, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let s1_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s1_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0xFFFFFF, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAdd(
                            builder,
                            llvm::core::LLVMBuildMul(
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u32(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, inst.abs, inst.neg, 2);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstInt(ty_i32, 0xFFFFFF, 0),
                            empty_name.as_ptr(),
                        );

                        let s1_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s1_value,
                            llvm::core::LLVMConstInt(ty_i32, 0xFFFFFF, 0),
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildAdd(
                            builder,
                            llvm::core::LLVMBuildMul(
                                builder,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            ),
                            s2_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_MAX_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            ),
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            ),
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_MIN_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            ),
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, inst.abs, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, inst.abs, inst.neg, 1);

                        let d_value = llvm::core::LLVMBuildSelect(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                s0_value,
                                s1_value,
                                empty_name.as_ptr(),
                            ),
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_vop3sd(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOP3SD,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

                        let s2_value =
                            emitter.emit_vector_source_operand_u64xn::<N>(&inst.src2, i, mask);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, 0, inst.neg, 2);

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

                        let intrinsic = emitter
                            .get_intrinsic_declaration("llvm.uadd.with.overflow.", &[ty_i64xn]);
                        let add_overflow = intrinsic.emit_call(
                            llvm::core::LLVMStructTypeInContext(
                                context,
                                [ty_i64xn, ty_i1xn].as_mut_ptr(),
                                2,
                                0,
                            ),
                            &[muled, s2_value],
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

                    let intrinsic =
                        emitter.get_intrinsic_declaration("llvm.vector.reduce.or.", &[ty_i32xn]);
                    let d_value = intrinsic.emit_call(ty_i32, &[agg_value]);

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

                        let s2_value = emitter.emit_vector_source_operand_u64(&inst.src2, elem);
                        let s2_value = emitter.emit_abs_neg_bits(s2_value, 0, inst.neg, 2);

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
                        let muled = llvm::core::LLVMBuildMul(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter
                            .get_intrinsic_declaration("llvm.uadd.with.overflow.", &[ty_i64]);
                        let add_overflow = intrinsic.emit_call(
                            llvm::core::LLVMStructTypeInContext(
                                context,
                                [ty_i64, ty_i1].as_mut_ptr(),
                                2,
                                0,
                            ),
                            &[muled, s2_value],
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
            I::V_DIV_SCALE_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                    let mut vcc_values = Vec::new();

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask);

                        let s2_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src2, i, mask);

                        let s0_value = emitter.emit_abs_neg_f32xn::<N>(s0_value, 0, inst.neg, 0);

                        let s1_value = emitter.emit_abs_neg_f32xn::<N>(s1_value, 0, inst.neg, 1);

                        let s2_value = emitter.emit_abs_neg_f32xn::<N>(s2_value, 0, inst.neg, 2);

                        let (d_value, vcc_value) =
                            emitter.emit_div_scale_f32(s0_value, s1_value, s2_value);

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);

                        vcc_values.push(vcc_value);
                    }

                    let vcc_value = emitter.emit_concat::<N>(&vcc_values);
                    let vcc_value = llvm::core::LLVMBuildBitCast(
                        builder,
                        vcc_value,
                        ty_i32,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, vcc_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f32(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f32(&inst.src2, elem);

                        let (d0_value, d1_value) =
                            emitter.emit_div_scale_f32(s0_value, s1_value, s2_value);

                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d0_value);

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

                        let s0_value = emitter.emit_abs_neg_f64xn::<N>(s0_value, 0, inst.neg, 0);

                        let s1_value = emitter.emit_abs_neg_f64xn::<N>(s1_value, 0, inst.neg, 1);

                        let s2_value = emitter.emit_abs_neg_f64xn::<N>(s2_value, 0, inst.neg, 2);

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

                        let d_value = emitter.emit_vop3_omod_clamp(inst.omod, inst.cm, d_value);
                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                    emitter.emit_store_sgpr_u32(
                        inst.sdst as u32,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    );
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let s1_value = emitter.emit_vector_source_operand_f64(&inst.src1, elem);

                        let s2_value = emitter.emit_vector_source_operand_f64(&inst.src2, elem);

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
            I::V_SUBREV_CO_CI_U32 => {
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
                        let carry = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);
                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

                        // tmp = S1 - S0 - VCC, carry out = S0 + VCC > S1
                        let s0_wide = llvm::core::LLVMBuildZExt(
                            builder,
                            s0_value,
                            ty_i64xn,
                            empty_name.as_ptr(),
                        );
                        let s1_wide = llvm::core::LLVMBuildZExt(
                            builder,
                            s1_value,
                            ty_i64xn,
                            empty_name.as_ptr(),
                        );
                        let carry_wide = llvm::core::LLVMBuildZExt(
                            builder,
                            carry,
                            ty_i64xn,
                            empty_name.as_ptr(),
                        );

                        let borrowed = llvm::core::LLVMBuildAdd(
                            builder,
                            s0_wide,
                            carry_wide,
                            empty_name.as_ptr(),
                        );
                        let d_wide = llvm::core::LLVMBuildSub(
                            builder,
                            s1_wide,
                            borrowed,
                            empty_name.as_ptr(),
                        );
                        let d_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            d_wide,
                            ty_i32xn,
                            empty_name.as_ptr(),
                        );

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntUGT,
                            borrowed,
                            s1_wide,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);

                        cmp_values.push(cmp_value);
                    }

                    let cmp_value = emitter.emit_concat::<N>(&cmp_values);

                    let carry_out = llvm::core::LLVMBuildBitCast(
                        builder,
                        cmp_value,
                        ty_i32,
                        empty_name.as_ptr(),
                    );

                    emitter.emit_store_sgpr_u32(inst.sdst as u32, carry_out);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);
                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

                        let s2_value = emitter.emit_scalar_source_operand_u32(&inst.src2);
                        let index_mask = llvm::core::LLVMBuildShl(
                            builder,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            elem,
                            empty_name.as_ptr(),
                        );
                        let masked = llvm::core::LLVMBuildAnd(
                            builder,
                            s2_value,
                            index_mask,
                            empty_name.as_ptr(),
                        );
                        let carry = llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntNE,
                            masked,
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        );

                        let s0_wide = llvm::core::LLVMBuildZExt(
                            builder,
                            s0_value,
                            ty_i64,
                            empty_name.as_ptr(),
                        );
                        let s1_wide = llvm::core::LLVMBuildZExt(
                            builder,
                            s1_value,
                            ty_i64,
                            empty_name.as_ptr(),
                        );
                        let carry_wide =
                            llvm::core::LLVMBuildZExt(builder, carry, ty_i64, empty_name.as_ptr());

                        let borrowed = llvm::core::LLVMBuildAdd(
                            builder,
                            s0_wide,
                            carry_wide,
                            empty_name.as_ptr(),
                        );
                        let d_wide = llvm::core::LLVMBuildSub(
                            builder,
                            s1_wide,
                            borrowed,
                            empty_name.as_ptr(),
                        );
                        let d_value = llvm::core::LLVMBuildTrunc(
                            builder,
                            d_wide,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let cmp_value = llvm::core::LLVMBuildICmp(
                            builder,
                            llvm::LLVMIntPredicate::LLVMIntUGT,
                            borrowed,
                            s1_wide,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        (bb, cmp_value)
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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

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
                    bb = self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s2_value = emitter.emit_scalar_source_operand_u32(&inst.src2);

                        let index_mask = llvm::core::LLVMBuildShl(
                            builder,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            elem,
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

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

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

                        let d0_value =
                            llvm::core::LLVMBuildTrunc(builder, added, ty_i32, empty_name.as_ptr());

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
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);

                        let s1_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src1, i, mask);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

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
                    bb = self.emit_vop_update_sgpr(bb, inst.sdst as u32, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s0_value = emitter.emit_abs_neg_bits(s0_value, 0, inst.neg, 0);

                        let s1_value = emitter.emit_vector_source_operand_u32(&inst.src1, elem);
                        let s1_value = emitter.emit_abs_neg_bits(s1_value, 0, inst.neg, 1);

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

                        let d0_value =
                            llvm::core::LLVMBuildTrunc(builder, added, ty_i32, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d0_value);

                        (bb, cmp_value)
                    });
                }
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_vop3p(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOP3P,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
            I::V_WMMA_F32_16X16X16_F16 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    let matrix_a = emitter.matrix.matrix_a_ptr;
                    let matrix_b = emitter.matrix.matrix_b_ptr;
                    let matrix_c = emitter.matrix.matrix_c_ptr;
                    let matrix_d = emitter.matrix.matrix_d_ptr;

                    assert!(N <= 16);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f16xn_vec::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_vector_source_operand_f16xn_vec::<N>(&inst.src1, i, mask);

                        let s2_value =
                            emitter.emit_vector_source_operand_f32xn_vec::<N>(&inst.src2, i, mask);

                        for j in 0..2 {
                            for k in 0..2 {
                                for l in 0..2 {
                                    let col = llvm::core::LLVMConstInt(
                                        ty_i32,
                                        (l + k * 2 + j * 8 + (i / 16) * 4) as u64,
                                        0,
                                    );

                                    let row = llvm::core::LLVMConstInt(ty_i32, (i % 16) as u64, 0);

                                    let value = llvm::core::LLVMBuildFPExt(
                                        builder,
                                        s0_value[(l + k * 2 + j * 4) as usize],
                                        ty_f32xn,
                                        empty_name.as_ptr(),
                                    );

                                    let idx = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            col,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        row,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_a,
                                        [idx].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    llvm::core::LLVMBuildStore(builder, value, ptr);
                                }
                            }
                        }

                        for j in 0..2 {
                            for k in 0..2 {
                                for l in 0..2 {
                                    let row = llvm::core::LLVMConstInt(
                                        ty_i32,
                                        (l + k * 2 + j * 8 + (i / 16) * 4) as u64,
                                        0,
                                    );

                                    let col = llvm::core::LLVMConstInt(ty_i32, (i % 16) as u64, 0);

                                    let value = llvm::core::LLVMBuildFPExt(
                                        builder,
                                        s1_value[(l + k * 2 + j * 4) as usize],
                                        ty_f32xn,
                                        empty_name.as_ptr(),
                                    );

                                    let idx = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            row,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        col,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr: *mut llvm_sys::LLVMValue = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_b,
                                        [idx].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    llvm::core::LLVMBuildStore(builder, value, ptr);
                                }
                            }
                        }

                        for j in 0..8 {
                            let row =
                                llvm::core::LLVMConstInt(ty_i32, (j + (i / 16) * 8) as u64, 0);

                            let col = llvm::core::LLVMConstInt(ty_i32, (i % 16) as u64, 0);

                            let value = s2_value[j as usize];

                            let idx = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    row,
                                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                    empty_name.as_ptr(),
                                ),
                                col,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_c,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            llvm::core::LLVMBuildStore(builder, value, ptr);
                        }
                    }

                    bb = emitter.emit_loop(bb, 16, |emitter, bb, i| {
                        let bb = emitter.emit_loop(bb, 16, |emitter, bb, j| {
                            let idx = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    i,
                                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                    empty_name.as_ptr(),
                                ),
                                j,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_c,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            let value = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_f32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            let (bb, value) =
                                emitter.emit_loop_reduce(bb, value, 16, |emitter, bb, value, k| {
                                    let idx_a = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            k,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        i,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr_a = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_a,
                                        [idx_a].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    let a = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        ty_f32,
                                        ptr_a,
                                        empty_name.as_ptr(),
                                    );
                                    let idx_b = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            k,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        j,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr_b = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_b,
                                        [idx_b].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    let b = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        ty_f32,
                                        ptr_b,
                                        empty_name.as_ptr(),
                                    );

                                    let intrinsic =
                                        emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                                    let value = intrinsic.emit_call(ty_f32, &[a, b, value]);

                                    (bb, value)
                                });

                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_d,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            llvm::core::LLVMBuildStore(builder, value, ptr);

                            bb
                        });
                        bb
                    });

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        for j in 0..8 {
                            let row =
                                llvm::core::LLVMConstInt(ty_i32, (j + (i / 16) * 8) as u64, 0);

                            let col = llvm::core::LLVMConstInt(ty_i32, (i % 16) as u64, 0);

                            let idx = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    row,
                                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                    empty_name.as_ptr(),
                                ),
                                col,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_d,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let value = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_f32xn,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f32xn::<N>(
                                inst.vdst as u32 + j,
                                i,
                                value,
                                mask,
                            );
                        }
                    }
                } else {
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                    let matrix_a = self.matrix.matrix_a_ptr;
                    let matrix_b = self.matrix.matrix_b_ptr;
                    let matrix_c = self.matrix.matrix_c_ptr;
                    let matrix_d = self.matrix.matrix_d_ptr;

                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_value = emitter.emit_vector_source_operand_f16_vec(&inst.src0, elem);
                        let s1_value = emitter.emit_vector_source_operand_f16_vec(&inst.src1, elem);
                        let s2_value = emitter.emit_vector_source_operand_f32_vec(&inst.src2, elem);

                        for i in 0..2 {
                            for j in 0..2 {
                                for k in 0..2 {
                                    let col = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMConstInt(
                                            ty_i32,
                                            (k + j * 2 + i * 8) as u64,
                                            0,
                                        ),
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            llvm::core::LLVMBuildUDiv(
                                                builder,
                                                elem,
                                                llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                                empty_name.as_ptr(),
                                            ),
                                            llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        empty_name.as_ptr(),
                                    );
                                    let row = llvm::core::LLVMBuildURem(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                        empty_name.as_ptr(),
                                    );

                                    let value = llvm::core::LLVMBuildFPExt(
                                        builder,
                                        s0_value[k + j * 2 + i * 4],
                                        ty_f32,
                                        empty_name.as_ptr(),
                                    );

                                    let idx = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            row,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        col,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_a,
                                        [idx].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    llvm::core::LLVMBuildStore(builder, value, ptr);
                                }
                            }
                        }

                        for i in 0..2 {
                            for j in 0..2 {
                                for k in 0..2 {
                                    let row = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMConstInt(
                                            ty_i32,
                                            (k + j * 2 + i * 8) as u64,
                                            0,
                                        ),
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            llvm::core::LLVMBuildUDiv(
                                                builder,
                                                elem,
                                                llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                                empty_name.as_ptr(),
                                            ),
                                            llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        empty_name.as_ptr(),
                                    );
                                    let col = llvm::core::LLVMBuildURem(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                        empty_name.as_ptr(),
                                    );

                                    let value = llvm::core::LLVMBuildFPExt(
                                        builder,
                                        s1_value[k + j * 2 + i * 4],
                                        ty_f32,
                                        empty_name.as_ptr(),
                                    );

                                    let idx = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            row,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        col,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_b,
                                        [idx].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    llvm::core::LLVMBuildStore(builder, value, ptr);
                                }
                            }
                        }

                        for i in 0..8 {
                            let row = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMConstInt(ty_i32, i as u64, 0),
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    llvm::core::LLVMBuildUDiv(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                        empty_name.as_ptr(),
                                    ),
                                    llvm::core::LLVMConstInt(ty_i32, 8, 0),
                                    empty_name.as_ptr(),
                                ),
                                empty_name.as_ptr(),
                            );

                            let col = llvm::core::LLVMBuildURem(
                                builder,
                                elem,
                                llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                empty_name.as_ptr(),
                            );

                            let value = s2_value[i];

                            let idx = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    row,
                                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                    empty_name.as_ptr(),
                                ),
                                col,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_c,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            llvm::core::LLVMBuildStore(builder, value, ptr);
                        }

                        bb
                    });

                    bb = self.emit_loop(bb, 16, |emitter, bb, i| {
                        let bb = emitter.emit_loop(bb, 16, |emitter, bb, j| {
                            let idx = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    i,
                                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                    empty_name.as_ptr(),
                                ),
                                j,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_c,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            let value = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_f32,
                                ptr,
                                empty_name.as_ptr(),
                            );
                            let (bb, value) =
                                emitter.emit_loop_reduce(bb, value, 16, |_, bb, value, k| {
                                    let idx_a = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            i,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        k,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr_a = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_a,
                                        [idx_a].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    let a = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        ty_f32,
                                        ptr_a,
                                        empty_name.as_ptr(),
                                    );
                                    let idx_b = llvm::core::LLVMBuildAdd(
                                        builder,
                                        llvm::core::LLVMBuildMul(
                                            builder,
                                            k,
                                            llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                            empty_name.as_ptr(),
                                        ),
                                        j,
                                        empty_name.as_ptr(),
                                    );
                                    let ptr_b = llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        matrix_b,
                                        [idx_b].as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    );
                                    let b = llvm::core::LLVMBuildLoad2(
                                        builder,
                                        ty_f32,
                                        ptr_b,
                                        empty_name.as_ptr(),
                                    );
                                    let mul = llvm::core::LLVMBuildFMul(
                                        builder,
                                        a,
                                        b,
                                        empty_name.as_ptr(),
                                    );
                                    let value = llvm::core::LLVMBuildFAdd(
                                        builder,
                                        value,
                                        mul,
                                        empty_name.as_ptr(),
                                    );
                                    (bb, value)
                                });

                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_d,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );
                            llvm::core::LLVMBuildStore(builder, value, ptr);

                            bb
                        });
                        bb
                    });

                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        for i in 0..8 {
                            let row = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMConstInt(ty_i32, i as u64, 0),
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    llvm::core::LLVMBuildUDiv(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                        empty_name.as_ptr(),
                                    ),
                                    llvm::core::LLVMConstInt(ty_i32, 8, 0),
                                    empty_name.as_ptr(),
                                ),
                                empty_name.as_ptr(),
                            );

                            let col = llvm::core::LLVMBuildURem(
                                builder,
                                elem,
                                llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                empty_name.as_ptr(),
                            );

                            let idx = llvm::core::LLVMBuildAdd(
                                builder,
                                llvm::core::LLVMBuildMul(
                                    builder,
                                    row,
                                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                                    empty_name.as_ptr(),
                                ),
                                col,
                                empty_name.as_ptr(),
                            );
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_f32,
                                matrix_d,
                                [idx].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let value = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_f32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_f32(inst.vdst as u32 + i, elem, value);
                        }

                        bb
                    });
                }
            }
            I::V_FMA_MIXLO_F16 => {
                if USE_SIMD {
                    let emitter = self;
                    let ty_f16 = llvm::core::LLVMHalfTypeInContext(context);
                    let ty_f16xn = llvm::core::LLVMVectorType(ty_f16, SIMD_WIDTH as u32);
                    let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                    let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, SIMD_WIDTH as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, SIMD_WIDTH as u32);
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, SIMD_WIDTH as u32);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let opsel_hi = (inst.opsel_hi2 << 2) | inst.opsel_hi;

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        // {OPSEL_HI[i], OPSEL[i]}: HI=0 → f32; HI=1 & SEL=1 → hi f16;
                        // HI=1 & SEL=0 → lo f16 (RDNA4 ISA §V_FMA_MIXLO_F16).
                        let s0_value = if opsel_hi & 1 == 0 {
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask)
                        } else if inst.opsel & 1 != 0 {
                            let value = emitter
                                .emit_vector_source_operand_hi_f16xn::<N>(&inst.src0, i, mask);

                            llvm::core::LLVMBuildFPExt(
                                builder,
                                value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            )
                        } else {
                            let value =
                                emitter.emit_vector_source_operand_f16xn::<N>(&inst.src0, i, mask);
                            llvm::core::LLVMBuildFPExt(
                                builder,
                                value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            )
                        };

                        let s1_value = if opsel_hi & 2 == 0 {
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src1, i, mask)
                        } else if inst.opsel & 2 != 0 {
                            let value = emitter
                                .emit_vector_source_operand_hi_f16xn::<N>(&inst.src1, i, mask);

                            llvm::core::LLVMBuildFPExt(
                                builder,
                                value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            )
                        } else {
                            let value =
                                emitter.emit_vector_source_operand_f16xn::<N>(&inst.src1, i, mask);
                            llvm::core::LLVMBuildFPExt(
                                builder,
                                value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            )
                        };

                        let s2_value = if opsel_hi & 4 == 0 {
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src2, i, mask)
                        } else if inst.opsel & 4 != 0 {
                            let value = emitter
                                .emit_vector_source_operand_hi_f16xn::<N>(&inst.src2, i, mask);

                            llvm::core::LLVMBuildFPExt(
                                builder,
                                value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            )
                        } else {
                            let value =
                                emitter.emit_vector_source_operand_f16xn::<N>(&inst.src2, i, mask);
                            llvm::core::LLVMBuildFPExt(
                                builder,
                                value,
                                ty_f32xn,
                                empty_name.as_ptr(),
                            )
                        };

                        let s0_value =
                            emitter.emit_abs_neg_f32xn::<N>(s0_value, inst.neg_hi, inst.neg, 0);
                        let s1_value =
                            emitter.emit_abs_neg_f32xn::<N>(s1_value, inst.neg_hi, inst.neg, 1);
                        let s2_value =
                            emitter.emit_abs_neg_f32xn::<N>(s2_value, inst.neg_hi, inst.neg, 2);

                        let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, s2_value);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            d_value,
                            ty_f16xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            d_value,
                            ty_i16xn,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildZExt(
                            builder,
                            d_value,
                            ty_i32xn,
                            empty_name.as_ptr(),
                        );
                        // MIXLO writes vdst[15:0]; preserve the existing high 16 bits.
                        let old = emitter.emit_load_vgpr_u32xn::<N>(inst.vdst as u32, i, mask);
                        let himask = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 0xffff_0000, 0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let hi = llvm::core::LLVMBuildAnd(builder, old, himask, empty_name.as_ptr());
                        let d_value = llvm::core::LLVMBuildOr(builder, hi, d_value, empty_name.as_ptr());
                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                        let ty_f16 = llvm::core::LLVMHalfTypeInContext(context);
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let opsel_hi = (inst.opsel_hi2 << 2) | inst.opsel_hi;

                        // {OPSEL_HI[i], OPSEL[i]}: HI=0 → f32; HI=1 & SEL=1 → hi f16;
                        // HI=1 & SEL=0 → lo f16 (RDNA4 ISA §V_FMA_MIXLO_F16).
                        let s0_value = if opsel_hi & 1 == 0 {
                            emitter.emit_vector_source_operand_f32(&inst.src0, elem)
                        } else if inst.opsel & 1 != 0 {
                            let value = emitter.emit_vector_source_operand_hi_f16(&inst.src0, elem);

                            llvm::core::LLVMBuildFPExt(builder, value, ty_f32, empty_name.as_ptr())
                        } else {
                            let value = emitter.emit_vector_source_operand_f16(&inst.src0, elem);
                            llvm::core::LLVMBuildFPExt(builder, value, ty_f32, empty_name.as_ptr())
                        };

                        let s1_value = if opsel_hi & 2 == 0 {
                            emitter.emit_vector_source_operand_f32(&inst.src1, elem)
                        } else if inst.opsel & 2 != 0 {
                            let value = emitter.emit_vector_source_operand_hi_f16(&inst.src1, elem);

                            llvm::core::LLVMBuildFPExt(builder, value, ty_f32, empty_name.as_ptr())
                        } else {
                            let value = emitter.emit_vector_source_operand_f16(&inst.src1, elem);
                            llvm::core::LLVMBuildFPExt(builder, value, ty_f32, empty_name.as_ptr())
                        };

                        let s2_value = if opsel_hi & 4 == 0 {
                            emitter.emit_vector_source_operand_f32(&inst.src2, elem)
                        } else if inst.opsel & 4 != 0 {
                            let value = emitter.emit_vector_source_operand_hi_f16(&inst.src2, elem);

                            llvm::core::LLVMBuildFPExt(builder, value, ty_f32, empty_name.as_ptr())
                        } else {
                            let value = emitter.emit_vector_source_operand_f16(&inst.src2, elem);
                            llvm::core::LLVMBuildFPExt(builder, value, ty_f32, empty_name.as_ptr())
                        };

                        let s0_value = emitter.emit_abs_neg_f32(inst.neg_hi, inst.neg, s0_value, 0);
                        let s1_value = emitter.emit_abs_neg_f32(inst.neg_hi, inst.neg, s1_value, 1);
                        let s2_value = emitter.emit_abs_neg_f32(inst.neg_hi, inst.neg, s2_value, 2);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value, s2_value]);

                        let d_value = llvm::core::LLVMBuildFPTrunc(
                            builder,
                            d_value,
                            ty_f16,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildBitCast(
                            builder,
                            d_value,
                            ty_i16,
                            empty_name.as_ptr(),
                        );

                        let d_value = llvm::core::LLVMBuildZExt(
                            builder,
                            d_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        // MIXLO writes vdst[15:0]; preserve the existing high 16 bits.
                        let old = emitter.emit_load_vgpr_u32(inst.vdst as u32, elem);
                        let hi = llvm::core::LLVMBuildAnd(
                            builder,
                            old,
                            llvm::core::LLVMConstInt(ty_i32, 0xffff_0000, 0),
                            empty_name.as_ptr(),
                        );
                        let d_value = llvm::core::LLVMBuildOr(builder, hi, d_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_vopd(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOPD,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        let emitter = self;
        let mut opx_results = Vec::new();
        let mut opy_results = Vec::new();
        let exec_value = emitter.emit_load_sgpr_u32(126);

        let vdstx = inst.vdstx as u32;
        let vdsty = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as u32;

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

                    let s1_value = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1x as u32, i, mask);

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
            I::V_DUAL_ADD_NC_U32 => {
                let empty_name = std::ffi::CString::new("").unwrap();

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildAdd(builder, s0_value, s1_value, empty_name.as_ptr());

                    opx_results.push(d_value);
                }
            }
            I::V_DUAL_ADD_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildFAdd(builder, s0_value, s1_value, empty_name.as_ptr());

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_MAX_NUM_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    let intrinsic = emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f32xn]);
                    let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value, s1_value]);

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_FMAMK_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                    let k_value = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstReal(
                            ty_f32,
                            f32::from_bits(inst.literal_constant.unwrap()) as f64,
                        ); N]
                            .as_mut_ptr(),
                        N as u32,
                    );

                    let d_value = emitter.emit_fma_f32xn::<N>(s0_value, k_value, s1_value);

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_SUB_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildFSub(builder, s0_value, s1_value, empty_name.as_ptr());
                    let d_value = emitter.emit_sub_nan_sign(d_value, s0_value, s1_value);

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_MUL_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildFMul(builder, s0_value, s1_value, empty_name.as_ptr());

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_FMAC_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let d_value = emitter.emit_load_vgpr_f32xn::<N>(vdstx, i, mask);

                    let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, d_value);

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_FMAAK_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0x, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1x as u32, i, mask);

                    let k_value = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstReal(
                            ty_f32,
                            f32::from_bits(inst.literal_constant.unwrap()) as f64,
                        ); N]
                            .as_mut_ptr(),
                        N as u32,
                    );

                    let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, k_value);

                    opx_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }
        match inst.opy {
            I::V_DUAL_MAX_NUM_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                    let intrinsic = emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f32xn]);
                    let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value, s1_value]);

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_FMAMK_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                    let k_value = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstReal(
                            ty_f32,
                            f32::from_bits(inst.literal_constant.unwrap()) as f64,
                        ); N]
                            .as_mut_ptr(),
                        N as u32,
                    );

                    let d_value = emitter.emit_fma_f32xn::<N>(s0_value, k_value, s1_value);

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_CNDMASK_B32 => {
                let empty_name = std::ffi::CString::new("").unwrap();

                let vcc_value = emitter.emit_load_sgpr_u32(106);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                    let cond = emitter.emit_bits_to_mask_u32xn::<N>(vcc_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

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

                    let s1_value = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildAdd(builder, s0_value, s1_value, empty_name.as_ptr());

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

                    let s1_value = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let s0_value = llvm::core::LLVMBuildAnd(
                        builder,
                        s0_value,
                        llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 0x1F, 0); N].as_mut_ptr(),
                            N as u32,
                        ),
                        empty_name.as_ptr(),
                    );

                    let d_value =
                        llvm::core::LLVMBuildShl(builder, s1_value, s0_value, empty_name.as_ptr());

                    opy_results.push(d_value);
                }
            }
            I::V_DUAL_AND_B32 => {
                let empty_name = std::ffi::CString::new("").unwrap();

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, s0_value, s1_value, empty_name.as_ptr());

                    opy_results.push(d_value);
                }
            }
            I::V_DUAL_ADD_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildFAdd(builder, s0_value, s1_value, empty_name.as_ptr());

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_SUB_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildFSub(builder, s0_value, s1_value, empty_name.as_ptr());
                    let d_value = emitter.emit_sub_nan_sign(d_value, s0_value, s1_value);

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_MUL_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let d_value =
                        llvm::core::LLVMBuildFMul(builder, s0_value, s1_value, empty_name.as_ptr());

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_FMAC_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let d_value = emitter.emit_load_vgpr_f32xn::<N>(vdsty, i, mask);

                    let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, d_value);

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            I::V_DUAL_FMAAK_F32 => {
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let s0_value =
                        emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0y, i, mask);

                    let s1_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1y as u32, i, mask);

                    let k_value = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstReal(
                            ty_f32,
                            f32::from_bits(inst.literal_constant.unwrap()) as f64,
                        ); N]
                            .as_mut_ptr(),
                        N as u32,
                    );

                    let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, k_value);

                    opy_results.push(llvm::core::LLVMBuildBitCast(
                        builder,
                        d_value,
                        ty_i32xn,
                        empty_name.as_ptr(),
                    ));
                }
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        for i in (0..32).step_by(N) {
            let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

            emitter.emit_store_vgpr_u32xn::<N>(vdstx, i, opx_results[i as usize / N], mask);
            emitter.emit_store_vgpr_u32xn::<N>(vdsty, i, opy_results[i as usize / N], mask);
        }

        bb
    }
}
