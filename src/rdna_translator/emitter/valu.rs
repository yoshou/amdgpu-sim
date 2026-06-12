use crate::rdna_translator::*;

use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_vopc(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOPC,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
            I::V_CMP_GT_U32 | I::V_CMP_LT_U32 | I::V_CMP_EQ_U32 | I::V_CMP_NE_U32 => {
                let pred = match inst.op {
                    I::V_CMP_GT_U32 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    I::V_CMP_LT_U32 => llvm::LLVMIntPredicate::LLVMIntULT,
                    I::V_CMP_EQ_U32 => llvm::LLVMIntPredicate::LLVMIntEQ,
                    I::V_CMP_NE_U32 => llvm::LLVMIntPredicate::LLVMIntNE,
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

                        let s1_value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    emitter.emit_store_sgpr_u32(106, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
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
            I::V_CMPX_NE_U32 | I::V_CMPX_LT_U32 | I::V_CMPX_GT_U32 | I::V_CMPX_EQ_U32 => {
                let pred = match inst.op {
                    I::V_CMPX_NE_U32 => llvm::LLVMIntPredicate::LLVMIntNE,
                    I::V_CMPX_LT_U32 => llvm::LLVMIntPredicate::LLVMIntULT,
                    I::V_CMPX_GT_U32 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    I::V_CMPX_EQ_U32 => llvm::LLVMIntPredicate::LLVMIntEQ,
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

                        let s1_value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    emitter.emit_store_sgpr_u32(126, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
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

                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, d_value, exec_value, empty_name.as_ptr());

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
            I::V_CMP_GT_F32
            | I::V_CMP_GE_F32
            | I::V_CMP_LE_F32
            | I::V_CMP_LT_F32
            | I::V_CMP_NGT_F32 => {
                let pred = match inst.op {
                    I::V_CMP_GT_F32 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_GE_F32 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMP_LE_F32 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMP_LT_F32 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMP_NGT_F32 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    _ => unreachable!(),
                };
                let not = match inst.op {
                    I::V_CMP_NGT_F32 => true,
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
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    emitter.emit_store_sgpr_u32(106, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
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
            I::V_CMP_GT_U64 | I::V_CMP_EQ_U64 => {
                let pred = match inst.op {
                    I::V_CMP_GT_U64 => llvm::LLVMIntPredicate::LLVMIntUGT,
                    I::V_CMP_EQ_U64 => llvm::LLVMIntPredicate::LLVMIntEQ,
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

                        let s1_value =
                            emitter.emit_load_vgpr_u64xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    emitter.emit_store_sgpr_u32(106, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u64(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u64(inst.vsrc1 as u32, elem);
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
            I::V_CMP_GT_F64
            | I::V_CMP_LT_F64
            | I::V_CMP_NLT_F64
            | I::V_CMP_NGT_F64
            | I::V_CMP_LE_F64 => {
                let pred = match inst.op {
                    I::V_CMP_GT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_LT_F64 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMP_LE_F64 => llvm::LLVMRealPredicate::LLVMRealOLE,
                    I::V_CMP_NGT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    I::V_CMP_NLT_F64 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    _ => unreachable!(),
                };
                let not = match inst.op {
                    I::V_CMP_NGT_F64 | I::V_CMP_NLT_F64 => true,
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
                            emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    emitter.emit_store_sgpr_u32(106, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
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
            I::V_CMPX_NGE_F64 | I::V_CMPX_NGT_F64 => {
                let pred = match inst.op {
                    I::V_CMPX_LT_I32 => llvm::LLVMRealPredicate::LLVMRealOLT,
                    I::V_CMPX_NGE_F64 => llvm::LLVMRealPredicate::LLVMRealOGE,
                    I::V_CMPX_NGT_F64 => llvm::LLVMRealPredicate::LLVMRealOGT,
                    _ => unreachable!(),
                };
                let not = match inst.op {
                    I::V_CMPX_NGE_F64 | I::V_CMPX_NGT_F64 => true,
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
                            emitter.emit_load_vgpr_f64xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    let d_value =
                        llvm::core::LLVMBuildAnd(builder, d_value, exec_value, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(126, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 126, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);
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
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_vop1(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOP1,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
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
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                        let d_value = llvm::core::LLVMBuildUIToFP(
                            builder,
                            s0_value,
                            ty_f64,
                            empty_name.as_ptr(),
                        );

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

                        let d_value =
                            llvm::core::LLVMBuildNot(builder, s0_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);

                        let d_value =
                            llvm::core::LLVMBuildNot(builder, s0_value, empty_name.as_ptr());

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

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
            I::V_SQRT_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

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

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f32xn]);
                        let d_value = intrinsic.emit_call(ty_f32xn, &[s0_value]);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value]);

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

                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.sqrt.", &[ty_f64]);
                        let sqrt_value = intrinsic.emit_call(ty_f64, &[s0_value]);
                        let d_value = llvm::core::LLVMBuildFDiv(
                            builder,
                            llvm::core::LLVMConstReal(ty_f64, 1.0),
                            sqrt_value,
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

                    let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
                    let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f64xn::<N>(&inst.src0, i, mask);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.roundeven.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

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

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value]);

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

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.floor.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value]);

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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);

                        let d_value = llvm::core::LLVMBuildFPToSI(
                            builder,
                            s0_value,
                            ty_i32,
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let d_value = llvm::core::LLVMBuildFPToUI(
                            builder,
                            s0_value,
                            ty_i32,
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

                        let d_value = llvm::core::LLVMBuildSIToFP(
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

                        let d_value = llvm::core::LLVMBuildSIToFP(
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
            I::V_CVT_I32_F32 => {
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let d_value = llvm::core::LLVMBuildFPToSI(
                            builder,
                            s0_value,
                            ty_i32,
                            empty_name.as_ptr(),
                        );

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
            I::V_FREXP_MANT_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let d_value = emitter.emit_fract_f32xn::<N>(s0_value);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let d_value = emitter.emit_fract_f32(s0_value);

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

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let d_value = emitter.emit_exp_f32xn::<N>(s0_value);

                        emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);

                        let d_value = emitter.emit_exp_f32(s0_value);

                        emitter.emit_store_vgpr_u32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_READFIRSTLANE_B32 => {
                let emitter = self;
                let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                const N: usize = SIMD_WIDTH;

                let exec_value = emitter.emit_load_sgpr_u32(126);

                let intrinsic = emitter.get_intrinsic_declaration("llvm.cttz.", &[ty_i32]);
                let elem = intrinsic
                    .emit_call(ty_i32, &[exec_value, llvm::core::LLVMConstInt(ty_i1, 0, 0)]);

                let elem = llvm::core::LLVMBuildAnd(
                    builder,
                    elem,
                    llvm::core::LLVMConstInt(ty_i32, 31, 0),
                    empty_name.as_ptr(),
                );

                let d_value = if USE_SIMD {
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
                    llvm::core::LLVMBuildExtractElement(builder, value, elem, empty_name.as_ptr())
                } else {
                    emitter.emit_vector_source_operand_u32(&inst.src0, elem)
                };

                emitter.emit_store_sgpr_u32(inst.vdst as u32, d_value);
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
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_vop2(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VOP2,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
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
            I::V_SUB_NC_U32 => {
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
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
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
            I::V_MUL_U32_U24 => {
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
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
            I::V_SUBREV_NC_U32 => {
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

                        let d_value = llvm::core::LLVMBuildSub(
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
                        let d_value = llvm::core::LLVMBuildSub(
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

                        let s1_value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

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
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);

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

                        let s1_value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

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
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);

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
            I::V_ADD_CO_CI_U32 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);
                    let vcc_value = emitter.emit_load_sgpr_u32(106);

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
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc1 as u32, i, mask);

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

                    emitter.emit_store_sgpr_u32(106, d_value);
                } else {
                    bb = self.emit_vop_update_sgpr(bb, 106, |emitter, bb, elem| {
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s2_value = emitter.emit_load_sgpr_u32(106);

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

                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);

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
            I::V_OR_B32 => {
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

                        let d_value = llvm::core::LLVMBuildOr(
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
                        let d_value = llvm::core::LLVMBuildOr(
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
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
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u32(inst.vsrc1 as u32, elem);
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
            I::V_LSHLREV_B64 => {
                if USE_SIMD {
                    let emitter = self;
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                    for i in (0..32).step_by(N) {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_u32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_load_vgpr_u64xn::<N>(inst.vsrc1 as u32, i, mask);

                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 63, 0); N].as_mut_ptr(),
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
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_u32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_u64(inst.vsrc1 as u32, elem);
                        let s0_value = llvm::core::LLVMBuildAnd(
                            builder,
                            s0_value,
                            llvm::core::LLVMConstInt(ty_i32, 63, 0),
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

                        let vcc_value = emitter.emit_vcc_bit(elem);

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
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

                        let d_value = emitter.emit_fadd(s0_value, s1_value);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
                        let d_value = llvm::core::LLVMBuildFAdd(
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
            I::V_SUB_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

                        let d_value = llvm::core::LLVMBuildFSub(
                            builder,
                            s0_value,
                            s1_value,
                            empty_name.as_ptr(),
                        );

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
                        let d_value = llvm::core::LLVMBuildFSub(
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
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

                        let d_value = emitter.emit_load_vgpr_f32xn::<N>(inst.vdst as u32, i, mask);

                        let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, d_value);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
                        let d_value = emitter.emit_load_vgpr_f32(inst.vdst as u32, elem);

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value, d_value]);

                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FMAMK_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

                        let k_value = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(
                                ty_f32,
                                f32::from_bits(inst.literal_constant.unwrap()) as f64,
                            ); N]
                                .as_mut_ptr(),
                            N as u32,
                        );

                        let d_value = emitter.emit_fma_f32xn::<N>(s0_value, k_value, s1_value);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
                        let k_value = llvm::core::LLVMConstReal(
                            ty_f32,
                            f32::from_bits(inst.literal_constant.unwrap()) as f64,
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, k_value, s1_value]);

                        emitter.emit_store_vgpr_f32(inst.vdst as u32, elem, d_value);

                        bb
                    });
                }
            }
            I::V_FMAAK_F32 => {
                if USE_SIMD {
                    let emitter = self;
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    const N: usize = SIMD_WIDTH;

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let s0_value =
                            emitter.emit_vector_source_operand_f32xn::<N>(&inst.src0, i, mask);

                        let s1_value =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vsrc1 as u32, i, mask);

                        let k_value = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstReal(
                                ty_f32,
                                f32::from_bits(inst.literal_constant.unwrap()) as f64,
                            ); N]
                                .as_mut_ptr(),
                            N as u32,
                        );

                        let d_value = emitter.emit_fma_f32xn::<N>(s0_value, s1_value, k_value);

                        emitter.emit_store_vgpr_f32xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f32(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f32(inst.vsrc1 as u32, elem);
                        let k_value = llvm::core::LLVMConstReal(
                            ty_f32,
                            f32::from_bits(inst.literal_constant.unwrap()) as f64,
                        );

                        let intrinsic = emitter.get_intrinsic_declaration("llvm.fma.", &[ty_f32]);
                        let d_value = intrinsic.emit_call(ty_f32, &[s0_value, s1_value, k_value]);

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

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f64xn]);
                        let d_value = intrinsic.emit_call(ty_f64xn, &[s0_value, s1_value]);

                        emitter.emit_store_vgpr_f64xn::<N>(inst.vdst as u32, i, d_value, mask);
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

                        let s0_value = emitter.emit_vector_source_operand_f64(&inst.src0, elem);
                        let s1_value = emitter.emit_load_vgpr_f64(inst.vsrc1 as u32, elem);

                        let intrinsic =
                            emitter.get_intrinsic_declaration("llvm.maxnum.", &[ty_f64]);
                        let d_value = intrinsic.emit_call(ty_f64, &[s0_value, s1_value]);
                        emitter.emit_store_vgpr_f64(inst.vdst as u32, elem, d_value);

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
}
