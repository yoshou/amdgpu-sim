use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_sopp(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &SOPP,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        match inst.op {
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
        }

        bb
    }

    pub(crate) unsafe fn emit_smem(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &SMEM,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        match inst.op {
            I::S_LOAD_U16 => {
                let emitter = self;
                let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                let empty_name = std::ffi::CString::new("").unwrap();

                let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32 * 2);

                {
                    let offset = llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                    let addr =
                        llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());

                    let ptr =
                        llvm::core::LLVMBuildIntToPtr(builder, addr, ty_p0, empty_name.as_ptr());

                    let data =
                        llvm::core::LLVMBuildLoad2(builder, ty_i16, ptr, empty_name.as_ptr());

                    let data =
                        llvm::core::LLVMBuildZExt(builder, data, ty_i32, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdata as u32, data);
                }
            }
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

                    let ptr =
                        llvm::core::LLVMBuildIntToPtr(builder, addr, ty_p0, empty_name.as_ptr());

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
                    let offset = llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                    let addr =
                        llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                    let ptr =
                        llvm::core::LLVMBuildIntToPtr(builder, addr, ty_p0, empty_name.as_ptr());
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
                    let offset = llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                    let addr =
                        llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                    let ptr =
                        llvm::core::LLVMBuildIntToPtr(builder, addr, ty_p0, empty_name.as_ptr());
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
                    let offset = llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                    let addr =
                        llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                    let ptr =
                        llvm::core::LLVMBuildIntToPtr(builder, addr, ty_p0, empty_name.as_ptr());
                    let data =
                        llvm::core::LLVMBuildLoad2(builder, ty_i32, ptr, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                }
            }
            I::S_LOAD_B256 => {
                let emitter = self;
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                let empty_name = std::ffi::CString::new("").unwrap();

                let sbase = emitter.emit_load_sgpr_u64(inst.sbase as u32 * 2);

                for i in 0..8 {
                    let offset = llvm::core::LLVMConstInt(ty_i64, (inst.ioffset + i * 4) as u64, 0);
                    let addr =
                        llvm::core::LLVMBuildAdd(builder, sbase, offset, empty_name.as_ptr());
                    let ptr =
                        llvm::core::LLVMBuildIntToPtr(builder, addr, ty_p0, empty_name.as_ptr());
                    let data =
                        llvm::core::LLVMBuildLoad2(builder, ty_i32, ptr, empty_name.as_ptr());

                    emitter.emit_store_sgpr_u32(inst.sdata as u32 + i, data);
                }
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_sop1(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &SOP1,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        match inst.op {
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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let d_value =
                    llvm::core::LLVMBuildAnd(builder, s0_value, not1_value, empty_name.as_ptr());

                emitter.emit_store_sgpr_u32(126, d_value);

                let cmp = llvm::core::LLVMBuildICmp(
                    builder,
                    llvm::LLVMIntPredicate::LLVMIntNE,
                    d_value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let intrinsic = emitter.get_intrinsic_declaration("llvm.cttz.", &[ty_i32]);
                let d_value =
                    intrinsic.emit_call(ty_i32, &[s0_value, llvm::core::LLVMConstInt(ty_i1, 0, 0)]);

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
            I::S_CVT_F32_I32 => {
                let emitter = self;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                let d_value =
                    llvm::core::LLVMBuildSIToFP(builder, s0_value, ty_f32, empty_name.as_ptr());

                emitter.emit_store_sgpr_f32(inst.sdst as u32, d_value);
            }
            I::S_CVT_F32_U32 => {
                let emitter = self;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);

                let d_value =
                    llvm::core::LLVMBuildUIToFP(builder, s0_value, ty_f32, empty_name.as_ptr());

                emitter.emit_store_sgpr_f32(inst.sdst as u32, d_value);
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_sop2(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &SOP2,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        match inst.op {
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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let d_value =
                    llvm::core::LLVMBuildAnd(builder, s0_value, not1_value, empty_name.as_ptr());

                emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                let cmp = llvm::core::LLVMBuildICmp(
                    builder,
                    llvm::LLVMIntPredicate::LLVMIntNE,
                    d_value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                emitter.emit_store_scc_u8(scc_value);
            }
            I::S_BFE_U32 => {
                let emitter = self;
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                // offset = s1[4:0], width = s1[22:16]
                let offset = llvm::core::LLVMBuildAnd(
                    builder,
                    s1_value,
                    llvm::core::LLVMConstInt(ty_i32, 0x1F, 0),
                    empty_name.as_ptr(),
                );
                let width = llvm::core::LLVMBuildAnd(
                    builder,
                    llvm::core::LLVMBuildLShr(
                        builder,
                        s1_value,
                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                        empty_name.as_ptr(),
                    ),
                    llvm::core::LLVMConstInt(ty_i32, 0x7F, 0),
                    empty_name.as_ptr(),
                );

                let shifted =
                    llvm::core::LLVMBuildLShr(builder, s0_value, offset, empty_name.as_ptr());

                // mask = (1 << width) - 1
                let one = llvm::core::LLVMConstInt(ty_i32, 1, 0);
                let mask = llvm::core::LLVMBuildSub(
                    builder,
                    llvm::core::LLVMBuildShl(builder, one, width, empty_name.as_ptr()),
                    one,
                    empty_name.as_ptr(),
                );

                let d_value = llvm::core::LLVMBuildAnd(builder, shifted, mask, empty_name.as_ptr());

                emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);

                let cmp = llvm::core::LLVMBuildICmp(
                    builder,
                    llvm::LLVMIntPredicate::LLVMIntNE,
                    d_value,
                    llvm::core::LLVMConstInt(ty_i32, 0, 0),
                    empty_name.as_ptr(),
                );

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                emitter.emit_store_scc_u8(scc_value);
            }
            I::S_BFM_B32 => {
                let emitter = self;
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                let s0_value = llvm::core::LLVMBuildAnd(
                    builder,
                    s0_value,
                    llvm::core::LLVMConstInt(ty_i32, 31, 0),
                    empty_name.as_ptr(),
                );
                let s1_value = llvm::core::LLVMBuildAnd(
                    builder,
                    s1_value,
                    llvm::core::LLVMConstInt(ty_i32, 31, 0),
                    empty_name.as_ptr(),
                );

                let d_value = llvm::core::LLVMBuildShl(
                    builder,
                    llvm::core::LLVMBuildSub(
                        builder,
                        llvm::core::LLVMBuildShl(
                            builder,
                            llvm::core::LLVMConstInt(ty_i32, 1, 0),
                            s0_value,
                            empty_name.as_ptr(),
                        ),
                        llvm::core::LLVMConstInt(ty_i32, 1, 0),
                        empty_name.as_ptr(),
                    ),
                    s1_value,
                    empty_name.as_ptr(),
                );

                emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
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

                let intrinsic =
                    emitter.get_intrinsic_declaration("llvm.sadd.with.overflow.", &[ty_i32]);

                let mut return_tys = vec![ty_i32, ty_i1];
                let add_overflow = intrinsic.emit_call(
                    llvm::core::LLVMStructTypeInContext(
                        context,
                        return_tys.as_mut_ptr(),
                        return_tys.len() as u32,
                        0,
                    ),
                    &[s0_value, s1_value],
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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let intrinsic =
                    emitter.get_intrinsic_declaration("llvm.ssub.with.overflow.", &[ty_i32]);

                let mut return_tys = vec![ty_i32, ty_i1];
                let add_overflow = intrinsic.emit_call(
                    llvm::core::LLVMStructTypeInContext(
                        context,
                        return_tys.as_mut_ptr(),
                        return_tys.len() as u32,
                        0,
                    ),
                    &[s0_value, s1_value],
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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                emitter.emit_store_scc_u8(scc_value);
            }
            I::S_LSHL_B64 => {
                let emitter = self;
                let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                let s0_value = emitter.emit_scalar_source_operand_u64(&inst.ssrc0);
                let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                let s1_value = llvm::core::LLVMBuildAnd(
                    builder,
                    s1_value,
                    llvm::core::LLVMConstInt(ty_i32, 63, 0),
                    empty_name.as_ptr(),
                );
                let s1_value =
                    llvm::core::LLVMBuildZExt(builder, s1_value, ty_i64, empty_name.as_ptr());

                let d_value =
                    llvm::core::LLVMBuildShl(builder, s0_value, s1_value, empty_name.as_ptr());

                emitter.emit_store_sgpr_u64(inst.sdst as u32, d_value);

                let cmp = llvm::core::LLVMBuildICmp(
                    builder,
                    llvm::LLVMIntPredicate::LLVMIntNE,
                    d_value,
                    llvm::core::LLVMConstInt(ty_i64, 0, 0),
                    empty_name.as_ptr(),
                );

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                emitter.emit_store_scc_u8(scc_value);
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_sopc(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &SOPC,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        match inst.op {
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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                emitter.emit_store_scc_u8(scc_value);
            }
            I::S_CMP_GT_U32 => {
                let emitter = self;
                let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                let empty_name = std::ffi::CString::new("").unwrap();

                let s0_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc0);
                let s1_value = emitter.emit_scalar_source_operand_u32(&inst.ssrc1);

                let cmp = llvm::core::LLVMBuildICmp(
                    builder,
                    llvm::LLVMIntPredicate::LLVMIntUGT,
                    s0_value,
                    s1_value,
                    empty_name.as_ptr(),
                );

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

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

                let scc_value = llvm::core::LLVMBuildZExt(builder, cmp, ty_i8, empty_name.as_ptr());

                emitter.emit_store_scc_u8(scc_value);
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }

    pub(crate) unsafe fn emit_sopk(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &SOPK,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;

        match inst.op {
            I::S_MOVK_I32 => {
                let emitter = self;
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

                let d_value = llvm::core::LLVMConstInt(ty_i32, inst.simm16 as i16 as i64 as u64, 0);

                emitter.emit_store_sgpr_u32(inst.sdst as u32, d_value);
            }
            _ => {
                panic!("Unsupported instruction: {:?}", inst);
            }
        }

        bb
    }
}
