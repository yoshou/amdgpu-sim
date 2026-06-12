use crate::rdna_translator::*;

use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_vflat(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VFLAT,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
            I::FLAT_LOAD_B32 | I::FLAT_LOAD_B64 | I::FLAT_LOAD_B128 => {
                let num_words = match inst.op {
                    I::FLAT_LOAD_B32 => 1,
                    I::FLAT_LOAD_B64 => 2,
                    I::FLAT_LOAD_B128 => 4,
                    _ => unreachable!(),
                };

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

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let ioffset_value =
                        llvm::core::LLVMConstInt(ty_i64, inst.ioffset as i32 as i64 as u64, 0);

                    let zero_vec = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                        N as u32,
                    );
                    let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                    let voffset_value = llvm::core::LLVMBuildInsertElement(
                        builder,
                        poison,
                        ioffset_value,
                        llvm::core::LLVMConstInt(ty_i64, 0, 0),
                        empty_name.as_ptr(),
                    );
                    let voffset_value = llvm::core::LLVMBuildShuffleVector(
                        builder,
                        voffset_value,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    );

                    let vscratch_base_value = llvm::core::LLVMBuildInsertElement(
                        builder,
                        poison,
                        emitter.scratch_base,
                        llvm::core::LLVMConstInt(ty_i64, 0, 0),
                        empty_name.as_ptr(),
                    );
                    let vscratch_base_value = llvm::core::LLVMBuildShuffleVector(
                        builder,
                        vscratch_base_value,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    );

                    let vscratch_limit_value = llvm::core::LLVMBuildAdd(
                        builder,
                        vscratch_base_value,
                        llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i64, emitter.scratch_size as u64, 0); N]
                                .as_mut_ptr(),
                            N as u32,
                        ),
                        empty_name.as_ptr(),
                    );

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let addr_value =
                            emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i as u32, mask);

                        let global_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            addr_value,
                            voffset_value,
                            empty_name.as_ptr(),
                        );

                        let is_scratch_range = llvm::core::LLVMBuildAnd(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                global_offset,
                                vscratch_base_value,
                                empty_name.as_ptr(),
                            ),
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                global_offset,
                                vscratch_limit_value,
                                empty_name.as_ptr(),
                            ),
                            empty_name.as_ptr(),
                        );

                        let scratch_offset = llvm::core::LLVMBuildSub(
                            builder,
                            global_offset,
                            vscratch_base_value,
                            empty_name.as_ptr(),
                        );

                        let scratch_offset = llvm::core::LLVMBuildMul(
                            builder,
                            scratch_offset,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 32, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let elem_offset = llvm::core::LLVMConstVector(
                            (0..N)
                                .map(|n| llvm::core::LLVMConstInt(ty_i64, (i + n as u64) * 4, 0))
                                .collect::<Vec<_>>()
                                .as_mut_ptr(),
                            N as u32,
                        );

                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            scratch_offset,
                            elem_offset,
                            empty_name.as_ptr(),
                        );

                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            vscratch_base_value,
                            scratch_offset,
                            empty_name.as_ptr(),
                        );

                        for j in 0..num_words {
                            let global_offset = llvm::core::LLVMBuildAdd(
                                builder,
                                global_offset,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, (j as u64) * 4, 0); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let scratch_offset = llvm::core::LLVMBuildAdd(
                                builder,
                                scratch_offset,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, (j as u64) * 4 * 32, 0); N]
                                        .as_mut_ptr(),
                                    N as u32,
                                ),
                                empty_name.as_ptr(),
                            );

                            let offset = llvm::core::LLVMBuildSelect(
                                builder,
                                is_scratch_range,
                                scratch_offset,
                                global_offset,
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                offset,
                                ty_p0xn,
                                empty_name.as_ptr(),
                            );

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_i32xn, ty_p0xn],
                            );
                            let data = intrinsic.emit_call(
                                ty_i32xn,
                                &[
                                    ptr,
                                    llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i32xn),
                                ],
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(
                                inst.vdst as u32 + j,
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

                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

                        let ioffset_value =
                            llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                        let addr_value = emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem);
                        let global_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            addr_value,
                            ioffset_value,
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildSub(
                            builder,
                            global_offset,
                            emitter.scratch_base,
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildMul(
                            builder,
                            scratch_offset,
                            llvm::core::LLVMConstInt(ty_i64, 32, 0),
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            scratch_offset,
                            llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            emitter.scratch_base,
                            scratch_offset,
                            empty_name.as_ptr(),
                        );
                        let is_scratch_range = llvm::core::LLVMBuildAnd(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                global_offset,
                                emitter.scratch_base,
                                empty_name.as_ptr(),
                            ),
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                global_offset,
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    emitter.scratch_base,
                                    llvm::core::LLVMConstInt(
                                        ty_i64,
                                        emitter.scratch_size as u64,
                                        0,
                                    ),
                                    empty_name.as_ptr(),
                                ),
                                empty_name.as_ptr(),
                            ),
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

                        for j in 0..num_words {
                            let global_offset = llvm::core::LLVMBuildAdd(
                                builder,
                                global_offset,
                                llvm::core::LLVMConstInt(ty_i64, (j as u64) * 4, 0),
                                empty_name.as_ptr(),
                            );
                            let scratch_offset = llvm::core::LLVMBuildAdd(
                                builder,
                                scratch_offset,
                                llvm::core::LLVMConstInt(ty_i64, (j as u64) * 4 * 32, 0),
                                empty_name.as_ptr(),
                            );
                            let offset = llvm::core::LLVMBuildSelect(
                                builder,
                                is_scratch_range,
                                scratch_offset,
                                global_offset,
                                empty_name.as_ptr(),
                            );

                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                offset,
                                ty_p0,
                                empty_name.as_ptr(),
                            );

                            let data = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32 + j, elem, data);
                        }

                        llvm::core::LLVMBuildBr(builder, bb_cont);
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        bb = bb_cont;
                    }
                }
            }
            I::FLAT_STORE_B32 => {
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
                    let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let ioffset_value =
                        llvm::core::LLVMConstInt(ty_i64, inst.ioffset as i32 as i64 as u64, 0);

                    let zero_vec = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstInt(ty_i64, 0, 0); N].as_mut_ptr(),
                        N as u32,
                    );
                    let poison = llvm::core::LLVMGetPoison(ty_i64xn);

                    let voffset_value = llvm::core::LLVMBuildInsertElement(
                        builder,
                        poison,
                        ioffset_value,
                        llvm::core::LLVMConstInt(ty_i64, 0, 0),
                        empty_name.as_ptr(),
                    );
                    let voffset_value = llvm::core::LLVMBuildShuffleVector(
                        builder,
                        voffset_value,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    );

                    let vscratch_base_value = llvm::core::LLVMBuildInsertElement(
                        builder,
                        poison,
                        emitter.scratch_base,
                        llvm::core::LLVMConstInt(ty_i64, 0, 0),
                        empty_name.as_ptr(),
                    );
                    let vscratch_base_value = llvm::core::LLVMBuildShuffleVector(
                        builder,
                        vscratch_base_value,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    );

                    let vscratch_limit_value = llvm::core::LLVMBuildAdd(
                        builder,
                        vscratch_base_value,
                        llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i64, 0x1_0000_0000, 0); N].as_mut_ptr(),
                            N as u32,
                        ),
                        empty_name.as_ptr(),
                    );

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let addr_value =
                            emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i as u32, mask);

                        let global_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            addr_value,
                            voffset_value,
                            empty_name.as_ptr(),
                        );

                        let scratch_offset = llvm::core::LLVMBuildSub(
                            builder,
                            global_offset,
                            vscratch_base_value,
                            empty_name.as_ptr(),
                        );

                        let scratch_offset = llvm::core::LLVMBuildMul(
                            builder,
                            scratch_offset,
                            llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i64, 32, 0); N].as_mut_ptr(),
                                N as u32,
                            ),
                            empty_name.as_ptr(),
                        );

                        let elem_offset = llvm::core::LLVMConstVector(
                            (0..N)
                                .map(|n| llvm::core::LLVMConstInt(ty_i64, (i + n as u64) * 4, 0))
                                .collect::<Vec<_>>()
                                .as_mut_ptr(),
                            N as u32,
                        );

                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            scratch_offset,
                            elem_offset,
                            empty_name.as_ptr(),
                        );

                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            vscratch_base_value,
                            scratch_offset,
                            empty_name.as_ptr(),
                        );

                        let is_scratch_range = llvm::core::LLVMBuildAnd(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                global_offset,
                                vscratch_base_value,
                                empty_name.as_ptr(),
                            ),
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                global_offset,
                                vscratch_limit_value,
                                empty_name.as_ptr(),
                            ),
                            empty_name.as_ptr(),
                        );

                        let offset = llvm::core::LLVMBuildSelect(
                            builder,
                            is_scratch_range,
                            scratch_offset,
                            global_offset,
                            empty_name.as_ptr(),
                        );

                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            offset,
                            ty_p0xn,
                            empty_name.as_ptr(),
                        );

                        {
                            let value =
                                emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc as u32, i as u32, mask);

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.scatter.",
                                &[ty_i32xn, ty_p0xn],
                            );
                            intrinsic.emit_call(
                                ty_void,
                                &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
                            );
                        }
                    }
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

                        let ioffset_value =
                            llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
                        let addr_value = emitter.emit_load_vgpr_u64(inst.vaddr as u32, elem);
                        let global_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            addr_value,
                            ioffset_value,
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildSub(
                            builder,
                            global_offset,
                            emitter.scratch_base,
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildMul(
                            builder,
                            scratch_offset,
                            llvm::core::LLVMConstInt(ty_i64, 32, 0),
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            scratch_offset,
                            llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                            empty_name.as_ptr(),
                        );
                        let scratch_offset = llvm::core::LLVMBuildAdd(
                            builder,
                            emitter.scratch_base,
                            scratch_offset,
                            empty_name.as_ptr(),
                        );
                        let is_scratch_range = llvm::core::LLVMBuildAnd(
                            builder,
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntUGE,
                                global_offset,
                                emitter.scratch_base,
                                empty_name.as_ptr(),
                            ),
                            llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntULT,
                                global_offset,
                                llvm::core::LLVMBuildAdd(
                                    builder,
                                    emitter.scratch_base,
                                    llvm::core::LLVMConstInt(ty_i64, 0x1_0000_0000, 0),
                                    empty_name.as_ptr(),
                                ),
                                empty_name.as_ptr(),
                            ),
                            empty_name.as_ptr(),
                        );
                        let offset = llvm::core::LLVMBuildSelect(
                            builder,
                            is_scratch_range,
                            scratch_offset,
                            global_offset,
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
                            let ptr = llvm::core::LLVMBuildIntToPtr(
                                builder,
                                offset,
                                ty_p0,
                                empty_name.as_ptr(),
                            );

                            let value = emitter.emit_load_vgpr_u32(inst.vsrc as u32, elem);

                            llvm::core::LLVMBuildStore(builder, value, ptr);
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
        }

        bb
    }

    pub(crate) unsafe fn emit_vglobal(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VGLOBAL,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
            I::GLOBAL_WB => {}
            I::GLOBAL_INV => {}
            I::GLOBAL_ATOMIC_ADD_U32 => {
                let emitter = self;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                const N: usize = SIMD_WIDTH;
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                let exec_value = emitter.emit_load_sgpr_u32(126);

                // Splat the uniform scalar base address across the lanes.
                let saddr_vec = if inst.saddr != 124 {
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
                    llvm::core::LLVMBuildShuffleVector(
                        builder,
                        saddr_value,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    )
                } else {
                    std::ptr::null_mut()
                };

                let ioffset_vec = llvm::core::LLVMConstVector(
                    [llvm::core::LLVMConstInt(
                        ty_i64,
                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                        0,
                    ); N]
                        .as_mut_ptr(),
                    N as u32,
                );

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let addr_vec = if inst.saddr != 124 {
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
                            saddr_vec,
                            vaddr_value,
                            empty_name.as_ptr(),
                        )
                    } else {
                        emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr as u32, i, mask)
                    };
                    let addr_vec = llvm::core::LLVMBuildAdd(
                        builder,
                        addr_vec,
                        ioffset_vec,
                        empty_name.as_ptr(),
                    );

                    let data_vec = emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc as u32, i, mask);

                    let mut result_vec = llvm::core::LLVMGetPoison(ty_i32xn);
                    for lane in 0..N {
                        let lane_idx = llvm::core::LLVMConstInt(ty_i32, lane as u64, 0);
                        let active = llvm::core::LLVMBuildExtractElement(
                            builder,
                            mask,
                            lane_idx,
                            empty_name.as_ptr(),
                        );
                        let addr = llvm::core::LLVMBuildExtractElement(
                            builder,
                            addr_vec,
                            lane_idx,
                            empty_name.as_ptr(),
                        );
                        let data = llvm::core::LLVMBuildExtractElement(
                            builder,
                            data_vec,
                            lane_idx,
                            empty_name.as_ptr(),
                        );

                        let bb_pre = llvm::core::LLVMGetInsertBlock(builder);
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

                        llvm::core::LLVMBuildCondBr(builder, active, bb_exec, bb_cont);

                        // Only active lanes perform the atomic (it has side effects).
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_exec);
                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            addr,
                            ty_p0,
                            empty_name.as_ptr(),
                        );
                        let old = llvm::core::LLVMBuildAtomicRMW(
                            builder,
                            llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                            ptr,
                            data,
                            llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                            0,
                        );
                        let result_exec = llvm::core::LLVMBuildInsertElement(
                            builder,
                            result_vec,
                            old,
                            lane_idx,
                            empty_name.as_ptr(),
                        );
                        llvm::core::LLVMBuildBr(builder, bb_cont);

                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        let phi = llvm::core::LLVMBuildPhi(builder, ty_i32xn, empty_name.as_ptr());
                        llvm::core::LLVMAddIncoming(
                            phi,
                            [result_exec, result_vec].as_mut_ptr(),
                            [bb_exec, bb_pre].as_mut_ptr(),
                            2,
                        );
                        result_vec = phi;
                    }

                    emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, i, result_vec, mask);
                }

                bb = llvm::core::LLVMGetInsertBlock(builder);
            }
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
                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_i8xn, ty_p0xn],
                            );
                            let data = intrinsic.emit_call(
                                ty_i8xn,
                                &[
                                    ptr,
                                    llvm::core::LLVMConstInt(ty_i32, 1, 0),
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i8xn),
                                ],
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
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_i16xn, ty_p0xn],
                            );
                            let data = intrinsic.emit_call(
                                ty_i16xn,
                                &[
                                    ptr,
                                    llvm::core::LLVMConstInt(ty_i32, 2, 0),
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i16xn),
                                ],
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
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_i32xn, ty_p0xn],
                            );
                            let data = intrinsic.emit_call(
                                ty_i32xn,
                                &[
                                    ptr,
                                    llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i32xn),
                                ],
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(
                                inst.vdst as u32 + j as u32,
                                i,
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

                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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

                        for j in 0..1 {
                            let addr = llvm::core::LLVMBuildAdd(
                                builder,
                                offset,
                                llvm::core::LLVMConstInt(
                                    ty_i64,
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
                                ty_i32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32 + j as u32, elem, data);
                        }

                        llvm::core::LLVMBuildBr(builder, bb_cont);
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        bb = bb_cont;
                    }
                }
            }
            I::GLOBAL_LOAD_B64 => {
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i32xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_call(
                                    ty_i32xn,
                                    &[
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i32xn),
                                    ],
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_call(
                                    ty_i64xn,
                                    &[
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i64xn),
                                    ],
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
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                    let mut offsets = Vec::new();
                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                        offsets.push(offset);
                    }

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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
                                    ty_i64,
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
                                ty_i32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32 + j as u32, elem, data);
                        }

                        llvm::core::LLVMBuildBr(builder, bb_cont);
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        bb = bb_cont;
                    }
                }
            }
            I::GLOBAL_LOAD_B128 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_call(
                                    ty_i64xn,
                                    &[
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i64xn),
                                    ],
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_call(
                                    ty_i64xn,
                                    &[
                                        ptr,
                                        llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i64xn),
                                    ],
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
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                    let mut offsets = Vec::new();
                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                        offsets.push(offset);
                    }

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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
                                    ty_i64,
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
                                ty_i32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32 + j as u32, elem, data);
                        }

                        llvm::core::LLVMBuildBr(builder, bb_cont);
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        bb = bb_cont;
                    }
                }
            }
            I::GLOBAL_STORE_B16 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                    let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                    let ty_void = llvm::core::LLVMVoidTypeInContext(context);

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

                        let value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vsrc as u32, i as u32, mask);

                        let value = llvm::core::LLVMBuildTrunc(
                            builder,
                            value,
                            ty_i16xn,
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter.get_intrinsic_declaration(
                            "llvm.masked.scatter.",
                            &[ty_i16xn, ty_p0xn],
                        );
                        intrinsic.emit_call(
                            ty_void,
                            &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 2, 0), mask],
                        );
                    }
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                    let mut offsets = Vec::new();
                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                        offsets.push(offset);
                    }

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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

                        let data = emitter.emit_load_vgpr_u32(inst.vsrc as u32, elem);

                        let data =
                            llvm::core::LLVMBuildTrunc(builder, data, ty_i16, empty_name.as_ptr());

                        llvm::core::LLVMBuildStore(builder, data, ptr);

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

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.scatter.",
                                &[ty_i32xn, ty_p0xn],
                            );
                            intrinsic.emit_call(
                                ty_void,
                                &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
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
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                        offsets.push(offset);
                    }

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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
                                    ty_i64,
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let value = emitter.emit_load_vgpr_u64xn::<N>(
                                    inst.vsrc as u32 + j as u32,
                                    i as u32,
                                    mask,
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.scatter.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                intrinsic.emit_call(
                                    ty_void,
                                    &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let value = emitter.emit_load_vgpr_u64xn::<N>(
                                    inst.vsrc as u32 + j as u32,
                                    i as u32,
                                    mask,
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.scatter.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                intrinsic.emit_call(
                                    ty_void,
                                    &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
                                );
                            }
                        }
                    }
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                    let mut offsets = Vec::new();
                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                        offsets.push(offset);
                    }

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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
                                    ty_i64,
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let value = emitter.emit_load_vgpr_u64xn::<N>(
                                    inst.vsrc as u32 + j as u32,
                                    i as u32,
                                    mask,
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.scatter.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                intrinsic.emit_call(
                                    ty_void,
                                    &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
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
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let value = emitter.emit_load_vgpr_u64xn::<N>(
                                    inst.vsrc as u32 + j as u32,
                                    i as u32,
                                    mask,
                                );

                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.scatter.",
                                    &[ty_i64xn, ty_p0xn],
                                );
                                intrinsic.emit_call(
                                    ty_void,
                                    &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
                                );
                            }
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
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = if inst.saddr != 124 {
                            let saddr_value = emitter.emit_load_sgpr_u64(inst.saddr as u32);
                            let vaddr_value = emitter.emit_load_vgpr_u32(inst.vaddr as u32, elem);
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
                        offsets.push(offset);
                    }

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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
                                    ty_i64,
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
        }

        bb
    }

    pub(crate) unsafe fn emit_vscratch(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VSCRATCH,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
            I::SCRATCH_STORE_B32 | I::SCRATCH_STORE_B64 | I::SCRATCH_STORE_B128 => {
                let num_words = match inst.op {
                    I::SCRATCH_STORE_B32 => 1,
                    I::SCRATCH_STORE_B64 => 2,
                    I::SCRATCH_STORE_B96 => 3,
                    I::SCRATCH_STORE_B128 => 4,
                    _ => unreachable!(),
                };

                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                    let ioffset_value = llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
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
                        emitter.scratch_base,
                        offset,
                        empty_name.as_ptr(),
                    );

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let offset = llvm::core::LLVMBuildAdd(
                            builder,
                            offset,
                            llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                            empty_name.as_ptr(),
                        );
                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            offset,
                            ty_p0,
                            empty_name.as_ptr(),
                        );
                        for j in 0..num_words {
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i32,
                                ptr,
                                [llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0)].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let value = emitter.emit_load_vgpr_u32xn::<N>(
                                inst.vsrc as u32 + j as u32,
                                i,
                                mask,
                            );

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.store.",
                                &[ty_i32xn, ty_p0],
                            );
                            intrinsic.emit_call(
                                ty_void,
                                &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 4, 0), mask],
                            );
                        }
                    }
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);

                    let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                    let ioffset_value = llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
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
                        emitter.scratch_base,
                        offset,
                        empty_name.as_ptr(),
                    );

                    for i in 0..32 {
                        let empty_name = std::ffi::CString::new("").unwrap();
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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

                        let offset = llvm::core::LLVMBuildAdd(
                            builder,
                            offset,
                            llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                            empty_name.as_ptr(),
                        );

                        for j in 0..num_words {
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
                                [llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0)].as_mut_ptr(),
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
            I::SCRATCH_LOAD_B32
            | I::SCRATCH_LOAD_B64
            | I::SCRATCH_LOAD_B96
            | I::SCRATCH_LOAD_B128 => {
                let num_words: usize = match inst.op {
                    I::SCRATCH_LOAD_B32 => 1,
                    I::SCRATCH_LOAD_B64 => 2,
                    I::SCRATCH_LOAD_B96 => 3,
                    I::SCRATCH_LOAD_B128 => 4,
                    _ => unreachable!(),
                };

                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                    let ioffset_value = llvm::core::LLVMConstInt(ty_i64, inst.ioffset as u64, 0);
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
                        emitter.scratch_base,
                        offset,
                        empty_name.as_ptr(),
                    );

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let offset = llvm::core::LLVMBuildAdd(
                            builder,
                            offset,
                            llvm::core::LLVMConstInt(ty_i64, (i as u64) * 4, 0),
                            empty_name.as_ptr(),
                        );
                        let ptr = llvm::core::LLVMBuildIntToPtr(
                            builder,
                            offset,
                            ty_p0,
                            empty_name.as_ptr(),
                        );
                        for j in 0..num_words {
                            let ptr = llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i32,
                                ptr,
                                [llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0)].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let intrinsic = emitter
                                .get_intrinsic_declaration("llvm.masked.load.", &[ty_i32xn, ty_p0]);
                            let value = intrinsic.emit_call(
                                ty_i32xn,
                                &[
                                    ptr,
                                    llvm::core::LLVMConstInt(ty_i32, 4, 0),
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i32xn),
                                ],
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(
                                inst.vdst as u32 + j as u32,
                                i as u32,
                                value,
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
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

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

                        for j in 0..num_words {
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
                                [llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0)].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            );

                            let data = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i32,
                                ptr,
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdst as u32 + j as u32, elem, data);
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
        }

        bb
    }

    pub(crate) unsafe fn emit_ds(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &DS,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
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
                    let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let offset = emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, i, mask);

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

                        let value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.data0 as u32, i as u32, mask);

                        let value = llvm::core::LLVMBuildTrunc(
                            builder,
                            value,
                            ty_i8xn,
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter
                            .get_intrinsic_declaration("llvm.masked.scatter.", &[ty_i8xn, ty_p0xn]);
                        intrinsic.emit_call(
                            ty_void,
                            &[value, ptr, llvm::core::LLVMConstInt(ty_i32, 1, 0), mask],
                        );
                    }
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

                    for i in 0..32 {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);

                        let offset = emitter.emit_load_vgpr_u32(inst.addr as u32, elem);
                        let offset =
                            llvm::core::LLVMBuildZExt(builder, offset, ty_i64, empty_name.as_ptr());
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

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let offset = emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, i, mask);

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
                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_i8xn, ty_p0xn],
                            );
                            let data = intrinsic.emit_call(
                                ty_i8xn,
                                &[
                                    ptr,
                                    llvm::core::LLVMConstInt(ty_i32, 1, 0),
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i8xn),
                                ],
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
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        let offset = emitter.emit_load_vgpr_u32(inst.addr as u32, elem);

                        let offset =
                            llvm::core::LLVMBuildZExt(builder, offset, ty_i64, empty_name.as_ptr());

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
        }

        bb
    }

    pub(crate) unsafe fn emit_vimage(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VIMAGE,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;
        let mut bb = bb;

        match inst.op {
            I::IMAGE_BVH64_INTERSECT_RAY => {
                if USE_SIMD {
                    let emitter = self;
                    let context = emitter.context;

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let node_addr =
                            emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr0 as u32, i, mask);
                        let ray_extent =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr1 as u32, i, mask);

                        let ray_origin_x =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr2 as u32, i, mask);
                        let ray_origin_y =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr2 as u32 + 1, i, mask);
                        let ray_origin_z =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr2 as u32 + 2, i, mask);
                        let ray_dir_x =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr3 as u32, i, mask);
                        let ray_dir_y =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr3 as u32 + 1, i, mask);
                        let ray_dir_z =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr3 as u32 + 2, i, mask);
                        let ray_inv_dir_x =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr4 as u32, i, mask);
                        let ray_inv_dir_y =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr4 as u32 + 1, i, mask);
                        let ray_inv_dir_z =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr4 as u32 + 2, i, mask);

                        let values = [
                            ray_extent,
                            ray_origin_x,
                            ray_origin_y,
                            ray_origin_z,
                            ray_dir_x,
                            ray_dir_y,
                            ray_dir_z,
                            ray_inv_dir_x,
                            ray_inv_dir_y,
                            ray_inv_dir_z,
                        ];

                        llvm::core::LLVMBuildStore(
                            builder,
                            node_addr,
                            llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i64,
                                emitter.ray.node_addr_ptr,
                                [llvm::core::LLVMConstInt(ty_i64, i as u64, 0)].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            ),
                        );

                        for (j, value) in values.iter().enumerate() {
                            llvm::core::LLVMBuildStore(
                                builder,
                                *value,
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_f32,
                                    emitter.ray.values_ptr,
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        (i as u64) + j as u64 * 32,
                                        0,
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                ),
                            );
                        }
                    }

                    bb = emitter.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let node_addr = llvm::core::LLVMBuildLoad2(
                            builder,
                            ty_i64,
                            llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i64,
                                emitter.ray.node_addr_ptr,
                                [elem].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            ),
                            empty_name.as_ptr(),
                        );

                        let values = (0..10)
                            .map(|j| {
                                llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_f32,
                                    llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        emitter.ray.values_ptr,
                                        [llvm::core::LLVMBuildAdd(
                                            builder,
                                            elem,
                                            llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0),
                                            empty_name.as_ptr(),
                                        )]
                                        .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    ),
                                    empty_name.as_ptr(),
                                )
                            })
                            .collect::<Vec<_>>();

                        let ray_extent = values[0];
                        let ray_origin_x = values[1];
                        let ray_origin_y = values[2];
                        let ray_origin_z = values[3];
                        let ray_dir_x = values[4];
                        let ray_dir_y = values[5];
                        let ray_dir_z = values[6];
                        let ray_inv_dir_x = values[7];
                        let ray_inv_dir_y = values[8];
                        let ray_inv_dir_z = values[9];

                        let image_bvh64_intersect_ray_func = llvm::core::LLVMGetNamedFunction(
                            emitter.module,
                            "image_bvh64_intersect_ray\0".as_ptr() as *const _,
                        );

                        let mut param_tys = vec![
                            ty_p0, ty_p0, ty_p0, ty_p0, ty_i64, ty_f32, ty_f32, ty_f32, ty_f32,
                            ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32,
                        ];
                        let image_bvh64_intersect_ray_func_ty = llvm::core::LLVMFunctionType(
                            ty_void,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as u32,
                            0,
                        );
                        let image_bvh64_intersect_ray_func =
                            if image_bvh64_intersect_ray_func.is_null() {
                                llvm::core::LLVMAddFunction(
                                    emitter.module,
                                    "image_bvh64_intersect_ray\0".as_ptr() as *const _,
                                    image_bvh64_intersect_ray_func_ty,
                                )
                            } else {
                                image_bvh64_intersect_ray_func
                            };

                        let results_ptr = (0..4)
                            .map(|j| {
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    emitter.ray.results_ptr,
                                    [llvm::core::LLVMBuildAdd(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0),
                                        empty_name.as_ptr(),
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                )
                            })
                            .collect::<Vec<_>>();

                        llvm::core::LLVMBuildCall2(
                            builder,
                            image_bvh64_intersect_ray_func_ty,
                            image_bvh64_intersect_ray_func,
                            [
                                results_ptr[0],
                                results_ptr[1],
                                results_ptr[2],
                                results_ptr[3],
                                node_addr,
                                ray_extent,
                                ray_origin_x,
                                ray_origin_y,
                                ray_origin_z,
                                ray_dir_x,
                                ray_dir_y,
                                ray_dir_z,
                                ray_inv_dir_x,
                                ray_inv_dir_y,
                                ray_inv_dir_z,
                            ]
                            .as_mut_ptr(),
                            param_tys.len() as u32,
                            empty_name.as_ptr(),
                        );

                        bb
                    });

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                        for j in 0..4 {
                            let result = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i32xn,
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    emitter.ray.results_ptr,
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        (i as u64) + j as u64 * 32,
                                        0,
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                ),
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(
                                inst.vdata as u32 + j as u32,
                                i,
                                result,
                                mask,
                            );
                        }
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let node_addr = emitter.emit_load_vgpr_u64(inst.vaddr0 as u32, elem);
                        let ray_extent = emitter.emit_load_vgpr_f32(inst.vaddr1 as u32, elem);

                        let ray_origin_x = emitter.emit_load_vgpr_f32(inst.vaddr2 as u32, elem);
                        let ray_origin_y = emitter.emit_load_vgpr_f32(inst.vaddr2 as u32 + 1, elem);
                        let ray_origin_z = emitter.emit_load_vgpr_f32(inst.vaddr2 as u32 + 2, elem);
                        let ray_dir_x = emitter.emit_load_vgpr_f32(inst.vaddr3 as u32, elem);
                        let ray_dir_y = emitter.emit_load_vgpr_f32(inst.vaddr3 as u32 + 1, elem);
                        let ray_dir_z = emitter.emit_load_vgpr_f32(inst.vaddr3 as u32 + 2, elem);
                        let ray_inv_dir_x = emitter.emit_load_vgpr_f32(inst.vaddr4 as u32, elem);
                        let ray_inv_dir_y =
                            emitter.emit_load_vgpr_f32(inst.vaddr4 as u32 + 1, elem);
                        let ray_inv_dir_z =
                            emitter.emit_load_vgpr_f32(inst.vaddr4 as u32 + 2, elem);

                        let image_bvh64_intersect_ray_func = llvm::core::LLVMGetNamedFunction(
                            emitter.module,
                            "image_bvh64_intersect_ray\0".as_ptr() as *const _,
                        );

                        let mut param_tys = vec![
                            ty_p0, ty_p0, ty_p0, ty_p0, ty_i64, ty_f32, ty_f32, ty_f32, ty_f32,
                            ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32,
                        ];
                        let image_bvh64_intersect_ray_func_ty = llvm::core::LLVMFunctionType(
                            ty_void,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as u32,
                            0,
                        );
                        let image_bvh64_intersect_ray_func =
                            if image_bvh64_intersect_ray_func.is_null() {
                                llvm::core::LLVMAddFunction(
                                    emitter.module,
                                    "image_bvh64_intersect_ray\0".as_ptr() as *const _,
                                    image_bvh64_intersect_ray_func_ty,
                                )
                            } else {
                                image_bvh64_intersect_ray_func
                            };

                        let results_ptr = (0..4)
                            .map(|j| {
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    emitter.ray.results_ptr,
                                    [llvm::core::LLVMBuildAdd(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0),
                                        empty_name.as_ptr(),
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                )
                            })
                            .collect::<Vec<_>>();

                        llvm::core::LLVMBuildCall2(
                            builder,
                            image_bvh64_intersect_ray_func_ty,
                            image_bvh64_intersect_ray_func,
                            [
                                results_ptr[0],
                                results_ptr[1],
                                results_ptr[2],
                                results_ptr[3],
                                node_addr,
                                ray_extent,
                                ray_origin_x,
                                ray_origin_y,
                                ray_origin_z,
                                ray_dir_x,
                                ray_dir_y,
                                ray_dir_z,
                                ray_inv_dir_x,
                                ray_inv_dir_y,
                                ray_inv_dir_z,
                            ]
                            .as_mut_ptr(),
                            param_tys.len() as u32,
                            empty_name.as_ptr(),
                        );

                        for i in 0..4 {
                            let result = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i32,
                                results_ptr[i],
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdata as u32 + i as u32, elem, result);
                        }

                        bb
                    });
                }
            }
            I::IMAGE_BVH8_INTERSECT_RAY => {
                if USE_SIMD {
                    let emitter = self;
                    let context = emitter.context;

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                    let empty_name = std::ffi::CString::new("").unwrap();

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                        let node_base =
                            emitter.emit_load_vgpr_u64xn::<N>(inst.vaddr0 as u32, i, mask);
                        let ray_extent =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr1 as u32, i, mask);

                        let ray_origin_x =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr2 as u32, i, mask);
                        let ray_origin_y =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr2 as u32 + 1, i, mask);
                        let ray_origin_z =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr2 as u32 + 2, i, mask);
                        let ray_dir_x =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr3 as u32, i, mask);
                        let ray_dir_y =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr3 as u32 + 1, i, mask);
                        let ray_dir_z =
                            emitter.emit_load_vgpr_f32xn::<N>(inst.vaddr3 as u32 + 2, i, mask);
                        let node_index =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr4 as u32, i, mask);

                        let values = [
                            ray_extent,
                            ray_origin_x,
                            ray_origin_y,
                            ray_origin_z,
                            ray_dir_x,
                            ray_dir_y,
                            ray_dir_z,
                            node_index,
                        ];

                        llvm::core::LLVMBuildStore(
                            builder,
                            node_base,
                            llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i64,
                                emitter.ray.node_addr_ptr,
                                [llvm::core::LLVMConstInt(ty_i64, i as u64, 0)].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            ),
                        );

                        for (j, value) in values.iter().enumerate() {
                            llvm::core::LLVMBuildStore(
                                builder,
                                *value,
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_f32,
                                    emitter.ray.values_ptr,
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        (i as u64) + j as u64 * 32,
                                        0,
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                ),
                            );
                        }
                    }

                    bb = emitter.emit_vop(bb, |emitter, bb, elem| {
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let node_base = llvm::core::LLVMBuildLoad2(
                            builder,
                            ty_i64,
                            llvm::core::LLVMBuildGEP2(
                                builder,
                                ty_i64,
                                emitter.ray.node_addr_ptr,
                                [elem].as_mut_ptr(),
                                1,
                                empty_name.as_ptr(),
                            ),
                            empty_name.as_ptr(),
                        );

                        let values = (0..10)
                            .map(|j| {
                                llvm::core::LLVMBuildLoad2(
                                    builder,
                                    ty_f32,
                                    llvm::core::LLVMBuildGEP2(
                                        builder,
                                        ty_f32,
                                        emitter.ray.values_ptr,
                                        [llvm::core::LLVMBuildAdd(
                                            builder,
                                            elem,
                                            llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0),
                                            empty_name.as_ptr(),
                                        )]
                                        .as_mut_ptr(),
                                        1,
                                        empty_name.as_ptr(),
                                    ),
                                    empty_name.as_ptr(),
                                )
                            })
                            .collect::<Vec<_>>();

                        let ray_extent = values[0];
                        let ray_origin_x = values[1];
                        let ray_origin_y = values[2];
                        let ray_origin_z = values[3];
                        let ray_dir_x = values[4];
                        let ray_dir_y = values[5];
                        let ray_dir_z = values[6];
                        let node_index = llvm::core::LLVMBuildBitCast(
                            builder,
                            values[7],
                            ty_i32,
                            empty_name.as_ptr(),
                        );

                        let image_bvh8_intersect_ray_func = llvm::core::LLVMGetNamedFunction(
                            emitter.module,
                            "image_bvh8_intersect_ray\0".as_ptr() as *const _,
                        );

                        let mut param_tys = vec![
                            ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0,
                            ty_i64, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_i32,
                        ];
                        let image_bvh8_intersect_ray_func_ty = llvm::core::LLVMFunctionType(
                            ty_void,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as u32,
                            0,
                        );
                        let image_bvh8_intersect_ray_func =
                            if image_bvh8_intersect_ray_func.is_null() {
                                llvm::core::LLVMAddFunction(
                                    emitter.module,
                                    "image_bvh8_intersect_ray\0".as_ptr() as *const _,
                                    image_bvh8_intersect_ray_func_ty,
                                )
                            } else {
                                image_bvh8_intersect_ray_func
                            };

                        let results_ptr = (0..10)
                            .map(|j| {
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    emitter.ray.results_ptr,
                                    [llvm::core::LLVMBuildAdd(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0),
                                        empty_name.as_ptr(),
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                )
                            })
                            .collect::<Vec<_>>();

                        llvm::core::LLVMBuildCall2(
                            builder,
                            image_bvh8_intersect_ray_func_ty,
                            image_bvh8_intersect_ray_func,
                            [
                                results_ptr[0],
                                results_ptr[1],
                                results_ptr[2],
                                results_ptr[3],
                                results_ptr[4],
                                results_ptr[5],
                                results_ptr[6],
                                results_ptr[7],
                                results_ptr[8],
                                results_ptr[9],
                                node_base,
                                ray_extent,
                                ray_origin_x,
                                ray_origin_y,
                                ray_origin_z,
                                ray_dir_x,
                                ray_dir_y,
                                ray_dir_z,
                                node_index,
                            ]
                            .as_mut_ptr(),
                            param_tys.len() as u32,
                            empty_name.as_ptr(),
                        );

                        bb
                    });

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);
                        for j in 0..10 {
                            let result = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i32xn,
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    emitter.ray.results_ptr,
                                    [llvm::core::LLVMConstInt(
                                        ty_i64,
                                        (i as u64) + j as u64 * 32,
                                        0,
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                ),
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32xn::<N>(
                                inst.vdata as u32 + j as u32,
                                i,
                                result,
                                mask,
                            );
                        }
                    }
                } else {
                    bb = self.emit_vop(bb, |emitter, bb, elem| {
                        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
                        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                        let ty_void = llvm::core::LLVMVoidTypeInContext(context);
                        let empty_name = std::ffi::CString::new("").unwrap();

                        let node_base = emitter.emit_load_vgpr_u64(inst.vaddr0 as u32, elem);
                        let ray_extent = emitter.emit_load_vgpr_f32(inst.vaddr1 as u32, elem);
                        let ray_origin_x = emitter.emit_load_vgpr_f32(inst.vaddr2 as u32, elem);
                        let ray_origin_y = emitter.emit_load_vgpr_f32(inst.vaddr2 as u32 + 1, elem);
                        let ray_origin_z = emitter.emit_load_vgpr_f32(inst.vaddr2 as u32 + 2, elem);
                        let ray_dir_x = emitter.emit_load_vgpr_f32(inst.vaddr3 as u32, elem);
                        let ray_dir_y = emitter.emit_load_vgpr_f32(inst.vaddr3 as u32 + 1, elem);
                        let ray_dir_z = emitter.emit_load_vgpr_f32(inst.vaddr3 as u32 + 2, elem);
                        let node_index = emitter.emit_load_vgpr_u32(inst.vaddr4 as u32, elem);

                        let image_bvh8_intersect_ray_func = llvm::core::LLVMGetNamedFunction(
                            emitter.module,
                            "image_bvh8_intersect_ray\0".as_ptr() as *const _,
                        );

                        let mut param_tys = vec![
                            ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0, ty_p0,
                            ty_i64, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_f32, ty_i32,
                        ];
                        let image_bvh8_intersect_ray_func_ty = llvm::core::LLVMFunctionType(
                            ty_void,
                            param_tys.as_mut_ptr(),
                            param_tys.len() as u32,
                            0,
                        );
                        let image_bvh8_intersect_ray_func =
                            if image_bvh8_intersect_ray_func.is_null() {
                                llvm::core::LLVMAddFunction(
                                    emitter.module,
                                    "image_bvh8_intersect_ray\0".as_ptr() as *const _,
                                    image_bvh8_intersect_ray_func_ty,
                                )
                            } else {
                                image_bvh8_intersect_ray_func
                            };

                        let results_ptr = (0..10)
                            .map(|j| {
                                llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    emitter.ray.results_ptr,
                                    [llvm::core::LLVMBuildAdd(
                                        builder,
                                        elem,
                                        llvm::core::LLVMConstInt(ty_i32, j as u64 * 32, 0),
                                        empty_name.as_ptr(),
                                    )]
                                    .as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                )
                            })
                            .collect::<Vec<_>>();

                        llvm::core::LLVMBuildCall2(
                            builder,
                            image_bvh8_intersect_ray_func_ty,
                            image_bvh8_intersect_ray_func,
                            [
                                results_ptr[0],
                                results_ptr[1],
                                results_ptr[2],
                                results_ptr[3],
                                results_ptr[4],
                                results_ptr[5],
                                results_ptr[6],
                                results_ptr[7],
                                results_ptr[8],
                                results_ptr[9],
                                node_base,
                                ray_extent,
                                ray_origin_x,
                                ray_origin_y,
                                ray_origin_z,
                                ray_dir_x,
                                ray_dir_y,
                                ray_dir_z,
                                node_index,
                            ]
                            .as_mut_ptr(),
                            param_tys.len() as u32,
                            empty_name.as_ptr(),
                        );

                        for i in 0..10 {
                            let result = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_i32,
                                results_ptr[i],
                                empty_name.as_ptr(),
                            );

                            emitter.emit_store_vgpr_u32(inst.vdata as u32 + i as u32, elem, result);
                        }

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

    pub(crate) unsafe fn emit_vsample(
        &mut self,
        bb: llvm::prelude::LLVMBasicBlockRef,
        inst: &VSAMPLE,
    ) -> llvm::prelude::LLVMBasicBlockRef {
        let context = self.context;
        let builder = self.builder;

        match inst.op {
            I::IMAGE_SAMPLE_LZ => {
                let emitter = self;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

                const N: usize = SIMD_WIDTH;
                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

                let exec_value = emitter.emit_load_sgpr_u32(126);

                // The 8-dword image resource descriptor is uniform across lanes.
                let rsrc_values = (0..8)
                    .map(|i| emitter.emit_load_sgpr_u32(inst.rsrc as u32 + i))
                    .collect::<Vec<_>>();

                // Declare (or find) the runtime helper `image_sample_lz`, resolved
                // by the JIT through the process symbol table.
                let mut param_tys = vec![
                    ty_i32, ty_i32, ty_i32, ty_i32, ty_i32, ty_i32, ty_i32, ty_i32, ty_f32, ty_f32,
                ];
                let func_ty = llvm::core::LLVMFunctionType(
                    ty_i32,
                    param_tys.as_mut_ptr(),
                    param_tys.len() as u32,
                    0,
                );
                let func = llvm::core::LLVMGetNamedFunction(
                    emitter.module,
                    "image_sample_lz\0".as_ptr() as *const _,
                );
                let func = if func.is_null() {
                    llvm::core::LLVMAddFunction(
                        emitter.module,
                        "image_sample_lz\0".as_ptr() as *const _,
                        func_ty,
                    )
                } else {
                    func
                };

                let vaddr0 = inst.vaddr0 as u32;
                let vaddr1 = inst.vaddr1 as u32;
                let vdata = inst.vdata as u32;

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i);

                    let u_vec = emitter.emit_load_vgpr_f32xn::<N>(vaddr0, i, mask);
                    let v_vec = emitter.emit_load_vgpr_f32xn::<N>(vaddr1, i, mask);

                    // The fetch is side-effect free, so sampling every lane and
                    // discarding inactive ones via the masked store is safe.
                    let mut result_vec = llvm::core::LLVMGetPoison(ty_i32xn);
                    for lane in 0..N {
                        let lane_idx = llvm::core::LLVMConstInt(ty_i32, lane as u64, 0);
                        let u = llvm::core::LLVMBuildExtractElement(
                            builder,
                            u_vec,
                            lane_idx,
                            empty_name.as_ptr(),
                        );
                        let v = llvm::core::LLVMBuildExtractElement(
                            builder,
                            v_vec,
                            lane_idx,
                            empty_name.as_ptr(),
                        );

                        let mut args = vec![
                            rsrc_values[0],
                            rsrc_values[1],
                            rsrc_values[2],
                            rsrc_values[3],
                            rsrc_values[4],
                            rsrc_values[5],
                            rsrc_values[6],
                            rsrc_values[7],
                            u,
                            v,
                        ];
                        let result = llvm::core::LLVMBuildCall2(
                            builder,
                            func_ty,
                            func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            empty_name.as_ptr(),
                        );

                        result_vec = llvm::core::LLVMBuildInsertElement(
                            builder,
                            result_vec,
                            result,
                            lane_idx,
                            empty_name.as_ptr(),
                        );
                    }

                    emitter.emit_store_vgpr_u32xn::<N>(vdata, i, result_vec, mask);
                }
            }
            _ => unimplemented!(),
        }

        bb
    }
}
