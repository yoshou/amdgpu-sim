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
            I::FLAT_LOAD_B32
            | I::FLAT_LOAD_B64
            | I::FLAT_LOAD_B96
            | I::FLAT_LOAD_B128
            | I::FLAT_LOAD_U8
            | I::FLAT_LOAD_I8
            | I::FLAT_LOAD_U16
            | I::FLAT_LOAD_I16 => {
                let num_words = match inst.op {
                    I::FLAT_LOAD_B64 => 2,
                    I::FLAT_LOAD_B96 => 3,
                    I::FLAT_LOAD_B128 => 4,
                    _ => 1,
                };
                // A sub-dword load reads its own width and widens it into the
                // destination, signed or not according to the opcode.
                let (load_bits, load_signed) = match inst.op {
                    I::FLAT_LOAD_U8 => (8, false),
                    I::FLAT_LOAD_I8 => (8, true),
                    I::FLAT_LOAD_U16 => (16, false),
                    I::FLAT_LOAD_I16 => (16, true),
                    _ => (32, false),
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

                    let ioffset_value = llvm::core::LLVMConstInt(
                        ty_i64,
                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                        0,
                    );

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

                            let ty_loadxn = llvm::core::LLVMVectorType(
                                llvm::core::LLVMIntTypeInContext(context, load_bits),
                                N as u32,
                            );
                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_loadxn, ty_p0xn],
                            );
                            let data = intrinsic.emit_masked_call(
                                ty_loadxn,
                                &[ptr, mask, llvm::core::LLVMGetPoison(ty_loadxn)],
                                0,
                                4,
                            );
                            let data = if load_bits == 32 {
                                data
                            } else if load_signed {
                                llvm::core::LLVMBuildSExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                )
                            };

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

                        let ioffset_value = llvm::core::LLVMConstInt(
                            ty_i64,
                            ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                            0,
                        );
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

                            let ty_load = llvm::core::LLVMIntTypeInContext(context, load_bits);
                            let data = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_load,
                                ptr,
                                empty_name.as_ptr(),
                            );
                            let data = if load_bits == 32 {
                                data
                            } else if load_signed {
                                llvm::core::LLVMBuildSExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                )
                            };

                            emitter.emit_store_vgpr_u32(inst.vdst as u32 + j, elem, data);
                        }

                        llvm::core::LLVMBuildBr(builder, bb_cont);
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        bb = bb_cont;
                    }
                }
            }
            I::FLAT_STORE_B8
            | I::FLAT_STORE_B16
            | I::FLAT_STORE_B32
            | I::FLAT_STORE_B64
            | I::FLAT_STORE_B96
            | I::FLAT_STORE_B128 => {
                let num_words = match inst.op {
                    I::FLAT_STORE_B64 => 2,
                    I::FLAT_STORE_B96 => 3,
                    I::FLAT_STORE_B128 => 4,
                    _ => 1,
                };
                // A sub-dword store writes the low bits of the source register.
                let store_bits = match inst.op {
                    I::FLAT_STORE_B8 => 8,
                    I::FLAT_STORE_B16 => 16,
                    _ => 32,
                };
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

                    let ioffset_value = llvm::core::LLVMConstInt(
                        ty_i64,
                        ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                        0,
                    );

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

                        for j in 0..num_words {
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
                            let ty_storexn = llvm::core::LLVMVectorType(
                                llvm::core::LLVMIntTypeInContext(context, store_bits),
                                N as u32,
                            );
                            let value = if store_bits == 32 {
                                value
                            } else {
                                llvm::core::LLVMBuildTrunc(
                                    builder,
                                    value,
                                    ty_storexn,
                                    empty_name.as_ptr(),
                                )
                            };

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.scatter.",
                                &[ty_storexn, ty_p0xn],
                            );
                            intrinsic.emit_masked_call(
                                ty_void,
                                &[value, ptr, mask], 1, 4);
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

                        let ioffset_value = llvm::core::LLVMConstInt(
                            ty_i64,
                            ((((inst.ioffset << 8) as i32) >> 8) as i64) as u64,
                            0,
                        );
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

                            for j in 0..num_words {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );
                                let value =
                                    emitter.emit_load_vgpr_u32(inst.vsrc as u32 + j as u32, elem);
                                let value = if store_bits == 32 {
                                    value
                                } else {
                                    llvm::core::LLVMBuildTrunc(
                                        builder,
                                        value,
                                        llvm::core::LLVMIntTypeInContext(context, store_bits),
                                        empty_name.as_ptr(),
                                    )
                                };

                                llvm::core::LLVMBuildStore(builder, value, ptr);
                            }
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
            I::GLOBAL_LOAD_I8 => {
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
                            let data = intrinsic.emit_masked_call(
                                ty_i8xn,
                                &[
                                    ptr,
                                    mask,
                                    llvm::core::LLVMGetPoison(ty_i8xn),
                                ], 0, 1,
                            );

                            let data = llvm::core::LLVMBuildSExt(
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
                            let data = llvm::core::LLVMBuildSExt(
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
                            let data = intrinsic.emit_masked_call(
                                ty_i8xn,
                                &[ptr, mask, llvm::core::LLVMGetPoison(ty_i8xn)],
                                0,
                                1,
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
            I::GLOBAL_LOAD_I16 => {
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
                            let data = intrinsic.emit_masked_call(
                                ty_i16xn,
                                &[ptr, mask, llvm::core::LLVMGetPoison(ty_i16xn)],
                                0,
                                2,
                            );

                            let data = llvm::core::LLVMBuildSExt(
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
                            let data = llvm::core::LLVMBuildSExt(
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
                            let data = intrinsic.emit_masked_call(
                                ty_i16xn,
                                &[ptr, mask, llvm::core::LLVMGetPoison(ty_i16xn)],
                                0,
                                2,
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
                            let data = intrinsic.emit_masked_call(
                                ty_i32xn,
                                &[ptr, mask, llvm::core::LLVMGetPoison(ty_i32xn)],
                                0,
                                4,
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
                                let data = intrinsic.emit_masked_call(
                                    ty_i32xn,
                                    &[ptr, mask, llvm::core::LLVMGetPoison(ty_i32xn)],
                                    0,
                                    4,
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
                                let data = intrinsic.emit_masked_call(
                                    ty_i64xn,
                                    &[
                                        ptr,
                                        mask,
                                        llvm::core::LLVMGetPoison(ty_i64xn),
                                    ], 0, 4,
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
            I::GLOBAL_LOAD_B96 => {
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

                    const NUM_WORDS: usize = 3;

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

                            // Three words is an odd count, so they are gathered one at a
                            // time rather than in pairs.
                            for j in 0..NUM_WORDS {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i32xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_masked_call(
                                    ty_i32xn,
                                    &[ptr, mask, llvm::core::LLVMGetPoison(ty_i32xn)],
                                    0,
                                    4,
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

                            // Three words is an odd count, so they are gathered one at a
                            // time rather than in pairs.
                            for j in 0..NUM_WORDS {
                                let ptr = llvm::core::LLVMBuildGEP2(
                                    builder,
                                    ty_i32,
                                    ptr,
                                    [llvm::core::LLVMConstInt(ty_i32, j as u64, 0)].as_mut_ptr(),
                                    1,
                                    empty_name.as_ptr(),
                                );

                                let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i32xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_masked_call(
                                    ty_i32xn,
                                    &[ptr, mask, llvm::core::LLVMGetPoison(ty_i32xn)],
                                    0,
                                    4,
                                );

                                emitter.emit_store_vgpr_u32xn::<N>(
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

                        for j in 0..3 {
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
                                let data = intrinsic.emit_masked_call(
                                    ty_i64xn,
                                    &[ptr, mask, llvm::core::LLVMGetPoison(ty_i64xn)],
                                    0,
                                    4,
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
                                let data = intrinsic.emit_masked_call(
                                    ty_i64xn,
                                    &[ptr, mask, llvm::core::LLVMGetPoison(ty_i64xn)],
                                    0,
                                    4,
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
            I::GLOBAL_STORE_B8 => {
                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i8xn = llvm::core::LLVMVectorType(ty_i8, N as u32);
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
                            ty_i8xn,
                            empty_name.as_ptr(),
                        );

                        let intrinsic = emitter
                            .get_intrinsic_declaration("llvm.masked.scatter.", &[ty_i8xn, ty_p0xn]);
                        intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 2);
                    }
                } else {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
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
                            llvm::core::LLVMBuildTrunc(builder, data, ty_i8, empty_name.as_ptr());

                        llvm::core::LLVMBuildStore(builder, data, ptr);

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
                        intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 2);
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
                            intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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
                                intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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
                                intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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
            I::GLOBAL_STORE_B96 => {
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

                    const NUM_WORDS: usize = 3;

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
                                intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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
                                intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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

                        for j in 0..3 {
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
                                intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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
                                intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 4);
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

        match inst.op {
            I::SCRATCH_LOAD_U8
            | I::SCRATCH_LOAD_I8
            | I::SCRATCH_LOAD_U16
            | I::SCRATCH_LOAD_I16
            | I::SCRATCH_LOAD_B32
            | I::SCRATCH_LOAD_B64
            | I::SCRATCH_LOAD_B96
            | I::SCRATCH_LOAD_B128
            | I::SCRATCH_STORE_B8
            | I::SCRATCH_STORE_B16
            | I::SCRATCH_STORE_B32
            | I::SCRATCH_STORE_B64
            | I::SCRATCH_STORE_B96
            | I::SCRATCH_STORE_B128 => {
                // How wide the access is, and whether a sub-dword load widens
                // signed. A store never widens.
                let (bits, signed): (usize, bool) = match inst.op {
                    I::SCRATCH_LOAD_U8 | I::SCRATCH_STORE_B8 => (8, false),
                    I::SCRATCH_LOAD_I8 => (8, true),
                    I::SCRATCH_LOAD_U16 | I::SCRATCH_STORE_B16 => (16, false),
                    I::SCRATCH_LOAD_I16 => (16, true),
                    I::SCRATCH_LOAD_B64 | I::SCRATCH_STORE_B64 => (64, false),
                    I::SCRATCH_LOAD_B96 | I::SCRATCH_STORE_B96 => (96, false),
                    I::SCRATCH_LOAD_B128 | I::SCRATCH_STORE_B128 => (128, false),
                    _ => (32, false),
                };
                let is_load = matches!(
                    inst.op,
                    I::SCRATCH_LOAD_U8
                        | I::SCRATCH_LOAD_I8
                        | I::SCRATCH_LOAD_U16
                        | I::SCRATCH_LOAD_I16
                        | I::SCRATCH_LOAD_B32
                        | I::SCRATCH_LOAD_B64
                        | I::SCRATCH_LOAD_B96
                        | I::SCRATCH_LOAD_B128
                );
                // SVE says whether the VGPR takes part in the address; SADDR
                // being NULL says the same of the SGPR.
                let use_vaddr = inst.sve != 0;
                let use_saddr = inst.saddr != 0x7C;
                let ioffset = (((inst.ioffset << 8) as i32) >> 8) as i64;

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
                let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                let splat64 = |value: i64| {
                    llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstInt(ty_i64, value as u64, 0); N].as_mut_ptr(),
                        N as u32,
                    )
                };

                let exec_value = emitter.emit_load_sgpr_u32(126);

                for i in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                    // The byte the lane's access starts at, in its own view of
                    // the private segment.
                    let mut start = splat64(ioffset);
                    if use_vaddr {
                        let vaddr_value =
                            emitter.emit_load_vgpr_u32xn::<N>(inst.vaddr as u32, i as u32, mask);
                        let vaddr_value = llvm::core::LLVMBuildZExt(
                            builder,
                            vaddr_value,
                            ty_i64xn,
                            empty_name.as_ptr(),
                        );
                        start = llvm::core::LLVMBuildAdd(
                            builder,
                            start,
                            vaddr_value,
                            empty_name.as_ptr(),
                        );
                    }
                    if use_saddr {
                        let saddr_value = emitter.emit_load_sgpr_u32(inst.saddr as u32);
                        // The scratch SGPR offset is a signed byte offset.
                        let saddr_value = llvm::core::LLVMBuildSExt(
                            builder,
                            saddr_value,
                            ty_i64,
                            empty_name.as_ptr(),
                        );
                        let saddr_value = llvm::core::LLVMBuildInsertElement(
                            builder,
                            llvm::core::LLVMGetPoison(ty_i64xn),
                            saddr_value,
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        );
                        let zero_vec = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let saddr_value = llvm::core::LLVMBuildShuffleVector(
                            builder,
                            saddr_value,
                            llvm::core::LLVMGetPoison(ty_i64xn),
                            zero_vec,
                            empty_name.as_ptr(),
                        );
                        start = llvm::core::LLVMBuildAdd(
                            builder,
                            start,
                            saddr_value,
                            empty_name.as_ptr(),
                        );
                    }

                    // The lane's slot within a swizzled dword.
                    let lane_offset = llvm::core::LLVMConstVector(
                        (0..N)
                            .map(|n| llvm::core::LLVMConstInt(ty_i64, ((i + n) * 4) as u64, 0))
                            .collect::<Vec<_>>()
                            .as_mut_ptr(),
                        N as u32,
                    );

                    // The private segment is swizzled across the lanes a dword
                    // at a time, so an access is put together a byte at a time:
                    // an unaligned one runs on into the next dword's slot, which
                    // is a lane-stride away rather than next door.
                    let byte_pointer = |emitter: &mut Self, offset: llvm::prelude::LLVMValueRef| {
                        let dword = llvm::core::LLVMBuildAShr(
                            builder,
                            offset,
                            splat64(2),
                            empty_name.as_ptr(),
                        );
                        let dword = llvm::core::LLVMBuildMul(
                            builder,
                            dword,
                            splat64(128),
                            empty_name.as_ptr(),
                        );
                        let byte = llvm::core::LLVMBuildAnd(
                            builder,
                            offset,
                            splat64(3),
                            empty_name.as_ptr(),
                        );
                        let base = llvm::core::LLVMBuildInsertElement(
                            builder,
                            llvm::core::LLVMGetPoison(ty_i64xn),
                            emitter.scratch_base,
                            llvm::core::LLVMConstInt(ty_i32, 0, 0),
                            empty_name.as_ptr(),
                        );
                        let zero_vec = llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                            N as u32,
                        );
                        let base = llvm::core::LLVMBuildShuffleVector(
                            builder,
                            base,
                            llvm::core::LLVMGetPoison(ty_i64xn),
                            zero_vec,
                            empty_name.as_ptr(),
                        );
                        let address =
                            llvm::core::LLVMBuildAdd(builder, base, dword, empty_name.as_ptr());
                        let address = llvm::core::LLVMBuildAdd(
                            builder,
                            address,
                            lane_offset,
                            empty_name.as_ptr(),
                        );
                        let address =
                            llvm::core::LLVMBuildAdd(builder, address, byte, empty_name.as_ptr());
                        llvm::core::LLVMBuildIntToPtr(
                            builder,
                            address,
                            ty_p0xn,
                            empty_name.as_ptr(),
                        )
                    };

                    let words = bits.div_ceil(32);
                    for word in 0..words {
                        let piece = if bits < 32 { bits / 8 } else { 4 };
                        if is_load {
                            let mut value = llvm::core::LLVMConstVector(
                                [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                                N as u32,
                            );
                            for byte in 0..piece {
                                let offset = llvm::core::LLVMBuildAdd(
                                    builder,
                                    start,
                                    splat64((word * 4 + byte) as i64),
                                    empty_name.as_ptr(),
                                );
                                let ptr = byte_pointer(emitter, offset);
                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.gather.",
                                    &[ty_i8xn, ty_p0xn],
                                );
                                let data = intrinsic.emit_masked_call(
                                    ty_i8xn,
                                    &[ptr, mask, llvm::core::LLVMGetPoison(ty_i8xn)],
                                    0,
                                    1,
                                );
                                let data = llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildShl(
                                    builder,
                                    data,
                                    llvm::core::LLVMConstVector(
                                        [llvm::core::LLVMConstInt(ty_i32, (byte * 8) as u64, 0); N]
                                            .as_mut_ptr(),
                                        N as u32,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                value = llvm::core::LLVMBuildOr(
                                    builder,
                                    value,
                                    data,
                                    empty_name.as_ptr(),
                                );
                            }
                            let value = if bits >= 32 {
                                value
                            } else {
                                // A sub-dword load widens into the destination.
                                let narrow = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    value,
                                    llvm::core::LLVMVectorType(
                                        llvm::core::LLVMIntTypeInContext(context, bits as u32),
                                        N as u32,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                if signed {
                                    llvm::core::LLVMBuildSExt(
                                        builder,
                                        narrow,
                                        ty_i32xn,
                                        empty_name.as_ptr(),
                                    )
                                } else {
                                    llvm::core::LLVMBuildZExt(
                                        builder,
                                        narrow,
                                        ty_i32xn,
                                        empty_name.as_ptr(),
                                    )
                                }
                            };
                            emitter.emit_store_vgpr_u32xn::<N>(
                                inst.vdst as u32 + word as u32,
                                i as u32,
                                value,
                                mask,
                            );
                        } else {
                            let value = emitter.emit_load_vgpr_u32xn::<N>(
                                inst.vsrc as u32 + word as u32,
                                i as u32,
                                mask,
                            );
                            for byte in 0..piece {
                                let offset = llvm::core::LLVMBuildAdd(
                                    builder,
                                    start,
                                    splat64((word * 4 + byte) as i64),
                                    empty_name.as_ptr(),
                                );
                                let ptr = byte_pointer(emitter, offset);
                                let data = llvm::core::LLVMBuildLShr(
                                    builder,
                                    value,
                                    llvm::core::LLVMConstVector(
                                        [llvm::core::LLVMConstInt(ty_i32, (byte * 8) as u64, 0); N]
                                            .as_mut_ptr(),
                                        N as u32,
                                    ),
                                    empty_name.as_ptr(),
                                );
                                let data = llvm::core::LLVMBuildTrunc(
                                    builder,
                                    data,
                                    ty_i8xn,
                                    empty_name.as_ptr(),
                                );
                                let intrinsic = emitter.get_intrinsic_declaration(
                                    "llvm.masked.scatter.",
                                    &[ty_i8xn, ty_p0xn],
                                );
                                intrinsic.emit_masked_call(ty_void, &[data, ptr, mask], 1, 1);
                            }
                        }
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
            I::DS_BPERMUTE_B32 => {
                let emitter = self;
                let empty_name = std::ffi::CString::new("").unwrap();
                let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
                let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                const N: usize = SIMD_WIDTH;

                let true_mask = llvm::core::LLVMConstVector(
                    [llvm::core::LLVMConstInt(ty_i1, 1, 0); N].as_mut_ptr(),
                    N as u32,
                );

                let source_active = (0..32)
                    .map(|i| {
                        let elem = llvm::core::LLVMConstInt(ty_i32, i as u64, 0);
                        emitter.emit_exec_bit(elem)
                    })
                    .collect::<Vec<_>>();

                let source_values = [
                    emitter.emit_load_vgpr_u32xn::<N>(inst.data0 as u32, 0, true_mask),
                    emitter.emit_load_vgpr_u32xn::<N>(inst.data0 as u32, N as u32, true_mask),
                ];
                let exec_value = emitter.emit_load_sgpr_u32(126);

                for base in (0..32).step_by(N) {
                    let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, base);
                    let addr_values =
                        emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, base as u32, true_mask);
                    let mut result = llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                        N as u32,
                    );

                    for dst_lane in 0..N {
                        let index = llvm::core::LLVMConstInt(ty_i32, dst_lane as u64, 0);
                        let addr = llvm::core::LLVMBuildExtractElement(
                            builder,
                            addr_values,
                            index,
                            empty_name.as_ptr(),
                        );
                        let addr = llvm::core::LLVMBuildAdd(
                            builder,
                            addr,
                            llvm::core::LLVMConstInt(ty_i32, inst.offset0 as u64, 0),
                            empty_name.as_ptr(),
                        );
                        let lane = llvm::core::LLVMBuildLShr(
                            builder,
                            addr,
                            llvm::core::LLVMConstInt(ty_i32, 2, 0),
                            empty_name.as_ptr(),
                        );
                        let lane = llvm::core::LLVMBuildAnd(
                            builder,
                            lane,
                            llvm::core::LLVMConstInt(ty_i32, 31, 0),
                            empty_name.as_ptr(),
                        );

                        let mut value = llvm::core::LLVMConstInt(ty_i32, 0, 0);
                        for src_lane in 0..32 {
                            let is_lane = llvm::core::LLVMBuildICmp(
                                builder,
                                llvm::LLVMIntPredicate::LLVMIntEQ,
                                lane,
                                llvm::core::LLVMConstInt(ty_i32, src_lane as u64, 0),
                                empty_name.as_ptr(),
                            );
                            let is_lane_active = llvm::core::LLVMBuildAnd(
                                builder,
                                is_lane,
                                source_active[src_lane],
                                empty_name.as_ptr(),
                            );
                            let source = llvm::core::LLVMBuildExtractElement(
                                builder,
                                source_values[src_lane / N],
                                llvm::core::LLVMConstInt(ty_i32, (src_lane % N) as u64, 0),
                                empty_name.as_ptr(),
                            );
                            value = llvm::core::LLVMBuildSelect(
                                builder,
                                is_lane_active,
                                source,
                                value,
                                empty_name.as_ptr(),
                            );
                        }

                        result = llvm::core::LLVMBuildInsertElement(
                            builder,
                            result,
                            value,
                            index,
                            empty_name.as_ptr(),
                        );
                    }

                    emitter.emit_store_vgpr_u32xn::<N>(inst.vdst as u32, base as u32, result, mask);
                }
            }
            I::DS_STORE_B8
            | I::DS_STORE_B16
            | I::DS_STORE_B32
            | I::DS_STORE_B64
            | I::DS_STORE_B96
            | I::DS_STORE_B128
            | I::DS_STORE_2ADDR_B32
            | I::DS_STORE_2ADDR_B64 => {
                // What the instruction moves, as (register, byte offset within
                // LDS, width). A two-address form names its second register
                // explicitly and indexes its offsets by the size it moves; every
                // other form takes the whole 16-bit offset and runs on from it.
                let whole = ((inst.offset1 as u64) << 8) | (inst.offset0 as u64);
                let pieces: Vec<(u32, u64, u32)> = match inst.op {
                    I::DS_STORE_B8 => vec![(inst.data0 as u32, whole, 8)],
                    I::DS_STORE_B16 => vec![(inst.data0 as u32, whole, 16)],
                    I::DS_STORE_B32 => vec![(inst.data0 as u32, whole, 32)],
                    I::DS_STORE_B64 | I::DS_STORE_B96 | I::DS_STORE_B128 => {
                        let words = match inst.op {
                            I::DS_STORE_B64 => 2,
                            I::DS_STORE_B96 => 3,
                            _ => 4,
                        };
                        (0..words)
                            .map(|w| (inst.data0 as u32 + w, whole + (w as u64) * 4, 32))
                            .collect()
                    }
                    I::DS_STORE_2ADDR_B32 => vec![
                        (inst.data0 as u32, (inst.offset0 as u64) * 4, 32),
                        (inst.data1 as u32, (inst.offset1 as u64) * 4, 32),
                    ],
                    I::DS_STORE_2ADDR_B64 => vec![
                        (inst.data0 as u32, (inst.offset0 as u64) * 8, 32),
                        (inst.data0 as u32 + 1, (inst.offset0 as u64) * 8 + 4, 32),
                        (inst.data1 as u32, (inst.offset1 as u64) * 8, 32),
                        (inst.data1 as u32 + 1, (inst.offset1 as u64) * 8 + 4, 32),
                    ],
                    _ => unreachable!(),
                };

                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
                    let ty_void = llvm::core::LLVMVoidTypeInContext(context);

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let base = emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, i, mask);
                        let base =
                            llvm::core::LLVMBuildZExt(builder, base, ty_i64xn, empty_name.as_ptr());

                        for (reg, byte, bits) in pieces.iter().copied() {
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                base,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, byte, 0); N].as_mut_ptr(),
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

                            let value = emitter.emit_load_vgpr_u32xn::<N>(reg, i as u32, mask);
                            let ty_storexn = llvm::core::LLVMVectorType(
                                llvm::core::LLVMIntTypeInContext(context, bits),
                                N as u32,
                            );
                            let value = if bits == 32 {
                                value
                            } else {
                                llvm::core::LLVMBuildTrunc(
                                    builder,
                                    value,
                                    ty_storexn,
                                    empty_name.as_ptr(),
                                )
                            };

                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.scatter.",
                                &[ty_storexn, ty_p0xn],
                            );
                            intrinsic.emit_masked_call(ty_void, &[value, ptr, mask], 1, 1);
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

                        let base = emitter.emit_load_vgpr_u32(inst.addr as u32, elem);
                        let base =
                            llvm::core::LLVMBuildZExt(builder, base, ty_i64, empty_name.as_ptr());

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

                        for (reg, byte, bits) in pieces.iter().copied() {
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                base,
                                llvm::core::LLVMConstInt(ty_i64, byte, 0),
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

                            let data = emitter.emit_load_vgpr_u32(reg, elem);
                            let data = if bits == 32 {
                                data
                            } else {
                                llvm::core::LLVMBuildTrunc(
                                    builder,
                                    data,
                                    llvm::core::LLVMIntTypeInContext(context, bits),
                                    empty_name.as_ptr(),
                                )
                            };

                            llvm::core::LLVMBuildStore(builder, data, ptr);
                        }

                        llvm::core::LLVMBuildBr(builder, bb_cont);
                        llvm::core::LLVMPositionBuilderAtEnd(builder, bb_cont);
                        bb = bb_cont;
                    }
                }
            }
            I::DS_LOAD_U8
            | I::DS_LOAD_I8
            | I::DS_LOAD_U16
            | I::DS_LOAD_I16
            | I::DS_LOAD_B32
            | I::DS_LOAD_B64
            | I::DS_LOAD_B96
            | I::DS_LOAD_B128
            | I::DS_LOAD_2ADDR_B32
            | I::DS_LOAD_2ADDR_B64 => {
                // What the instruction moves, as (register, byte offset within
                // LDS, width). A two-address form names its second register
                // explicitly and indexes its offsets by the size it moves; every
                // other form takes the whole 16-bit offset and runs on from it.
                let whole = ((inst.offset1 as u64) << 8) | (inst.offset0 as u64);
                let pieces: Vec<(u32, u64, u32)> = match inst.op {
                    I::DS_LOAD_U8 | I::DS_LOAD_I8 => vec![(inst.vdst as u32, whole, 8)],
                    I::DS_LOAD_U16 | I::DS_LOAD_I16 => vec![(inst.vdst as u32, whole, 16)],
                    I::DS_LOAD_B32 => vec![(inst.vdst as u32, whole, 32)],
                    I::DS_LOAD_B64 | I::DS_LOAD_B96 | I::DS_LOAD_B128 => {
                        let words = match inst.op {
                            I::DS_LOAD_B64 => 2,
                            I::DS_LOAD_B96 => 3,
                            _ => 4,
                        };
                        (0..words)
                            .map(|w| (inst.vdst as u32 + w, whole + (w as u64) * 4, 32))
                            .collect()
                    }
                    I::DS_LOAD_2ADDR_B32 => vec![
                        (inst.vdst as u32, (inst.offset0 as u64) * 4, 32),
                        (inst.vdst as u32 + 1, (inst.offset1 as u64) * 4, 32),
                    ],
                    I::DS_LOAD_2ADDR_B64 => vec![
                        (inst.vdst as u32, (inst.offset0 as u64) * 8, 32),
                        (inst.vdst as u32 + 1, (inst.offset0 as u64) * 8 + 4, 32),
                        (inst.vdst as u32 + 2, (inst.offset1 as u64) * 8, 32),
                        (inst.vdst as u32 + 3, (inst.offset1 as u64) * 8 + 4, 32),
                    ],
                    _ => unreachable!(),
                };
                // A sub-dword load widens into the destination, signed or not
                // according to the opcode.
                let signed = matches!(inst.op, I::DS_LOAD_I8 | I::DS_LOAD_I16);

                if USE_SIMD {
                    let emitter = self;
                    let empty_name = std::ffi::CString::new("").unwrap();

                    const N: usize = SIMD_WIDTH;

                    let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
                    let ty_p0xn = llvm::core::LLVMVectorType(ty_p0, N as u32);
                    let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
                    let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
                    let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
                    let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
                    let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

                    let exec_value = emitter.emit_load_sgpr_u32(126);

                    for i in (0..32).step_by(N) {
                        let mask = emitter.emit_bits_to_mask_u32xn::<N>(exec_value, i as u32);

                        let base = emitter.emit_load_vgpr_u32xn::<N>(inst.addr as u32, i, mask);
                        let base =
                            llvm::core::LLVMBuildZExt(builder, base, ty_i64xn, empty_name.as_ptr());

                        for (reg, byte, bits) in pieces.iter().copied() {
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                base,
                                llvm::core::LLVMConstVector(
                                    [llvm::core::LLVMConstInt(ty_i64, byte, 0); N].as_mut_ptr(),
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

                            let ty_loadxn = llvm::core::LLVMVectorType(
                                llvm::core::LLVMIntTypeInContext(context, bits),
                                N as u32,
                            );
                            let intrinsic = emitter.get_intrinsic_declaration(
                                "llvm.masked.gather.",
                                &[ty_loadxn, ty_p0xn],
                            );
                            let data = intrinsic.emit_masked_call(
                                ty_loadxn,
                                &[ptr, mask, llvm::core::LLVMGetPoison(ty_loadxn)],
                                0,
                                1,
                            );
                            let data = if bits == 32 {
                                data
                            } else if signed {
                                llvm::core::LLVMBuildSExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32xn,
                                    empty_name.as_ptr(),
                                )
                            };

                            emitter.emit_store_vgpr_u32xn::<N>(reg, i as u32, data, mask);
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

                        let base = emitter.emit_load_vgpr_u32(inst.addr as u32, elem);
                        let base =
                            llvm::core::LLVMBuildZExt(builder, base, ty_i64, empty_name.as_ptr());

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

                        for (reg, byte, bits) in pieces.iter().copied() {
                            let offset = llvm::core::LLVMBuildAdd(
                                builder,
                                base,
                                llvm::core::LLVMConstInt(ty_i64, byte, 0),
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

                            let ty_load = llvm::core::LLVMIntTypeInContext(context, bits);
                            let data = llvm::core::LLVMBuildLoad2(
                                builder,
                                ty_load,
                                ptr,
                                empty_name.as_ptr(),
                            );
                            let data = if bits == 32 {
                                data
                            } else if signed {
                                llvm::core::LLVMBuildSExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                )
                            } else {
                                llvm::core::LLVMBuildZExt(
                                    builder,
                                    data,
                                    ty_i32,
                                    empty_name.as_ptr(),
                                )
                            };

                            emitter.emit_store_vgpr_u32(reg, elem, data);
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
            op => unimplemented!("{:?}", op),
        }

        bb
    }
}
