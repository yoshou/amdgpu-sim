use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_scalar_source_operand_u32(
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
            SourceOperand::FloatConstant(value) => {
                llvm::core::LLVMConstInt(ty_i32, f32::to_bits(*value as f32) as u64, 0)
            }
            SourceOperand::ScalarRegister(value) => self.emit_load_sgpr_u32(*value as u32),
            _ => panic!("Unsupported source operand type"),
        }
    }

    pub(crate) unsafe fn emit_scalar_source_operand_u64(
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
            SourceOperand::PrivateBase => self.scratch_base,
            _ => panic!("Unsupported source operand type: {:?}", operand),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_u32(
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
            SourceOperand::FloatConstant(value) => {
                llvm::core::LLVMConstInt(ty_i32, f32::to_bits(*value as f32) as u64, 0)
            }
            SourceOperand::ScalarRegister(value) => self.emit_load_sgpr_u32(*value as u32),
            SourceOperand::VectorRegister(value) => self.emit_load_vgpr_u32(*value as u32, elem),
            _ => panic!("Unsupported source operand type"),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f32(
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
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f32_vec(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                vec![llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value) as f64); 8]
            }
            SourceOperand::IntegerConstant(value) => {
                vec![llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value as u32) as f64); 8]
            }
            SourceOperand::FloatConstant(value) => {
                vec![llvm::core::LLVMConstReal(ty_f32, *value); 8]
            }
            SourceOperand::ScalarRegister(value) => {
                let mut values = Vec::new();
                for i in 0..8 {
                    let value = self.emit_load_sgpr_u32(*value as u32 + i);
                    let value =
                        llvm::core::LLVMBuildBitCast(builder, value, ty_f32, empty_name.as_ptr());
                    values.push(value);
                }
                values
            }
            SourceOperand::VectorRegister(value) => {
                let mut values = Vec::new();
                for i in 0..8 {
                    let value = self.emit_load_vgpr_u32(*value as u32 + i, elem);
                    let value = llvm::core::LLVMBuildBitCast(
                        self.builder,
                        value,
                        ty_f32,
                        empty_name.as_ptr(),
                    );
                    values.push(value);
                }
                values
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f16(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(self.context);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value) as f64)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value as u32) as f64)
            }
            SourceOperand::FloatConstant(value) => llvm::core::LLVMConstReal(ty_f16, *value),
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);
                let value = llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                llvm::core::LLVMBuildBitCast(builder, value, ty_f16, empty_name.as_ptr())
            }
            SourceOperand::VectorRegister(value) => {
                let value = self.emit_load_vgpr_u32(*value as u32, elem);
                let value = llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                llvm::core::LLVMBuildBitCast(self.builder, value, ty_f16, empty_name.as_ptr())
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_hi_f16(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(self.context);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(self.context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value) as f64)
            }
            SourceOperand::IntegerConstant(value) => {
                llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value as u32) as f64)
            }
            SourceOperand::FloatConstant(value) => llvm::core::LLVMConstReal(ty_f16, *value),
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);
                let value = llvm::core::LLVMBuildLShr(
                    builder,
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                    empty_name.as_ptr(),
                );
                let value = llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                llvm::core::LLVMBuildBitCast(builder, value, ty_f16, empty_name.as_ptr())
            }
            SourceOperand::VectorRegister(value) => {
                let value = self.emit_load_vgpr_u32(*value as u32, elem);
                let value = llvm::core::LLVMBuildLShr(
                    builder,
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                    empty_name.as_ptr(),
                );
                let value = llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                llvm::core::LLVMBuildBitCast(self.builder, value, ty_f16, empty_name.as_ptr())
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f16_vec(
        &mut self,
        operand: &SourceOperand,
        elem: llvm::prelude::LLVMValueRef,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(self.context);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(self.context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                vec![llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value) as f64); 8]
            }
            SourceOperand::IntegerConstant(value) => {
                vec![llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value as u32) as f64); 8]
            }
            SourceOperand::FloatConstant(value) => {
                vec![llvm::core::LLVMConstReal(ty_f16, *value); 8]
            }
            SourceOperand::ScalarRegister(value) => {
                let mut values = Vec::new();
                for i in 0..4 {
                    let value = self.emit_load_sgpr_u32(*value as u32 + i);
                    let value_lo =
                        llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                    let value_hi = llvm::core::LLVMBuildLShr(
                        builder,
                        value,
                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                        empty_name.as_ptr(),
                    );
                    let value_hi =
                        llvm::core::LLVMBuildTrunc(builder, value_hi, ty_i16, empty_name.as_ptr());
                    let value_lo = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_lo,
                        ty_f16,
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_hi,
                        ty_f16,
                        empty_name.as_ptr(),
                    );
                    values.push(value_lo);
                    values.push(value_hi);
                }
                values
            }
            SourceOperand::VectorRegister(value) => {
                let mut values = Vec::new();
                for i in 0..4 {
                    let value = self.emit_load_vgpr_u32(*value as u32 + i, elem);
                    let value_lo =
                        llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                    let value_hi = llvm::core::LLVMBuildLShr(
                        builder,
                        value,
                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                        empty_name.as_ptr(),
                    );
                    let value_hi =
                        llvm::core::LLVMBuildTrunc(builder, value_hi, ty_i16, empty_name.as_ptr());
                    let value_lo = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_lo,
                        ty_f16,
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_hi,
                        ty_f16,
                        empty_name.as_ptr(),
                    );
                    values.push(value_lo);
                    values.push(value_hi);
                }
                values
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_u64(
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
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f64(
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
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_bit_mask_u32xn<const N: usize>(
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

    pub(crate) unsafe fn emit_bits_to_mask_u32xn<const N: usize>(
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

    pub(crate) unsafe fn emit_vector_source_operand_u64xn<const N: usize>(
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
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f64xn<const N: usize>(
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
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_u32xn<const N: usize>(
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
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f32xn<const N: usize>(
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
                self.emit_load_vgpr_f32xn::<N>(*value as u32, elem, mask)
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f32xn_vec<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
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
                vec![value; 8]
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f32, f32::from_bits(*value as u32) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                vec![value; 8]
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f32, *value);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                vec![value; 8]
            }
            SourceOperand::ScalarRegister(value) => {
                let mut values = Vec::new();
                for i in 0..8 {
                    let value = self.emit_load_sgpr_u32(*value as u32 + i);
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
                    values.push(value);
                }
                values
            }
            SourceOperand::VectorRegister(value) => {
                let mut values = Vec::new();
                for i in 0..8 {
                    let value = self.emit_load_vgpr_f32xn::<N>(*value as u32 + i, elem, mask);
                    values.push(value);
                }
                values
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f16xn<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(self.context);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(self.context);
        let ty_f16xn = llvm::core::LLVMVectorType(ty_f16, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_f16xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value as u32) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, *value);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);
                let value = llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                let value =
                    llvm::core::LLVMBuildBitCast(builder, value, ty_f16, empty_name.as_ptr());

                let value = llvm::core::LLVMBuildInsertElement(
                    builder,
                    llvm::core::LLVMGetUndef(ty_f16xn),
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
                self.emit_load_vgpr_f16xn::<N>(*value as u32, elem, mask)
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_hi_f16xn<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(self.context);
        let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(self.context);
        let ty_f16xn = llvm::core::LLVMVectorType(ty_f16, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_f16xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value as u32) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, *value);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                value
            }
            SourceOperand::ScalarRegister(value) => {
                let value = self.emit_load_sgpr_u32(*value as u32);
                let value = llvm::core::LLVMBuildLShr(
                    builder,
                    value,
                    llvm::core::LLVMConstInt(ty_i32, 16, 0),
                    empty_name.as_ptr(),
                );
                let value = llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                let value =
                    llvm::core::LLVMBuildBitCast(builder, value, ty_f16, empty_name.as_ptr());

                let value = llvm::core::LLVMBuildInsertElement(
                    builder,
                    llvm::core::LLVMGetUndef(ty_f16xn),
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
                let value = self.emit_load_vgpr_u32xn::<N>(*value as u32, elem, mask);
                let value = llvm::core::LLVMBuildLShr(
                    builder,
                    value,
                    llvm::core::LLVMConstVector(
                        [llvm::core::LLVMConstInt(ty_i32, 16, 0); N].as_mut_ptr(),
                        N as u32,
                    ),
                    empty_name.as_ptr(),
                );
                let value =
                    llvm::core::LLVMBuildTrunc(builder, value, ty_i16xn, empty_name.as_ptr());
                let value =
                    llvm::core::LLVMBuildBitCast(builder, value, ty_f16xn, empty_name.as_ptr());
                value
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    pub(crate) unsafe fn emit_vector_source_operand_f16xn_vec<const N: usize>(
        &mut self,
        operand: &SourceOperand,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(self.context);
        let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(self.context);
        let ty_f16xn = llvm::core::LLVMVectorType(ty_f16, N as u32);

        let zero_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
            N as u32,
        );
        let poison = llvm::core::LLVMGetPoison(ty_f16xn);

        match operand {
            SourceOperand::LiteralConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                vec![value; 8]
            }
            SourceOperand::IntegerConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, f32::from_bits(*value as u32) as f64);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                vec![value; 8]
            }
            SourceOperand::FloatConstant(value) => {
                let value = llvm::core::LLVMConstReal(ty_f16, *value);
                let value = llvm::core::LLVMConstVector([value; N].as_mut_ptr(), N as u32);
                vec![value; 8]
            }
            SourceOperand::ScalarRegister(value) => {
                let mut values = Vec::new();
                for i in 0..4 {
                    let value = self.emit_load_sgpr_u32(*value as u32 + i);
                    let value_lo =
                        llvm::core::LLVMBuildTrunc(builder, value, ty_i16, empty_name.as_ptr());
                    let value_lo = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_lo,
                        ty_f16,
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildLShr(
                        builder,
                        value,
                        llvm::core::LLVMConstInt(ty_i32, 16, 0),
                        empty_name.as_ptr(),
                    );
                    let value_hi =
                        llvm::core::LLVMBuildTrunc(builder, value_hi, ty_i16, empty_name.as_ptr());
                    let value_hi = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_hi,
                        ty_f16,
                        empty_name.as_ptr(),
                    );

                    let value_lo = llvm::core::LLVMBuildInsertElement(
                        builder,
                        llvm::core::LLVMGetUndef(ty_f16xn),
                        value_lo,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );

                    let value_lo = llvm::core::LLVMBuildShuffleVector(
                        builder,
                        value_lo,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    );

                    let value_hi = llvm::core::LLVMBuildInsertElement(
                        builder,
                        llvm::core::LLVMGetUndef(ty_f16xn),
                        value_hi,
                        llvm::core::LLVMConstInt(ty_i32, 0, 0),
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildShuffleVector(
                        builder,
                        value_hi,
                        poison,
                        zero_vec,
                        empty_name.as_ptr(),
                    );
                    values.push(value_lo);
                    values.push(value_hi);
                }
                values
            }
            SourceOperand::VectorRegister(value) => {
                let mut values = Vec::new();
                for i in 0..4 {
                    let value = self.emit_load_vgpr_u32xn::<N>(*value as u32 + i, elem, mask);

                    let value_lo =
                        llvm::core::LLVMBuildTrunc(builder, value, ty_i16xn, empty_name.as_ptr());
                    let value_lo = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_lo,
                        ty_f16xn,
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildLShr(
                        builder,
                        value,
                        llvm::core::LLVMConstVector(
                            [llvm::core::LLVMConstInt(ty_i32, 16, 0); N].as_mut_ptr(),
                            N as u32,
                        ),
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildTrunc(
                        builder,
                        value_hi,
                        ty_i16xn,
                        empty_name.as_ptr(),
                    );
                    let value_hi = llvm::core::LLVMBuildBitCast(
                        builder,
                        value_hi,
                        ty_f16xn,
                        empty_name.as_ptr(),
                    );

                    values.push(value_lo);
                    values.push(value_hi);
                }
                values
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }
}
