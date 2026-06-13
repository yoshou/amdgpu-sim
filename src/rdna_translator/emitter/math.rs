use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_abs_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty_f32]);
        let abs_value = intrinsic.emit_call(ty_f32, &[value]);

        abs_value
    }

    pub(crate) unsafe fn emit_abs_f64(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty_f64]);
        let abs_value = intrinsic.emit_call(ty_f64, &[value]);

        abs_value
    }

    pub(crate) unsafe fn emit_abs_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty_f32xn]);
        let abs_value = intrinsic.emit_call(ty_f32xn, &[value]);

        abs_value
    }

    pub(crate) unsafe fn emit_fract_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let intrinsic = self.get_intrinsic_declaration("llvm.frexp.", &[ty_f32, ty_i32]);
        let mut return_tys = vec![ty_f32, ty_i32];
        let frexp_value = intrinsic.emit_call(
            llvm::core::LLVMStructTypeInContext(
                context,
                return_tys.as_mut_ptr(),
                return_tys.len() as u32,
                0,
            ),
            &[value],
        );

        let fract_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 0, empty_name.as_ptr());
        fract_value
    }

    pub(crate) unsafe fn emit_exp_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let intrinsic = self.get_intrinsic_declaration("llvm.frexp.", &[ty_f32, ty_i32]);
        let mut return_tys = vec![ty_f32, ty_i32];
        let frexp_value = intrinsic.emit_call(
            llvm::core::LLVMStructTypeInContext(
                context,
                return_tys.as_mut_ptr(),
                return_tys.len() as u32,
                0,
            ),
            &[value],
        );

        let exp_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 1, empty_name.as_ptr());
        exp_value
    }

    pub(crate) unsafe fn emit_exp_f64(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let intrinsic = self.get_intrinsic_declaration("llvm.frexp.", &[ty_f64, ty_i32]);
        let mut return_tys = vec![ty_f64, ty_i32];
        let frexp_value = intrinsic.emit_call(
            llvm::core::LLVMStructTypeInContext(
                context,
                return_tys.as_mut_ptr(),
                return_tys.len() as u32,
                0,
            ),
            &[value],
        );

        let exp_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 1, empty_name.as_ptr());
        exp_value
    }

    pub(crate) unsafe fn _emit_exp_f64xn<const N: usize>(
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

        let intrinsic = self.get_intrinsic_declaration("llvm.frexp.", &[ty_f64xn, ty_i32xn]);
        let mut return_tys = vec![ty_f64xn, ty_i32xn];
        let frexp_value = intrinsic.emit_call(
            llvm::core::LLVMStructTypeInContext(
                context,
                return_tys.as_mut_ptr(),
                return_tys.len() as u32,
                0,
            ),
            &[value],
        );

        let exp_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 1, empty_name.as_ptr());
        exp_value
    }

    pub(crate) unsafe fn emit_fract_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.frexp.", &[ty_f32xn, ty_i32xn]);
        let mut return_tys = vec![ty_f32xn, ty_i32xn];
        let frexp_value = intrinsic.emit_call(
            llvm::core::LLVMStructTypeInContext(
                context,
                return_tys.as_mut_ptr(),
                return_tys.len() as u32,
                0,
            ),
            &[value],
        );

        let fract_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 0, empty_name.as_ptr());
        fract_value
    }

    pub(crate) unsafe fn emit_exp_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.frexp.", &[ty_f32xn, ty_i32xn]);
        let mut return_tys = vec![ty_f32xn, ty_i32xn];
        let frexp_value = intrinsic.emit_call(
            llvm::core::LLVMStructTypeInContext(
                context,
                return_tys.as_mut_ptr(),
                return_tys.len() as u32,
                0,
            ),
            &[value],
        );

        let exp_value =
            llvm::core::LLVMBuildExtractValue(builder, frexp_value, 1, empty_name.as_ptr());
        exp_value
    }

    pub(crate) unsafe fn emit_abs_neg_f32(
        &mut self,
        abs: u8,
        neg: u8,
        value: llvm::prelude::LLVMValueRef,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);

        let value = if (abs >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f32);
            self.emit_abs_f32(value)
        } else {
            value
        };

        let value = if (neg >> idx) & 1 != 0 {
            assert!(llvm::core::LLVMTypeOf(value) == ty_f32);
            llvm::core::LLVMBuildFNeg(builder, value, empty_name.as_ptr())
        } else {
            value
        };

        value
    }

    pub(crate) unsafe fn emit_abs_neg_f64(
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
            self.emit_abs_f64(value)
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

    pub(crate) unsafe fn emit_abs_neg_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        abs: u8,
        neg: u8,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = if (abs >> idx) & 1 != 0 {
            self.emit_abs_f32xn::<N>(value)
        } else {
            value
        };

        let value = if (neg >> idx) & 1 != 0 {
            llvm::core::LLVMBuildFNeg(builder, value, empty_name.as_ptr())
        } else {
            value
        };

        value
    }

    pub(crate) unsafe fn emit_abs_neg_f64xn<const N: usize>(
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

    pub(crate) unsafe fn emit_omod_clamp(
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

            let intrinsic = self.get_intrinsic_declaration("llvm.minnum.", &[ty_f64]);
            let min_value = intrinsic.emit_call(ty_f64, &[value, one]);

            let intrinsic = self.get_intrinsic_declaration("llvm.maxnum.", &[ty_f64]);
            let max_value = intrinsic.emit_call(ty_f64, &[min_value, zero]);

            max_value
        } else {
            value
        };

        value
    }

    pub(crate) unsafe fn emit_omod_clamp_f64xn<const N: usize>(
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

            let intrinsic = self.get_intrinsic_declaration("llvm.minnum.", &[ty_f64xn]);
            let min_value = intrinsic.emit_call(ty_f64xn, &[value, one]);

            let intrinsic = self.get_intrinsic_declaration("llvm.maxnum.", &[ty_f64xn]);
            let max_value = intrinsic.emit_call(ty_f64xn, &[min_value, zero]);

            max_value
        } else {
            value
        };

        value
    }

    pub(crate) unsafe fn emit_fma_f32xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
        value2: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.fma.", &[ty_f32xn]);
        let fma_value = intrinsic.emit_call(ty_f32xn, &[value0, value1, value2]);
        fma_value
    }

    pub(crate) unsafe fn emit_fma_f64xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
        value2: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.fma.", &[ty_f64xn]);
        let fma_value = intrinsic.emit_call(ty_f64xn, &[value0, value1, value2]);
        fma_value
    }

    pub(crate) unsafe fn emit_fadd(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let add_value = llvm::core::LLVMBuildFAdd(builder, value0, value1, empty_name.as_ptr());
        add_value
    }

    pub(crate) unsafe fn emit_u32_to_f64xn<const N: usize>(
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

    pub(crate) unsafe fn _emit_i32_to_f64xn<const N: usize>(
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

    pub(crate) unsafe fn emit_exp2_f32xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.exp2.", &[ty_f32xn]);
        let exp2_value = intrinsic.emit_call(ty_f32xn, &[value0]);
        exp2_value
    }

    pub(crate) unsafe fn _emit_exp2_f64xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.exp2.", &[ty_f64xn]);
        let exp2_value = intrinsic.emit_call(ty_f64xn, &[value0]);
        exp2_value
    }

    pub(crate) unsafe fn _emit_ldexp_f64xn<const N: usize>(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);

        let intrinsic = self.get_intrinsic_declaration("llvm.ldexp.", &[ty_f64xn, ty_i32xn]);
        let ldexp_value = intrinsic.emit_call(ty_f64xn, &[value0, value1]);
        ldexp_value
    }

    // x86 has no vector lowering for llvm.ldexp/llvm.exp2, so those scalarize
    // into per-lane scalbn libcalls. Compute x * 2^n inline instead, as three
    // clamped power-of-two multiplies so overflow, underflow and denormals
    // still round correctly over the full i32 exponent range.
    pub(crate) unsafe fn emit_ldexp_f64xn<const N: usize>(
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
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

        let splat_i32 = |v: i64| {
            llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstInt(ty_i32, v as u64, 1); N].as_mut_ptr(),
                N as u32,
            )
        };
        let splat_i64 = |v: u64| {
            llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstInt(ty_i64, v, 0); N].as_mut_ptr(),
                N as u32,
            )
        };

        let smin = self.get_intrinsic_declaration("llvm.smin.", &[ty_i32xn]);
        let smax = self.get_intrinsic_declaration("llvm.smax.", &[ty_i32xn]);

        let mut result = value0;
        let mut remaining = value1;

        for _ in 0..3 {
            let step = smin.emit_call(ty_i32xn, &[remaining, splat_i32(1023)]);
            let step = smax.emit_call(ty_i32xn, &[step, splat_i32(-1022)]);
            remaining =
                llvm::core::LLVMBuildSub(builder, remaining, step, empty_name.as_ptr());

            let step = llvm::core::LLVMBuildSExt(builder, step, ty_i64xn, empty_name.as_ptr());
            let biased =
                llvm::core::LLVMBuildAdd(builder, step, splat_i64(1023), empty_name.as_ptr());
            let bits =
                llvm::core::LLVMBuildShl(builder, biased, splat_i64(52), empty_name.as_ptr());
            let scale =
                llvm::core::LLVMBuildBitCast(builder, bits, ty_f64xn, empty_name.as_ptr());

            result = llvm::core::LLVMBuildFMul(builder, result, scale, empty_name.as_ptr());
        }

        result
    }

    pub(crate) unsafe fn emit_ldexp_f32(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let intrinsic = self.get_intrinsic_declaration("llvm.ldexp.", &[ty_f32, ty_i32]);
        let ldexp_value = intrinsic.emit_call(ty_f32, &[value0, value1]);
        ldexp_value
    }

    pub(crate) unsafe fn emit_fmul(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let mul_value = llvm::core::LLVMBuildFMul(builder, value0, value1, empty_name.as_ptr());
        mul_value
    }

    pub(crate) unsafe fn emit_concat_pair(
        &mut self,
        values: &Vec<llvm::prelude::LLVMValueRef>,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let context = self.context;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let len = values.len() as u32;

        let mut result = Vec::new();
        for i in (0..len).step_by(2) {
            let value1 = values[i as usize];
            let value2 = values[i as usize + 1];

            let value1_len = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(value1));
            let value2_len = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(value2));

            let mut index_values = Vec::new();
            for i in 0..(value1_len + value2_len) {
                index_values.push(llvm::core::LLVMConstInt(ty_i32, i as u64, 0));
            }

            let indices =
                llvm::core::LLVMConstVector(index_values.as_mut_ptr(), index_values.len() as u32);

            let cmp_value = llvm::core::LLVMBuildShuffleVector(
                builder,
                value1,
                value2,
                indices,
                empty_name.as_ptr(),
            );
            result.push(cmp_value);
        }
        result
    }

    pub(crate) unsafe fn emit_concat<const N: usize>(
        &mut self,
        values: &Vec<llvm::prelude::LLVMValueRef>,
    ) -> llvm::prelude::LLVMValueRef {
        let mut len = values.len() as u32;
        let mut values = values.clone();
        while len > 1 {
            let new_values = self.emit_concat_pair(&values);
            values = new_values;
            len = values.len() as u32;
        }
        values[0]
    }

    pub(crate) unsafe fn emit_split<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> Vec<llvm::prelude::LLVMValueRef> {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let context = self.context;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);

        let len = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(value)) as usize;

        let mut values = Vec::new();
        for i in (0..len).step_by(N) {
            let mut index_values = Vec::new();
            for j in 0..N {
                index_values.push(llvm::core::LLVMConstInt(ty_i32, (i + j) as u64, 0));
            }
            let indices =
                llvm::core::LLVMConstVector(index_values.as_mut_ptr(), index_values.len() as u32);
            let value = llvm::core::LLVMBuildShuffleVector(
                builder,
                value,
                llvm::core::LLVMGetUndef(llvm::core::LLVMTypeOf(value)),
                indices,
                empty_name.as_ptr(),
            );
            values.push(value);
        }
        values
    }
}
