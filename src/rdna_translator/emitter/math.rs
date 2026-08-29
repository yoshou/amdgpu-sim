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
        // FREXP splits the operand; this wrapper keeps the significand.
        self.emit_frexp(value).0
    }

    pub(crate) unsafe fn emit_exp_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        // FREXP splits the operand; this wrapper keeps the exponent.
        self.emit_frexp(value).1
    }

    pub(crate) unsafe fn _emit_exp_f64(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        // FREXP splits the operand; this wrapper keeps the exponent.
        self.emit_frexp(value).1
    }

    pub(crate) unsafe fn _emit_exp_f64xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        // FREXP splits the operand; this wrapper keeps the exponent.
        self.emit_frexp(value).1
    }

    pub(crate) unsafe fn emit_fract_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        // FREXP splits the operand; this wrapper keeps the significand.
        self.emit_frexp(value).0
    }

    pub(crate) unsafe fn emit_exp_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        // FREXP splits the operand; this wrapper keeps the exponent.
        self.emit_frexp(value).1
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

        let sign_mask_vec = llvm::core::LLVMConstVector(
            [llvm::core::LLVMConstInt(ty_i64, 0x8000000000000000, 0); N].as_mut_ptr(),
            N as u32,
        );

        let value = if (abs >> idx) & 1 != 0 {
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i64xn, empty_name.as_ptr());
            let value = llvm::core::LLVMBuildAnd(
                builder,
                value,
                llvm::core::LLVMBuildNot(builder, sign_mask_vec, empty_name.as_ptr()),
                empty_name.as_ptr(),
            );
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_f64xn, empty_name.as_ptr());
            value
        } else {
            value
        };

        let value = if (neg >> idx) & 1 != 0 {
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i64xn, empty_name.as_ptr());
            let value =
                llvm::core::LLVMBuildXor(builder, value, sign_mask_vec, empty_name.as_ptr());
            let value = llvm::core::LLVMBuildBitCast(builder, value, ty_f64xn, empty_name.as_ptr());
            value
        } else {
            value
        };

        value
    }

    /// The ISA requires a FRACT result below 1.0, but `x - floor(x)` rounds up to
    /// exactly 1.0 for a tiny negative x, so it is held at the value just below.
    pub(crate) unsafe fn emit_fract_below_one(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let ty = llvm::core::LLVMTypeOf(value);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let elem_ty = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let below_one = llvm::core::LLVMConstReal(elem_ty, f64::from_bits(0x3FEF_FFFF_FFFF_FFFF));
        let limit = if is_vector {
            let lanes = llvm::core::LLVMGetVectorSize(ty);
            llvm::core::LLVMConstVector(vec![below_one; lanes as usize].as_mut_ptr(), lanes)
        } else {
            below_one
        };
        // A NaN is not below the limit and must stay a NaN, which minnum would
        // instead replace with the limit.
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let at_limit = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOGE,
            value,
            limit,
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildSelect(builder, at_limit, limit, value, empty_name.as_ptr())
    }

    /// x - rint(x) is +0 even where x is -0, so the sign of a zero operand has to
    /// be put back before sin() can carry it into the result. A negative whole
    /// turn is not a zero operand, and the hardware returns +0 for it.
    pub(crate) unsafe fn emit_keep_turn_sign(
        &mut self,
        reduced: llvm::prelude::LLVMValueRef,
        operand: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(reduced);
        let intrinsic = self.get_intrinsic_declaration("llvm.copysign.", &[ty]);
        let signed = intrinsic.emit_call(ty, &[reduced, operand]);
        let zero = {
            let elem_ty =
                if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
                    llvm::core::LLVMGetElementType(ty)
                } else {
                    ty
                };
            let constant = llvm::core::LLVMConstReal(elem_ty, 0.0);
            if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };
        let is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            operand,
            zero,
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildSelect(builder, is_zero, signed, reduced, empty_name.as_ptr())
    }

    /// The sine unit returns exact results at the quarter turns: zero at a half
    /// turn and +-1 at a quarter turn, whatever the library sine of the scaled
    /// angle would round to. A whole turn keeps the sign the library sine gives.
    pub(crate) unsafe fn emit_sin_exact_turns(
        &mut self,
        turns: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(value);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let elem_ty = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let splat = |v: f64| {
            let constant = llvm::core::LLVMConstReal(elem_ty, v);
            if is_vector {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let magnitude = intrinsic.emit_call(ty, &[turns]);

        let mut result = value;
        for (probe, exact, on_magnitude) in
            [(0.5, 0.0, true), (0.25, 1.0, false), (-0.25, -1.0, false)]
        {
            let subject = if on_magnitude { magnitude } else { turns };
            let is_probe = llvm::core::LLVMBuildFCmp(
                builder,
                llvm::LLVMRealPredicate::LLVMRealOEQ,
                subject,
                splat(probe),
                empty_name.as_ptr(),
            );
            result = llvm::core::LLVMBuildSelect(
                builder,
                is_probe,
                splat(exact),
                result,
                empty_name.as_ptr(),
            );
        }
        result
    }

    /// The hardware subtracts by negating and adding: a NaN operand comes through
    /// quieted, and the negated one comes through with its sign flipped.
    pub(crate) unsafe fn emit_sub_f32(
        &mut self,
        a: llvm::prelude::LLVMValueRef,
        b: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let context = self.context;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(a);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let int_ty = if is_vector {
            llvm::core::LLVMVectorType(ty_i32, llvm::core::LLVMGetVectorSize(ty))
        } else {
            ty_i32
        };

        let negated = llvm::core::LLVMBuildFNeg(builder, b, empty_name.as_ptr());
        let sum = llvm::core::LLVMBuildFAdd(builder, a, negated, empty_name.as_ptr());

        let quiet = |value: llvm::prelude::LLVMValueRef| {
            let bits = llvm::core::LLVMBuildBitCast(builder, value, int_ty, empty_name.as_ptr());
            let quiet_bit = llvm::core::LLVMConstInt(ty_i32, 0x0040_0000, 0);
            let quiet_bit = if is_vector {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![quiet_bit; lanes as usize].as_mut_ptr(), lanes)
            } else {
                quiet_bit
            };
            let bits = llvm::core::LLVMBuildOr(builder, bits, quiet_bit, empty_name.as_ptr());
            llvm::core::LLVMBuildBitCast(builder, bits, ty, empty_name.as_ptr())
        };

        let quiet_negated = quiet(negated);
        let is_nan_negated = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            negated,
            negated,
            empty_name.as_ptr(),
        );
        let result = llvm::core::LLVMBuildSelect(
            builder,
            is_nan_negated,
            quiet_negated,
            sum,
            empty_name.as_ptr(),
        );

        let quiet_a = quiet(a);
        let is_nan_a = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            a,
            a,
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildSelect(builder, is_nan_a, quiet_a, result, empty_name.as_ptr())
    }

    /// The logarithm of a negative operand is the negative quiet NaN, not the
    /// positive one a library log2 returns.
    pub(crate) unsafe fn emit_negative_log_nan(
        &mut self,
        operand: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let context = self.context;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(value);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let splat = |bits: u64, elem_ty: llvm::prelude::LLVMTypeRef| {
            let constant = llvm::core::LLVMConstInt(elem_ty, bits, 0);
            if is_vector {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };
        let nan = llvm::core::LLVMBuildBitCast(
            builder,
            splat(0xFFC0_0000, ty_i32),
            ty,
            empty_name.as_ptr(),
        );

        let zero = {
            let elem_ty = if is_vector {
                llvm::core::LLVMGetElementType(ty)
            } else {
                ty
            };
            let constant = llvm::core::LLVMConstReal(elem_ty, 0.0);
            if is_vector {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };
        let is_negative = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOLT,
            operand,
            zero,
            empty_name.as_ptr(),
        );
        llvm::core::LLVMBuildSelect(builder, is_negative, nan, value, empty_name.as_ptr())
    }

    /// The significand and exponent of a float, as FREXP defines them: the
    /// significand lies in [0.5, 1) with the operand's sign, and a zero, an
    /// infinity or a NaN comes back unchanged with a zero exponent. A denormal
    /// is normalised, which is why this counts leading zeros rather than reading
    /// the exponent field alone.
    pub(crate) unsafe fn emit_frexp(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> (llvm::prelude::LLVMValueRef, llvm::prelude::LLVMValueRef) {
        let builder = self.builder;
        let context = self.context;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(value);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let lanes = if is_vector {
            llvm::core::LLVMGetVectorSize(ty)
        } else {
            0
        };
        let elem_ty = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let is_double =
            llvm::core::LLVMGetTypeKind(elem_ty) == llvm::LLVMTypeKind::LLVMDoubleTypeKind;

        let (bits, frac_bits, exp_mask, half_exp) = if is_double {
            (64u32, 52u32, 0x7FFu64, 1022u64)
        } else {
            (32, 23, 0xFF, 126)
        };
        let int_elem = llvm::core::LLVMIntTypeInContext(context, bits);
        let int_ty = if is_vector {
            llvm::core::LLVMVectorType(int_elem, lanes)
        } else {
            int_elem
        };
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let i32_ty = if is_vector {
            llvm::core::LLVMVectorType(ty_i32, lanes)
        } else {
            ty_i32
        };
        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);

        let splat = |elem: llvm::prelude::LLVMTypeRef, v: u64| {
            let constant = llvm::core::LLVMConstInt(elem, v, 0);
            if is_vector {
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };
        let build = |op: unsafe extern "C" fn(
            llvm::prelude::LLVMBuilderRef,
            llvm::prelude::LLVMValueRef,
            llvm::prelude::LLVMValueRef,
            *const std::os::raw::c_char,
        ) -> llvm::prelude::LLVMValueRef,
                     a,
                     b| op(builder, a, b, empty_name.as_ptr());

        let raw = llvm::core::LLVMBuildBitCast(builder, value, int_ty, empty_name.as_ptr());
        let sign = build(
            llvm::core::LLVMBuildAnd,
            raw,
            splat(int_elem, 1u64 << (bits - 1)),
        );
        let frac = build(
            llvm::core::LLVMBuildAnd,
            raw,
            splat(int_elem, (1u64 << frac_bits) - 1),
        );
        let exp_field = build(
            llvm::core::LLVMBuildAnd,
            build(
                llvm::core::LLVMBuildLShr,
                raw,
                splat(int_elem, frac_bits as u64),
            ),
            splat(int_elem, exp_mask),
        );

        // The significand keeps the fraction and takes the exponent of 0.5.
        let half = splat(int_elem, half_exp << frac_bits);
        let mant_normal = build(
            llvm::core::LLVMBuildOr,
            build(llvm::core::LLVMBuildOr, sign, half),
            frac,
        );
        let exp_normal = build(
            llvm::core::LLVMBuildSub,
            llvm::core::LLVMBuildTrunc(builder, exp_field, i32_ty, empty_name.as_ptr()),
            splat(ty_i32, half_exp),
        );

        // A denormal is shifted up until its highest set bit reaches the
        // significand's, and the exponent counts how far it moved.
        let intrinsic = self.get_intrinsic_declaration("llvm.ctlz.", &[int_ty]);
        let leading = intrinsic.emit_call(int_ty, &[frac, llvm::core::LLVMConstInt(ty_i1, 0, 0)]);
        let shift = build(
            llvm::core::LLVMBuildSub,
            leading,
            splat(int_elem, (bits - frac_bits - 1) as u64),
        );
        let frac_denorm = build(
            llvm::core::LLVMBuildAnd,
            build(llvm::core::LLVMBuildShl, frac, shift),
            splat(int_elem, (1u64 << frac_bits) - 1),
        );
        let mant_denorm = build(
            llvm::core::LLVMBuildOr,
            build(llvm::core::LLVMBuildOr, sign, half),
            frac_denorm,
        );
        let exp_denorm = build(
            llvm::core::LLVMBuildSub,
            splat(
                ty_i32,
                ((bits as i64) - (half_exp as i64) - (frac_bits as i64)) as u64,
            ),
            llvm::core::LLVMBuildTrunc(builder, leading, i32_ty, empty_name.as_ptr()),
        );

        let is_denorm = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            exp_field,
            splat(int_elem, 0),
            empty_name.as_ptr(),
        );

        let mant = llvm::core::LLVMBuildSelect(
            builder,
            is_denorm,
            mant_denorm,
            mant_normal,
            empty_name.as_ptr(),
        );
        let exp = llvm::core::LLVMBuildSelect(
            builder,
            is_denorm,
            exp_denorm,
            exp_normal,
            empty_name.as_ptr(),
        );

        // A zero, an infinity and a NaN come through unchanged.
        let magnitude = build(
            llvm::core::LLVMBuildAnd,
            raw,
            splat(int_elem, (1u64 << (bits - 1)) - 1),
        );
        let is_zero = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            magnitude,
            splat(int_elem, 0),
            empty_name.as_ptr(),
        );
        let is_special = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            exp_field,
            splat(int_elem, exp_mask),
            empty_name.as_ptr(),
        );
        let untouched = llvm::core::LLVMBuildOr(builder, is_zero, is_special, empty_name.as_ptr());

        // A signalling NaN comes back quiet.
        let quieted = build(
            llvm::core::LLVMBuildOr,
            raw,
            splat(int_elem, 1u64 << (frac_bits - 1)),
        );
        let is_nan = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntUGT,
            magnitude,
            splat(int_elem, exp_mask << frac_bits),
            empty_name.as_ptr(),
        );
        let raw = llvm::core::LLVMBuildSelect(builder, is_nan, quieted, raw, empty_name.as_ptr());
        let mant = llvm::core::LLVMBuildSelect(builder, untouched, raw, mant, empty_name.as_ptr());
        let mant = llvm::core::LLVMBuildBitCast(builder, mant, ty, empty_name.as_ptr());
        let exp = llvm::core::LLVMBuildSelect(
            builder,
            untouched,
            splat(ty_i32, 0),
            exp,
            empty_name.as_ptr(),
        );
        (mant, exp)
    }

    /// An integer constant shaped like `like`, which is either a scalar or a
    /// vector of the same integer type.
    pub(crate) unsafe fn emit_splat_u32(
        &mut self,
        value: u64,
        like: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let ty = llvm::core::LLVMTypeOf(like);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let elem_ty = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let constant = llvm::core::LLVMConstInt(elem_ty, value, 0);
        if is_vector {
            let lanes = llvm::core::LLVMGetVectorSize(ty);
            llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
        } else {
            constant
        }
    }

    /// ABS and NEG act on the sign bit of the operand, whatever the instruction
    /// reads it as, so an integer compare sees them too.
    pub(crate) unsafe fn emit_abs_neg_bits(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        abs: u8,
        neg: u8,
        idx: u32,
    ) -> llvm::prelude::LLVMValueRef {
        if ((abs >> idx) & 1) == 0 && ((neg >> idx) & 1) == 0 {
            return value;
        }
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(value);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let elem_ty = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let width = llvm::core::LLVMGetIntTypeWidth(elem_ty);
        let splat = |bits: u64| {
            let constant = llvm::core::LLVMConstInt(elem_ty, bits, 0);
            if is_vector {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };
        let sign = 1u64 << (width - 1);

        let mut result = value;
        if ((abs >> idx) & 1) != 0 {
            let mask = splat(!sign & ((1u128 << width) - 1) as u64);
            result = llvm::core::LLVMBuildAnd(builder, result, mask, empty_name.as_ptr());
        }
        if ((neg >> idx) & 1) != 0 {
            let mask = splat(sign);
            result = llvm::core::LLVMBuildXor(builder, result, mask, empty_name.as_ptr());
        }
        result
    }

    /// The transcendental unit flushes denormals on input and on output, whatever
    /// the mode register says (ISA §V_RCP_F32, §V_SQRT_F32, §V_RSQ_F32). The
    /// value keeps its sign, so a negative denormal becomes -0.
    pub(crate) unsafe fn emit_ftz_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(value);
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let elem_ty = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let splat = |v: f64| {
            let constant = llvm::core::LLVMConstReal(elem_ty, v);
            if is_vector {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let magnitude = intrinsic.emit_call(ty, &[value]);
        let smallest = splat(f32::MIN_POSITIVE as f64);
        let is_denorm = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOLT,
            magnitude,
            smallest,
            empty_name.as_ptr(),
        );

        let intrinsic = self.get_intrinsic_declaration("llvm.copysign.", &[ty]);
        let zero = intrinsic.emit_call(ty, &[splat(0.0), value]);

        llvm::core::LLVMBuildSelect(builder, is_denorm, zero, value, empty_name.as_ptr())
    }

    /// The VOP3 output modifiers: OMOD scales the result by 1, 2, 4 or 1/2, and
    /// CLAMP then holds it in [0,1]. OMOD is a two-bit enum here, unlike the
    /// per-half selector the packed helpers below take.
    pub(crate) unsafe fn emit_vop3_omod_clamp(
        &mut self,
        omod: u8,
        clamp: u8,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty = llvm::core::LLVMTypeOf(value);
        let elem_ty = if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };

        let splat = |v: f64| {
            let constant = llvm::core::LLVMConstReal(elem_ty, v);
            if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
                let lanes = llvm::core::LLVMGetVectorSize(ty);
                llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
            } else {
                constant
            }
        };

        let value = match omod {
            1 => {
                let scale = splat(2.0);
                llvm::core::LLVMBuildFMul(builder, value, scale, empty_name.as_ptr())
            }
            2 => {
                let scale = splat(4.0);
                llvm::core::LLVMBuildFMul(builder, value, scale, empty_name.as_ptr())
            }
            3 => {
                let scale = splat(0.5);
                llvm::core::LLVMBuildFMul(builder, value, scale, empty_name.as_ptr())
            }
            _ => value,
        };

        if clamp == 0 {
            return value;
        }
        let zero = splat(0.0);
        let one = splat(1.0);
        // CLAMP turns a NaN into zero, which minnum/maxnum would instead drop.
        let is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            value,
            value,
            empty_name.as_ptr(),
        );
        let intrinsic = self.get_intrinsic_declaration("llvm.minnum.", &[ty]);
        let clamped = intrinsic.emit_call(ty, &[value, one]);
        let intrinsic = self.get_intrinsic_declaration("llvm.maxnum.", &[ty]);
        let clamped = intrinsic.emit_call(ty, &[clamped, zero]);
        // CLAMP holds the result inside [0.0, 1.0], and the zero it lands on is
        // the positive one. Which zero maxnum answers with is not specified, so
        // the sign is settled here rather than left to the target.
        let clamped = llvm::core::LLVMBuildFAdd(builder, clamped, zero, empty_name.as_ptr());
        llvm::core::LLVMBuildSelect(builder, is_nan, zero, clamped, empty_name.as_ptr())
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

    /// Splat a floating point constant to the shape and element type of `ty`.
    unsafe fn const_fp_like(
        &mut self,
        ty: llvm::prelude::LLVMTypeRef,
        value: f64,
    ) -> llvm::prelude::LLVMValueRef {
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let ty_elem = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let constant = llvm::core::LLVMConstReal(ty_elem, value);

        if is_vector {
            let lanes = llvm::core::LLVMGetVectorSize(ty);
            llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
        } else {
            constant
        }
    }

    /// Splat an i32 constant to the shape of `ty` (scalar or vector of i32).
    unsafe fn const_i32_like(
        &mut self,
        ty: llvm::prelude::LLVMTypeRef,
        value: i32,
    ) -> llvm::prelude::LLVMValueRef {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let constant = llvm::core::LLVMConstInt(ty_i32, value as u64, 1);

        if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            let lanes = llvm::core::LLVMGetVectorSize(ty);
            llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
        } else {
            constant
        }
    }

    /// True when `ty` is an f64, or a vector of f64.
    unsafe fn is_double_like(&mut self, ty: llvm::prelude::LLVMTypeRef) -> bool {
        let ty_elem = if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };

        llvm::core::LLVMGetTypeKind(ty_elem) == llvm::LLVMTypeKind::LLVMDoubleTypeKind
    }

    /// Float to signed integer conversion with RDNA saturation semantics:
    /// out-of-range inputs (including infinity) saturate and NaN converts to 0.
    ///
    /// The operand is clamped into the destination range before the conversion
    /// because plain `fptosi` is poison outside it. `llvm.fptosi.sat` has the
    /// semantics we want but scalarizes into one conversion per lane, so the
    /// clamp is open coded to keep the sequence packed.
    pub(crate) unsafe fn emit_fp_to_si_sat(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        ty_dst: llvm::prelude::LLVMTypeRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty_src = llvm::core::LLVMTypeOf(value);
        let is_double = self.is_double_like(ty_src);

        // An f64 holds every i32 exactly, so clamping alone saturates. An f32
        // cannot represent i32::MAX, so it is clamped to the largest float below
        // 2^31 and the saturated case is restored afterwards.
        let upper = if is_double {
            2147483647.0
        } else {
            2147483520.0
        };

        let bound = self.const_fp_like(ty_src, upper);
        let intrinsic = self.get_intrinsic_declaration("llvm.minnum.", &[ty_src]);
        let clamped = intrinsic.emit_call(ty_src, &[value, bound]);

        let bound = self.const_fp_like(ty_src, -2147483648.0);
        let intrinsic = self.get_intrinsic_declaration("llvm.maxnum.", &[ty_src]);
        let clamped = intrinsic.emit_call(ty_src, &[clamped, bound]);

        let is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            value,
            value,
            empty_name.as_ptr(),
        );
        let zero = self.const_fp_like(ty_src, 0.0);
        let clamped =
            llvm::core::LLVMBuildSelect(builder, is_nan, zero, clamped, empty_name.as_ptr());

        let d_value = llvm::core::LLVMBuildFPToSI(builder, clamped, ty_dst, empty_name.as_ptr());

        if is_double {
            return d_value;
        }

        let bound = self.const_fp_like(ty_src, 2147483648.0);
        let overflows = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOGE,
            value,
            bound,
            empty_name.as_ptr(),
        );
        let saturated = self.const_i32_like(ty_dst, i32::MAX);

        llvm::core::LLVMBuildSelect(builder, overflows, saturated, d_value, empty_name.as_ptr())
    }

    /// Float to unsigned integer conversion with RDNA saturation semantics.
    pub(crate) unsafe fn emit_fp_to_ui_sat(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        ty_dst: llvm::prelude::LLVMTypeRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty_src = llvm::core::LLVMTypeOf(value);
        let is_double = self.is_double_like(ty_src);

        // maxnum returns the operand that is not NaN, which also maps NaN to 0.
        let zero = self.const_fp_like(ty_src, 0.0);
        let intrinsic = self.get_intrinsic_declaration("llvm.maxnum.", &[ty_src]);
        let clamped = intrinsic.emit_call(ty_src, &[value, zero]);

        let upper = if is_double {
            4294967295.0
        } else {
            4294967040.0
        };
        let bound = self.const_fp_like(ty_src, upper);
        let intrinsic = self.get_intrinsic_declaration("llvm.minnum.", &[ty_src]);
        let clamped = intrinsic.emit_call(ty_src, &[clamped, bound]);

        let d_value = llvm::core::LLVMBuildFPToUI(builder, clamped, ty_dst, empty_name.as_ptr());

        if is_double {
            return d_value;
        }

        let bound = self.const_fp_like(ty_src, 4294967296.0);
        let overflows = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOGE,
            value,
            bound,
            empty_name.as_ptr(),
        );
        let saturated = self.const_i32_like(ty_dst, -1);

        llvm::core::LLVMBuildSelect(builder, overflows, saturated, d_value, empty_name.as_ptr())
    }

    /// Integer type with the same shape and width as the float type `ty`.
    unsafe fn int_type_like(
        &mut self,
        ty: llvm::prelude::LLVMTypeRef,
    ) -> llvm::prelude::LLVMTypeRef {
        let ty_int = if self.is_double_like(ty) {
            llvm::core::LLVMInt64TypeInContext(self.context)
        } else {
            llvm::core::LLVMInt32TypeInContext(self.context)
        };

        if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            llvm::core::LLVMVectorType(ty_int, llvm::core::LLVMGetVectorSize(ty))
        } else {
            ty_int
        }
    }

    /// Splat an integer constant to the shape and width of the integer type `ty`.
    unsafe fn const_int_like(
        &mut self,
        ty: llvm::prelude::LLVMTypeRef,
        value: i64,
    ) -> llvm::prelude::LLVMValueRef {
        let is_vector = llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind;
        let ty_elem = if is_vector {
            llvm::core::LLVMGetElementType(ty)
        } else {
            ty
        };
        let constant = llvm::core::LLVMConstInt(ty_elem, value as u64, 1);

        if is_vector {
            let lanes = llvm::core::LLVMGetVectorSize(ty);
            llvm::core::LLVMConstVector(vec![constant; lanes as usize].as_mut_ptr(), lanes)
        } else {
            constant
        }
    }

    /// RDNA computes a subtract as S0 + (-S1), so a NaN coming from the second
    /// operand is propagated with its sign bit flipped. LLVM folds the negation
    /// into the subtract and x86 then returns that NaN unchanged, so the sign is
    /// restored here. A NaN in the first operand, and a NaN created by the
    /// operation itself, already agree.
    pub(crate) unsafe fn emit_sub_nan_sign(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        minuend: llvm::prelude::LLVMValueRef,
        subtrahend: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(value);
        let ty_int = self.int_type_like(ty);

        let minuend_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            minuend,
            minuend,
            empty_name.as_ptr(),
        );
        let subtrahend_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            subtrahend,
            subtrahend,
            empty_name.as_ptr(),
        );
        let flips = llvm::core::LLVMBuildAnd(
            builder,
            subtrahend_is_nan,
            llvm::core::LLVMBuildNot(builder, minuend_is_nan, empty_name.as_ptr()),
            empty_name.as_ptr(),
        );

        let sign_bit = if self.is_double_like(ty) {
            i64::MIN
        } else {
            i32::MIN as i64
        };
        let sign = llvm::core::LLVMBuildSelect(
            builder,
            flips,
            self.const_int_like(ty_int, sign_bit),
            self.const_int_like(ty_int, 0),
            empty_name.as_ptr(),
        );

        let bits = llvm::core::LLVMBuildBitCast(builder, value, ty_int, empty_name.as_ptr());
        let bits = llvm::core::LLVMBuildXor(builder, bits, sign, empty_name.as_ptr());

        llvm::core::LLVMBuildBitCast(builder, bits, ty, empty_name.as_ptr())
    }

    /// f64 type with the same shape as the value type `ty`.
    unsafe fn double_type_like(
        &mut self,
        ty: llvm::prelude::LLVMTypeRef,
    ) -> llvm::prelude::LLVMTypeRef {
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);

        if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            llvm::core::LLVMVectorType(ty_f64, llvm::core::LLVMGetVectorSize(ty))
        } else {
            ty_f64
        }
    }

    /// Biased exponent field of an f32 value, as an integer of matching shape.
    unsafe fn emit_exponent_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(value);
        let ty_int = self.int_type_like(ty);

        let bits = llvm::core::LLVMBuildBitCast(builder, value, ty_int, empty_name.as_ptr());
        let shift = self.const_i32_like(ty_int, 23);
        let shifted = llvm::core::LLVMBuildLShr(builder, bits, shift, empty_name.as_ptr());
        let mask = self.const_i32_like(ty_int, 0xff);

        llvm::core::LLVMBuildAnd(builder, shifted, mask, empty_name.as_ptr())
    }

    /// Biased exponent field of an f64 value, as an i32. This is what the ISA
    /// pseudo code calls `exponent()`; `llvm.frexp` returns a different
    /// normalization and must not be used where the ISA compares against the raw
    /// field.
    pub(crate) unsafe fn emit_exponent_f64(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

        let bits = llvm::core::LLVMBuildBitCast(builder, value, ty_i64, empty_name.as_ptr());
        let shifted = llvm::core::LLVMBuildLShr(
            builder,
            bits,
            llvm::core::LLVMConstInt(ty_i64, 52, 0),
            empty_name.as_ptr(),
        );
        let exponent = llvm::core::LLVMBuildAnd(
            builder,
            shifted,
            llvm::core::LLVMConstInt(ty_i64, 0x7ff, 0),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildTrunc(builder, exponent, ty_i32, empty_name.as_ptr())
    }

    /// `V_DIV_FIXUP_F32`: apply the division corner cases of the RDNA ISA to a
    /// quotient produced by the reciprocal/Newton-Raphson macro. Operands follow
    /// the ISA order (S0 quotient, S1 denominator, S2 numerator) and may be
    /// scalar or vector f32 values.
    pub(crate) unsafe fn emit_div_fixup_f32(
        &mut self,
        quotient: llvm::prelude::LLVMValueRef,
        denominator: llvm::prelude::LLVMValueRef,
        numerator: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(quotient);
        let ty_int = self.int_type_like(ty);

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_quotient = intrinsic.emit_call(ty, &[quotient]);
        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_denominator = intrinsic.emit_call(ty, &[denominator]);
        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_numerator = intrinsic.emit_call(ty, &[numerator]);

        let denominator_bits =
            llvm::core::LLVMBuildBitCast(builder, denominator, ty_int, empty_name.as_ptr());
        let numerator_bits =
            llvm::core::LLVMBuildBitCast(builder, numerator, ty_int, empty_name.as_ptr());

        // sign_out = sign(S1) ^ sign(S2)
        let sign_bits = llvm::core::LLVMBuildXor(
            builder,
            denominator_bits,
            numerator_bits,
            empty_name.as_ptr(),
        );
        let zero_int = self.const_i32_like(ty_int, 0);
        let sign_out = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLT,
            sign_bits,
            zero_int,
            empty_name.as_ptr(),
        );

        let zero = self.const_fp_like(ty, 0.0);
        let infinity = self.const_fp_like(ty, f64::INFINITY);

        let denominator_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            denominator,
            denominator,
            empty_name.as_ptr(),
        );
        let numerator_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            numerator,
            numerator,
            empty_name.as_ptr(),
        );
        let denominator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            denominator,
            zero,
            empty_name.as_ptr(),
        );
        let numerator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            numerator,
            zero,
            empty_name.as_ptr(),
        );
        let denominator_is_inf = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            abs_denominator,
            infinity,
            empty_name.as_ptr(),
        );
        let numerator_is_inf = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            abs_numerator,
            infinity,
            empty_name.as_ptr(),
        );

        // A NaN input is propagated quiet.
        let quiet_bit = self.const_i32_like(ty_int, 0x0040_0000);
        let quiet_denominator = llvm::core::LLVMBuildBitCast(
            builder,
            llvm::core::LLVMBuildOr(builder, denominator_bits, quiet_bit, empty_name.as_ptr()),
            ty,
            empty_name.as_ptr(),
        );
        let quiet_numerator = llvm::core::LLVMBuildBitCast(
            builder,
            llvm::core::LLVMBuildOr(builder, numerator_bits, quiet_bit, empty_name.as_ptr()),
            ty,
            empty_name.as_ptr(),
        );

        // 0/0 and inf/inf produce the canonical negative quiet NaN.
        let canonical_nan = llvm::core::LLVMBuildBitCast(
            builder,
            self.const_i32_like(ty_int, 0xffc00000u32 as i32),
            ty,
            empty_name.as_ptr(),
        );

        let negative_infinity = self.const_fp_like(ty, f64::NEG_INFINITY);
        let signed_inf = llvm::core::LLVMBuildSelect(
            builder,
            sign_out,
            negative_infinity,
            infinity,
            empty_name.as_ptr(),
        );
        let negative_zero = self.const_fp_like(ty, -0.0);
        let signed_zero = llvm::core::LLVMBuildSelect(
            builder,
            sign_out,
            negative_zero,
            zero,
            empty_name.as_ptr(),
        );

        // Underflow rounds to a signed zero and overflow to a signed infinity
        // under the default round-to-nearest-even mode.
        let denominator_exponent = self.emit_exponent_f32(denominator);
        let numerator_exponent = self.emit_exponent_f32(numerator);
        let exponent_delta = llvm::core::LLVMBuildSub(
            builder,
            numerator_exponent,
            denominator_exponent,
            empty_name.as_ptr(),
        );
        let underflow_bound = self.const_i32_like(ty_int, -150);
        let underflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLT,
            exponent_delta,
            underflow_bound,
            empty_name.as_ptr(),
        );
        let overflow_bound = self.const_i32_like(ty_int, 255);
        let overflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            denominator_exponent,
            overflow_bound,
            empty_name.as_ptr(),
        );

        let negated = llvm::core::LLVMBuildFNeg(builder, abs_quotient, empty_name.as_ptr());
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            sign_out,
            negated,
            abs_quotient,
            empty_name.as_ptr(),
        );

        // A NaN quotient means the division overflowed, which the fixup turns
        // into an infinity of the quotient's sign rather than passing it on.
        let quotient_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            quotient,
            quotient,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            quotient_is_nan,
            signed_inf,
            d_value,
            empty_name.as_ptr(),
        );

        // The selects are applied in reverse so that earlier ISA cases win.
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            overflows,
            signed_inf,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            underflows,
            signed_zero,
            d_value,
            empty_name.as_ptr(),
        );
        // x/inf, 0/y
        let to_zero = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_inf,
            numerator_is_zero,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            to_zero,
            signed_zero,
            d_value,
            empty_name.as_ptr(),
        );
        // x/0, inf/y
        let to_inf = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_zero,
            numerator_is_inf,
            empty_name.as_ptr(),
        );
        let d_value =
            llvm::core::LLVMBuildSelect(builder, to_inf, signed_inf, d_value, empty_name.as_ptr());
        // inf/inf
        let inf_over_inf = llvm::core::LLVMBuildAnd(
            builder,
            denominator_is_inf,
            numerator_is_inf,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            inf_over_inf,
            canonical_nan,
            d_value,
            empty_name.as_ptr(),
        );
        // 0/0
        let zero_over_zero = llvm::core::LLVMBuildAnd(
            builder,
            denominator_is_zero,
            numerator_is_zero,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            zero_over_zero,
            canonical_nan,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            denominator_is_nan,
            quiet_denominator,
            d_value,
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildSelect(
            builder,
            numerator_is_nan,
            quiet_numerator,
            d_value,
            empty_name.as_ptr(),
        )
    }

    /// `V_DIV_FIXUP_F64`: apply the division corner cases of the RDNA ISA to a
    /// quotient produced by the reciprocal/Newton-Raphson macro. Operands follow
    /// the ISA order (S0 quotient, S1 denominator, S2 numerator) and may be
    /// scalar or vector f32 values.
    pub(crate) unsafe fn emit_div_fixup_f64(
        &mut self,
        quotient: llvm::prelude::LLVMValueRef,
        denominator: llvm::prelude::LLVMValueRef,
        numerator: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(quotient);
        let ty_int = self.int_type_like(ty);

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_quotient = intrinsic.emit_call(ty, &[quotient]);
        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_denominator = intrinsic.emit_call(ty, &[denominator]);
        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_numerator = intrinsic.emit_call(ty, &[numerator]);

        let denominator_bits =
            llvm::core::LLVMBuildBitCast(builder, denominator, ty_int, empty_name.as_ptr());
        let numerator_bits =
            llvm::core::LLVMBuildBitCast(builder, numerator, ty_int, empty_name.as_ptr());

        // sign_out = sign(S1) ^ sign(S2)
        let sign_bits = llvm::core::LLVMBuildXor(
            builder,
            denominator_bits,
            numerator_bits,
            empty_name.as_ptr(),
        );
        let zero_int = self.const_int_like(ty_int, 0);
        let sign_out = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLT,
            sign_bits,
            zero_int,
            empty_name.as_ptr(),
        );

        let zero = self.const_fp_like(ty, 0.0);
        let infinity = self.const_fp_like(ty, f64::INFINITY);

        let denominator_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            denominator,
            denominator,
            empty_name.as_ptr(),
        );
        let numerator_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            numerator,
            numerator,
            empty_name.as_ptr(),
        );
        let denominator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            denominator,
            zero,
            empty_name.as_ptr(),
        );
        let numerator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            numerator,
            zero,
            empty_name.as_ptr(),
        );
        let denominator_is_inf = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            abs_denominator,
            infinity,
            empty_name.as_ptr(),
        );
        let numerator_is_inf = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            abs_numerator,
            infinity,
            empty_name.as_ptr(),
        );

        // A NaN input is propagated quiet.
        let quiet_bit = self.const_int_like(ty_int, 0x0008_0000_0000_0000);
        let quiet_denominator = llvm::core::LLVMBuildBitCast(
            builder,
            llvm::core::LLVMBuildOr(builder, denominator_bits, quiet_bit, empty_name.as_ptr()),
            ty,
            empty_name.as_ptr(),
        );
        let quiet_numerator = llvm::core::LLVMBuildBitCast(
            builder,
            llvm::core::LLVMBuildOr(builder, numerator_bits, quiet_bit, empty_name.as_ptr()),
            ty,
            empty_name.as_ptr(),
        );

        // 0/0 and inf/inf produce the canonical negative quiet NaN.
        let canonical_nan = llvm::core::LLVMBuildBitCast(
            builder,
            self.const_int_like(ty_int, 0xfff8_0000_0000_0000u64 as i64),
            ty,
            empty_name.as_ptr(),
        );

        let negative_infinity = self.const_fp_like(ty, f64::NEG_INFINITY);
        let signed_inf = llvm::core::LLVMBuildSelect(
            builder,
            sign_out,
            negative_infinity,
            infinity,
            empty_name.as_ptr(),
        );
        let negative_zero = self.const_fp_like(ty, -0.0);
        let signed_zero = llvm::core::LLVMBuildSelect(
            builder,
            sign_out,
            negative_zero,
            zero,
            empty_name.as_ptr(),
        );

        // Underflow rounds to a signed zero and overflow to a signed infinity
        // under the default round-to-nearest-even mode.
        let exponent_shift = self.const_int_like(ty_int, 52);
        let exponent_mask = self.const_int_like(ty_int, 0x7FF);
        let denominator_exponent = llvm::core::LLVMBuildAnd(
            builder,
            llvm::core::LLVMBuildLShr(
                builder,
                denominator_bits,
                exponent_shift,
                empty_name.as_ptr(),
            ),
            exponent_mask,
            empty_name.as_ptr(),
        );
        let numerator_exponent = llvm::core::LLVMBuildAnd(
            builder,
            llvm::core::LLVMBuildLShr(builder, numerator_bits, exponent_shift, empty_name.as_ptr()),
            exponent_mask,
            empty_name.as_ptr(),
        );
        let exponent_delta = llvm::core::LLVMBuildSub(
            builder,
            numerator_exponent,
            denominator_exponent,
            empty_name.as_ptr(),
        );
        let underflow_bound = self.const_int_like(ty_int, -1075);
        let underflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLT,
            exponent_delta,
            underflow_bound,
            empty_name.as_ptr(),
        );
        let overflow_bound = self.const_int_like(ty_int, 2047);
        let overflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            denominator_exponent,
            overflow_bound,
            empty_name.as_ptr(),
        );

        let negated = llvm::core::LLVMBuildFNeg(builder, abs_quotient, empty_name.as_ptr());
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            sign_out,
            negated,
            abs_quotient,
            empty_name.as_ptr(),
        );

        // A NaN quotient means the division overflowed, which the fixup turns
        // into an infinity of the quotient's sign rather than passing it on.
        let quotient_is_nan = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealUNO,
            quotient,
            quotient,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            quotient_is_nan,
            signed_inf,
            d_value,
            empty_name.as_ptr(),
        );

        // The selects are applied in reverse so that earlier ISA cases win.
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            overflows,
            signed_inf,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            underflows,
            signed_zero,
            d_value,
            empty_name.as_ptr(),
        );
        // x/inf, 0/y
        let to_zero = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_inf,
            numerator_is_zero,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            to_zero,
            signed_zero,
            d_value,
            empty_name.as_ptr(),
        );
        // x/0, inf/y
        let to_inf = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_zero,
            numerator_is_inf,
            empty_name.as_ptr(),
        );
        let d_value =
            llvm::core::LLVMBuildSelect(builder, to_inf, signed_inf, d_value, empty_name.as_ptr());
        // inf/inf
        let inf_over_inf = llvm::core::LLVMBuildAnd(
            builder,
            denominator_is_inf,
            numerator_is_inf,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            inf_over_inf,
            canonical_nan,
            d_value,
            empty_name.as_ptr(),
        );
        // 0/0
        let zero_over_zero = llvm::core::LLVMBuildAnd(
            builder,
            denominator_is_zero,
            numerator_is_zero,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            zero_over_zero,
            canonical_nan,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            denominator_is_nan,
            quiet_denominator,
            d_value,
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildSelect(
            builder,
            numerator_is_nan,
            quiet_numerator,
            d_value,
            empty_name.as_ptr(),
        )
    }

    /// `V_DIV_FMAS_F32`: fused multiply-add that post-scales the quotient when
    /// `V_DIV_SCALE_F32` reported that an operand was scaled.
    ///
    /// The ISA document states a fixed `2**32` factor, but gfx1201 hardware
    /// applies `2**+-64` and picks the direction from the magnitude of the
    /// addend: the quotient estimate is at least 2.0 only when the denominator
    /// was scaled up, and below 2.0 when the numerator was. The scale is part of
    /// the same rounding step as the multiply-add, so it is computed in f64 and
    /// rounded once; rounding the f32 result first and scaling afterwards is a
    /// double rounding and lands 1 ULP off once the result is subnormal.
    pub(crate) unsafe fn emit_div_fmas_f32(
        &mut self,
        value0: llvm::prelude::LLVMValueRef,
        value1: llvm::prelude::LLVMValueRef,
        value2: llvm::prelude::LLVMValueRef,
        condition: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(value2);

        let intrinsic = self.get_intrinsic_declaration("llvm.fma.", &[ty]);
        let fma_value = intrinsic.emit_call(ty, &[value0, value1, value2]);

        let intrinsic = self.get_intrinsic_declaration("llvm.fabs.", &[ty]);
        let abs_value2 = intrinsic.emit_call(ty, &[value2]);
        let threshold = self.const_fp_like(ty, 2.0);
        let scales_up = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOGE,
            abs_value2,
            threshold,
            empty_name.as_ptr(),
        );

        let ty_wide = self.double_type_like(ty);
        let value0_wide = llvm::core::LLVMBuildFPExt(builder, value0, ty_wide, empty_name.as_ptr());
        let value1_wide = llvm::core::LLVMBuildFPExt(builder, value1, ty_wide, empty_name.as_ptr());
        let value2_wide = llvm::core::LLVMBuildFPExt(builder, value2, ty_wide, empty_name.as_ptr());

        let intrinsic = self.get_intrinsic_declaration("llvm.fma.", &[ty_wide]);
        let fma_wide = intrinsic.emit_call(ty_wide, &[value0_wide, value1_wide, value2_wide]);

        let scale_up = self.const_fp_like(ty_wide, 18446744073709551616.0);
        let scale_down = self.const_fp_like(ty_wide, 5.421010862427522e-20);
        let factor = llvm::core::LLVMBuildSelect(
            builder,
            scales_up,
            scale_up,
            scale_down,
            empty_name.as_ptr(),
        );

        let scaled_wide = llvm::core::LLVMBuildFMul(builder, fma_wide, factor, empty_name.as_ptr());
        let scaled_value =
            llvm::core::LLVMBuildFPTrunc(builder, scaled_wide, ty, empty_name.as_ptr());

        llvm::core::LLVMBuildSelect(
            builder,
            condition,
            scaled_value,
            fma_value,
            empty_name.as_ptr(),
        )
    }

    /// The two halves of a packed pair of half-precision values, widened to
    /// f32: the one in the low 16 bits first.
    pub(crate) unsafe fn emit_f16_to_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> (llvm::prelude::LLVMValueRef, llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty_i16xn = llvm::core::LLVMVectorType(
            llvm::core::LLVMInt16TypeInContext(context),
            N as u32,
        );
        let ty_f16xn = llvm::core::LLVMVectorType(
            llvm::core::LLVMHalfTypeInContext(context),
            N as u32,
        );
        let ty_f32xn = llvm::core::LLVMVectorType(
            llvm::core::LLVMFloatTypeInContext(context),
            N as u32,
        );
        let ty_i32xn = llvm::core::LLVMTypeOf(value);

        let mut widen = |bits: llvm::prelude::LLVMValueRef| {
            let narrow = llvm::core::LLVMBuildTrunc(builder, bits, ty_i16xn, empty_name.as_ptr());
            let half = llvm::core::LLVMBuildBitCast(builder, narrow, ty_f16xn, empty_name.as_ptr());
            llvm::core::LLVMBuildFPExt(builder, half, ty_f32xn, empty_name.as_ptr())
        };

        let low = widen(value);
        let shift = self.const_int_like(ty_i32xn, 16);
        let high = widen(llvm::core::LLVMBuildLShr(
            builder,
            value,
            shift,
            empty_name.as_ptr(),
        ));
        (low, high)
    }

    /// The same for brain floats, which are the top half of an f32.
    pub(crate) unsafe fn emit_bf16_to_f32xn<const N: usize>(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
    ) -> (llvm::prelude::LLVMValueRef, llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty_f32xn = llvm::core::LLVMVectorType(
            llvm::core::LLVMFloatTypeInContext(context),
            N as u32,
        );
        let ty_i32xn = llvm::core::LLVMTypeOf(value);

        let shift = self.const_int_like(ty_i32xn, 16);
        let low_bits = llvm::core::LLVMBuildShl(builder, value, shift, empty_name.as_ptr());
        let high_mask = self.const_int_like(ty_i32xn, 0xFFFF_0000u32 as i32 as i64);
        let high_bits = llvm::core::LLVMBuildAnd(builder, value, high_mask, empty_name.as_ptr());

        (
            llvm::core::LLVMBuildBitCast(builder, low_bits, ty_f32xn, empty_name.as_ptr()),
            llvm::core::LLVMBuildBitCast(builder, high_bits, ty_f32xn, empty_name.as_ptr()),
        )
    }

    /// `V_DIV_SCALE_F32`: scale an operand of the division macro so that no
    /// subnormal terms appear during the Newton-Raphson correction. Operands
    /// follow the ISA order (S0 value to scale, S1 denominator, S2 numerator).
    /// Returns the scaled value and the VCC mask that tells `V_DIV_FMAS_F32`
    /// whether the quotient needs post-scaling.
    ///
    /// The branch conditions were measured on gfx1201 hardware over a sweep of
    /// the whole exponent range; they differ from the ISA pseudo code in that
    /// the quotient-underflow case triggers on the exponent difference,
    /// symmetric with the overflow case, and that the reciprocal-is-subnormal
    /// test is an f32 test, not f64.
    pub(crate) unsafe fn emit_div_scale_f32(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        denominator: llvm::prelude::LLVMValueRef,
        numerator: llvm::prelude::LLVMValueRef,
    ) -> (llvm::prelude::LLVMValueRef, llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(value);
        let ty_int = self.int_type_like(ty);
        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
        let ty_bool = if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            llvm::core::LLVMVectorType(ty_i1, llvm::core::LLVMGetVectorSize(ty))
        } else {
            ty_i1
        };
        let true_value = llvm::core::LLVMConstAllOnes(ty_bool);
        let false_value = llvm::core::LLVMConstNull(ty_bool);

        let zero = self.const_fp_like(ty, 0.0);

        let denominator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            denominator,
            zero,
            empty_name.as_ptr(),
        );
        let numerator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            numerator,
            zero,
            empty_name.as_ptr(),
        );
        let returns_nan = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_zero,
            numerator_is_zero,
            empty_name.as_ptr(),
        );

        let denominator_exponent = self.emit_exponent_f32(denominator);
        let numerator_exponent = self.emit_exponent_f32(numerator);
        let exponent_delta = llvm::core::LLVMBuildSub(
            builder,
            numerator_exponent,
            denominator_exponent,
            empty_name.as_ptr(),
        );

        // The quotient leaves the representable range in either direction.
        let overflow_bound = self.const_i32_like(ty_int, 96);
        let quotient_overflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSGE,
            exponent_delta,
            overflow_bound,
            empty_name.as_ptr(),
        );
        let underflow_bound = self.const_i32_like(ty_int, -96);
        let quotient_underflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLE,
            exponent_delta,
            underflow_bound,
            empty_name.as_ptr(),
        );
        // 1.0 / S1 is subnormal, so the reciprocal loses precision.
        let reciprocal_bound = self.const_i32_like(ty_int, 253);
        let reciprocal_is_denorm = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSGE,
            denominator_exponent,
            reciprocal_bound,
            empty_name.as_ptr(),
        );
        // S1 is subnormal, or the numerator is so small that the residual terms
        // of the correction would be subnormal.
        let zero_exponent = self.const_i32_like(ty_int, 0);
        let denominator_is_denorm = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            denominator_exponent,
            zero_exponent,
            empty_name.as_ptr(),
        );
        let tiny_bound = self.const_i32_like(ty_int, 23);
        let numerator_is_tiny = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLE,
            numerator_exponent,
            tiny_bound,
            empty_name.as_ptr(),
        );
        let operands_are_tiny = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_denorm,
            numerator_is_tiny,
            empty_name.as_ptr(),
        );

        // 2**64 and 2**-64 are exact in f32, so a multiply matches ldexp bit for
        // bit while staying a packed operation.
        let scale_up = self.const_fp_like(ty, 18446744073709551616.0);
        let scaled_up = llvm::core::LLVMBuildFMul(builder, value, scale_up, empty_name.as_ptr());
        let scale_down = self.const_fp_like(ty, 5.421010862427522e-20);
        let scaled_down =
            llvm::core::LLVMBuildFMul(builder, value, scale_down, empty_name.as_ptr());

        // A zero operand returns the negative quiet NaN, not the positive one
        // the pseudo code names.
        let nan = llvm::core::LLVMBuildBitCast(
            builder,
            self.const_i32_like(ty_int, 0xffc00000u32 as i32),
            ty,
            empty_name.as_ptr(),
        );
        let scales_denominator = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            value,
            denominator,
            empty_name.as_ptr(),
        );
        let scales_numerator = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            value,
            numerator,
            empty_name.as_ptr(),
        );

        // Only the operand this invocation was handed is modified; the other one
        // passes through unchanged.
        let denominator_up = llvm::core::LLVMBuildSelect(
            builder,
            scales_denominator,
            scaled_up,
            value,
            empty_name.as_ptr(),
        );
        let denominator_down = llvm::core::LLVMBuildSelect(
            builder,
            scales_denominator,
            scaled_down,
            value,
            empty_name.as_ptr(),
        );
        let numerator_up = llvm::core::LLVMBuildSelect(
            builder,
            scales_numerator,
            scaled_up,
            value,
            empty_name.as_ptr(),
        );

        // Within range the quotient is left alone, but both operands are shifted
        // together when they sit at the edge of the exponent range.
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            operands_are_tiny,
            scaled_up,
            value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            reciprocal_is_denorm,
            scaled_down,
            d_value,
            empty_name.as_ptr(),
        );
        // Quotient underflow: scale the numerator up, unless the reciprocal is
        // already subnormal, in which case the denominator is scaled down.
        let underflow_value = llvm::core::LLVMBuildSelect(
            builder,
            reciprocal_is_denorm,
            denominator_down,
            numerator_up,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            quotient_underflows,
            underflow_value,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            quotient_overflows,
            denominator_up,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value =
            llvm::core::LLVMBuildSelect(builder, returns_nan, nan, d_value, empty_name.as_ptr());

        let needs_post_scale = llvm::core::LLVMBuildOr(
            builder,
            quotient_overflows,
            quotient_underflows,
            empty_name.as_ptr(),
        );
        let vcc_value = llvm::core::LLVMBuildSelect(
            builder,
            needs_post_scale,
            true_value,
            false_value,
            empty_name.as_ptr(),
        );

        (d_value, vcc_value)
    }

    /// `V_DIV_SCALE_F64`: scale an operand of the division macro so that no
    /// subnormal terms appear during the Newton-Raphson correction. Operands
    /// follow the ISA order (S0 value to scale, S1 denominator, S2 numerator).
    /// Returns the scaled value and the VCC mask that tells `V_DIV_FMAS_F64`
    /// whether the quotient needs post-scaling.
    ///
    /// The branch conditions were measured on gfx1201 hardware over a sweep of
    /// the whole exponent range; they differ from the ISA pseudo code in that
    /// the quotient-underflow case triggers on the exponent difference,
    /// symmetric with the overflow case, and that the reciprocal-is-subnormal
    /// test is an f32 test, not f64.
    pub(crate) unsafe fn emit_div_scale_f64(
        &mut self,
        value: llvm::prelude::LLVMValueRef,
        denominator: llvm::prelude::LLVMValueRef,
        numerator: llvm::prelude::LLVMValueRef,
    ) -> (llvm::prelude::LLVMValueRef, llvm::prelude::LLVMValueRef) {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();

        let ty = llvm::core::LLVMTypeOf(value);
        let ty_int = self.int_type_like(ty);
        let ty_i1 = llvm::core::LLVMInt1TypeInContext(context);
        let ty_bool = if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            llvm::core::LLVMVectorType(ty_i1, llvm::core::LLVMGetVectorSize(ty))
        } else {
            ty_i1
        };
        let true_value = llvm::core::LLVMConstAllOnes(ty_bool);
        let false_value = llvm::core::LLVMConstNull(ty_bool);

        let zero = self.const_fp_like(ty, 0.0);

        let denominator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            denominator,
            zero,
            empty_name.as_ptr(),
        );
        let numerator_is_zero = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            numerator,
            zero,
            empty_name.as_ptr(),
        );
        let returns_nan = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_zero,
            numerator_is_zero,
            empty_name.as_ptr(),
        );

        let exponent_shift = self.const_int_like(ty_int, 52);
        let exponent_mask = self.const_int_like(ty_int, 0x7FF);
        let mut exponent = |operand: llvm::prelude::LLVMValueRef| {
            let bits = llvm::core::LLVMBuildBitCast(builder, operand, ty_int, empty_name.as_ptr());
            llvm::core::LLVMBuildAnd(
                builder,
                llvm::core::LLVMBuildLShr(builder, bits, exponent_shift, empty_name.as_ptr()),
                exponent_mask,
                empty_name.as_ptr(),
            )
        };
        let denominator_exponent = exponent(denominator);
        let numerator_exponent = exponent(numerator);
        let exponent_delta = llvm::core::LLVMBuildSub(
            builder,
            numerator_exponent,
            denominator_exponent,
            empty_name.as_ptr(),
        );

        // The quotient leaves the representable range in either direction.
        let overflow_bound = self.const_int_like(ty_int, 768);
        let quotient_overflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSGE,
            exponent_delta,
            overflow_bound,
            empty_name.as_ptr(),
        );
        let underflow_bound = self.const_int_like(ty_int, -768);
        let quotient_underflows = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLE,
            exponent_delta,
            underflow_bound,
            empty_name.as_ptr(),
        );
        // 1.0 / S1 is subnormal, so the reciprocal loses precision.
        let reciprocal_bound = self.const_int_like(ty_int, 2045);
        let reciprocal_is_denorm = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSGE,
            denominator_exponent,
            reciprocal_bound,
            empty_name.as_ptr(),
        );
        // S1 is subnormal, or the numerator is so small that the residual terms
        // of the correction would be subnormal.
        let zero_exponent = self.const_int_like(ty_int, 0);
        let denominator_is_denorm = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntEQ,
            denominator_exponent,
            zero_exponent,
            empty_name.as_ptr(),
        );
        let tiny_bound = self.const_int_like(ty_int, 53);
        let numerator_is_tiny = llvm::core::LLVMBuildICmp(
            builder,
            llvm::LLVMIntPredicate::LLVMIntSLE,
            numerator_exponent,
            tiny_bound,
            empty_name.as_ptr(),
        );
        let operands_are_tiny = llvm::core::LLVMBuildOr(
            builder,
            denominator_is_denorm,
            numerator_is_tiny,
            empty_name.as_ptr(),
        );

        // 2**64 and 2**-64 are exact in f32, so a multiply matches ldexp bit for
        // bit while staying a packed operation.
        let scale_up = self.const_fp_like(ty, f64::from_bits(0x47F0_0000_0000_0000));
        let scaled_up = llvm::core::LLVMBuildFMul(builder, value, scale_up, empty_name.as_ptr());
        let scale_down = self.const_fp_like(ty, f64::from_bits(0x37F0_0000_0000_0000));
        let scaled_down =
            llvm::core::LLVMBuildFMul(builder, value, scale_down, empty_name.as_ptr());

        // A zero operand returns the negative quiet NaN, not the positive one
        // the pseudo code names.
        let nan = llvm::core::LLVMBuildBitCast(
            builder,
            self.const_int_like(ty_int, 0xfff8_0000_0000_0000u64 as i64),
            ty,
            empty_name.as_ptr(),
        );
        let scales_denominator = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            value,
            denominator,
            empty_name.as_ptr(),
        );
        let scales_numerator = llvm::core::LLVMBuildFCmp(
            builder,
            llvm::LLVMRealPredicate::LLVMRealOEQ,
            value,
            numerator,
            empty_name.as_ptr(),
        );

        // Only the operand this invocation was handed is modified; the other one
        // passes through unchanged.
        let denominator_up = llvm::core::LLVMBuildSelect(
            builder,
            scales_denominator,
            scaled_up,
            value,
            empty_name.as_ptr(),
        );
        let denominator_down = llvm::core::LLVMBuildSelect(
            builder,
            scales_denominator,
            scaled_down,
            value,
            empty_name.as_ptr(),
        );
        let numerator_up = llvm::core::LLVMBuildSelect(
            builder,
            scales_numerator,
            scaled_up,
            value,
            empty_name.as_ptr(),
        );

        // Within range the quotient is left alone, but both operands are shifted
        // together when they sit at the edge of the exponent range.
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            operands_are_tiny,
            scaled_up,
            value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            reciprocal_is_denorm,
            scaled_down,
            d_value,
            empty_name.as_ptr(),
        );
        // Quotient underflow: scale the numerator up, unless the reciprocal is
        // already subnormal, in which case the denominator is scaled down.
        let underflow_value = llvm::core::LLVMBuildSelect(
            builder,
            reciprocal_is_denorm,
            denominator_down,
            numerator_up,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            quotient_underflows,
            underflow_value,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value = llvm::core::LLVMBuildSelect(
            builder,
            quotient_overflows,
            denominator_up,
            d_value,
            empty_name.as_ptr(),
        );
        let d_value =
            llvm::core::LLVMBuildSelect(builder, returns_nan, nan, d_value, empty_name.as_ptr());

        let needs_post_scale = llvm::core::LLVMBuildOr(
            builder,
            quotient_overflows,
            quotient_underflows,
            empty_name.as_ptr(),
        );
        let vcc_value = llvm::core::LLVMBuildSelect(
            builder,
            needs_post_scale,
            true_value,
            false_value,
            empty_name.as_ptr(),
        );

        (d_value, vcc_value)
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
        let ty_f32 = llvm::core::LLVMTypeOf(value0);
        let ty_i32 = llvm::core::LLVMTypeOf(value1);

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
