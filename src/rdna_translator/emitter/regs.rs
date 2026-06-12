use crate::rdna_translator::*;

use llvm_sys as llvm;

use super::*;

impl IREmitter {
    pub(crate) unsafe fn emit_load_scc_u8(&mut self) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let scc_ptr = if self.use_scc_cache {
            self.local_scc_ptr
        } else {
            self.scc_ptr
        };

        let ty_i8 = llvm::core::LLVMInt8TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildLoad2(builder, ty_i8, scc_ptr, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_store_scc_u8(&mut self, value: llvm::prelude::LLVMValueRef) {
        let builder = self.builder;
        let scc_ptr = if self.use_scc_cache {
            self.local_scc_ptr
        } else {
            self.scc_ptr
        };

        llvm::core::LLVMBuildStore(builder, value, scc_ptr);
    }

    pub(crate) unsafe fn emit_load_vgpr_u32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(elem) != ty_i32 {
            panic!("Type of elem is not i32");
        }

        if self.use_vgpr_cache {
            panic!("Not implemented");
        }

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(ty_i32, reg as u64 * 32, 0),
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

    pub(crate) unsafe fn emit_load_vgpr_f32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let value = self.emit_load_vgpr_u32(reg, elem);
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f32, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_load_vgpr_u64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);

        let value_lo = self.emit_load_vgpr_u32(reg, elem);
        let value_hi = self.emit_load_vgpr_u32(reg + 1, elem);

        let value_lo = llvm::core::LLVMBuildZExt(builder, value_lo, ty_i64, empty_name.as_ptr());
        let value_hi = llvm::core::LLVMBuildZExt(builder, value_hi, ty_i64, empty_name.as_ptr());
        llvm::core::LLVMBuildOr(
            builder,
            llvm::core::LLVMBuildShl(
                builder,
                value_hi,
                llvm::core::LLVMConstInt(ty_i64, 32, 0),
                empty_name.as_ptr(),
            ),
            value_lo,
            empty_name.as_ptr(),
        )
    }

    pub(crate) unsafe fn emit_load_vgpr_f64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let value = self.emit_load_vgpr_u64(reg, elem);
        let context = self.context;
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_load_vgpr_u64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(self.context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);

        let value_lo = self.emit_load_vgpr_u32xn::<N>(reg, elem, mask);
        let value_hi = self.emit_load_vgpr_u32xn::<N>(reg + 1, elem, mask);

        let mut shuffle_indices = Vec::new();
        for i in 0..N {
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, i as u64, 0));
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, (i + N) as u64, 0));
        }

        let value_lo_hi = llvm::core::LLVMBuildShuffleVector(
            builder,
            value_lo,
            value_hi,
            llvm::core::LLVMConstVector(shuffle_indices.as_mut_ptr(), N as u32 * 2),
            empty_name.as_ptr(),
        );

        llvm::core::LLVMBuildBitCast(builder, value_lo_hi, ty_i64xn, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_load_vgpr_f64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);

        if self.use_vgpr_cache {
            let value = self.vgpr_reg_f64_map.get(&reg).unwrap()[elem as usize / N];
            if value != std::ptr::null_mut() {
                return value;
            }
        }

        let value = self.emit_load_vgpr_u64xn::<N>(reg, elem, mask);
        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f64xn, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_load_stack_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        if reg == 124 {
            return llvm::core::LLVMConstVector(
                [llvm::core::LLVMConstInt(ty_i32, 0, 0); N].as_mut_ptr(),
                N as u32,
            );
        }
        self.vgpr_reg_map.get(&reg).unwrap()[elem as usize / N]
    }

    pub(crate) unsafe fn emit_load_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let empty_name = std::ffi::CString::new("").unwrap();
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
        let alignment = llvm::core::LLVMConstInt(ty_i32, 4, 0);

        if self.use_vgpr_cache {
            return self.emit_load_stack_vgpr_u32xn::<N>(reg, elem);
        }

        let elem = llvm::core::LLVMConstInt(ty_i32, elem as u64, 0);

        let intrinsic = self.get_intrinsic_declaration("llvm.masked.load.", &[ty_i32xn, ty_p0]);

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(ty_i32, reg as u64 * 32, 0),
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

        let value = intrinsic.emit_call(
            ty_i32xn,
            &[
                value_ptr,
                alignment,
                mask,
                llvm::core::LLVMGetPoison(ty_i32xn),
            ],
        );
        value
    }

    pub(crate) unsafe fn emit_load_vgpr_f32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = self.emit_load_vgpr_u32xn::<N>(reg, elem, mask);

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f32xn, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_load_vgpr_f16xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        mask: llvm::prelude::LLVMValueRef,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let ty_i16 = llvm::core::LLVMInt16TypeInContext(context);
        let ty_i16xn = llvm::core::LLVMVectorType(ty_i16, N as u32);
        let ty_f16 = llvm::core::LLVMHalfTypeInContext(context);
        let ty_f16xn = llvm::core::LLVMVectorType(ty_f16, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = self.emit_load_vgpr_u32xn::<N>(reg, elem, mask);
        let value = llvm::core::LLVMBuildTrunc(self.builder, value, ty_i16xn, empty_name.as_ptr());

        llvm::core::LLVMBuildBitCast(self.builder, value, ty_f16xn, empty_name.as_ptr())
    }

    pub(crate) unsafe fn emit_store_stack_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        _mask: llvm::prelude::LLVMValueRef,
    ) {
        if reg == 124 {
            return;
        }

        self.vgpr_reg_f64_map.get_mut(&reg).unwrap()[elem as usize / N] = std::ptr::null_mut();

        self.vgpr_reg_map.get_mut(&reg).unwrap()[elem as usize / N] = value;
    }

    pub(crate) unsafe fn emit_store_vgpr_u32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(elem) != ty_i32 {
            panic!("Type of elem is not i32");
        }

        if self.use_vgpr_cache {
            panic!("Not implemented");
        }

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(ty_i32, reg as u64 * 32, 0),
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

    pub(crate) unsafe fn emit_store_vgpr_u32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let vgprs_ptr = self.vgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
        let ty_p0 = llvm::core::LLVMPointerTypeInContext(context, 0);
        let ty_void = llvm::core::LLVMVoidTypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();
        let alignment = llvm::core::LLVMConstInt(ty_i32, 4, 0);

        if self.use_vgpr_cache {
            return self.emit_store_stack_vgpr_u32xn::<N>(reg, elem, value, mask);
        }

        let elem = llvm::core::LLVMConstInt(ty_i32, elem as u64, 0);

        let index = llvm::core::LLVMBuildAdd(
            builder,
            llvm::core::LLVMConstInt(ty_i32, reg as u64 * 32, 0),
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

        let intrinsic = self.get_intrinsic_declaration("llvm.masked.store.", &[ty_i32xn, ty_p0]);
        intrinsic.emit_call(ty_void, &[value, value_ptr, alignment, mask]);
    }

    pub(crate) unsafe fn emit_store_vgpr_u64(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(elem) != ty_i32 {
            panic!("Type of elem is not i32");
        }

        if llvm::core::LLVMTypeOf(value) != ty_i64 {
            panic!("Type of value is not i64");
        }

        let value_lo = llvm::core::LLVMBuildTrunc(builder, value, ty_i32, empty_name.as_ptr());
        let value_hi = llvm::core::LLVMBuildLShr(
            builder,
            value,
            llvm::core::LLVMConstInt(ty_i64, 32, 0),
            empty_name.as_ptr(),
        );
        let value_hi = llvm::core::LLVMBuildTrunc(builder, value_hi, ty_i32, empty_name.as_ptr());
        self.emit_store_vgpr_u32(reg, elem, value_lo);
        self.emit_store_vgpr_u32(reg + 1, elem, value_hi);
    }

    pub(crate) unsafe fn emit_store_vgpr_f32(
        &mut self,
        reg: u32,
        elem: llvm::prelude::LLVMValueRef,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(context);
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(value) != ty_f32 {
            panic!("Type of value is not f32");
        }

        let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i32, empty_name.as_ptr());
        self.emit_store_vgpr_u32(reg, elem, value);
    }

    pub(crate) unsafe fn emit_store_vgpr_f64(
        &mut self,
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

    pub(crate) unsafe fn emit_store_vgpr_u64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let builder = self.builder;
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32x2n = llvm::core::LLVMVectorType(ty_i32, N as u32 * 2);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i32x2n, empty_name.as_ptr());

        let mut shuffle_indices = Vec::new();
        for i in 0..N {
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, (i * 2) as u64, 0));
        }

        let value_lo = llvm::core::LLVMBuildShuffleVector(
            builder,
            value,
            llvm::core::LLVMGetPoison(ty_i32x2n),
            llvm::core::LLVMConstVector(shuffle_indices.as_mut_ptr(), N as u32),
            empty_name.as_ptr(),
        );

        let mut shuffle_indices = Vec::new();
        for i in 0..N {
            shuffle_indices.push(llvm::core::LLVMConstInt(ty_i32, (i * 2 + 1) as u64, 0));
        }

        let value_hi = llvm::core::LLVMBuildShuffleVector(
            builder,
            value,
            llvm::core::LLVMGetPoison(ty_i32x2n),
            llvm::core::LLVMConstVector(shuffle_indices.as_mut_ptr(), N as u32),
            empty_name.as_ptr(),
        );

        self.emit_store_vgpr_u32xn::<N>(reg, elem, value_lo, mask);
        self.emit_store_vgpr_u32xn::<N>(reg + 1, elem, value_hi, mask);
    }

    pub(crate) unsafe fn emit_store_vgpr_f32xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let ty_i32 = llvm::core::LLVMInt32TypeInContext(self.context);
        let ty_i32xn = llvm::core::LLVMVectorType(ty_i32, N as u32);
        let ty_f32 = llvm::core::LLVMFloatTypeInContext(self.context);
        let ty_f32xn = llvm::core::LLVMVectorType(ty_f32, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(value) != ty_f32xn {
            panic!("Type of value is not i32xn");
        }

        let value =
            llvm::core::LLVMBuildBitCast(self.builder, value, ty_i32xn, empty_name.as_ptr());
        self.emit_store_vgpr_u32xn::<N>(reg, elem, value, mask);
    }

    pub(crate) unsafe fn emit_store_vgpr_f64xn<const N: usize>(
        &mut self,
        reg: u32,
        elem: u32,
        value: llvm::prelude::LLVMValueRef,
        mask: llvm::prelude::LLVMValueRef,
    ) {
        let ty_i64 = llvm::core::LLVMInt64TypeInContext(self.context);
        let ty_i64xn = llvm::core::LLVMVectorType(ty_i64, N as u32);
        let ty_f64 = llvm::core::LLVMDoubleTypeInContext(self.context);
        let ty_f64xn = llvm::core::LLVMVectorType(ty_f64, N as u32);
        let empty_name = std::ffi::CString::new("").unwrap();

        if llvm::core::LLVMTypeOf(value) != ty_f64xn {
            panic!("Type of value is not f64xn");
        }

        let value_u64xn =
            llvm::core::LLVMBuildBitCast(self.builder, value, ty_i64xn, empty_name.as_ptr());
        self.emit_store_vgpr_u64xn::<N>(reg, elem, value_u64xn, mask);

        if self.use_vgpr_cache {
            self.vgpr_reg_f64_map.get_mut(&reg).unwrap()[elem as usize / N] = value;
        }
    }

    pub(crate) unsafe fn emit_load_stack_sgpr_u32(
        &mut self,
        reg: u32,
    ) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if reg == 124 {
            llvm::core::LLVMConstInt(ty_i32, 0, 0)
        } else {
            let value_ptr = *self.sgpr_ptr_map.get(&reg).unwrap();
            llvm::core::LLVMBuildLoad2(builder, ty_i32, value_ptr, empty_name.as_ptr())
        }
    }

    pub(crate) unsafe fn emit_load_sgpr_u32(&mut self, reg: u32) -> llvm::prelude::LLVMValueRef {
        let context = self.context;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if USE_SGPR_CACHE {
            return self.emit_load_stack_sgpr_u32(reg);
        }

        if reg == 124 {
            llvm::core::LLVMConstInt(ty_i32, 0, 0)
        } else {
            let mut indices = vec![llvm::core::LLVMConstInt(ty_i32, reg as u64, 0)];
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

    pub(crate) unsafe fn emit_load_sgpr_u64(&mut self, reg: u32) -> llvm::prelude::LLVMValueRef {
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

    pub(crate) unsafe fn emit_store_stack_sgpr_u32(
        &mut self,
        reg: u32,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let builder = self.builder;

        if reg != 124 {
            let value_ptr = *self.sgpr_ptr_map.get(&reg).unwrap();
            llvm::core::LLVMBuildStore(builder, value, value_ptr);
        }
    }

    pub(crate) unsafe fn emit_store_sgpr_u32(
        &mut self,
        reg: u32,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;
        let sgprs_ptr = self.sgprs_ptr;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        if USE_SGPR_CACHE {
            return self.emit_store_stack_sgpr_u32(reg, value);
        }

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

    pub(crate) unsafe fn emit_store_sgpr_f32(
        &mut self,
        reg: u32,
        value: llvm::prelude::LLVMValueRef,
    ) {
        let context = self.context;
        let builder = self.builder;

        let ty_i32 = llvm::core::LLVMInt32TypeInContext(context);
        let empty_name = std::ffi::CString::new("").unwrap();

        let value = llvm::core::LLVMBuildBitCast(builder, value, ty_i32, empty_name.as_ptr());

        self.emit_store_sgpr_u32(reg, value);
    }

    pub(crate) unsafe fn emit_store_sgpr_u64(
        &mut self,
        reg: u32,
        value: llvm::prelude::LLVMValueRef,
    ) {
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
}
