use llvm_sys::core::*;
use llvm_sys::prelude::*;
use std::ffi::CString;

use super::ir::{FloatPred, IntPred};

pub mod jit;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Atomic {
    Monotonic,
    Acquire,
    Release,
    SequentiallyConsistent,
}

fn int_predicate(p: IntPred) -> llvm_sys::LLVMIntPredicate {
    use llvm_sys::LLVMIntPredicate::*;
    match p {
        IntPred::Eq => LLVMIntEQ,
        IntPred::Ne => LLVMIntNE,
        IntPred::Ult => LLVMIntULT,
        IntPred::Ugt => LLVMIntUGT,
        IntPred::Ule => LLVMIntULE,
        IntPred::Uge => LLVMIntUGE,
        IntPred::Slt => LLVMIntSLT,
        IntPred::Sgt => LLVMIntSGT,
        IntPred::Sle => LLVMIntSLE,
        IntPred::Sge => LLVMIntSGE,
    }
}

fn real_predicate(p: FloatPred) -> llvm_sys::LLVMRealPredicate {
    use llvm_sys::LLVMRealPredicate::*;
    match p {
        FloatPred::Oeq => LLVMRealOEQ,
        FloatPred::Ogt => LLVMRealOGT,
        FloatPred::Oge => LLVMRealOGE,
        FloatPred::Olt => LLVMRealOLT,
        FloatPred::Ole => LLVMRealOLE,
        FloatPred::One => LLVMRealONE,
        FloatPred::Ord => LLVMRealORD,
        FloatPred::Uno => LLVMRealUNO,
        FloatPred::Ueq => LLVMRealUEQ,
        FloatPred::Ugt => LLVMRealUGT,
        FloatPred::Uge => LLVMRealUGE,
        FloatPred::Ult => LLVMRealULT,
        FloatPred::Ule => LLVMRealULE,
        FloatPred::Une => LLVMRealUNE,
    }
}

fn ordering(order: Atomic) -> llvm_sys::LLVMAtomicOrdering {
    use llvm_sys::LLVMAtomicOrdering::*;
    match order {
        Atomic::Monotonic => LLVMAtomicOrderingMonotonic,
        Atomic::Acquire => LLVMAtomicOrderingAcquire,
        Atomic::Release => LLVMAtomicOrderingRelease,
        Atomic::SequentiallyConsistent => LLVMAtomicOrderingSequentiallyConsistent,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Value(LLVMValueRef);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Type(LLVMTypeRef);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct BasicBlock(LLVMBasicBlockRef);

impl Value {
    pub const fn from_raw(raw: LLVMValueRef) -> Self {
        Value(raw)
    }
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
    pub fn ty(self) -> Type {
        unsafe { Type(LLVMTypeOf(self.0)) }
    }
    pub fn is_vector(self) -> bool {
        self.ty().is_vector()
    }
    pub fn param(self, index: u32) -> Value {
        unsafe { Value(LLVMGetParam(self.0, index)) }
    }
    pub fn set_alignment(self, align: u32) -> Value {
        unsafe {
            LLVMSetAlignment(self.0, align);
        }
        self
    }
    pub fn set_initializer(self, value: Value) {
        unsafe {
            LLVMSetInitializer(self.0, value.0);
        }
    }
    pub fn set_constant(self) {
        unsafe {
            LLVMSetGlobalConstant(self.0, 1);
        }
    }
    pub fn set_private(self) {
        unsafe {
            LLVMSetLinkage(self.0, llvm_sys::LLVMLinkage::LLVMPrivateLinkage);
        }
    }
    pub fn replace_all_uses_with(self, other: Value) {
        unsafe {
            LLVMReplaceAllUsesWith(self.0, other.0);
        }
    }
    pub fn erase(self) {
        unsafe {
            LLVMInstructionEraseFromParent(self.0);
        }
    }
    pub fn add_incoming(self, incoming: &[(Value, BasicBlock)]) {
        let mut values: Vec<LLVMValueRef> = incoming.iter().map(|(v, _)| v.0).collect();
        let mut blocks: Vec<LLVMBasicBlockRef> = incoming.iter().map(|(_, b)| b.0).collect();
        unsafe {
            LLVMAddIncoming(
                self.0,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );
        }
    }
    pub fn entry_block(self) -> BasicBlock {
        unsafe { BasicBlock(LLVMGetEntryBasicBlock(self.0)) }
    }
}

impl Type {
    fn is_vector(self) -> bool {
        unsafe { LLVMGetTypeKind(self.0) == llvm_sys::LLVMTypeKind::LLVMVectorTypeKind }
    }
    pub fn is_float(self) -> bool {
        unsafe { LLVMGetTypeKind(self.0) == llvm_sys::LLVMTypeKind::LLVMFloatTypeKind }
    }
    pub fn vector_size(self) -> u32 {
        unsafe { LLVMGetVectorSize(self.0) }
    }
    pub fn element(self) -> Type {
        unsafe { Type(LLVMGetElementType(self.0)) }
    }
    pub fn vector(self, lanes: u32) -> Type {
        unsafe { Type(LLVMVectorType(self.0, lanes)) }
    }
    pub fn array(self, count: u64) -> Type {
        unsafe { Type(LLVMArrayType2(self.0, count)) }
    }
    pub fn poison(self) -> Value {
        unsafe { Value(LLVMGetPoison(self.0)) }
    }
    pub fn undef(self) -> Value {
        unsafe { Value(LLVMGetUndef(self.0)) }
    }
    pub fn null(self) -> Value {
        unsafe { Value(LLVMConstNull(self.0)) }
    }
    pub fn const_int(self, bits: u64) -> Value {
        unsafe { Value(LLVMConstInt(self.0, bits, 0)) }
    }
    pub fn function(self, params: &[Type]) -> Type {
        let mut params: Vec<LLVMTypeRef> = params.iter().map(|t| t.0).collect();
        unsafe {
            Type(LLVMFunctionType(
                self.0,
                params.as_mut_ptr(),
                params.len() as u32,
                0,
            ))
        }
    }
}

impl BasicBlock {
    pub fn function(self) -> Value {
        unsafe { Value(LLVMGetBasicBlockParent(self.0)) }
    }
    pub fn first_instruction(self) -> Option<Value> {
        let first = unsafe { LLVMGetFirstInstruction(self.0) };
        if first.is_null() {
            None
        } else {
            Some(Value(first))
        }
    }
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}
const ANON: *const std::ffi::c_char = b"\0".as_ptr() as *const std::ffi::c_char;

#[derive(Clone, Copy)]
pub struct Builder {
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    b: LLVMBuilderRef,
}

impl Builder {
    pub fn new(ctx: LLVMContextRef, module: LLVMModuleRef, b: LLVMBuilderRef) -> Self {
        Builder { ctx, module, b }
    }
    pub fn detached(&self) -> Builder {
        unsafe {
            Builder {
                ctx: self.ctx,
                module: self.module,
                b: LLVMCreateBuilderInContext(self.ctx),
            }
        }
    }
    pub fn dispose(self) {
        unsafe {
            LLVMDisposeBuilder(self.b);
        }
    }

    pub fn void(&self) -> Type {
        unsafe { Type(LLVMVoidTypeInContext(self.ctx)) }
    }
    pub fn i1(&self) -> Type {
        unsafe { Type(LLVMInt1TypeInContext(self.ctx)) }
    }
    pub fn i8(&self) -> Type {
        unsafe { Type(LLVMInt8TypeInContext(self.ctx)) }
    }
    pub fn i16(&self) -> Type {
        unsafe { Type(LLVMInt16TypeInContext(self.ctx)) }
    }
    pub fn i32(&self) -> Type {
        unsafe { Type(LLVMInt32TypeInContext(self.ctx)) }
    }
    pub fn i64(&self) -> Type {
        unsafe { Type(LLVMInt64TypeInContext(self.ctx)) }
    }
    pub fn int(&self, bits: u32) -> Type {
        unsafe { Type(LLVMIntTypeInContext(self.ctx, bits)) }
    }
    pub fn f16(&self) -> Type {
        unsafe { Type(LLVMHalfTypeInContext(self.ctx)) }
    }
    pub fn f32(&self) -> Type {
        unsafe { Type(LLVMFloatTypeInContext(self.ctx)) }
    }
    pub fn f64(&self) -> Type {
        unsafe { Type(LLVMDoubleTypeInContext(self.ctx)) }
    }
    pub fn ptr(&self) -> Type {
        unsafe { Type(LLVMPointerTypeInContext(self.ctx, 0)) }
    }
    pub fn structure(&self, fields: &[Type]) -> Type {
        let mut fields: Vec<LLVMTypeRef> = fields.iter().map(|t| t.0).collect();
        unsafe {
            Type(LLVMStructTypeInContext(
                self.ctx,
                fields.as_mut_ptr(),
                fields.len() as u32,
                0,
            ))
        }
    }

    pub fn ci1(&self, v: bool) -> Value {
        self.i1().const_int(v as u64)
    }
    pub fn ci32(&self, v: u32) -> Value {
        self.i32().const_int(v as u64)
    }
    pub fn ci64(&self, v: u64) -> Value {
        self.i64().const_int(v)
    }
    pub fn const_vector(&self, elements: &[Value]) -> Value {
        let mut elements: Vec<LLVMValueRef> = elements.iter().map(|v| v.0).collect();
        unsafe {
            Value(LLVMConstVector(
                elements.as_mut_ptr(),
                elements.len() as u32,
            ))
        }
    }
    pub fn const_i32_vector(&self, elements: &[u32]) -> Value {
        let elements: Vec<Value> = elements.iter().map(|&v| self.ci32(v)).collect();
        self.const_vector(&elements)
    }
    pub fn const_array(&self, elem: Type, elements: &[Value]) -> Value {
        let mut elements: Vec<LLVMValueRef> = elements.iter().map(|v| v.0).collect();
        unsafe {
            Value(LLVMConstArray2(
                elem.0,
                elements.as_mut_ptr(),
                elements.len() as u64,
            ))
        }
    }
    pub fn const_bitcast(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMConstBitCast(v.0, ty.0)) }
    }

    pub fn add_function(&self, name: &str, ty: Type) -> Value {
        let name = cstr(name);
        unsafe { Value(LLVMAddFunction(self.module, name.as_ptr(), ty.0)) }
    }
    pub fn function(&self, name: &str, ty: Type) -> Value {
        let name = cstr(name);
        unsafe {
            let f = LLVMGetNamedFunction(self.module, name.as_ptr());
            Value(if f.is_null() {
                LLVMAddFunction(self.module, name.as_ptr(), ty.0)
            } else {
                f
            })
        }
    }
    pub fn global(&self, name: &str) -> Option<Value> {
        let name = cstr(name);
        let g = unsafe { LLVMGetNamedGlobal(self.module, name.as_ptr()) };
        if g.is_null() {
            None
        } else {
            Some(Value(g))
        }
    }
    pub fn add_global(&self, name: &str, ty: Type) -> Value {
        let name = cstr(name);
        unsafe { Value(LLVMAddGlobal(self.module, ty.0, name.as_ptr())) }
    }
    fn intrinsic(&self, prefix: &str, overloads: &[Type]) -> (Value, Type) {
        let mut overloads: Vec<LLVMTypeRef> = overloads.iter().map(|t| t.0).collect();
        unsafe {
            let id = LLVMLookupIntrinsicID(prefix.as_ptr() as *const _, prefix.len());
            let f = LLVMGetIntrinsicDeclaration(
                self.module,
                id,
                overloads.as_mut_ptr(),
                overloads.len(),
            );
            (Value(f), Type(LLVMGlobalGetValueType(f)))
        }
    }

    pub fn append_block(&self, function: Value, name: &str) -> BasicBlock {
        let name = cstr(name);
        unsafe {
            BasicBlock(LLVMAppendBasicBlockInContext(
                self.ctx,
                function.0,
                name.as_ptr(),
            ))
        }
    }
    pub fn position_at_end(&self, block: BasicBlock) {
        unsafe {
            LLVMPositionBuilderAtEnd(self.b, block.0);
        }
    }
    pub fn position_before(&self, inst: Value) {
        unsafe {
            LLVMPositionBuilderBefore(self.b, inst.0);
        }
    }
    pub fn insert_block(&self) -> BasicBlock {
        unsafe { BasicBlock(LLVMGetInsertBlock(self.b)) }
    }
    pub fn current_function(&self) -> Value {
        self.insert_block().function()
    }

    pub fn add(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildAdd(self.b, a.0, b.0, ANON)) }
    }
    pub fn sub(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildSub(self.b, a.0, b.0, ANON)) }
    }
    pub fn mul(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildMul(self.b, a.0, b.0, ANON)) }
    }
    pub fn srem(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildSRem(self.b, a.0, b.0, ANON)) }
    }
    pub fn and(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildAnd(self.b, a.0, b.0, ANON)) }
    }
    pub fn or(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildOr(self.b, a.0, b.0, ANON)) }
    }
    pub fn xor(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildXor(self.b, a.0, b.0, ANON)) }
    }
    pub fn not(&self, a: Value) -> Value {
        unsafe { Value(LLVMBuildNot(self.b, a.0, ANON)) }
    }
    pub fn shl(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildShl(self.b, a.0, b.0, ANON)) }
    }
    pub fn lshr(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildLShr(self.b, a.0, b.0, ANON)) }
    }
    pub fn ashr(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildAShr(self.b, a.0, b.0, ANON)) }
    }
    pub fn fadd(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildFAdd(self.b, a.0, b.0, ANON)) }
    }
    pub fn fsub(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildFSub(self.b, a.0, b.0, ANON)) }
    }
    pub fn fmul(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildFMul(self.b, a.0, b.0, ANON)) }
    }
    pub fn fdiv(&self, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildFDiv(self.b, a.0, b.0, ANON)) }
    }
    pub fn fneg(&self, a: Value) -> Value {
        unsafe { Value(LLVMBuildFNeg(self.b, a.0, ANON)) }
    }
    pub fn freeze(&self, a: Value) -> Value {
        unsafe { Value(LLVMBuildFreeze(self.b, a.0, ANON)) }
    }
    pub fn icmp(&self, predicate: IntPred, a: Value, b: Value) -> Value {
        unsafe {
            Value(LLVMBuildICmp(
                self.b,
                int_predicate(predicate),
                a.0,
                b.0,
                ANON,
            ))
        }
    }
    pub fn fcmp(&self, predicate: FloatPred, a: Value, b: Value) -> Value {
        unsafe {
            Value(LLVMBuildFCmp(
                self.b,
                real_predicate(predicate),
                a.0,
                b.0,
                ANON,
            ))
        }
    }
    pub fn select(&self, c: Value, a: Value, b: Value) -> Value {
        unsafe { Value(LLVMBuildSelect(self.b, c.0, a.0, b.0, ANON)) }
    }

    pub fn zext(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildZExt(self.b, v.0, ty.0, ANON)) }
    }
    pub fn sext(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildSExt(self.b, v.0, ty.0, ANON)) }
    }
    pub fn trunc(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildTrunc(self.b, v.0, ty.0, ANON)) }
    }
    pub fn bitcast(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildBitCast(self.b, v.0, ty.0, ANON)) }
    }
    pub fn sitofp(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildSIToFP(self.b, v.0, ty.0, ANON)) }
    }
    pub fn uitofp(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildUIToFP(self.b, v.0, ty.0, ANON)) }
    }
    pub fn fpext(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildFPExt(self.b, v.0, ty.0, ANON)) }
    }
    pub fn fptrunc(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildFPTrunc(self.b, v.0, ty.0, ANON)) }
    }
    pub fn inttoptr(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildIntToPtr(self.b, v.0, ty.0, ANON)) }
    }
    pub fn ptrtoint(&self, v: Value, ty: Type) -> Value {
        unsafe { Value(LLVMBuildPtrToInt(self.b, v.0, ty.0, ANON)) }
    }

    pub fn extract(&self, v: Value, index: Value) -> Value {
        unsafe { Value(LLVMBuildExtractElement(self.b, v.0, index.0, ANON)) }
    }
    pub fn extract_at(&self, v: Value, index: u32) -> Value {
        self.extract(v, self.ci32(index))
    }
    fn insert(&self, v: Value, element: Value, index: Value) -> Value {
        unsafe {
            Value(LLVMBuildInsertElement(
                self.b, v.0, element.0, index.0, ANON,
            ))
        }
    }
    pub fn insert_at(&self, v: Value, element: Value, index: u32) -> Value {
        self.insert(v, element, self.ci32(index))
    }
    pub fn shuffle(&self, a: Value, b: Value, mask: Value) -> Value {
        unsafe { Value(LLVMBuildShuffleVector(self.b, a.0, b.0, mask.0, ANON)) }
    }
    pub fn shuffle_by(&self, a: Value, b: Value, mask: &[u32]) -> Value {
        self.shuffle(a, b, self.const_i32_vector(mask))
    }
    pub fn splat(&self, v: Value, lanes: u32) -> Value {
        let vty = v.ty().vector(lanes);
        let poison = vty.poison();
        let ins = self.insert_at(poison, v, 0);
        self.shuffle(ins, poison, self.i32().vector(lanes).null())
    }

    pub fn alloca(&self, ty: Type, name: &str) -> Value {
        let name = cstr(name);
        unsafe { Value(LLVMBuildAlloca(self.b, ty.0, name.as_ptr())) }
    }
    pub fn array_alloca(&self, ty: Type, count: Value, name: &str) -> Value {
        let name = cstr(name);
        unsafe { Value(LLVMBuildArrayAlloca(self.b, ty.0, count.0, name.as_ptr())) }
    }
    pub fn load(&self, ty: Type, ptr: Value) -> Value {
        unsafe { Value(LLVMBuildLoad2(self.b, ty.0, ptr.0, ANON)) }
    }
    pub fn store(&self, v: Value, ptr: Value) -> Value {
        unsafe { Value(LLVMBuildStore(self.b, v.0, ptr.0)) }
    }
    pub fn gep(&self, ty: Type, ptr: Value, indices: &[Value]) -> Value {
        let mut indices: Vec<LLVMValueRef> = indices.iter().map(|v| v.0).collect();
        unsafe {
            Value(LLVMBuildGEP2(
                self.b,
                ty.0,
                ptr.0,
                indices.as_mut_ptr(),
                indices.len() as u32,
                ANON,
            ))
        }
    }
    pub fn struct_gep(&self, ty: Type, ptr: Value, field: u32) -> Value {
        unsafe { Value(LLVMBuildStructGEP2(self.b, ty.0, ptr.0, field, ANON)) }
    }
    pub fn atomic_add(&self, ptr: Value, v: Value, order: Atomic) -> Value {
        unsafe {
            Value(LLVMBuildAtomicRMW(
                self.b,
                llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                ptr.0,
                v.0,
                ordering(order),
                0,
            ))
        }
    }
    pub fn fence(&self, order: Atomic) -> Value {
        unsafe { Value(LLVMBuildFence(self.b, ordering(order), 0, ANON)) }
    }

    pub fn br(&self, block: BasicBlock) -> Value {
        unsafe { Value(LLVMBuildBr(self.b, block.0)) }
    }
    pub fn cond_br(&self, c: Value, yes: BasicBlock, no: BasicBlock) -> Value {
        unsafe { Value(LLVMBuildCondBr(self.b, c.0, yes.0, no.0)) }
    }
    pub fn ret(&self, v: Value) -> Value {
        unsafe { Value(LLVMBuildRet(self.b, v.0)) }
    }
    pub fn ret_void(&self) -> Value {
        unsafe { Value(LLVMBuildRetVoid(self.b)) }
    }
    pub fn unreachable(&self) -> Value {
        unsafe { Value(LLVMBuildUnreachable(self.b)) }
    }
    pub fn phi(&self, ty: Type) -> Value {
        unsafe { Value(LLVMBuildPhi(self.b, ty.0, ANON)) }
    }

    pub fn call(&self, ty: Type, function: Value, args: &[Value]) -> Value {
        let mut args: Vec<LLVMValueRef> = args.iter().map(|v| v.0).collect();
        unsafe {
            Value(LLVMBuildCall2(
                self.b,
                ty.0,
                function.0,
                args.as_mut_ptr(),
                args.len() as u32,
                ANON,
            ))
        }
    }
    pub fn call_named(&self, name: &str, ret: Type, params: &[Type], args: &[Value]) -> Value {
        let ty = ret.function(params);
        let function = self.function(name, ty);
        self.call(ty, function, args)
    }
    pub fn call_intrinsic(&self, prefix: &str, overloads: &[Type], args: &[Value]) -> Value {
        let (function, ty) = self.intrinsic(prefix, overloads);
        self.call(ty, function, args)
    }
    pub fn set_call_align(&self, call: Value, argument: u32, align: u64) {
        let name = b"align";
        unsafe {
            let kind = LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
            let attr = LLVMCreateEnumAttribute(self.ctx, kind, align);
            LLVMAddCallSiteAttribute(call.0, argument, attr);
        }
    }
}
