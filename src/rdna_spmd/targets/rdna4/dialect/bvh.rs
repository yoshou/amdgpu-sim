//! BVH64 target lowering. Inputs and outputs are values; no ISA decoding or
//! architectural register access occurs here. Preserve the existing native paths.
//! Ray operands are explicit bit-pattern values. Form their floating-point
//! views where each native path consumes them, preserving the original order.
use super::*;
use llvm_sys as llvm;
use std::ffi::CString;
fn cstr(s:&str)->CString {CString::new(s).unwrap()}
#[derive(Clone,Copy)]
pub(in crate::rdna_spmd) struct Storage {
    pub scratch: LLVMValueRef,
    pub packet: LLVMValueRef,
    pub packet_ty: LLVMTypeRef,
}
struct Native {
    b:LLVMBuilderRef, module:LLVMModuleRef,ctx:LLVMContextRef,w:u32,
    i1:LLVMTypeRef,i32t:LLVMTypeRef,i64t:LLVMTypeRef,f32t:LLVMTypeRef,ptr:LLVMTypeRef,
    vi1:LLVMTypeRef,vi32:LLVMTypeRef,vi64:LLVMTypeRef,vf32:LLVMTypeRef,
    bvh_packet:LLVMValueRef,bvh_packet_ty:LLVMTypeRef,
}
pub(in crate::rdna_spmd) unsafe fn lowering_state(e:&Emitter,sink:LLVMValueRef)->Box<dyn std::any::Any> {
    let (packet,packet_ty)=if e.width().is_some() {
        let ctx=e.ctx;let i32t=LLVMInt32TypeInContext(ctx);let i64t=LLVMInt64TypeInContext(ctx);let f32t=LLVMFloatTypeInContext(ctx);
        let lanes=crate::rdna_translator::bvh::BVH_RAY_PACKET_LANES as u64;
        let packet_i64=LLVMArrayType2(i64t,lanes);let packet_f32=LLVMArrayType2(f32t,lanes);let packet_i32=LLVMArrayType2(i32t,lanes);
        let mut fields=[packet_i64;15];fields[1..11].fill(packet_f32);fields[11..15].fill(packet_i32);
        let packet_ty=LLVMStructTypeInContext(ctx,fields.as_mut_ptr(),fields.len() as u32,0);
        let packet=LLVMBuildAlloca(e.b,packet_ty,b"\0".as_ptr().cast());
        LLVMSetAlignment(packet,64);
        (packet,packet_ty)
    } else {(std::ptr::null_mut(),std::ptr::null_mut())};
    Box::new(Storage {scratch:sink,packet,packet_ty})
}
pub(super) unsafe fn lower(e:&Emitter,a:&[LLVMValueRef])->Vec<LLVMValueRef> {
    let storage=*e.state::<Storage>().expect("BVH lowering requires native scratch storage");
    let ctx=e.ctx;let b=e.b;let module=LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(b)));
    let i1=LLVMInt1TypeInContext(ctx);let i32t=LLVMInt32TypeInContext(ctx);let i64t=LLVMInt64TypeInContext(ctx);let f32t=LLVMFloatTypeInContext(ctx);let ptr=LLVMPointerTypeInContext(ctx,0);
    let cg=Native {b,module,ctx,w:e.width().unwrap_or(1),i1,i32t,i64t,f32t,ptr,vi1:e.ty(Ty::I1),vi32:e.ty(Ty::I32),vi64:e.ty(Ty::I64),vf32:e.ty(Ty::F32),bvh_packet:storage.packet,bvh_packet_ty:storage.packet_ty};
    if e.width().is_some() {
        let resource=[LLVMBuildExtractElement(b,a[0],cg.ci32(0),cg.n()),LLVMBuildExtractElement(b,a[1],cg.ci32(0),cg.n())];
        let width=LLVMIntTypeInContext(ctx,cg.w);
        let mask=LLVMBuildZExt(b,LLVMBuildBitCast(b,a[13],width,cg.n()),i32t,cg.n());
        cg.packet(a,resource,mask)
    } else {
        let scratch_ptr=|k:u32|LLVMBuildGEP2(b,i32t,storage.scratch,[cg.ci32(k)].as_mut_ptr(),1,cg.n());
        let params=[ptr,ptr,ptr,ptr,i32t,i32t,i64t,f32t,f32t,f32t,f32t,f32t,f32t,f32t,f32t,f32t,f32t];
        let args=[scratch_ptr(0),scratch_ptr(1),scratch_ptr(2),scratch_ptr(3),a[0],a[1],a[2],cg.vf32_of(a[3]),cg.vf32_of(a[4]),cg.vf32_of(a[5]),cg.vf32_of(a[6]),cg.vf32_of(a[7]),cg.vf32_of(a[8]),cg.vf32_of(a[9]),cg.vf32_of(a[10]),cg.vf32_of(a[11]),cg.vf32_of(a[12])];
        cg.call("image_bvh64_intersect_ray",LLVMVoidTypeInContext(ctx),&params,&args);
        (0..4).map(|k|LLVMBuildLoad2(b,i32t,scratch_ptr(k),cg.n())).collect()
    }
}
pub(super) unsafe fn lower8(e:&Emitter,a:&[LLVMValueRef])->Vec<LLVMValueRef> {
    let ctx=e.ctx;let b=e.b;
    let module=LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(b)));
    let i1=LLVMInt1TypeInContext(ctx);let i32t=LLVMInt32TypeInContext(ctx);let i64t=LLVMInt64TypeInContext(ctx);let f32t=LLVMFloatTypeInContext(ctx);let ptr=LLVMPointerTypeInContext(ctx,0);
    let cg=Native {b,module,ctx,w:e.width().unwrap_or(1),i1,i32t,i64t,f32t,ptr,vi1:e.ty(Ty::I1),vi32:e.ty(Ty::I32),vi64:e.ty(Ty::I64),vf32:e.ty(Ty::F32),bvh_packet:std::ptr::null_mut(),bvh_packet_ty:std::ptr::null_mut()};
    let n=cg.n();
    let packed=e.width().is_some();
    let lanes=e.width().unwrap_or(1);
    const RESULTS:u32=10;
    let slots=cg.entry_alloca(LLVMArrayType2(i32t,(RESULTS*lanes) as u64));
    let lane=|v:LLVMValueRef,l:u32|if packed {LLVMBuildExtractElement(b,v,cg.ci32(l),n)} else {v};
    let params=[ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,i32t,i32t,i64t,f32t,i32t,f32t,f32t,f32t,f32t,f32t,f32t,i32t];
    let func=LLVMGetBasicBlockParent(LLVMGetInsertBlock(b));
    for l in 0..lanes {
        let slot=|k:u32|LLVMBuildGEP2(b,i32t,slots,[cg.ci32(l*RESULTS+k)].as_mut_ptr(),1,n);
        let ptrs:Vec<LLVMValueRef>=(0..RESULTS).map(slot).collect();
        let call_bb=LLVMAppendBasicBlockInContext(ctx,func,cstr("bvh8.lane").as_ptr());
        let join=LLVMAppendBasicBlockInContext(ctx,func,cstr("bvh8.join").as_ptr());
        LLVMBuildCondBr(b,lane(a[12],l),call_bb,join);
        LLVMPositionBuilderAtEnd(b,call_bb);
        let float=|v:LLVMValueRef|LLVMBuildBitCast(b,lane(v,l),f32t,n);
        let args=[ptrs[0],ptrs[1],ptrs[2],ptrs[3],ptrs[4],ptrs[5],ptrs[6],ptrs[7],ptrs[8],ptrs[9],
            lane(a[0],l),lane(a[1],l),lane(a[2],l),float(a[3]),lane(a[4],l),
            float(a[5]),float(a[6]),float(a[7]),float(a[8]),float(a[9]),float(a[10]),lane(a[11],l)];
        cg.call("image_bvh8_intersect_ray",LLVMVoidTypeInContext(ctx),&params,&args);
        LLVMBuildBr(b,join);
        LLVMPositionBuilderAtEnd(b,join);
    }
    (0..RESULTS).map(|k| {
        if !packed {
            return LLVMBuildLoad2(b,i32t,LLVMBuildGEP2(b,i32t,slots,[cg.ci32(k)].as_mut_ptr(),1,n),n);
        }
        let mut out=LLVMGetPoison(cg.vi32);
        for l in 0..lanes {
            let value=LLVMBuildLoad2(b,i32t,LLVMBuildGEP2(b,i32t,slots,[cg.ci32(l*RESULTS+k)].as_mut_ptr(),1,n),n);
            out=LLVMBuildInsertElement(b,out,value,cg.ci32(l),n);
        }
        out
    }).collect()
}

impl Native {
    unsafe fn entry_alloca(&self,ty:LLVMTypeRef)->LLVMValueRef {
        let entry=LLVMGetEntryBasicBlock(LLVMGetBasicBlockParent(LLVMGetInsertBlock(self.b)));
        let builder=LLVMCreateBuilderInContext(self.ctx);
        match LLVMGetFirstInstruction(entry) {
            first if first.is_null() => LLVMPositionBuilderAtEnd(builder,entry),
            first => LLVMPositionBuilderBefore(builder,first),
        }
        let value=LLVMBuildAlloca(builder,ty,self.n());
        LLVMSetAlignment(value,64);
        LLVMDisposeBuilder(builder);
        value
    }
    unsafe fn n(&self) -> *const std::ffi::c_char {
        b"\0".as_ptr() as *const std::ffi::c_char
    }
    unsafe fn get_func(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef]) -> (LLVMValueRef, LLVMTypeRef) {
        let cname = cstr(name);
        let mut f = llvm::core::LLVMGetNamedFunction(self.module, cname.as_ptr());
        let fty = llvm::core::LLVMFunctionType(ret, params.as_ptr() as *mut _, params.len() as u32, 0);
        if f.is_null() {
            f = llvm::core::LLVMAddFunction(self.module, cname.as_ptr(), fty);
        }
        (f, fty)
    }
    unsafe fn call(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef], args: &[LLVMValueRef]) -> LLVMValueRef {
        let (f, fty) = self.get_func(name, ret, params);
        llvm::core::LLVMBuildCall2(self.b, fty, f, args.as_ptr() as *mut _, args.len() as u32, self.n())
    }
    unsafe fn ci32(&self, v: u32) -> LLVMValueRef { llvm::core::LLVMConstInt(self.i32t, v as u64, 0) }
    unsafe fn ci64(&self, v: u64) -> LLVMValueRef { llvm::core::LLVMConstInt(self.i64t, v, 0) }
    unsafe fn splat(&self, v: LLVMValueRef, vty: LLVMTypeRef) -> LLVMValueRef {
        let poison = llvm::core::LLVMGetPoison(vty);
        let ins = llvm::core::LLVMBuildInsertElement(self.b, poison, v, self.ci32(0), self.n());
        let mask = llvm::core::LLVMConstNull(llvm::core::LLVMVectorType(self.i32t, llvm::core::LLVMGetVectorSize(vty)));
        llvm::core::LLVMBuildShuffleVector(self.b, ins, poison, mask, self.n())
    }
    unsafe fn vci32(&self, v: u32) -> LLVMValueRef { self.splat(self.ci32(v), self.vi32) }
    unsafe fn zext64s(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildZExt(self.b, v, self.i64t, self.n())
    }
    unsafe fn vf32_bits(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, v, self.vi32, self.n())
    }
    unsafe fn vf32_of(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, v, self.vf32, self.n())
    }
    unsafe fn vminnum_raw(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        let c = llvm::core::LLVMBuildFCmp(self.b, llvm::LLVMRealPredicate::LLVMRealOLT, a, b, self.n());
        llvm::core::LLVMBuildSelect(self.b, c, a, b, self.n())
    }
    unsafe fn vmaxnum_raw(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        let c = llvm::core::LLVMBuildFCmp(self.b, llvm::LLVMRealPredicate::LLVMRealOGT, a, b, self.n());
        llvm::core::LLVMBuildSelect(self.b, c, a, b, self.n())
    }
    unsafe fn vfadd(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildFAdd(self.b, a, b, self.n()) }
    unsafe fn vfmul(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildFMul(self.b, a, b, self.n()) }
    unsafe fn vfsub(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildFSub(self.b, a, b, self.n())
    }
    unsafe fn packet(&self,a:&[LLVMValueRef],resource:[LLVMValueRef;2],mask:LLVMValueRef)->Vec<LLVMValueRef> {
                use llvm::LLVMIntPredicate::*;
                use llvm::LLVMRealPredicate::*;
                let n = self.n();
                let func = llvm::core::LLVMGetBasicBlockParent(
                    llvm::core::LLVMGetInsertBlock(self.b),
                );
                let bb = |name: &str| {
                    llvm::core::LLVMAppendBasicBlockInContext(
                        self.ctx,
                        func,
                        cstr(name).as_ptr(),
                    )
                };
                let uni_bb = bb("bvh.uniform");
                let fast = bb("bvh.box");
                let tri_bb = bb("bvh.tri");
                let slow = bb("bvh.general");
                let join = bb("bvh.join");

                // The resource names the base of the BVH in 256-byte units,
                // and says whether the children that point at triangle nodes
                // sort before the ones that point at boxes.
                let base_hi = llvm::core::LLVMBuildShl(
                    self.b,
                    self.zext64s(llvm::core::LLVMBuildAnd(
                        self.b,
                        resource[1],
                        self.ci32(0xFF),
                        n,
                    )),
                    self.ci64(32),
                    n,
                );
                let bvh_base = llvm::core::LLVMBuildShl(
                    self.b,
                    llvm::core::LLVMBuildOr(
                        self.b,
                        self.zext64s(resource[0]),
                        base_hi,
                        n,
                    ),
                    self.ci64(8),
                    n,
                );
                let sorts_boxes = llvm::core::LLVMBuildICmp(
                    self.b,
                    LLVMIntNE,
                    llvm::core::LLVMBuildAnd(
                        self.b,
                        llvm::core::LLVMBuildLShr(
                            self.b,
                            resource[1],
                            self.ci32(31),
                            n,
                        ),
                        self.ci32(1),
                        n,
                    ),
                    self.ci32(0),
                    n,
                );
                let sorts_triangles_first = llvm::core::LLVMBuildICmp(
                    self.b,
                    LLVMIntNE,
                    llvm::core::LLVMBuildAnd(
                        self.b,
                        llvm::core::LLVMBuildLShr(
                            self.b,
                            resource[1],
                            self.ci32(20),
                            n,
                        ),
                        self.ci32(1),
                        n,
                    ),
                    self.ci32(0),
                    n,
                );
                let addr = a[2];
                let extent = self.vf32_of(a[3]);
                let origin: Vec<LLVMValueRef> = (0..3)
                    .map(|k| self.vf32_of(a[4 + k as usize]))
                    .collect();
                let inv: Vec<LLVMValueRef> = (0..3)
                    .map(|k| self.vf32_of(a[10 + k as usize]))
                    .collect();

                // Representative node address over the active lanes, the same
                // umax idiom `emit_vglobal_cluster` uses (the block only runs
                // with EXEC != 0, so at least one lane contributes).
                let exec = a[13];
                let masked = llvm::core::LLVMBuildSelect(
                    self.b,
                    exec,
                    addr,
                    llvm::core::LLVMConstNull(self.vi64),
                    n,
                );
                let rep = self.call(
                    &format!("llvm.vector.reduce.umax.v{}i64", self.w),
                    self.i64t,
                    &[self.vi64],
                    &[masked],
                );
                let same = llvm::core::LLVMBuildICmp(
                    self.b,
                    LLVMIntEQ,
                    addr,
                    self.splat(rep, self.vi64),
                    n,
                );
                let same_or_off = llvm::core::LLVMBuildOr(
                    self.b,
                    same,
                    llvm::core::LLVMBuildNot(self.b, exec, n),
                    n,
                );
                let uniform = self.call(
                    &format!("llvm.vector.reduce.and.v{}i1", self.w),
                    self.i1,
                    &[self.vi1],
                    &[same_or_off],
                );
                let ntype = llvm::core::LLVMBuildAnd(self.b, rep, self.ci64(7), n);
                let is_box = llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, ntype, self.ci64(5), n);
                let is_tri = llvm::core::LLVMBuildICmp(self.b, LLVMIntULT, ntype, self.ci64(2), n);
                let known = llvm::core::LLVMBuildOr(self.b, is_box, is_tri, n);
                // With EXEC == 0 the reduction above has no active lane to pick,
                // so `rep` would be 0 and the type test would accept it as a
                // triangle node at address 0. Blocks are not supposed to run
                // with EXEC == 0, but the fast path must not dereference a null
                // node if one ever does.
                let any_active = llvm::core::LLVMBuildICmp(
                    self.b,
                    LLVMIntNE,
                    mask,
                    self.ci32(0),
                    n,
                );
                // The inline path sorts the children by the time the ray
                // reaches them and nothing else, which is what a resource that
                // sorts its boxes and leaves triangle nodes where they are
                // asks for; the helper knows the rest of Table 65.
                let plainly_sorted = llvm::core::LLVMBuildAnd(
                    self.b,
                    sorts_boxes,
                    llvm::core::LLVMBuildNot(self.b, sorts_triangles_first, n),
                    n,
                );
                let take = llvm::core::LLVMBuildAnd(
                    self.b,
                    llvm::core::LLVMBuildAnd(
                        self.b,
                        llvm::core::LLVMBuildAnd(self.b, uniform, known, n),
                        any_active,
                        n,
                    ),
                    plainly_sorted,
                    n,
                );
                llvm::core::LLVMBuildCondBr(self.b, take, uni_bb, slow);
                llvm::core::LLVMPositionBuilderAtEnd(self.b, uni_bb);
                llvm::core::LLVMBuildCondBr(self.b, is_box, fast, tri_bb);

                // ---- every active lane at the same box node ----------------
                llvm::core::LLVMPositionBuilderAtEnd(self.b, fast);
                let node_ptr = llvm::core::LLVMBuildAdd(
                    self.b,
                    bvh_base,
                    llvm::core::LLVMBuildShl(
                        self.b,
                        llvm::core::LLVMBuildAnd(self.b, rep, self.ci64(!0x7u64), n),
                        self.ci64(3),
                        n,
                    ),
                    n,
                );
                // Box4Node: child_index[4], then aabb[4] of { min[3], max[3] }.
                let field = |off: u64, ty: LLVMTypeRef| -> LLVMValueRef {
                    let a = llvm::core::LLVMBuildAdd(self.b, node_ptr, self.ci64(off), n);
                    let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, n);
                    let ld = llvm::core::LLVMBuildLoad2(self.b, ty, p, n);
                    llvm::core::LLVMSetAlignment(ld, 4);
                    ld
                };
                let vzero = llvm::core::LLVMConstNull(self.vf32);
                let mut child = [llvm::core::LLVMConstNull(self.vi32); 4];
                let mut dist = [vzero; 4];
                let mut nan_acc: Option<LLVMValueRef> = None;
                for c in 0..4u64 {
                    let mut hi3 = [vzero; 3];
                    let mut lo3 = [vzero; 3];
                    for axis in 0..3u64 {
                        let base = 16 + c * 24 + axis * 4;
                        let bhi = self.splat(field(base + 12, self.f32t), self.vf32);
                        let blo = self.splat(field(base, self.f32t), self.vf32);
                        let f = self.vfmul(self.vfsub(bhi, origin[axis as usize]), inv[axis as usize]);
                        let g = self.vfmul(self.vfsub(blo, origin[axis as usize]), inv[axis as usize]);
                        for v in [f, g] {
                            let u = llvm::core::LLVMBuildFCmp(self.b, LLVMRealUNO, v, v, n);
                            nan_acc = Some(match nan_acc {
                                None => u,
                                Some(p) => llvm::core::LLVMBuildOr(self.b, p, u, n),
                            });
                        }
                        hi3[axis as usize] = self.vmaxnum_raw(f, g);
                        lo3[axis as usize] = self.vminnum_raw(f, g);
                    }
                    let t1 = self.vminnum_raw(
                        hi3[0],
                        self.vminnum_raw(hi3[1], self.vminnum_raw(hi3[2], extent)),
                    );
                    let t0 = self.vmaxnum_raw(
                        lo3[0],
                        self.vmaxnum_raw(lo3[1], self.vmaxnum_raw(lo3[2], vzero)),
                    );
                    let hit = llvm::core::LLVMBuildFCmp(self.b, LLVMRealOLE, t0, t1, n);
                    let ci = self.splat(field(c * 4, self.i32t), self.vi32);
                    child[c as usize] = llvm::core::LLVMBuildSelect(
                        self.b,
                        hit,
                        ci,
                        self.vci32(0xFFFF_FFFF),
                        n,
                    );
                    dist[c as usize] = t0;
                }
                let ones = self.vci32(0xFFFF_FFFF);
                let swap_pair = |a: usize, b: usize,
                                     child: &mut [LLVMValueRef; 4],
                                     dist: &mut [LLVMValueRef; 4]| {
                    let b_valid =
                        llvm::core::LLVMBuildICmp(self.b, LLVMIntNE, child[b], ones, n);
                    let closer =
                        llvm::core::LLVMBuildFCmp(self.b, LLVMRealOLT, dist[b], dist[a], n);
                    let a_empty =
                        llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, child[a], ones, n);
                    let sw = llvm::core::LLVMBuildOr(
                        self.b,
                        llvm::core::LLVMBuildAnd(self.b, b_valid, closer, n),
                        a_empty,
                        n,
                    );
                    let ca = llvm::core::LLVMBuildSelect(self.b, sw, child[b], child[a], n);
                    let cb = llvm::core::LLVMBuildSelect(self.b, sw, child[a], child[b], n);
                    let da = llvm::core::LLVMBuildSelect(self.b, sw, dist[b], dist[a], n);
                    let db = llvm::core::LLVMBuildSelect(self.b, sw, dist[a], dist[b], n);
                    child[a] = ca;
                    child[b] = cb;
                    dist[a] = da;
                    dist[b] = db;
                };
                swap_pair(0, 2, &mut child, &mut dist);
                swap_pair(1, 3, &mut child, &mut dist);
                swap_pair(0, 1, &mut child, &mut dist);
                swap_pair(2, 3, &mut child, &mut dist);
                swap_pair(1, 2, &mut child, &mut dist);

                // A NaN slab value makes the ordered min/max above differ from
                // minNum; that lane set is rare enough to redo in the helper.
                let any_nan = self.call(
                    &format!("llvm.vector.reduce.or.v{}i1", self.w),
                    self.i1,
                    &[self.vi1],
                    &[nan_acc.unwrap()],
                );
                llvm::core::LLVMBuildCondBr(self.b, any_nan, slow, join);
                let fast_end = llvm::core::LLVMGetInsertBlock(self.b);

                // ---- every active lane at the same triangle-pair node ------
                llvm::core::LLVMPositionBuilderAtEnd(self.b, tri_bb);
                let tnode = llvm::core::LLVMBuildAdd(
                    self.b,
                    bvh_base,
                    llvm::core::LLVMBuildShl(
                        self.b,
                        llvm::core::LLVMBuildAnd(self.b, rep, self.ci64(!0x7u64), n),
                        self.ci64(3),
                        n,
                    ),
                    n,
                );
                let tfield = |off: u64, ty: LLVMTypeRef| -> LLVMValueRef {
                    let a = llvm::core::LLVMBuildAdd(self.b, tnode, self.ci64(off), n);
                    let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, n);
                    let ld = llvm::core::LLVMBuildLoad2(self.b, ty, p, n);
                    llvm::core::LLVMSetAlignment(ld, 4);
                    ld
                };
                // TrianglePairNode: v0,v1,v2,v3 (3 f32 each), pad, prim_index[2], flags.
                let odd = llvm::core::LLVMBuildICmp(
                    self.b,
                    LLVMIntNE,
                    llvm::core::LLVMBuildAnd(self.b, rep, self.ci64(1), n),
                    self.ci64(0),
                    n,
                );
                let vtx = |slot: u64, axis: u64| tfield(slot * 12 + axis * 4, self.f32t);
                // tri = odd ? [v1, v3, v2] : [v0, v1, v2]
                let pick = |a: u64, b: u64, axis: u64| {
                    llvm::core::LLVMBuildSelect(self.b, odd, vtx(a, axis), vtx(b, axis), n)
                };
                let t0v: Vec<LLVMValueRef> = (0..3).map(|k| self.splat(pick(1, 0, k), self.vf32)).collect();
                let t1v: Vec<LLVMValueRef> = (0..3).map(|k| self.splat(pick(3, 1, k), self.vf32)).collect();
                let t2v: Vec<LLVMValueRef> = (0..3).map(|k| self.splat(pick(2, 2, k), self.vf32)).collect();
                let flags_raw = tfield(60, self.i32t);
                let flags = llvm::core::LLVMBuildLShr(
                    self.b,
                    flags_raw,
                    llvm::core::LLVMBuildSelect(self.b, odd, self.ci32(8), self.ci32(0), n),
                    n,
                );

                let dir: Vec<LLVMValueRef> = (0..3)
                    .map(|k| self.vf32_of(a[7 + k as usize]))
                    .collect();
                let sub3 = |a: &[LLVMValueRef], b: &[LLVMValueRef]| -> Vec<LLVMValueRef> {
                    (0..3).map(|k| self.vfsub(a[k], b[k])).collect()
                };
                // Same association as `intersect_triangle_frac`: cross uses
                // a1*b2 - a2*b1, dot is (x + y) + z. No contraction.
                let cross = |a: &[LLVMValueRef], b: &[LLVMValueRef]| -> Vec<LLVMValueRef> {
                    vec![
                        self.vfsub(self.vfmul(a[1], b[2]), self.vfmul(a[2], b[1])),
                        self.vfsub(self.vfmul(a[2], b[0]), self.vfmul(a[0], b[2])),
                        self.vfsub(self.vfmul(a[0], b[1]), self.vfmul(a[1], b[0])),
                    ]
                };
                let dot = |a: &[LLVMValueRef], b: &[LLVMValueRef]| -> LLVMValueRef {
                    self.vfadd(
                        self.vfadd(self.vfmul(a[0], b[0]), self.vfmul(a[1], b[1])),
                        self.vfmul(a[2], b[2]),
                    )
                };

                let e1 = sub3(&t1v, &t0v);
                let e2 = sub3(&t2v, &t0v);
                let s1 = cross(&dir, &e2);
                let denom = dot(&s1, &e1);
                let dv = sub3(&origin, &t0v);
                let b_y = dot(&dv, &s1);
                let s2 = cross(&dv, &e1);
                let b_z = dot(&dir, &s2);
                let t_hit = dot(&e2, &s2);
                let b_x = self.vfsub(self.vfsub(denom, b_y), b_z);

                let zero = llvm::core::LLVMConstNull(self.vf32);
                let fc = |p, a, b| llvm::core::LLVMBuildFCmp(self.b, p, a, b, n);
                let or = |a, b| llvm::core::LLVMBuildOr(self.b, a, b, n);
                let and = |a, b| llvm::core::LLVMBuildAnd(self.b, a, b, n);
                let byz = self.vfadd(b_y, b_z);
                let reject_pos = and(
                    fc(LLVMRealOGT, denom, zero),
                    or(
                        or(
                            or(fc(LLVMRealOLT, b_y, zero), fc(LLVMRealOGT, b_y, denom)),
                            or(fc(LLVMRealOLT, b_z, zero), fc(LLVMRealOGT, byz, denom)),
                        ),
                        fc(LLVMRealOLT, t_hit, zero),
                    ),
                );
                let reject_neg = and(
                    fc(LLVMRealOLT, denom, zero),
                    or(
                        or(
                            or(fc(LLVMRealOGT, b_y, zero), fc(LLVMRealOLT, b_y, denom)),
                            or(fc(LLVMRealOGT, b_z, zero), fc(LLVMRealOLT, byz, denom)),
                        ),
                        fc(LLVMRealOGT, t_hit, zero),
                    ),
                );
                // A ray in the plane of the triangle meets nothing either.
                let miss = or(or(reject_pos, reject_neg), fc(LLVMRealOEQ, denom, zero));
                let sel = |c, a, b| llvm::core::LLVMBuildSelect(self.b, c, a, b, n);
                // The numerator a miss returns is infinite, signed with the
                // denominator so that the quotient is positive infinity
                // whichever way round the triangle faces.
                let missed = self.vf32_of(llvm::core::LLVMBuildOr(
                    self.b,
                    llvm::core::LLVMBuildAnd(
                        self.b,
                        self.vf32_bits(denom),
                        self.vci32(0x8000_0000u32 as u32),
                        n,
                    ),
                    self.vci32(0x7F80_0000),
                    n,
                ));
                // `flags` picks a barycentric per output; it is uniform, so the
                // index is a scalar and the selects are scalar-controlled.
                let bary = |shift: u32| -> LLVMValueRef {
                    let idx = llvm::core::LLVMBuildAnd(
                        self.b,
                        llvm::core::LLVMBuildLShr(self.b, flags, self.ci32(shift), n),
                        self.ci32(3),
                        n,
                    );
                    let is1 = llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, idx, self.ci32(1), n);
                    let is2 = llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, idx, self.ci32(2), n);
                    // The fourth encoding is reserved, and the part answers it
                    // with the first barycentric.
                    sel(is1, b_y, sel(is2, b_z, b_x))
                };
                let tri_res = [sel(miss, missed, t_hit), denom, bary(0), bary(2)];
                let tri_bits: Vec<LLVMValueRef> =
                    tri_res.iter().map(|&v| self.vf32_bits(v)).collect();
                llvm::core::LLVMBuildBr(self.b, join);
                let tri_end = llvm::core::LLVMGetInsertBlock(self.b);


                // ---- divergent nodes, triangle nodes, or NaN --------------
                llvm::core::LLVMPositionBuilderAtEnd(self.b, slow);
                let field_ptr = |f: u32| -> LLVMValueRef {
                    llvm::core::LLVMBuildStructGEP2(
                        self.b,
                        self.bvh_packet_ty,
                        self.bvh_packet,
                        f,
                        n,
                    )
                };
                let inputs = [
                    addr,
                    extent,
                    origin[0],
                    origin[1],
                    origin[2],
                    self.vf32_of(a[7]),
                    self.vf32_of(a[8]),
                    self.vf32_of(a[9]),
                    inv[0],
                    inv[1],
                    inv[2],
                ];
                let step = self.w.min(crate::rdna_translator::bvh::BVH_RAY_PACKET_LANES as u32);
                let slice = |v: LLVMValueRef, off: u32, len: u32| -> LLVMValueRef {
                    if off == 0 && len == self.w { return v; }
                    let mut idx: Vec<LLVMValueRef> = (0..len).map(|k| self.ci32(off + k)).collect();
                    let mask = llvm::core::LLVMConstVector(idx.as_mut_ptr(), len);
                    llvm::core::LLVMBuildShuffleVector(self.b, v, llvm::core::LLVMGetPoison(llvm::core::LLVMTypeOf(v)), mask, n)
                };
                let mut chunks: Vec<Vec<LLVMValueRef>> = Vec::new();
                let mut off = 0;
                while off < self.w {
                    let len = step.min(self.w - off);
                    for (f, value) in inputs.iter().copied().enumerate() {
                        let store = llvm::core::LLVMBuildStore(self.b, slice(value, off, len), field_ptr(f as u32));
                        llvm::core::LLVMSetAlignment(store, if f == 0 { 8 } else { 4 });
                    }
                    self.call(
                        "image_bvh64_intersect_ray_packet",
                        llvm::core::LLVMVoidTypeInContext(self.ctx),
                        &[self.ptr, self.i32t, self.i32t, self.i32t, self.i32t],
                        &[
                            self.bvh_packet,
                            self.ci32(len),
                            llvm::core::LLVMBuildLShr(self.b, mask, self.ci32(off), n),
                            resource[0],
                            resource[1],
                        ],
                    );
                    chunks.push((0..4).map(|k| {
                        let ty = llvm::core::LLVMVectorType(self.i32t, len);
                        let ld = llvm::core::LLVMBuildLoad2(self.b, ty, field_ptr(11 + k), n);
                        llvm::core::LLVMSetAlignment(ld, 4);
                        ld
                    }).collect());
                    off += len;
                }
                let slow_res: Vec<LLVMValueRef> = (0..4).map(|k| {
                    let mut value = chunks[0][k as usize];
                    let mut have = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(value));
                    for chunk in &chunks[1..] {
                        let next = chunk[k as usize];
                        let more = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(next));
                        let wide = have.max(more);
                        let pad = |v: LLVMValueRef, n_have: u32| if n_have == wide { v } else {
                            let mut idx: Vec<LLVMValueRef> = (0..wide).map(|j| self.ci32(j.min(n_have - 1))).collect();
                            let m = llvm::core::LLVMConstVector(idx.as_mut_ptr(), wide);
                            llvm::core::LLVMBuildShuffleVector(self.b, v, llvm::core::LLVMGetPoison(llvm::core::LLVMTypeOf(v)), m, n)
                        };
                        let (a, b) = (pad(value, have), pad(next, more));
                        let mut idx: Vec<LLVMValueRef> = (0..have).chain(wide..wide + more).map(|j| self.ci32(j)).collect();
                        let m = llvm::core::LLVMConstVector(idx.as_mut_ptr(), idx.len() as u32);
                        value = llvm::core::LLVMBuildShuffleVector(self.b, a, b, m, n);
                        have += more;
                    }
                    value
                }).collect();
                llvm::core::LLVMBuildBr(self.b, join);
                let slow_end = llvm::core::LLVMGetInsertBlock(self.b);

                llvm::core::LLVMPositionBuilderAtEnd(self.b, join);
                // All phis must sit at the top of the block, so build them
                // before any of the register writes.
                let phis: Vec<LLVMValueRef> = (0..4)
                    .map(|k| {
                        let phi = llvm::core::LLVMBuildPhi(self.b, self.vi32, n);
                        let mut vals = [child[k], tri_bits[k], slow_res[k]];
                        let mut blocks = [fast_end, tri_end, slow_end];
                        llvm::core::LLVMAddIncoming(
                            phi,
                            vals.as_mut_ptr(),
                            blocks.as_mut_ptr(),
                            3,
                        );
                        phi
                    })
                    .collect();
                phis
    }
}
