//! Packet physical lowering of memory effects, retaining pairs and transpose.
use super::*;
use crate::rdna_spmd::{
    ir::typed::{effect::*, Ty},
    lift::memory::{Address, Parameter, Plan},
    typed_codegen::Values,
};
impl Cg {
    /// Serialize each equal-address group of an unused-result atomic once.
    /// The pending mask contains only active lanes, so neither the selected
    /// pointer nor any gathered addend can come from an inactive lane.
    unsafe fn emit_grouped_atomic_add(&self, addresses: LLVMValueRef, values: LLVMValueRef, exec: LLVMValueRef) {
        use llvm::core::*;
        let b = self.b; let n = self.n();
        // Inactive pointer lanes may be poison. Comparing then packing them
        // before the active-mask AND would otherwise poison the whole mask.
        let addresses = LLVMBuildFreeze(b, addresses, n);
        let entry = LLVMGetInsertBlock(b); let function = LLVMGetBasicBlockParent(entry);
        let header = LLVMAppendBasicBlockInContext(self.ctx, function, cstr("atomic.groups").as_ptr());
        let body = LLVMAppendBasicBlockInContext(self.ctx, function, cstr("atomic.group").as_ptr());
        let done = LLVMAppendBasicBlockInContext(self.ctx, function, cstr("atomic.done").as_ptr());
        let initial = self.vec_to_mask(exec);
        LLVMBuildBr(b, header); LLVMPositionBuilderAtEnd(b, header);
        let pending = LLVMBuildPhi(b, self.i32t, n);
        LLVMAddIncoming(pending, [initial].as_mut_ptr(), [entry].as_mut_ptr(), 1);
        let nonempty = LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntNE, pending, self.ci32(0), n);
        LLVMBuildCondBr(b, nonempty, body, done); LLVMPositionBuilderAtEnd(b, body);
        let lane = self.call("llvm.cttz.i32", self.i32t, &[self.i32t, self.i1],
            &[pending, LLVMConstInt(self.i1, 1, 0)]);
        let address = LLVMBuildExtractElement(b, addresses, lane, n);
        let equal = LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntEQ, addresses, self.splat(address, self.vi64), n);
        let members = LLVMBuildAnd(b, self.vec_to_mask(equal), pending, n);
        let mask = self.mask_to_vec(members);
        let addends = LLVMBuildSelect(b, mask, values, LLVMConstNull(self.vi32), n);
        let sum = self.call(&format!("llvm.vector.reduce.add.v{}i32", self.w), self.i32t, &[self.vi32], &[addends]);
        let pointer = LLVMBuildIntToPtr(b, address, self.ptr, n);
        LLVMBuildAtomicRMW(b, llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
            pointer, sum, llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent, 0);
        let remaining = LLVMBuildAnd(b, pending, LLVMBuildNot(b, members, n), n);
        let backedge = LLVMGetInsertBlock(b);
        LLVMBuildBr(b, header); LLVMAddIncoming(pending, [remaining].as_mut_ptr(), [backedge].as_mut_ptr(), 1);
        LLVMPositionBuilderAtEnd(b, done);
    }
    pub(super) unsafe fn memory_parameter(&self, p: &Parameter, scalar: bool) -> LLVMValueRef {
        match p {
            Parameter::Register(input) if scalar => match input.ty {
                Ty::I64 => self.ssrc_u64(input.source.operand()),
                Ty::I32 => self.ssrc_u32(input.source.operand()),
                _ => unreachable!(),
            },
            Parameter::Register(input) => self.typed_input(input, false),
            Parameter::Exec => self.exec_vec(),
            Parameter::ScratchBase => self.splat(self.scratch_base_scalar, self.vi64),
            Parameter::ScratchSize => self.splat(self.scratch_stride, self.vi64),
        }
    }
    pub(super) unsafe fn emit_memory(
        &self,
        plan: &Plan,
        values: &Values,
        data: impl Fn(u32) -> LLVMValueRef,
    ) {
        let m = &plan.memory;
        assert!(
            m.space() != Space::Lds || self.coop,
            "LDS requires cooperative dispatch"
        );
        if m.op == MemoryOp::Fence {
            let order = match m.semantics.ordering {
                Ordering::Acquire => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingAcquire,
                Ordering::Release => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingRelease,
                _ => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
            };
            llvm::core::LLVMBuildFence(self.b, order, 0, self.n());
            return;
        }
        let base = values.value(plan.base);
        let mut addr = values.value(plan.address);
        let exec = values.value(plan.mask);
        let elem = match m.size().bytes() {
            1 => llvm::core::LLVMInt8TypeInContext(self.ctx),
            2 => llvm::core::LLVMInt16TypeInContext(self.ctx),
            _ => self.i32t,
        };
        if m.scalar() {
            for k in 0..m.words {
                let a = llvm::core::LLVMBuildAdd(self.b, addr, self.ci64(m.word_offset(k) as u64), self.n());
                let ptr = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
                let ld = llvm::core::LLVMBuildLoad2(self.b, elem, ptr, self.n());
                llvm::core::LLVMSetAlignment(ld, if m.size() == MemSize::B32 { 4 } else { 1 });
                let value = if m.size() == MemSize::B32 {
                    ld
                } else if m.size().signed() {
                    llvm::core::LLVMBuildSExt(self.b, ld, self.i32t, self.n())
                } else {
                    llvm::core::LLVMBuildZExt(self.b, ld, self.i32t, self.n())
                };
                self.st_sgpr32(m.dest + k, value);
            }
            return;
        }
        if m.space() == Space::Scratch {
            addr = self.v_add(
                self.scratch_vec,
                llvm::core::LLVMBuildSExt(self.b, addr, self.vi64, self.n()),
            );
        }
        if m.space() == Space::Lds {
            addr = self.v_add(self.splat(self.lds_base, self.vi64), self.zext64v(addr));
        }
        if let Some((_, inside, _, _)) = &plan.flat {
            // Fuse the aperture's I32 offset with the physical lane base. A
            // defined private access stays within its allocated segment, so
            // narrowing and re-extending the offset cannot change these bits.
            // False masks still suppress invalid addresses at the memory op.
            let lane_offset = llvm::core::LLVMBuildSub(
                self.b, self.scratch_vec,
                self.splat(self.scratch_base_scalar,self.vi64), self.n());
            let physical = self.v_add(addr, lane_offset);
            addr = llvm::core::LLVMBuildSelect(
                self.b,
                values.value(*inside),
                physical,
                addr,
                self.n(),
            );
        }
        if m.op == MemoryOp::AtomicAdd {
            if plan.group_atomics {
                self.emit_grouped_atomic_add(addr, data(0), exec);
                return;
            }
            // Atomics scalarize by lane. Pack the typed I1 mask once, retaining
            // the existing scalar bit tests instead of expanding a vector mask.
            let packed_exec = self.vec_to_mask(exec);
            let ptrs = self.ptr_at_vec(addr, 0);
            let mut result = llvm::core::LLVMGetPoison(self.vi32);
            for k in 0..self.w {
                let bit = llvm::core::LLVMBuildAnd(self.b,
                    llvm::core::LLVMBuildLShr(self.b,packed_exec,self.ci32(k),self.n()),self.ci32(1),self.n());
                let active = llvm::core::LLVMBuildICmp(self.b,
                    llvm::LLVMIntPredicate::LLVMIntNE,bit,self.ci32(0),self.n());
                let ptr = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(k), self.n());
                let ptr =
                    llvm::core::LLVMBuildSelect(self.b, active, ptr, self.bvh_scratch, self.n());
                let value =
                    llvm::core::LLVMBuildExtractElement(self.b, data(0), self.ci32(k), self.n());
                let value =
                    llvm::core::LLVMBuildSelect(self.b, active, value, self.ci32(0), self.n());
                let old = llvm::core::LLVMBuildAtomicRMW(
                    self.b,
                    llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                    ptr,
                    value,
                    llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                    0,
                );
                result =
                    llvm::core::LLVMBuildInsertElement(self.b, result, old, self.ci32(k), self.n());
            }
            if m.returns {
                self.st_vgpr32(m.dest, result);
            }
            return;
        }
        let affine = matches!(m.address, Address::Scratch { vector: None, .. });
        if m.stores() {
            for k in 0..m.words {
                let ptrs = self.ptr_at_vec(addr, m.word_offset(k) as u64);
                let value = data(k);
                if m.size() != MemSize::B32 {
                    let value = llvm::core::LLVMBuildTrunc(
                        self.b,
                        value,
                        llvm::core::LLVMVectorType(elem, self.w),
                        self.n(),
                    );
                    self.masked_scatter_ty(value, ptrs, exec, elem);
                } else if m.space() == Space::Lds {
                    self.masked_scatter_ty(value, ptrs, exec, self.i32t);
                } else if affine {
                    self.affine_store(value, ptrs, exec);
                } else {
                    self.masked_scatter(value, ptrs, exec);
                }
            }
            return;
        }
        if m.size() != MemSize::B32 {
            let value = self.masked_gather_ty(self.ptr_at_vec(addr, 0), exec, elem);
            let value = if m.size().signed() {
                llvm::core::LLVMBuildSExt(self.b, value, self.vi32, self.n())
            } else {
                llvm::core::LLVMBuildZExt(self.b, value, self.vi32, self.n())
            };
            self.st_vgpr32(m.dest, value);
            return;
        }
        if m.private_load_end().is_some() && m.words >= 2 {
            let tile = if self.w % 4 == 0 {
                4
            } else if self.w % 2 == 0 {
                2
            } else {
                1
            };
            let rowty = llvm::core::LLVMVectorType(self.i32t, m.words);
            let rows: Vec<_> = (0..self.w)
                .map(|l| {
                    let a =
                        llvm::core::LLVMBuildExtractElement(self.b, addr, self.ci32(l), self.n());
                    let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
                    // min_private_bytes is part of the compiled kernel's ABI;
                    // all lanes own this cell, including padding/inactive lanes.
                    let load=llvm::core::LLVMBuildLoad2(self.b,rowty,p,self.n());
                    llvm::core::LLVMSetAlignment(load,4);load
                })
                .collect();
            let cols = self.transpose_rows(&rows, m.words, tile);
            for k in 0..m.words {
                self.st_vgpr32(m.dest + k, cols[k as usize]);
            }
            return;
        }
        let global = matches!(m.address, Address::Global { .. });
        let words = m.words;
        let uniform_addr = global && self.global_load.get() == GlobalLoad::Broadcast;
        // The plan proved the frame bounds and alignment. Emit the same grouped
        // contiguous loads and transpose; row crossings still use groups <= 8.
        if let (
            true,
            GlobalLoad::Frame {
                stride_words: sp4,
                offset_words: ioff_w,
            },
        ) = (global, self.global_load.get())
        {
            let grp = self.w.min(8); // lanes per contiguous group (W-aligned, ≤8)
            let nblk = self.w / grp;
            let blkty = llvm::core::LLVMVectorType(self.i32t, grp * sp4);
            let poison_blk = llvm::core::LLVMGetPoison(blkty);
            // Per-group contiguous load from the group's first lane's address.
            let blocks: Vec<LLVMValueRef> = (0..nblk)
                .map(|g| {
                    let a = llvm::core::LLVMBuildExtractElement(
                        self.b,
                        base,
                        self.ci32(g * grp),
                        self.n(),
                    );
                    let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
                    let ld = llvm::core::LLVMBuildLoad2(self.b, blkty, p, self.n());
                    llvm::core::LLVMSetAlignment(ld, 4);
                    ld
                })
                .collect();
            let extract = |fw: u32| -> LLVMValueRef {
                // Transpose field `fw` out of each group, then concat groups.
                let parts: Vec<LLVMValueRef> = blocks
                    .iter()
                    .map(|&blk| {
                        let mut idx: Vec<LLVMValueRef> =
                            (0..grp).map(|lane| self.ci32(lane * sp4 + fw)).collect();
                        let mask = llvm::core::LLVMConstVector(idx.as_mut_ptr(), grp);
                        llvm::core::LLVMBuildShuffleVector(self.b, blk, poison_blk, mask, self.n())
                    })
                    .collect();
                self.vconcat_i32(&parts)
            };
            let mut k = 0u32;
            while k < words {
                if k + 1 < words {
                    let lo = extract(ioff_w as u32 + k);
                    let hi = extract(ioff_w as u32 + k + 1);
                    let lo64 = self.zext64v(lo);
                    let hi64 = llvm::core::LLVMBuildShl(
                        self.b,
                        self.zext64v(hi),
                        self.splat(self.ci64(32), self.vi64),
                        self.n(),
                    );
                    let u = self.v_or(hi64, lo64);
                    let d = llvm::core::LLVMBuildBitCast(self.b, u, self.vf64, self.n());
                    self.st_vgpr_f64(m.dest + k, d);
                    k += 2;
                } else {
                    self.st_vgpr32(m.dest + k, extract(ioff_w as u32 + k));
                    k += 1;
                }
            }
            return;
        }
        let mut k = 0u32;
        while k < words {
            if global && k + 1 < words {
                let ptrs = self.ptr_at_vec(addr, m.word_offset(k) as u64);
                let d = if uniform_addr {
                    self.bcast_load_f64(ptrs)
                } else {
                    self.masked_gather_f64(ptrs, exec)
                };
                self.st_vgpr_f64(m.dest + k, d);
                k += 2;
            } else {
                let ptrs = self.ptr_at_vec(addr, m.word_offset(k) as u64);
                let d = if uniform_addr {
                    self.bcast_load_i32(ptrs)
                } else if m.space() == Space::Lds {
                    self.masked_gather_ty(ptrs, exec, self.i32t)
                } else if affine {
                    self.affine_load(ptrs,m.private_load_end().is_some())
                } else {
                    self.masked_gather(ptrs, exec)
                };
                self.st_vgpr32(m.dest + k, d);
                k += 1;
            }
        }
    }
}
