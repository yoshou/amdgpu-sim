//! Scalar physical lowering of verified memory effects.
use super::*;
use crate::rdna_spmd::{
    ir::typed::effect::*,
    lift::memory::{Parameter, Plan},
    memory_shape::ScalarShape,
    typed_codegen::Values,
};
impl Cg {
    pub(super) unsafe fn memory_parameter(&self, p: &Parameter, _: bool) -> LLVMValueRef {
        match p {
            Parameter::Register(input) => self.typed_input(input),
            Parameter::Exec => llvm::core::LLVMBuildICmp(
                self.b,
                llvm::LLVMIntPredicate::LLVMIntNE,
                self.b_and(self.ld_sgpr32(EXEC), self.ci32(1)),
                self.ci32(0),
                self.n(),
            ),
            Parameter::ScratchBase => self.private_base,
            Parameter::ScratchSize => self.private_size,
        }
    }
    pub(super) unsafe fn emit_memory(
        &self,
        plan: &Plan,
        shape: ScalarShape,
        values: &Values,
        data: impl Fn(u32) -> LLVMValueRef,
    ) {
        let m = &plan.memory;
        assert!(
            m.space() != Space::Lds || self.coop,
            "LDS requires cooperative dispatch"
        );
        if shape == ScalarShape::Fence {
            let order = match m.semantics.ordering {
                Ordering::Acquire => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingAcquire,
                Ordering::Release => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingRelease,
                _ => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
            };
            llvm::core::LLVMBuildFence(self.b, order, 0, self.n());
            return;
        }
        let mut addr = values.value(plan.address);
        if let Some((_,inside,..))=&plan.flat {
            let offset=llvm::core::LLVMBuildSub(self.b,self.scratch_base,self.private_base,self.n());
            let physical=self.b_add(addr,offset);
            addr=llvm::core::LLVMBuildSelect(self.b,values.value(*inside),physical,addr,self.n());
        }
        if m.space() != Space::Global {
            addr = self.b_add(
                if m.space() == Space::Scratch {
                    self.scratch_base
                } else {
                    self.lds_base
                },
                if m.space() == Space::Scratch {
                    llvm::core::LLVMBuildSExt(self.b, addr, self.i64t, self.n())
                } else {
                    self.zext64(addr)
                },
            );
        }
        // A false load mask must not dereference a lane's possibly invalid pointer.
        let load_addr = if !m.scalar() && self.predicate.get() {
            self.store_addr(addr)
        } else {
            addr
        };
        if shape == ScalarShape::AtomicAdd {
            let ptr = self.ptr_at(self.store_addr(addr), 0);
            let old = llvm::core::LLVMBuildAtomicRMW(
                self.b,
                llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                ptr,
                data(0),
                llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                0,
            );
            if m.returns {
                self.st_vgpr32(m.dest, old);
            }
            return;
        }
        let elem = match m.size().bytes() {
            1 => self.i8,
            2 => self.i16ty(),
            _ => self.i32t,
        };
        if shape == ScalarShape::Store {
            for k in 0..m.words {
                let value = if m.size() == MemSize::B32 {
                    data(k)
                } else {
                    llvm::core::LLVMBuildTrunc(self.b, data(k), elem, self.n())
                };
                let p = self.ptr_at(
                    self.store_addr(self.b_add(addr, self.ci64(m.word_offset(k) as u64))),
                    0,
                );
                let store = llvm::core::LLVMBuildStore(self.b, value, p);
                llvm::core::LLVMSetAlignment(store, if m.space() == Space::Lds { 1 } else { m.size().bytes() });
                llvm::core::LLVMSetVolatile(store, m.semantics.volatile as i32);
            }
            return;
        }
        let ScalarShape::Words { pairs } = shape else { unreachable!() };
        let mut k = 0;
        while k < m.words {
            if pairs && k + 1 < m.words {
                let value = llvm::core::LLVMBuildLoad2(
                    self.b,
                    self.f64t,
                    self.ptr_at(load_addr, m.word_offset(k) as u64),
                    self.n(),
                );
                llvm::core::LLVMSetAlignment(value, 4);
                self.st_vgpr_f64(m.dest + k, value);
                k += 2;
            } else {
                let load = llvm::core::LLVMBuildLoad2(
                    self.b,
                    elem,
                    self.ptr_at(load_addr, m.word_offset(k) as u64),
                    self.n(),
                );
                llvm::core::LLVMSetAlignment(load, if m.space() == Space::Lds || m.size() != MemSize::B32 { 1 } else { 4 });
                llvm::core::LLVMSetVolatile(load, m.semantics.volatile as i32);
                let value = if m.size() == MemSize::B32 {
                    load
                } else if m.size().signed() {
                    llvm::core::LLVMBuildSExt(self.b, load, self.i32t, self.n())
                } else {
                    llvm::core::LLVMBuildZExt(self.b, load, self.i32t, self.n())
                };
                if m.scalar() {
                    let (raw,result)=plan.scalar_results[k as usize];
                    let (ty,value)=values.effect_result(raw,result,value);
                    self.typed_output(if ty==super::super::ir::typed::Ty::I1 {super::super::lift::Output::MaskBit(m.dest+k)}
                        else {super::super::lift::Output::Scalar(m.dest+k,ty)},value);
                } else {
                    self.st_vgpr32(m.dest + k, value);
                }
                k += 1;
            }
        }
    }
}
