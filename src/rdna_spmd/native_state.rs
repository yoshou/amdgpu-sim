//! SSA definitions for physical register representations in the native CFG.
//!
//! A representation has a type, never an address. Reads at block entry create
//! incomplete phis; sealing uses the completed native CFG, including mask
//! bridges, specialized blocks and cooperative resume edges.
use llvm_sys::{core::*, prelude::*};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CellId(usize);

#[derive(Default)]
pub(super) struct State {
    types: Vec<LLVMTypeRef>,
    definitions: BTreeMap<(LLVMBasicBlockRef, CellId), LLVMValueRef>,
    incomplete: Vec<(LLVMBasicBlockRef, CellId, LLVMValueRef)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify before optimization: O3 must not be needed to repair SSA edges.
    unsafe fn verify(module: LLVMModuleRef) {
        let mut message = std::ptr::null_mut();
        let status = llvm_sys::analysis::LLVMVerifyModule(
            module, llvm_sys::analysis::LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut message,
        );
        let diagnostic = if message.is_null() { String::new() } else {
            let s = std::ffi::CStr::from_ptr(message).to_string_lossy().into_owned();
            LLVMDisposeMessage(message);
            s
        };
        assert_eq!(status, 0, "{}", diagnostic);
    }

    #[test]
    fn loop_reads_previous_iteration_and_keeps_later_definition() {
        unsafe {
            let ctx = LLVMContextCreate();
            let module = LLVMModuleCreateWithNameInContext(b"test\0".as_ptr().cast(), ctx);
            let ty = LLVMInt32TypeInContext(ctx);
            let fty = LLVMFunctionType(ty, std::ptr::null_mut(), 0, 0);
            let func = LLVMAddFunction(module, b"loop\0".as_ptr().cast(), fty);
            let blocks: Vec<_> = (0..3).map(|_| LLVMAppendBasicBlockInContext(ctx, func, b"\0".as_ptr().cast())).collect();
            let b = LLVMCreateBuilderInContext(ctx);
            let mut state = State::default();
            let reg = state.add(ty);
            let zero = LLVMConstInt(ty, 0, 0);
            let one = LLVMConstInt(ty, 1, 0);
            LLVMPositionBuilderAtEnd(b, blocks[0]);
            state.write(b, reg, zero);
            LLVMBuildBr(b, blocks[1]);
            LLVMPositionBuilderAtEnd(b, blocks[1]);
            let before = state.read(b, reg);
            let after = LLVMBuildAdd(b, before, one, b"\0".as_ptr().cast());
            state.write(b, reg, after);
            let cond = LLVMBuildICmp(b, llvm_sys::LLVMIntPredicate::LLVMIntULT, after, LLVMConstInt(ty, 4, 0), b"\0".as_ptr().cast());
            LLVMBuildCondBr(b, cond, blocks[1], blocks[2]);
            LLVMPositionBuilderAtEnd(b, blocks[2]);
            let result = state.read(b, reg);
            let ret = LLVMBuildRet(b, result);
            state.finish(func);
            verify(module);
            assert_eq!(LLVMCountIncoming(before), 2);
            assert_eq!(LLVMGetIncomingValue(before, 0), zero);
            assert_eq!(LLVMGetIncomingValue(before, 1), after);
            assert_eq!(LLVMGetOperand(ret, 0), after);
            LLVMDisposeBuilder(b);
            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
        }
    }

    #[test]
    fn duplicate_resume_edges_and_undefined_state_are_valid_ssa() {
        unsafe {
            let ctx = LLVMContextCreate();
            let module = LLVMModuleCreateWithNameInContext(b"test\0".as_ptr().cast(), ctx);
            let ty = LLVMInt32TypeInContext(ctx);
            let mut params = [ty];
            let fty = LLVMFunctionType(ty, params.as_mut_ptr(), 1, 0);
            let func = LLVMAddFunction(module, b"resume\0".as_ptr().cast(), fty);
            let blocks: Vec<_> = (0..4).map(|_| LLVMAppendBasicBlockInContext(ctx, func, b"\0".as_ptr().cast())).collect();
            let b = LLVMCreateBuilderInContext(ctx);
            let mut state = State::default();
            let reg = state.add(ty);
            LLVMPositionBuilderAtEnd(b, blocks[0]);
            state.write(b, reg, LLVMConstInt(ty, 7, 0));
            let switch = LLVMBuildSwitch(b, LLVMGetParam(func, 0), blocks[2], 2);
            LLVMAddCase(switch, LLVMConstInt(ty, 0, 0), blocks[2]);
            LLVMAddCase(switch, LLVMConstInt(ty, 1, 0), blocks[1]);
            LLVMPositionBuilderAtEnd(b, blocks[1]);
            state.write(b, reg, LLVMConstInt(ty, 9, 0));
            LLVMBuildBr(b, blocks[2]);
            LLVMPositionBuilderAtEnd(b, blocks[2]);
            let result = state.read(b, reg);
            LLVMBuildRet(b, result);
            // An unreachable block may read a representation never initialized.
            LLVMPositionBuilderAtEnd(b, blocks[3]);
            let undef = state.read(b, reg);
            let ret = LLVMBuildRet(b, undef);
            state.finish(func);
            verify(module);
            assert_eq!(LLVMCountIncoming(result), 3);
            assert_ne!(LLVMIsUndef(LLVMGetOperand(ret, 0)), 0);
            LLVMDisposeBuilder(b);
            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
        }
    }
}

impl State {
    pub fn add(&mut self, ty: LLVMTypeRef) -> CellId {
        let cell = CellId(self.types.len());
        self.types.push(ty);
        cell
    }

    pub unsafe fn read(&mut self, builder: LLVMBuilderRef, cell: CellId) -> LLVMValueRef {
        self.read_at(LLVMGetInsertBlock(builder), cell)
    }

    unsafe fn read_at(&mut self, block: LLVMBasicBlockRef, cell: CellId) -> LLVMValueRef {
        if let Some(&value) = self.definitions.get(&(block, cell)) {
            return value;
        }
        let builder = LLVMCreateBuilderInContext(LLVMGetTypeContext(self.types[cell.0]));
        let first = LLVMGetFirstInstruction(block);
        if first.is_null() {
            LLVMPositionBuilderAtEnd(builder, block);
        } else {
            LLVMPositionBuilderBefore(builder, first);
        }
        let phi = LLVMBuildPhi(builder, self.types[cell.0], b"\0".as_ptr().cast());
        LLVMDisposeBuilder(builder);
        self.definitions.insert((block, cell), phi);
        self.incomplete.push((block, cell, phi));
        phi
    }

    pub unsafe fn write(&mut self, builder: LLVMBuilderRef, cell: CellId, value: LLVMValueRef) {
        assert_eq!(LLVMTypeOf(value), self.types[cell.0], "register SSA type mismatch");
        self.definitions.insert((LLVMGetInsertBlock(builder), cell), value);
    }

    /// Complete all incoming edges before simplifying any phi. Keeping the
    /// final block definitions alive until then handles backedges and values
    /// read before a later definition in the same block.
    pub unsafe fn finish(&mut self, function: LLVMValueRef) {
        let mut predecessors: BTreeMap<LLVMBasicBlockRef, Vec<LLVMBasicBlockRef>> = BTreeMap::new();
        let mut block = LLVMGetFirstBasicBlock(function);
        while !block.is_null() {
            let term = LLVMGetBasicBlockTerminator(block);
            assert!(!term.is_null(), "unfinished native SSA block");
            for edge in 0..LLVMGetNumSuccessors(term) {
                predecessors.entry(LLVMGetSuccessor(term, edge)).or_default().push(block);
            }
            block = LLVMGetNextBasicBlock(block);
        }
        let mut index = 0;
        while index < self.incomplete.len() {
            let (block, cell, phi) = self.incomplete[index];
            let mut blocks = predecessors.get(&block).cloned().unwrap_or_default();
            let mut values: Vec<_> = blocks.iter().map(|&pred| self.read_at(pred, cell)).collect();
            LLVMAddIncoming(phi, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
            index += 1;
        }
        self.definitions.clear();
        // Removing a trivial phi can make its users trivial, including phis
        // already visited. LLVM's general optimization handles nontrivial SCCs.
        loop {
            let mut changed = false;
            for (_, cell, phi) in &mut self.incomplete {
                if phi.is_null() { continue; }
                let mut same = None;
                let mut trivial = true;
                for index in 0..LLVMCountIncoming(*phi) {
                    let value = LLVMGetIncomingValue(*phi, index);
                    if value == *phi { continue; }
                    match same {
                        Some(previous) if previous != value => { trivial = false; break; }
                        _ => same = Some(value),
                    }
                }
                if trivial {
                    let value = match same {
                        Some(value) => value,
                        None => LLVMGetUndef(self.types[cell.0]),
                    };
                    LLVMReplaceAllUsesWith(*phi, value);
                    LLVMInstructionEraseFromParent(*phi);
                    *phi = std::ptr::null_mut();
                    changed = true;
                }
            }
            if !changed { break; }
        }
        // Canonical register order keeps LLVM's scheduling and register
        // allocation independent of the order in which lazy reads discovered
        // the phis. Insert in ascending cell order at the block start, giving
        // descending cell order in the final IR, as register promotion did.
        let mut phis: Vec<_> = self.incomplete.iter()
            .filter(|x| !x.2.is_null()).copied().collect();
        phis.sort_by_key(|x| (x.0, x.1));
        for (block, _, phi) in phis {
            let first = LLVMGetFirstInstruction(block);
            if first == phi { continue; }
            LLVMInstructionRemoveFromParent(phi);
            let b = LLVMCreateBuilderInContext(LLVMGetTypeContext(LLVMTypeOf(phi)));
            LLVMPositionBuilderBefore(b, first);
            LLVMInsertIntoBuilder(b, phi);
            LLVMDisposeBuilder(b);
        }
        self.incomplete.clear();
    }
}
