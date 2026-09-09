use super::*;
use super::super::engine::yields::Argument;

impl<'a> Cg<'a> {
    unsafe fn spill_slot_ptr(&self, vgpr: u32, lane: u32) -> LLVMValueRef {
        let idx = {
            let mut m = self.spill.borrow_mut();
            let next = m.len();
            *m.entry((vgpr, lane)).or_insert(next)
        };
        assert!(idx < super::super::engine::kernel::COOP_SPILL_SLOTS, "too many writelane/readlane spill slots");
        LLVMBuildGEP2(self.b, self.i32t, self.spill_base, [self.ci32(idx as u32)].as_mut_ptr(), 1, self.n())
    }
    unsafe fn lane_constant(&self, lane: ValueId) -> u32 {
        self.p.constants[lane.0].expect("cross-lane access requires a constant lane") as u32
    }

    pub(super) unsafe fn emit_local_wave(&mut self, op: EffectOp, inputs: &[ValueId], outputs: &[(ValueId, Ty)]) {
        let n = self.n();
        match op {
            EffectOp::Wave(WaveOp::Any) | EffectOp::Wave(WaveOp::Ballot) => {
                let bits = match self.p.width {
                    Some(w) => { let v = self.vector(inputs[0]); LLVMBuildZExt(self.b, LLVMBuildBitCast(self.b, v, LLVMIntTypeInContext(self.ctx, w), n), self.i32t, n) }
                    None => { let v = self.scalar(inputs[0]); LLVMBuildZExt(self.b, v, self.i32t, n) }
                };
                let result = if op == EffectOp::Wave(WaveOp::Any) { LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bits, self.ci32(0), n) } else { bits };
                self.define(outputs[0].0, result);
            }
            EffectOp::Wave(WaveOp::ReadFirstLane) => {
                let result = if let Some(w) = self.p.width {
                    let src = self.vector(inputs[0]);
                    let exec = self.vector(inputs[1]);
                    let word = self.vec_to_mask(exec);
                    let tz = self.call_i32("llvm.cttz.i32", &[self.i32t, self.i1], &[word, LLVMConstInt(self.i1, 0, 0)]);
                    let over = LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGE, tz, self.ci32(w), n);
                    let idx = LLVMBuildSelect(self.b, over, self.ci32(0), tz, n);
                    LLVMBuildExtractElement(self.b, src, idx, n)
                } else { self.scalar(inputs[0]) };
                self.define(outputs[0].0, result);
            }
            EffectOp::Wave(WaveOp::WriteLane) => {
                let lane = self.lane_constant(inputs[1]);
                let reg = self.lane_constant(inputs[3]);
                let value = self.scalar(inputs[0]);
                let slot = self.spill_slot_ptr(reg, lane);
                LLVMBuildStore(self.b, value, slot);
                let old = self.values[inputs[2].0];
                self.define(outputs[0].0, old);
            }
            EffectOp::Wave(WaveOp::ReadLane) => {
                let lane = self.lane_constant(inputs[1]);
                let reg = self.lane_constant(inputs[2]);
                assert!(reg != u32::MAX, "readlane source must be a VGPR");
                let slot = self.spill_slot_ptr(reg, lane);
                let value = LLVMBuildLoad2(self.b, self.i32t, slot, n);
                self.define(outputs[0].0, value);
            }
            _ => panic!("wave operation {:?} requires cooperative dispatch", op),
        }
    }

    unsafe fn call_i32(&self, name: &str, params: &[LLVMTypeRef], args: &[LLVMValueRef]) -> LLVMValueRef {
        let cname = cstr(name);
        let mut f = LLVMGetNamedFunction(self.module, cname.as_ptr());
        let fty = LLVMFunctionType(self.i32t, params.as_ptr() as *mut _, params.len() as u32, 0);
        if f.is_null() { f = LLVMAddFunction(self.module, cname.as_ptr(), fty); }
        LLVMBuildCall2(self.b, fty, f, args.as_ptr() as *mut _, args.len() as u32, self.n())
    }

    pub(super) unsafe fn emit_yield(&mut self, provenance: u64, inputs: &[ValueId], outputs: &[(ValueId, Ty)]) {
        let n = self.n();
        let layout = self.p.yields.get(&provenance).expect("scheduled effect lacks a yield layout").clone();
        let resume = self.p.resume_index(provenance);
        let width = self.width() as u64;
        let frame = self.yield_frame;
        let pointer = |cg: &Self, slot: usize| LLVMBuildGEP2(cg.b, cg.i32t, frame, [LLVMConstInt(cg.i32t, slot as u64 * width, 0)].as_mut_ptr(), 1, n);
        for (i, (&id, &ty)) in inputs.iter().zip(&layout.inputs).enumerate() {
            if matches!(layout.arguments[i], Argument::Constant(_)) { continue; }
            let uniform = matches!(layout.arguments[i], Argument::Uniform);
            let value = self.shaped(id, uniform || self.p.width.is_none());
            let int_ty = if uniform || self.p.width.is_none() { self.i32t } else { self.vi32() };
            let bits = match ty {
                Ty::I1 => LLVMBuildZExt(self.b, value, int_ty, n),
                Ty::F32 => LLVMBuildBitCast(self.b, value, int_ty, n),
                Ty::I32 => value,
                _ => unreachable!("wide wave operand"),
            };
            LLVMSetAlignment(LLVMBuildStore(self.b, bits, pointer(self, i)), 4);
        }
        let i64t = self.i64t; let ptr = self.ptr;
        let ty = LLVMFunctionType(LLVMVoidTypeInContext(self.ctx), [ptr, i64t, ptr].as_mut_ptr(), 3, 0);
        let name = b"amdgpu_sim_fiber_yield_values\0".as_ptr().cast();
        let mut function = LLVMGetNamedFunction(self.module, name);
        if function.is_null() { function = LLVMAddFunction(self.module, name, ty); }
        let context = LLVMGetParam(self.func, 7);
        LLVMBuildCall2(self.b, ty, function, [context, LLVMConstInt(i64t, resume as u64, 0), frame].as_mut_ptr(), 3, n);
        let uniform = layout.uniform_result();
        for (i, &(id, ty)) in outputs.iter().enumerate() {
            let int_ty = if uniform || self.p.width.is_none() { self.i32t } else { self.vi32() };
            let bits = LLVMBuildLoad2(self.b, int_ty, pointer(self, layout.output_base + i), n);
            LLVMSetAlignment(bits, 4);
            let bool_ty = if uniform || self.p.width.is_none() { self.i1 } else { self.vi1() };
            let f32_ty = if uniform || self.p.width.is_none() { self.f32t } else { self.vec_ty(self.f32t) };
            let result = match ty {
                Ty::I1 => LLVMBuildTrunc(self.b, bits, bool_ty, n),
                Ty::F32 => LLVMBuildBitCast(self.b, bits, f32_ty, n),
                Ty::I32 => bits,
                _ => unreachable!("wide wave result"),
            };
            self.define(id, result);
        }
    }
}
