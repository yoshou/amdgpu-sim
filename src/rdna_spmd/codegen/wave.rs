use super::super::engine::yields::Argument;
use super::*;

impl<'a> Cg<'a> {
    fn spill_slot_ptr(&self, vgpr: u32, lane: u32) -> Value {
        let idx = {
            let mut m = self.spill.borrow_mut();
            let next = m.len();
            *m.entry((vgpr, lane)).or_insert(next)
        };
        assert!(
            idx < super::super::engine::kernel::COOP_SPILL_SLOTS,
            "too many writelane/readlane spill slots"
        );
        self.register_slot(self.spill_base, idx as u32)
    }
    fn lane_constant(&self, lane: ValueId) -> u32 {
        self.p.constants[lane.0].expect("cross-lane access requires a constant lane") as u32
    }

    pub(super) fn emit_local_wave(
        &mut self,
        op: EffectOp,
        inputs: &[ValueId],
        outputs: &[(ValueId, Ty)],
    ) {
        let ir = self.ir;
        match op {
            EffectOp::Wave(WaveOp::Any) | EffectOp::Wave(WaveOp::Ballot) => {
                let bits = match self.p.width {
                    Some(_) => {
                        let v = self.vector(inputs[0]);
                        self.vec_to_mask(v)
                    }
                    None => {
                        let v = self.scalar(inputs[0]);
                        ir.zext(v, ir.i32())
                    }
                };
                let result = if op == EffectOp::Wave(WaveOp::Any) {
                    ir.icmp(IntPred::Ne, bits, self.ci32(0))
                } else {
                    bits
                };
                self.define(outputs[0].0, result);
            }
            EffectOp::Wave(WaveOp::ReadFirstLane) => {
                let result = if let Some(w) = self.p.width {
                    let src = self.vector(inputs[0]);
                    let exec = self.vector(inputs[1]);
                    let exec = self.em.to_bool(exec);
                    let word = self.vec_to_mask(exec);
                    let tz = ir.call_named(
                        "llvm.cttz.i32",
                        ir.i32(),
                        &[ir.i32(), ir.i1()],
                        &[word, ir.ci1(false)],
                    );
                    let over = ir.icmp(IntPred::Uge, tz, self.ci32(w));
                    let idx = ir.select(over, self.ci32(0), tz);
                    ir.extract(src, idx)
                } else {
                    self.scalar(inputs[0])
                };
                self.define(outputs[0].0, result);
            }
            EffectOp::Wave(WaveOp::WriteLane) => {
                let lane = self.lane_constant(inputs[1]);
                let reg = self.lane_constant(inputs[3]);
                let value = self.scalar(inputs[0]);
                let slot = self.spill_slot_ptr(reg, lane);
                ir.store(value, slot);
                let old = self.values[inputs[2].0];
                self.define(outputs[0].0, old);
            }
            EffectOp::Wave(WaveOp::ReadLane) => {
                let lane = self.lane_constant(inputs[1]);
                let reg = self.lane_constant(inputs[2]);
                assert!(reg != u32::MAX, "readlane source must be a VGPR");
                let slot = self.spill_slot_ptr(reg, lane);
                let value = ir.load(ir.i32(), slot);
                self.define(outputs[0].0, value);
            }
            _ => panic!("wave operation {:?} requires cooperative dispatch", op),
        }
    }

    pub(super) fn emit_yield(&mut self, members: &[(u64, Vec<ValueId>, Vec<(ValueId, Ty)>)]) {
        let ir = self.ir;
        let resume = self.p.resume_index(members[0].0);
        let width = self.width() as u64;
        let frame = self.yield_frame;
        let pointer =
            |slot: usize| ir.gep(ir.i32(), frame, &[ir.ci32((slot as u64 * width) as u32)]);
        for (provenance, inputs, _) in members {
            let layout = self
                .p
                .yields
                .get(provenance)
                .expect("scheduled effect lacks a yield layout")
                .clone();
            for (i, (&id, &ty)) in inputs.iter().zip(&layout.inputs).enumerate() {
                if matches!(layout.arguments[i], Argument::Constant(_)) {
                    continue;
                }
                let uniform = matches!(layout.arguments[i], Argument::Uniform);
                let value = self.shaped(id, uniform || self.p.width.is_none());
                let int_ty = if uniform || self.p.width.is_none() {
                    ir.i32()
                } else {
                    self.vi32()
                };
                let bits = match ty {
                    Ty::I1 => ir.zext(self.em.to_bool(value), int_ty),
                    Ty::F32 => ir.bitcast(value, int_ty),
                    Ty::I32 => value,
                    _ => unreachable!("wide wave operand"),
                };
                ir.store(bits, pointer(layout.base + i)).set_alignment(4);
            }
        }
        let ty = ir.void().function(&[ir.ptr(), ir.i64(), ir.ptr()]);
        let function = ir.function("amdgpu_sim_fiber_yield_values", ty);
        let context = self.func.param(7);
        ir.call(ty, function, &[context, ir.ci64(resume as u64), frame]);
        for (provenance, _, outputs) in members {
            let layout = self
                .p
                .yields
                .get(provenance)
                .expect("scheduled effect lacks a yield layout")
                .clone();
            let uniform = layout.uniform_result();
            for (i, &(id, ty)) in outputs.iter().enumerate() {
                let int_ty = if uniform || self.p.width.is_none() {
                    ir.i32()
                } else {
                    self.vi32()
                };
                let bits = ir
                    .load(int_ty, pointer(layout.base + layout.output_base + i))
                    .set_alignment(4);
                let bool_ty = if uniform || self.p.width.is_none() {
                    ir.i1()
                } else {
                    self.vi1()
                };
                let f32_ty = if uniform || self.p.width.is_none() {
                    ir.f32()
                } else {
                    self.vec_ty(ir.f32())
                };
                let result = match ty {
                    Ty::I1 => self.em.from_bool(ir.trunc(bits, bool_ty)),
                    Ty::F32 => ir.bitcast(bits, f32_ty),
                    Ty::I32 => bits,
                    _ => unreachable!("wide wave result"),
                };
                self.define(id, result);
            }
        }
    }
}
