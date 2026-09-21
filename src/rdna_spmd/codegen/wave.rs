use super::super::engine::yields::Argument;
use super::*;

impl<'a> Cg<'a> {
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
                    Ty::I1 => ir.zext(value, int_ty),
                    Ty::F32 => ir.bitcast(value, int_ty),
                    Ty::I32 => value,
                    _ => unreachable!("wide wave operand"),
                };
                ir.store(bits, pointer(layout.base + i)).set_alignment(4);
            }
        }
        let ty = ir.void().function(&[ir.ptr(), ir.i64(), ir.ptr()]);
        let function = ir.function("amdgpu_sim_fiber_yield_values", ty);
        let context = self.func.param(6);
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
                    Ty::I1 => ir.trunc(bits, bool_ty),
                    Ty::F32 => ir.bitcast(bits, f32_ty),
                    Ty::I32 => bits,
                    _ => unreachable!("wide wave result"),
                };
                self.define(id, result);
            }
        }
    }
}
