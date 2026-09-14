//! BVH64 target lowering. Inputs and outputs are values; no ISA decoding or
//! architectural register access occurs here. Preserve the existing native paths.
//! Ray operands are explicit bit-pattern values. Form their floating-point
//! views where each native path consumes them, preserving the original order.
use super::*;
use crate::rdna_spmd::native::{BasicBlock, Builder, Type};

#[derive(Clone, Copy)]
pub(in crate::rdna_spmd) struct Storage {
    pub scratch: Value,
    pub packet: Option<(Value, Type)>,
}

struct Ray {
    addr: Value,
    extent: Value,
    origin: [Value; 3],
    inv: [Value; 3],
}

struct Native {
    ir: Builder,
    w: u32,
    vi32: Type,
    vi64: Type,
    vf32: Type,
    bvh_packet: Option<(Value, Type)>,
}

pub(in crate::rdna_spmd) fn lowering_state(e: &Emitter, sink: Value) -> Box<dyn std::any::Any> {
    let ir = e.ir;
    let packet = e.width().map(|_| {
        let lanes = crate::rdna_translator::bvh::BVH_RAY_PACKET_LANES as u64;
        let (packet_i64, packet_f32, packet_i32) = (ir.i64().array(lanes), ir.f32().array(lanes), ir.i32().array(lanes));
        let mut fields = [packet_i64; 15];
        fields[1..11].fill(packet_f32);
        fields[11..15].fill(packet_i32);
        let packet_ty = ir.structure(&fields);
        let packet = ir.alloca(packet_ty, "").set_alignment(64);
        (packet, packet_ty)
    });
    Box::new(Storage { scratch: sink, packet })
}

fn native(e: &Emitter, bvh_packet: Option<(Value, Type)>) -> Native {
    Native {
        ir: e.ir,
        w: e.width().unwrap_or(1),
        vi32: e.ty(Ty::I32),
        vi64: e.ty(Ty::I64),
        vf32: e.ty(Ty::F32),
        bvh_packet,
    }
}

pub(super) fn lower(e: &Emitter, a: &[Value]) -> Vec<Value> {
    let storage = *e.state::<Storage>().expect("BVH lowering requires native scratch storage");
    let ir = e.ir;
    let cg = native(e, storage.packet);
    if e.width().is_some() {
        let resource = [ir.extract_at(a[0], 0), ir.extract_at(a[1], 0)];
        let mask = ir.zext(ir.bitcast(a[13], ir.int(cg.w)), ir.i32());
        return cg.packet(a, resource, mask);
    }
    let scratch_ptr = |k: u32| ir.gep(ir.i32(), storage.scratch, &[ir.ci32(k)]);
    let (ptr, i32t, i64t, f32t) = (ir.ptr(), ir.i32(), ir.i64(), ir.f32());
    let params = [ptr, ptr, ptr, ptr, i32t, i32t, i64t, f32t, f32t, f32t, f32t, f32t, f32t, f32t, f32t, f32t, f32t];
    let args = [
        scratch_ptr(0), scratch_ptr(1), scratch_ptr(2), scratch_ptr(3), a[0], a[1], a[2],
        cg.vf32_of(a[3]), cg.vf32_of(a[4]), cg.vf32_of(a[5]), cg.vf32_of(a[6]), cg.vf32_of(a[7]),
        cg.vf32_of(a[8]), cg.vf32_of(a[9]), cg.vf32_of(a[10]), cg.vf32_of(a[11]), cg.vf32_of(a[12]),
    ];
    let func = ir.current_function();
    let call_bb = ir.append_block(func, "bvh.lane");
    let join = ir.append_block(func, "bvh.join");
    ir.cond_br(a[13], call_bb, join);
    ir.position_at_end(call_bb);
    ir.call_named("image_bvh64_intersect_ray", ir.void(), &params, &args);
    ir.br(join);
    ir.position_at_end(join);
    (0..4).map(|k| ir.load(i32t, scratch_ptr(k))).collect()
}

pub(super) fn lower8(e: &Emitter, a: &[Value]) -> Vec<Value> {
    let ir = e.ir;
    let cg = native(e, None);
    let packed = e.width().is_some();
    let lanes = e.width().unwrap_or(1);
    const RESULTS: u32 = 10;
    let (ptr, i32t, i64t, f32t) = (ir.ptr(), ir.i32(), ir.i64(), ir.f32());
    let slots = cg.entry_alloca(i32t.array((RESULTS * lanes) as u64));
    let lane = |v: Value, l: u32| if packed { ir.extract_at(v, l) } else { v };
    let params = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32t, i32t, i64t, f32t, i32t, f32t, f32t, f32t, f32t, f32t, f32t, i32t];
    let func = ir.current_function();
    for l in 0..lanes {
        let slot = |k: u32| ir.gep(i32t, slots, &[ir.ci32(l * RESULTS + k)]);
        let ptrs: Vec<Value> = (0..RESULTS).map(slot).collect();
        let call_bb = ir.append_block(func, "bvh8.lane");
        let join = ir.append_block(func, "bvh8.join");
        ir.cond_br(lane(a[12], l), call_bb, join);
        ir.position_at_end(call_bb);
        let float = |v: Value| ir.bitcast(lane(v, l), f32t);
        let args = [
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], ptrs[8], ptrs[9],
            lane(a[0], l), lane(a[1], l), lane(a[2], l), float(a[3]), lane(a[4], l),
            float(a[5]), float(a[6]), float(a[7]), float(a[8]), float(a[9]), float(a[10]), lane(a[11], l),
        ];
        ir.call_named("image_bvh8_intersect_ray", ir.void(), &params, &args);
        ir.br(join);
        ir.position_at_end(join);
    }
    (0..RESULTS).map(|k| {
        if !packed {
            return ir.load(i32t, ir.gep(i32t, slots, &[ir.ci32(k)]));
        }
        let mut out = cg.vi32.poison();
        for l in 0..lanes {
            let value = ir.load(i32t, ir.gep(i32t, slots, &[ir.ci32(l * RESULTS + k)]));
            out = ir.insert_at(out, value, l);
        }
        out
    }).collect()
}

impl Native {
    fn entry_alloca(&self, ty: Type) -> Value {
        let entry = self.ir.current_function().entry_block();
        let builder = self.ir.detached();
        match entry.first_instruction() {
            None => builder.position_at_end(entry),
            Some(first) => builder.position_before(first),
        }
        let value = builder.alloca(ty, "").set_alignment(64);
        builder.dispose();
        value
    }
    fn ci32(&self, v: u32) -> Value { self.ir.ci32(v) }
    fn ci64(&self, v: u64) -> Value { self.ir.ci64(v) }
    fn splat(&self, v: Value, vty: Type) -> Value { self.ir.splat(v, vty.vector_size()) }
    fn vci32(&self, v: u32) -> Value { self.splat(self.ci32(v), self.vi32) }
    fn zext64s(&self, v: Value) -> Value { self.ir.zext(v, self.ir.i64()) }
    fn vf32_bits(&self, v: Value) -> Value { self.ir.bitcast(v, self.vi32) }
    fn vf32_of(&self, v: Value) -> Value { self.ir.bitcast(v, self.vf32) }
    fn vminnum_raw(&self, a: Value, b: Value) -> Value {
        let c = self.ir.fcmp(FloatPred::Olt, a, b);
        self.ir.select(c, a, b)
    }
    fn vmaxnum_raw(&self, a: Value, b: Value) -> Value {
        let c = self.ir.fcmp(FloatPred::Ogt, a, b);
        self.ir.select(c, a, b)
    }
    fn vfadd(&self, a: Value, b: Value) -> Value { self.ir.fadd(a, b) }
    fn vfmul(&self, a: Value, b: Value) -> Value { self.ir.fmul(a, b) }
    fn vfsub(&self, a: Value, b: Value) -> Value { self.ir.fsub(a, b) }
    fn reduce(&self, operation: &str, ret: Type, operand: Value) -> Value {
        self.ir.call_named(&format!("llvm.vector.reduce.{operation}"), ret, &[operand.ty()], &[operand])
    }
    fn node_field(&self, node: Value, off: u64, ty: Type) -> Value {
        let a = self.ir.add(node, self.ci64(off));
        let p = self.ir.inttoptr(a, self.ir.ptr());
        self.ir.load(ty, p).set_alignment(4)
    }
    fn node_address(&self, bvh_base: Value, rep: Value) -> Value {
        let offset = self.ir.shl(self.ir.and(rep, self.ci64(!0x7u64)), self.ci64(3));
        self.ir.add(bvh_base, offset)
    }

    fn packet(&self, a: &[Value], resource: [Value; 2], mask: Value) -> Vec<Value> {
        let ir = self.ir;
        let func = ir.current_function();
        let uni_bb = ir.append_block(func, "bvh.uniform");
        let fast = ir.append_block(func, "bvh.box");
        let tri_bb = ir.append_block(func, "bvh.tri");
        let slow = ir.append_block(func, "bvh.general");
        let join = ir.append_block(func, "bvh.join");

        // The resource names the base of the BVH in 256-byte units,
        // and says whether the children that point at triangle nodes
        // sort before the ones that point at boxes.
        let base_hi = ir.shl(self.zext64s(ir.and(resource[1], self.ci32(0xFF))), self.ci64(32));
        let bvh_base = ir.shl(ir.or(self.zext64s(resource[0]), base_hi), self.ci64(8));
        let resource_bit = |shift: u32| {
            let bit = ir.and(ir.lshr(resource[1], self.ci32(shift)), self.ci32(1));
            ir.icmp(IntPred::Ne, bit, self.ci32(0))
        };
        let sorts_boxes = resource_bit(31);
        let sorts_triangles_first = resource_bit(20);
        let ray = Ray {
            addr: a[2],
            extent: self.vf32_of(a[3]),
            origin: [self.vf32_of(a[4]), self.vf32_of(a[5]), self.vf32_of(a[6])],
            inv: [self.vf32_of(a[10]), self.vf32_of(a[11]), self.vf32_of(a[12])],
        };

        // Representative node address over the active lanes, the same
        // umax idiom `emit_vglobal_cluster` uses (the block only runs
        // with EXEC != 0, so at least one lane contributes).
        let exec = a[13];
        let masked = ir.select(exec, ray.addr, self.vi64.null());
        let rep = self.reduce(&format!("umax.v{}i64", self.w), ir.i64(), masked);
        let same = ir.icmp(IntPred::Eq, ray.addr, self.splat(rep, self.vi64));
        let same_or_off = ir.or(same, ir.not(exec));
        let uniform = self.reduce(&format!("and.v{}i1", self.w), ir.i1(), same_or_off);
        let ntype = ir.and(rep, self.ci64(7));
        let is_box = ir.icmp(IntPred::Eq, ntype, self.ci64(5));
        let is_tri = ir.icmp(IntPred::Ult, ntype, self.ci64(2));
        let known = ir.or(is_box, is_tri);
        // With EXEC == 0 the reduction above has no active lane to pick,
        // so `rep` would be 0 and the type test would accept it as a
        // triangle node at address 0. Blocks are not supposed to run
        // with EXEC == 0, but the fast path must not dereference a null
        // node if one ever does.
        let any_active = ir.icmp(IntPred::Ne, mask, self.ci32(0));
        // The inline path sorts the children by the time the ray
        // reaches them and nothing else, which is what a resource that
        // sorts its boxes and leaves triangle nodes where they are
        // asks for; the helper knows the rest of Table 65.
        let plainly_sorted = ir.and(sorts_boxes, ir.not(sorts_triangles_first));
        let take = ir.and(ir.and(ir.and(uniform, known), any_active), plainly_sorted);
        ir.cond_br(take, uni_bb, slow);
        ir.position_at_end(uni_bb);
        ir.cond_br(is_box, fast, tri_bb);

        ir.position_at_end(fast);
        let (child, fast_end) = self.box_path(bvh_base, rep, &ray, slow, join);
        ir.position_at_end(tri_bb);
        let (tri_bits, tri_end) = self.triangle_path(a, bvh_base, rep, &ray, join);
        ir.position_at_end(slow);
        let (slow_res, slow_end) = self.general_path(a, resource, mask, &ray, join);

        ir.position_at_end(join);
        // All phis must sit at the top of the block, so build them
        // before any of the register writes.
        (0..4).map(|k| {
            let phi = ir.phi(self.vi32);
            phi.add_incoming(&[(child[k], fast_end), (tri_bits[k], tri_end), (slow_res[k], slow_end)]);
            phi
        }).collect()
    }

    fn box_path(&self, bvh_base: Value, rep: Value, ray: &Ray, slow: BasicBlock, join: BasicBlock) -> ([Value; 4], BasicBlock) {
        let ir = self.ir;
        let node_ptr = self.node_address(bvh_base, rep);
        // Box4Node: child_index[4], then aabb[4] of { min[3], max[3] }.
        let field = |off: u64, ty: Type| self.node_field(node_ptr, off, ty);
        let vzero = self.vf32.null();
        let mut child = [self.vi32.null(); 4];
        let mut dist = [vzero; 4];
        let mut nan_acc: Option<Value> = None;
        for c in 0..4u64 {
            let mut hi3 = [vzero; 3];
            let mut lo3 = [vzero; 3];
            for axis in 0..3u64 {
                let base = 16 + c * 24 + axis * 4;
                let bhi = self.splat(field(base + 12, ir.f32()), self.vf32);
                let blo = self.splat(field(base, ir.f32()), self.vf32);
                let f = self.vfmul(self.vfsub(bhi, ray.origin[axis as usize]), ray.inv[axis as usize]);
                let g = self.vfmul(self.vfsub(blo, ray.origin[axis as usize]), ray.inv[axis as usize]);
                for v in [f, g] {
                    let u = ir.fcmp(FloatPred::Uno, v, v);
                    nan_acc = Some(match nan_acc {
                        None => u,
                        Some(p) => ir.or(p, u),
                    });
                }
                hi3[axis as usize] = self.vmaxnum_raw(f, g);
                lo3[axis as usize] = self.vminnum_raw(f, g);
            }
            let t1 = self.vminnum_raw(hi3[0], self.vminnum_raw(hi3[1], self.vminnum_raw(hi3[2], ray.extent)));
            let t0 = self.vmaxnum_raw(lo3[0], self.vmaxnum_raw(lo3[1], self.vmaxnum_raw(lo3[2], vzero)));
            let hit = ir.fcmp(FloatPred::Ole, t0, t1);
            let ci = self.splat(field(c * 4, ir.i32()), self.vi32);
            child[c as usize] = ir.select(hit, ci, self.vci32(0xFFFF_FFFF));
            dist[c as usize] = t0;
        }
        let ones = self.vci32(0xFFFF_FFFF);
        let swap_pair = |a: usize, b: usize, child: &mut [Value; 4], dist: &mut [Value; 4]| {
            let b_valid = ir.icmp(IntPred::Ne, child[b], ones);
            let closer = ir.fcmp(FloatPred::Olt, dist[b], dist[a]);
            let a_empty = ir.icmp(IntPred::Eq, child[a], ones);
            let sw = ir.or(ir.and(b_valid, closer), a_empty);
            let ca = ir.select(sw, child[b], child[a]);
            let cb = ir.select(sw, child[a], child[b]);
            let da = ir.select(sw, dist[b], dist[a]);
            let db = ir.select(sw, dist[a], dist[b]);
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
        let any_nan = self.reduce(&format!("or.v{}i1", self.w), ir.i1(), nan_acc.unwrap());
        ir.cond_br(any_nan, slow, join);
        (child, ir.insert_block())
    }

    fn triangle_path(&self, a: &[Value], bvh_base: Value, rep: Value, ray: &Ray, join: BasicBlock) -> ([Value; 4], BasicBlock) {
        let ir = self.ir;
        let tnode = self.node_address(bvh_base, rep);
        let tfield = |off: u64, ty: Type| self.node_field(tnode, off, ty);
        // TrianglePairNode: v0,v1,v2,v3 (3 f32 each), pad, prim_index[2], flags.
        let odd = ir.icmp(IntPred::Ne, ir.and(rep, self.ci64(1)), self.ci64(0));
        let vtx = |slot: u64, axis: u64| tfield(slot * 12 + axis * 4, ir.f32());
        // tri = odd ? [v1, v3, v2] : [v0, v1, v2]
        let pick = |a: u64, b: u64, axis: u64| ir.select(odd, vtx(a, axis), vtx(b, axis));
        let t0v: Vec<Value> = (0..3).map(|k| self.splat(pick(1, 0, k), self.vf32)).collect();
        let t1v: Vec<Value> = (0..3).map(|k| self.splat(pick(3, 1, k), self.vf32)).collect();
        let t2v: Vec<Value> = (0..3).map(|k| self.splat(pick(2, 2, k), self.vf32)).collect();
        let flags_raw = tfield(60, ir.i32());
        let flags = ir.lshr(flags_raw, ir.select(odd, self.ci32(8), self.ci32(0)));

        let dir: Vec<Value> = (0..3).map(|k| self.vf32_of(a[7 + k])).collect();
        let sub3 = |a: &[Value], b: &[Value]| -> Vec<Value> { (0..3).map(|k| self.vfsub(a[k], b[k])).collect() };
        // Same association as `intersect_triangle_frac`: cross uses
        // a1*b2 - a2*b1, dot is (x + y) + z. No contraction.
        let cross = |a: &[Value], b: &[Value]| -> Vec<Value> {
            vec![
                self.vfsub(self.vfmul(a[1], b[2]), self.vfmul(a[2], b[1])),
                self.vfsub(self.vfmul(a[2], b[0]), self.vfmul(a[0], b[2])),
                self.vfsub(self.vfmul(a[0], b[1]), self.vfmul(a[1], b[0])),
            ]
        };
        let dot = |a: &[Value], b: &[Value]| -> Value {
            self.vfadd(self.vfadd(self.vfmul(a[0], b[0]), self.vfmul(a[1], b[1])), self.vfmul(a[2], b[2]))
        };

        let e1 = sub3(&t1v, &t0v);
        let e2 = sub3(&t2v, &t0v);
        let s1 = cross(&dir, &e2);
        let denom = dot(&s1, &e1);
        let dv = sub3(&ray.origin, &t0v);
        let b_y = dot(&dv, &s1);
        let s2 = cross(&dv, &e1);
        let b_z = dot(&dir, &s2);
        let t_hit = dot(&e2, &s2);
        let b_x = self.vfsub(self.vfsub(denom, b_y), b_z);

        let zero = self.vf32.null();
        let fc = |p, a, b| ir.fcmp(p, a, b);
        let or = |a, b| ir.or(a, b);
        let and = |a, b| ir.and(a, b);
        let byz = self.vfadd(b_y, b_z);
        let reject_pos = and(
            fc(FloatPred::Ogt, denom, zero),
            or(
                or(
                    or(fc(FloatPred::Olt, b_y, zero), fc(FloatPred::Ogt, b_y, denom)),
                    or(fc(FloatPred::Olt, b_z, zero), fc(FloatPred::Ogt, byz, denom)),
                ),
                fc(FloatPred::Olt, t_hit, zero),
            ),
        );
        let reject_neg = and(
            fc(FloatPred::Olt, denom, zero),
            or(
                or(
                    or(fc(FloatPred::Ogt, b_y, zero), fc(FloatPred::Olt, b_y, denom)),
                    or(fc(FloatPred::Ogt, b_z, zero), fc(FloatPred::Olt, byz, denom)),
                ),
                fc(FloatPred::Ogt, t_hit, zero),
            ),
        );
        // A ray in the plane of the triangle meets nothing either.
        let miss = or(or(reject_pos, reject_neg), fc(FloatPred::Oeq, denom, zero));
        let sel = |c, a, b| ir.select(c, a, b);
        // The numerator a miss returns is infinite, signed with the
        // denominator so that the quotient is positive infinity
        // whichever way round the triangle faces.
        let missed = self.vf32_of(ir.or(ir.and(self.vf32_bits(denom), self.vci32(0x8000_0000)), self.vci32(0x7F80_0000)));
        // `flags` picks a barycentric per output; it is uniform, so the
        // index is a scalar and the selects are scalar-controlled.
        let bary = |shift: u32| -> Value {
            let idx = ir.and(ir.lshr(flags, self.ci32(shift)), self.ci32(3));
            let is1 = ir.icmp(IntPred::Eq, idx, self.ci32(1));
            let is2 = ir.icmp(IntPred::Eq, idx, self.ci32(2));
            // The fourth encoding is reserved, and the part answers it
            // with the first barycentric.
            sel(is1, b_y, sel(is2, b_z, b_x))
        };
        let tri_res = [sel(miss, missed, t_hit), denom, bary(0), bary(2)];
        let tri_bits = [self.vf32_bits(tri_res[0]), self.vf32_bits(tri_res[1]), self.vf32_bits(tri_res[2]), self.vf32_bits(tri_res[3])];
        ir.br(join);
        (tri_bits, ir.insert_block())
    }

    fn general_path(&self, a: &[Value], resource: [Value; 2], mask: Value, ray: &Ray, join: BasicBlock) -> ([Value; 4], BasicBlock) {
        let ir = self.ir;
        let (bvh_packet, bvh_packet_ty) = self.bvh_packet.expect("packet lowering requires the ray packet frame");
        let field_ptr = |f: u32| ir.struct_gep(bvh_packet_ty, bvh_packet, f);
        let inputs = [
            ray.addr, ray.extent, ray.origin[0], ray.origin[1], ray.origin[2],
            self.vf32_of(a[7]), self.vf32_of(a[8]), self.vf32_of(a[9]),
            ray.inv[0], ray.inv[1], ray.inv[2],
        ];
        let step = self.w.min(crate::rdna_translator::bvh::BVH_RAY_PACKET_LANES as u32);
        let slice = |v: Value, off: u32, len: u32| -> Value {
            if off == 0 && len == self.w { return v; }
            let idx: Vec<u32> = (0..len).map(|k| off + k).collect();
            ir.shuffle_by(v, v.ty().poison(), &idx)
        };
        let mut chunks: Vec<Vec<Value>> = Vec::new();
        let mut off = 0;
        while off < self.w {
            let len = step.min(self.w - off);
            for (f, value) in inputs.iter().copied().enumerate() {
                ir.store(slice(value, off, len), field_ptr(f as u32)).set_alignment(if f == 0 { 8 } else { 4 });
            }
            ir.call_named(
                "image_bvh64_intersect_ray_packet",
                ir.void(),
                &[ir.ptr(), ir.i32(), ir.i32(), ir.i32(), ir.i32()],
                &[bvh_packet, self.ci32(len), ir.lshr(mask, self.ci32(off)), resource[0], resource[1]],
            );
            chunks.push((0..4).map(|k| ir.load(ir.i32().vector(len), field_ptr(11 + k)).set_alignment(4)).collect());
            off += len;
        }
        let joined = |k: usize| {
            let mut value = chunks[0][k];
            let mut have = value.ty().vector_size();
            for chunk in &chunks[1..] {
                let next = chunk[k];
                let more = next.ty().vector_size();
                let wide = have.max(more);
                let pad = |v: Value, n_have: u32| if n_have == wide { v } else {
                    let idx: Vec<u32> = (0..wide).map(|j| j.min(n_have - 1)).collect();
                    ir.shuffle_by(v, v.ty().poison(), &idx)
                };
                let (a, b) = (pad(value, have), pad(next, more));
                let idx: Vec<u32> = (0..have).chain(wide..wide + more).collect();
                value = ir.shuffle_by(a, b, &idx);
                have += more;
            }
            value
        };
        let slow_res = [joined(0), joined(1), joined(2), joined(3)];
        ir.br(join);
        (slow_res, ir.insert_block())
    }
}
