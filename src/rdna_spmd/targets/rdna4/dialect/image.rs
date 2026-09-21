use super::*;

fn call(e: &Emitter, name: &str, result: Type, args: &[Value]) -> Value {
    let types = args.iter().map(|v| v.ty()).collect::<Vec<_>>();
    e.ir.call_named(name, result, &types, args)
}

fn require(e: &Emitter, mut valid: Value) {
    let ir = e.ir;
    if let Some(w) = e.width() {
        valid = call(
            e,
            &format!("llvm.vector.reduce.and.v{w}i1"),
            ir.i1(),
            &[valid],
        );
    }
    let f = ir.current_function();
    let next = ir.append_block(f, "image.valid");
    let bad = ir.append_block(f, "image.unsupported");
    ir.cond_br(valid, next, bad);
    ir.position_at_end(bad);
    call(e, "llvm.trap", ir.void(), &[]);
    ir.unreachable();
    ir.position_at_end(next);
}

pub(super) fn sample(e: &Emitter, a: &[Value]) -> Value {
    let ir = e.ir;
    let k = |v| e.constant(Ty::I32, v);
    let eq = |a, c| ir.icmp(IntPred::Eq, a, c);
    let and = |a, c| ir.and(a, c);
    let or = |a, c| ir.or(a, c);
    let select = |p, a, c| ir.select(p, a, c);
    let shr = |a, bits| ir.lshr(a, k(bits));
    let field = |words: &[Value], start: usize, size: u32| {
        let low = shr(words[start / 32], (start % 32) as u64);
        let value = if start % 32 + size as usize > 32 {
            or(
                low,
                ir.shl(words[start / 32 + 1], k((32 - start % 32) as u64)),
            )
        } else {
            low
        };
        and(value, k((1u64 << size) - 1))
    };
    let format = field(&a[..8], 49, 8);
    let width = ir.add(field(&a[..8], 62, 16), k(1));
    let height = ir.add(field(&a[..8], 78, 16), k(1));
    let pitch = field(&a[..8], 128, 16);
    let row = select(eq(pitch, k(0)), width, ir.add(pitch, k(1)));
    let row = and(ir.add(row, k(127)), k(0xffff_ff80));
    let component_shift = ir.mul(a[12], k(3));
    let component = and(ir.lshr(a[3], component_shift), k(7));
    let data_component = ir.icmp(IntPred::Uge, component, k(4));
    let filter = field(&a[8..12], 84, 2);
    let selector_valid = or(data_component, ir.icmp(IntPred::Ule, component, k(1)));
    require(e, and(eq(filter, k(0)), selector_valid));

    let unrm = or(a[13], eq(field(&a[8..12], 15, 1), k(1)));
    let axis = |value, size, shift| {
        let extent = ir.uitofp(size, e.ty(Ty::F32));
        let coordinate = select(unrm, value, ir.fmul(value, extent));
        let integral = e.call(
            &format!("llvm.floor.{}", e.suffix(Ty::F32)),
            Ty::F32,
            &[coordinate],
        );
        let coordinate = e.call(
            &format!(
                "llvm.fptosi.sat.{}.{}",
                e.suffix(Ty::I32),
                e.suffix(Ty::F32)
            ),
            Ty::I32,
            &[integral],
        );
        let encoded = and(shr(a[8], shift), k(7));
        let repeat = ir.icmp(IntPred::Ult, encoded, k(2));
        let mode = select(and(unrm, repeat), ir.add(encoded, k(2)), encoded);
        let mirror = eq(and(mode, k(1)), k(1));
        let negative = ir.icmp(IntPred::Slt, coordinate, k(0));
        let reflected = select(
            and(mirror, negative),
            ir.xor(coordinate, k(0xffff_ffff)),
            coordinate,
        );
        let last = ir.sub(size, k(1));
        let clamped = select(ir.icmp(IntPred::Slt, reflected, k(0)), k(0), reflected);
        let clamped = select(ir.icmp(IntPred::Sgt, clamped, last), last, clamped);
        let period = select(mirror, ir.shl(size, k(1)), size);
        let rem = ir.srem(coordinate, period);
        let rem = select(ir.icmp(IntPred::Slt, rem, k(0)), ir.add(rem, period), rem);
        let folded = ir.sub(ir.sub(period, k(1)), rem);
        let repeated = select(ir.icmp(IntPred::Sge, rem, size), folded, rem);
        let repeat = ir.icmp(IntPred::Ult, mode, k(2));
        let coord = select(repeat, repeated, clamped);
        let border_mode = ir.icmp(IntPred::Uge, mode, k(6));
        let in_range = ir.icmp(IntPred::Ult, reflected, size);
        (coord, or(ir.not(border_mode), in_range))
    };
    let (x, x_valid) = axis(a[14], width, 0);
    let (y, y_valid) = axis(a[15], height, 3);
    let inside = and(x_valid, y_valid);
    let load = and(inside, data_component);
    let supported_format = or(
        or(eq(format, k(1)), eq(format, k(2))),
        or(eq(format, k(5)), eq(format, k(6))),
    );
    require(e, or(ir.not(load), supported_format));
    let border = field(&a[8..12], 126, 2);
    let border_read = and(data_component, ir.not(inside));
    require(
        e,
        or(ir.not(border_read), ir.icmp(IntPred::Ult, border, k(3))),
    );

    let wide = |v| ir.zext(v, e.ty(Ty::I64));
    let base = or(
        ir.shl(wide(a[0]), e.constant(Ty::I64, 8)),
        ir.shl(wide(and(a[1], k(255))), e.constant(Ty::I64, 40)),
    );
    let offset = ir.add(ir.mul(wide(y), wide(row)), wide(x));
    let address = ir.add(base, offset);

    let w = e.width().unwrap_or(1);
    let byte_vector = ir.i8().vector(w);
    let pointers = ir.ptr().vector(w);
    let (addresses, mask) = if e.width().is_some() {
        (address, load)
    } else {
        (
            ir.insert_at(ir.i64().vector(1).undef(), address, 0),
            ir.insert_at(ir.i1().vector(1).undef(), load, 0),
        )
    };
    let pointers = ir.inttoptr(addresses, pointers);
    let raw = call(
        e,
        &format!("llvm.masked.gather.v{w}i8.v{w}p0"),
        byte_vector,
        &[pointers, mask, byte_vector.null()],
    );
    ir.set_call_align(raw, 1, 1);
    let raw = if e.width().is_some() {
        raw
    } else {
        ir.extract_at(raw, 0)
    };
    let unsigned = ir.zext(raw, e.ty(Ty::I32));
    let signed = ir.sext(raw, e.ty(Ty::I32));
    let normalized_signed = select(
        ir.icmp(IntPred::Slt, signed, k((-127i32) as u32 as u64)),
        k((-127i32) as u32 as u64),
        signed,
    );
    let unorm = ir.fdiv(
        ir.uitofp(unsigned, e.ty(Ty::F32)),
        e.constant(Ty::F32, 255f32.to_bits() as u64),
    );
    let snorm = ir.fdiv(
        ir.sitofp(normalized_signed, e.ty(Ty::F32)),
        e.constant(Ty::F32, 127f32.to_bits() as u64),
    );
    let mut value = select(eq(format, k(6)), signed, unsigned);
    value = select(eq(format, k(1)), ir.bitcast(unorm, e.ty(Ty::I32)), value);
    value = select(eq(format, k(2)), ir.bitcast(snorm, e.ty(Ty::I32)), value);
    let one = select(
        or(eq(format, k(1)), eq(format, k(2))),
        k(1f32.to_bits() as u64),
        k(1),
    );
    let border_value = select(eq(border, k(2)), one, k(0));
    value = select(inside, value, border_value);
    value = select(eq(component, k(1)), one, value);
    select(eq(component, k(0)), k(0), value)
}
