use std::sync::OnceLock;

use super::super::native::Value;

type ApplyFn = unsafe extern "C" fn(*const *mut u32);

static APPLY: [OnceLock<super::super::native::jit::NativeCode>; 6] = [
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
];

fn apply_fn(width: usize) -> ApplyFn {
    assert!(
        matches!(width, 1 | 2 | 4 | 8 | 16 | 32),
        "unsupported packet width {}",
        width
    );
    let addr = APPLY[width.trailing_zeros() as usize]
        .get_or_init(|| compile(width as u32))
        .address();
    unsafe { std::mem::transmute::<u64, ApplyFn>(addr) }
}

pub(in crate::rdna_spmd) fn warm(width: usize) {
    let _ = apply_fn(width);
}

pub(in crate::rdna_spmd) unsafe fn apply_values(width: usize, packets: *const *mut u32) {
    apply_fn(width)(packets);
}

fn compile(w: u32) -> super::super::native::jit::NativeCode {

    let name = format!("wmma_apply_w{w}");
    let native = super::super::native::jit::Module::new(&name);
    let ir = native.builder();

    let (i32t, ptr) = (ir.i32(), ir.ptr());
    let packet_i32 = i32t.vector(w);
    let wave_i32 = i32t.vector(32);
    let wave_i16 = ir.i16().vector(32);
    let wave_f16 = ir.f16().vector(32);
    let wave_f32 = ir.f32().vector(32);

    let func = ir.add_function(&name, ir.void().function(&[ptr]));
    ir.position_at_end(ir.append_block(func, ""));

    let konst = |v: u32| ir.ci32(v);
    let packets: Vec<Value> = (0..32 / w)
        .map(|p| ir.load(ptr, ir.gep(ptr, func.param(0), &[konst(p)])))
        .collect();

    let reg_of = |n: u32, k: u32| konst(n + k);

    let load_reg = |reg: Value| {
        let offset = ir.mul(reg, konst(w));
        let mut parts: Vec<Value> = packets
            .iter()
            .map(|&p| {
                ir.load(packet_i32, ir.gep(i32t, p, &[offset]))
                    .set_alignment(4)
            })
            .collect();
        let mut size = w;
        while size < 32 {
            parts = parts
                .chunks(2)
                .map(|pair| {
                    let mask: Vec<u32> = (0..2 * size).collect();
                    ir.shuffle_by(pair[0], pair[1], &mask)
                })
                .collect();
            size *= 2;
        }
        parts[0]
    };

    let fragments = |n: u32| -> Vec<Value> {
        (0..8u32)
            .map(|e| {
                let mut bits = load_reg(reg_of(n, e / 2));
                if e % 2 == 1 {
                    let shift = ir.const_vector(&vec![konst(16); 32]);
                    bits = ir.lshr(bits, shift);
                }
                let half = ir.bitcast(ir.trunc(bits, wave_i16), wave_f16);
                ir.fpext(half, wave_f32)
            })
            .collect()
    };

    let a_frag = fragments(0);
    let b_frag = fragments(4);
    let mut acc: Vec<Value> = (0..8u32)
        .map(|m| ir.bitcast(load_reg(reg_of(8, m)), wave_f32))
        .collect();

    let pick = |v: Value, from: &dyn Fn(u32) -> u32| {
        let mask: Vec<u32> = (0..32).map(|lane| from(lane)).collect();
        ir.shuffle_by(v, wave_f32.poison(), &mask)
    };

    for k in 0..16u32 {

        let e = ((k % 4) + 4 * (k / 8)) as usize;
        let g = (k / 4) % 2;

        let b_k = pick(b_frag[e], &|lane| lane % 16 + 16 * g);
        for m in 0..8u32 {

            let a_mk = pick(a_frag[e], &|lane| m + 8 * (lane / 16) + 16 * g);
            let product = ir.fmul(a_mk, b_k);
            acc[m as usize] = ir.fadd(acc[m as usize], product);
        }
    }

    for m in 0..8u32 {
        let value = ir.bitcast(acc[m as usize], wave_i32);
        let offset = ir.mul(reg_of(8, m), konst(w));
        for (p, &packet) in packets.iter().enumerate() {
            let mask: Vec<u32> = (0..w).map(|lane| p as u32 * w + lane).collect();
            let lanes = ir.shuffle_by(value, wave_i32.poison(), &mask);
            ir.store(lanes, ir.gep(i32t, packet, &[offset]))
                .set_alignment(4);
        }
    }
    ir.ret_void();
    native
        .optimize(super::super::native::jit::Mode::Packet)
        .compile(&name)
}
