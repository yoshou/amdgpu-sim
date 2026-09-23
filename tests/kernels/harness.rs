use amdgpu_sim::rdna_spmd::{self as spmd, Buffer, Function, Launch, Module, Pod};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

pub(crate) const WIDTHS: [u32; 6] = [1, 2, 4, 8, 16, 32];

pub(crate) struct Kernels {
    module: Module,
    functions: RefCell<Vec<(String, Rc<Function>)>>,
}

pub(crate) enum Arg<'a> {
    In(&'a [u8]),
    Out(*mut u8, usize, PhantomData<&'a mut [u8]>),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl<'a> Arg<'a> {
    pub(crate) fn input<T: Pod>(values: &'a [T]) -> Self {
        Arg::In(unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        })
    }

    pub(crate) fn output<T: Pod>(values: &'a mut [T]) -> Self {
        Arg::Out(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values), PhantomData)
    }
}

pub(crate) struct Run<'a> {
    pub(crate) kernel: &'a str,
    pub(crate) wg: [u32; 3],
    pub(crate) grid: [u32; 3],
    pub(crate) args: &'a [Arg<'a>],
}

impl Kernels {
    pub(crate) fn load() -> Self {
        Self::open("kernels_gfx1200.o", "tests/kernels/build.sh")
    }

    pub(crate) fn open(object: &str, builder: &str) -> Self {
        let path = format!("{}/tests/data/{}", env!("CARGO_MANIFEST_DIR"), object);
        let module = Module::open(&path).unwrap_or_else(|e| panic!("{} (run {})", e, builder));
        Kernels {
            module,
            functions: RefCell::new(Vec::new()),
        }
    }

    fn function(&self, kernel: &str) -> Rc<Function> {
        let mut functions = self.functions.borrow_mut();
        if let Some((_, f)) = functions.iter().find(|(name, _)| name == kernel) {
            return f.clone();
        }
        let f = Rc::new(
            self.module
                .function(kernel)
                .unwrap_or_else(|e| panic!("{}: {}", kernel, e)),
        );
        functions.push((kernel.to_string(), f.clone()));
        f
    }

    pub(crate) fn run(&self, spec: &Run, width: u32) {
        self.run_threaded(spec, width, 1)
    }

    pub(crate) fn run_threaded(&self, spec: &Run, width: u32, threads: usize) {
        let function = self.function(spec.kernel);
        let mut buffers: Vec<Option<Buffer>> = spec
            .args
            .iter()
            .map(|arg| match *arg {
                Arg::In(bytes) => Some(Buffer::from_slice(bytes)),
                Arg::Out(at, len, _) => {
                    Some(Buffer::from_slice(unsafe { std::slice::from_raw_parts(at, len) }))
                }
                _ => None,
            })
            .collect();
        let bindings: Vec<spmd::Arg> = spec
            .args
            .iter()
            .zip(buffers.iter_mut())
            .map(|(arg, buffer)| match *arg {
                Arg::In(_) => spmd::Arg::read(buffer.as_ref().unwrap()),
                Arg::Out(..) => spmd::Arg::write(buffer.as_mut().unwrap()),
                Arg::U32(v) => spmd::Arg::value(v),
                Arg::I32(v) => spmd::Arg::value(v),
                Arg::F32(v) => spmd::Arg::value(v),
            })
            .collect();
        let launch = Launch::new(spec.grid, spec.wg).width(width).threads(threads);
        function
            .launch(&launch, &bindings)
            .unwrap_or_else(|e| panic!("{}: {}", spec.kernel, e));
        drop(bindings);
        for (arg, buffer) in spec.args.iter().zip(&buffers) {
            if let Arg::Out(at, len, _) = *arg {
                let out = unsafe { std::slice::from_raw_parts_mut(at, len) };
                out.copy_from_slice(buffer.as_ref().unwrap().as_slice::<u8>());
            }
        }
    }
}

pub(crate) fn same<T: PartialEq + std::fmt::Debug>(
    kernel: &str,
    width: u32,
    against: &str,
    got: &[T],
    want: &[T],
) {
    if got == want {
        return;
    }
    let (at, (g, w)) = got
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (g, w))| g != w)
        .unwrap_or_else(|| panic!("{}: W={} produced {} values, {} has {}", kernel, width, got.len(), against, want.len()));
    let differing = got.iter().zip(want).filter(|(g, w)| g != w).count();
    panic!(
        "{}: W={} differs from {} at {} of {} places; \
         first at index {}: got {:?}, {} gave {:?}",
        kernel, width, against, differing, got.len(), at, g, against, w
    );
}

pub(crate) fn close(kernel: &str, width: u32, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (at, (&g, &w)) in got.iter().zip(want).enumerate() {
        let slack = tol * w.abs().max(1.0);
        if (g - w).abs() > slack {
            panic!(
                "{}: W={} differs from the reference at index {}: got {}, expected {} (tolerance {})",
                kernel, width, at, g, w, slack
            );
        }
    }
}

pub(crate) fn close64(kernel: &str, width: u32, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (at, (&g, &w)) in got.iter().zip(want).enumerate() {
        let slack = tol * w.abs().max(1.0);
        if (g - w).abs() > slack {
            panic!(
                "{}: W={} differs from the reference at index {}: got {}, expected {} (tolerance {})",
                kernel, width, at, g, w, slack
            );
        }
    }
}

pub(crate) fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

pub(crate) fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    sign * match e {
        0 => m * 2f32.powi(-24),
        0x1f if m == 0.0 => f32::INFINITY,
        0x1f => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

pub(crate) fn round_f16(v: f64) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v as f32;
    }
    let e = ((v.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let q = (e - 10).max(-24);
    let scale = 2f64.powi(q);
    let r = (v / scale).round_ties_even() * scale;
    if r.abs() >= 65520.0 {
        return if r > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY };
    }
    r as f32
}

pub(crate) fn halves(values: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = values.iter().map(|&v| f16_bits(v)).collect();
    let exact = bits.iter().map(|&b| f16_value(b)).collect();
    (bits, exact)
}
