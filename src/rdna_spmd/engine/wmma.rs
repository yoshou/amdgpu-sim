use std::sync::OnceLock;

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
        .get_or_init(|| super::super::codegen::compile_wmma(width as u32))
        .address();
    unsafe { std::mem::transmute::<u64, ApplyFn>(addr) }
}

pub fn warm_wmma(width: usize) {
    let _ = apply_fn(width);
}

pub unsafe fn apply_values(width: usize, packets: *const *mut u32) {
    apply_fn(width)(packets);
}
