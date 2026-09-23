use std::mem::{size_of, size_of_val};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod sealed {
    pub trait Sealed {}
}

pub trait Pod: Copy + 'static + sealed::Sealed {}

macro_rules! pod {
    ($($t:ty),*) => {
        $(
            impl sealed::Sealed for $t {}
            impl Pod for $t {}
        )*
    };
}
pod!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl<T: Pod, const N: usize> sealed::Sealed for [T; N] {}
impl<T: Pod, const N: usize> Pod for [T; N] {}

pub(crate) fn bytes_of<T: Pod>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, size_of_val(values)) }
}

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

pub struct Buffer {
    base: NonNull<u8>,
    len: usize,
    allocation: u64,
    exposed: AtomicBool,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(len: usize) -> Self {
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len.max(1),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(mapped != libc::MAP_FAILED, "cannot map a buffer of {} bytes", len);
        let base = NonNull::new(mapped as *mut u8).expect("a mapping at a nonzero address");
        Self {
            base,
            len,
            allocation: ALLOCATIONS.fetch_add(1, Ordering::Relaxed),
            exposed: AtomicBool::new(false),
        }
    }

    pub fn zeroed<T: Pod>(count: usize) -> Self {
        Self::new(count * size_of::<T>())
    }

    pub fn from_slice<T: Pod>(values: &[T]) -> Self {
        let mut buffer = Self::new(size_of_val(values));
        buffer.as_mut_slice::<u8>().copy_from_slice(bytes_of(values));
        buffer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice<T: Pod>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.base.as_ptr() as *const T, self.count::<T>()) }
    }

    pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr() as *mut T, self.count::<T>()) }
    }

    pub fn to_vec<T: Pod>(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    fn count<T: Pod>(&self) -> usize {
        let size = size_of::<T>();
        assert!(
            size > 0 && self.len.is_multiple_of(size),
            "a buffer of {} bytes does not hold values of {} bytes",
            self.len,
            size
        );
        self.len / size
    }

    pub fn address(&self) -> u64 {
        self.exposed.store(true, Ordering::Relaxed);
        self.base()
    }

    pub(crate) fn base(&self) -> u64 {
        self.base.as_ptr() as u64
    }

    pub(crate) fn allocation(&self) -> u64 {
        self.allocation
    }

    pub(crate) fn exposed(&self) -> bool {
        self.exposed.load(Ordering::Relaxed)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr() as *mut libc::c_void, self.len.max(1));
        }
    }
}
