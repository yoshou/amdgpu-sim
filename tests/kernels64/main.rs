#[path = "../kernels/cases.rs"]
mod cases;
#[path = "../kernels/harness.rs"]
mod harness;

pub(crate) const OBJECT: &str = "kernels_gfx1200_w64.o";
pub(crate) const LANES: usize = 64;
