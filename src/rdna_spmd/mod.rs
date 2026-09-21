mod analysis;
mod codegen;
mod compiler;
mod decompile;
mod dialect;
mod engine;
mod host;
mod ir;
mod lockstep;
mod native;
mod pass;
mod program;
mod targets;

pub use compiler::{compile, decode_program, CompileOptions};
pub use engine::{
    dispatch,
    dispatch::GridDims,
    kernel::{Kernel, Scheduler},
};
pub use program::Program;

pub fn default_width() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return 16;
        }
    }
    0
}
