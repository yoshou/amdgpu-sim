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
mod rdna4;

pub use compiler::{compile, decode_program, CompileOptions};
pub use engine::{
    dispatch,
    dispatch::GridDims,
    kernel::{Kernel, Scheduler},
};
pub use program::Program;
