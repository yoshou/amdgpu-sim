mod analysis;
mod codegen;
mod compiler;
mod decompile;
mod dialect;
mod engine;
mod environment;
mod gcn3;
mod hash;
mod host;
mod ir;
mod lockstep;
mod native;
mod pass;
mod program;
mod rdna4;
mod runtime;

pub(crate) const SHARED_APERTURE: u64 = 1 << 48;

pub use runtime::{Arg, Buffer, Error, Function, Launch, Module, Pod};
