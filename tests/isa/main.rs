//! Unit tests for the RDNA4 instruction implementations.
//!
//! Every expected value in these tests was captured on an AMD gfx1200 part. The
//! comparison is bit-exact unless the ISA manual grants a tolerance for that
//! opcode, or the operation cannot be bit-exact by its nature -- in which case
//! the threshold is quoted from the manual or derived from measurement, and the
//! comment says which. Special values (NaN, +-0, denormals, +-inf) are always
//! compared bit-exactly, because the manual pins them down in every case.
//!
//! Each instruction is tested in every encoding it has, because the simulator
//! implements the encodings independently: an opcode can work in one and abort
//! in another.

mod compare;
mod encoding;
mod harness;

mod vop1;
mod vop2;
mod vop3;
mod mem;
mod salu;
mod vopc;
mod ds;
mod scratch;
