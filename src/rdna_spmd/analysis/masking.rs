//! How a program carries the mask of the lanes each of its operations runs
//! for.
//!
//! A program lifted from the wave keeps the mask in the EXEC register. The
//! register is a bit parameter of every block, found by its position among the
//! bit parameters, and a register write only some lanes make is a choice
//! between the new and the old word, which the analyses of predication
//! recognise. A program the lockstep lowering made keeps each mask as an
//! ordinary value that the operations it guards take as an operand; its bit
//! parameters are values like any other.
//!
//! The analyses and passes both kinds of program go through are generic over
//! the kind and ask it what the masks mean. Each kind has its own pipeline,
//! which fixes the kind for every analysis and pass in it.

use super::super::ir::{Cvt, IntOp, Op, Ty, *};
use super::dataflow::{Cfg, Sparse};
use super::masks::any_of;
use super::{Access, Analyses, Exec, Masks};

pub(crate) trait Masking: 'static {
    /// Values chosen from words the lanes may disagree on that every lane
    /// reading the value agrees on.
    fn guarded(f: &Func, analyses: &Analyses) -> Vec<bool>;
    /// Whether the mask each access runs under always holds a lane.
    fn holds_a_lane(f: &Func, analyses: &Analyses, accesses: &[Access]) -> Vec<bool>;
    /// Whether an access whose mask is the constant `mask` works on the word
    /// of a scalar register, once for the packet.
    fn scalar_word(mask: Option<u64>) -> bool;
    /// Whether a parameter of `ty` stays for its position, read or not.
    fn positional(ty: Ty) -> bool;
}

/// Masks kept in the EXEC register.
pub(crate) struct ExecRegister;

impl Masking for ExecRegister {
    /// A register write the lanes it leaves out never read the old word of.
    fn guarded(f: &Func, analyses: &Analyses) -> Vec<bool> {
        analyses.get::<Masks>(f).guarded.clone()
    }
    /// An access runs under the register where it stands.
    fn holds_a_lane(f: &Func, analyses: &Analyses, accesses: &[Access]) -> Vec<bool> {
        let exec = analyses.get::<Exec>(f);
        accesses
            .iter()
            .map(|a| exec.nonempty_at(a.block, a.effects[0]))
            .collect()
    }
    /// The lifter masks an access to a scalar register's word with the
    /// constant one.
    fn scalar_word(mask: Option<u64>) -> bool {
        mask == Some(1)
    }
    /// The analyses of the register find it by its position among the bit
    /// parameters.
    fn positional(ty: Ty) -> bool {
        ty == Ty::I1
    }
}

/// Masks kept as values.
pub(crate) struct MaskValues;

impl Masking for MaskValues {
    /// The lowering chooses only where lanes that computed different values
    /// meet, and every lane that meets there reads the choice.
    fn guarded(f: &Func, _: &Analyses) -> Vec<bool> {
        vec![false; f.types.len()]
    }
    /// An access names the mask it runs under.
    fn holds_a_lane(f: &Func, analyses: &Analyses, accesses: &[Access]) -> Vec<bool> {
        let ctx = analyses.context();
        let nonempty = nonempty(f, ctx.exec_index, ctx.exec_initial);
        accesses.iter().map(|a| nonempty[a.mask.0]).collect()
    }
    /// A mask of the constant one is every lane of the packet.
    fn scalar_word(_: Option<u64>) -> bool {
        false
    }
    /// A mask is found by the value an operation names, never by position.
    fn positional(_: Ty) -> bool {
        false
    }
}

/// Masks at least one lane is always in: the mask the dispatch started with,
/// a mask a query has already answered for, and what those two build up.
fn nonempty(f: &Func, exec_index: usize, initial: bool) -> Vec<bool> {
    let cfg = Cfg::new(f);
    let defs = f.definitions();
    let queries = any_of(f);
    let entry_exec = f.blocks[&f.entry].params[exec_index].0;
    let boundary = |p: ValueId| p == entry_exec && initial;
    let edge = |edge: &Edge, src: usize, position: usize, facts: &[bool]| {
        let arg = edge.args[position];
        if facts[arg.0] {
            return true;
        }
        match &cfg.blocks[src].term {
            Term::CondBr { cond, yes, .. } => {
                std::ptr::eq(edge, yes) && queries[cond.0] == Some(arg)
            }
            _ => false,
        }
    };
    let transfer = |_: &Inst, v: ValueId, facts: &[bool]| match defs[v.0] {
        Some(Op::Const(Ty::I1, bits)) => bits & 1 != 0,
        Some(Op::Int(IntOp::Or, a, b)) => facts[a.0] || facts[b.0],
        Some(Op::Convert(Cvt::Bitcast, Ty::I1, a)) => facts[a.0],
        _ => false,
    };
    Sparse {
        cfg: &cfg,
        start: true,
        boundary: &boundary,
        edge: &edge,
        transfer: &transfer,
    }
    .solve(f.types.len())
}
