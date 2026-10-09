use super::super::super::logic::{lane_test, projected_word, Atom, Choice, Logic};
use super::super::queries::Queries;
use super::decide::Decisions;
use super::forms::{Form, Forms};
use super::joint::{Desc, LANE, WAVE};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Write {
    pub(super) at: (BlockId, usize),
    pub(super) space: Space,
    pub(super) op: MemoryOp,
    pub(super) address: Option<usize>,
    pub(super) data: Option<usize>,
    pub(super) mask: Bdd,
    pub(super) semantics: MemorySemantics,
    pub(super) partnered: bool,
}

pub(super) struct Evaluation {
    pub(super) side: usize,
    pub(super) block: BlockId,
    pub(super) cond: Bdd,
    pub(super) descs: HashMap<ValueId, Desc>,
    pub(super) writes: Vec<Write>,
    pub(super) stored: bool,
}

pub(super) struct Eval<'c, 'a, Q: Queries<'a>> {
    pub(super) q: &'c mut Q,
    pub(super) forms: Forms<'a>,
    leaves: Vec<Bdd>,
    decisions: Decisions,
}

impl<'c, 'a, Q: Queries<'a>> Eval<'c, 'a, Q> {
    pub(super) fn new(q: &'c mut Q, assume: Bdd) -> Self {
        let forms = Forms::new(q.program().f, q.program().facts);
        Self {
            q,
            forms,
            leaves: Vec::new(),
            decisions: Decisions::new(assume),
        }
    }

    #[inline]
    pub(super) fn logic(&mut self) -> &mut Logic {
        self.q.logic()
    }

    #[inline]
    pub(super) fn assume(&self) -> Bdd {
        self.decisions.assume()
    }

    #[inline]
    pub(super) fn fresh(&mut self, side: usize, v: ValueId, position: u32) -> Bdd {
        self.q.logic().atom(Atom::Fresh(side, v, position))
    }

    #[inline]
    pub(super) fn decide(&mut self, g: Bdd, cond: Bdd, side: usize) -> Option<bool> {
        self.decisions.decide(self.q, g, cond, side)
    }

    #[inline]
    pub(super) fn weaken(&mut self, g: Bdd, side: usize) -> Bdd {
        self.decisions.weaken(self.q, g, side)
    }

    #[inline]
    pub(super) fn varies(&mut self, wave: Bdd, lane: Bdd) -> Bdd {
        self.decisions.varies(self.q, wave, lane)
    }

    #[inline]
    pub(super) fn leaves(&self, t: usize) -> Bdd {
        self.leaves[t]
    }

    pub(super) fn intern(&mut self, ty: Ty, form: Form) -> usize {
        let t = self.forms.intern(ty, form);
        while self.leaves.len() < self.forms.len() {
            let leaves = match self.forms.term(self.leaves.len()) {
                Form::Value(v) => self.q.whole(*v),
                Form::Core(_, op) => {
                    let mut children = Vec::new();
                    op.map(|c| {
                        children.push(c.0);
                        c
                    });
                    let mut h = Bdd::FALSE;
                    for c in children {
                        h = self.q.or(h, self.leaves[c]);
                    }
                    h
                }
                Form::Target(_, args, _) => {
                    let mut h = Bdd::FALSE;
                    for &c in args {
                        h = self.q.or(h, self.leaves[c]);
                    }
                    h
                }
                Form::Load(_, _, a) => self.leaves[*a],
                Form::Hazard(block, index, t) => {
                    let loaded = self.q.loaded((*block, *index)).unwrap();
                    self.q.or(self.leaves[*t], loaded)
                }
                Form::Opaque(..) => Bdd::TRUE,
                Form::Linear(_, terms, _) => {
                    let mut h = Bdd::FALSE;
                    for &(c, _) in terms {
                        h = self.q.or(h, self.leaves[c]);
                    }
                    h
                }
            };
            self.leaves.push(leaves);
        }
        t
    }

    pub(super) fn start(&mut self, v: ValueId) -> Desc {
        let (f, facts) = (self.q.program().f, self.q.program().facts);
        let ty = f.types[v.0];
        let bits = match ty {
            Ty::I1 => Some(self.q.bit(v)),
            Ty::I32 | Ty::I64 if facts.viewed[v.0] => Some(self.q.view(v)),
            _ => None,
        };
        let same = Some(self.intern(ty, Form::Value(v)));
        Desc { same, bits }
    }

    pub(super) fn start_wave(&mut self, v: ValueId) -> Desc {
        let d = self.start(v);
        let Some(bits) = d.bits else {
            return d;
        };
        let h = self.q.h(v);
        if self.decisions.reliable(self.q, h) {
            return d;
        }
        let atom = if self.q.program().f.types[v.0] == Ty::I1 {
            Atom::Bit(v)
        } else {
            Atom::View(v)
        };
        let unknown = self.q.logic().atom(atom);
        let bits = self.q.logic().m.ite(h, unknown, bits);
        Desc {
            same: d.same,
            bits: Some(bits),
        }
    }

    fn loaded(&mut self, ty: Ty, v: ValueId, t: usize) -> usize {
        match self.q.program().facts.site[v.0] {
            Site::Inst { block, index } if self.q.loaded((block, index)).is_some() => {
                self.intern(ty, Form::Hazard(block, index, t))
            }
            _ => t,
        }
    }

    fn form_bits(&mut self, side: usize, v: ValueId, t: usize, view: bool) -> Bdd {
        if self.decisions.reliable(self.q, self.leaves[t]) {
            self.q.logic().atom(Atom::Term(t, view))
        } else {
            self.fresh(side, v, 0)
        }
    }

    fn formed(&mut self, side: usize, v: ValueId, t: Option<usize>) -> Desc {
        let (f, facts) = (self.q.program().f, self.q.program().facts);
        let ty = f.types[v.0];
        if ty == Ty::I1 || (matches!(ty, Ty::I32 | Ty::I64) && facts.viewed[v.0]) {
            let bits = match t {
                Some(t) => self.form_bits(side, v, t, ty != Ty::I1),
                None => self.fresh(side, v, 0),
            };
            Desc {
                same: t,
                bits: Some(bits),
            }
        } else {
            Desc {
                same: t,
                bits: None,
            }
        }
    }

    pub(super) fn get(&mut self, ev: &mut Evaluation, v: ValueId) -> Desc {
        if let Some(&d) = ev.descs.get(&v) {
            return d;
        }
        let inst = self
            .q
            .program()
            .facts
            .inst(self.q.program().f, v)
            .expect("a detour reads a value its block does not define");
        self.compute(ev, inst);
        ev.descs[&v]
    }

    fn bits_of(&mut self, ev: &mut Evaluation, v: ValueId) -> Bdd {
        match self.get(ev, v).bits {
            Some(g) => g,
            None => self.fresh(ev.side, v, 0),
        }
    }

    fn operand(&mut self, ev: &mut Evaluation, v: ValueId) -> usize {
        match self.get(ev, v).same {
            Some(t) => t,
            None => {
                let ty = self.q.program().f.types[v.0];
                self.intern(ty, Form::Opaque(ev.side, v))
            }
        }
    }

    fn form(&mut self, ev: &mut Evaluation, ty: Ty, op: Op) -> usize {
        let mut children = Vec::new();
        op.map(|c| {
            children.push(c);
            c
        });
        let terms: Vec<usize> = children.into_iter().map(|c| self.operand(ev, c)).collect();
        let mut next = terms.into_iter();
        let mapped = op.map(|_| ValueId(next.next().unwrap()));
        self.intern(ty, Form::Core(ty, mapped))
    }

    fn compute(&mut self, ev: &mut Evaluation, inst: &Inst) {
        let (f, facts) = (self.q.program().f, self.q.program().facts);
        let (side, cond) = (ev.side, ev.cond);
        match inst {
            Inst::Core { value, ty, op } => {
                let (value, ty, op) = (*value, *ty, *op);
                let lane_only = match ty {
                    Ty::I1 => true,
                    Ty::I32 | Ty::I64 => facts.viewed[value.0],
                    _ => false,
                };
                let lanes = if lane_only { self.logic().lane_function(f, facts, value) } else { None };
                let desc = match op {
                    _ if lanes.is_some() => {
                        let values = lanes.unwrap();
                        let bits = if ty == Ty::I1 {
                            self.logic().lanes(|l| values[l as usize] & 1 == 1)
                        } else {
                            self.logic().lanes(|l| values[l as usize] >> (l & 31) & 1 == 1)
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits: Some(bits),
                        }
                    }
                    Op::Const(_, k) => {
                        let bits = match ty {
                            Ty::I1 => Some(Manager::constant(k != 0)),
                            Ty::I32 | Ty::I64 if facts.viewed[value.0] => Some(self.logic().word_of(ty, k)),
                            _ => None,
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits,
                        }
                    }
                    Op::Env(Env::ValidLane) => Desc {
                        same: Some(self.form(ev, ty, op)),
                        bits: Some(Bdd::TRUE),
                    },
                    Op::Select(c, a, b) => {
                        let gc = self.bits_of(ev, c);
                        match self.decide(gc, cond, side) {
                            Some(true) => self.get(ev, a),
                            Some(false) => self.get(ev, b),
                            None => {
                                let (da, db) = (self.get(ev, a), self.get(ev, b));
                                if da == db {
                                    da
                                } else {
                                    let same = self.form(ev, ty, op);
                                    let bits = match (da.bits, db.bits) {
                                        (Some(ga), Some(gb)) => {
                                            Some(self.logic().m.ite(gc, ga, gb))
                                        }
                                        _ => self.formed(side, value, Some(same)).bits,
                                    };
                                    Desc {
                                        same: Some(same),
                                        bits,
                                    }
                                }
                            }
                        }
                    }
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b)
                        if matches!(ty, Ty::I1 | Ty::I32 | Ty::I64) =>
                    {
                        if ty != Ty::I1 && !facts.viewed[value.0] {
                            Desc {
                                same: Some(self.form(ev, ty, op)),
                                bits: None,
                            }
                        } else {
                            let ga = self.bits_of(ev, a);
                            let gb = self.bits_of(ev, b);
                            let m = &mut self.logic().m;
                            let g = match k {
                                IntOp::And => m.and(ga, gb),
                                IntOp::Or => m.or(ga, gb),
                                _ => m.xor(ga, gb),
                            };
                            Desc {
                                same: Some(self.form(ev, ty, op)),
                                bits: Some(g),
                            }
                        }
                    }
                    Op::Convert(Cvt::Bitcast, to, a) if f.types[a.0] == to => self.get(ev, a),
                    Op::Pack64(a, b) if facts.viewed[value.0] => {
                        let low = self.bits_of(ev, a);
                        let bits = if f.lanes == 32 {
                            low
                        } else {
                            let high = self.bits_of(ev, b);
                            let upper = self.logic().atom(Atom::Lane(5));
                            self.logic().m.ite(upper, high, low)
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits: Some(bits),
                        }
                    }
                    Op::UnpackLo(a) | Op::UnpackHi(a) if facts.viewed[value.0] => {
                        let high = matches!(op, Op::UnpackHi(_));
                        let other = self.fresh(side, value, 0);
                        let bits = if high && f.lanes == 32 {
                            other
                        } else {
                            let own = self.bits_of(ev, a);
                            self.logic().half(high, own, other)
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits: Some(bits),
                        }
                    }
                    Op::Convert(Cvt::Trunc, Ty::I1, s) if projected_word(f, facts, s).is_some() => {
                        let w = projected_word(f, facts, s).unwrap();
                        let same = self.form(ev, ty, op);
                        Desc {
                            same: Some(same),
                            bits: Some(self.bits_of(ev, w)),
                        }
                    }
                    Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b) if f.types[a.0] == Ty::I1 => {
                        let ga = self.bits_of(ev, a);
                        let gb = self.bits_of(ev, b);
                        let m = &mut self.logic().m;
                        let g = if p == IntPred::Ne {
                            m.xor(ga, gb)
                        } else {
                            m.iff(ga, gb)
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits: Some(g),
                        }
                    }
                    Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b)
                        if lane_test(f, facts, a, b).is_some_and(|w| !facts.materialized[w.0]) =>
                    {
                        let w = lane_test(f, facts, a, b).unwrap();
                        let bit = self.bits_of(ev, w);
                        let answer = self.answer(side, value, bit, cond);
                        let g = if p == IntPred::Ne {
                            answer
                        } else {
                            self.logic().m.not(answer)
                        };
                        let mode = self.logic().materialized(facts, w);
                        let g = if mode == Bdd::FALSE {
                            g
                        } else {
                            let t = self.form(ev, ty, op);
                            let full = self.formed(side, value, Some(t)).bits.unwrap();
                            self.logic().m.ite(mode, full, g)
                        };
                        Desc {
                            same: None,
                            bits: Some(g),
                        }
                    }
                    _ => {
                        let t = self.form(ev, ty, op);
                        self.formed(side, value, Some(t))
                    }
                };
                ev.descs.insert(value, desc);
            }
            Inst::Effect {
                op,
                inputs,
                outputs,
                ..
            } => match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let out = outputs[0].0;
                    let local = self.logic().local(Choice::Query(out));
                    let answer = if local == Bdd::FALSE {
                        Bdd::FALSE
                    } else {
                        let bit = self.bits_of(ev, inputs[0]);
                        self.answer(side, out, bit, cond)
                    };
                    let g = if local == Bdd::TRUE {
                        answer
                    } else {
                        let wave = self.fresh(side, out, 0);
                        self.logic().m.ite(local, answer, wave)
                    };
                    ev.descs.insert(
                        out,
                        Desc {
                            same: None,
                            bits: Some(g),
                        },
                    );
                }
                EffectOp::Wave(WaveOp::Ballot { high }) => {
                    let own = self.bits_of(ev, inputs[0]);
                    let g = if f.lanes == 32 {
                        own
                    } else {
                        let other = self.fresh(side, outputs[0].0, 1);
                        self.logic().half(*high, own, other)
                    };
                    ev.descs.insert(
                        outputs[0].0,
                        Desc {
                            same: None,
                            bits: Some(g),
                        },
                    );
                }
                EffectOp::Memory {
                    space,
                    op: MemoryOp::Load(size),
                    ..
                } => {
                    let (out, ty) = outputs[0];
                    let pred = self.bits_of(ev, inputs[1]);
                    let same = if !self.stored_before(ev, out) && self.decide(pred, cond, side) == Some(true) {
                        let a = self.operand(ev, inputs[0]);
                        let t = self.intern(ty, Form::Load(*space, *size, a));
                        Some(self.loaded(ty, out, t))
                    } else {
                        None
                    };
                    let d = self.formed(side, out, same);
                    ev.descs.insert(out, d);
                }
                _ => {
                    for &(v, _) in outputs {
                        let d = self.formed(side, v, None);
                        ev.descs.insert(v, d);
                    }
                }
            },
            Inst::Target {
                op, args, outputs, ..
            } => {
                if outputs.first().is_some_and(|&(v, _)| self.reads_memory(v) && self.stored_before(ev, v)) {
                    for &(v, _) in outputs {
                        let d = self.formed(side, v, None);
                        ev.descs.insert(v, d);
                    }
                    return;
                }
                let terms: Vec<usize> =
                    args.values().iter().map(|&a| self.operand(ev, a)).collect();
                for (i, &(v, ty)) in outputs.iter().enumerate() {
                    let t = self.intern(ty, Form::Target(*op, terms.clone(), i));
                    let t = self.loaded(ty, v, t);
                    let d = self.formed(side, v, Some(t));
                    ev.descs.insert(v, d);
                }
            }
            Inst::Packet { .. } => unreachable!("a packet query in a wave program"),
        }
    }

    fn stored_before(&self, ev: &Evaluation, v: ValueId) -> bool {
        let Site::Inst { index, .. } = self.q.program().facts.site[v.0] else {
            return ev.stored || ev.writes.iter().any(|w| w.op != MemoryOp::Fence);
        };
        ev.stored || ev.writes.iter().any(|w| w.op != MemoryOp::Fence && w.at.1 < index)
    }

    fn reads_memory(&self, v: ValueId) -> bool {
        let Site::Inst { block, index } = self.q.program().facts.site[v.0] else {
            return true;
        };
        self.q.program().hazards.accesses.iter().any(|a| (a.block, a.index) == (block, index))
    }

    pub(super) fn check_effects(&mut self, ev: &mut Evaluation) {
        let (f, facts) = (self.q.program().f, self.q.program().facts);
        for (index, inst) in f.blocks[&ev.block].insts.iter().enumerate() {
            if let Some(&m) = self.q.program().meetings.get(&(ev.block, index)) {
                let local = self.logic().local(Choice::Meet(m));
                let kept = self.q.not(local);
                let differs = self.q.and(ev.cond, kept);
                self.q.require(
                    ev.block,
                    index,
                    "a retained meeting may have different participating lanes",
                    differs,
                );
                if self.q.stopped() {
                    return;
                }
            }
            let Inst::Effect {
                op,
                inputs,
                outputs,
                ..
            } = inst
            else {
                continue;
            };
            let collective = match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let local = self.logic().local(Choice::Query(outputs[0].0));
                    self.q.not(local)
                }
                EffectOp::Wave(WaveOp::Ballot { .. }) => {
                    self.logic().materialized(facts, outputs[0].0)
                }
                EffectOp::Wave(WaveOp::ReadFirstLane) => {
                    Manager::constant(!facts.uniform[inputs[0].0])
                }
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                    Bdd::TRUE
                }
                EffectOp::Memory { .. } => Bdd::FALSE,
            };
            if collective != Bdd::FALSE {
                let mut differs = self.q.and(ev.cond, collective);
                if ev.side == WAVE && matches!(op, EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. })) {
                    let bit = self.bits_of(ev, inputs[0]);
                    let present = self.weaken(bit, WAVE);
                    differs = self.q.and(differs, present);
                }
                self.q.require(
                    ev.block,
                    index,
                    "a retained collective may have different participating lanes",
                    differs,
                );
                if self.q.stopped() {
                    return;
                }
            }
            if let EffectOp::Memory {
                space,
                op: MemoryOp::Fence,
                semantics,
            } = op
            {
                let at = (ev.block, index);
                ev.writes.push(Write {
                    at,
                    space: *space,
                    op: MemoryOp::Fence,
                    address: None,
                    data: None,
                    mask: Bdd::TRUE,
                    semantics: *semantics,
                    partnered: false,
                });
            }
            if let EffectOp::Memory {
                op:
                    memory @ (MemoryOp::Store(_)
                    | MemoryOp::AtomicAdd(_)
                    | MemoryOp::AtomicRmw(_)
                    | MemoryOp::AtomicCmpSwap),
                semantics,
                ..
            } = op
            {
                let pred = self.bits_of(ev, inputs[memory.mask_input()]);
                if self.decide(pred, ev.cond, ev.side) != Some(false) {
                    let at = (ev.block, index);
                    let space = match op {
                        EffectOp::Memory { space, .. } => *space,
                        _ => unreachable!(),
                    };
                    let address = self.get(ev, inputs[0]).same;
                    let data = match memory {
                        MemoryOp::AtomicCmpSwap => match (self.get(ev, inputs[1]).same, self.get(ev, inputs[2]).same) {
                            (Some(d), Some(c)) => Some(self.intern(Ty::I64, Form::Core(Ty::I64, Op::Pack64(ValueId(d), ValueId(c))))),
                            _ => None,
                        },
                        _ => self.get(ev, inputs[1]).same,
                    };
                    ev.writes.push(Write {
                        at,
                        space,
                        op: *memory,
                        address,
                        data,
                        mask: pred,
                        semantics: *semantics,
                        partnered: self.q.program().partnered(at),
                    });
                }
            }
        }
    }

    fn answer(&mut self, side: usize, v: ValueId, bit: Bdd, cond: Bdd) -> Bdd {
        if side == LANE {
            return bit;
        }
        if self.decide(bit, cond, side) == Some(true) {
            return Bdd::TRUE;
        }
        let others = self.fresh(side, v, 0);
        self.logic().m.or(bit, others)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::super::hazard::Hazards;
    use super::super::super::super::testing::*;
    use super::super::super::differences::Differences;
    use super::super::super::program::Program;
    use super::super::super::Mode;
    use super::*;
    use crate::rdna_spmd::analysis::facts::Facts;
    use crate::rdna_spmd::analysis::loops::Loops;
    use std::collections::BTreeSet;

    fn queried() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let a = b.wave(e, WaveOp::Any, vec![c]);
        (b, a)
    }

    #[test]
    fn the_wave_side_starts_from_the_lane_value_only_where_the_programs_agree() {
        let (b, a) = queried();
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let loops = Loops::new(f, &facts).unwrap();
        let hazards = Hazards {
            accesses: Vec::new(),
            together: BTreeSet::new(),
            apart: BTreeSet::new(),
            idle: BTreeSet::new(),
            meetings: Vec::new(),
        };
        let mut r = Random::new(29);
        let mut wrong = Vec::new();
        let mut unknown_somewhere = 0;
        for trial in 0..300 {
            let logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
            let program = Program::new(f, &facts, &b.inputs, Some(0), &loops, &hazards);
            let mut q = Differences::new(program, logic, Mode::Search);
            let lane_bits = q.bit(a);
            let mut pool: Vec<Atom> = q.logic().support(lane_bits).iter().map(|&v| q.logic().atom_of(v)).collect();
            pool.extend([Atom::Lane(0), Atom::Lane(1)]);
            let h = random_function(q.logic(), &mut r, &pool, 3);
            q.raise_h(a, h);
            let assume = random_function(q.logic(), &mut r, &pool, 3);
            if assume == Bdd::FALSE {
                continue;
            }
            let own = variable(q.logic(), Atom::Bit(a));
            let vars: Vec<u32> = pool.iter().map(|&atom| variable(q.logic(), atom)).collect();
            let mut eval = Eval::new(&mut q, assume);
            let wave = eval.start_wave(a);
            let lane = eval.start(a);
            if wave.same != lane.same {
                wrong.push(format!("trial {} changes the form", trial));
            }
            let bits = wave.bits.unwrap();
            let (mut every, mut none) = (true, true);
            for row in 0..1u32 << vars.len() {
                let at = |var: u32| vars.iter().position(|&v| v == var).is_some_and(|i| row >> i & 1 == 1);
                if !evaluate(&eval.q.logic().m, assume, &at) {
                    continue;
                }
                let differs = evaluate(&eval.q.logic().m, h, &at);
                let expected: BTreeSet<bool> = if differs {
                    unknown_somewhere += 1;
                    BTreeSet::from([false, true])
                } else {
                    BTreeSet::from([evaluate(&eval.q.logic().m, lane_bits, &at)])
                };
                let taken: BTreeSet<bool> = [false, true]
                    .iter()
                    .map(|&x| evaluate(&eval.q.logic().m, bits, &|var| if var == own { x } else { at(var) }))
                    .collect();
                if taken != expected {
                    wrong.push(format!("trial {} row {} takes {:?}, expected {:?}", trial, row, taken, expected));
                }
                every &= expected.iter().all(|&x| x);
                none &= expected.iter().all(|&x| !x);
            }
            let expected = if every { Some(true) } else if none { Some(false) } else { None };
            let decided = eval.decide(bits, assume, WAVE);
            if decided != expected {
                wrong.push(format!("trial {} decides {:?}, expected {:?}", trial, decided, expected));
            }
        }
        assert!(unknown_somewhere > 0, "some trial leaves the wave's answer unknown");
        assert!(wrong.is_empty(), "the wave side must start from what the lane program shows where nothing differs: {:?}", &wrong[..wrong.len().min(8)]);
    }
}
