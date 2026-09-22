use super::logic::{lane_test, Atom, Kept, Logic};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::TargetOp;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::hash::HashMap;
use std::collections::BTreeMap;

pub fn prove(
    f: &Func,
    facts: &Facts,
    inputs: &[crate::rdna_spmd::ir::Parameter],
    exec_index: Option<usize>,
) -> (Kept, std::collections::BTreeSet<u64>) {
    let mut all_local = Logic::new(f, facts, &Default::default());
    let mut exhausted = match analyse(f, facts, &mut all_local, Bdd::TRUE, inputs, exec_index) {
        Ok(chosen) => return chosen,
        Err(exhausted) => exhausted,
    };
    drop(all_local);
    let mut others = Logic::policies(f, facts);
    let rest = others.m.not(others.all_local());
    if rest != Bdd::FALSE {
        match analyse(f, facts, &mut others, rest, inputs, exec_index) {
            Ok(chosen) => return chosen,
            Err(last) => exhausted = last,
        }
    }
    let (block, index, reason) = exhausted;
    panic!(
        "b{}:{}: {} under every conversion policy",
        block.0, index, reason
    )
}

type Exhausted = (BlockId, usize, &'static str);

fn analyse(
    f: &Func,
    facts: &Facts,
    logic: &mut Logic,
    safe: Bdd,
    inputs: &[crate::rdna_spmd::ir::Parameter],
    exec_index: Option<usize>,
) -> Result<(Kept, std::collections::BTreeSet<u64>), Exhausted> {
    let exec = exec_index.map(|index| f.blocks[&f.entry].params[index].0);
    let mut proof = Proof {
        f,
        facts,
        logic,
        safe,
        exhausted: None,
        demands: BTreeMap::new(),
        masked: vec![false; f.types.len()],
        faithful: vec![false; f.types.len()],
        h: vec![Bdd::FALSE; f.types.len()],
        words: vec![Bdd::FALSE; f.types.len()],
        arrivals: BTreeMap::new(),
        reach: BTreeMap::new(),
        rank: facts
            .order
            .iter()
            .enumerate()
            .map(|(r, &b)| (b, r))
            .collect(),
        loops: crate::rdna_spmd::analysis::loops::Loops::new(f, facts).unwrap_or_else(|block| {
            panic!(
                "b{}: control flow enters a cycle other than through its header",
                block.0
            )
        }),
    };
    let start = match exec {
        Some(e) => proof.logic.atom(Atom::Bit(e)),
        None => Bdd::TRUE,
    };
    proof.reach = proof.logic.policy_reach(f, facts, f.entry, start);
    for r in proof.reach.values_mut() {
        *r = proof.logic.m.and(*r, safe);
    }
    proof.solve_masked(inputs, exec_index);
    loop {
        proof.settle();
        if let Some(exhausted) = proof.exhausted {
            return Err(exhausted);
        }
        let arrivals = proof.detours();
        if let Some(exhausted) = proof.exhausted {
            return Err(exhausted);
        }
        let mut changed = false;
        for (block, contributions) in arrivals {
            let old = proof
                .arrivals
                .entry(block)
                .or_insert_with(|| vec![Bdd::FALSE; contributions.len()]);
            for (slot, c) in contributions.into_iter().enumerate() {

                let c = proof.logic.m.and(c, proof.safe);
                let joined = proof.logic.m.or(old[slot], c);
                if joined != old[slot] {
                    old[slot] = joined;
                    changed = true;
                }
            }
        }
        if !changed {
            let kept = proof.logic.choose(proof.safe);
            let disabled: Vec<_> = kept.queries.iter().chain(&kept.words).copied().collect();
            let everyone = std::mem::take(&mut proof.demands)
                .into_iter()
                .filter_map(|(p, condition)| {
                    (proof.logic.settled(condition, &disabled) == Bdd::TRUE).then_some(p)
                })
                .collect();
            return Ok((kept, everyone));
        }
    }
}

struct Proof<'a> {
    f: &'a Func,
    facts: &'a Facts,
    logic: &'a mut Logic,
    safe: Bdd,

    exhausted: Option<Exhausted>,

    demands: BTreeMap<u64, Bdd>,

    masked: Vec<bool>,

    faithful: Vec<bool>,
    h: Vec<Bdd>,

    words: Vec<Bdd>,
    arrivals: BTreeMap<BlockId, Vec<Bdd>>,
    reach: BTreeMap<BlockId, Bdd>,
    rank: BTreeMap<BlockId, usize>,
    loops: crate::rdna_spmd::analysis::loops::Loops,
}

impl Proof<'_> {
    fn reachable(&self, id: BlockId) -> Bdd {
        self.reach.get(&id).copied().unwrap_or(Bdd::FALSE)
    }
    fn bit(&mut self, v: ValueId) -> Bdd {
        self.logic.bit(self.f, self.facts, v)
    }
    fn view(&mut self, v: ValueId) -> Bdd {
        self.logic.view(self.f, self.facts, v)
    }
    fn or(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.logic.m.or(a, b)
    }
    fn and(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.logic.m.and(a, b)
    }
    fn not(&mut self, a: Bdd) -> Bdd {
        self.logic.m.not(a)
    }

    fn whole(&self, v: ValueId) -> Bdd {
        if self.facts.lane_word[v.0] {
            self.words[v.0]
        } else {
            self.h[v.0]
        }
    }

    fn solve_masked(
        &mut self,
        inputs: &[crate::rdna_spmd::ir::Parameter],
        exec_index: Option<usize>,
    ) {
                let (f, facts) = (self.f, self.facts);
        let word = |v: ValueId| f.types[v.0] == Ty::I32 && facts.lane_word[v.0];
        let Some(exec_index) = exec_index else {
            self.masked = vec![true; f.types.len()];
            self.faithful = (0..f.types.len()).map(|v| word(ValueId(v))).collect();
            return;
        };
        for (&id, block) in &f.blocks {
            let exec = block.params[exec_index].0;
            for (index, &(param, ty)) in block.params.iter().enumerate() {

                let cleared = matches!(
                    inputs.get(index).map(|p| p.source),
                    Some(ParameterSource::MaskBit(_))
                );
                let assumed = id != f.entry || param == exec || cleared;
                self.masked[param.0] = assumed && (ty == Ty::I1 || word(param));
                self.faithful[param.0] = assumed && word(param);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &facts.order {
                let block = &f.blocks[&id];
                let exec = block.params[exec_index].0;
                let mut active = self.logic.atom(Atom::Bit(exec));
                for &(param, ty) in &block.params {
                    if param != exec && self.masked[param.0] {
                        let atom = if ty == Ty::I1 {
                            Atom::Bit(param)
                        } else {
                            Atom::View(param)
                        };
                        let known = self.logic.atom(atom);
                        active = self.or(active, known);
                    }
                }
                let mut lockstep: HashMap<ValueId, Option<Bdd>> = HashMap::default();
                for inst in &block.insts {
                    for v in inst.outputs() {
                        let bit = f.types[v.0] == Ty::I1;
                        if !bit && f.types[v.0] != Ty::I32 {
                            continue;
                        }
                        let formula = if bit { self.bit(v) } else { self.view(v) };
                        let masked = self.logic.m.implies(formula, active);
                        if self.masked[v.0] != masked {
                            self.masked[v.0] = masked;
                            changed = true;
                        }
                        if word(v) {
                            let faithful = self
                                .lockstep_view(v, active, &mut lockstep)
                                .is_some_and(|l| self.logic.m.iff(l, formula) == Bdd::TRUE);
                            if self.faithful[v.0] != faithful {
                                self.faithful[v.0] = faithful;
                                changed = true;
                            }
                        }
                    }
                }
            }
            for &id in &facts.order {
                if id == f.entry {
                    continue;
                }
                let block = &f.blocks[&id];
                let exec = block.params[exec_index].0;
                for (index, &(param, _)) in block.params.iter().enumerate() {
                    if param == exec {
                        continue;
                    }
                    let mut arguments = facts.arguments(f, id, index);
                    if self.masked[param.0] && !arguments.all(|a| self.masked[a.0]) {
                        self.masked[param.0] = false;
                        changed = true;
                    }

                    let mut arguments = facts.arguments(f, id, index);
                    if self.faithful[param.0] && !arguments.all(|a| !word(a) || self.faithful[a.0])
                    {
                        self.faithful[param.0] = false;
                        changed = true;
                    }
                }
            }
        }
    }

    fn lockstep_view(
        &mut self,
        w: ValueId,
        active: Bdd,
        memo: &mut HashMap<ValueId, Option<Bdd>>,
    ) -> Option<Bdd> {
        if let Some(&l) = memo.get(&w) {
            return l;
        }
        let (f, facts) = (self.f, self.facts);
        let l = if !facts.lane_word[w.0] {
            Some(self.view(w))
        } else {
            match facts.inst(f, w) {
                None => self.faithful[w.0].then(|| self.view(w)),
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot),
                    inputs,
                    ..
                }) => {
                    let x = self.bit(inputs[0]);
                    Some(self.and(x, active))
                }
                Some(Inst::Core { op, .. }) => match *op {
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                        let a = self.lockstep_view(a, active, memo)?;
                        let b = self.lockstep_view(b, active, memo)?;
                        Some(match k {
                            IntOp::And => self.and(a, b),
                            IntOp::Or => self.or(a, b),
                            _ => self.logic.m.xor(a, b),
                        })
                    }
                    Op::Select(c, a, b) => {
                        let c = self.bit(c);
                        let a = self.lockstep_view(a, active, memo)?;
                        let b = self.lockstep_view(b, active, memo)?;
                        Some(self.logic.m.ite(c, a, b))
                    }
                    Op::Convert(Cvt::Bitcast, Ty::I32, a) => self.lockstep_view(a, active, memo),
                    _ => Some(self.view(w)),
                },
                Some(_) => None,
            }
        };
        memo.insert(w, l);
        l
    }

    fn require(
        &mut self,
        block: BlockId,
        index: usize,
        reason: &'static str,
        difference: Bdd,
    ) {
        let difference = self.and(difference, self.safe);
        if difference == Bdd::FALSE {
            return;
        }
        let bad = self.logic.possible_policies(difference);
        let good = self.not(bad);
        self.safe = self.and(self.safe, good);
        for h in self.h.iter_mut().chain(&mut self.words) {
            *h = self.logic.m.and(*h, self.safe);
        }
        for r in self.reach.values_mut() {
            *r = self.logic.m.and(*r, self.safe);
        }
        for incoming in self.arrivals.values_mut() {
            for h in incoming {
                *h = self.logic.m.and(*h, self.safe);
            }
        }
        if self.safe == Bdd::FALSE {
            self.exhausted.get_or_insert((block, index, reason));
        }
    }

    fn demand(&mut self, provenance: u64, condition: Bdd) {
        let old = self.demands.get(&provenance).copied().unwrap_or(Bdd::FALSE);
        let condition = self.or(old, condition);
        self.demands.insert(provenance, condition);
    }

    fn settle(&mut self) {
        let order = &self.facts.order;
        let mut stale = vec![true; order.len()];
        loop {
            let mut changed = false;
            for index in 0..order.len() {
                if !std::mem::take(&mut stale[index]) {
                    continue;
                }
                let id = order[index];
                let safe = self.safe;
                let rose = self.transfer(id);
                if self.safe != safe {
                    stale.fill(true);
                    changed = true;
                }
                if rose {
                    changed = true;
                    for edge in self.f.blocks[&id].term.edges() {
                        stale[self.rank[&edge.dst]] = true;
                    }
                }
            }
            if !changed {
                return;
            }
        }
    }

    fn raise(&mut self, v: ValueId, h: Bdd) -> bool {
        let h = self.and(h, self.safe);
        let joined = self.or(self.h[v.0], h);
        if joined == self.h[v.0] {
            return false;
        }
        self.h[v.0] = joined;
        true
    }

    fn raise_word(&mut self, v: ValueId, h: Bdd) -> bool {
        let h = self.and(h, self.safe);
        let joined = self.or(self.words[v.0], h);
        if joined == self.words[v.0] {
            return false;
        }
        self.words[v.0] = joined;
        true
    }

    fn word_difference(&mut self, inst: &Inst, value: ValueId) -> Bdd {
        match inst {
            Inst::Effect {
                provenance,
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            } => {
                if !self.faithful[value.0] {
                    let whole = self.logic.materialized(self.facts, value);
                    self.demand(*provenance, whole);
                }

                let h = self.h[inputs[0].0];
                let varying: Vec<_> = self
                    .logic
                    .support(h)
                    .into_iter()
                    .filter(|&var| !self.logic.uniform_atom(self.facts, var))
                    .collect();
                self.logic.m.exists(h, &|var| varying.contains(&var))
            }
            Inst::Core {
                op: Op::Select(c, a, b),
                ..
            } => {
                let cbit = self.bit(*c);
                let arms = self.logic.m.ite(cbit, self.whole(*a), self.whole(*b));
                self.or(self.h[c.0], arms)
            }
            _ => {
                let mut h = Bdd::FALSE;
                for a in inst.operands() {
                    h = self.or(h, self.whole(a));
                }
                h
            }
        }
    }

    fn edge_difference(&mut self, pred: BlockId, slot: usize, difference: Bdd) -> Bdd {
        if difference == Bdd::FALSE {
            return difference;
        }
        let mut guard = self.reachable(pred);
        if let Term::CondBr { cond, .. } = self.f.blocks[&pred].term {
            let bit = self.bit(cond);
            let taken = if slot == 0 { bit } else { self.not(bit) };
            guard = self.and(guard, taken);
        }
        let difference = self.and(difference, guard);
        self.logic.image(self.f, self.facts, pred, slot, difference)
    }

    fn transfer(&mut self, id: BlockId) -> bool {
        let (f, facts) = (self.f, self.facts);
        let block = &f.blocks[&id];
        let mut changed = false;
        if id != f.entry {
            let reachable = self.reachable(id);
            for (k, &(param, _)) in block.params.iter().enumerate() {
                let mode = self.logic.materialized(facts, param);
                let mut h = self.arrivals.get(&id).map_or(Bdd::FALSE, |a| a[k]);
                let mut word = h;
                for &(pred, slot) in &facts.incoming[&id] {
                    let arg = f.blocks[&pred].term.edges().nth(slot).unwrap().args[k];
                    let d = self.h[arg.0];
                    if d != Bdd::FALSE {
                        let image = self.edge_difference(pred, slot, d);
                        h = self.or(h, image);
                    }
                    if facts.lane_word[param.0] && mode != Bdd::FALSE {
                        let d = self.whole(arg);
                        if d != Bdd::FALSE {
                            let image = self.edge_difference(pred, slot, d);
                            word = self.or(word, image);
                        }
                    }
                }
                let h = self.and(h, reachable);
                changed |= self.raise(param, h);
                let word = self.and(word, reachable);
                let word = self.logic.m.ite(mode, word, h);
                changed |= self.raise_word(param, word);
            }
        }
        for (index, inst) in block.insts.iter().enumerate() {
            let h = self.inst(id, index, inst);
            for v in inst.outputs() {
                changed |= self.raise(v, h);
                if facts.lane_word[v.0] {
                    let mode = self.logic.materialized(facts, v);
                    let word = if mode == Bdd::FALSE {
                        h
                    } else {
                        let word = self.word_difference(inst, v);
                        self.logic.m.ite(mode, word, h)
                    };
                    changed |= self.raise_word(v, word);
                }
            }
        }
        changed
    }

    fn query(&mut self, x: Bdd, hx: Bdd) -> Bdd {
        let (logic, facts) = (&mut *self.logic, self.facts);
        let set = logic.m.or(x, hx);
        let support = logic.support(set);
        let varying: Vec<u32> = support
            .into_iter()
            .filter(|&var| !logic.uniform_atom(facts, var))
            .collect();
        let others = logic.m.exists(set, &|var| varying.contains(&var));
        let absent = logic.m.not(x);
        let differs = logic.m.and(absent, others);
        logic.m.or(hx, differs)
    }

    fn inst(&mut self, id: BlockId, index: usize, inst: &Inst) -> Bdd {
        let any_of = |p: &mut Self, values: &[ValueId]| {
            let mut h = Bdd::FALSE;
            for &v in values {
                let w = p.whole(v);
                h = p.or(h, w);
            }
            h
        };
        match inst {
            Inst::Core { value, ty, op } => match *op {
                Op::Const(..) | Op::Env(_) => Bdd::FALSE,
                Op::Select(c, a, b) => {
                    let fc = self.bit(c);
                    let (ha, hb, hc) = if self.facts.lane_word[value.0] {
                        (self.h[a.0], self.h[b.0], self.h[c.0])
                    } else {
                        (self.whole(a), self.whole(b), self.h[c.0])
                    };
                    let taken = self.and(fc, ha);
                    let nfc = self.not(fc);
                    let other = self.and(nfc, hb);
                    let arms = self.or(taken, other);
                    self.or(hc, arms)
                }
                Op::Int(k @ (IntOp::And | IntOp::Or), a, b)
                    if *ty == Ty::I1 || self.facts.lane_word[value.0] =>
                {
                    let (fa, fb) = if *ty == Ty::I1 {
                        (self.bit(a), self.bit(b))
                    } else {
                        (self.view(a), self.view(b))
                    };
                    let (ha, hb) = (self.h[a.0], self.h[b.0]);
                    let absorbs = |p: &mut Self, fx: Bdd, hx: Bdd| {
                        let settled = p.not(hx);
                        let value = if k == IntOp::And { p.not(fx) } else { fx };
                        p.and(value, settled)
                    };
                    let za = absorbs(self, fa, ha);
                    let zb = absorbs(self, fb, hb);
                    let either = self.or(ha, hb);
                    let nza = self.not(za);
                    let nzb = self.not(zb);
                    let open = self.and(nza, nzb);
                    self.and(either, open)
                }
                Op::Int(IntOp::Xor, a, b) if self.facts.lane_word[value.0] => {
                    self.or(self.h[a.0], self.h[b.0])
                }
                Op::Convert(Cvt::Bitcast, Ty::I32, a) if self.facts.lane_word[value.0] => {
                    self.h[a.0]
                }
                Op::Convert(Cvt::Trunc, Ty::I1, s) => match self.facts.op(self.f, s) {
                    Some(Op::Int(IntOp::LShr, w, lane))
                        if self.facts.lane_word[w.0] && self.facts.is_lane_id(self.f, lane) =>
                    {
                        self.h[w.0]
                    }
                    _ => self.h[s.0],
                },
                Op::Int(IntOp::LShr, w, lane)
                    if self.facts.lane_word[w.0] && self.facts.is_lane_id(self.f, lane) =>
                {
                    self.h[w.0]
                }
                Op::Cmp(IntPred::Eq | IntPred::Ne, a, b)
                    if lane_test(self.f, self.facts, a, b).is_some() =>
                {
                    let w = lane_test(self.f, self.facts, a, b).unwrap();
                    let mode = self.logic.materialized(self.facts, w);
                    if mode == Bdd::TRUE {
                        self.whole(w)
                    } else {
                        let fw = self.view(w);
                        let hw = self.h[w.0];
                        let local = self.query(fw, hw);
                        self.logic.m.ite(mode, self.whole(w), local)
                    }
                }
                _ => any_of(self, &inst.operands()),
            },
            Inst::Target { args, .. } => any_of(self, args.values()),
            Inst::Packet { .. } => unreachable!("a packet query in a wave program"),
            Inst::Effect {
                provenance,
                op,
                inputs,
                outputs,
            } => match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let local = self.logic.local(outputs[0].0);
                    if !self.masked[inputs[0].0] {
                        let kept = self.not(local);
                        self.demand(*provenance, kept);
                    }
                    let fx = self.bit(inputs[0]);
                    let hx = self.h[inputs[0].0];
                    let query = self.query(fx, hx);
                    self.logic.m.ite(local, query, hx)
                }
                EffectOp::Wave(WaveOp::Ballot) => self.h[inputs[0].0],
                EffectOp::Wave(WaveOp::ReadFirstLane) => {
                    if self.facts.uniform[inputs[0].0] {
                        self.whole(inputs[0])
                    } else {
                        if !self.masked[inputs[1].0] {
                            self.demand(*provenance, Bdd::TRUE);
                        }
                        self.h[inputs[0].0]
                    }
                }
                EffectOp::Memory {
                    op: MemoryOp::Load(_),
                    ..
                } => {
                    let fp = self.bit(inputs[1]);
                    let absent = self.not(fp);
                    let h = any_of(self, &inputs[..2]);
                    self.or(h, absent)
                }
                EffectOp::Memory {
                    op:
                        memory @ (MemoryOp::Store(_)
                        | MemoryOp::AtomicAdd(_)
                        | MemoryOp::AtomicRmw(_)
                        | MemoryOp::AtomicCmpSwap),
                    ..
                } => {
                    let pred = inputs[memory.mask_input()];
                    let reachable = self.reachable(id);
                    let happens = self.and(self.h[pred.0], reachable);
                    if happens != Bdd::FALSE {
                        self.require(
                            id,
                            index,
                            "whether a store happens depends on the other lanes",
                            happens,
                        );
                    }
                    let fp = self.bit(pred);
                    let operands = any_of(self, &inputs[..memory.mask_input()]);
                    let performed = self.and(fp, reachable);
                    let writes = self.and(performed, operands);
                    if writes != Bdd::FALSE {
                        self.require(
                            id,
                            index,
                            "what a store writes depends on the other lanes",
                            writes,
                        );
                    }
                    let absent = self.not(fp);
                    self.or(operands, absent)
                }
                EffectOp::Memory {
                    op: MemoryOp::Fence,
                    ..
                } => Bdd::FALSE,

                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                    any_of(self, inputs)
                }
            },
        }
    }

    fn detours(&mut self) -> BTreeMap<BlockId, Vec<Bdd>> {
        let (f, facts) = (self.f, self.facts);
        let mut arrivals: BTreeMap<BlockId, Vec<Bdd>> = BTreeMap::new();
        for (position, &b) in facts.order.iter().enumerate() {
            let Term::CondBr { cond, yes, no } = &f.blocks[&b].term else {
                continue;
            };
            let hc = self.h[cond.0];
            if hc == Bdd::FALSE || yes.dst == no.dst {
                continue;
            }
            let fc = self.bit(*cond);
            let nfc = self.not(fc);
            let reachable = self.reachable(b);
            for (wave, lane, taken) in [(0, 1, nfc), (1, 0, fc)] {
                let assume = self.and(hc, taken);
                let assume = self.and(assume, reachable);
                let assume = self.and(assume, self.safe);
                if assume == Bdd::FALSE {
                    continue;
                }
                let all_local = self.logic.all_local();
                let other = self.not(all_local);

                for (part, slice) in [all_local, other].iter().enumerate() {
                    let assume = self.and(assume, *slice);
                    if assume == Bdd::FALSE {
                        continue;
                    }
                    Explore {
                        proof: self,
                        id: (position * 2 + wave) * 2 + part,
                        branch: b,
                        assume,
                        unreliable: HashMap::default(),
                        terms: Vec::new(),
                        types: Vec::new(),
                        leaves: Vec::new(),
                        index: HashMap::default(),
                    }
                    .run(wave, lane, &mut arrivals);
                }
            }
        }
        arrivals
    }
}

const WAVE: usize = 0;
const LANE: usize = 1;

#[derive(Clone, PartialEq, Eq, Hash)]
enum Form {
    Value(ValueId),
    Core(Ty, Op),
    Target(TargetOp, Vec<usize>, usize),
    Load(Space, MemSize, usize),
    Opaque(usize, ValueId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Desc {
    same: Option<usize>,
    bits: Option<Bdd>,
}

#[derive(Clone, PartialEq)]
struct Pair {
    cond: Bdd,
    sides: [Vec<Desc>; 2],
}

type Key = [Option<BlockId>; 2];

struct Explore<'p, 'a> {
    proof: &'p mut Proof<'a>,
    id: usize,
    branch: BlockId,
    assume: Bdd,
    unreliable: HashMap<u32, bool>,
    terms: Vec<Form>,
    types: Vec<Ty>,
    leaves: Vec<Bdd>,
    index: HashMap<Form, usize>,
}

impl Explore<'_, '_> {
    fn logic(&mut self) -> &mut Logic {
        self.proof.logic
    }

    fn fresh(&mut self, side: usize, v: ValueId) -> Bdd {
        let d = self.id * 2 + side;
        self.logic().atom(Atom::Fresh(d, v, 0))
    }

    fn constant_of(&self, t: usize) -> Option<u64> {
        match self.terms[t] {
            Form::Core(_, Op::Const(_, k)) => Some(k),
            _ => None,
        }
    }

    fn intern(&mut self, ty: Ty, form: Form) -> usize {
        let ones = |ty: Ty| {
            if ty == Ty::I64 {
                u64::MAX
            } else {
                (1u64 << ty.bits()) - 1
            }
        };
        match &form {
            Form::Core(_, Op::Convert(Cvt::Bitcast, _, a)) => {
                let a = a.0;
                if self.types[a] == ty {
                    return a;
                }
                if let Form::Core(_, Op::Convert(Cvt::Bitcast, _, b)) = self.terms[a] {
                    if self.types[b.0] == ty {
                        return b.0;
                    }
                }
            }
            Form::Core(_, Op::UnpackLo(p) | Op::UnpackHi(p)) => {
                let low = matches!(form, Form::Core(_, Op::UnpackLo(_)));
                let half = |e: &mut Self, x: ValueId| {
                    let op = if low {
                        Op::UnpackLo(x)
                    } else {
                        Op::UnpackHi(x)
                    };
                    e.intern(Ty::I32, Form::Core(Ty::I32, op))
                };
                match self.terms[p.0].clone() {
                    Form::Core(_, Op::Pack64(lo, hi)) => return if low { lo.0 } else { hi.0 },
                    Form::Core(_, Op::Const(_, k)) => {
                        let word = if low { k & 0xffff_ffff } else { k >> 32 };
                        return self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, word)));
                    }
                    Form::Core(_, Op::Convert(Cvt::ZExt, _, a)) if self.types[a.0] == Ty::I32 => {
                        return if low {
                            a.0
                        } else {
                            self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, 0)))
                        };
                    }
                    Form::Core(_, Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y)) => {
                        let (x, y) = (half(self, x), half(self, y));
                        return self.intern(
                            Ty::I32,
                            Form::Core(Ty::I32, Op::Int(k, ValueId(x), ValueId(y))),
                        );
                    }
                    Form::Core(_, Op::Int(k @ (IntOp::LShr | IntOp::Shl), x, s))
                        if self.constant_of(s.0) == Some(32) =>
                    {
                        let other = match (k, low) {
                            (IntOp::LShr, true) => Some(Op::UnpackHi(x)),
                            (IntOp::Shl, false) => Some(Op::UnpackLo(x)),
                            _ => None,
                        };
                        return match other {
                            Some(op) => self.intern(Ty::I32, Form::Core(Ty::I32, op)),
                            None => {
                                self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, 0)))
                            }
                        };
                    }
                    _ => {}
                }
            }
            Form::Core(_, Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y)) => {
                for (a, b) in [(x.0, y.0), (y.0, x.0)] {
                    match (k, self.constant_of(a)) {
                        (IntOp::And, Some(0)) => return a,
                        (IntOp::And, Some(c)) if c == ones(ty) => return b,
                        (IntOp::Or | IntOp::Xor, Some(0)) => return b,
                        _ => {}
                    }
                }
            }
            Form::Core(_, Op::Pack64(lo, hi)) => {
                if let (Form::Core(_, Op::UnpackLo(a)), Form::Core(_, Op::UnpackHi(b))) =
                    (&self.terms[lo.0], &self.terms[hi.0])
                {
                    if a == b {
                        return a.0;
                    }
                }
            }
            _ => {}
        }
        if let Some(&t) = self.index.get(&form) {
            return t;
        }
        let leaves = match &form {
            Form::Value(v) => self.proof.whole(*v),
            Form::Core(_, op) => {
                let mut h = Bdd::FALSE;
                let mut children = Vec::new();
                op.map(|c| {
                    children.push(c.0);
                    c
                });
                for c in children {
                    h = self.proof.logic.m.or(h, self.leaves[c]);
                }
                h
            }
            Form::Target(_, args, _) => {
                let mut h = Bdd::FALSE;
                for &c in args {
                    h = self.proof.logic.m.or(h, self.leaves[c]);
                }
                h
            }
            Form::Load(_, _, a) => self.leaves[*a],
            Form::Opaque(..) => Bdd::TRUE,
        };
        let t = self.terms.len();
        self.terms.push(form.clone());
        self.types.push(ty);
        self.leaves.push(leaves);
        self.index.insert(form, t);
        t
    }

    fn operand(&mut self, side: usize, descs: &HashMap<ValueId, Desc>, v: ValueId) -> usize {
        match descs[&v].same {
            Some(t) => t,
            None => {
                let ty = self.proof.f.types[v.0];
                self.intern(ty, Form::Opaque(side, v))
            }
        }
    }

    fn reliable(&mut self, t: usize) -> bool {
        let assume = self.proof.and(self.assume, self.proof.safe);
        let leaves = self.leaves[t];
        self.logic().m.and(assume, leaves) == Bdd::FALSE
    }

    fn form_bits(&mut self, side: usize, v: ValueId, t: usize, view: bool) -> Bdd {
        if self.reliable(t) {
            let id = self.id;
            self.logic().atom(Atom::Term(id, t, view))
        } else {
            self.fresh(side, v)
        }
    }

    fn is_unreliable(&mut self, var: u32) -> bool {
        if let Some(&u) = self.unreliable.get(&var) {
            return u;
        }
        let u = match self.proof.logic.atom_of(var) {
            Atom::Fresh(..) | Atom::Term(..) | Atom::Constant(_) | Atom::Marker(_) => false,
            Atom::Bit(v) | Atom::View(v) => {
                let h = self.proof.h[v.0];
                let assume = self.assume;
                self.logic().m.and(assume, h) != Bdd::FALSE
            }
        };
        self.unreliable.insert(var, u);
        u
    }

    fn unknowns(&mut self, g: Bdd, side: usize, forms: bool) -> Vec<u32> {
        let support = self.logic().support(g);
        support
            .into_iter()
            .filter(|&v| match self.proof.logic.atom_of(v) {
                Atom::Fresh(..) => true,
                Atom::Term(..) => forms,
                _ => side == WAVE && self.is_unreliable(v),
            })
            .collect()
    }

    fn decide(&mut self, g: Bdd, cond: Bdd, side: usize) -> Option<bool> {
        let cond = self.proof.and(cond, self.proof.safe);
        if let Some(k) = g.constant() {
            return Some(k);
        }
        let unknown = self.unknowns(g, side, true);
        let m = &mut self.proof.logic.m;
        let holds = m.forall(g, &|v| unknown.contains(&v));
        if m.implies(cond, holds) {
            return Some(true);
        }
        let ng = m.not(g);
        let fails = m.forall(ng, &|v| unknown.contains(&v));
        if m.implies(cond, fails) {
            return Some(false);
        }
        None
    }

    fn weaken(&mut self, g: Bdd, side: usize) -> Bdd {
        let unknown = self.unknowns(g, side, false);
        self.logic().m.exists(g, &|v| unknown.contains(&v))
    }

    fn start(&mut self, v: ValueId) -> Desc {
        let (f, facts) = (self.proof.f, self.proof.facts);
        let bits = match f.types[v.0] {
            Ty::I1 => Some(self.proof.logic.bit(f, facts, v)),
            Ty::I32 => Some(self.proof.logic.view(f, facts, v)),
            _ => None,
        };
        let same = Some(self.intern(f.types[v.0], Form::Value(v)));
        Desc { same, bits }
    }

    fn canon(&mut self, side: usize, param: ValueId, g: Bdd) -> Bdd {
        let support = self.logic().support(g);
        let fresh: Vec<u32> = support
            .into_iter()
            .filter(|&v| matches!(self.proof.logic.atom_of(v), Atom::Fresh(..)))
            .collect();
        if fresh.is_empty() {
            return g;
        }
        let d = self.id * 2 + side;
        let renamed: Vec<Bdd> = (0..fresh.len())
            .map(|i| self.logic().atom(Atom::Fresh(d, param, i as u32 + 1)))
            .collect();
        self.logic().m.compose(g, &|v| {
            fresh.iter().position(|&x| x == v).map(|i| renamed[i])
        })
    }

    fn merge(&mut self, side: usize, param: ValueId, cond: Bdd, old: Desc, new: Desc) -> Desc {
        let same = if old.same == new.same { old.same } else { None };
        let bits = match (old.bits, new.bits) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(a), Some(b)) => match (self.decide(a, cond, side), self.decide(b, cond, side)) {
                (Some(x), Some(y)) if x == y => Some(Manager::constant(x)),
                _ => {
                    let d = self.id * 2 + side;
                    Some(self.logic().atom(Atom::Fresh(d, param, 0)))
                }
            },
            _ => None,
        };
        Desc { same, bits }
    }

    fn run(
        &mut self,
        wave: usize,
        lane: usize,
        arrivals: &mut BTreeMap<BlockId, Vec<Bdd>>,
    ) {
        let f = self.proof.f;
        let branch = &f.blocks[&self.branch];
        let edges = [
            branch.term.edges().nth(wave).unwrap().clone(),
            branch.term.edges().nth(lane).unwrap().clone(),
        ];
        let sides = [
            edges[0].args.iter().map(|&a| self.start(a)).collect(),
            edges[1].args.iter().map(|&a| self.start(a)).collect(),
        ];
        let first: Key = [Some(edges[0].dst), Some(edges[1].dst)];
        let mut pairs: BTreeMap<Key, Pair> = BTreeMap::from([(
            first,
            Pair {
                cond: self.assume,
                sides,
            },
        )]);
        let mut worklist = vec![first];
        while let Some(key) = worklist.pop() {
            let mut pair = pairs[&key].clone();
            pair.cond = self.proof.and(pair.cond, self.proof.safe);
            if pair.cond == Bdd::FALSE {
                continue;
            }
            let side = match key {
                [None, None] => continue,
                [Some(a), Some(b)] if a == b => {
                    self.meet(a, &pair, arrivals);
                    continue;
                }
                [Some(a), Some(b)] => {
                    let (ra, rb) = (self.proof.rank[&a], self.proof.rank[&b]);
                    if self.proof.loops.before(ra, rb) {
                        WAVE
                    } else {
                        LANE
                    }
                }
                [Some(_), None] => WAVE,
                [None, Some(_)] => LANE,
            };
            let block = key[side].unwrap();
            for (dst, descs, constraint) in self.step(side, block, &pair.sides[side], pair.cond) {
                let cond = self.logic().m.and(pair.cond, constraint);
                let cond = self.proof.and(cond, self.proof.safe);
                if cond == Bdd::FALSE {
                    continue;
                }
                let mut next = key;
                next[side] = dst;
                let mut sides = pair.sides.clone();
                sides[side] = descs;
                let incoming = Pair { cond, sides };
                let merged = match pairs.get(&next) {
                    None => incoming,
                    Some(old) => {
                        let cond = self.logic().m.or(old.cond, incoming.cond);
                        let mut sides = old.sides.clone();
                        for s in [WAVE, LANE] {
                            let Some(block) = next[s] else { continue };
                            for (k, &(param, _)) in f.blocks[&block].params.iter().enumerate() {
                                sides[s][k] = self.merge(
                                    s,
                                    param,
                                    cond,
                                    old.sides[s][k],
                                    incoming.sides[s][k],
                                );
                            }
                        }
                        let merged = Pair { cond, sides };
                        if &merged == old {
                            continue;
                        }
                        merged
                    }
                };
                pairs.insert(next, merged);
                if !worklist.contains(&next) {
                    worklist.push(next);
                }
            }
        }
    }

    fn step(
        &mut self,
        side: usize,
        x: BlockId,
        params: &[Desc],
        cond: Bdd,
    ) -> Vec<(Option<BlockId>, Vec<Desc>, Bdd)> {
        let f = self.proof.f;
        let descs = self.evaluate(side, x, params, cond);
        let block = &f.blocks[&x];
        let mut out = Vec::new();
        let followed: Vec<(usize, Bdd)> = match &block.term {
            Term::Ret(_) => return vec![(None, vec![], Bdd::TRUE)],
            Term::Br(_) => vec![(0, Bdd::TRUE)],
            Term::CondBr { cond: c, .. } => {
                let g = match descs[c].bits {
                    Some(g) => g,
                    None => self.fresh(side, *c),
                };
                match self.decide(g, cond, side) {
                    Some(true) => vec![(0, Bdd::TRUE)],
                    Some(false) => vec![(1, Bdd::TRUE)],
                    None => {
                        let ng = self.logic().m.not(g);
                        vec![(0, self.weaken(g, side)), (1, self.weaken(ng, side))]
                    }
                }
            }
        };
        for (slot, constraint) in followed {
            let edge = block.term.edges().nth(slot).unwrap();
            let dst = &f.blocks[&edge.dst];
            let mut args = Vec::with_capacity(edge.args.len());
            for (&a, &(param, _)) in edge.args.iter().zip(&dst.params) {
                let mut d = descs[&a];
                if let Some(g) = d.bits {
                    d.bits = Some(self.canon(side, param, g));
                }
                args.push(d);
            }
            out.push((Some(edge.dst), args, constraint));
        }
        out
    }

    fn evaluate(
        &mut self,
        side: usize,
        x: BlockId,
        params: &[Desc],
        cond: Bdd,
    ) -> HashMap<ValueId, Desc> {
        let (f, facts) = (self.proof.f, self.proof.facts);
        let block = &f.blocks[&x];
        let mut descs: HashMap<ValueId, Desc> = HashMap::default();
        for (&(v, _), &d) in block.params.iter().zip(params) {
            descs.insert(v, d);
        }
        let bits_of =
            |e: &mut Self, descs: &HashMap<ValueId, Desc>, v: ValueId| match descs[&v].bits {
                Some(g) => g,
                None => e.fresh(side, v),
            };
        let formed = |e: &mut Self, v: ValueId, t: Option<usize>| -> Desc {
            let ty = f.types[v.0];
            let bits = match (ty, t) {
                (Ty::I1 | Ty::I32, Some(t)) => Some(e.form_bits(side, v, t, ty == Ty::I32)),
                (Ty::I1 | Ty::I32, None) => Some(e.fresh(side, v)),
                _ => None,
            };
            Desc { same: t, bits }
        };
        for (index, inst) in block.insts.iter().enumerate() {

            let collective = match inst {
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    outputs,
                    ..
                } => {
                    let local = self.logic().local(outputs[0].0);
                    self.logic().m.not(local)
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot),
                    outputs,
                    ..
                } => self.logic().materialized(facts, outputs[0].0),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadFirstLane),
                    inputs,
                    ..
                } => Manager::constant(!facts.uniform[inputs[0].0]),
                Inst::Effect {
                    op: EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait,
                    ..
                } => Bdd::TRUE,
                _ => Bdd::FALSE,
            };
            if collective != Bdd::FALSE {
                let differs = self.proof.and(cond, collective);
                self.proof.require(
                    x,
                    index,
                    "a retained collective may have different participating lanes",
                    differs,
                );
            }
            match inst {
                Inst::Core { value, ty, op } => {
                    let mut children = Vec::new();
                    op.map(|c| {
                        children.push(c);
                        c
                    });
                    let form = |e: &mut Self, descs: &HashMap<ValueId, Desc>| -> Option<usize> {
                        let mut forms = Vec::new();
                        for &c in &children {
                            forms.push(e.operand(side, descs, c));
                        }
                        let mut next = forms.into_iter();
                        let mapped = op.map(|_| ValueId(next.next().unwrap()));
                        Some(e.intern(*ty, Form::Core(*ty, mapped)))
                    };
                    let desc = match *op {
                        Op::Const(_, k) => Desc {
                            same: form(self, &descs),
                            bits: match ty {
                                Ty::I1 => Some(Manager::constant(k != 0)),
                                Ty::I32 => Some(match k as u32 {
                                    0 => Bdd::FALSE,
                                    u32::MAX => Bdd::TRUE,
                                    k => self.logic().atom(Atom::Constant(k)),
                                }),
                                _ => None,
                            },
                        },
                        Op::Env(Env::ValidLane) => Desc {
                            same: form(self, &descs),
                            bits: Some(Bdd::TRUE),
                        },
                        Op::Select(c, a, b) => {
                            let gc = bits_of(self, &descs, c);
                            match self.decide(gc, cond, side) {
                                Some(true) => descs[&a],
                                Some(false) => descs[&b],
                                None if descs[&a] == descs[&b] => descs[&a],
                                None => {
                                    let same = form(self, &descs);
                                    let bits = match (descs[&a].bits, descs[&b].bits) {
                                        (Some(ga), Some(gb)) => {
                                            Some(self.logic().m.ite(gc, ga, gb))
                                        }
                                        _ => formed(self, *value, same).bits,
                                    };
                                    Desc { same, bits }
                                }
                            }
                        }
                        Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b)
                            if matches!(ty, Ty::I1 | Ty::I32) =>
                        {
                            let ga = bits_of(self, &descs, a);
                            let gb = bits_of(self, &descs, b);
                            let m = &mut self.proof.logic.m;
                            let g = match k {
                                IntOp::And => m.and(ga, gb),
                                IntOp::Or => m.or(ga, gb),
                                _ => m.xor(ga, gb),
                            };
                            Desc {
                                same: form(self, &descs),
                                bits: Some(g),
                            }
                        }
                        Op::Convert(Cvt::Bitcast, to, a) if f.types[a.0] == to => descs[&a],
                        Op::Convert(Cvt::Trunc, Ty::I1, s) if matches!(facts.op(f, s), Some(Op::Int(IntOp::LShr, _, lane)) if facts.is_lane_id(f, lane)) =>
                        {
                            let Some(Op::Int(IntOp::LShr, w, _)) = facts.op(f, s) else {
                                unreachable!()
                            };
                            Desc {
                                same: form(self, &descs),
                                bits: Some(bits_of(self, &descs, w)),
                            }
                        }
                        Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b)
                            if f.types[a.0] == Ty::I1 =>
                        {
                            let ga = bits_of(self, &descs, a);
                            let gb = bits_of(self, &descs, b);
                            let m = &mut self.proof.logic.m;
                            let g = if p == IntPred::Ne {
                                m.xor(ga, gb)
                            } else {
                                m.iff(ga, gb)
                            };
                            Desc {
                                same: form(self, &descs),
                                bits: Some(g),
                            }
                        }
                        Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b)
                            if lane_test(f, facts, a, b)
                                .is_some_and(|w| !facts.materialized[w.0]) =>
                        {
                            let w = lane_test(f, facts, a, b).unwrap();
                            let bit = bits_of(self, &descs, w);
                            let set = self.answer(side, *value, bit, cond);
                            let g = if p == IntPred::Ne {
                                set
                            } else {
                                self.logic().m.not(set)
                            };
                            let mode = self.logic().materialized(facts, w);
                            let g = if mode == Bdd::FALSE {
                                g
                            } else {
                                let t = form(self, &descs);
                                let full = formed(self, *value, t).bits.unwrap();
                                self.logic().m.ite(mode, full, g)
                            };
                            Desc {
                                same: None,
                                bits: Some(g),
                            }
                        }
                        _ => {
                            let same = form(self, &descs);
                            formed(self, *value, same)
                        }
                    };
                    descs.insert(*value, desc);
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    inputs,
                    outputs,
                    ..
                } => {
                    let local = self.logic().local(outputs[0].0);
                    let bit = bits_of(self, &descs, inputs[0]);
                    let answer = self.answer(side, outputs[0].0, bit, cond);
                    let wave = self.fresh(side, outputs[0].0);
                    let g = self.logic().m.ite(local, answer, wave);
                    descs.insert(
                        outputs[0].0,
                        Desc {
                            same: None,
                            bits: Some(g),
                        },
                    );
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot),
                    inputs,
                    outputs,
                    ..
                } => {
                    let bit = bits_of(self, &descs, inputs[0]);
                    descs.insert(
                        outputs[0].0,
                        Desc {
                            same: None,
                            bits: Some(bit),
                        },
                    );
                }
                Inst::Effect {
                    op:
                        EffectOp::Memory {
                            space,
                            op: MemoryOp::Load(size),
                            ..
                        },
                    inputs,
                    outputs,
                    ..
                } => {
                    let pred = bits_of(self, &descs, inputs[1]);
                    let performed = self.decide(pred, cond, side) == Some(true);
                    let same = if performed {
                        let a = self.operand(side, &descs, inputs[0]);
                        let ty = outputs[0].1;
                        Some(self.intern(ty, Form::Load(*space, *size, a)))
                    } else {
                        None
                    };
                    let d = formed(self, outputs[0].0, same);
                    descs.insert(outputs[0].0, d);
                }
                Inst::Effect {
                    op:
                        EffectOp::Memory {
                            op:
                                memory @ (MemoryOp::Store(_)
                                | MemoryOp::AtomicAdd(_)
                                | MemoryOp::AtomicRmw(_)
                                | MemoryOp::AtomicCmpSwap),
                            ..
                        },
                    inputs,
                    outputs,
                    ..
                } => {
                    let pred = bits_of(self, &descs, inputs[memory.mask_input()]);
                    if self.decide(pred, cond, side) != Some(false) {
                        self.proof.require(
                            x,
                            index,
                            if side == WAVE {
                                "the wave program may store while the programs are apart"
                            } else {
                                "the lane program may store while the programs are apart"
                            },
                            cond,
                        );
                    }
                    for &(v, _) in outputs {
                        let d = formed(self, v, None);
                        descs.insert(v, d);
                    }
                }
                Inst::Target {
                    op, args, outputs, ..
                } => {
                    let terms: Vec<usize> = args
                        .values()
                        .iter()
                        .map(|&a| self.operand(side, &descs, a))
                        .collect();
                    for (i, &(v, ty)) in outputs.iter().enumerate() {
                        let same = Some(self.intern(ty, Form::Target(*op, terms.clone(), i)));
                        let d = formed(self, v, same);
                        descs.insert(v, d);
                    }
                }
                _ => {
                    for v in inst.outputs() {
                        let d = formed(self, v, None);
                        descs.insert(v, d);
                    }
                }
            }
        }
        descs
    }

    fn answer(&mut self, side: usize, v: ValueId, bit: Bdd, cond: Bdd) -> Bdd {
        if side == LANE {
            return bit;
        }
        if self.decide(bit, cond, side) == Some(true) {
            return Bdd::TRUE;
        }
        let others = self.fresh(side, v);
        self.logic().m.or(bit, others)
    }

    fn meet(&mut self, block: BlockId, pair: &Pair, arrivals: &mut BTreeMap<BlockId, Vec<Bdd>>) {
        let (f, facts) = (self.proof.f, self.proof.facts);
        let dst = &f.blocks[&block];
        let cond = pair.cond;
        let mut relation = Bdd::TRUE;
        for (k, &(param, ty)) in dst.params.iter().enumerate() {
            let atom = match ty {
                Ty::I1 => Atom::Bit(param),
                Ty::I32 if facts.viewed[param.0] => Atom::View(param),
                _ => continue,
            };
            if let Some(g) = pair.sides[LANE][k].bits {
                let a = self.logic().atom(atom);
                let link = self.logic().m.iff(a, g);
                relation = self.logic().m.and(relation, link);
            }
        }
        let mut contributions = Vec::with_capacity(dst.params.len());
        for (k, &(param, ty)) in dst.params.iter().enumerate() {
            let (w, l) = (pair.sides[WAVE][k], pair.sides[LANE][k]);
            let boolean =
                ty == Ty::I1 || (facts.lane_word[param.0] && !facts.materialized[param.0]);
            let differs = if w.same.is_some() && w.same == l.same {
                let leaves = self.leaves[w.same.unwrap()];
                self.logic().m.and(cond, leaves)
            } else if let (true, Some(gw), Some(gl)) = (boolean, w.bits, l.bits) {
                let equal = self.logic().m.iff(gw, gl);
                if self.decide(equal, cond, WAVE) == Some(true) {
                    let mut support = self.logic().support(gw);
                    support.extend(self.logic().support(gl));
                    let mut hs = Bdd::FALSE;
                    for var in support {
                        if let Atom::Bit(v) | Atom::View(v) = self.proof.logic.atom_of(var) {
                            let hv = self.proof.h[v.0];
                            hs = self.logic().m.or(hs, hv);
                        }
                    }
                    self.logic().m.and(cond, hs)
                } else {
                    cond
                }
            } else {
                cond
            };
            let differs = if ty == Ty::I32
                && facts.lane_word[param.0]
                && !(w.same.is_some() && w.same == l.same)
            {
                let mode = self.logic().materialized(facts, param);
                let full = self.logic().m.and(cond, mode);
                self.logic().m.or(differs, full)
            } else {
                differs
            };
            contributions.push(differs);
        }
        let entry = arrivals
            .entry(block)
            .or_insert_with(|| vec![Bdd::FALSE; dst.params.len()]);
        for (k, c) in contributions.into_iter().enumerate() {
            if c == Bdd::FALSE {
                continue;
            }
            let joint = self.logic().m.and(c, relation);
            let logic = &mut *self.proof.logic;

            let foreign: Vec<u32> = logic
                .support(joint)
                .into_iter()
                .filter(|&v| {
                    !matches!(logic.atom_of(v), Atom::Marker(_))
                        && logic.scope(facts, v) != Some(block)
                })
                .collect();
            let restated = logic.m.exists(joint, &|v| foreign.contains(&v));
            entry[k] = self.proof.logic.m.or(entry[k], restated);
        }
    }
}
