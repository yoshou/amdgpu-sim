use super::form::*;
use super::queries::*;
use super::HashSet;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct Origins {
    coarse: HashMap<ValueKey, Assumed<Regions>>,
    refined: HashMap<ValueKey, Assumed<Regions>>,
    active: HashSet<(ValueId, u8, Option<ValueId>, bool)>,
    looped: BTreeMap<(ValueId, u8, bool), usize>,
    checked: BTreeSet<(ValueId, u8, bool)>,
    grown: HashMap<(ValueId, u8, bool), Regions>,
    split: HashSet<(ValueId, bool)>,
}

impl Origins {
    pub(super) fn enter(&mut self) {
        self.coarse.clear();
        self.refined.clear();
        self.looped.clear();
        self.checked.clear();
    }

    pub(super) fn regions<'a, Q: Queries<'a>>(&mut self, q: &mut Q, v: ValueId, lane: usize, assume: Option<ValueId>, refine: bool) -> Regions {
        self.assumed_regions(q, v, lane, assume, refine).0
    }

    pub(super) fn read_regions<'a, Q: Queries<'a>>(&mut self, q: &mut Q, at: (BlockId, usize), lane: usize, exec: Option<ValueId>, refine: bool) -> Regions {
        let mut set = Regions::one(None);
        if let Inst::Target { args, .. } = &q.program().f.blocks[&at.0].insts[at.1] {
            for &x in args.values() {
                match q.program().provenance.known[x.0] {
                    Some(r) => set.add(r),
                    None => set.union(&self.regions(q, x, lane, exec, refine)),
                }
            }
        }
        set
    }

    fn assumed_regions<'a, Q: Queries<'a>>(&mut self, q: &mut Q, v: ValueId, lane: usize, assume: Option<ValueId>, refine: bool) -> Assumed<Regions> {
        if let Some(r) = q.program().provenance.known[v.0] {
            return unassumed(Regions::one(r));
        }
        let lane = q.symbols().canonical(v, lane);
        let l = lane as u8;
        let cache = if refine { &self.refined } else { &self.coarse };
        let open = |e: &Assumed<Regions>| assume.is_some() && e.1.open;
        let hit = match (cache.get(&(v, ALL, None)), cache.get(&(v, l, None))) {
            (Some(e), _) if !open(e) => Some(e.clone()),
            (_, Some(e)) if !open(e) => Some(e.clone()),
            _ => assume.and_then(|_| cache.get(&(v, ALL, assume)).or_else(|| cache.get(&(v, l, assume))).cloned()),
        };
        if let Some(r) = hit {
            return r;
        }
        let key = (v, l, assume, refine);
        if !self.active.insert(key) {
            return in_lane(Regions::any());
        }
        let mut reliance = Reliance::default();
        let set = self.compute_regions(q, v, lane, assume, refine, &mut reliance);
        self.active.remove(&key);
        let r = (set, reliance);
        let cache = if refine { &mut self.refined } else { &mut self.coarse };
        let lanes = if reliance.lane { l } else { ALL };
        if assume.is_some() {
            cache.insert((v, lanes, assume), r.clone());
        }
        if !reliance.used {
            cache.insert((v, lanes, None), r.clone());
        }
        r
    }

    fn compute_regions<'a, Q: Queries<'a>>(
        &mut self,
        q: &mut Q,
        v: ValueId,
        lane: usize,
        assume: Option<ValueId>,
        refine: bool,
        reliance: &mut Reliance,
    ) -> Regions {
        macro_rules! regions {
            ($x:expr) => {{
                let (r, u) = self.assumed_regions(q, $x, lane, assume, refine);
                *reliance |= u;
                r
            }};
        }
        let none = || Regions::one(None);
        if let Some(&root) = q.program().copies.get(&v) {
            return regions!(root);
        }
        let (f, facts) = (q.program().f, q.program().facts);
        match facts.site[v.0] {
            Site::Param { block, index } if block == f.entry => match q.program().inputs[index].source {
                ParameterSource::Vgpr(n) if n != 0 => Regions::default(),
                ParameterSource::Sgpr(n) if Some(n) == q.program().entry.kernarg_ptr => Regions::one(Some(Region::Kernarg)),
                ParameterSource::Sgpr(n) if Some(n) == q.program().entry.dispatch_ptr => Regions::one(Some(Region::Dispatch)),
                _ => none(),
            },
            Site::Param { block, index } => {
                let header = q.program().headers.contains(&block);
                let own = q.program().rank[&block];
                let mut set = Regions::default();
                for &e in &facts.incoming[&block] {
                    if header && q.program().rank[&e.0] >= own {
                        continue;
                    }
                    let (r, u) = self.assumed_regions(q, q.program().edge_arg(e, index), lane, None, refine);
                    reliance.lane |= u.lane;
                    set.union(&r);
                }
                if q.program().provenance.plain[v.0] {
                    set.add(None);
                }
                if q.program().provenance.carried.contains_key(&v) {
                    self.carry(q, v, lane, refine, &mut set, reliance);
                }
                set
            }
            Site::Inst { block, index } => match &f.blocks[&block].insts[index] {
                Inst::Core { op, .. } => match *op {
                    Op::Int(IntOp::Add, a, b) => regions!(a).combine(&regions!(b), |x, y| match (x, y) {
                        (Some(r), None) | (None, Some(r)) => Some(r),
                        _ => None,
                    }),
                    Op::Int(IntOp::Sub, a, b) => regions!(a).combine(&regions!(b), |x, y| match (x, y) {
                        (Some(r), None) => Some(r),
                        _ => None,
                    }),
                    Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a)
                        if to.bits() >= 32 && f.types[a.0].bits() >= 32 =>
                    {
                        regions!(a)
                    }
                    Op::Pack64(lo, _) | Op::UnpackLo(lo) => regions!(lo),
                    Op::UnpackHi(x) => match facts.op(f, x) {
                        Some(Op::Pack64(_, hi)) => regions!(hi),
                        _ => none(),
                    },
                    Op::Select(c, a, b) => {
                        let root = q.program().copies.get(&c).copied().unwrap_or(c);
                        if assume.is_some_and(|p| q.reason(|conditions, program| conditions.assumes(program, p, root))) {
                            reliance.used = true;
                            return regions!(a);
                        }
                        let (x, y) = (regions!(a), regions!(b));
                        if x.single().is_some() && x == y {
                            return x;
                        }
                        if refine {
                            let (bit, u) = q.bit(c, lane, assume);
                            *reliance |= u;
                            reliance.lane |= !facts.uniform[c.0];
                            match bit {
                                Some(true) => return x,
                                Some(false) => return y,
                                None => {}
                            }
                        }
                        reliance.open = true;
                        x.joined(&y)
                    }
                    Op::Int(IntOp::And, a, b) => {
                        let mask = |x: ValueId| match facts.op(f, x) {
                            Some(Op::Const(_, k)) => Some(k as u32),
                            _ => None,
                        };
                        match (mask(a), mask(b)) {
                            (None, Some(m)) if aligns(m) => regions!(a),
                            (Some(m), None) if aligns(m) => regions!(b),
                            (None, None) => {
                                let mut set = regions!(a).combine(&regions!(b), |x, y| match (x, y) {
                                    (Some(r), None) | (None, Some(r)) => Some(r),
                                    _ => None,
                                });
                                set.add(None);
                                set
                            }
                            _ => none(),
                        }
                    }
                    Op::Int(IntOp::Or | IntOp::Xor, a, b) => {
                        let (x, y) = (regions!(a), regions!(b));
                        if x.any || y.any {
                            return Regions::any();
                        }
                        let mut set = Regions::default();
                        for &r in x.list.iter().chain(&y.list) {
                            if r.is_some() {
                                set.add(r);
                            }
                        }
                        if x.list.contains(&None) && y.list.contains(&None) {
                            set.add(None);
                        }
                        set
                    }
                    Op::Int(IntOp::LShr, a, _) if f.types[v.0] != Ty::I64 => {
                        let mut set = regions!(a);
                        set.add(None);
                        set
                    }
                    Op::Env(Env::ScratchBase) => Regions::one(Some(Region::Private)),
                    _ => none(),
                },
                Inst::Effect {
                    op: EffectOp::Memory {
                        op: MemoryOp::Load(_),
                        space,
                        ..
                    },
                    inputs,
                    ..
                } => {
                    let address = regions!(inputs[0]);
                    let kernarg = address.any || address.list.contains(&Some(Region::Kernarg));
                    if *space == Space::Scratch || kernarg {
                        let (value, u) = q.value(v, lane, assume);
                        *reliance |= u;
                        reliance.lane |= !facts.uniform[v.0];
                        let mut set = Regions::one(value.region);
                        if (q.program().provenance.carried.contains_key(&v) || q.program().provenance.loaded.contains_key(&v)) && q.symbols().unresolved(v, lane, &value) {
                            match q.program().provenance.loaded.get(&v) {
                                Some(loaded) => set.union(loaded),
                                None => self.carry(q, v, lane, refine, &mut set, reliance),
                            }
                        }
                        set
                    } else {
                        none()
                    }
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadFirstLane),
                    inputs,
                    ..
                } => self.lanes_regions(q, inputs[0], false),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadLane),
                    inputs,
                    ..
                } => {
                    reliance.lane = true;
                    let last = q.program().lanes() as u32 - 1;
                    match q.value(inputs[1], lane, None).0.form.as_constant() {
                        Some(k) if !q.symbols().valid((k & last) as usize) => none(),
                        Some(k) => self.assumed_regions(q, inputs[0], (k & last) as usize, None, refine).0,
                        None => {
                            let partial = !(0..q.program().lanes()).all(|l| q.symbols().valid(l));
                            self.lanes_regions(q, inputs[0], partial)
                        }
                    }
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::WriteLane),
                    inputs,
                    ..
                } => {
                    reliance.lane = true;
                    let last = q.program().lanes() as u32 - 1;
                    match q.value(inputs[1], lane, None).0.form.as_constant() {
                        Some(k) if (k & last) as usize == lane => self.assumed_regions(q, inputs[0], lane, None, refine).0,
                        Some(_) => self.assumed_regions(q, inputs[2], lane, None, refine).0,
                        None => {
                            let mut set = self.assumed_regions(q, inputs[0], lane, None, refine).0;
                            set.union(&self.assumed_regions(q, inputs[2], lane, None, refine).0);
                            set
                        }
                    }
                }
                Inst::Effect {
                    op: EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)),
                    inputs,
                    ..
                } => {
                    reliance.lane = true;
                    let silent = *op == WaveOp::Bpermute || !(0..q.program().lanes()).all(|l| q.symbols().valid(l));
                    self.lanes_regions(q, inputs[1], silent)
                }
                _ => none(),
            },
            Site::Unreached => none(),
        }
    }

    fn carry<'a, Q: Queries<'a>>(&mut self, q: &mut Q, v: ValueId, lane: usize, refine: bool, set: &mut Regions, reliance: &mut Reliance) {
        for tier in [false, true] {
            if tier && !refine {
                continue;
            }
            if let Some(grown) = self.grown.get(&(v, ALL, tier)) {
                set.union(grown);
            }
            if self.split.contains(&(v, tier)) {
                reliance.lane = true;
                if let Some(grown) = self.grown.get(&(v, lane as u8, tier)) {
                    set.union(grown);
                }
            }
        }
        let bound = &q.program().provenance.carried[&v];
        if !bound[if bound.len() == 1 { 0 } else { lane }].within(set) {
            let key = (v, if reliance.lane { lane as u8 } else { ALL }, refine);
            self.looped.entry(key).or_insert(lane);
        }
    }

    fn stored_regions<'a, Q: Queries<'a>>(&mut self, q: &mut Q, v: ValueId, lane: usize, refine: bool, held: &Regions) -> Assumed<Regions> {
        let mut set = Regions::default();
        let Site::Inst { block, index } = q.program().facts.site[v.0] else {
            return (set, Reliance::default());
        };
        let Inst::Effect {
            op: EffectOp::Memory {
                op: MemoryOp::Load(size), ..
            },
            inputs,
            ..
        } = &q.program().f.blocks[&block].insts[index]
        else {
            return (set, Reliance::default());
        };
        let (address, mut reliance) = q.value(inputs[0], lane, Some(inputs[1]));
        reliance.lane |= !q.program().facts.uniform[inputs[0].0];
        let words = q.symbols().bounds(&address.form)
            .map(|(low, high)| ((low / 4) as u32, (high + size.bytes() as u64).div_ceil(4) as u32));
        let sources = q.program().sources.get(&v).cloned().unwrap_or_default();
        for i in sources {
            if let (Some((a, b)), Some((c, d))) = (words, q.program().provenance.spills[i].words) {
                if b <= c || d <= a {
                    continue;
                }
            }
            if q.program().provenance.spills[i].part(lane).within(held) {
                continue;
            }
            let mask = q.program().provenance.spills[i].mask;
            for k in 0..q.program().provenance.spills[i].data.len() {
                let (r, u) = self.assumed_regions(q, q.program().provenance.spills[i].data[k], lane, Some(mask), refine);
                reliance |= u;
                set.union(&r);
            }
        }
        (set, reliance)
    }

    fn feeding<'a, Q: Queries<'a>>(&mut self, q: &mut Q, v: ValueId, lane: usize, refine: bool, held: &Regions) -> Assumed<Regions> {
        match q.program().facts.site[v.0] {
            Site::Param { .. } => self.back_regions(q, v, lane, refine),
            _ => self.stored_regions(q, v, lane, refine, held),
        }
    }

    fn back_regions<'a, Q: Queries<'a>>(&mut self, q: &mut Q, v: ValueId, lane: usize, refine: bool) -> Assumed<Regions> {
        let mut set = Regions::default();
        let mut reliance = Reliance::default();
        let Site::Param { block, index } = q.program().facts.site[v.0] else {
            return (set, reliance);
        };
        let own = q.program().rank[&block];
        let back: Vec<(BlockId, usize)> = q.program().facts.incoming[&block]
            .iter()
            .copied()
            .filter(|&(pred, _)| q.program().rank[&pred] >= own)
            .collect();
        for e in back {
            let (r, u) = self.assumed_regions(q, q.program().edge_arg(e, index), lane, None, refine);
            reliance |= u;
            set.union(&r);
        }
        (set, reliance)
    }

    pub(super) fn settle_loops<'a, Q: Queries<'a>>(&mut self, q: &mut Q) -> bool {
        let mut grew = false;
        loop {
            let pending: Vec<((ValueId, u8, bool), usize)> = self
                .looped
                .iter()
                .filter(|(key, _)| !self.checked.contains(key))
                .map(|(&key, &lane)| (key, lane))
                .collect();
            if pending.is_empty() {
                break;
            }
            for ((v, l, refine), first) in pending {
                self.checked.insert((v, l, refine));
                let held = self.regions(q, v, first, None, refine);
                let bound = q.program().provenance.carried[&v].clone();
                let all: Vec<usize> = (0..q.program().lanes()).filter(|&k| q.symbols().valid(k)).collect();
                let (lanes, mut known) = match (l == ALL, bound.len() > 1) {
                    (true, true) => (all, None),
                    (true, false) => {
                        let (found, u) = self.feeding(q, v, first, false, &held);
                        (if u.lane { all } else { vec![first] }, Some((found, u)))
                    }
                    _ => (vec![first], None),
                };
                for lane in lanes {
                    let own = &bound[if bound.len() == 1 { 0 } else { lane }];
                    if own.within(&held) {
                        continue;
                    }
                    let (found, u) = if !own.any && own.list.iter().all(Option::is_none) {
                        (own.clone(), in_lane(()).1)
                    } else {
                        let cached = if lane == first { known.take() } else { None };
                        let coarse = match cached {
                            Some(coarse) => coarse,
                            None => self.feeding(q, v, lane, false, &held),
                        };
                        if coarse.0.within(&held) {
                            continue;
                        }
                        if refine {
                            self.feeding(q, v, lane, true, &held)
                        } else {
                            coarse
                        }
                    };
                    if found.within(&held) {
                        continue;
                    }
                    let key = if l == ALL && !u.lane {
                        (v, ALL, refine)
                    } else {
                        self.split.insert((v, refine));
                        (v, lane as u8, refine)
                    };
                    let grown = self.grown.entry(key).or_default();
                    if !found.within(grown) {
                        grown.union(&found);
                        grew = true;
                    }
                }
            }
        }
        if grew {
            self.coarse.clear();
            self.refined.clear();
            self.looped.clear();
            self.checked.clear();
        }
        grew
    }

    pub(super) fn pending(&self) -> BTreeSet<(ValueId, u8, bool)> {
        self.looped.keys().copied().collect()
    }

    pub(super) fn forget(&mut self, kept: &BTreeSet<(ValueId, u8, bool)>) {
        self.looped.retain(|key, _| kept.contains(key));
        self.coarse.clear();
        self.refined.clear();
    }

    pub(super) fn expose<'a, Q: Queries<'a>>(&mut self, q: &mut Q, open: &[u64], stake: &[u64]) -> Vec<u64> {
        let mut found: Vec<u64> = Vec::new();
        let lanes: Vec<usize> = (0..q.program().lanes()).filter(|&l| q.symbols().valid(l)).collect();
        for e in q.program().provenance.exposing.clone() {
            let mut fresh: Vec<u64> = e
                .candidates
                .iter()
                .copied()
                .filter(|id| stake.contains(id) && !open.contains(id) && !found.contains(id))
                .collect();
            for &lane in &lanes {
                for &x in &e.operands {
                    if fresh.is_empty() {
                        break;
                    }
                    let meets = |set: &Regions, id: u64| set.any || set.list.contains(&Some(Region::Allocation(id)));
                    let coarse = self.regions(q, x, lane, e.assume, false);
                    if !fresh.iter().any(|&id| meets(&coarse, id)) {
                        continue;
                    }
                    let refined = self.regions(q, x, lane, e.assume, true);
                    fresh.retain(|&id| {
                        let hit = meets(&refined, id);
                        if hit {
                            found.push(id);
                        }
                        !hit
                    });
                }
            }
        }
        found
    }

    fn lanes_regions<'a, Q: Queries<'a>>(&mut self, q: &mut Q, x: ValueId, silent: bool) -> Regions {
        let mut set = Regions::default();
        if silent {
            set.add(None);
        }
        for l in 0..q.program().lanes() {
            if q.symbols().valid(l) {
                set.union(&self.regions(q, x, l, None, false));
            }
        }
        set
    }
}
