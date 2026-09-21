use crate::rdna_spmd::analysis::bdd::Bdd;
use super::cost::Costs;
use super::mask::Masks;
use super::structure::{Structure, Unit};
use crate::rdna_spmd::ir::{DialectRegistry, Effect};
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Val {
    Value(ValueId),
    Bit(Bdd),
}

struct Arrival {
    mask: Bdd,

    args: Vec<Option<Val>>,
}

struct Frame {
    l: usize,
    header: BlockId,
    latches: Vec<Arrival>,
    exits: BTreeMap<BlockId, Vec<Arrival>>,
}

struct Leave {
    target: BlockId,

    carried: Vec<(usize, ValueId, Ty)>,
    list: Vec<Arrival>,

    held: Bdd,

    kept: Vec<Kept>,
}

enum Kept {

    Variable(usize),

    Slot(Val),
}

fn agreed(arrivals: &[Arrival], index: usize) -> Option<Val> {
    let first = arrivals.first()?.args[index]?;
    arrivals
        .iter()
        .all(|a| a.args[index] == Some(first))
        .then_some(first)
}

struct Span {
    merge: BlockId,
    guard: BlockId,
    mask: Bdd,
    values: usize,
    atoms: usize,
    knowledge: super::mask::Knowledge,
    pending: BTreeMap<BlockId, Vec<Arrival>>,
    latches: Vec<Arrival>,
    exits: BTreeMap<BlockId, Vec<Arrival>>,
}

pub struct Lowering {
    pub ir: Func,
    pub masks: Masks,
}

pub fn lower(
    q: &Func,
    s: &Structure,
    registry: &DialectRegistry,
    uniform: &[bool],
    exec: usize,
    costs: &Costs,
    everyone: &BTreeSet<u64>,
) -> Lowering {
    let mut e = Emit {
        q,
        s,
        registry,
        uniform,
        costs,
        out: Func::new(BlockId(0), crate::rdna_spmd::ir::Presence::Wave),
        cur: BlockId(0),
        masks: Masks::new(),
        lane: vec![None; q.types.len()],
        pending: BTreeMap::new(),
        frames: Vec::new(),
        scopes: vec![Scope::default()],
        blocks: 0,
        flat: HashMap::new(),
        at: q.entry,
        root: Bdd::TRUE,
        full: Vec::new(),
        wanted: Vec::new(),
        demanded: everyone,
        provenance: q
            .blocks
            .values()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match inst {
                Inst::Effect { provenance, .. } => Some(*provenance),
                Inst::Target { provenance, .. } => *provenance,
                _ => None,
            })
            .max()
            .map_or(0, |p| p + 1),
    };
    let entry = &q.blocks[&q.entry];
    let types: Vec<Ty> = entry.params.iter().map(|p| p.1).collect();
    let (block, params) = e.block(&types);
    e.out.entry = block;
    e.cur = block;
    let mut args = Vec::with_capacity(params.len());

    let valid = e.masks.atom(params[exec], false, None);
    e.masks.assume_a_lane(valid);
    e.root = valid;
    e.full.push(valid);
    let mask = valid;
    for (index, (&(p, ty), &value)) in entry.params.iter().zip(&params).enumerate() {
        let val = if ty == Ty::I1 {
            if index == exec {

                Val::Bit(Bdd::TRUE)
            } else {
                Val::Bit(e.masks.atom(value, uniform[p.0], Some(p)))
            }
        } else {
            Val::Value(value)
        };
        args.push(Some(val));
    }
    e.pending.insert(q.entry, vec![Arrival { mask, args }]);
    e.region(None);
    assert!(
        e.pending.is_empty(),
        "the lowering left lanes waiting at a block it never ran"
    );
    e.terminate(Term::Ret(vec![]));
    let mut ir = e.out;
    ir.one_region(ir.presence_needed());
    Lowering {
        ir,
        masks: e.masks,
    }
}

#[derive(Default)]
struct Scope {
    bits: HashMap<Bdd, ValueId>,
    queries: HashMap<Bdd, Bdd>,
    constants: HashMap<(Ty, u64), ValueId>,
}

struct Emit<'a> {
    q: &'a Func,
    s: &'a Structure,
    registry: &'a DialectRegistry,
    uniform: &'a [bool],
    costs: &'a Costs,
    out: Func,
    cur: BlockId,
    masks: Masks,
    lane: Vec<Option<Val>>,
    pending: BTreeMap<BlockId, Vec<Arrival>>,
    frames: Vec<Frame>,
    scopes: Vec<Scope>,
    blocks: usize,

    flat: HashMap<u64, Option<ValueId>>,

    provenance: u64,

    at: BlockId,

    root: Bdd,

    full: Vec<Bdd>,

    wanted: Vec<usize>,

    demanded: &'a BTreeSet<u64>,
}

impl Emit<'_> {

    fn block(&mut self, types: &[Ty]) -> (BlockId, Vec<ValueId>) {
        let id = BlockId(self.blocks);
        self.blocks += 1;
        let params: Vec<ValueId> = types.iter().map(|&ty| self.out.value(ty)).collect();
        self.out.blocks.insert(
            id,
            Block {
                params: params.iter().zip(types).map(|(&v, &ty)| (v, ty)).collect(),
                insts: Vec::new(),
                term: Term::Ret(vec![]),
            },
        );
        (id, params)
    }

    fn push(&mut self, inst: Inst) {
        self.out.blocks.get_mut(&self.cur).unwrap().insts.push(inst);
    }

    fn core(&mut self, ty: Ty, op: Op) -> ValueId {
        let value = self.out.value(ty);
        self.push(Inst::Core { value, ty, op });
        value
    }

    fn terminate(&mut self, term: Term) {
        self.out.blocks.get_mut(&self.cur).unwrap().term = term;
    }

    fn constant(&mut self, ty: Ty, bits: u64) -> ValueId {
        for scope in self.scopes.iter().rev() {
            if let Some(&value) = scope.constants.get(&(ty, bits)) {
                return value;
            }
        }
        let value = self.core(ty, Op::Const(ty, bits));
        self.scopes
            .last_mut()
            .unwrap()
            .constants
            .insert((ty, bits), value);
        value
    }

    fn constant_in(&mut self, block: BlockId, ty: Ty, bits: u64) -> ValueId {
        let value = self.out.value(ty);
        self.out
            .blocks
            .get_mut(&block)
            .unwrap()
            .insts
            .push(Inst::Core {
                value,
                ty,
                op: Op::Const(ty, bits),
            });
        value
    }

    fn materialize(&mut self, f: Bdd) -> ValueId {
        let f = self.masks.exact(f);
        self.build(f)
    }

    fn build(&mut self, f: Bdd) -> ValueId {
        if let Some(value) = f.constant() {
            return self.constant(Ty::I1, value as u64);
        }
        for scope in self.scopes.iter().rev() {
            if let Some(&value) = scope.bits.get(&f) {
                return value;
            }
        }
        let (var, low, high) = self.masks.m.decompose(f).expect("a non-constant mask");
        let x = self.masks.at(var).value;
        let value = match (low.constant(), high.constant()) {
            (Some(false), Some(true)) => x,
            (Some(true), Some(false)) => self.negate(x),
            (Some(false), None) => {
                let h = self.build(high);
                self.core(Ty::I1, Op::Int(IntOp::And, x, h))
            }
            (None, Some(false)) => {
                let nx = self.negate(x);
                let l = self.build(low);
                self.core(Ty::I1, Op::Int(IntOp::And, nx, l))
            }
            (None, Some(true)) => {
                let l = self.build(low);
                self.core(Ty::I1, Op::Int(IntOp::Or, x, l))
            }
            (Some(true), None) => {
                let nx = self.negate(x);
                let h = self.build(high);
                self.core(Ty::I1, Op::Int(IntOp::Or, nx, h))
            }
            _ => {
                let flipped = self.masks.not(high);
                if flipped == low {
                    let l = self.build(low);
                    self.core(Ty::I1, Op::Int(IntOp::Xor, x, l))
                } else {
                    let h = self.build(high);
                    let l = self.build(low);
                    self.core(Ty::I1, Op::Select(x, h, l))
                }
            }
        };
        self.scopes.last_mut().unwrap().bits.insert(f, value);
        value
    }

    fn building(&self, f: Bdd) -> usize {
        if f.constant().is_some() || self.scopes.iter().any(|s| s.bits.contains_key(&f)) {
            return 0;
        }
        let (_, low, high) = self.masks.m.decompose(f).unwrap();
        match (low.constant(), high.constant()) {
            (Some(false), Some(true)) => 0,
            (Some(true), Some(false)) => 1,
            (Some(_), None) => 1 + self.building(high),
            (None, Some(_)) => 1 + self.building(low),
            _ => 1 + self.building(low) + self.building(high),
        }
    }

    fn negate(&mut self, value: ValueId) -> ValueId {
        let one = self.constant(Ty::I1, 1);
        self.core(Ty::I1, Op::Int(IntOp::Xor, value, one))
    }

    fn query(&mut self, f: Bdd) -> Bdd {
        let f = self.masks.exact(f);
        if f.constant().is_some() {
            return f;
        }
        if self.masks.holds_a_lane(f) {
            return Bdd::TRUE;
        }
        if let Some(var) = self.uniform_var(f) {
            let high = self.masks.m.cofactor(f, var, true);
            let low = self.masks.m.cofactor(f, var, false);
            let high = self.query(high);
            let low = self.query(low);
            let test = self.masks.m.var(var);
            return self.masks.ite(test, high, low);
        }
        for scope in self.scopes.iter().rev() {
            if let Some(&answer) = scope.queries.get(&f) {
                return answer;
            }
        }
        let input = self.build(f);
        let output = self.out.value(Ty::I1);
        self.push(Inst::Packet {
            op: PacketOp::Any,
            input,
            output,
        });
        let answer = self.masks.atom(output, true, None);
        self.scopes.last_mut().unwrap().queries.insert(f, answer);
        answer
    }

    fn query_is_scalar(&mut self, f: Bdd) -> bool {
        let f = self.masks.exact(f);
        if f.constant().is_some() || self.masks.holds_a_lane(f) {
            return true;
        }
        match self.uniform_var(f) {
            Some(var) => {
                let high = self.masks.m.cofactor(f, var, true);
                let low = self.masks.m.cofactor(f, var, false);
                self.query_is_scalar(high) && self.query_is_scalar(low)
            }
            None => false,
        }
    }

    fn uniform_var(&mut self, f: Bdd) -> Option<u32> {
        self.masks
            .m
            .support(f)
            .into_iter()
            .find(|&var| self.masks.at(var).uniform)
    }

    fn val(&mut self, v: ValueId) -> Val {
        let v = self.s.resolve(v);
        self.lane[v.0].expect("a lane value the lowering has not reached")
    }

    fn value(&mut self, v: ValueId) -> ValueId {
        match self.val(v) {
            Val::Value(value) => value,
            Val::Bit(bit) => self.materialize(bit),
        }
    }

    fn bit(&mut self, v: ValueId) -> Bdd {
        match self.val(v) {
            Val::Bit(bit) => bit,
            Val::Value(value) => {
                let atom = self.masks.atom(value, self.uniform[v.0], Some(v));
                self.lane[self.s.resolve(v).0] = Some(Val::Bit(atom));
                atom
            }
        }
    }

    fn define(&mut self, v: ValueId, ty: Ty, value: ValueId) {
        let val = if ty == Ty::I1 {
            Val::Bit(self.masks.atom(value, self.uniform[v.0], Some(v)))
        } else {
            Val::Value(value)
        };
        self.lane[v.0] = Some(val);
    }

    fn zero(&mut self, ty: Ty) -> Val {
        if ty == Ty::I1 {
            Val::Bit(Bdd::FALSE)
        } else {
            Val::Value(self.constant(ty, 0))
        }
    }

    fn region(&mut self, l: Option<usize>) {
        let region = l.map_or(0, |l| l + 1);
        self.subtree(region, 0, false);
    }

    fn subtree(&mut self, region: usize, u: usize, skippable: bool) {
        let unit = self.s.regions[region].units[u];
        let header = match unit {
            Unit::Block(b) => b,
            Unit::Loop(c) => self.s.header(c),
        };
        let Some(arrivals) = self.pending.remove(&header) else {
            return;
        };
        let mask = self.union(&arrivals);
        if mask == Bdd::FALSE {
            return;
        }
        let span = skippable.then(|| self.decide(region, u, mask)).flatten();
        match unit {
            Unit::Block(b) => self.emit_block(b, arrivals, mask),
            Unit::Loop(c) => self.emit_loop(c, arrivals, mask),
        }
        for child in self.s.regions[region].children[u].clone() {
            self.subtree(region, child, true);
        }
        if let Some(span) = span {
            self.close(span);
        }
    }

    fn union(&mut self, arrivals: &[Arrival]) -> Bdd {
        let mut mask = Bdd::FALSE;
        for a in arrivals {
            mask = self.masks.or(mask, a.mask);
        }
        self.masks.exact(mask)
    }

    fn decide(&mut self, region: usize, u: usize, mask: Bdd) -> Option<Span> {
        if self.masks.holds_a_lane(mask) {
            return None;
        }

        if self.keeps_wave(self.s.regions[region].span[u].iter().copied()) {
            let full = *self.full.last().unwrap();
            let own = self.masks.m.support(full);
            return self.all_or_none(mask, &own).then(|| self.open(mask));
        }
        let deciding = if self.query_is_scalar(mask) {
            1
        } else {
            self.costs.lanes()
        };
        let empty: f64 = (1..=deciding)
            .map(|k| (2 * k - 1) as f64 / (2 * k) as f64)
            .product();
        let query = self.query_work(mask) + 1;
        let (work, merge) = self.span_work(region, u);
        (work as f64 > query as f64 / empty + merge as f64).then(|| self.open(mask))
    }

    fn query_work(&mut self, f: Bdd) -> u64 {
        let f = self.masks.exact(f);
        if f.constant().is_some() || self.masks.holds_a_lane(f) {
            return 0;
        }
        match self.uniform_var(f) {
            Some(var) => {
                let high = self.masks.m.cofactor(f, var, true);
                let low = self.masks.m.cofactor(f, var, false);
                1 + self.query_work(high) + self.query_work(low)
            }
            None => {
                let reduction = self.costs.registers(Ty::I1, false);
                self.building(f) as u64 * reduction + reduction
            }
        }
    }

    fn span_work(&self, region: usize, u: usize) -> (u64, u64) {
        let blocks = &self.s.regions[region].span[u];
        let inside: BTreeSet<BlockId> = blocks.iter().copied().collect();
        let mut work = 0;
        let mut merge = 0;
        for &b in blocks {
            let block = &self.q.blocks[&b];
            for inst in &block.insts {
                work += self.costs.instruction(inst, self.uniform);
            }
            for edge in block.term.edges() {
                if inside.contains(&edge.dst) {
                    continue;
                }
                merge += self.costs.registers(Ty::I1, false);
                for (&arg, &(param, ty)) in edge.args.iter().zip(&self.q.blocks[&edge.dst].params) {
                    if !self.s.live[param.0] || self.s.same[param.0].is_some() {
                        continue;
                    }
                    let source = self.s.resolve(arg);
                    if self
                        .s
                        .defined_in(source)
                        .is_some_and(|d| inside.contains(&d))
                    {
                        merge += self.costs.registers(ty, self.uniform[source.0]);
                    }
                }
            }
        }
        (work, merge)
    }

    fn open(&mut self, mask: Bdd) -> Span {
        let answer = self.query(mask);
        let cond = self.materialize(answer);
        let (run, _) = self.block(&[]);
        let (merge, _) = self.block(&[]);
        let guard = self.cur;
        self.terminate(Term::CondBr {
            cond,
            yes: Edge {
                dst: run,
                args: Vec::new(),
            },
            no: Edge {
                dst: merge,
                args: Vec::new(),
            },
        });
        let span = Span {
            merge,
            guard,
            mask,
            values: self.out.types.len(),
            atoms: self.masks.atoms().len(),
            knowledge: self.masks.knowledge(),
            pending: std::mem::take(&mut self.pending),
            latches: self
                .frames
                .last_mut()
                .map_or_else(Vec::new, |f| std::mem::take(&mut f.latches)),
            exits: self
                .frames
                .last_mut()
                .map_or_else(BTreeMap::new, |f| std::mem::take(&mut f.exits)),
        };
        self.cur = run;
        self.scopes.push(Scope::default());
        self.masks.assume_a_lane(mask);

        let shared = self.shared_part(mask);
        self.masks.assume(shared);
        span
    }

    fn shared_part(&mut self, f: Bdd) -> Bdd {
        let f = self.masks.exact(f);
        let varying: Vec<u32> = self
            .masks
            .m
            .support(f)
            .into_iter()
            .filter(|&var| !self.masks.at(var).uniform)
            .collect();
        self.masks.m.exists(f, &|var| varying.contains(&var))
    }

    fn close(&mut self, span: Span) {
        let Span {
            merge,
            guard,
            mask,
            values,
            atoms,
            knowledge,
            pending,
            latches,
            exits,
        } = span;
        let mut inside: Vec<(Route, BlockId, Vec<Arrival>)> = std::mem::take(&mut self.pending)
            .into_iter()
            .map(|(target, list)| (Route::Pending, target, list))
            .collect();
        if let Some(frame) = self.frames.last_mut() {
            let header = frame.header;
            let latched = std::mem::take(&mut frame.latches);
            let left = std::mem::take(&mut frame.exits);
            inside.push((Route::Latch, header, latched));
            inside.extend(left.into_iter().map(|(t, list)| (Route::Exit, t, list)));
        }

        let mut crossing = Crossing::default();
        for (_, _, list) in inside.iter_mut() {
            for arrival in list.iter_mut() {
                if self.inner(arrival.mask, atoms) {
                    let uniform = self.masks.uniform(arrival.mask);
                    let value = self.materialize(arrival.mask);
                    crossing.add(value, Ty::I1, uniform);
                    arrival.mask = self.masks.atom(value, uniform, None);
                }
                for arg in arrival.args.iter_mut() {
                    match *arg {
                        Some(Val::Value(value)) if value.0 >= values => {
                            let ty = self.out.types[value.0];
                            let uniform = self.uniform_value(value);
                            crossing.add(value, ty, uniform);
                        }
                        Some(Val::Bit(bit)) if self.inner(bit, atoms) => {
                            let uniform = self.masks.uniform(bit);
                            let value = self.materialize(bit);
                            crossing.add(value, Ty::I1, uniform);
                            *arg = Some(Val::Value(value));
                        }
                        _ => {}
                    }
                }
            }
        }
        let types: Vec<Ty> = crossing.entries.iter().map(|c| c.1).collect();
        let params = self.add_params(merge, &types);
        let run: Vec<ValueId> = crossing.entries.iter().map(|c| c.0).collect();
        self.terminate(Term::Br(Edge {
            dst: merge,
            args: run,
        }));
        let skipped: Vec<ValueId> = types
            .iter()
            .map(|&ty| self.constant_in(guard, ty, 0))
            .collect();
        match &mut self.out.blocks.get_mut(&guard).unwrap().term {
            Term::CondBr { no, .. } => no.args = skipped,
            _ => unreachable!("a span is opened by a conditional branch"),
        }
        self.scopes.pop();
        self.masks.restore(knowledge);
        self.cur = merge;
        let mut renamed: BTreeMap<ValueId, Val> = BTreeMap::new();
        let mut atoms_of: BTreeMap<u32, Bdd> = BTreeMap::new();
        for (&(inner, ty, uniform), &param) in crossing.entries.iter().zip(&params) {
            if ty == Ty::I1 {
                let atom = self.masks.atom(param, uniform, None);
                renamed.insert(inner, Val::Bit(atom));
            } else {
                renamed.insert(inner, Val::Value(param));
            }
        }
        for atom in atoms..self.masks.atoms().len() {
            let value = self.masks.at(atom as u32).value;
            if let Some(&Val::Bit(bdd)) = renamed.get(&value) {
                atoms_of.insert(atom as u32, bdd);
            }
        }
        self.pending = pending;
        if let Some(frame) = self.frames.last_mut() {
            frame.latches = latches;
            frame.exits = exits;
        }
        for (route, target, mut list) in inside {
            for arrival in list.iter_mut() {
                if self.inner(arrival.mask, atoms) {
                    let replace = |var: u32| atoms_of.get(&var).copied();
                    let outside = self.masks.m.compose(arrival.mask, &replace);
                    arrival.mask = self.masks.and(mask, outside);
                }
                for arg in arrival.args.iter_mut() {
                    if let Some(Val::Value(value)) = *arg {
                        if let Some(&val) = renamed.get(&value) {
                            *arg = Some(val);
                        }
                    }
                }
            }
            match route {
                Route::Pending => self.pending.entry(target).or_default().extend(list),
                Route::Latch => self.frames.last_mut().unwrap().latches.extend(list),
                Route::Exit => self
                    .frames
                    .last_mut()
                    .unwrap()
                    .exits
                    .entry(target)
                    .or_default()
                    .extend(list),
            }
        }
    }

    fn inner(&mut self, f: Bdd, atoms: usize) -> bool {
        self.masks
            .m
            .support(f)
            .into_iter()
            .any(|var| var as usize >= atoms)
    }

    fn uniform_value(&self, value: ValueId) -> bool {
        self.masks
            .atoms()
            .iter()
            .any(|a| a.value == value && a.uniform)
    }

    fn add_params(&mut self, block: BlockId, types: &[Ty]) -> Vec<ValueId> {
        let params: Vec<ValueId> = types.iter().map(|&ty| self.out.value(ty)).collect();
        let target = self.out.blocks.get_mut(&block).unwrap();
        for (&v, &ty) in params.iter().zip(types) {
            target.params.push((v, ty));
        }
        params
    }

    fn emit_block(&mut self, id: BlockId, arrivals: Vec<Arrival>, mask: Bdd) {
        self.at = id;
        let q = self.q;
        let block = &q.blocks[&id];
        for (index, &(p, ty)) in block.params.iter().enumerate() {
            if !self.s.live[p.0] || self.s.same[p.0].is_some() {
                continue;
            }
            let vals: Vec<Option<Val>> = arrivals.iter().map(|a| a.args[index]).collect();
            let masks: Vec<Bdd> = arrivals.iter().map(|a| a.mask).collect();
            let val = self.blend(ty, &masks, &vals, mask);
            self.lane[p.0] = Some(val);
        }
        for (index, inst) in block.insts.iter().enumerate() {
            self.emit_inst(&block.insts[index + 1..], inst, mask);
        }
        match &block.term {
            Term::Ret(_) => {}
            Term::Br(edge) => {
                let args = self.edge_args(edge);
                self.arrive(edge.dst, mask, args);
            }
            Term::CondBr { cond, yes, no } => {
                let c = self.bit(*cond);
                let taken = self.masks.and(mask, c);
                let flipped = self.masks.not(c);
                let other = self.masks.and(mask, flipped);
                let yes_args = self.edge_args(yes);
                let no_args = self.edge_args(no);
                self.arrive(yes.dst, taken, yes_args);
                self.arrive(no.dst, other, no_args);
            }
        }
    }

    fn carried(&self, block: BlockId) -> Vec<(usize, ValueId, Ty)> {
        self.q.blocks[&block]
            .params
            .iter()
            .enumerate()
            .filter(|(_, &(p, _))| self.s.live[p.0] && self.s.same[p.0].is_none())
            .map(|(index, &(p, ty))| (index, p, ty))
            .collect()
    }

    fn edge_value(&mut self, val: Val) -> ValueId {
        match val {
            Val::Value(value) => value,
            Val::Bit(bit) => self.materialize(bit),
        }
    }

    fn incoming(&mut self, block: BlockId, arrivals: &[Arrival], care: Bdd) -> Vec<Val> {
        let masks: Vec<Bdd> = arrivals.iter().map(|a| a.mask).collect();
        self.carried(block)
            .into_iter()
            .map(|(index, _, ty)| {
                let vals: Vec<Option<Val>> = arrivals.iter().map(|a| a.args[index]).collect();
                self.blend(ty, &masks, &vals, care)
            })
            .collect()
    }

    fn param_val(&mut self, value: ValueId, lane: ValueId, ty: Ty) -> Val {
        if ty == Ty::I1 {
            Val::Bit(self.masks.atom(value, self.uniform[lane.0], Some(lane)))
        } else {
            Val::Value(value)
        }
    }

    fn extend_edge(&mut self, block: BlockId, dst: BlockId, args: &[ValueId]) {
        let term = &mut self.out.blocks.get_mut(&block).unwrap().term;
        for edge in term.edges_mut() {
            if edge.dst == dst {
                edge.args.extend_from_slice(args);
                return;
            }
        }
        unreachable!("the block does not branch to that target");
    }

    fn emit_loop(&mut self, l: usize, arrivals: Vec<Arrival>, entered: Bdd) {
        let header = self.s.header(l);
        let carried = self.carried(header);
        let entry_vals = self.incoming(header, &arrivals, entered);
        let mut types = vec![Ty::I1];
        types.extend(carried.iter().map(|&(_, _, ty)| ty));
        let (head, params) = self.block(&types);
        let (after, _) = self.block(&[]);
        let mut entry_args = vec![self.materialize(entered)];
        for &val in &entry_vals {
            let value = self.edge_value(val);
            entry_args.push(value);
        }
        let guard = self.cur;
        let held = self.masks.holds_a_lane(entered);

        let wave = self.keeps_wave(self.loop_blocks(l));
        if held {
            self.terminate(Term::Br(Edge {
                dst: head,
                args: entry_args,
            }));
        } else {
            let answer = if wave {
                let provenance = self.fresh_provenance();
                self.wave_query(provenance, entered)
            } else {
                self.query(entered)
            };
            let cond = self.materialize(answer);
            self.terminate(Term::CondBr {
                cond,
                yes: Edge {
                    dst: head,
                    args: entry_args,
                },
                no: Edge {
                    dst: after,
                    args: Vec::new(),
                },
            });
        }

        self.cur = head;
        self.scopes.push(Scope::default());
        let knowledge = self.masks.knowledge();
        let mask = self.masks.atom(params[0], false, None);
        self.masks.assume_a_lane(mask);
        self.masks.assume_within(mask, entered);

        let together = self.masks.exact(entered) == *self.full.last().unwrap();
        self.full.push(if together { mask } else { Bdd::FALSE });
        let mut args: Vec<Option<Val>> = vec![None; self.q.blocks[&header].params.len()];
        let mut current = Vec::with_capacity(carried.len());
        for (k, &(index, p, ty)) in carried.iter().enumerate() {
            let val = self.param_val(params[k + 1], p, ty);
            current.push(val);
            args[index] = Some(val);
        }
        let outer = std::mem::take(&mut self.pending);
        self.pending.insert(header, vec![Arrival { mask, args }]);
        self.frames.push(Frame {
            l,
            header,
            latches: Vec::new(),
            exits: BTreeMap::new(),
        });
        self.region(Some(l));
        assert!(
            self.pending.is_empty(),
            "the lowering left lanes waiting inside a loop"
        );
        self.pending = outer;
        let frame = self.frames.pop().unwrap();
        let continuing = self.union(&frame.latches);
        let depth = self.full.len() - 1;
        self.full.pop();
        if self.wanted.iter().any(|&d| d >= depth) {
            assert!(
                self.go_around_together(mask, continuing),
                "b{}: an operation over every lane in a loop the lanes leave apart",
                header.0
            );
            self.wanted.retain(|&d| d < depth);
        }
        let answer = if wave {
            let provenance = self.fresh_provenance();
            self.wave_query(provenance, continuing)
        } else {
            self.query(continuing)
        };
        let cond = self.materialize(answer);

        let around: Vec<Option<Val>> = carried
            .iter()
            .map(|&(index, _, _)| agreed(&frame.latches, index))
            .collect();
        let mut kept_around = vec![false; carried.len()];
        let mut leaves: Vec<Leave> = Vec::new();
        for (target, list) in frame.exits {
            let left = self.union(&list);

            let early = self.masks.coexist(left, continuing);
            let mut leave = Leave {
                target,
                carried: self.carried(target),
                list,
                held: Bdd::FALSE,
                kept: Vec::new(),
            };
            if early {
                let gone = self.add_params(head, &[Ty::I1])[0];
                let zero = self.constant_in(guard, Ty::I1, 0);
                self.extend_edge(guard, head, &[zero]);
                let gone = self.masks.atom(gone, false, None);
                let both = self.masks.and(gone, mask);
                let apart = self.masks.not(both);
                self.masks.assume(apart);
                self.masks.assume_within(gone, entered);
                leave.held = gone;
                for (index, p, ty) in leave.carried.clone() {
                    let out = agreed(&leave.list, index);
                    let same = out.and_then(|v| around.iter().position(|&a| a == Some(v)));
                    let kept = match same {

                        Some(k) => {
                            kept_around[k] = true;
                            Kept::Variable(k)
                        }
                        None => {
                            let slot = self.add_params(head, &[ty])[0];
                            let zero = self.constant_in(guard, ty, 0);
                            self.extend_edge(guard, head, &[zero]);
                            Kept::Slot(self.param_val(slot, p, ty))
                        }
                    };
                    leave.kept.push(kept);
                }
            }
            leaves.push(leave);
        }
        let variables: Vec<Option<Val>> = carried
            .iter()
            .enumerate()
            .map(|(k, &(_, _, ty))| {
                kept_around[k].then(|| self.choose(ty, mask, around[k].unwrap(), current[k]))
            })
            .collect();
        let flipped = self.masks.not(answer);
        let back = self.loop_edge(
            header,
            &frame.latches,
            &leaves,
            &variables,
            continuing,
            answer,
            true,
        );
        let out = self.loop_edge(
            header,
            &frame.latches,
            &leaves,
            &variables,
            continuing,
            flipped,
            false,
        );
        match answer.constant() {
            Some(true) => self.terminate(Term::Br(Edge {
                dst: head,
                args: back,
            })),
            Some(false) => self.terminate(Term::Br(Edge {
                dst: after,
                args: out,
            })),
            None => self.terminate(Term::CondBr {
                cond,
                yes: Edge {
                    dst: head,
                    args: back,
                },
                no: Edge {
                    dst: after,
                    args: out,
                },
            }),
        }
        self.masks.restore(knowledge);
        self.scopes.pop();

        let types: Vec<Ty> = leaves
            .iter()
            .flat_map(|leave| {
                std::iter::once(Ty::I1).chain(leave.carried.iter().map(|&(_, _, ty)| ty))
            })
            .collect();
        let slots = self.add_params(after, &types);
        if !held {
            let zeros: Vec<ValueId> = types
                .iter()
                .map(|&ty| self.constant_in(guard, ty, 0))
                .collect();
            self.extend_edge(guard, after, &zeros);
        }
        self.cur = after;

        let mut taken = Bdd::FALSE;
        let mut at = 0usize;
        let count = leaves.len();
        for (index, leave) in leaves.into_iter().enumerate() {
            let last = index + 1 == count;
            let mut mask = if last {
                let rest = self.masks.not(taken);
                self.masks.and(entered, rest)
            } else {
                self.masks.atom(slots[at], false, None)
            };
            mask = self.masks.exact(mask);
            self.masks.assume_within(mask, entered);
            if !last {
                let overlap = self.masks.and(taken, mask);
                let apart = self.masks.not(overlap);
                self.masks.assume(apart);
                taken = self.masks.or(taken, mask);
            }
            at += 1;
            let mut args: Vec<Option<Val>> = vec![None; self.q.blocks[&leave.target].params.len()];
            for &(index, p, ty) in &leave.carried {
                let val = self.param_val(slots[at], p, ty);
                args[index] = Some(val);
                at += 1;
            }
            self.arrive(leave.target, mask, args);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn loop_edge(
        &mut self,
        header: BlockId,
        latches: &[Arrival],
        leaves: &[Leave],
        variables: &[Option<Val>],
        continuing: Bdd,
        decided: Bdd,
        around: bool,
    ) -> Vec<ValueId> {
        let saved = self.masks.knowledge();
        self.masks.assume(decided);
        let mut args = Vec::new();
        if around {
            let going = self.masks.exact(continuing);
            args.push(self.materialize(going));
            let carried = self.incoming(header, latches, going);
            for (k, val) in carried.into_iter().enumerate() {
                let val = variables[k].unwrap_or(val);
                let value = self.edge_value(val);
                args.push(value);
            }
        }
        for leave in leaves {
            if around && leave.held == Bdd::FALSE {
                continue;
            }
            let left = self.union(&leave.list);
            let reached = self.masks.or(leave.held, left);
            let reached = self.masks.exact(reached);
            args.push(self.materialize(reached));
            for (position, &(index, _, ty)) in leave.carried.iter().enumerate() {
                let mut masks: Vec<Bdd> = leave.list.iter().map(|a| a.mask).collect();
                let mut vals: Vec<Option<Val>> = leave.list.iter().map(|a| a.args[index]).collect();
                match leave.kept.get(position) {
                    Some(Kept::Variable(k)) => {
                        if !around {
                            let value = self.edge_value(variables[*k].unwrap());
                            args.push(value);
                        }
                        continue;
                    }
                    Some(Kept::Slot(acc)) => {
                        masks.push(leave.held);
                        vals.push(Some(*acc));
                    }
                    None => {}
                }
                let blended = self.blend(ty, &masks, &vals, reached);
                let value = self.edge_value(blended);
                args.push(value);
            }
        }
        self.masks.restore(saved);
        args
    }

    fn edge_args(&mut self, edge: &Edge) -> Vec<Option<Val>> {
        let q = self.q;
        let params = &q.blocks[&edge.dst].params;
        let mut args = Vec::with_capacity(params.len());
        for (&arg, &(p, _)) in edge.args.iter().zip(params) {
            if !self.s.live[p.0] || self.s.same[p.0].is_some() {
                args.push(None);
            } else {
                args.push(Some(self.val(arg)));
            }
        }
        args
    }

    fn arrive(&mut self, target: BlockId, mask: Bdd, args: Vec<Option<Val>>) {
        let mask = self.masks.exact(mask);
        if mask == Bdd::FALSE {
            return;
        }
        let arrival = Arrival { mask, args };
        let route = match self.frames.last() {
            Some(frame) => {
                if target == frame.header {
                    Route::Latch
                } else if self.s.within(Some(frame.l), target) {
                    Route::Pending
                } else {
                    Route::Exit
                }
            }
            None => Route::Pending,
        };
        match route {
            Route::Latch => self.frames.last_mut().unwrap().latches.push(arrival),
            Route::Exit => self
                .frames
                .last_mut()
                .unwrap()
                .exits
                .entry(target)
                .or_default()
                .push(arrival),
            Route::Pending => self.pending.entry(target).or_default().push(arrival),
        }
    }

    fn blend(&mut self, ty: Ty, masks: &[Bdd], vals: &[Option<Val>], care: Bdd) -> Val {
        let mut groups: Vec<(Val, Bdd)> = Vec::new();
        for (&m, val) in masks.iter().zip(vals) {
            let Some(val) = val else { continue };
            match groups.iter_mut().find(|(v, _)| v == val) {
                Some((_, mask)) => *mask = self.masks.or(*mask, m),
                None => groups.push((*val, m)),
            }
        }
        if groups.is_empty() {
            return self.zero(ty);
        }
        if groups.len() == 1 {
            return groups[0].0;
        }
        let default = groups.len() - 1;
        let mut left = care;
        let mut choices: Vec<(Val, Bdd)> = Vec::new();
        for (index, &(val, mask)) in groups.iter().enumerate() {
            if index == default {
                continue;
            }
            let discriminator = self.masks.choice(mask, left);
            let rest = self.masks.not(mask);
            left = self.masks.and(left, rest);
            choices.push((val, discriminator));
        }
        let mut result = groups[default].0;
        for &(val, discriminator) in choices.iter().rev() {
            result = self.choose(ty, discriminator, val, result);
        }
        result
    }

    fn choose(&mut self, ty: Ty, discriminator: Bdd, taken: Val, other: Val) -> Val {
        if taken == other || discriminator == Bdd::TRUE {
            return taken;
        }
        if discriminator == Bdd::FALSE {
            return other;
        }
        if ty == Ty::I1 {
            let (Val::Bit(a), Val::Bit(b)) = (taken, other) else {
                unreachable!("a bit is held as a mask");
            };
            return Val::Bit(self.masks.ite(discriminator, a, b));
        }
        let (Val::Value(a), Val::Value(b)) = (taken, other) else {
            unreachable!("a value wider than a bit is held as a value");
        };

        let flipped = self.masks.not(discriminator);
        if self.building(flipped) < self.building(discriminator) {
            let c = self.build(flipped);
            return Val::Value(self.core(ty, Op::Select(c, b, a)));
        }
        let c = self.build(discriminator);
        Val::Value(self.core(ty, Op::Select(c, a, b)))
    }

    fn access_predicate(
        &mut self,
        rest: &[Inst],
        provenance: u64,
        op: EffectOp,
        predicate: ValueId,
        mask: Bdd,
    ) -> Option<ValueId> {
        if let Some(predicate) = self.flat.remove(&provenance) {
            return predicate;
        }
        let own = self.bit(predicate);
        let EffectOp::Memory {
            space: Space::Global,
            op: memory,
            ..
        } = op
        else {
            let active = self.masks.and(mask, own);
            let active = self.masks.exact(active);
            return (active != Bdd::FALSE).then(|| self.materialize(active));
        };
        let partner = rest.iter().find_map(|inst| match inst {
            Inst::Effect {
                provenance: p,
                op:
                    EffectOp::Memory {
                        space: Space::Scratch,
                        op: o,
                        ..
                    },
                inputs,
                ..
            } if p >> 8 == provenance >> 8 && *o == memory => {
                let index = match memory {
                    MemoryOp::Load(_) => 1,
                    _ => 2,
                };
                Some((*p, inputs[index]))
            }
            _ => None,
        });
        let Some((private, inside_predicate)) = partner else {
            let active = self.masks.and(mask, own);
            let active = self.masks.exact(active);
            return (active != Bdd::FALSE).then(|| self.materialize(active));
        };
        let inside = self.bit(inside_predicate);
        let either = self.masks.or(own, inside);
        let active = self.masks.and(mask, either);
        let active = self.masks.exact(active);
        if active == Bdd::FALSE {
            self.flat.insert(private, None);
            return None;
        }
        let chosen = self.masks.choice(inside, active);
        let lanes = self.materialize(active);
        let within = self.build(chosen);
        let one = self.constant(Ty::I1, 1);
        let outside = self.core(Ty::I1, Op::Int(IntOp::Xor, within, one));
        let global = self.core(Ty::I1, Op::Int(IntOp::And, lanes, outside));
        let scratch = self.core(Ty::I1, Op::Int(IntOp::And, lanes, within));
        self.flat.insert(private, Some(scratch));
        Some(global)
    }

    fn emit_inst(&mut self, rest: &[Inst], inst: &Inst, mask: Bdd) {
        match inst {
            Inst::Core { value, ty, op } => {
                if *ty == Ty::I1 {
                    if let Some(bit) = self.bit_op(*op) {
                        self.lane[value.0] = Some(Val::Bit(bit));
                        return;
                    }
                }
                let mapped = self.map_op(*op);
                let out = self.core(*ty, mapped);
                self.define(*value, *ty, out);
            }
            Inst::Target {
                provenance,
                op,
                args,
                outputs,
            } => {
                let spec = self
                    .registry
                    .operation(*op)
                    .expect("a lane program only names registered operations");
                let effect = spec.effect;
                let immediates: Vec<usize> = spec.immediates.iter().map(|&(i, _)| i).collect();
                let values: Vec<ValueId> = args.values().to_vec();
                let mut mapped = Vec::with_capacity(values.len());
                for (index, &v) in values.iter().enumerate() {
                    let ty = self.q.types[self.s.resolve(v).0];
                    let value = match effect {
                        Effect::ReadGlobal { every_lane: false } if ty == Ty::I1 => {
                            let bit = self.bit(v);
                            let active = self.masks.and(mask, bit);
                            self.materialize(active)
                        }
                        Effect::ReadGlobal { every_lane: true }
                            if ty != Ty::I1
                                && !immediates.contains(&index)
                                && mask != Bdd::TRUE =>
                        {

                            let operand = self.value(v);
                            let zero = self.constant(ty, 0);
                            let active = self.materialize(mask);
                            self.core(ty, Op::Select(active, operand, zero))
                        }
                        _ => self.value(v),
                    };
                    mapped.push(value);
                }
                let mut next = mapped.into_iter();
                let args = args.map(|_| next.next().unwrap());
                let outputs: Vec<(ValueId, Ty)> = outputs
                    .iter()
                    .map(|&(_, ty)| (self.out.value(ty), ty))
                    .collect();
                self.push(Inst::Target {
                    provenance: *provenance,
                    op: *op,
                    args,
                    outputs: outputs.clone(),
                });
                for (&(lane, ty), &(value, _)) in inst_outputs(inst).iter().zip(&outputs) {
                    self.define(lane, ty, value);
                }
            }
            Inst::Effect {
                provenance,
                op: EffectOp::Wave(wave),
                inputs,
                outputs,
            } => self.emit_wave(*provenance, *wave, inputs, outputs, mask),
            Inst::Effect {
                provenance,
                op: op @ (EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait),
                inputs,
                outputs,
            } => {
                self.everyone(mask);
                self.emit_effect(*provenance, *op, inputs, outputs);
            }
            Inst::Effect {
                provenance,
                op:
                    op @ EffectOp::Memory {
                        op: MemoryOp::Fence,
                        ..
                    },
                inputs,
                outputs,
            } => self.emit_effect(*provenance, *op, inputs, outputs),
            Inst::Effect {
                provenance,
                op,
                inputs,
                outputs,
            } => {
                let EffectOp::Memory { op: memory, .. } = op else {
                    unreachable!("every wave operation and barrier is lowered above")
                };
                let predicate = match memory {
                    MemoryOp::Load(_) => 1,
                    MemoryOp::Store(_) | MemoryOp::AtomicAdd(_) => 2,
                    MemoryOp::Fence => unreachable!("a fence is lowered above"),
                };
                let active = self.access_predicate(rest, *provenance, *op, inputs[predicate], mask);
                let Some(active) = active else {
                    for &(lane, ty) in outputs {
                        let zero = self.constant(ty, 0);
                        self.define(lane, ty, zero);
                    }
                    return;
                };
                let mut mapped = Vec::with_capacity(inputs.len());
                for (index, &v) in inputs.iter().enumerate() {
                    if index == predicate {
                        mapped.push(active);
                    } else {
                        mapped.push(self.value(v));
                    }
                }
                let results: Vec<(ValueId, Ty)> = outputs
                    .iter()
                    .map(|&(_, ty)| (self.out.value(ty), ty))
                    .collect();
                self.push(Inst::Effect {
                    provenance: *provenance,
                    op: *op,
                    inputs: mapped,
                    outputs: results.clone(),
                });
                for (&(lane, ty), &(value, _)) in outputs.iter().zip(&results) {
                    self.define(lane, ty, value);
                }
            }
            Inst::Packet { .. } => unreachable!("a lane program queries no packet"),
        }
    }

    fn keeps_wave(&self, blocks: impl Iterator<Item = BlockId>) -> bool {
        blocks.into_iter().any(|b| {
            self.q.blocks[&b].insts.iter().any(|inst| {
                matches!(
                    inst,
                    Inst::Effect {
                        op: EffectOp::Wave(_)
                            | EffectOp::BarrierSignal { .. }
                            | EffectOp::BarrierWait,
                        ..
                    }
                )
            })
        })
    }

    fn everyone(&mut self, mask: Bdd) {
        let full = *self.full.last().unwrap();
        assert!(
            full != Bdd::FALSE && self.masks.exact(mask) == full,
            "b{}: an operation over every lane where lanes may be elsewhere",
            self.at.0
        );
        self.wanted.push(self.full.len() - 1);
    }

    fn go_around_together(&mut self, mask: Bdd, continuing: Bdd) -> bool {
        let own: BTreeSet<u32> = self.masks.m.support(mask);
        self.all_or_none(continuing, &own)
    }

    fn all_or_none(&mut self, f: Bdd, own: &BTreeSet<u32>) -> bool {
        let f = self.masks.exact(f);
        self.masks
            .m
            .support(f)
            .into_iter()
            .all(|var| own.contains(&var) || self.masks.at(var).uniform)
    }

    fn emit_effect(
        &mut self,
        provenance: u64,
        op: EffectOp,
        inputs: &[ValueId],
        outputs: &[(ValueId, Ty)],
    ) {
        let inputs: Vec<ValueId> = inputs.iter().map(|&v| self.value(v)).collect();
        let results: Vec<(ValueId, Ty)> = outputs
            .iter()
            .map(|&(_, ty)| (self.out.value(ty), ty))
            .collect();
        self.push(Inst::Effect {
            provenance,
            op,
            inputs,
            outputs: results.clone(),
        });
        for (&(lane, ty), &(value, _)) in outputs.iter().zip(&results) {
            self.define(lane, ty, value);
        }
    }

    fn loop_blocks(&self, l: usize) -> impl Iterator<Item = BlockId> + '_ {
        self.q
            .blocks
            .keys()
            .copied()
            .filter(move |&b| self.s.within(Some(l), b))
    }

    fn fresh_provenance(&mut self) -> u64 {
        let p = self.provenance;
        self.provenance += 1;
        p
    }

    fn emit_wave(
        &mut self,
        provenance: u64,
        op: WaveOp,
        inputs: &[ValueId],
        outputs: &[(ValueId, Ty)],
        mask: Bdd,
    ) {
        let (lane, ty) = outputs[0];
        match op {
            WaveOp::Any => {
                if self.demanded.contains(&provenance) {
                    self.everyone(mask);
                }
                let x = self.bit(inputs[0]);
                let held = self.masks.and(mask, x);
                let answer = self.wave_query(provenance, held);
                self.lane[lane.0] = Some(Val::Bit(answer));
            }
            WaveOp::Ballot => {
                if self.demanded.contains(&provenance) {
                    self.everyone(mask);
                }
                let x = self.bit(inputs[0]);
                let held = self.masks.and(mask, x);
                let input = self.materialize(held);
                let word = self.out.value(ty);
                self.push(Inst::Effect {
                    provenance,
                    op: EffectOp::Wave(op),
                    inputs: vec![input],
                    outputs: vec![(word, ty)],
                });
                self.define(lane, ty, word);
            }
            WaveOp::ReadFirstLane => {
                if self.demanded.contains(&provenance) {
                    self.everyone(mask);
                }
                let value = self.value(inputs[0]);
                let e = self.bit(inputs[1]);
                let held = self.masks.and(mask, e);
                let active = self.materialize(held);
                let word = self.out.value(ty);
                self.push(Inst::Effect {
                    provenance,
                    op: EffectOp::Wave(op),
                    inputs: vec![value, active],
                    outputs: vec![(word, ty)],
                });
                self.define(lane, ty, word);
            }
            WaveOp::ReadLane
            | WaveOp::WriteLane
            | WaveOp::Bpermute
            | WaveOp::BpermuteFi
            | WaveOp::Wmma => {
                self.everyone(mask);
                self.emit_effect(provenance, EffectOp::Wave(op), inputs, outputs);
            }
        }
    }

    fn wave_query(&mut self, provenance: u64, f: Bdd) -> Bdd {
        let f = self.masks.exact(f);
        if f.constant().is_some() {
            return f;
        }
        if self.masks.holds_a_lane(f) {
            return Bdd::TRUE;
        }
        let input = self.build(f);
        let output = self.out.value(Ty::I1);
        self.push(Inst::Effect {
            provenance,
            op: EffectOp::Wave(WaveOp::Any),
            inputs: vec![input],
            outputs: vec![(output, Ty::I1)],
        });
        self.masks.atom(output, true, None)
    }

    fn bit_op(&mut self, op: Op) -> Option<Bdd> {
        let bits = |emit: &mut Self, a: ValueId, b: ValueId| (emit.bit(a), emit.bit(b));
        let i1 = |emit: &Self, v: ValueId| emit.q.types[emit.s.resolve(v).0] == Ty::I1;
        match op {
            Op::Const(Ty::I1, k) => Some(if k & 1 != 0 { Bdd::TRUE } else { Bdd::FALSE }),
            Op::Int(IntOp::And, a, b) => {
                let (a, b) = bits(self, a, b);
                Some(self.masks.and(a, b))
            }
            Op::Int(IntOp::Or, a, b) => {
                let (a, b) = bits(self, a, b);
                Some(self.masks.or(a, b))
            }
            Op::Int(IntOp::Xor, a, b) => {
                let (a, b) = bits(self, a, b);
                let flipped = self.masks.not(b);
                Some(self.masks.ite(a, flipped, b))
            }
            Op::Select(c, a, b) if i1(self, a) => {
                let c = self.bit(c);
                let (a, b) = bits(self, a, b);
                Some(self.masks.ite(c, a, b))
            }
            Op::Convert(Cvt::Bitcast, Ty::I1, a) if i1(self, a) => Some(self.bit(a)),
            _ => None,
        }
    }

    fn map_op(&mut self, op: Op) -> Op {
        let mut mapped = op;
        mapped = mapped.map(|v| self.value(v));
        mapped
    }
}

#[derive(Clone, Copy)]
enum Route {
    Latch,
    Exit,
    Pending,
}

#[derive(Default)]
struct Crossing {
    entries: Vec<(ValueId, Ty, bool)>,
}

impl Crossing {
    fn add(&mut self, value: ValueId, ty: Ty, uniform: bool) {
        if !self.entries.iter().any(|e| e.0 == value) {
            self.entries.push((value, ty, uniform));
        }
    }
}

fn inst_outputs(inst: &Inst) -> Vec<(ValueId, Ty)> {
    match inst {
        Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => outputs.clone(),
        Inst::Core { value, ty, .. } => vec![(*value, *ty)],
        Inst::Packet { output, op, .. } => vec![(*output, op.result_type())],
    }
}
