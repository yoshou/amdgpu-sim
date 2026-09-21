use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Exit {
    pub from: BlockId,
    pub edge: Option<usize>,
    pub to: Option<BlockId>,
}

pub(super) struct Regions {
    pub entries: Vec<BlockId>,
    pub children: Vec<Vec<usize>>,
    pub home: BTreeMap<BlockId, usize>,
    inside: Vec<BTreeSet<BlockId>>,
    pub live: Vec<Vec<ValueId>>,
    pub rebuilt: Vec<Vec<ValueId>>,
    pub exits: Vec<Vec<Exit>>,
    pub frame_words: usize,
    scalar: Vec<bool>,
    width: u32,
}

pub(super) const FRAME_BASE: u32 = 16;

pub(super) fn scalar_values(p: &Prepared) -> Vec<bool> {
    let scalars = std::env::var("AMDGPU_SIM_NOSCALAR").map_or(true, |x| x != "1");
    (0..p.ir.func().types.len())
        .map(|v| p.width.is_none() || (p.uniform[v] && scalars))
        .collect()
}

fn outputs(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { value, .. } | Inst::Packet { output: value, .. } => f(*value),
        Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => {
            for o in outputs {
                f(o.0);
            }
        }
    }
}

fn term_uses(term: &Term, mut f: impl FnMut(ValueId)) {
    match term {
        Term::Br(edge) => edge.args.iter().for_each(|&v| f(v)),
        Term::CondBr { cond, yes, no } => {
            f(*cond);
            yes.args.iter().chain(&no.args).for_each(|&v| f(v));
        }
        Term::Ret(args) => args.iter().for_each(|&v| f(v)),
    }
}

fn unseen_uses(
    p: &Prepared,
    definitions: &[Option<Op>],
) -> BTreeMap<(BlockId, usize), Vec<ValueId>> {
    let f = p.ir.func();
    let lane_word = |shifted: ValueId| match definitions[shifted.0] {
        Some(Op::Int(IntOp::LShr, word, lane))
            if matches!(definitions[lane.0], Some(Op::Env(Env::LaneId))) =>
        {
            Some(word)
        }
        _ => None,
    };
    let mut unseen: BTreeMap<(BlockId, usize), Vec<ValueId>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            let word = match inst {
                Inst::Core {
                    op: Op::Convert(Cvt::Trunc, Ty::I1, shifted),
                    ..
                } => lane_word(*shifted),
                Inst::Packet { input, .. } => {
                    let bit = match definitions[input.0] {
                        Some(Op::Int(IntOp::And, a, b))
                            if matches!(definitions[b.0], Some(Op::Env(Env::ValidLane))) =>
                        {
                            a
                        }
                        Some(Op::Int(IntOp::And, a, b))
                            if matches!(definitions[a.0], Some(Op::Env(Env::ValidLane))) =>
                        {
                            b
                        }
                        _ => *input,
                    };
                    match definitions[bit.0] {
                        Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => lane_word(shifted),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(word) = word {
                unseen.entry((id, index)).or_default().push(word);
            }
        }
    }
    for access in p.accesses.iter() {
        let list = unseen
            .entry((access.block, access.effects[0]))
            .or_default();
        list.extend([access.base, access.address, access.mask]);
        list.extend(access.inside);
        list.extend(access.data.iter().copied());
    }
    unseen
}

impl Regions {
    pub fn new(p: &Prepared) -> Self {
        let f = p.ir.func();
        let doms = Dominators::of(f);
        let held = f.regions_of(&doms);
        let mut entries = vec![f.entry];
        entries.extend(f.regions.keys().copied().filter(|&e| e != f.entry));
        let index: BTreeMap<BlockId, usize> =
            entries.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let home: BTreeMap<BlockId, usize> =
            held.iter().map(|(&b, e)| (b, index[e])).collect();
        let mut children = vec![Vec::new(); entries.len()];
        for (r, &entry) in entries.iter().enumerate().skip(1) {
            let above = doms
                .parent(entry)
                .expect("a nested region is entered below another block");
            children[home[&above]].push(r);
        }
        let inside: Vec<BTreeSet<BlockId>> = entries
            .iter()
            .map(|&e| {
                doms.order
                    .iter()
                    .copied()
                    .filter(|&b| doms.dominates(e, b))
                    .collect()
            })
            .collect();
        let values = f.types.len();
        let mut definitions = vec![None; values];
        for block in f.blocks.values() {
            for inst in &block.insts {
                if let Inst::Core { value, op, .. } = inst {
                    definitions[value.0] = Some(*op);
                }
            }
        }
        let unseen = unseen_uses(p, &definitions);
        let mut live = vec![Vec::new(); entries.len()];
        let mut rebuilt = vec![Vec::new(); entries.len()];
        for r in 1..entries.len() {
            let mut defined: BTreeSet<ValueId> = BTreeSet::new();
            let mut used: BTreeSet<ValueId> = BTreeSet::new();
            for &id in &inside[r] {
                let block = &f.blocks[&id];
                defined.extend(block.params.iter().map(|p| p.0));
                for (at, inst) in block.insts.iter().enumerate() {
                    outputs(inst, |v| {
                        defined.insert(v);
                    });
                    super::super::pass::dce::operands(inst, |v| {
                        used.insert(v);
                    });
                    used.extend(unseen.get(&(id, at)).into_iter().flatten().copied());
                }
                term_uses(&block.term, |v| {
                    used.insert(v);
                });
            }
            for v in used.difference(&defined).copied() {
                let mut operands = 0;
                if let Some(op) = definitions[v.0] {
                    op.map(|o| {
                        operands += 1;
                        o
                    });
                }
                if definitions[v.0].is_some() && operands == 0 {
                    rebuilt[r].push(v);
                } else {
                    live[r].push(v);
                }
            }
        }
        let exits: Vec<Vec<Exit>> = inside
            .iter()
            .map(|blocks| {
                let mut exits = Vec::new();
                for &from in blocks {
                    let term = &f.blocks[&from].term;
                    if matches!(term, Term::Ret(_)) {
                        exits.push(Exit {
                            from,
                            edge: None,
                            to: None,
                        });
                    }
                    for (edge, e) in term.edges().enumerate() {
                        if !blocks.contains(&e.dst) {
                            exits.push(Exit {
                                from,
                                edge: Some(edge),
                                to: Some(e.dst),
                            });
                        }
                    }
                }
                exits
            })
            .collect();
        let mut regions = Self {
            entries,
            children,
            home,
            inside,
            live,
            rebuilt,
            exits,
            frame_words: 0,
            scalar: scalar_values(p),
            width: p.width.expect("regions belong to a packet program"),
        };
        let mut words = FRAME_BASE;
        for r in 1..regions.entries.len() {
            let inputs = regions.inputs(f, r);
            words = words.max(regions.end(f, &inputs));
        }
        for block in f.blocks.values() {
            let params: Vec<ValueId> = block.params.iter().map(|p| p.0).collect();
            words = words.max(regions.end(f, &params));
        }
        regions.frame_words = words.div_ceil(16) as usize * 16;
        regions
    }

    pub fn inputs(&self, f: &Func, region: usize) -> Vec<ValueId> {
        f.blocks[&self.entries[region]]
            .params
            .iter()
            .map(|p| p.0)
            .chain(self.live[region].iter().copied())
            .collect()
    }

    fn size(&self, f: &Func, v: ValueId) -> u32 {
        match (f.types[v.0], self.scalar[v.0]) {
            (Ty::I1, _) => 1,
            (ty, true) => ty.bits() / 32,
            (ty, false) => ty.bits() / 32 * self.width,
        }
    }

    pub fn slots(&self, f: &Func, values: &[ValueId]) -> Vec<u32> {
        let mut next = FRAME_BASE;
        values
            .iter()
            .map(|&v| {
                let size = self.size(f, v);
                let align = size.min(16).next_power_of_two();
                next = next.div_ceil(align) * align;
                let at = next;
                next += size;
                at
            })
            .collect()
    }

    fn end(&self, f: &Func, values: &[ValueId]) -> u32 {
        self.slots(f, values)
            .last()
            .zip(values.last())
            .map_or(FRAME_BASE, |(&at, &v)| at + self.size(f, v))
    }

    pub fn holds(&self, region: usize, block: BlockId) -> bool {
        self.inside[region].contains(&block)
    }

    pub fn exit(&self, region: usize, from: BlockId, edge: Option<usize>) -> usize {
        self.exits[region]
            .iter()
            .position(|x| x.from == from && x.edge == edge)
            .expect("an edge leaves a region without being one of its exits")
    }

    pub fn own(&self, region: usize) -> BTreeSet<BlockId> {
        self.home
            .iter()
            .filter(|&(_, &r)| r == region)
            .map(|(&b, _)| b)
            .collect()
    }
}

impl<'a> Cg<'a> {
    fn frame_slot(&self, word: u32) -> Value {
        self.register_slot(self.frame, word)
    }

    fn stored_type(&self, shape: ValueId) -> Type {
        match (self.types[shape.0], self.param_scalar[shape.0]) {
            (Ty::I1, _) => self.ir.i32(),
            (ty, true) => self.sem.ty(ty),
            (ty, false) => self.em.ty(ty),
        }
    }

    fn store_value(&self, shape: ValueId, value: Value, word: u32) {
        let ir = self.ir;
        let scalar = self.param_scalar[shape.0];
        let value = match (scalar, value.is_vector()) {
            (true, true) => ir.extract_at(value, 0),
            (false, false) => self.splat(value),
            _ => value,
        };
        let bits = match (self.types[shape.0], scalar) {
            (Ty::I1, true) => ir.zext(value, ir.i32()),
            (Ty::I1, false) => self.vec_to_mask(value),
            _ => value,
        };
        ir.store(bits, self.frame_slot(word)).set_alignment(4);
    }

    fn load_value(&self, shape: ValueId, word: u32) -> Value {
        let ir = self.ir;
        let stored = ir
            .load(self.stored_type(shape), self.frame_slot(word))
            .set_alignment(4);
        match (self.types[shape.0], self.param_scalar[shape.0]) {
            (Ty::I1, true) => ir.trunc(stored, ir.i1()),
            (Ty::I1, false) => self.mask_to_vec(stored),
            _ => stored,
        }
    }

    fn as_incoming(&self, param: ValueId, value: Value) -> Value {
        if !self.param_scalar[param.0] && self.types[param.0] == Ty::I1 && self.mask_words() {
            self.ir.bitcast(value, self.ir.int(self.width()))
        } else {
            value
        }
    }

    pub(super) fn enter_region(&mut self) {
        let regions = self.regions;
        let f = self.p.ir.func();
        let entry = regions.entries[self.region];
        let params = f.blocks[&entry].params.len();
        let inputs = regions.inputs(f, self.region);
        let slots = regions.slots(f, &inputs);
        let mut args = Vec::new();
        for (i, (&v, &slot)) in inputs.iter().zip(&slots).enumerate() {
            let value = self.load_value(v, slot);
            if i < params {
                args.push(self.as_incoming(v, value));
            } else {
                self.define(v, value);
            }
        }
        for &v in &regions.rebuilt[self.region] {
            let op = self.definitions[v.0].expect("a rebuilt value is a core operation");
            self.emit_core(v, self.types[v.0], op);
        }
        let from = self.ir.insert_block();
        self.incoming.entry(entry).or_default().push((from, args));
    }

    pub(super) fn leave_region(&mut self, from: BlockId, ordinal: usize, edge: &Edge) -> BasicBlock {
        let regions = self.regions;
        let f = self.p.ir.func();
        let ir = self.ir;
        let exit = regions.exit(self.region, from, Some(ordinal));
        let params: Vec<ValueId> = f.blocks[&edge.dst].params.iter().map(|p| p.0).collect();
        let slots = regions.slots(f, &params);
        let values: Vec<Value> = edge
            .args
            .iter()
            .zip(&params)
            .map(|(&arg, &param)| self.shaped(arg, self.param_scalar[param.0]))
            .collect();
        let here = ir.insert_block();
        let leave = ir.append_block(self.func, "leave");
        ir.position_at_end(leave);
        for ((&param, &slot), &value) in params.iter().zip(&slots).zip(&values) {
            self.store_value(param, value, slot);
        }
        ir.ret(self.ci64(super::super::engine::kernel::COOP_LEAVE | exit as u64));
        ir.position_at_end(here);
        leave
    }

    pub(super) fn emit_child(&mut self, ordinal: usize, child: usize) {
        let regions = self.regions;
        let f = self.p.ir.func();
        let ir = self.ir;
        let entry = regions.entries[child];
        let block = &f.blocks[&entry];
        ir.position_at_end(self.bbs[&entry]);
        let words = self.mask_words();
        let mut phis = Vec::new();
        let mut held = Vec::new();
        for &(v, ty) in &block.params {
            let scalar = self.param_scalar[v.0];
            let t = if scalar {
                self.sem.ty(ty)
            } else if ty == Ty::I1 && words {
                ir.int(self.width())
            } else {
                self.em.ty(ty)
            };
            let phi = ir.phi(t);
            phis.push(phi);
            held.push(if !scalar && ty == Ty::I1 && words {
                ir.bitcast(phi, self.em.ty(Ty::I1))
            } else {
                phi
            });
        }
        self.phis.insert(entry, phis);
        let inputs = regions.inputs(f, child);
        let slots = regions.slots(f, &inputs);
        for (i, (&v, &slot)) in inputs.iter().zip(&slots).enumerate() {
            let value = match held.get(i) {
                Some(&value) => value,
                None => {
                    let value = self.values[v.0];
                    assert!(
                        !value.is_null(),
                        "a region is entered before v{} is defined: {}",
                        v.0,
                        self.describe(v)
                    );
                    value
                }
            };
            self.store_value(v, value, slot);
        }
        let ty = ir.void().function(&[ir.ptr(), ir.i64(), ir.ptr()]);
        let function = ir.function("amdgpu_sim_fiber_yield_values", ty);
        let enter = super::super::engine::kernel::COOP_ENTER | ordinal as u64;
        ir.call(
            ty,
            function,
            &[self.func.param(6), ir.ci64(enter), self.yield_frame],
        );
        let left = ir.load(ir.i32(), self.frame_slot(0)).set_alignment(4);
        let exits = &regions.exits[child];
        assert!(!exits.is_empty(), "a region that is never left");
        for (k, exit) in exits.iter().enumerate() {
            let stub = ir.append_block(self.func, "left");
            if k + 1 < exits.len() {
                let next = ir.append_block(self.func, "left.next");
                let taken = ir.icmp(IntPred::Eq, left, self.ci32(k as u32));
                ir.cond_br(taken, stub, next);
                ir.position_at_end(stub);
                self.emit_left(exit);
                ir.position_at_end(next);
            } else {
                ir.br(stub);
                ir.position_at_end(stub);
                self.emit_left(exit);
            }
        }
    }

    fn emit_left(&mut self, exit: &Exit) {
        let regions = self.regions;
        let f = self.p.ir.func();
        let ir = self.ir;
        match exit.to {
            Some(to) if regions.holds(self.region, to) => {
                let params: Vec<ValueId> = f.blocks[&to].params.iter().map(|p| p.0).collect();
                let slots = regions.slots(f, &params);
                let values: Vec<Value> = params
                    .iter()
                    .zip(&slots)
                    .map(|(&param, &slot)| self.as_incoming(param, self.load_value(param, slot)))
                    .collect();
                let from = ir.insert_block();
                self.incoming.entry(to).or_default().push((from, values));
                ir.br(self.bbs[&to]);
            }
            _ if self.region == 0 => {
                assert!(exit.to.is_none(), "an edge leaves the outermost region");
                ir.ret(self.ci64(super::super::engine::kernel::COOP_DONE));
            }
            _ => {
                let outer = regions.exit(self.region, exit.from, exit.edge);
                ir.ret(self.ci64(super::super::engine::kernel::COOP_LEAVE | outer as u64));
            }
        }
    }
}
