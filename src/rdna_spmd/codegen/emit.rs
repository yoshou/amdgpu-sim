use super::ops::{Emitter, Lowerings};
use super::region::Regions;
use super::{Prepared, Shared, DONE, LEAVE};
use crate::rdna_spmd::ir::{Cvt, Env, IntOp, Op, Ty, ValueId, *};
use crate::rdna_spmd::native::{Atomic, BasicBlock, Builder, Type, Value};
use std::collections::BTreeMap;

pub(super) fn lane_word(
    definitions: &[Option<Op>],
    uniform: &[bool],
    shifted: ValueId,
) -> Option<ValueId> {
    match definitions[shifted.0] {
        Some(Op::Int(IntOp::LShr, word, lane))
            if matches!(definitions[lane.0], Some(Op::Env(Env::LaneId))) && uniform[word.0] =>
        {
            Some(word)
        }
        _ => None,
    }
}

pub(super) fn queried_word(
    definitions: &[Option<Op>],
    uniform: &[bool],
    input: ValueId,
) -> Option<(ValueId, bool)> {
    let (bit, valid) = match definitions[input.0] {
        Some(Op::Int(IntOp::And, a, b))
            if matches!(definitions[b.0], Some(Op::Env(Env::ValidLane))) =>
        {
            (a, true)
        }
        Some(Op::Int(IntOp::And, a, b))
            if matches!(definitions[a.0], Some(Op::Env(Env::ValidLane))) =>
        {
            (b, true)
        }
        _ => (input, false),
    };
    let Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) = definitions[bit.0] else {
        return None;
    };
    Some((lane_word(definitions, uniform, shifted)?, valid))
}

fn reverse_postorder(f: &Func) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut visited = std::collections::BTreeSet::new();
    let mut stack: Vec<(BlockId, usize)> = vec![(f.entry, 0)];
    visited.insert(f.entry);
    while let Some((id, next)) = stack.last_mut() {
        let edges: Vec<&Edge> = f.blocks[id].term.edges().collect();
        if *next < edges.len() {
            let dst = edges[*next].dst;
            *next += 1;
            if visited.insert(dst) {
                stack.push((dst, 0));
            }
        } else {
            order.push(*id);
            stack.pop();
        }
    }
    order.reverse();
    for &id in f.blocks.keys() {
        if visited.insert(id) {
            order.push(id);
        }
    }
    order
}

pub(super) const UNDEFINED: Value = Value::from_raw(std::ptr::null_mut());

pub(super) struct Cg<'a> {
    pub(super) p: &'a Prepared,
    pub(super) ir: Builder,
    pub(super) func: Value,
    pub(super) em: Emitter,
    pub(super) sem: Emitter,
    pub(super) values: Vec<Value>,
    pub(super) vectors: Vec<Value>,
    pub(super) scalars: Vec<Value>,
    pub(super) bbs: BTreeMap<BlockId, BasicBlock>,
    pub(super) phis: BTreeMap<BlockId, Vec<Value>>,
    pub(super) incoming: BTreeMap<BlockId, Vec<(BasicBlock, Vec<Value>)>>,
    pub(super) param_scalar: Vec<bool>,
    pub(super) types: Vec<Ty>,
    pub(super) definitions: Vec<Option<Op>>,
    pub(super) access_at: BTreeMap<(BlockId, usize), usize>,
    pub(super) skip: std::collections::BTreeSet<(BlockId, usize)>,
    pub(super) current: BlockId,
    pub(super) sgprs_p: Value,
    pub(super) vgprs_p: Value,
    pub(super) scratch_base_scalar: Value,
    pub(super) scratch_vec: Value,
    pub(super) lds_base: Value,
    pub(super) regions: &'a Regions,
    pub(super) region: usize,
    pub(super) frame: Value,
    pub(super) loaded_pairs: BTreeMap<(ValueId, ValueId), Value>,
    pub(super) valid_mask: Value,
    pub(super) lane_base: Value,
    pub(super) yield_frame: Value,
    pub(super) sink: Value,
    pub(super) store_sink: Value,
    pub(super) tile_sink: Value,
}

pub(super) fn emit_function(
    p: &Prepared,
    lowerings: &std::sync::Arc<Lowerings>,
    ir: Builder,
    symbol: &str,
    scope: (&Regions, usize),
    shared: &Shared,
) {
    let (i32t, i64t, ptr) = (ir.i32(), ir.i64(), ir.ptr());
    let func = ir.add_function(
        symbol,
        i64t.function(&[ptr, ptr, i64t, i64t, i64t, i64t, ptr, i32t, ptr]),
    );
    let entry = ir.append_block(func, "entry");
    ir.position_at_end(entry);
    let sgprs_p = func.param(0);
    let vgprs_p = func.param(1);
    let scratch_base = func.param(2);
    let scratch_stride = func.param(3);
    let lds_base = func.param(4);
    let lane_base = func.param(5);
    let valid_mask = func.param(7);
    let width_lanes = p.width;
    let scratch_base_scalar = {
        let aperture = ir.and(scratch_base, ir.ci64(0xffff_ffff_0000_0000));
        let sized = ir.icmp(IntPred::Ne, scratch_stride, ir.ci64(0));
        ir.select(sized, aperture, scratch_base)
    };
    let sink = ir.array_alloca(i32t, ir.ci32(10), "");
    let mut em = Emitter::new(ir, Some(p.width), p.registry.clone(), lowerings.clone());
    em.state = lowerings.state(&em, sink);
    let mut sem = Emitter::new(ir, None, p.registry.clone(), lowerings.clone());
    sem.state = lowerings.state(&sem, sink);
    em.scratch = Some((scratch_base_scalar, scratch_stride));
    sem.scratch = em.scratch;
    let cells = p
        .yields
        .values()
        .map(|l| l.base + l.cells())
        .max()
        .unwrap_or(0);
    let yield_frame = if cells == 0 {
        ptr.null()
    } else {
        ir.alloca(
            i32t.array(cells as u64 * width_lanes as u64),
            "yield.values",
        )
        .set_alignment(64)
    };
    let f = p.ir.func();
    let mut definitions = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core { value, op, .. } = inst {
                definitions[value.0] = Some(*op);
            }
        }
    }
    let mut access_at = BTreeMap::new();
    let mut skip = std::collections::BTreeSet::new();
    for (index, access) in p.accesses.iter().enumerate() {
        access_at.insert((access.block, access.effects[0]), index);
        for &e in &access.effects[1..] {
            skip.insert((access.block, e));
        }
    }
    for (&start, cluster) in &p.clusters {
        for member in start + 1..start + cluster.members {
            for &e in &p.accesses[member].effects {
                skip.insert((p.accesses[member].block, e));
            }
        }
    }
    let mut cg = Cg {
        p,
        ir,
        func,
        em,
        sem,
        values: vec![UNDEFINED; f.types.len()],
        vectors: vec![UNDEFINED; f.types.len()],
        scalars: vec![UNDEFINED; f.types.len()],
        bbs: BTreeMap::new(),
        phis: BTreeMap::new(),
        incoming: BTreeMap::new(),
        param_scalar: p.uniform.clone(),
        types: f.types.clone(),
        definitions,
        access_at,
        skip,
        current: f.entry,
        sgprs_p,
        vgprs_p,
        scratch_base_scalar,
        scratch_vec: scratch_base,
        lds_base,
        regions: scope.0,
        region: scope.1,
        frame: func.param(8),
        loaded_pairs: BTreeMap::new(),
        valid_mask,
        lane_base: ir.trunc(lane_base, i32t),
        yield_frame,
        sink,
        store_sink: ir.alloca(i64t, "store_sink"),
        tile_sink: ir.array_alloca(i32t, ir.ci32(64), "tile_sink"),
    };
    {
        let lanes: Vec<Value> = (0..p.width).map(|k| ir.ci64(k as u64)).collect();
        let lane_idx = ir.const_vector(&lanes);
        let base_v = cg.splat(scratch_base);
        let stride_v = cg.splat(scratch_stride);
        let lane_base_v = cg.splat(lane_base);
        let scratch_lane = ir.add(lane_base_v, lane_idx);
        let off = ir.mul(scratch_lane, stride_v);
        cg.scratch_vec = ir.add(base_v, off);
    }
    let packet_valid = cg.lane_base_word(valid_mask);
    let valid_vec = cg.mask_to_vec(packet_valid);
    cg.em.valid_lane = Some(valid_vec);
    cg.em.set_lane_id(lane_base);
    cg.sem.set_lane_id(lane_base);
    cg.sem.valid_lane = Some(ir.icmp(IntPred::Ne, ir.and(packet_valid, ir.ci32(1)), ir.ci32(0)));
    let (regions, r) = scope;
    let own = regions.own(r);
    let mine = |id: &BlockId| own.contains(id);
    for &id in f.blocks.keys().filter(|id| mine(id)) {
        cg.bbs
            .insert(id, ir.append_block(func, &format!("b{:x}", id.0)));
    }
    for &child in &regions.children[r] {
        let entry = regions.entries[child];
        cg.bbs
            .insert(entry, ir.append_block(func, &format!("enter{:x}", entry.0)));
    }
    if r > 0 {
        cg.enter_region();
    } else {
        cg.load_entry();
    }
    ir.br(cg.bbs[&regions.entries[r]]);
    for id in reverse_postorder(f).into_iter().filter(|id| mine(id)) {
        let block = &f.blocks[&id];
        ir.position_at_end(cg.bbs[&id]);
        cg.current = id;
        cg.begin_block(id, block);
        if let Some((_, global, index)) = &shared.counts {
            let slot = ir.gep(i64t, *global, &[ir.ci64(index[&id] as u64)]);
            ir.atomic_add(slot, ir.ci64(1), Atomic::Monotonic);
        }
        for (index, inst) in block.insts.iter().enumerate() {
            if cg.skip.contains(&(id, index)) {
                continue;
            }
            cg.emit_inst(id, index, inst);
        }
        cg.emit_term(id, block);
    }
    for (ordinal, &child) in regions.children[r].iter().enumerate() {
        cg.emit_child(ordinal, child);
    }
    cg.finish_phis();
}

impl<'a> Cg<'a> {
    pub(super) fn regs(&self) -> crate::rdna_spmd::ir::Registers {
        self.p.registry.registers()
    }

    pub(super) fn width(&self) -> u32 {
        self.p.width
    }
    pub(super) fn ci32(&self, v: u32) -> Value {
        self.ir.ci32(v)
    }
    pub(super) fn ci64(&self, v: u64) -> Value {
        self.ir.ci64(v)
    }
    pub(super) fn vec_ty(&self, scalar: Type) -> Type {
        scalar.vector(self.p.width)
    }
    pub(super) fn splat(&self, v: Value) -> Value {
        self.ir.splat(v, self.p.width)
    }
    pub(super) fn mask_to_vec(&self, word: Value) -> Value {
        let bits = self.ir.trunc(word, self.ir.int(self.p.width));
        self.ir.bitcast(bits, self.ir.i1().vector(self.p.width))
    }
    pub(super) fn vec_to_mask(&self, v: Value) -> Value {
        let bits = self.ir.bitcast(v, self.ir.int(self.p.width));
        self.ir.zext(bits, self.ir.i32())
    }
    pub(super) fn describe(&self, v: ValueId) -> String {
        let f = self.p.ir.func();
        for (&id, block) in &f.blocks {
            if let Some(index) = block.params.iter().position(|p| p.0 == v) {
                return format!("parameter {index} of b{:x}", id.0);
            }
            for (index, inst) in block.insts.iter().enumerate() {
                let defined = match inst {
                    Inst::Core { value, .. } | Inst::Packet { output: value, .. } => *value == v,
                    Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => {
                        outputs.iter().any(|o| o.0 == v)
                    }
                };
                if defined {
                    let accesses: Vec<String> = self
                        .p
                        .accesses
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| {
                            a.block == id && a.effects.contains(&index)
                                || a.block == id && (a.start..a.end).contains(&index)
                        })
                        .map(|(k, a)| {
                            format!(
                                "access {k} effects {:?} shape {:?} cluster {:?} results {:?}",
                                a.effects,
                                self.p.shapes[k],
                                self.p.clusters.get(&k).map(|c| c.members),
                                a.results
                            )
                        })
                        .collect();
                    return format!(
                        "b{:x}:{index} {} (current b{:x}, skipped {}) {:?}",
                        id.0,
                        crate::rdna_spmd::ir::print::inst(&self.p.registry, &f.types, inst),
                        self.current.0,
                        self.skip.contains(&(id, index)),
                        accesses
                    );
                }
            }
        }
        "undefined".into()
    }
    pub(super) fn vector(&mut self, v: ValueId) -> Value {
        let value = self.values[v.0];
        assert!(
            !value.is_null(),
            "undefined SSA value v{}: {}",
            v.0,
            self.describe(v)
        );
        if value.is_vector() {
            return value;
        }
        if !self.vectors[v.0].is_null() {
            return self.vectors[v.0];
        }
        let out = self.splat(value);
        let out = if self.types[v.0] == Ty::I1 {
            out
        } else {
            out
        };
        self.vectors[v.0] = out;
        out
    }
    pub(super) fn scalar(&mut self, v: ValueId) -> Value {
        let value = self.values[v.0];
        assert!(
            !value.is_null(),
            "undefined SSA value v{}: {}",
            v.0,
            self.describe(v)
        );
        if !value.is_vector() {
            return value;
        }
        if !self.scalars[v.0].is_null() {
            return self.scalars[v.0];
        }
        let out = self.ir.extract_at(value, 0);
        self.scalars[v.0] = out;
        out
    }
    pub(super) fn shaped(&mut self, v: ValueId, scalar: bool) -> Value {
        if scalar {
            self.scalar(v)
        } else {
            self.vector(v)
        }
    }
    pub(super) fn define(&mut self, v: ValueId, value: Value) {
        self.values[v.0] = value;
        self.vectors[v.0] = UNDEFINED;
        self.scalars[v.0] = UNDEFINED;
    }

    pub(super) fn register_slot(&self, file: Value, index: u32) -> Value {
        self.ir.gep(self.ir.i32(), file, &[self.ci32(index)])
    }

    pub(super) fn load_entry(&mut self) {
        let f = self.p.ir.func();
        let entry = &f.blocks[&f.entry];
        let ir = self.ir;
        for (index, &(id, ty)) in entry.params.iter().enumerate() {
            let input = &self.p.inputs[index];
            let value = match input.source {
                ParameterSource::Vgpr(r) => {
                    let w = self.p.width;
                    ir.load(ir.i32().vector(w), self.register_slot(self.vgprs_p, r * w))
                        .set_alignment(4)
                }
                ParameterSource::Sgpr(r) => {
                    if r == self.regs().null {
                        self.ci32(0)
                    } else {
                        ir.load(ir.i32(), self.register_slot(self.sgprs_p, r))
                    }
                }
                ParameterSource::MaskBit(r) => {
                    let word = ir.load(ir.i32(), self.register_slot(self.sgprs_p, r));
                    let word = if r == self.regs().exec {
                        ir.and(word, self.lane_base_word(self.valid_mask))
                    } else {
                        word
                    };
                    self.mask_to_vec(word)
                }
                ParameterSource::Scc => {
                    let word = ir.load(
                        ir.i32(),
                        self.register_slot(self.sgprs_p, self.regs().scc_slot),
                    );
                    ir.icmp(IntPred::Ne, word, self.ci32(0))
                }
            };
            assert_eq!(ty, input.ty);
            self.define(id, value);
        }
    }

    pub(super) fn begin_block(&mut self, id: BlockId, block: &Block) {
        let f = self.p.ir.func();
        if id == f.entry {
            return;
        }
        let mut phis = Vec::new();
        for &(v, ty) in &block.params {
            let t = if self.param_scalar[v.0] {
                self.sem.ty(ty)
            } else {
                self.em.ty(ty)
            };
            let phi = self.ir.phi(t);
            phis.push(phi);
            self.define(v, phi);
        }
        self.phis.insert(id, phis);
    }

    pub(super) fn finish_phis(&mut self) {
        for (&id, phis) in &self.phis {
            let incoming = self.incoming.get(&id).cloned().unwrap_or_default();
            for (index, &phi) in phis.iter().enumerate() {
                let edges: Vec<(Value, BasicBlock)> = incoming
                    .iter()
                    .map(|(bb, args)| (args[index], *bb))
                    .collect();
                if edges.is_empty() {
                    phi.replace_all_uses_with(phi.ty().undef());
                    phi.erase();
                    continue;
                }
                phi.add_incoming(&edges);
            }
        }
    }

    pub(super) fn edge_args(&mut self, edge: &Edge) -> Vec<Value> {
        let f = self.p.ir.func();
        let params: Vec<_> = f.blocks[&edge.dst].params.iter().map(|p| p.0).collect();
        edge.args
            .iter()
            .zip(params)
            .map(|(&arg, param)| self.shaped(arg, self.param_scalar[param.0]))
            .collect()
    }

    pub(super) fn branch_to(&mut self, from: BlockId, ordinal: usize, edge: &Edge) -> BasicBlock {
        if !self.regions.holds(self.region, edge.dst) {
            return self.leave_region(from, ordinal, edge);
        }
        let args = self.edge_args(edge);
        let from = self.ir.insert_block();
        self.incoming
            .entry(edge.dst)
            .or_default()
            .push((from, args));
        self.bbs[&edge.dst]
    }

    pub(super) fn emit_term(&mut self, id: BlockId, block: &Block) {
        match &block.term {
            Term::Br(edge) => {
                let bb = self.branch_to(id, 0, edge);
                self.ir.br(bb);
            }
            Term::CondBr { cond, yes, no } => {
                let c = self.scalar(*cond);
                let yes_bb = self.branch_to(id, 0, yes);
                let no_bb = self.branch_to(id, 1, no);
                self.ir.cond_br(c, yes_bb, no_bb);
            }
            Term::Ret(args) => {
                assert!(args.is_empty(), "a packet program returns no register");
                let left = match self.region {
                    0 => DONE,
                    _ => {
                        let exit = self.regions.exit(self.region, id, None);
                        LEAVE | exit as u64
                    }
                };
                self.ir.ret(self.ci64(left));
            }
        }
    }

    pub(super) fn lane_base_word(&self, word: Value) -> Value {
        self.ir.lshr(word, self.lane_base)
    }

    pub(super) fn any_of_word(&mut self, input: ValueId) -> Option<Value> {
        let w = self.p.width;
        let (word, valid) = queried_word(&self.definitions, &self.p.uniform, input)?;
        let word = self.scalar(word);
        let word = self.lane_base_word(word);
        let mut bits = self.ir.and(word, self.ci32(((1u64 << w) - 1) as u32));
        if valid {
            let packet = self.lane_base_word(self.valid_mask);
            bits = self.ir.and(bits, packet);
        }
        Some(self.ir.icmp(IntPred::Ne, bits, self.ci32(0)))
    }

    pub(super) fn emit_inst(&mut self, id: BlockId, index: usize, inst: &Inst) {
        match inst {
            Inst::Core { value, ty, op } => self.emit_core(*value, *ty, *op),
            Inst::Target {
                op, args, outputs, ..
            } => {
                let values: Vec<ValueId> = args.values().to_vec();
                let mut table = self.values.clone();
                for a in &values {
                    table[a.0] = self.vector(*a);
                }
                let results = self.em.target(*op, *args, &table);
                for (&(out, _), result) in outputs.iter().zip(results) {
                    self.define(out, result);
                }
            }
            Inst::Packet { op, input, output } => {
                if *op == PacketOp::Any {
                    if let Some(result) = self.any_of_word(*input) {
                        self.define(*output, result);
                        return;
                    }
                }
                let bits = {
                    let v = self.vector(*input);
                    self.vec_to_mask(v)
                };
                let result = if *op == PacketOp::Any {
                    self.ir.icmp(IntPred::Ne, bits, self.ci32(0))
                } else {
                    bits
                };
                self.define(*output, result);
            }
            Inst::Effect { provenance, op, .. } => match op {
                EffectOp::Memory { .. } => {
                    let access = self.access_at[&(id, index)];
                    if let Some(cluster) = self.p.clusters.get(&access) {
                        let members: Vec<usize> = (access..access + cluster.members).collect();
                        self.emit_cluster(&members, cluster);
                    } else {
                        self.emit_memory(access);
                    }
                }
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                    let Some(group) = self.p.group_at(*provenance) else {
                        return;
                    };
                    let mut members = Vec::with_capacity(group.len());
                    let mut wanted = group.iter();
                    let mut next = wanted.next();
                    for inst in &self.p.ir.func().blocks[&id].insts[index..] {
                        let Some(&want) = next else { break };
                        let Inst::Effect {
                            provenance: at,
                            inputs,
                            outputs,
                            ..
                        } = inst
                        else {
                            continue;
                        };
                        if *at != want {
                            continue;
                        }
                        members.push((want, inputs.clone(), outputs.clone()));
                        next = wanted.next();
                    }
                    assert_eq!(members.len(), group.len(), "a yield group lost a member");
                    self.emit_yield(&members);
                }
            },
        }
    }

    pub(super) fn emit_core(&mut self, value: ValueId, ty: Ty, op: Op) {
        if let Op::Pack64(a, b) = op {
            if let Some(&loaded) = self.loaded_pairs.get(&(a, b)) {
                let result = self.ir.bitcast(loaded, self.vec_ty(self.ir.i64()));
                self.define(value, result);
                return;
            }
        }
        if let Op::Convert(Cvt::Trunc, Ty::I1, shift) = op {
            if let Some(word) = lane_word(&self.definitions, &self.p.uniform, shift) {
                let w = self.scalar(word);
                let w = self.lane_base_word(w);
                let out = self.mask_to_vec(w);
                self.define(value, out);
                return;
            }
        }
        if let Op::Select(_, a, b) = op {
            if a == b {
                let v = self.values[a.0];
                self.define(value, v);
                return;
            }
        }
        let mut args = vec![];
        op.map(|id| {
            if !args.contains(&id) {
                args.push(id);
            }
            id
        });
        let scalar = !matches!(op, Op::Env(Env::LaneId | Env::ValidLane))
            && args.iter().all(|a| !self.values[a.0].is_vector());
        let mut table = self.values.clone();
        for a in &args {
            table[a.0] = self.shaped(*a, scalar);
        }
        let result = if scalar {
            self.sem.op(ty, op, &table)
        } else {
            self.em.op(ty, op, &table)
        };
        let result = if matches!(op, Op::Convert(Cvt::Trunc, Ty::I1, _)) {
            result
        } else {
            result
        };
        self.define(value, result);
    }
}
