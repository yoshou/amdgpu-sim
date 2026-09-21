pub(super) mod memory;
pub(super) mod ops;
pub(super) mod prepare;
pub(super) mod region;
pub(super) mod wave;
pub(super) mod wmma;
pub(super) mod yields;

use std::collections::BTreeMap;

use super::analysis::Access;
use super::ir::{Cvt, Env, IntOp, Op, Ty, ValueId, *};
use super::native::{Atomic, BasicBlock, Builder, Type, Value};
use ops::Emitter;
use std::rc::Rc;

pub(super) struct Cluster {
    pub members: usize,
    pub lo: i64,
    pub span: u32,
    pub tile: u32,
}

pub(in crate::rdna_spmd) const DONE: u64 = u64::MAX;

pub(in crate::rdna_spmd) const ENTER: u64 = 1 << 32;

pub(in crate::rdna_spmd) const LEAVE: u64 = 1 << 33;

pub(in crate::rdna_spmd) const YIELD: &str = "amdgpu_sim_fiber_yield_values";

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

pub(super) struct Prepared {
    pub registry: std::sync::Arc<super::ir::DialectRegistry>,
    pub ir: VerifiedFunc,
    pub inputs: Vec<Parameter>,
    pub width: u32,
    pub uniform: Vec<bool>,

    pub holds_a_lane: Vec<bool>,
    pub accesses: Rc<Vec<Access>>,
    pub shapes: Vec<memory::Shape>,
    pub clusters: BTreeMap<usize, Cluster>,
    pub yields: BTreeMap<u64, yields::YieldValues>,
    pub groups: Vec<Vec<u64>>,
    pub min_private_bytes: usize,
    pub num_vgprs: usize,
}

impl Prepared {
    pub fn resume_layouts(&self) -> Vec<Vec<yields::YieldValues>> {
        self.groups
            .iter()
            .map(|g| g.iter().map(|p| self.yields[p].clone()).collect())
            .collect()
    }
    pub fn resume_index(&self, provenance: u64) -> usize {
        self.groups
            .iter()
            .position(|g| g[0] == provenance)
            .expect("scheduled effect lacks a yield layout")
    }
    pub fn group_at(&self, provenance: u64) -> Option<&[u64]> {
        self.groups
            .iter()
            .find(|g| g[0] == provenance)
            .map(|g| g.as_slice())
    }
}

pub(super) fn reverse_postorder(f: &Func) -> Vec<BlockId> {
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

const UNDEFINED: Value = Value::from_raw(std::ptr::null_mut());

pub(super) struct Cg<'a> {
    p: &'a Prepared,
    ir: Builder,
    func: Value,
    em: Emitter,
    sem: Emitter,
    values: Vec<Value>,
    vectors: Vec<Value>,
    scalars: Vec<Value>,
    bbs: BTreeMap<BlockId, BasicBlock>,
    phis: BTreeMap<BlockId, Vec<Value>>,
    incoming: BTreeMap<BlockId, Vec<(BasicBlock, Vec<Value>)>>,
    param_scalar: Vec<bool>,
    types: Vec<Ty>,
    definitions: Vec<Option<Op>>,
    access_at: BTreeMap<(BlockId, usize), usize>,
    skip: std::collections::BTreeSet<(BlockId, usize)>,
    current: BlockId,
    sgprs_p: Value,
    vgprs_p: Value,
    scratch_base_scalar: Value,
    scratch_vec: Value,
    lds_base: Value,
    regions: &'a region::Regions,
    region: usize,
    frame: Value,
    loaded_pairs: BTreeMap<(ValueId, ValueId), Value>,
    valid_mask: Value,
    lane_base: Value,
    yield_frame: Value,
    sink: Value,
    store_sink: Value,
    tile_sink: Value,
}

pub(super) struct RegionCode {
    pub address: u64,
    pub children: Vec<usize>,
    pub blocks: std::collections::BTreeSet<BlockId>,
}

pub(super) struct Compiled {
    pub code: super::native::jit::NativeCode,
    pub regions: Vec<RegionCode>,
    pub frame_words: usize,
}

struct Shared {
    counts: Option<(String, Value, BTreeMap<BlockId, usize>)>,
}

impl Shared {
    fn new(ir: Builder, f: &Func) -> Self {
        let counts = std::env::var("AMDGPU_SIM_BLOCK_COUNTS").ok().map(|path| {
            let ty = ir.i64().array(f.blocks.len() as u64);
            let global = ir.add_global("block_counts", ty);
            global.set_initializer(ty.null());
            let index: BTreeMap<BlockId, usize> = f
                .blocks
                .keys()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();
            let text: String = f
                .blocks
                .keys()
                .map(|id| format!("{} b{:x}\n", index[id], id.0))
                .collect();
            std::fs::write(format!("{path}.blocks"), text).unwrap();
            (path, global, index)
        });
        Self {
            counts,
        }
    }

    fn finish(
        self,
        native: super::native::jit::Module,
        symbol: &str,
        runtime: &[(&str, u64)],
    ) -> super::native::jit::NativeCode {
        let mut code = native.optimize().compile(symbol, runtime);
        if let Some((path, _, index)) = self.counts {
            code.block_counts = Some((path, index.len()));
        }
        code
    }
}

pub(super) fn compile_regions(
    p: &Prepared,
    lowerings: std::sync::Arc<ops::Lowerings>,
    name: &str,
    runtime: &[(&str, u64)],
) -> Compiled {
    let regions = region::Regions::new(p);
    let native = super::native::jit::Module::new(name);
    let shared = Shared::new(native.builder(), p.ir.func());
    let symbols: Vec<String> = (0..regions.entries.len())
        .map(|r| format!("kernel_{r}"))
        .collect();
    for (r, symbol) in symbols.iter().enumerate() {
        emit_function(p, &lowerings, native.builder(), symbol, (&regions, r), &shared);
    }
    let code = shared.finish(native, &symbols[0], runtime);
    let compiled = symbols
        .iter()
        .enumerate()
        .map(|(r, symbol)| RegionCode {
            address: code.lookup(symbol),
            children: regions.children[r].clone(),
            blocks: regions.own(r),
        })
        .collect();
    Compiled {
        code,
        regions: compiled,
        frame_words: regions.frame_words,
    }
}

fn emit_function(
    p: &Prepared,
    lowerings: &std::sync::Arc<ops::Lowerings>,
    ir: Builder,
    symbol: &str,
    scope: (&region::Regions, usize),
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
    fn regs(&self) -> super::ir::Registers {
        self.p.registry.registers()
    }

    fn width(&self) -> u32 {
        self.p.width
    }
    fn ci32(&self, v: u32) -> Value {
        self.ir.ci32(v)
    }
    fn ci64(&self, v: u64) -> Value {
        self.ir.ci64(v)
    }
    fn vec_ty(&self, scalar: Type) -> Type {
        scalar.vector(self.p.width)
    }
    fn splat(&self, v: Value) -> Value {
        self.ir.splat(v, self.p.width)
    }
    fn mask_to_vec(&self, word: Value) -> Value {
        let bits = self.ir.trunc(word, self.ir.int(self.p.width));
        self.ir.bitcast(bits, self.ir.i1().vector(self.p.width))
    }
    fn vec_to_mask(&self, v: Value) -> Value {
        let bits = self.ir.bitcast(v, self.ir.int(self.p.width));
        self.ir.zext(bits, self.ir.i32())
    }
    fn describe(&self, v: ValueId) -> String {
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
                        super::ir::print::inst(&self.p.registry, &f.types, inst),
                        self.current.0,
                        self.skip.contains(&(id, index)),
                        accesses
                    );
                }
            }
        }
        "undefined".into()
    }
    fn vector(&mut self, v: ValueId) -> Value {
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
    fn scalar(&mut self, v: ValueId) -> Value {
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
    fn shaped(&mut self, v: ValueId, scalar: bool) -> Value {
        if scalar {
            self.scalar(v)
        } else {
            self.vector(v)
        }
    }
    fn define(&mut self, v: ValueId, value: Value) {
        self.values[v.0] = value;
        self.vectors[v.0] = UNDEFINED;
        self.scalars[v.0] = UNDEFINED;
    }

    fn register_slot(&self, file: Value, index: u32) -> Value {
        self.ir.gep(self.ir.i32(), file, &[self.ci32(index)])
    }

    fn load_entry(&mut self) {
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

    fn begin_block(&mut self, id: BlockId, block: &Block) {
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

    fn finish_phis(&mut self) {
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

    fn edge_args(&mut self, edge: &Edge) -> Vec<Value> {
        let f = self.p.ir.func();
        let params: Vec<_> = f.blocks[&edge.dst].params.iter().map(|p| p.0).collect();
        edge.args
            .iter()
            .zip(params)
            .map(|(&arg, param)| self.shaped(arg, self.param_scalar[param.0]))
            .collect()
    }

    fn branch_to(&mut self, from: BlockId, ordinal: usize, edge: &Edge) -> BasicBlock {
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

    fn emit_term(&mut self, id: BlockId, block: &Block) {
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

    fn lane_base_word(&self, word: Value) -> Value {
        self.ir.lshr(word, self.lane_base)
    }

    fn any_of_word(&mut self, input: ValueId) -> Option<Value> {
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

    fn emit_inst(&mut self, id: BlockId, index: usize, inst: &Inst) {
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

    fn emit_core(&mut self, value: ValueId, ty: Ty, op: Op) {
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
