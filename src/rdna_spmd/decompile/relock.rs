use super::facts::{operands, outputs, Facts};
use super::loops::Loops;
use crate::rdna_spmd::compiler::exec_index;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::LiftedFunction;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum State {
    Register(usize, Ty),
    Escaped(ValueId),
    Mask,
    Waiting(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Node {
    Entry,
    Guard(usize),
    Body(usize),
    End(usize),
    Check(usize, usize),
    Exit,
}

type Values = BTreeMap<State, ValueId>;

struct Relock<'q> {
    q: &'q Func,
    out: Func,
    layout: Vec<BlockId>,
    position: BTreeMap<BlockId, usize>,
    closing: Vec<Vec<usize>>,
    ids: BTreeMap<Node, BlockId>,
    shapes: BTreeMap<Node, Vec<State>>,
    defs: BTreeMap<ValueId, Op>,
    escaping: BTreeSet<ValueId>,
}

fn escaping(q: &Func) -> BTreeSet<ValueId> {
    let mut out = BTreeSet::new();
    for block in q.blocks.values() {
        let mut local: BTreeSet<ValueId> = block.params.iter().map(|p| p.0).collect();
        let mut used = Vec::new();
        for inst in &block.insts {
            used.extend(operands(inst));
            local.extend(outputs(inst));
        }
        if let Term::CondBr { cond, .. } = &block.term {
            used.push(*cond);
        }
        for edge in block.term.edges() {
            used.extend(edge.args.iter().copied());
        }
        out.extend(used.into_iter().filter(|v| !local.contains(v)));
    }
    out
}

pub(crate) fn relock(lane: &LiftedFunction) -> LiftedFunction {
    let q = &lane.ir;
    let facts = Facts::new(q, &lane.parameter_inputs);
    let loops = Loops::new(q, &facts).expect("a lane program is reducible");
    let n = facts.order.len();
    let (by_rank, ranges) = loops.layout(n);
    let layout: Vec<BlockId> = by_rank.iter().map(|&r| facts.order[r]).collect();
    let position = layout.iter().enumerate().map(|(p, &b)| (b, p)).collect();
    let mut closing = vec![Vec::new(); n];
    let mut by_length = ranges;
    by_length.sort_by_key(|&(first, last)| last - first);
    for (first, last) in by_length {
        closing[last].push(first);
    }
    let mut relock = Relock {
        q,
        out: Func {
            entry: BlockId(0),
            blocks: BTreeMap::new(),
            types: Vec::new(),
        },
        layout,
        position,
        closing,
        ids: BTreeMap::new(),
        shapes: BTreeMap::new(),
        defs: BTreeMap::new(),
        escaping: escaping(q),
    };
    relock.shape();
    relock.emit(exec_index(&lane.parameter_inputs, &lane.registry));
    let mut out = relock.out;
    out.compact();
    LiftedFunction {
        registry: lane.registry.clone(),
        ir: out,
        parameter_inputs: lane.parameter_inputs.clone(),
        revision: lane.revision + 1,
    }
}

impl Relock<'_> {
    fn next(&self, p: usize, i: usize) -> Node {
        if i < self.closing[p].len() {
            Node::Check(p, i)
        } else if p + 1 < self.layout.len() {
            Node::Guard(p + 1)
        } else {
            Node::Exit
        }
    }

    fn flow(&self, node: Node, waiting: &BTreeSet<usize>) -> Vec<(Node, BTreeSet<usize>)> {
        let without = |p: usize| {
            let mut rest = waiting.clone();
            rest.remove(&p);
            rest
        };
        match node {
            Node::Entry => vec![(Node::Guard(0), waiting.clone())],
            Node::Guard(p) => vec![(Node::Body(p), without(p)), (Node::End(p), without(p))],
            Node::Body(p) => {
                let mut out = waiting.clone();
                out.extend(
                    self.q.blocks[&self.layout[p]]
                        .term
                        .edges()
                        .map(|e| self.position[&e.dst]),
                );
                vec![(Node::End(p), out)]
            }
            Node::End(p) => vec![(self.next(p, 0), waiting.clone())],
            Node::Check(p, i) => {
                let header = self.closing[p][i];
                vec![
                    (Node::Guard(header), waiting.clone()),
                    (self.next(p, i + 1), without(header)),
                ]
            }
            Node::Exit => vec![],
        }
    }

    fn shape(&mut self) {
        let start = self.position[&self.q.entry];
        let mut entering: BTreeMap<Node, BTreeSet<usize>> = BTreeMap::new();
        entering.insert(Node::Entry, [start].into());
        let mut work = vec![Node::Entry];
        while let Some(node) = work.pop() {
            let waiting = entering[&node].clone();
            for (succ, out) in self.flow(node, &waiting) {
                let known = entering.contains_key(&succ);
                let set = entering.entry(succ).or_default();
                let before = set.len();
                set.extend(out);
                if !known || set.len() != before {
                    work.push(succ);
                }
            }
        }
        let registers: BTreeSet<(usize, Ty)> = self
            .q
            .blocks
            .values()
            .flat_map(|b| b.params.iter().enumerate().map(|(k, &(_, ty))| (k, ty)))
            .collect();
        let mut common: Vec<State> = registers
            .into_iter()
            .map(|(k, ty)| State::Register(k, ty))
            .collect();
        common.push(State::Mask);
        common.extend(self.escaping.iter().map(|&v| State::Escaped(v)));
        for (next, (&node, waiting)) in entering.iter().enumerate() {
            self.ids.insert(node, BlockId(next));
            let mut shape = common.clone();
            shape.extend(waiting.iter().map(|&p| State::Waiting(p)));
            self.shapes.insert(node, shape);
        }
    }

    fn ty(&self, s: State) -> Ty {
        match s {
            State::Register(_, ty) => ty,
            State::Escaped(v) => self.q.types[v.0],
            State::Mask | State::Waiting(_) => Ty::I1,
        }
    }

    fn open(&mut self, node: Node) -> Values {
        let mut params = Vec::new();
        let mut state = BTreeMap::new();
        for s in self.shapes[&node].clone() {
            let ty = self.ty(s);
            let v = self.out.value(ty);
            params.push((v, ty));
            state.insert(s, v);
        }
        self.out.blocks.insert(
            self.ids[&node],
            Block {
                params,
                insts: Vec::new(),
                term: Term::Ret(vec![]),
            },
        );
        state
    }

    fn inst(&mut self, node: Node, inst: Inst) {
        let id = self.ids[&node];
        self.out.blocks.get_mut(&id).unwrap().insts.push(inst);
    }

    fn core(&mut self, node: Node, ty: Ty, op: Op) -> ValueId {
        let value = self.out.value(ty);
        self.defs.insert(value, op);
        self.inst(node, Inst::Core { value, ty, op });
        value
    }

    fn masked(&mut self, node: Node, predicate: ValueId, mask: ValueId) -> ValueId {
        match self.defs.get(&predicate) {
            Some(&Op::Int(IntOp::And, first, last)) => {
                let first = self.core(node, Ty::I1, Op::Int(IntOp::And, first, mask));
                self.core(node, Ty::I1, Op::Int(IntOp::And, first, last))
            }
            _ => self.core(node, Ty::I1, Op::Int(IntOp::And, mask, predicate)),
        }
    }

    fn any(&mut self, node: Node, input: ValueId) -> ValueId {
        let output = self.out.value(Ty::I1);
        self.inst(
            node,
            Inst::Packet {
                op: PacketOp::Any,
                input,
                output,
            },
        );
        output
    }

    fn waiting(&mut self, node: Node, state: &Values, p: usize) -> ValueId {
        match state.get(&State::Waiting(p)) {
            Some(&v) => v,
            None => self.core(node, Ty::I1, Op::Const(Ty::I1, 0)),
        }
    }

    fn edge(&mut self, src: Node, dst: Node, state: &Values) -> Edge {
        let shape = self.shapes[&dst].clone();
        let args = shape
            .into_iter()
            .map(|s| match state.get(&s) {
                Some(&v) => v,
                None => {
                    let ty = self.ty(s);
                    self.core(src, ty, Op::Const(ty, 0))
                }
            })
            .collect();
        Edge {
            dst: self.ids[&dst],
            args,
        }
    }

    fn terminate(&mut self, node: Node, term: Term) {
        let id = self.ids[&node];
        self.out.blocks.get_mut(&id).unwrap().term = term;
    }

    fn branch(&mut self, node: Node, cond: ValueId, yes: (Node, &Values), no: (Node, &Values)) {
        let yes = self.edge(node, yes.0, yes.1);
        let no = self.edge(node, no.0, no.1);
        self.terminate(node, Term::CondBr { cond, yes, no });
    }

    fn jump(&mut self, node: Node, dst: Node, state: &Values) {
        let edge = self.edge(node, dst, state);
        self.terminate(node, Term::Br(edge));
    }

    fn emit(&mut self, exec: usize) {
        let nodes: Vec<Node> = self.ids.keys().copied().collect();
        for node in nodes {
            match node {
                Node::Entry => self.entry(exec),
                Node::Guard(p) => {
                    let state = self.open(node);
                    let waiting = self.waiting(node, &state, p);
                    let any = self.any(node, waiting);
                    let mut running = state.clone();
                    running.insert(State::Mask, waiting);
                    running.remove(&State::Waiting(p));
                    self.branch(node, any, (Node::Body(p), &running), (Node::End(p), &state));
                }
                Node::Body(p) => self.body(p),
                Node::End(p) => {
                    let state = self.open(node);
                    self.jump(node, self.next(p, 0), &state);
                }
                Node::Check(p, i) => {
                    let state = self.open(node);
                    let header = self.closing[p][i];
                    let waiting = self.waiting(node, &state, header);
                    let any = self.any(node, waiting);
                    let mut left = state.clone();
                    left.remove(&State::Waiting(header));
                    let onward = self.next(p, i + 1);
                    self.branch(node, any, (Node::Guard(header), &state), (onward, &left));
                }
                Node::Exit => {
                    self.open(node);
                }
            }
        }
    }

    fn entry(&mut self, exec: usize) {
        let entry = &self.q.blocks[&self.q.entry];
        let params: Vec<(ValueId, Ty)> = entry
            .params
            .iter()
            .map(|&(_, ty)| (self.out.value(ty), ty))
            .collect();
        let mut state: Values = params
            .iter()
            .enumerate()
            .map(|(k, &(v, ty))| (State::Register(k, ty), v))
            .collect();
        let held = params[exec].0;
        state.insert(State::Mask, held);
        state.insert(State::Waiting(self.position[&self.q.entry]), held);
        self.out.blocks.insert(
            self.ids[&Node::Entry],
            Block {
                params,
                insts: Vec::new(),
                term: Term::Ret(vec![]),
            },
        );
        self.jump(Node::Entry, Node::Guard(0), &state);
    }

    fn body(&mut self, p: usize) {
        let node = Node::Body(p);
        let q = self.q;
        let block = &q.blocks[&self.layout[p]];
        let mut state = self.open(node);
        let mask = state[&State::Mask];
        let mut map: BTreeMap<ValueId, ValueId> = self
            .escaping
            .iter()
            .map(|&v| (v, state[&State::Escaped(v)]))
            .collect();
        for (k, &(v, ty)) in block.params.iter().enumerate() {
            map.insert(v, state[&State::Register(k, ty)]);
        }
        for inst in &block.insts {
            let inst = match inst {
                Inst::Core { value, ty, op } => {
                    let op = op.map(|v| map[&v]);
                    let out = self.out.value(*ty);
                    self.defs.insert(out, op);
                    map.insert(*value, out);
                    Inst::Core {
                        value: out,
                        ty: *ty,
                        op,
                    }
                }
                Inst::Target {
                    provenance,
                    op,
                    args,
                    outputs,
                } => Inst::Target {
                    provenance: *provenance,
                    op: *op,
                    args: args.map(|v| map[&v]),
                    outputs: self.outputs(outputs, &mut map),
                },
                Inst::Effect {
                    provenance,
                    op,
                    inputs,
                    outputs,
                } => {
                    let EffectOp::Memory { op: memory, .. } = op else {
                        unreachable!("a lane program exchanges nothing between lanes")
                    };
                    let mut inputs: Vec<ValueId> = inputs.iter().map(|v| map[v]).collect();
                    let predicate = match memory {
                        MemoryOp::Load(_) => Some(1),
                        MemoryOp::Store(_) | MemoryOp::AtomicAdd => Some(2),
                        MemoryOp::Fence => None,
                    };
                    if let Some(i) = predicate {
                        inputs[i] = self.masked(node, inputs[i], mask);
                    }
                    Inst::Effect {
                        provenance: *provenance,
                        op: *op,
                        inputs,
                        outputs: self.outputs(outputs, &mut map),
                    }
                }
                Inst::Packet { .. } => unreachable!("a lane program queries no packet"),
            };
            self.inst(node, inst);
        }
        let defined: Vec<ValueId> = block
            .params
            .iter()
            .map(|p| p.0)
            .chain(block.insts.iter().flat_map(outputs))
            .filter(|v| self.escaping.contains(v))
            .collect();
        for v in defined {
            let (new, old) = (map[&v], state[&State::Escaped(v)]);
            let ty = q.types[v.0];
            let kept = self.core(node, ty, Op::Select(mask, new, old));
            state.insert(State::Escaped(v), kept);
        }
        let edges: Vec<(&Edge, ValueId)> = match &block.term {
            Term::Ret(_) => vec![],
            Term::Br(e) => vec![(e, mask)],
            Term::CondBr { cond, yes, no } => {
                let c = map[cond];
                let taken = self.core(node, Ty::I1, Op::Int(IntOp::And, mask, c));
                let one = self.core(node, Ty::I1, Op::Const(Ty::I1, 1));
                let not = self.core(node, Ty::I1, Op::Int(IntOp::Xor, c, one));
                let other = self.core(node, Ty::I1, Op::Int(IntOp::And, mask, not));
                vec![(yes, taken), (no, other)]
            }
        };
        for (edge, lanes) in edges {
            let dst = &q.blocks[&edge.dst];
            for (k, (arg, &(_, ty))) in edge.args.iter().zip(&dst.params).enumerate() {
                let slot = State::Register(k, ty);
                let (new, old) = (map[arg], state[&slot]);
                if new != old {
                    let chosen = self.core(node, ty, Op::Select(lanes, new, old));
                    state.insert(slot, chosen);
                }
            }
            let at = State::Waiting(self.position[&edge.dst]);
            let joined = match state.get(&at) {
                Some(&waiting) => self.core(node, Ty::I1, Op::Int(IntOp::Or, waiting, lanes)),
                None => lanes,
            };
            state.insert(at, joined);
        }
        self.jump(node, Node::End(p), &state);
    }

    fn outputs(&mut self, outputs: &[(ValueId, Ty)], map: &mut BTreeMap<ValueId, ValueId>) -> Vec<(ValueId, Ty)> {
        outputs
            .iter()
            .map(|&(v, ty)| {
                let out = self.out.value(ty);
                map.insert(v, out);
                (out, ty)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::compiler::compile_packet;
    use crate::rdna_spmd::ir::parse;
    use crate::rdna_spmd::program::{Parameter, ParameterSource, Program};
    use std::sync::Arc;

    #[test]
    fn lanes_leaving_a_loop_after_different_trips_keep_what_they_computed() {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        let ir = parse::func(
            &registry,
            "func entry b0
             b0(v0: i32, v1: i32, v2: i32, v3: i1):
               v4: i32 = const i32 0x2
               v5: i32 = int lshr v0, v4
               v6: i32 = const i32 0x7
               v7: i32 = const i32 0x0
               br b1(v5, v7)
             b1(v8: i32, v9: i32):
               v10: i1 = cmp eq v8, v7
               condbr v10, b3(), b2(v8, v9)
             b2(v11: i32, v12: i32):
               v13: i32 = const i32 0x1
               v14: i32 = int sub v11, v13
               v15: i32 = const i32 0x3
               v16: i32 = int add v12, v15
               br b1(v14, v16)
             b3():
               v17: i64 = pack64 v1, v2
               v18: i64 = convert zext i64 v0
               v19: i64 = int add v17, v18
               v20: i32 = int mul v5, v6
               v21: i32 = int add v20, v9
               effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v19, v21, v3)
               ret",
        )
        .unwrap();
        let input = |source: ParameterSource, ty: Ty| Parameter { source, ty };
        let lane = LiftedFunction {
            registry: Arc::new(registry),
            ir,
            parameter_inputs: vec![
                input(ParameterSource::Vgpr(0), Ty::I32),
                input(ParameterSource::Sgpr(0), Ty::I32),
                input(ParameterSource::Sgpr(1), Ty::I32),
                input(ParameterSource::MaskBit(126), Ty::I1),
            ],
            revision: 0,
        };
        let packet = relock(&lane);
        packet.ir.check(&packet.registry).unwrap();
        for width in [1u32, 2, 4, 8, 16, 32] {
            let lanes = width as usize;
            let mut output = vec![0u32; lanes];
            let mut sgprs = [0u32; 128];
            let address = output.as_mut_ptr() as u64;
            sgprs[0] = address as u32;
            sgprs[1] = (address >> 32) as u32;
            let mut vgprs = vec![0u32; 256 * lanes];
            for l in 0..lanes {
                vgprs[l] = l as u32 * 4;
            }
            unsafe {
                compile_packet(Program { function: packet.clone() }, 256, width, None).run(
                    sgprs.as_mut_ptr(),
                    vgprs.as_mut_ptr(),
                    0,
                    0,
                    u32::MAX,
                    0,
                );
            }
            let expected: Vec<u32> = (0..lanes as u32).map(|l| 10 * l).collect();
            assert_eq!(output, expected, "width {}", width);
        }
    }
}
