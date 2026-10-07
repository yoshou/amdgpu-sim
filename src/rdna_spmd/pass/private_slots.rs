use super::super::analysis::{Analyses, Constants};
use super::super::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub struct PrivateSlots;
impl super::Pass for PrivateSlots {
    fn name(&self) -> &str {
        "private_slots"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let constants = analyses.get::<Constants>(f);
        run(f, &constants) > 0
    }
}

#[derive(Clone, Copy)]
enum Access {
    Load(usize),
    Store(usize),
}

impl Access {
    fn slot(self) -> usize {
        match self {
            Self::Load(s) | Self::Store(s) => s,
        }
    }
}

fn accesses(f: &Func, constants: &[Option<u64>]) -> Option<BTreeMap<(BlockId, usize), Access>> {
    let mut found = BTreeMap::new();
    let mut offsets = BTreeSet::new();
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            match inst {
                Inst::Core {
                    op: Op::Env(Env::ScratchBase | Env::ScratchSize),
                    ..
                } => return None,
                Inst::Effect {
                    op: EffectOp::Memory { space: Space::Scratch, op, semantics },
                    inputs,
                    outputs,
                    ..
                } => {
                    if semantics.volatile {
                        return None;
                    }
                    let offset = constants[inputs[0].0].filter(|&k| k % 4 == 0 && k < 1 << 32)?;
                    let access = match op {
                        MemoryOp::Load(MemSize::B32) if outputs.len() == 1 && outputs[0].1 == Ty::I32 => {
                            Access::Load(offset as usize)
                        }
                        MemoryOp::Store(MemSize::B32) if f.types[inputs[1].0] == Ty::I32 => {
                            Access::Store(offset as usize)
                        }
                        _ => return None,
                    };
                    offsets.insert(offset);
                    found.insert((id, index), access);
                }
                _ => {}
            }
        }
    }
    if found.is_empty() {
        return None;
    }
    let slot: BTreeMap<u64, usize> = offsets.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    Some(
        found
            .into_iter()
            .map(|(at, access)| {
                let renumbered = match access {
                    Access::Load(k) => Access::Load(slot[&(k as u64)]),
                    Access::Store(k) => Access::Store(slot[&(k as u64)]),
                };
                (at, renumbered)
            })
            .collect(),
    )
}

fn live(f: &Func, accesses: &BTreeMap<(BlockId, usize), Access>, slots: usize) -> BTreeMap<BlockId, Vec<bool>> {
    let mut live: BTreeMap<BlockId, Vec<bool>> = f.blocks.keys().map(|&id| (id, vec![false; slots])).collect();
    for (&(id, _), access) in accesses {
        live.get_mut(&id).unwrap()[access.slot()] = true;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (&id, block) in &f.blocks {
            for edge in block.term.edges() {
                for s in 0..slots {
                    if live[&edge.dst][s] && !live[&id][s] {
                        live.get_mut(&id).unwrap()[s] = true;
                        changed = true;
                    }
                }
            }
        }
    }
    live
}

fn run(f: &mut Func, constants: &[Option<u64>]) -> usize {
    if f.blocks.values().any(|b| b.term.edges().any(|e| e.dst == f.entry)) {
        return 0;
    }
    let Some(accesses) = accesses(f, constants) else {
        return 0;
    };
    let slots = accesses.values().map(|a| a.slot() + 1).max().unwrap();
    let live = live(f, &accesses, slots);
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    let mut added: BTreeMap<BlockId, Vec<(usize, ValueId)>> = BTreeMap::new();
    for &id in &ids {
        if id == f.entry {
            continue;
        }
        let params: Vec<(usize, ValueId)> = (0..slots)
            .filter(|&s| live[&id][s])
            .map(|s| (s, f.value(Ty::I32)))
            .collect();
        let block = f.blocks.get_mut(&id).unwrap();
        block.params.extend(params.iter().map(|&(_, v)| (v, Ty::I32)));
        added.insert(id, params);
    }
    let zero = f.value(Ty::I32);
    let mut renamed: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for &id in &ids {
        let mut current: Vec<Option<ValueId>> = vec![None; slots];
        if id == f.entry {
            current = vec![Some(zero); slots];
        } else {
            for &(s, v) in &added[&id] {
                current[s] = Some(v);
            }
        }
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut insts = Vec::with_capacity(old.len() + 1);
        if id == f.entry {
            insts.push(Inst::Core {
                value: zero,
                ty: Ty::I32,
                op: Op::Const(Ty::I32, 0),
            });
        }
        for (index, inst) in old.into_iter().enumerate() {
            match (accesses.get(&(id, index)), &inst) {
                (Some(&Access::Load(s)), Inst::Effect { outputs, .. }) => {
                    renamed.insert(outputs[0].0, current[s].unwrap());
                }
                (Some(&Access::Store(s)), Inst::Effect { inputs, .. }) => {
                    let value = f.value(Ty::I32);
                    insts.push(Inst::Core {
                        value,
                        ty: Ty::I32,
                        op: Op::Select(inputs[2], inputs[1], current[s].unwrap()),
                    });
                    current[s] = Some(value);
                }
                _ => insts.push(inst),
            }
        }
        let block = f.blocks.get_mut(&id).unwrap();
        block.insts = insts;
        let mut edges: Vec<&mut Edge> = match &mut block.term {
            Term::Br(e) => vec![e],
            Term::CondBr { yes, no, .. } => vec![yes, no],
            Term::Ret(_) => Vec::new(),
        };
        for edge in &mut edges {
            for &(s, _) in &added[&edge.dst] {
                edge.args.push(current[s].unwrap());
            }
        }
    }
    f.rename(&renamed);
    f.compact();
    accesses.len()
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    fn memory(space: Space, op: MemoryOp, volatile: bool) -> EffectOp {
        EffectOp::Memory {
            space,
            op,
            semantics: MemorySemantics {
                scope: Scope::WorkItem,
                ordering: Ordering::Relaxed,
                cache_policy: CachePolicy::Temporal,
                volatile,
                deferred_scope: false,
            },
        }
    }

    struct Builder {
        f: Func,
        next: u64,
    }

    impl Builder {
        fn effect(&mut self, insts: &mut Vec<Inst>, op: EffectOp, inputs: Vec<ValueId>, outputs: Vec<(ValueId, Ty)>) {
            insts.push(Inst::Effect {
                provenance: self.next << 8,
                op,
                inputs,
                outputs,
            });
            self.next += 1;
        }

        fn store(&mut self, insts: &mut Vec<Inst>, address: ValueId, data: ValueId, mask: ValueId) {
            self.effect(insts, memory(Space::Scratch, MemoryOp::Store(MemSize::B32), false), vec![address, data, mask], Vec::new());
        }

        fn load(&mut self, insts: &mut Vec<Inst>, address: ValueId, mask: ValueId) -> ValueId {
            let v = self.f.value(Ty::I32);
            self.effect(insts, memory(Space::Scratch, MemoryOp::Load(MemSize::B32), false), vec![address, mask], vec![(v, Ty::I32)]);
            v
        }
    }

    fn random_program(r: &mut Rng, lanes: u32) -> Func {
        let mut b = Builder {
            f: Func::new(BlockId(0), Presence::Wave, lanes),
            next: 1,
        };
        let (entry, header, exit) = (BlockId(0), BlockId(1), BlockId(2));
        let x = b.f.value(Ty::I32);
        let exec = b.f.value(Ty::I1);
        let mut start = Vec::new();
        let zero = core(&mut b.f, &mut start, Ty::I32, Op::Const(Ty::I32, 0));
        let lane = core(&mut b.f, &mut start, Ty::I32, Op::Env(Env::LaneId));
        let carried = 2;
        let counter = b.f.value(Ty::I32);
        let (xh, laneh, zeroh) = (b.f.value(Ty::I32), b.f.value(Ty::I32), b.f.value(Ty::I32));
        let params: Vec<ValueId> = (0..carried).map(|_| b.f.value(Ty::I32)).collect();
        let slots = 1 + r.below(4);
        let offset = |b: &mut Builder, insts: &mut Vec<Inst>, k: usize| {
            let base = core(&mut b.f, insts, Ty::I32, Op::Const(Ty::I32, 0));
            let step = core(&mut b.f, insts, Ty::I32, Op::Const(Ty::I32, 4 * k as u64));
            core(&mut b.f, insts, Ty::I32, Op::Int(IntOp::Add, base, step))
        };
        for k in 0..r.below(3) {
            let at = offset(&mut b, &mut start, k % slots);
            b.store(&mut start, at, x, exec);
        }
        let mut body = Vec::new();
        let mut pool: Vec<ValueId> = vec![zeroh, xh, laneh, counter];
        pool.extend(params.iter().copied());
        let mut bits: Vec<ValueId> = Vec::new();
        for _ in 0..10 {
            let a = pool[r.below(pool.len())];
            let c = pool[r.below(pool.len())];
            match r.below(4) {
                0 => {
                    let m = core(&mut b.f, &mut body, Ty::I1, Op::Cmp(IntPred::Ult, a, c));
                    bits.push(m);
                }
                1 => {
                    let v = core(&mut b.f, &mut body, Ty::I32, Op::Int(IntOp::Add, a, c));
                    pool.push(v);
                }
                2 => {
                    let mask = bits.get(r.below(bits.len() + 1)).copied().unwrap_or(exec_bit(&mut b, &mut body));
                    let at = offset(&mut b, &mut body, r.below(slots));
                    b.store(&mut body, at, a, mask);
                }
                _ => {
                    let mask = bits.get(r.below(bits.len() + 1)).copied().unwrap_or(exec_bit(&mut b, &mut body));
                    let at = offset(&mut b, &mut body, r.below(slots));
                    let v = b.load(&mut body, at, mask);
                    let kept = core(&mut b.f, &mut body, Ty::I32, Op::Select(mask, v, a));
                    pool.push(kept);
                }
            }
        }
        let one = core(&mut b.f, &mut body, Ty::I32, Op::Const(Ty::I32, 1));
        let next = core(&mut b.f, &mut body, Ty::I32, Op::Int(IntOp::Add, counter, one));
        let limit = core(&mut b.f, &mut body, Ty::I32, Op::Const(Ty::I32, 3));
        let more = core(&mut b.f, &mut body, Ty::I1, Op::Cmp(IntPred::Ult, next, limit));
        let backs: Vec<ValueId> = vec![next, xh, laneh, zeroh].into_iter().chain((0..carried).map(|_| pool[r.below(pool.len())])).collect();
        let starts: Vec<ValueId> = vec![zero, x, lane, zero].into_iter().chain((0..carried).map(|_| [zero, x, lane][r.below(3)])).collect();
        let out = b.f.value(Ty::I32);
        let mut last = Vec::new();
        let full = core(&mut b.f, &mut last, Ty::I1, Op::Const(Ty::I1, 1));
        let at = offset(&mut b, &mut last, r.below(slots));
        let read = b.load(&mut last, at, full);
        let exits = vec![pool[r.below(pool.len())]];
        let mut header_params = vec![(counter, Ty::I32), (xh, Ty::I32), (laneh, Ty::I32), (zeroh, Ty::I32)];
        header_params.extend(params.iter().map(|&v| (v, Ty::I32)));
        b.f.blocks.insert(
            entry,
            Block {
                params: vec![(x, Ty::I32), (exec, Ty::I1)],
                insts: start,
                term: Term::Br(Edge { dst: header, args: starts }),
            },
        );
        b.f.blocks.insert(
            header,
            Block {
                params: header_params,
                insts: body,
                term: Term::CondBr {
                    cond: more,
                    yes: Edge { dst: header, args: backs },
                    no: Edge { dst: exit, args: exits },
                },
            },
        );
        b.f.blocks.insert(
            exit,
            Block {
                params: vec![(out, Ty::I32)],
                insts: last,
                term: Term::Ret(vec![out, read]),
            },
        );
        b.f
    }

    fn exec_bit(b: &mut Builder, insts: &mut Vec<Inst>) -> ValueId {
        core(&mut b.f, insts, Ty::I1, Op::Const(Ty::I1, 1))
    }

    fn inputs(lanes: usize, r: &mut Rng) -> Vec<Vec<u64>> {
        vec![
            (0..lanes).map(|_| r.below(8) as u64).collect(),
            (0..lanes).map(|_| (r.below(4) != 0) as u64).collect(),
        ]
    }

    fn constants(f: &Func) -> Vec<Option<u64>> {
        let mut known = vec![None; f.types.len()];
        for block in f.blocks.values() {
            for inst in &block.insts {
                if let Inst::Core { value, op, .. } = *inst {
                    known[value.0] = match op {
                        Op::Const(_, k) => Some(k),
                        Op::Int(IntOp::Add, a, b) => known[a.0].zip(known[b.0]).map(|(x, y)| (x + y) & 0xffff_ffff),
                        _ => None,
                    };
                }
            }
        }
        known
    }

    fn promoted(f: &Func) -> (Func, usize) {
        let mut promoted = f.clone();
        let constants = constants(&promoted);
        let count = run(&mut promoted, &constants);
        (promoted, count)
    }

    #[test]
    fn promoted_slots_return_what_the_private_memory_returns() {
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let registry = registry();
        for trial in 0..400 {
            let lanes = if trial % 2 == 0 { 32 } else { 64 };
            let f = random_program(&mut r, lanes);
            f.check(&registry).unwrap_or_else(|e| panic!("trial {}: the generated program: {}", trial, e));
            let (promoted, count) = promoted(&f);
            assert!(count > 0, "trial {}: the program touches its private memory", trial);
            promoted.check(&registry).unwrap_or_else(|e| panic!("trial {}: {}", trial, e));
            let left = promoted.blocks.values().flat_map(|b| &b.insts).any(|i| {
                matches!(i, Inst::Effect { op: EffectOp::Memory { space: Space::Scratch, .. }, .. })
            });
            assert!(!left, "trial {}: no private access is left", trial);
            for _ in 0..4 {
                let entry = inputs(lanes as usize, &mut r);
                let want = simulate(&f, &entry, &mut |_, _| {});
                let got = simulate(&promoted, &entry, &mut |_, _| {});
                assert_eq!(want, got, "trial {}: the promoted program returns other words", trial);
            }
        }
    }

    fn single(op: EffectOp, address: u64, extra: impl FnOnce(&mut Func, &mut Vec<Inst>) -> ValueId) -> Func {
        let mut f = Func::new(BlockId(0), Presence::Wave, 32);
        let exec = f.value(Ty::I1);
        let mut insts = Vec::new();
        let at = extra(&mut f, &mut insts);
        let at = if address == u64::MAX { at } else { core(&mut f, &mut insts, Ty::I32, Op::Const(Ty::I32, address)) };
        let data = core(&mut f, &mut insts, Ty::I32, Op::Const(Ty::I32, 7));
        let inputs = match op {
            EffectOp::Memory { op: MemoryOp::Load(_), .. } => vec![at, exec],
            _ => vec![at, data, exec],
        };
        let outputs = match op {
            EffectOp::Memory { op: MemoryOp::Load(MemSize::B64), .. } => vec![(f.value(Ty::I64), Ty::I64)],
            EffectOp::Memory { op: MemoryOp::Load(_), .. } => vec![(f.value(Ty::I32), Ty::I32)],
            _ => Vec::new(),
        };
        insts.push(Inst::Effect { provenance: 1 << 8, op, inputs, outputs });
        f.blocks.insert(BlockId(0), Block { params: vec![(exec, Ty::I1)], insts, term: Term::Ret(Vec::new()) });
        f
    }

    #[test]
    fn private_memory_stays_unless_every_access_is_an_aligned_word_at_a_known_place() {
        let store = memory(Space::Scratch, MemoryOp::Store(MemSize::B32), false);
        let lane = |f: &mut Func, insts: &mut Vec<Inst>| core(f, insts, Ty::I32, Op::Env(Env::LaneId));
        let nothing = |f: &mut Func, insts: &mut Vec<Inst>| core(f, insts, Ty::I32, Op::Const(Ty::I32, 0));
        let base = |f: &mut Func, insts: &mut Vec<Inst>| {
            core(f, insts, Ty::I64, Op::Env(Env::ScratchBase));
            core(f, insts, Ty::I32, Op::Const(Ty::I32, 0))
        };
        let cases: Vec<(&str, Func, usize)> = vec![
            ("a word at a known place", single(store, 8, nothing), 1),
            ("a lane's own place", single(store, u64::MAX, lane), 0),
            ("a misaligned word", single(store, 6, nothing), 0),
            ("a volatile word", single(memory(Space::Scratch, MemoryOp::Store(MemSize::B32), true), 8, nothing), 0),
            ("a wide word", single(memory(Space::Scratch, MemoryOp::Load(MemSize::B64), false), 8, nothing), 0),
            ("a short word", single(memory(Space::Scratch, MemoryOp::Store(MemSize::U8), false), 8, nothing), 0),
            ("a pointer to the private memory", single(store, 8, base), 0),
        ];
        for (name, f, expected) in cases {
            let (promoted, count) = promoted(&f);
            assert_eq!(count, expected, "{}", name);
            if expected == 0 {
                assert_eq!(promoted, f, "{}: the function is left alone", name);
            }
        }
    }

    #[test]
    fn private_memory_stays_when_the_entry_is_a_loop() {
        let mut f = single(memory(Space::Scratch, MemoryOp::Store(MemSize::B32), false), 8, |f, insts| {
            core(f, insts, Ty::I32, Op::Const(Ty::I32, 0))
        });
        let exec = f.blocks[&BlockId(0)].params[0].0;
        f.blocks.get_mut(&BlockId(0)).unwrap().term = Term::CondBr {
            cond: exec,
            yes: Edge { dst: BlockId(0), args: vec![exec] },
            no: Edge { dst: BlockId(1), args: Vec::new() },
        };
        f.blocks.insert(BlockId(1), Block { params: Vec::new(), insts: Vec::new(), term: Term::Ret(Vec::new()) });
        let (promoted, count) = promoted(&f);
        assert_eq!(count, 0);
        assert_eq!(promoted, f);
    }
}
