use super::super::ir::*;

pub(super) const UNSET: u64 = 0xdead_beef;

pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

pub(super) fn core(f: &mut Func, insts: &mut Vec<Inst>, ty: Ty, op: Op) -> ValueId {
    let value = f.value(ty);
    insts.push(Inst::Core { value, ty, op });
    value
}

fn mask(ty: Ty) -> u64 {
    match ty {
        Ty::I1 => 1,
        Ty::I32 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

fn evaluate(op: Op, ty: Ty, lane: usize, g: &dyn Fn(ValueId) -> u64) -> u64 {
    let x = match op {
        Op::Const(_, k) => k,
        Op::Env(Env::LaneId) => lane as u64,
        Op::Env(Env::ValidLane) => 1,
        Op::Int(k, a, b) => {
            let (a, b) = (g(a), g(b));
            match k {
                IntOp::Add => a.wrapping_add(b),
                IntOp::Sub => a.wrapping_sub(b),
                IntOp::And => a & b,
                IntOp::Or => a | b,
                IntOp::Xor => a ^ b,
                IntOp::Shl => a.checked_shl(b as u32).unwrap_or(0),
                IntOp::LShr => a.checked_shr(b as u32).unwrap_or(0),
                other => panic!("an integer operation the simulation lacks: {:?}", other),
            }
        }
        Op::Cmp(p, a, b) => {
            let (a, b) = (g(a), g(b));
            (match p {
                IntPred::Eq => a == b,
                IntPred::Ne => a != b,
                IntPred::Ult => a < b,
                IntPred::Ugt => a > b,
                other => panic!("a comparison the simulation lacks: {:?}", other),
            }) as u64
        }
        Op::Select(c, a, b) => {
            if g(c) & 1 != 0 {
                g(a)
            } else {
                g(b)
            }
        }
        Op::Pack64(lo, hi) => (g(lo) & 0xffff_ffff) | (g(hi) << 32),
        Op::UnpackLo(a) => g(a) & 0xffff_ffff,
        Op::UnpackHi(a) => g(a) >> 32,
        Op::Convert(Cvt::Trunc | Cvt::ZExt | Cvt::Bitcast, _, a) => g(a),
        other => panic!("an operation the simulation lacks: {:?}", other),
    };
    x & mask(ty)
}

pub(super) fn simulate(
    f: &Func,
    entry: &[Vec<u64>],
    visit: &mut dyn FnMut(BlockId, &[Vec<u64>]),
) -> Vec<Vec<u64>> {
    let lanes = f.lanes as usize;
    let mut values: Vec<Vec<u64>> = vec![Vec::new(); f.types.len()];
    let mut scratch: Vec<std::collections::BTreeMap<u64, u64>> = vec![Default::default(); lanes];
    for (&(p, _), v) in f.blocks[&f.entry].params.iter().zip(entry) {
        values[p.0] = v.clone();
    }
    let mut block = f.entry;
    for _ in 0..512 {
        let b = &f.blocks[&block];
        for inst in &b.insts {
            match inst {
                Inst::Core { value, ty, op } => {
                    let out: Vec<u64> = (0..lanes)
                        .map(|l| evaluate(*op, *ty, l, &|v: ValueId| values[v.0][l]))
                        .collect();
                    values[value.0] = out;
                }
                Inst::Effect { op, inputs, outputs, .. } => match op {
                    EffectOp::Wave(WaveOp::Any) => {
                        let set = values[inputs[0].0].iter().any(|&x| x & 1 != 0) as u64;
                        values[outputs[0].0 .0] = vec![set; lanes];
                    }
                    EffectOp::Wave(WaveOp::Ballot { high }) => {
                        let first = if *high { 32 } else { 0 };
                        let word = (0..32)
                            .filter(|k| first + k < lanes && values[inputs[0].0][first + k] & 1 != 0)
                            .fold(0u64, |w, k| w | 1 << k);
                        values[outputs[0].0 .0] = vec![word; lanes];
                    }
                    EffectOp::Memory {
                        space: Space::Scratch,
                        op: MemoryOp::Load(MemSize::B32),
                        ..
                    } => {
                        let out: Vec<u64> = (0..lanes)
                            .map(|l| {
                                if values[inputs[1].0][l] & 1 != 0 {
                                    scratch[l].get(&values[inputs[0].0][l]).copied().unwrap_or(0)
                                } else {
                                    UNSET
                                }
                            })
                            .collect();
                        values[outputs[0].0 .0] = out;
                    }
                    EffectOp::Memory {
                        space: Space::Scratch,
                        op: MemoryOp::Store(MemSize::B32),
                        ..
                    } => {
                        for (l, memory) in scratch.iter_mut().enumerate() {
                            if values[inputs[2].0][l] & 1 != 0 {
                                memory.insert(values[inputs[0].0][l], values[inputs[1].0][l]);
                            }
                        }
                    }
                    other => panic!("an effect the simulation lacks: {:?}", other),
                },
                other => panic!("an instruction the simulation lacks: {:?}", other),
            }
        }
        visit(block, &values);
        let edge = match &b.term {
            Term::Ret(args) => return args.iter().map(|a| values[a.0].clone()).collect(),
            Term::Br(e) => e,
            Term::CondBr { cond, yes, no } => {
                let c = &values[cond.0];
                assert!(c.iter().all(|&x| x == c[0]), "the wave branches on a bit every lane shares");
                if c[0] & 1 != 0 {
                    yes
                } else {
                    no
                }
            }
        };
        let args: Vec<Vec<u64>> = edge.args.iter().map(|a| values[a.0].clone()).collect();
        for (&(p, _), arg) in f.blocks[&edge.dst].params.iter().zip(args) {
            values[p.0] = arg;
        }
        block = edge.dst;
    }
    panic!("the program does not end");
}

pub(super) fn registry() -> DialectRegistry {
    let mut registry = DialectRegistry::default();
    registry.set_registers(Registers {
        exec: 126,
        vcc: 106,
        null: 124,
        scc_slot: 128,
        sgprs: 128,
        vgprs: 256,
    });
    registry
}
