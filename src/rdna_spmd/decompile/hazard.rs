use super::address::{Addresses, Form, Region, Regions, UnknownInfo, Value, LANES};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::Program;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Commutes {
    Add,
    Rmw(Rmw),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Read,
    Write,
    Update {
        commutes: Option<Commutes>,
        used: bool,
    },
}

impl Kind {
    pub fn writes(self) -> bool {
        self != Kind::Read
    }

    pub fn reads(self) -> bool {
        !matches!(self, Kind::Write | Kind::Update { used: false, .. })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub block: BlockId,
    pub index: usize,
    pub instruction: u64,
    pub kind: Kind,
    pub space: Option<Space>,
    pub bytes: u32,
    pub address: Option<ValueId>,
    pub predicate: Option<ValueId>,
    pub output: Option<ValueId>,
    pub exec: Option<ValueId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hazards {
    pub accesses: Vec<Access>,
    pub together: BTreeSet<(usize, usize)>,
    pub apart: BTreeSet<(usize, usize)>,
    pub idle: BTreeSet<(usize, usize, bool, usize)>,
    pub meetings: Vec<(BlockId, usize)>,
}

type Judgments = [(bool, [bool; 2]); 2];

const RESOURCE_BYTES: u32 = 64;

impl Hazards {
    pub fn find(program: &Program, env: &Environment) -> Self {
        let f = &program.ir;
        let facts = Facts::new(f, &program.parameter_inputs, &BTreeSet::new());
        let exec = program.registry.registers().exec;
        let exec_index = program.parameter_inputs.iter().position(
            |p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec),
        );
        let accesses = accesses(program, &facts, exec_index);
        let mut hazards = Hazards {
            accesses,
            together: BTreeSet::new(),
            apart: BTreeSet::new(),
            idle: BTreeSet::new(),
            meetings: Vec::new(),
        };
        if !hazards
            .accesses
            .iter()
            .any(|a| a.kind.writes() && a.space != Some(Space::Scratch))
        {
            return hazards;
        }
        let loops = Loops::new(f, &facts).unwrap_or_else(|block| {
            panic!(
                "b{}: control flow enters a cycle other than through its header",
                block.0
            )
        });
        let rank: crate::rdna_spmd::hash::HashMap<BlockId, usize> =
            facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let headers: BTreeSet<BlockId> = (0..loops.count())
            .map(|l| facts.order[loops.header(l)])
            .collect();
        let mut addresses = Addresses::new(
            f,
            &facts,
            &program.parameter_inputs,
            exec,
            program.entry,
            env,
            headers,
        );
        let n = hazards.accesses.len();
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for p in 0..n {
            for q in p..n {
                if relevant(&hazards.accesses, &loops, &rank, p, q) {
                    candidates.push((p, q));
                }
            }
        }
        if candidates.is_empty() {
            return hazards;
        }
        let waves = addresses.waves();
        let around: Vec<Vec<usize>> = (0..facts.order.len()).map(|r| containing(&loops, r)).collect();
        let loops_of: Vec<Vec<usize>> = hazards
            .accesses
            .iter()
            .map(|a| containing(&loops, rank[&a.block]))
            .collect();
        let mut state: BTreeMap<(usize, usize), Judgments> = candidates
            .into_iter()
            .map(|pair| (pair, [(false, [true; 2]); 2]))
            .collect();
        let settled = |s: &Judgments| s.iter().all(|&(found, idle)| found && idle == [false; 2]);
        for wave in 0..waves {
            if state.values().all(settled) {
                break;
            }
            addresses.enter(wave);
            let pairs: Vec<(usize, usize)> = state
                .iter()
                .filter(|(_, s)| !settled(s))
                .map(|(&pair, _)| pair)
                .collect();
            let involved: BTreeSet<usize> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
            let mut regions: Vec<Vec<Option<Regions>>> = hazards
                .accesses
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    (0..LANES)
                        .map(|lane| {
                            (involved.contains(&i) && addresses.valid(lane))
                                .then(|| region_of(&mut addresses, a, lane, false))
                        })
                        .collect()
                })
                .collect();
            let mut sharing: BTreeSet<(usize, usize)> = pairs
                .iter()
                .copied()
                .filter(|&(p, q)| may_share(env, &regions[p], &regions[q]))
                .collect();
            let mut refined: BTreeSet<usize> = BTreeSet::new();
            loop {
                let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
                for &(p, q) in &sharing {
                    for i in [p, q] {
                        if !refined.contains(&i) && regions[i].iter().flatten().any(|r| !r.single_region()) {
                            *counts.entry(i).or_default() += 1;
                        }
                    }
                }
                let Some((&next, _)) = counts.iter().max_by_key(|&(&i, &n)| (n, std::cmp::Reverse(i))) else {
                    break;
                };
                refined.insert(next);
                let a = &hazards.accesses[next];
                regions[next] = (0..LANES)
                    .map(|lane| addresses.valid(lane).then(|| region_of(&mut addresses, a, lane, true)))
                    .collect();
                sharing.retain(|&(p, q)| (p != next && q != next) || may_share(env, &regions[p], &regions[q]));
            }
            let precise: BTreeSet<usize> = sharing.iter().flat_map(|&(p, q)| [p, q]).collect();
            let lanes: Vec<Vec<Option<Place>>> = hazards
                .accesses
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    if precise.contains(&i) {
                        places(&mut addresses, a)
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            for (p, q) in sharing {
                let common = loops_of[p].iter().any(|l| loops_of[q].contains(l));
                let mut now = state[&(p, q)];
                for (k, together) in [(0, true), (1, false)] {
                    if !together && !common {
                        now[1] = now[0];
                        continue;
                    }
                    let (lp, lq) = (&loops_of[p], &loops_of[q]);
                    let variant = |info: &UnknownInfo| {
                        around[info.rank].iter().any(|l| {
                            let (in_p, in_q) = (lp.contains(l), lq.contains(l));
                            if together {
                                in_p != in_q
                            } else {
                                in_p || in_q
                            }
                        })
                    };
                    let found = meet(
                        &mut addresses,
                        env,
                        (&hazards.accesses[p], &lanes[p]),
                        (&hazards.accesses[q], &lanes[q]),
                        &variant,
                    );
                    if let Some(idle) = found {
                        now[k].0 = true;
                        now[k].1 = [now[k].1[0] && idle[0], now[k].1[1] && idle[1]];
                    }
                }
                state.insert((p, q), now);
            }
        }
        for ((p, q), s) in state {
            for (k, apart) in [(0, false), (1, true)] {
                let (found, idle) = s[k];
                if !found {
                    continue;
                }
                if apart {
                    hazards.apart.insert((p, q));
                } else {
                    hazards.together.insert((p, q));
                }
                for (side, &all) in idle.iter().enumerate() {
                    if all {
                        hazards.idle.insert((p, q, apart, side));
                    }
                }
            }
        }
        hazards.meetings = hazards.positions(&facts.order);
        hazards
    }

    pub fn conflicts(&self) -> BTreeSet<(usize, usize)> {
        self.together.union(&self.apart).copied().collect()
    }

    pub fn involved(&self) -> BTreeSet<usize> {
        self.conflicts().iter().flat_map(|&(p, q)| [p, q]).collect()
    }

    fn positions(&self, order: &[BlockId]) -> Vec<(BlockId, usize)> {
        let mut positions = BTreeSet::new();
        for i in self.involved() {
            let a = &self.accesses[i];
            let first = self
                .accesses
                .iter()
                .filter(|x| x.block == a.block && x.instruction == a.instruction)
                .map(|x| x.index)
                .min()
                .unwrap_or(a.index);
            positions.insert((a.block, first));
        }
        let rank: BTreeMap<BlockId, usize> =
            order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let mut positions: Vec<(BlockId, usize)> = positions.into_iter().collect();
        positions.sort_by_key(|&(b, i)| (rank[&b], i));
        positions
    }
}

fn accesses(program: &Program, facts: &Facts, exec_index: Option<usize>) -> Vec<Access> {
    let f = &program.ir;
    let mut out = Vec::new();
    for &b in &facts.order {
        let exec = exec_index.map(|i| f.blocks[&b].params[i].0);
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            match inst {
                Inst::Effect {
                    provenance,
                    op: EffectOp::Memory { space, op, .. },
                    inputs,
                    outputs,
                } => {
                    let output = outputs.first().map(|o| o.0);
                    let used = output.is_some_and(|o| !facts.uses[o.0].is_empty());
                    let (kind, bytes) = match *op {
                        MemoryOp::Fence => continue,
                        MemoryOp::Load(size) => (Kind::Read, size.bytes()),
                        MemoryOp::Store(size) => (Kind::Write, size.bytes()),
                        MemoryOp::AtomicAdd(Numeric::Unsigned) => (
                            Kind::Update {
                                commutes: Some(Commutes::Add),
                                used,
                            },
                            4,
                        ),
                        MemoryOp::AtomicRmw(rmw) => (
                            Kind::Update {
                                commutes: Some(Commutes::Rmw(rmw)),
                                used,
                            },
                            4,
                        ),
                        MemoryOp::AtomicAdd(Numeric::Float) | MemoryOp::AtomicCmpSwap => (
                            Kind::Update {
                                commutes: None,
                                used,
                            },
                            4,
                        ),
                    };
                    out.push(Access {
                        block: b,
                        index,
                        instruction: provenance >> 8,
                        kind,
                        space: Some(*space),
                        bytes,
                        address: Some(inputs[0]),
                        predicate: Some(inputs[op.mask_input()]),
                        output,
                        exec,
                    });
                }
                Inst::Target {
                    provenance,
                    op,
                    outputs,
                    ..
                } => {
                    let reads = program.registry.operation(*op).is_ok_and(|spec| {
                        matches!(spec.effect, Effect::ReadGlobal { .. })
                    });
                    if reads {
                        out.push(Access {
                            block: b,
                            index,
                            instruction: provenance.map_or(u64::MAX, |p| p >> 8),
                            kind: Kind::Read,
                            space: None,
                            bytes: RESOURCE_BYTES,
                            address: None,
                            predicate: None,
                            output: outputs.first().map(|o| o.0),
                            exec,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn containing(loops: &Loops, rank: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut l = loops.innermost(rank);
    while let Some(x) = l {
        out.push(x);
        l = loops.parent(x);
    }
    out
}

fn relevant(
    accesses: &[Access],
    loops: &Loops,
    rank: &crate::rdna_spmd::hash::HashMap<BlockId, usize>,
    p: usize,
    q: usize,
) -> bool {
    let (a, b) = (&accesses[p], &accesses[q]);
    if !a.kind.writes() && !b.kind.writes() {
        return false;
    }
    if a.space == Some(Space::Scratch) || b.space == Some(Space::Scratch) {
        return false;
    }
    if let (
        Kind::Update {
            commutes: Some(x),
            used: false,
        },
        Kind::Update {
            commutes: Some(y),
            used: false,
        },
    ) = (a.kind, b.kind)
    {
        if x == y {
            return false;
        }
    }
    let lds = |x: &Access| x.space == Some(Space::Lds);
    if lds(a) != lds(b) {
        return false;
    }
    if a.instruction == b.instruction {
        return loops.innermost(rank[&a.block]).is_some();
    }
    true
}

#[derive(Clone, Debug)]
struct Place {
    region: Region,
    address: Option<Value>,
}

fn places(addresses: &mut Addresses, a: &Access) -> Vec<Option<Place>> {
    (0..LANES)
        .map(|lane| {
            if !addresses.valid(lane) {
                return None;
            }
            let Some(address) = a.address else {
                return Some(Place {
                    region: Region::Exposed,
                    address: None,
                });
            };
            let (value, _) = addresses.operand(address, a.block, lane, a.predicate);
            let region = match a.space {
                Some(Space::Lds) => Region::Lds,
                Some(Space::Scratch) => Region::Private,
                _ => value.region.unwrap_or(Region::Exposed),
            };
            Some(Place {
                region,
                address: Some(value),
            })
        })
        .collect()
}

fn runs(addresses: &mut Addresses, access: &Access, lane: usize) -> bool {
    addresses.reaches_block(access.block)
        && access
            .predicate
            .is_none_or(|p| addresses.bit(p, lane, None).0 != Some(false))
}

fn idles(addresses: &mut Addresses, access: &Access, lane: usize) -> bool {
    access.kind == Kind::Read && access.exec.is_some_and(|e| addresses.bit(e, lane, None).0 == Some(false))
}

fn region_of(addresses: &mut Addresses, a: &Access, lane: usize, refine: bool) -> Regions {
    match a.space {
        Some(Space::Lds) => Regions::one(Some(Region::Lds)),
        Some(Space::Scratch) => Regions::one(Some(Region::Private)),
        _ => match a.address {
            Some(x) => addresses.regions(x, lane, a.predicate, refine),
            None => Regions::one(None),
        },
    }
}

fn may_share(env: &Environment, p: &[Option<Regions>], q: &[Option<Regions>]) -> bool {
    let meets = |x: Option<Region>, y: Option<Region>| {
        overlapping(env, x.unwrap_or(Region::Exposed), y.unwrap_or(Region::Exposed))
    };
    let mut all_p = Regions::default();
    for x in p.iter().flatten() {
        all_p.union(x);
    }
    let mut all_q = Regions::default();
    for y in q.iter().flatten() {
        all_q.union(y);
    }
    if !all_p.overlaps(&all_q, meets) {
        return false;
    }
    p.iter().enumerate().any(|(a, x)| {
        x.as_ref().is_some_and(|x| {
            q.iter()
                .enumerate()
                .any(|(b, y)| a != b && y.as_ref().is_some_and(|y| x.overlaps(y, meets)))
        })
    })
}

fn overlapping(env: &Environment, a: Region, b: Region) -> bool {
    use Region::*;
    let exposed = |id: u64| env.exposed.contains(&id);
    match (a, b) {
        (Private, _) | (_, Private) => false,
        (Kernarg | Dispatch, _) | (_, Kernarg | Dispatch) => false,
        (Lds, Lds) => true,
        (Lds, _) | (_, Lds) => false,
        (Allocation(x), Allocation(y)) => x == y,
        (Allocation(x), Exposed) | (Exposed, Allocation(x)) => exposed(x),
        (Exposed, Exposed) => true,
    }
}

fn meet(
    addresses: &mut Addresses,
    env: &Environment,
    (pa, p): (&Access, &[Option<Place>]),
    (qa, q): (&Access, &[Option<Place>]),
    variant: &dyn Fn(&UnknownInfo) -> bool,
) -> Option<[bool; 2]> {
    let mut idle: Option<[bool; 2]> = None;
    for (a, x) in p.iter().enumerate() {
        let Some(x) = x else { continue };
        for (b, y) in q.iter().enumerate() {
            let Some(y) = y else { continue };
            if a == b || !overlapping(env, x.region, y.region) {
                continue;
            }
            let (Some(xa), Some(ya)) = (&x.address, &y.address) else {
                if !runs(addresses, pa, a) || !runs(addresses, qa, b) {
                    continue;
                }
                let both = [idles(addresses, pa, a), idles(addresses, qa, b)];
                let old = idle.unwrap_or([true; 2]);
                idle = Some([old[0] && both[0], old[1] && both[1]]);
                continue;
            };
            let unknowns = &addresses.unknowns;
            if xa.form.terms == ya.form.terms && xa.form.terms.iter().all(|&(u, _)| !variant(&unknowns[u as usize])) {
                let d = xa.form.constant.wrapping_sub(ya.form.constant);
                if d >= qa.bytes && d.wrapping_neg() >= pa.bytes {
                    continue;
                }
            } else if !may_overlap(unknowns, &xa.form, &ya.form, pa.bytes, qa.bytes, variant) {
                continue;
            }
            if !runs(addresses, pa, a) || !runs(addresses, qa, b) {
                continue;
            }
            if excluded(addresses, (pa, a, &xa.form), (qa, b, &ya.form), variant) {
                continue;
            }
            let both = [idles(addresses, pa, a), idles(addresses, qa, b)];
            let old = idle.unwrap_or([true; 2]);
            idle = Some([old[0] && both[0], old[1] && both[1]]);
            if idle == Some([false; 2]) {
                return idle;
            }
        }
    }
    idle
}

fn excluded(
    addresses: &mut Addresses,
    (pa, a, x): (&Access, usize, &Form),
    (qa, b, y): (&Access, usize, &Form),
    variant: &dyn Fn(&UnknownInfo) -> bool,
) -> bool {
    let difference = x.sub(y);
    if difference.terms.len() != 1 {
        return false;
    }
    let (u, c) = difference.terms[0];
    let info = addresses.unknowns[u as usize].clone();
    if !info.shared || variant(&info) {
        return false;
    }
    let Some(v) = addresses.program_value(u) else {
        return false;
    };
    let shift = c.trailing_zeros();
    let odd = c >> shift;
    let inverse = (0..5).fold(odd, |inv, _| inv.wrapping_mul(2u32.wrapping_sub(odd.wrapping_mul(inv))));
    let low_mask = if shift == 0 { u32::MAX } else { (1u32 << (32 - shift)) - 1 };
    let mut candidates = Vec::new();
    for t in -(pa.bytes as i64) + 1..qa.bytes as i64 {
        let target = (t as u32).wrapping_sub(difference.constant);
        if target & ((1u64 << shift) - 1) as u32 != 0 {
            continue;
        }
        let base = (target >> shift).wrapping_mul(inverse) & low_mask;
        let step = low_mask as u64 + 1;
        let (low, high) = info.range.map_or((0u64, u32::MAX as u64), |(l, h)| (l as u64, h as u64));
        let first = if low > base as u64 { (low - base as u64).div_ceil(step) } else { 0 };
        let last = if high >= base as u64 { (high - base as u64) / step } else { continue };
        if last < first {
            continue;
        }
        for j in first..=last {
            candidates.push((base as u64 + j * step) as u32);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates.into_iter().all(|value| {
        addresses.with_value(v, value, |this| {
            let runs = |this: &mut Addresses, access: &Access, lane: usize| {
                access
                    .predicate
                    .is_none_or(|p| this.bit(p, lane, None).0 != Some(false))
            };
            !(runs(this, pa, a) && runs(this, qa, b))
        })
    })
}

fn may_overlap(
    unknowns: &[UnknownInfo],
    x: &Form,
    y: &Form,
    x_bytes: u32,
    y_bytes: u32,
    variant: &dyn Fn(&UnknownInfo) -> bool,
) -> bool {
    let constant = x.constant.wrapping_sub(y.constant) as i64;
    let (mut low, mut high) = (0i64, 0i64);
    let mut modulus: u64 = 1 << 32;
    let mut divisor: u64 = 0;
    let mut term = |c: u32, range: Option<(u32, u32)>| {
        if c == 0 {
            return;
        }
        let signed = c as i32 as i64;
        match range {
            Some((lo, hi)) if (hi as i64 - lo as i64) * signed.abs() < 1 << 32 => {
                let (a, b) = (signed * lo as i64, signed * hi as i64);
                low += a.min(b);
                high += a.max(b);
                divisor = gcd(divisor, signed.unsigned_abs());
            }
            _ => modulus = modulus.min(1u64 << c.trailing_zeros()),
        }
    };
    let (mut i, mut j) = (0, 0);
    while i < x.terms.len() || j < y.terms.len() {
        let u = match (x.terms.get(i), y.terms.get(j)) {
            (Some(&(a, _)), Some(&(b, _))) => a.min(b),
            (Some(&(a, _)), None) => a,
            (None, Some(&(b, _))) => b,
            (None, None) => break,
        };
        let cx = match x.terms.get(i) {
            Some(&(a, c)) if a == u => {
                i += 1;
                c
            }
            _ => 0,
        };
        let cy = match y.terms.get(j) {
            Some(&(b, c)) if b == u => {
                j += 1;
                c
            }
            _ => 0,
        };
        let info = &unknowns[u as usize];
        if variant(info) {
            term(cx, info.range);
            term(cy.wrapping_neg(), info.range);
        } else {
            term(cx.wrapping_sub(cy), info.range);
        }
    }
    let step = gcd(divisor, modulus) as i64;
    let window = -(x_bytes as i64) + 1..y_bytes as i64;
    if !window.clone().any(|t| (t - constant).rem_euclid(step) == 0) {
        return false;
    }
    if high - low + 1 >= modulus as i64 {
        return true;
    }
    let m = modulus as i64;
    window.into_iter().any(|t| {
        let target = (t - constant).rem_euclid(m);
        let start = low.rem_euclid(m);
        let span = high - low;
        let offset = (target - start).rem_euclid(m);
        offset <= span
    })
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
