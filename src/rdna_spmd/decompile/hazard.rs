use super::address::{Addresses, Form, Region, Regions, Unknown, UnknownInfo, Value, LANES};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    Anywhere,
    Node {
        offset: &'static [(usize, u64)],
        shift: u32,
        bytes: u32,
        kinds: Option<(usize, u64, &'static [(u64, u32)])>,
    },
    Image,
}

fn reach(registry: &DialectRegistry, op: TargetOp) -> Reach {
    let Ok(spec) = registry.operation(op) else {
        return Reach::Anywhere;
    };
    match (registry.dialect_name(op.dialect()), spec.name) {
        (Some("rdna4"), "image_sample_lz") => Reach::Image,
        (Some("rdna4"), "image_bvh64_intersect_ray") => Reach::Node {
            offset: &[(2, !7)],
            shift: 3,
            bytes: 128,
            kinds: Some((2, 7, &[(0, 64), (1, 64), (5, 128)])),
        },
        (Some("rdna4"), "image_bvh8_intersect_ray") => Reach::Node {
            offset: &[(2, u64::MAX), (11, !15)],
            shift: 3,
            bytes: 128,
            kinds: None,
        },
        _ => Reach::Anywhere,
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
    pub reach: Reach,
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
        super::assert_exec_position(f, &facts, exec_index);
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
            &program.registry,
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
        let successors: Vec<Vec<usize>> = facts
            .order
            .iter()
            .map(|b| f.blocks[b].term.edges().map(|e| rank[&e.dst]).collect())
            .collect();
        let reaches: Vec<Vec<bool>> = (0..facts.order.len())
            .map(|start| {
                let mut seen = vec![false; facts.order.len()];
                let mut stack = successors[start].clone();
                while let Some(x) = stack.pop() {
                    if !seen[x] {
                        seen[x] = true;
                        stack.extend(successors[x].iter().copied());
                    }
                }
                seen
            })
            .collect();
        let loops_of: Vec<Vec<usize>> = hazards
            .accesses
            .iter()
            .map(|a| containing(&loops, rank[&a.block]))
            .collect();
        let start: BTreeMap<(usize, usize), Judgments> = candidates
            .into_iter()
            .map(|pair| (pair, [(false, [true; 2]); 2]))
            .collect();
        let mut state = start.clone();
        let settled = |s: &Judgments| s.iter().all(|&(found, idle)| found && idle == [false; 2]);
        let mut open = Opening::default();
        let mut wave = 0;
        while wave < waves {
            if state.values().all(settled) {
                break;
            }
            let env = open.wide.as_ref().unwrap_or(env);
            addresses.enter(wave);
            let pairs: Vec<(usize, usize)> = state
                .iter()
                .filter(|(pair, s)| !settled(s) && open.only.as_ref().is_none_or(|o| o.contains(pair)))
                .map(|(&pair, _)| pair)
                .collect();
            let Some((mut regions, sharing)) = shared(&mut addresses, &hazards.accesses, env, &pairs, wave == 0) else {
                state = start.clone();
                open.only = None;
                wave = 0;
                continue;
            };
            let precise: BTreeSet<usize> = sharing.iter().flat_map(|&(p, q)| [p, q]).collect();
            let lanes: Vec<Vec<Option<Place>>> = hazards
                .accesses
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    if precise.contains(&i) {
                        places(&mut addresses, a, &regions[i])
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            for &(p, q) in &sharing {
                let common = loops_of[p].iter().any(|l| loops_of[q].contains(l));
                let (rp, rq) = (rank[&hazards.accesses[p].block], rank[&hazards.accesses[q].block]);
                if !common && rp != rq && !reaches[rp][rq] && !reaches[rq][rp] {
                    continue;
                }
                let mut now = state[&(p, q)];
                for (k, together) in [(0, true), (1, false)] {
                    if !together && !common {
                        now[1] = now[0];
                        continue;
                    }
                    let (lp, lq) = (&loops_of[p], &loops_of[q]);
                    let outer: Vec<usize> = lp.iter().rev().copied().filter(|l| lq.contains(l)).collect();
                    let cases: Vec<(&[usize], Option<Unknown>)> = if together {
                        vec![(&[][..], None)]
                    } else {
                        (0..outer.len())
                            .map(|j| (&outer[..j], addresses.trip(facts.order[loops.header(outer[j])])))
                            .collect()
                    };
                    for (same, differ) in cases {
                        let variant = |info: &UnknownInfo| {
                            around[info.rank].iter().any(|l| {
                                let (in_p, in_q) = (lp.contains(l), lq.contains(l));
                                if together {
                                    in_p != in_q
                                } else {
                                    (in_p || in_q) && !same.contains(l)
                                }
                            })
                        };
                        let found = meet(
                            &mut addresses,
                            env,
                            (&hazards.accesses[p], &lanes[p]),
                            (&hazards.accesses[q], &lanes[q]),
                            &variant,
                            differ,
                        );
                        if let Some(idle) = found {
                            now[k].0 = true;
                            now[k].1 = [now[k].1[0] && idle[0], now[k].1[1] && idle[1]];
                        }
                    }
                }
                state.insert((p, q), now);
            }
            let judged = Judged {
                accesses: &hazards.accesses,
                lanes: &lanes,
                pairs: &pairs,
                sharing: &sharing,
            };
            let outcome = widened(&mut addresses, env, &judged, &mut regions, &mut open.stake, wave == 0);
            wave = open.after(outcome, &mut state, &start, wave);
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

#[cfg(test)]
pub(super) type Position = (BlockId, usize);

#[cfg(test)]
impl Hazards {
    pub(super) fn given(
        program: &Program,
        together: &[(Position, Position)],
        apart: &[(Position, Position)],
        idle: &[(Position, Position, bool)],
    ) -> Self {
        let facts = Facts::new(&program.ir, &program.parameter_inputs, &BTreeSet::new());
        let exec = program.registry.registers().exec;
        let exec_index = program.parameter_inputs.iter().position(
            |p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec),
        );
        let accesses = accesses(program, &facts, exec_index);
        let at = |x: Position| accesses.iter().position(|a| (a.block, a.index) == x).expect("an access at this position");
        let key = |p: Position, q: Position| {
            let (a, b) = (at(p), at(q));
            (a.min(b), a.max(b))
        };
        let mut hazards = Hazards {
            together: together.iter().map(|&(p, q)| key(p, q)).collect(),
            apart: apart.iter().map(|&(p, q)| key(p, q)).collect(),
            idle: idle
                .iter()
                .map(|&(reader, other, apart)| {
                    let (a, b) = key(reader, other);
                    (a, b, apart, if at(reader) == a { 0 } else { 1 })
                })
                .collect(),
            accesses: Vec::new(),
            meetings: Vec::new(),
        };
        hazards.accesses = accesses;
        hazards.meetings = hazards.positions(&facts.order);
        hazards
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
                        reach: Reach::Anywhere,
                    });
                }
                Inst::Target {
                    provenance,
                    op,
                    args,
                    outputs,
                } => {
                    let effect = program.registry.operation(*op).map(|spec| spec.effect);
                    if let Ok(Effect::ReadGlobal { every_lane }) = effect {
                        let predicate = (!every_lane)
                            .then(|| args.values().iter().copied().find(|a| f.types[a.0] == Ty::I1))
                            .flatten();
                        out.push(Access {
                            block: b,
                            index,
                            instruction: provenance.map_or(u64::MAX, |p| p >> 8),
                            kind: Kind::Read,
                            space: None,
                            bytes: RESOURCE_BYTES,
                            address: None,
                            predicate,
                            output: outputs.first().map(|o| o.0),
                            exec,
                            reach: reach(&program.registry, *op),
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
    region: Option<Region>,
    within: Regions,
    address: Option<Value>,
    bytes: u32,
    high: Option<Form>,
}

impl Place {
    fn touches(&self, other: &Place, env: &Environment) -> bool {
        let meets = |x: Option<Region>, y: Option<Region>| {
            overlapping(env, x.unwrap_or(Region::Exposed), y.unwrap_or(Region::Exposed))
        };
        match (self.region, other.region) {
            (Some(x), Some(y)) => overlapping(env, x, y),
            (Some(x), None) => other.within.reaches(Some(x), meets),
            (None, Some(y)) => self.within.reaches(Some(y), meets),
            (None, None) => self.within.overlaps(&other.within, meets),
        }
    }
}

fn places(addresses: &mut Addresses, a: &Access, found: &[Option<Regions>]) -> Vec<Option<Place>> {
    let within = |lane: usize| found[lane].clone().unwrap_or_else(|| Regions::one(None));
    (0..LANES)
        .map(|lane| {
            if !addresses.valid(lane) {
                return None;
            }
            let Some(address) = a.address else {
                let reached = addresses.resource_span((a.block, a.index), a.reach, lane);
                return Some(Place {
                    region: None,
                    within: within(lane),
                    address: reached.as_ref().map(|r| r.0.clone()),
                    bytes: reached.as_ref().map_or(a.bytes, |r| r.1),
                    high: reached.and_then(|r| r.2),
                });
            };
            let (value, _) = addresses.operand(address, a.block, lane, a.predicate);
            let region = match a.space {
                Some(Space::Lds) => Some(Region::Lds),
                Some(Space::Scratch) => Some(Region::Private),
                _ => value.region,
            };
            Some(Place {
                region,
                within: if region.is_some() { Regions::default() } else { within(lane) },
                address: Some(value),
                bytes: a.bytes,
                high: None,
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

fn shared(
    addresses: &mut Addresses,
    accesses: &[Access],
    env: &Environment,
    pairs: &[(usize, usize)],
    first: bool,
) -> Option<(Vec<Vec<Option<Regions>>>, BTreeSet<(usize, usize)>)> {
    let involved: BTreeSet<usize> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
    loop {
        let mut regions: Vec<Vec<Option<Regions>>> = accesses
            .iter()
            .enumerate()
            .map(|(i, a)| {
                (0..LANES)
                    .map(|lane| {
                        (involved.contains(&i) && addresses.valid(lane)).then(|| region_of(addresses, a, lane, false))
                    })
                    .collect()
            })
            .collect();
        if addresses.settle_loops() {
            if first {
                continue;
            }
            return None;
        }
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
            let a = &accesses[next];
            regions[next] = (0..LANES)
                .map(|lane| addresses.valid(lane).then(|| region_of(addresses, a, lane, true)))
                .collect();
            sharing.retain(|&(p, q)| (p != next && q != next) || may_share(env, &regions[p], &regions[q]));
        }
        if addresses.settle_loops() {
            if first {
                continue;
            }
            return None;
        }
        return Some((regions, sharing));
    }
}

#[derive(Default)]
struct Opening {
    wide: Option<Environment>,
    stake: Vec<u64>,
    only: Option<BTreeSet<(usize, usize)>>,
}

impl Opening {
    fn after(
        &mut self,
        outcome: Option<(Environment, Option<BTreeSet<(usize, usize)>>)>,
        state: &mut BTreeMap<(usize, usize), Judgments>,
        start: &BTreeMap<(usize, usize), Judgments>,
        wave: usize,
    ) -> usize {
        let Some((wider, affected)) = outcome else {
            self.only = None;
            return wave + 1;
        };
        self.wide = Some(wider);
        match affected {
            Some(affected) => {
                for pair in &affected {
                    state.insert(*pair, start[pair]);
                }
                self.only = Some(affected);
                wave
            }
            None => {
                *state = start.clone();
                self.only = None;
                0
            }
        }
    }
}

struct Judged<'a> {
    accesses: &'a [Access],
    lanes: &'a [Vec<Option<Place>>],
    pairs: &'a [(usize, usize)],
    sharing: &'a BTreeSet<(usize, usize)>,
}

fn widened(
    addresses: &mut Addresses,
    env: &Environment,
    judged: &Judged,
    regions: &mut [Vec<Option<Regions>>],
    stake: &mut Vec<u64>,
    first: bool,
) -> Option<(Environment, Option<BTreeSet<(usize, usize)>>)> {
    let known = stake.len();
    let mut refined: BTreeSet<usize> = BTreeSet::new();
    let mut open = env.exposed.clone();
    loop {
        for id in addresses.exposable() {
            if !stake.contains(&id) && !open.contains(&id) && !judged.matters(addresses, env, regions, id, &mut refined).is_empty() {
                stake.push(id);
            }
        }
        let kept = addresses.pending();
        let found = addresses.expose(&open, stake);
        let complete = !found.is_empty() && stake.iter().all(|id| open.contains(id) || found.contains(id));
        open.extend(found);
        if complete {
            addresses.forget(&kept);
        }
        if !addresses.settle_loops() {
            break;
        }
        for &i in &refined {
            regions[i] = judged.refined(addresses, i);
        }
    }
    let found: Vec<u64> = open[env.exposed.len()..].to_vec();
    if found.is_empty() && (first || stake.len() == known) {
        return None;
    }
    let affected = first.then(|| {
        found
            .iter()
            .flat_map(|&id| judged.matters(addresses, env, regions, id, &mut refined))
            .collect()
    });
    let mut wider = env.clone();
    wider.exposed = open;
    Some((wider, affected))
}

impl Judged<'_> {
    fn refined(&self, addresses: &mut Addresses, i: usize) -> Vec<Option<Regions>> {
        (0..LANES)
            .map(|lane| addresses.valid(lane).then(|| region_of(addresses, &self.accesses[i], lane, true)))
            .collect()
    }

    fn matters(
        &self,
        addresses: &mut Addresses,
        env: &Environment,
        regions: &mut [Vec<Option<Regions>>],
        id: u64,
        refined: &mut BTreeSet<usize>,
    ) -> Vec<(usize, usize)> {
        let target = Some(Region::Allocation(id));
        let lost = |x: &Place| x.region.is_none() && x.within.lost();
        let reaches = |x: &Place| x.region == target || (x.region.is_none() && x.within.reaches(target, |a, b| a == b));
        let faces = |a: usize, b: usize| self.lanes[a].iter().flatten().any(lost) && self.lanes[b].iter().flatten().any(reaches);
        let mut wider = env.clone();
        wider.exposed.push(id);
        let mut out = Vec::new();
        for &(p, q) in self.pairs {
            let placed = !self.lanes[p].is_empty() && !self.lanes[q].is_empty();
            if placed || self.sharing.contains(&(p, q)) {
                if faces(p, q) || faces(q, p) {
                    out.push((p, q));
                }
                continue;
            }
            if !may_share(&wider, &regions[p], &regions[q]) {
                continue;
            }
            for i in [p, q] {
                if refined.insert(i) {
                    regions[i] = self.refined(addresses, i);
                }
            }
            if may_share(&wider, &regions[p], &regions[q]) {
                out.push((p, q));
            }
        }
        out
    }
}

fn region_of(addresses: &mut Addresses, a: &Access, lane: usize, refine: bool) -> Regions {
    match a.space {
        Some(Space::Lds) => Regions::one(Some(Region::Lds)),
        Some(Space::Scratch) => Regions::one(Some(Region::Private)),
        _ => match a.address {
            Some(x) => addresses.regions(x, lane, a.predicate, refine),
            None => addresses.read_regions((a.block, a.index), lane, a.exec, refine),
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
    differ: Option<Unknown>,
) -> Option<[bool; 2]> {
    let mut idle: Option<[bool; 2]> = None;
    for (a, x) in p.iter().enumerate() {
        let Some(x) = x else { continue };
        for (b, y) in q.iter().enumerate() {
            let Some(y) = y else { continue };
            if a == b || !x.touches(y, env) {
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
                if d >= y.bytes && d.wrapping_neg() >= x.bytes {
                    continue;
                }
            } else if !may_overlap(unknowns, &xa.form, &ya.form, x.bytes, y.bytes, variant, differ) {
                continue;
            }
            if above(addresses, (pa, a, xa, x.high.as_ref(), x.bytes), (qa, b, ya, y.high.as_ref(), y.bytes), variant, differ) {
                continue;
            }
            if !runs(addresses, pa, a) || !runs(addresses, qa, b) {
                continue;
            }
            if excluded(addresses, (pa, a, &xa.form, x.bytes), (qa, b, &ya.form, y.bytes), variant) {
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
    (pa, a, x, x_bytes): (&Access, usize, &Form, u32),
    (qa, b, y, y_bytes): (&Access, usize, &Form, u32),
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
    for t in -(x_bytes as i64) + 1..y_bytes as i64 {
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
                this.reaches_block_with_value(access.block)
                    && access
                        .predicate
                        .is_none_or(|p| this.bit(p, lane, None).0 != Some(false))
            };
            !(runs(this, pa, a) && runs(this, qa, b))
        })
    })
}

fn above(
    addresses: &mut Addresses,
    (pa, a, x, xh, x_bytes): (&Access, usize, &Value, Option<&Form>, u32),
    (qa, b, y, yh, y_bytes): (&Access, usize, &Value, Option<&Form>, u32),
    variant: &dyn Fn(&UnknownInfo) -> bool,
    differ: Option<Unknown>,
) -> bool {
    let mut high = |access: &Access, lane: usize, given: Option<&Form>| match access.address {
        Some(v) if addresses.wide(v) => Some(addresses.high(v, lane)),
        Some(_) => None,
        None => given.cloned(),
    };
    let (Some(hx), Some(hy)) = (high(pa, a, xh), high(qa, b, yh)) else {
        return false;
    };
    let unknowns = &addresses.unknowns;
    if let (Some(lx), Some(ly)) = (x.form.as_constant(), y.form.as_constant()) {
        let low = ly as i64 - lx as i64;
        return [-1i64, 0, 1].iter().all(|&k| {
            let d = k * (1i64 << 32) + low;
            !(d > -(y_bytes as i64) && d < x_bytes as i64)
                || !may_overlap(unknowns, &hx.add(&Form::constant(k as u32)), &hy, 1, 1, variant, differ)
        });
    }
    let same = x.form == y.form && x.form.terms.iter().all(|&(u, _)| !variant(&unknowns[u as usize]));
    let window = if same { 1 } else { 2 };
    !may_overlap(unknowns, &hx, &hy, window, window, variant, differ)
}

fn may_overlap(
    unknowns: &[UnknownInfo],
    x: &Form,
    y: &Form,
    x_bytes: u32,
    y_bytes: u32,
    variant: &dyn Fn(&UnknownInfo) -> bool,
    differ: Option<Unknown>,
) -> bool {
    let mut sum = Sum::default();
    let mut apart = None;
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
        let set = if info.values.is_some() { u } else { NO_SET };
        if variant(info) {
            if let (true, true, Some((lo, hi)), NO_SET) = (Some(u) == differ, cx == cy, info.range, set) {
                apart = Some((cx, hi - lo));
                continue;
            }
            sum.add(cx, info.range, set);
            sum.add(cy.wrapping_neg(), info.range, set);
        } else {
            sum.add(cx.wrapping_sub(cy), info.range, set);
        }
    }
    let constant = x.constant.wrapping_sub(y.constant) as i64;
    match apart {
        None => sum.within(unknowns, constant, x_bytes, y_bytes),
        Some((_, 0)) => false,
        Some((c, span)) => [c, c.wrapping_neg()].iter().any(|&c| {
            let mut sum = sum;
            sum.add(c, Some((1, span)), NO_SET);
            sum.within(unknowns, constant, x_bytes, y_bytes)
        }),
    }
}

const BOUNDED: usize = 8;
const PARTIAL_SUMS: usize = 4096;
const NO_SET: Unknown = Unknown::MAX;

#[derive(Clone, Copy)]
struct Sum {
    low: i64,
    high: i64,
    modulus: u64,
    divisor: u64,
    free: u64,
    terms: [(u32, u32, u32, Unknown); BOUNDED],
    count: usize,
}

impl Default for Sum {
    fn default() -> Self {
        Sum {
            low: 0,
            high: 0,
            modulus: 1 << 32,
            divisor: 0,
            free: 1 << 32,
            terms: [(0, 0, 0, NO_SET); BOUNDED],
            count: 0,
        }
    }
}

impl Sum {
    fn add(&mut self, c: u32, range: Option<(u32, u32)>, set: Unknown) {
        if c == 0 {
            return;
        }
        let signed = c as i32 as i64;
        match range {
            Some((lo, hi)) if (hi as i64 - lo as i64) * signed.abs() < 1 << 32 => {
                let (a, b) = (signed * lo as i64, signed * hi as i64);
                self.low = self.low.wrapping_add(a.min(b));
                self.high = self.high.wrapping_add(a.max(b));
                self.divisor = gcd(self.divisor, signed.unsigned_abs());
            }
            _ => self.modulus = self.modulus.min(1u64 << c.trailing_zeros()),
        }
        match range {
            Some((lo, hi)) if self.count < BOUNDED => {
                self.terms[self.count] = (c, lo, hi, set);
                self.count += 1;
            }
            Some(_) => self.count = BOUNDED + 1,
            None => self.free = self.free.min(1u64 << c.trailing_zeros()),
        }
    }

    fn within(&self, unknowns: &[UnknownInfo], constant: i64, x_bytes: u32, y_bytes: u32) -> bool {
        self.roughly_within(constant, x_bytes, y_bytes) && self.exactly_within(unknowns, constant, x_bytes, y_bytes)
    }

    fn roughly_within(&self, constant: i64, x_bytes: u32, y_bytes: u32) -> bool {
        let step = gcd(self.divisor, self.modulus) as i64;
        let window = -(x_bytes as i64) + 1..y_bytes as i64;
        if !window.clone().any(|t| (t - constant).rem_euclid(step) == 0) {
            return false;
        }
        let span = self.high.wrapping_sub(self.low);
        if span + 1 >= self.modulus as i64 {
            return true;
        }
        let m = self.modulus as i64;
        window.into_iter().any(|t| {
            let target = (t - constant).rem_euclid(m);
            let start = self.low.rem_euclid(m);
            let offset = (target - start).rem_euclid(m);
            offset <= span
        })
    }

    fn exactly_within(&self, unknowns: &[UnknownInfo], constant: i64, x_bytes: u32, y_bytes: u32) -> bool {
        if self.count == 0 || self.count > BOUNDED {
            return true;
        }
        let m = self.free;
        let mask = m - 1;
        let terms = &self.terms[..self.count];
        let mut widest: Vec<usize> = (0..terms.len()).filter(|&k| terms[k].3 == NO_SET).collect();
        widest.sort_by_key(|&k| std::cmp::Reverse(terms[k].2 - terms[k].1));
        let (first, second) = (widest.first().copied(), widest.get(1).copied());
        let hits = |s: u64| {
            (-(x_bytes as i64) + 1..y_bytes as i64).any(|t| {
                let target = ((t - constant).rem_euclid(m as i64) as u64).wrapping_sub(s) & mask;
                let Some(first) = first else {
                    return target == 0;
                };
                let (c1, lo1, hi1, _) = terms[first];
                match second {
                    None => solvable(c1, lo1, hi1, target, m),
                    Some(k) => {
                        let (c2, lo2, hi2, _) = terms[k];
                        let start = (c1 as u64).wrapping_mul(lo1 as u64).wrapping_add((c2 as u64).wrapping_mul(lo2 as u64));
                        let target = target.wrapping_sub(start) & mask;
                        two_solvable(c1 as u64, (hi1 - lo1) as u64, c2 as u64, (hi2 - lo2) as u64, target, m)
                    }
                }
            })
        };
        let mut sums: Vec<u64> = vec![0];
        for (k, &(c, lo, hi, set)) in terms.iter().enumerate() {
            if Some(k) == first || Some(k) == second {
                continue;
            }
            let values = (set != NO_SET).then(|| unknowns[set as usize].values.clone()).flatten();
            let width = values.as_ref().map_or((hi - lo) as usize + 1, |v| v.len());
            if width > PARTIAL_SUMS || sums.len() * width > PARTIAL_SUMS * 4 {
                return true;
            }
            let mut next: Vec<u64> = Vec::with_capacity(sums.len() * width);
            for &s in &sums {
                match &values {
                    Some(values) => {
                        for &u in values.iter() {
                            next.push(s.wrapping_add((c as u64).wrapping_mul(u as u64)) & mask);
                        }
                    }
                    None => {
                        for u in lo..=hi {
                            next.push(s.wrapping_add((c as u64).wrapping_mul(u as u64)) & mask);
                        }
                    }
                }
            }
            next.sort_unstable();
            next.dedup();
            if next.len() > PARTIAL_SUMS {
                return true;
            }
            sums = next;
        }
        sums.into_iter().any(hits)
    }
}

fn odd_inverse(odd: u64) -> u64 {
    (0..6).fold(odd, |inv: u64, _| inv.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(inv))))
}

fn floor_sum(mut n: u128, mut m: u128, mut a: u128, mut b: u128) -> u128 {
    let mut sum = 0u128;
    loop {
        if a >= m {
            sum += n * n.saturating_sub(1) / 2 * (a / m);
            a %= m;
        }
        if b >= m {
            sum += n * (b / m);
            b %= m;
        }
        let top = a * n + b;
        if top < m {
            return sum;
        }
        n = top / m;
        b = top % m;
        std::mem::swap(&mut m, &mut a);
    }
}

fn some_low_residue(n: u64, modulus: u64, a: u64, b: u64, bound: u64) -> bool {
    if n == 0 {
        return false;
    }
    if bound >= modulus - 1 {
        return true;
    }
    let (n, m, a, b, bound) = (n as u128, modulus as u128, a as u128, b as u128, bound as u128);
    floor_sum(n, m, b, a) + n > floor_sum(n, m, b, a + m - 1 - bound)
}

fn two_solvable(c1: u64, w1: u64, c2: u64, w2: u64, target: u64, m: u64) -> bool {
    let mask = m - 1;
    let (c1, c2, target) = (c1 & mask, c2 & mask, target & mask);
    if c1 == 0 {
        return solvable(c2 as u32, 0, w2 as u32, target, m);
    }
    if c2 == 0 {
        return solvable(c1 as u32, 0, w1 as u32, target, m);
    }
    let g = 1u64 << c1.trailing_zeros();
    let g2 = (1u64 << c2.trailing_zeros()).min(g);
    if target % g2 != 0 {
        return false;
    }
    let step = g / g2;
    let first = if step == 1 {
        0
    } else {
        ((target / g2) % step).wrapping_mul(odd_inverse(c2 / g2)) & (step - 1)
    };
    if first > w2 {
        return false;
    }
    let count = (w2 - first) / step + 1;
    let big = m / g;
    let low = big - 1;
    let inverse = odd_inverse(c1 / g) & low;
    let base = target.wrapping_sub(c2.wrapping_mul(first)) & mask;
    let a = (base / g).wrapping_mul(inverse) & low;
    let delta = (c2.wrapping_mul(step) & mask) / g;
    let b = (big - (delta & low)).wrapping_mul(inverse) & low;
    some_low_residue(count, big, a, b, w1)
}

fn solvable(c: u32, lo: u32, hi: u32, target: u64, m: u64) -> bool {
    let c = c as u64 % m;
    if c == 0 {
        return target % m == 0;
    }
    let g = 1u64 << c.trailing_zeros();
    if target % g != 0 {
        return false;
    }
    let reduced = m / g;
    let mask = reduced - 1;
    let odd = c / g;
    let inverse = (0..6).fold(odd, |inv: u64, _| inv.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(inv))));
    let r = ((target / g).wrapping_mul(inverse)) & mask;
    let first = lo as u64 + (r.wrapping_sub(lo as u64) & mask);
    first <= hi as u64
}

#[cfg(test)]
pub(super) fn may_overlap_for_tests(unknowns: &[UnknownInfo], x: &Form, y: &Form) -> bool {
    may_overlap(unknowns, x, y, 1, 1, &|_: &UnknownInfo| false, None)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::rdna_spmd::engine::WORKGROUP_ID_X;

    fn pair(h: &Hazards, p: (BlockId, usize), q: (BlockId, usize)) -> (usize, usize) {
        let at = |x: (BlockId, usize)| {
            h.accesses
                .iter()
                .position(|a| (a.block, a.index) == x)
                .expect("an access at this position")
        };
        let (p, q) = (at(p), at(q));
        (p.min(q), p.max(q))
    }

    fn overlaps(x: u32, x_bytes: u32, y: u32, y_bytes: u32) -> bool {
        let d = x.wrapping_sub(y) as u64;
        d < y_bytes as u64 || d > (1u64 << 32) - x_bytes as u64
    }

    fn info(rank: usize, range: Option<(u32, u32)>) -> UnknownInfo {
        UnknownInfo {
            shared: true,
            block: BlockId(0),
            rank,
            range,
            through: Vec::new(),
            values: None,
        }
    }

    fn values(info: &UnknownInfo) -> Vec<u32> {
        if let Some(set) = &info.values {
            return set.to_vec();
        }
        let (lo, hi) = info.range.expect("a small range");
        (lo..=hi).collect()
    }

    fn evaluate(form: &Form, values: &[u32]) -> u32 {
        form.terms
            .iter()
            .fold(form.constant, |acc, &(u, c)| acc.wrapping_add(c.wrapping_mul(values[u as usize])))
    }

    fn exact_overlap(unknowns: &[UnknownInfo], variant: &[bool], x: &Form, y: &Form, x_bytes: u32, y_bytes: u32) -> bool {
        let n = unknowns.len();
        let mut choices: Vec<Vec<u32>> = Vec::new();
        for info in unknowns {
            choices.push(values(info));
        }
        for (u, info) in unknowns.iter().enumerate() {
            choices.push(if variant[u] { values(info) } else { vec![0] });
        }
        let mut index = vec![0usize; 2 * n];
        loop {
            let xs: Vec<u32> = (0..n).map(|u| choices[u][index[u]]).collect();
            let ys: Vec<u32> = (0..n)
                .map(|u| if variant[u] { choices[n + u][index[n + u]] } else { xs[u] })
                .collect();
            if overlaps(evaluate(x, &xs), x_bytes, evaluate(y, &ys), y_bytes) {
                return true;
            }
            let mut k = 0;
            loop {
                if k == 2 * n {
                    return false;
                }
                index[k] += 1;
                if index[k] < choices[k].len() {
                    break;
                }
                index[k] = 0;
                k += 1;
            }
        }
    }

    fn random_form(r: &mut Random, n: usize) -> Form {
        const COEFFICIENTS: [u32; 10] = [1, 2, 3, 4, 6, 8, 0xffff_fffc, 0xffff_ffff, 0x8000_0000, 0x4000_0000];
        let mut terms = Vec::new();
        for u in 0..n {
            if r.below(3) != 0 {
                terms.push((u as u32, COEFFICIENTS[r.below(COEFFICIENTS.len() as u64) as usize]));
            }
        }
        Form {
            constant: match r.below(3) {
                0 => r.below(16) as u32,
                1 => (r.below(16) as u32).wrapping_neg(),
                _ => r.next() as u32,
            },
            terms,
        }
    }

    fn random_case(r: &mut Random, top: bool) -> (Vec<UnknownInfo>, Vec<bool>, Form, Form, u32, u32) {
        let n = 1 + r.below(3) as usize;
        let unknowns: Vec<UnknownInfo> = (0..n)
            .map(|u| {
                let lo = match r.below(3) {
                    0 => 0,
                    1 => r.below(64) as u32,
                    _ if top => u32::MAX - 8,
                    _ => 1 << 16,
                };
                info(u, Some((lo, lo + r.below(6) as u32)))
            })
            .collect();
        let variant: Vec<bool> = (0..n).map(|_| r.below(2) == 0).collect();
        let (x, y) = (random_form(r, n), random_form(r, n));
        let sizes = [1, 2, 4, 8, 64];
        let (a, b) = (sizes[r.below(5) as usize], sizes[r.below(5) as usize]);
        (unknowns, variant, x, y, a, b)
    }

    fn random_set_case(r: &mut Random) -> (Vec<UnknownInfo>, Vec<bool>, Form, Form, u32, u32) {
        let (mut unknowns, variant, x, y, a, b) = random_case(r, false);
        for (u, info) in unknowns.iter_mut().enumerate() {
            if r.below(3) == 0 {
                continue;
            }
            let lo = info.range.unwrap().0;
            let mut set: Vec<u32> = (0..1 + r.below(5)).map(|_| lo.wrapping_add(1 << r.below(6)).wrapping_sub(1)).collect();
            set.sort_unstable();
            set.dedup();
            *info = UnknownInfo {
                range: Some((set[0], set[set.len() - 1])),
                values: Some(set.into()),
                ..self::info(u, None)
            };
        }
        (unknowns, variant, x, y, a, b)
    }

    #[test]
    fn may_overlap_never_misses_an_overlap_over_value_sets() {
        let mut r = Random::new(13);
        for _ in 0..20000 {
            let (unknowns, variant, x, y, a, b) = random_set_case(&mut r);
            let exact = exact_overlap(&unknowns, &variant, &x, &y, a, b);
            let found = may_overlap(&unknowns, &x, &y, a, b, &|i: &UnknownInfo| variant[i.rank], None);
            assert!(
                found || !exact,
                "{:?} ({} bytes) and {:?} ({} bytes) over {:?} with variant {:?} overlap, but may_overlap says they do not",
                x, a, y, b, unknowns.iter().map(|i| (i.range, i.values.clone())).collect::<Vec<_>>(), variant
            );
        }
    }

    #[test]
    fn may_overlap_is_exact_over_value_sets() {
        let mut r = Random::new(17);
        for _ in 0..20000 {
            let (unknowns, variant, x, y, a, b) = random_set_case(&mut r);
            let exact = exact_overlap(&unknowns, &variant, &x, &y, a, b);
            let found = may_overlap(&unknowns, &x, &y, a, b, &|i: &UnknownInfo| variant[i.rank], None);
            assert_eq!(
                found, exact,
                "{:?} ({} bytes) and {:?} ({} bytes) over {:?} with variant {:?}",
                x, a, y, b, unknowns.iter().map(|i| (i.range, i.values.clone())).collect::<Vec<_>>(), variant
            );
        }
    }

    #[test]
    fn may_overlap_never_misses_an_overlap() {
        let mut r = Random::new(7);
        for _ in 0..20000 {
            let (unknowns, variant, x, y, a, b) = random_case(&mut r, false);
            let exact = exact_overlap(&unknowns, &variant, &x, &y, a, b);
            let found = may_overlap(&unknowns, &x, &y, a, b, &|i: &UnknownInfo| variant[i.rank], None);
            assert!(
                found || !exact,
                "{:?} ({} bytes) and {:?} ({} bytes) over {:?} with variant {:?} overlap, but may_overlap says they do not",
                x, a, y, b, unknowns.iter().map(|i| i.range).collect::<Vec<_>>(), variant
            );
        }
    }

    #[test]
    fn may_overlap_is_exact_on_small_ranges() {
        let mut r = Random::new(11);
        for _ in 0..20000 {
            let (unknowns, variant, x, y, a, b) = random_case(&mut r, true);
            let exact = exact_overlap(&unknowns, &variant, &x, &y, a, b);
            let found = may_overlap(&unknowns, &x, &y, a, b, &|i: &UnknownInfo| variant[i.rank], None);
            assert_eq!(
                found, exact,
                "{:?} ({} bytes) and {:?} ({} bytes) over {:?} with variant {:?}",
                x, a, y, b, unknowns.iter().map(|i| i.range).collect::<Vec<_>>(), variant
            );
        }
    }

    #[test]
    fn may_overlap_takes_large_terms_near_the_top_of_the_word() {
        let unknowns = [info(0, Some((u32::MAX - 1, u32::MAX))), info(1, Some((u32::MAX - 1, u32::MAX)))];
        let x = Form {
            constant: 0,
            terms: vec![(0, 0x8000_0000), (1, 0x8000_0000)],
        };
        let y = Form::constant(0);
        assert!(exact_overlap(&unknowns, &[false, false], &x, &y, 1, 1));
        assert!(may_overlap(&unknowns, &x, &y, 1, 1, &|_: &UnknownInfo| false, None));
    }

    #[test]
    fn two_solvable_matches_enumeration() {
        let mut r = Random::new(53);
        let mut wrong = Vec::new();
        for _ in 0..3000 {
            let m = 1u64 << (1 + r.below(32));
            let pick = |r: &mut Random| match r.below(4) {
                0 => r.below(8),
                1 => (r.next() as u32 as u64) << r.below(8),
                2 => r.next() as u32 as u64,
                _ => 1u64 << r.below(32),
            };
            let (c1, c2, target) = (pick(&mut r) & (m - 1), pick(&mut r) & (m - 1), r.next() as u32 as u64 & (m - 1));
            let (w1, w2) = (r.below(70), r.below(70));
            let brute = (0..=w1).any(|u| (0..=w2).any(|v| (c1.wrapping_mul(u).wrapping_add(c2.wrapping_mul(v)).wrapping_sub(target)) & (m - 1) == 0));
            if two_solvable(c1, w1, c2, w2, target, m) != brute {
                wrong.push(format!("{} u + {} v = {} mod {:#x}, u <= {}, v <= {}: expected {}", c1, c2, target, m, w1, w2, brute));
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(5)]);
    }

    #[test]
    fn some_low_residue_matches_enumeration() {
        let mut r = Random::new(59);
        let mut wrong = Vec::new();
        for _ in 0..3000 {
            let bits = r.below(20);
            let m = 1 + r.below(1 << bits);
            let (a, b, bound, n) = (r.below(m), r.below(m), r.below(m), r.below(200));
            let brute = (0..n).any(|j| (a + b * j) % m <= bound);
            if some_low_residue(n, m, a, b, bound) != brute {
                wrong.push(format!("({} + {} j) mod {} <= {} for j < {}: expected {}", a, b, m, bound, n, brute));
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(5)]);
    }

    #[test]
    fn may_overlap_finds_four_u_plus_six_v_hitting_ten_over_wide_ranges() {
        let unknowns = [info(0, Some((0, 10000))), info(1, Some((0, 10000)))];
        let x = Form {
            constant: 0,
            terms: vec![(0, 4), (1, 6)],
        };
        let y = Form::constant(10);
        assert!(may_overlap(&unknowns, &x, &y, 1, 1, &|_: &UnknownInfo| false, None), "u = 1, v = 1 gives 10");
    }

    #[test]
    fn may_overlap_sees_that_four_u_plus_six_v_never_hits_two_over_wide_ranges() {
        let unknowns = [info(0, Some((0, 10000))), info(1, Some((0, 10000)))];
        let x = Form {
            constant: 0,
            terms: vec![(0, 4), (1, 6)],
        };
        let y = Form::constant(2);
        assert!(!may_overlap(&unknowns, &x, &y, 1, 1, &|_: &UnknownInfo| false, None), "4u + 6v is 0, 4, 6, 8, ... for u, v >= 0, never 2");
    }

    fn three_wide_terms(target: u32) -> bool {
        let unknowns = [info(0, Some((0, 10000))), info(1, Some((0, 10000))), info(2, Some((0, 10000)))];
        let x = Form {
            constant: 0,
            terms: vec![(0, 4), (1, 6), (2, 10)],
        };
        may_overlap(&unknowns, &x, &Form::constant(target), 1, 1, &|_: &UnknownInfo| false, None)
    }

    #[test]
    fn may_overlap_finds_four_u_plus_six_v_plus_ten_w_hitting_fourteen_over_wide_ranges() {
        assert!(three_wide_terms(14), "u = w = 1, v = 0 gives 14");
    }

    #[test]
    fn may_overlap_sees_that_four_u_plus_six_v_plus_ten_w_never_hits_two_over_wide_ranges() {
        assert!(!three_wide_terms(2), "2u + 3v + 5w = 1 has no solution with u, v, w >= 0");
    }

    #[test]
    fn may_overlap_sees_that_four_u_plus_six_v_never_hits_two() {
        let unknowns = [info(0, Some((0, 1))), info(1, Some((0, 1)))];
        let x = Form {
            constant: 0,
            terms: vec![(0, 4), (1, 6)],
        };
        let y = Form::constant(2);
        assert!(!exact_overlap(&unknowns, &[false, false], &x, &y, 1, 1));
        assert!(!may_overlap(&unknowns, &x, &y, 1, 1, &|_: &UnknownInfo| false, None));
    }

    #[test]
    fn may_overlap_without_ranges_follows_the_power_of_two_in_the_coefficient() {
        let unknowns = [info(0, None)];
        let x = Form {
            constant: 0,
            terms: vec![(0, 8)],
        };
        for (y, bytes, expected) in [(4, 4, false), (4, 5, true), (0x1_0000_0000u64 as u32, 1, true), (7, 1, false), (16, 1, true)] {
            assert_eq!(
                may_overlap(&unknowns, &x, &Form::constant(y), 4, bytes, &|_: &UnknownInfo| false, None),
                expected,
                "8u (4 bytes) against {} ({} bytes)",
                y,
                bytes
            );
        }
    }

    #[test]
    fn overlapping_follows_the_region_table() {
        use Region::*;
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut env = env;
        env.exposed = vec![1];
        let all = [Allocation(1), Allocation(2), Exposed, Kernarg, Dispatch, Lds, Private];
        let expected = |a: Region, b: Region| match (a, b) {
            (Lds, Lds) => true,
            (Allocation(x), Allocation(y)) => x == y,
            (Allocation(1), Exposed) | (Exposed, Allocation(1)) | (Exposed, Exposed) => true,
            _ => false,
        };
        for a in all {
            for b in all {
                assert_eq!(overlapping(&env, a, b), expected(a, b), "{:?} and {:?}", a, b);
            }
        }
    }

    fn uniform_stores(lanes: u32) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(lanes, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_two_lanes_storing_one_word() {
        assert!(uniform_stores(32));
        assert!(uniform_stores(2));
    }

    #[test]
    fn find_reports_nothing_when_the_workgroup_has_one_lane() {
        assert!(!uniform_stores(1));
    }

    #[test]
    fn find_keeps_lanes_apart_that_store_their_own_words() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let address = byte_offset(&mut b, e, buf, lane, 4);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, address, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, address, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)));
    }

    #[test]
    fn find_reports_neighbours_whose_words_meet() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let four = b.constant(e, Ty::I64, 4);
        let next = b.int(e, IntOp::Add, own, four);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, own, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, next, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(h.together.contains(&pair(&h, s1, s2)));
    }

    #[test]
    fn find_reports_lanes_that_share_an_x_across_rows() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let mask = b.constant(e, Ty::I32, 0x3ff);
        let x = b.int(e, IntOp::And, k.item, mask);
        let address = byte_offset(&mut b, e, buf, x, 4);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, address, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, address, zero, k.exec);
        let mut env = environment(16, &[(0, 1, 0x1000)]);
        env.block = [16, 2, 1];
        let h = Hazards::find(&b.program(), &env);
        assert!(h.together.contains(&pair(&h, s1, s2)));
    }

    #[test]
    fn find_sign_extends_a_true_bit_to_all_ones() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, other, lane, 4);
        let v = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Eq, v, zero);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, c));
        let four = b.constant(e, Ty::I64, 4);
        let offset = b.int(e, IntOp::Mul, wide, four);
        let low = b.int(e, IntOp::Add, buf, offset);
        let minus_four = b.constant(e, Ty::I64, (-4i64) as u64);
        let below = b.int(e, IntOp::Add, buf, minus_four);
        let s1 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, low, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, below, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        assert!(
            h.together.contains(&pair(&h, s1, s2)),
            "a lane whose word is zero stores at buf - 4 through sext(true) * 4, where every lane stores next"
        );
    }

    #[test]
    fn find_reads_a_projected_bit_from_the_lane_own_word() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let loaded = b.load(e, Space::Global, MemSize::B32, other, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let three = b.constant(e, Ty::I32, 3);
        let third = b.cmp(e, IntPred::Eq, lane, three);
        let zero = b.constant(e, Ty::I32, 0);
        let word = b.core(e, Ty::I32, Op::Select(third, zero, loaded));
        let shifted = b.int(e, IntOp::LShr, word, three);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let s1 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, zero, bit);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        assert!(
            h.together.contains(&pair(&h, s1, s2)),
            "every lane but lane 3 stores when bit 3 of the loaded word is set"
        );
    }

    fn counted_loop(trips: u64, lane_step: bool) -> (Hazards, (usize, usize)) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero]);
        let index = if lane_step {
            let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
            b.int(body, IntOp::Add, p[1], lane)
        } else {
            p[1]
        };
        let address = byte_offset(&mut b, body, buf, index, 4);
        let data = b.constant(body, Ty::I32, 0);
        let s = b.here(body);
        b.store(body, Space::Global, MemSize::B32, address, data, p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[1], one);
        let limit = b.constant(body, Ty::I32, trips);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, s, s);
        (h, key)
    }

    struct Looped {
        b: Build,
        buf: ValueId,
        body: BlockId,
        exec: ValueId,
        index: ValueId,
        exit: BlockId,
        last: ValueId,
    }

    fn looped(trips: u64) -> Looped {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32]);
        let (latch, l) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, x) = b.block(&[Ty::I1, Ty::I32]);
        b.br(e, body, vec![k.exec, zero]);
        b.br(body, latch, vec![p[0], p[1]]);
        let one = b.constant(latch, Ty::I32, 1);
        let next = b.int(latch, IntOp::Add, l[1], one);
        let limit = b.constant(latch, Ty::I32, trips);
        let again = b.cmp(latch, IntPred::Ult, next, limit);
        b.cond_br(latch, again, (body, vec![l[0], next]), (exit, vec![l[0], l[1]]));
        Looped {
            b,
            buf,
            body,
            exec: p[0],
            index: p[1],
            exit,
            last: x[1],
        }
    }

    fn store_at(b: &mut Build, block: BlockId, address: ValueId, mask: ValueId) -> (BlockId, usize) {
        let zero = b.constant(block, Ty::I32, 0);
        let at = b.here(block);
        b.store(block, Space::Global, MemSize::B32, address, zero, mask);
        at
    }

    #[test]
    fn find_reports_lanes_that_meet_through_a_mask_of_the_loop_index() {
        let Looped {
            mut b,
            buf,
            body,
            exec,
            index,
            ..
        } = looped(4);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let slid = b.int(body, IntOp::Add, index, lane);
        let three = b.constant(body, Ty::I32, 3);
        let slot = b.int(body, IntOp::And, slid, three);
        let address = byte_offset(&mut b, body, buf, slot, 4);
        let s = store_at(&mut b, body, address, exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(h.together.contains(&pair(&h, s, s)), "lanes 0 and 4 store to one word in every iteration");
    }

    #[test]
    fn find_keeps_apart_the_rows_of_a_loop_that_strides_by_the_wave() {
        let Looped {
            mut b,
            buf,
            body,
            exec,
            index,
            ..
        } = looped(4);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let row = b.constant(body, Ty::I32, 32);
        let base = b.int(body, IntOp::Mul, index, row);
        let item = b.int(body, IntOp::Add, base, lane);
        let address = byte_offset(&mut b, body, buf, item, 4);
        let s = store_at(&mut b, body, address, exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s, s)), "every lane stores its own word of its own row");
    }

    #[test]
    fn find_reports_the_index_a_loop_leaves_behind() {
        let Looped {
            mut b,
            buf,
            exit,
            last,
            ..
        } = looped(4);
        let exec = b.f.blocks[&exit].params[0].0;
        let address = byte_offset(&mut b, exit, buf, last, 4);
        let s1 = store_at(&mut b, exit, address, exec);
        let s2 = store_at(&mut b, exit, address, exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(h.together.contains(&pair(&h, s1, s2)), "every lane stores to the word the last index names");
    }

    #[test]
    fn find_reports_a_word_read_one_iteration_after_another_lane_wrote_it() {
        let Looped {
            mut b,
            buf,
            body,
            exec,
            index,
            ..
        } = looped(4);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let slid = b.int(body, IntOp::Add, index, lane);
        let address = byte_offset(&mut b, body, buf, slid, 4);
        let read = b.here(body);
        b.load(body, Space::Global, MemSize::B32, address, exec);
        let four = b.constant(body, Ty::I64, 4);
        let ahead = b.int(body, IntOp::Add, address, four);
        let s = store_at(&mut b, body, ahead, exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, read, s);
        assert!(h.together.contains(&key), "lane a + 1 reads in iteration i the word lane a writes in iteration i");
        assert!(h.apart.contains(&key), "lane a reads in iteration i + 2 the word lane a + 1 wrote in iteration i");
    }

    #[test]
    fn find_follows_a_pointer_the_loop_advances() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let start = byte_offset(&mut b, e, buf, lane, 4);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, start, zero]);
        let s = store_at(&mut b, body, p[1], p[0]);
        let stride = b.constant(body, Ty::I64, 128);
        let moved = b.int(body, IntOp::Add, p[1], stride);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let limit = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], moved, next]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s, s)), "each lane walks its own column of 128-byte rows");
    }

    #[test]
    fn find_reports_a_pointer_the_loop_advances_by_less_than_the_wave() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let start = byte_offset(&mut b, e, buf, lane, 4);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, start, zero]);
        let s = store_at(&mut b, body, p[1], p[0]);
        let stride = b.constant(body, Ty::I64, 4);
        let moved = b.int(body, IntOp::Add, p[1], stride);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let limit = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], moved, next]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, s, s);
        assert!(h.apart.contains(&key), "lane a + 1 stores in iteration i the word lane a stores in iteration i + 1");
        assert!(!h.together.contains(&key), "within one iteration the lanes store their own words");
    }

    fn nested(item: impl Fn(&mut Build, BlockId, ValueId, ValueId) -> ValueId) -> (Hazards, (usize, usize)) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let zero = b.constant(e, Ty::I32, 0);
        let (outer, o) = b.block(&[Ty::I1, Ty::I32]);
        let (inner, i) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (step, t) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, outer, vec![k.exec, zero]);
        let z = b.constant(outer, Ty::I32, 0);
        b.br(outer, inner, vec![o[0], o[1], z]);
        let four = b.constant(inner, Ty::I32, 4);
        let item = item(&mut b, inner, i[1], i[2]);
        let address = byte_offset(&mut b, inner, buf, item, 4);
        let s = store_at(&mut b, inner, address, i[0]);
        let one = b.constant(inner, Ty::I32, 1);
        let next = b.int(inner, IntOp::Add, i[2], one);
        let again = b.cmp(inner, IntPred::Ult, next, four);
        b.cond_br(inner, again, (inner, vec![i[0], i[1], next]), (step, vec![i[0], i[1]]));
        let one = b.constant(step, Ty::I32, 1);
        let next = b.int(step, IntOp::Add, t[1], one);
        let four = b.constant(step, Ty::I32, 4);
        let again = b.cmp(step, IntPred::Ult, next, four);
        b.cond_br(step, again, (outer, vec![t[0], next]), (exit, vec![t[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, s, s);
        (h, key)
    }

    #[test]
    fn find_reports_nested_loops_within_one_inner_iteration_only() {
        let (h, key) = nested(|b, inner, o, i| {
            let four = b.constant(inner, Ty::I32, 4);
            let row = b.int(inner, IntOp::Mul, o, four);
            b.int(inner, IntOp::Add, row, i)
        });
        assert!(h.together.contains(&key), "all lanes store to one word in each inner iteration");
        assert!(!h.apart.contains(&key), "different inner iterations store to different words");
    }

    #[test]
    fn find_reports_nested_loops_whose_outer_iterations_reuse_the_inner_words() {
        let (h, key) = nested(|_, _, _, i| i);
        assert!(h.together.contains(&key), "all lanes store to one word in each inner iteration");
        assert!(h.apart.contains(&key), "the same inner iteration of two outer iterations stores to one word");
    }

    #[test]
    fn find_reports_a_loop_whose_iterations_pair_up_on_a_halved_index() {
        let (h, key) = nested(|b, inner, _, i| {
            let one = b.constant(inner, Ty::I32, 1);
            b.int(inner, IntOp::LShr, i, one)
        });
        assert!(h.apart.contains(&key), "inner iterations 0 and 1 both store to word 0");
    }

    #[test]
    fn find_reports_stores_whose_indices_scale_the_iteration_differently() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero]);
        let first = byte_offset(&mut b, body, buf, p[1], 4);
        let s1 = store_at(&mut b, body, first, p[0]);
        let two = b.constant(body, Ty::I32, 2);
        let doubled = b.int(body, IntOp::Mul, p[1], two);
        let second = byte_offset(&mut b, body, buf, doubled, 4);
        let s2 = store_at(&mut b, body, second, p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[1], one);
        let limit = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, s1, s2);
        assert!(h.apart.contains(&key), "iteration 2 of the first store and iteration 1 of the second both store to word 2");
    }

    #[test]
    fn find_reports_no_other_iteration_of_a_loop_that_runs_once() {
        let (h, key) = counted_loop(1, false);
        assert!(h.together.contains(&key), "all lanes store to word 0");
        assert!(!h.apart.contains(&key), "the loop runs once, so no two iterations exist");
    }

    fn guarded(distance: u64) -> (Hazards, (usize, usize)) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let v = b.load(e, Space::Global, MemSize::B32, other, yes);
        let three = b.constant(e, Ty::I32, 3);
        let chosen = b.cmp(e, IntPred::Eq, v, three);
        let mask = b.int(e, IntOp::And, chosen, k.exec);
        let s1 = store_at(&mut b, e, buf, mask);
        let scaled = byte_offset(&mut b, e, buf, v, 4);
        let back = b.constant(e, Ty::I64, distance.wrapping_neg());
        let address = b.int(e, IntOp::Add, scaled, back);
        let s2 = store_at(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        let key = pair(&h, s1, s2);
        (h, key)
    }

    #[test]
    fn find_reports_stores_that_meet_when_the_guarded_one_runs() {
        let (h, key) = guarded(12);
        assert!(h.together.contains(&key), "with v = 3 both store to buf");
    }

    #[test]
    fn find_excludes_stores_that_meet_only_when_the_guarded_one_is_off() {
        let (h, key) = guarded(8);
        assert!(!h.conflicts().contains(&key), "the stores meet only at v = 2, where the first does not run");
    }

    fn narrowed(bound: u64) -> (Hazards, (usize, usize), usize) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let limit = b.constant(e, Ty::I32, bound);
        let inside = b.cmp(e, IntPred::Ult, lane, limit);
        let exec = b.int(e, IntOp::And, inside, k.exec);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        b.br(e, then, vec![exec, buf]);
        let yes = b.constant(then, Ty::I1, 1);
        let read = b.here(then);
        b.load(then, Space::Global, MemSize::B32, t[1], yes);
        let s = store_at(&mut b, then, t[1], t[0]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, read, s);
        let side = if key.0 == pair(&h, read, read).0 { 0 } else { 1 };
        (h, key, side)
    }

    #[test]
    fn find_marks_a_conflict_idle_when_only_inactive_lanes_read() {
        let (h, key, side) = narrowed(1);
        assert!(h.together.contains(&key), "lane 0 stores the word every other lane reads");
        assert!(h.idle.contains(&(key.0, key.1, false, side)), "every lane that reads lane 0's word has exec clear");
    }

    #[test]
    fn find_keeps_a_conflict_busy_when_an_active_lane_reads() {
        let (h, key, side) = narrowed(16);
        assert!(h.together.contains(&key));
        assert!(!h.idle.contains(&(key.0, key.1, false, side)), "lane 1 has exec set and reads the word lane 0 stores");
    }

    fn spilled(halves: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let slot = b.constant(e, Ty::I32, 16);
        let reloaded = if halves {
            let lo = b.core(e, Ty::I32, Op::UnpackLo(buf));
            let hi = b.core(e, Ty::I32, Op::UnpackHi(buf));
            let next = b.constant(e, Ty::I32, 20);
            b.store(e, Space::Scratch, MemSize::B32, slot, lo, k.exec);
            b.store(e, Space::Scratch, MemSize::B32, next, hi, k.exec);
            let lo = b.load(e, Space::Scratch, MemSize::B32, slot, k.exec);
            let hi = b.load(e, Space::Scratch, MemSize::B32, next, k.exec);
            b.core(e, Ty::I64, Op::Pack64(lo, hi))
        } else {
            b.store(e, Space::Scratch, MemSize::B64, slot, buf, k.exec);
            b.load(e, Space::Scratch, MemSize::B64, slot, k.exec)
        };
        let s1 = store_at(&mut b, e, reloaded, k.exec);
        let s2 = store_at(&mut b, e, buf, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_follows_a_pointer_through_a_private_spill() {
        assert!(spilled(false), "the pointer read back from the 8-byte slot is buf");
        assert!(spilled(true), "the pointer read back from two 4-byte slots is buf");
    }

    fn escaped(space: Space) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let place = match space {
            Space::Lds => b.constant(e, Ty::I32, 64),
            _ => table,
        };
        b.store(e, space, MemSize::B64, place, buf, k.exec);
        let reloaded = b.load(e, space, MemSize::B64, place, k.exec);
        let s1 = store_at(&mut b, e, reloaded, k.exec);
        let s2 = store_at(&mut b, e, buf, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_follows_a_pointer_written_to_global_memory_and_read_back() {
        assert!(escaped(Space::Global));
    }

    #[test]
    fn find_follows_a_pointer_written_to_lds_and_read_back() {
        assert!(escaped(Space::Lds));
    }

    #[test]
    fn find_follows_a_pointer_through_operations_that_lose_its_form() {
        let names = ["xor twice", "mul by one", "shl by zero", "or with zero", "and with all ones", "select between itself"];
        let lost: Vec<&str> = names
            .iter()
            .enumerate()
            .filter(|&(case, _)| {
                let (mut b, k) = Build::kernel();
                let e = BlockId(0);
                let buf = k.buffer(&mut b, e, 0);
                let other = k.buffer(&mut b, e, 8);
                let yes = b.constant(e, Ty::I1, 1);
                let v = b.load(e, Space::Global, MemSize::B32, other, yes);
                let zero = b.constant(e, Ty::I32, 0);
                let unknown = b.cmp(e, IntPred::Eq, v, zero);
                let moved = match case {
                    0 => {
                        let k = b.constant(e, Ty::I64, 0x5a5a);
                        let x = b.int(e, IntOp::Xor, buf, k);
                        b.int(e, IntOp::Xor, x, k)
                    }
                    1 => {
                        let one = b.constant(e, Ty::I64, 1);
                        b.int(e, IntOp::Mul, buf, one)
                    }
                    2 => {
                        let z = b.constant(e, Ty::I64, 0);
                        b.int(e, IntOp::Shl, buf, z)
                    }
                    3 => {
                        let z = b.constant(e, Ty::I64, 0);
                        b.int(e, IntOp::Or, buf, z)
                    }
                    4 => {
                        let all = b.constant(e, Ty::I64, u64::MAX);
                        b.int(e, IntOp::And, buf, all)
                    }
                    _ => b.core(e, Ty::I64, Op::Select(unknown, buf, buf)),
                };
                let s1 = store_at(&mut b, e, moved, k.exec);
                let s2 = store_at(&mut b, e, buf, k.exec);
                let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
                !h.together.contains(&pair(&h, s1, s2))
            })
            .map(|(_, &name)| name)
            .collect();
        assert!(lost.is_empty(), "{:?}: the address is still buf, where every lane stores next", lost);
    }

    #[test]
    fn find_reports_a_conflict_only_the_second_wave_has() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let wave = b.constant(e, Ty::I32, 32);
        let later = b.cmp(e, IntPred::Uge, k.item, wave);
        let own = byte_offset(&mut b, e, buf, k.item, 4);
        let address = b.core(e, Ty::I64, Op::Select(later, buf, own));
        let s1 = store_at(&mut b, e, address, k.exec);
        let s2 = store_at(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &environment(64, &[(0, 1, 0x1000)]));
        assert!(h.together.contains(&pair(&h, s1, s2)), "every lane of the second wave stores to buf");
    }

    #[test]
    fn find_reports_lanes_that_share_an_lds_word() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let half = b.int(e, IntOp::LShr, lane, one);
        let four = b.constant(e, Ty::I32, 4);
        let address = b.int(e, IntOp::Mul, half, four);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Lds, MemSize::B32, address, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Lds, MemSize::B32, address, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[]));
        assert!(h.together.contains(&pair(&h, s1, s2)), "lanes 0 and 1 store to LDS word 0");
    }

    fn walked(stride: u64) -> (Hazards, (usize, usize)) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let start = byte_offset(&mut b, e, buf, lane, 4);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, start, zero]);
        let s = store_at(&mut b, body, p[1], p[0]);
        let half = b.constant(body, Ty::I64, stride / 2);
        let midway = b.int(body, IntOp::Add, p[1], half);
        let moved = b.int(body, IntOp::Add, midway, half);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], moved, next]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        let key = pair(&h, s, s);
        (h, key)
    }

    #[test]
    fn find_reports_a_pointer_advanced_in_two_steps_that_meets_another_lane_later() {
        let (h, key) = walked(8);
        assert!(h.apart.contains(&key), "lane a + 2 stores in iteration i where lane a stores in iteration i + 1");
        assert!(!h.together.contains(&key));
    }

    #[test]
    fn find_keeps_apart_a_pointer_advanced_in_two_steps_past_the_wave() {
        let (h, key) = walked(128);
        assert!(!h.conflicts().contains(&key), "every lane walks its own column of 128-byte rows");
    }

    #[test]
    fn find_reports_a_pointer_that_swaps_between_two_buffers() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I64, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, first, first, second, zero]);
        let s1 = store_at(&mut b, body, p[1], p[0]);
        let s2 = store_at(&mut b, body, p[2], p[0]);
        let at_first = b.cmp(body, IntPred::Eq, p[1], p[2]);
        let swapped = b.core(body, Ty::I64, Op::Select(at_first, p[3], p[2]));
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[4], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], swapped, p[2], p[3], next]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        assert!(h.together.contains(&pair(&h, s1, s2)), "in iteration 0 every lane stores to the first buffer twice");
    }

    fn halving(target: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let sixteen = b.constant(e, Ty::I32, 16);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64]);
        let (exit, x) = b.block(&[Ty::I1, Ty::I64]);
        b.br(e, body, vec![k.exec, sixteen, buf]);
        let address = byte_offset(&mut b, body, p[2], p[1], 4);
        let s = store_at(&mut b, body, address, p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let half = b.int(body, IntOp::LShr, p[1], one);
        let zero = b.constant(body, Ty::I32, 0);
        let again = b.cmp(body, IntPred::Ne, half, zero);
        b.cond_br(body, again, (body, vec![p[0], half, p[2]]), (exit, vec![p[0], p[2]]));
        let at = b.constant(exit, Ty::I64, target * 4);
        let there = b.int(exit, IntOp::Add, x[1], at);
        let t = store_at(&mut b, exit, there, x[0]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.conflicts().contains(&pair(&h, s, t))
    }

    #[test]
    fn find_reports_a_halving_index_that_reaches_a_word_stored_after_the_loop() {
        assert!(halving(8), "the index runs 16, 8, 4, 2, 1");
    }

    #[test]
    fn find_bounds_a_halving_index_by_where_it_starts() {
        assert!(!halving(100), "the index never exceeds 16");
    }

    fn flagged_loop(start: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let flag = b.constant(e, Ty::I1, start);
        let (body, p) = b.block(&[Ty::I1, Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, flag, zero, buf, other]);
        let yes = b.constant(body, Ty::I1, 1);
        let v = b.load(body, Space::Global, MemSize::B32, p[4], yes);
        let z = b.constant(body, Ty::I32, 0);
        let fresh = b.cmp(body, IntPred::Ne, v, z);
        let mask = b.int(body, IntOp::And, p[1], p[0]);
        let s1 = store_at(&mut b, body, p[3], mask);
        let s2 = store_at(&mut b, body, p[3], p[0]);
        let still = b.int(body, IntOp::And, p[1], fresh);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], still, next, p[3], p[4]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_store_masked_by_a_flag_that_starts_set() {
        assert!(flagged_loop(1), "in iteration 0 the flag is set and every lane stores twice");
    }

    #[test]
    fn find_drops_a_store_masked_by_a_flag_that_starts_clear_and_can_only_stay_clear() {
        assert!(!flagged_loop(0), "the flag starts clear and each iteration ands it with something");
    }

    fn joined_spill(same: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let v = uniform_word(&mut b, &k, e, 16, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1, Ty::I64]);
        b.cond_br(e, c, (then, vec![k.exec, first, second]), (other, vec![k.exec, first, second]));
        let slot = b.constant(then, Ty::I32, 16);
        b.store(then, Space::Scratch, MemSize::B64, slot, t[1], t[0]);
        b.br(then, join, vec![t[0], t[1]]);
        let slot = b.constant(other, Ty::I32, 16);
        let spilled = if same { o[1] } else { o[2] };
        b.store(other, Space::Scratch, MemSize::B64, slot, spilled, o[0]);
        b.br(other, join, vec![o[0], o[1]]);
        let slot = b.constant(join, Ty::I32, 16);
        let reloaded = b.load(join, Space::Scratch, MemSize::B64, slot, j[0]);
        let s1 = store_at(&mut b, join, reloaded, j[0]);
        let s2 = store_at(&mut b, join, j[1], j[0]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_follows_a_pointer_spilled_on_both_paths_into_a_join() {
        assert!(joined_spill(true), "both paths spill the first buffer, which every lane stores to next");
        assert!(joined_spill(false), "a wave that takes the first arm spills the first buffer, which every lane stores to next");
    }

    fn by_workgroup(other: u64, grid: u32) -> bool {
        let (mut b, k, extra) = Build::kernel_with(&[(ParameterSource::Sgpr(WORKGROUP_ID_X), Ty::I32)]);
        b.entry.workgroup_id_x = true;
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let address = byte_offset(&mut b, e, buf, extra[0], 4);
        let s1 = store_at(&mut b, e, address, k.exec);
        let at = b.constant(e, Ty::I64, other * 4);
        let there = b.int(e, IntOp::Add, buf, at);
        let s2 = store_at(&mut b, e, there, k.exec);
        let mut env = environment(32, &[(0, 1, 0x1000)]);
        env.grid = [grid, 1, 1];
        let h = Hazards::find(&b.program(), &env);
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_word_the_workgroup_id_can_name() {
        assert!(by_workgroup(3, 4), "workgroup 3 of 4 stores to word 3 from every lane");
    }

    #[test]
    fn find_bounds_the_workgroup_id_by_the_grid() {
        assert!(!by_workgroup(4, 4), "there is no workgroup 4 in a grid of 4");
    }

    fn by_packet(offset: u64, bytes: MemSize, block: [u32; 3], shift: u64) -> bool {
        let (mut b, k, extra) = Build::kernel_with(&[(ParameterSource::Sgpr(0), Ty::I32), (ParameterSource::Sgpr(1), Ty::I32)]);
        b.entry.dispatch_ptr = Some(0);
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let packet = b.core(e, Ty::I64, Op::Pack64(extra[0], extra[1]));
        let at = b.constant(e, Ty::I64, offset);
        let field = b.int(e, IntOp::Add, packet, at);
        let yes = b.constant(e, Ty::I1, 1);
        let size = b.load(e, Space::Global, bytes, field, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let past = b.int(e, IntOp::Add, lane, size);
        let high = byte_offset(&mut b, e, buf, past, 4);
        let s1 = store_at(&mut b, e, high, k.exec);
        let k2 = b.constant(e, Ty::I32, shift);
        let low = b.int(e, IntOp::Add, lane, k2);
        let low = byte_offset(&mut b, e, buf, low, 4);
        let s2 = store_at(&mut b, e, low, k.exec);
        let mut env = environment(block.iter().product(), &[(0, 1, 0x1000)]);
        env.block = block;
        env.grid = [2, 1, 1];
        let h = Hazards::find(&b.program(), &env);
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reads_the_workgroup_size_from_the_dispatch_packet() {
        assert!(!by_packet(4, MemSize::U16, [32, 1, 1], 0), "lanes store past the 32 words the wave covers");
        assert!(by_packet(4, MemSize::U16, [32, 1, 1], 1), "lane 1's word 32 is lane 0's word 32");
        assert!(!by_packet(4, MemSize::U16, [16, 1, 1], 0), "lanes store past the 16 words the workgroup covers");
        assert!(by_packet(4, MemSize::U16, [16, 1, 1], 1), "lane 1's word 16 is lane 0's word 16");
    }

    #[test]
    fn find_reads_the_grid_size_from_the_dispatch_packet() {
        assert!(!by_packet(12, MemSize::B32, [32, 1, 1], 0), "the grid has 64 items, past the 32 words");
        assert!(by_packet(12, MemSize::B32, [32, 1, 1], 33), "lane 0 stores word 64, lane 31 stores 31 + 33");
    }

    fn moved(mover: usize, other: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let index = match mover {
            0 => b.wave(e, WaveOp::ReadFirstLane, vec![lane, k.exec]),
            1 => {
                let five = b.constant(e, Ty::I32, 5);
                let r = b.constant(e, Ty::I32, 0);
                b.wave(e, WaveOp::ReadLane, vec![lane, five, r])
            }
            2 => {
                let nine = b.constant(e, Ty::I32, 9);
                let three = b.constant(e, Ty::I32, 3);
                let r = b.constant(e, Ty::I32, 0);
                b.wave(e, WaveOp::WriteLane, vec![nine, three, lane, r])
            }
            _ => {
                let four = b.constant(e, Ty::I32, 4);
                let low = b.cmp(e, IntPred::Ult, lane, four);
                b.wave(e, WaveOp::Ballot, vec![low])
            }
        };
        let address = byte_offset(&mut b, e, buf, index, 4);
        let s1 = store_at(&mut b, e, address, k.exec);
        let at = b.constant(e, Ty::I64, other * 4);
        let there = b.int(e, IntOp::Add, buf, at);
        let s2 = store_at(&mut b, e, there, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_follows_indices_that_lanes_read_from_each_other() {
        let cases = [
            ("read first lane", 0, 0, true),
            ("read first lane", 0, 1, false),
            ("read lane 5", 1, 5, true),
            ("read lane 5", 1, 6, false),
            ("write 9 into lane 3", 2, 9, true),
            ("write 9 into lane 3", 2, 3, false),
            ("ballot of lanes below 4", 3, 15, true),
            ("ballot of lanes below 4", 3, 14, false),
        ];
        let wrong: Vec<(&str, u64, bool)> = cases
            .iter()
            .filter(|&&(_, mover, other, expected)| moved(mover, other) != expected)
            .map(|&(name, _, other, expected)| (name, other, expected))
            .collect();
        assert!(wrong.is_empty(), "(index, word the other store names, whether they meet): {:?}", wrong);
    }

    fn atomics(first: MemoryOp, second: MemoryOp, use_first: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let out = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let data = |op: MemoryOp| match op {
            MemoryOp::AtomicCmpSwap => vec![lane, lane],
            _ => vec![lane],
        };
        let a1 = b.here(e);
        let inputs = [vec![buf], data(first), vec![k.exec]].concat();
        let old = b.effect(e, memory(Space::Global, first), inputs)[0];
        let a2 = b.here(e);
        let inputs = [vec![buf], data(second), vec![k.exec]].concat();
        b.effect(e, memory(Space::Global, second), inputs);
        if use_first {
            let own = byte_offset(&mut b, e, out, lane, 4);
            b.store(e, Space::Global, MemSize::B32, own, old, k.exec);
        }
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.together.contains(&pair(&h, a1, a2))
    }

    #[test]
    fn find_orders_atomics_only_when_their_order_can_show() {
        use MemoryOp::*;
        let cases = [
            ("two unsigned adds nobody reads", AtomicAdd(Numeric::Unsigned), AtomicAdd(Numeric::Unsigned), false, false),
            ("an unsigned add whose old value is kept", AtomicAdd(Numeric::Unsigned), AtomicAdd(Numeric::Unsigned), true, true),
            ("two float adds nobody reads", AtomicAdd(Numeric::Float), AtomicAdd(Numeric::Float), false, true),
            ("two unsigned maxima nobody reads", AtomicRmw(Rmw::UnsignedMax), AtomicRmw(Rmw::UnsignedMax), false, false),
            ("a maximum and a minimum nobody reads", AtomicRmw(Rmw::UnsignedMax), AtomicRmw(Rmw::UnsignedMin), false, true),
            ("an unsigned add and a maximum nobody reads", AtomicAdd(Numeric::Unsigned), AtomicRmw(Rmw::UnsignedMax), false, true),
            ("two compare-and-swaps", AtomicCmpSwap, AtomicCmpSwap, false, true),
        ];
        let wrong: Vec<(&str, bool)> = cases
            .iter()
            .filter(|&&(_, a, b, used, expected)| atomics(a, b, used) != expected)
            .map(|&(name, .., expected)| (name, expected))
            .collect();
        assert!(wrong.is_empty(), "(pair, whether its order matters): {:?}", wrong);
    }

    #[test]
    fn find_leaves_private_stores_to_each_lane() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let slot = b.constant(e, Ty::I32, 16);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let s1 = b.here(e);
        b.store(e, Space::Scratch, MemSize::B32, slot, lane, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Scratch, MemSize::B32, slot, lane, k.exec);
        let buf = k.buffer(&mut b, e, 0);
        store_at(&mut b, e, buf, k.exec);
        store_at(&mut b, e, buf, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "each lane has its own private memory");
    }

    #[test]
    fn find_keeps_lds_and_global_words_apart() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Lds, MemSize::B32, zero, zero, k.exec);
        let wide = b.constant(e, Ty::I64, 0);
        let s2 = b.here(e);
        b.store(e, Space::Global, MemSize::B32, wide, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)));
    }

    #[test]
    fn find_ignores_the_order_of_lanes_within_one_instruction_outside_loops() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let s = store_at(&mut b, e, buf, k.exec);
        let other = k.buffer(&mut b, e, 8);
        store_at(&mut b, e, other, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        assert!(!h.conflicts().contains(&pair(&h, s, s)), "the lanes of one instruction have no order in the wave either");
    }

    #[test]
    fn find_counts_no_fence_and_no_pair_of_reads() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let fence = b.here(e);
        b.effect(e, memory(Space::Global, MemoryOp::Fence), vec![]);
        let l1 = b.here(e);
        b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        let l2 = b.here(e);
        b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        store_at(&mut b, e, buf, k.exec);
        store_at(&mut b, e, buf, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.accesses.iter().any(|a| (a.block, a.index) == fence));
        assert!(!h.conflicts().contains(&pair(&h, l1, l2)));
    }

    fn resource(same: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let s = store_at(&mut b, e, buf, k.exec);
        let r = b.here(e);
        b.resource(e, if same { buf } else { other });
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.together.contains(&pair(&h, s, r))
    }

    #[test]
    fn find_orders_a_resource_read_after_stores_to_its_buffer() {
        assert!(resource(true), "the resource read may read the word every lane stores");
    }

    #[test]
    fn find_keeps_a_resource_read_apart_from_stores_to_another_buffer() {
        assert!(!resource(false));
    }

    fn rdna4(b: &mut Build, name: &str) -> TargetOp {
        b.registry = crate::rdna_spmd::rdna4::dialect().registry;
        b.registry.lookup(0x5244_4e34, name).unwrap()
    }

    fn base_units(b: &mut Build, e: BlockId, buf: ValueId) -> ValueId {
        let eight = b.constant(e, Ty::I64, 8);
        let units = b.int(e, IntOp::LShr, buf, eight);
        b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, units))
    }

    fn node_read_at(node: u64, offset: u64, active: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_bvh64_intersect_ray");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let mut args = [b.constant(e, Ty::I32, 0); 14];
        args[0] = base_units(&mut b, e, buf);
        args[2] = b.constant(e, Ty::I64, node);
        args[13] = if active { k.exec } else { b.constant(e, Ty::I1, 0) };
        let r = b.here(e);
        b.target(e, op, Arguments::Fourteen(args), &[Ty::I32; 4]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s, r))
    }

    fn node_read(offset: u64) -> bool {
        node_read_at(5, offset, true)
    }

    #[test]
    fn find_orders_a_node_read_four_gigabytes_up_after_stores_into_that_node() {
        assert!(node_read_at((1 << 29) + 5, (1 << 32) + 64, true), "node 2^29 + 5 starts 2^32 bytes into the BVH");
    }

    #[test]
    fn find_keeps_a_node_read_four_gigabytes_up_apart_from_stores_into_the_first_node() {
        assert!(!node_read_at((1 << 29) + 5, 64, true), "node 2^29 + 5 reads 2^32 bytes past the store");
    }

    #[test]
    fn find_orders_a_node_read_after_stores_into_the_node() {
        assert!(node_read(64), "the box node at the base of the BVH spans 128 bytes");
    }

    #[test]
    fn find_keeps_a_node_read_apart_from_stores_past_the_node() {
        assert!(!node_read(4096), "the box node at the base of the BVH ends 128 bytes in");
    }

    #[test]
    fn find_orders_a_read_of_a_later_node_after_stores_into_it() {
        assert!(node_read_at(0x2d, 0x140 + 124, true), "node 0x2d of type 5 sits 0x28 << 3 = 0x140 bytes in and spans 128 bytes");
    }

    #[test]
    fn find_keeps_a_read_of_a_later_node_apart_from_stores_before_it() {
        assert!(!node_read_at(0x2d, 0x140 - 4, true), "node 0x2d starts 0x140 bytes in");
    }

    fn lane_nodes(offset: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_bvh64_intersect_ray");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let scaled = b.int(e, IntOp::Mul, lane, sixteen);
        let five = b.constant(e, Ty::I32, 5);
        let node = b.int(e, IntOp::Add, scaled, five);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, node));
        let mut args = [b.constant(e, Ty::I32, 0); 14];
        args[0] = base_units(&mut b, e, buf);
        args[2] = wide;
        args[13] = k.exec;
        let r = b.here(e);
        b.target(e, op, Arguments::Fourteen(args), &[Ty::I32; 4]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s, r))
    }

    #[test]
    fn find_orders_reads_of_lanes_own_nodes_after_stores_into_one_of_them() {
        assert!(lane_nodes(128 * 3 + 4), "lane 3 reads node 3 * 16 + 5, 128 * 3 bytes in");
    }

    #[test]
    fn find_keeps_reads_of_lanes_own_nodes_apart_from_stores_past_the_last() {
        assert!(!lane_nodes(128 * 32 + 4), "the 32 lanes read the first 128 * 32 bytes");
    }

    #[test]
    fn find_orders_a_triangle_node_read_after_stores_into_the_node() {
        assert!(node_read_at(0, 60, true), "a triangle pair node spans 64 bytes");
    }

    #[test]
    fn find_keeps_a_triangle_node_read_apart_from_stores_past_the_node() {
        assert!(!node_read_at(0, 64, true), "a triangle pair node ends 64 bytes in");
    }

    #[test]
    fn find_keeps_a_texel_read_of_the_first_row_apart_from_stores_into_a_later_row() {
        assert!(!texel_read(5 * 128 + 4), "every lane reads row 0 when v is 0");
    }

    #[test]
    fn find_keeps_a_node_read_no_lane_makes_apart_from_stores_into_the_node() {
        assert!(!node_read_at(5, 64, false), "the read's own exec is false in every lane");
    }

    fn node8_read(offset: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_bvh8_intersect_ray");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let mut args = [b.constant(e, Ty::I32, 0); 13];
        args[0] = base_units(&mut b, e, buf);
        args[2] = b.constant(e, Ty::I64, 0x20);
        args[11] = b.constant(e, Ty::I32, 0x13);
        args[12] = k.exec;
        let r = b.here(e);
        b.target(e, op, Arguments::Thirteen(args), &[Ty::I32; 10]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s, r))
    }

    #[test]
    fn find_orders_an_eight_wide_node_read_after_stores_into_the_node() {
        assert!(node8_read(0x180 + 124), "base 0x20 plus index 0x13 & !15 = 0x30 gives 0x30 << 3 = 0x180 bytes in, 128 bytes long");
    }

    #[test]
    fn find_keeps_an_eight_wide_node_read_apart_from_stores_before_the_node() {
        assert!(!node8_read(0x180 - 4), "the node starts 0x180 bytes in");
    }

    fn texel_read(offset: u64) -> bool {
        texel_read_in(offset, false)
    }

    fn texel_read_at_row(offset: u64, v: f32) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_sample_lz");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let mut args = [zero; 16];
        args[0] = base_units(&mut b, e, buf);
        args[1] = b.constant(e, Ty::I32, 3 << 30 | 5 << 17);
        args[2] = b.constant(e, Ty::I32, 15 << 14 | 3);
        args[3] = b.constant(e, Ty::I32, 4);
        args[13] = b.constant(e, Ty::I1, 1);
        args[14] = b.core(e, Ty::F32, Op::Convert(Cvt::UnsignedToFloatRte, Ty::F32, k.item));
        args[15] = b.constant(e, Ty::F32, v.to_bits() as u64);
        let r = b.here(e);
        b.target(e, op, Arguments::Sixteen(args), &[Ty::I32]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s, r))
    }

    #[test]
    fn find_orders_a_texel_read_clamped_to_the_last_row_after_stores_into_it() {
        assert!(texel_read_at_row(15 * 128 + 3, 20.0), "v = 20 clamps to row 15, whose texel 3 the store writes");
    }

    #[test]
    fn find_keeps_a_texel_read_clamped_to_the_last_row_apart_from_stores_into_row_fourteen() {
        assert!(!texel_read_at_row(14 * 128 + 3, 20.0), "v = 20 clamps to row 15, and the store writes row 14");
    }

    #[test]
    fn find_orders_a_texel_read_of_row_three_after_stores_into_it() {
        assert!(texel_read_at_row(3 * 128 + 5, 3.5), "v = 3.5 reads row 3, whose texel 5 the store writes");
    }

    fn texel_read_in(offset: u64, rows: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_sample_lz");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let mut args = [zero; 16];
        args[0] = base_units(&mut b, e, buf);
        args[1] = b.constant(e, Ty::I32, 3 << 30 | 5 << 17);
        args[2] = b.constant(e, Ty::I32, 15 << 14 | 3);
        args[3] = b.constant(e, Ty::I32, 4);
        args[13] = b.constant(e, Ty::I1, 1);
        args[14] = b.core(e, Ty::F32, Op::Convert(Cvt::UnsignedToFloatRte, Ty::F32, k.item));
        args[15] = if rows { args[14] } else { b.constant(e, Ty::F32, 0) };
        let r = b.here(e);
        b.target(e, op, Arguments::Sixteen(args), &[Ty::I32]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        h.together.contains(&pair(&h, s, r))
    }

    #[test]
    fn find_orders_a_texel_read_after_stores_into_the_image() {
        assert!(texel_read(8), "lanes 8 to 11 read the texels of the first row the store writes");
    }

    #[test]
    fn find_keeps_a_texel_read_apart_from_stores_past_the_image() {
        assert!(!texel_read(4096), "a 16 x 16 image of bytes with 128-byte rows ends 1936 bytes in");
    }

    #[test]
    fn find_orders_a_texel_read_after_stores_into_the_last_row() {
        assert!(texel_read_in(15 * 128 + 12, true), "lanes 15 to 31 read texel (15, 15), byte 15 * 128 + 15");
    }

    #[test]
    fn find_keeps_a_texel_read_apart_from_stores_just_past_the_last_texel() {
        assert!(!texel_read_in(15 * 128 + 16, true), "the last texel is byte 15 * 128 + 15");
    }

    fn twice(b: &mut Build, block: BlockId, address: ValueId, mask: ValueId) -> ((BlockId, usize), (BlockId, usize)) {
        (store_at(b, block, address, mask), store_at(b, block, address, mask))
    }

    fn uniform_word(b: &mut Build, k: &Kernel, block: BlockId, offset: u64, size: MemSize) -> ValueId {
        let table = k.buffer(b, block, 8);
        let at = b.constant(block, Ty::I64, offset);
        let address = b.int(block, IntOp::Add, table, at);
        let yes = b.constant(block, Ty::I1, 1);
        b.load(block, Space::Global, size, address, yes)
    }

    fn env2() -> Environment {
        environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)])
    }

    fn scaled_lanes(scale: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let factor = scale(&mut b, e, u);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let index = b.int(e, IntOp::Mul, lane, factor);
        let address = byte_offset(&mut b, e, buf, index, 4);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    fn lane_indexed(index: impl Fn(&mut Build, BlockId, &Kernel, ValueId) -> ValueId) -> bool {
        lane_indexed_by(4, index)
    }

    fn lane_indexed_by(scale: u64, index: impl Fn(&mut Build, BlockId, &Kernel, ValueId) -> ValueId) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let index = index(&mut b, e, &k, lane);
        let address = byte_offset(&mut b, e, buf, index, scale);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_keeps_apart_lanes_that_shift_a_word_they_offset_by_four_lanes() {
        let collide = lane_indexed(|b, e, k, lane| {
            let u = uniform_word(b, k, e, 0, MemSize::B32);
            let four = b.constant(e, Ty::I32, 4);
            let step = b.int(e, IntOp::Mul, lane, four);
            let sum = b.int(e, IntOp::Add, u, step);
            let two = b.constant(e, Ty::I32, 2);
            b.int(e, IntOp::LShr, sum, two)
        });
        assert!(!collide, "(u + 4 lane) >> 2 differs by the lane distance modulo 2^30, so the lanes name distinct words");
    }

    #[test]
    fn find_reports_lanes_that_shift_a_word_they_offset_by_one_lane() {
        let collide = lane_indexed(|b, e, k, lane| {
            let u = uniform_word(b, k, e, 0, MemSize::B32);
            let sum = b.int(e, IntOp::Add, u, lane);
            let two = b.constant(e, Ty::I32, 2);
            b.int(e, IntOp::LShr, sum, two)
        });
        assert!(collide, "u = 0 puts lanes 0 to 3 on word 0");
    }

    fn shifted_lanes(lane_zero: u64, lane_one: u64, scale: u64) -> bool {
        lane_indexed_by(scale, |b, e, k, lane| {
            let u = uniform_word(b, k, e, 0, MemSize::B32);
            let one = b.constant(e, Ty::I32, 1);
            let second = b.cmp(e, IntPred::Eq, lane, one);
            let special = b.constant(e, Ty::I32, lane_one);
            let far = b.constant(e, Ty::I32, 1024);
            let spread = b.int(e, IntOp::Mul, lane, far);
            let zero = b.constant(e, Ty::I32, 0);
            let first = b.cmp(e, IntPred::Eq, lane, zero);
            let zeroth = b.constant(e, Ty::I32, lane_zero);
            let low = b.core(e, Ty::I32, Op::Select(first, zeroth, spread));
            let offset = b.core(e, Ty::I32, Op::Select(second, special, low));
            let sum = b.int(e, IntOp::Add, u, offset);
            let two = b.constant(e, Ty::I32, 2);
            b.int(e, IntOp::LShr, sum, two)
        })
    }

    #[test]
    fn find_reports_lanes_whose_shifted_indices_meet_through_a_carry() {
        assert!(shifted_lanes(3, 4, 4), "(u + 3) >> 2 and (u + 4) >> 2 name one word whenever u mod 4 is not 0");
    }

    #[test]
    fn find_keeps_apart_lanes_whose_shifted_offsets_are_a_word_apart() {
        assert!(!shifted_lanes(3, 0xffff_ffff, 4), "(u + 3) >> 2 and (u - 1) >> 2 always differ by exactly 1");
    }

    #[test]
    fn find_reports_lanes_whose_shifted_indices_meet_through_a_wrap() {
        assert!(shifted_lanes(1, 0xffff_ffff, 1), "(u + 1) >> 2 and (u - 1) >> 2 are one index when u mod 4 is 1 or 2");
    }

    #[test]
    fn find_keeps_apart_the_lanes_after_a_select_of_two_offsets() {
        let collide = lane_indexed(|b, e, k, lane| {
            let v = uniform_word(b, k, e, 0, MemSize::B32);
            let zero = b.constant(e, Ty::I32, 0);
            let c = b.cmp(e, IntPred::Ne, v, zero);
            let two = b.constant(e, Ty::I32, 2);
            let even = b.int(e, IntOp::Mul, lane, two);
            let one = b.constant(e, Ty::I32, 1);
            let odd = b.int(e, IntOp::Add, even, one);
            b.core(e, Ty::I32, Op::Select(c, even, odd))
        });
        assert!(!collide, "every lane takes 2 lane or every lane 2 lane + 1, and no two lanes meet either way");
    }

    #[test]
    fn find_reports_the_lanes_after_a_select_where_one_arm_halves_the_index() {
        let collide = lane_indexed(|b, e, k, lane| {
            let v = uniform_word(b, k, e, 0, MemSize::B32);
            let zero = b.constant(e, Ty::I32, 0);
            let c = b.cmp(e, IntPred::Ne, v, zero);
            let one = b.constant(e, Ty::I32, 1);
            let half = b.int(e, IntOp::LShr, lane, one);
            b.core(e, Ty::I32, Op::Select(c, lane, half))
        });
        assert!(collide, "on the second arm lanes 0 and 1 both name word 0");
    }

    #[test]
    fn find_keeps_apart_lanes_that_permute_their_neighbours_lane_ids() {
        let collide = lane_indexed(|b, e, k, lane| {
            let one = b.constant(e, Ty::I32, 1);
            let neighbour = b.int(e, IntOp::Xor, lane, one);
            let two = b.constant(e, Ty::I32, 2);
            let byte = b.int(e, IntOp::Shl, neighbour, two);
            b.wave(e, WaveOp::BpermuteFi, vec![byte, lane, k.exec])
        });
        assert!(!collide, "lane l reads lane l ^ 1's lane id, so the lanes name distinct words");
    }

    #[test]
    fn find_reports_lanes_that_permute_one_lanes_id() {
        let collide = lane_indexed(|b, e, k, lane| {
            let zero = b.constant(e, Ty::I32, 0);
            b.wave(e, WaveOp::BpermuteFi, vec![zero, lane, k.exec])
        });
        assert!(collide, "every lane reads lane 0's id and names word 0");
    }

    #[test]
    fn find_keeps_apart_lanes_scaled_by_an_odd_word() {
        let collide = scaled_lanes(|b, e, u| {
            let one = b.constant(e, Ty::I32, 1);
            b.int(e, IntOp::Or, u, one)
        });
        assert!(!collide, "an odd multiplier keeps the lanes 0 to 31 on distinct words");
    }

    #[test]
    fn find_reports_lanes_scaled_by_a_word_with_its_low_bit_cleared() {
        let collide = scaled_lanes(|b, e, u| {
            let even = b.constant(e, Ty::I32, !1u32 as u64);
            b.int(e, IntOp::And, u, even)
        });
        assert!(collide, "u = 2^31 makes every lane's word 0");
    }

    #[test]
    fn find_keeps_apart_lanes_scaled_by_a_word_with_its_second_bit_set() {
        let collide = scaled_lanes(|b, e, u| {
            let two = b.constant(e, Ty::I32, 2);
            b.int(e, IntOp::Or, u, two)
        });
        assert!(!collide, "u | 2 is odd or twice an odd number, so 4 lane (u | 2) differs between the lanes 0 to 31");
    }

    #[test]
    fn find_keeps_apart_lanes_whose_addresses_differ_above_bit_31() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        let thirty_two = b.constant(e, Ty::I64, 32);
        let high = b.int(e, IntOp::Shl, wide, thirty_two);
        let address = b.int(e, IntOp::Add, buf, high);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "the lanes' addresses differ by multiples of 2^32");
    }

    #[test]
    fn find_reports_lanes_whose_words_meet_across_a_four_gigabyte_boundary() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let edge = b.constant(e, Ty::I64, 0xffff_fffe);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        let two = b.constant(e, Ty::I64, 2);
        let step = b.int(e, IntOp::Mul, wide, two);
        let address = b.int(e, IntOp::Add, edge, step);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(h.conflicts().contains(&pair(&h, s1, s2)), "lane 0 writes bytes 2^32 - 2 to 2^32 + 1 and lane 1 writes 2^32 to 2^32 + 3");
    }

    fn read_back_index(extra: impl Fn(&mut Build, &Kernel, BlockId, ValueId, ValueId), shared: bool, mask: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let slot = if shared {
            let one = b.constant(e, Ty::I32, 1);
            b.int(e, IntOp::And, lane, one)
        } else {
            lane
        };
        let own = byte_offset(&mut b, e, table, slot, 4);
        let stored = if mask {
            let sixteen = b.constant(e, Ty::I32, 16);
            let low = b.cmp(e, IntPred::Ult, lane, sixteen);
            b.int(e, IntOp::And, low, k.exec)
        } else {
            k.exec
        };
        b.store(e, Space::Global, MemSize::B32, own, lane, stored);
        extra(&mut b, &k, e, table, lane);
        let back = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let address = byte_offset(&mut b, e, buf, back, 4);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_an_index_another_store_may_overwrite() {
        let collide = read_back_index(
            |b, k, e, table, lane| {
                let one = b.constant(e, Ty::I32, 1);
                let next = b.int(e, IntOp::Xor, lane, one);
                let neighbour = byte_offset(b, e, table, next, 4);
                let zero = b.constant(e, Ty::I32, 0);
                b.store(e, Space::Global, MemSize::B32, neighbour, zero, k.exec);
            },
            false,
            false,
        );
        assert!(collide, "the second store writes 0 into every word, so every lane may read back 0");
    }

    #[test]
    fn find_reports_an_index_lanes_share_a_word_for() {
        assert!(read_back_index(|_, _, _, _, _| {}, true, false), "lanes 0 and 2 both write word 0, so lane 2 may read back 0");
    }

    #[test]
    fn find_reports_an_index_only_some_lanes_stored() {
        assert!(read_back_index(|_, _, _, _, _| {}, false, true), "lanes 16 to 31 read words nothing in this wave wrote");
    }

    fn squared_back(shared: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let slot = if shared {
            let one = b.constant(e, Ty::I32, 1);
            b.int(e, IntOp::And, lane, one)
        } else {
            lane
        };
        let own = byte_offset(&mut b, e, table, slot, 4);
        let square = b.int(e, IntOp::Mul, lane, lane);
        b.store(e, Space::Global, MemSize::B32, own, square, k.exec);
        let back = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let address = byte_offset(&mut b, e, buf, back, 4);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_square_lanes_share_a_word_for() {
        assert!(squared_back(true), "lanes 0 and 2 both write word 0, so lane 2 may read back lane 0's square 0");
    }

    #[test]
    fn find_follows_a_square_the_lane_stored_and_loaded_back() {
        assert!(!squared_back(false), "word w of the table only ever holds w * w, so each lane reads back its own square");
    }

    fn loaded_back_in_the_next_block(overwritten: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, lane, k.exec);
        let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        b.br(e, next, vec![k.exec, buf, own]);
        if overwritten {
            let zero = b.constant(next, Ty::I32, 0);
            b.store(next, Space::Global, MemSize::B32, p[2], zero, p[0]);
        }
        let back = b.load(next, Space::Global, MemSize::B32, p[2], p[0]);
        let address = byte_offset(&mut b, next, p[1], back, 4);
        let (s1, s2) = twice(&mut b, next, address, p[0]);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_an_index_the_next_block_overwrites_before_loading_it_back() {
        assert!(loaded_back_in_the_next_block(true), "the next block writes 0 into every lane's word, so every lane reads back 0");
    }

    #[test]
    fn find_follows_an_index_the_lane_stored_and_loaded_back_in_the_next_block() {
        assert!(!loaded_back_in_the_next_block(false), "word w of the table only ever holds w, so each lane reads back its own lane id");
    }

    #[test]
    fn find_follows_an_index_the_lane_stored_and_loaded_back() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, lane, k.exec);
        let back = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let address = byte_offset(&mut b, e, buf, back, 4);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "word w of the table only ever holds w, so each lane reads back its own lane id");
    }

    #[test]
    fn find_reports_the_lanes_after_a_join_where_one_path_halves_the_index() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let v = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let half = b.int(e, IntOp::LShr, lane, one);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        let (join, j) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, c, (then, vec![k.exec, lane]), (other, vec![k.exec, half]));
        b.br(then, join, vec![t[0], t[1]]);
        b.br(other, join, vec![o[0], o[1]]);
        let address = byte_offset(&mut b, join, buf, j[1], 4);
        let (s1, s2) = twice(&mut b, join, address, j[0]);
        let h = Hazards::find(&b.program(), &env2());
        assert!(h.conflicts().contains(&pair(&h, s1, s2)), "on the second path lanes 0 and 1 both store to word 0");
    }

    #[test]
    fn find_reports_indices_a_path_into_a_private_slot_halves() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let v = uniform_word(&mut b, &k, e, 16, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let slot = b.constant(e, Ty::I32, 16);
        let one = b.constant(e, Ty::I32, 1);
        let half = b.int(e, IntOp::LShr, lane, one);
        b.store(e, Space::Scratch, MemSize::B32, slot, lane, k.exec);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (last, l) = b.block(&[Ty::I1, Ty::I64]);
        b.cond_br(e, c, (then, vec![k.exec, buf]), (other, vec![k.exec, buf, half]));
        let slot_other = b.constant(other, Ty::I32, 16);
        b.store(other, Space::Scratch, MemSize::B32, slot_other, o[2], o[0]);
        b.br(other, last, vec![o[0], o[1]]);
        b.br(then, last, vec![t[0], t[1]]);
        let slot = b.constant(last, Ty::I32, 16);
        let index = b.load(last, Space::Scratch, MemSize::B32, slot, l[0]);
        let address = byte_offset(&mut b, last, l[1], index, 4);
        let s1 = store_at(&mut b, last, address, l[0]);
        let s2 = store_at(&mut b, last, address, l[0]);
        let h = Hazards::find(&b.program(), &env2());
        assert!(h.conflicts().contains(&pair(&h, s1, s2)), "on the second path lanes 0 and 1 reload 0 and store to word 0");
    }

    #[test]
    fn find_keeps_apart_the_lanes_after_a_join_of_two_offsets() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let v = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let base = byte_offset(&mut b, e, buf, lane, 8);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1, Ty::I64]);
        b.cond_br(e, c, (then, vec![k.exec, base]), (other, vec![k.exec, base]));
        b.br(then, join, vec![t[0], t[1]]);
        let four = b.constant(other, Ty::I64, 4);
        let shifted = b.int(other, IntOp::Add, o[1], four);
        b.br(other, join, vec![o[0], shifted]);
        let (s1, s2) = twice(&mut b, join, j[1], j[0]);
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "lane a stores to 8a or 8a + 4, which no other lane touches");
    }

    #[test]
    fn find_keeps_a_doubling_index_off_the_words_it_never_names() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, one, zero, buf]);
        let address = byte_offset(&mut b, body, p[3], p[1], 4);
        let s1 = store_at(&mut b, body, address, p[0]);
        let three = b.constant(body, Ty::I64, 12);
        let word = b.int(body, IntOp::Add, p[3], three);
        let s2 = store_at(&mut b, body, word, p[0]);
        let doubled = b.int(body, IntOp::Add, p[1], p[1]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], doubled, next, p[3]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.together.contains(&pair(&h, s1, s2)), "the index runs 1, 2, 4, 8 and never names word 3");
    }

    fn doubling_index(word: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, one, zero, buf]);
        let address = byte_offset(&mut b, body, p[3], p[1], 4);
        let s1 = store_at(&mut b, body, address, p[0]);
        let at = b.constant(body, Ty::I64, word * 4);
        let fixed = b.int(body, IntOp::Add, p[3], at);
        let s2 = store_at(&mut b, body, fixed, p[0]);
        let doubled = b.int(body, IntOp::Add, p[1], p[1]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], doubled, next, p[3]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_doubling_index_on_a_word_it_names() {
        assert!(doubling_index(8), "the index runs 1, 2, 4, 8 and names word 8 in the last iteration");
    }

    #[test]
    fn find_keeps_a_doubling_index_off_a_word_past_its_last_value() {
        assert!(!doubling_index(16), "the index stops at 8");
    }

    fn joined_twice(same: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::U16);
        let v = uniform_word(&mut b, &k, e, 4, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        let (join, j) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        b.cond_br(e, c, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
        let one = b.constant(then, Ty::I32, 1);
        let next = b.int(then, IntOp::Add, t[1], one);
        let second = if same { t[1] } else { next };
        b.br(then, join, vec![t[0], t[1], second]);
        let one = b.constant(other, Ty::I32, 1);
        let next = b.int(other, IntOp::Add, o[1], one);
        let two = b.constant(other, Ty::I32, 2);
        let after = b.int(other, IntOp::Add, o[1], two);
        let second = if same { next } else { after };
        b.br(other, join, vec![o[0], next, second]);
        let first_word = byte_offset(&mut b, join, buf, j[1], 4);
        let second_word = byte_offset(&mut b, join, buf, j[2], 4);
        let s1 = store_at(&mut b, join, first_word, j[0]);
        let s2 = store_at(&mut b, join, second_word, j[0]);
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_two_joins_of_one_branch_that_take_the_same_word() {
        assert!(joined_twice(true), "both joins bring u on one path and u + 1 on the other");
    }

    #[test]
    fn find_keeps_apart_two_joins_of_one_branch_one_word_apart() {
        assert!(!joined_twice(false), "the second join is always the first plus one");
    }

    fn joined_with_another_term(apart: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let v = uniform_word(&mut b, &k, e, 4, MemSize::U8);
        let flag = uniform_word(&mut b, &k, e, 8, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (join, j) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
        b.cond_br(e, c, (then, vec![k.exec, u, v]), (other, vec![k.exec, u, v]));
        b.br(then, join, vec![t[0], t[1], t[1], t[2]]);
        let sum = b.int(other, IntOp::Add, o[1], o[2]);
        b.br(other, join, vec![o[0], sum, o[1], o[2]]);
        let offset = b.int(join, IntOp::Sub, j[1], j[2]);
        let first = byte_offset(&mut b, join, buf, offset, 4);
        let shift = b.constant(join, Ty::I32, apart);
        let moved = b.int(join, IntOp::Add, j[3], shift);
        let second = byte_offset(&mut b, join, buf, moved, 4);
        let s1 = store_at(&mut b, join, first, j[0]);
        let s2 = store_at(&mut b, join, second, j[0]);
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_join_of_different_terms_that_meets_the_other_word() {
        assert!(joined_with_another_term(0), "on the second path the first word is v, the second word's index");
    }

    #[test]
    fn find_keeps_apart_a_join_of_different_terms_from_a_word_it_never_takes() {
        assert!(!joined_with_another_term(300), "the first index is 0 or v < 256, the second v + 300");
    }

    fn spilled_across_blocks(overwritten: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let at = uniform_word(&mut b, &k, e, 16, MemSize::U8);
        let four = b.constant(e, Ty::I32, 4);
        let slot = b.int(e, IntOp::Mul, at, four);
        let sixty_four = b.constant(e, Ty::I32, 64);
        let elsewhere = b.int(e, IntOp::Add, slot, sixty_four);
        let spill = |b: &mut Build, pointer: ValueId, at: ValueId| {
            let low = b.core(e, Ty::I32, Op::UnpackLo(pointer));
            let high = b.core(e, Ty::I32, Op::UnpackHi(pointer));
            let four = b.constant(e, Ty::I32, 4);
            let above = b.int(e, IntOp::Add, at, four);
            b.store(e, Space::Scratch, MemSize::B32, at, low, k.exec);
            b.store(e, Space::Scratch, MemSize::B32, above, high, k.exec);
        };
        spill(&mut b, first, slot);
        spill(&mut b, second, if overwritten { slot } else { elsewhere });
        let (next, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64]);
        b.br(e, next, vec![k.exec, slot, second]);
        let four = b.constant(next, Ty::I32, 4);
        let above = b.int(next, IntOp::Add, p[1], four);
        let low = b.load(next, Space::Scratch, MemSize::B32, p[1], p[0]);
        let high = b.load(next, Space::Scratch, MemSize::B32, above, p[0]);
        let pointer = b.core(next, Ty::I64, Op::Pack64(low, high));
        let lane = b.core(next, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, next, pointer, lane, 4);
        let s1 = store_at(&mut b, next, own, p[0]);
        let zero = b.constant(next, Ty::I32, 0);
        let word = byte_offset(&mut b, next, p[2], zero, 4);
        let s2 = store_at(&mut b, next, word, p[0]);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_pointer_reloaded_in_a_later_block_from_a_slot_a_second_spill_overwrote() {
        assert!(spilled_across_blocks(true), "the second spill overwrites the slot, so the reload is the second buffer");
    }

    #[test]
    fn find_keeps_a_pointer_reloaded_in_a_later_block_off_a_buffer_spilled_elsewhere() {
        assert!(!spilled_across_blocks(false), "the slot holds the first buffer; the second went 64 bytes further");
    }

    fn bounded_pair(below: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let v = uniform_word(&mut b, &k, e, 4, MemSize::B32);
        let three = b.constant(e, Ty::I32, 3);
        let small = b.cmp(e, IntPred::Ult, u, three);
        let ten = b.constant(e, Ty::I32, 10);
        let other = if below { b.cmp(e, IntPred::Ult, v, three) } else { b.cmp(e, IntPred::Ugt, v, ten) };
        let first_mask = b.int(e, IntOp::And, small, k.exec);
        let second_mask = b.int(e, IntOp::And, other, k.exec);
        let first = byte_offset(&mut b, e, buf, u, 4);
        let second = byte_offset(&mut b, e, buf, v, 4);
        let s1 = store_at(&mut b, e, first, first_mask);
        let s2 = store_at(&mut b, e, second, second_mask);
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_stores_whose_predicates_let_their_indices_meet() {
        assert!(bounded_pair(true), "u = v = 0 passes both predicates");
    }

    #[test]
    fn find_keeps_apart_stores_whose_predicates_keep_their_indices_apart() {
        assert!(!bounded_pair(false), "the first index is below 3 and the second above 10");
    }

    fn workgroup_branch(entered: u64) -> bool {
        let (mut b, k, extra) = Build::kernel_with(&[(ParameterSource::Sgpr(WORKGROUP_ID_X), Ty::I32)]);
        b.entry.workgroup_id_x = true;
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let entered = b.constant(e, Ty::I32, entered);
        let first = b.cmp(e, IntPred::Eq, extra[0], entered);
        let address = byte_offset(&mut b, e, buf, extra[0], 4);
        let four = b.constant(e, Ty::I64, 4);
        let word = b.int(e, IntOp::Add, buf, four);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, first, (then, vec![k.exec, address, word]), (exit, vec![k.exec]));
        let s1 = store_at(&mut b, then, t[1], t[0]);
        let s2 = store_at(&mut b, then, t[2], t[0]);
        let mut env = env2();
        env.grid = [4, 1, 1];
        let h = Hazards::find(&b.program(), &env);
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    fn branch_some_lanes_decide(cond: impl Fn(&mut Build, BlockId, ValueId, ValueId) -> ValueId) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let five = b.constant(e, Ty::I32, 5);
        let is_five = b.cmp(e, IntPred::Eq, u, five);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let c = cond(&mut b, e, lane, is_five);
        let (then, t) = b.block(&[Ty::I1]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, c, (then, vec![k.exec]), (other, vec![k.exec, buf]));
        b.br(then, join, vec![t[0]]);
        let s1 = store_at(&mut b, other, o[1], o[0]);
        let s2 = store_at(&mut b, other, o[1], o[0]);
        b.br(other, join, vec![o[0]]);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_decides_a_uniform_branch_that_some_lanes_decide() {
        let collide = branch_some_lanes_decide(|b, e, lane, is_five| {
            let sixteen = b.constant(e, Ty::I32, 16);
            let low = b.cmp(e, IntPred::Ult, lane, sixteen);
            b.int(e, IntOp::Or, low, is_five)
        });
        assert!(!collide, "lanes 0 to 15 take the branch, so the uniform branch always skips the storing arm");
    }

    #[test]
    fn find_reports_the_arm_a_uniform_branch_takes_when_some_lanes_decide() {
        let collide = branch_some_lanes_decide(|b, e, lane, is_five| {
            let sixteen = b.constant(e, Ty::I32, 16);
            let low = b.cmp(e, IntPred::Ult, lane, sixteen);
            b.int(e, IntOp::And, low, is_five)
        });
        assert!(collide, "lanes 16 to 31 skip the branch, so the uniform branch always takes the storing arm");
        let unknown = branch_some_lanes_decide(|_, _, _, is_five| is_five);
        assert!(unknown, "u == 5 is unknown in every lane, so both arms may run");
    }

    fn spilled_pointer(later: bool, known_slot: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let at = if known_slot { lane } else { uniform_word(&mut b, &k, e, 16, MemSize::U8) };
        let slot = b.int(e, IntOp::Mul, at, four);
        let low = b.core(e, Ty::I32, Op::UnpackLo(first));
        let high = b.core(e, Ty::I32, Op::UnpackHi(first));
        let eight = b.constant(e, Ty::I32, 8);
        let slot_high = b.int(e, IntOp::Add, slot, eight);
        b.store(e, Space::Scratch, MemSize::B32, slot, low, k.exec);
        b.store(e, Space::Scratch, MemSize::B32, slot_high, high, k.exec);
        let reload_low = b.load(e, Space::Scratch, MemSize::B32, slot, k.exec);
        let reload_high = b.load(e, Space::Scratch, MemSize::B32, slot_high, k.exec);
        let pointer = b.core(e, Ty::I64, Op::Pack64(reload_low, reload_high));
        let own = byte_offset(&mut b, e, pointer, lane, 4);
        let s1 = store_at(&mut b, e, own, k.exec);
        if later {
            let low = b.core(e, Ty::I32, Op::UnpackLo(second));
            let high = b.core(e, Ty::I32, Op::UnpackHi(second));
            b.store(e, Space::Scratch, MemSize::B32, slot, low, k.exec);
            b.store(e, Space::Scratch, MemSize::B32, slot_high, high, k.exec);
        }
        let zero = b.constant(e, Ty::I32, 0);
        let word = byte_offset(&mut b, e, second, zero, 4);
        let s2 = store_at(&mut b, e, word, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    fn spilled_in_a_loop(respill_first: bool) -> bool {
        spilled_in_a_loop_when(respill_first, false)
    }

    fn spilled_in_a_loop_when(respill_first: bool, sometimes: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let at = uniform_word(&mut b, &k, e, 16, MemSize::U8);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero, at, first, second]);
        let four = b.constant(body, Ty::I32, 4);
        let slot = b.int(body, IntOp::Mul, p[2], four);
        let eight = b.constant(body, Ty::I32, 8);
        let slot_high = b.int(body, IntOp::Add, slot, eight);
        let five = b.constant(body, Ty::I32, 5);
        let is_five = b.cmp(body, IntPred::Eq, p[2], five);
        let rarely = b.int(body, IntOp::And, is_five, p[0]);
        let spill = |b: &mut Build, pointer: ValueId, mask: ValueId| {
            let low = b.core(body, Ty::I32, Op::UnpackLo(pointer));
            let high = b.core(body, Ty::I32, Op::UnpackHi(pointer));
            b.store(body, Space::Scratch, MemSize::B32, slot, low, mask);
            b.store(body, Space::Scratch, MemSize::B32, slot_high, high, mask);
        };
        if respill_first {
            spill(&mut b, p[3], if sometimes { rarely } else { p[0] });
        }
        let reload_low = b.load(body, Space::Scratch, MemSize::B32, slot, p[0]);
        let reload_high = b.load(body, Space::Scratch, MemSize::B32, slot_high, p[0]);
        let pointer = b.core(body, Ty::I64, Op::Pack64(reload_low, reload_high));
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, body, pointer, lane, 4);
        let s1 = store_at(&mut b, body, own, p[0]);
        spill(&mut b, p[4], p[0]);
        let word = byte_offset(&mut b, body, p[4], zero, 4);
        let s2 = store_at(&mut b, body, word, p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[1], one);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], next, p[2], p[3], p[4]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_keeps_a_reloaded_pointer_off_a_buffer_the_loop_spills_after_the_reload() {
        assert!(!spilled_in_a_loop(true), "each iteration spills the first buffer right before the reload, so the reload never returns the second");
    }

    #[test]
    fn find_reports_a_reloaded_pointer_a_conditional_spill_may_leave_behind() {
        assert!(spilled_in_a_loop_when(true, true), "when u != 5 the spill before the reload does not run, so iteration 2 reloads the second buffer");
    }

    #[test]
    fn find_reports_a_reloaded_pointer_the_previous_iteration_spilled() {
        assert!(spilled_in_a_loop(false), "without the spill before the reload, iteration 2 reloads the second buffer");
    }

    #[test]
    fn find_keeps_a_reloaded_pointer_off_a_buffer_only_a_later_spill_names() {
        assert!(!spilled_pointer(true, true), "the reload can only return the first buffer; the second is spilled after it");
        assert!(!spilled_pointer(true, false), "the same holds when the slot's address is an unknown word");
    }

    #[test]
    fn find_keeps_a_reloaded_pointer_off_a_buffer_no_spill_names() {
        assert!(!spilled_pointer(false, true), "only the first buffer is ever spilled");
        assert!(!spilled_pointer(false, false), "only the first buffer is ever spilled");
    }

    #[test]
    fn find_uses_what_the_branch_into_a_block_says_about_the_workgroup() {
        assert!(!workgroup_branch(0), "only workgroup 0 enters the block, and it stores to words 0 and 1");
    }

    #[test]
    fn find_reports_the_workgroup_the_branch_into_a_block_lets_in() {
        assert!(workgroup_branch(1), "workgroup 1 enters the block and stores to word 1 twice");
    }

    fn two_halves(bump: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::U16);
        let one = b.constant(e, Ty::I32, 1);
        let bump = b.constant(e, Ty::I32, bump);
        let low = b.int(e, IntOp::LShr, u, one);
        let bumped = b.int(e, IntOp::Add, u, bump);
        let high = b.int(e, IntOp::LShr, bumped, one);
        let a1 = byte_offset(&mut b, e, buf, low, 4);
        let a2 = byte_offset(&mut b, e, buf, high, 4);
        let s1 = store_at(&mut b, e, a1, k.exec);
        let s2 = store_at(&mut b, e, a2, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_relates_two_indices_derived_from_one_word() {
        assert!(!two_halves(2), "(u + 2) >> 1 is always (u >> 1) + 1");
    }

    #[test]
    fn find_reports_two_indices_derived_from_one_word_that_can_meet() {
        assert!(two_halves(1), "(u + 1) >> 1 is u >> 1 whenever u is even");
    }

    #[test]
    fn find_keeps_apart_pointers_reloaded_from_one_slot_at_different_times() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero, first, second]);
        let slot = b.constant(body, Ty::I32, 16);
        b.store(body, Space::Scratch, MemSize::B64, slot, p[2], p[0]);
        let r1 = b.load(body, Space::Scratch, MemSize::B64, slot, p[0]);
        let s1 = store_at(&mut b, body, r1, p[0]);
        b.store(body, Space::Scratch, MemSize::B64, slot, p[3], p[0]);
        let s3 = store_at(&mut b, body, p[3], p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[1], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], next, p[2], p[3]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &env2());
        assert!(
            !h.conflicts().contains(&pair(&h, s1, s3)),
            "each iteration spills the first buffer right before reloading it, and the other store is to the second"
        );
    }

    fn at_named_words(first: u64, second: u64, pred: IntPred) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::U8);
        let w = uniform_word(&mut b, &k, e, 4, MemSize::U8);
        let first = b.constant(e, Ty::I32, first);
        let second = b.constant(e, Ty::I32, second);
        let at_first = b.cmp(e, pred, u, first);
        let at_second = b.cmp(e, pred, w, second);
        let m1 = b.int(e, IntOp::And, at_first, k.exec);
        let m2 = b.int(e, IntOp::And, at_second, k.exec);
        let a1 = byte_offset(&mut b, e, buf, u, 4);
        let a2 = byte_offset(&mut b, e, buf, w, 4);
        let s1 = store_at(&mut b, e, a1, m1);
        let s2 = store_at(&mut b, e, a2, m2);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_excludes_stores_that_each_run_only_at_a_different_word() {
        assert!(!at_named_words(7, 8, IntPred::Eq), "the first store runs only at word 7, the second only at word 8");
    }

    #[test]
    fn find_reports_stores_that_both_run_at_one_named_word() {
        assert!(at_named_words(7, 7, IntPred::Eq), "both stores run at word 7 when both loaded words are 7");
    }

    #[test]
    fn find_reports_an_unguarded_store_at_a_word_another_store_names_under_its_guard() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::U8);
        let seven = b.constant(e, Ty::I32, 7);
        let at_seven = b.cmp(e, IntPred::Eq, u, seven);
        let m1 = b.int(e, IntOp::And, at_seven, k.exec);
        let address = byte_offset(&mut b, e, buf, u, 4);
        store_at(&mut b, e, address, m1);
        let s2 = store_at(&mut b, e, address, k.exec);
        let eight = b.constant(e, Ty::I64, 32);
        let word = b.int(e, IntOp::Add, buf, eight);
        let s3 = store_at(&mut b, e, word, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(h.conflicts().contains(&pair(&h, s2, s3)), "the unguarded store runs at word 8 when u is 8");
    }

    #[test]
    fn find_reports_stores_that_skip_different_words() {
        assert!(at_named_words(7, 8, IntPred::Ne), "u = w = 3 runs both stores at word 3");
    }

    #[test]
    fn find_keeps_apart_stores_on_exclusive_arms_of_a_uniform_branch() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let v = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64]);
        b.cond_br(e, c, (then, vec![k.exec, buf]), (other, vec![k.exec, buf]));
        let s1 = store_at(&mut b, then, t[1], t[0]);
        let s2 = store_at(&mut b, other, o[1], o[0]);
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "a wave takes one arm of a uniform branch, so no two lanes run both stores");
    }

    #[test]
    fn find_follows_a_pointer_whose_halves_move_between_lanes() {
        let names = ["read lane 3", "read first lane", "backward permute from lane 0", "write lane 2"];
        let lost: Vec<&str> = names
            .iter()
            .enumerate()
            .filter(|&(case, _)| {
                let (mut b, k) = Build::kernel();
                let e = BlockId(0);
                let buf = k.buffer(&mut b, e, 0);
                let lo = b.core(e, Ty::I32, Op::UnpackLo(buf));
                let hi = b.core(e, Ty::I32, Op::UnpackHi(buf));
                let zero = b.constant(e, Ty::I32, 0);
                let moved: Vec<ValueId> = [lo, hi]
                    .iter()
                    .map(|&half| match case {
                        0 => {
                            let three = b.constant(e, Ty::I32, 3);
                            b.wave(e, WaveOp::ReadLane, vec![half, three, zero])
                        }
                        1 => b.wave(e, WaveOp::ReadFirstLane, vec![half, k.exec]),
                        2 => b.wave(e, WaveOp::Bpermute, vec![zero, half, k.exec]),
                        _ => {
                            let two = b.constant(e, Ty::I32, 2);
                            b.wave(e, WaveOp::WriteLane, vec![half, two, half, zero])
                        }
                    })
                    .collect();
                let rebuilt = b.core(e, Ty::I64, Op::Pack64(moved[0], moved[1]));
                let s1 = store_at(&mut b, e, rebuilt, k.exec);
                let s2 = store_at(&mut b, e, buf, k.exec);
                let h = Hazards::find(&b.program(), &env2());
                !h.together.contains(&pair(&h, s1, s2))
            })
            .map(|(_, &name)| name)
            .collect();
        assert!(lost.is_empty(), "{:?}: the rebuilt pointer is buf, where every lane stores next", lost);
    }

    #[test]
    fn find_exposes_a_pointer_a_masked_store_leaks() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let mask = b.int(e, IntOp::And, first, k.exec);
        let null = b.constant(e, Ty::I64, 0);
        let leaked = b.core(e, Ty::I64, Op::Select(first, buf, null));
        b.store(e, Space::Global, MemSize::B64, table, leaked, mask);
        let yes = b.constant(e, Ty::I1, 1);
        let reloaded = b.load(e, Space::Global, MemSize::B64, table, yes);
        let s1 = store_at(&mut b, e, reloaded, k.exec);
        let s2 = store_at(&mut b, e, buf, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(h.together.contains(&pair(&h, s1, s2)), "lane 0 writes buf into the table, and every lane stores through what it reads back");
    }

    fn slot_index(join: bool, same: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let v = uniform_word(&mut b, &k, e, 16, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let slot = b.constant(e, Ty::I32, 16);
        let wave = b.constant(e, Ty::I32, 32);
        let shifted = b.int(e, IntOp::Add, lane, wave);
        b.store(e, Space::Scratch, MemSize::B32, slot, lane, k.exec);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (last, l) = b.block(&[Ty::I1, Ty::I64]);
        if join {
            let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
            b.cond_br(e, c, (then, vec![k.exec, buf, shifted]), (other, vec![k.exec, buf, shifted]));
            let slot = b.constant(other, Ty::I32, 16);
            if !same {
                b.store(other, Space::Scratch, MemSize::B32, slot, o[2], o[0]);
            }
            b.br(other, last, vec![o[0], o[1]]);
        } else {
            b.br(e, then, vec![k.exec, buf, shifted]);
        }
        b.br(then, last, vec![t[0], t[1]]);
        let slot = b.constant(last, Ty::I32, 16);
        let index = b.load(last, Space::Scratch, MemSize::B32, slot, l[0]);
        let address = byte_offset(&mut b, last, l[1], index, 4);
        let s1 = store_at(&mut b, last, address, l[0]);
        let s2 = store_at(&mut b, last, address, l[0]);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_follows_an_index_through_private_memory_into_later_blocks() {
        assert!(!slot_index(false, true), "each lane reloads its own lane id");
        assert!(!slot_index(true, true), "both paths keep the lane id in the slot");
    }

    #[test]
    fn find_keeps_apart_indices_that_differ_between_the_paths_into_a_private_slot() {
        assert!(!slot_index(true, false), "every lane reloads its lane id, or every lane its lane id + 32, which no other lane stores to");
    }

    fn ballot_bit(bit: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let low = b.cmp(e, IntPred::Ult, lane, four);
        let w = b.wave(e, WaveOp::Ballot, vec![low]);
        let k2 = b.constant(e, Ty::I32, bit);
        let shifted = b.int(e, IntOp::LShr, w, k2);
        let set = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let mask = b.int(e, IntOp::And, set, k.exec);
        let (s1, s2) = twice(&mut b, e, buf, mask);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reads_a_fixed_bit_of_a_ballot() {
        assert!(ballot_bit(3), "lane 3 is below 4, so every lane stores");
        assert!(!ballot_bit(5), "lane 5 is not below 4, so no lane stores");
    }

    fn ballot_carried(start: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let flag = b.constant(e, Ty::I1, start);
        let (body, p) = b.block(&[Ty::I1, Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, flag, zero, buf, other]);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, body, p[4], lane, 4);
        let yes = b.constant(body, Ty::I1, 1);
        let v = b.load(body, Space::Global, MemSize::B32, own, yes);
        let z = b.constant(body, Ty::I32, 0);
        let fresh = b.cmp(body, IntPred::Ne, v, z);
        let both = b.int(body, IntOp::And, p[1], fresh);
        let w = b.wave(body, WaveOp::Ballot, vec![both]);
        let shifted = b.int(body, IntOp::LShr, w, lane);
        let still = b.core(body, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let mask = b.int(body, IntOp::And, p[1], p[0]);
        let s1 = store_at(&mut b, body, p[3], mask);
        let s2 = store_at(&mut b, body, p[3], p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], still, next, p[3], p[4]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_carries_a_lane_bit_of_a_ballot_around_a_loop() {
        assert!(ballot_carried(1), "the flag starts set, so every lane stores twice in iteration 0");
        assert!(!ballot_carried(0), "the flag starts clear and each iteration keeps the lane's own bit of flag & fresh");
    }

    const PATHS: [&str; 16] = [
        "an 8-byte private spill",
        "two 4-byte private spills",
        "global memory",
        "lds",
        "xor twice",
        "mul by one",
        "shl by zero",
        "or with zero",
        "and with all ones",
        "select between itself",
        "read lane 3",
        "read first lane",
        "backward permute from lane 0",
        "write lane 2",
        "a store only lane 0 makes",
        "nothing",
    ];

    fn carried_to(path: usize, moved_is_first: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let table = k.buffer(&mut b, e, 16);
        let pointer = if moved_is_first { first } else { second };
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let yes = b.constant(e, Ty::I1, 1);
        let halves = |b: &mut Build, f: &dyn Fn(&mut Build, ValueId) -> ValueId| {
            let lo = b.core(e, Ty::I32, Op::UnpackLo(pointer));
            let hi = b.core(e, Ty::I32, Op::UnpackHi(pointer));
            let (lo, hi) = (f(b, lo), f(b, hi));
            b.core(e, Ty::I64, Op::Pack64(lo, hi))
        };
        let moved = match path {
            0 => {
                let slot = b.constant(e, Ty::I32, 16);
                b.store(e, Space::Scratch, MemSize::B64, slot, pointer, k.exec);
                b.load(e, Space::Scratch, MemSize::B64, slot, k.exec)
            }
            1 => {
                let lo = b.core(e, Ty::I32, Op::UnpackLo(pointer));
                let hi = b.core(e, Ty::I32, Op::UnpackHi(pointer));
                let (s0, s1) = (b.constant(e, Ty::I32, 16), b.constant(e, Ty::I32, 20));
                b.store(e, Space::Scratch, MemSize::B32, s0, lo, k.exec);
                b.store(e, Space::Scratch, MemSize::B32, s1, hi, k.exec);
                let lo = b.load(e, Space::Scratch, MemSize::B32, s0, k.exec);
                let hi = b.load(e, Space::Scratch, MemSize::B32, s1, k.exec);
                b.core(e, Ty::I64, Op::Pack64(lo, hi))
            }
            2 => {
                b.store(e, Space::Global, MemSize::B64, table, pointer, k.exec);
                b.load(e, Space::Global, MemSize::B64, table, yes)
            }
            3 => {
                let place = b.constant(e, Ty::I32, 64);
                b.store(e, Space::Lds, MemSize::B64, place, pointer, k.exec);
                b.load(e, Space::Lds, MemSize::B64, place, yes)
            }
            4 => {
                let key = b.constant(e, Ty::I64, 0x5a5a);
                let x = b.int(e, IntOp::Xor, pointer, key);
                b.int(e, IntOp::Xor, x, key)
            }
            5 => {
                let one = b.constant(e, Ty::I64, 1);
                b.int(e, IntOp::Mul, pointer, one)
            }
            6 => {
                let z = b.constant(e, Ty::I64, 0);
                b.int(e, IntOp::Shl, pointer, z)
            }
            7 => {
                let z = b.constant(e, Ty::I64, 0);
                b.int(e, IntOp::Or, pointer, z)
            }
            8 => {
                let all = b.constant(e, Ty::I64, u64::MAX);
                b.int(e, IntOp::And, pointer, all)
            }
            9 => {
                let v = b.load(e, Space::Global, MemSize::B32, table, yes);
                let c = b.cmp(e, IntPred::Eq, v, zero);
                b.core(e, Ty::I64, Op::Select(c, pointer, pointer))
            }
            10 => halves(&mut b, &|b, h| {
                let three = b.constant(e, Ty::I32, 3);
                let z = b.constant(e, Ty::I32, 0);
                b.wave(e, WaveOp::ReadLane, vec![h, three, z])
            }),
            11 => halves(&mut b, &|b, h| b.wave(e, WaveOp::ReadFirstLane, vec![h, k.exec])),
            12 => halves(&mut b, &|b, h| {
                let z = b.constant(e, Ty::I32, 0);
                b.wave(e, WaveOp::Bpermute, vec![z, h, k.exec])
            }),
            13 => halves(&mut b, &|b, h| {
                let two = b.constant(e, Ty::I32, 2);
                let z = b.constant(e, Ty::I32, 0);
                b.wave(e, WaveOp::WriteLane, vec![h, two, h, z])
            }),
            14 => {
                let only = b.cmp(e, IntPred::Eq, lane, zero);
                let mask = b.int(e, IntOp::And, only, k.exec);
                let null = b.constant(e, Ty::I64, 0);
                let leaked = b.core(e, Ty::I64, Op::Select(only, pointer, null));
                b.store(e, Space::Global, MemSize::B64, table, leaked, mask);
                b.load(e, Space::Global, MemSize::B64, table, yes)
            }
            _ => pointer,
        };
        let s1 = store_at(&mut b, e, moved, k.exec);
        let s2 = store_at(&mut b, e, first, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000), (16, 3, 0x3000)]));
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_follows_a_pointer_to_the_same_buffer_through_every_path() {
        let missed: Vec<&str> = (0..PATHS.len()).filter(|&p| !carried_to(p, true)).map(|p| PATHS[p]).collect();
        assert!(missed.is_empty(), "{:?}: the pointer that arrives is the first buffer, where every lane stores next", missed);
    }

    #[test]
    fn find_keeps_apart_a_pointer_to_another_buffer_through_every_path() {
        let loose: Vec<&str> = (0..PATHS.len()).filter(|&p| carried_to(p, false)).map(|p| PATHS[p]).collect();
        assert!(loose.is_empty(), "{:?}: the pointer that arrives is the second buffer, and no one leaks the first", loose);
    }

    #[test]
    fn find_keeps_apart_lanes_of_one_row() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let mask = b.constant(e, Ty::I32, 0x3ff);
        let x = b.int(e, IntOp::And, k.item, mask);
        let address = byte_offset(&mut b, e, buf, x, 4);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "in a 32 x 1 block every lane has its own x");
    }

    #[test]
    fn find_keeps_a_sign_extended_offset_off_words_it_cannot_reach() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, other, lane, 4);
        let v = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Eq, v, zero);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, c));
        let four = b.constant(e, Ty::I64, 4);
        let offset = b.int(e, IntOp::Mul, wide, four);
        let low = b.int(e, IntOp::Add, buf, offset);
        let eight = b.constant(e, Ty::I64, 8);
        let far = b.int(e, IntOp::Add, buf, eight);
        let s1 = store_at(&mut b, e, low, k.exec);
        let s2 = store_at(&mut b, e, far, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "sext(c) * 4 is 0 or -4, never 8");
    }

    #[test]
    fn find_drops_a_store_masked_by_a_projected_bit_that_is_always_clear() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let three = b.constant(e, Ty::I32, 3);
        let third = b.cmp(e, IntPred::Eq, lane, three);
        let zero = b.constant(e, Ty::I32, 0);
        let seven = b.constant(e, Ty::I32, 7);
        let word = b.core(e, Ty::I32, Op::Select(third, zero, seven));
        let shifted = b.int(e, IntOp::LShr, word, three);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let s1 = store_at(&mut b, e, buf, bit);
        let s2 = store_at(&mut b, e, buf, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "bit 3 of 0 and of 7 is clear, so the first store never runs");
    }

    #[test]
    fn find_keeps_apart_the_lanes_of_a_single_wave() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let wave = b.constant(e, Ty::I32, 32);
        let later = b.cmp(e, IntPred::Uge, k.item, wave);
        let own = byte_offset(&mut b, e, buf, k.item, 4);
        let address = b.core(e, Ty::I64, Op::Select(later, buf, own));
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "with 32 lanes every lane stores to its own word");
    }

    #[test]
    fn find_keeps_apart_lanes_with_their_own_lds_words() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let address = b.int(e, IntOp::Mul, lane, four);
        let zero = b.constant(e, Ty::I32, 0);
        let s1 = b.here(e);
        b.store(e, Space::Lds, MemSize::B32, address, zero, k.exec);
        let s2 = b.here(e);
        b.store(e, Space::Lds, MemSize::B32, address, zero, k.exec);
        let h = Hazards::find(&b.program(), &environment(32, &[]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)));
    }

    #[test]
    fn find_keeps_apart_lanes_that_a_mask_of_the_loop_index_keeps_distinct() {
        let Looped {
            mut b,
            buf,
            body,
            exec,
            index,
            ..
        } = looped(4);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let slid = b.int(body, IntOp::Add, index, lane);
        let mask = b.constant(body, Ty::I32, 31);
        let slot = b.int(body, IntOp::And, slid, mask);
        let address = byte_offset(&mut b, body, buf, slot, 4);
        let s = store_at(&mut b, body, address, exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.together.contains(&pair(&h, s, s)), "(i + lane) & 31 differs between the lanes of one iteration");
    }

    #[test]
    fn find_keeps_apart_the_lanes_after_a_loop_when_each_adds_its_lane() {
        let Looped {
            mut b,
            buf,
            exit,
            last,
            ..
        } = looped(4);
        let exec = b.f.blocks[&exit].params[0].0;
        let lane = b.core(exit, Ty::I32, Op::Env(Env::LaneId));
        let index = b.int(exit, IntOp::Add, last, lane);
        let address = byte_offset(&mut b, exit, buf, index, 4);
        let s1 = store_at(&mut b, exit, address, exec);
        let s2 = store_at(&mut b, exit, address, exec);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s2)), "last + lane differs between the lanes");
    }

    #[test]
    fn find_keeps_a_pointer_that_swaps_between_two_buffers_off_a_third() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let third = k.buffer(&mut b, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I64, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, first, first, second, zero, third]);
        let s1 = store_at(&mut b, body, p[1], p[0]);
        let s3 = store_at(&mut b, body, p[5], p[0]);
        let at_first = b.cmp(body, IntPred::Eq, p[1], p[2]);
        let swapped = b.core(body, Ty::I64, Op::Select(at_first, p[3], p[2]));
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[4], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], swapped, p[2], p[3], next, p[5]]), (exit, vec![p[0]]));
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000), (16, 3, 0x3000)]));
        assert!(!h.conflicts().contains(&pair(&h, s1, s3)), "the pointer is only ever the first or the second buffer");
    }

    #[test]
    fn find_reports_an_index_every_lane_reloads_from_a_private_slot() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let slot = b.constant(e, Ty::I32, 16);
        let seven = b.constant(e, Ty::I32, 7);
        b.store(e, Space::Scratch, MemSize::B32, slot, seven, k.exec);
        let (next, n) = b.block(&[Ty::I1, Ty::I64]);
        b.br(e, next, vec![k.exec, buf]);
        let slot = b.constant(next, Ty::I32, 16);
        let index = b.load(next, Space::Scratch, MemSize::B32, slot, n[0]);
        let address = byte_offset(&mut b, next, n[1], index, 4);
        let (s1, s2) = twice(&mut b, next, address, n[0]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000)]));
        assert!(h.together.contains(&pair(&h, s1, s2)), "every lane reloads 7 and stores to word 7");
    }

    #[test]
    fn positions_put_each_meeting_before_the_first_access_of_its_instruction() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let other = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let first_part = b.here(e);
        for (address, provenance) in [(other, 0x7700u64), (buf, 0x7701)] {
            b.f.blocks.get_mut(&e).unwrap().insts.push(Inst::Effect {
                provenance,
                op: memory(Space::Global, MemoryOp::Store(MemSize::B32)),
                inputs: vec![address, zero, k.exec],
                outputs: vec![],
            });
        }
        let second_part = (e, first_part.1 + 1);
        let later = store_at(&mut b, e, buf, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(second_part, later)], &[], &[]);
        assert_eq!(hazards.meetings, vec![first_part, later], "one meeting before the whole instruction, one before the later store");
    }

    fn first_lane(mask: usize, other: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let three = b.constant(e, Ty::I32, 3);
        let mask = match mask {
            0 => b.cmp(e, IntPred::Uge, lane, three),
            1 => b.constant(e, Ty::I1, 0),
            _ => {
                let flags = k.buffer(&mut b, e, 8);
                let own = byte_offset(&mut b, e, flags, lane, 4);
                let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
                let zero = b.constant(e, Ty::I32, 0);
                b.cmp(e, IntPred::Ne, flag, zero)
            }
        };
        let index = b.wave(e, WaveOp::ReadFirstLane, vec![lane, mask]);
        let address = byte_offset(&mut b, e, buf, index, 4);
        let s1 = store_at(&mut b, e, address, k.exec);
        let at = b.constant(e, Ty::I64, other * 4);
        let there = b.int(e, IntOp::Add, buf, at);
        let s2 = store_at(&mut b, e, there, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reads_the_first_lane_whose_mask_is_set() {
        let cases = [
            ("mask lane >= 3, other word 3", 0, 3, true),
            ("mask lane >= 3, other word 0", 0, 0, false),
            ("empty mask, other word 0", 1, 0, true),
            ("empty mask, other word 1", 1, 1, false),
            ("loaded mask, other word 5", 2, 5, true),
        ];
        let wrong: Vec<(&str, bool)> = cases
            .iter()
            .filter(|&&(_, mask, other, expected)| first_lane(mask, other) != expected)
            .map(|&(name, .., expected)| (name, expected))
            .collect();
        assert!(wrong.is_empty(), "(case, whether the stores meet): {:?}", wrong);
    }

    #[test]
    fn find_reports_an_arm_and_the_join_after_it() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let v = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, v, zero);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1, Ty::I64]);
        b.cond_br(e, c, (then, vec![k.exec, buf]), (join, vec![k.exec, buf]));
        let s1 = store_at(&mut b, then, t[1], t[0]);
        b.br(then, join, vec![t[0], t[1]]);
        let s2 = store_at(&mut b, join, j[1], j[0]);
        let h = Hazards::find(&b.program(), &env2());
        assert!(h.together.contains(&pair(&h, s1, s2)), "a wave that takes the arm stores there and then at the join");
    }

    #[test]
    fn find_reports_exclusive_arms_that_different_iterations_take() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let (head, h) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (latch, l) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, head, vec![k.exec, zero, buf, table]);
        let at = byte_offset(&mut b, head, h[3], h[1], 4);
        let yes = b.constant(head, Ty::I1, 1);
        let v = b.load(head, Space::Global, MemSize::B32, at, yes);
        let z = b.constant(head, Ty::I32, 0);
        let c = b.cmp(head, IntPred::Ne, v, z);
        b.cond_br(head, c, (then, h.clone()), (other, h.clone()));
        let s1 = store_at(&mut b, then, t[2], t[0]);
        b.br(then, latch, t.clone());
        let s2 = store_at(&mut b, other, o[2], o[0]);
        b.br(other, latch, o.clone());
        let one = b.constant(latch, Ty::I32, 1);
        let next = b.int(latch, IntOp::Add, l[1], one);
        let four = b.constant(latch, Ty::I32, 4);
        let again = b.cmp(latch, IntPred::Ult, next, four);
        b.cond_br(latch, again, (head, vec![l[0], next, l[2], l[3]]), (exit, vec![l[0]]));
        let hz = Hazards::find(&b.program(), &env2());
        assert!(hz.apart.contains(&pair(&hz, s1, s2)), "one iteration may take the first arm and another the second");
    }

    #[test]
    fn find_reports_a_uniform_store_in_a_loop_within_one_iteration_only() {
        let (h, key) = counted_loop(4, false);
        assert!(h.together.contains(&key));
        assert!(!h.apart.contains(&key));
    }

    #[test]
    fn find_reports_a_sliding_store_in_a_loop_across_iterations_only() {
        let (h, key) = counted_loop(4, true);
        assert!(!h.together.contains(&key));
        assert!(h.apart.contains(&key));
    }

    #[test]
    fn find_reports_no_second_iteration_of_a_loop_that_runs_once() {
        let (h, key) = counted_loop(1, true);
        assert!(!h.conflicts().contains(&key));
    }

    fn two_index_read_back(shared: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let u = uniform_word(&mut b, &k, e, 16, MemSize::U8);
        let column = byte_offset(&mut b, e, table, lane, 4);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
        let row_bytes = b.constant(e, Ty::I64, 128);
        let row = b.int(e, IntOp::Mul, wide, row_bytes);
        let own = b.int(e, IntOp::Add, column, row);
        let thirty_two = b.constant(e, Ty::I32, 32);
        let scaled = b.int(e, IntOp::Mul, u, thirty_two);
        let value = if shared { scaled } else { b.int(e, IntOp::Add, lane, scaled) };
        b.store(e, Space::Global, MemSize::B32, own, value, k.exec);
        let back = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let address = byte_offset(&mut b, e, buf, back, 4);
        let (s1, s2) = twice(&mut b, e, address, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_word_two_indices_address_that_every_lane_shares() {
        assert!(two_index_read_back(true), "every lane stores 32 u and reads it back");
    }

    #[test]
    fn find_follows_a_word_two_indices_address_stored_and_loaded_back() {
        assert!(!two_index_read_back(false), "the word at 4 lane + 128 u only ever holds lane + 32 u, which differs between lanes");
    }

    fn joined_three_ways(apart: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let f1 = uniform_word(&mut b, &k, e, 4, MemSize::B32);
        let f2 = uniform_word(&mut b, &k, e, 8, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c1 = b.cmp(e, IntPred::Ne, f1, zero);
        let c2 = b.cmp(e, IntPred::Ne, f2, zero);
        let (one, a) = b.block(&[Ty::I1, Ty::I32]);
        let (rest, r) = b.block(&[Ty::I1, Ty::I32, Ty::I1]);
        let (two, t) = b.block(&[Ty::I1, Ty::I32]);
        let (three, h) = b.block(&[Ty::I1, Ty::I32]);
        let (join, j) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        b.cond_br(e, c1, (one, vec![k.exec, u]), (rest, vec![k.exec, u, c2]));
        b.cond_br(rest, r[2], (two, vec![r[0], r[1]]), (three, vec![r[0], r[1]]));
        b.br(one, join, vec![a[0], a[1], a[1]]);
        let step = b.constant(two, Ty::I32, 1);
        let next = b.int(two, IntOp::Add, t[1], step);
        b.br(two, join, vec![t[0], next, t[1]]);
        let step = b.constant(three, Ty::I32, 5);
        let next = b.int(three, IntOp::Add, h[1], step);
        b.br(three, join, vec![h[0], next, h[1]]);
        let first = byte_offset(&mut b, join, buf, j[1], 4);
        let shift = b.constant(join, Ty::I32, apart);
        let moved = b.int(join, IntOp::Add, j[2], shift);
        let second = byte_offset(&mut b, join, buf, moved, 4);
        let s1 = store_at(&mut b, join, first, j[0]);
        let s2 = store_at(&mut b, join, second, j[0]);
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_three_way_join_that_meets_the_other_word() {
        assert!(joined_three_ways(5), "the third path brings u + 5");
    }

    #[test]
    fn find_keeps_apart_a_three_way_join_from_a_word_it_never_takes() {
        assert!(!joined_three_ways(3), "the join is u, u + 1 or u + 5, never u + 3");
    }

    fn selected_with_another_term(apart: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::B32);
        let v = uniform_word(&mut b, &k, e, 4, MemSize::U8);
        let flag = uniform_word(&mut b, &k, e, 8, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let sum = b.int(e, IntOp::Add, u, v);
        let chosen = b.core(e, Ty::I32, Op::Select(c, u, sum));
        let offset = b.int(e, IntOp::Sub, chosen, u);
        let first = byte_offset(&mut b, e, buf, offset, 4);
        let shift = b.constant(e, Ty::I32, apart);
        let moved = b.int(e, IntOp::Add, v, shift);
        let second = byte_offset(&mut b, e, buf, moved, 4);
        let s1 = store_at(&mut b, e, first, k.exec);
        let s2 = store_at(&mut b, e, second, k.exec);
        let h = Hazards::find(&b.program(), &env2());
        h.together.contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_select_of_different_terms_that_meets_the_other_word() {
        assert!(selected_with_another_term(0), "when the flag is clear the first word is v, the second word's index");
    }

    #[test]
    fn find_keeps_apart_a_select_of_different_terms_from_a_word_it_never_takes() {
        assert!(!selected_with_another_term(300), "the first index is 0 or v < 256, the second v + 300");
    }

    enum Reload {
        Chain(usize),
        ThreeWays,
        Loop,
    }

    fn spilled_before(shape: Reload, overwritten: bool) -> bool {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let at = uniform_word(&mut b, &k, e, 16, MemSize::U8);
        let f1 = uniform_word(&mut b, &k, e, 20, MemSize::B32);
        let f2 = uniform_word(&mut b, &k, e, 24, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let c1 = b.cmp(e, IntPred::Ne, f1, zero);
        let c2 = b.cmp(e, IntPred::Ne, f2, zero);
        let four = b.constant(e, Ty::I32, 4);
        let slot = b.int(e, IntOp::Mul, at, four);
        let sixty_four = b.constant(e, Ty::I32, 64);
        let elsewhere = b.int(e, IntOp::Add, slot, sixty_four);
        let spill = |b: &mut Build, block: BlockId, exec: ValueId, pointer: ValueId, at: ValueId| {
            let low = b.core(block, Ty::I32, Op::UnpackLo(pointer));
            let high = b.core(block, Ty::I32, Op::UnpackHi(pointer));
            let four = b.constant(block, Ty::I32, 4);
            let above = b.int(block, IntOp::Add, at, four);
            b.store(block, Space::Scratch, MemSize::B32, at, low, exec);
            b.store(block, Space::Scratch, MemSize::B32, above, high, exec);
        };
        spill(&mut b, e, k.exec, first, slot);
        let looping = matches!(shape, Reload::Loop);
        spill(&mut b, e, k.exec, second, if overwritten && !looping { slot } else { elsewhere });
        let carried = [Ty::I1, Ty::I32, Ty::I64];
        let (last, p) = match shape {
            Reload::Chain(n) => {
                let mut from = e;
                let mut args = vec![k.exec, slot, second];
                let mut block = (e, Vec::new());
                for _ in 0..n {
                    block = b.block(&carried);
                    b.br(from, block.0, args.clone());
                    from = block.0;
                    args = block.1.clone();
                }
                block
            }
            Reload::ThreeWays => {
                let (rest, r) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I1]);
                let (one, a) = b.block(&carried);
                let (two, t) = b.block(&carried);
                let (three, h) = b.block(&carried);
                let join = b.block(&carried);
                b.cond_br(e, c1, (one, vec![k.exec, slot, second]), (rest, vec![k.exec, slot, second, c2]));
                b.cond_br(rest, r[3], (two, vec![r[0], r[1], r[2]]), (three, vec![r[0], r[1], r[2]]));
                for (block, q) in [(one, a), (two, t), (three, h)] {
                    b.br(block, join.0, vec![q[0], q[1], q[2]]);
                }
                join
            }
            Reload::Loop => b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I32]),
        };
        if looping {
            let zero = b.constant(e, Ty::I32, 0);
            b.br(e, last, vec![k.exec, slot, second, zero]);
        }
        let four = b.constant(last, Ty::I32, 4);
        let above = b.int(last, IntOp::Add, p[1], four);
        let low = b.load(last, Space::Scratch, MemSize::B32, p[1], p[0]);
        let high = b.load(last, Space::Scratch, MemSize::B32, above, p[0]);
        let pointer = b.core(last, Ty::I64, Op::Pack64(low, high));
        let lane = b.core(last, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, last, pointer, lane, 4);
        let s1 = store_at(&mut b, last, own, p[0]);
        let zero = b.constant(last, Ty::I32, 0);
        let word = byte_offset(&mut b, last, p[2], zero, 4);
        let s2 = store_at(&mut b, last, word, p[0]);
        if looping {
            if overwritten {
                spill(&mut b, last, p[0], p[2], p[1]);
            }
            let one = b.constant(last, Ty::I32, 1);
            let next = b.int(last, IntOp::Add, p[3], one);
            let three = b.constant(last, Ty::I32, 3);
            let again = b.cmp(last, IntPred::Ult, next, three);
            let (exit, _) = b.block(&[Ty::I1]);
            b.cond_br(last, again, (last, vec![p[0], p[1], p[2], next]), (exit, vec![p[0]]));
        }
        let h = Hazards::find(&b.program(), &env2());
        h.conflicts().contains(&pair(&h, s1, s2))
    }

    #[test]
    fn find_reports_a_pointer_reloaded_nine_blocks_later_from_a_slot_a_second_spill_overwrote() {
        assert!(spilled_before(Reload::Chain(9), true), "the second spill overwrites the slot");
    }

    #[test]
    fn find_keeps_a_pointer_reloaded_nine_blocks_later_off_a_buffer_spilled_elsewhere() {
        assert!(!spilled_before(Reload::Chain(9), false), "the slot holds the first buffer nine blocks later");
    }

    #[test]
    fn find_reports_a_pointer_reloaded_after_a_three_way_join_from_a_slot_a_second_spill_overwrote() {
        assert!(spilled_before(Reload::ThreeWays, true), "the second spill overwrites the slot");
    }

    #[test]
    fn find_keeps_a_pointer_reloaded_after_a_three_way_join_off_a_buffer_spilled_elsewhere() {
        assert!(!spilled_before(Reload::ThreeWays, false), "no path into the join writes the slot, which holds the first buffer");
    }

    #[test]
    fn find_reports_a_pointer_reloaded_in_a_loop_that_spills_the_second_buffer_into_the_slot() {
        assert!(spilled_before(Reload::Loop, true), "the second iteration reloads the second buffer the first spilled");
    }

    #[test]
    fn find_keeps_a_pointer_reloaded_in_a_loop_off_a_buffer_spilled_before_it() {
        assert!(!spilled_before(Reload::Loop, false), "the loop never writes the slot, which holds the first buffer");
    }

    fn loaded_node_far_up(offset: u64) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_bvh64_intersect_ray");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let u = uniform_word(&mut b, &k, e, 0, MemSize::U16);
        let sixteen = b.constant(e, Ty::I32, 16);
        let scaled = b.int(e, IntOp::Mul, u, sixteen);
        let base = b.constant(e, Ty::I32, (1 << 29) + 5);
        let node = b.int(e, IntOp::Add, scaled, base);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, node));
        let mut args = [b.constant(e, Ty::I32, 0); 14];
        args[0] = base_units(&mut b, e, buf);
        args[2] = wide;
        args[13] = k.exec;
        let r = b.here(e);
        b.target(e, op, Arguments::Fourteen(args), &[Ty::I32; 4]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.together.contains(&pair(&h, s, r))
    }

    #[test]
    fn find_orders_a_read_of_a_loaded_node_four_gigabytes_up_after_stores_into_one_it_may_name() {
        assert!(loaded_node_far_up((1 << 32) + 128 * 3 + 4), "u = 3 names node 2^29 + 53, 2^32 + 128 * 3 bytes in");
    }

    #[test]
    fn find_keeps_a_read_of_a_loaded_node_four_gigabytes_up_apart_from_stores_into_the_first_node() {
        assert!(!loaded_node_far_up(64), "every node the read may name starts 2^32 bytes or more past the store");
    }

    fn texel_read_with(offset: u64, coordinates: impl Fn(&mut Build, BlockId, &Kernel) -> (ValueId, ValueId)) -> bool {
        let (mut b, k) = Build::kernel();
        let op = rdna4(&mut b, "image_sample_lz");
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, buf, at);
        let s = store_at(&mut b, e, address, k.exec);
        let (x, y) = coordinates(&mut b, e, &k);
        let zero = b.constant(e, Ty::I32, 0);
        let mut args = [zero; 16];
        args[0] = base_units(&mut b, e, buf);
        args[1] = b.constant(e, Ty::I32, 3 << 30 | 5 << 17);
        args[2] = b.constant(e, Ty::I32, 15 << 14 | 3);
        args[3] = b.constant(e, Ty::I32, 4);
        args[13] = b.constant(e, Ty::I1, 1);
        args[14] = b.core(e, Ty::F32, Op::Convert(Cvt::UnsignedToFloatRte, Ty::F32, x));
        args[15] = b.core(e, Ty::F32, Op::Convert(Cvt::UnsignedToFloatRte, Ty::F32, y));
        let r = b.here(e);
        b.target(e, op, Arguments::Sixteen(args), &[Ty::I32]);
        let h = Hazards::find(&b.program(), &environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]));
        h.together.contains(&pair(&h, s, r))
    }

    fn texel_read_of_a_small_row(offset: u64) -> bool {
        texel_read_with(offset, |b, e, k| {
            let w = uniform_word(b, k, e, 0, MemSize::B32);
            let three = b.constant(e, Ty::I32, 3);
            (k.item, b.int(e, IntOp::And, w, three))
        })
    }

    #[test]
    fn find_orders_a_texel_read_of_a_loaded_row_after_stores_into_row_two() {
        assert!(texel_read_of_a_small_row(2 * 128 + 4), "lane 4 reads texel (4, 2) when w & 3 is 2");
    }

    #[test]
    fn find_keeps_a_texel_read_of_a_loaded_row_below_four_apart_from_stores_into_row_ten() {
        assert!(!texel_read_of_a_small_row(10 * 128 + 4), "the rows read are w & 3, below 4");
    }

    fn texel_read_of_two_rows(offset: u64) -> bool {
        texel_read_with(offset, |b, e, k| {
            let three = b.constant(e, Ty::I32, 3);
            let two = b.constant(e, Ty::I32, 2);
            (b.int(e, IntOp::And, k.item, three), b.int(e, IntOp::And, k.item, two))
        })
    }

    #[test]
    fn find_orders_a_texel_read_of_rows_zero_and_two_after_stores_into_row_two() {
        assert!(texel_read_of_two_rows(2 * 128 + 2), "lane 2 reads texel (2, 2)");
    }

    #[test]
    fn find_keeps_a_texel_read_of_rows_zero_and_two_apart_from_stores_into_row_one() {
        assert!(!texel_read_of_two_rows(128 + 10), "the lanes read columns 0 to 3 of rows 0 and 2 only");
    }
}
