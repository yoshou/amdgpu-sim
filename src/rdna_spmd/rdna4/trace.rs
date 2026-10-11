use super::decode::{call_op, Decoded};
use super::lift;
use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};
use crate::rdna_spmd::dialect::DialectRegistry;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub parent: Option<usize>,
    pub site: usize,
    pub entry: usize,
    pub ret: usize,
}

pub struct Traced {
    pub contexts: Vec<Context>,
    pub calls: BTreeMap<(usize, usize), usize>,
    pub reached: BTreeSet<(usize, usize)>,
    pub missing: BTreeSet<usize>,
}

pub fn id(span: usize, context: usize, pc: usize) -> usize {
    context * span + pc
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value {
    Unknown,
    Known(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    sgprs: [Value; 128],
    scc: Value,
    lanes: BTreeMap<(u8, u8), Value>,
}

impl State {
    fn entry() -> State {
        State {
            sgprs: [Value::Unknown; 128],
            scc: Value::Unknown,
            lanes: BTreeMap::new(),
        }
    }

    fn merge(&self, other: &State) -> State {
        let mut merged = self.clone();
        for (a, b) in merged.sgprs.iter_mut().zip(&other.sgprs) {
            if *a != *b {
                *a = Value::Unknown;
            }
        }
        if merged.scc != other.scc {
            merged.scc = Value::Unknown;
        }
        merged.lanes.retain(|slot, value| other.lanes.get(slot) == Some(value));
        merged
    }

    fn operand(&self, source: &SourceOperand) -> Value {
        match *source {
            SourceOperand::ScalarRegister(r) => self.sgprs[r as usize & 127],
            SourceOperand::IntegerConstant(k) => Value::Known(k as u32),
            SourceOperand::LiteralConstant(k) => Value::Known(k),
            _ => Value::Unknown,
        }
    }

    fn pair(&self, source: &SourceOperand) -> Option<u64> {
        let SourceOperand::ScalarRegister(r) = *source else {
            return None;
        };
        match (self.sgprs[r as usize & 127], self.sgprs[(r as usize + 1) & 127]) {
            (Value::Known(low), Value::Known(high)) => Some((high as u64) << 32 | low as u64),
            _ => None,
        }
    }

    fn write(&mut self, r: u8, value: Value) {
        self.sgprs[r as usize & 127] = value;
    }

    fn write_pair(&mut self, r: u8, value: u64) {
        self.write(r, Value::Known(value as u32));
        self.write(r.wrapping_add(1), Value::Known((value >> 32) as u32));
    }

    fn forget_vgprs(&mut self, inst: &InstFormat, registry: &DialectRegistry, lanes: u32) {
        let inner = match inst {
            InstFormat::DPP(d) => &*d.inner,
            other => other,
        };
        match lift::instruction_with_registry(inner, registry, lanes) {
            Ok(lowering) => {
                let io = lift::footprint(&lowering, lanes);
                let written: Vec<u32> = io.writes.vgprs().collect();
                self.lanes.retain(|&(v, _), _| !written.contains(&(v as u32)));
            }
            Err(_) => self.lanes.clear(),
        }
    }

    fn step(&mut self, next_pc: usize, inst: &InstFormat, registry: &DialectRegistry, lanes: u32) {
        let previous = self.clone();
        let dests = lift::scalar_dests(inst);
        for r in 0..128u8 {
            if dests >> r & 1 != 0 {
                self.write(r, Value::Unknown);
            }
        }
        if matches!(inst, InstFormat::SOP1(_) | InstFormat::SOP2(_) | InstFormat::SOPC(_) | InstFormat::SOPK(_)) {
            self.scc = Value::Unknown;
        }
        let scalar_only = matches!(
            inst,
            InstFormat::SOP1(_) | InstFormat::SOP2(_) | InstFormat::SOPC(_) | InstFormat::SOPK(_) | InstFormat::SOPP(_) | InstFormat::SMEM(_)
        );
        let writelane = matches!(inst, InstFormat::VOP3(i) if matches!(i.op, I::V_WRITELANE_B32));
        if !scalar_only && !writelane {
            self.forget_vgprs(inst, registry, lanes);
        }
        match inst {
            InstFormat::SOP1(i) => match i.op {
                I::S_GETPC_B64 | I::S_SWAPPC_B64 => self.write_pair(i.sdst, next_pc as u64),
                I::S_MOV_B32 => {
                    let value = previous.operand(&i.ssrc0);
                    self.write(i.sdst, value);
                    self.scc = previous.scc;
                }
                I::S_MOV_B64 => {
                    let value = match i.ssrc0 {
                        SourceOperand::ScalarRegister(_) => previous.pair(&i.ssrc0),
                        SourceOperand::IntegerConstant(k) => Some(k),
                        SourceOperand::LiteralConstant(k) => Some(k as i32 as i64 as u64),
                        _ => None,
                    };
                    match value {
                        Some(value) => self.write_pair(i.sdst, value),
                        None => {
                            self.write(i.sdst, Value::Unknown);
                            self.write(i.sdst.wrapping_add(1), Value::Unknown);
                        }
                    }
                    self.scc = previous.scc;
                }
                I::S_SEXT_I32_I16 => {
                    let value = match previous.operand(&i.ssrc0) {
                        Value::Known(v) => Value::Known(v as u16 as i16 as i32 as u32),
                        Value::Unknown => Value::Unknown,
                    };
                    self.write(i.sdst, value);
                    self.scc = previous.scc;
                }
                _ => {}
            },
            InstFormat::SOP2(i) => match i.op {
                I::S_ADD_CO_U32 | I::S_ADD_CO_CI_U32 | I::S_SUB_CO_U32 | I::S_SUB_CO_CI_U32 => {
                    let carried = matches!(i.op, I::S_ADD_CO_CI_U32 | I::S_SUB_CO_CI_U32);
                    let carry = if carried { previous.scc } else { Value::Known(0) };
                    let (result, out) = match (previous.operand(&i.ssrc0), previous.operand(&i.ssrc1), carry) {
                        (Value::Known(a), Value::Known(b), Value::Known(c)) => {
                            let wide = if matches!(i.op, I::S_ADD_CO_U32 | I::S_ADD_CO_CI_U32) {
                                a as u64 + b as u64 + c as u64
                            } else {
                                (a as u64).wrapping_sub(b as u64).wrapping_sub(c as u64)
                            };
                            let out = if matches!(i.op, I::S_ADD_CO_U32 | I::S_ADD_CO_CI_U32) {
                                wide >> 32 != 0
                            } else {
                                (b as u64 + c as u64) > a as u64
                            };
                            (Value::Known(wide as u32), Value::Known(out as u32))
                        }
                        _ => (Value::Unknown, Value::Unknown),
                    };
                    self.write(i.sdst, result);
                    self.scc = out;
                }
                _ => {}
            },
            InstFormat::VOP3(i) if matches!(i.op, I::V_WRITELANE_B32) => {
                let value = previous.operand(&i.src0);
                match previous.operand(&i.src1) {
                    Value::Known(lane) => {
                        self.lanes.insert((i.vdst, (lane & 63) as u8), value);
                    }
                    Value::Unknown => self.lanes.retain(|&(v, _), _| v != i.vdst),
                }
            }
            InstFormat::VOP3(i) if matches!(i.op, I::V_READLANE_B32) => {
                let read = match (&i.src0, previous.operand(&i.src1)) {
                    (SourceOperand::VectorRegister(v), Value::Known(lane)) => {
                        previous.lanes.get(&(*v, (lane & 63) as u8)).copied()
                    }
                    _ => None,
                };
                self.write(i.vdst, read.unwrap_or(Value::Unknown));
            }
            _ => {}
        }
    }
}

impl Traced {
    fn chain(&self, mut context: usize) -> impl Iterator<Item = &Context> {
        std::iter::from_fn(move || {
            let c = self.contexts.get(context)?;
            context = c.parent.unwrap_or(usize::MAX);
            Some(c)
        })
    }
}

pub fn trace(decoded: &Decoded, registry: &DialectRegistry, lanes: u32) -> Result<Traced, String> {
    let mut traced = Traced {
        contexts: vec![Context {
            parent: None,
            site: usize::MAX,
            entry: decoded.entry_pc,
            ret: usize::MAX,
        }],
        calls: BTreeMap::new(),
        reached: BTreeSet::new(),
        missing: BTreeSet::new(),
    };
    let mut at_entry: BTreeMap<(usize, usize), State> = BTreeMap::new();
    at_entry.insert((0, decoded.entry_pc), State::entry());
    let mut work = vec![(0usize, decoded.entry_pc)];
    while let Some(node) = work.pop() {
        let (context, start) = node;
        let Some(block) = decoded.blocks.get(&start) else {
            traced.missing.insert(start);
            continue;
        };
        traced.reached.insert(node);
        let mut state = at_entry[&node].clone();
        let last = block.insts.len() - 1;
        for (k, inst) in block.insts.iter().enumerate() {
            if k == last && call_op(inst).is_some() {
                break;
            }
            let next_pc = block.pcs.get(k + 1).copied().unwrap_or(block.end);
            state.step(next_pc, inst, registry, lanes);
        }
        let pc = block.pcs[last];
        let mut flow = |to: (usize, usize), state: State, work: &mut Vec<(usize, usize)>| {
            let merged = match at_entry.get(&to) {
                Some(old) => old.merge(&state),
                None => state,
            };
            if at_entry.get(&to) != Some(&merged) {
                at_entry.insert(to, merged);
                work.push(to);
            }
        };
        let InstFormat::SOP1(call) = &block.insts[last] else {
            for &next in &block.next_pcs {
                flow((context, next), state.clone(), &mut work);
            }
            continue;
        };
        match call_op(&block.insts[last]) {
            Some(I::S_SWAPPC_B64) => {
                let target = state
                    .pair(&call.ssrc0)
                    .ok_or_else(|| format!("{pc:#x}: a call to an address the program does not determine"))?
                    as usize;
                if traced.chain(context).any(|c| c.entry == target) {
                    return Err(format!("{pc:#x}: a recursive call to {target:#x}"));
                }
                let callee = match traced
                    .contexts
                    .iter()
                    .position(|c| c.parent == Some(context) && c.site == pc)
                {
                    Some(callee) if traced.contexts[callee].entry == target => callee,
                    Some(_) => return Err(format!("{pc:#x}: a call site whose target varies")),
                    None => {
                        traced.contexts.push(Context {
                            parent: Some(context),
                            site: pc,
                            entry: target,
                            ret: block.end,
                        });
                        traced.contexts.len() - 1
                    }
                };
                traced.calls.insert(node, callee);
                state.write_pair(call.sdst, block.end as u64);
                flow((callee, target), state, &mut work);
            }
            Some(_) => {
                let frame = traced.contexts[context];
                let parent = frame
                    .parent
                    .ok_or_else(|| format!("{pc:#x}: a kernel that jumps through a register"))?;
                if state.pair(&call.ssrc0) != Some(frame.ret as u64) {
                    return Err(format!(
                        "{pc:#x}: a function that may return somewhere other than {:#x}",
                        frame.ret
                    ));
                }
                flow((parent, frame.ret), state, &mut work);
            }
            None => {
                for &next in &block.next_pcs {
                    flow((context, next), state.clone(), &mut work);
                }
            }
        }
    }
    Ok(traced)
}

pub fn explore(entry_pc: usize, memory: &[u8], registry: &DialectRegistry, lanes: u32) -> Result<(Decoded, Traced), String> {
    let mut entries = vec![entry_pc];
    loop {
        let decoded = super::decode::program(&entries, memory)?;
        let traced = trace(&decoded, registry, lanes)?;
        if traced.missing.is_empty() {
            return Ok((decoded, traced));
        }
        entries.extend(traced.missing.iter().copied());
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode::{body, DecodedBlock};
    use super::*;
    use crate::rdna_instructions::{SOP1, SOP2, SOPP, VOP3};

    fn sop1(op: I, sdst: u8, ssrc0: SourceOperand) -> InstFormat {
        InstFormat::SOP1(SOP1 { ssrc0, op, sdst })
    }

    fn sop2(op: I, sdst: u8, ssrc0: SourceOperand, ssrc1: SourceOperand) -> InstFormat {
        InstFormat::SOP2(SOP2 { ssrc0, ssrc1, sdst, op })
    }

    fn sopp(op: I) -> InstFormat {
        InstFormat::SOPP(SOPP { simm16: 0, op })
    }

    fn lane_move(op: I, vdst: u8, src0: SourceOperand, lane: u64) -> InstFormat {
        InstFormat::VOP3(VOP3 {
            vdst,
            abs: 0,
            opsel: 0,
            cm: 0,
            op,
            src0,
            src1: SourceOperand::IntegerConstant(lane),
            src2: SourceOperand::IntegerConstant(0),
            omod: 0,
            neg: 0,
        })
    }

    fn s(r: u8) -> SourceOperand {
        SourceOperand::ScalarRegister(r)
    }

    fn lit(k: u32) -> SourceOperand {
        SourceOperand::LiteralConstant(k)
    }

    fn block(start: usize, insts: Vec<InstFormat>, next_pcs: Vec<usize>) -> (usize, DecodedBlock) {
        let pcs: Vec<usize> = (0..insts.len()).map(|k| start + 4 * k).collect();
        let end = start + 4 * insts.len();
        (start, DecodedBlock { insts, pcs, end, next_pcs })
    }

    fn call_sequence(at: usize, target: usize, pair: u8) -> Vec<InstFormat> {
        let next = at + 4;
        let low = (target as u64).wrapping_sub(next as u64) as u32;
        let high = ((target as u64).wrapping_sub(next as u64) >> 32) as u32;
        vec![
            sop1(I::S_GETPC_B64, pair, s(0)),
            sop1(I::S_SEXT_I32_I16, pair + 1, s(pair + 1)),
            sop2(I::S_ADD_CO_U32, pair, s(pair), lit(low)),
            sop2(I::S_ADD_CO_CI_U32, pair + 1, s(pair + 1), lit(high)),
            sop1(I::S_SWAPPC_B64, 30, s(pair)),
        ]
    }

    fn saving_callee(start: usize, middle: Vec<InstFormat>) -> (usize, DecodedBlock) {
        let mut insts = vec![
            lane_move(I::V_WRITELANE_B32, 4, s(30), 0),
            lane_move(I::V_WRITELANE_B32, 4, s(31), 1),
        ];
        insts.extend(middle);
        insts.extend([
            lane_move(I::V_READLANE_B32, 30, SourceOperand::VectorRegister(4), 0),
            lane_move(I::V_READLANE_B32, 31, SourceOperand::VectorRegister(4), 1),
            sop1(I::S_SETPC_B64, 0, s(30)),
        ]);
        block(start, insts, vec![])
    }

    fn traced(blocks: Vec<(usize, DecodedBlock)>) -> Result<Traced, String> {
        let decoded = Decoded {
            entry_pc: 0x100,
            blocks: blocks.into_iter().collect(),
        };
        let registry = super::super::dialect().registry;
        trace(&decoded, &registry, 32)
    }

    #[test]
    fn a_call_through_a_computed_address_opens_a_context_that_returns_to_the_call_site() {
        let kernel = block(0x100, call_sequence(0x100, 0x200, 0), vec![0x114]);
        let after = block(0x114, vec![sopp(I::S_ENDPGM)], vec![]);
        let callee = saving_callee(0x200, vec![sop1(I::S_MOV_B32, 0, lit(7))]);
        let t = traced(vec![kernel, after, callee]).unwrap();
        assert_eq!(t.contexts.len(), 2);
        assert_eq!(t.contexts[1], Context { parent: Some(0), site: 0x110, entry: 0x200, ret: 0x114 });
        assert_eq!(t.calls.get(&(0, 0x100)), Some(&1));
        assert!(t.reached.contains(&(1, 0x200)) && t.reached.contains(&(0, 0x114)));
        assert!(t.missing.is_empty());
    }

    #[test]
    fn a_callee_that_is_not_decoded_yet_is_reported_as_missing() {
        let kernel = block(0x100, call_sequence(0x100, 0x300, 0), vec![0x114]);
        let after = block(0x114, vec![sopp(I::S_ENDPGM)], vec![]);
        let t = traced(vec![kernel, after]).unwrap();
        assert_eq!(t.missing.into_iter().collect::<Vec<_>>(), vec![0x300]);
    }

    #[test]
    fn a_recursive_call_is_refused() {
        let kernel = block(0x100, call_sequence(0x100, 0x200, 0), vec![0x114]);
        let after = block(0x114, vec![sopp(I::S_ENDPGM)], vec![]);
        let callee = block(0x200, call_sequence(0x200, 0x200, 2), vec![0x214]);
        let error = traced(vec![kernel, after, callee]).err().unwrap();
        assert_eq!(error, "0x210: a recursive call to 0x200");
    }

    #[test]
    fn a_return_through_a_clobbered_address_is_refused() {
        let kernel = block(0x100, call_sequence(0x100, 0x200, 0), vec![0x114]);
        let after = block(0x114, vec![sopp(I::S_ENDPGM)], vec![]);
        let callee = block(0x200, vec![sop1(I::S_MOV_B32, 30, lit(7)), sop1(I::S_SETPC_B64, 0, s(30))], vec![]);
        let error = traced(vec![kernel, after, callee]).err().unwrap();
        assert_eq!(error, "0x204: a function that may return somewhere other than 0x114");
    }

    #[test]
    fn a_call_to_an_undetermined_address_is_refused() {
        let kernel = block(0x100, vec![sop1(I::S_SWAPPC_B64, 30, s(8))], vec![0x104]);
        let error = traced(vec![kernel]).err().unwrap();
        assert_eq!(error, "0x100: a call to an address the program does not determine");
    }

    #[test]
    fn a_saved_return_address_survives_vector_writes_to_other_registers_only() {
        let moved = |vdst: u8| {
            InstFormat::VOP1(crate::rdna_instructions::VOP1 {
                src0: SourceOperand::VectorRegister(1),
                op: I::V_MOV_B32,
                vdst,
            })
        };
        let run = |vdst: u8| {
            let kernel = block(0x100, call_sequence(0x100, 0x200, 0), vec![0x114]);
            let after = block(0x114, vec![sopp(I::S_ENDPGM)], vec![]);
            traced(vec![kernel, after, saving_callee(0x200, vec![moved(vdst)])])
        };
        assert_eq!(run(5).unwrap().contexts.len(), 2, "v5 holds nothing the return needs");
        assert_eq!(run(4).err().unwrap(), "0x214: a function that may return somewhere other than 0x114");
    }

    #[test]
    fn the_block_body_replaces_the_program_counter_read_with_its_value() {
        let insts = vec![
            sop1(I::S_GETPC_B64, 2, s(0)),
            InstFormat::SOPP(SOPP { simm16: 0, op: I::S_DELAY_ALU }),
            sop2(I::S_ADD_CO_U32, 2, s(2), lit(8)),
        ];
        let lowered = body(&insts, &[0x100, 0x104, 0x108], 0x10c);
        assert_eq!(lowered.len(), 2);
        assert!(matches!(
            &lowered[0],
            InstFormat::SOP1(SOP1 { op: I::S_MOV_B64, sdst: 2, ssrc0: SourceOperand::IntegerConstant(0x104) })
        ));
        assert!(matches!(&lowered[1], InstFormat::SOP2(SOP2 { op: I::S_ADD_CO_U32, .. })));
    }
}
