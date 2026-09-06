//! Whole-function SSA construction over the mixed migration stream.
//!
//! Register words are SSA variables at the adapter boundary. Each CFG edge
//! passes their current definitions to destination block parameters. A legacy
//! operation consumes the previous state and defines only its written words;
//! typed ALU operations share the same value namespace. The backend lowers
//! boundary words/block arguments through explicit native SSA definitions, so
//! this adds no runtime register-file traffic or lane helper calls.
use super::*;
use crate::rdna_spmd::boundary::BoundaryIo;
use crate::rdna_spmd::ir::typed::cfg::*;
use crate::rdna_spmd::ir::{ScalarProgram, Terminator};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use super::state::{Word, Words, footprint, words as register_words};

pub(in crate::rdna_spmd) struct Alu {
    pub inputs: Vec<(Input, ValueId)>,
    pub core: Range<usize>,
    pub outputs: Vec<(Output, ValueId)>,
    pub scalar: bool,
    /// A native canonical pair can satisfy its explicit Pack64 and float view
    /// together. This is representation coalescing, not another SSA definition.
    pub pairs: Vec<(ValueId, ValueId)>,
}
pub(in crate::rdna_spmd) struct BlockPlan {
    pub instructions: Vec<Option<Alu>>,
    pub memory: BTreeMap<usize, super::memory::Plan>,
    pub wave: BTreeMap<usize,super::wave::YieldAction>,
    pub outgoing: Vec<ValueId>,
    pub yield_values: Option<super::wave::Plan>,
    pub condition: Option<Condition>,
    yielding: bool,
}
pub(in crate::rdna_spmd) struct Condition {
    pub input: Input,
    pub input_value: ValueId,
    pub core: Range<usize>,
}
pub(in crate::rdna_spmd) enum Control {
    Return,
    Jump(usize),
    Branch { cond: ValueId, taken: usize, fallthrough: usize },
    Yield { resume: usize },
}
pub(in crate::rdna_spmd) struct Function {
    pub registry: std::sync::Arc<DialectRegistry>,
    /// Architectural words assigned by this compiled function. Unassigned
    /// words remain in the scheduler's private state buffer across a call.
    pub written: crate::rdna_spmd::boundary::RegSet,
    pub ir: VerifiedFunc,
    pub blocks: BTreeMap<usize, BlockPlan>,
}
impl Function {
    pub fn new<'a>(
        registry: std::sync::Arc<DialectRegistry>,
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering<'a>>>,
        boundary: Option<&BTreeMap<usize, BoundaryIo>>,
    ) -> Self {
        let mut regs = BTreeSet::new();
        let mut written = crate::rdna_spmd::boundary::RegSet::default();
        for block in lowerings.values() {
            for lowering in block {
                let io = footprint(lowering);
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
                for r in io.writes.sgprs() { if r != 124 { written.add_sgpr(r); } }
                for r in io.writes.vgprs() { written.add_vgpr(r); }
                written.scc |= io.writes.scc;
            }
        }
        for block in program.blocks.values() {
            if let Terminator::Yield { action, .. } = &block.term {
                let io = action.io();
                for r in io.writes.sgprs() { if r != 124 { written.add_sgpr(r); } }
                for r in io.writes.vgprs() { written.add_vgpr(r); }
                written.scc |= io.writes.scc;
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
            }
        }
        if let Some(boundary) = boundary {
            for io in boundary.values() {
                regs.extend(register_words(&io.reads));
                regs.extend(register_words(&io.writes));
            }
        }
        let regs: Vec<_> = regs.into_iter().collect();
        let mut f = Func {
            entry: BlockId(program.entry_pc),
            blocks: BTreeMap::new(),
            types: vec![],
        };
        // Block parameters make all incoming definitions, including backedges,
        // explicit before any block body is lifted.
        for &pc in program.blocks.keys() {
            let mut params: Vec<_> = regs.iter().map(|_| (f.value(Ty::I32), Ty::I32)).collect();
            params.push((f.value(Ty::I1), Ty::I1));
            f.blocks.insert(
                BlockId(pc),
                Block {
                    params,
                    insts: vec![],
                    term: Term::Ret,
                },
            );
        }
        let mut plans = BTreeMap::new();
        let mut provenance = 0;
        for (&pc, source) in &program.blocks {
            let mut block = f.blocks.remove(&BlockId(pc)).unwrap();
            let mut words: Words = regs
                .iter()
                .copied()
                .zip(block.params.iter().map(|p| p.0))
                .collect();
            let mut views: BTreeMap<(Word, Ty, bool), ValueId> = BTreeMap::new();
            let mut scc = block.params.last().unwrap().0;
            let mut instructions = Vec::with_capacity(source.body.len());
            let mut memory = BTreeMap::new();
            let mut wave = BTreeMap::new();
            for lowering in &lowerings[&pc] {
                let io = footprint(lowering);
                let writes: Vec<_> = register_words(&io.writes).collect();
                let plan = match lowering {
                    Lowering::Wave(action) => {
                        for (dest, value) in action.lift(&mut f,&mut block,&mut words,&mut provenance).definitions {
                            if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                        }
                        invalidate(&mut views, &writes);
                        wave.insert(instructions.len(),action.clone());
                        None
                    }
                    Lowering::Memory(m) => {
                        let plan=m.lift(&mut f, &mut block, &mut words, &mut views, &mut provenance);
                        memory.insert(instructions.len(),plan);
                        invalidate(&mut views, &writes);
                        None
                    }
                    Lowering::TypedAlu {
                        inputs,
                        outputs,
                        scalar,
                        expr,
                    } => {
                        let mut operands = super::state::Operands::default();
                        let args: Vec<_> = inputs.iter().map(|input| operands.read(
                            input, *scalar, Some(scc), &mut f, &mut block, &words, &mut views,
                        )).collect();
                        let mut values = args;
                        let start = block.insts.len();
                        block.insts.extend(operands.core);
                        for inst in &expr.expr().insts {
                            match inst {
                                ExprInst::Core(ty, op) => {
                                    let op = op.map(|v| values[v.0]);
                                    let value = f.value(*ty);
                                    block.insts.push(Inst::Core { value, ty: *ty, op });
                                    values.push(value);
                                }
                                ExprInst::Target { op, args, outputs } => {
                                    let args = args.map(|v| values[v.0]);
                                    let outputs = outputs.iter().map(|&ty| (f.value(ty), ty)).collect::<Vec<_>>();
                                    values.extend(outputs.iter().map(|&(id, _)| id));
                                    let effect = registry.operation(*op).unwrap().effect;
                                    let id = if effect == crate::rdna_spmd::dialect::Effect::Pure { None }
                                        else { let id = provenance; provenance += 1; Some(id) };
                                    block.insts.push(Inst::Target { provenance: id, op: *op, args, outputs });
                                }
                            }
                        }
                        let end = block.insts.len();
                        let results: Vec<_> = outputs.iter().copied()
                            .zip(expr.expr().results.iter().map(|id| values[id.0])).collect();
                        for &(output, value) in &results {
                            if matches!(output, Output::Scc) { scc = value; }
                        }
                        // Predication is an explicit SSA select. The mask
                        // still comes from the coupled architectural-mask
                        // boundary; ordinary destination words are pure values.
                        let exec = if outputs.iter().any(|out| matches!(out, Output::Vgpr(..))) {
                            let value = f.value(Ty::I1);
                            block.insts.push(Inst::Boundary { inputs: vec![], outputs: vec![(value, Ty::I1)] });
                            Some(value)
                        } else { None };
                        let mut stored = Vec::new();
                        let mut word_defs = BTreeMap::new();
                        for &(output, result) in &results {
                            let ty = output.ty();
                            let value = match output {
                                Output::Vgpr(reg, _) => {
                                    let mut old = words[&Word::Vgpr(reg)];
                                    if ty.bits() == 64 {
                                        let value = f.value(Ty::I64);
                                        block.insts.push(Inst::Core { value, ty: Ty::I64,
                                            op: Op::Pack64(old, words[&Word::Vgpr(reg + 1)]) });
                                        old = value;
                                    }
                                    if matches!(ty, Ty::F32 | Ty::F64) {
                                        let value = f.value(ty);
                                        block.insts.push(Inst::Core { value, ty,
                                            op: Op::Convert(Cvt::Bitcast, ty, old) });
                                        old = value;
                                    }
                                    let value = f.value(ty);
                                    block.insts.push(Inst::Core { value, ty,
                                        op: Op::Select(exec.unwrap(), result, old) });
                                    value
                                }
                                Output::Scalar(reg, _) if Word::scalar(reg).is_some_and(|word| word.ordinary_span(ty)) => result,
                                Output::Scc => result,
                                _ => {
                                    // Mask observation and special word writes
                                    // migrate together with full-wave Ballot.
                                    let value = f.value(ty);
                                    let mut inputs = vec![result];
                                    inputs.extend(writes.iter().filter_map(|r| words.get(r).copied()));
                                    block.insts.push(Inst::Boundary { inputs, outputs: vec![(value, ty)] });
                                    value
                                }
                            };
                            stored.push((value, ty));
                            let register = match output {
                                Output::Vgpr(r, _) => Some((r, false)),
                                Output::Scalar(r, _) => Some((r, true)),
                                _ => None,
                            };
                            let Some((reg, scalar)) = register else { continue; };
                            let word = |offset| if scalar { Word::scalar(reg + offset) }
                                else { Some(Word::Vgpr(reg + offset)) };
                            let bits = if matches!(ty, Ty::F32 | Ty::F64) {
                                let int_ty = if ty == Ty::F32 { Ty::I32 } else { Ty::I64 };
                                let bits = f.value(int_ty);
                                block.insts.push(Inst::Core { value: bits, ty: int_ty, op: Op::Convert(Cvt::Bitcast, int_ty, value) });
                                bits
                            } else { value };
                            if ty.bits() == 32 {
                                if let Some(word) = word(0) { word_defs.insert(word, bits); }
                            } else {
                                for (offset, op) in [(0, Op::UnpackLo(bits)), (1, Op::UnpackHi(bits))] {
                                    if let Some(word) = word(offset) {
                                        let value = f.value(Ty::I32);
                                        block.insts.push(Inst::Core { value, ty: Ty::I32, op });
                                        word_defs.insert(word, value);
                                    }
                                }
                            }
                        }
                        for &r in &writes {
                            let value = if let Some(&value) = word_defs.get(&r) { value } else {
                                let value = f.value(Ty::I32);
                                block.insts.push(Inst::Boundary {
                                    inputs: results.iter().map(|x| x.1).chain(words.get(&r).copied()).collect(),
                                    outputs: vec![(value, Ty::I32)],
                                });
                                value
                            };
                            words.insert(r, value);
                        }
                        invalidate(&mut views, &writes);
                        for (&output, &(stored, _)) in outputs.iter().zip(&stored) {
                            match output {
                                Output::Vgpr(reg, ty) => { views.insert((Word::Vgpr(reg), ty, false), stored); }
                                Output::Scalar(reg, ty) => if let Some(reg) = Word::scalar(reg).filter(|r| r.ordinary_span(ty)) {
                                    views.insert((reg, ty, true), stored);
                                },
                                _ => {}
                            }
                        }
                        Some(Alu {
                            inputs: operands.bindings,
                            core: start..end,
                            outputs: results,
                            scalar: *scalar,
                            pairs: operands.pairs,
                        })
                    }
                    Lowering::Legacy(_) => {
                        // Remaining register/control and target-specific adapters
                        // retain their original order during state migration.
                        let mut deps: Vec<_> = register_words(&io.reads)
                            .filter_map(|r| words.get(&r).copied()).collect();
                        deps.extend(writes.iter().filter_map(|r| words.get(r).copied()));
                        let mut outputs = vec![];
                        for &r in &writes {
                            let v = f.value(Ty::I32);
                            words.insert(r, v);
                            outputs.push((v, Ty::I32));
                        }
                        // The remaining mask SALU defines SCC; ordinary moves
                        // and vector operations preserve its previous definition.
                        if io.writes.scc {
                            deps.push(scc);
                            scc = f.value(Ty::I1);
                            outputs.push((scc, Ty::I1));
                        }
                        block.insts.push(Inst::Boundary {
                            inputs: deps,
                            outputs,
                        });
                        invalidate(&mut views, &writes);
                        None
                    }
                };
                instructions.push(plan);
            }
            let yield_values = if let Terminator::Yield { ref action, .. } = source.term {
                let plan = action.lift(&mut f, &mut block, &mut words, &mut provenance);
                for &(dest, value) in &plan.definitions {
                    if matches!(dest, super::wave::Destination::Scc) { scc = value; }
                }
                Some(plan)
            } else { None };
            if let Terminator::Barrier { resume } = source.term {
                if let Some(io) = boundary.and_then(|map| map.get(&resume)) {
                    // A host-side wave operation runs between yield and resume.
                    // Its register outputs are new definitions on this edge,
                    // not the values written back before yielding.
                    let mut inputs: Vec<_> = register_words(&io.reads)
                        .chain(register_words(&io.writes))
                        .map(|r| words[&r])
                        .collect();
                    let mut outputs: Vec<_> = register_words(&io.writes)
                        .map(|r| {
                            let value = f.value(Ty::I32);
                            words.insert(r, value);
                            (value, Ty::I32)
                        })
                        .collect();
                    if io.reads.scc || io.writes.scc { inputs.push(scc); }
                    if io.writes.scc {
                        scc = f.value(Ty::I1);
                        outputs.push((scc, Ty::I1));
                    }
                    block.insts.push(Inst::Boundary { inputs, outputs });
                }
            }
            let edge = |pc| Edge {
                dst: BlockId(pc),
                args: regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect(),
            };
            let mut condition = None;
            block.term = match source.term {
                Terminator::Return => Term::Ret,
                Terminator::Jump(pc) | Terminator::Barrier { resume: pc } | Terminator::Yield { resume: pc, .. } => Term::Br(edge(pc)),
                Terminator::Branch {
                    cond, taken, fallthrough,
                } => {
                    use crate::rdna_spmd::ir::Cond;
                    let input_source = match cond {
                        Cond::Scc0 | Cond::Scc1 => InputSource::Scc,
                        Cond::ExecZ | Cond::ExecNz => InputSource::PacketMaskAny(126),
                        Cond::VccZ | Cond::VccNz => InputSource::PacketMaskAny(106),
                    };
                    let input_value = if matches!(input_source, InputSource::Scc) { scc } else {
                        let value = f.value(Ty::I1);
                        block.insts.push(Inst::Boundary { inputs: vec![], outputs: vec![(value, Ty::I1)] });
                        value
                    };
                    let start = block.insts.len();
                    let result = if matches!(cond, Cond::Scc0 | Cond::ExecZ | Cond::VccZ) {
                        let zero = f.value(Ty::I1);
                        block.insts.push(Inst::Core { value: zero, ty: Ty::I1, op: Op::Const(Ty::I1, 0) });
                        let result = f.value(Ty::I1);
                        block.insts.push(Inst::Core { value: result, ty: Ty::I1, op: Op::Cmp(IntPred::Eq, input_value, zero) });
                        result
                    } else { input_value };
                    condition = Some(Condition { input: Input { source: input_source, ty: Ty::I1 },
                        input_value, core: start..block.insts.len() });
                    Term::CondBr {
                        cond: result,
                        yes: edge(taken),
                        no: edge(fallthrough),
                    }
                }
            };
            f.blocks.insert(BlockId(pc), block);
            plans.insert(
                pc,
                BlockPlan {
                    instructions,
                    memory,
                    wave,
                    yield_values,
                    condition,
                    yielding: matches!(source.term, Terminator::Yield { .. } | Terminator::Barrier { .. }),
                    outgoing: regs.iter().map(|r| words[r]).chain(std::iter::once(scc)).collect(),
                },
            );
        }
        for (&pc, block) in &mut plans {
            if let Some(plan) = &mut block.yield_values {
                if let Some(end) = crate::rdna_spmd::passes::local_write_lane(&mut f,BlockId(pc),plan.core.end) {
                    plan.core.end = end;
                    plan.local = true;
                }
            }
        }
        let ir = f.verify_with(&registry).expect("invalid function SSA lift");
        let constants = crate::rdna_spmd::analysis::constants(&ir);
        for block in plans.values_mut() {
            if let Some(plan) = &mut block.yield_values {
                for (index, id) in plan.arguments.iter().enumerate() {
                    use crate::rdna_spmd::ir::typed::effect::{EffectOp, WaveOp};
                    if plan.layout.op == EffectOp::Wave(WaveOp::Wmma)
                        || plan.layout.op == EffectOp::Wave(WaveOp::WriteLane) && index == 2 { continue; }
                    if let Some(bits) = constants[id.0] {
                        plan.layout.arguments[index] = crate::rdna_spmd::yield_values::Argument::Constant(bits as u32);
                    }
                }
            }
        }
        Self {
            ir,
            registry,
            written,
            blocks: plans,
        }
    }
    pub fn min_private_bytes(&self) -> usize {
        self.blocks.values().flat_map(|b|b.memory.values())
            .filter_map(|p|p.memory.private_load_end()).max().unwrap_or(0) as usize
    }
    pub fn value_yields(&self) -> BTreeMap<usize, crate::rdna_spmd::yield_values::YieldValues> {
        self.blocks.iter().filter_map(|(&pc, block)| {
            let plan = block.yield_values.as_ref()?;
            if plan.local { return None; }
            let Term::Br(edge) = &self.ir.func().blocks[&BlockId(pc)].term else { panic!("yield lacks resume edge") };
            Some((edge.dst.0, plan.layout.clone()))
        }).collect()
    }
    /// Preserve the ABI's cooperative yield while taking targets from typed CFG.
    pub fn terminator(&self, pc: usize) -> Control {
        let term = &self.ir.func().blocks[&BlockId(pc)].term;
        // Coalesce each parallel edge copy with the register-boundary slot:
        // every destination parameter and its incoming definition occupy the
        // same word slot. Check the mapping rather than silently ignoring SSA
        // arguments; an IR rewrite requiring moves must lower those explicitly.
        for edge in term.edges() {
            assert_eq!(
                edge.args, self.blocks[&pc].outgoing,
                "unlowered SSA edge copies"
            );
        }
        match term {
            Term::Ret => Control::Return,
            Term::Br(e) if self.blocks[&pc].yielding => Control::Yield { resume: e.dst.0 },
            Term::Br(e) => Control::Jump(e.dst.0),
            Term::CondBr { cond, yes, no } => Control::Branch {
                cond: *cond,
                taken: yes.dst.0,
                fallthrough: no.dst.0,
            },
        }
    }
}
fn invalidate(views: &mut BTreeMap<(Word, Ty, bool), ValueId>, writes: &[Word]) {
    views.retain(|&(r, t, _), _| {
        !(0..t.bits().div_ceil(32)).filter_map(|k| r.offset(k)).any(|r| writes.contains(&r))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::ir::ScalarBlock;
    #[test]
    fn ordinary_alu_memory_and_wave_values_have_no_opaque_word_boundaries() {
        use crate::rdna_instructions::{VGLOBAL, VOP1};
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
            pc: 0, body: vec![
                InstFormat::VOP1(VOP1 { op: I::V_MOV_B32, src0: SourceOperand::IntegerConstant(17), vdst: 2 }),
                InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_LOAD_B64, saddr: 0, vaddr: 2,
                    vsrc: 0, vdst: 4, scope: 0, th: 0, ioffset: 0, sve: 0 }),
                InstFormat::VOP1(VOP1 { op: I::V_READFIRSTLANE_B32, src0: SourceOperand::VectorRegister(4), vdst: 6 }),
                InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_STORE_B32, saddr: 0, vaddr: 2,
                    vsrc: 5, vdst: 0, scope: 0, th: 0, ioffset: 0, sve: 0 }),
            ], term: Terminator::Return,
        })]) };
        let lowerings: Vec<_> = program.blocks[&0].body.iter().map(super::super::instruction).collect();
        let f = Function::new(std::sync::Arc::new(DialectRegistry::rdna4()), &program,
            &BTreeMap::from([(0, lowerings.iter().collect())]), None);
        let block = &f.ir.func().blocks[&BlockId(0)];
        for inst in &block.insts {
            if let Inst::Boundary { outputs, .. } = inst {
                assert!(outputs.iter().all(|(_, ty)| *ty == Ty::I1), "opaque ordinary word: {inst:?}");
            }
        }
        let load = f.blocks[&0].memory[&1].effects[0];
        let Inst::Effect { outputs, .. } = &block.insts[load] else { unreachable!() };
        let loaded_word = outputs[0].0;
        let selected = block.insts.iter().find_map(|inst| match inst {
            Inst::Core { value, op: Op::Select(_, loaded, _), .. } if *loaded == loaded_word => Some(*value),
            _ => None,
        }).expect("loaded value must explicitly preserve inactive lanes");
        assert!(block.insts.iter().any(|inst| matches!(inst,
            Inst::Effect { op: effect::EffectOp::Wave(effect::WaveOp::ReadFirstLane), inputs, .. }
            if inputs[0] == selected)));
    }

    #[test]
    fn host_written_resume_values_are_new_edge_definitions() {
        let program = ScalarProgram {
            entry_pc: 0,
            blocks: BTreeMap::from([
                (
                    0,
                    ScalarBlock {
                        pc: 0,
                        body: vec![],
                        term: Terminator::Barrier { resume: 1 },
                    },
                ),
                (
                    1,
                    ScalarBlock {
                        pc: 1,
                        body: vec![],
                        term: Terminator::Return,
                    },
                ),
            ]),
        };
        let mut io = BoundaryIo::default();
        io.reads.add_vgpr(7);
        io.writes.add_vgpr(23);
        io.reads.add_sgpr(7);
        io.writes.add_sgpr(23);
        io.writes.scc = true;
        let boundary = BTreeMap::from([(1, io)]);
        let f = Function::new(
            std::sync::Arc::new(DialectRegistry::rdna4()),
            &program,
            &BTreeMap::from([(0, vec![]), (1, vec![])]),
            Some(&boundary),
        );
        let block = &f.ir.func().blocks[&BlockId(0)];
        let Inst::Boundary { inputs, outputs } = &block.insts[0] else {
            panic!("missing host boundary")
        };
        assert_eq!(inputs.len(), 5);
        assert_eq!(outputs.len(), 3);
        let edge = block.term.edges()[0];
        assert_eq!(edge.args[0], block.params[0].0);
        assert_eq!(edge.args[1], outputs[0].0);
        assert_ne!(edge.args[1], block.params[1].0);
        assert_eq!(edge.args[2], block.params[2].0);
        assert_eq!(edge.args[3], outputs[1].0);
        assert_ne!(edge.args[3], block.params[3].0);
        assert_eq!(edge.args[4], outputs[2].0);
        assert_ne!(edge.args[4], block.params[4].0);
        assert_eq!(outputs[2].1, Ty::I1);
    }
}
