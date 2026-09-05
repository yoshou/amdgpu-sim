//! Whole-function SSA construction over the mixed migration stream.
//!
//! Register words are SSA variables at the adapter boundary. Each CFG edge
//! passes their current definitions to destination block parameters. A legacy
//! operation consumes the previous state and defines only its written words;
//! typed ALU operations share the same value namespace. The backend lowers
//! boundary words/block arguments through its existing promotable allocas, so
//! this adds no runtime register-file traffic or lane helper calls.
use super::*;
use crate::rdna_spmd::boundary::BoundaryIo;
use crate::rdna_spmd::ir::typed::cfg::*;
use crate::rdna_spmd::ir::{ScalarProgram, Terminator};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

pub(in crate::rdna_spmd) struct Alu {
    pub inputs: Vec<(Input, ValueId)>,
    pub core: Range<usize>,
    pub result: ValueId,
    pub output: Output,
}
pub(in crate::rdna_spmd) struct BlockPlan {
    pub instructions: Vec<Option<Alu>>,
    pub memory: BTreeMap<usize, super::memory::Plan>,
    pub wave: BTreeMap<usize,super::wave::YieldAction>,
    pub outgoing: Vec<ValueId>,
    pub yield_action: Option<(u64, super::wave::YieldAction)>,
}
pub(in crate::rdna_spmd) struct Function {
    pub ir: VerifiedFunc,
    pub blocks: BTreeMap<usize, BlockPlan>,
}
impl Function {
    pub fn new<'a>(
        program: &ScalarProgram,
        lowerings: &BTreeMap<usize, Vec<&Lowering<'a>>>,
        boundary: Option<&BTreeMap<usize, BoundaryIo>>,
    ) -> Self {
        let mut regs = BTreeSet::new();
        for block in program.blocks.values() {
            for inst in &block.body {
                regs.extend(
                    crate::rdna_spmd::vec_live::vgpr_reads(inst)
                        .into_iter()
                        .filter(|&r| r < 256),
                );
                regs.extend(crate::rdna_spmd::freshness::vgpr_writes(inst));
            }
        }
        for block in program.blocks.values() {
            if let Terminator::Yield { action, .. } = &block.term {
                let io = action.io();regs.extend(io.reads.vgprs());regs.extend(io.writes.vgprs());
            }
        }
        if let Some(boundary) = boundary {
            for io in boundary.values() {
                regs.extend(io.reads.vgprs());
                regs.extend(io.writes.vgprs());
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
            let params = regs.iter().map(|_| (f.value(Ty::I32), Ty::I32)).collect();
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
            let mut words: BTreeMap<u32, ValueId> = regs
                .iter()
                .copied()
                .zip(block.params.iter().map(|p| p.0))
                .collect();
            let mut views: BTreeMap<(u32, Ty), ValueId> = BTreeMap::new();
            let mut instructions = Vec::with_capacity(source.body.len());
            let mut memory = BTreeMap::new();
            let mut wave = BTreeMap::new();
            for (inst, lowering) in source.body.iter().zip(&lowerings[&pc]) {
                let plan = match lowering {
                    Lowering::Wave(action) => {
                        action.lift(&mut f,&mut block,&mut words,&mut provenance);
                        invalidate(&mut views,&action.io().writes.vgprs().collect::<Vec<_>>());
                        wave.insert(instructions.len(),action.clone());
                        None
                    }
                    Lowering::Memory(m) => {
                        let plan=m.lift(&mut f, &mut block, &mut words, &mut provenance);
                        memory.insert(instructions.len(),plan);
                        invalidate(&mut views, &m.writes());
                        None
                    }
                    Lowering::TypedAlu {
                        inputs,
                        output,
                        expr,
                    } => {
                        let mut args = Vec::new();
                        for input in inputs {
                            let key = match input.source {
                                SourceOperand::VectorRegister(r) => Some((r as u32, input.ty)),
                                _ => None,
                            };
                            let known = key.and_then(|k| views.get(&k).copied());
                            let value = if let Some(value) = known {
                                value
                            } else {
                                let deps = key
                                    .map(|(reg, ty)| {
                                        (0..ty.bits().div_ceil(32))
                                            .filter_map(|k| words.get(&(reg + k)).copied())
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                let value = f.value(input.ty);
                                block.insts.push(Inst::Boundary {
                                    inputs: deps,
                                    outputs: vec![(value, input.ty)],
                                });
                                if let Some(k) = key {
                                    views.insert(k, value);
                                }
                                value
                            };
                            args.push((input.clone(), value));
                        }
                        let mut values: Vec<_> = args.iter().map(|p| p.1).collect();
                        let start = block.insts.len();
                        for &(ty, op) in &expr.expr().insts {
                            let op = op.map(|v| values[v.0]);
                            let value = f.value(ty);
                            block.insts.push(Inst::Core { value, ty, op });
                            values.push(value);
                        }
                        let end = block.insts.len();
                        let result = values[expr.expr().result.0];
                        let stored = f.value(output.ty());
                        let writes = crate::rdna_spmd::freshness::vgpr_writes(inst);
                        let mut deps = vec![result];
                        deps.extend(writes.iter().filter_map(|r| words.get(r).copied()));
                        let mut defs = vec![(stored, output.ty())];
                        for &r in &writes {
                            let v = f.value(Ty::I32);
                            words.insert(r, v);
                            defs.push((v, Ty::I32));
                        }
                        block.insts.push(Inst::Boundary {
                            inputs: deps,
                            outputs: defs,
                        });
                        invalidate(&mut views, &writes);
                        if let Output::Vgpr(reg, ty) = *output {
                            views.insert((reg, ty), stored);
                        }
                        Some(Alu {
                            inputs: args,
                            core: start..end,
                            result,
                            output: *output,
                        })
                    }
                    Lowering::Legacy(_) => {
                        let writes = crate::rdna_spmd::freshness::vgpr_writes(inst);
                        // Remaining register/control and target-specific adapters
                        // retain their original order during state migration.
                        let mut deps: Vec<_> = crate::rdna_spmd::vec_live::vgpr_reads(inst)
                            .iter()
                            .filter_map(|r| words.get(r).copied())
                            .collect();
                        deps.extend(writes.iter().filter_map(|r| words.get(r).copied()));
                        let mut outputs = vec![];
                        for &r in &writes {
                            let v = f.value(Ty::I32);
                            words.insert(r, v);
                            outputs.push((v, Ty::I32));
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
            let yield_action = if let Terminator::Yield { ref action, .. } = source.term {
                Some((provenance,action.as_ref().clone()))
            } else {None};
            if let Terminator::Yield { ref action, .. } = source.term {
                action.lift(&mut f, &mut block, &mut words, &mut provenance);
            }
            if let Terminator::Barrier { resume } = source.term {
                if let Some(io) = boundary.and_then(|map| map.get(&resume)) {
                    // A host-side wave operation runs between yield and resume.
                    // Its register outputs are new definitions on this edge,
                    // not the values written back before yielding.
                    let inputs = io
                        .reads
                        .vgprs()
                        .chain(io.writes.vgprs())
                        .map(|r| words[&r])
                        .collect();
                    let outputs = io
                        .writes
                        .vgprs()
                        .map(|r| {
                            let value = f.value(Ty::I32);
                            words.insert(r, value);
                            (value, Ty::I32)
                        })
                        .collect();
                    block.insts.push(Inst::Boundary { inputs, outputs });
                }
            }
            let edge = |pc| Edge {
                dst: BlockId(pc),
                args: regs.iter().map(|r| words[r]).collect(),
            };
            block.term = match source.term {
                Terminator::Return => Term::Ret,
                Terminator::Jump(pc) | Terminator::Barrier { resume: pc } | Terminator::Yield { resume: pc, .. } => Term::Br(edge(pc)),
                Terminator::Branch {
                    taken, fallthrough, ..
                } => {
                    // Lane-mask reduction and scalar condition conventions
                    // remain an adapter boundary until wave-op lifting.
                    let cond = f.value(Ty::I1);
                    block.insts.push(Inst::Boundary {
                        inputs: vec![],
                        outputs: vec![(cond, Ty::I1)],
                    });
                    Term::CondBr {
                        cond,
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
                    yield_action,
                    outgoing: regs.iter().map(|r| words[r]).collect(),
                },
            );
        }
        Self {
            ir: f.verify().expect("invalid function SSA lift"),
            blocks: plans,
        }
    }
    pub fn min_private_bytes(&self) -> usize {
        self.blocks.values().flat_map(|b|b.memory.values())
            .filter_map(|p|p.memory.private_load_end()).max().unwrap_or(0) as usize
    }
    /// Bind the verified SSA effect to the scheduler's register adapter. The
    /// opcode and resume edge come from the same IR consumed by codegen.
    pub fn yields(&self) -> BTreeMap<usize, super::wave::YieldAction> {
        self.blocks.iter().filter_map(|(&pc,plan)| {
            let (id,binding)=plan.yield_action.as_ref()?;
            let block=&self.ir.func().blocks[&BlockId(pc)];
            let op=block.insts.iter().find_map(|inst|match inst {
                Inst::Effect{provenance,op,..} if provenance==id=>Some(*op),_=>None,
            }).expect("yield binding lacks verified effect");
            let Term::Br(edge)=&block.term else {panic!("yield lacks resume edge")};
            Some((edge.dst.0,super::wave::YieldAction::new(op,binding.inputs.clone(),binding.outputs.clone())))
        }).collect()
    }
    /// Preserve the ABI's cooperative yield while taking targets from typed CFG.
    pub fn terminator(&self, pc: usize, source: &Terminator) -> Terminator {
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
        match (term, source) {
            (Term::Ret, Terminator::Return) => Terminator::Return,
            (Term::Br(e), Terminator::Yield { action, .. }) => Terminator::Yield { resume: e.dst.0, action: action.clone() },
            (Term::Br(e), Terminator::Barrier { .. }) => Terminator::Barrier { resume: e.dst.0 },
            (Term::Br(e), Terminator::Jump(_)) => Terminator::Jump(e.dst.0),
            (Term::CondBr { yes, no, .. }, Terminator::Branch { cond, .. }) => Terminator::Branch {
                cond: *cond,
                taken: yes.dst.0,
                fallthrough: no.dst.0,
            },
            _ => unreachable!("mismatched migration CFG"),
        }
    }
}
fn invalidate(views: &mut BTreeMap<(u32, Ty), ValueId>, writes: &[u32]) {
    views.retain(|&(r, t), _| {
        !writes
            .iter()
            .any(|&w| w >= r && w < r + t.bits().div_ceil(32))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::ir::ScalarBlock;
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
        let boundary = BTreeMap::from([(1, io)]);
        let f = Function::new(
            &program,
            &BTreeMap::from([(0, vec![]), (1, vec![])]),
            Some(&boundary),
        );
        let block = &f.ir.func().blocks[&BlockId(0)];
        let Inst::Boundary { inputs, outputs } = &block.insts[0] else {
            panic!("missing host boundary")
        };
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);
        let edge = block.term.edges()[0];
        assert_eq!(edge.args[0], block.params[0].0);
        assert_eq!(edge.args[1], outputs[0].0);
        assert_ne!(edge.args[1], block.params[1].0);
    }
}
