use crate::instructions::I;
use crate::rdna4_decoder::{decode_rdna4, InstStream};
use crate::rdna_instructions::InstFormat;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cond {
    ExecZ,
    ExecNz,
    VccZ,
    VccNz,
    Scc0,
    Scc1,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Terminator {

    Return,

    Jump(usize),

    Branch {
        cond: Cond,
        taken: usize,
        fallthrough: usize,
    },

    Barrier { resume: usize },

    Yield {
        resume: usize,
        action: Box<super::lift::YieldAction>,
    },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScalarBlock {
    pub pc: usize,

    pub body: Vec<InstFormat>,
    pub term: Terminator,
}

#[derive(Debug, Clone)]
pub struct ScalarProgram {
    pub entry_pc: usize,
    pub blocks: BTreeMap<usize, ScalarBlock>,
}

fn is_noop(inst: &InstFormat) -> bool {
    match inst {
        InstFormat::SOPP(i) => matches!(
            i.op,
            I::S_DELAY_ALU
                | I::S_WAIT_ALU
                | I::S_WAIT_LOADCNT
                | I::S_WAIT_KMCNT
                | I::S_WAIT_DSCNT
                | I::S_WAIT_STORECNT
                | I::S_WAIT_STORECNT_DSCNT
                | I::S_WAIT_LOADCNT_DSCNT
                | I::S_WAIT_SAMPLECNT
                | I::S_WAIT_BVHCNT
                | I::S_WAIT_EXPCNT
                | I::S_WAIT_EVENT
                | I::S_WAIT_IDLE
                | I::S_WAITCNT
                | I::S_NOP
                | I::S_CLAUSE
                | I::S_SENDMSG
        ),
        _ => false,
    }
}

fn lower_terminator(last: &InstFormat, next_pcs: &[usize]) -> Terminator {
    if let InstFormat::SOPP(i) = last {
        match i.op {
            I::S_ENDPGM => return Terminator::Return,
            I::S_BRANCH => return Terminator::Jump(next_pcs[0]),
            I::S_CBRANCH_EXECZ => {
                return Terminator::Branch {
                    cond: Cond::ExecZ,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_EXECNZ => {
                return Terminator::Branch {
                    cond: Cond::ExecNz,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_VCCZ => {
                return Terminator::Branch {
                    cond: Cond::VccZ,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_VCCNZ => {
                return Terminator::Branch {
                    cond: Cond::VccNz,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_SCC0 => {
                return Terminator::Branch {
                    cond: Cond::Scc0,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            I::S_CBRANCH_SCC1 => {
                return Terminator::Branch {
                    cond: Cond::Scc1,
                    taken: next_pcs[1],
                    fallthrough: next_pcs[0],
                }
            }
            _ => {}
        }
    }

    Terminator::Jump(next_pcs[0])
}

pub fn lower_block(pc: usize, insts: &[InstFormat], next_pcs: &[usize]) -> ScalarBlock {
    let (last, head) = insts.split_last().expect("empty block");

    let term = lower_terminator(last, next_pcs);

    let last_is_term = matches!(
        last,
        InstFormat::SOPP(i) if matches!(
            i.op,
            I::S_ENDPGM
                | I::S_BRANCH
                | I::S_CBRANCH_EXECZ
                | I::S_CBRANCH_EXECNZ
                | I::S_CBRANCH_VCCZ
                | I::S_CBRANCH_VCCNZ
                | I::S_CBRANCH_SCC0
                | I::S_CBRANCH_SCC1
        )
    );

    let body_src: &[InstFormat] = if last_is_term { head } else { insts };
    let body: Vec<InstFormat> = body_src.iter().filter(|i| !is_noop(i)).cloned().collect();
    ScalarBlock { pc, body, term }
}

pub struct DecodedBlock {
    pub insts: Vec<InstFormat>,
    pub next_pcs: Vec<usize>,
}

pub struct Decoded {
    pub entry_pc: usize,
    pub blocks: BTreeMap<usize, DecodedBlock>,
}

fn is_terminator(inst: &InstFormat) -> bool {
    matches!(inst, InstFormat::SOPP(i) if matches!(i.op,
        I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 | I::S_CBRANCH_VCCZ | I::S_CBRANCH_VCCNZ | I::S_CBRANCH_EXECZ
        | I::S_CBRANCH_EXECNZ | I::S_BRANCH | I::S_BARRIER_WAIT | I::S_ENDPGM))
}

fn successors(pc: usize, inst: &InstFormat) -> Vec<usize> {
    let target = |simm16: u16| ((pc as i64) + (simm16 as i16 as i64) * 4) as usize;
    match inst {
        InstFormat::SOPP(i) => match i.op {
            I::S_CBRANCH_EXECZ
            | I::S_CBRANCH_EXECNZ
            | I::S_CBRANCH_VCCZ
            | I::S_CBRANCH_VCCNZ
            | I::S_CBRANCH_SCC0
            | I::S_CBRANCH_SCC1 => vec![pc, target(i.simm16)],
            I::S_BRANCH => vec![target(i.simm16)],
            I::S_ENDPGM => vec![],
            _ => vec![pc],
        },
        _ => vec![pc],
    }
}

fn decode_at(memory: &[u8], pc: usize) -> Result<(InstFormat, usize), String> {
    if pc + 8 > memory.len() {
        return Err(format!(
            "instruction at {pc:#x} is outside the loaded object"
        ));
    }
    decode_rdna4(InstStream {
        insts: &memory[pc..],
    })
    .map_err(|_| format!("undecodable instruction at {pc:#x}"))
}

struct Search<'a> {
    memory: &'a [u8],
    ranges: BTreeSet<(usize, usize)>,
}

impl Search<'_> {
    fn containing(&self, pc: usize) -> Option<(usize, usize)> {
        self.ranges
            .iter()
            .copied()
            .find(|&(start, end)| pc >= start && pc < end)
    }
    fn walk(&mut self, start: usize) -> Result<(), String> {
        let mut pc = start;
        let mut last;
        loop {
            let (inst, size) = decode_at(self.memory, pc)?;
            pc += size;
            let stop = is_terminator(&inst)
                || self.containing(pc).is_some()
                || super::lift::writes_exec(&inst);
            last = inst;
            if stop {
                break;
            }
        }
        let next = successors(pc, &last);
        self.ranges.insert((start, pc));
        for next_pc in next {
            if let Some((range_start, range_end)) = self.containing(next_pc) {
                if range_start < next_pc {
                    self.ranges.remove(&(range_start, range_end));
                    self.ranges.insert((range_start, next_pc));
                    self.ranges.insert((next_pc, range_end));
                }
            } else {
                self.walk(next_pc)?;
            }
        }
        Ok(())
    }
}

pub fn program(entry_pc: usize, memory: &[u8]) -> Result<Decoded, String> {
    let mut search = Search {
        memory,
        ranges: BTreeSet::new(),
    };
    search.walk(entry_pc)?;
    let ranges: Vec<_> = search.ranges.into_iter().collect();
    for pair in ranges.windows(2) {
        if pair[0].1 != pair[1].0 {
            return Err(format!(
                "decoded ranges are not contiguous: {:#x}..{:#x} and {:#x}..{:#x}",
                pair[0].0, pair[0].1, pair[1].0, pair[1].1
            ));
        }
    }
    let mut blocks = BTreeMap::new();
    for (start, end) in ranges {
        let mut insts = Vec::new();
        let mut pc = start;
        while pc < end {
            let (inst, size) = decode_at(memory, pc)?;
            insts.push(inst);
            pc += size;
        }
        let next_pcs = successors(
            end,
            insts
                .last()
                .ok_or_else(|| format!("empty range at {start:#x}"))?,
        );
        blocks.insert(start, DecodedBlock { insts, next_pcs });
    }
    Ok(Decoded { entry_pc, blocks })
}
