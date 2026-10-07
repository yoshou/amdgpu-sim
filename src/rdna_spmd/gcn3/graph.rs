use super::decode::{decode, Form, Inst};
use super::state::writes_exec;
use crate::instructions::I;
use std::collections::{BTreeMap, BTreeSet};

pub struct Block {
    pub insts: Vec<(usize, Inst)>,
    pub next: Vec<usize>,
}

pub struct Graph {
    pub entry: usize,
    pub blocks: BTreeMap<usize, Block>,
}

impl Graph {
    pub fn span(&self) -> usize {
        self.blocks
            .values()
            .filter_map(|b| b.insts.last())
            .map(|(pc, inst)| pc + inst.size)
            .max()
            .unwrap_or(0)
    }
}

fn branch(op: I) -> bool {
    matches!(
        op,
        I::S_CBRANCH_SCC0
            | I::S_CBRANCH_SCC1
            | I::S_CBRANCH_VCCZ
            | I::S_CBRANCH_VCCNZ
            | I::S_CBRANCH_EXECZ
            | I::S_CBRANCH_EXECNZ
    )
}

pub fn terminates(op: I) -> bool {
    branch(op) || matches!(op, I::S_BRANCH | I::S_ENDPGM | I::S_SWAPPC_B64 | I::S_SETPC_B64)
}

fn ends(op: I) -> bool {
    terminates(op) || matches!(op, I::S_BARRIER)
}

fn successors(end: usize, inst: &Inst) -> Vec<usize> {
    match inst.op {
        I::S_SETPC_B64 => return vec![],
        I::S_SWAPPC_B64 => return vec![end],
        _ => {}
    }
    let Form::Sopp { simm16 } = inst.form else {
        return vec![end];
    };
    let target = (end as i64 + simm16 as i16 as i64 * 4) as usize;
    match inst.op {
        op if branch(op) => vec![end, target],
        I::S_BRANCH => vec![target],
        I::S_ENDPGM => vec![],
        _ => vec![end],
    }
}

fn scc_branch_follows(memory: &[u8], mut pc: usize) -> bool {
    loop {
        let Ok(inst) = decode(memory, pc) else {
            return false;
        };
        match inst.op {
            I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 => return true,
            I::S_NOP => pc += inst.size,
            _ => return false,
        }
    }
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
        let last = loop {
            let inst = decode(self.memory, pc)?;
            pc += inst.size;
            if ends(inst.op)
                || self.containing(pc).is_some()
                || writes_exec(&inst) && !scc_branch_follows(self.memory, pc)
            {
                break inst;
            }
        };
        self.ranges.insert((start, pc));
        for next in successors(pc, &last) {
            match self.containing(next) {
                Some((first, end)) if first < next => {
                    self.ranges.remove(&(first, end));
                    self.ranges.insert((first, next));
                    self.ranges.insert((next, end));
                }
                Some(_) => {}
                None => self.walk(next)?,
            }
        }
        Ok(())
    }
}

pub fn discover(entries: &[usize], memory: &[u8]) -> Result<Graph, String> {
    let mut search = Search {
        memory,
        ranges: BTreeSet::new(),
    };
    for &entry in entries {
        match search.containing(entry) {
            Some((first, end)) if first < entry => {
                search.ranges.remove(&(first, end));
                search.ranges.insert((first, entry));
                search.ranges.insert((entry, end));
            }
            Some(_) => {}
            None => search.walk(entry)?,
        }
    }
    let mut blocks = BTreeMap::new();
    for (start, end) in search.ranges {
        let mut insts = Vec::new();
        let mut pc = start;
        while pc < end {
            let inst = decode(memory, pc)?;
            insts.push((pc, inst));
            pc += inst.size;
        }
        if pc != end {
            return Err(format!("an instruction at {start:#x}..{end:#x} straddles a block boundary"));
        }
        let next = successors(end, &insts.last().ok_or_else(|| format!("empty block at {start:#x}"))?.1);
        blocks.insert(start, Block { insts, next });
    }
    Ok(Graph {
        entry: entries[0],
        blocks,
    })
}
