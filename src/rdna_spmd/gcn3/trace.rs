use super::decode::Form;
use super::graph::{discover, Graph};
use super::state::{State, Value};
use crate::instructions::I;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub parent: Option<usize>,
    pub site: usize,
    pub entry: usize,
    pub ret: usize,
}

pub struct Trace {
    pub contexts: Vec<Context>,
    pub states: BTreeMap<(usize, usize), Vec<State>>,
    pub calls: BTreeMap<(usize, usize), usize>,
    pub missing: BTreeSet<usize>,
}

pub fn id(span: usize, context: usize, pc: usize) -> usize {
    context * span + pc
}

fn address(state: &State, register: u8) -> Option<usize> {
    match (state.sgprs.get(register as usize), state.sgprs.get(register as usize + 1)) {
        (Some(&Value::Known(low)), Some(&Value::Known(high))) => Some((high as u64) << 32 | low as u64).map(|a| a as usize),
        _ => None,
    }
}

impl Trace {
    pub fn chain(&self, mut context: usize) -> impl Iterator<Item = &Context> {
        std::iter::from_fn(move || {
            let c = self.contexts.get(context)?;
            context = c.parent.unwrap_or(usize::MAX);
            Some(c)
        })
    }
}

pub fn trace(graph: &Graph, entry: State) -> Result<Trace, String> {
    let mut trace = Trace {
        contexts: vec![Context {
            parent: None,
            site: usize::MAX,
            entry: graph.entry,
            ret: usize::MAX,
        }],
        states: BTreeMap::new(),
        calls: BTreeMap::new(),
        missing: BTreeSet::new(),
    };
    let mut at_entry: BTreeMap<(usize, usize), State> = BTreeMap::new();
    at_entry.insert((0, graph.entry), entry);
    let mut work = vec![(0usize, graph.entry)];
    while let Some(node) = work.pop() {
        let (context, start) = node;
        let Some(block) = graph.blocks.get(&start) else {
            trace.missing.insert(start);
            continue;
        };
        let mut state = at_entry[&node].clone();
        let mut states = Vec::with_capacity(block.insts.len());
        for (pc, inst) in &block.insts {
            states.push(state.clone());
            state.step(*pc, inst);
        }
        let (pc, last) = *block.insts.last().expect("a block without instructions");
        let before = states.last().unwrap().clone();
        trace.states.insert(node, states);
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
        match (last.op, last.form) {
            (I::S_SWAPPC_B64, Form::Sop1 { ssrc0, .. }) => {
                let target = (ssrc0 < 127)
                    .then(|| address(&before, ssrc0 as u8))
                    .flatten()
                    .ok_or_else(|| format!("{pc:#x}: a call to an address the program does not determine"))?;
                if trace.chain(context).any(|c| c.entry == target) {
                    return Err(format!("{pc:#x}: a recursive call to {target:#x}"));
                }
                let callee = match trace
                    .contexts
                    .iter()
                    .position(|c| c.parent == Some(context) && c.site == pc)
                {
                    Some(callee) if trace.contexts[callee].entry == target => callee,
                    Some(_) => return Err(format!("{pc:#x}: a call site whose target varies")),
                    None => {
                        trace.contexts.push(Context {
                            parent: Some(context),
                            site: pc,
                            entry: target,
                            ret: pc + last.size,
                        });
                        trace.contexts.len() - 1
                    }
                };
                trace.calls.insert(node, callee);
                flow((callee, target), state, &mut work);
            }
            (I::S_SETPC_B64, Form::Sop1 { ssrc0, .. }) => {
                let frame = trace.contexts[context];
                let parent = frame
                    .parent
                    .ok_or_else(|| format!("{pc:#x}: a kernel that jumps through a register"))?;
                let target = (ssrc0 < 127).then(|| address(&before, ssrc0 as u8)).flatten();
                if target != Some(frame.ret) {
                    return Err(format!(
                        "{pc:#x}: a function that may return somewhere other than {:#x}",
                        frame.ret
                    ));
                }
                flow((parent, frame.ret), state, &mut work);
            }
            _ => {
                for &next in &block.next {
                    flow((context, next), state.clone(), &mut work);
                }
            }
        }
    }
    Ok(trace)
}

pub fn explore(entry_pc: usize, memory: &[u8], entry: State) -> Result<(Graph, Trace), String> {
    let mut entries = vec![entry_pc];
    loop {
        let graph = discover(&entries, memory)?;
        let traced = trace(&graph, entry.clone())?;
        if traced.missing.is_empty() {
            return Ok((graph, traced));
        }
        entries.extend(traced.missing.iter().copied());
    }
}
