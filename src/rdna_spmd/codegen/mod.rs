mod access;
mod emit;
mod memory;
mod ops;
mod prepare;
mod region;
mod wave;
mod wmma;
mod yields;

pub use ops::{Emitter, Implementation, Lowerings};
pub use prepare::prepare;
pub use wmma::compile_wmma;
pub use yields::{Argument, YieldValues};

use std::collections::BTreeMap;

use super::ir::{BlockId, Func, Parameter, VerifiedFunc};
use super::native::{Builder, Value};
use access::Access;
use emit::emit_function;

struct Cluster {
    pub members: usize,
    pub lo: i64,
    pub span: u32,
    pub tile: u32,
}

pub const DONE: u64 = u64::MAX;

pub const ENTER: u64 = 1 << 32;

pub const LEAVE: u64 = 1 << 33;

pub const YIELD: &str = "amdgpu_sim_fiber_yield_values";

pub struct Prepared {
    registry: std::sync::Arc<super::ir::DialectRegistry>,
    ir: VerifiedFunc,
    inputs: Vec<Parameter>,
    width: u32,
    uniform: Vec<bool>,

    holds_a_lane: Vec<bool>,
    accesses: Vec<Access>,
    shapes: Vec<memory::Shape>,
    clusters: BTreeMap<usize, Cluster>,
    yields: BTreeMap<u64, yields::YieldValues>,
    groups: Vec<Vec<u64>>,
    min_private_bytes: usize,
    num_vgprs: usize,
}

impl Prepared {
    pub fn func(&self) -> &Func {
        self.ir.func()
    }
    pub fn registers(&self) -> super::ir::Registers {
        self.registry.registers()
    }
    pub fn num_vgprs(&self) -> usize {
        self.num_vgprs
    }
    pub fn min_private_bytes(&self) -> usize {
        self.min_private_bytes
    }
    pub fn resume_layouts(&self) -> Vec<Vec<yields::YieldValues>> {
        self.groups
            .iter()
            .map(|g| g.iter().map(|p| self.yields[p].clone()).collect())
            .collect()
    }
    fn resume_index(&self, provenance: u64) -> usize {
        self.groups
            .iter()
            .position(|g| g[0] == provenance)
            .expect("scheduled effect lacks a yield layout")
    }
    fn group_at(&self, provenance: u64) -> Option<&[u64]> {
        self.groups
            .iter()
            .find(|g| g[0] == provenance)
            .map(|g| g.as_slice())
    }
}

pub struct RegionCode {
    pub address: u64,
    pub children: Vec<usize>,
    pub blocks: std::collections::BTreeSet<BlockId>,
}

pub struct Compiled {
    pub code: super::native::jit::NativeCode,
    pub regions: Vec<RegionCode>,
    pub frame_words: usize,
}

struct Shared {
    counts: Option<(String, Value, BTreeMap<BlockId, usize>)>,
}

impl Shared {
    fn new(ir: Builder, f: &Func, defines: bool) -> Self {
        let counts = std::env::var("AMDGPU_SIM_BLOCK_COUNTS").ok().map(|path| {
            let ty = ir.i64().array(f.blocks.len() as u64);
            let global = ir.add_global("block_counts", ty);
            let index: BTreeMap<BlockId, usize> = f
                .blocks
                .keys()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();
            if defines {
                global.set_initializer(ty.null());
                let text: String = f
                    .blocks
                    .keys()
                    .map(|id| format!("{} b{:x}\n", index[id], id.0))
                    .collect();
                std::fs::write(format!("{path}.blocks"), text).unwrap();
            }
            (path, global, index)
        });
        Self {
            counts,
        }
    }
}

fn each_in_parallel<T: Send, R: Send>(items: Vec<T>, work: impl Fn(T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let slots: Vec<Mutex<Option<T>>> = items.into_iter().map(|item| Mutex::new(Some(item))).collect();
    let results: Vec<Mutex<Option<R>>> = (0..slots.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, slots.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= slots.len() {
                    break;
                }
                let item = slots[k].lock().unwrap().take().unwrap();
                *results[k].lock().unwrap() = Some(work(item));
            });
        }
    });
    results
        .into_iter()
        .map(|result| result.into_inner().unwrap().unwrap())
        .collect()
}

pub fn compile_regions(
    p: &Prepared,
    lowerings: std::sync::Arc<ops::Lowerings>,
    name: &str,
    runtime: &[(&str, u64)],
) -> Compiled {
    let regions = region::Regions::new(p);
    let symbols: Vec<String> = (0..regions.count())
        .map(|r| format!("kernel_{r}"))
        .collect();
    let mut counts = None;
    let modules: Vec<super::native::jit::Module> = symbols
        .iter()
        .enumerate()
        .map(|(r, symbol)| {
            let native = super::native::jit::Module::new(&format!("{name}_{r}"));
            let shared = Shared::new(native.builder(), p.ir.func(), r == 0);
            emit_function(p, &lowerings, native.builder(), symbol, (&regions, r), &shared);
            if r == 0 {
                counts = shared.counts.map(|(path, _, index)| (path, index.len()));
            }
            native
        })
        .collect();
    let objects = each_in_parallel(modules, |module| module.optimize().object());
    let mut code = super::native::jit::link(objects, &symbols[0], runtime);
    if let Some((path, blocks)) = counts {
        code.count_blocks(path, blocks);
    }
    let compiled = symbols
        .iter()
        .enumerate()
        .map(|(r, symbol)| RegionCode {
            address: code.lookup(symbol),
            children: regions.children(r).to_vec(),
            blocks: regions.own(r),
        })
        .collect();
    Compiled {
        code,
        regions: compiled,
        frame_words: regions.frame_words(),
    }
}
