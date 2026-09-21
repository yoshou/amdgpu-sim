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
    fn new(ir: Builder, f: &Func) -> Self {
        let counts = std::env::var("AMDGPU_SIM_BLOCK_COUNTS").ok().map(|path| {
            let ty = ir.i64().array(f.blocks.len() as u64);
            let global = ir.add_global("block_counts", ty);
            global.set_initializer(ty.null());
            let index: BTreeMap<BlockId, usize> = f
                .blocks
                .keys()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();
            let text: String = f
                .blocks
                .keys()
                .map(|id| format!("{} b{:x}\n", index[id], id.0))
                .collect();
            std::fs::write(format!("{path}.blocks"), text).unwrap();
            (path, global, index)
        });
        Self {
            counts,
        }
    }

    fn finish(
        self,
        native: super::native::jit::Module,
        symbol: &str,
        runtime: &[(&str, u64)],
    ) -> super::native::jit::NativeCode {
        let mut code = native.optimize().compile(symbol, runtime);
        if let Some((path, _, index)) = self.counts {
            code.count_blocks(path, index.len());
        }
        code
    }
}

pub fn compile_regions(
    p: &Prepared,
    lowerings: std::sync::Arc<ops::Lowerings>,
    name: &str,
    runtime: &[(&str, u64)],
) -> Compiled {
    let regions = region::Regions::new(p);
    let native = super::native::jit::Module::new(name);
    let shared = Shared::new(native.builder(), p.ir.func());
    let symbols: Vec<String> = (0..regions.count())
        .map(|r| format!("kernel_{r}"))
        .collect();
    for (r, symbol) in symbols.iter().enumerate() {
        emit_function(p, &lowerings, native.builder(), symbol, (&regions, r), &shared);
    }
    let code = shared.finish(native, &symbols[0], runtime);
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
