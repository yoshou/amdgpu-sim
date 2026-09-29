use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::environment::{Binding, Environment};
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::Program;
use std::sync::Arc;

pub(super) const EXEC: u32 = 126;
pub(super) const KERNARG: u32 = 4;

pub(super) struct Build {
    pub(super) f: Func,
    pub(super) inputs: Vec<Parameter>,
    pub(super) registry: DialectRegistry,
    pub(super) entry: EntryLayout,
    next: u64,
}

impl Build {
    pub(super) fn new(sources: &[(ParameterSource, Ty)]) -> (Self, Vec<ValueId>) {
        let mut f = Func::new(BlockId(0), Presence::Wave);
        let params: Vec<(ValueId, Ty)> = sources.iter().map(|&(_, ty)| (f.value(ty), ty)).collect();
        f.blocks.insert(
            BlockId(0),
            Block {
                params: params.clone(),
                insts: Vec::new(),
                term: Term::Ret(Vec::new()),
            },
        );
        let inputs = sources.iter().map(|&(source, ty)| Parameter { source, ty }).collect();
        let mut registry = DialectRegistry::default();
        registry.set_registers(Registers {
            exec: EXEC,
            vcc: 106,
            null: 124,
            scc_slot: 128,
            sgprs: 128,
            vgprs: 256,
        });
        let entry = EntryLayout {
            kernarg_ptr: Some(KERNARG),
            ..EntryLayout::default()
        };
        (
            Self {
                f,
                inputs,
                registry,
                entry,
                next: 1,
            },
            params.into_iter().map(|p| p.0).collect(),
        )
    }

    pub(super) fn kernel() -> (Self, Kernel) {
        let (b, k, _) = Self::kernel_with(&[]);
        (b, k)
    }

    pub(super) fn kernel_with(extra: &[(ParameterSource, Ty)]) -> (Self, Kernel, Vec<ValueId>) {
        let mut sources = vec![
            (ParameterSource::MaskBit(EXEC), Ty::I1),
            (ParameterSource::Vgpr(0), Ty::I32),
            (ParameterSource::Sgpr(KERNARG), Ty::I32),
            (ParameterSource::Sgpr(KERNARG + 1), Ty::I32),
        ];
        sources.extend_from_slice(extra);
        let (b, p) = Self::new(&sources);
        (
            b,
            Kernel {
                exec: p[0],
                item: p[1],
                kernarg: (p[2], p[3]),
            },
            p[4..].to_vec(),
        )
    }

    pub(super) fn resource(&mut self, b: BlockId, descriptor: ValueId) -> ValueId {
        let op = self.registry.lookup(7, "resource_read").unwrap_or_else(|_| {
            self.registry
                .register(
                    7,
                    1,
                    Operation {
                        name: "resource_read",
                        inputs: &[Ty::I64],
                        outputs: vec![Ty::I32],
                        effect: Effect::ReadGlobal { every_lane: false },
                        immediates: &[],
                    },
                )
                .unwrap()
        });
        let value = self.f.value(Ty::I32);
        let provenance = Some(self.next << 8);
        self.next += 1;
        self.f.blocks.get_mut(&b).unwrap().insts.push(Inst::Target {
            provenance,
            op,
            args: Arguments::Unary(descriptor),
            outputs: vec![(value, Ty::I32)],
        });
        value
    }

    pub(super) fn target(&mut self, b: BlockId, op: TargetOp, args: Arguments, types: &[Ty]) -> Vec<ValueId> {
        let outputs: Vec<(ValueId, Ty)> = types.iter().map(|&ty| (self.f.value(ty), ty)).collect();
        let provenance = Some(self.next << 8);
        self.next += 1;
        self.f.blocks.get_mut(&b).unwrap().insts.push(Inst::Target {
            provenance,
            op,
            args,
            outputs: outputs.clone(),
        });
        outputs.into_iter().map(|o| o.0).collect()
    }

    pub(super) fn block(&mut self, types: &[Ty]) -> (BlockId, Vec<ValueId>) {
        let id = BlockId(self.f.blocks.keys().last().map_or(0, |b| b.0 + 1));
        let params: Vec<(ValueId, Ty)> = types.iter().map(|&ty| (self.f.value(ty), ty)).collect();
        self.f.blocks.insert(
            id,
            Block {
                params: params.clone(),
                insts: Vec::new(),
                term: Term::Ret(Vec::new()),
            },
        );
        (id, params.into_iter().map(|p| p.0).collect())
    }

    pub(super) fn here(&self, b: BlockId) -> (BlockId, usize) {
        (b, self.f.blocks[&b].insts.len())
    }

    pub(super) fn core(&mut self, b: BlockId, ty: Ty, op: Op) -> ValueId {
        let value = self.f.value(ty);
        self.f.blocks.get_mut(&b).unwrap().insts.push(Inst::Core { value, ty, op });
        value
    }

    pub(super) fn constant(&mut self, b: BlockId, ty: Ty, k: u64) -> ValueId {
        self.core(b, ty, Op::Const(ty, k))
    }

    pub(super) fn int(&mut self, b: BlockId, op: IntOp, x: ValueId, y: ValueId) -> ValueId {
        let ty = self.f.types[x.0];
        self.core(b, ty, Op::Int(op, x, y))
    }

    pub(super) fn cmp(&mut self, b: BlockId, pred: IntPred, x: ValueId, y: ValueId) -> ValueId {
        self.core(b, Ty::I1, Op::Cmp(pred, x, y))
    }

    pub(super) fn effect(&mut self, b: BlockId, op: EffectOp, inputs: Vec<ValueId>) -> Vec<ValueId> {
        let (_, results) = op.signature();
        let outputs: Vec<(ValueId, Ty)> = results.into_iter().map(|ty| (self.f.value(ty), ty)).collect();
        let provenance = self.next << 8;
        self.next += 1;
        self.f.blocks.get_mut(&b).unwrap().insts.push(Inst::Effect {
            provenance,
            op,
            inputs,
            outputs: outputs.clone(),
        });
        outputs.into_iter().map(|o| o.0).collect()
    }

    pub(super) fn load(&mut self, b: BlockId, space: Space, size: MemSize, address: ValueId, mask: ValueId) -> ValueId {
        self.effect(b, memory(space, MemoryOp::Load(size)), vec![address, mask])[0]
    }

    pub(super) fn store(&mut self, b: BlockId, space: Space, size: MemSize, address: ValueId, data: ValueId, mask: ValueId) {
        self.effect(b, memory(space, MemoryOp::Store(size)), vec![address, data, mask]);
    }

    pub(super) fn wave(&mut self, b: BlockId, op: WaveOp, inputs: Vec<ValueId>) -> ValueId {
        self.effect(b, EffectOp::Wave(op), inputs)[0]
    }

    pub(super) fn br(&mut self, b: BlockId, dst: BlockId, args: Vec<ValueId>) {
        self.f.blocks.get_mut(&b).unwrap().term = Term::Br(Edge { dst, args });
    }

    pub(super) fn cond_br(&mut self, b: BlockId, cond: ValueId, yes: (BlockId, Vec<ValueId>), no: (BlockId, Vec<ValueId>)) {
        self.f.blocks.get_mut(&b).unwrap().term = Term::CondBr {
            cond,
            yes: Edge { dst: yes.0, args: yes.1 },
            no: Edge { dst: no.0, args: no.1 },
        };
    }

    pub(super) fn program(self) -> Program {
        Program {
            registry: Arc::new(self.registry),
            ir: self.f,
            parameter_inputs: self.inputs,
            entry: self.entry,
        }
    }
}

pub(super) struct Kernel {
    pub(super) exec: ValueId,
    pub(super) item: ValueId,
    pub(super) kernarg: (ValueId, ValueId),
}

impl Kernel {
    pub(super) fn buffer(&self, b: &mut Build, block: BlockId, offset: u64) -> ValueId {
        let base = b.core(block, Ty::I64, Op::Pack64(self.kernarg.0, self.kernarg.1));
        let at = b.constant(block, Ty::I64, offset);
        let address = b.int(block, IntOp::Add, base, at);
        let yes = b.constant(block, Ty::I1, 1);
        b.load(block, Space::Global, MemSize::B64, address, yes)
    }
}

pub(super) fn byte_offset(b: &mut Build, block: BlockId, base: ValueId, index: ValueId, scale: u64) -> ValueId {
    let wide = b.core(block, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, index));
    let k = b.constant(block, Ty::I64, scale);
    let offset = b.int(block, IntOp::Mul, wide, k);
    b.int(block, IntOp::Add, base, offset)
}

pub(super) struct Random(u64);

impl Random {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

pub(super) fn memory(space: Space, op: MemoryOp) -> EffectOp {
    EffectOp::Memory {
        space,
        op,
        semantics: MemorySemantics {
            scope: Scope::Device,
            ordering: Ordering::Relaxed,
            cache_policy: CachePolicy::Temporal,
            volatile: false,
            deferred_scope: false,
        },
    }
}

pub(super) fn environment(lanes: u32, buffers: &[(u32, u64, u64)]) -> Environment {
    Environment {
        grid: [1, 1, 1],
        block: [lanes, 1, 1],
        kernarg: vec![0; 64],
        bindings: buffers
            .iter()
            .map(|&(offset, allocation, pointer)| Binding {
                offset,
                allocation,
                pointer,
            })
            .collect(),
        exposed: Vec::new(),
    }
}
