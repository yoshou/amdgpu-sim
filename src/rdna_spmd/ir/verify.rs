use super::*;
use std::collections::BTreeSet;

pub(crate) struct VerifiedFunc(Func);
impl Func {
    pub fn verify_with(
        self,
        registry: &crate::rdna_spmd::dialect::DialectRegistry,
    ) -> Result<VerifiedFunc, &'static str> {
        self.check(registry)?;
        Ok(VerifiedFunc(self))
    }
    pub fn check(
        &self,
        registry: &crate::rdna_spmd::dialect::DialectRegistry,
    ) -> Result<(), &'static str> {
        if !self.blocks.contains_key(&self.entry) {
            return Err("missing entry");
        }
        let count = self.types.len();
        let mut defined = vec![false; count];
        let mut definitions = 0usize;
        let mut effects = BTreeSet::new();
        let mut constants: Vec<Option<u64>> = vec![None; count];
        let mut local = vec![0u32; count];
        let mut generation = 0u32;
        for block in self.blocks.values() {
            generation += 1;
            let seen = |id: ValueId, local: &[u32]| id.0 < count && local[id.0] == generation;
            let define = |id: ValueId,
                          ty: Ty,
                          local: &mut Vec<u32>,
                          defined: &mut Vec<bool>,
                          definitions: &mut usize| {
                if self.types.get(id.0) != Some(&ty) {
                    return Err("invalid value type");
                }
                if defined[id.0] {
                    return Err("duplicate SSA definition");
                }
                defined[id.0] = true;
                *definitions += 1;
                local[id.0] = generation;
                Ok(())
            };
            for &(id, ty) in &block.params {
                define(id, ty, &mut local, &mut defined, &mut definitions)?;
            }
            for inst in &block.insts {
                match inst {
                    Inst::Packet { op, input, output } => {
                        if !seen(*input, &local) {
                            return Err("non-dominating packet input");
                        }
                        if self.types[input.0] != Ty::I1 {
                            return Err("packet query requires a predicate");
                        }
                        define(
                            *output,
                            op.result_type(),
                            &mut local,
                            &mut defined,
                            &mut definitions,
                        )?;
                    }
                    Inst::Target {
                        provenance,
                        op,
                        args,
                        outputs,
                    } => {
                        if args.values().iter().any(|v| !seen(*v, &local)) {
                            return Err("non-dominating target input");
                        }
                        let spec = registry.operation(*op)?;
                        spec.verify_immediates(*args, |v| constants.get(v.0).copied().flatten())?;
                        if (spec.effect == crate::rdna_spmd::dialect::Effect::Pure)
                            != provenance.is_none()
                        {
                            return Err("target effect provenance mismatch");
                        }
                        if let Some(id) = provenance {
                            if !effects.insert(*id) {
                                return Err("duplicate effect provenance");
                            }
                        }
                        let expected = registry.result_types(*op, *args, &self.types)?;
                        if expected.len() != outputs.len()
                            || outputs
                                .iter()
                                .zip(expected)
                                .any(|((_, ty), expected)| ty != expected)
                        {
                            return Err("target result signature mismatch");
                        }
                        for &(id, ty) in outputs {
                            define(id, ty, &mut local, &mut defined, &mut definitions)?;
                        }
                    }
                    Inst::Effect {
                        provenance,
                        op,
                        inputs,
                        outputs,
                    } => {
                        if !effects.insert(*provenance) {
                            return Err("duplicate effect provenance");
                        }
                        if inputs.iter().any(|v| !seen(*v, &local)) {
                            return Err("non-dominating effect input");
                        }
                        op.verify(inputs, outputs, &self.types)?;
                        for &(id, ty) in outputs {
                            define(id, ty, &mut local, &mut defined, &mut definitions)?;
                        }
                    }
                    Inst::Core { value, ty, op } => {
                        let mut valid = true;
                        op.map(|v| {
                            valid &= seen(v, &local);
                            v
                        });
                        if !valid {
                            return Err("value does not dominate use; pass it as a block argument");
                        }
                        if op.result_type(&self.types)? != *ty {
                            return Err("incorrect core result type");
                        }
                        define(*value, *ty, &mut local, &mut defined, &mut definitions)?;
                        if let Op::Const(_, bits) = op {
                            constants[value.0] = Some(*bits);
                        }
                    }
                }
            }
            if let Term::CondBr { cond, .. } = block.term {
                if !seen(cond, &local) || self.types[cond.0] != Ty::I1 {
                    return Err("branch requires a dominating i1");
                }
            }
            if let Term::Ret(args) = &block.term {
                if args.iter().any(|arg| !seen(*arg, &local)) {
                    return Err("invalid return argument");
                }
            }
            for e in block.term.edges() {
                let dst = self.blocks.get(&e.dst).ok_or("missing branch target")?;
                if e.args.len() != dst.params.len() {
                    return Err("block argument count mismatch");
                }
                for (&arg, &(_, ty)) in e.args.iter().zip(&dst.params) {
                    if !seen(arg, &local) || self.types[arg.0] != ty {
                        return Err("invalid block argument");
                    }
                }
            }
        }
        if definitions != count {
            return Err("undefined SSA value");
        }
        self.check_regions()
    }

    fn check_regions(&self) -> Result<(), &'static str> {
        let doms = Dominators::of(self);
        if self.regions.get(&self.entry).is_none() {
            return Err("no region is entered at the function's entry");
        }
        let reached: BTreeSet<BlockId> = doms.order.iter().copied().collect();
        for (&entry, &present) in &self.regions {
            if !reached.contains(&entry) {
                return Err("a region is entered at a block the function does not reach");
            }
            let enclosing = self
                .regions
                .iter()
                .filter(|&(&other, _)| self.encloses(&doms, other, entry))
                .map(|(_, &p)| p)
                .min();
            if enclosing.is_some_and(|outer| present > outer) {
                return Err("a region has more lanes present than the one enclosing it");
            }
        }
        let held = self.regions_of(&doms);
        for &id in &doms.order {
            let present = self.regions[&held[&id]];
            for inst in &self.blocks[&id].insts {
                if lanes_read(inst) > Some(present) {
                    return Err("an operation reads lanes its region does not have present");
                }
            }
        }
        Ok(())
    }
}

pub(super) fn lanes_read(inst: &Inst) -> Option<Presence> {
    match inst {
        Inst::Packet { .. }
        | Inst::Core {
            op: Op::Env(Env::PacketLaneId),
            ..
        } => Some(Presence::Packet),
        Inst::Effect {
            op: EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait,
            ..
        } => Some(Presence::Workgroup),
        Inst::Effect {
            op: EffectOp::Wave(_),
            ..
        } => Some(Presence::Wave),
        _ => None,
    }
}
impl VerifiedFunc {
    pub fn func(&self) -> &Func {
        &self.0
    }
}

pub(crate) struct VerifiedExpr(Expr);
impl Expr {
    pub fn verify_with(
        self,
        registry: &crate::rdna_spmd::dialect::DialectRegistry,
    ) -> Result<VerifiedExpr, &'static str> {
        let mut types = self.params.clone();
        let mut constants = std::collections::BTreeMap::new();
        for inst in &self.insts {
            match inst {
                ExprInst::Core(declared, op) => {
                    if op.result_type(&types)? != *declared {
                        return Err("result type mismatch");
                    }
                    if let Op::Const(_, bits) = op {
                        constants.insert(ValueId(types.len()), *bits);
                    }
                }
                ExprInst::Target { op, args, outputs } => {
                    if registry.result_types(*op, *args, &types)? != outputs {
                        return Err("target result type mismatch");
                    }
                    registry
                        .operation(*op)?
                        .verify_immediates(*args, |v| constants.get(&v).copied())?;
                }
            }
            types.extend_from_slice(inst.result_types());
        }
        if self.results.iter().any(|v| v.0 >= types.len()) {
            return Err("undefined result");
        }
        Ok(VerifiedExpr(self))
    }
}
impl VerifiedExpr {
    pub fn expr(&self) -> &Expr {
        &self.0
    }
}
