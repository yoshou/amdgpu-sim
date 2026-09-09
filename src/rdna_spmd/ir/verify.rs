use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct VerifiedFunc(Func);
impl Func {
    #[cfg(test)]
    pub fn verify(self) -> Result<VerifiedFunc, &'static str> { self.verify_with(&crate::rdna_spmd::dialect::DialectRegistry::rdna4()) }
    pub fn verify_with(self, registry: &crate::rdna_spmd::dialect::DialectRegistry) -> Result<VerifiedFunc, &'static str> {
        if !self.blocks.contains_key(&self.entry) {
            return Err("missing entry");
        }
        let mut definitions = BTreeSet::new();
        let mut effects = BTreeSet::new();
        let mut constants = BTreeMap::new();
        for block in self.blocks.values() {
            let mut local = BTreeSet::new();
            let define = |id: ValueId,
                          ty: Ty,
                          local: &mut BTreeSet<ValueId>,
                          all: &mut BTreeSet<ValueId>| {
                if self.types.get(id.0) != Some(&ty) {
                    return Err("invalid value type");
                }
                if !all.insert(id) {
                    return Err("duplicate SSA definition");
                }
                local.insert(id);
                Ok(())
            };
            for &(id, ty) in &block.params {
                define(id, ty, &mut local, &mut definitions)?;
            }
            for inst in &block.insts {
                match inst {
                    Inst::Packet {op,input,output}=>{
                        if !local.contains(input) {return Err("non-dominating packet input");}
                        if self.types[input.0]!=Ty::I1 {return Err("packet query requires a predicate");}
                        define(*output,op.result_type(),&mut local,&mut definitions)?;
                    },
                    Inst::Target { provenance, op, args, outputs } => {
                        if args.values().iter().any(|v| !local.contains(v)) { return Err("non-dominating target input"); }
                        let spec = registry.operation(*op)?;
                        spec.verify_immediates(*args, |v| constants.get(&v).copied())?;
                        if (spec.effect == crate::rdna_spmd::dialect::Effect::Pure) != provenance.is_none() {
                            return Err("target effect provenance mismatch");
                        }
                        if let Some(id) = provenance { if !effects.insert(*id) { return Err("duplicate effect provenance"); } }
                        let expected = registry.result_types(*op, *args, &self.types)?;
                        if expected.len() != outputs.len() || outputs.iter().zip(expected).any(|((_, ty), expected)| ty != expected) {
                            return Err("target result signature mismatch");
                        }
                        for &(id, ty) in outputs { define(id, ty, &mut local, &mut definitions)?; }
                    }
                    Inst::Effect { provenance, op, inputs, outputs } => {
                        if !effects.insert(*provenance) { return Err("duplicate effect provenance"); }
                        if inputs.iter().any(|v| !local.contains(v)) { return Err("non-dominating effect input"); }
                        op.verify(inputs, outputs, &self.types)?;
                        for &(id, ty) in outputs { define(id, ty, &mut local, &mut definitions)?; }
                    }
                    Inst::Core { value, ty, op } => {
                        let mut valid = true;
                        op.map(|v| {
                            valid &= local.contains(&v);
                            v
                        });
                        if !valid {
                            return Err("value does not dominate use; pass it as a block argument");
                        }
                        if op.result_type(&self.types)? != *ty {
                            return Err("incorrect core result type");
                        }
                        define(*value, *ty, &mut local, &mut definitions)?;
                        if let Op::Const(_, bits) = op { constants.insert(*value, *bits); }
                    }

                }
            }
            if let Term::CondBr { cond, .. } = block.term {
                if !local.contains(&cond) || self.types[cond.0] != Ty::I1 {
                    return Err("branch requires a dominating i1");
                }
            }
            if let Term::Ret(args) = &block.term {
                if args.iter().any(|arg| !local.contains(arg)) { return Err("invalid return argument"); }
            }
            for e in block.term.edges() {
                let dst = self.blocks.get(&e.dst).ok_or("missing branch target")?;
                if e.args.len() != dst.params.len() {
                    return Err("block argument count mismatch");
                }
                for (&arg, &(_, ty)) in e.args.iter().zip(&dst.params) {
                    if !local.contains(&arg) || self.types[arg.0] != ty {
                        return Err("invalid block argument");
                    }
                }
            }
        }
        if definitions.len() != self.types.len() {
            return Err("undefined SSA value");
        }
        Ok(VerifiedFunc(self))
    }
}
impl VerifiedFunc {
    pub fn func(&self) -> &Func {
        &self.0
    }
}

pub(crate) struct VerifiedExpr(Expr);
impl Expr {
    #[cfg(test)]
    pub fn verify(self) -> Result<VerifiedExpr, &'static str> { self.verify_with(&crate::rdna_spmd::dialect::DialectRegistry::rdna4()) }
    pub fn verify_with(self, registry: &crate::rdna_spmd::dialect::DialectRegistry) -> Result<VerifiedExpr, &'static str> {
        let mut types = self.params.clone();
        let mut constants = std::collections::BTreeMap::new();
        for inst in &self.insts {
            match inst {
                ExprInst::Core(declared, op) => {
                    if op.result_type(&types)? != *declared { return Err("result type mismatch"); }
                    if let Op::Const(_, bits) = op { constants.insert(ValueId(types.len()), *bits); }
                },
                ExprInst::Target { op, args, outputs } => {
                    if registry.result_types(*op, *args, &types)? != outputs { return Err("target result type mismatch"); }
                    registry.operation(*op)?.verify_immediates(*args, |v| constants.get(&v).copied())?;
                },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_multistep_ssa_with_boolean_result_and_select() {
        Expr {
            params: vec![Ty::I32, Ty::I32],
            insts: vec![
                ExprInst::Core(Ty::I1, Op::Cmp(IntPred::Ult, ValueId(0), ValueId(1))),
                ExprInst::Core(Ty::I32, Op::Select(ValueId(2), ValueId(0), ValueId(1))),
            ],
            results: vec![ValueId(3)],
        }
        .verify()
        .unwrap();
    }

    #[test]
    fn rejects_forward_self_and_missing_references() {
        for value in [1, 2, 99] {
            assert!(Expr {
                params: vec![Ty::I32],
                insts: vec![ExprInst::Core(Ty::I32, Op::Int(IntOp::Add, ValueId(0), ValueId(value)))],
                results: vec![ValueId(1)]
            }
            .verify()
            .is_err());
        }
        assert!(Expr {
            params: vec![],
            insts: vec![],
            results: vec![ValueId(0)]
        }
        .verify()
        .is_err());
    }

    #[test]
    fn rejects_invalid_operand_and_result_types() {
        for (ty, op) in [
            (Ty::I32, Op::Int(IntOp::Add, ValueId(0), ValueId(1))),
            (Ty::I1, Op::Int(IntOp::Add, ValueId(1), ValueId(1))),
            (Ty::I32, Op::Cmp(IntPred::Ult, ValueId(0), ValueId(0))),
            (Ty::I32, Op::Select(ValueId(0), ValueId(0), ValueId(0))),
            (Ty::I32, Op::Select(ValueId(1), ValueId(0), ValueId(1))),
        ] {
            assert!(Expr {
                params: vec![Ty::I32, Ty::I1],
                insts: vec![ExprInst::Core(ty, op)],
                results: vec![ValueId(2)]
            }
            .verify()
            .is_err());
        }
    }
}
