use super::super::analysis::masks::Predication;
use super::super::ir::Func;
use super::{Analyses, Pass};

pub(crate) trait Idiom: Send + Sync {
    fn rewrite(&self, f: &mut Func, predication: &Predication, constants: &[Option<u64>]) -> usize;
}

pub(crate) struct Idioms;
impl Pass for Idioms {
    fn name(&self) -> &str { "idioms" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        for idiom in analyses.context().registry.idioms() {
            if idiom.rewrite(f, analyses.predication(f), analyses.constants(f)) > 0 { return true; }
        }
        false
    }
}
