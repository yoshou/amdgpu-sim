use super::super::analysis::{Analyses, Constants, Predication};
use super::super::ir::Func;
use super::Pass;

pub(crate) trait Idiom: Send + Sync {
    fn rewrite(&self, f: &mut Func, predication: &Predication, constants: &[Option<u64>]) -> usize;
}

pub(crate) struct Idioms;
impl Pass for Idioms {
    fn name(&self) -> &str { "idioms" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let (predication, constants) = (analyses.get::<Predication>(f), analyses.get::<Constants>(f));
        for idiom in analyses.context().registry.idioms() {
            if idiom.rewrite(f, &predication, &constants) > 0 { return true; }
        }
        false
    }
}
