use super::super::analysis::{Analyses, Exec, Predication};
use super::super::ir::{*, Cvt, Op};

pub(crate) struct Active;
impl super::Pass for Active {
    fn name(&self) -> &str { "active" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let (masks, exec) = (analyses.get::<Predication>(f), analyses.get::<Exec>(f));
        run(f, &masks, &exec) > 0
    }
}

pub(crate) fn run(f: &mut Func, masks: &Predication, exec: &Exec) -> usize {
    let mut count = 0;
    for (&id, block) in f.blocks.iter_mut() {
        for (index, inst) in block.insts.iter_mut().enumerate() {
            let Inst::Core { value, ty, op } = inst else { continue; };
            let replacement = match *op {
                Op::Select(_, new, _) if masks.predicated[value.0].is_some() => Some(new),
                Op::Int(..) if masks.masked[value.0] => masks.masked_result[value.0],
                _ => None,
            };
            let Some(raw) = replacement else { continue; };
            if !exec.active_at(id, index) { continue; }
            *op = Op::Convert(Cvt::Bitcast, *ty, raw);
            count += 1;
        }
    }
    count
}
