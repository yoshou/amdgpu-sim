pub(crate) mod adjacency;
pub(crate) mod dce;
pub(crate) mod entry;
pub(crate) mod idioms;
pub(crate) mod pairs;
pub(crate) mod simplify;
pub(crate) mod uniform_queries;

use super::analysis::{Analyses, Preserved};
use super::dialect::DialectRegistry;
use super::ir::Func;

pub(crate) trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool;
    fn preserves(&self) -> Preserved {
        Preserved::none()
    }
}

pub(super) struct Driver {
    pub(super) trace: bool,
    verify_each: bool,
}

impl Driver {
    pub fn new() -> Self {
        let trace = std::env::var_os("AMDGPU_SIM_PRINT_IR").is_some();
        Self {
            trace,
            verify_each: trace || cfg!(test),
        }
    }
    pub fn pipeline(
        &self,
        f: &mut Func,
        analyses: &mut Analyses,
        passes: &[&dyn Pass],
    ) -> Result<bool, String> {
        let mut any = false;
        for pass in passes {
            let changed = pass.run(f, analyses);
            if changed {
                analyses.invalidate(&pass.preserves());
                if self.verify_each {
                    analyses.audit(f, pass.name());
                }
                any = true;
            }
            if self.verify_each {
                self.check(f, analyses.context().registry, pass.name())?;
            }
        }
        if !self.verify_each {
            self.check(
                f,
                analyses.context().registry,
                passes.last().map_or("pipeline", |p| p.name()),
            )?;
        }
        Ok(any)
    }
    pub fn fixpoint(
        &self,
        f: &mut Func,
        analyses: &mut Analyses,
        name: &str,
        limit: usize,
        passes: &[&dyn Pass],
    ) -> Result<(), String> {
        if limit == 0 {
            return Err(format!(
                "{name}: fixpoint group requires a nonzero iteration limit"
            ));
        }
        for _ in 0..limit {
            if !self.pipeline(f, analyses, passes)? {
                return Ok(());
            }
        }
        Err(format!("{name}: no fixed point within {limit} iterations"))
    }
    fn check(&self, f: &Func, registry: &DialectRegistry, name: &str) -> Result<(), String> {
        if self.trace {
            eprintln!("; after {name}\n{}", super::ir::print::func(registry, f));
        }
        f.check(registry).map_err(|e| format!("{name}: {e}"))?;
        Ok(())
    }
}
