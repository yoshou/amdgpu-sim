use super::target::Target;
use std::sync::Arc;

pub(crate) mod rdna4;

pub(crate) fn select(arch: &str) -> Option<Arc<dyn Target>> {
    let rdna4 = rdna4::Rdna4::new();
    if rdna4.supports(arch) {
        return Some(Arc::new(rdna4));
    }
    None
}
