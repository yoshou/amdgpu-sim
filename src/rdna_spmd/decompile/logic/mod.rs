mod kernel;
mod rules;

#[cfg(test)]
mod tests;

pub use kernel::{choices, constant_choices, lane_test, projected_word, Atom, Choice, Kept, PATH};
#[cfg(test)]
pub(super) use rules::float_compare;
pub use rules::Logic;

type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;
