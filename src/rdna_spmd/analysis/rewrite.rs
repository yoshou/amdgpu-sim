//! SSA observations for the existing instruction-window rewrite policies.
//! Native slots constrain the original pattern shapes; uses and full-definition
//! boundaries identify SSA values, including overlapping word pairs.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rdna_spmd) enum Kind { Rsq, Rcp, Mul, Fma, DivScale, DivFmas, DivFixup }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rdna_spmd) enum ConstantEncoding { Float, Integer, Literal }
#[derive(Clone, Copy, Debug)]
pub(in crate::rdna_spmd) struct Operand {
    pub slot: Option<u32>,
    pub scalar: bool,
    pub constant: Option<(ConstantEncoding, u64)>,
    pub words: [Option<ValueId>; 2],
}
impl Operand {
    pub fn vector(self) -> Option<u32> { if self.scalar { None } else { self.slot } }
    pub fn float(self, value: f64) -> bool { self.constant == Some((ConstantEncoding::Float, value.to_bits())) }
    pub fn same_encoding(self, other: Self) -> bool {
        (self.slot.is_some() || self.constant.is_some()) && self.slot == other.slot && self.scalar == other.scalar && self.constant == other.constant
    }
}
#[derive(Clone, Debug)]
pub(in crate::rdna_spmd) struct Math {
    pub kind: Kind,
    pub destination: u32,
    pub scalar_destination: Option<u32>,
    pub inputs: [Operand; 3],
    pub neg: u8,
    pub abs: u8,
    pub local_sqrt: bool,
    pub local_div: bool,
    pub cross_sqrt: bool,
}
#[derive(Clone)]
pub(in crate::rdna_spmd) struct Observation {
    pub reads: Vec<ValueId>,
    pub definitions: Vec<(u32, ValueId)>,
    pub defined_words: Vec<(u32, ValueId)>,
    pub replaced: Vec<ValueId>,
    pub known: bool,
    pub removable: bool,
    pub math: Option<Math>,
}
/// Definitions are indexed by their SSA pair, so no register-based reaching
/// definition search is needed. Unknown proof scopes remain search barriers.
pub(in crate::rdna_spmd) struct Definitions {
    pairs: BTreeMap<(ValueId, ValueId), usize>,
    versions: BTreeMap<u32, Vec<(usize, ValueId)>>,
}
impl Definitions {
    pub fn new(body: &[Observation]) -> Self {
        let mut pairs = BTreeMap::new();
        let mut versions = BTreeMap::<u32, Vec<_>>::new();
        for (index, site) in body.iter().enumerate() {
            for &(slot, value) in &site.defined_words { versions.entry(slot).or_default().push((index, value)); }
            for &(slot, lo) in &site.definitions {
                if slot < 512 { continue; }
                if let Some(&(_, hi)) = site.definitions.iter().find(|&&(r, _)| r == slot+1) {
                    pairs.insert((lo, hi), index);
                }
            }
        }
        Self { pairs, versions }
    }
    pub fn before(&self, body: &[Observation], before: usize, reg: u32) -> Option<usize> {
        // The existing expansion contract also asks for a definition at a
        // scheduling point that does not itself read the pair. Resolve that
        // point in the SSA definition timeline, rather than scanning opcodes.
        let value = |slot| {
            let versions = self.versions.get(&slot)?;
            let index = versions.partition_point(|&(index, _)| index < before);
            index.checked_sub(1).map(|i| versions[i].1)
        };
        let pair = (value(reg+512)?, value(reg+513)?);
        let &index = self.pairs.get(&pair)?;
        (index < before && body[index+1..before].iter().all(|s| s.known)).then_some(index)
    }
}
