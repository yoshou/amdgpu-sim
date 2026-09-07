//! Native storage slot set. Representation proofs live in `analysis::state`.

pub type RegSet = [u128; 2];

#[inline]
pub fn bget(s: &RegSet, r: u32) -> bool {
    let r = (r & 255) as usize;
    (s[r / 128] >> (r % 128)) & 1 == 1
}
