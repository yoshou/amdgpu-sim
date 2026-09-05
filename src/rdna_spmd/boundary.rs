//! Register footprints exchanged with host wave operations.

/// A set of SGPRs and VGPRs.
#[derive(Clone, Copy, Default, PartialEq)]
pub(super) struct RegSet {
    pub(super) scc: bool,
    sgpr: u128,
    vgpr: [u128; 2],
}

impl RegSet {
    /// Registers outside the architectural files (128 SGPRs, 256 VGPRs) are
    /// dropped: operand encodings can name reserved indices, and aliasing one
    /// onto a real register would be worse than ignoring it.
    pub(super) fn add_sgpr(&mut self, reg: u32) {
        if reg < 128 {
            self.sgpr |= 1 << reg;
        }
    }
    pub(super) fn add_vgpr(&mut self, reg: u32) {
        if reg < 256 {
            self.vgpr[(reg >> 7) as usize] |= 1 << (reg & 127);
        }
    }
    pub(super) fn has_sgpr(&self, reg: u32) -> bool {
        reg < 128 && self.sgpr & (1 << reg) != 0
    }
    pub(super) fn has_vgpr(&self, reg: u32) -> bool {
        reg < 256 && self.vgpr[(reg >> 7) as usize] & (1 << (reg & 127)) != 0
    }
    pub(super) fn vgprs(&self) -> impl Iterator<Item = u32> + '_ {
        (0..256u32).filter(move |&reg| self.has_vgpr(reg))
    }
}

/// What the host-applied wave-level op at a boundary touches: the registers it
/// reads out of the packet (which the kernel stores before yielding) and the
/// ones it writes back (which the kernel reloads afterwards). A partial write
/// — writelane touches one lane of the packed vector — belongs in `reads` too,
/// so the lanes it leaves alone survive the round trip.
#[derive(Clone, Copy, Default)]
pub(super) struct BoundaryIo {
    pub(super) reads: RegSet,
    pub(super) writes: RegSet,
}
