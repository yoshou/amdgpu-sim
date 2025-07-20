use crate::instructions::*;

#[derive(Debug, Clone)]
pub struct SMEM {
    pub sbase: u8,
    pub sdata: u8,
    pub glc: u8,
    pub imm: u8,
    pub op: I,
    pub offset: u8,
}

pub const SMEM_ENCODE: u32 = 0b110000;

#[derive(Debug)]
pub enum InstFormat {
    SOP1(SOP1),
    SOP2(SOP2),
    SOPK(SOPK),
    SOPC(SOPC),
    SOPP(SOPP),
    SMRD(SMRD),
    SMEM(SMEM),
    VOP1(VOP1),
    VOP2(VOP2),
    VOPC(VOPC),
    VOP3A(VOP3A),
    VOP3B(VOP3B),
    FLAT(FLAT),
    MUBUF(MUBUF),
}
