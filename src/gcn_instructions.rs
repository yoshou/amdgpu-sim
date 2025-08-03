use crate::instructions::*;

#[derive(Debug, Clone)]
pub struct VOP2 {
    pub src0: u16,
    pub vsrc1: u8,
    pub vdst: u8,
    pub op: I,
}

#[derive(Debug, Clone)]
pub struct VOP1 {
    pub src0: u16,
    pub op: I,
    pub vdst: u8,
}

#[derive(Debug, Clone)]
pub struct VOPC {
    pub src0: u16,
    pub vsrc1: u8,
    pub op: I,
}

#[derive(Debug, Clone)]
pub struct VOP3A {
    pub vdst: u8,
    pub abs: u8,
    pub clamp: u8,
    pub op: I,
    pub src0: u16,
    pub src1: u16,
    pub src2: u16,
    pub omod: u8,
    pub neg: u8,
}

#[derive(Debug, Clone)]
pub struct VOP3B {
    pub vdst: u8,
    pub sdst: u8,
    pub clamp: u8,
    pub op: I,
    pub src0: u16,
    pub src1: u16,
    pub src2: u16,
    pub omod: u8,
    pub neg: u8,
}

pub const VOPC_ENCODE: u32 = 0b0111110;
pub const VOP1_ENCODE: u32 = 0b0111111;
pub const VOP2_ENCODE: u32 = 0b0;
pub const VOP3AB_ENCODE: u32 = 0b110100;

#[derive(Debug, Clone)]
pub struct SOP2 {
    pub ssrc0: u8,
    pub ssrc1: u8,
    pub sdst: u8,
    pub op: I,
}

#[derive(Debug, Clone)]
pub struct SOPK {
    pub simm16: u16,
    pub sdst: u8,
    pub op: I,
}

#[derive(Debug, Clone)]
pub struct SOP1 {
    pub ssrc0: u8,
    pub op: I,
    pub sdst: u8,
}

#[derive(Debug, Clone)]
pub struct SOPC {
    pub ssrc0: u8,
    pub ssrc1: u8,
    pub op: I,
}

#[derive(Debug, Clone)]
pub struct SOPP {
    pub simm16: u16,
    pub op: I,
}

pub const SOP2_ENCODE: u32 = 0b10;
pub const SOPK_ENCODE: u32 = 0b1011;
pub const SOP1_ENCODE: u32 = 0b101111101;
pub const SOPC_ENCODE: u32 = 0b101111110;
pub const SOPP_ENCODE: u32 = 0b101111111;

#[derive(Debug, Clone)]
pub struct SMRD {
    pub offset: u8,
    pub imm: u8,
    pub sbase: u8,
    pub sdst: u8,
    pub op: u8,
}

pub const SMRD_ENCODE: u32 = 0b11000;

#[derive(Debug, Clone)]
pub struct FLAT {
    pub glc: u8,
    pub slc: u8,
    pub op: I,
    pub addr: u8,
    pub data: u8,
    pub tfe: u8,
    pub vdst: u8,
}

pub const FLAT_ENCODE: u32 = 0b110111;

#[derive(Debug, Clone)]
pub struct MUBUF {
    pub offset: u16,
    pub offen: u8,
    pub idxen: u8,
    pub glc: u8,
    pub lds: u8,
    pub slc: u8,
    pub op: I,
    pub vaddr: u8,
    pub vdata: u8,
    pub srsrc: u8,
    pub tfe: u8,
    pub soffset: u8,
}

pub const MUBUF_ENCODE: u32 = 0b111000;
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
