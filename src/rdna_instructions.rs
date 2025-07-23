use crate::instructions::*;

#[derive(Debug, Clone)]
pub struct SMEM {
    pub op: I,
    pub sdata: u8,
    pub sbase: u8,
    pub ioffset: u32,
    pub soffset: u8,
    pub scope: u8,
    pub th: u8,
}

pub const SMEM_ENCODE: u32 = 0b111101;

#[derive(Debug, Clone)]
pub struct VOP3 {
    pub vdst: u8,
    pub abs: u8,
    pub opsel: u8,
    pub cm: u8,
    pub op: I,
    pub src0: u16,
    pub src1: u16,
    pub src2: u16,
    pub omod: u8,
    pub neg: u8,
}

#[derive(Debug, Clone)]
pub struct VOP3SD {
    pub vdst: u8,
    pub sdst: u8,
    pub cm: u8,
    pub op: I,
    pub src0: u16,
    pub src1: u16,
    pub src2: u16,
    pub omod: u8,
    pub neg: u8,
}

pub const VOP3_ENCODE: u32 = 0b110101;

#[derive(Debug, Clone)]
pub struct VFLAT {
    pub op: I,
    pub vaddr: u8,
    pub vsrc: u8,
    pub vdst: u8,
    pub scope: u8,
    pub th: u8,
    pub ioffset: u32,
    pub saddr: u8,
    pub sve: u8,
}

pub const VFLAT_ENCODE: u32 = 0b11101100;

#[derive(Debug, Clone)]
pub struct VGLOBAL {
    pub op: I,
    pub vaddr: u8,
    pub vsrc: u8,
    pub vdst: u8,
    pub scope: u8,
    pub th: u8,
    pub ioffset: u32,
    pub saddr: u8,
    pub sve: u8,
}

pub const VGLOBAL_ENCODE: u32 = 0b11101110;

#[derive(Debug, Clone)]
pub struct VSCRATCH {
    pub op: I,
    pub vaddr: u8,
    pub vsrc: u8,
    pub vdst: u8,
    pub scope: u8,
    pub th: u8,
    pub ioffset: u32,
    pub saddr: u8,
    pub sve: u8,
}

pub const VSCRATCH_ENCODE: u32 = 0b11101101;

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
    VOP3(VOP3),
    VOP3SD(VOP3SD),
    VOPC(VOPC),
    VFLAT(VFLAT),
    VGLOBAL(VGLOBAL),
    VSCRATCH(VSCRATCH),
}
