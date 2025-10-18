use crate::instructions::*;

#[derive(Debug, Copy, Clone)]
pub enum SourceOperand {
    ScalarRegister(u8),
    VectorRegister(u8),
    IntegerConstant(u64),
    FloatConstant(f64),
    LiteralConstant(u32),
    PrivateBase,
}

#[derive(Debug, Clone)]
pub struct SOP2 {
    pub ssrc0: SourceOperand,
    pub ssrc1: SourceOperand,
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
    pub ssrc0: SourceOperand,
    pub op: I,
    pub sdst: u8,
}

#[derive(Debug, Clone)]
pub struct SOPC {
    pub ssrc0: SourceOperand,
    pub ssrc1: SourceOperand,
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
pub struct VOP2 {
    pub src0: SourceOperand,
    pub vsrc1: u8,
    pub vdst: u8,
    pub op: I,
    pub literal_constant: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct VOP1 {
    pub src0: SourceOperand,
    pub op: I,
    pub vdst: u8,
}

#[derive(Debug, Clone)]
pub struct VOPC {
    pub src0: SourceOperand,
    pub vsrc1: u8,
    pub op: I,
}

#[derive(Debug, Clone)]
pub struct VOP3 {
    pub vdst: u8,
    pub abs: u8,
    pub opsel: u8,
    pub cm: u8,
    pub op: I,
    pub src0: SourceOperand,
    pub src1: SourceOperand,
    pub src2: SourceOperand,
    pub omod: u8,
    pub neg: u8,
}

#[derive(Debug, Clone)]
pub struct VOP3SD {
    pub vdst: u8,
    pub sdst: u8,
    pub cm: u8,
    pub op: I,
    pub src0: SourceOperand,
    pub src1: SourceOperand,
    pub src2: SourceOperand,
    pub omod: u8,
    pub neg: u8,
}

#[derive(Debug, Clone)]
pub struct VOP3P {
    pub vdst: u8,
    pub neg_hi: u8,
    pub opsel: u8,
    pub opsel_hi2: u8,
    pub cm: u8,
    pub op: I,
    pub src0: SourceOperand,
    pub src1: SourceOperand,
    pub src2: SourceOperand,
    pub opsel_hi: u8,
    pub neg: u8,
}

pub const VOPC_ENCODE: u32 = 0b0111110;
pub const VOP1_ENCODE: u32 = 0b0111111;
pub const VOP2_ENCODE: u32 = 0b0;
pub const VOP3_ENCODE: u32 = 0b110101;
pub const VOP3P_ENCODE: u32 = 0b11001100;
pub const VOPD_ENCODE: u32 = 0b110010;

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

#[derive(Debug, Clone)]
pub struct VOPD {
    pub opx: I,
    pub opy: I,
    pub src0x: SourceOperand,
    pub src0y: SourceOperand,
    pub vsrc1x: u8,
    pub vsrc1y: u8,
    pub vdstx: u8,
    pub vdsty: u8,
    pub literal_constant: Option<u32>,
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
pub struct VIMAGE {
    pub op: I,
    pub dim: u8,
    pub r128: u8,
    pub d16: u8,
    pub a16: u8,
    pub dmask: u8,
    pub vdata: u8,
    pub rsrc: u16,
    pub scope: u8,
    pub th: u8,
    pub tfe: u8,
    pub vaddr4: u8,
    pub vaddr0: u8,
    pub vaddr1: u8,
    pub vaddr2: u8,
    pub vaddr3: u8,
}

pub const VIMAGE_ENCODE: u32 = 0b110100;

#[derive(Debug, Clone)]
pub struct DS {
    pub offset0: u8,
    pub offset1: u8,
    pub op: I,
    pub addr: u8,
    pub data0: u8,
    pub data1: u8,
    pub vdst: u8,
}

pub const DS_ENCODE: u32 = 0b110110;

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

#[derive(Debug, Clone)]
pub enum InstFormat {
    SOP1(SOP1),
    SOP2(SOP2),
    SOPK(SOPK),
    SOPC(SOPC),
    SOPP(SOPP),
    SMEM(SMEM),
    VOP1(VOP1),
    VOP2(VOP2),
    VOP3(VOP3),
    VOP3SD(VOP3SD),
    VOP3P(VOP3P),
    VOPC(VOPC),
    VOPD(VOPD),
    VFLAT(VFLAT),
    VGLOBAL(VGLOBAL),
    VSCRATCH(VSCRATCH),
    VIMAGE(VIMAGE),
    DS(DS),
}
