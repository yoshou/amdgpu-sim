use super::decode::{Ext, Form, Inst, Sdwa, LITERAL};
use super::flush::flushed;
use super::state::{carry, compare, compare_exec, hwreg, State, Value, EXEC, HW_REG_MODE, M0, VCC};
use crate::instructions::I;
use crate::rdna_instructions::{self as rdna, InstFormat, SourceOperand};
use crate::rdna_spmd::dialect::DialectRegistry;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::rdna4::lift::{
    self, input, packed::narrow_wide_half, Builder, InputSource, Lowering, Operand, Output, YieldAction,
};

pub struct Target<'a> {
    pub registry: &'a DialectRegistry,
    pub lanes: u32,
    pub descriptor: Option<u16>,
}

#[derive(Default)]
pub struct Lowered {
    pub scan: Vec<InstFormat>,
    pub lowerings: Vec<Lowering>,
}

impl Lowered {
    fn one(scan: Option<InstFormat>, lowering: Lowering) -> Self {
        Lowered {
            scan: scan.into_iter().collect(),
            lowerings: vec![lowering],
        }
    }
}

fn refuse(inst: &Inst, why: &str) -> String {
    format!("{:?} {why}", inst.op)
}

pub fn source(code: u16, literal: Option<u32>) -> Result<SourceOperand, String> {
    Ok(match code {
        0..=103 | 106 | 107 | 126 | 127 => SourceOperand::ScalarRegister(code as u8),
        M0 => SourceOperand::ScalarRegister(125),
        128..=192 => SourceOperand::IntegerConstant((code - 128) as u64),
        193..=208 => SourceOperand::IntegerConstant((192 - code as i64) as u64),
        240..=247 => SourceOperand::FloatConstant([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][(code - 240) as usize]),
        248 => SourceOperand::FloatConstant(f64::from_bits(0x3fc4_5f30_6dc9_c882)),
        LITERAL => SourceOperand::LiteralConstant(literal.ok_or("a literal operand without its literal")?),
        256..=511 => SourceOperand::VectorRegister((code - 256) as u8),
        _ => return Err(format!("operand {code} has no SPMD model")),
    })
}

pub fn scalar_dst(code: u8) -> Result<u8, String> {
    match code as u16 {
        0..=103 | 106 | 107 | 126 | 127 => Ok(code),
        M0 => Ok(125),
        _ => Err(format!("scalar destination {code} has no SPMD model")),
    }
}

fn name(op: I) -> String {
    format!("{:?}", op)
}

fn wide_integer(op: I) -> bool {
    let n = name(op);
    n.contains("_B64") || n.contains("_U64") || n.contains("_I64")
}

#[derive(Clone, Copy)]
struct Denormals {
    inputs: bool,
    outputs: bool,
}

fn denormals(inst: &Inst, state: &State) -> Result<Denormals, String> {
    let n = name(inst.op);
    let single = n.contains("F32");
    let other = n.contains("F64") || n.contains("F16");
    if !single && !other {
        return Ok(Denormals {
            inputs: false,
            outputs: false,
        });
    }
    if state.mode.field(0, 4) != Some(0) {
        return Err(refuse(inst, "runs under an unknown or non-nearest rounding mode"));
    }
    if state.mode.field(8, 2) != Some(3) {
        return Err(refuse(inst, "runs without both DX10 clamping and IEEE mode"));
    }
    if other && state.mode.field(6, 2) != Some(3) {
        return Err(refuse(inst, "runs while double and half denormals are flushed"));
    }
    if !single {
        return Ok(Denormals {
            inputs: false,
            outputs: false,
        });
    }
    let mode = state
        .mode
        .field(4, 2)
        .ok_or_else(|| refuse(inst, "runs under an unknown single precision denormal mode"))?;
    Ok(Denormals {
        inputs: mode & 1 == 0,
        outputs: mode & 2 == 0,
    })
}

fn checked_modifiers(inst: &Inst, lowering: &Lowering, abs: u8, neg: u8, clamp: bool, omod: u8) -> Result<(), String> {
    let Lowering::TypedAlu { inputs, outputs, .. } = lowering else {
        return Ok(());
    };
    for (k, input) in inputs.iter().enumerate().take(3) {
        if (abs | neg) >> k & 1 != 0 && !matches!(input.ty, Ty::F32 | Ty::F64) {
            return Err(refuse(inst, "modifies an integer operand"));
        }
    }
    if (clamp || omod != 0) && !outputs.iter().any(|o| matches!(o.ty(), Ty::F32 | Ty::F64)) {
        return Err(refuse(inst, "clamps or scales an integer result"));
    }
    Ok(())
}

fn plain_scalar(op: I) -> Option<I> {
    Some(match op {
        I::S_ADD_U32 => I::S_ADD_CO_U32,
        I::S_SUB_U32 => I::S_SUB_CO_U32,
        I::S_ADD_I32 => I::S_ADD_CO_I32,
        I::S_SUB_I32 => I::S_SUB_CO_I32,
        I::S_ADDC_U32 => I::S_ADD_CO_CI_U32,
        I::S_SUBB_U32 => I::S_SUB_CO_CI_U32,
        I::S_ANDN2_B32 => I::S_AND_NOT1_B32,
        I::S_ANDN2_B64 => I::S_AND_NOT1_B64,
        I::S_ORN2_B32 => I::S_OR_NOT1_B32,
        I::S_ORN2_B64 => I::S_OR_NOT1_B64,
        I::S_FF1_I32_B32 => I::S_CTZ_I32_B32,
        I::S_FF1_I32_B64 => I::S_CTZ_I32_B64,
        I::S_ANDN2_SAVEEXEC_B64 => I::S_AND_NOT1_SAVEEXEC_B64,
        I::S_ORN2_SAVEEXEC_B64 => I::S_OR_NOT1_SAVEEXEC_B64,
        I::S_CMP_NE_U64 => I::S_CMP_LG_U64,
        I::S_ADDK_I32 => I::S_ADDK_CO_I32,
        I::S_MAX_U32
        | I::S_MUL_I32
        | I::S_CSELECT_B32
        | I::S_CSELECT_B64
        | I::S_LSHL_B32
        | I::S_LSHL_B64
        | I::S_LSHR_B32
        | I::S_LSHR_B64
        | I::S_ASHR_I32
        | I::S_ASHR_I64
        | I::S_BFM_B32
        | I::S_BFE_U32
        | I::S_AND_B32
        | I::S_AND_B64
        | I::S_OR_B32
        | I::S_OR_B64
        | I::S_XOR_B32
        | I::S_XOR_B64
        | I::S_NAND_B32
        | I::S_NAND_B64
        | I::S_NOR_B32
        | I::S_NOR_B64
        | I::S_XNOR_B32
        | I::S_XNOR_B64
        | I::S_MOV_B32
        | I::S_MOV_B64
        | I::S_CMOV_B32
        | I::S_CMOV_B64
        | I::S_NOT_B32
        | I::S_NOT_B64
        | I::S_BREV_B32
        | I::S_BREV_B64
        | I::S_BCNT0_I32_B32
        | I::S_BCNT0_I32_B64
        | I::S_BCNT1_I32_B32
        | I::S_BCNT1_I32_B64
        | I::S_SEXT_I32_I16
        | I::S_AND_SAVEEXEC_B64
        | I::S_OR_SAVEEXEC_B64
        | I::S_XOR_SAVEEXEC_B64
        | I::S_NAND_SAVEEXEC_B64
        | I::S_NOR_SAVEEXEC_B64
        | I::S_XNOR_SAVEEXEC_B64
        | I::S_CMP_EQ_I32
        | I::S_CMP_LG_I32
        | I::S_CMP_GT_I32
        | I::S_CMP_GE_I32
        | I::S_CMP_LT_I32
        | I::S_CMP_LE_I32
        | I::S_CMP_EQ_U32
        | I::S_CMP_LG_U32
        | I::S_CMP_GT_U32
        | I::S_CMP_GE_U32
        | I::S_CMP_LT_U32
        | I::S_CMP_LE_U32
        | I::S_CMP_EQ_U64
        | I::S_MOVK_I32
        | I::S_CMOVK_I32
        | I::S_MULK_I32 => op,
        _ => return None,
    })
}

fn immediate_compare(op: I) -> Option<(I, bool)> {
    Some(match op {
        I::S_CMPK_EQ_I32 => (I::S_CMP_EQ_I32, true),
        I::S_CMPK_LG_I32 => (I::S_CMP_LG_I32, true),
        I::S_CMPK_GT_I32 => (I::S_CMP_GT_I32, true),
        I::S_CMPK_GE_I32 => (I::S_CMP_GE_I32, true),
        I::S_CMPK_LT_I32 => (I::S_CMP_LT_I32, true),
        I::S_CMPK_LE_I32 => (I::S_CMP_LE_I32, true),
        I::S_CMPK_EQ_U32 => (I::S_CMP_EQ_U32, false),
        I::S_CMPK_LG_U32 => (I::S_CMP_LG_U32, false),
        I::S_CMPK_GT_U32 => (I::S_CMP_GT_U32, false),
        I::S_CMPK_GE_U32 => (I::S_CMP_GE_U32, false),
        I::S_CMPK_LT_U32 => (I::S_CMP_LT_U32, false),
        I::S_CMPK_LE_U32 => (I::S_CMP_LE_U32, false),
        _ => return None,
    })
}

fn through(registry: &DialectRegistry, lanes: u32, inst: InstFormat) -> Lowered {
    let lowering = lift::instruction_with_registry(&inst, registry, lanes);
    Lowered::one(Some(inst), lowering)
}

fn scalar(inst: &Inst, state: &State, target: &Target) -> Result<Lowered, String> {
    let literal = inst.literal;
    let wide_literal = |codes: &[u16]| wide_integer(inst.op) && codes.contains(&LITERAL);
    match inst.form {
        Form::Sop1 { sdst, ssrc0 } => {
            let op = plain_scalar(inst.op).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
            if wide_literal(&[ssrc0]) {
                return Err(refuse(inst, "extends a literal to 64 bits"));
            }
            let format = InstFormat::SOP1(rdna::SOP1 {
                ssrc0: source(ssrc0, literal)?,
                op,
                sdst: scalar_dst(sdst)?,
            });
            Ok(through(target.registry, target.lanes, format))
        }
        Form::Sop2 { sdst, ssrc0, ssrc1 } => {
            let op = plain_scalar(inst.op).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
            if wide_literal(&[ssrc0, ssrc1]) {
                return Err(refuse(inst, "extends a literal to 64 bits"));
            }
            let format = InstFormat::SOP2(rdna::SOP2 {
                ssrc0: source(ssrc0, literal)?,
                ssrc1: source(ssrc1, literal)?,
                sdst: scalar_dst(sdst)?,
                op,
            });
            Ok(through(target.registry, target.lanes, format))
        }
        Form::Sopc { ssrc0, ssrc1 } => {
            let op = plain_scalar(inst.op).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
            if wide_literal(&[ssrc0, ssrc1]) {
                return Err(refuse(inst, "extends a literal to 64 bits"));
            }
            let format = InstFormat::SOPC(rdna::SOPC {
                ssrc0: source(ssrc0, literal)?,
                ssrc1: source(ssrc1, literal)?,
                op,
            });
            Ok(through(target.registry, target.lanes, format))
        }
        Form::Sopk { sdst, simm16 } => {
            if let Some((op, signed)) = immediate_compare(inst.op) {
                let value = if signed { simm16 as i16 as i32 as u32 } else { simm16 as u32 };
                let format = InstFormat::SOPC(rdna::SOPC {
                    ssrc0: source(sdst as u16, None)?,
                    ssrc1: SourceOperand::LiteralConstant(value),
                    op,
                });
                return Ok(through(target.registry, target.lanes, format));
            }
            match inst.op {
                I::S_SETREG_B32 | I::S_SETREG_IMM32_B32 => {
                    let (id, _, _) = hwreg(simm16);
                    if id != HW_REG_MODE {
                        return Err(refuse(inst, "writes a hardware register other than MODE"));
                    }
                    Ok(Lowered::default())
                }
                I::S_GETREG_B32 => {
                    let (id, offset, size) = hwreg(simm16);
                    let value = (id == HW_REG_MODE)
                        .then(|| state.mode.field(offset, size.min(32 - offset)))
                        .flatten()
                        .ok_or_else(|| refuse(inst, "reads a hardware register the program does not determine"))?;
                    let format = InstFormat::SOP1(rdna::SOP1 {
                        ssrc0: SourceOperand::LiteralConstant(value),
                        op: I::S_MOV_B32,
                        sdst: scalar_dst(sdst)?,
                    });
                    Ok(through(target.registry, target.lanes, format))
                }
                _ => {
                    let op = plain_scalar(inst.op).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
                    let format = InstFormat::SOPK(rdna::SOPK {
                        simm16,
                        sdst: scalar_dst(sdst)?,
                        op,
                    });
                    Ok(through(target.registry, target.lanes, format))
                }
            }
        }
        _ => unreachable!(),
    }
}

fn sopp(inst: &Inst) -> Result<Lowered, String> {
    match inst.op {
        I::S_NOP | I::S_WAITCNT | I::S_SETPRIO | I::S_SLEEP => Ok(Lowered::default()),
        I::S_BARRIER => Ok(Lowered {
            scan: vec![],
            lowerings: [EffectOp::BarrierSignal { is_first: false }, EffectOp::BarrierWait]
                .iter()
                .map(|&op| {
                    Lowering::Wave(YieldAction::new(
                        op,
                        vec![Operand::Source(SourceOperand::LiteralConstant(u32::MAX))],
                        vec![],
                    ))
                })
                .collect(),
        }),
        _ => Err(refuse(inst, "has no SPMD lowering")),
    }
}

fn smem(inst: &Inst, target: &Target) -> Result<Lowered, String> {
    let Form::Smem {
        sdata,
        sbase,
        offset,
        imm,
        glc,
    } = inst.form
    else {
        unreachable!()
    };
    let op = match inst.op {
        I::S_LOAD_DWORD => I::S_LOAD_B32,
        I::S_LOAD_DWORDX2 => I::S_LOAD_B64,
        I::S_LOAD_DWORDX4 => I::S_LOAD_B128,
        I::S_LOAD_DWORDX8 => I::S_LOAD_B256,
        I::S_LOAD_DWORDX16 => I::S_LOAD_B512,
        _ => return Err(refuse(inst, "has no SPMD lowering")),
    };
    let words = super::state::scalar_words(inst.op) as u16;
    if (sdata as u16..sdata as u16 + words).any(|r| r > 101) {
        return Err(refuse(inst, "loads into special scalar registers"));
    }
    let (ioffset, soffset) = if imm {
        (offset, 124)
    } else {
        (0, scalar_dst(offset as u8)?)
    };
    let format = InstFormat::SMEM(rdna::SMEM {
        op,
        sdata,
        sbase,
        ioffset,
        soffset,
        scope: if glc { 2 } else { 0 },
        th: 0,
    });
    Ok(through(target.registry, target.lanes, format))
}

struct Vector {
    vdst: u8,
    sdst: Option<u8>,
    src: [u16; 3],
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
}

fn vector_operands(inst: &Inst) -> Vector {
    let none = 128;
    match inst.form {
        Form::Vop1 { vdst, src0, .. } => Vector {
            vdst,
            sdst: None,
            src: [src0, none, none],
            abs: 0,
            neg: 0,
            clamp: false,
            omod: 0,
        },
        Form::Vop2 { vdst, src0, vsrc1, .. } => {
            let carry_in = matches!(inst.op, I::V_ADDC_U32 | I::V_SUBB_U32 | I::V_SUBBREV_U32 | I::V_CNDMASK_B32);
            Vector {
                vdst,
                sdst: carry(inst.op).then_some(VCC as u8),
                src: [src0, 256 + vsrc1 as u16, if carry_in { VCC } else { none }],
                abs: 0,
                neg: 0,
                clamp: false,
                omod: 0,
            }
        }
        Form::Vopc { src0, vsrc1, .. } => Vector {
            vdst: VCC as u8,
            sdst: None,
            src: [src0, 256 + vsrc1 as u16, none],
            abs: 0,
            neg: 0,
            clamp: false,
            omod: 0,
        },
        Form::Vop3 {
            vdst,
            sdst,
            src,
            abs,
            neg,
            clamp,
            omod,
        } => Vector {
            vdst,
            sdst,
            src,
            abs,
            neg,
            clamp,
            omod,
        },
        _ => unreachable!(),
    }
}

fn routed(op: I) -> Option<I> {
    Some(match op {
        I::V_ADD_U32 => I::V_ADD_CO_U32,
        I::V_SUB_U32 => I::V_SUB_CO_U32,
        I::V_SUBREV_U32 => I::V_SUBREV_CO_U32,
        I::V_ADDC_U32 => I::V_ADD_CO_CI_U32,
        I::V_SUBB_U32 => I::V_SUB_CO_CI_U32,
        I::V_SUBBREV_U32 => I::V_SUBREV_CO_CI_U32,
        I::V_MAD_U64_U32 => I::V_MAD_CO_U64_U32,
        I::V_FFBH_U32 => I::V_CLZ_I32_U32,
        I::V_MIN_F64 => I::V_MIN_NUM_F64,
        I::V_MAX_F64 => I::V_MAX_NUM_F64,
        I::V_CNDMASK_B32
        | I::V_ADD_F32
        | I::V_SUB_F32
        | I::V_SUBREV_F32
        | I::V_MUL_F32
        | I::V_MUL_I32_I24
        | I::V_MUL_U32_U24
        | I::V_MIN_F32
        | I::V_MAX_F32
        | I::V_MIN_I32
        | I::V_MAX_I32
        | I::V_MIN_U32
        | I::V_MAX_U32
        | I::V_LSHRREV_B32
        | I::V_ASHRREV_I32
        | I::V_LSHLREV_B32
        | I::V_AND_B32
        | I::V_OR_B32
        | I::V_XOR_B32
        | I::V_MOV_B32
        | I::V_READFIRSTLANE_B32
        | I::V_CVT_I32_F64
        | I::V_CVT_F64_I32
        | I::V_CVT_F32_I32
        | I::V_CVT_F32_U32
        | I::V_CVT_U32_F32
        | I::V_CVT_I32_F32
        | I::V_CVT_F32_F64
        | I::V_CVT_F64_F32
        | I::V_CVT_F32_UBYTE0
        | I::V_CVT_F32_UBYTE1
        | I::V_CVT_F32_UBYTE2
        | I::V_CVT_F32_UBYTE3
        | I::V_CVT_U32_F64
        | I::V_CVT_F64_U32
        | I::V_RCP_F32
        | I::V_RCP_IFLAG_F32
        | I::V_RCP_F64
        | I::V_RSQ_F32
        | I::V_RSQ_F64
        | I::V_SQRT_F32
        | I::V_SQRT_F64
        | I::V_FLOOR_F32
        | I::V_FLOOR_F64
        | I::V_CEIL_F32
        | I::V_TRUNC_F32
        | I::V_TRUNC_F64
        | I::V_RNDNE_F32
        | I::V_RNDNE_F64
        | I::V_FRACT_F64
        | I::V_FREXP_MANT_F32
        | I::V_FREXP_MANT_F64
        | I::V_FREXP_EXP_I32_F32
        | I::V_FREXP_EXP_I32_F64
        | I::V_EXP_F32
        | I::V_LOG_F32
        | I::V_SIN_F32
        | I::V_COS_F32
        | I::V_NOT_B32
        | I::V_BFREV_B32
        | I::V_MAD_I32_I24
        | I::V_MAD_U32_U24
        | I::V_BFE_U32
        | I::V_BFI_B32
        | I::V_FMA_F32
        | I::V_FMA_F64
        | I::V_ALIGNBIT_B32
        | I::V_MIN3_I32
        | I::V_MIN3_U32
        | I::V_MAX3_I32
        | I::V_MAX3_U32
        | I::V_MED3_I32
        | I::V_MED3_U32
        | I::V_DIV_FIXUP_F32
        | I::V_DIV_FIXUP_F64
        | I::V_DIV_SCALE_F32
        | I::V_DIV_SCALE_F64
        | I::V_DIV_FMAS_F32
        | I::V_DIV_FMAS_F64
        | I::V_PERM_B32
        | I::V_ADD_F64
        | I::V_MUL_F64
        | I::V_LDEXP_F64
        | I::V_LDEXP_F32
        | I::V_MUL_LO_U32
        | I::V_MUL_HI_U32
        | I::V_READLANE_B32
        | I::V_WRITELANE_B32
        | I::V_BCNT_U32_B32
        | I::V_MBCNT_LO_U32_B32
        | I::V_MBCNT_HI_U32_B32
        | I::V_LSHLREV_B64
        | I::V_LSHRREV_B64
        | I::V_ASHRREV_I64
        | I::V_TRIG_PREOP_F64 => op,
        I::V_CMP_F32(_)
        | I::V_CMP_F64(_)
        | I::V_CMP_I32(_)
        | I::V_CMP_U32(_)
        | I::V_CMP_I64(_)
        | I::V_CMP_U64(_)
        | I::V_CMP_I16(_)
        | I::V_CMP_U16(_)
        | I::V_CMP_CLASS_F32
        | I::V_CMP_CLASS_F64 => op,
        _ => return None,
    })
}

fn quiet(op: I) -> I {
    match op {
        I::V_CMPX_F32(p) => I::V_CMP_F32(p),
        I::V_CMPX_F64(p) => I::V_CMP_F64(p),
        I::V_CMPX_I32(p) => I::V_CMP_I32(p),
        I::V_CMPX_U32(p) => I::V_CMP_U32(p),
        I::V_CMPX_I64(p) => I::V_CMP_I64(p),
        I::V_CMPX_U64(p) => I::V_CMP_U64(p),
        I::V_CMPX_I16(p) => I::V_CMP_I16(p),
        I::V_CMPX_U16(p) => I::V_CMP_U16(p),
        I::V_CMPX_CLASS_F32 => I::V_CMP_CLASS_F32,
        I::V_CMPX_CLASS_F64 => I::V_CMP_CLASS_F64,
        op => op,
    }
}

fn wide_operands(op: I) -> [bool; 3] {
    match op {
        I::V_LSHLREV_B64 | I::V_LSHRREV_B64 | I::V_ASHRREV_I64 => [false, true, false],
        I::V_MAD_U64_U32 => [false, false, true],
        I::V_CMP_I64(_) | I::V_CMP_U64(_) | I::V_CMPX_I64(_) | I::V_CMPX_U64(_) => [true, true, false],
        _ => [false; 3],
    }
}

fn routed_lowering(inst: &Inst, v: &Vector, target: &Target) -> Result<(InstFormat, Lowering), String> {
    let op = routed(quiet(inst.op)).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
    let literal = inst.literal;
    for (k, &wide) in wide_operands(inst.op).iter().enumerate() {
        if wide && v.src[k] == LITERAL {
            return Err(refuse(inst, "extends a literal to 64 bits"));
        }
    }
    let [src0, src1, src2] = [source(v.src[0], literal)?, source(v.src[1], literal)?, source(v.src[2], literal)?];
    let format = if matches!(inst.op, I::V_READFIRSTLANE_B32) {
        InstFormat::VOP1(rdna::VOP1 {
            src0,
            op,
            vdst: scalar_dst(v.vdst)?,
        })
    } else if let Some(sdst) = v.sdst {
        if carry(inst.op) && v.clamp {
            return Err(refuse(inst, "saturates a carry"));
        }
        if sdst as u16 == M0 || sdst as u16 == EXEC {
            return Err(refuse(inst, "writes a lane mask into M0 or EXEC"));
        }
        InstFormat::VOP3SD(rdna::VOP3SD {
            vdst: v.vdst,
            sdst: scalar_dst(sdst)?,
            cm: v.clamp as u8,
            op,
            src0,
            src1,
            src2,
            omod: v.omod,
            neg: v.neg,
        })
    } else if compare(inst.op) {
        if v.vdst as u16 == M0 {
            return Err(refuse(inst, "writes a lane mask into M0"));
        }
        InstFormat::VOP3(rdna::VOP3 {
            vdst: scalar_dst(v.vdst)?,
            abs: v.abs,
            opsel: 0,
            cm: 0,
            op,
            src0,
            src1,
            src2: SourceOperand::IntegerConstant(0),
            omod: 0,
            neg: v.neg,
        })
    } else {
        let vdst = if matches!(inst.op, I::V_READLANE_B32) {
            scalar_dst(v.vdst)?
        } else {
            v.vdst
        };
        InstFormat::VOP3(rdna::VOP3 {
            vdst,
            abs: v.abs,
            opsel: 0,
            cm: v.clamp as u8,
            op,
            src0,
            src1,
            src2,
            omod: v.omod,
            neg: v.neg,
        })
    };
    let lowering = lift::instruction_with_registry(&format, target.registry, target.lanes);
    if !compare(inst.op) {
        checked_modifiers(inst, &lowering, v.abs, v.neg, v.clamp, v.omod)?;
    }
    Ok((format, lowering))
}

fn unfused(inst: &Inst, v: &Vector, registry: &DialectRegistry) -> Result<Lowering, String> {
    let literal = inst.literal;
    let k = || SourceOperand::LiteralConstant(literal.unwrap());
    let (a, b, c) = match (inst.op, inst.form) {
        (I::V_MAC_F32, _) => {
            if (v.abs | v.neg) & 4 != 0 {
                return Err(refuse(inst, "modifies its accumulator"));
            }
            (source(v.src[0], literal)?, source(v.src[1], literal)?, SourceOperand::VectorRegister(v.vdst))
        }
        (I::V_MAD_F32, _) => (source(v.src[0], literal)?, source(v.src[1], literal)?, source(v.src[2], literal)?),
        (I::V_MADMK_F32, _) => (source(v.src[0], None)?, k(), source(v.src[1], None)?),
        (I::V_MADAK_F32, _) => (source(v.src[0], None)?, source(v.src[1], None)?, k()),
        _ => unreachable!(),
    };
    let mut q = Builder::new(registry, vec![input(a, Ty::F32), input(b, Ty::F32), input(c, Ty::F32)]);
    let x = q.float_mod(Ty::F32, ValueId(0), v.abs, v.neg, 0);
    let y = q.float_mod(Ty::F32, ValueId(1), v.abs, v.neg, 1);
    let z = q.float_mod(Ty::F32, ValueId(2), v.abs, v.neg, 2);
    let product = q.push(Ty::F32, Op::Float(FloatOp::Mul, x, y));
    let sum = q.push(Ty::F32, Op::Float(FloatOp::Add, product, z));
    let result = q.output_mod(Ty::F32, sum, v.clamp as u8, v.omod);
    Ok(q.finish(Output::Vgpr(v.vdst as u32, Ty::F32), result))
}

fn halfword(op: I) -> bool {
    matches!(
        op,
        I::V_ADD_U16
            | I::V_SUB_U16
            | I::V_SUBREV_U16
            | I::V_MUL_LO_U16
            | I::V_LSHLREV_B16
            | I::V_LSHRREV_B16
            | I::V_ASHRREV_I16
            | I::V_MAX_U16
            | I::V_MAX_I16
            | I::V_MIN_U16
            | I::V_MIN_I16
            | I::V_MAD_U16
            | I::V_MAD_I16
    )
}

fn half_integer(inst: &Inst, v: &Vector, registry: &DialectRegistry) -> Result<Lowering, String> {
    if v.abs != 0 || v.neg != 0 || v.omod != 0 {
        return Err(refuse(inst, "modifies a 16-bit integer operation"));
    }
    let saturating = matches!(inst.op, I::V_ADD_U16 | I::V_SUB_U16 | I::V_SUBREV_U16 | I::V_MAD_U16 | I::V_MAD_I16);
    if v.clamp && !saturating {
        return Err(refuse(inst, "clamps a 16-bit result it cannot saturate"));
    }
    let mad = matches!(inst.op, I::V_MAD_U16 | I::V_MAD_I16);
    let count = if mad { 3 } else { 2 };
    let inputs = (0..count)
        .map(|k| Ok(input(source(v.src[k], inst.literal)?, Ty::I32)))
        .collect::<Result<Vec<_>, String>>()?;
    let mut q = Builder::new(registry, inputs);
    let mask = q.k(Ty::I32, 0xffff);
    let sixteen = q.k(Ty::I32, 16);
    let fifteen = q.k(Ty::I32, 15);
    let low = |q: &mut Builder, x: ValueId| q.int(IntOp::And, x, mask);
    let signed = |q: &mut Builder, x: ValueId| {
        let high = q.int(IntOp::Shl, x, sixteen);
        q.int(IntOp::AShr, high, sixteen)
    };
    let (a, b) = (ValueId(0), ValueId(1));
    let pick = |q: &mut Builder, pred: IntPred, x: ValueId, y: ValueId| {
        let cond = q.push(Ty::I1, Op::Cmp(pred, x, y));
        q.push(Ty::I32, Op::Select(cond, x, y))
    };
    let raw = match inst.op {
        I::V_ADD_U16 => {
            let (x, y) = (low(&mut q, a), low(&mut q, b));
            let sum = q.int(IntOp::Add, x, y);
            if v.clamp {
                pick(&mut q, IntPred::Ult, sum, mask)
            } else {
                sum
            }
        }
        I::V_SUB_U16 | I::V_SUBREV_U16 => {
            let (x, y) = if matches!(inst.op, I::V_SUB_U16) { (a, b) } else { (b, a) };
            let (x, y) = (low(&mut q, x), low(&mut q, y));
            let difference = q.int(IntOp::Sub, x, y);
            if v.clamp {
                let borrow = q.push(Ty::I1, Op::Cmp(IntPred::Ult, x, y));
                let zero = q.k(Ty::I32, 0);
                q.push(Ty::I32, Op::Select(borrow, zero, difference))
            } else {
                difference
            }
        }
        I::V_MUL_LO_U16 => q.int(IntOp::Mul, a, b),
        I::V_LSHLREV_B16 => {
            let amount = q.int(IntOp::And, a, fifteen);
            let x = low(&mut q, b);
            q.int(IntOp::Shl, x, amount)
        }
        I::V_LSHRREV_B16 => {
            let amount = q.int(IntOp::And, a, fifteen);
            let x = low(&mut q, b);
            q.int(IntOp::LShr, x, amount)
        }
        I::V_ASHRREV_I16 => {
            let amount = q.int(IntOp::And, a, fifteen);
            let x = signed(&mut q, b);
            q.int(IntOp::AShr, x, amount)
        }
        I::V_MAX_U16 | I::V_MIN_U16 => {
            let (x, y) = (low(&mut q, a), low(&mut q, b));
            pick(&mut q, if matches!(inst.op, I::V_MAX_U16) { IntPred::Ugt } else { IntPred::Ult }, x, y)
        }
        I::V_MAX_I16 | I::V_MIN_I16 => {
            let (x, y) = (signed(&mut q, a), signed(&mut q, b));
            pick(&mut q, if matches!(inst.op, I::V_MAX_I16) { IntPred::Sgt } else { IntPred::Slt }, x, y)
        }
        I::V_MAD_U16 => {
            let (x, y, z) = (low(&mut q, a), low(&mut q, b), low(&mut q, ValueId(2)));
            let product = q.int(IntOp::Mul, x, y);
            let sum = q.int(IntOp::Add, product, z);
            if v.clamp {
                pick(&mut q, IntPred::Ult, sum, mask)
            } else {
                sum
            }
        }
        I::V_MAD_I16 => {
            let (x, y, z) = (signed(&mut q, a), signed(&mut q, b), signed(&mut q, ValueId(2)));
            let product = q.int(IntOp::Mul, x, y);
            let sum = q.int(IntOp::Add, product, z);
            if v.clamp {
                let least = q.k(Ty::I32, i16::MIN as i32 as u32 as u64);
                let most = q.k(Ty::I32, i16::MAX as u64);
                let above = pick(&mut q, IntPred::Sgt, sum, least);
                pick(&mut q, IntPred::Slt, above, most)
            } else {
                sum
            }
        }
        _ => unreachable!(),
    };
    let result = low(&mut q, raw);
    Ok(q.finish(Output::Vgpr(v.vdst as u32, Ty::I32), result))
}

fn half_conversion(inst: &Inst, v: &Vector, target: &Target) -> Result<(InstFormat, Lowering), String> {
    let format = InstFormat::VOP3(rdna::VOP3 {
        vdst: v.vdst,
        abs: v.abs,
        opsel: 0,
        cm: v.clamp as u8,
        op: inst.op,
        src0: source(v.src[0], inst.literal)?,
        src1: SourceOperand::IntegerConstant(0),
        src2: SourceOperand::IntegerConstant(0),
        omod: v.omod,
        neg: v.neg,
    });
    let mut lowering = lift::instruction_with_registry(&format, target.registry, target.lanes);
    if matches!(inst.op, I::V_CVT_F16_F32) {
        let Lowering::TypedAlu { inputs, .. } = &mut lowering else {
            unreachable!()
        };
        inputs[1].source = InputSource::Operand(SourceOperand::IntegerConstant(0));
    }
    Ok((format, lowering))
}

fn half_fma(inst: &Inst, v: &Vector, registry: &DialectRegistry) -> Result<Lowering, String> {
    if v.omod != 0 {
        return Err(refuse(inst, "scales a half precision result"));
    }
    let inputs = (0..3)
        .map(|k| Ok(input(lift::half_source(source(v.src[k], inst.literal)?), Ty::I32)))
        .collect::<Result<Vec<_>, String>>()?;
    let mut q = Builder::new(registry, inputs);
    let widen = crate::rdna_spmd::rdna4::dialect::unary(registry, I::V_CVT_F32_F16).unwrap();
    let wide: Vec<ValueId> = (0..3)
        .map(|k| {
            let bits = q.bits_mod(Ty::I32, 16, ValueId(k), v.abs, v.neg, k);
            let single = q.target_one(widen, Arguments::Unary(bits));
            q.push(Ty::F64, Op::Convert(Cvt::FloatResizeRte, Ty::F64, single))
        })
        .collect();
    let product = q.push(Ty::F64, Op::Float(FloatOp::Mul, wide[0], wide[1]));
    let sum = q.push(Ty::F64, Op::Float(FloatOp::Add, product, wide[2]));
    let back = q.push(Ty::F64, Op::Float(FloatOp::Sub, sum, product));
    let first = q.push(Ty::F64, Op::Float(FloatOp::Sub, sum, back));
    let first = q.push(Ty::F64, Op::Float(FloatOp::Sub, product, first));
    let second = q.push(Ty::F64, Op::Float(FloatOp::Sub, wide[2], back));
    let error = q.push(Ty::F64, Op::Float(FloatOp::Add, first, second));
    let zero = q.k(Ty::F64, 0);
    let inexact = q.push(Ty::I1, Op::FCmp(FloatPred::One, error, zero));
    let bits = q.push(Ty::I64, Op::Convert(Cvt::Bitcast, Ty::I64, sum));
    let one = q.k(Ty::I64, 1);
    let parity = q.push(Ty::I64, Op::Int(IntOp::And, bits, one));
    let zero_bits = q.k(Ty::I64, 0);
    let even = q.push(Ty::I1, Op::Cmp(IntPred::Eq, parity, zero_bits));
    let error_positive = q.push(Ty::I1, Op::FCmp(FloatPred::Ogt, error, zero));
    let sum_positive = q.push(Ty::I1, Op::FCmp(FloatPred::Ogt, sum, zero));
    let outward = q.push(Ty::I1, Op::Int(IntOp::Xor, error_positive, sum_positive));
    let up = q.push(Ty::I64, Op::Int(IntOp::Add, bits, one));
    let down = q.push(Ty::I64, Op::Int(IntOp::Sub, bits, one));
    let moved = q.push(Ty::I64, Op::Select(outward, down, up));
    let adjust = q.push(Ty::I1, Op::Int(IntOp::And, inexact, even));
    let odd = q.push(Ty::I64, Op::Select(adjust, moved, bits));
    let odd = q.push(Ty::F64, Op::Convert(Cvt::Bitcast, Ty::F64, odd));
    let clamped = q.output_mod(Ty::F64, odd, v.clamp as u8, 0);
    let half = narrow_wide_half(&mut q, clamped);
    let mask = q.k(Ty::I32, 0xffff);
    let result = q.int(IntOp::And, half, mask);
    Ok(q.finish(Output::Vgpr(v.vdst as u32, Ty::I32), result))
}

fn with_exec(lowering: Lowering, registry: &DialectRegistry) -> Lowering {
    let Lowering::TypedAlu {
        inputs,
        mut outputs,
        scalar,
        expr,
    } = lowering
    else {
        unreachable!()
    };
    let mut expr = expr.expr().clone();
    let result = expr.results[0];
    outputs.push(Output::Compare(EXEC as u32));
    expr.results.push(result);
    Lowering::TypedAlu {
        inputs,
        outputs,
        scalar,
        expr: expr.verify_with(registry).expect("a compare with its EXEC write"),
    }
}

fn selected(q: &mut Builder, value: ValueId, sel: u8, sext: bool) -> Result<ValueId, String> {
    let (shift, bits) = match sel {
        0..=3 => (8 * sel as u64, 8),
        4 | 5 => (16 * (sel - 4) as u64, 16),
        6 => return Ok(value),
        _ => return Err("a reserved SDWA selection".to_string()),
    };
    let amount = q.k(Ty::I32, shift);
    let moved = q.int(IntOp::LShr, value, amount);
    let width = q.k(Ty::I32, 32 - bits);
    let high = q.int(IntOp::Shl, moved, width);
    Ok(q.int(if sext { IntOp::AShr } else { IntOp::LShr }, high, width))
}

fn sdwa(inst: &Inst, s: &Sdwa, v: &Vector, base: Lowering, registry: &DialectRegistry) -> Result<Lowering, String> {
    let Lowering::TypedAlu {
        inputs,
        outputs,
        scalar,
        expr,
    } = base
    else {
        return Err(refuse(inst, "has no SDWA form"));
    };
    if s.clamp || s.src0_neg || s.src0_abs || s.src1_neg || s.src1_abs {
        return Err(refuse(inst, "modifies an SDWA operand"));
    }
    let binary = !matches!(inst.form, Form::Vop1 { .. });
    let count = if binary { 2 } else { 1 };
    for k in 0..count {
        let expected = InputSource::Operand(source(v.src[k], None)?);
        if format!("{:?}", inputs[k].source) != format!("{:?}", expected) || inputs[k].ty != Ty::I32 {
            return Err(refuse(inst, "has an SDWA operand of another type"));
        }
    }
    let partial = s.dst_sel != 6 && !matches!(inst.form, Form::Vopc { .. });
    let preserve = partial && s.dst_unused == 2;
    if s.dst_unused == 3 {
        return Err(refuse(inst, "uses a reserved SDWA destination format"));
    }
    let mut given = inputs.clone();
    if preserve {
        given.push(input(SourceOperand::VectorRegister(v.vdst), Ty::I32));
    }
    let old = ValueId(inputs.len());
    let mut q = Builder::new(registry, given);
    let mut map: Vec<ValueId> = (0..inputs.len()).map(ValueId).collect();
    map[0] = selected(&mut q, ValueId(0), s.src0_sel, s.src0_sext)?;
    if binary {
        map[1] = selected(&mut q, ValueId(1), s.src1_sel, s.src1_sext)?;
    }
    for inst in &expr.expr().insts {
        match inst {
            ExprInst::Core(ty, op) => {
                let value = q.push(*ty, op.map(|x| map[x.0]));
                map.push(value);
            }
            ExprInst::Target { op, args, .. } => {
                let values = q.target(*op, args.map(|x| map[x.0]));
                map.extend(values);
            }
        }
    }
    let mut results = vec![];
    for (&output, result) in outputs.iter().zip(&expr.expr().results) {
        let value = map[result.0];
        let (output, value) = match output {
            Output::Vgpr(r, Ty::I32) if partial => {
                let (shift, bits) = match s.dst_sel {
                    0..=3 => (8 * s.dst_sel as u64, 8),
                    _ => (16 * (s.dst_sel - 4) as u64, 16),
                };
                let width = q.k(Ty::I32, 32 - bits);
                let high = q.int(IntOp::Shl, value, width);
                let field = q.int(if s.dst_unused == 1 { IntOp::AShr } else { IntOp::LShr }, high, width);
                let low_mask = q.k(Ty::I32, (1u64 << bits) - 1);
                let amount = q.k(Ty::I32, shift);
                let placed = match s.dst_unused {
                    1 => q.int(IntOp::Shl, field, amount),
                    _ => {
                        let clean = q.int(IntOp::And, field, low_mask);
                        q.int(IntOp::Shl, clean, amount)
                    }
                };
                let value = if preserve {
                    let hole = q.k(Ty::I32, !(((1u64 << bits) - 1) << shift) & 0xffff_ffff);
                    let kept = q.int(IntOp::And, old, hole);
                    q.int(IntOp::Or, kept, placed)
                } else {
                    placed
                };
                (Output::Vgpr(r, Ty::I32), value)
            }
            Output::Vgpr(_, ty) if partial && ty != Ty::I32 => return Err(refuse(inst, "writes part of a non-integer result")),
            other => (other, value),
        };
        results.push((output, value));
    }
    Ok(q.finish_many(scalar, results))
}

fn vector(inst: &Inst, state: &State, target: &Target) -> Result<Lowered, String> {
    let registry = target.registry;
    let denormals = denormals(inst, state)?;
    let v = vector_operands(inst);
    let ext = match inst.form {
        Form::Vop1 { ext, .. } | Form::Vop2 { ext, .. } | Form::Vopc { ext, .. } => ext,
        _ => Ext::Plain,
    };
    let mut always = false;
    let (scan, lowering) = match inst.op {
        I::V_MAC_F32 | I::V_MAD_F32 | I::V_MADMK_F32 | I::V_MADAK_F32 => {
            always = true;
            (None, unfused(inst, &v, registry)?)
        }
        op if halfword(op) => (None, half_integer(inst, &v, registry)?),
        I::V_CVT_F16_F32 | I::V_CVT_F32_F16 => {
            let (format, lowering) = half_conversion(inst, &v, target)?;
            (Some(format), lowering)
        }
        I::V_FMA_F16 => (None, half_fma(inst, &v, registry)?),
        _ => {
            let (format, lowering) = routed_lowering(inst, &v, target)?;
            (Some(format), lowering)
        }
    };
    let lowering = match ext {
        Ext::Plain => lowering,
        Ext::Sdwa(s) => sdwa(inst, &s, &v, lowering, registry)?,
        Ext::Dpp(_) => return Err(refuse(inst, "uses DPP")),
    };
    let lowering = if compare_exec(inst.op) {
        with_exec(lowering, registry)
    } else {
        lowering
    };
    let lowering = flushed(
        lowering,
        registry,
        always || denormals.inputs,
        always || denormals.outputs,
    );
    Ok(Lowered {
        scan: if ext == Ext::Plain { scan.into_iter().collect() } else { vec![] },
        lowerings: vec![lowering],
    })
}

fn global(inst: &Inst) -> Option<I> {
    Some(match inst.op {
        I::FLAT_LOAD_UBYTE => I::GLOBAL_LOAD_U8,
        I::FLAT_LOAD_SBYTE => I::GLOBAL_LOAD_I8,
        I::FLAT_LOAD_USHORT => I::GLOBAL_LOAD_U16,
        I::FLAT_LOAD_SSHORT => I::GLOBAL_LOAD_I16,
        I::FLAT_LOAD_DWORD => I::GLOBAL_LOAD_B32,
        I::FLAT_LOAD_DWORDX2 => I::GLOBAL_LOAD_B64,
        I::FLAT_LOAD_DWORDX3 => I::GLOBAL_LOAD_B96,
        I::FLAT_LOAD_DWORDX4 => I::GLOBAL_LOAD_B128,
        I::FLAT_STORE_BYTE => I::GLOBAL_STORE_B8,
        I::FLAT_STORE_SHORT => I::GLOBAL_STORE_B16,
        I::FLAT_STORE_DWORD => I::GLOBAL_STORE_B32,
        I::FLAT_STORE_DWORDX2 => I::GLOBAL_STORE_B64,
        I::FLAT_STORE_DWORDX3 => I::GLOBAL_STORE_B96,
        I::FLAT_STORE_DWORDX4 => I::GLOBAL_STORE_B128,
        I::FLAT_ATOMIC_ADD => I::GLOBAL_ATOMIC_ADD_U32,
        I::FLAT_ATOMIC_SMIN => I::GLOBAL_ATOMIC_MIN_I32,
        I::FLAT_ATOMIC_SMAX => I::GLOBAL_ATOMIC_MAX_I32,
        I::FLAT_ATOMIC_UMIN => I::GLOBAL_ATOMIC_MIN_U32,
        I::FLAT_ATOMIC_UMAX => I::GLOBAL_ATOMIC_MAX_U32,
        I::FLAT_ATOMIC_CMPSWAP => I::GLOBAL_ATOMIC_CMPSWAP_B32,
        _ => return None,
    })
}

fn flat(inst: &Inst, target: &Target) -> Result<Lowered, String> {
    let Form::Flat {
        glc,
        slc,
        addr,
        data,
        tfe,
        vdst,
    } = inst.form
    else {
        unreachable!()
    };
    let op = global(inst).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
    if tfe {
        return Err(refuse(inst, "reports texture faults"));
    }
    let atomic = format!("{:?}", inst.op).contains("ATOMIC");
    let (scope, th) = if atomic {
        (2, glc as u8 | (slc as u8) << 1)
    } else {
        (if glc { 2 } else { 0 }, slc as u8)
    };
    let format = InstFormat::VGLOBAL(rdna::VGLOBAL {
        op,
        vaddr: addr,
        vsrc: data,
        vdst,
        scope,
        th,
        ioffset: 0,
        saddr: 124,
        sve: 0,
    });
    Ok(through(target.registry, target.lanes, format))
}

fn local(op: I) -> Option<I> {
    Some(match op {
        I::DS_READ_B32 => I::DS_LOAD_B32,
        I::DS_READ_B64 => I::DS_LOAD_B64,
        I::DS_READ_B96 => I::DS_LOAD_B96,
        I::DS_READ_B128 => I::DS_LOAD_B128,
        I::DS_READ_U8 => I::DS_LOAD_U8,
        I::DS_READ_I8 => I::DS_LOAD_I8,
        I::DS_READ_U16 => I::DS_LOAD_U16,
        I::DS_READ_I16 => I::DS_LOAD_I16,
        I::DS_READ2_B32 => I::DS_LOAD_2ADDR_B32,
        I::DS_READ2_B64 => I::DS_LOAD_2ADDR_B64,
        I::DS_READ2ST64_B32 => I::DS_LOAD_2ADDR_STRIDE64_B32,
        I::DS_READ2ST64_B64 => I::DS_LOAD_2ADDR_STRIDE64_B64,
        I::DS_WRITE_B8 => I::DS_STORE_B8,
        I::DS_WRITE_B16 => I::DS_STORE_B16,
        I::DS_WRITE_B32 => I::DS_STORE_B32,
        I::DS_WRITE_B64 => I::DS_STORE_B64,
        I::DS_WRITE_B96 => I::DS_STORE_B96,
        I::DS_WRITE_B128 => I::DS_STORE_B128,
        I::DS_WRITE2_B32 => I::DS_STORE_2ADDR_B32,
        I::DS_WRITE2_B64 => I::DS_STORE_2ADDR_B64,
        I::DS_WRITE2ST64_B32 => I::DS_STORE_2ADDR_STRIDE64_B32,
        I::DS_WRITE2ST64_B64 => I::DS_STORE_2ADDR_STRIDE64_B64,
        I::DS_ADD_U32 | I::DS_ADD_RTN_U32 | I::DS_ADD_F32 | I::DS_BPERMUTE_B32 => op,
        _ => return None,
    })
}

fn ds(inst: &Inst, state: &State, target: &Target) -> Result<Lowered, String> {
    let Form::Ds {
        offset0,
        offset1,
        gds,
        addr,
        data0,
        data1,
        vdst,
    } = inst.form
    else {
        unreachable!()
    };
    let op = local(inst.op).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
    if gds {
        return Err(refuse(inst, "accesses the global data share"));
    }
    if !matches!(inst.op, I::DS_BPERMUTE_B32) {
        match state.sgprs[M0 as usize] {
            Value::Known(m0) if m0 & 0x1_ffff >= 0x1_0000 => {}
            _ => return Err(refuse(inst, "runs while M0 may clamp the local data share")),
        }
    }
    let format = InstFormat::DS(rdna::DS {
        offset0,
        offset1,
        op,
        addr,
        data0,
        data1,
        vdst,
    });
    Ok(through(target.registry, target.lanes, format))
}

fn private(op: I) -> Option<I> {
    Some(match op {
        I::BUFFER_LOAD_UBYTE => I::SCRATCH_LOAD_U8,
        I::BUFFER_LOAD_SBYTE => I::SCRATCH_LOAD_I8,
        I::BUFFER_LOAD_USHORT => I::SCRATCH_LOAD_U16,
        I::BUFFER_LOAD_SSHORT => I::SCRATCH_LOAD_I16,
        I::BUFFER_LOAD_DWORD => I::SCRATCH_LOAD_B32,
        I::BUFFER_LOAD_DWORDX2 => I::SCRATCH_LOAD_B64,
        I::BUFFER_LOAD_DWORDX3 => I::SCRATCH_LOAD_B96,
        I::BUFFER_LOAD_DWORDX4 => I::SCRATCH_LOAD_B128,
        I::BUFFER_STORE_BYTE => I::SCRATCH_STORE_B8,
        I::BUFFER_STORE_SHORT => I::SCRATCH_STORE_B16,
        I::BUFFER_STORE_DWORD => I::SCRATCH_STORE_B32,
        I::BUFFER_STORE_DWORDX2 => I::SCRATCH_STORE_B64,
        I::BUFFER_STORE_DWORDX3 => I::SCRATCH_STORE_B96,
        I::BUFFER_STORE_DWORDX4 => I::SCRATCH_STORE_B128,
        _ => return None,
    })
}

fn mubuf(inst: &Inst, state: &State, target: &Target) -> Result<Lowered, String> {
    let Form::Mubuf {
        offset,
        offen,
        idxen,
        lds,
        vaddr,
        vdata,
        srsrc,
        tfe,
        soffset,
        ..
    } = inst.form
    else {
        unreachable!()
    };
    let op = private(inst.op).ok_or_else(|| refuse(inst, "has no SPMD lowering"))?;
    if idxen || lds || tfe {
        return Err(refuse(inst, "indexes, loads into LDS or reports faults"));
    }
    let descriptor = target
        .descriptor
        .ok_or_else(|| refuse(inst, "addresses a buffer without the private segment buffer"))?;
    let first = srsrc as usize * 4;
    let entry = (0..4).all(|k| state.sgprs[first + k] == Value::Entry(descriptor + k as u16));
    if !entry {
        return Err(refuse(inst, "addresses a buffer other than the private segment"));
    }
    let shift = match state.operand(soffset, None) {
        Value::Known(bytes) if bytes % 256 == 0 => bytes / 64,
        _ => return Err(refuse(inst, "offsets the private segment by an unknown or unaligned amount")),
    };
    let ioffset = offset as u32 + shift;
    if ioffset >= 1 << 23 {
        return Err(refuse(inst, "offsets the private segment beyond the signed 24-bit range"));
    }
    let format = InstFormat::VSCRATCH(rdna::VSCRATCH {
        op,
        vaddr,
        vsrc: vdata,
        vdst: vdata,
        scope: 0,
        th: 0,
        ioffset,
        saddr: 124,
        sve: offen as u8,
    });
    Ok(through(target.registry, target.lanes, format))
}

fn image(inst: &Inst, target: &Target) -> Result<Lowered, String> {
    let Form::Mimg {
        dmask,
        unorm,
        da,
        r128,
        tfe,
        lwe,
        vaddr,
        vdata,
        srsrc,
        ssamp,
        d16,
        ..
    } = inst.form
    else {
        unreachable!()
    };
    if !matches!(inst.op, I::IMAGE_SAMPLE_LZ) {
        return Err(refuse(inst, "has no SPMD lowering"));
    }
    if tfe || lwe {
        return Err(refuse(inst, "reports texture faults"));
    }
    if d16 {
        return Err(refuse(inst, "returns half-precision data"));
    }
    if r128 {
        return Err(refuse(inst, "takes a 128-bit resource, which holds no pitch"));
    }
    if da {
        return Err(refuse(inst, "samples a slice of an array"));
    }
    let register = |code: u16| match source(code, None)? {
        SourceOperand::ScalarRegister(r) if r < 102 => Ok(r),
        _ => Err(refuse(inst, "takes a resource from outside the scalar registers")),
    };
    let mut rsrc = [0u8; 8];
    for (k, r) in rsrc.iter_mut().enumerate() {
        *r = register(srsrc as u16 * 4 + k as u16)?;
    }
    let mut samp = [0u8; 4];
    for (k, r) in samp.iter_mut().enumerate() {
        *r = register(ssamp as u16 * 4 + k as u16)?;
    }
    let op = crate::rdna_spmd::rdna4::dialect::image_sample_gcn3(target.registry);
    let lowering = lift::image_sample(target.registry, op, rsrc, samp, [vaddr, vaddr + 1], vdata, dmask, unorm);
    Ok(Lowered::one(None, lowering))
}

pub fn link(register: u8, address: usize, target: &Target) -> Result<Lowered, String> {
    let format = InstFormat::SOP1(rdna::SOP1 {
        ssrc0: SourceOperand::IntegerConstant(address as u64),
        op: I::S_MOV_B64,
        sdst: scalar_dst(register)?,
    });
    Ok(through(target.registry, target.lanes, format))
}

pub fn lower(pc: usize, inst: &Inst, state: &State, target: &Target) -> Result<Lowered, String> {
    if let (I::S_GETPC_B64, Form::Sop1 { sdst, .. }) = (inst.op, inst.form) {
        return link(sdst, pc + inst.size, target);
    }
    match inst.form {
        Form::Sopp { .. } => sopp(inst),
        Form::Sop1 { .. } | Form::Sop2 { .. } | Form::Sopc { .. } | Form::Sopk { .. } => scalar(inst, state, target),
        Form::Smem { .. } => smem(inst, target),
        Form::Vop1 { .. } | Form::Vop2 { .. } | Form::Vopc { .. } | Form::Vop3 { .. } => vector(inst, state, target),
        Form::Ds { .. } => ds(inst, state, target),
        Form::Flat { .. } => flat(inst, target),
        Form::Mubuf { .. } => mubuf(inst, state, target),
        Form::Mimg { .. } => image(inst, target),
    }
}
