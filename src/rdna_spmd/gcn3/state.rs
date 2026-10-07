use super::decode::{memory_words, vector_words, Form, Inst, LITERAL};
use crate::instructions::I;
use std::collections::BTreeMap;

pub const M0: u16 = 124;
pub const VCC: u16 = 106;
pub const EXEC: u16 = 126;

fn name(op: I) -> String {
    format!("{:?}", op)
}

fn wide_result(op: I) -> bool {
    let n = name(op);
    !(n.contains("_I32_B64") || n.contains("_I32_I64") || n.starts_with("S_BITCMP"))
        && (n.ends_with("_B64") || n.ends_with("_U64") || n.ends_with("_I64"))
}

pub fn compare(op: I) -> bool {
    name(op).starts_with("V_CMP")
}

pub fn compare_exec(op: I) -> bool {
    name(op).starts_with("V_CMPX")
}

pub fn saveexec(op: I) -> bool {
    name(op).contains("_SAVEEXEC_")
}

pub fn carry(op: I) -> bool {
    matches!(
        op,
        I::V_ADD_U32 | I::V_SUB_U32 | I::V_SUBREV_U32 | I::V_ADDC_U32 | I::V_SUBB_U32 | I::V_SUBBREV_U32
    )
}

pub fn scalar_words(op: I) -> u32 {
    match op {
        I::S_LOAD_DWORDX2 | I::S_BUFFER_LOAD_DWORDX2 | I::S_MEMTIME | I::S_MEMREALTIME => 2,
        I::S_LOAD_DWORDX4 | I::S_BUFFER_LOAD_DWORDX4 => 4,
        I::S_LOAD_DWORDX8 | I::S_BUFFER_LOAD_DWORDX8 => 8,
        I::S_LOAD_DWORDX16 | I::S_BUFFER_LOAD_DWORDX16 => 16,
        _ => 1,
    }
}

pub fn scalar_writes(inst: &Inst) -> Vec<u16> {
    let span = |first: u8, words: u32| (0..words).map(|k| first as u16 + k as u16).collect::<Vec<_>>();
    let pair = |first: u8| span(first, 2);
    let mut writes = match inst.form {
        Form::Sop1 { sdst, .. } => match inst.op {
            I::S_SETPC_B64 | I::S_CBRANCH_JOIN | I::S_RFE_B64 => vec![],
            I::S_SET_GPR_IDX_IDX => vec![M0],
            op if saveexec(op) => pair(sdst),
            op if wide_result(op) => pair(sdst),
            _ => span(sdst, 1),
        },
        Form::Sop2 { sdst, .. } => match inst.op {
            op if wide_result(op) => pair(sdst),
            _ => span(sdst, 1),
        },
        Form::Sopk { sdst, .. } => match inst.op {
            I::S_MOVK_I32 | I::S_CMOVK_I32 | I::S_ADDK_I32 | I::S_MULK_I32 | I::S_GETREG_B32 => span(sdst, 1),
            _ => vec![],
        },
        Form::Sopc { .. } => match inst.op {
            I::S_SET_GPR_IDX_ON => vec![M0],
            _ => vec![],
        },
        Form::Sopp { .. } => match inst.op {
            I::S_SET_GPR_IDX_MODE => vec![M0],
            _ => vec![],
        },
        Form::Smem { sdata, .. } => {
            if name(inst.op).contains("STORE") || name(inst.op).contains("DCACHE") || name(inst.op).contains("ATC") {
                vec![]
            } else {
                span(sdata, scalar_words(inst.op))
            }
        }
        Form::Vop1 { vdst, .. } if matches!(inst.op, I::V_READFIRSTLANE_B32) => span(vdst, 1),
        Form::Vop1 { .. } => vec![],
        Form::Vop2 { .. } if carry(inst.op) => pair(VCC as u8),
        Form::Vop2 { .. } => vec![],
        Form::Vopc { .. } => pair(VCC as u8),
        Form::Vop3 { vdst, sdst, .. } => match (inst.op, sdst) {
            (_, Some(sdst)) => pair(sdst),
            (I::V_READLANE_B32 | I::V_READFIRSTLANE_B32, _) => span(vdst, 1),
            (op, _) if compare(op) => pair(vdst),
            _ => vec![],
        },
        Form::Ds { .. } | Form::Mubuf { .. } | Form::Mimg { .. } | Form::Flat { .. } => vec![],
    };
    if saveexec(inst.op) || compare_exec(inst.op) {
        writes.extend([EXEC, EXEC + 1]);
    }
    writes.sort_unstable();
    writes.dedup();
    writes
}

pub fn vector_writes(inst: &Inst) -> Option<Vec<u16>> {
    let n = name(inst.op);
    if n.starts_with("V_MOVREL") || n.starts_with("S_SET_GPR_IDX") {
        return None;
    }
    let span = |first: u8, words: u32| Some((0..words).map(|k| first as u16 + k as u16).collect());
    match inst.form {
        Form::Vop1 { vdst, .. } => match inst.op {
            I::V_READFIRSTLANE_B32 | I::V_NOP | I::V_CLREXCP => Some(vec![]),
            op => span(vdst, vector_words(op).0),
        },
        Form::Vop2 { vdst, .. } => span(vdst, vector_words(inst.op).0),
        Form::Vop3 { vdst, .. } => match inst.op {
            op if compare(op) => Some(vec![]),
            I::V_READLANE_B32 | I::V_READFIRSTLANE_B32 => Some(vec![]),
            op => span(vdst, vector_words(op).0),
        },
        Form::Ds { vdst, .. } => {
            let returns = ["READ", "RTN", "SWIZZLE", "PERMUTE", "CONSUME", "APPEND", "ORDERED"].iter().any(|k| n.contains(k));
            let pairs = n.contains("READ2") || n.contains("XCHG2");
            match returns {
                true => span(vdst, memory_words(inst.op) * if pairs { 2 } else { 1 }),
                false => Some(vec![]),
            }
        }
        Form::Flat { vdst, tfe, .. } if !n.contains("STORE") => span(vdst, memory_words(inst.op) + tfe as u32),
        Form::Mubuf { vdata, tfe, .. } if !n.contains("STORE") => span(vdata, memory_words(inst.op) + tfe as u32),
        Form::Mimg { vdata, tfe, .. } if !n.contains("STORE") => span(vdata, 4 + tfe as u32),
        _ => Some(vec![]),
    }
}

pub fn writes_exec(inst: &Inst) -> bool {
    scalar_writes(inst).iter().any(|&r| r == EXEC || r == EXEC + 1)
}

pub fn writes_scc(inst: &Inst) -> bool {
    match inst.form {
        Form::Sop1 { .. } => !matches!(
            inst.op,
            I::S_MOV_B32
                | I::S_MOV_B64
                | I::S_CMOV_B32
                | I::S_CMOV_B64
                | I::S_BREV_B32
                | I::S_BREV_B64
                | I::S_FF0_I32_B32
                | I::S_FF0_I32_B64
                | I::S_FF1_I32_B32
                | I::S_FF1_I32_B64
                | I::S_FLBIT_I32_B32
                | I::S_FLBIT_I32_B64
                | I::S_FLBIT_I32
                | I::S_FLBIT_I32_I64
                | I::S_SEXT_I32_I8
                | I::S_SEXT_I32_I16
                | I::S_BITSET0_B32
                | I::S_BITSET0_B64
                | I::S_BITSET1_B32
                | I::S_BITSET1_B64
                | I::S_GETPC_B64
                | I::S_SETPC_B64
                | I::S_SWAPPC_B64
        ),
        Form::Sop2 { .. } => !matches!(inst.op, I::S_CSELECT_B32 | I::S_CSELECT_B64 | I::S_BFM_B32 | I::S_BFM_B64 | I::S_MUL_I32),
        Form::Sopk { .. } => matches!(inst.op, I::S_ADDK_I32) || name(inst.op).starts_with("S_CMPK"),
        Form::Sopc { .. } => !matches!(inst.op, I::S_SETVSKIP | I::S_SET_GPR_IDX_ON),
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Unknown,
    Known(u32),
    Entry(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bits {
    pub known: u32,
    pub value: u32,
}

impl Bits {
    pub fn field(&self, offset: u32, size: u32) -> Option<u32> {
        let mask = if size == 32 { u32::MAX } else { (1 << size) - 1 } << offset;
        (self.known & mask == mask).then_some((self.value & mask) >> offset)
    }

    fn set(&mut self, offset: u32, size: u32, value: Option<u32>) {
        let mask = if size == 32 { u32::MAX } else { (1 << size) - 1 } << offset;
        match value {
            Some(v) => {
                self.known |= mask;
                self.value = self.value & !mask | v << offset & mask;
            }
            None => self.known &= !mask,
        }
    }

    fn merge(self, other: Bits) -> Bits {
        let known = self.known & other.known & !(self.value ^ other.value);
        Bits {
            known,
            value: self.value & known,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub sgprs: [Value; 128],
    pub scc: Value,
    pub mode: Bits,
    pub lanes: BTreeMap<(u16, u8), Value>,
}

pub const HW_REG_MODE: u16 = 1;

pub fn hwreg(simm16: u16) -> (u16, u32, u32) {
    (simm16 & 63, (simm16 >> 6 & 31) as u32, ((simm16 >> 11) + 1) as u32)
}

impl State {
    pub fn entry(mode: u32, known: Vec<(u16, u32)>) -> State {
        let mut sgprs = [Value::Unknown; 128];
        for (r, v) in sgprs.iter_mut().enumerate() {
            *v = Value::Entry(r as u16);
        }
        sgprs[M0 as usize] = Value::Unknown;
        for (r, value) in known {
            sgprs[r as usize] = Value::Known(value);
        }
        State {
            sgprs,
            scc: Value::Unknown,
            mode: Bits {
                known: 0x3ff,
                value: mode & 0x3ff,
            },
            lanes: BTreeMap::new(),
        }
    }

    pub fn operand(&self, code: u16, literal: Option<u32>) -> Value {
        match code {
            0..=127 => self.sgprs[code as usize],
            128..=192 => Value::Known((code - 128) as u32),
            193..=208 => Value::Known((192i32 - code as i32) as u32),
            240..=247 => Value::Known([0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][(code - 240) as usize].to_bits()),
            248 => Value::Known(0x3e22_f983),
            LITERAL => literal.map_or(Value::Unknown, Value::Known),
            _ => Value::Unknown,
        }
    }

    fn wide_operand(&self, code: u16) -> [Value; 2] {
        match code {
            0..=126 => [self.sgprs[code as usize], self.sgprs[code as usize + 1]],
            128..=192 => [Value::Known((code - 128) as u32), Value::Known(0)],
            193..=208 => [Value::Known((192i32 - code as i32) as u32), Value::Known(u32::MAX)],
            _ => [Value::Unknown; 2],
        }
    }

    pub fn merge(&self, other: &State) -> State {
        let mut merged = self.clone();
        for (a, b) in merged.sgprs.iter_mut().zip(&other.sgprs) {
            if *a != *b {
                *a = Value::Unknown;
            }
        }
        if merged.scc != other.scc {
            merged.scc = Value::Unknown;
        }
        merged.mode = self.mode.merge(other.mode);
        merged.lanes.retain(|slot, value| other.lanes.get(slot) == Some(value));
        merged
    }

    fn write(&mut self, r: u16, value: Value) {
        if (r as usize) < self.sgprs.len() {
            self.sgprs[r as usize] = value;
        }
    }

    pub fn step(&mut self, pc: usize, inst: &Inst) {
        let writes = scalar_writes(inst);
        let scc = writes_scc(inst);
        let previous = self.clone();
        for &r in &writes {
            self.write(r, Value::Unknown);
        }
        if scc {
            self.scc = Value::Unknown;
        }
        let value = |code: u16| previous.operand(code, inst.literal);
        if !matches!(inst.op, I::V_WRITELANE_B32) {
            match vector_writes(inst) {
                None => self.lanes.clear(),
                Some(written) => self.lanes.retain(|&(v, _), _| !written.contains(&v)),
            }
        }
        match (inst.op, inst.form) {
            (I::V_WRITELANE_B32, Form::Vop3 { vdst, src, .. }) => match value(src[1]) {
                Value::Known(lane) => {
                    self.lanes.insert((vdst as u16, (lane & 63) as u8), value(src[0]));
                }
                _ => self.lanes.retain(|&(v, _), _| v != vdst as u16),
            },
            (I::V_READLANE_B32, Form::Vop3 { vdst, src, .. }) => {
                let read = match (src[0], value(src[1])) {
                    (256..=511, Value::Known(lane)) => previous.lanes.get(&(src[0] - 256, (lane & 63) as u8)).copied(),
                    _ => None,
                };
                self.write(vdst as u16, read.unwrap_or(Value::Unknown));
            }
            (I::S_GETPC_B64 | I::S_SWAPPC_B64, Form::Sop1 { sdst, .. }) => {
                let next = (pc + inst.size) as u64;
                self.write(sdst as u16, Value::Known(next as u32));
                self.write(sdst as u16 + 1, Value::Known((next >> 32) as u32));
            }
            (I::S_MOV_B32, Form::Sop1 { sdst, ssrc0 }) => self.write(sdst as u16, value(ssrc0)),
            (I::S_MOV_B64, Form::Sop1 { sdst, ssrc0 }) => {
                let [low, high] = previous.wide_operand(ssrc0);
                self.write(sdst as u16, low);
                self.write(sdst as u16 + 1, high);
            }
            (I::S_CMOV_B32, Form::Sop1 { sdst, ssrc0 }) => {
                let chosen = match previous.scc {
                    Value::Known(0) => previous.sgprs[sdst as usize],
                    Value::Known(_) => value(ssrc0),
                    _ if value(ssrc0) == previous.sgprs[sdst as usize] => value(ssrc0),
                    _ => Value::Unknown,
                };
                self.write(sdst as u16, chosen);
            }
            (I::S_MOVK_I32, Form::Sopk { sdst, simm16 }) => self.write(sdst as u16, Value::Known(simm16 as i16 as i32 as u32)),
            (I::S_ADD_U32 | I::S_ADDC_U32, Form::Sop2 { sdst, ssrc0, ssrc1 }) => {
                let carry = if matches!(inst.op, I::S_ADDC_U32) { previous.scc } else { Value::Known(0) };
                let (result, out) = match (value(ssrc0), value(ssrc1), carry) {
                    (Value::Known(a), Value::Known(b), Value::Known(c)) => {
                        let sum = a as u64 + b as u64 + c as u64;
                        (Value::Known(sum as u32), Value::Known((sum >> 32) as u32))
                    }
                    (x, Value::Known(0), Value::Known(0)) | (Value::Known(0), x, Value::Known(0)) => (x, Value::Known(0)),
                    _ => (Value::Unknown, Value::Unknown),
                };
                self.write(sdst as u16, result);
                self.scc = out;
            }
            (I::S_ADD_I32 | I::S_SUB_I32 | I::S_SUB_U32, Form::Sop2 { sdst, ssrc0, ssrc1 }) => {
                if let (Value::Known(a), Value::Known(b)) = (value(ssrc0), value(ssrc1)) {
                    let (result, flag) = match inst.op {
                        I::S_ADD_I32 => {
                            let (r, o) = (a as i32).overflowing_add(b as i32);
                            (r as u32, o)
                        }
                        I::S_SUB_I32 => {
                            let (r, o) = (a as i32).overflowing_sub(b as i32);
                            (r as u32, o)
                        }
                        _ => (a.wrapping_sub(b), b > a),
                    };
                    self.write(sdst as u16, Value::Known(result));
                    self.scc = Value::Known(flag as u32);
                }
            }
            (I::S_AND_B32 | I::S_OR_B32 | I::S_XOR_B32 | I::S_LSHL_B32 | I::S_LSHR_B32 | I::S_MUL_I32, Form::Sop2 { sdst, ssrc0, ssrc1 }) => {
                if let (Value::Known(a), Value::Known(b)) = (value(ssrc0), value(ssrc1)) {
                    let result = match inst.op {
                        I::S_AND_B32 => a & b,
                        I::S_OR_B32 => a | b,
                        I::S_XOR_B32 => a ^ b,
                        I::S_LSHL_B32 => a << (b & 31),
                        I::S_LSHR_B32 => a >> (b & 31),
                        _ => a.wrapping_mul(b),
                    };
                    self.write(sdst as u16, Value::Known(result));
                    if scc {
                        self.scc = Value::Known((result != 0) as u32);
                    }
                }
            }
            (I::S_SETREG_IMM32_B32 | I::S_SETREG_B32, Form::Sopk { sdst, simm16 }) => {
                let (id, offset, size) = hwreg(simm16);
                if id == HW_REG_MODE {
                    let given = if matches!(inst.op, I::S_SETREG_IMM32_B32) {
                        inst.literal.map(Value::Known).unwrap_or(Value::Unknown)
                    } else {
                        previous.sgprs[sdst as usize]
                    };
                    let size = size.min(32 - offset);
                    let bits = match given {
                        Value::Known(v) => Some(if size == 32 { v } else { v & ((1 << size) - 1) }),
                        _ => None,
                    };
                    self.mode.set(offset, size, bits);
                }
            }
            (I::S_GETREG_B32, Form::Sopk { sdst, simm16 }) => {
                let (id, offset, size) = hwreg(simm16);
                if id == HW_REG_MODE {
                    let size = size.min(32 - offset);
                    self.write(sdst as u16, previous.mode.field(offset, size).map_or(Value::Unknown, Value::Known));
                }
            }
            _ => {}
        }
    }
}
