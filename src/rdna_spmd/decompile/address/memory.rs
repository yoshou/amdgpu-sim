use super::control::edges_into;
use super::form::*;
use super::graph::Store;
use super::queries::*;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

#[derive(Clone, PartialEq)]
pub(super) enum Reached {
    Value(Value),
    Same,
}

fn extend(word: u32, size: MemSize) -> u32 {
    match size {
        MemSize::U8 => word & 0xff,
        MemSize::U16 => word & 0xffff,
        MemSize::I8 => word as u8 as i8 as i32 as u32,
        MemSize::I16 => word as u16 as i16 as i32 as u32,
        MemSize::B32 | MemSize::B64 => word,
    }
}

pub(super) fn load<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, address: &Value, size: MemSize, lane: usize) -> Value {
    let offset = match address.region {
        Some(r @ (Region::Kernarg | Region::Dispatch)) => address.form.sub(&q.symbols_mut().base(r).form).as_constant(),
        _ => None,
    };
    let bytes = size.bytes().min(4);
    if let Site::Inst { block, index } = q.program().facts.site[v.0] {
        if let Inst::Effect {
            op:
                EffectOp::Memory {
                    space: Space::Scratch,
                    ..
                },
            ..
        } = &q.program().f.blocks[&block].insts[index]
        {
            if let (MemSize::B32 | MemSize::B64, Some(a)) = (size, address.form.as_constant()) {
                if let Some(value) = slot_before(q, (block, index), a, size.bytes(), lane) {
                    return q.symbols_mut().leave(value, block, lane);
                }
            } else if matches!(size, MemSize::B32 | MemSize::B64) {
                if let Some(value) = slot_in_block(q, (block, index), &address.form, size.bytes(), lane) {
                    return q.symbols_mut().leave(value, block, lane);
                }
            }
            return q.symbols_mut().opaque(v, lane, None);
        }
    }
    match (address.region, offset) {
        (Some(Region::Kernarg), Some(at)) => {
            if bytes == 4 {
                if let Some(binding) = q.program().env.binding(at) {
                    return Value {
                        form: Form::constant(binding.pointer as u32),
                        region: Some(Region::Allocation(binding.allocation)),
                    };
                }
                if let Some(binding) = at.checked_sub(4).and_then(|o| q.program().env.binding(o)) {
                    return Value::constant((binding.pointer >> 32) as u32);
                }
            }
            let word = q.program().env.kernarg_word(at, bytes);
            Value::constant(extend(word, size))
        }
        (Some(Region::Dispatch), Some(at)) => match q.program().dispatch_word(at, bytes) {
            Some(word) => Value::constant(extend(word, size)),
            None => q.symbols_mut().opaque(v, lane, None),
        },
        _ => {
            let range = match size {
                MemSize::U8 => Some((0, 0xff)),
                MemSize::U16 => Some((0, 0xffff)),
                _ => None,
            };
            q.symbols_mut().opaque(v, lane, range)
        }
    }
}

fn slot_in_block<'a, Q: Queries<'a>>(q: &mut Q, at: (BlockId, usize), address: &Form, bytes: u32, lane: usize) -> Option<Value> {
    let mut memo = HashMap::default();
    match symbolic_slot(q, at, address, bytes, lane, None, &mut memo)? {
        Reached::Value(value) => Some(value),
        Reached::Same => None,
    }
}

fn symbolic_slot<'a, Q: Queries<'a>>(
    q: &mut Q,
    at: (BlockId, usize),
    address: &Form,
    bytes: u32,
    lane: usize,
    boundary: Option<BlockId>,
    memo: &mut HashMap<(BlockId, Option<BlockId>), Option<Reached>>,
) -> Option<Reached> {
    let (block, index) = at;
    let stores: Vec<Store> = q
        .program().stores
        .get(&block)
        .map(|list| list.iter().filter(|w| w.index < index).rev().copied().collect())
        .unwrap_or_default();
    for w in stores {
        let ran = q.bit(w.predicate, lane, None).0;
        if ran == Some(false) {
            continue;
        }
        let target = q.value(w.address, lane, Some(w.predicate)).0.form;
        if target.terms != address.terms {
            return None;
        }
        let d = target.constant.wrapping_sub(address.constant);
        if d != 0 {
            if d >= bytes && d.wrapping_neg() >= w.bytes {
                continue;
            }
            return None;
        }
        if w.bytes != bytes || ran != Some(true) {
            return None;
        }
        return Some(Reached::Value(q.value(w.data?, lane, Some(w.predicate)).0));
    }
    if Some(block) == boundary {
        return Some(Reached::Same);
    }
    if block == q.program().f.entry {
        return None;
    }
    if let Some(known) = memo.get(&(block, boundary)) {
        return known.clone();
    }
    memo.insert((block, boundary), None);
    let found = reaching_block_start(q, block, address, bytes, lane, boundary, memo);
    memo.insert((block, boundary), found.clone());
    found
}

fn reaching_block_start<'a, Q: Queries<'a>>(
    q: &mut Q,
    block: BlockId,
    address: &Form,
    bytes: u32,
    lane: usize,
    boundary: Option<BlockId>,
    memo: &mut HashMap<(BlockId, Option<BlockId>), Option<Reached>>,
) -> Option<Reached> {
    let (entering, back) = edges_into(q, block);
    if entering.is_empty() {
        return None;
    }
    let mut found: Option<Reached> = None;
    for (pred, _) in entering {
        let end = q.program().f.blocks[&pred].insts.len();
        let reached = symbolic_slot(q, (pred, end), address, bytes, lane, boundary, memo)?;
        match &found {
            Some(old) if *old != reached => return None,
            _ => found = Some(reached),
        }
    }
    for (pred, _) in back {
        let end = q.program().f.blocks[&pred].insts.len();
        match symbolic_slot(q, (pred, end), address, bytes, lane, Some(block), memo)? {
            Reached::Same => {}
            Reached::Value(v) if found == Some(Reached::Value(v.clone())) => {}
            _ => return None,
        }
    }
    found
}

pub(super) fn slot_before<'a, Q: Queries<'a>>(q: &mut Q, at: (BlockId, usize), address: u32, bytes: u32, lane: usize) -> Option<Value> {
    let (block, index) = at;
    let stores: Vec<Store> = q
        .program().stores
        .get(&block)
        .map(|list| list.iter().filter(|w| w.index < index).rev().copied().collect())
        .unwrap_or_default();
    for w in stores {
        let ran = q.bit(w.predicate, lane, None).0;
        if ran == Some(false) {
            continue;
        }
        let target = q.value(w.address, lane, Some(w.predicate)).0.form.as_constant()?;
        let (t, a) = (target as u64, address as u64);
        if t + w.bytes as u64 <= a || a + bytes as u64 <= t {
            continue;
        }
        if target != address || w.bytes != bytes {
            return None;
        }
        let data = q.value(w.data?, lane, Some(w.predicate)).0;
        if ran == Some(true) {
            return Some(data);
        }
        let before = slot_before(q, (block, w.index), address, bytes, lane)?;
        return q.symbols_mut().merge_held(vec![before, data], (block, address, bytes, lane as u8), w.index + 1);
    }
    q.slot_entry(block, address, bytes, lane)
}

pub(super) fn join_slot<'a, Q: Queries<'a>>(q: &mut Q, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value> {
    let (_, back) = edges_into(q, block);
    if back.is_empty() {
        return entering_slot(q, block, address, bytes, lane);
    }
    q.recur(block, lane, Target::Slot(address, bytes))
}

pub(super) fn entering_slot<'a, Q: Queries<'a>>(q: &mut Q, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value> {
    let (entering, _) = edges_into(q, block);
    let mut values: Vec<Value> = Vec::new();
    for (pred, _) in entering {
        let end = q.program().f.blocks[&pred].insts.len();
        values.push(slot_before(q, (pred, end), address, bytes, lane)?);
    }
    q.symbols_mut().merge_held(values, (block, address, bytes, lane as u8), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_widens_each_size_as_the_load_does() {
        for word in [0u32, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff, 0x1234_5678, 0xffff_ffff] {
            assert_eq!(extend(word, MemSize::U8), word & 0xff);
            assert_eq!(extend(word, MemSize::I8), word as u8 as i8 as i32 as u32);
            assert_eq!(extend(word, MemSize::U16), word & 0xffff);
            assert_eq!(extend(word, MemSize::I16), word as u16 as i16 as i32 as u32);
            assert_eq!(extend(word, MemSize::B32), word);
        }
    }
}
