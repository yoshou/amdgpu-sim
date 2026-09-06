//! Typed values exchanged by a suspended packet and its wave scheduler.
//! Slots describe effect arguments/results, never architectural registers.
use super::{fiber::Fiber, ir::typed::{Ty, effect::{EffectOp, WaveOp}}};

#[derive(Clone, Copy)]
pub(crate) enum Argument {
    Lane,
    Uniform,
    Constant(u32),
}

#[derive(Clone)]
pub(crate) struct YieldValues {
    pub op: EffectOp,
    pub inputs: Vec<Ty>,
    pub outputs: Vec<Ty>,
    pub output_base: usize,
    pub uniform_selector: bool,
    pub arguments: Vec<Argument>,
}
impl YieldValues {
    pub fn new(op: EffectOp) -> Self {
        let (inputs, outputs) = op.signature();
        assert!(inputs.len() <= 24 && outputs.len() <= 8);
        assert!(inputs.iter().chain(&outputs).all(|ty| ty.bits() <= 32));
        // Every input is captured before any result is written. Reusing an
        // old-value slot avoids copying WriteLane/WMMA accumulators at yields.
        let output_base = match op {
            EffectOp::Wave(WaveOp::WriteLane) => 2,
            EffectOp::Wave(WaveOp::Bpermute | WaveOp::BpermuteFi) => 1,
            EffectOp::Wave(WaveOp::Wmma) => 8,
            _ => 0,
        };
        let arguments = vec![Argument::Lane; inputs.len()];
        Self { op, inputs, outputs, output_base, uniform_selector: false, arguments }
    }
    pub fn cells(&self) -> usize { self.inputs.len().max(self.output_base + self.outputs.len()) }
    pub fn uniform_result(&self) -> bool {
        match self.op {
            EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane) => true,
            EffectOp::Wave(WaveOp::ReadLane) => self.uniform_selector,
            EffectOp::BarrierSignal { is_first: true } => true,
            _ => false,
        }
    }
    pub fn is_wave(&self) -> bool { matches!(self.op, EffectOp::Wave(_)) }

    fn argument(&self, index: usize, lane: usize, width: usize, fibers: &[Fiber]) -> u32 {
        let offset = match self.arguments[index] {
            Argument::Constant(value) => return value,
            Argument::Uniform => index * width,
            Argument::Lane => index * width + lane % width,
        };
        unsafe { *fibers[lane / width].yield_values().add(offset) }
    }

    pub fn uniform_id(&self, width: usize, valid: u32, fibers: &[Fiber]) -> u32 {
        assert_ne!(valid, 0);
        let first = valid.trailing_zeros() as usize;
        let read = |lane: usize| self.argument(0, lane, width, fibers);
        let id = read(first);
        for lane in 0..32 { if valid >> lane & 1 != 0 { assert_eq!(read(lane), id, "nonuniform barrier ID"); } }
        id
    }
    pub fn broadcast_result(&self, width: usize, valid: u32, fibers: &[Fiber], value: u32) {
        assert_eq!(self.outputs.len(), 1);
        if self.uniform_result() {
            // A uniform result has one scalar cell per packet. Native code
            // broadcasts that cell only when a varying consumer needs it.
            for (packet, fiber) in fibers.iter().enumerate() {
                if valid >> (packet*width) & ((1u32 << width)-1) != 0 {
                    unsafe { *fiber.yield_values().add(self.output_base*width) = value; }
                }
            }
        } else {
            for lane in 0..32 { if valid >> lane & 1 != 0 {
                unsafe { *fibers[lane / width].yield_values().add(self.output_base * width + lane % width) = value; }
            } }
        }
    }

    pub fn apply_wave(&self, width: usize, valid: u32, fibers: &[Fiber]) {
        assert_eq!(fibers.len(), 32 / width);
        assert_ne!(valid, 0);
        let op = match self.op { EffectOp::Wave(op) => op, _ => panic!("not a wave effect") };
        let arg = |index: usize, lane: usize| if valid >> lane & 1 == 0 { 0 } else {
            self.argument(index, lane, width, fibers)
        };
        // WriteLane's signature requires wave-uniform value and selector,
        // checked by lift. Its old-value input already occupies the result
        // slot, so only the selected cell changes.
        if op == WaveOp::WriteLane {
            let first = valid.trailing_zeros() as usize;
            let value = arg(0, first); let lane = (arg(1, first) & 31) as usize;
            if valid >> lane & 1 != 0 {
                unsafe { *fibers[lane/width].yield_values().add(self.output_base*width + lane%width) = value; }
            }
            return;
        }
        if op == WaveOp::ReadLane && self.uniform_selector {
            let first = valid.trailing_zeros() as usize;
            let value = arg(0, (arg(1, first) & 31) as usize);
            self.broadcast_result(width, valid, fibers, value);
            return;
        }
        if op == WaveOp::Wmma {
            let mut padding = (valid != u32::MAX).then(|| vec![0u32; self.cells() * width]);
            let mut pointers = [std::ptr::null_mut(); 32];
            for packet in 0..fibers.len() {
                pointers[packet] = if valid >> (packet * width) & ((1u32 << width) - 1) != 0 {
                    fibers[packet].yield_values()
                } else { padding.as_mut().unwrap().as_mut_ptr() };
            }
            if valid != u32::MAX {
                for lane in 0..32 { if valid >> lane & 1 == 0 {
                    for arg in 0..self.inputs.len() {
                        unsafe { *pointers[lane / width].add(arg * width + lane % width) = 0; }
                    }
                } }
            }
            // The existing fused native WMMA consumes the dense A/B/C slots.
            unsafe { super::wmma::apply_values(width, pointers.as_ptr()); }
            return;
        }
        assert!(self.inputs.len() <= 3 && self.outputs.len() == 1);
        let result = evaluate(op, valid, |index, lane| self.argument(index, lane, width, fibers));
        if self.uniform_result() { self.broadcast_result(width, valid, fibers, result[0]); }
        else { for lane in 0..32 { if valid >> lane & 1 != 0 {
            unsafe { *fibers[lane / width].yield_values().add(self.output_base * width + lane % width) = result[lane]; }
        } } }
    }
}

/// Complete a wave operation before its caller writes any aliased result.
/// `read` receives only allocated lanes; padding never contributes a value.
pub(crate) fn evaluate(op: WaveOp, valid: u32, read: impl Fn(usize, usize) -> u32) -> [u32; 32] {
    assert_ne!(valid, 0, "empty wave");
    let arg = |index, lane| if valid >> lane & 1 != 0 { read(index, lane) } else { 0 };
    // Capture just the values used by this effect. The result is complete
    // before the caller writes any destination, including overlapping ones.
    let mut out = [0; 32];
    match op {
        WaveOp::Any => out.fill((0..32).any(|lane| arg(0, lane) != 0) as u32),
        WaveOp::Ballot => {
            let mask = (0..32).fold(0, |mask, lane| mask | ((arg(0, lane) != 0) as u32) << lane);
            out.fill(mask);
        }
        WaveOp::ReadFirstLane => {
            let lane = (0..32).find(|&lane| arg(1, lane) != 0).unwrap_or(0);
            out.fill(arg(0, lane));
        }
        WaveOp::ReadLane => {
            out = std::array::from_fn(|lane| arg(0, (arg(1, lane) & 31) as usize));
        }
        WaveOp::WriteLane => {
            let first = valid.trailing_zeros() as usize;
            assert!(first < 32, "empty wave");
            let value = arg(0, first);
            let selector = arg(1, first);
            out = std::array::from_fn(|lane| {
                if valid >> lane & 1 != 0 {
                    assert_eq!(arg(0, lane), value, "nonuniform writelane value");
                    assert_eq!(arg(1, lane), selector, "nonuniform writelane lane");
                }
                arg(2, lane)
            });
            out[(selector & 31) as usize] = value;
        }
        WaveOp::Bpermute | WaveOp::BpermuteFi => {
            out = std::array::from_fn(|lane| {
                let src = ((arg(0, lane) >> 2) & 31) as usize;
                if op == WaveOp::BpermuteFi || arg(2, src) != 0 { arg(1, src) } else { 0 }
            });
        }
        WaveOp::Wmma => panic!("WMMA uses the existing fragment lowering"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::{Compiler, ScalarBlock, ScalarProgram, Terminator};
    use crate::rdna_spmd::fiber::{KernelArgs, FIBER_DONE};
    use crate::rdna_spmd::lift::wave::{Destination, Operand, YieldAction};
    use crate::rdna_instructions::SourceOperand;
    use std::collections::BTreeMap;

    // Invoke the compiled packet entry on real fibers, including its native
    // argument stores and result loads. Register buffers are observed only at
    // entry/return, never used to apply the yielded effect.
    fn run(program: &ScalarProgram, width: usize, count: usize,
           sgprs: &mut [[u32; 129]], vgprs: &mut [Vec<u32>]) {
        let kernel = if width == 0 { Compiler::default().compile_cooperative(program, 24) }
            else { Compiler::default().compile_cooperative_vec(program, 24, width as u32) };
        let width = width.max(1);
        let valid = (u32::MAX as u64 >> (32-count)) as u32;
        let mut fibers: Vec<_> = (0..32/width).map(|_| Fiber::new(256*1024)).collect();
        let mut spill = vec![vec![0u32; 8192]; fibers.len()];
        for packet in 0..count.div_ceil(width) {
            fibers[packet].start(KernelArgs {
                entry: kernel.addr(), sgprs: sgprs[packet].as_mut_ptr(),
                vgprs: vgprs[packet].as_mut_ptr(), spill: spill[packet].as_mut_ptr(),
                scratch_base: 0, scratch_stride: 0, lane_base: (packet*width) as u64,
                lds_base: 0, valid_mask: (valid >> (packet*width)) & ((1 << width)-1),
            });
        }
        loop {
            let stops: Vec<_> = fibers[..count.div_ceil(width)].iter_mut().map(Fiber::resume).collect();
            assert!(stops.iter().all(|pc| *pc == stops[0]));
            if stops[0] == FIBER_DONE { break; }
            kernel.yields[&(stops[0] as usize)].apply_wave(width, valid, &fibers);
        }
    }
    fn program(actions: Vec<YieldAction>) -> ScalarProgram {
        let n = actions.len();
        let mut blocks = BTreeMap::new();
        for (pc, action) in actions.into_iter().enumerate() {
            blocks.insert(pc, ScalarBlock { pc, body: vec![],
                term: Terminator::Yield { resume: pc+1, action: Box::new(action) } });
        }
        blocks.insert(n, ScalarBlock { pc: n, body: vec![], term: Terminator::Return });
        ScalarProgram { entry_pc: 0, blocks }
    }
    #[test]
    fn local_writelane_uses_global_lane_id_and_ignores_exec() {
        let v = |r| Operand::Source(SourceOperand::VectorRegister(r));
        let s = |r| Operand::Source(SourceOperand::ScalarRegister(r));
        let p = program(vec![
            YieldAction::new(EffectOp::Wave(WaveOp::WriteLane),vec![s(10),s(11),v(0)],vec![Destination::Vgpr(0)]),
            YieldAction::new(EffectOp::Wave(WaveOp::ReadLane),vec![v(0),s(11)],vec![Destination::Sgpr(12)]),
        ]);
        for code_width in [0,1,2,4,8,16] {
            let width = code_width.max(1);
            for count in [23,32] {
                for selector in [0u32,17,31,63] {
                    let mut s = vec![[0;129];32/width];
                    let mut v = vec![vec![0;24*width];32/width];
                    for packet in 0..32/width {
                        s[packet][10]=0x12345678; s[packet][11]=selector;
                        // EXEC is empty throughout: WriteLane still applies.
                        for lane in 0..width { v[packet][lane]=100+(packet*width+lane) as u32; }
                    }
                    run(&p,code_width,count,&mut s,&mut v);
                    let selected=(selector&31) as usize;
                    for lane in 0..count {
                        assert_eq!(v[lane/width][lane%width],if lane==selected {0x12345678} else {100+lane as u32});
                        assert_eq!(s[lane/width][12],if selected<count {0x12345678} else {0});
                    }
                }
            }
        }
    }
    #[test]
    fn native_any_and_ballot_collect_full_wave_with_empty_exec_and_padding() {
        let p = program(vec![
            YieldAction::new(EffectOp::Wave(WaveOp::Any), vec![Operand::Exec], vec![Destination::Sgpr(10)]),
            YieldAction::new(EffectOp::Wave(WaveOp::Ballot), vec![Operand::Exec], vec![Destination::Sgpr(11)]),
        ]);
        for code_width in [0,1,2,4,8,16] {
            let width = code_width.max(1);
            for count in [1usize,23,32] {
                let valid = (u32::MAX as u64 >> (32-count)) as u32;
                for mask in [0,1,1<<21,1<<31,0xaaaaaaaa,u32::MAX] {
                    let mut s = vec![[0;129];32/width];
                    let mut v = vec![vec![0;24*width];32/width];
                    for (packet, regs) in s.iter_mut().enumerate() {
                        regs[126] = (mask >> (packet*width)) & ((1<<width)-1);
                    }
                    run(&p,code_width,count,&mut s,&mut v);
                    for regs in &s[..count.div_ceil(width)] {
                        assert_eq!((regs[10],regs[11]), ((mask&valid != 0) as u32,mask&valid),
                            "width={}, count={}, mask={:#x}",width,count,mask);
                    }
                }
            }
        }
    }
    #[test]
    fn native_wmma_frames_preserve_float_accumulators_and_source_overlap() {
        let v = |r| Operand::Source(SourceOperand::VectorRegister(r));
        // D aliases A/B, then the next effect uses that D as its C. This
        // exposes capture order and transport of typed f32 results to words.
        let args = |c| (0..8).map(v).chain((c..c+8).map(v)).collect();
        let p = program(vec![
            YieldAction::new(EffectOp::Wave(WaveOp::Wmma), args(8), (0..8).map(Destination::Vgpr).collect()),
            YieldAction::new(EffectOp::Wave(WaveOp::Wmma),
                (16..24).map(v).chain((0..8).map(v)).collect(), (8..16).map(Destination::Vgpr).collect()),
        ]);
        for code_width in [0,1,2,4,8,16] {
            let width = code_width.max(1);
            for count in [23usize,32] {
                let mut s = vec![[0;129];32/width]; // WMMA ignores empty EXEC.
                let mut regs = vec![vec![0xdeadbeef;24*width];32/width];
                for lane in 0..count {
                    for r in 0..24 {
                        regs[lane/width][r*width+lane%width] = match r {
                            0..=3 | 16..=19 => 0x3c003c00, // A = 1
                            4..=7 | 20..=23 => 0x40004000, // B = 2
                            _ => (lane as f32 * 0.25 + r as f32).to_bits(),
                        };
                    }
                }
                run(&p,code_width,count,&mut s,&mut regs);
                for lane in 0..count { for m in 0..8 {
                    // For a partial wave, missing A/B fragments are zero.
                    // Both source lanes must exist for a k term to contribute.
                    let row = m + 8*(lane/16); let col = lane%16;
                    let terms = (0..16).filter(|k| {
                        let group=(k/4)%2;
                        row+16*group < count && col+16*group < count
                    }).count() as f32;
                    let c = lane as f32*0.25 + (8+m) as f32;
                    assert_eq!(regs[lane/width][m*width+lane%width], (c+2.0*terms).to_bits());
                    assert_eq!(regs[lane/width][(8+m)*width+lane%width], (c+4.0*terms).to_bits());
                } }
            }
        }
    }
}
