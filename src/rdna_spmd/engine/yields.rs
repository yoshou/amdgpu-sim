//! Typed values exchanged by a suspended packet and its wave scheduler.
//! Slots describe effect arguments/results, never architectural registers.
use super::fiber::Fiber;

fn lanes(width: usize) -> u32 { if width >= 32 { u32::MAX } else { (1u32 << width) - 1 } }
use super::super::ir::{Ty, EffectOp, WaveOp};

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
    pub base: usize,
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
        Self { op, inputs, outputs, output_base, base: 0, uniform_selector: false, arguments }
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

    // Keep the argument loads in the wave handler, as in the pre-SSA native code.
    #[inline]
    fn argument(&self, index: usize, lane: usize, width: usize, fibers: &[Fiber]) -> u32 {
        let offset = match self.arguments[index] {
            Argument::Constant(value) => return value,
            Argument::Uniform => (self.base + index) * width,
            Argument::Lane => (self.base + index) * width + lane % width,
        };
        unsafe { *fibers[lane / width].yield_values().add(offset) }
    }

    /// Collect nonzero lanes, stopping at the first match for Any/ReadFirstLane.
    /// Test the argument layout once, and address each packet's frame once.
    #[inline(never)]
    fn query_bits(&self, index: usize, width: usize, valid: u32, fibers: &[Fiber], first_only: bool) -> u32 {
        // Preserve the scalar scan without SIMD setup or a packet-width specialization.
        #[cfg(target_arch = "x86_64")]
        if width >= 4 {
            return self.query_bits_simd(index, width, valid, fibers, first_only);
        }
        self.query_bits_scalar(index, width, valid, fibers, first_only)
    }

    #[inline(never)]
    fn query_bits_scalar(&self, index: usize, width: usize, valid: u32, fibers: &[Fiber], first_only: bool) -> u32 {
        if let Argument::Constant(value) = self.arguments[index] {
            return if value == 0 { 0 } else { valid };
        }
        let uniform = matches!(self.arguments[index], Argument::Uniform);
        let mut result = 0;
        for (packet, fiber) in fibers.iter().enumerate() {
            let shift = packet * width;
            let mask = (valid >> shift) & lanes(width);
            if mask == 0 { continue; }
            let ptr = unsafe { fiber.yield_values().add((self.base + index) * width) };
            if uniform {
                if unsafe { *ptr } != 0 { result |= mask << shift; }
            } else {
                for lane in 0..width {
                    if mask >> lane & 1 != 0 && unsafe { *ptr.add(lane) } != 0 {
                        result |= 1 << (shift + lane);
                        if first_only { return result; }
                    }
                }
            }
            if first_only && result != 0 { return result; }
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    fn query_bits_simd(&self, index: usize, width: usize, valid: u32, fibers: &[Fiber], first_only: bool) -> u32 {
        if let Argument::Constant(value) = self.arguments[index] {
            return if value == 0 { 0 } else { valid };
        }
        let uniform = matches!(self.arguments[index], Argument::Uniform);
        let mut result = 0;
        for (packet, fiber) in fibers.iter().enumerate() {
            let shift = packet * width;
            let mask = (valid >> shift) & lanes(width);
            if mask == 0 { continue; }
            let ptr = unsafe { fiber.yield_values().add((self.base + index) * width) };
            #[cfg(target_arch = "x86_64")]
            if !uniform && width >= 4 && mask == lanes(width) {
                // SSE2 is available on every x86-64 host. A full packet lets
                // us read four initialized lanes at once; padding falls back.
                use std::arch::x86_64::{_mm_loadu_si128, _mm_cmpeq_epi32, _mm_setzero_si128, _mm_movemask_ps, _mm_castsi128_ps};
                let mut lane = 0;
                while lane < width {
                    let bits = unsafe {
                        let values = _mm_loadu_si128(ptr.add(lane).cast());
                        let zeros = _mm_cmpeq_epi32(values, _mm_setzero_si128());
                        (_mm_movemask_ps(_mm_castsi128_ps(zeros)) as u32) ^ 15
                    };
                    result |= bits << (shift + lane);
                    if first_only && result != 0 { return result; }
                    lane += 4;
                }
                continue;
            }
            if uniform {
                if unsafe { *ptr } != 0 { result |= mask << shift; }
            } else {
                for lane in 0..width {
                    if mask >> lane & 1 != 0 && unsafe { *ptr.add(lane) } != 0 {
                        result |= 1 << (shift + lane);
                        if first_only { return result; }
                    }
                }
            }
            if first_only && result != 0 { return result; }
        }
        result
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
                if valid >> (packet*width) & lanes(width) != 0 {
                    unsafe { *fiber.yield_values().add((self.base + self.output_base)*width) = value; }
                }
            }
        } else {
            for lane in 0..32 { if valid >> lane & 1 != 0 {
                unsafe { *fibers[lane / width].yield_values().add((self.base + self.output_base) * width + lane % width) = value; }
            } }
        }
    }

    pub fn apply_wave(&self, width: usize, valid: u32, fibers: &[Fiber]) {
        assert_eq!(fibers.len(), 32 / width);
        assert_ne!(valid, 0);
        let op = match self.op { EffectOp::Wave(op) => op, _ => panic!("not a wave effect") };
        let answer = match op {
            WaveOp::Any => Some((self.query_bits(0, width, valid, fibers, true) != 0) as u32),
            WaveOp::Ballot => Some(self.query_bits(0, width, valid, fibers, false)),
            WaveOp::ReadFirstLane => {
                let bits = self.query_bits(1, width, valid, fibers, true);
                // Empty EXEC reads lane 0, even if another valid lane exists.
                let lane = if bits == 0 { 0 } else { bits.trailing_zeros() as usize };
                Some(if valid >> lane & 1 != 0 { self.argument(0, lane, width, fibers) } else { 0 })
            }
            _ => None,
        };
        if let Some(answer) = answer {
            // Finish all input reads before overwriting the aliased result.
            self.broadcast_result(width, valid, fibers, answer);
            return;
        }
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
                unsafe { *fibers[lane/width].yield_values().add((self.base + self.output_base)*width + lane%width) = value; }
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
                pointers[packet] = if valid >> (packet * width) & lanes(width) != 0 {
                    unsafe { fibers[packet].yield_values().add(self.base * width) }
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
            unsafe { *fibers[lane / width].yield_values().add((self.base + self.output_base) * width + lane % width) = result[lane]; }
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
    use crate::rdna_spmd::engine::fiber::{KernelArgs, FIBER_DONE};
    use crate::rdna_spmd::targets::rdna4::lift::wave::{Destination, Operand, YieldAction};
    use crate::rdna_instructions::SourceOperand;
    use std::collections::BTreeMap;

    #[test]
    fn reduced_queries_match_wave_semantics_for_all_layouts_and_valid_lanes() {
        use crate::rdna_spmd::engine::fiber::{FiberCtx, amdgpu_sim_fiber_yield_values};
        unsafe extern "C" fn yield_frame(
            frame: *mut u32, _: *mut u32, _: u64, _: u64, _: *mut u32,
            _: u64, _: u64, ctx: *mut FiberCtx, _: u32,
        ) -> u64 {
            amdgpu_sim_fiber_yield_values(ctx, 0, frame);
            FIBER_DONE
        }
        let predicates: Vec<u32> = [0, 0x5555_5555, u32::MAX].iter().copied()
            .chain((0..32).map(|lane| 1 << lane)).collect();
        for width in [1, 2, 4, 8, 16, 32] {
            let mut fibers = Fiber::batch(32 / width, 64 * 1024);
            for valid in [1, 0x1ffff, 0x80000000, 0xaaaa_aaaa, u32::MAX] {
                for &predicate in &predicates {
                    for source_kind in [Argument::Lane, Argument::Uniform, Argument::Constant(0x12345678)] {
                        for mask_kind in [Argument::Lane, Argument::Uniform, Argument::Constant(0), Argument::Constant(1)] {
                            for op in [WaveOp::Any, WaveOp::Ballot, WaveOp::ReadFirstLane] {
                                let mut layout = YieldValues::new(EffectOp::Wave(op));
                                layout.base = 1; // An effect can follow others in a shared frame.
                                layout.arguments = if op == WaveOp::ReadFirstLane { vec![source_kind, mask_kind] } else { vec![mask_kind] };
                                let mut frames = vec![vec![0xdeadbeef; 4 * width]; fibers.len()];
                                let mut logical = [[0u32; 32]; 2];
                                for (index, &kind) in layout.arguments.iter().enumerate() {
                                    let is_mask = op != WaveOp::ReadFirstLane || index == 1;
                                    for lane in 0..32 {
                                        let packet = lane / width;
                                        let representative = if matches!(kind, Argument::Uniform) { packet * width } else { lane };
                                        let value = match kind {
                                            Argument::Constant(value) => value,
                                            _ if is_mask => (predicate >> representative & 1) * 0x80000000,
                                            _ => 100 + representative as u32,
                                        };
                                        logical[index][lane] = value;
                                        match kind {
                                            Argument::Constant(_) => {},
                                            Argument::Uniform => frames[packet][(layout.base + index) * width] = value,
                                            Argument::Lane if valid >> lane & 1 != 0 => frames[packet][(layout.base + index) * width + lane % width] = value,
                                            _ => {}, // Poison padding; it must never contribute.
                                        }
                                    }
                                }
                                let before = frames.clone();
                                for (packet, fiber) in fibers.iter_mut().enumerate() {
                                    if valid >> (packet * width) & lanes(width) == 0 { continue; }
                                    fiber.start(KernelArgs { entry: yield_frame as *const () as u64,
                                        sgprs: frames[packet].as_mut_ptr(), vgprs: std::ptr::null_mut(), spill: std::ptr::null_mut(),
                                        scratch_base: 0, scratch_stride: 0, lane_base: (packet * width) as u64,
                                        lds_base: 0, valid_mask: valid });
                                    assert_eq!(fiber.resume(), 0);
                                }
                                let expected = evaluate(op, valid, |index, lane| logical[index][lane])[0];
                                layout.apply_wave(width, valid, &fibers);
                                for (packet, fiber) in fibers.iter_mut().enumerate() {
                                    let active = valid >> (packet * width) & lanes(width) != 0;
                                    for (cell, &value) in frames[packet].iter().enumerate() {
                                        let expected = if active && cell == layout.base * width { expected } else { before[packet][cell] };
                                        assert_eq!(value, expected, "{op:?} width={width} valid={valid:#x} predicate={predicate:#x} packet={packet} cell={cell}");
                                    }
                                    if active { assert_eq!(fiber.resume(), FIBER_DONE); }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Invoke the compiled packet entry on real fibers, including its native
    // argument stores and result loads. Register buffers are observed only at
    // entry/return, never used to apply the yielded effect.
    fn run(program: &ScalarProgram, width: usize, count: usize,
           sgprs: &mut [[u32; 129]], vgprs: &mut [Vec<u32>]) {
        run_private(program,width,count,sgprs,vgprs,0,0);
    }
    fn run_private(program: &ScalarProgram, width: usize, count: usize,
           sgprs: &mut [[u32; 129]], vgprs: &mut [Vec<u32>],scratch_base:u64,scratch_stride:u64) {
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
                scratch_base, scratch_stride, lane_base: (packet*width) as u64,
                lds_base: 0, valid_mask: valid,
            });
        }
        loop {
            let stops: Vec<_> = fibers[..count.div_ceil(width)].iter_mut().map(Fiber::resume).collect();
            assert!(stops.iter().all(|pc| *pc == stops[0]));
            if stops[0] == FIBER_DONE { break; }
            for action in &kernel.yields[stops[0] as usize] { action.apply_wave(width, valid, &fibers); }
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
    fn private_pointer_broadcast_keeps_storage_private_to_each_lane() {
        use crate::instructions::I;
        use crate::rdna_instructions::{InstFormat,SOP1,SOP2,VOP1,VOP2,VFLAT};
        let read=|source,destination|YieldAction {op:EffectOp::Wave(WaveOp::ReadFirstLane),
            inputs:vec![Operand::Source(SourceOperand::VectorRegister(source)),Operand::Exec],
            outputs:vec![Destination::Sgpr(destination)]};
        let vcopy=|vdst,src|InstFormat::VOP1(VOP1 {op:I::V_MOV_B32,vdst,src0:SourceOperand::ScalarRegister(src)});
        let p=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([
            (0,ScalarBlock {pc:0,body:vec![
                InstFormat::VOP2(VOP2 {op:I::V_ADD_NC_U32,src0:SourceOperand::LiteralConstant(100),vsrc1:0,vdst:2,literal_constant:None}),
                InstFormat::SOP1(SOP1 {op:I::S_MOV_B64,sdst:4,ssrc0:SourceOperand::PrivateBase}),
                InstFormat::SOP2(SOP2 {op:I::S_LSHR_B64,sdst:4,ssrc0:SourceOperand::ScalarRegister(4),ssrc1:SourceOperand::IntegerConstant(32)}),
                InstFormat::SOP2(SOP2 {op:I::S_LSHL_B64,sdst:4,ssrc0:SourceOperand::ScalarRegister(4),ssrc1:SourceOperand::IntegerConstant(32)}),
                vcopy(0,4),vcopy(1,5),
                InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst:126,ssrc0:SourceOperand::LiteralConstant(1<<21)}),
            ],term:Terminator::Yield {resume:1,action:Box::new(read(0,10))}}),
            (1,ScalarBlock {pc:1,body:vec![],term:Terminator::Yield {resume:2,action:Box::new(read(1,11))}}),
            (2,ScalarBlock {pc:2,body:vec![
                InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst:126,ssrc0:SourceOperand::LiteralConstant(u32::MAX)}),
                vcopy(4,10),vcopy(5,11),
                InstFormat::VFLAT(VFLAT {op:I::FLAT_STORE_B32,vaddr:4,vsrc:2,vdst:0,saddr:124,ioffset:4,scope:0,th:0,sve:0}),
            ],term:Terminator::Return}),
        ])};
        for code_width in [0usize,1,2,4,8,16] {
            let width=code_width.max(1);
            for (count,base) in [(23usize,0usize),(32,32*16)] {
                let mut s=vec![[0;129];32/width];
                let mut v=vec![vec![0;24*width];32/width];
                let mut scratch=vec![0xdeadbeef;64*16];
                for lane in 0..count {s[lane/width][126]|=1<<(lane%width);v[lane/width][lane%width]=lane as u32;}
                run_private(&p,code_width,count,&mut s,&mut v,unsafe {scratch.as_mut_ptr().add(base)} as u64,64);
                for lane in 0usize..64 {for word in 0..16 {
                    let local=lane.wrapping_sub(base/16);
                    let expected=if local<count && word==1 {100+local as u32} else {0xdeadbeef};
                    assert_eq!(scratch[lane*16+word],expected,"width={code_width} lane={lane} word={word}");
                }}
            }
        }
    }
    #[test]
    fn memory_and_wave_results_use_wave_mask_words_before_saveexec() {
        use crate::instructions::I;
        use crate::rdna_instructions::{InstFormat,SMEM,SOP1};
        let mov=|sdst,ssrc0|InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst,ssrc0});
        let p=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([
            (0,ScalarBlock {pc:0,body:vec![
                InstFormat::SMEM(SMEM {op:I::S_LOAD_B32,sbase:2,sdata:126,soffset:124,ioffset:0,scope:0,th:0}),
                mov(10,SourceOperand::ScalarRegister(126)),
            ],term:Terminator::Yield {resume:1,action:Box::new(YieldAction {
                op:EffectOp::Wave(WaveOp::ReadLane),
                inputs:vec![Operand::Source(SourceOperand::VectorRegister(2)),Operand::Source(SourceOperand::IntegerConstant(21)),Operand::Source(SourceOperand::IntegerConstant(2))],
                outputs:vec![Destination::Sgpr(106)],
            })}}),
            (1,ScalarBlock {pc:1,body:vec![
                mov(11,SourceOperand::ScalarRegister(106)),
                InstFormat::SOP1(SOP1 {op:I::S_AND_SAVEEXEC_B32,ssrc0:SourceOperand::ScalarRegister(106),sdst:20}),
                mov(12,SourceOperand::ScalarRegister(126)),
            ],term:Terminator::Return}),
        ])};
        for code_width in [0usize,1,2,4,8,16] {
            let width=code_width.max(1);
            for count in [1usize,23,32] {
                let valid=(u32::MAX as u64 >> (32-count)) as u32;
                let word=0x80a02005u32;
                let ptr=&word as *const u32 as u64;
                let chosen=0x80200004u32;
                let ballot=if count>21 {chosen & valid} else {0};
                let mut s=vec![[0;129];32/width];
                let mut v=vec![vec![0;24*width];32/width];
                for packet in 0..count.div_ceil(width) {
                    s[packet][4]=ptr as u32;s[packet][5]=(ptr>>32) as u32;
                    s[packet][126]=(valid>>(packet*width)) & ((1<<width)-1);
                    for lane in 0..width {v[packet][2*width+lane]=chosen;}
                }
                run(&p,code_width,count,&mut s,&mut v);
                for packet in 0..count.div_ceil(width) {
                    assert_eq!(s[packet][10],word&valid,"width={code_width} count={count} packet={packet}");
                    assert_eq!(s[packet][11],ballot,"width={code_width} count={count} packet={packet}");
                    assert_eq!(s[packet][20],word&valid);
                    assert_eq!(s[packet][12],word&ballot&valid);
                    assert_eq!(s[packet][128],(word&ballot&valid!=0) as u32);
                }
            }
        }
    }
    #[test]
    fn architectural_mask_words_and_branches_hold_wave_bits() {
        use crate::instructions::I;
        use crate::rdna_instructions::{InstFormat,SOP1};
        use crate::rdna_spmd::targets::rdna4::decode::Cond;
        let mov = |sdst,ssrc0| InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst,ssrc0});
        let p = ScalarProgram {entry_pc:0,blocks:BTreeMap::from([
            (0,ScalarBlock {pc:0,body:vec![
                mov(10,SourceOperand::ScalarRegister(126)),
                mov(126,SourceOperand::LiteralConstant(0x80000001)),
                mov(11,SourceOperand::ScalarRegister(126)),
                mov(126,SourceOperand::ScalarRegister(10)),
            ],term:Terminator::Branch {cond:Cond::ExecNz,taken:1,fallthrough:2}}),
            (1,ScalarBlock {pc:1,body:vec![mov(12,SourceOperand::IntegerConstant(1))],term:Terminator::Jump(3)}),
            (2,ScalarBlock {pc:2,body:vec![mov(12,SourceOperand::IntegerConstant(0))],term:Terminator::Jump(3)}),
            (3,ScalarBlock {pc:3,body:vec![],term:Terminator::Return}),
        ])};
        for code_width in [0,1,2,4,8,16] {
            let width=code_width.max(1);
            for count in [1usize,23,32] {
                let valid=(u32::MAX as u64 >> (32-count)) as u32;
                for mask in [0,1<<21,0xaaaaaaaa,u32::MAX] {
                    let mut s=vec![[0;129];32/width];
                    let mut v=vec![vec![0;256*width];32/width];
                    for (packet,regs) in s.iter_mut().enumerate() {
                        regs[126]=(mask >> (packet*width)) & ((1<<width)-1);
                    }
                    run(&p,code_width,count,&mut s,&mut v);
                    for (packet,regs) in s[..count.div_ceil(width)].iter().enumerate() {
                        let packet_mask=((mask&valid)>>(packet*width))&((1<<width)-1);
                        assert_eq!(regs[10],mask&valid,"saved wave mask after entry validity clipping");
                        assert_eq!(regs[11],0x80000001&valid,"wave lane projection");
                        assert_eq!(regs[12],(mask&valid!=0) as u32,"wave branch width={code_width} packet={packet}");
                        assert_eq!(regs[126],packet_mask);
                    }
                }
            }
        }
    }
    #[test]
    fn exec_and_vcc_branches_reduce_over_the_wave_and_scc_stays_scalar() {
        use crate::instructions::I;
        use crate::rdna_instructions::{InstFormat,SOP1};
        use crate::rdna_spmd::targets::rdna4::decode::Cond;
        let mov=|value|InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst:12,
            ssrc0:SourceOperand::IntegerConstant(value)});
        for cond in [Cond::ExecZ,Cond::ExecNz,Cond::VccZ,Cond::VccNz,Cond::Scc0,Cond::Scc1] {
            let program=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([
                (0,ScalarBlock {pc:0,body:vec![],term:Terminator::Branch {cond,taken:1,fallthrough:2}}),
                (1,ScalarBlock {pc:1,body:vec![mov(1)],term:Terminator::Return}),
                (2,ScalarBlock {pc:2,body:vec![mov(0)],term:Terminator::Return}),
            ])};
            for code_width in [0usize,1,2,4,8,16] {for count in [1usize,23,32] {
                let width=code_width.max(1);
                let mut s=vec![[0u32;129];32/width];
                let mut v=vec![vec![0u32;24*width];32/width];
                for (packet,regs) in s.iter_mut().enumerate() {
                    regs[126]=((1u32<<21)>>(packet*width))&((1<<width)-1);
                    regs[106]=((1u32<<3)>>(packet*width))&((1<<width)-1);
                    regs[128]=(packet%2) as u32;
                }
                run(&program,code_width,count,&mut s,&mut v);
                let valid=(u32::MAX as u64>>(32-count)) as u32;
                for (packet,regs) in s[..count.div_ceil(width)].iter().enumerate() {
                    let nonzero=match cond {
                        Cond::ExecZ|Cond::ExecNz=>(1u32<<21)&valid!=0,
                        Cond::VccZ|Cond::VccNz=>(1u32<<3)&valid!=0,
                        Cond::Scc0|Cond::Scc1=>regs[128]!=0,
                    };
                    let inverted=matches!(cond,Cond::ExecZ|Cond::VccZ|Cond::Scc0);
                    assert_eq!(regs[12],(nonzero^inverted) as u32,
                        "{cond:?} width={code_width} count={count} packet={packet}");
                }
            }}
        }
    }
    #[test]
    fn local_writelane_uses_global_lane_id_and_ignores_exec() {
        let v = |r| Operand::Source(SourceOperand::VectorRegister(r));
        let s = |r| Operand::Source(SourceOperand::ScalarRegister(r));
        let p = program(vec![
            YieldAction::new(EffectOp::Wave(WaveOp::WriteLane),vec![s(10),s(11),v(0),Operand::Source(SourceOperand::IntegerConstant(0))],vec![Destination::Vgpr(0)]),
            YieldAction::new(EffectOp::Wave(WaveOp::ReadLane),vec![v(0),s(11),Operand::Source(SourceOperand::IntegerConstant(0))],vec![Destination::Sgpr(12)]),
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
