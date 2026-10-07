use amdgpu_sim::rdna_spmd::{Arg, Buffer, Launch, Module};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const LANES: usize = 64;
pub const WIDTHS: [u32; 6] = [1, 2, 4, 8, 16, 32];

pub struct Probe {
    pub inputs: usize,
    pub outputs: usize,
    pub body: String,
    pub denorm32: u32,
    pub lds: u32,
    pub private: u32,
    pub wave_offset: bool,
    pub texture: Option<Vec<u8>>,
}

fn tool(name: &str) -> PathBuf {
    let prefix = std::env::var("LLVM_SYS_221_PREFIX").unwrap_or_else(|_| "/usr/lib/llvm-22".to_string());
    PathBuf::from(prefix).join("bin").join(name)
}

static NEXT: AtomicUsize = AtomicUsize::new(0);

impl Probe {
    pub fn new(inputs: usize, outputs: usize, body: &str) -> Self {
        Probe {
            inputs,
            outputs,
            body: body.to_string(),
            denorm32: 3,
            lds: 0,
            private: 0,
            wave_offset: false,
            texture: None,
        }
    }

    pub fn texture(mut self, bytes: Vec<u8>) -> Self {
        self.texture = Some(bytes);
        self
    }

    pub fn flushing(mut self) -> Self {
        self.denorm32 = 0;
        self
    }

    pub fn shared(mut self, bytes: u32) -> Self {
        self.lds = bytes;
        self
    }

    pub fn private(mut self, bytes: u32, wave_offset: bool) -> Self {
        self.private = bytes;
        self.wave_offset = wave_offset;
        self
    }

    fn source(&self) -> String {
        let mut text = String::new();
        let line = |text: &mut String, s: &str| {
            text.push_str("  ");
            text.push_str(s);
            text.push('\n');
        };
        text.push_str(".amdgcn_target \"amdgcn-amd-amdhsa--gfx803\"\n.text\n.globl probe\n.p2align 8\n.type probe,@function\nprobe:\n");
        line(&mut text, "s_load_dwordx4 s[40:43], s[4:5], 0x0");
        if self.texture.is_some() {
            line(&mut text, "s_load_dwordx2 s[46:47], s[4:5], 0x10");
        }
        line(&mut text, "s_lshl_b32 s44, s6, 6");
        line(&mut text, "v_add_u32 v44, vcc, s44, v0");
        line(&mut text, &format!("v_mul_u32_u24 v40, {}, v44", self.inputs.max(1) * 4));
        line(&mut text, &format!("v_mul_u32_u24 v41, {}, v44", self.outputs.max(1) * 4));
        line(&mut text, "s_waitcnt lgkmcnt(0)");
        line(&mut text, "v_mov_b32 v43, s41");
        line(&mut text, "v_add_u32 v42, vcc, s40, v40");
        line(&mut text, "v_addc_u32 v43, vcc, v43, 0, vcc");
        for k in 0..self.inputs {
            line(&mut text, &format!("flat_load_dword v{}, v[42:43]", k + 1));
            line(&mut text, "v_add_u32 v42, vcc, 4, v42");
            line(&mut text, "v_addc_u32 v43, vcc, 0, v43, vcc");
        }
        line(&mut text, "s_waitcnt vmcnt(0)");
        for body in self.body.lines() {
            line(&mut text, body.trim());
        }
        line(&mut text, "v_mov_b32 v43, s43");
        line(&mut text, "v_add_u32 v42, vcc, s42, v41");
        line(&mut text, "v_addc_u32 v43, vcc, v43, 0, vcc");
        for k in 0..self.outputs {
            line(&mut text, &format!("flat_store_dword v[42:43], v{}", 20 + k));
            line(&mut text, "v_add_u32 v42, vcc, 4, v42");
            line(&mut text, "v_addc_u32 v43, vcc, 0, v43, vcc");
        }
        line(&mut text, "s_endpgm");
        text.push_str(".Lend:\n.size probe, .Lend-probe\n\n.rodata\n.p2align 6\n.amdhsa_kernel probe\n");
        line(&mut text, ".amdhsa_user_sgpr_private_segment_buffer 1");
        line(&mut text, ".amdhsa_user_sgpr_kernarg_segment_ptr 1");
        line(&mut text, ".amdhsa_system_sgpr_workgroup_id_x 1");
        line(&mut text, &format!(".amdhsa_system_sgpr_private_segment_wavefront_offset {}", self.wave_offset as u32));
        line(&mut text, &format!(".amdhsa_private_segment_fixed_size {}", self.private));
        line(&mut text, &format!(".amdhsa_group_segment_fixed_size {}", self.lds));
        line(&mut text, ".amdhsa_next_free_vgpr 48");
        line(&mut text, ".amdhsa_next_free_sgpr 48");
        line(&mut text, &format!(".amdhsa_float_denorm_mode_32 {}", self.denorm32));
        text.push_str(".end_amdhsa_kernel\n\n.amdgpu_metadata\n---\namdhsa.version: [ 1, 2 ]\namdhsa.target: amdgcn-amd-amdhsa--gfx803\namdhsa.kernels:\n");
        let kernarg = if self.texture.is_some() { 24 } else { 16 };
        text.push_str(&format!("  - .name: probe\n    .symbol: probe.kd\n    .kernarg_segment_size: {}\n", kernarg));
        text.push_str(&format!("    .group_segment_fixed_size: {}\n    .private_segment_fixed_size: {}\n", self.lds, self.private));
        text.push_str("    .kernarg_segment_align: 8\n    .wavefront_size: 64\n    .sgpr_count: 48\n    .vgpr_count: 48\n    .max_flat_workgroup_size: 64\n");
        text.push_str("    .args:\n      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }\n      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }\n");
        if self.texture.is_some() {
            text.push_str("      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }\n");
        }
        text.push_str("...\n.end_amdgpu_metadata\n");
        text
    }

    pub fn object(&self) -> Vec<u8> {
        let directory = std::env::temp_dir().join(format!(
            "gcn3-probe-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("probe.s");
        let object = directory.join("probe.o");
        let shared = directory.join("probe.co");
        std::fs::write(&source, self.source()).unwrap();
        let assembled = Command::new(tool("llvm-mc"))
            .args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx803", "-filetype=obj"])
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .output()
            .expect("llvm-mc runs");
        assert!(assembled.status.success(), "{}\n{}", String::from_utf8_lossy(&assembled.stderr), self.source());
        let linked = Command::new(tool("ld.lld"))
            .arg("-shared")
            .arg(&object)
            .arg("-o")
            .arg(&shared)
            .output()
            .expect("ld.lld runs");
        assert!(linked.status.success(), "{}", String::from_utf8_lossy(&linked.stderr));
        let bytes = std::fs::read(&shared).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        bytes
    }

    pub fn refusal(&self) -> String {
        let module = Module::load(&self.object()).unwrap_or_else(|e| panic!("{}", e));
        match module.function("probe") {
            Ok(_) => panic!("the frontend accepted\n{}", self.body),
            Err(e) => e.to_string(),
        }
    }

    pub fn run(&self, inputs: &[u32]) -> Vec<Vec<u32>> {
        let waves = if self.inputs == 0 { 1 } else { inputs.len() / (LANES * self.inputs) };
        assert_eq!(inputs.len(), waves * LANES * self.inputs);
        let module = Module::load(&self.object()).unwrap_or_else(|e| panic!("{}", e));
        let function = module.function("probe").unwrap_or_else(|e| panic!("{}", e));
        WIDTHS
            .iter()
            .map(|&width| {
                let given = Buffer::from_slice(if inputs.is_empty() { &[0u32][..] } else { inputs });
                let mut taken = Buffer::from_slice(&vec![0u32; waves * LANES * self.outputs]);
                let launch = Launch::new([waves as u32, 1, 1], [LANES as u32, 1, 1]).width(width);
                let texture = self.texture.as_ref().map(|bytes| Buffer::from_slice(bytes));
                let mut args = vec![Arg::read(&given), Arg::write(&mut taken)];
                if let Some(texture) = &texture {
                    args.push(Arg::read(texture));
                }
                function
                    .launch(&launch, &args)
                    .unwrap_or_else(|e| panic!("W={}: {}", width, e));
                taken.as_slice::<u32>().to_vec()
            })
            .collect()
    }

    pub fn check(&self, inputs: &[u32], expected: impl Fn(usize, &[u32]) -> Vec<u32>) {
        self.compare(inputs, expected, |w| w)
    }

    pub fn check_float(&self, inputs: &[u32], expected: impl Fn(usize, &[u32]) -> Vec<u32>) {
        self.compare(inputs, expected, |w| if w & 0x7f80_0000 == 0x7f80_0000 && w & 0x007f_ffff != 0 { 0x7fc0_0000 } else { w })
    }

    fn compare(&self, inputs: &[u32], expected: impl Fn(usize, &[u32]) -> Vec<u32>, canonical: fn(u32) -> u32) {
        let lanes = if self.inputs == 0 { LANES } else { inputs.len() / self.inputs };
        let want: Vec<u32> = (0..lanes)
            .flat_map(|lane| expected(lane % LANES, &inputs[lane * self.inputs..(lane + 1) * self.inputs]))
            .map(canonical)
            .collect();
        for (got, width) in self.run(inputs).iter().zip(WIDTHS) {
            let got: Vec<u32> = got.iter().map(|&w| canonical(w)).collect();
            if got == want {
                continue;
            }
            let at = got.iter().zip(&want).position(|(g, w)| g != w).unwrap();
            let lane = at / self.outputs;
            panic!(
                "W={}: lane {} output {} is {:#010x}, expected {:#010x}; inputs {:x?}\n{}",
                width,
                lane,
                at % self.outputs,
                got[at],
                want[at],
                &inputs[lane * self.inputs..(lane + 1) * self.inputs],
                self.body
            );
        }
    }
}
