use amdgpu_sim::buffer::*;
use amdgpu_sim::processor::*;
use amdgpu_sim::rdna_spmd::{compile, decode_program, dispatch, CompileOptions, GridDims};
use object::*;
use std::fs::File;
use std::io::Read;

pub(crate) const WIDTHS: [u32; 6] = [1, 2, 4, 8, 16, 32];

fn align(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

#[derive(serde::Deserialize)]
struct ArgMeta {
    #[serde(alias = ".offset")]
    offset: i64,
    #[serde(alias = ".value_kind")]
    value_kind: String,
}

#[derive(serde::Deserialize)]
struct KernelMeta {
    #[serde(alias = ".name")]
    name: String,
    #[serde(alias = ".args")]
    args: Option<Vec<ArgMeta>>,
    #[serde(alias = ".kernarg_segment_size")]
    kernarg_segment_size: i64,
    #[serde(alias = ".private_segment_fixed_size")]
    private_segment_fixed_size: i64,
    #[serde(alias = ".group_segment_fixed_size")]
    group_segment_fixed_size: i64,
}

pub(crate) struct Layout {
    pub(crate) kernarg_size: usize,
    pub(crate) private: usize,
    pub(crate) group: usize,
    pub(crate) explicit: Vec<usize>,
    hidden: Vec<(String, usize)>,
}

impl Layout {
    fn at(&self, name: &str) -> Option<usize> {
        self.hidden
            .iter()
            .find(|(n, _)| n == name)
            .map(|&(_, o)| o)
    }
}

#[derive(serde::Deserialize)]
struct Meta {
    #[serde(alias = "amdhsa.kernels")]
    amdhsa_kernels: Vec<KernelMeta>,
}

fn layout(note: &[u8], kernel: &str) -> Layout {
    let mut pos = 0;
    while pos < note.len() {
        let name_size = get_u32(note, pos) as usize;
        let data_size = get_u32(note, pos + 4) as usize;
        let note_type = get_u32(note, pos + 8) as usize;
        pos = align(pos + 12 + name_size, 4);
        let data = get_bytes(note, pos, data_size);
        pos = align(pos + data_size, 4);
        if note_type == 32 {
            let map: Meta = rmp_serde::from_slice(&data).unwrap();
            let meta = map
                .amdhsa_kernels
                .iter()
                .find(|k| k.name == kernel)
                .unwrap_or_else(|| panic!("no metadata for {}", kernel));
            let args = meta.args.as_deref().unwrap_or(&[]);
            return Layout {
                kernarg_size: meta.kernarg_segment_size as usize,
                private: meta.private_segment_fixed_size as usize,
                group: meta.group_segment_fixed_size as usize,
                explicit: args
                    .iter()
                    .filter(|a| !a.value_kind.starts_with("hidden_"))
                    .map(|a| a.offset as usize)
                    .collect(),
                hidden: args
                    .iter()
                    .filter(|a| a.value_kind.starts_with("hidden_"))
                    .map(|a| (a.value_kind.clone(), a.offset as usize))
                    .collect(),
            };
        }
    }
    panic!("no MessagePack metadata note in the kernel object");
}

pub(crate) struct Kernels {
    mem: Vec<u8>,
    symbols: Vec<(String, usize)>,
    note: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum Arg {
    Ptr(u64),
    U32(u32),
    I32(i32),
    F32(f32),
}

pub(crate) struct Run<'a> {
    pub(crate) kernel: &'a str,
    pub(crate) wg: [u32; 3],
    pub(crate) grid: [u32; 3],
    pub(crate) args: &'a [Arg],
}

impl Kernels {
    pub(crate) fn load() -> Self {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/kernels_gfx1200.o");
        let mut data = vec![];
        File::open(path)
            .unwrap_or_else(|e| panic!("{}: {} (run tests/kernels/build.sh)", path, e))
            .read_to_end(&mut data)
            .unwrap();
        let elf = ElfFile::parse(&data).expect("the kernel object parses");

        let mut mem = Vec::<u8>::new();
        for segment in elf.segments() {
            let offset = segment.address() as usize;
            let size = segment.size() as usize;
            let end = mem.len().max(offset + size);
            mem.resize(end, 0);
            let given = size.min(segment.data().len());
            mem[offset..offset + given].copy_from_slice(&segment.data()[..given]);
        }

        let note_data = elf
            .sections()
            .find(|s| s.name() == Some(".note"))
            .expect("the kernel object has a .note section")
            .data()
            .to_vec();
        let note = mem.len();
        mem.extend_from_slice(&note_data);

        let symbols = elf
            .symbols()
            .filter_map(|s| s.name().map(|n| (n.to_string(), s.address() as usize)))
            .collect();

        Kernels { mem, symbols, note }
    }

    fn address(&self, name: &str) -> usize {
        self.symbols
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no symbol {} in the kernel object", name))
            .1
    }

    pub(crate) fn run(&self, spec: &Run, width: u32) {
        self.run_threaded(spec, width, 1)
    }

    pub(crate) fn run_threaded(&self, spec: &Run, width: u32, threads: usize) {
        let layout = layout(&self.mem[self.note..], spec.kernel);
        let kernel_addr = self.address(&format!("{}.kd", spec.kernel));
        let kd = decode_kernel_desc(&self.mem[kernel_addr..kernel_addr + 64]);

        let mut arg = vec![0u8; layout.kernarg_size.max(256)];
        assert_eq!(
            spec.args.len(),
            layout.explicit.len(),
            "{} takes {} arguments, not {}",
            spec.kernel,
            layout.explicit.len(),
            spec.args.len()
        );
        for (value, &at) in spec.args.iter().zip(&layout.explicit) {
            match *value {
                Arg::Ptr(p) => set_u64(&mut arg, at, p),
                Arg::U32(v) => set_u32(&mut arg, at, v),
                Arg::I32(v) => set_u32(&mut arg, at, v as u32),
                Arg::F32(v) => set_f32(&mut arg, at, v),
            }
        }

        let counts = ["hidden_block_count_x", "hidden_block_count_y", "hidden_block_count_z"];
        for (name, value) in counts.iter().zip(spec.grid) {
            if let Some(at) = layout.at(name) {
                set_u32(&mut arg, at, value);
            }
        }
        let sizes = ["hidden_group_size_x", "hidden_group_size_y", "hidden_group_size_z"];
        for (name, value) in sizes.iter().zip(spec.wg) {
            if let Some(at) = layout.at(name) {
                set_u16(&mut arg, at, value as u16);
            }
        }
        let (private, group) = (layout.private, layout.group);

        let aql = HsaKernelDispatchPacket {
            header: 0,
            setup: 0,
            workgroup_size_x: spec.wg[0] as u16,
            workgroup_size_y: spec.wg[1] as u16,
            workgroup_size_z: spec.wg[2] as u16,
            grid_size_x: spec.wg[0] * spec.grid[0],
            grid_size_y: spec.wg[1] * spec.grid[1],
            grid_size_z: spec.wg[2] * spec.grid[2],
            private_segment_size: private as u32,
            group_segment_size: group as u32,
            kernel_object: Pointer::new(&self.mem, kernel_addr),
            kernarg_address: Pointer::new(&arg, 0),
        };

        let entry = kernel_addr + kd.kernel_code_entry_byte_offset;
        let program = decode_program("gfx1200", entry, &self.mem)
            .unwrap_or_else(|e| panic!("{} decode: {}", spec.kernel, e));
        let kernel = compile(
            &program,
            CompileOptions {
                width,
                num_vgprs: kd.granulated_workitem_vgpr_count,
                workgroup_x: Some(spec.wg[0]),
            },
        );
        dispatch(
            &kernel,
            &kd,
            arg.as_ptr() as u64,
            &aql as *const HsaKernelDispatchPacket as u64,
            GridDims {
                num_wg_x: spec.grid[0],
                num_wg_y: spec.grid[1],
                num_wg_z: spec.grid[2],
                wg_x: spec.wg[0],
                wg_y: spec.wg[1],
                wg_z: spec.wg[2],
            },
            private as u32,
            group,
            threads,
        );
    }
}

pub(crate) fn same<T: PartialEq + std::fmt::Debug>(
    kernel: &str,
    width: u32,
    against: &str,
    got: &[T],
    want: &[T],
) {
    if got == want {
        return;
    }
    let (at, (g, w)) = got
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (g, w))| g != w)
        .unwrap_or_else(|| panic!("{}: W={} produced {} values, {} has {}", kernel, width, got.len(), against, want.len()));
    let differing = got.iter().zip(want).filter(|(g, w)| g != w).count();
    panic!(
        "{}: W={} differs from {} at {} of {} places; \
         first at index {}: got {:?}, {} gave {:?}",
        kernel, width, against, differing, got.len(), at, g, against, w
    );
}

pub(crate) fn close(kernel: &str, width: u32, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (at, (&g, &w)) in got.iter().zip(want).enumerate() {
        let slack = tol * w.abs().max(1.0);
        if (g - w).abs() > slack {
            panic!(
                "{}: W={} differs from the reference at index {}: got {}, expected {} (tolerance {})",
                kernel, width, at, g, w, slack
            );
        }
    }
}
