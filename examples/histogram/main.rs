use yaml_rust::yaml::*;

use amdgpu_sim::buffer::*;
use amdgpu_sim::gcn_processor::*;
use amdgpu_sim::processor::*;
use getopts::Options;
use object::*;
use std::env;
use std::fs::File;
use std::io::*;

fn align(value: usize, align: usize) -> usize {
    ((value + align - 1) / align) * align
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct KernelArgumentMetadataMapV5 {
    #[serde(alias = ".name")]
    name: Option<String>,
    #[serde(alias = ".type_name")]
    type_name: Option<String>,
    #[serde(alias = ".size")]
    size: i32,
    #[serde(alias = ".offset")]
    offset: i32,
    #[serde(alias = ".value_kind")]
    value_kind: String,
    #[serde(alias = ".value_type")]
    value_type: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct KernelMetadataMapV5 {
    #[serde(alias = ".name")]
    name: String,
    #[serde(alias = ".symbol")]
    symbol: String,
    #[serde(alias = ".language")]
    language: Option<String>,
    #[serde(alias = ".language_version")]
    language_version: Option<Vec<i32>>,
    #[serde(alias = ".args")]
    args: Option<Vec<KernelArgumentMetadataMapV5>>,
    #[serde(alias = ".kernarg_segment_size")]
    kernarg_segment_size: i64,
    #[serde(alias = ".group_segment_fixed_size")]
    group_segment_fixed_size: i64,
    #[serde(alias = ".private_segment_fixed_size")]
    private_segment_fixed_size: i64,
    #[serde(alias = ".kernarg_segment_align")]
    kernarg_segment_align: i64,
    #[serde(alias = ".wavefront_size")]
    wavefront_size: i64,
    #[serde(alias = ".sgpr_count")]
    sgpr_count: i64,
    #[serde(alias = ".vgpr_count")]
    vgpr_count: i64,
    #[serde(alias = ".agpr_count")]
    agpr_count: Option<i64>,
    #[serde(alias = ".max_flat_workgroup_size")]
    max_flat_workgroup_size: i64,
    #[serde(alias = ".uses_dynamic_stack")]
    uses_dynamic_stack: Option<bool>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct MetadataMapV5 {
    #[serde(alias = "amdhsa.version")]
    amdhsa_version: Vec<i32>,
    #[serde(alias = "amdhsa.printf")]
    amdhsa_printf: Option<Vec<String>>,
    #[serde(alias = "amdhsa.kernels")]
    amdhsa_kernels: Vec<KernelMetadataMapV5>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct MetadataMapVersion {
    #[serde(alias = "amdhsa.version")]
    amdhsa_version: Vec<i32>,
}

enum Metadata {
    Yaml(String),
    MessagePack(Vec<u8>),
}

fn decode_note_metadata(buffer: &[u8]) -> Option<Metadata> {
    let mut pos = 0;

    while pos < buffer.len() {
        let name_size = get_u32(buffer, pos) as usize;
        pos += 4;
        let data_size = get_u32(buffer, pos) as usize;
        pos += 4;
        let note_type = get_u32(buffer, pos) as usize;
        pos += 4;
        let _name = get_str(buffer, pos, name_size);
        pos += name_size;
        pos = align(pos, 4);
        let data = get_bytes(buffer, pos, data_size);
        pos += data_size;
        pos = align(pos, 4);

        if note_type == 10 {
            return Some(Metadata::Yaml(
                data.iter().map(|&s| s as char).collect::<String>(),
            ));
        }
        if note_type == 32 {
            return Some(Metadata::MessagePack(data));
        }
    }

    None
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [OPTIONS]", program);
    print!("{}", opts.usage(&brief));
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("", "arch", "Architecture", "ARCH");
    opts.optflag("h", "help", "Print help");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => {
            panic!("{}", f.to_string())
        }
    };
    if matches.opt_present("h") {
        print_usage(&program, opts);
        return Ok(());
    }

    let size = 1024 * 1024;
    let items_per_thread = 1024;
    let threads_per_block = 128;
    let bin_size = 256;
    let total_blocks = size / (items_per_thread * threads_per_block);

    let mut input = vec![0u8; size];

    for i in 0..size {
        let value = rand::random::<u8>();
        input[i] = value;
    }
    let input_ptr = (&input[0] as *const u8) as u64;
    println!("input_ptr: 0x{:16X}", input_ptr);

    let mut block_bins = vec![0u32; bin_size * total_blocks];
    let block_bins_ptr = (&block_bins[0] as *const u32) as u64;
    println!("block_bins_ptr: 0x{:16X}", block_bins_ptr);

    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx803".to_string()
    };

    let program_filename = format!("examples/histogram/kernel_{}.o", arch);
    let kernel_name = "_Z18histogram256_blockPhPji.kd";
    let mut file = File::open(program_filename).unwrap();
    let mut data = vec![];
    file.read_to_end(&mut data).unwrap();
    if let Ok(elffile) = ElfFile::parse(&data) {
        println!("Elf file was successfully loaded.");

        let note_section_data = elffile
            .sections()
            .find(|section| section.name() == Some(".note"))
            .unwrap();

        let metadata = decode_note_metadata(note_section_data.data()).unwrap();
        let (kernarg_seg_size, private_segment_size, wavefront_size) =
            if let Metadata::Yaml(metadata) = metadata {
                let metadatas = YamlLoader::load_from_str(&metadata).unwrap();
                let metadata = &metadatas[0];

                let kernarg_seg_size = if let Yaml::Integer(integer) =
                    metadata["Kernels"][0]["CodeProps"]["KernargSegmentSize"]
                {
                    integer
                } else {
                    1
                } as usize;
                let private_seg_fixed_size = if let Yaml::Integer(integer) =
                    metadata["Kernels"][0]["CodeProps"]["PrivateSegmentFixedSize"]
                {
                    integer
                } else {
                    1
                } as usize;
                let is_dynamic_call_stack = if let Yaml::Boolean(integer) =
                    metadata["Kernels"][0]["CodeProps"]["IsDynamicCallStack"]
                {
                    integer
                } else {
                    false
                };

                let stack_size = if is_dynamic_call_stack { 0x2000 } else { 0 };
                let private_segment_size = private_seg_fixed_size + stack_size;
                let wavefront_size = if let Yaml::Integer(integer) =
                    metadata["Kernels"][0]["CodeProps"]["WavefrontSize"]
                {
                    integer
                } else {
                    panic!("Wavefront size not found in metadata")
                } as usize;

                (kernarg_seg_size, private_segment_size, wavefront_size)
            } else if let Metadata::MessagePack(metadata) = metadata {
                let version: MetadataMapVersion = rmp_serde::from_slice(&metadata).unwrap();
                if version.amdhsa_version[0] == 1 && version.amdhsa_version[1] == 2 {
                    let map: MetadataMapV5 = rmp_serde::from_slice(&metadata).unwrap();
                    let kernarg_seg_size = map.amdhsa_kernels[0].kernarg_segment_size as usize;
                    let private_seg_fixed_size =
                        map.amdhsa_kernels[0].private_segment_fixed_size as usize;

                    let is_dynamic_call_stack =
                        if let Some(value) = map.amdhsa_kernels[0].uses_dynamic_stack {
                            value
                        } else {
                            false
                        };

                    let stack_size = if is_dynamic_call_stack { 0x2000 } else { 0 };
                    let private_segment_size = private_seg_fixed_size + stack_size;
                    let wavefront_size = map.amdhsa_kernels[0].wavefront_size as usize;

                    (kernarg_seg_size, private_segment_size, wavefront_size)
                } else {
                    panic!()
                }
            } else {
                panic!()
            };

        let mut arg_buffer = vec![0u8; kernarg_seg_size];

        println!("argument size: {}", arg_buffer.len());

        let kernel_arg_ptr = (&arg_buffer[0] as *const u8) as u64;

        println!("kernel_arg_ptr: 0x{:X}", kernel_arg_ptr);

        let mut mem = Vec::<u8>::new();
        for segment in elffile.segments() {
            let offset = segment.address() as usize;
            let size = segment.size() as usize;
            let new_size = mem.len().max(offset + size);
            mem.resize(new_size, 0);
            mem[offset..(offset + size.min(segment.data().len()))].copy_from_slice(segment.data());
        }

        if let Some(kernel_sym) = elffile
            .symbols()
            .find(|sym| sym.name() == Some(kernel_name))
        {
            let kernel_addr = kernel_sym.address() as usize;

            set_u64(&mut arg_buffer, 0, input_ptr);
            set_u64(&mut arg_buffer, 8, block_bins_ptr);
            set_u32(&mut arg_buffer, 16, items_per_thread as u32);

            set_u32(&mut arg_buffer, 24, total_blocks as u32);
            set_u32(&mut arg_buffer, 28, 1);
            set_u32(&mut arg_buffer, 32, 1);

            set_u16(&mut arg_buffer, 36, threads_per_block as u16);
            set_u16(&mut arg_buffer, 38, 1);
            set_u16(&mut arg_buffer, 40, 1);

            let aql = HsaKernelDispatchPacket {
                header: 0,
                setup: 0,
                workgroup_size_x: threads_per_block as u16,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                grid_size_x: (threads_per_block * total_blocks) as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: private_segment_size as u32,
                group_segment_size: 0,
                kernel_object: Pointer::new(&mem, kernel_addr),
                kernarg_address: Pointer::new(&arg_buffer, 0),
            };

            if (arch == "gfx803") && (wavefront_size != 64) {
                println!("Wavefront size must be 64 for gfx803 architectures.");
                return Ok(());
            }

            let mut processor = GCNProcessor::new(&aql, 16, &mem);
            processor.execute();
        }
    }

    let mut bins = vec![0u32; bin_size];
    for i in 0..total_blocks {
        for j in 0..bin_size {
            bins[j] += block_bins[i * bin_size + j];
        }
    }

    let mut verfy_bins = vec![0u32; bin_size];
    for i in 0..size {
        let value = input[i] as usize;
        verfy_bins[value] += 1;
    }

    let mut error = 0;
    for i in 0..bin_size {
        if bins[i] != verfy_bins[i] {
            error += 1;
        }
    }

    if error == 0 {
        println!("Validation passed.");
    } else {
        println!("Validation failed.");
    }

    Ok(())
}
