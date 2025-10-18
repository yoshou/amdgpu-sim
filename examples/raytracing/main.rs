use yaml_rust::yaml::*;

use amdgpu_sim::buffer::*;
use amdgpu_sim::processor::*;
use amdgpu_sim::rdna_processor::*;
use getopts::Options;
use object::*;
use png::*;
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

fn ceil_div<T>(x: T, y: T) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + From<u32>
        + Copy,
{
    (x + y - T::from(1)) / y
}

fn load_geometry(filename: &str) -> Result<aligned_vec::AVec<u8>> {
    let mut file = File::open(filename)?;
    let mut data = vec![];
    file.read_to_end(&mut data)?;

    let mut aligned_data = aligned_vec::AVec::new(128);
    aligned_data.extend_from_slice(&data);

    Ok(aligned_data)
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

    let res_x = 512;
    let res_y = 512;
    let ao_radius = 200f32;

    let mut pixels = vec![0u8; res_x * res_y * 4];

    for i in 0..(res_x * res_y) {
        pixels[i * 4 + 0] = 0;
        pixels[i * 4 + 1] = 0;
        pixels[i * 4 + 2] = 0;
        pixels[i * 4 + 3] = 255;
    }

    let pixels_ptr = pixels.as_mut_ptr() as u64;
    println!("pixels_ptr: 0x{:016X}", pixels_ptr);

    let mut geometry = load_geometry("examples/raytracing/cornellbox.bin")?;

    println!("geometry size: {}", geometry.len());
    let geometry_ptr = geometry.as_ptr() as u64;
    println!("geometry_ptr: 0x{:016X}", geometry_ptr);

    let box_nodes = get_u64(&geometry, 0);
    set_u64(&mut geometry, 0, box_nodes + geometry_ptr);
    let prim_nodes = get_u64(&geometry, 8);
    set_u64(&mut geometry, 8, prim_nodes + geometry_ptr);

    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx1200".to_string()
    };

    let program_filename = format!("examples/raytracing/kernel_{}.o", arch);
    let kernel_name = "_Z24ambient_occlusion_kernelP14_hiprtGeometryPh15HIP_vector_typeIiLj2EEf.kd";
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
        let (kernarg_seg_size, private_segment_size, _) = if let Metadata::Yaml(metadata) = metadata
        {
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

            set_u64(&mut arg_buffer, 0, geometry_ptr as u64);
            set_u64(&mut arg_buffer, 8, pixels_ptr as u64);
            set_u32(&mut arg_buffer, 16, res_x as u32);
            set_u32(&mut arg_buffer, 20, res_y as u32);
            set_u32(&mut arg_buffer, 24, f32::to_bits(ao_radius));

            let nx = res_x as u32;
            let ny = res_y as u32;
            let bx = 8u32;
            let by = 8u32;
            let nbx = ceil_div(nx, bx);
            let nby = ceil_div(ny, by);

            let block_dim = [bx, by, 1];
            let grid_dim = [nbx, nby, 1];

            set_u32(&mut arg_buffer, 32, grid_dim[0]);
            set_u32(&mut arg_buffer, 36, grid_dim[1]);
            set_u32(&mut arg_buffer, 40, grid_dim[2]);

            set_u16(&mut arg_buffer, 44, block_dim[0] as u16);
            set_u16(&mut arg_buffer, 46, block_dim[1] as u16);
            set_u16(&mut arg_buffer, 48, block_dim[2] as u16);

            let aql = HsaKernelDispatchPacket {
                header: 0,
                setup: 0,
                workgroup_size_x: block_dim[0] as u16,
                workgroup_size_y: block_dim[1] as u16,
                workgroup_size_z: block_dim[2] as u16,
                grid_size_x: grid_dim[0],
                grid_size_y: grid_dim[1],
                grid_size_z: grid_dim[2],
                private_segment_size: private_segment_size as u32,
                group_segment_size: 0,
                kernel_object: Pointer::new(&mem, kernel_addr),
                kernarg_address: Pointer::new(&arg_buffer, 0),
            };

            if "gfx1200" == arch {
                let mut processor = RDNAProcessor::new(&aql, 32, 32, &mem);
                processor.execute();
            } else {
                println!("Unsupported architecture: {}", arch);
                return Ok(());
            }
        }
    }

    let file = std::fs::File::create("image.png")?;
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, res_x as u32, res_y as u32);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(pixels.as_slice())?;

    Ok(())
}
