use yaml_rust::yaml::*;

use amdgpu_sim::buffer::*;
use amdgpu_sim::gcn_processor::*;
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

fn clamp(x: f32, low: f32, high: f32) -> f32 {
    if x > high {
        high
    } else if x < low {
        low
    } else {
        x
    }
}

fn gamma(x: f32) -> f32 {
    x.powf(1.0 / 2.2)
}

fn to_byte(x: f32) -> u8 {
    clamp(255.0 * x, 0.0, 255.0) as u8
}

fn write_png(width: usize, height: usize, data: &[f32], fname: &str) -> Result<()> {
    let file = std::fs::File::create(fname)?;
    let ref mut w = BufWriter::new(file);

    let mut bytes = vec![0; width * height * 4];
    for i in (0..bytes.len()).step_by(4) {
        bytes[i] = to_byte(gamma(data[i]));
        bytes[i + 1] = to_byte(gamma(data[i + 1]));
        bytes[i + 2] = to_byte(gamma(data[i + 2]));
        bytes[i + 3] = 255;
    }

    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(&bytes)?;

    Ok(())
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

#[repr(C)]
pub enum ReflectionType {
    Diffuse,
    Specular,
    Refractive,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

#[repr(C)]
pub struct Sphere {
    r: f64,
    p: Vector3,
    e: Vector3,
    f: Vector3,
    reflection_type: ReflectionType,
}

static SPHERES: [Sphere; 9] = [
    Sphere {
        r: 1e5,
        p: Vector3 {
            x: 1e5 + 1.0,
            y: 40.8,
            z: 81.6,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.75,
            y: 0.25,
            z: 0.25,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Left
    Sphere {
        r: 1e5,
        p: Vector3 {
            x: -1e5 + 99.0,
            y: 40.8,
            z: 81.6,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.25,
            y: 0.25,
            z: 0.75,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Right
    Sphere {
        r: 1e5,
        p: Vector3 {
            x: 50.0,
            y: 40.8,
            z: 1e5,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.75,
            y: 0.75,
            z: 0.75,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Back
    Sphere {
        r: 1e5,
        p: Vector3 {
            x: 50.0,
            y: 40.8,
            z: -1e5 + 170.0,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Front
    Sphere {
        r: 1e5,
        p: Vector3 {
            x: 50.0,
            y: 1e5,
            z: 81.6,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.75,
            y: 0.75,
            z: 0.75,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Bottom
    Sphere {
        r: 1e5,
        p: Vector3 {
            x: 50.0,
            y: -1e5 + 81.6,
            z: 81.6,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.75,
            y: 0.75,
            z: 0.75,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Top
    Sphere {
        r: 16.5,
        p: Vector3 {
            x: 27.0,
            y: 16.5,
            z: 47.0,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.999,
            y: 0.999,
            z: 0.999,
        },
        reflection_type: ReflectionType::Specular,
    }, //Mirror
    Sphere {
        r: 16.5,
        p: Vector3 {
            x: 73.0,
            y: 16.5,
            z: 78.0,
        },
        e: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f: Vector3 {
            x: 0.999,
            y: 0.999,
            z: 0.999,
        },
        reflection_type: ReflectionType::Refractive,
    }, //Glass
    Sphere {
        r: 600.0,
        p: Vector3 {
            x: 50.0,
            y: 681.6 - 0.27,
            z: 81.6,
        },
        e: Vector3 {
            x: 12.0,
            y: 12.0,
            z: 12.0,
        },
        f: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        reflection_type: ReflectionType::Diffuse,
    }, //Light
];

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [OPTIONS]", program);
    print!("{}", opts.usage(&brief));
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("", "nb_samples", "Number of samples", "NUM");
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

    let nb_samples = if matches.opt_present("nb_samples") {
        matches
            .opt_str("nb_samples")
            .unwrap()
            .parse::<i32>()
            .unwrap()
            / 4
    } else {
        1
    };
    let arch = if matches.opt_present("arch") {
        matches.opt_str("arch").unwrap()
    } else {
        "gfx803".to_string()
    };

    let program_filename = format!("examples/smallpt/kernel_{}.o", arch);
    let width = 1024;
    let height = 768;
    let nb_pixels = width * height;
    let kernel_name = "_ZN7smallptL6kernelEPKNS_6SphereEmjjPNS_7Vector3Ej.kd";

    let ls = vec![
        Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        nb_pixels
    ];

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

        let ls_ptr = (&ls[0] as *const Vector3) as u64;
        let spheres_ptr = (&SPHERES[0] as *const Sphere) as u64;

        set_u64(&mut arg_buffer, 0, spheres_ptr);
        set_u64(&mut arg_buffer, 8, SPHERES.len() as u64);
        set_u32(&mut arg_buffer, 16, width as u32);
        set_u32(&mut arg_buffer, 20, height as u32);
        set_u64(&mut arg_buffer, 24, ls_ptr);
        set_u32(&mut arg_buffer, 32, nb_samples as u32);

        set_u32(&mut arg_buffer, 40, (width / 16) as u32);
        set_u32(&mut arg_buffer, 44, (height / 16) as u32);
        set_u32(&mut arg_buffer, 48, 1);

        set_u16(&mut arg_buffer, 52, 16);
        set_u16(&mut arg_buffer, 54, 16);
        set_u16(&mut arg_buffer, 56, 1);

        println!("spheres_ptr: {}", spheres_ptr);
        println!("ls_ptr: {}", ls_ptr);

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

            let aql = HsaKernelDispatchPacket {
                header: 0,
                setup: 0,
                workgroup_size_x: 16,
                workgroup_size_y: 16,
                workgroup_size_z: 1,
                grid_size_x: (width / 16) as u32,
                grid_size_y: (height / 16) as u32,
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

            if arch == "gfx803" {
                let mut processor = GCNProcessor::new(&aql, 16, &mem);

                use std::time::Instant;
                let start = Instant::now();
                processor.execute();
                let end = start.elapsed();
                println!(
                    "Elapsed time: {}.{:03} [ms]",
                    end.as_secs(),
                    end.subsec_nanos() / 1_000
                );
            } else if "gfx1200" == arch {
                let mut processor = RDNAProcessor::new(&aql, 32, 32, &mem);
                processor.execute();
            } else {
                println!("Unsupported architecture: {}", arch);
                return Ok(());
            }
        }
    }

    let mut img = vec![0.0f32; width * height * 4];
    for i in 0..ls.len() {
        img[i * 4] = ls[i].x as f32;
        img[i * 4 + 1] = ls[i].y as f32;
        img[i * 4 + 2] = ls[i].z as f32;
    }

    write_png(width, height, &img, "image.png")?;

    Ok(())
}
