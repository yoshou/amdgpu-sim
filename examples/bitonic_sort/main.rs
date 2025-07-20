use yaml_rust::yaml::*;

use amdgpu_sim::gcn_processor::*;
use amdgpu_sim::processor::*;
use object::*;
use std::env;
use std::fs::File;
use std::io::*;

fn get_u8(buffer: &[u8], offset: usize) -> u8 {
    buffer[offset]
}

fn get_u32(buffer: &[u8], offset: usize) -> u32 {
    let b0 = buffer[offset] as u32;
    let b1 = buffer[offset + 1] as u32;
    let b2 = buffer[offset + 2] as u32;
    let b3 = buffer[offset + 3] as u32;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

fn get_u64(buffer: &[u8], offset: usize) -> u64 {
    let b0 = buffer[offset] as u64;
    let b1 = buffer[offset + 1] as u64;
    let b2 = buffer[offset + 2] as u64;
    let b3 = buffer[offset + 3] as u64;
    let b4 = buffer[offset + 4] as u64;
    let b5 = buffer[offset + 5] as u64;
    let b6 = buffer[offset + 6] as u64;
    let b7 = buffer[offset + 7] as u64;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
}

fn get_f32(buffer: &[u8], offset: usize) -> f32 {
    let arr: [u8; 4] = [
        buffer[offset],
        buffer[offset + 1],
        buffer[offset + 2],
        buffer[offset + 3],
    ];
    unsafe { std::mem::transmute::<[u8; 4], f32>(arr) }
}

fn set_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset] = (value & 0xFF) as u8;
    buffer[offset + 1] = ((value >> 8) & 0xFF) as u8;
}

fn set_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset] = (value & 0xFF) as u8;
    buffer[offset + 1] = ((value >> 8) & 0xFF) as u8;
    buffer[offset + 2] = ((value >> 16) & 0xFF) as u8;
    buffer[offset + 3] = ((value >> 24) & 0xFF) as u8;
}

fn set_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset] = (value & 0xFF) as u8;
    buffer[offset + 1] = ((value >> 8) & 0xFF) as u8;
    buffer[offset + 2] = ((value >> 16) & 0xFF) as u8;
    buffer[offset + 3] = ((value >> 24) & 0xFF) as u8;
    buffer[offset + 4] = ((value >> 32) & 0xFF) as u8;
    buffer[offset + 5] = ((value >> 40) & 0xFF) as u8;
    buffer[offset + 6] = ((value >> 48) & 0xFF) as u8;
    buffer[offset + 7] = ((value >> 56) & 0xFF) as u8;
}

fn set_f32(buffer: &mut [u8], offset: usize, value: f32) {
    unsafe { set_u32(buffer, offset, std::mem::transmute::<f32, u32>(value)) };
}

fn get_str(buffer: &[u8], offset: usize, size: usize) -> String {
    let bytes: &[u8] = &buffer[offset..(offset + size)];
    bytes.iter().map(|&s| s as char).collect::<String>()
}

fn get_bytes(buffer: &[u8], offset: usize, size: usize) -> Vec<u8> {
    let bytes: &[u8] = &buffer[offset..(offset + size)];
    bytes.to_vec()
}

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
        let name = get_str(buffer, pos, name_size);
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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Must pass program file.");
        return Ok(());
    }

    let length = 128;

    let num_stages = 31 - (length as u32).leading_zeros();

    let mut input = vec![0u8; length * 4];

    println!("---------------------------------------------");
    for i in 0..length {
        let value = rand::random::<u32>();
        set_u32(&mut input, i * 4, value);
        println!("{}", value);
    }
    println!("---------------------------------------------");
    let input_ptr = (&input[0] as *const u8) as u64;
    println!("input_ptr: 0x{:16X}", input_ptr);

    let program_filename = &args[1];
    let kernel_name = "bitonicSort";
    let mut file = File::open(program_filename).unwrap();
    let mut data = vec![];
    file.read_to_end(&mut data).unwrap();
    if let Ok(elffile) = ElfFile::parse(&data) {
        println!("Elf file was successfully loaded.");

        let note_section_data = elffile
            .sections()
            .find(|section| section.name() == Some(".note"))
            .unwrap();

        let mut arg_buffer: Vec<u8> = Vec::new();
        let metadata = decode_note_metadata(note_section_data.data()).unwrap();
        if let Metadata::Yaml(metadata) = metadata {
            let metadatas = YamlLoader::load_from_str(&metadata).unwrap();
            let metadata = &metadatas[0];
            let args_desc = &metadata["Kernels"][0]["Args"];

            if let Yaml::Array(array) = args_desc {
                for arg in array {
                    let size = if let Yaml::Integer(integer) = arg["Size"] {
                        integer
                    } else {
                        0
                    } as usize;
                    let alignment = if let Yaml::Integer(integer) = arg["Align"] {
                        integer
                    } else {
                        1
                    } as usize;
                    let arg_size = arg_buffer.len();
                    let arg_size_align = align(arg_size, alignment);
                    for _ in arg_size..arg_size_align {
                        arg_buffer.push(0);
                    }
                    for _ in 0..size {
                        arg_buffer.push(0);
                    }
                }
            }
        } else {
            panic!()
        }

        set_u64(&mut arg_buffer, 0, input_ptr);

        let sort_increasing = 1;
        set_u32(&mut arg_buffer, 16, sort_increasing);

        println!("argument size: {}", arg_buffer.len());

        let kernel_arg_ptr = (&arg_buffer[0] as *const u8) as u64;

        println!("kernel_arg_ptr: 0x{:X}", kernel_arg_ptr);

        let mut mem = Vec::<u8>::new();
        for segment in elffile.segments() {
            let offset = segment.address() as usize;
            let size = segment.size() as usize;
            let new_size = mem.len().max(offset + size);
            mem.resize(new_size, 0);
            mem[offset..(offset + size)].copy_from_slice(segment.data());
        }

        if let Some(text_section) = elffile
            .sections()
            .find(|section| section.name() == Some(".text"))
        {
            let text_addr = text_section.address() as usize;
            let text_len = text_section.size() as usize;
            if let Some(kernel_sym) = elffile
                .symbols()
                .find(|sym| sym.name() == Some(kernel_name))
            {
                let kernel_addr = kernel_sym.address() as usize;

                if (kernel_addr < text_addr) || (kernel_addr >= (text_addr + text_len)) {
                    println!("Invalid kernel address.");
                    return Ok(());
                }
                for stage in 0..num_stages {
                    set_u32(&mut arg_buffer, 8, stage);

                    for pass_of_stage in 0..(stage + 1) {
                        set_u32(&mut arg_buffer, 12, pass_of_stage);
                        let aql = hsa_kernel_dispatch_packet_s {
                            header: 0,
                            setup: 0,
                            workgroup_size_x: 256,
                            workgroup_size_y: 1,
                            workgroup_size_z: 1,
                            grid_size_x: (length / 2) as u32,
                            grid_size_y: 1,
                            grid_size_z: 1,
                            private_segment_size: 0,
                            group_segment_size: 0,
                            kernel_object: Pointer::new(&data, kernel_addr),
                            kernarg_address: Pointer::new(&arg_buffer, 0),
                        };
                        let mut processor = GCNProcessor::new(&aql, 1, &mem);
                        processor.execute();
                    }
                }
            }
        }

        println!("---------------------------------------------");
        for i in 0..length {
            let value = get_u32(&input, i * 4);
            println!("{}", value);
        }
        println!("---------------------------------------------");
    }

    Ok(())
}
