use super::buffer::{bytes_of, Buffer, Pod};
use super::module::{Image, KernelMetadata};
use super::Error;
use crate::processor::decode_kernel_desc;
use crate::rdna_spmd::compiler::{decode_program, Jit};
use crate::rdna_spmd::engine::{dispatch, GridDims, Kernel};
use crate::rdna_spmd::environment::{Binding, Environment};
use std::sync::{Arc, Mutex};

const DYNAMIC_STACK: u32 = 0x2000;

#[derive(Clone, Debug)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub width: u32,
    pub threads: usize,
    pub dynamic_shared: u32,
}

impl Launch {
    pub fn new(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            width: 16,
            threads: std::thread::available_parallelism().map_or(8, |n| n.get()),
            dynamic_shared: 0,
        }
    }

    pub fn width(self, width: u32) -> Self {
        Self { width, ..self }
    }

    pub fn threads(self, threads: usize) -> Self {
        Self {
            threads: threads.max(1),
            ..self
        }
    }

    pub fn dynamic_shared(self, bytes: u32) -> Self {
        Self {
            dynamic_shared: bytes,
            ..self
        }
    }
}

pub enum Arg<'a> {
    Read(&'a Buffer, usize),
    Write(&'a mut Buffer, usize),
    Value(Vec<u8>),
}

impl<'a> Arg<'a> {
    pub fn read(buffer: &'a Buffer) -> Self {
        Arg::Read(buffer, 0)
    }

    pub fn write(buffer: &'a mut Buffer) -> Self {
        Arg::Write(buffer, 0)
    }

    pub fn value<T: Pod>(value: T) -> Self {
        Arg::Value(bytes_of(std::slice::from_ref(&value)).to_vec())
    }

    fn pointer(&self) -> Option<(&Buffer, usize)> {
        match self {
            Arg::Read(buffer, offset) => Some((buffer, *offset)),
            Arg::Write(buffer, offset) => Some((buffer, *offset)),
            Arg::Value(_) => None,
        }
    }
}

#[repr(C)]
struct DispatchPacket {
    header: u16,
    setup: u16,
    workgroup_size: [u16; 3],
    reserved0: u16,
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    reserved2: u64,
    completion_signal: u64,
}

struct Bound {
    kernarg: Vec<u64>,
    packet: Box<DispatchPacket>,
    dims: GridDims,
    private: u32,
    group: usize,
    environment: Environment,
}

pub struct Function {
    name: String,
    image: Arc<Image>,
    metadata: KernelMetadata,
    descriptor_address: usize,
    jit: Mutex<Jit>,
}

impl Function {
    pub(crate) fn new(image: Arc<Image>, target: &str, metadata: KernelMetadata) -> Result<Self, Error> {
        let name = metadata.name.clone();
        let descriptor_address = image
            .symbol(&metadata.symbol)
            .ok_or_else(|| Error::new(format!("{}: no kernel descriptor symbol {}", name, metadata.symbol)))?;
        let bytes = image
            .memory
            .get(descriptor_address..descriptor_address + 64)
            .ok_or_else(|| Error::new(format!("{}: the kernel descriptor lies outside the code object", name)))?;
        let descriptor = decode_kernel_desc(bytes);
        let entry = descriptor_address + descriptor.kernel_code_entry_byte_offset;
        let program = decode_program(target, &descriptor, entry, &image.memory, metadata.wavefront_size)
            .map_err(|e| Error::new(format!("{}: {}", name, e)))?;
        let jit = Mutex::new(Jit::new(program, descriptor.granulated_workitem_vgpr_count));
        Ok(Self {
            name,
            image,
            metadata,
            descriptor_address,
            jit,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn prepare(&self, launch: &Launch, args: &[Arg]) -> Result<(), Error> {
        let bound = self.bind(launch, args)?;
        self.kernel(launch, &bound.environment);
        Ok(())
    }

    pub fn launch(&self, launch: &Launch, args: &[Arg]) -> Result<(), Error> {
        let bound = self.bind(launch, args)?;
        let kernel = self.kernel(launch, &bound.environment);
        dispatch(
            &kernel,
            bound.kernarg.as_ptr() as u64,
            &*bound.packet as *const DispatchPacket as u64,
            bound.dims,
            bound.private,
            bound.group,
            launch.threads,
        );
        Ok(())
    }

    fn kernel(&self, launch: &Launch, environment: &Environment) -> Arc<Kernel> {
        self.jit.lock().unwrap().kernel(launch.width, environment)
    }

    fn bind(&self, launch: &Launch, args: &[Arg]) -> Result<Bound, Error> {
        let name = &self.name;
        if !matches!(launch.width, 1 | 2 | 4 | 8 | 16 | 32) {
            return Err(Error::new(format!("{}: unsupported packet width {}", name, launch.width)));
        }
        let size: u64 = launch.block.iter().map(|&n| n as u64).product();
        if size == 0 || size > self.metadata.max_flat_workgroup_size as u64 {
            return Err(Error::new(format!(
                "{}: a workgroup of {:?} work-items, but the kernel takes at most {}",
                name, launch.block, self.metadata.max_flat_workgroup_size
            )));
        }
        if launch.grid.contains(&0) {
            return Err(Error::new(format!("{}: an empty grid {:?}", name, launch.grid)));
        }
        let segment = self.metadata.kernarg_segment_size as usize;
        let mut kernarg = vec![0u64; segment.div_ceil(8).max(1)];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(kernarg.as_mut_ptr() as *mut u8, kernarg.len() * 8)
        };
        let mut put = |offset: u32, value: &[u8]| {
            let at = offset as usize;
            bytes[at..at + value.len()].copy_from_slice(value);
        };
        let explicit: Vec<_> = self
            .metadata
            .args
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden_"))
            .collect();
        if explicit.len() != args.len() {
            return Err(Error::new(format!(
                "{} takes {} arguments, not {}",
                name,
                explicit.len(),
                args.len()
            )));
        }
        let mut bindings = Vec::new();
        let mut exposed = Vec::new();
        for (index, (meta, arg)) in explicit.iter().zip(args).enumerate() {
            match (meta.value_kind.as_str(), arg) {
                ("global_buffer", arg) if arg.pointer().is_some() => {
                    let (buffer, offset) = arg.pointer().unwrap();
                    bindings.push(Binding {
                        offset: meta.offset,
                        allocation: buffer.allocation(),
                        pointer: buffer.base() + offset as u64,
                    });
                    if buffer.exposed() && !exposed.contains(&buffer.allocation()) {
                        exposed.push(buffer.allocation());
                    }
                    if offset > buffer.len() {
                        return Err(Error::new(format!(
                            "{}: argument {} points {} bytes into a buffer of {}",
                            name,
                            index,
                            offset,
                            buffer.len()
                        )));
                    }
                    put(meta.offset, &(buffer.base() + offset as u64).to_le_bytes());
                }
                ("by_value", Arg::Value(value)) if value.len() == meta.size as usize => {
                    put(meta.offset, value)
                }
                (kind, _) => {
                    return Err(Error::new(format!(
                        "{}: argument {} is a {} of {} bytes, which the given argument is not",
                        name, index, kind, meta.size
                    )))
                }
            }
        }
        let dims = launch.grid.iter().zip(&launch.block).filter(|&(&g, &b)| g * b > 1).count();
        for meta in self.metadata.args.iter().filter(|a| a.value_kind.starts_with("hidden_")) {
            let axis = |suffix: &str| ["x", "y", "z"].iter().position(|&a| a == suffix);
            let kind = meta.value_kind.as_str();
            let (field, suffix) = kind.rsplit_once('_').unwrap_or((kind, ""));
            let value: Vec<u8> = match (field, axis(suffix)) {
                ("hidden_block_count", Some(a)) => launch.grid[a].to_le_bytes().to_vec(),
                ("hidden_group_size", Some(a)) => (launch.block[a] as u16).to_le_bytes().to_vec(),
                ("hidden_grid", None) if suffix == "dims" => ((dims.max(1)) as u16).to_le_bytes().to_vec(),
                ("hidden_dynamic_lds", None) if suffix == "size" => {
                    launch.dynamic_shared.to_le_bytes().to_vec()
                }
                _ => continue,
            };
            put(meta.offset, &value[..(meta.size as usize).min(value.len())]);
        }
        let private = self.metadata.private_segment_fixed_size
            + if self.metadata.uses_dynamic_stack {
                DYNAMIC_STACK
            } else {
                0
            };
        let group = self.metadata.group_segment_fixed_size as usize + launch.dynamic_shared as usize;
        let environment = Environment {
            grid: launch.grid,
            block: launch.block,
            kernarg: bytes.to_vec(),
            bindings,
            exposed,
        };
        let packet = Box::new(DispatchPacket {
            header: 0,
            setup: dims.max(1) as u16,
            workgroup_size: launch.block.map(|n| n as u16),
            reserved0: 0,
            grid_size: [0, 1, 2].map(|a| launch.grid[a] * launch.block[a]),
            private_segment_size: private,
            group_segment_size: group as u32,
            kernel_object: self.image.memory.as_ptr() as u64 + self.descriptor_address as u64,
            kernarg_address: kernarg.as_ptr() as u64,
            reserved2: 0,
            completion_signal: 0,
        });
        Ok(Bound {
            kernarg,
            packet,
            dims: GridDims {
                num_wg_x: launch.grid[0],
                num_wg_y: launch.grid[1],
                num_wg_z: launch.grid[2],
                wg_x: launch.block[0],
                wg_y: launch.block[1],
                wg_z: launch.block[2],
            },
            private,
            group,
            environment,
        })
    }
}
