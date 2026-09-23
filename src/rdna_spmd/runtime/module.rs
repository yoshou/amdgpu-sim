use super::function::Function;
use super::Error;
use object::{ElfFile, Object, ObjectSection, ObjectSegment};
use serde::Deserialize;
use std::sync::Arc;

const NT_AMDGPU_METADATA: u32 = 32;

#[derive(Deserialize)]
struct Metadata {
    #[serde(rename = "amdhsa.target")]
    target: Option<String>,
    #[serde(rename = "amdhsa.kernels")]
    kernels: Vec<KernelMetadata>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct KernelMetadata {
    #[serde(rename = ".name")]
    pub name: String,
    #[serde(rename = ".symbol")]
    pub symbol: String,
    #[serde(rename = ".args", default)]
    pub args: Vec<ArgMetadata>,
    #[serde(rename = ".kernarg_segment_size")]
    pub kernarg_segment_size: u32,
    #[serde(rename = ".group_segment_fixed_size")]
    pub group_segment_fixed_size: u32,
    #[serde(rename = ".private_segment_fixed_size")]
    pub private_segment_fixed_size: u32,
    #[serde(rename = ".uses_dynamic_stack", default)]
    pub uses_dynamic_stack: bool,
    #[serde(rename = ".max_flat_workgroup_size")]
    pub max_flat_workgroup_size: u32,
    #[serde(rename = ".wavefront_size")]
    pub wavefront_size: u32,
}

#[derive(Clone, Deserialize)]
pub(crate) struct ArgMetadata {
    #[serde(rename = ".offset")]
    pub offset: u32,
    #[serde(rename = ".size")]
    pub size: u32,
    #[serde(rename = ".value_kind")]
    pub value_kind: String,
}

pub(crate) struct Image {
    pub memory: Vec<u8>,
    symbols: Vec<(String, usize)>,
}

impl Image {
    pub fn symbol(&self, name: &str) -> Option<usize> {
        self.symbols
            .iter()
            .find(|(n, _)| n == name)
            .map(|&(_, address)| address)
    }
}

pub struct Module {
    image: Arc<Image>,
    target: String,
    kernels: Vec<KernelMetadata>,
}

impl Module {
    pub fn load(code_object: &[u8]) -> Result<Self, Error> {
        let elf = ElfFile::parse(code_object)
            .map_err(|e| Error::new(format!("not a code object: {}", e)))?;
        let mut memory = Vec::new();
        for segment in elf.segments() {
            let start = segment.address() as usize;
            let end = start + segment.size() as usize;
            if memory.len() < end {
                memory.resize(end, 0);
            }
            let data = segment.data();
            let given = data.len().min(end - start);
            memory[start..start + given].copy_from_slice(&data[..given]);
        }
        let symbols = elf
            .symbols()
            .filter_map(|s| s.name().map(|n| (n.to_string(), s.address() as usize)))
            .collect();
        let notes = elf
            .sections()
            .find(|s| s.name() == Some(".note"))
            .ok_or_else(|| Error::new("the code object has no .note section"))?;
        let metadata = metadata(notes.data())?;
        let target = metadata
            .target
            .as_deref()
            .and_then(|t| t.split("--").nth(1))
            .and_then(|t| t.split(':').next())
            .ok_or_else(|| Error::new("the code object metadata names no target"))?
            .to_string();
        Ok(Self {
            image: Arc::new(Image { memory, symbols }),
            target,
            kernels: metadata.kernels,
        })
    }

    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::new(format!("{}: {}", path.display(), e)))?;
        Self::load(&bytes)
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn function(&self, name: &str) -> Result<Function, Error> {
        let metadata = self
            .kernels
            .iter()
            .find(|k| k.name == name || k.symbol == name || k.symbol == format!("{}.kd", name))
            .ok_or_else(|| Error::new(format!("the code object has no kernel {}", name)))?;
        Function::new(self.image.clone(), &self.target, metadata.clone())
    }
}

fn metadata(notes: &[u8]) -> Result<Metadata, Error> {
    let word = |at: usize| -> Result<u32, Error> {
        notes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| Error::new("a truncated note"))
    };
    let align = |n: usize| n.div_ceil(4) * 4;
    let mut at = 0;
    while at < notes.len() {
        let name_size = word(at)? as usize;
        let data_size = word(at + 4)? as usize;
        let kind = word(at + 8)?;
        let data = align(at + 12 + name_size);
        at = align(data + data_size);
        if kind == NT_AMDGPU_METADATA {
            let bytes = notes
                .get(data..data + data_size)
                .ok_or_else(|| Error::new("a truncated metadata note"))?;
            return rmp_serde::from_slice(bytes)
                .map_err(|e| Error::new(format!("unreadable kernel metadata: {}", e)));
        }
    }
    Err(Error::new("the code object has no MessagePack metadata"))
}
