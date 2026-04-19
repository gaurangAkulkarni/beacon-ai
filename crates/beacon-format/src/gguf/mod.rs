//! GGUF v3 file parser.
//!
//! Opens a GGUF file via mmap, parses the header, metadata key-value pairs,
//! and tensor info entries. Provides access to tensor data as byte slices
//! directly from the mmap (zero-copy).

mod config_map;
mod header;
mod metadata;
mod tensor_info;
pub mod types;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::config::ModelConfig;
use crate::error::FormatError;
use crate::tensor_meta::TensorMeta;

pub use types::GgufValue;

/// A parsed GGUF v3 file.
///
/// The file is memory-mapped; tensor data is accessed zero-copy from the mmap.
#[derive(Debug)]
pub struct GgufFile {
    /// GGUF format version (always 3 for supported files).
    pub version: u32,
    /// All metadata key-value pairs.
    pub metadata: BTreeMap<String, GgufValue>,
    /// Tensor metadata (offsets are absolute file positions).
    pub tensors: Vec<TensorMeta>,
    mmap: Arc<memmap2::Mmap>,
}

impl GgufFile {
    /// Open and parse a GGUF v3 file.
    pub fn open(path: &Path) -> Result<Self, FormatError> {
        let file = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);
        let data: &[u8] = &mmap;

        // 1. Fixed header.
        let hdr = header::parse_header(data)?;

        // 2. Metadata KV pairs (start right after the 24-byte header).
        let (metadata, after_meta) = metadata::parse_metadata(data, 24, hdr.metadata_kv_count)?;

        // 3. Tensor info entries.
        let (mut tensors, after_tensors) =
            tensor_info::parse_tensor_infos(data, after_meta, hdr.tensor_count)?;

        // 4. Compute absolute tensor data section offset.
        let alignment = tensor_info::alignment_from_metadata(&metadata);
        let data_section_start =
            tensor_info::compute_data_section_offset(after_tensors as u64, alignment);

        // 5. Resolve relative offsets → absolute file offsets.
        tensor_info::resolve_tensor_offsets(&mut tensors, data_section_start);

        Ok(Self {
            version: hdr.version,
            metadata,
            tensors,
            mmap,
        })
    }

    /// Extract a [`ModelConfig`] from the GGUF metadata.
    pub fn model_config(&self) -> Result<ModelConfig, FormatError> {
        config_map::model_config_from_metadata(&self.metadata)
    }

    /// Get the raw byte slice for a tensor's data.
    #[allow(clippy::cast_possible_truncation)]
    pub fn tensor_data(&self, tensor: &TensorMeta) -> &[u8] {
        let start = tensor.data_offset as usize;
        let end = start + tensor.data_length as usize;
        &self.mmap[start..end]
    }

    /// Get the underlying mmap.
    pub fn mmap(&self) -> &Arc<memmap2::Mmap> {
        &self.mmap
    }

    /// Serialize the tokenizer-related GGUF metadata into a JSON string.
    ///
    /// This captures `tokenizer.ggml.*` keys so the `.beacon` file can store
    /// tokenizer data without re-parsing the GGUF later (used by Step 5).
    pub fn tokenizer_json(&self) -> Result<String, FormatError> {
        let mut tok_meta = serde_json::Map::new();
        for (key, value) in &self.metadata {
            if key.starts_with("tokenizer.") {
                tok_meta.insert(key.clone(), gguf_value_to_json(value));
            }
        }
        Ok(serde_json::to_string(&tok_meta)?)
    }
}

/// Convert a `GgufValue` to a `serde_json::Value` for tokenizer JSON
/// serialization.
fn gguf_value_to_json(v: &GgufValue) -> serde_json::Value {
    match v {
        GgufValue::Uint8(n) => serde_json::Value::from(*n),
        GgufValue::Int8(n) => serde_json::Value::from(*n),
        GgufValue::Uint16(n) => serde_json::Value::from(*n),
        GgufValue::Int16(n) => serde_json::Value::from(*n),
        GgufValue::Uint32(n) => serde_json::Value::from(*n),
        GgufValue::Int32(n) => serde_json::Value::from(*n),
        GgufValue::Float32(n) => serde_json::Value::from(*n),
        GgufValue::Bool(b) => serde_json::Value::from(*b),
        GgufValue::String(s) => serde_json::Value::from(s.as_str()),
        GgufValue::Uint64(n) => serde_json::Value::from(*n),
        GgufValue::Int64(n) => serde_json::Value::from(*n),
        GgufValue::Float64(n) => serde_json::Value::from(*n),
        GgufValue::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(gguf_value_to_json).collect())
        }
    }
}
