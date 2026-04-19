//! `.beacon` file reader.
//!
//! Memory-maps the file, parses the header eagerly, and provides zero-copy
//! access to tensor data via the mmap.

use std::path::Path;
use std::sync::Arc;

use crate::beacon::writer::{BEACON_MAGIC, BEACON_VERSION};
use crate::config::ModelConfig;
use crate::dtype::BeaconDtype;
use crate::error::FormatError;
use crate::tensor_meta::TensorMeta;

/// A parsed `.beacon` model file.
///
/// The file is memory-mapped; tensor data is served zero-copy from the mmap.
#[derive(Debug)]
pub struct BeaconFile {
    /// Parsed model configuration.
    pub config: ModelConfig,
    /// Serialised tokenizer data (JSON).
    pub tokenizer_json: String,
    /// Tensor metadata (offsets are absolute file positions in the mmap).
    pub tensors: Vec<TensorMeta>,
    mmap: Arc<memmap2::Mmap>,
}

impl BeaconFile {
    /// Open and parse a `.beacon` file.
    pub fn open(path: &Path) -> Result<Self, FormatError> {
        let file = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);
        Self::from_mmap(mmap)
    }

    /// Parse from an existing mmap.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_mmap(mmap: Arc<memmap2::Mmap>) -> Result<Self, FormatError> {
        let data: &[u8] = &mmap;
        let mut pos: usize = 0;

        // --- Fixed header (24 bytes) ---
        check_len(data, pos, 24)?;
        let magic: [u8; 4] = data[0..4].try_into().unwrap();
        if magic != BEACON_MAGIC {
            return Err(FormatError::InvalidMagic {
                expected: &BEACON_MAGIC,
                got: magic,
            });
        }
        pos += 4;

        let version = read_u32(data, pos);
        pos += 4;
        if version != BEACON_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }

        let _header_size = read_u64(data, pos);
        pos += 8;

        let tensor_count = read_u64(data, pos) as usize;
        pos += 8;

        // --- Config JSON ---
        let (config_bytes, new_pos) = read_length_prefixed(data, pos)?;
        pos = new_pos;
        let config: ModelConfig = serde_json::from_slice(config_bytes)?;

        // --- Tokenizer JSON ---
        let (tok_bytes, new_pos) = read_length_prefixed(data, pos)?;
        pos = new_pos;
        let tokenizer_json = String::from_utf8(tok_bytes.to_vec()).map_err(|e| {
            FormatError::InvalidMetadata(format!("invalid UTF-8 in tokenizer JSON: {e}"))
        })?;

        // --- Tensor metadata table ---
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let (name_bytes, new_pos) = read_length_prefixed(data, pos)?;
            pos = new_pos;
            let name = String::from_utf8(name_bytes.to_vec()).map_err(|e| {
                FormatError::InvalidMetadata(format!("invalid UTF-8 in tensor name: {e}"))
            })?;

            check_len(data, pos, 8)?; // dtype(4) + ndim(4)
            let dtype_u32 = read_u32(data, pos);
            pos += 4;
            let dtype = BeaconDtype::from_u32(dtype_u32)
                .ok_or(FormatError::UnsupportedGgufType(dtype_u32))?;

            let ndim = read_u32(data, pos) as usize;
            pos += 4;

            check_len(data, pos, ndim * 8 + 16)?; // dims + offset + length
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(read_u64(data, pos));
                pos += 8;
            }

            let data_offset = read_u64(data, pos);
            pos += 8;
            let data_length = read_u64(data, pos);
            pos += 8;

            tensors.push(TensorMeta {
                name,
                dtype,
                shape,
                data_offset,
                data_length,
            });
        }

        Ok(Self {
            config,
            tokenizer_json,
            tensors,
            mmap,
        })
    }

    /// Get the raw byte slice for a tensor's data (zero-copy from the mmap).
    #[allow(clippy::cast_possible_truncation)]
    pub fn tensor_data(&self, tensor: &TensorMeta) -> &[u8] {
        let start = tensor.data_offset as usize;
        let end = start + tensor.data_length as usize;
        &self.mmap[start..end]
    }

    /// Get the underlying mmap (for passing to `beacon-mlx` tensor creation).
    pub fn mmap(&self) -> &Arc<memmap2::Mmap> {
        &self.mmap
    }
}

// --- Helpers -----------------------------------------------------------------

fn read_u32(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap())
}

fn read_u64(data: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap())
}

#[allow(clippy::cast_possible_truncation)]
fn read_length_prefixed(data: &[u8], pos: usize) -> Result<(&[u8], usize), FormatError> {
    check_len(data, pos, 8)?;
    let len = read_u64(data, pos) as usize;
    let start = pos + 8;
    check_len(data, start, len)?;
    Ok((&data[start..start + len], start + len))
}

fn check_len(data: &[u8], pos: usize, needed: usize) -> Result<(), FormatError> {
    if pos + needed > data.len() {
        return Err(FormatError::Truncated {
            needed: needed as u64,
            offset: pos as u64,
            file_len: data.len() as u64,
        });
    }
    Ok(())
}
