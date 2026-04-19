//! `.beacon` file writer.
//!
//! Writes a self-contained `.beacon` file from a model config, tokenizer JSON,
//! tensor metadata, and tensor data. Tensor data is 64-byte aligned.

use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::config::ModelConfig;
use crate::error::FormatError;
use crate::tensor_meta::TensorMeta;

/// `.beacon` format magic bytes.
pub const BEACON_MAGIC: [u8; 4] = *b"BCN1";

/// Current `.beacon` format version.
pub const BEACON_VERSION: u32 = 1;

/// Alignment for tensor data (bytes).
const DATA_ALIGNMENT: u64 = 64;

/// Write a `.beacon` file.
///
/// `tensor_data` must be parallel to `tensors` — `tensor_data[i]` is the raw
/// byte content for `tensors[i]`.
#[allow(clippy::cast_possible_truncation)]
pub fn write_beacon(
    path: &Path,
    config: &ModelConfig,
    tokenizer_json: &str,
    tensors: &[TensorMeta],
    tensor_data: &[&[u8]],
) -> Result<(), FormatError> {
    assert_eq!(tensors.len(), tensor_data.len());

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // --- Fixed header (24 bytes) ---
    // magic (4) + version (4) + header_size placeholder (8) + tensor_count (8)
    w.write_all(&BEACON_MAGIC)?;
    w.write_all(&BEACON_VERSION.to_le_bytes())?;
    let header_size_pos = stream_pos(&mut w)?;
    w.write_all(&0u64.to_le_bytes())?; // placeholder — filled in later
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;

    // --- Config JSON (length-prefixed) ---
    let config_json = serde_json::to_string(config)?;
    write_length_prefixed(&mut w, config_json.as_bytes())?;

    // --- Tokenizer JSON (length-prefixed) ---
    write_length_prefixed(&mut w, tokenizer_json.as_bytes())?;

    // --- Tensor metadata table ---
    // Pre-compute where the tensor data section starts and each tensor's offset.
    let meta_table_start = stream_pos(&mut w)?;
    let meta_table_size: u64 = tensors
        .iter()
        .map(|t| 8 + t.name.len() as u64 + 4 + 4 + 8 * t.shape.len() as u64 + 8 + 8)
        .sum();
    let header_end = meta_table_start + meta_table_size;
    let data_section_start = align_up(header_end, DATA_ALIGNMENT);

    // Compute absolute data offsets for each tensor.
    let mut data_offsets = Vec::with_capacity(tensors.len());
    let mut cursor = data_section_start;
    for t in tensors {
        cursor = align_up(cursor, DATA_ALIGNMENT);
        data_offsets.push(cursor);
        cursor += t.data_length;
    }

    // Write tensor metadata entries.
    for (i, t) in tensors.iter().enumerate() {
        write_length_prefixed(&mut w, t.name.as_bytes())?;
        w.write_all(&(t.dtype as u32).to_le_bytes())?;
        w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
        for &dim in &t.shape {
            w.write_all(&dim.to_le_bytes())?;
        }
        w.write_all(&data_offsets[i].to_le_bytes())?;
        w.write_all(&t.data_length.to_le_bytes())?;
    }

    // --- Fill in header_size ---
    let current_pos = stream_pos(&mut w)?;
    debug_assert_eq!(current_pos, header_end);
    w.seek(SeekFrom::Start(header_size_pos))?;
    w.write_all(&header_end.to_le_bytes())?;
    w.seek(SeekFrom::Start(current_pos))?;

    // --- Padding to data section ---
    let pad = (data_section_start - current_pos) as usize;
    if pad > 0 {
        w.write_all(&vec![0u8; pad])?;
    }

    // --- Tensor data ---
    for (i, data) in tensor_data.iter().enumerate() {
        // Pad to alignment.
        let current = stream_pos(&mut w)?;
        let target = data_offsets[i];
        if current < target {
            w.write_all(&vec![0u8; (target - current) as usize])?;
        }
        w.write_all(data)?;
    }

    w.flush()?;
    Ok(())
}

fn write_length_prefixed(w: &mut impl Write, data: &[u8]) -> Result<(), FormatError> {
    w.write_all(&(data.len() as u64).to_le_bytes())?;
    w.write_all(data)?;
    Ok(())
}

fn stream_pos(w: &mut (impl Seek + ?Sized)) -> Result<u64, FormatError> {
    Ok(w.stream_position()?)
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}
