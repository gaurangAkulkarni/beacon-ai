//! GGUF tensor info entry parser.
//!
//! Each tensor info entry follows the metadata section and describes a tensor's
//! name, shape, type, and data offset (relative to the tensor data section).

use crate::dtype::BeaconDtype;
use crate::error::FormatError;
use crate::gguf::types::GgufTensorType;
use crate::tensor_meta::TensorMeta;

/// Parse `tensor_count` tensor info entries starting at `offset` in `data`.
///
/// Returns the parsed entries and the byte position after the last entry.
/// The `data_offset` in each returned `TensorMeta` is relative to the start of
/// the tensor data section (not the file start). The caller must add the
/// absolute tensor data section offset.
#[allow(clippy::cast_possible_truncation)]
pub fn parse_tensor_infos(
    data: &[u8],
    offset: usize,
    tensor_count: u64,
) -> Result<(Vec<TensorMeta>, usize), FormatError> {
    let mut pos = offset;
    let mut tensors = Vec::with_capacity(tensor_count as usize);

    for _ in 0..tensor_count {
        // Name: u64 length + UTF-8 bytes.
        let name_len = read_u64(data, pos)? as usize;
        pos += 8;
        let name_bytes = read_slice(data, pos, name_len)?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|e| {
            FormatError::InvalidMetadata(format!("invalid UTF-8 in tensor name: {e}"))
        })?;
        pos += name_len;

        // ndim: u32
        let ndim = read_u32(data, pos)? as usize;
        pos += 4;

        // dims: u64 * ndim
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(data, pos)?);
            pos += 8;
        }

        // GGUF type: u32
        let gguf_type_id = read_u32(data, pos)?;
        pos += 4;
        let gguf_type = GgufTensorType::from_u32(gguf_type_id)
            .ok_or(FormatError::UnsupportedGgufType(gguf_type_id))?;
        let dtype = gguf_type.to_beacon_dtype()?;

        // Offset (relative to tensor data section start): u64
        let rel_offset = read_u64(data, pos)?;
        pos += 8;

        let num_elements: u64 = shape.iter().copied().product();
        let data_length = gguf_type.data_byte_length(num_elements);

        tensors.push(TensorMeta {
            name,
            dtype,
            shape,
            data_offset: rel_offset,
            data_length,
        });
    }

    Ok((tensors, pos))
}

/// Compute the absolute offset of the tensor data section.
///
/// GGUF v3 aligns the tensor data section to `alignment` bytes (default 32)
/// after the metadata + tensor info entries.
pub fn compute_data_section_offset(end_of_tensor_infos: u64, alignment: u64) -> u64 {
    end_of_tensor_infos.div_ceil(alignment) * alignment
}

// --- Helpers -----------------------------------------------------------------

fn read_u32(data: &[u8], pos: usize) -> Result<u32, FormatError> {
    let slice = read_slice(data, pos, 4)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(data: &[u8], pos: usize) -> Result<u64, FormatError> {
    let slice = read_slice(data, pos, 8)?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_slice(data: &[u8], pos: usize, len: usize) -> Result<&[u8], FormatError> {
    if pos + len > data.len() {
        return Err(FormatError::Truncated {
            needed: len as u64,
            offset: pos as u64,
            file_len: data.len() as u64,
        });
    }
    Ok(&data[pos..pos + len])
}

/// Resolve the alignment value from GGUF metadata.
///
/// The `general.alignment` key overrides the default (32). If absent, returns
/// the GGUF v3 default alignment.
pub fn alignment_from_metadata(
    metadata: &std::collections::BTreeMap<String, crate::gguf::types::GgufValue>,
) -> u64 {
    metadata
        .get("general.alignment")
        .and_then(super::types::GgufValue::as_u64)
        .unwrap_or(32)
}

/// Convert relative tensor offsets to absolute file offsets.
pub fn resolve_tensor_offsets(tensors: &mut [TensorMeta], data_section_start: u64) {
    for t in tensors {
        t.data_offset += data_section_start;
    }
}

/// Check that a `BeaconDtype` round-trips correctly from a GGUF type.
#[allow(dead_code)]
pub(crate) fn beacon_dtype_for_gguf_type(gguf_type_id: u32) -> Result<BeaconDtype, FormatError> {
    let gguf_type = GgufTensorType::from_u32(gguf_type_id)
        .ok_or(FormatError::UnsupportedGgufType(gguf_type_id))?;
    gguf_type.to_beacon_dtype()
}
