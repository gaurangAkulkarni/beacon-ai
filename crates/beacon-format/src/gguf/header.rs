//! GGUF v3 fixed header parsing.

use crate::error::FormatError;

/// The GGUF magic bytes: ASCII `"GGUF"`.
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Parsed GGUF fixed header (first 24 bytes).
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// Parse the fixed 24-byte header from a byte slice (the start of the file).
pub fn parse_header(data: &[u8]) -> Result<GgufHeader, FormatError> {
    if data.len() < 24 {
        return Err(FormatError::Truncated {
            needed: 24,
            offset: 0,
            file_len: data.len() as u64,
        });
    }

    let magic: [u8; 4] = data[0..4].try_into().unwrap();
    if magic != GGUF_MAGIC {
        return Err(FormatError::InvalidMagic {
            expected: &GGUF_MAGIC,
            got: magic,
        });
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 3 {
        return Err(FormatError::UnsupportedVersion(version));
    }

    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}
