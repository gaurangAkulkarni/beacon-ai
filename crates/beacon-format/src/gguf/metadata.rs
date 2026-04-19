//! GGUF metadata key-value parser.
//!
//! Parses the variable-length metadata section that follows the fixed header.
//! All values are little-endian.

use std::collections::BTreeMap;

use crate::error::FormatError;
use crate::gguf::types::{GgufMetaType, GgufValue};

/// A cursor over a byte slice for sequential little-endian reads.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        if self.remaining() < n {
            return Err(FormatError::Truncated {
                needed: n as u64,
                offset: self.pos as u64,
                file_len: self.data.len() as u64,
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.read_bytes(1)?[0])
    }

    #[allow(clippy::cast_possible_wrap)]
    fn read_i8(&mut self) -> Result<i8, FormatError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, FormatError> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, FormatError> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, FormatError> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, FormatError> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, FormatError> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, FormatError> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_string(&mut self) -> Result<String, FormatError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| FormatError::InvalidMetadata(format!("invalid UTF-8 in GGUF string: {e}")))
    }

    fn read_bool(&mut self) -> Result<bool, FormatError> {
        Ok(self.read_u8()? != 0)
    }
}

/// Read a single typed value.
#[allow(clippy::cast_possible_truncation)]
fn read_value(r: &mut Reader<'_>, vtype: GgufMetaType) -> Result<GgufValue, FormatError> {
    match vtype {
        GgufMetaType::Uint8 => Ok(GgufValue::Uint8(r.read_u8()?)),
        GgufMetaType::Int8 => Ok(GgufValue::Int8(r.read_i8()?)),
        GgufMetaType::Uint16 => Ok(GgufValue::Uint16(r.read_u16()?)),
        GgufMetaType::Int16 => Ok(GgufValue::Int16(r.read_i16()?)),
        GgufMetaType::Uint32 => Ok(GgufValue::Uint32(r.read_u32()?)),
        GgufMetaType::Int32 => Ok(GgufValue::Int32(r.read_i32()?)),
        GgufMetaType::Float32 => Ok(GgufValue::Float32(r.read_f32()?)),
        GgufMetaType::Bool => Ok(GgufValue::Bool(r.read_bool()?)),
        GgufMetaType::String => Ok(GgufValue::String(r.read_string()?)),
        GgufMetaType::Uint64 => Ok(GgufValue::Uint64(r.read_u64()?)),
        GgufMetaType::Int64 => Ok(GgufValue::Int64(r.read_i64()?)),
        GgufMetaType::Float64 => Ok(GgufValue::Float64(r.read_f64()?)),
        GgufMetaType::Array => {
            let elem_type_id = r.read_u32()?;
            let elem_type = GgufMetaType::from_u32(elem_type_id).ok_or_else(|| {
                FormatError::InvalidMetadata(format!(
                    "unknown GGUF array element type: {elem_type_id}"
                ))
            })?;
            let count = r.read_u64()? as usize;
            let mut items = Vec::with_capacity(count.min(1024 * 1024)); // cap pre-alloc
            for _ in 0..count {
                items.push(read_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
    }
}

/// Parse `kv_count` metadata key-value pairs starting at `offset` in `data`.
///
/// Returns the parsed map and the byte position after the last KV pair.
pub fn parse_metadata(
    data: &[u8],
    offset: usize,
    kv_count: u64,
) -> Result<(BTreeMap<String, GgufValue>, usize), FormatError> {
    let mut r = Reader::new(data, offset);
    let mut map = BTreeMap::new();

    for _ in 0..kv_count {
        let key = r.read_string()?;
        let vtype_id = r.read_u32()?;
        let vtype = GgufMetaType::from_u32(vtype_id).ok_or_else(|| {
            FormatError::InvalidMetadata(format!(
                "unknown GGUF metadata value type {vtype_id} for key {key:?}"
            ))
        })?;
        let value = read_value(&mut r, vtype)?;
        map.insert(key, value);
    }

    Ok((map, r.pos))
}
