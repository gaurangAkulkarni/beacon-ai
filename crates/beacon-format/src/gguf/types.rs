//! GGUF type definitions.

use crate::dtype::BeaconDtype;
use crate::error::FormatError;

/// GGUF tensor data type IDs (from the GGUF spec / llama.cpp `ggml_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgufTensorType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Convert to the equivalent `BeaconDtype`, if supported.
    pub fn to_beacon_dtype(self) -> Result<BeaconDtype, FormatError> {
        match self {
            Self::F32 => Ok(BeaconDtype::F32),
            Self::F16 => Ok(BeaconDtype::F16),
            Self::BF16 => Ok(BeaconDtype::BF16),
            Self::I8 => Ok(BeaconDtype::I8),
            Self::I32 => Ok(BeaconDtype::I32),
            Self::Q4_0 => Ok(BeaconDtype::Q4_0),
            Self::Q4_1 => Ok(BeaconDtype::Q4_1),
            Self::Q5_0 => Ok(BeaconDtype::Q5_0),
            Self::Q5_1 => Ok(BeaconDtype::Q5_1),
            Self::Q8_0 => Ok(BeaconDtype::Q8_0),
            Self::Q2K => Ok(BeaconDtype::Q2K),
            Self::Q3K => Ok(BeaconDtype::Q3K),
            Self::Q4K => Ok(BeaconDtype::Q4K),
            Self::Q5K => Ok(BeaconDtype::Q5K),
            Self::Q6K => Ok(BeaconDtype::Q6K),
            Self::Q8K => Ok(BeaconDtype::Q8K),
            other => Err(FormatError::UnsupportedGgufType(other as u32)),
        }
    }

    /// Bytes per block for this GGUF type (used to compute tensor data sizes).
    fn bytes_per_block(self) -> u64 {
        match self {
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::F32 | Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Elements per block.
    fn block_size(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Compute the byte length for `num_elements` elements of this type.
    pub fn data_byte_length(self, num_elements: u64) -> u64 {
        let bs = self.block_size();
        let num_blocks = num_elements.div_ceil(bs);
        num_blocks * self.bytes_per_block()
    }
}

/// GGUF metadata value type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetaType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetaType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A parsed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Try to extract as a `u32`.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(u32::from(*v)),
            Self::Uint16(v) => Some(u32::from(*v)),
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => u32::try_from(*v).ok(),
            Self::Int32(v) => u32::try_from(*v).ok(),
            Self::Int64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract as a `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(u64::from(*v)),
            Self::Uint16(v) => Some(u64::from(*v)),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Uint64(v) => Some(*v),
            Self::Int32(v) => u64::try_from(*v).ok(),
            Self::Int64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract as a `usize`.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|v| usize::try_from(v).ok())
    }

    /// Try to extract as `f32`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to extract as a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as an array.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}
