//! Beacon dtype — the on-disk type tag used in `.beacon` files.
//!
//! Discriminant values match the C ABI's `BeaconDtype` so the Rust format
//! layer and the MLX shim agree without conversion.

/// Data type stored in a `.beacon` tensor metadata entry.
///
/// Standard types plus GGUF-style quantisation types. The discriminant values
/// are the canonical on-disk representation (little-endian `u32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u32)]
pub enum BeaconDtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    I8 = 4,
    Q4_0 = 100,
    Q4K = 101,
    Q5K = 102,
    Q6K = 103,
    Q8_0 = 104,
}

impl BeaconDtype {
    /// Decode from an on-disk `u32`.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::BF16),
            3 => Some(Self::I32),
            4 => Some(Self::I8),
            100 => Some(Self::Q4_0),
            101 => Some(Self::Q4K),
            102 => Some(Self::Q5K),
            103 => Some(Self::Q6K),
            104 => Some(Self::Q8_0),
            _ => None,
        }
    }

    /// Number of elements per quantisation block (1 for non-quantised types).
    pub fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I32 | Self::I8 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Bytes per block.
    pub fn bytes_per_block(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::Q4_0 => 18, // 2-byte scale + 16 bytes of 4-bit weights (32 elements)
            Self::Q8_0 => 34, // 2-byte scale + 32 bytes of 8-bit weights
            Self::Q4K => 144, // 256-element block
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Compute the byte length of tensor data given the total number of elements.
    pub fn data_byte_length(self, num_elements: u64) -> u64 {
        let bs = self.block_size();
        // Number of blocks, rounding up.
        let num_blocks = num_elements.div_ceil(bs);
        num_blocks * self.bytes_per_block()
    }
}
