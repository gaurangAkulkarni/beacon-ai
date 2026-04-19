//! Beacon dtype enum mirroring `BeaconDtype` from the C ABI.

use crate::ffi;

/// Data type for tensors.
///
/// Standard types (F32, F16, BF16, I32, I8) plus GGUF-style quantization types
/// that the shim preserves as logical metadata even when the underlying MLX
/// storage differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Dtype {
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

impl Dtype {
    pub(crate) fn to_ffi(self) -> ffi::BeaconDtype {
        match self {
            Self::F32 => ffi::BeaconDtype::BEACON_DTYPE_F32,
            Self::F16 => ffi::BeaconDtype::BEACON_DTYPE_F16,
            Self::BF16 => ffi::BeaconDtype::BEACON_DTYPE_BF16,
            Self::I32 => ffi::BeaconDtype::BEACON_DTYPE_I32,
            Self::I8 => ffi::BeaconDtype::BEACON_DTYPE_I8,
            Self::Q4_0 => ffi::BeaconDtype::BEACON_DTYPE_Q4_0,
            Self::Q4K => ffi::BeaconDtype::BEACON_DTYPE_Q4_K,
            Self::Q5K => ffi::BeaconDtype::BEACON_DTYPE_Q5_K,
            Self::Q6K => ffi::BeaconDtype::BEACON_DTYPE_Q6_K,
            Self::Q8_0 => ffi::BeaconDtype::BEACON_DTYPE_Q8_0,
        }
    }

    pub(crate) fn from_ffi(d: ffi::BeaconDtype) -> Self {
        match d {
            ffi::BeaconDtype::BEACON_DTYPE_F32 => Self::F32,
            ffi::BeaconDtype::BEACON_DTYPE_F16 => Self::F16,
            ffi::BeaconDtype::BEACON_DTYPE_BF16 => Self::BF16,
            ffi::BeaconDtype::BEACON_DTYPE_I32 => Self::I32,
            ffi::BeaconDtype::BEACON_DTYPE_I8 => Self::I8,
            ffi::BeaconDtype::BEACON_DTYPE_Q4_0 => Self::Q4_0,
            ffi::BeaconDtype::BEACON_DTYPE_Q4_K => Self::Q4K,
            ffi::BeaconDtype::BEACON_DTYPE_Q5_K => Self::Q5K,
            ffi::BeaconDtype::BEACON_DTYPE_Q6_K => Self::Q6K,
            ffi::BeaconDtype::BEACON_DTYPE_Q8_0 => Self::Q8_0,
        }
    }
}
