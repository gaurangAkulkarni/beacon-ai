//! Tensor metadata shared between GGUF and `.beacon` formats.

use crate::dtype::BeaconDtype;

/// Metadata for a single tensor.
///
/// Used both in GGUF parsing (where offsets are relative to the GGUF tensor
/// data section) and in `.beacon` files (where offsets are absolute file
/// positions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    /// Tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Data type / quantisation scheme.
    pub dtype: BeaconDtype,
    /// Dimensions (outermost first).
    pub shape: Vec<u64>,
    /// Byte offset of this tensor's data.
    pub data_offset: u64,
    /// Byte length of this tensor's data.
    pub data_length: u64,
}

impl TensorMeta {
    /// Total number of logical elements.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().copied().product()
    }
}
