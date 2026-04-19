//! Beacon engine orchestration and model loading.
//!
//! This crate ties together `beacon-mlx` (MLX compute backend),
//! `beacon-format` (model file parsing), and `beacon-kernels` (CPU kernels)
//! into a working inference engine.
//!
//! The primary type is [`Engine`], parameterised over a [`ComputeBackend`].
//! For Apple Silicon, use [`MlxBackend`]; CPU and CUDA backends will be
//! added in later steps.

mod backend;
mod cpu_backend;
mod engine;
mod error;
mod kv_cache;
mod mlx_backend;
mod weights;

/// Re-export dequantization from beacon-format (canonical location).
pub mod dequant {
    pub use beacon_format::dequant::*;
}

pub use backend::ComputeBackend;
pub use cpu_backend::{CpuBackend, CpuStream, CpuTensor};
pub use engine::Engine;
pub use error::EngineError;
pub use kv_cache::KvCache;
pub use mlx_backend::MlxBackend;
pub use weights::{
    beacon_dtype_to_mlx, AttentionWeights, FfnWeights, LayerWeights, ModelWeights, ProjectionWeight,
};

// Re-export key types from beacon-format so downstream consumers do not
// need to depend on beacon-format directly.
pub use beacon_format::{Architecture, ModelConfig};

#[cfg(test)]
mod tests;
