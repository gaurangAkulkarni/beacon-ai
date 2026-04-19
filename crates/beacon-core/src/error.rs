//! Engine-level error types.

use beacon_format::FormatError;
use beacon_mlx::MlxError;

/// Errors produced by the inference engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Error from a compute backend.
    #[error("backend error: {0}")]
    Backend(String),

    /// Error from the model format layer.
    #[error(transparent)]
    Format(#[from] FormatError),

    /// A required weight tensor was not found in the model file.
    #[error("weight not found: {0}")]
    WeightNotFound(String),

    /// Tensor shape does not match expectations.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// The KV cache has been exhausted (context overflow).
    #[error("context overflow: KV cache exhausted")]
    ContextOverflow,

    /// Error from the MLX backend.
    #[error(transparent)]
    Mlx(#[from] MlxError),
}
