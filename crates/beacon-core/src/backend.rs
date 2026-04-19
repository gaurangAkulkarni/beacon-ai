//! The `ComputeBackend` trait — abstraction over MLX, CPU, and CUDA backends.
//!
//! The forward pass is written generically against this trait and monomorphised
//! per backend at compile time, so there is zero runtime dispatch cost.
//! See architecture doc section 5.

use crate::error::EngineError;

/// Trait abstracting compute operations for a single backend.
///
/// Each backend (`MlxBackend`, future `CpuBackend`, future `CudaBackend`)
/// implements this trait. The engine is parameterised over `B: ComputeBackend`
/// and the forward pass is monomorphised at compile time.
pub trait ComputeBackend {
    /// Tensor type for this backend.
    type Tensor;
    /// Execution stream type for this backend.
    type Stream;

    /// Create a new execution stream for scheduling ops.
    fn new_stream(&self) -> Result<Self::Stream, EngineError>;

    /// Matrix multiplication: `out = a @ b`.
    fn matmul(
        &self,
        stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError>;

    /// Quantized matrix multiplication: `out = x @ dequant(w, scales, biases)`.
    fn quantized_matmul(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        w: &Self::Tensor,
        scales: &Self::Tensor,
        biases: Option<&Self::Tensor>,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Tensor, EngineError>;

    /// RMS normalisation: `out = rms_norm(x, weight, eps)`.
    fn rms_norm(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, EngineError>;

    /// Rotary position embedding.
    fn rope(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        position_offset: i32,
        theta: f32,
        dim: i32,
    ) -> Result<Self::Tensor, EngineError>;

    /// Fused scaled dot-product attention with GQA support.
    ///
    /// - `q`: `[batch, n_heads, seq_len, head_dim]`
    /// - `k`: `[batch, n_kv_heads, kv_seq_len, head_dim]`
    /// - `v`: `[batch, n_kv_heads, kv_seq_len, head_dim]`
    /// - `mask`: optional attention mask.
    fn attention(
        &self,
        stream: &Self::Stream,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: Option<&Self::Tensor>,
        scale: f32,
    ) -> Result<Self::Tensor, EngineError>;

    /// `SiLU` activation: `out = x * sigmoid(x)`.
    fn silu(&self, stream: &Self::Stream, x: &Self::Tensor) -> Result<Self::Tensor, EngineError>;

    /// Softmax along `axis`.
    fn softmax(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        axis: i32,
    ) -> Result<Self::Tensor, EngineError>;

    /// Element-wise addition: `out = a + b`.
    fn add(
        &self,
        stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError>;

    /// Element-wise multiplication: `out = a * b`.
    fn mul(
        &self,
        stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError>;

    /// Reshape a tensor (total element count must be unchanged).
    fn reshape(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        shape: &[i64],
    ) -> Result<Self::Tensor, EngineError>;

    /// Swap two axes of a tensor.
    fn swapaxes(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        axis1: i32,
        axis2: i32,
    ) -> Result<Self::Tensor, EngineError>;

    /// Transpose the last two dimensions of a tensor.
    fn transpose(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError>;

    /// Embedding lookup: select rows from `weight` by `indices`.
    fn embedding(
        &self,
        stream: &Self::Stream,
        weight: &Self::Tensor,
        indices: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError>;

    /// Force evaluation of the lazy computation graph for this tensor.
    fn eval(&self, t: &Self::Tensor, stream: &Self::Stream) -> Result<(), EngineError>;

    /// Read tensor values into a host `Vec<f32>`.
    fn read_f32(&self, t: &Self::Tensor, n: usize) -> Result<Vec<f32>, EngineError>;

    /// Update KV cache at `position` with new K/V values.
    ///
    /// Writes `new_k` and `new_v` into the cache tensors at the given position,
    /// then returns views (or copies) covering `[0..=position]` for attention.
    fn kv_cache_update(
        &self,
        stream: &Self::Stream,
        cache_k: &Self::Tensor,
        cache_v: &Self::Tensor,
        new_k: &Self::Tensor,
        new_v: &Self::Tensor,
        position: i64,
    ) -> Result<(Self::Tensor, Self::Tensor), EngineError>;

    /// Create a 1-D tensor of token IDs for embedding lookup.
    ///
    /// The exact storage format is backend-specific (I32 for MLX, f32 indices
    /// for CPU).
    fn create_token_tensor(&self, tokens: &[u32]) -> Result<Self::Tensor, EngineError>;
}
