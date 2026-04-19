//! MLX backend implementation — delegates to `beacon_mlx::ops`.
//!
//! This is the primary compute backend for Apple Silicon. Each method maps
//! directly to the corresponding `beacon-mlx` op, converting `MlxError` to
//! `EngineError` at the boundary.

use std::sync::Arc;

use beacon_mlx::{Dtype, MlxContext, MlxStream, MlxTensor};

use crate::backend::ComputeBackend;
use crate::error::EngineError;

/// MLX compute backend for Apple Silicon.
///
/// Wraps an `MlxContext` and delegates all ops to `beacon_mlx::ops`.
#[derive(Debug)]
pub struct MlxBackend {
    ctx: Arc<MlxContext>,
}

impl MlxBackend {
    /// Create a new MLX backend from an existing context.
    pub fn new(ctx: Arc<MlxContext>) -> Self {
        Self { ctx }
    }

    /// Get a reference to the underlying `MlxContext`.
    pub fn context(&self) -> &Arc<MlxContext> {
        &self.ctx
    }
}

impl ComputeBackend for MlxBackend {
    type Tensor = MlxTensor;
    type Stream = MlxStream;

    fn new_stream(&self) -> Result<Self::Stream, EngineError> {
        self.ctx.new_stream().map_err(EngineError::from)
    }

    fn matmul(
        &self,
        stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::matmul(stream, a, b).map_err(EngineError::from)
    }

    fn quantized_matmul(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        w: &Self::Tensor,
        scales: &Self::Tensor,
        biases: Option<&Self::Tensor>,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::quantized_matmul(stream, x, w, scales, biases, group_size, bits)
            .map_err(EngineError::from)
    }

    fn rms_norm(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::rms_norm(stream, x, weight, eps).map_err(EngineError::from)
    }

    fn rope(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        position_offset: i32,
        theta: f32,
        dim: i32,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::rope(stream, x, position_offset, theta, dim).map_err(EngineError::from)
    }

    fn attention(
        &self,
        stream: &Self::Stream,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: Option<&Self::Tensor>,
        scale: f32,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::attention(stream, q, k, v, mask, scale).map_err(EngineError::from)
    }

    fn silu(&self, stream: &Self::Stream, x: &Self::Tensor) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::silu(stream, x).map_err(EngineError::from)
    }

    fn softmax(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        axis: i32,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::softmax(stream, x, axis).map_err(EngineError::from)
    }

    fn add(
        &self,
        stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::add(stream, a, b).map_err(EngineError::from)
    }

    fn mul(
        &self,
        stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::elementwise_mul(stream, a, b).map_err(EngineError::from)
    }

    fn reshape(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        shape: &[i64],
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::reshape(stream, x, shape).map_err(EngineError::from)
    }

    fn swapaxes(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
        axis1: i32,
        axis2: i32,
    ) -> Result<Self::Tensor, EngineError> {
        Ok(beacon_mlx::ops::swapaxes(stream, x, axis1, axis2)?)
    }

    fn transpose(
        &self,
        stream: &Self::Stream,
        x: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        Ok(beacon_mlx::ops::transpose(stream, x)?)
    }

    fn embedding(
        &self,
        stream: &Self::Stream,
        weight: &Self::Tensor,
        indices: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::embedding(stream, weight, indices).map_err(EngineError::from)
    }

    fn eval(&self, t: &Self::Tensor, stream: &Self::Stream) -> Result<(), EngineError> {
        t.eval(stream).map_err(EngineError::from)
    }

    fn read_f32(&self, t: &Self::Tensor, n: usize) -> Result<Vec<f32>, EngineError> {
        t.read_f32(n).map_err(EngineError::from)
    }

    fn kv_cache_update(
        &self,
        stream: &Self::Stream,
        cache_k: &Self::Tensor,
        cache_v: &Self::Tensor,
        new_k: &Self::Tensor,
        new_v: &Self::Tensor,
        position: i64,
    ) -> Result<(Self::Tensor, Self::Tensor), EngineError> {
        beacon_mlx::ops::kv_cache_update(stream, cache_k, cache_v, new_k, new_v, position)
            .map_err(EngineError::from)
    }

    /// Create an `MlxTensor` containing token IDs as I32.
    ///
    /// Uses an anonymous mmap so the data stays in memory (no disk I/O).
    /// This satisfies the non-negotiable rule about no blocking in the decode
    /// hot path — anonymous mmap is a pure memory allocation.
    #[allow(clippy::cast_possible_wrap)]
    fn create_token_tensor(&self, tokens: &[u32]) -> Result<Self::Tensor, EngineError> {
        let n = tokens.len();
        let byte_len = n * 4;

        let mut mmap_mut = memmap2::MmapMut::map_anon(byte_len)
            .map_err(|e| EngineError::Backend(format!("anonymous mmap failed: {e}")))?;

        for (i, &tok) in tokens.iter().enumerate() {
            let val = tok as i32;
            let offset = i * 4;
            mmap_mut[offset..offset + 4].copy_from_slice(&val.to_ne_bytes());
        }

        let mmap = mmap_mut
            .make_read_only()
            .map_err(|e| EngineError::Backend(format!("mmap make_read_only failed: {e}")))?;
        let mmap = Arc::new(mmap);

        let shape = [n as i64];
        MlxTensor::from_mmap(Arc::clone(&self.ctx), mmap, 0, &shape, Dtype::I32)
            .map_err(EngineError::from)
    }
}
