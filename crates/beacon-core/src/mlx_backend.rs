//! MLX backend implementation — delegates to `beacon_mlx::ops`.
//!
//! This is the primary compute backend for Apple Silicon. Each method maps
//! directly to the corresponding `beacon-mlx` op, converting `MlxError` to
//! `EngineError` at the boundary.

use std::sync::Arc;

use beacon_mlx::{MlxContext, MlxStream, MlxTensor};

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
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Tensor, EngineError> {
        beacon_mlx::ops::quantized_matmul(stream, x, w, scales, group_size, bits)
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
}
