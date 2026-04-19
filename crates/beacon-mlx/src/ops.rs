//! MLX operations exposed as free functions.
//!
//! Each function maps 1:1 to a `beacon_op_*` C ABI call. All ops take an
//! [`MlxStream`] to control execution ordering and return a new [`MlxTensor`].

use std::sync::Arc;

use crate::error::{status_to_result, MlxError};
use crate::ffi;
use crate::stream::MlxStream;
use crate::tensor::MlxTensor;

/// Matrix multiplication: `out = a @ b`.
pub fn matmul(stream: &MlxStream, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&a.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status =
        unsafe { ffi::beacon_op_matmul(ctx.inner, stream.inner, a.inner, b.inner, &raw mut out) };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Quantized matrix multiplication: `out = x @ dequant(w, scales)`.
pub fn quantized_matmul(
    stream: &MlxStream,
    x: &MlxTensor,
    w: &MlxTensor,
    scales: &MlxTensor,
    group_size: i32,
    bits: i32,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_quantized_matmul(
            ctx.inner,
            stream.inner,
            x.inner,
            w.inner,
            scales.inner,
            group_size,
            bits,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// RMS normalisation: `out = rms_norm(x, weight, eps)`.
pub fn rms_norm(
    stream: &MlxStream,
    x: &MlxTensor,
    weight: &MlxTensor,
    eps: f32,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_rms_norm(
            ctx.inner,
            stream.inner,
            x.inner,
            weight.inner,
            eps,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Rotary position embedding.
pub fn rope(
    stream: &MlxStream,
    x: &MlxTensor,
    position_offset: i32,
    theta: f32,
    dim: i32,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_rope(
            ctx.inner,
            stream.inner,
            x.inner,
            position_offset,
            theta,
            dim,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// `SiLU` activation: `out = x * sigmoid(x)`.
pub fn silu(stream: &MlxStream, x: &MlxTensor) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe { ffi::beacon_op_silu(ctx.inner, stream.inner, x.inner, &raw mut out) };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Softmax along `axis`.
pub fn softmax(stream: &MlxStream, x: &MlxTensor, axis: i32) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status =
        unsafe { ffi::beacon_op_softmax(ctx.inner, stream.inner, x.inner, axis, &raw mut out) };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Fused scaled dot-product attention with GQA support.
///
/// - `q`: `[batch, n_heads, seq_len, head_dim]`
/// - `k`: `[batch, n_kv_heads, kv_seq_len, head_dim]`
/// - `v`: `[batch, n_kv_heads, kv_seq_len, head_dim]`
/// - `mask`: optional attention mask (nullable).
pub fn attention(
    stream: &MlxStream,
    q: &MlxTensor,
    k: &MlxTensor,
    v: &MlxTensor,
    mask: Option<&MlxTensor>,
    scale: f32,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&q.ctx);
    let mask_ptr = mask.map_or(std::ptr::null(), |m| m.inner.cast_const());
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_attention(
            ctx.inner,
            stream.inner,
            q.inner,
            k.inner,
            v.inner,
            mask_ptr,
            scale,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Element-wise addition: `out = a + b`.
pub fn add(stream: &MlxStream, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&a.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status =
        unsafe { ffi::beacon_op_add(ctx.inner, stream.inner, a.inner, b.inner, &raw mut out) };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Element-wise multiplication: `out = a * b`.
pub fn elementwise_mul(
    stream: &MlxStream,
    a: &MlxTensor,
    b: &MlxTensor,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&a.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_elementwise_mul(ctx.inner, stream.inner, a.inner, b.inner, &raw mut out)
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Update KV cache at `position` with new K/V values.
///
/// Returns views over `[0..=position]` for attention to consume.
pub fn kv_cache_update(
    stream: &MlxStream,
    cache_k: &MlxTensor,
    cache_v: &MlxTensor,
    new_k: &MlxTensor,
    new_v: &MlxTensor,
    position: i64,
) -> Result<(MlxTensor, MlxTensor), MlxError> {
    let ctx = Arc::clone(&cache_k.ctx);
    let mut out_k: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let mut out_v: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_kv_cache_update(
            ctx.inner,
            stream.inner,
            cache_k.inner,
            cache_v.inner,
            new_k.inner,
            new_v.inner,
            position,
            &raw mut out_k,
            &raw mut out_v,
        )
    };
    status_to_result(status)?;
    Ok((
        MlxTensor::from_raw(Arc::clone(&ctx), out_k),
        MlxTensor::from_raw(ctx, out_v),
    ))
}

/// Reshape a tensor to `new_shape` (total elements must be unchanged).
pub fn reshape(
    stream: &MlxStream,
    x: &MlxTensor,
    new_shape: &[i64],
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_reshape(
            ctx.inner,
            stream.inner,
            x.inner,
            new_shape.as_ptr(),
            new_shape.len(),
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Transpose the last two dimensions of a tensor.
pub fn transpose(stream: &MlxStream, x: &MlxTensor) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status =
        unsafe { ffi::beacon_op_transpose(ctx.inner, stream.inner, x.inner, &raw mut out) };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Embedding lookup: select rows from `weight` by `indices`.
///
/// `weight`: `[vocab_size, hidden_dim]`, `indices`: `[seq_len]` (i32).
/// Returns `[seq_len, hidden_dim]`.
pub fn embedding(
    stream: &MlxStream,
    weight: &MlxTensor,
    indices: &MlxTensor,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&weight.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_embedding(
            ctx.inner,
            stream.inner,
            weight.inner,
            indices.inner,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Custom Metal kernel: Q4 dequant + matmul (stub in v0.1).
pub fn kernel_q4_dequant_mul(
    stream: &MlxStream,
    x: &MlxTensor,
    w_q4: &MlxTensor,
    scales: &MlxTensor,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_kernel_q4_dequant_mul(
            ctx.inner,
            stream.inner,
            x.inner,
            w_q4.inner,
            scales.inner,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}
