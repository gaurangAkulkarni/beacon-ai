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

/// Quantized matrix multiplication: `out = x @ dequant(w, scales, biases)`.
///
/// The `biases` parameter is optional — pass `None` for the legacy path.
pub fn quantized_matmul(
    stream: &MlxStream,
    x: &MlxTensor,
    w: &MlxTensor,
    scales: &MlxTensor,
    biases: Option<&MlxTensor>,
    group_size: i32,
    bits: i32,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let biases_ptr = biases.map_or(std::ptr::null(), |b| b.inner.cast_const());
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_quantized_matmul(
            ctx.inner,
            stream.inner,
            x.inner,
            w.inner,
            scales.inner,
            biases_ptr,
            group_size,
            bits,
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(ctx, out))
}

/// Result of [`quantize`]: three tensors produced by MLX's `quantize()`.
#[derive(Debug)]
pub struct QuantizedTensors {
    /// Packed weight data as uint32.
    pub packed: MlxTensor,
    /// Per-group scales.
    pub scales: MlxTensor,
    /// Per-group biases.
    pub biases: MlxTensor,
}

/// Quantize a float matrix into MLX's internal quantized format.
///
/// Takes a `[rows, cols]` float tensor and returns packed weights, scales, and
/// biases suitable for [`quantized_matmul`].
pub fn quantize(
    stream: &MlxStream,
    w: &MlxTensor,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedTensors, MlxError> {
    let ctx = Arc::clone(&w.ctx);
    let mut out_packed: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let mut out_scales: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let mut out_biases: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_quantize(
            ctx.inner,
            stream.inner,
            w.inner,
            group_size,
            bits,
            &raw mut out_packed,
            &raw mut out_scales,
            &raw mut out_biases,
        )
    };
    status_to_result(status)?;
    Ok(QuantizedTensors {
        packed: MlxTensor::from_raw(Arc::clone(&ctx), out_packed),
        scales: MlxTensor::from_raw(Arc::clone(&ctx), out_scales),
        biases: MlxTensor::from_raw(ctx, out_biases),
    })
}

/// Dequantize raw GGUF quantized bytes to an F32 `MlxTensor` using gguflib.
///
/// `gguf_type` is the GGUF tensor type ID (2=`Q4_0`, 8=`Q8_0`, 12=`Q4_K`, etc.).
/// The returned tensor has the given `shape` and dtype F32.
#[allow(clippy::cast_possible_truncation)]
pub fn dequantize_gguf(
    stream: &MlxStream,
    ctx: &std::sync::Arc<crate::MlxContext>,
    data: &[u8],
    gguf_type: u32,
    num_elements: u64,
    shape: &[i64],
) -> Result<MlxTensor, MlxError> {
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_dequantize_gguf(
            ctx.inner,
            stream.inner,
            data.as_ptr().cast(),
            data.len(),
            #[allow(clippy::cast_possible_wrap)]
            {
                gguf_type as i32
            },
            num_elements,
            shape.as_ptr(),
            shape.len(),
            &raw mut out,
        )
    };
    status_to_result(status)?;
    Ok(MlxTensor::from_raw(std::sync::Arc::clone(ctx), out))
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
///
/// `freqs` is an optional tensor of custom `RoPE` frequencies (e.g.
/// `rope_freqs.weight` from Llama 3.2). When `None`, standard frequencies
/// derived from `theta` are used.
pub fn rope(
    stream: &MlxStream,
    x: &MlxTensor,
    position_offset: i32,
    theta: f32,
    dim: i32,
    freqs: Option<&MlxTensor>,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let freqs_ptr = freqs.map_or(std::ptr::null(), |f| f.inner.cast_const());
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_rope(
            ctx.inner,
            stream.inner,
            x.inner,
            position_offset,
            theta,
            dim,
            freqs_ptr,
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

/// Swap two axes of a tensor.
pub fn swapaxes(
    stream: &MlxStream,
    x: &MlxTensor,
    axis1: i32,
    axis2: i32,
) -> Result<MlxTensor, MlxError> {
    let ctx = Arc::clone(&x.ctx);
    let mut out: *mut ffi::BeaconTensor = std::ptr::null_mut();
    let status = unsafe {
        ffi::beacon_op_swapaxes(ctx.inner, stream.inner, x.inner, axis1, axis2, &raw mut out)
    };
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
