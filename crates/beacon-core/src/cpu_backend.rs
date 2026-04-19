//! CPU compute backend using `beacon_kernels` ops.
//!
//! This is the fallback backend for systems without MLX (non-Apple-Silicon).
//! All computation is eager (no lazy graph), using f32 precision throughout.

use crate::backend::ComputeBackend;
use crate::error::EngineError;

/// A simple f32 tensor for the CPU backend.
///
/// Data is stored in row-major order. Shape dimensions are signed i64 to
/// match the `ComputeBackend` trait's reshape API.
#[derive(Debug, Clone)]
pub struct CpuTensor {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions (e.g. `[batch, heads, seq, dim]`).
    pub shape: Vec<i64>,
}

impl CpuTensor {
    /// Total number of elements.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }
}

/// CPU stream — no-op (CPU execution is eager).
#[derive(Debug)]
pub struct CpuStream;

/// CPU compute backend using `beacon_kernels` ops.
#[derive(Debug)]
pub struct CpuBackend;

/// Helper: convert i64 shape dim to usize, used pervasively in this module.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn dim(v: i64) -> usize {
    v as usize
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
impl ComputeBackend for CpuBackend {
    type Tensor = CpuTensor;
    type Stream = CpuStream;

    fn new_stream(&self) -> Result<Self::Stream, EngineError> {
        Ok(CpuStream)
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul(
        &self,
        _stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        // Support 2-D matmul: a=[m,k], b=[k,n] -> out=[m,n].
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(EngineError::ShapeMismatch(format!(
                "matmul requires 2-D tensors, got {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let m = dim(a.shape[0]);
        let k = dim(a.shape[1]);
        let k2 = dim(b.shape[0]);
        let n = dim(b.shape[1]);
        if k != k2 {
            return Err(EngineError::ShapeMismatch(format!(
                "matmul inner dim mismatch: {k} vs {k2}"
            )));
        }
        let mut out = vec![0.0f32; m * n];
        beacon_kernels::ops::matmul_f32(&a.data, &b.data, &mut out, m, k, n);
        Ok(CpuTensor {
            data: out,
            shape: vec![m as i64, n as i64],
        })
    }

    fn quantized_matmul(
        &self,
        _stream: &Self::Stream,
        _x: &Self::Tensor,
        _w: &Self::Tensor,
        _scales: &Self::Tensor,
        _group_size: i32,
        _bits: i32,
    ) -> Result<Self::Tensor, EngineError> {
        Err(EngineError::Backend(
            "quantized_matmul not implemented for CPU backend in v0.1".to_owned(),
        ))
    }

    fn rms_norm(
        &self,
        _stream: &Self::Stream,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, EngineError> {
        // Handle both 1-D [dim] and 2-D [seq, dim] inputs.
        // RMS norm is applied per-row (last dimension).
        let d = dim(*x
            .shape
            .last()
            .ok_or_else(|| EngineError::ShapeMismatch("rms_norm: empty shape".to_owned()))?);

        if weight.data.len() != d {
            return Err(EngineError::ShapeMismatch(format!(
                "rms_norm weight length {} != dim {d}",
                weight.data.len()
            )));
        }

        let rows = x.data.len() / d;
        let mut out = vec![0.0f32; x.data.len()];
        for r in 0..rows {
            let offset = r * d;
            beacon_kernels::ops::rms_norm(
                &x.data[offset..offset + d],
                &weight.data,
                &mut out[offset..offset + d],
                eps,
            );
        }
        Ok(CpuTensor {
            data: out,
            shape: x.shape.clone(),
        })
    }

    fn rope(
        &self,
        _stream: &Self::Stream,
        x: &Self::Tensor,
        position_offset: i32,
        theta: f32,
        rope_dim: i32,
    ) -> Result<Self::Tensor, EngineError> {
        // x may be [batch, heads, seq, head_dim] or [seq, dim].
        // RoPE applies to the last two dimensions: seq_len rows of `dim` width.
        let dim_u = rope_dim as usize;
        let total = x.data.len();
        let seq_len = if x.shape.len() >= 2 {
            dim(x.shape[x.shape.len() - 2])
        } else {
            1
        };

        // Number of independent groups (batch * heads).
        let groups = total / (seq_len * dim_u);

        let mut data = x.data.clone();
        for g in 0..groups {
            let offset = g * seq_len * dim_u;
            beacon_kernels::ops::rope_inplace(
                &mut data[offset..offset + seq_len * dim_u],
                seq_len,
                dim_u,
                position_offset as usize,
                theta,
            );
        }
        Ok(CpuTensor {
            data,
            shape: x.shape.clone(),
        })
    }

    fn attention(
        &self,
        _stream: &Self::Stream,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: Option<&Self::Tensor>,
        scale: f32,
    ) -> Result<Self::Tensor, EngineError> {
        cpu_attention(q, k, v, mask, scale)
    }

    fn silu(&self, _stream: &Self::Stream, x: &Self::Tensor) -> Result<Self::Tensor, EngineError> {
        let mut data = x.data.clone();
        beacon_kernels::ops::silu_inplace(&mut data);
        Ok(CpuTensor {
            data,
            shape: x.shape.clone(),
        })
    }

    fn softmax(
        &self,
        _stream: &Self::Stream,
        x: &Self::Tensor,
        _axis: i32,
    ) -> Result<Self::Tensor, EngineError> {
        // Apply softmax over the last dimension (matching common transformer usage).
        let last_dim = dim(*x
            .shape
            .last()
            .ok_or_else(|| EngineError::ShapeMismatch("softmax: empty shape".to_owned()))?);
        let mut data = x.data.clone();
        let rows = data.len() / last_dim;
        for r in 0..rows {
            let offset = r * last_dim;
            beacon_kernels::ops::softmax_inplace(&mut data[offset..offset + last_dim]);
        }
        Ok(CpuTensor {
            data,
            shape: x.shape.clone(),
        })
    }

    fn add(
        &self,
        _stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        if a.data.len() != b.data.len() {
            return Err(EngineError::ShapeMismatch(format!(
                "add length mismatch: {} vs {}",
                a.data.len(),
                b.data.len()
            )));
        }
        let mut out = vec![0.0f32; a.data.len()];
        beacon_kernels::ops::add(&a.data, &b.data, &mut out);
        Ok(CpuTensor {
            data: out,
            shape: a.shape.clone(),
        })
    }

    fn mul(
        &self,
        _stream: &Self::Stream,
        a: &Self::Tensor,
        b: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        if a.data.len() != b.data.len() {
            return Err(EngineError::ShapeMismatch(format!(
                "mul length mismatch: {} vs {}",
                a.data.len(),
                b.data.len()
            )));
        }
        let mut out = vec![0.0f32; a.data.len()];
        beacon_kernels::ops::mul(&a.data, &b.data, &mut out);
        Ok(CpuTensor {
            data: out,
            shape: a.shape.clone(),
        })
    }

    fn reshape(
        &self,
        _stream: &Self::Stream,
        x: &Self::Tensor,
        shape: &[i64],
    ) -> Result<Self::Tensor, EngineError> {
        let new_numel: usize = shape.iter().map(|&d| d as usize).product();
        let old_numel = x.numel();
        if new_numel != old_numel {
            return Err(EngineError::ShapeMismatch(format!(
                "reshape: element count mismatch {old_numel} vs {new_numel}"
            )));
        }
        Ok(CpuTensor {
            data: x.data.clone(),
            shape: shape.to_vec(),
        })
    }

    fn embedding(
        &self,
        _stream: &Self::Stream,
        weight: &Self::Tensor,
        indices: &Self::Tensor,
    ) -> Result<Self::Tensor, EngineError> {
        // weight: [vocab, dim], indices: [seq] (f32 values representing integer indices).
        if weight.shape.len() != 2 {
            return Err(EngineError::ShapeMismatch(
                "embedding: weight must be 2-D".to_owned(),
            ));
        }
        let embed_dim = dim(weight.shape[1]);
        let seq = indices.data.len();
        let mut out = Vec::with_capacity(seq * embed_dim);
        for &idx_f in &indices.data {
            let idx = idx_f as usize;
            let start = idx * embed_dim;
            let end = start + embed_dim;
            if end > weight.data.len() {
                return Err(EngineError::ShapeMismatch(format!(
                    "embedding index {idx} out of bounds for vocab {}",
                    weight.shape[0]
                )));
            }
            out.extend_from_slice(&weight.data[start..end]);
        }
        Ok(CpuTensor {
            data: out,
            shape: vec![seq as i64, embed_dim as i64],
        })
    }

    fn eval(&self, _t: &Self::Tensor, _stream: &Self::Stream) -> Result<(), EngineError> {
        // CPU execution is eager — nothing to evaluate.
        Ok(())
    }

    fn read_f32(&self, t: &Self::Tensor, n: usize) -> Result<Vec<f32>, EngineError> {
        if n > t.data.len() {
            return Err(EngineError::ShapeMismatch(format!(
                "read_f32: requested {n} elements but tensor has {}",
                t.data.len()
            )));
        }
        Ok(t.data[..n].to_vec())
    }

    fn kv_cache_update(
        &self,
        _stream: &Self::Stream,
        cache_k: &Self::Tensor,
        cache_v: &Self::Tensor,
        new_k: &Self::Tensor,
        new_v: &Self::Tensor,
        position: i64,
    ) -> Result<(Self::Tensor, Self::Tensor), EngineError> {
        // cache shape: [max_ctx, n_kv_heads, head_dim]
        // new_k/v shape: [1, n_kv_heads, seq, head_dim]
        // We copy new_k/v into the cache at the given position, then return
        // slices [0..position+seq_len].

        let n_kv_heads = dim(cache_k.shape[1]);
        let head_dim = dim(cache_k.shape[2]);
        let row_size = n_kv_heads * head_dim;
        let pos = position as usize;

        // Determine seq_len of new data.
        let new_elements = new_k.data.len();
        let seq_len = new_elements / row_size;

        // Copy cache data and insert new K/V.
        let mut k_data = cache_k.data.clone();
        let mut v_data = cache_v.data.clone();

        for s in 0..seq_len {
            let cache_offset = (pos + s) * row_size;
            let new_offset = s * row_size;
            k_data[cache_offset..cache_offset + row_size]
                .copy_from_slice(&new_k.data[new_offset..new_offset + row_size]);
            v_data[cache_offset..cache_offset + row_size]
                .copy_from_slice(&new_v.data[new_offset..new_offset + row_size]);
        }

        // Return the slice [0..pos+seq_len] reshaped for attention:
        // [1, n_kv_heads, pos+seq_len, head_dim]
        let out_len = pos + seq_len;
        let out_k = CpuTensor {
            data: k_data[..out_len * row_size].to_vec(),
            shape: vec![1, n_kv_heads as i64, out_len as i64, head_dim as i64],
        };
        let out_v = CpuTensor {
            data: v_data[..out_len * row_size].to_vec(),
            shape: vec![1, n_kv_heads as i64, out_len as i64, head_dim as i64],
        };
        Ok((out_k, out_v))
    }

    fn create_token_tensor(&self, tokens: &[u32]) -> Result<Self::Tensor, EngineError> {
        // Store token IDs as f32 for embedding lookup (cast, not reinterpret).
        let data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        Ok(CpuTensor {
            data,
            shape: vec![tokens.len() as i64],
        })
    }
}

/// CPU attention: Q @ K^T * scale, optional mask, softmax, @ V.
///
/// Handles GQA by mapping query heads to KV heads.
///
/// Shapes:
/// - q: `[batch, n_heads, seq_q, head_dim]`
/// - k: `[batch, n_kv_heads, seq_kv, head_dim]`
/// - v: `[batch, n_kv_heads, seq_kv, head_dim]`
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]
fn cpu_attention(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    mask: Option<&CpuTensor>,
    scale: f32,
) -> Result<CpuTensor, EngineError> {
    if q.shape.len() != 4 || k.shape.len() != 4 || v.shape.len() != 4 {
        return Err(EngineError::ShapeMismatch(
            "attention: q/k/v must be 4-D".to_owned(),
        ));
    }

    let batch = dim(q.shape[0]);
    let n_heads = dim(q.shape[1]);
    let seq_q = dim(q.shape[2]);
    let head_dim = dim(q.shape[3]);

    let n_kv_heads = dim(k.shape[1]);
    let seq_kv = dim(k.shape[2]);

    if n_heads % n_kv_heads != 0 {
        return Err(EngineError::ShapeMismatch(format!(
            "attention: n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )));
    }
    let heads_per_kv = n_heads / n_kv_heads;

    let mut out = vec![0.0f32; batch * n_heads * seq_q * head_dim];

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;

            for sq in 0..seq_q {
                // Compute scores: Q[b,h,sq,:] @ K[b,kv_h,:,:]^T * scale
                let mut scores = vec![0.0f32; seq_kv];
                let q_offset = ((b * n_heads + h) * seq_q + sq) * head_dim;

                for sk in 0..seq_kv {
                    let k_offset = ((b * n_kv_heads + kv_h) * seq_kv + sk) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q.data[q_offset + d] * k.data[k_offset + d];
                    }
                    scores[sk] = dot * scale;
                }

                // Apply mask if present.
                if let Some(m) = mask {
                    let mask_numel = m.data.len();
                    let mask_row_len = if mask_numel >= seq_q * seq_kv {
                        seq_kv
                    } else {
                        mask_numel
                    };
                    if mask_row_len == seq_kv {
                        let m_offset = sq * seq_kv;
                        for sk in 0..seq_kv {
                            if m_offset + sk < m.data.len() {
                                scores[sk] += m.data[m_offset + sk];
                            }
                        }
                    }
                }

                // Softmax over scores.
                beacon_kernels::ops::softmax_inplace(&mut scores);

                // Weighted sum: scores @ V[b,kv_h,:,:]
                let out_offset = ((b * n_heads + h) * seq_q + sq) * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for sk in 0..seq_kv {
                        let v_offset = ((b * n_kv_heads + kv_h) * seq_kv + sk) * head_dim;
                        val += scores[sk] * v.data[v_offset + d];
                    }
                    out[out_offset + d] = val;
                }
            }
        }
    }

    Ok(CpuTensor {
        data: out,
        shape: vec![batch as i64, n_heads as i64, seq_q as i64, head_dim as i64],
    })
}
