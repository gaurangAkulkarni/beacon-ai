//! Scalar reference implementations of transformer ops.
//!
//! These are correct but unoptimized. SIMD-accelerated versions in `neon` and
//! `avx2` modules override them on supported platforms.

use crate::q4;

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

/// Dense FP32 matrix multiplication: `C = A × B`.
///
/// `a`: `[m, k]` row-major, `b`: `[k, n]` row-major, `c`: `[m, n]` output.
#[allow(clippy::many_single_char_names)]
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// `Q4_0` dequantize-multiply: `out[i] += sum_j(dequant(w_q4)[j] * x[j])`.
///
/// Computes one output row: the dot product of `x` (f32, length `k`) with a
/// `Q4_0`-quantized weight row (`w_q4`, `k` elements packed in `Q4_0` blocks).
///
/// `k` must be a multiple of 32 (`Q4_0` block size).
pub fn q4_dot_f32(x: &[f32], w_q4: &[u8], k: usize) -> f32 {
    debug_assert_eq!(k % q4::Q4_0_BLOCK_SIZE, 0);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(w_q4.len(), k / q4::Q4_0_BLOCK_SIZE * q4::Q4_0_BLOCK_BYTES);

    let num_blocks = k / q4::Q4_0_BLOCK_SIZE;
    let mut sum = 0.0f32;
    let mut deq = [0.0f32; q4::Q4_0_BLOCK_SIZE];

    for b in 0..num_blocks {
        let block = &w_q4[b * q4::Q4_0_BLOCK_BYTES..(b + 1) * q4::Q4_0_BLOCK_BYTES];
        q4::dequantize_q4_0_block(block, &mut deq);
        let x_offset = b * q4::Q4_0_BLOCK_SIZE;
        for i in 0..q4::Q4_0_BLOCK_SIZE {
            sum += deq[i] * x[x_offset + i];
        }
    }
    sum
}

/// `Q4_0` x FP32 matrix-vector multiply: `out = W_q4 × x`.
///
/// `w_q4`: `[m, k]` `Q4_0`-quantized weight matrix (row-major, `m` rows of `k`
/// elements each). `x`: `[k]` input vector. `out`: `[m]` output vector.
pub fn q4_matmul_f32(w_q4: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let row_bytes = k / q4::Q4_0_BLOCK_SIZE * q4::Q4_0_BLOCK_BYTES;
    debug_assert_eq!(w_q4.len(), m * row_bytes);

    for i in 0..m {
        let row = &w_q4[i * row_bytes..(i + 1) * row_bytes];
        out[i] = q4_dot_f32(x, row, k);
    }
}

// ---------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------

/// RMS normalization: `out[i] = x[i] * weight[i] / rms(x)`.
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    debug_assert_eq!(out.len(), n);

    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    #[allow(clippy::cast_precision_loss)]
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/// `SiLU` activation (in-place): `x[i] = x[i] * sigmoid(x[i])`.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// Softmax (in-place) along the last `n` elements of `x`.
///
/// For numerical stability, subtracts the max before exponentiating.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Rotary position embedding (`RoPE`).
///
/// Applies `RoPE` to `x` (shape `[seq_len, dim]`) starting at `position_offset`.
/// `theta` is the base frequency (e.g. 10000.0). `dim` is the embedding
/// dimension (must be even).
#[allow(clippy::cast_precision_loss)]
pub fn rope_inplace(x: &mut [f32], seq_len: usize, dim: usize, position_offset: usize, theta: f32) {
    debug_assert_eq!(dim % 2, 0);
    debug_assert_eq!(x.len(), seq_len * dim);

    for s in 0..seq_len {
        let pos = (position_offset + s) as f32;
        let row = &mut x[s * dim..(s + 1) * dim];
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / dim as f32);
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x0 = row[i];
            let x1 = row[i + 1];
            row[i] = x0 * cos_a - x1 * sin_a;
            row[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// Element-wise addition: `out[i] = a[i] + b[i]`.
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
pub fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}
