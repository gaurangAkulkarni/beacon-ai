//! NEON-optimized kernels for aarch64 (Apple Silicon, Linux ARM).
//!
//! These are only compiled on `aarch64` targets. The dispatch layer falls back
//! to scalar on other architectures.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

use crate::q4;

/// NEON-accelerated `Q4_0` dot product: `dot(x, dequant(w_q4))`.
///
/// Processes 32 elements per block using NEON SIMD.
///
/// # Safety
///
/// Requires aarch64 NEON (always available on Apple Silicon and `ARMv8+`).
#[target_feature(enable = "neon")]
pub unsafe fn q4_dot_f32_neon(x: &[f32], w_q4: &[u8], k: usize) -> f32 {
    debug_assert_eq!(k % q4::Q4_0_BLOCK_SIZE, 0);

    let num_blocks = k / q4::Q4_0_BLOCK_SIZE;
    // SAFETY: All operations below use NEON intrinsics which are safe on aarch64.
    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        for b in 0..num_blocks {
            let block = &w_q4[b * q4::Q4_0_BLOCK_BYTES..];
            let scale = q4_block_scale_f32(block);
            let scale_vec = vdupq_n_f32(scale);

            // Dequantize: unpack 16 bytes → 32 × f32, then dot with x.
            // For simplicity and correctness, dequantize to a temp buffer
            // and use NEON fma for the dot product.
            let mut deq = [0.0f32; q4::Q4_0_BLOCK_SIZE];
            q4::dequantize_q4_0_block(
                &w_q4[b * q4::Q4_0_BLOCK_BYTES..(b + 1) * q4::Q4_0_BLOCK_BYTES],
                &mut deq,
            );

            let x_base = x.as_ptr().add(b * q4::Q4_0_BLOCK_SIZE);

            // 32 elements = 8 × f32x4 lanes.
            for chunk in 0..8 {
                let offset = chunk * 4;
                let w_f32 = vld1q_f32(deq.as_ptr().add(offset));
                let x_f32 = vld1q_f32(x_base.add(offset));
                sum = vfmaq_f32(sum, w_f32, x_f32);
            }
            // Scale is already baked into the dequantized values, so no
            // separate multiply needed. But we used dequantize_q4_0_block
            // which already applies scale, so this is correct.
            let _ = scale_vec; // scale already applied in dequantization
        }

        vaddvq_f32(sum)
    }
}

/// Extract the `f16` scale from a `Q4_0` block header and convert to `f32`.
#[inline]
fn q4_block_scale_f32(block: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([block[0], block[1]]);
    q4::dequantize_q4_0_block_scale(bits)
}

/// NEON-accelerated `Q4_0` matrix-vector multiply.
///
/// # Safety
///
/// Requires aarch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn q4_matmul_f32_neon(w_q4: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let row_bytes = k / q4::Q4_0_BLOCK_SIZE * q4::Q4_0_BLOCK_BYTES;
    for i in 0..m {
        let row = &w_q4[i * row_bytes..(i + 1) * row_bytes];
        // SAFETY: caller guarantees NEON availability.
        out[i] = unsafe { q4_dot_f32_neon(x, row, k) };
    }
}
