//! Correctness tests for CPU kernels.
//!
//! Tests verify scalar implementations against known results, and on aarch64,
//! verify that NEON kernels produce the same results as scalar.

use crate::ops;
use crate::q4;

// ---------------------------------------------------------------------------
// Q4 dequantization
// ---------------------------------------------------------------------------

/// Build a `Q4_0` block from a known scale and weight pattern.
fn build_q4_block(scale_f32: f32, weights: &[i8; 32]) -> [u8; q4::Q4_0_BLOCK_BYTES] {
    let mut block = [0u8; q4::Q4_0_BLOCK_BYTES];
    // Encode scale as f16 (approximate).
    let scale_f16 = f32_to_f16_approx(scale_f32);
    block[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    // Pack weights: each pair (lo, hi) into one byte.
    for i in 0..16 {
        #[allow(clippy::cast_sign_loss)]
        let lo = (weights[i * 2] + 8) as u8;
        #[allow(clippy::cast_sign_loss)]
        let hi = (weights[i * 2 + 1] + 8) as u8;
        block[2 + i] = (hi << 4) | (lo & 0x0F);
    }
    block
}

/// Approximate `f32` to `f16` conversion for test data.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn f32_to_f16_approx(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let man = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign as u16;
    }
    if exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    #[allow(clippy::cast_sign_loss)]
    let result = sign | ((exp as u32) << 10) | man;
    result as u16
}

#[test]
fn q4_dequantize_known_values() {
    // Scale = 0.5, all weights = 1 → dequantized = 0.5 * 1 = 0.5
    let mut weights = [1i8; 32];
    let block = build_q4_block(0.5, &weights);
    let mut out = [0.0f32; 32];
    q4::dequantize_q4_0_block(&block, &mut out);
    for &v in &out {
        assert!((v - 0.5).abs() < 0.01, "expected ~0.5, got {v}");
    }

    // Scale = 2.0, weights = [-7, -3, 0, 3, 7, ...]
    weights = [0i8; 32];
    weights[0] = -7;
    weights[1] = -3;
    weights[2] = 0;
    weights[3] = 3;
    weights[4] = 7;
    let block = build_q4_block(2.0, &weights);
    q4::dequantize_q4_0_block(&block, &mut out);
    assert!((out[0] - (-14.0)).abs() < 0.1);
    assert!((out[1] - (-6.0)).abs() < 0.1);
    assert!(out[2].abs() < 0.1);
    assert!((out[3] - 6.0).abs() < 0.1);
    assert!((out[4] - 14.0).abs() < 0.1);
}

// ---------------------------------------------------------------------------
// Q4 dot product
// ---------------------------------------------------------------------------

#[test]
fn q4_dot_scalar() {
    // 32-element dot product: x = [1.0; 32], w = all-ones with scale=1.0
    // Expected: sum of 32 × (1.0 × 1.0) = 32.0
    let x = vec![1.0f32; 32];
    let block = build_q4_block(1.0, &[1i8; 32]);
    let result = ops::q4_dot_f32(&x, &block, 32);
    assert!((result - 32.0).abs() < 1.0, "expected ~32.0, got {result}");
}

#[test]
fn q4_matmul_scalar() {
    let k = 32;
    let m = 2;
    let x = vec![1.0f32; k];
    // Two rows of Q4 weights, both all-ones with scale=1.0
    let block = build_q4_block(1.0, &[1i8; 32]);
    let mut w_q4 = Vec::new();
    w_q4.extend_from_slice(&block);
    w_q4.extend_from_slice(&block);

    let mut out = vec![0.0f32; m];
    ops::q4_matmul_f32(&w_q4, &x, &mut out, m, k);
    for &v in &out {
        assert!((v - 32.0).abs() < 1.0, "expected ~32.0, got {v}");
    }
}

// ---------------------------------------------------------------------------
// FP32 matmul
// ---------------------------------------------------------------------------

#[test]
fn matmul_f32_known() {
    // [[1,2,3],[4,5,6]] × [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0f32; 4];
    ops::matmul_f32(&a, &b, &mut c, 2, 3, 2);
    let expected = [58.0, 64.0, 139.0, 154.0];
    for (i, (&got, &want)) in c.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < f32::EPSILON,
            "matmul_f32[{i}]: got {got}, expected {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------

#[test]
fn rms_norm_known() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0; 4];
    let mut out = [0.0f32; 4];
    ops::rms_norm(&x, &weight, &mut out, 1e-5);

    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    let rms = (7.5f32 + 1e-5).sqrt();
    for (i, &v) in out.iter().enumerate() {
        let expected = x[i] / rms;
        assert!(
            (v - expected).abs() < 1e-5,
            "rms_norm[{i}]: got {v}, expected {expected}"
        );
    }
}

#[test]
fn silu_known() {
    let mut x = [0.0f32, 1.0, -1.0, 2.0];
    let expected: Vec<f32> = x.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect();
    ops::silu_inplace(&mut x);
    for (i, (&got, &want)) in x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "silu[{i}]: got {got}, expected {want}"
        );
    }
}

#[test]
fn softmax_known() {
    let mut x = [1.0f32, 2.0, 3.0];
    ops::softmax_inplace(&mut x);
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax should sum to 1, got {sum}"
    );
    // Check ordering: softmax preserves relative order.
    assert!(x[0] < x[1]);
    assert!(x[1] < x[2]);
}

#[test]
fn rope_known() {
    // Single position, dim=4, offset=0, theta=10000.
    let mut x = [1.0, 0.0, 1.0, 0.0];
    ops::rope_inplace(&mut x, 1, 4, 0, 10000.0);
    // At position 0, all angles are 0, so cos=1, sin=0 → x unchanged.
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!(x[1].abs() < 1e-6);
    assert!((x[2] - 1.0).abs() < 1e-6);
    assert!(x[3].abs() < 1e-6);

    // At position 1, the rotation should be non-trivial.
    let mut x2 = [1.0, 0.0, 1.0, 0.0];
    ops::rope_inplace(&mut x2, 1, 4, 1, 10000.0);
    // First pair rotated by freq = 1/10000^(0/4) = 1.0, angle = 1.0
    assert!((x2[0] - 1.0_f32.cos()).abs() < 1e-5);
    assert!((x2[1] - 1.0_f32.sin()).abs() < 1e-5);
}

#[test]
fn add_known() {
    let a = [1.0, 2.0, 3.0];
    let b = [10.0, 20.0, 30.0];
    let mut out = [0.0f32; 3];
    ops::add(&a, &b, &mut out);
    let expected = [11.0, 22.0, 33.0];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < f32::EPSILON,
            "add[{i}]: got {got}, expected {want}"
        );
    }
}

#[test]
fn mul_known() {
    let a = [2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0];
    let mut out = [0.0f32; 3];
    ops::mul(&a, &b, &mut out);
    let expected = [10.0, 18.0, 28.0];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < f32::EPSILON,
            "mul[{i}]: got {got}, expected {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// NEON vs scalar cross-check (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[test]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn neon_q4_dot_matches_scalar() {
    let k = 64; // 2 blocks
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

    // Build 2 blocks with varied weights.
    let mut w_q4 = Vec::new();
    let mut weights1 = [0i8; 32];
    let mut weights2 = [0i8; 32];
    for i in 0..32 {
        weights1[i] = (i as i8 % 7) - 3;
        weights2[i] = -(i as i8 % 5) + 2;
    }
    w_q4.extend_from_slice(&build_q4_block(0.5, &weights1));
    w_q4.extend_from_slice(&build_q4_block(1.5, &weights2));

    let scalar_result = ops::q4_dot_f32(&x, &w_q4, k);
    let neon_result = unsafe { crate::neon::q4_dot_f32_neon(&x, &w_q4, k) };

    assert!(
        (scalar_result - neon_result).abs() < 0.5,
        "NEON ({neon_result}) should match scalar ({scalar_result})"
    );
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#[test]
fn dispatch_detect() {
    let level = crate::dispatch::detect();
    // Should be Neon on aarch64, Scalar or Avx2/Avx512 on x86_64.
    #[cfg(target_arch = "aarch64")]
    assert_eq!(level, crate::SimdLevel::Neon);

    // Just verify it doesn't panic on any platform.
    let _ = format!("{level}");
}
