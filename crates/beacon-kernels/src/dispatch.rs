//! Runtime CPU feature detection and kernel dispatch.
//!
//! On aarch64, NEON is always available (mandatory in `ARMv8`). On `x86_64`,
//! AVX2 and AVX-512 are detected at runtime.

/// Detected CPU SIMD capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD — scalar fallback.
    Scalar,
    /// ARM NEON (aarch64, always available).
    Neon,
    /// `x86_64` AVX2.
    Avx2,
    /// `x86_64` AVX-512.
    Avx512,
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            Self::Neon => write!(f, "NEON"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
        }
    }
}

/// Detect the best available SIMD level for the current CPU.
pub fn detect() -> SimdLevel {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64.
        return SimdLevel::Neon;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::Avx2;
        }
    }

    #[allow(unreachable_code)]
    SimdLevel::Scalar
}

/// `Q4_0` matrix-vector multiply dispatched to the best available SIMD path.
///
/// `w_q4`: `Q4_0`-quantized weight matrix `[m, k]`, `x`: input `[k]`,
/// `out`: output `[m]`. `k` must be a multiple of 32.
pub fn q4_matmul_f32(w_q4: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    match detect() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => {
            // SAFETY: NEON is mandatory on aarch64.
            unsafe { crate::neon::q4_matmul_f32_neon(w_q4, x, out, m, k) }
        }
        // AVX2 and AVX-512 use scalar for now; optimized implementations
        // will land in a follow-up when we have x86_64 hardware to benchmark.
        _ => crate::ops::q4_matmul_f32(w_q4, x, out, m, k),
    }
}

/// `Q4_0` single-row dot product dispatched to the best SIMD path.
pub fn q4_dot_f32(x: &[f32], w_q4: &[u8], k: usize) -> f32 {
    match detect() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => {
            // SAFETY: NEON is mandatory on aarch64.
            unsafe { crate::neon::q4_dot_f32_neon(x, w_q4, k) }
        }
        _ => crate::ops::q4_dot_f32(x, w_q4, k),
    }
}
