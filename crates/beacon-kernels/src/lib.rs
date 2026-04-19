//! CPU SIMD kernels for the Beacon CPU backend.
//!
//! Provides optimized implementations of the core transformer operations:
//! Q4 dequantize-multiply, FP32 matmul, RMS normalization, `SiLU` activation,
//! softmax, and rotary position embedding.
//!
//! The [`dispatch`] module selects the best SIMD path at runtime:
//! NEON on aarch64, AVX2/AVX-512 on `x86_64`, scalar fallback elsewhere.
//! The scalar implementations in [`ops`] serve as the reference for
//! correctness testing.

pub mod dispatch;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod ops;
pub mod q4;

/// Re-export the detected SIMD level.
pub use dispatch::SimdLevel;

#[cfg(test)]
mod tests;
