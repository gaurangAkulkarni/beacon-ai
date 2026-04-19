//! `Q4_0` block layout and dequantization.
//!
//! `Q4_0` stores 32 elements per block as 16 bytes of packed 4-bit weights plus
//! a 2-byte `f16` scale factor (total 18 bytes/block). The layout matches
//! llama.cpp's `block_q4_0` from `ggml-quants.c`.
//!
//! ```text
//! Block layout (18 bytes):
//!   [0..2)   f16 scale (little-endian)
//!   [2..18)  16 bytes = 32 × 4-bit weights, packed low-nibble-first
//! ```
//!
//! Each 4-bit weight is an unsigned integer [0, 15] that represents the
//! signed value `(weight - 8) * scale`.

/// Bytes per `Q4_0` block.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Elements per `Q4_0` block.
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Dequantize a single `Q4_0` block into 32 `f32` values.
///
/// `block` must be exactly 18 bytes.
#[inline]
pub fn dequantize_q4_0_block(block: &[u8], out: &mut [f32; Q4_0_BLOCK_SIZE]) {
    debug_assert_eq!(block.len(), Q4_0_BLOCK_BYTES);

    // Scale is stored as f16 in the first 2 bytes.
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    // 16 bytes of packed 4-bit weights, 2 per byte, low nibble first.
    for i in 0..16 {
        let byte = block[2 + i];
        #[allow(clippy::cast_possible_wrap)]
        let lo = (byte & 0x0F) as i8 - 8;
        #[allow(clippy::cast_possible_wrap)]
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        out[i * 2] = f32::from(lo) * scale;
        out[i * 2 + 1] = f32::from(hi) * scale;
    }
}

/// Extract and convert the `f16` scale from `Q4_0` block header bits.
#[inline]
pub fn dequantize_q4_0_block_scale(bits: u16) -> f32 {
    f16_to_f32(bits)
}

/// Convert a 16-bit IEEE 754 half-precision float to `f32`.
#[inline]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    if exponent == 0 {
        if mantissa == 0 {
            // ±0
            return f32::from_bits(sign);
        }
        // Subnormal: convert to normalised f32.
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_man = m << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }

    if exponent == 31 {
        // Inf / NaN
        let f32_bits = sign | 0x7F80_0000 | (mantissa << 13);
        return f32::from_bits(f32_bits);
    }

    // Normal
    let f32_exp = (exponent + 112) << 23; // 112 = 127 - 15
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        // Test a few known f16 values.
        assert!((f16_to_f32(0x0000)).abs() < f32::EPSILON);
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < f32::EPSILON);
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < f32::EPSILON);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dequantize_q4_0_known() {
        // Build a block where scale=1.0 (f16: 0x3C00) and all weights = 8
        // (meaning value 0 after subtracting 8).
        let mut block = [0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00; // f16 1.0 = 0x3C00
        block[1] = 0x3C;
        // Weight nibbles all set to 8: byte = 0x88
        for b in &mut block[2..18] {
            *b = 0x88;
        }

        let mut out = [0.0f32; Q4_0_BLOCK_SIZE];
        dequantize_q4_0_block(&block, &mut out);

        // All values should be (8-8)*1.0 = 0.0
        for &v in &out {
            assert!(v.abs() < f32::EPSILON);
        }
    }
}
