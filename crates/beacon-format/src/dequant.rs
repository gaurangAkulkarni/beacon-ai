//! Dequantization of GGUF quantized weights to F16.
//!
//! Quantized weight data (`Q4_0`, `Q8_0`, `Q4_K`, etc.) is expanded to F16
//! during GGUF-to-beacon conversion so that `.beacon` files contain only
//! non-quantized data ready for zero-copy loading.
//!
//! This module implements dequantization for every GGUF quantization type
//! encountered in practice. A native quantized-matmul bridge (avoiding the
//! F16 expansion entirely) is planned for v0.2.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use crate::BeaconDtype;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Dequantize raw quantized bytes to F16 bit patterns (`u16` values).
///
/// Returns a `Vec<u16>` where each element is an IEEE 754 half-precision
/// float stored as its raw bit pattern (little-endian).
///
/// # Panics
///
/// Panics if `data.len()` is not consistent with `dtype` and `num_elements`.
pub fn dequantize_to_f16(data: &[u8], dtype: BeaconDtype, num_elements: u64) -> Vec<u16> {
    let n = num_elements as usize;
    let mut out = vec![0u16; n];

    match dtype {
        BeaconDtype::Q4_0 => dequant_q4_0(data, &mut out),
        BeaconDtype::Q4_1 => dequant_q4_1(data, &mut out),
        BeaconDtype::Q5_0 => dequant_q5_0(data, &mut out),
        BeaconDtype::Q5_1 => dequant_q5_1(data, &mut out),
        BeaconDtype::Q8_0 => dequant_q8_0(data, &mut out),
        BeaconDtype::Q4K => dequant_q4_k(data, &mut out),
        BeaconDtype::Q5K => dequant_q5_k(data, &mut out),
        BeaconDtype::Q6K => dequant_q6_k(data, &mut out),
        BeaconDtype::Q2K => dequant_q2_k(data, &mut out),
        BeaconDtype::Q3K => dequant_q3_k(data, &mut out),
        BeaconDtype::Q8K => dequant_q8_k(data, &mut out),
        // Non-quantized types should never reach here.
        BeaconDtype::F32
        | BeaconDtype::F16
        | BeaconDtype::BF16
        | BeaconDtype::I32
        | BeaconDtype::I8 => {
            unreachable!("dequantize_to_f16 called on non-quantized dtype {dtype:?}")
        }
    }

    out
}

/// Returns `true` if the dtype is quantized and needs dequantization.
pub fn is_quantized(dtype: BeaconDtype) -> bool {
    !matches!(
        dtype,
        BeaconDtype::F32
            | BeaconDtype::F16
            | BeaconDtype::BF16
            | BeaconDtype::I32
            | BeaconDtype::I8
    )
}

// ---------------------------------------------------------------------------
// F16 / F32 conversion helpers
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (16-bit) to single-precision (32-bit).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal
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
        return f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13));
    }

    let f32_exp = (exponent + 112) << 23;
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}

/// Convert single-precision (32-bit) to half-precision (16-bit).
///
/// Uses truncation (round-toward-zero). Sufficient for weight dequantization
/// where the values are already quantized and precision is limited.
#[inline]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Inf / NaN
        if mantissa == 0 {
            return sign | 0x7C00; // Inf
        }
        return sign | 0x7C00 | ((mantissa >> 13) as u16).max(1); // NaN
    }

    // Rebias exponent: f32 bias is 127, f16 bias is 15.
    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        // Overflow to Inf
        return sign | 0x7C00;
    }

    if new_exp <= 0 {
        // Subnormal or underflow
        if new_exp < -10 {
            return sign; // Too small, flush to zero
        }
        // Subnormal: shift mantissa (with implicit 1 bit)
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
        return sign | m as u16;
    }

    // Normal
    let exp_bits = (new_exp as u16) << 10;
    let man_bits = (mantissa >> 13) as u16;
    sign | exp_bits | man_bits
}

// ---------------------------------------------------------------------------
// Simple quant types (32 elements per block)
// ---------------------------------------------------------------------------

/// `Q4_0`: 18 bytes/block, 32 elements.
/// Layout: `[f16 scale (2 bytes)] [16 bytes packed 4-bit weights]`
fn dequant_q4_0(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_SIZE: usize = 32;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dst = &mut out[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = block[2 + i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            dst[i * 2] = f32_to_f16(f32::from(lo) * scale);
            dst[i * 2 + 1] = f32_to_f16(f32::from(hi) * scale);
        }
    }
}

/// `Q4_1`: 20 bytes/block, 32 elements.
/// Layout: `[f16 scale (2)] [f16 min (2)] [16 bytes packed 4-bit weights]`
fn dequant_q4_1(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 20;
    const BLOCK_SIZE: usize = 32;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min_val = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let dst = &mut out[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = block[4 + i];
            let lo = f32::from(byte & 0x0F);
            let hi = f32::from(byte >> 4);
            dst[i * 2] = f32_to_f16(lo * scale + min_val);
            dst[i * 2 + 1] = f32_to_f16(hi * scale + min_val);
        }
    }
}

/// `Q5_0`: 22 bytes/block, 32 elements.
/// Layout: `[f16 scale (2)] [u32 high-bits (4)] [16 bytes packed 4-bit low weights]`
fn dequant_q5_0(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 22;
    const BLOCK_SIZE: usize = 32;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let dst = &mut out[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = block[6 + i];
            let mut lo = u32::from(byte & 0x0F);
            let mut hi = u32::from(byte >> 4);
            // Add 5th bit from high-bits word
            lo |= ((high_bits >> i) & 1) << 4;
            hi |= ((high_bits >> (i + 16)) & 1) << 4;
            // Values are unsigned [0, 31], centred at 16
            dst[i * 2] = f32_to_f16((lo as f32 - 16.0) * scale);
            dst[i * 2 + 1] = f32_to_f16((hi as f32 - 16.0) * scale);
        }
    }
}

/// `Q5_1`: 24 bytes/block, 32 elements.
/// Layout: `[f16 scale (2)] [f16 min (2)] [u32 high-bits (4)] [16 bytes packed 4-bit]`
fn dequant_q5_1(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 24;
    const BLOCK_SIZE: usize = 32;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min_val = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let high_bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let dst = &mut out[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = block[8 + i];
            let mut lo = u32::from(byte & 0x0F);
            let mut hi = u32::from(byte >> 4);
            lo |= ((high_bits >> i) & 1) << 4;
            hi |= ((high_bits >> (i + 16)) & 1) << 4;
            dst[i * 2] = f32_to_f16(lo as f32 * scale + min_val);
            dst[i * 2 + 1] = f32_to_f16(hi as f32 * scale + min_val);
        }
    }
}

/// `Q8_0`: 34 bytes/block, 32 elements.
/// Layout: `[f16 scale (2)] [32 bytes of i8 weights]`
fn dequant_q8_0(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 34;
    const BLOCK_SIZE: usize = 32;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dst = &mut out[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..32 {
            let weight = block[2 + i] as i8;
            dst[i] = f32_to_f16(f32::from(weight) * scale);
        }
    }
}

// ---------------------------------------------------------------------------
// K-quant types (256 elements per block)
// ---------------------------------------------------------------------------

/// Decode the 12-byte packed 6-bit scale/min arrays used by `Q4_K` and `Q5_K`.
///
/// Matches llama.cpp's `get_scale_min_k4` exactly. The 12 bytes encode
/// 8 scales and 8 mins:
/// - For j < 4: scale[j] = q[j] & 63, min[j] = q[j+4] & 63
/// - For j >= 4: scale[j] and min[j] are assembled from bytes q[j-4],
///   q[j], and q[j+4] using 6-bit packing.
///
/// Returns `([scales; 8], [mins; 8])` as raw `u8` values.
#[inline]
fn decode_k_scales_mins(q: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    // j < 4: low 6 bits directly
    for j in 0..4 {
        scales[j] = q[j] & 0x3F;
        mins[j] = q[j + 4] & 0x3F;
    }

    // j >= 4: assembled from the upper 2 bits of q[j-4] / q[j] and
    // the low/high nibbles of q[j+4] (= q[8..11]).
    for j in 4..8 {
        scales[j] = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mins[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }

    (scales, mins)
}

/// `Q4_K`: 144 bytes/block, 256 elements.
///
/// Block layout (from `ggml-quants.c` `block_q4_K`):
/// ```text
/// [0..2)    f16 d (super-block scale)
/// [2..4)    f16 dmin (super-block min)
/// [4..16)   12 bytes: packed 6-bit sub-block scales and mins
/// [16..144) 128 bytes: 256 x 4-bit quantized weights
/// ```
fn dequant_q4_k(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 144;
    const BLOCK_SIZE: usize = 256;

    let num_blocks = out.len() / BLOCK_SIZE;
    for blk in 0..num_blocks {
        let block = &data[blk * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let (raw_scales, raw_mins) = decode_k_scales_mins(&block[4..16]);

        // Pre-scale the sub-block scales and mins.
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            scales[j] = f32::from(raw_scales[j]) * d;
            mins[j] = f32::from(raw_mins[j]) * dmin;
        }

        let qs = &block[16..144]; // 128 bytes

        // Process in pairs of sub-blocks (64 elements per iteration).
        // Each 32 bytes yields 64 weights: low nibbles → sub-block b,
        // high nibbles → sub-block b+1. Reference: gguflib gguf_q4_k_to_float.
        let mut dst_idx = blk * BLOCK_SIZE;
        let mut q_ptr = 0;
        for b in (0..8).step_by(2) {
            let sc = scales[b];
            let mn = mins[b];
            // First 32 elements: low nibbles with scale[b].
            for j in 0..32 {
                let w = f32::from(qs[q_ptr + j] & 0x0F);
                out[dst_idx] = f32_to_f16(w * sc - mn);
                dst_idx += 1;
            }
            let sc2 = scales[b + 1];
            let mn2 = mins[b + 1];
            // Next 32 elements: high nibbles with scale[b+1].
            for j in 0..32 {
                let w = f32::from(qs[q_ptr + j] >> 4);
                out[dst_idx] = f32_to_f16(w * sc2 - mn2);
                dst_idx += 1;
            }
            q_ptr += 32;
        }
    }
}

/// `Q5_K`: 176 bytes/block, 256 elements.
///
/// Block layout (from `ggml-quants.c` `block_q5_K`):
/// ```text
/// [0..2)    f16 d (super-block scale)
/// [2..4)    f16 dmin (super-block min)
/// [4..16)   12 bytes: packed 6-bit sub-block scales and mins
/// [16..48)  32 bytes: high bits (1 bit per element, 256 bits = 32 bytes)
/// [48..176) 128 bytes: 256 x 4-bit low quantized weights
/// ```
fn dequant_q5_k(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 176;
    const BLOCK_SIZE: usize = 256;

    let num_blocks = out.len() / BLOCK_SIZE;
    for blk in 0..num_blocks {
        let block = &data[blk * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let (raw_scales, raw_mins) = decode_k_scales_mins(&block[4..16]);
        let qh = &block[16..48]; // 32 bytes of high bits
        let ql = &block[48..176]; // 128 bytes of low nibbles

        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            scales[j] = f32::from(raw_scales[j]) * d;
            mins[j] = f32::from(raw_mins[j]) * dmin;
        }

        // Same pattern as Q4_K: pairs of sub-blocks, 32 bytes → 64 elements.
        // Low nibbles get scale[b], high nibbles get scale[b+1].
        // High bit from qh array adds the 5th bit.
        let mut dst_idx = blk * BLOCK_SIZE;
        let mut q_ptr = 0usize;
        for b in (0..8).step_by(2) {
            let sc = scales[b];
            let mn = mins[b];
            // First 32 elements: low nibbles
            for j in 0..32 {
                let lo4 = u32::from(ql[q_ptr + j] & 0x0F);
                let elem = dst_idx - blk * BLOCK_SIZE;
                let hb = u32::from((qh[elem / 8] >> (elem % 8)) & 1) << 4;
                let w = (lo4 | hb) as f32 * sc - mn;
                out[dst_idx] = f32_to_f16(w);
                dst_idx += 1;
            }
            let sc2 = scales[b + 1];
            let mn2 = mins[b + 1];
            // Next 32 elements: high nibbles
            for j in 0..32 {
                let hi4 = u32::from(ql[q_ptr + j] >> 4);
                let elem = dst_idx - blk * BLOCK_SIZE;
                let hb = u32::from((qh[elem / 8] >> (elem % 8)) & 1) << 4;
                let w = (hi4 | hb) as f32 * sc2 - mn2;
                out[dst_idx] = f32_to_f16(w);
                dst_idx += 1;
            }
            q_ptr += 32;
        }
    }
}

/// `Q6_K`: 210 bytes/block, 256 elements.
///
/// Block layout (from `ggml-quants.c` `block_q6_K`):
/// ```text
/// [0..128)   128 bytes: low 4 bits of 6-bit weights (256 elements, 2 per byte)
/// [128..192) 64 bytes: high 2 bits of 6-bit weights (256 elements, 4 per byte)
/// [192..208) 16 bytes: i8 sub-block scales (16 sub-blocks of 16 elements)
/// [208..210) f16 d (super-block scale)
/// ```
fn dequant_q6_k(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 210;
    const BLOCK_SIZE: usize = 256;

    let num_blocks = out.len() / BLOCK_SIZE;
    for blk in 0..num_blocks {
        let block = &data[blk * BLOCK_BYTES..][..BLOCK_BYTES];
        // Reference: gguflib gguf_q6_k_to_float
        let super_scale = f16_to_f32(u16::from_le_bytes([
            block[128 + 64 + 16],
            block[128 + 64 + 17],
        ]));

        let mut dst_idx = blk * BLOCK_SIZE;
        let mut l_ptr = 0usize; // into L (low 4 bits, 128 bytes at block[0..128])
        let mut h_ptr = 128usize; // into H (high 2 bits, 64 bytes at block[128..192])
        let mut sc_ptr = 192usize; // into scales (16 bytes at block[192..208])

        for _cluster in 0..2 {
            for j in 0..128u32 {
                // Scale for this element's sub-block (16 elements per sub-block)
                let scale = f32::from(block[sc_ptr + (j / 16) as usize] as i8) * super_scale;

                // Low 4 bits: L array, two halves interleaved
                let low4 = (block[l_ptr + (j % 64) as usize] >> ((j / 64) * 4)) & 0x0F;
                // High 2 bits: H array
                let high2 = (block[h_ptr + (j % 32) as usize] >> ((j / 32) * 2)) & 0x03;

                let q6 = (low4 | (high2 << 4)) as i8 - 32;
                out[dst_idx] = f32_to_f16(f32::from(q6) * scale);
                dst_idx += 1;
            }
            l_ptr += 64;
            h_ptr += 32;
            sc_ptr += 8;
        }
    }
}

/// `Q2_K`: 84 bytes/block, 256 elements.
///
/// Block layout (from `ggml-quants.c` `block_q2_K`):
/// ```text
/// [0..16)  16 bytes: packed 4-bit sub-block scales (16 sub-blocks)
/// [16..80) 64 bytes: 256 x 2-bit weights (4 per byte)
/// [80..82) f16 d (super-block scale)
/// [82..84) f16 dmin (super-block min)
/// ```
fn dequant_q2_k(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 84;
    const BLOCK_SIZE: usize = 256;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let sc_data = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        for (sub, &sc_byte) in sc_data.iter().enumerate().take(16) {
            let sc = f32::from(sc_byte & 0x0F) * d;
            let mn = f32::from(sc_byte >> 4) * dmin;
            let dst_off = sub * 16;
            let qs_off = sub * 4; // 16 elements, 2 bits each = 4 bytes

            for i in 0..16 {
                let byte_idx = qs_off + i / 4;
                let shift = (i % 4) * 2;
                let q = f32::from((qs[byte_idx] >> shift) & 0x03);
                out[b * BLOCK_SIZE + dst_off + i] = f32_to_f16(q * sc - mn);
            }
        }
    }
}

/// `Q3_K`: 110 bytes/block, 256 elements.
///
/// Block layout (from `ggml-quants.c` `block_q3_K`):
/// ```text
/// [0..32)    32 bytes: high bits (256 x 1 bit = 32 bytes)
/// [32..96)   64 bytes: 256 x 2-bit low weights (4 per byte)
/// [96..108)  12 bytes: packed sub-block scales
/// [108..110) f16 d (super-block scale)
/// ```
fn dequant_q3_k(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 110;
    const BLOCK_SIZE: usize = 256;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let hmask = &block[0..32]; // high bits
        let qs = &block[32..96]; // 2-bit low weights
        let sc_data = &block[96..108]; // 12 bytes packed scales
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        // Decode 16 sub-block scales from 12 bytes.
        // Each scale is a 6-bit unsigned value, centred at 32.
        let mut sc = [0i32; 16];
        for i in 0..8 {
            sc[i] = i32::from(sc_data[i] & 0x0F);
            sc[i + 8] = i32::from(sc_data[i] >> 4);
        }
        // Apply upper bits from bytes 8..11
        for i in 0..4 {
            let upper = sc_data[8 + i];
            sc[2 * i] |= i32::from(upper & 0x03) << 4;
            sc[2 * i + 1] |= i32::from((upper >> 2) & 0x03) << 4;
            sc[2 * i + 8] |= i32::from((upper >> 4) & 0x03) << 4;
            sc[2 * i + 9] |= i32::from((upper >> 6) & 0x03) << 4;
        }

        for (sub, sub_sc) in sc.iter().enumerate() {
            let sub_scale = (*sub_sc - 32) as f32 * d;
            let dst_off = sub * 16;

            for i in 0..16 {
                let elem = dst_off + i;
                // 2-bit low value
                let byte_idx = elem / 4;
                let shift = (elem % 4) * 2;
                let lo = u32::from((qs[byte_idx] >> shift) & 0x03);
                // 1-bit high value
                let hi_bit = u32::from((hmask[elem / 8] >> (elem % 8)) & 1);
                let q = lo | (hi_bit << 2);
                // 3-bit value [0, 7], centred at 4
                let val = (q as f32 - 4.0) * sub_scale;
                out[b * BLOCK_SIZE + elem] = f32_to_f16(val);
            }
        }
    }
}

/// `Q8_K`: 292 bytes/block, 256 elements.
///
/// Block layout (from `ggml-quants.c` `block_q8_K`):
/// ```text
/// [0..4)     f32 d (super-block scale)
/// [4..260)   256 bytes: i8 quantized weights
/// [260..292) 32 bytes: 16 x i16 sub-block sums (unused for dequant)
/// ```
fn dequant_q8_k(data: &[u8], out: &mut [u16]) {
    const BLOCK_BYTES: usize = 292;
    const BLOCK_SIZE: usize = 256;

    let num_blocks = out.len() / BLOCK_SIZE;
    for b in 0..num_blocks {
        let block = &data[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];

        for i in 0..256 {
            let weight = qs[i] as i8;
            out[b * BLOCK_SIZE + i] = f32_to_f16(f32::from(weight) * d);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: check that all values are within tolerance of reference.
    fn assert_close(actual: &[u16], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let a_f32 = f16_to_f32(a);
            assert!(
                (a_f32 - e).abs() <= tol,
                "element {i}: got {a_f32}, expected {e}, diff {}",
                (a_f32 - e).abs()
            );
        }
    }

    #[test]
    fn test_f32_f16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 42.0, 0.001];
        for &v in &values {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!(
                (back - v).abs() < 0.01 || (back - v).abs() / v.abs().max(1e-6) < 0.01,
                "roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_0() {
        // Build a block: scale=1.0 (f16: 0x3C00), all nibbles = 10
        // value = (10 - 8) * 1.0 = 2.0
        let mut block = [0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        for b in &mut block[2..18] {
            *b = 0xAA; // lo=10, hi=10
        }

        let mut out = vec![0u16; 32];
        dequant_q4_0(&block, &mut out);

        let expected = vec![2.0f32; 32];
        assert_close(&out, &expected, 0.01);
    }

    #[test]
    fn test_dequant_q8_0() {
        // Build a block: scale=0.5 (f16: 0x3800), all weights = 4 (as i8)
        // value = 4 * 0.5 = 2.0
        let mut block = [0u8; 34];
        block[0] = 0x00;
        block[1] = 0x38; // f16 0.5
        for b in &mut block[2..34] {
            *b = 4; // i8 = 4
        }

        let mut out = vec![0u16; 32];
        dequant_q8_0(&block, &mut out);

        let expected = vec![2.0f32; 32];
        assert_close(&out, &expected, 0.01);
    }

    #[test]
    fn test_dequant_q4_0_zeros() {
        // scale=1.0, all nibbles = 8 (value = 0)
        let mut block = [0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C;
        for b in &mut block[2..18] {
            *b = 0x88;
        }

        let mut out = vec![0u16; 32];
        dequant_q4_0(&block, &mut out);

        let expected = vec![0.0f32; 32];
        assert_close(&out, &expected, 0.001);
    }

    #[test]
    fn test_is_quantized() {
        assert!(!is_quantized(BeaconDtype::F32));
        assert!(!is_quantized(BeaconDtype::F16));
        assert!(!is_quantized(BeaconDtype::BF16));
        assert!(!is_quantized(BeaconDtype::I32));
        assert!(!is_quantized(BeaconDtype::I8));
        assert!(is_quantized(BeaconDtype::Q4_0));
        assert!(is_quantized(BeaconDtype::Q4_1));
        assert!(is_quantized(BeaconDtype::Q5_0));
        assert!(is_quantized(BeaconDtype::Q8_0));
        assert!(is_quantized(BeaconDtype::Q4K));
        assert!(is_quantized(BeaconDtype::Q5K));
        assert!(is_quantized(BeaconDtype::Q6K));
        assert!(is_quantized(BeaconDtype::Q2K));
        assert!(is_quantized(BeaconDtype::Q3K));
        assert!(is_quantized(BeaconDtype::Q8K));
    }
}
