//! Criterion benchmarks for beacon-kernels CPU ops.
//!
//! Covers the core transformer operations at representative LLM sizes
//! (`hidden_size=4096`, single-token decode). These benchmarks run on every PR
//! for regression detection per architecture section 13.4.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use beacon_kernels::{dispatch, ops, q4};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a valid `Q4_0` weight buffer for `rows x cols` elements.
///
/// Each block has scale = 1.0 (f16 0x3C00) and all nibbles set to 0x88
/// (value 0 after the -8 bias). The data is not random but exercises the
/// full dequantization path, which is what matters for throughput.
fn make_q4_weights(rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % q4::Q4_0_BLOCK_SIZE, 0);
    let blocks_per_row = cols / q4::Q4_0_BLOCK_SIZE;
    let total_blocks = rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * q4::Q4_0_BLOCK_BYTES];

    for b in 0..total_blocks {
        let off = b * q4::Q4_0_BLOCK_BYTES;
        // f16 1.0 = 0x3C00 little-endian
        buf[off] = 0x00;
        buf[off + 1] = 0x3C;
        // nibbles: 0x88 => both lo and hi = 8 => value 0
        for byte in &mut buf[off + 2..off + q4::Q4_0_BLOCK_BYTES] {
            *byte = 0x88;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

#[allow(clippy::many_single_char_names)]
fn bench_matmul_f32(c: &mut Criterion) {
    // Single-token decode: m=1, k=4096 (hidden_size), n=4096.
    let rows = 1;
    let cols_inner = 4096;
    let cols_out = 4096;
    let mat_a = vec![0.1_f32; rows * cols_inner];
    let mat_b = vec![0.2_f32; cols_inner * cols_out];
    let mut out = vec![0.0_f32; rows * cols_out];

    c.bench_function("matmul_f32_1x4096x4096", |bench| {
        bench.iter(|| {
            ops::matmul_f32(
                black_box(&mat_a),
                black_box(&mat_b),
                &mut out,
                rows,
                cols_inner,
                cols_out,
            );
        });
    });

    // Smaller size for quick sanity: 512x512.
    let rows_s = 1;
    let inner_s = 512;
    let out_s = 512;
    let lhs_small = vec![0.1_f32; rows_s * inner_s];
    let rhs_small = vec![0.2_f32; inner_s * out_s];
    let mut out_small = vec![0.0_f32; rows_s * out_s];

    c.bench_function("matmul_f32_1x512x512", |bench| {
        bench.iter(|| {
            ops::matmul_f32(
                black_box(&lhs_small),
                black_box(&rhs_small),
                &mut out_small,
                rows_s,
                inner_s,
                out_s,
            );
        });
    });
}

fn bench_q4_matmul(c: &mut Criterion) {
    // Single-token decode with Q4_0 weights: m output rows, k=4096 input dim.
    let m = 4096; // output dimension (e.g. projecting to hidden_size)
    let k = 4096; // input dimension (hidden_size, must be multiple of 32)
    let x = vec![0.1_f32; k];
    let w_q4 = make_q4_weights(m, k);
    let mut out = vec![0.0_f32; m];

    c.bench_function("q4_matmul_4096x4096_dispatched", |bench| {
        bench.iter(|| {
            dispatch::q4_matmul_f32(black_box(&w_q4), black_box(&x), &mut out, m, k);
        });
    });

    // Scalar-only for comparison.
    c.bench_function("q4_matmul_4096x4096_scalar", |bench| {
        bench.iter(|| {
            ops::q4_matmul_f32(black_box(&w_q4), black_box(&x), &mut out, m, k);
        });
    });
}

fn bench_rms_norm(c: &mut Criterion) {
    let n = 4096; // hidden_size
    let x = vec![0.3_f32; n];
    let weight = vec![1.0_f32; n];
    let mut out = vec![0.0_f32; n];
    let eps = 1e-5_f32;

    c.bench_function("rms_norm_4096", |bench| {
        bench.iter(|| {
            ops::rms_norm(black_box(&x), black_box(&weight), &mut out, eps);
        });
    });
}

fn bench_silu(c: &mut Criterion) {
    let n = 4096; // hidden_size
    let original = vec![0.5_f32; n];
    let mut x = original.clone();

    c.bench_function("silu_inplace_4096", |bench| {
        bench.iter(|| {
            x.copy_from_slice(&original);
            ops::silu_inplace(black_box(&mut x));
        });
    });

    // Larger: intermediate_size for FFN gate output.
    let n_large = 11008; // typical LLaMA 7B intermediate_size
    let original_large = vec![0.5_f32; n_large];
    let mut x_large = original_large.clone();

    c.bench_function("silu_inplace_11008", |bench| {
        bench.iter(|| {
            x_large.copy_from_slice(&original_large);
            ops::silu_inplace(black_box(&mut x_large));
        });
    });
}

fn bench_softmax(c: &mut Criterion) {
    // Softmax over attention scores for one head: seq_len logits.
    let seq_len = 2048;
    let original = vec![0.1_f32; seq_len];
    let mut x = original.clone();

    c.bench_function("softmax_inplace_2048", |bench| {
        bench.iter(|| {
            x.copy_from_slice(&original);
            ops::softmax_inplace(black_box(&mut x));
        });
    });

    // Shorter sequence.
    let seq_short = 512;
    let original_short = vec![0.1_f32; seq_short];
    let mut x_short = original_short.clone();

    c.bench_function("softmax_inplace_512", |bench| {
        bench.iter(|| {
            x_short.copy_from_slice(&original_short);
            ops::softmax_inplace(black_box(&mut x_short));
        });
    });
}

fn bench_rope(c: &mut Criterion) {
    let dim = 128; // head_dim
    let seq_len = 1;
    let original = vec![0.5_f32; seq_len * dim];
    let mut x = original.clone();

    c.bench_function("rope_inplace_1x128", |bench| {
        bench.iter(|| {
            x.copy_from_slice(&original);
            ops::rope_inplace(black_box(&mut x), seq_len, dim, 0, 10000.0);
        });
    });
}

fn bench_elementwise(c: &mut Criterion) {
    let n = 4096;
    let a = vec![0.5_f32; n];
    let b = vec![0.3_f32; n];
    let mut out = vec![0.0_f32; n];

    c.bench_function("add_4096", |bench| {
        bench.iter(|| {
            ops::add(black_box(&a), black_box(&b), &mut out);
        });
    });

    c.bench_function("mul_4096", |bench| {
        bench.iter(|| {
            ops::mul(black_box(&a), black_box(&b), &mut out);
        });
    });
}

criterion_group!(
    benches,
    bench_matmul_f32,
    bench_q4_matmul,
    bench_rms_norm,
    bench_silu,
    bench_softmax,
    bench_rope,
    bench_elementwise,
);
criterion_main!(benches);
