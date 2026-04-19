//! Integration tests for beacon-mlx.
//!
//! These tests exercise the full path: Rust → C ABI → MLX → result readback.
//! They require macOS ARM64 with Metal; on other platforms they are
//! compile-only stubs.
//!
//! **Thread safety:** MLX's Metal backend has global state that is not safe to
//! access concurrently from multiple OS threads creating separate contexts.
//! All tests acquire a shared mutex so `cargo test` (which runs tests in
//! parallel threads) does not trigger undefined behaviour. In production,
//! a single `MlxContext` is created per process per the architecture spec.

use std::sync::Mutex;

use crate::{ops, Dtype, MlxContext, MlxTensor};

/// Serialise all MLX tests to avoid concurrent Metal device access.
static MLX_LOCK: Mutex<()> = Mutex::new(());

/// Step 3 success criterion: create MLX tensors, perform matmul, verify result.
///
/// We compute:
///   A (2x3) = [[1, 2, 3],
///              [4, 5, 6]]
///   B (3x2) = [[7, 8],
///              [9, 10],
///              [11, 12]]
///   C = A @ B = [[58, 64],
///                [139, 154]]
#[test]
fn matmul_correctness() {
    let _lock = MLX_LOCK.lock().unwrap();

    let ctx = MlxContext::new().expect("failed to create MlxContext");
    let stream = ctx.new_stream().expect("failed to create MlxStream");

    // Use zeros to verify lifecycle works.
    let z = MlxTensor::zeros(ctx.clone(), &[2, 3], Dtype::F32).expect("zeros failed");
    assert_eq!(z.shape(), vec![2, 3]);
    assert_eq!(z.dtype(), Dtype::F32);

    // For the actual matmul test, create tensors from mmap'd buffers.
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = tensor_from_f32_slice(&ctx, &a_data, &[2, 3]);
    let b = tensor_from_f32_slice(&ctx, &b_data, &[3, 2]);

    let c = ops::matmul(&stream, &a, &b).expect("matmul failed");

    assert_eq!(c.shape(), vec![2, 2]);

    let result = c.read_f32(4).expect("read_f32 failed");
    let expected = [58.0, 64.0, 139.0, 154.0];

    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "matmul result mismatch: got {got}, want {want}"
        );
    }
}

/// Test that eval + `read_f32` on a zeros tensor returns all zeros.
#[test]
fn zeros_readback() {
    let _lock = MLX_LOCK.lock().unwrap();

    let ctx = MlxContext::new().expect("failed to create MlxContext");
    let stream = ctx.new_stream().expect("failed to create MlxStream");

    let z = MlxTensor::zeros(ctx, &[3, 4], Dtype::F32).expect("zeros failed");
    z.eval(&stream).expect("eval failed");

    let data = z.read_f32(12).expect("read_f32 failed");
    assert_eq!(data.len(), 12);
    assert!(data.iter().all(|&v| v == 0.0));
}

/// Test element-wise add.
#[test]
fn add_correctness() {
    let _lock = MLX_LOCK.lock().unwrap();

    let ctx = MlxContext::new().expect("failed to create MlxContext");
    let stream = ctx.new_stream().expect("failed to create MlxStream");

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

    let a = tensor_from_f32_slice(&ctx, &a_data, &[2, 2]);
    let b = tensor_from_f32_slice(&ctx, &b_data, &[2, 2]);

    let c = ops::add(&stream, &a, &b).expect("add failed");
    let result = c.read_f32(4).expect("read_f32 failed");

    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

/// Test `SiLU` activation.
#[test]
fn silu_correctness() {
    let _lock = MLX_LOCK.lock().unwrap();

    let ctx = MlxContext::new().expect("failed to create MlxContext");
    let stream = ctx.new_stream().expect("failed to create MlxStream");

    let data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0];
    let x = tensor_from_f32_slice(&ctx, &data, &[4]);

    let y = ops::silu(&stream, &x).expect("silu failed");
    let result = y.read_f32(4).expect("read_f32 failed");

    // SiLU(x) = x * sigmoid(x)
    for (i, &x_val) in data.iter().enumerate() {
        let expected = x_val * (1.0 / (1.0 + (-x_val).exp()));
        assert!(
            (result[i] - expected).abs() < 1e-4,
            "silu mismatch at {i}: got {}, want {expected}",
            result[i]
        );
    }
}

// --- Helper ------------------------------------------------------------------

/// Create a tensor from an f32 slice by writing data to a temporary file and
/// mmap-ing it, exercising the real `MlxTensor::from_mmap` production path.
fn tensor_from_f32_slice(
    ctx: &std::sync::Arc<crate::MlxContext>,
    data: &[f32],
    shape: &[i64],
) -> MlxTensor {
    use std::sync::Arc;

    let byte_len = std::mem::size_of_val(data);

    let tmp = tempfile::tempfile().expect("failed to create tempfile");
    tmp.set_len(byte_len as u64)
        .expect("failed to set tempfile length");
    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&tmp) }.expect("mmap_mut failed");
    let src_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
    mmap[..byte_len].copy_from_slice(src_bytes);
    mmap.flush().expect("flush failed");

    let mmap = mmap.make_read_only().expect("make_read_only failed");
    let mmap = Arc::new(mmap);

    MlxTensor::from_mmap(Arc::clone(ctx), mmap, 0, shape, Dtype::F32).expect("from_mmap failed")
}
