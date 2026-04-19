//! Tests for the beacon-core crate.

use beacon_format::BeaconDtype;
use beacon_mlx::Dtype;

use crate::backend::ComputeBackend;
use crate::cpu_backend::{CpuBackend, CpuStream, CpuTensor};

/// Verify that the `Engine<MlxBackend>` type is constructible (compilation test).
///
/// This does not create an actual engine (that requires a real model file),
/// but ensures the generic type machinery compiles correctly.
#[test]
fn engine_type_compiles_with_mlx_backend() {
    // The existence of this function proves that `Engine<MlxBackend>` compiles.
    fn assert_engine_is_sized(_e: &crate::Engine<crate::MlxBackend>) {}
    // We cannot call it without a real engine, but its compilation is the test.
    let _ = assert_engine_is_sized;
}

/// Verify that `Engine<CpuBackend>` type is constructible (compilation test).
#[test]
fn engine_type_compiles_with_cpu_backend() {
    fn assert_engine_is_sized(_e: &crate::Engine<crate::CpuBackend>) {}
    let _ = assert_engine_is_sized;
}

/// Verify that all `BeaconDtype` variants map to the corresponding `Dtype`.
#[test]
fn dtype_conversion_round_trip() {
    let pairs = [
        (BeaconDtype::F32, Dtype::F32),
        (BeaconDtype::F16, Dtype::F16),
        (BeaconDtype::BF16, Dtype::BF16),
        (BeaconDtype::I32, Dtype::I32),
        (BeaconDtype::I8, Dtype::I8),
        (BeaconDtype::Q4_0, Dtype::Q4_0),
        (BeaconDtype::Q4K, Dtype::Q4K),
        (BeaconDtype::Q5K, Dtype::Q5K),
        (BeaconDtype::Q6K, Dtype::Q6K),
        (BeaconDtype::Q8_0, Dtype::Q8_0),
    ];
    for (bd, expected) in pairs {
        assert_eq!(
            crate::beacon_dtype_to_mlx(bd),
            expected,
            "mismatch for {bd:?}"
        );
    }
}

/// Verify that `ComputeBackend` is usable as a generic bound.
#[test]
fn compute_backend_trait_is_usable() {
    fn assert_backend_generic<B: crate::ComputeBackend>() {}
    assert_backend_generic::<crate::MlxBackend>();
    assert_backend_generic::<crate::CpuBackend>();
}

// ── CPU backend unit tests ──────────────────────────────────────────────

/// Helper to create a `CpuTensor` from flat data and shape.
fn cpu_tensor(data: Vec<f32>, shape: Vec<i64>) -> CpuTensor {
    CpuTensor { data, shape }
}

#[test]
fn cpu_matmul_2x2() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // A @ B = [[19, 22], [43, 50]]
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cpu_tensor(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = backend.matmul(&stream, &a, &b).unwrap();

    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn cpu_matmul_non_square() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // A = [[1, 2, 3]], B = [[4], [5], [6]] => [[32]]
    let a = cpu_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = cpu_tensor(vec![4.0, 5.0, 6.0], vec![3, 1]);
    let c = backend.matmul(&stream, &a, &b).unwrap();

    assert_eq!(c.shape, vec![1, 1]);
    assert!((c.data[0] - 32.0).abs() < 1e-6);
}

#[test]
fn cpu_rms_norm() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let x = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let weight = cpu_tensor(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let result = backend.rms_norm(&stream, &x, &weight, 1e-5).unwrap();

    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ~ 2.7386
    // inv_rms ~ 0.3651
    let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
    let inv = 1.0 / rms;
    #[allow(clippy::cast_precision_loss)]
    for (i, &v) in result.data.iter().enumerate() {
        let expected = (i as f32 + 1.0) * inv;
        assert!(
            (v - expected).abs() < 1e-5,
            "rms_norm mismatch at {i}: {v} vs {expected}"
        );
    }
}

#[test]
fn cpu_rms_norm_2d() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // 2 rows of dim=2: normalize each row independently.
    let x = cpu_tensor(vec![3.0, 4.0, 1.0, 0.0], vec![2, 2]);
    let weight = cpu_tensor(vec![1.0, 1.0], vec![2]);
    let result = backend.rms_norm(&stream, &x, &weight, 1e-5).unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data.len(), 4);
}

#[test]
fn cpu_silu() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let x = cpu_tensor(vec![0.0, 1.0, -1.0], vec![3]);
    let result = backend.silu(&stream, &x).unwrap();

    // silu(0) = 0, silu(1) = 1 * sigmoid(1) ~ 0.7311, silu(-1) ~ -0.2689
    assert!((result.data[0]).abs() < 1e-6);
    assert!((result.data[1] - 0.7311).abs() < 0.001);
    assert!((result.data[2] - (-0.2689)).abs() < 0.001);
}

#[test]
fn cpu_softmax() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let x = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = backend.softmax(&stream, &x, -1).unwrap();

    // Softmax values should sum to 1.
    let sum: f32 = result.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum: {sum}");
    // Values should be monotonically increasing.
    assert!(result.data[0] < result.data[1]);
    assert!(result.data[1] < result.data[2]);
}

#[test]
fn cpu_add() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let a = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b = cpu_tensor(vec![4.0, 5.0, 6.0], vec![3]);
    let result = backend.add(&stream, &a, &b).unwrap();
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn cpu_mul() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let a = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b = cpu_tensor(vec![4.0, 5.0, 6.0], vec![3]);
    let result = backend.mul(&stream, &a, &b).unwrap();
    assert_eq!(result.data, vec![4.0, 10.0, 18.0]);
}

#[test]
fn cpu_reshape() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let x = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = backend.reshape(&stream, &x, &[3, 2]).unwrap();
    assert_eq!(result.shape, vec![3, 2]);
    assert_eq!(result.data, x.data);
}

#[test]
fn cpu_reshape_mismatch() {
    let backend = CpuBackend;
    let stream = CpuStream;

    let x = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = backend.reshape(&stream, &x, &[2, 2]);
    assert!(result.is_err());
}

#[test]
fn cpu_embedding() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // vocab=3, dim=2: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    let weight = cpu_tensor(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2]);
    // Look up indices [2, 0]
    let indices = cpu_tensor(vec![2.0, 0.0], vec![2]);
    let result = backend.embedding(&stream, &weight, &indices).unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert!((result.data[0] - 0.5).abs() < 1e-6);
    assert!((result.data[1] - 0.6).abs() < 1e-6);
    assert!((result.data[2] - 0.1).abs() < 1e-6);
    assert!((result.data[3] - 0.2).abs() < 1e-6);
}

#[test]
fn cpu_attention_single_head() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // batch=1, n_heads=1, seq=1, head_dim=2
    let q = cpu_tensor(vec![1.0, 0.0], vec![1, 1, 1, 2]);
    let k = cpu_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
    let v = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

    let scale = 1.0;
    let result = backend.attention(&stream, &q, &k, &v, None, scale).unwrap();

    assert_eq!(result.shape, vec![1, 1, 1, 2]);
    // Q=[1,0], K=[[1,0],[0,1]] => scores = [1, 0] * scale => softmax([1,0])
    // softmax([1,0]) ~ [0.7311, 0.2689]
    // out = 0.7311 * [1,2] + 0.2689 * [3,4] = [1.538, 2.538]
    let s0 = 1.0f32.exp() / (1.0f32.exp() + 1.0);
    let s1 = 1.0 - s0;
    let expected_0 = s0 * 1.0 + s1 * 3.0;
    let expected_1 = s0 * 2.0 + s1 * 4.0;
    assert!(
        (result.data[0] - expected_0).abs() < 1e-4,
        "got {} expected {}",
        result.data[0],
        expected_0
    );
    assert!(
        (result.data[1] - expected_1).abs() < 1e-4,
        "got {} expected {}",
        result.data[1],
        expected_1
    );
}

#[test]
fn cpu_attention_gqa() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // GQA: 2 query heads, 1 KV head. Both query heads should use the same K/V.
    // batch=1, seq_q=1, seq_kv=1, head_dim=2
    let q = cpu_tensor(
        vec![1.0, 0.0, 0.0, 1.0], // head 0: [1,0], head 1: [0,1]
        vec![1, 2, 1, 2],
    );
    let k = cpu_tensor(vec![1.0, 1.0], vec![1, 1, 1, 2]);
    let v = cpu_tensor(vec![2.0, 3.0], vec![1, 1, 1, 2]);

    let result = backend.attention(&stream, &q, &k, &v, None, 1.0).unwrap();

    assert_eq!(result.shape, vec![1, 2, 1, 2]);
    // With single KV token, softmax([score]) = [1.0] for both heads.
    // Both heads' outputs should be the V vector [2.0, 3.0].
    assert!((result.data[0] - 2.0).abs() < 1e-5);
    assert!((result.data[1] - 3.0).abs() < 1e-5);
    assert!((result.data[2] - 2.0).abs() < 1e-5);
    assert!((result.data[3] - 3.0).abs() < 1e-5);
}

#[test]
fn cpu_create_token_tensor() {
    let backend = CpuBackend;
    let result = backend.create_token_tensor(&[0, 1, 42]).unwrap();
    assert_eq!(result.shape, vec![3]);
    // Token IDs stored as f32 cast — verify values.
    assert!((result.data[0] - 0.0).abs() < 1e-6);
    assert!((result.data[1] - 1.0).abs() < 1e-6);
    assert!((result.data[2] - 42.0).abs() < 1e-6);
}

#[test]
fn cpu_kv_cache_update() {
    let backend = CpuBackend;
    let stream = CpuStream;

    // cache: [4, 1, 2] (max_ctx=4, 1 kv_head, head_dim=2)
    let cache_k = cpu_tensor(vec![0.0; 8], vec![4, 1, 2]);
    let cache_v = cpu_tensor(vec![0.0; 8], vec![4, 1, 2]);

    // new_k/v: [1, 1, 1, 2] (batch=1, 1 head, seq=1, dim=2)
    let new_k = cpu_tensor(vec![1.0, 2.0], vec![1, 1, 1, 2]);
    let new_v = cpu_tensor(vec![3.0, 4.0], vec![1, 1, 1, 2]);

    let (k_out, v_out) = backend
        .kv_cache_update(&stream, &cache_k, &cache_v, &new_k, &new_v, 0)
        .unwrap();

    // Should return [1, 1, 1, 2] with the new data at position 0.
    assert_eq!(k_out.shape, vec![1, 1, 1, 2]);
    assert_eq!(k_out.data, vec![1.0, 2.0]);
    assert_eq!(v_out.data, vec![3.0, 4.0]);
}

#[test]
fn cpu_eval_is_noop() {
    let backend = CpuBackend;
    let stream = CpuStream;
    let t = cpu_tensor(vec![1.0], vec![1]);
    // Should not error.
    backend.eval(&t, &stream).unwrap();
}

#[test]
fn cpu_read_f32() {
    let backend = CpuBackend;
    let t = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let v = backend.read_f32(&t, 2).unwrap();
    assert_eq!(v, vec![1.0, 2.0]);
}

/// Integration test: load a real `.beacon` model and run a forward pass.
///
/// Gated behind the `BEACON_TEST_MODEL` environment variable which should
/// point to a `.beacon` file.
#[test]
#[ignore = "requires BEACON_TEST_MODEL env var pointing to a .beacon file"]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn real_model_forward_pass() {
    use std::path::Path;
    use std::sync::Arc;

    let model_path =
        std::env::var("BEACON_TEST_MODEL").expect("set BEACON_TEST_MODEL to a .beacon file path");

    let beacon = beacon_format::BeaconFile::open(Path::new(&model_path))
        .expect("failed to open .beacon file");

    let ctx = beacon_mlx::MlxContext::new().expect("failed to create MLX context");
    let backend = crate::MlxBackend::new(Arc::clone(&ctx));

    let mut engine = crate::Engine::load(&beacon, backend).expect("failed to load model");

    // Run a simple forward pass with a single token (token 1 = common BOS).
    let logits = engine.forward(&[1], 0).expect("forward pass failed");
    let shape = logits.shape();
    assert_eq!(shape.len(), 2, "logits should be 2D: [seq, vocab]");
    assert_eq!(shape[0], 1, "single token input should produce 1 row");
    assert_eq!(
        shape[1] as usize, engine.config.vocab_size,
        "logits width should match vocab size"
    );

    // Read logits and verify they are finite.
    let values = logits
        .read_f32(engine.config.vocab_size)
        .expect("failed to read logits");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "all logits should be finite"
    );

    // Greedy decode one token.
    let next = engine.generate_next_token(1, 1).expect("generate failed");
    assert!(
        (next as usize) < engine.config.vocab_size,
        "generated token should be within vocab"
    );
}
