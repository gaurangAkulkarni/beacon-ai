//! Tests for the beacon-core crate.

use beacon_format::BeaconDtype;
use beacon_mlx::Dtype;

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
