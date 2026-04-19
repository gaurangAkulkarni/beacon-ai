//! Tests for sampling and generation.

use crate::params::GenerationParams;
use crate::sampling;

// ---------------------------------------------------------------------------
// Sampling tests
// ---------------------------------------------------------------------------

#[test]
fn greedy_sampling() {
    let mut logits = vec![1.0, 3.0, 2.0, 0.5];
    let params = GenerationParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut rng = rand::rng();
    let token = sampling::sample(&mut logits, &[], &params, &mut rng);
    assert_eq!(token, 1, "greedy should pick index 1 (highest logit)");
}

#[test]
fn argmax_basic() {
    assert_eq!(sampling::argmax(&[1.0, 5.0, 3.0]), 1);
    assert_eq!(sampling::argmax(&[9.0, 1.0, 2.0]), 0);
    assert_eq!(sampling::argmax(&[0.0, 0.0, 1.0]), 2);
}

#[test]
fn temperature_sampling() {
    // With very low temperature, should behave like greedy.
    let mut logits = vec![1.0, 10.0, 2.0];
    let params = GenerationParams {
        temperature: 0.01,
        ..Default::default()
    };
    let mut rng = rand::rng();
    let token = sampling::sample(&mut logits, &[], &params, &mut rng);
    assert_eq!(token, 1, "very low temperature should pick highest logit");
}

#[test]
fn repeat_penalty() {
    // Token 1 was previously generated; with penalty > 1, it should be
    // penalised. Token 0 has higher logit but token 1 was the max.
    let mut logits = vec![4.9, 5.0, 1.0];
    let previous = vec![1u32]; // token 1 was generated before
    let params = GenerationParams {
        temperature: 0.0,
        repeat_penalty: 2.0,
        ..Default::default()
    };
    let mut rng = rand::rng();
    let token = sampling::sample(&mut logits, &previous, &params, &mut rng);
    // Token 1's logit (5.0) should be divided by 2.0 → 2.5, making token 0 (4.9) win.
    assert_eq!(token, 0, "repeat penalty should suppress token 1");
}

#[test]
fn top_k_sampling() {
    // With top_k=2 and temperature=1.0, only top 2 tokens are candidates.
    let mut logits = vec![1.0, 10.0, 5.0, 0.1];
    let params = GenerationParams {
        temperature: 0.001, // near-greedy to make test deterministic
        top_k: Some(2),
        ..Default::default()
    };
    let mut rng = rand::rng();
    let token = sampling::sample(&mut logits, &[], &params, &mut rng);
    assert!(
        token == 1 || token == 2,
        "top_k=2 should only allow tokens 1 and 2, got {token}"
    );
}

#[test]
fn default_params_are_greedy() {
    let params = GenerationParams::default();
    assert!(
        params.temperature == 0.0,
        "default should be greedy (temperature=0)"
    );
    assert_eq!(params.max_tokens, 512);
    assert!((params.repeat_penalty - 1.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Streaming generation tests
// ---------------------------------------------------------------------------

#[test]
fn generate_stream_sync_stops_on_stop_token() {
    let params = GenerationParams {
        temperature: 0.0,
        max_tokens: 100,
        stop_tokens: vec![99],
        ..Default::default()
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    // Mock forward function: always returns logits where token 99 wins.
    let mut forward_fn = |_token: u32, _pos: usize| -> Result<Vec<f32>, crate::SchedulerError> {
        let mut logits = vec![0.0f32; 100];
        logits[99] = 10.0; // token 99 is the winner
        Ok(logits)
    };

    crate::generate_stream_sync(0, 0, &params, &mut forward_fn, &tx).unwrap();

    // Should have received exactly 1 token (the stop token, with is_last=true).
    let token = rx.try_recv().unwrap().unwrap();
    assert_eq!(token.token_id, 99);
    assert!(token.is_last);

    // No more tokens.
    assert!(rx.try_recv().is_err());
}

#[test]
fn generate_stream_sync_respects_max_tokens() {
    let params = GenerationParams {
        temperature: 0.0,
        max_tokens: 3,
        stop_tokens: vec![], // no stop tokens
        ..Default::default()
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    // Mock: always returns logits where token 5 wins.
    let mut forward_fn = |_token: u32, _pos: usize| -> Result<Vec<f32>, crate::SchedulerError> {
        let mut logits = vec![0.0f32; 10];
        logits[5] = 10.0;
        Ok(logits)
    };

    crate::generate_stream_sync(0, 0, &params, &mut forward_fn, &tx).unwrap();

    let mut count = 0;
    while let Ok(result) = rx.try_recv() {
        let token = result.unwrap();
        assert_eq!(token.token_id, 5);
        count += 1;
        if token.is_last {
            break;
        }
    }
    assert_eq!(count, 3, "should generate exactly max_tokens=3 tokens");
}

#[test]
fn generate_stream_sync_increments_position() {
    let params = GenerationParams {
        temperature: 0.0,
        max_tokens: 3,
        stop_tokens: vec![],
        ..Default::default()
    };

    let (tx, _rx) = tokio::sync::mpsc::channel(32);

    let mut positions_seen = Vec::new();
    let mut forward_fn = |_token: u32, pos: usize| -> Result<Vec<f32>, crate::SchedulerError> {
        positions_seen.push(pos);
        let mut logits = vec![0.0f32; 10];
        logits[1] = 10.0;
        Ok(logits)
    };

    crate::generate_stream_sync(0, 5, &params, &mut forward_fn, &tx).unwrap();

    assert_eq!(
        positions_seen,
        vec![5, 6, 7],
        "positions should increment from start_position"
    );
}
