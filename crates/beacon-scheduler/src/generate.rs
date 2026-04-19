//! Streaming generation loop.
//!
//! Drives the forward pass, applies sampling, emits tokens via a
//! `tokio::sync::mpsc` channel, and enforces stop conditions.

use tokio::sync::mpsc;

use crate::error::SchedulerError;
use crate::params::GenerationParams;
use crate::sampling;

/// A generated token emitted by the streaming loop.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// The token ID.
    pub token_id: u32,
    /// Whether this is the last token (stop condition met).
    pub is_last: bool,
}

/// Result type for the generation callback.
///
/// The `generate_fn` closure receives a token ID and position and returns the
/// logits for the next token as a `Vec<f32>`.
pub type ForwardFn<'a> = &'a mut dyn FnMut(u32, usize) -> Result<Vec<f32>, SchedulerError>;

/// Run a streaming generation loop synchronously.
///
/// Calls `forward_fn(token, position)` for each decode step, samples the
/// next token, and sends it through `tx`. Stops when a stop token is
/// generated, `max_tokens` is reached, or the channel is closed.
///
/// This is the sync entry point — the caller can wrap it in a `tokio::task`
/// for async streaming.
#[allow(clippy::cast_possible_truncation)]
pub fn generate_stream_sync(
    initial_token: u32,
    start_position: usize,
    params: &GenerationParams,
    forward_fn: ForwardFn<'_>,
    tx: &mpsc::Sender<Result<GeneratedToken, SchedulerError>>,
) -> Result<(), SchedulerError> {
    let mut rng = rand::rng();
    let mut current_token = initial_token;
    let mut position = start_position;
    let mut previous_tokens: Vec<u32> = Vec::new();

    #[allow(clippy::explicit_counter_loop)]
    for step in 0..params.max_tokens {
        // Run forward pass.
        let mut logits = forward_fn(current_token, position)?;

        // Apply sampling pipeline.
        let next_token = sampling::sample(&mut logits, &previous_tokens, params, &mut rng);

        // Check stop condition.
        let is_last = params.stop_tokens.contains(&next_token) || step + 1 >= params.max_tokens;

        let gen_token = GeneratedToken {
            token_id: next_token,
            is_last,
        };

        // Send the token. If the receiver is dropped, stop.
        if tx.blocking_send(Ok(gen_token)).is_err() {
            return Ok(()); // Receiver dropped = cancellation.
        }

        if is_last {
            return Ok(());
        }

        previous_tokens.push(next_token);
        current_token = next_token;
        position += 1;
    }

    Ok(())
}

/// Run streaming generation asynchronously.
///
/// Returns a receiver that yields tokens as they are generated. The
/// generation runs in a background `tokio::task::spawn_blocking` because
/// the forward pass is CPU/GPU-bound and should not block the async
/// runtime.
pub fn generate_stream(
    initial_token: u32,
    start_position: usize,
    params: GenerationParams,
    mut forward_fn: Box<dyn FnMut(u32, usize) -> Result<Vec<f32>, SchedulerError> + Send>,
    buffer_size: usize,
) -> mpsc::Receiver<Result<GeneratedToken, SchedulerError>> {
    let (tx, rx) = mpsc::channel(buffer_size);

    tokio::task::spawn_blocking(move || {
        let result = generate_stream_sync(
            initial_token,
            start_position,
            &params,
            &mut *forward_fn,
            &tx,
        );

        // If the loop itself errored, send the error.
        if let Err(e) = result {
            let _ = tx.blocking_send(Err(e));
        }
    });

    rx
}
