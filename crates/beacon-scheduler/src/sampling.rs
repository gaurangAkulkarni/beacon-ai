//! Sampling over f32 logits.
//!
//! Implements the full sampling pipeline per architecture doc §11.2:
//! repeat penalty → temperature → top-k → top-p → min-p → multinomial.

use rand::Rng;

use crate::params::GenerationParams;

/// Apply the full sampling pipeline to logits and return the selected token ID.
///
/// `logits` is a mutable slice of vocab-sized f32 logits. `previous_tokens`
/// is the list of already-generated tokens (for repeat penalty).
#[allow(clippy::cast_possible_truncation)]
pub fn sample(
    logits: &mut [f32],
    previous_tokens: &[u32],
    params: &GenerationParams,
    rng: &mut impl Rng,
) -> u32 {
    // 1. Repeat penalty.
    if (params.repeat_penalty - 1.0).abs() > f32::EPSILON {
        apply_repeat_penalty(logits, previous_tokens, params.repeat_penalty);
    }

    // 2. Temperature.
    if params.temperature <= f32::EPSILON {
        // Greedy: return argmax.
        return argmax(logits) as u32;
    }
    if (params.temperature - 1.0).abs() > f32::EPSILON {
        apply_temperature(logits, params.temperature);
    }

    // 3. Top-k.
    if let Some(k) = params.top_k {
        apply_top_k(logits, k);
    }

    // 4. Top-p (nucleus).
    if let Some(p) = params.top_p {
        if p < 1.0 {
            apply_top_p(logits, p);
        }
    }

    // 5. Min-p.
    if let Some(min_p) = params.min_p {
        if min_p > 0.0 {
            apply_min_p(logits, min_p);
        }
    }

    // 6. Softmax + multinomial sample.
    softmax_inplace(logits);
    multinomial_sample(logits, rng) as u32
}

/// Greedy argmax over logits.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/// Penalise tokens that have already appeared.
fn apply_repeat_penalty(logits: &mut [f32], previous_tokens: &[u32], penalty: f32) {
    for &tok in previous_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Divide logits by temperature.
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
}

/// Zero out all logits except the top-k highest.
fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }

    // Find the k-th largest value.
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k];

    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Zero out tokens whose cumulative probability exceeds `p` (nucleus sampling).
fn apply_top_p(logits: &mut [f32], p: f32) {
    // Compute softmax to get probabilities.
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);

    // Sort indices by probability (descending).
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the cutoff index where cumulative probability exceeds p.
    let mut cumulative = 0.0f32;
    let mut cutoff_idx = indices.len();
    for (i, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            cutoff_idx = i + 1; // keep at least this many
            break;
        }
    }

    // Zero out everything below the cutoff.
    for &idx in &indices[cutoff_idx..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Zero out tokens whose probability is less than `min_p * max_prob`.
fn apply_min_p(logits: &mut [f32], min_p: f32) {
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);

    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;

    for (i, &prob) in probs.iter().enumerate() {
        if prob < threshold {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// In-place softmax.
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// Sample an index from a probability distribution.
fn multinomial_sample(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.random();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i;
        }
    }
    // Fallback to last token (rounding).
    probs.len().saturating_sub(1)
}
