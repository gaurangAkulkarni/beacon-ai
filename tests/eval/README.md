# Beacon Quality Eval Harness

Quality evaluation tests for the Beacon inference engine, as specified in
architecture section 13.3.

## Eval suite

| Eval | Description | Gate |
|---|---|---|
| Logit-level exactness | First-token logits at T=0 match HuggingFace reference to 3 decimal places | `BEACON_TEST_MODEL` |
| MMLU subset | 1,000 questions across 10 categories | `BEACON_TEST_EVAL` |
| HumanEval subset | 50 coding problems | `BEACON_TEST_EVAL` |
| Wiki perplexity | Perplexity on held-out Wikipedia slice | `BEACON_TEST_EVAL` |

## How to run

All eval tests require real model weights and are gated behind environment
variables. They are implemented as `#[ignore]` tests in the relevant crates.

### Logit-level check

Requires a downloaded model (GGUF or .beacon format):

```bash
BEACON_TEST_MODEL=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
    cargo test -p beacon-core -- --ignored logit
```

### Full eval suite

Requires model weights and eval datasets:

```bash
BEACON_TEST_EVAL=1 \
BEACON_TEST_MODEL=/path/to/model.gguf \
BEACON_EVAL_MMLU=/path/to/mmlu/ \
BEACON_EVAL_HUMANEVAL=/path/to/humaneval.jsonl \
BEACON_EVAL_WIKI=/path/to/wiki_holdout.txt \
    cargo test -p beacon-core -- --ignored eval
```

### Criterion benchmarks (kernel-level performance)

```bash
# Full benchmark run
./scripts/benchmark.sh

# Compile-check only (CI)
./scripts/benchmark.sh --quick

# Direct cargo invocation
cargo bench -p beacon-kernels
```

## Success criteria (from README)

- Beacon within 0.5% of HuggingFace reference on MMLU for each supported
  model.
- Logits match HuggingFace reference to 3 decimal places at temperature 0.

## Adding new evals

1. Add `#[ignore]` tests in the relevant crate (typically `beacon-core`),
   gated behind an environment variable.
2. Use `BEACON_TEST_EVAL` as the general gate, with specific dataset paths
   in dedicated env vars.
3. Document the new eval in this README.
