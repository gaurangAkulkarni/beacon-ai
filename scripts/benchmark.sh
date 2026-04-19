#!/usr/bin/env bash
# scripts/benchmark.sh — Run Criterion benchmarks and print a summary.
#
# Usage:
#   ./scripts/benchmark.sh            # Run all kernel benchmarks
#   ./scripts/benchmark.sh --quick    # Compile-check only (no full run)
#
# Results are saved to target/criterion/ with HTML reports.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Beacon Kernel Benchmarks ==="
echo "Platform: $(uname -sm)"
echo "Rust:     $(rustc --version)"
echo ""

if [[ "${1:-}" == "--quick" ]]; then
    echo "--- Quick mode: compile-check only ---"
    cargo bench -p beacon-kernels -- --test 2>&1
    echo ""
    echo "Benchmarks compile successfully."
    exit 0
fi

echo "Running full benchmarks (this may take a few minutes)..."
echo ""
cargo bench -p beacon-kernels 2>&1
echo ""
echo "=== Done ==="
echo "HTML reports saved to: target/criterion/"
echo "Open target/criterion/report/index.html in a browser to view."
