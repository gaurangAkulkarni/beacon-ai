#!/usr/bin/env bash
# Enforces non-negotiable rule §1: shim/src/ and shim/include/ stay under 2,000
# lines of C/C++ combined. Counted across *.cpp and *.h files.
#
# Usage: scripts/check-shim-lines.sh

set -euo pipefail

LIMIT=2000

cd "$(dirname "$0")/.."

total=$(find shim/src shim/include -type f \( -name '*.cpp' -o -name '*.h' \) \
    -print0 | xargs -0 cat | wc -l | tr -d ' ')

echo "shim line count: ${total} / ${LIMIT}"

if [ "${total}" -gt "${LIMIT}" ]; then
    echo "ERROR: shim exceeds the 2,000-line cap (non-negotiable rule §1)."
    echo "Breakdown:"
    find shim/src shim/include -type f \( -name '*.cpp' -o -name '*.h' \) \
        -print0 | xargs -0 wc -l | sort -nr
    exit 1
fi
