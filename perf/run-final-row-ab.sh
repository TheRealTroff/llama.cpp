#!/bin/bash
set -euo pipefail
B=${B:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/work-elimination-20260928}
for line in ${PICK_LINES:-q4 ud}; do
    for run in a1 b1 b2 a2; do
        arm=0
        case "$run" in b*) arm=1 ;; esac
        tag="fr-perf-$line-$run"
        LINE="$line" ARM="$arm" DEPTH=3 MODE=prefill TAG="$tag" \
            bash "$B/perf/run-final-row-check.sh" > "$OUT/$tag.console.log" 2>&1
    done
done
