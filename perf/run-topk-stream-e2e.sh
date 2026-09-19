#!/bin/bash
# perf/run-topk-stream-e2e.sh - e2e gate for the streaming top-k kernel (GGML_TOPK_STREAM=1, 2026-09-19,
# owner released the 2026-08-28 hold: "Go ahead"). The drafter's TOP_K [248320, width] -> 16 goes from the
# block-bitonic + 8-dispatch merge ladder (0.84 ms at width 4, 1.63 at width 8, test-backend-ops perf) to a
# strip scan + one merge (0.067 / 0.116 ms). Fixed depth pins the width: depth 3 = the picks' width 4,
# depth 7 = the controller's width 8. Arms interleaved x2 per (line, depth) through run-w8-decomp.sh
# (anchor step only = -lv 3 spec-prof round split), the pick env of each line, chat benchprompt, 300 tokens.
#   B=<tree with the kernel> LINES="ud q4" DEPTHS="3 7" REPS=2 perf/run-topk-stream-e2e.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-prod}
export B
LINES=${LINES:-ud q4}
DEPTHS=${DEPTHS:-3 7}
REPS=${REPS:-2}
TAG=${TAG:-topk-$(date +%m%d-%H%M)}
echo "=== topk stream e2e: $TAG  B=$B ($(git -C "$B" rev-parse --short HEAD)) lines [$LINES] depths [$DEPTHS] reps $REPS  $(date '+%F %T') ==="
for line in $LINES; do
  for depth in $DEPTHS; do
    for r in $(seq 1 "$REPS"); do
      TAG=$TAG-base-r$r   LINES=$line DEPTHS=$depth STEPS=anchor EXTRA_ENV=""                  "$B/perf/run-w8-decomp.sh" 2>&1 | grep -E '^\s+\[' | sed "s/^/base   r$r /"
      TAG=$TAG-stream-r$r LINES=$line DEPTHS=$depth STEPS=anchor EXTRA_ENV="GGML_TOPK_STREAM=1" "$B/perf/run-w8-decomp.sh" 2>&1 | grep -E '^\s+\[' | sed "s/^/stream r$r /"
    done
  done
done
echo "=== done $(date '+%F %T') ==="
