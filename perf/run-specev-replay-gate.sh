#!/bin/bash
# perf/run-specev-replay-gate.sh - the deterministic controller gate (spec-verify-narrow.md section 11, owner 2026-09-18:
# "make determinism an option"). The adaptive-depth controller's picks depend on wall time through its cost EMA, so a
# marginal round can verify a different width run to run and fork a near tie under a different kernel family - a
# sha gate of a kernel change under the controller is statistical. This gate records the pick sequence once on build A
# (LLAMA_SPEC_EV_TRACE) and replays it on build B (LLAMA_SPEC_EV_REPLAY) N times: the same kernels on the same
# tokens; a sha difference is then the kernel change, not the controller. Both builds must carry the trace code.
#   A=<tree> B=<tree> (default: both prod)  N=<replays, default 3>  NPRED (default 1200)  PICK_LINES (default q4)
#   ARMS (run-fuse-quick.sh arms, default turbo4-n3)  TAG  EXTRA_A / EXTRA_B (env appended to the record / replay runs,
#   e.g. EXTRA_A=GGML_TOPK_STREAM=0 records under the old selector and the replays run the pick - same tree, one flag)
# Output: the record run's sha, each replay's sha, and the summary's replay counts (desync / past-trace must be 0).
set -u
A=${A:-/Users/troff/play/llama.cpp-prod}
B=${B:-/Users/troff/play/llama.cpp-prod}
N=${N:-3}
export NPRED=${NPRED:-1200} PICK_LINES=${PICK_LINES:-q4} ARMS=${ARMS:-turbo4-n3} LV=5
TAG=${TAG:-replaygate-$(date +%m%d-%H%M)}
OUT=/Users/troff/play/kvquant-experiments/results/fuse-quick
echo "=== replay gate $TAG: record on $(cd $A && git rev-parse --short HEAD) ($A), replay x$N on $(cd $B && git rev-parse --short HEAD) ($B) ==="
# one trace PER LINE (2026-09-29, kv-layout.md: with PICK_LINES="q4 ud" the ud record overwrote the q4 trace and the q4
# replays ran the ud line's widths - a "fork" that was a pick difference, not a kernel one)
for line in $PICK_LINES; do
TRACE=$OUT/$TAG-$line.picks
B=$A TAG=$TAG-rec PICK_LINES=$line EXTRA="LLAMA_SPEC_EV_DBG=1 LLAMA_SPEC_EV_TRACE=$TRACE ${EXTRA_A:-}" "$A/perf/run-fuse-quick.sh" 2>&1 | grep -v "^==="
echo "  recorded $(wc -l < "$TRACE" | tr -d ' ') picks ($line)"
for i in $(seq 1 "$N"); do
  B=$B TAG=$TAG-rep$i PICK_LINES=$line EXTRA="LLAMA_SPEC_EV_DBG=1 LLAMA_SPEC_EV_REPLAY=$TRACE ${EXTRA_B:-}" "$B/perf/run-fuse-quick.sh" 2>&1 | grep -v "^==="
  for log in "$OUT"/$TAG-rep$i-$line-*.server.log; do
    printf "    %s\n" "$(grep -h 'spec-ev: k hist' "$log" | tail -1 | grep -o 'replay \[.*\]' || echo 'replay [NO REPLAYED PICKS]')"
    grep -h "spec-ev: replay" "$log" | grep -i "desync\|exhausted\|cannot" | head -2 | sed 's/^/    /'
  done
done
done
echo "=== REPLAY-GATE-COMPLETE $(date '+%T') ==="
