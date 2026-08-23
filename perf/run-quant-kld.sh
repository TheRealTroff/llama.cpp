#!/bin/bash
# How much quality does the weight format cost? The whole speed stack is gated on Q4_0
# (skinny and repack are hard-gated on GGML_TYPE_Q4_0), and that choice was never priced
# against a better quant -- uniform-Q4_0 was only ever PPL-checked against ANOTHER Q4_0
# (mtp-kv-results.md:262, 6.5286 vs unsloth 6.5879).
#
# This measures the test model against q8_0 reference logits. KLD, not PPL: at these chunk
# counts a PPL confidence interval is ~1.4% and the Q4_0-vs-q8_0 delta lives right at that
# edge, while KLD is per-token and resolves it easily. PPL comes out of the same runs anyway.
#
#   perf/run-quant-kld.sh                      # q8_0 ref vs uniform-Q4_0, 24 chunks
#   CHUNKS=48 perf/run-quant-kld.sh <model>... # more power, ~1.02 GB of logits per chunk
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
# W is the scoring corpus. Overridable so the same machinery can score model-GENERATED
# text instead of wikitext - see run-agreement.sh, which is what makes 'Same top p'
# read as a greedy acceptance rate rather than a teacher-forced wikitext statistic.
W=${W:-/Users/troff/play/kvquant-experiments/data/wikitext-2-raw/wiki.test.raw}
REF=${REF:-/Users/troff/play/Qwen3.8-27B-conv-q8_0.gguf}
CHUNKS=${CHUNKS:-24}
CTX=${CTX:-2048}
SCRATCH=${SCRATCH:-/private/tmp/claude-501/-Users-troff-play/5a19a1a2-8952-4f8d-95e5-99e5e3b07300/scratchpad}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-kld-$(date +%m%d-%H%M)}
mkdir -p "$OUT" "$SCRATCH"

TESTS=("$@")
[ ${#TESTS[@]} -eq 0 ] && TESTS=(/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf)

BASE="$SCRATCH/kld-base-$TAG.dat"
NEED=$(python3 -c "print(int(248320*$CTX*2*$CHUNKS/1e9)+2)")
FREE=$(df -g "$SCRATCH" | tail -1 | awk '{print $4}')
echo "=== weight-quant KLD: $TAG ==="
echo "ref    : $REF"
echo "tests  : ${TESTS[*]}"
echo "chunks : $CHUNKS at -c $CTX  (~$((CHUNKS*CTX)) tokens)"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "binary : $(date -r "$BIN/llama-perplexity" '+%Y-%m-%d %H:%M')"
if [ -s "$BASE" ]; then
  echo "logits : $BASE (reusing, $(du -g "$BASE" | cut -f1) GB on disk)"
else
  echo "logits : $BASE  (to generate, need ~${NEED} GB, ${FREE} GB free)"
fi
echo

# 1. reference logits. -fa on with f16 KV both sides so the ONLY variable is the weights.
# The space check belongs HERE, not above: an existing base file is reused and needs no
# room, and hoisting the check aborted a legitimate reuse run at 14 GB free on 2026-08-23.
if [ ! -s "$BASE" ]; then
  [ "$FREE" -lt "$NEED" ] && { echo "ABORT: not enough space for the base logits"; exit 1; }
  echo "--- generating reference logits from $(basename "$REF") ---"
  "$BIN/llama-perplexity" -m "$REF" -f "$W" -c "$CTX" --chunks "$CHUNKS" -fa on \
    -ctk f16 -ctv f16 --kl-divergence-base "$BASE" >"$OUT/$TAG-ref.log" 2>&1 \
    || { echo "FAILED, see $OUT/$TAG-ref.log"; tail -5 "$OUT/$TAG-ref.log"; exit 1; }
  grep -E 'Final estimate' "$OUT/$TAG-ref.log" | sed 's/^/  ref /'
fi

# 2. each test model against those logits
for M in "${TESTS[@]}"; do
  n=$(basename "$M" .gguf)
  echo
  echo "--- $n vs reference ---"
  "$BIN/llama-perplexity" -m "$M" -f "$W" -c "$CTX" --chunks "$CHUNKS" -fa on \
    -ctk f16 -ctv f16 --kl-divergence --kl-divergence-base "$BASE" \
    >"$OUT/$TAG-$n.log" 2>&1 \
    || { echo "FAILED, see $OUT/$TAG-$n.log"; tail -5 "$OUT/$TAG-$n.log"; continue; }
  grep -E 'Mean KLD|Maximum KLD|99.0%|99.9%|Median KLD|Mean Delta|top token|Same top|RMS|PPL ratio|Final estimate' \
    "$OUT/$TAG-$n.log" | sed 's/^/  /'
done

echo
echo "logits file kept at $BASE - delete it when done (~${NEED} GB)"
