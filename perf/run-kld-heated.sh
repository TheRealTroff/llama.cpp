#!/bin/bash
# KLD of the pick's numerics on greedy vs heated trajectories (perf/spec-heated.md). Scores the
# two run-agreement-heated.sh corpora against q8_0 (f16 cache) with three test arms of the q4
# pick file: f16 cache (weights), Turbo4 cache (weights + cache), f16 cache + acch (weights +
# the prefill half-accumulate). Reads the new 'Mean overlap' lines (llama-perplexity on
# spec-heated) next to Same top p.
#   GEN_TAG=agreeh-sep07 perf/run-kld-heated.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
export B=${B:-/Users/troff/play/llama.cpp-active}
GEN_TAG=${GEN_TAG:-agreeh-sep07}
DATA=/Users/troff/play/kvquant-experiments/data
Q4=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf
CHUNKS=${CHUNKS:-8}
for arm in t0 t07; do
  W=$DATA/generated-$GEN_TAG-$arm.txt
  [ -s "$W" ] || { echo "missing corpus $W"; continue; }
  TAG=kldh-$GEN_TAG-$arm
  echo; echo "########## corpus $arm ($W, $(wc -c <"$W" | tr -d ' ') bytes) ##########"
  W=$W CHUNKS=$CHUNKS TAG=$TAG "$B/perf/run-quant-kld.sh" "$Q4"
  W=$W CHUNKS=$CHUNKS TAG=$TAG KV=turbo4 LABEL=-turbo4 TURBO_AUTO_ASYMMETRIC=0 "$B/perf/run-quant-kld.sh" "$Q4"
  W=$W CHUNKS=$CHUNKS TAG=$TAG LABEL=-acch GGML_MM_ACC_HALF=1 "$B/perf/run-quant-kld.sh" "$Q4"
done
echo; echo "done: kldh-$GEN_TAG"
