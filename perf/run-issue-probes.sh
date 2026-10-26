#!/bin/bash
# run-issue-probes.sh - the two prediction arms of census-groups.md "Instructions per type" (exp/issue-probes, 2026-10-26)
#   (a) q4_0 SoA skinny tile is issue-bound at width 8: GGML_MM_SKINNY_PROBE=1 (no dequant: fewer instructions, same bytes) pays,
#       GGML_MM_SKINNY_PROBE=2 (cached weights: fewer DRAM bytes, same instructions) does not.
#   (b) the K-quant skinny tile is occupancy/latency-bound: GGML_MM_SKINNY_SMEM_PAD (fewer resident threadgroups, kernel unchanged)
#       costs q5_K proportionally and q4_0 SoA much less.
# Isolated test-backend-ops perf, REPS runs each, median of the us/run lines. Output: $OUT/issue-probes-<TAG>.txt
B=${B:-/Users/troff/play/llama.cpp-issueprobes}
BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-oct26}
REPS=${REPS:-3}
PADS=${PADS:-"0 4096 8192 12288 16384 24576"}
# the pick env of each tile (perf/pick.sh): the arms time the census kernels, not a variant
Q4_ENV=${Q4_ENV:-"GGML_MM_SKINNY_BSPLIT=2"}
GEN_ENV=${GEN_ENV:-"GGML_MM_SKINNY_GEN=6 GGML_MM_SKINNY_Q5K=1 GGML_MM_SKINNY_KQ2=1 GGML_MM_SKINNY_BSPLIT=2"}
LOG=$OUT/issue-probes-$TAG.txt
mkdir -p "$OUT"; : > "$LOG"

run() { # name envs filter
  local name=$1 envs=$2 filt=$3 us
  us=$(for r in $(seq $REPS); do (cd "$B" && env GGML_MV_REPACK=2 $envs "$BIN" perf -o MUL_MAT -b MTL0 -p "$filt" 2>&1) | grep -o '[0-9.]* us/run' | awk '{print $1}'; done | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  printf '%-52s %10s us/run   [%s]\n' "$name" "$us" "$envs" | tee -a "$LOG"
}

Q4A="type_a=q4_0_soa,type_b=f32,m=17408,n=8,k=5120,"
Q4B="type_a=q4_0_soa,type_b=f32,m=5120,n=8,k=17408,"
Q5="type_a=q5_K_soa,type_b=f32,m=17408,n=8,k=5120,"
echo "# arm (a): q4_0 SoA skinny, instructions vs bytes" | tee -a "$LOG"
for f in "$Q4A" "$Q4B"; do
  run "q4_0_soa $f base"           "$Q4_ENV"                        "$f"
  run "q4_0_soa $f PROBE=1 nodeq"  "$Q4_ENV GGML_MM_SKINNY_PROBE=1" "$f"
  run "q4_0_soa $f PROBE=2 cached" "$Q4_ENV GGML_MM_SKINNY_PROBE=2" "$f"
done
echo "# arm (b): occupancy pad ladder, q4_0 SoA vs q5_K generic skinny" | tee -a "$LOG"
for pad in $PADS; do
  run "q4_0_soa $Q4A pad=$pad" "$Q4_ENV GGML_MM_SKINNY_SMEM_PAD=$pad" "$Q4A"
  run "q5_K     $Q5 pad=$pad"  "$GEN_ENV GGML_MM_SKINNY_SMEM_PAD=$pad" "$Q5"
done
echo "done -> $LOG"
