#!/bin/bash
# ABAB e2e gate for GGML_FUSE_SMALL: the canonical harness (run-prod-pick.sh + benchprompt.txt), Turbo4 arms,
# same binary, base = manifest picks only, fused = + the proposed flag (PICK_PROPOSED=1). Both lines.
set -u
B=${B:-/Users/troff/play/llama.cpp-fuse}
LINES=${LINES:-"ud q4"}
REPS=${REPS:-2}
ARMS=${ARMS:-"turbo4-n3-600 turbo4-n3-300"}
DATE=$(date +%m%d-%H%M)
for line in $LINES; do
  for rep in $(seq 1 $REPS); do
    for arm in base fused; do
      prop=0; [ $arm = fused ] && prop=1
      env B="$B" LINE="$line" TURBO=1 ARMS="$ARMS" PICK_PROPOSED=$prop TAG="fusegate-$DATE-$line-$arm-r$rep" "$B/perf/run-prod-pick.sh" 2>&1 | grep -E "^\[|commit|env  "
    done
  done
done
