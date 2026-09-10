#!/bin/bash
# ABAB e2e gate for GGML_FUSE_SMALL: the canonical harness (run-prod-pick.sh + benchprompt.txt), Turbo4 arms,
# same binary, base = manifest picks, fused = base + GGML_FUSE_SMALL=<FUSE> in the environment. Both lines.
# NOT PICK_PROPOSED=1: that enables every proposed manifest entry, including SPEC-class ones (LLAMA_SPEC_EV=1)
# that fork the sha lineage on their own - a gate for one flag passes that flag explicitly.
set -u
B=${B:-/Users/troff/play/llama.cpp-fuse}
PICK_LINES=${PICK_LINES:-"ud q4"}
REPS=${REPS:-2}
ARMS=${ARMS:-"turbo4-n3-600 turbo4-n3-300"}
FUSE=${FUSE:-63}
DATE=$(date +%m%d-%H%M)
for line in $PICK_LINES; do
  for rep in $(seq 1 $REPS); do
    for arm in base fused; do
      if [ $arm = fused ]; then
        env B="$B" LINE="$line" TURBO=1 ARMS="$ARMS" PICK_PROPOSED=0 GGML_FUSE_SMALL=$FUSE TAG="fusegate-$DATE-$line-$arm-r$rep" "$B/perf/run-prod-pick.sh" 2>&1 | grep -E "^\[|commit"
      else
        env B="$B" LINE="$line" TURBO=1 ARMS="$ARMS" PICK_PROPOSED=0 TAG="fusegate-$DATE-$line-$arm-r$rep" "$B/perf/run-prod-pick.sh" 2>&1 | grep -E "^\[|commit"
      fi
    done
  done
done
