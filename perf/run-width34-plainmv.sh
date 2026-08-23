#!/bin/bash
# Widths 3-4 on plain mul_mv instead of mul_mv_ext.
#
# Under the prod env (GGML_MV_NC=2 GGML_MM_SKINNY=5) widths 3-4 fall in the gap between the
# nc route (ne11 <= min(GGML_MV_NC,4)) and skinny (ne11 >= max(2,GGML_MM_SKINNY)) and land on
# mul_mv_ext. Run 3 of width4-verify.md A/B'd ext against nc and skinny there, but never
# against the *plain* batch-1 mv kernel, which is what the final else branch of
# ggml_metal_op_mul_mat dispatches once every earlier gate declines.
#
# No code: GGML_MV_EXT_MAX=2 takes widths 3-4 out of the ext gate (ne11 <= ne11_mv_max), and
# GGML_MM_MIN stays 8 so mul_mm does not catch them either. Under the prod env this changes
# widths 3 and 4 ONLY - width 1 is not ext-eligible, width 2 is claimed by nc, widths 5-8 by
# skinny - so those widths are within-run controls that must not move.
#
# Plain mv dispatches ne11 in the grid y-dim with nr1=1, so it re-reads the weights once per
# column; ext streams them once for all r1ptg columns. The question is whether the smaller,
# higher-occupancy kernel beats that extra traffic at these widths.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-w34plainmv-$(date +%m%d-%H%M)}
REPS=${REPS:-3}
mkdir -p "$OUT"

BASE_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)
# the four 27B verify projections, widths 1-5; 1/2/5 are the untouched controls
SHAPES='m=(17408|5120|6144|3072),n=[1-5],k=(5120|17408)'

echo "=== widths 3-4 on plain mul_mv: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/test-backend-ops")"
echo "reps   : $REPS interleaved"
echo

echo "--- part 0: correctness of the plain-mv arm at widths 3-4 ---"
env "${BASE_ENV[@]}" GGML_MV_EXT_MAX=2 "$BIN/test-backend-ops" test -o MUL_MAT -b MTL0 \
    -p "$SHAPES" 2>&1 | tee "$OUT/$TAG-correctness.log" | tail -3
echo

echo "--- part 0b: which kernel each width lands on (one case at a time) ---"
for arm in ext plainmv; do
  # bash 3.2 + set -u: an empty array expansion is an error, so state the default
  extra=(GGML_MV_EXT_MAX=8); [ "$arm" = plainmv ] && extra=(GGML_MV_EXT_MAX=2)
  for n in 1 2 3 4 5; do
    ppl=$(env "${BASE_ENV[@]}" "${extra[@]}" "$BIN/test-backend-ops" perf -o MUL_MAT -b MTL0 \
            -p "m=5120,n=$n,k=17408" 2>&1 >/dev/null \
          | sed -n 's/.*compiling pipeline:.*name = .\(kernel_mul[^'"'"']*\).*/\1/p' | tail -1)
    printf '%-8s width %d  %s\n' "$arm" "$n" "$ppl"
  done
done | tee "$OUT/$TAG-routing.log"
echo

echo "--- part 1: test-backend-ops perf, us/run per shape and width ---"
for rep in $(seq 1 "$REPS"); do
  for arm in ext plainmv; do
    # bash 3.2 + set -u: an empty array expansion is an error, so state the default
  extra=(GGML_MV_EXT_MAX=8); [ "$arm" = plainmv ] && extra=(GGML_MV_EXT_MAX=2)
    env "${BASE_ENV[@]}" "${extra[@]}" "$BIN/test-backend-ops" perf -o MUL_MAT -b MTL0 \
        -p "$SHAPES" 2>/dev/null \
      | awk -v arm="$arm" -v rep="$rep" '
          /^  MUL_MAT/ { match($0, /m=[0-9]+,n=[0-9]+,k=[0-9]+/); shape=substr($0, RSTART, RLENGTH) }
          /us\/run/   { for (i=1;i<=NF;i++) if ($i=="us/run") us=$(i-1); print arm, rep, shape, us }
        '
  done
done | tee "$OUT/$TAG-perf.raw"
echo

echo "--- summary (median of $REPS, us/run) ---"
python3 "$B/perf/summarize-width34-plainmv.py" "$OUT/$TAG-perf.raw"
echo

echo "--- part 2: llama-bench ms/pass, whole model ---"
for arm in ext plainmv; do
  # bash 3.2 + set -u: an empty array expansion is an error, so state the default
  extra=(GGML_MV_EXT_MAX=8); [ "$arm" = plainmv ] && extra=(GGML_MV_EXT_MAX=2)
  echo "--- $arm ---"
  env "${BASE_ENV[@]}" GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1 "${extra[@]}" \
      "$BIN/llama-bench" -m "$M" -fa 1 -ctk f16 -ctv f16 -n 0 -p 1,2,3,4,5,6,7,8 -r 3 2>&1 \
    | tee "$OUT/$TAG-bench-$arm.log" | grep -E "^\|" | grep -vE "^\| *-"
  echo
done
