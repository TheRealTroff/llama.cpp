#!/bin/bash
# K-split sweep for mul_mv_ext at widths 3-4.
#
# Their verify_m4 splits K across simdgroups (K_PARTS 2-4) and reduces through threadgroup
# memory; we have never split K past the 32 lanes of one simdgroup. Two knobs reach it:
#
#   nxpsg  - threads along K inside a simdgroup (already exists, capped at the simd width)
#   kp     - simdgroups per row block, each taking a strided slice of ne00 (new, this branch)
#
# Effective lanes along K = nxpsg*kp, and rows per threadgroup = (32/nxpsg)*(nsg/kp)*nr0, so
# the same lane count is reachable at different threadgroup shapes. The sweep is laid out to
# separate those two: e.g. 32 lanes as (nx32,kp1), (nx16,kp2) and (nx8,kp4).
#
# Weight and activation traffic are identical in every cell - each row is read by exactly one
# threadgroup and each K slice by exactly one simdgroup - so this varies parallelism alone.
#
# attn_q is a control by construction: it sits below the f16y gate, and the ks kernel is a
# copy of the f16y kernel only, so kp cannot engage there. Widths 1/2/5 are controls too
# (mv, mv_nc and skinny under the prod env).
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ksplit-$(date +%m%d-%H%M)}
REPS=${REPS:-3}
mkdir -p "$OUT"

BASE_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)
SHAPES='m=(17408|5120|6144|3072),n=[1-5],k=(5120|17408)'

# label:env - lanes along K = nxpsg*kp, rows/TG = (32/nxpsg)*(nsg/kp)*nr0 at nr0=2
CONFIGS=(
  "nx8-kp1:GGML_MV_EXT_NSG=2 GGML_MV_EXT_NXPSG=8  GGML_MV_EXT_KP=1"
  "nx16-kp1:GGML_MV_EXT_NSG=2 GGML_MV_EXT_NXPSG=16 GGML_MV_EXT_KP=1"
  "nx32-kp1:GGML_MV_EXT_NSG=2 GGML_MV_EXT_NXPSG=32 GGML_MV_EXT_KP=1"
  "nx8-kp2:GGML_MV_EXT_NSG=2 GGML_MV_EXT_NXPSG=8  GGML_MV_EXT_KP=2"
  "nx16-kp2:GGML_MV_EXT_NSG=2 GGML_MV_EXT_NXPSG=16 GGML_MV_EXT_KP=2"
  "nx32-kp2:GGML_MV_EXT_NSG=2 GGML_MV_EXT_NXPSG=32 GGML_MV_EXT_KP=2"
  "nx8-kp4:GGML_MV_EXT_NSG=4 GGML_MV_EXT_NXPSG=8  GGML_MV_EXT_KP=4"
  "nx16-kp4:GGML_MV_EXT_NSG=4 GGML_MV_EXT_NXPSG=16 GGML_MV_EXT_KP=4"
  "nx32-kp4:GGML_MV_EXT_NSG=4 GGML_MV_EXT_NXPSG=32 GGML_MV_EXT_KP=4"
  "nx32-kp2-nsg4:GGML_MV_EXT_NSG=4 GGML_MV_EXT_NXPSG=32 GGML_MV_EXT_KP=2"
)

echo "=== K-split sweep: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/test-backend-ops")"
echo "reps   : $REPS interleaved"
echo

echo "--- part 0: correctness, every cell that engages kp ---"
for cfg in "${CONFIGS[@]}"; do
  label=${cfg%%:*}; envs=${cfg#*:}
  printf '%-14s ' "$label"
  env "${BASE_ENV[@]}" $envs "$BIN/test-backend-ops" test -o MUL_MAT -b MTL0 2>&1 \
    | grep -E "tests passed|backends passed" | tr '\n' ' '
  echo
done | tee "$OUT/$TAG-correctness.log"
echo

echo "--- part 0b: kernel actually dispatched at width 4 ---"
for cfg in "${CONFIGS[@]}"; do
  label=${cfg%%:*}; envs=${cfg#*:}
  ppl=$(env "${BASE_ENV[@]}" $envs "$BIN/test-backend-ops" perf -o MUL_MAT -b MTL0 \
          -p "m=5120,n=4,k=17408" 2>&1 >/dev/null \
        | sed -n 's/.*compiling pipeline:.*name = .\(kernel_mul[^'"'"']*\).*/\1/p' | tail -1)
  printf '%-14s %s\n' "$label" "$ppl"
done | tee "$OUT/$TAG-routing.log"
echo

echo "--- part 1: test-backend-ops perf, us/run ---"
for rep in $(seq 1 "$REPS"); do
  for cfg in "${CONFIGS[@]}"; do
    label=${cfg%%:*}; envs=${cfg#*:}
    env "${BASE_ENV[@]}" $envs "$BIN/test-backend-ops" perf -o MUL_MAT -b MTL0 -p "$SHAPES" 2>/dev/null \
      | awk -v cfg="$label" -v rep="$rep" '
          /^  MUL_MAT/ { match($0, /m=[0-9]+,n=[0-9]+,k=[0-9]+/); shape=substr($0, RSTART, RLENGTH) }
          /us\/run/   { for (i=1;i<=NF;i++) if ($i=="us/run") us=$(i-1); print cfg, rep, shape, us }
        '
  done
done | tee "$OUT/$TAG-perf.raw"
echo

echo "--- summary (median of $REPS, us/run, delta vs nx8-kp1) ---"
python3 "$B/perf/summarize-ksplit.py" "$OUT/$TAG-perf.raw"
