#!/bin/bash
# perf/run-iq4xs-lut.sh - the iq4_xs tile's table lookups out of device loads (branch exp/iq4xs-lut,
# perf/w8-decomp-sep18.md lever 2). GGML_MM_SKINNY_IQ4LUT: 0 = the constant table (32 device loads per K-step
# per thread), 1 = the table staged in threadgroup memory, 2 = the exact quartic. Three steps on the experiment
# tree's binaries:
#   test   test-backend-ops correctness for iq4_xs_soa at widths 6-8 under the tile route, forms 1 and 2
#   perf   per-call timings, forms 0/1/2 interleaved, on the width-8 shapes the census timed (+ width 6)
#   e2e    ud line, fixed depth 7 (every verify at width 8), pick env with the form 0 vs the best form,
#          interleaved REPS times through perf/run-w8-decomp.sh STEPS=anchor (sha + round timers)
#   STEPS="test perf e2e" REPS=2 FORMS="0 1" B=/Users/troff/play/llama.cpp-iq4lut
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
export B=${B:-/Users/troff/play/llama.cpp-iq4lut}
BIN=$B/build/bin
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-iq4lut-$(date +%m%d-%H%M)}
STEPS=${STEPS:-test perf e2e}
REPS=${REPS:-2}
FORMS=${FORMS:-0 1}      # e2e arms
E2E_VAR=${E2E_VAR:-GGML_MM_SKINNY_IQ4LUT}   # the env var the e2e arms set (GGML_MM_SKINNY_Q5K, GGML_MM_SKINNY_KQ2 ...)
PFORMS=${PFORMS:-0 1 2}  # perf arms
TFORMS=${TFORMS:-1 2}    # test arms
export PICK_SPEC_EV=0
source "$B/perf/pick.sh"
pick_env ud turbo4
ENVS="${PICK_ENV[*]}"
echo "=== iq4_xs tile table forms: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD)   binary: $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"

case " $STEPS " in *" test "*)
  echo; echo "--- test: iq4_xs_soa widths 6-8, tile route (GGML_MM_SKINNY_GEN=6), forms 1 and 2 ---"
  for form in $TFORMS; do
    for n in 6 7 8; do
      r=$(env $ENVS GGML_MV_REPACK=2 GGML_MM_SKINNY_IQ4LUT=$form "$BIN/test-backend-ops" test -b MTL0 -o MUL_MAT -p "type_a=iq4_xs_soa,type_b=f32,m=6144,n=$n,k=5120," 2>&1 | grep -E "tests passed|FAIL|Backend MTL0" | tail -1)
      printf "  form=%d n=%d  %s\n" "$form" "$n" "$r"
    done
  done
  ;;
esac

case " $STEPS " in *" perf "*)
  echo; echo "--- perf: per call, forms $PFORMS interleaved (us/run) ---"
  for f in "type_a=iq4_xs_soa,type_b=f32,m=17408,n=8,k=5120," "type_a=iq4_xs_soa,type_b=f32,m=5120,n=8,k=17408," "type_a=iq4_xs_soa,type_b=f32,m=17408,n=6,k=5120,"; do
    for rep in 1 2; do
      for form in $PFORMS; do
        out=$(env $ENVS GGML_MV_REPACK=2 GGML_MM_SKINNY_IQ4LUT=$form "$BIN/test-backend-ops" perf -b MTL0 -o MUL_MAT -p "$f" 2>&1)
        us=$(echo "$out" | grep -o "[0-9.]* us/run" | head -1)
        k=$(echo "$out" | grep -o "loaded kernel_mul_mm_skinny[^ ]*" | tail -1)
        printf "  %-52s form=%d rep%d  %-16s %s\n" "$f" "$form" "$rep" "$us" "$k"
      done
    done
  done
  ;;
esac

case " $STEPS " in *" e2e "*)
  echo; echo "--- e2e: ud fixed depth 7, pick env, forms $FORMS interleaved x$REPS (run-w8-decomp.sh anchors) ---"
  for r in $(seq 1 "$REPS"); do
    for form in $FORMS; do
      TAG="$TAG-f$form-r$r" LINES=ud DEPTHS=7 STEPS=anchor EXTRA_ENV="$E2E_VAR=$form" "$B/perf/run-w8-decomp.sh" 2>&1 | grep -E "^\s*\[|died|ABORT|timeout" | sed "s/^/  form=$form r$r /"
      grep -h -o "loaded kernel_mul_mm_skinny_[a-z0-9_]*_soa_ex[^ ]*" "$OUT/$TAG-f$form-r$r-ud-d7-anchor.server.log" | sort -u | sed 's/^/      /'
    done
  done
  ;;
esac
echo; echo "logs: $OUT/$TAG-*"
