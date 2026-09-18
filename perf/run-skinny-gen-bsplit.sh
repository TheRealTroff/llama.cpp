#!/bin/bash
# perf/run-skinny-gen-bsplit.sh - the B-split ported to the generic skinny tile (branch exp/skinny-gen-bsplit,
# perf/w8-decomp-sep18.md lever 1). Three steps on the experiment tree's binaries:
#   test   test-backend-ops correctness for the stored SoA types at widths 6-8 under the tile route + bsp 2
#   perf   per-call timings, bsp 0 vs 2, on the width-8 shapes the census timed (iq4_xs / q5_K / q4_K)
#   e2e    ud line, fixed depth 7 (every verify at width 8), pick env with GGML_MM_SKINNY_BSPLIT=0 vs 2,
#          interleaved REPS times through perf/run-w8-decomp.sh STEPS=anchor (sha + round timers)
#   STEPS="test perf e2e" REPS=2 B=/Users/troff/play/llama.cpp-w8
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
export B=${B:-/Users/troff/play/llama.cpp-w8}
BIN=$B/build/bin
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-genbsp-$(date +%m%d-%H%M)}
STEPS=${STEPS:-test perf e2e}
REPS=${REPS:-2}
export PICK_SPEC_EV=0
source "$B/perf/pick.sh"
pick_env ud turbo4
ENVS="${PICK_ENV[*]}"
echo "=== generic skinny tile B-split: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD)   binary: $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"

case " $STEPS " in *" test "*)
  echo; echo "--- test: stored SoA types, widths 6-8, tile route (GGML_MM_SKINNY_GEN=6) + bsp 2 ---"
  for t in iq4_xs_soa q4_K_soa q5_K_soa q6_K_soa q3_K_soa iq4_nl_soa iq3_s_soa; do
    for n in 6 7 8; do
      r=$(env $ENVS GGML_MV_REPACK=2 "$BIN/test-backend-ops" test -b MTL0 -o MUL_MAT -p "type_a=$t,type_b=f32,m=6144,n=$n,k=5120," 2>&1 | grep -E "tests passed|FAIL|Backend MTL0" | tail -1)
      printf "  %-12s n=%d  %s\n" "$t" "$n" "$r"
    done
  done
  ;;
esac

case " $STEPS " in *" perf "*)
  echo; echo "--- perf: per call, bsp 0 vs 2 (us/run) ---"
  for f in "type_a=iq4_xs_soa,type_b=f32,m=17408,n=8,k=5120," "type_a=q5_K_soa,type_b=f32,m=17408,n=8,k=5120," "type_a=q4_K_soa,type_b=f32,m=17408,n=8,k=5120," "type_a=iq4_xs_soa,type_b=f32,m=5120,n=8,k=17408," "type_a=q5_K_soa,type_b=f32,m=5120,n=8,k=17408," "type_a=q5_K_soa,type_b=f32,m=6144,n=8,k=5120," "type_a=iq4_xs_soa,type_b=f32,m=17408,n=6,k=5120,"; do
    for rep in 1 2; do
      for bsp in 0 2; do
        us=$(env $ENVS GGML_MV_REPACK=2 GGML_MM_SKINNY_BSPLIT=$bsp "$BIN/test-backend-ops" perf -b MTL0 -o MUL_MAT -p "$f" 2>&1 | grep -o "[0-9.]* us/run" | head -1)
        k=$(env $ENVS GGML_MV_REPACK=2 GGML_MM_SKINNY_BSPLIT=$bsp "$BIN/test-backend-ops" perf -b MTL0 -o MUL_MAT -p "$f" 2>&1 | grep -o "loaded kernel_mul_mm_skinny[^ ]*" | tail -1)
        printf "  %-52s bsp=%d rep%d  %-16s %s\n" "$f" "$bsp" "$rep" "$us" "$k"
      done
    done
  done
  ;;
esac

case " $STEPS " in *" e2e "*)
  echo; echo "--- e2e: ud fixed depth 7, pick env, bsp 0 vs 2 interleaved x$REPS (run-w8-decomp.sh anchors) ---"
  for r in $(seq 1 "$REPS"); do
    for bsp in 0 2; do
      TAG="$TAG-bsp$bsp-r$r" LINES=ud DEPTHS=7 STEPS=anchor EXTRA_ENV="GGML_MM_SKINNY_BSPLIT=$bsp" "$B/perf/run-w8-decomp.sh" 2>&1 | grep -E "^\s*\[|died|ABORT|timeout" | sed "s/^/  bsp=$bsp r$r /"
      grep -h -o "loaded kernel_mul_mm_skinny_iq4_xs[^ ]*" "$OUT/$TAG-bsp$bsp-r$r-ud-d7-anchor.server.log" | sort -u | sed 's/^/      /'
    done
  done
  ;;
esac
echo; echo "logs: $OUT/$TAG-*"
