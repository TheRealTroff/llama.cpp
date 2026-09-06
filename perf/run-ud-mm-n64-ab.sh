#!/bin/bash
# UD line prefill (perf/ud-model.md step 8): f32 64-column mul_mm tile vs the 32-column incumbent per
# format at the two FFN prefill shapes (n=512). test-backend-ops perf, interleaved reps, pipeline name
# from each invocation's own stderr. GGML_MM_N64=1 routes the f32 n64 tile when GGML_MM_ACC_HALF is
# unset; GGML_MM_N64_KMAX lifts the K<=6144 guard so ffn_down (K=17408) can be measured too.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-ud-soa}
BIN=$B/build/bin/test-backend-ops
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ud-mm-n64-$(date +%m%d-%H%M)}
REPS=${REPS:-2}
TYPES=${TYPES:-"q5_K q4_K iq4_xs q3_K q6_K q4_0"}
SHAPES=${SHAPES:-"17408:5120:ffn_gate_up 5120:17408:ffn_down"}
mkdir -p "$OUT"; exec > >(tee "$OUT/$TAG.log") 2>&1
echo "=== f32 n64 mul_mm A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$BIN")"
one() { # type m k envs -> "us kernel"
    local out
    out=$(cd "$B" && env $4 "$BIN" perf -o MUL_MAT -b MTL0 -p "type_a=$1,type_b=f32,m=$2,n=512,k=$3," 2>&1)
    echo "$(echo "$out" | grep -oE '[0-9]+\.[0-9]+ us/run' | head -1 | cut -d' ' -f1) $(echo "$out" | grep -oE 'loaded kernel_mul_mm_[A-Za-z0-9_]+' | tail -1 | sed 's/loaded //')"
}
printf '%-8s %-12s %-4s %-5s %10s  %s\n' type shape rep arm us_run kernel
for sh in $SHAPES; do
    m=${sh%%:*}; rest=${sh#*:}; k=${rest%%:*}; label=${rest#*:}
    for rep in $(seq 1 $REPS); do
        for t in $TYPES; do
            r=$(one $t $m $k "GGML_MV_REPACK=2");                                  printf '%-8s %-12s %-4s %-5s %10s  %s\n' $t $label $rep base ${r%% *} ${r#* }
            r=$(one $t $m $k "GGML_MV_REPACK=2 GGML_MM_N64=1 GGML_MM_N64_KMAX=20000"); printf '%-8s %-12s %-4s %-5s %10s  %s\n' $t $label $rep n64 ${r%% *} ${r#* }
        done
    done
done
