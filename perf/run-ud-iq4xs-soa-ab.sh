#!/bin/bash
# UD line step 6 (perf/ud-model.md): iq4_xs width-4 SoA kernel variants vs the incumbent
# ext r1_4 kernel, per projection shape, interleaved. test-backend-ops perf, GGML_MV_REPACK=2
# (test buffers), route = GGML_MV_SOA_IQ4XS=<variant>. Each arm's us/run is taken from the
# same invocation whose stderr names the pipeline, so the routing claim is in the log.
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-ud-soa}
BIN=$B/build/bin/test-backend-ops
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ud-iq4xs-soa-$(date +%m%d-%H%M)}
REPS=${REPS:-2}
VARIANTS=${VARIANTS:-"1 2 3 4 5 6"}
N=${N:-4}
mkdir -p "$OUT"
exec > >(tee "$OUT/$TAG.log") 2>&1
echo "=== iq4_xs SoA w$N A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$BIN")"
SHAPES=${SHAPES:-"17408:5120:ffn_gate_up 5120:17408:ffn_down 12288:5120:attn_q 6144:5120:attn_gate"}
one() {  # m k variant -> "us kernel"
    local m=$1 k=$2 v=$3 out
    out=$(env GGML_MV_REPACK=2 GGML_MV_SOA_IQ4XS=$v "$BIN" perf -o MUL_MAT -b MTL0 \
          -p "type_a=iq4_xs,type_b=f32,m=$m,n=$N,k=$k," 2>&1)
    local us=$(echo "$out" | grep -oE '[0-9]+\.[0-9]+ us/run' | head -1 | cut -d' ' -f1)
    local kn=$(echo "$out" | grep -oE 'loaded kernel_mul_mv_[a-z0-9_]+' | grep -v cpy | tail -1 | sed 's/loaded //')
    echo "$us $kn"
}
printf '%-14s %-4s %-4s %10s  %s\n' shape rep arm us_run kernel
for sh in $SHAPES; do
    m=${sh%%:*}; rest=${sh#*:}; k=${rest%%:*}; label=${rest#*:}
    for rep in $(seq 1 $REPS); do
        r=$(one $m $k 0); printf '%-14s %-4s %-4s %10s  %s\n' $label $rep base ${r%% *} ${r#* }
        for v in $VARIANTS; do
            r=$(one $m $k $v); printf '%-14s %-4s %-4s %10s  %s\n' $label $rep v$v ${r%% *} ${r#* }
        done
    done
done
