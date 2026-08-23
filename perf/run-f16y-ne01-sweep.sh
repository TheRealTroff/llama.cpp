#!/bin/bash
# Is the f16y size gate keyed on the wrong dimension?
#
# From run-f16y-ab.sh (2026-08-23): at width 4, turning f16y off costs +17.3% on ffn_down
# and +16.7% on gdn_qkv but only +0.6% on attn_q - inside noise. attn_q passes the gate
# comfortably (ne00*ne01 = 5120*3072 = 15.7M, gate is 8M), so the gate is letting through a
# shape that gains nothing.
#
# Hold ne00 fixed at 5120 and the only variable between gdn_qkv and attn_q is ne01: 6144
# wins 16.7%, 3072 wins nothing. That points at a dimensional error in the policy
# (ggml-metal-ops.cpp:2799-2800, `ne00*ne01 >= 8M`):
#
#   convert cost   ~ ne00*ne11          (one pass over the activations, independent of ne01)
#   matmul saving  ~ ne01*ne00*ne11     (halved y-loads, once per output row)
#   ratio          ~ ne01               <- ne00 cancels; the product gate is wrong-dimensioned
#
# PRE-REGISTERED PREDICTION: the f16y win should scale with ne01 and be near zero at small
# ne01 regardless of ne00. Crossover somewhere between ne01 3072 and 6144. If instead the win
# tracks ne00*ne01, m=4096 at ne00=5120 (21M) should win about as much as gdn_qkv (31M) and
# the ne01 story is wrong.
#
# Controls: m=1024 and m=1280 sit BELOW the 8M gate at ne00=5120 (5.2M, 6.5M), so f16y is
# off in both arms and both must read flat. If they do not, the harness is measuring
# something other than f16y.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin/test-backend-ops
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-f16yne01-$(date +%m%d-%H%M)}
LOG=$OUT/$TAG.log
mkdir -p "$OUT"

PROD_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)

exec > >(tee "$LOG") 2>&1

echo "=== f16y vs ne01, ne00 held at 5120: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$BIN")"
echo "date   : $(date)"
echo

one() {
    local m=$1 n=$2 f16y=$3
    env "${PROD_ENV[@]}" GGML_MV_EXT_F16Y="$f16y" "$BIN" perf -o MUL_MAT -b MTL0 \
        -p "type_a=q4_0,type_b=f32,m=$m,n=$n,k=5120," 2>/dev/null \
      | grep -oE '[0-9]+\.[0-9]+ us/run' | head -1 | cut -d' ' -f1
}

# ne01 values available as perf cases at k=5120 (tests/test-backend-ops.cpp:10218-10231)
#   1024, 1280        below the 8M gate  -> control, f16y inactive
#   3072 (attn_q)     15.7M              -> above gate, measured flat
#   4096              21.0M              -> the untested midpoint
#   6144 (gdn_qkv)    31.5M              -> above gate, measured +16.7%
#   17408 (ffn_g/u)   89.1M              -> above gate, but ne01 >= 8192 forces nr0=4, so
#                                           it is a different kernel config, not comparable
for width in 3 4; do
    echo "--- width $width, 3 interleaved reps ---"
    printf '%-8s %10s %10s %10s %9s %s\n' ne01 "prod(M)" "f16y=1" "f16y=0" "delta" note
    for m in 1024 1280 3072 4096 6144; do
        prod=$(python3 -c "print(f'{5120*$m/1e6:.1f}')")
        if [ "$m" -lt 1638 ]; then note="below gate (control)"; else note="above gate"; fi
        on=""; off=""
        for r in 1 2 3; do
            on="$on $(one "$m" "$width" 1)"
            off="$off $(one "$m" "$width" 0)"
        done
        read -r mon moff pct <<< "$(python3 -c "
on=[float(x) for x in '''$on'''.split()]
off=[float(x) for x in '''$off'''.split()]
a=sum(on)/len(on); b=sum(off)/len(off)
print(f'{a:.2f} {b:.2f} {100*(b-a)/a:+.1f}')")"
        printf '%-8s %10s %10s %10s %8s%% %s\n' "$m" "$prod" "$mon" "$moff" "$pct" "$note"
        echo "  ne01=$m w=$width f16y=1:$on f16y=0:$off" >> "$LOG.reps"
    done
    echo
done

echo "per-rep detail in $LOG.reps"
echo "=== done $(date) ==="
