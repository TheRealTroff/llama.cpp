#!/bin/bash
# Is the f16y convert dispatch part of the width 3-4 cliff?
#
# Found by dumping /private/tmp/perf-metal-67662.gputrace (perf/width4-verify.md): the ext
# f16y path encodes TWO dispatches per MUL_MAT node, not one -
#
#   "MUL_MAT" +- kernel_cpy_f32_f16              {272,1,1} x {256,1,1}
#             +- kernel_mul_mv_ext_q4_0_f16_r1_4 {320,1,1} x {32,2,1}
#
# with a ggml_metal_op_concurrency_reset between them (ggml-metal-ops.cpp:2856-2896). The
# convert is gated on ne11 >= 2 AND the ext family, and under prod routing
# (GGML_MV_NC=2 GGML_MM_SKINNY=5) only widths 3 and 4 land on ext. So widths 1-2 (mul_mv)
# and 5+ (skinny) never pay it and widths 3-4 always do - a structural discontinuity sitting
# exactly on the +113 us cliff, never A/B'd. GGML_MV_EXT_F16Y=0 turns it off, no code.
#
# PRE-REGISTERED BOUND (perf/width4-verify.md discipline - state it before measuring):
#   f16y is the shipping default, so the expectation is that f16y=1 WINS: the convert costs
#   ~418 KB of traffic (~2 us at 250 GB/s) plus one dispatch and a barrier, and buys halved
#   y-load instructions in the matmul. Predicting the convert+barrier overhead is under
#   20 us, i.e. under 18% of the 113 us cliff. f16y=0 coming out FASTER at widths 3-4 would
#   refute the default and is the outcome worth looking for.
#   Widths 1, 2 and 5 are controls: different kernel family, must be flat within noise.
#
# Interleaved arms inside one block - perf/width4-verify.md run 2 methodology note: across
# sessions the same arm drifts 3-8%, so only compare arms measured next to each other.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin/test-backend-ops
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-f16y-$(date +%m%d-%H%M)}
LOG=$OUT/$TAG.log
mkdir -p "$OUT"

# prod routing. Without it test-backend-ops sends everything to ext and the curve is a
# different shape - perf/width4-verify.md "Two traps in this harness".
PROD_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)

exec > >(tee "$LOG") 2>&1

echo "=== f16y A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$BIN")"
echo "env    : ${PROD_ENV[*]}"
echo "date   : $(date)"
echo

# m,k,label - the four 27B verify projections (tests/test-backend-ops.cpp:10218-10224)
SHAPES=(
    "17408 5120  ffn_gate_up"
    "5120  17408 ffn_down"
    "6144  5120  gdn_qkv"
    "3072  5120  attn_q"
)

# one perf case, one arm. echoes the us/run only.
one() {
    local m=$1 k=$2 n=$3 f16y=$4
    env "${PROD_ENV[@]}" GGML_MV_EXT_F16Y="$f16y" "$BIN" perf -o MUL_MAT -b MTL0 \
        -p "type_a=q4_0,type_b=f32,m=$m,n=$n,k=$k," 2>/dev/null \
      | grep -oE '[0-9]+\.[0-9]+ us/run' | head -1 | cut -d' ' -f1
}

# record the pipeline actually chosen, so the routing claim is in the log rather than assumed
echo "--- routing check (which kernel each width lands on, prod env) ---"
for n in 1 2 3 4 5; do
    p=$(env "${PROD_ENV[@]}" "$BIN" perf -o MUL_MAT -b MTL0 \
          -p "type_a=q4_0,type_b=f32,m=5120,n=$n,k=17408," 2>&1 \
        | grep -oE "loaded kernel_[a-z0-9_]+(_nsg=[0-9]+_nxpsg=[0-9]+_nr0=[0-9]+)?" \
        | sed 's/loaded //' | sort -u | tr '\n' ' ')
    echo "  ffn_down n=$n : $p"
done
echo

for width in 3 4 1 2 5; do
    # 3 interleaved reps at the ext widths, 1 at the controls
    if [ "$width" = "3" ] || [ "$width" = "4" ]; then REPS=3; ROLE="ext"; else REPS=1; ROLE="control"; fi
    echo "--- width $width ($ROLE, $REPS rep(s), arms interleaved) ---"
    printf '%-14s %10s %10s %9s\n' shape "f16y=1" "f16y=0" "delta"
    for s in "${SHAPES[@]}"; do
        read -r m k label <<< "$s"
        on=""; off=""
        for r in $(seq 1 $REPS); do
            a=$(one "$m" "$k" "$width" 1)
            b=$(one "$m" "$k" "$width" 0)
            on="$on $a"; off="$off $b"
        done
        # mean of the reps, and the percentage change from turning the convert off
        read -r mon moff pct <<< "$(python3 -c "
on=[float(x) for x in '''$on'''.split()]
off=[float(x) for x in '''$off'''.split()]
a=sum(on)/len(on); b=sum(off)/len(off)
print(f'{a:.2f} {b:.2f} {100*(b-a)/a:+.1f}')")"
        printf '%-14s %10s %10s %8s%%\n' "$label" "$mon" "$moff" "$pct"
        echo "    reps f16y=1:$on   f16y=0:$off" >> "$LOG.reps"
    done
    echo
done

echo "per-rep detail in $LOG.reps"
echo "=== done $(date) ==="
