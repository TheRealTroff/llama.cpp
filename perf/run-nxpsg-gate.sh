#!/bin/bash
# Does extending nxpsg=16 to widths 3-4 pay on a whole model pass?
#
# This replaces the first attempt (perf/run-nxpsg-confirm.sh, 2026-08-23), which forced
# GGML_MV_EXT_NXPSG=16 globally and thereby ALSO bypassed the `ne00 % 256 == 0` guard at
# ggml-metal-ops.cpp:2770. That guard is correctness, not a heuristic: forced, MUL_MAT fails
# with NaN on kernel_mul_mv_ext_f16_f32_r1_2 at k=128 (two cases, type_a=f16, m=64/83).
# Any llama-bench number from that arm was measuring a partly-wrong kernel.
#
# The honest version moves ONLY the ne11 cutoff, keeping the ne00 guard in both arms:
#   GGML_MV_EXT_NXPSG16_MAX=3   ne11 < 3  -> shipping: widths 1-2 get nxpsg=16 (branch default)
#   GGML_MV_EXT_NXPSG16_MAX=5   ne11 < 5  -> widths 1-4 get it, which is the proposal
# One binary, one flag, guard intact.
#
# PRE-REGISTERED BOUND (perf/width4-verify.md run 3 measured the per-shape split):
#   ffn_gate/up -7.4%/+0.1%, ffn_down -5.0%/-3.2%, gdn_qkv -0.6%/-2.5%, attn_q +2.5%/+8.3%
#   at widths 3/4. Three shapes win, attn_q loses and loses hardest at width 4. If the
#   aggregate follows weight bytes, expect N=3 and N=4 to improve ~3-5% and N=1,2,5..8 flat.
#   N=1 and N=2 are hard controls: they are already nxpsg=16 in both arms and MUST be flat.
#   A flat or worse N=3/N=4 means attn_q-class shapes dominate the pass and the cutoff has
#   to become per-shape rather than a single ne11 threshold.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-nxpsggate-$(date +%m%d-%H%M)}
LOG=$OUT/$TAG.log
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)

exec > >(tee "$LOG") 2>&1

echo "=== nxpsg=16 cutoff, ne11 < 3 vs ne11 < 5: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$BIN/llama-bench")"
echo "env    : ${PICK_ENV[*]}"
echo "date   : $(date)"
echo

echo "--- step 1: correctness, both arms, full MUL_MAT suite ---"
for mx in 3 5; do
    res=$(env "${PICK_ENV[@]}" GGML_MV_EXT_NXPSG16_MAX=$mx \
            "$BIN/test-backend-ops" test -o MUL_MAT -b MTL0 2>&1 \
          | grep -cE "^\s+MUL_MAT.*(FAIL|NaN)")
    tot=$(env "${PICK_ENV[@]}" GGML_MV_EXT_NXPSG16_MAX=$mx \
            "$BIN/test-backend-ops" test -o MUL_MAT -b MTL0 2>&1 \
          | grep -oE "[0-9]+/[0-9]+ backends passed" | head -1)
    echo "  NXPSG16_MAX=$mx : failing MUL_MAT cases = $res, $tot"
done
echo

echo "--- step 2: routing check, which nxpsg each width actually gets (ffn_down) ---"
for mx in 3 5; do
    echo "  NXPSG16_MAX=$mx"
    for n in 1 2 3 4 5; do
        p=$(env "${PICK_ENV[@]}" GGML_MV_EXT_NXPSG16_MAX=$mx "$BIN/test-backend-ops" \
              perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32,m=5120,n=$n,k=17408," 2>&1 \
            | grep -oE "loaded kernel_[a-z0-9_]+(_nsg=[0-9]+_nxpsg=[0-9]+_nr0=[0-9]+)?" \
            | sed 's/loaded //' | sort -u | tr '\n' ' ')
        echo "    n=$n : $p"
    done
done
echo

echo "--- step 3: llama-bench pass cost, N=1..8, arms interleaved by rep ---"
# Parse with the csv module, NOT awk -F, : llama-bench quotes its data rows and cpu_info is
# "Accelerate, Apple M4 Pro", so a bare comma split shifts every column after it by one.
# avg_ns is the pass cost in ns - the figure width4-verify.md's llama-bench signature
# (73.0/73.8/101.5/111.5/...) is stated in.
PARSE='
import csv, sys
r = csv.reader(sys.stdin)
head = next(r)
for row in r:
    d = dict(zip(head, row))
    print("  rep=%s max=%s N=%-2s ms=%.2f" % (sys.argv[1], sys.argv[2], d["n_prompt"], int(d["avg_ns"])/1e6))
'
# ALTERNATE the arm order between reps. The first version of this harness ran max=3 then
# max=5 in every rep, and the N=1 control - which is bit-identical routing in both arms and
# must read flat - came out +2.1% while the N=2 control read -0.3%. That is the machine
# warming inside a rep being charged to whichever arm runs second. Alternating cancels it to
# first order; the two controls agreeing is the check that it worked.
for rep in 1 2 3 4; do
    if [ $((rep % 2)) -eq 1 ]; then ORDER="3 5"; else ORDER="5 3"; fi
    for mx in $ORDER; do
        env "${PICK_ENV[@]}" GGML_MV_EXT_NXPSG16_MAX=$mx \
            "$BIN/llama-bench" -m "$M" -fa 1 -ctk f16 -ctv f16 \
            -n 0 -p 1,2,3,4,5,6,7,8 -r 1 -o csv 2>/dev/null \
          | python3 -c "$PARSE" "$rep" "$mx"
    done
done

echo
echo "=== done $(date) ==="
