#!/bin/bash
# perf/run-specev-tax.sh - where the controller's free-form round tax comes from (2026-09-17).
# The Sep 17 gate A/B (run-specev-pick-gate.sh) found the block-8 draft rounds ~+16 ms on benchprompt against
# the +4.4 ms of Sep 7 (spec-verify-narrow.md section 7): dec_sub_tg +4.6, dec_syn_tg +3.2, draft_call +1.5 in
# the spec-prof window. Arms on ONE prompt, LV=5 (per-round lines) + LLAMA_SPEC_EV_DBG=1 for the (block, k)
# accounting (specev-dbg-account.py); LLAMA_DECODE_PROF=1 for the per-decode apply/reuse/set_inputs/submit split
# (a target graph that cannot be reused because the verify width changed is rebuilt and reallocated):
#   fixed3     the pick at depth 3
#   hybrid     the controller as gated
#   forced83   block 8 drafted every round, verify 3 (LLAMA_SPEC_EV_BLOCK=full LLAMA_SPEC_EV_WIDTHS=3, depth 7)
#   forced43   block 4 every round, verify 3 (depth 4)
#   tiered     the tiered block rule (block 4 by default, 8 after a fully accepted >=4 round)
#   w37        hybrid with the verify widths restricted to {3, 7} (LLAMA_SPEC_EV_WIDTHS=3,7): one shape change per regime switch
#   w37full    block 8 every round + widths {3, 7}: no drafter shape change either
#   noasync    hybrid with DFLASH_ASYNC_INJECT=0 (is the tax the async inject serialized behind the target submit?)
# LINE=q4|ud, PROMPT=benchprompt, ARMS=...
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=/Users/troff/play/llama.cpp-prod
OUT=/Users/troff/play/kvquant-experiments/results
LINE=${LINE:-q4}
PROMPT=${PROMPT:-benchprompt}
ARMS=${ARMS:-"fixed3 hybrid forced83 forced43 tiered w37 w37full noasync fixed3b"}
TAGB=${TAGB:-specev-tax-$(date +%m%d)-$LINE}
export B LINE KV=turbo4 LV=5 NPRED=${NPRED:-300} PROMPTS=$PROMPT LLAMA_DECODE_PROF=1 PICK_SPEC_EV=0  # arms set the controller flags themselves
cd "$B"
echo "=== spec-ev tax: $TAGB line=$LINE prompt=$PROMPT arms [$ARMS]; prod $(git rev-parse --short HEAD) $(date '+%T') ==="
for arm in $ARMS; do
  tag=$TAGB-$arm
  case $arm in
    fixed3|fixed3b) env -u LLAMA_SPEC_EV TAG=$tag DEPTHS=3 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    hybrid)   LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=hybrid LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=7 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    forced83) LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=full LLAMA_SPEC_EV_WIDTHS=3 LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=7 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    forced43) LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=full LLAMA_SPEC_EV_WIDTHS=3 LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=4 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    tiered)   LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=tiered LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=7 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    w37)      LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=hybrid LLAMA_SPEC_EV_WIDTHS=3,7 LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=7 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    w37full)  LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=full LLAMA_SPEC_EV_WIDTHS=3,7 LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=7 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
    noasync)  DFLASH_ASYNC_INJECT=0 LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=hybrid LLAMA_SPEC_EV_DBG=1 TAG=$tag DEPTHS=7 perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' ;;
  esac
  log=$(ls -t $OUT/$tag-n*-$PROMPT-r1.server.log 2>/dev/null | head -1)
  [ -n "$log" ] && grep -h "spec-prof" "$log" | tail -10 | sed -E 's/^[0-9.]+ I (srv|slot) +[a-z_]+\(\)?:? ?//' | awk '{printf "%s %s | ", $2, $8} END {print ""}' | sed "s/^/  [$arm] /"
  [ -n "$log" ] && grep -h "spec-ev: k hist" "$log" | tail -1 | sed -E 's/^.*spec-ev: //' | cut -c1-120 | sed "s/^/  [$arm] /"
  [ -n "$log" ] && grep -h "dflash-prof lattice" "$log" | tail -1 | sed -E 's/^.*dflash-prof //' | sed "s/^/  [$arm] /"
  [ -n "$log" ] && grep -h "decode-prof" "$log" | awk '{c[$2]=$0} END {for (k in c) print c[k]}' | sed -E 's/^.*decode-prof //' | sed "s/^/  [$arm] /"
  [ -n "$log" ] && grep -h "graphs reused" "$log" | tail -1 | sed -E 's/^.*graphs reused/graphs reused/' | sed "s/^/  [$arm] /"
done
for arm in hybrid forced83 tiered w37 w37full noasync; do
  case " $ARMS " in *" $arm "*) echo "--- account $arm"; python3 perf/specev-dbg-account.py $TAGB-$arm 2>&1 | head -30 ;; esac
done
echo "=== TAX-COMPLETE $(date '+%T') ==="
