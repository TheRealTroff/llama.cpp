#!/bin/bash
# The view-slide A/B (exp/graph-reuse-cache, perf/cpu-round-overhead.md addendum 2026-09-29): the recurrent-state
# views of a cached graph slide to the round's source rows instead of the graph being rebuilt. Fixed depth 3
# (PICK_SPEC_EV=0: a deterministic sha), the Turbo4 pick arm, ABBA with LLAMA_RS_SLIDE=0 as A. Reports sha, t/s,
# `graphs reused`, the decode-prof split (LLAMA_DECODE_PROF=1, non-perturbing) and dec_sub_tg.
#   LINE=q4|ud [ARM=turbo4-n3-300] [TAG=..] bash perf/run-rs-slide-ab.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-graph-reuse}
LINE=${LINE:-q4}; ARM=${ARM:-turbo4-n3-300}; TAG=${TAG:-rsslide-$(date +%m%d-%H%M)}
OUT=/Users/troff/play/kvquant-experiments/results
S=${S:-/tmp}
for run in 1-off 2-on 3-on 4-off; do
  v=${run#*-}; on=0; [ "$v" = on ] && on=1
  T="$TAG-$LINE-$run"
  B="$B" LINE=$LINE TURBO=1 ARMS="$ARM" MULTISLOT=0 VISION=0 PICK_SPEC_EV=0 TAG=$T \
    EXTRA="LLAMA_DECODE_PROF=1 LLAMA_RS_SLIDE=$on" bash "$B/perf/run-prod-pick.sh" > "$S/$T.console.log" 2>&1
  slog="$OUT/$T-$ARM.server.log"
  sha=$(grep -o "[0-9a-f]\{12\}  *[0-9]* B  */tmp/prodpick-$ARM.txt" "$S/$T.console.log" | tail -1 | awk '{print $1}')
  tps=$(grep "eval time =" "$slog" | grep -v prompt | tail -1 | sed 's/.*(\(.*\)tokens per second.*/\1/' | awk '{print $NF}')
  reused=$(grep "graphs reused" "$slog" | tail -1 | awk '{print $NF}')
  sub=$(grep "spec-prof dec_sub_tg" "$slog" | tail -1 | sed 's/.*avg = *//; s/ ms.*//')
  dp=$(grep "decode-prof" "$slog" | grep -v "$(grep -m1 -o 'ctx=0x[0-9a-f]*' "$slog" | head -1)" | tail -1 | sed 's/.*avg ms: //')
  echo "$run slide=$on sha=$sha t/s=$tps reused=$reused dec_sub_tg=$sub | $dp"
done
