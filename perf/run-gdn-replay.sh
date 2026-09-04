#!/bin/bash
# Recompute-on-rollback (LLAMA_GDN_REPLAY=1) against the snapshot write-back it replaces, at the
# four points of the write-back ceiling table in parallel-streams.md: prompt 06, 400 tokens,
# Turbo4 SOA-V1 at 1/4/8 slots (DFlash depth 3, the wide slot budget 16 the table used) and the
# f16 pick at 1 slot depth 4. Each point runs off/on interleaved, REPS times. Per-round time comes
# from the server's spec-prof timers (perf/server-prof-parse.py), t/s and shas from the harness.
#
#   perf/run-gdn-replay.sh                      # all four points, REPS=2
#   POINTS="t4-n1" REPS=1 perf/run-gdn-replay.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-gdnreplay-$(date +%m%d-%H%M)}
REPS=${REPS:-2}
POINTS=${POINTS:-"t4-n1 t4-n4 t4-n8 f16-n1"}
export SETS=same SAME_IDX=5 NPRED=${NPRED:-400}
run_point() {  # point, replay(0|1), rep
  local p=$1 on=$2 rep=$3
  local arm=off; [ "$on" = 1 ] && arm=on
  local tag=$TAG-$p-$arm-r$rep
  case $p in
    t4-n1)  env_kv=turbo4; ns=1; depth=3;;
    t4-n4)  env_kv=turbo4; ns=4; depth=3;;
    t4-n8)  env_kv=turbo4; ns=8; depth=3;;
    f16-n1) env_kv=f16;    ns=1; depth=4;;
  esac
  local -a e=(KV=$env_kv NS=$ns DEPTH=$depth TAG=$tag)
  if [ "$env_kv" = turbo4 ]; then
    e+=(M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf)
  fi
  [ "$ns" -ge 4 ] && e+=(GGML_MM_SKINNY_N16=1 LLAMA_SPEC_SLOT_BUDGET_WIDE=16)
  [ "$on" = 1 ] && e+=(LLAMA_GDN_REPLAY=1)
  echo "### $tag: ${e[*]}"
  env "${e[@]}" "$B/perf/run-parallel-streams.sh" 2>&1 | grep -E "^\[|WARNING|died|error" 
  grep -c "recompute-on-rollback" "$OUT/$tag-same-n$ns.server.log" | sed 's/^/    replay log lines: /'
  echo "    sha: $(awk -F'\t' 'NR>1{print $13}' "$OUT/$tag.tsv" | sort -u | tr '\n' ' ')  t/s: $(awk -F'\t' 'NR>1{print $9}' "$OUT/$tag.tsv" | tr '\n' ' ')"
}
echo "=== gdn replay A/B: $TAG (commit $(git -C "$B" rev-parse --short HEAD)) ==="
for rep in $(seq 1 "$REPS"); do
  for p in $POINTS; do
    run_point "$p" 0 "$rep"
    run_point "$p" 1 "$rep"
  done
done
echo
echo "=== per-round ms from the last two spec-prof dumps (perf/server-prof-parse.py) ==="
tags=()
for rep in $(seq 1 "$REPS"); do for p in $POINTS; do tags+=("$TAG-$p-off-r$rep" "$TAG-$p-on-r$rep"); done; done
python3 "$B/perf/server-prof-parse.py" "${tags[@]}"
