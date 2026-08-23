#!/bin/bash
# Width 4: is the cost a half-filled simdgroup tile, or the vec kernel losing weight reuse?
#
# GGML_MM_SKINNY=5 excludes width 4 (ne11=4) from kernel_mul_mm_skinny, so width 4 runs on
# mul_mv_ext today. kernel_mul_mm_skinny accumulates into simdgroup_half8x8, so its column
# tile is fixed at 8: at ne11=4 it clamps nr1=4 but still issues full 8x8 MMAs and dispatches
# ((ne11+7)/8) threadgroups. If the tile-waste reading is right, skinny at width 4 should be
# poor in a specific way; if ext is losing weight reuse instead, skinny should WIN there.
# Nobody has measured skinny at width 4 - the flag has always excluded it.
#
# README warns 4-column batches misroute at SKINNY=4 unless repack is on, so the repack arm
# is the real candidate and the no-repack arm is run only to document the misroute.
# CORRECTNESS GATE: canonical output sha1 is 9ad7e023c6ab (slope-sweep.md, all 15 runs, both
# drafters, every depth). Any arm that differs is a wrong answer, not a fast one.
#
# Also fills the MTP width-4 cell: the MTP sweep ran d1/d2/d4/d6/d7/d8 and skipped d3.
#
# Controls run FIRST AND LAST. width4-verify.md run 6 was invalidated by an n6 control that
# came in -6.1% on a byte-identical workload; if the two controls disagree, the run is void.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PROMPT=/Users/troff/play/benchprompt.txt
PORT=8095
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-width4ab-$(date +%m%d-%H%M)}
COOL=${COOL:-90}
mkdir -p "$OUT"

BASE_ENV=(GGML_MV_NC=2 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)
CANON=9ad7e023c6ab

echo "=== width-4 skinny A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "canon  : sha1 $CANON  (any other sha = wrong answer)"
echo "cooldown: ${COOL}s between e2e arms"
echo

run_e2e() {
  local label=$1; shift
  local nenv=$1; shift          # count of leading env assignments
  local envs=("${@:1:$nenv}"); shift $nenv
  local slog="$OUT/$TAG-$label.server.log"
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port busy"; return 1; fi
  env "${BASE_ENV[@]}" "${envs[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on \
      -ctk f16 -ctv f16 "$@" --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }

  python3 -c "
import json
print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': 300, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d: print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
gen=t.get('predicted_n',0); dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
rounds=gen-da; tps=t.get('predicted_per_second',0)
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
print('[%-22s] %6.3f t/s  acc=%5.1f%%  committed/rd=%4.2f  rounds=%3d  ms/rd=%6.1f  sha=%s%s'
      % ('$label', tps, 100*da/dn if dn else 0, gen/rounds if rounds else 0, rounds,
         1000*(gen/rounds)/tps if tps and rounds else 0, sha,
         '' if sha=='$CANON' else '  <<< SHA MISMATCH'))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep "$COOL"
}

echo "--- part 1: e2e ---"
run_e2e "ctrl-n6-pre"       1 GGML_MM_SKINNY=5 -md "$MD" --spec-type draft-dflash --spec-draft-n-max 6
run_e2e "dflash-n3-skinny5" 1 GGML_MM_SKINNY=5 -md "$MD" --spec-type draft-dflash --spec-draft-n-max 3
run_e2e "dflash-n3-skinny4-repack" 2 GGML_MM_SKINNY=4 GGML_MV_REPACK=1 -md "$MD" --spec-type draft-dflash --spec-draft-n-max 3
run_e2e "dflash-n3-skinny4-norepack" 1 GGML_MM_SKINNY=4 -md "$MD" --spec-type draft-dflash --spec-draft-n-max 3
run_e2e "mtp-d3-skinny5"    1 GGML_MM_SKINNY=5 --spec-type draft-mtp --spec-draft-n-max 3
run_e2e "mtp-d3-skinny4-repack" 2 GGML_MM_SKINNY=4 GGML_MV_REPACK=1 --spec-type draft-mtp --spec-draft-n-max 3
run_e2e "ctrl-n6-post"      1 GGML_MM_SKINNY=5 -md "$MD" --spec-type draft-dflash --spec-draft-n-max 6
echo

echo "--- part 2: llama-bench ms/pass, the kernel in isolation ---"
bench() {
  local label=$1; shift
  echo "--- $label ---"
  env "${BASE_ENV[@]}" "$@" "$BIN/llama-bench" -m "$M" -fa 1 -ctk f16 -ctv f16 \
      -n 0 -p 1,2,3,4,5,6,7,8 -r 3 2>&1 \
    | tee "$OUT/$TAG-bench-$label.log" | grep -E "^\|" | grep -vE "^\| *-"
  echo
}
bench "skinny5"        GGML_MM_SKINNY=5
bench "skinny4-repack" GGML_MM_SKINNY=4 GGML_MV_REPACK=1
