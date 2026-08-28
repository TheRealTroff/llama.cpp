#!/bin/bash
# CPU-round-overhead diagnostics (perf/cpu-round-overhead.md). Runs the prod pick with
# LLAMA_DECODE_PROF=1 so llama-context prints the per-context CPU split of every small
# decode (apply/reuse/set_inputs/submit/rest) - target AND drafter ctx - plus the server
# spec-prof dump (dec_sub_tg et al) and the graphs-reused counter.
#
# Timings here are valid: LLAMA_DECODE_PROF adds two ggml_time_us() calls per decode and
# does NOT serialize the GPU (unlike GGML_METAL_PROFILE, which must stay out - README).
# t/s from these runs are anchors for sanity, not headlines; headline = run-prod-pick.sh.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
ARMS=${ARMS:-}
PORT=8093
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-cpuovh-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

# The prod pick env, copied from run-prod-pick.sh (keep in sync), plus the decode profiler.
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          LLAMA_DECODE_PROF=1)

PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 4)
BASE_SPEC=(--spec-type none)

echo "=== cpu-overhead diagnostics: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "env    : ${PICK_ENV[*]}"
echo

run_one() {
  local label=$1 npred=$2 specname=$3 envname=${4:-PICK_ENV}
  if [ -n "$ARMS" ]; then
    case " $ARMS " in *" $label "*) ;; *) return 0 ;; esac
  fi
  local slog="$OUT/$TAG-$label.server.log"
  local -a specv envv
  eval "specv=(\"\${$specname[@]}\")"
  eval "envv=(\"\${$envname[@]}\")"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  env "${envv[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    "${specv[@]}" --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }
  lsof -ti :$PORT 2>/dev/null | grep -qx "$pid" || {
    echo "[$label] ABORT: port $PORT is served by another process, not our server"
    kill -9 $pid 2>/dev/null; return 1; }

  python3 -c "
import json
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': $npred, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{})
c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
print('[%-16s] n_predict=%-4s %6.3f t/s  acc=%5.1f%%  n=%d  sha1=%s'
      % ('$label', '$npred', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null

  echo "  --- last decode-prof lines per ctx ($slog) ---"
  awk '/decode-prof/ {last[$2]=$0} END {for (k in last) print "  " last[k]}' "$slog"
  echo "  --- submit-prof windows per ctx (first window includes warmup) ---"
  grep "submit-prof" "$slog" | sed 's/^/  /'
  echo "  --- last spec-prof dump ---"
  grep "spec-prof" "$slog" | tail -8 | sed 's/^/  /'
  grep "graphs reused" "$slog" | tail -1 | sed 's/^/  /'
  echo
  sleep 5
}

# dprof arms: decode-prof only (the timing baseline). sprof arms add the GPU-timeline
# submit profiler; its handlers are cheap but it is kept out of the baseline arms anyway.
run_one "dprof-n4" 300 PICK_SPEC
run_one "dprof-b1" 300 BASE_SPEC
# 600 units so the target ctx fills at least two 64-graph windows (the second is clean)
SPROF_ENV=("${PICK_ENV[@]}" GGML_METAL_SUBMIT_PROF=1)
run_one "sprof-n4" 600 PICK_SPEC SPROF_ENV
run_one "sprof-b1" 300 BASE_SPEC SPROF_ENV

# get-memcpy A/B: deferred host memcpy vs the upstream blit for logits readback.
# GET_MEMCPY is in the pick since 2026-08-28 pm, so the ctrl arms now disable it
# explicitly (a trailing =0 wins). Interleaved reps - run-to-run spread is ~2%.
GMC_OFF_ENV=("${PICK_ENV[@]}" GGML_METAL_GET_MEMCPY=0)
run_one "gmc-ctrl-a" 600 PICK_SPEC GMC_OFF_ENV
run_one "gmc-on-a"   600 PICK_SPEC
run_one "gmc-ctrl-b" 600 PICK_SPEC GMC_OFF_ENV
run_one "gmc-on-b"   600 PICK_SPEC
run_one "gmc-b1-ctrl" 300 BASE_SPEC GMC_OFF_ENV
run_one "gmc-b1-on"   300 BASE_SPEC
