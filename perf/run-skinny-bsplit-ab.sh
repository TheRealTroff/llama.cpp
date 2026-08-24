#!/bin/bash
# A/B for GGML_MM_SKINNY_BSPLIT on the prod pick. The flag spreads mul_mm_skinny's B-tile
# load over all 32*TPR threads instead of a fixed 32; per-shape it is -1.4% to -2.6% on
# every projection at width 7 (skinny-tpr-refuted.md), which weights to ~-2.2 ms/round.
#
# Alternating arms in ONE process, because cross-session drift is ~3% and within-session
# repeats agree to <1% - a 1.4% effect is not measurable across sessions. n_predict 600 for
# the low-variance units (~166 rounds, controls spread ~0.02 t/s).
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=8093
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-bsplit-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
REPS=${REPS:-2}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)
# REPACK=1 routes skinny to the _di kernel, which is the only way to exercise its B stage:
# GGML_MV_REPACK is silently inert in test-backend-ops (README trap 3), so the _di path has
# no per-shape arm and e2e output identity is its correctness check.
[ "${REPACK:-0}" = 1 ] && PICK_ENV+=(GGML_MV_REPACK=1)
PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 6)

echo "=== skinny B-split A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "env    : ${PICK_ENV[*]}  (+ GGML_MM_SKINNY_BSPLIT per arm)"
echo "npred  : $NPRED, $REPS repeats per arm, alternating"
echo

run_one() {
  local label=$1 bsp=$2
  local slog="$OUT/$TAG-$label.server.log"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  env "${PICK_ENV[@]}" GGML_MM_SKINNY_BSPLIT="$bsp" "$BIN/llama-server" -m "$M" -c 10240 \
    -fa on -ctk f16 -ctv f16 "${PICK_SPEC[@]}" --port $PORT >"$slog" 2>&1 &
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

  # the kernel must actually be the one under test: bsp=1 compiles a _bsp=1 pipeline
  local tell
  tell=$(grep -o "kernel_mul_mm_skinny_q4_0_f32_[^ ]*" "$slog" | tail -1)

  python3 -c "
import json
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{})
c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
print('[%-14s] %6.3f t/s  acc=%5.1f%%  n=%d  sha1=%s  %s'
      % ('$label', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha, '$tell'))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

for r in $(seq 1 "$REPS"); do
  run_one "control-r$r" 0
  run_one "bsplit-r$r"  1
done
