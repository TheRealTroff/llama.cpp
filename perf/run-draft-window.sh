#!/bin/bash
# Drafter attention window probe (perf/drafter-graph-count.md, reframed question 2):
# the drafter's 5 FA calls/round run over the FULL ~8.4k KV (~1.6 ms/round profiled,
# ~2.3x stream floor, grows with context). LLAMA_DRAFT_WINDOW exists but was never
# benchmarked at the pick. Text is verify-gated so the canonical shas hold under ANY
# drafting change; the tradable quantity is acceptance vs the FA (and KV) savings.
# Note: the window path disables DFLASH_FUSED_INJECT (ring needs g on host), so these
# arms measure the window against plain ctrl - no fused/async in any arm.
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
TAG=${TAG:-draftwin-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1)
W512_ENV=("${PICK_ENV[@]}" LLAMA_DRAFT_WINDOW=512)
W1024_ENV=("${PICK_ENV[@]}" LLAMA_DRAFT_WINDOW=1024)
W2048_ENV=("${PICK_ENV[@]}" LLAMA_DRAFT_WINDOW=2048)

PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 4)

echo "=== drafter window probe: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo

run_one() {
  local label=$1 npred=$2 envname=$3
  if [ -n "$ARMS" ]; then
    case " $ARMS " in *" $label "*) ;; *) return 0 ;; esac
  fi
  local slog="$OUT/$TAG-$label.server.log"
  local -a envv
  eval "envv=(\"\${$envname[@]}\")"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  env "${envv[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    "${PICK_SPEC[@]}" --port $PORT >"$slog" 2>&1 &
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
open('/tmp/draftwin-$label.txt','w').write(c)
print('[%-10s] n_predict=%-4s %6.3f t/s  acc=%5.1f%%  n=%d  sha1=%s'
      % ('$label', '$npred', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
  grep -h "dflash-prof lattice\|drafter context window" "$slog" | tail -2 | sed 's/^/    /'
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

echo "--- interleaved at 600 (canonical 3776c0adb7ee must hold in EVERY arm) ---"
run_one "ctrl-a"   600 PICK_ENV
run_one "w512-a"   600 W512_ENV
run_one "w1024-a"  600 W1024_ENV
run_one "ctrl-b"   600 PICK_ENV
run_one "w512-b"   600 W512_ENV
run_one "w1024-b"  600 W1024_ENV
run_one "ctrl-c"   600 PICK_ENV
run_one "w2048-a"  600 W2048_ENV
run_one "ctrl-d"   600 PICK_ENV
run_one "w2048-b"  600 W2048_ENV

echo
echo "--- output identity ---"
for f in /tmp/draftwin-*.txt; do echo "  $(shasum "$f" | cut -c1-12)  $(wc -c <"$f" | tr -d ' ') B  $f"; done
