#!/bin/bash
# Step 1 of perf/drafter-pipelining.md: does dropping the explicit sync after the drafter
# inject decode buy anything?
#
# DFLASH_ASYNC_INJECT=1 removes `llama_synchronize(ctx_dft)` after the inject decode in
# process() (common/speculative.cpp) and in apply_window()'s flush. The wait is not removed,
# only moved to the next read of the drafter output, so the win is whatever CPU work sits
# between process() and the next draft() - the accept/sampling path. Expect small.
#
# Safety argument for the removal (see drafter-pipelining.md section 5): llama_decode copies
# the batch into device memory before returning, so the host batch_inject buffer is free to
# reuse; and both contexts share ONE Metal queue (ggml-metal-context.m:228), so graphs still
# execute in submit order.
#
# THE CORRECTNESS CHECK IS THE POINT, not the t/s: every arm at the same n_predict must emit
# the SAME output sha1. A differing sha means the async path raced, and the t/s is worthless.
#
# Arms alternate off/on/off/on so thermal drift cannot be read as an effect.
set -u

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PORT=8094
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-asyncinj-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)
PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 6)

echo "=== async-inject A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "env    : ${PICK_ENV[*]}"
echo "n_pred : $NPRED"
echo

run_one() {
  local label=$1 inj=$2
  local slog="$OUT/$TAG-$label.server.log"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  env "${PICK_ENV[@]}" DFLASH_ASYNC_INJECT=$inj "$BIN/llama-server" -m "$M" -c 10240 -fa on \
    -ctk f16 -ctv f16 "${PICK_SPEC[@]}" --port $PORT >"$slog" 2>&1 &
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

  # confirm the flag actually reached the drafter - it is silent when unset
  local seen
  seen=$(grep -c "drafter async inject" "$slog" 2>/dev/null || true)
  if [ "$inj" = "1" ] && [ "$seen" = "0" ]; then
    echo "[$label] WARNING: server never logged 'drafter async inject' - flag not read"
  fi

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
open('/tmp/asyncinj-$label.txt','w').write(c)
print('[%-14s] inject=%s %7.3f t/s  acc=%5.1f%%  n=%d  sha1=%s'
      % ('$label', '$inj', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
  # the in-tree profiler prints inject+sync every 32 calls; last line is the steady state
  grep "dflash-prof process" "$slog" 2>/dev/null | tail -1 | sed 's/^/    /'

  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 20   # cooldown, matching the other harnesses
}

run_one "off-r1" 0
run_one "on-r1"  1
run_one "off-r2" 0
run_one "on-r2"  1

echo
echo "--- output identity: ALL FOUR must share one sha, or the async path raced ---"
for f in /tmp/asyncinj-*.txt; do
  echo "  $(shasum "$f" | cut -c1-12)  $(wc -c <"$f" | tr -d ' ') B  $f"
done
echo
echo "distinct shas: $(shasum /tmp/asyncinj-*.txt | cut -c1-12 | sort -u | wc -l | tr -d ' ') (must be 1)"
