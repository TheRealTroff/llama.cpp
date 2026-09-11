#!/bin/bash
# quick sha + speed check for the small-op fusion tree: 3000-char prompt, NPRED tokens, temp 0.
# NOT the canonical harness (run-prod-pick.sh + benchprompt.txt); shas here are only comparable to each other.
set -u
B=${B:-/Users/troff/play/llama.cpp-fuse}
TAG=${TAG:-fq-$(date +%H%M%S)}
EXTRA=${EXTRA:-}
PICK_LINES=${PICK_LINES:-"ud q4"}   # not LINES: bash owns that name (terminal rows) and drops a two-word value
ARMS=${ARMS:-"turbo4-n3 batch1"}
NPRED=${NPRED:-96}
PROMPT=${PROMPT:-$B/perf/prompts/quickprompt-3000.txt}
PORT=${PORT:-8097}
OUT=/Users/troff/play/kvquant-experiments/results/fuse-quick
mkdir -p "$OUT"
source "$B/perf/pick.sh"
echo "=== $TAG  tree=$B commit=$(cd "$B" && git rev-parse --short HEAD) extra='$EXTRA'"
for line in $PICK_LINES; do
  for arm in $ARMS; do
    if [ "$arm" = turbo4-n3 ]; then pick_env "$line" turbo4; pick_args "$line" turbo4; spec=("${PICK_SPEC[@]}")
    else pick_env "$line" f16; pick_args "$line" f16; spec=(--spec-type none); fi
    label=$TAG-$line-$arm
    if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port busy"; exit 1; fi
    env "${PICK_ENV[@]}" $EXTRA "$B/build/bin/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" "${spec[@]}" ${LV:+-lv "$LV"} --port $PORT > "$OUT/$label.server.log" 2>&1 &
    pid=$!; ok=0
    for i in $(seq 1 150); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }; sleep 2; kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -3 "$OUT/$label.server.log"; break; }; done
    [ $ok = 1 ] || { kill -9 $pid 2>/dev/null; continue; }
    python3 -c "import json;print(json.dumps({'prompt':open('$PROMPT').read(),'n_predict':$NPRED,'temperature':0}))" \
      | curl -s -H "Content-Type: application/json" -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d: print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
open('$OUT/$label.txt','w').write(c)
print('[%-34s] %7.3f t/s  pp %6.1f t/s  acc=%5.1f%%  n=%d  sha1=%s' % ('$label', t.get('predicted_per_second',0), t.get('prompt_per_second',0), acc, t.get('predicted_n',0), hashlib.sha1(c.encode()).hexdigest()[:12]))"
    kill -TERM $pid 2>/dev/null; for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
    sleep 2
  done
done
