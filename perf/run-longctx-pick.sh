#!/bin/bash
# Long-context run of the CURRENT pick (2026-09-15): reads perf/pick.sh (line + cache), one fresh server,
# a prompt file and context of your choosing. Reports prompt eval s and t/s, decode t/s, acceptance, the
# spec-prof round time, and the output sha. Supersedes run-longctx.sh (which carries a hardcoded env of Sep 6).
#   LINE=ud KV=turbo4 CTX=102400 PROMPT=... NPRED=300 DEPTH=3 TAG=... EXTRA_ENV="GGML_METAL_PROFILE=1" perf/run-longctx-pick.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
LINE=${LINE:-ud}
KV=${KV:-turbo4}
CTX=${CTX:-102400}
PROMPT=${PROMPT:-/Users/troff/play/kvquant-experiments/data/longprompt-96k.txt}
NPRED=${NPRED:-300}
DEPTH=${DEPTH:-3}
PORT=${PORT:-8096}
LV=${LV:-}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-longctx-pick-$(date +%m%d-%H%M)}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p "$OUT"
slog="$OUT/$TAG.server.log"

source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
PROMPT=$(pick_prompt "$PROMPT")  # chat-templated since 2026-09-17 evening; PICK_CHAT=0 = the raw lineage
pick_env "$LINE" "$KV"
pick_args "$LINE" "$KV"
# context is the caller's, not the manifest's default
ARGS=(); skip=0
for a in "${PICK_ARGS[@]}"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  if [ "$a" = -c ]; then skip=1; continue; fi
  ARGS+=("$a")
done
if [ "$DEPTH" = 0 ]; then spec=(--spec-type none); else spec=("${PICK_SPEC[@]:0:2}" --spec-type draft-dflash --spec-draft-n-max "$DEPTH"); fi

echo "=== longctx-pick $TAG: line=$LINE kv=$KV ctx=$CTX depth=$DEPTH n_predict=$NPRED prompt=$(wc -c < "$PROMPT" | tr -d ' ') bytes"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD); extra: $EXTRA_ENV"
echo "model  : $PICK_MODEL"
echo "prompt : $PROMPT (PICK_CHAT=$PICK_CHAT)"
echo "env    : ${PICK_ENV[*]}"
for i in $(seq 1 90); do lsof -ti :$PORT >/dev/null 2>&1 || break; sleep 2; done
if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi

env "${PICK_ENV[@]}" $EXTRA_ENV "$BIN/llama-server" -m "$PICK_MODEL" -c "$CTX" "${ARGS[@]}" \
  "${spec[@]}" $EXTRA_ARGS ${LV:+-lv "$LV"} --port $PORT >"$slog" 2>&1 &
pid=$!; ok=0
echo "pid    : $pid  (log: $slog)"
for i in $(seq 1 300); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
  sleep 2; kill -0 $pid 2>/dev/null || { echo "server died:"; tail -6 "$slog"; exit 1; }
done
[ $ok = 1 ] || { echo "health timeout"; kill -9 $pid; exit 1; }
t0=$(date +%s)
python3 -c "
import json
print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -H "Content-Type: application/json" -X POST "http://127.0.0.1:$PORT/completion" -d @- --max-time "${MAXTIME:-14400}" > "$OUT/$TAG.json"
python3 -c "
import json,hashlib
d=json.load(open('$OUT/$TAG.json'))
if 'error' in d: print('ERROR', json.dumps(d['error'])[:300])
else:
    t=d.get('timings',{}); c=d.get('content','')
    acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
    sha=hashlib.sha1(c.encode()).hexdigest()[:12]
    open('/tmp/longctx-$TAG.txt','w').write(c)
    print('[%s] prompt %d tok  %.1f s  %.1f t/s | decode %.3f t/s  acc=%.1f%%  n=%d | sha1=%s' % ('$TAG', t.get('prompt_n',0), t.get('prompt_ms',0)/1000, t.get('prompt_per_second',0), t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
echo "wall   : $(( $(date +%s) - t0 )) s"
grep -E 'spec-prof (round|loop_body)' "$slog" | tail -2 | sed -E 's/^[0-9.]+ I srv +operator\(\): //'
kill -TERM $pid 2>/dev/null; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
echo "server log: $slog"
