#!/bin/bash
# perf/run-effort-pos.sh - the effort-line-position experiment end to end (perf/effort-line-position.md), detached:
#   nohup env B=/Users/troff/play/llama.cpp-prod E=/Users/troff/play/llama.cpp-effortpos TAG=effortpos-oct03 \
#     perf/run-effort-pos.sh > kvquant-experiments/logs/effortpos-oct03.log 2>&1 & disown
# One pick server per template (stock = embedded, tail = perf/qwen3.8-effort-tail.jinja), ud line, Turbo4, depth 3
# pinned (texts compare by sha), one slot of CTX. Each server: /apply-template renders of every prompt and level,
# then the three levels of every prompt at temp 0, max_tokens 8192 (xhigh can think 16K+ on a design prompt; a capped cell
# still sorts into the xhigh bucket). B = the tree whose binary and pick.sh serve; E = this experiment tree.
set -u
B=${B:?export B=<serving tree>}; E=${E:-$(cd "$(dirname "$0")/.." && pwd)}
TAG=${TAG:-effortpos-$(date +%b%d | tr A-Z a-z)}; LINE=${LINE:-ud}; CTX=${CTX:-32768}; DEPTH=${DEPTH:-3}; PORT=${PORT:-8101}
LEVELS=${LEVELS:-xhigh,low,medium}; TMPLS=${TMPLS:-stock tail}; PROMPTS=${PROMPTS:-$E/perf/effort-pos-prompts.json}; MAX_TOKENS=${MAX_TOKENS:-8192}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/effortpos/$TAG}; mkdir -p "$OUT"
SLOTDIR=$OUT/slots; mkdir -p "$SLOTDIR"

wait_health() {  # wait_health <port> <pid>
  local i
  for i in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || { echo "server $2 died"; return 1; }
    sleep 2
  done
  echo "server on $1 never became healthy"; return 1
}

for TMPL in $TMPLS; do
  EXTRA_ARGS=""
  [ "$TMPL" = tail ] && EXTRA_ARGS="--chat-template-file $E/perf/qwen3.8-effort-tail.jinja"
  [ "$TMPL" = sharp ] && EXTRA_ARGS="--chat-template-file $E/perf/sharp_chat_template.jinja"   # froggeric v22.1: terseness block at the end of the system text, effort line still at the head
  echo "=== $(date '+%H:%M:%S') template $TMPL: starting server on $PORT ($EXTRA_ARGS)"
  B=$B LINE=$LINE CTX=$CTX DEPTH=$DEPTH PORT=$PORT SLOTDIR=$SLOTDIR EXTRA_ARGS="$EXTRA_ARGS" \
    "$B/perf/run-prefix-server.sh" > "$OUT/server-$TMPL.log" 2>&1 &
  SPID=$!
  wait_health "$PORT" "$SPID" || exit 1
  grep -m1 -o 'chat_template.*' "$OUT/server-$TMPL.log" | head -c 200; echo
  python3 "$E/perf/effort-pos.py" render --port "$PORT" --prompts "$PROMPTS" --levels "$LEVELS" --out "$OUT/render-$TMPL.json"
  if [ -n "${VARIANTS:-}" ]; then   # user-message variants on this template at level medium (effort-pos.py VARIANTS)
    python3 "$E/perf/effort-pos.py" run --port "$PORT" --prompts "$PROMPTS" --tmpl "$TMPL" --variants "$VARIANTS" --max-tokens "$MAX_TOKENS" --out "$OUT"
  else
    python3 "$E/perf/effort-pos.py" run --port "$PORT" --prompts "$PROMPTS" --tmpl "$TMPL" --levels "$LEVELS" --max-tokens "$MAX_TOKENS" --out "$OUT"
  fi
  echo "=== $(date '+%H:%M:%S') template $TMPL done; stopping server $SPID"
  kill "$SPID"; wait "$SPID" 2>/dev/null
  sleep 5
done
python3 "$E/perf/effort-pos.py" report "$OUT" | tee "$OUT/report.md"
echo "=== $(date '+%H:%M:%S') all done -> $OUT"
