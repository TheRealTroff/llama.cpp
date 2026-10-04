#!/bin/bash
# perf/serve.sh - serve a pick line for real use, from the manifest (perf/pick.sh), detached. Owner-runnable, no agent needed.
#
#   perf/serve.sh q4            start the q4 line (192.168.64.1:8093)
#   perf/serve.sh ud            start the ud line (192.168.64.1:8094)
#   perf/serve.sh q4 stop       stop it (by pid file, else by port)
#   perf/serve.sh q4 status     pid, port, /health, the last log lines
#   perf/serve.sh q4 log        follow the log
#
# Knobs (environment): HOST, PORT, KV=turbo4|f16, CTX=<tokens> (replaces the pick's -c), NP=<slots> (-np, CTX each),
#   SIZES=a,b,c (--ctx-seq-sizes, overrides CTX/NP), EFFORT=medium (template default when a client sends no reasoning_effort;
#   EFFORT=default keeps the template's own xhigh), PREFIX=1 (prefix slot saves of agent base prompts, LLAMA_PREFIX_DIR +
#   LLAMA_PREFIX_AUTO - perf/prefix-slot-saves.md), B=<tree> (default llama.cpp-prod), LV=<log level>, EXTRA_ARGS="..."
#
# The pick's env flags, model, KV args and the speculative controller come from pick.sh, exactly as the mints use them.
# pick_check refuses a shell environment carrying a numerics flag outside the line's manifest. The harness port is 8093:
# run-prod-pick.sh aborts while a server holds it, so stop the q4 server before a mint.
set -u
LINE=${1:?usage: perf/serve.sh <q4|ud> [start|stop|status|log]}; CMD=${2:-start}
B=${B:-/Users/troff/play/llama.cpp-prod}; BIN=${BIN:-$B/build/bin}
HOST=${HOST:-192.168.64.1}; KV=${KV:-turbo4}; LV=${LV:-3}; EFFORT=${EFFORT:-medium}
case "$LINE" in q4) PORT=${PORT:-8093} ;; ud) PORT=${PORT:-8094} ;; *) echo "serve.sh: line must be q4 or ud" >&2; exit 1 ;; esac
RUN=/Users/troff/play/kvquant-experiments/run; LOGS=/Users/troff/play/kvquant-experiments/logs; mkdir -p "$RUN" "$LOGS"
PIDFILE=$RUN/serve-$LINE.pid; LOG=$LOGS/serve-$LINE.log
SLOTDIR=${SLOTDIR:-/Users/troff/play/kvquant-experiments/slots/serve-$LINE}

pid_of() {  # the server's pid from the pid file if alive, else whoever listens on the port
  local p
  if [ -f "$PIDFILE" ]; then p=$(cat "$PIDFILE"); kill -0 "$p" 2>/dev/null && { echo "$p"; return 0; }; fi
  lsof -ti :"$PORT" 2>/dev/null | head -1
}

case "$CMD" in
  stop)
    p=$(pid_of); [ -z "$p" ] && { echo "serve.sh: no $LINE server (port $PORT)"; rm -f "$PIDFILE"; exit 0; }
    kill -TERM "$p"; for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && { echo "still alive after 60 s, SIGKILL"; kill -9 "$p"; }
    rm -f "$PIDFILE"; echo "stopped $LINE server $p"; exit 0 ;;
  status)
    p=$(pid_of); if [ -z "$p" ]; then echo "$LINE: not running (port $PORT)"; exit 1; fi
    echo "$LINE: pid $p on $HOST:$PORT, health: $(curl -sf "http://$HOST:$PORT/health" || echo unreachable)"
    echo "started: $(ps -o lstart= -p "$p")"; echo "log: $LOG"; tail -3 "$LOG"; exit 0 ;;
  log) exec tail -f "$LOG" ;;
  start) ;;
  *) echo "serve.sh: unknown command $CMD" >&2; exit 1 ;;
esac

if p=$(pid_of) && [ -n "$p" ]; then echo "serve.sh: a server already holds port $PORT (pid $p); 'perf/serve.sh $LINE stop' first" >&2; exit 1; fi
[ -x "$BIN/llama-server" ] || { echo "serve.sh: no binary at $BIN/llama-server (build the tree first)" >&2; exit 1; }
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"
if [ -n "${SIZES:-}" ]; then CTXARGS=(--ctx-seq-sizes "$SIZES")
elif [ -n "${CTX:-}" ] || [ -n "${NP:-}" ]; then CTXARGS=(-c $(( ${CTX:-${PICK_ARGS[1]}} * ${NP:-1} )) -np "${NP:-1}")
else CTXARGS=("${PICK_ARGS[@]:0:2}"); fi
ARGS=("${CTXARGS[@]}" "${PICK_ARGS[@]:2}" "${PICK_SPEC[@]}")
[ "$EFFORT" != default ] && ARGS+=(--reasoning-effort "$EFFORT")   # perf/effort-line-position.md: a request with no level renders no sentence
ENV=("${PICK_ENV[@]}" GGML_METAL_SYNC_TIMEOUT=${SYNC_TIMEOUT:-120})
if [ "${PREFIX:-0}" = 1 ]; then mkdir -p "$SLOTDIR/prefix"; ENV+=(LLAMA_PREFIX_DIR="$SLOTDIR/prefix" LLAMA_PREFIX_AUTO=1); fi
mkdir -p "$SLOTDIR"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') serve $LINE: $HOST:$PORT kv=$KV effort=$EFFORT prefix=${PREFIX:-0} tree $B @ $(git -C "$B" rev-parse --short HEAD) binary $(date -r "$BIN/llama-server" '+%m-%d %H:%M')"
  echo "env : ${ENV[*]}"
  echo "args: ${ARGS[*]} ${EXTRA_ARGS:-}"
} >> "$LOG"
nohup env "${ENV[@]}" caffeinate -dimsu "$BIN/llama-server" -m "$PICK_MODEL" "${ARGS[@]}" --slot-save-path "$SLOTDIR" \
    -lv "$LV" --host "$HOST" --port "$PORT" ${EXTRA_ARGS:-} >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"; disown
echo "$LINE server starting: pid $(cat "$PIDFILE") on $HOST:$PORT, log $LOG"
for i in $(seq 1 300); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && { echo "healthy after $((i*2)) s"; exit 0; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server died:"; tail -5 "$LOG"; rm -f "$PIDFILE"; exit 1; }
  sleep 2
done
echo "not healthy after 600 s; see $LOG"; exit 1
