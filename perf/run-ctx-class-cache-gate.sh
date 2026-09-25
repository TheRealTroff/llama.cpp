#!/bin/bash
# The prompt-cache gate under per-slot context sizes (2026-09-25): four slots of k, 2k, 4k, 8k tokens
# (--ctx-seq-sizes, exp/kv-size-classes) and five multi-turn chat streams that exercise the attention-KV prefix
# reuse, the recurrent (GDN) checkpoints, the host-RAM prompt cache (--cache-ram) and the class routing, under the
# line's Turbo4 pick at fixed DFlash depth 3. Two fresh servers: the cached plan, then the control server that
# replays A1-A3 in slot (the RAM-cache round trip must be byte-identical) and every prompt uncached (the reference).
# Driver and plan: perf/ctx-class-cache-driver.py. Note: perf/ctx-class-cache.md.
#   LINE=q4 K=4096 bash perf/run-ctx-class-cache-gate.sh        (B= the tree whose pick.sh runs; BIN= the binary; PORT= default 8099)
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
SD=$(cd "$(dirname "$0")" && pwd)
LINE=${LINE:-q4}
KV=${KV:-turbo4}
K=${K:-4096}
PORT=${PORT:-8099}
NPRED=${NPRED:-48}
SYNC_TIMEOUT=${SYNC_TIMEOUT:-15}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ctxcache-$(date +%m%d-%H%M)}
CONTROL=${CONTROL:-full}   # short = the control server replays A1-A3 in slot only (no uncached reference; ~10 min per run)
mkdir -p "$OUT"

stuck_servers() { ps -axo pid=,stat=,command= | awk '$2 ~ /E/ && $0 ~ /llama-server/ {print $1}'; }
if [ -n "$(stuck_servers)" ]; then echo "ABORT: llama-server pid(s) $(stuck_servers | tr '\n' ' ')stuck in kernel exit - reboot first"; exit 3; fi
kill_server() {
  local pid=$1
  kill -TERM $pid 2>/dev/null; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; for i in $(seq 1 10); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  if kill -0 $pid 2>/dev/null; then echo "  SERVER PID $pid SURVIVED SIGKILL - REBOOT before the next GPU run"; exit 3; fi
  wait $pid 2>/dev/null
}

export PICK_SPEC_EV=0   # fixed depth = one verify width per round: the cached and uncached arms run the same kernels
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" "$KV"
pick_args "$LINE" "$KV"
ARGS=(); skip=0
for a in "${PICK_ARGS[@]}"; do   # the context comes from the size list, not the manifest
  if [ $skip = 1 ]; then skip=0; continue; fi
  if [ "$a" = -c ]; then skip=1; continue; fi
  ARGS+=("$a")
done
SIZES="$K,$((2*K)),$((4*K)),$((8*K))"

echo "=== ctx-class cache gate $TAG: line=$LINE kv=$KV --ctx-seq-sizes $SIZES, depth ${PICK_DEPTH_LINE}, n_predict $NPRED, control=$CONTROL; extra: $EXTRA_ENV $EXTRA_ARGS"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD); binary $BIN; driver $SD"
echo "model  : $PICK_MODEL"
echo "env    : ${PICK_ENV[*]}"

start_server() {  # start_server <log> -> pid in $pid
  local slog=$1
  for i in $(seq 1 90); do lsof -ti :$PORT >/dev/null 2>&1 || break; sleep 2; done
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
  env "${PICK_ENV[@]}" GGML_METAL_SYNC_TIMEOUT=$SYNC_TIMEOUT $EXTRA_ENV "$BIN/llama-server" -m "$PICK_MODEL" "${ARGS[@]}" \
    --ctx-seq-sizes "$SIZES" "${PICK_SPEC[@]}" $EXTRA_ARGS -lv 5 --port $PORT >"$slog" 2>&1 &
  pid=$!
  for i in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
    sleep 2; kill -0 $pid 2>/dev/null || { echo "server died:"; tail -6 "$slog"; return 1; }
  done
  return 1
}
check_server() {  # after a phase: died? sync guard?
  local slog=$1
  grep -ciE "error|failed" "$slog" | sed 's/^/  server log error lines: /'
  if ! kill -0 $pid 2>/dev/null; then echo "  SERVER DIED DURING THE PHASE (log $slog)"; grep -m3 "sync-guard" "$slog" | sed 's/^/    /'; fi
  grep -q "sync-guard: backend" "$slog" && { echo "  THE SYNC GUARD FIRED - GPU hang. Stopping."; kill_server $pid; exit 2; }
}

slog1="$OUT/$TAG.cached.server.log"
echo; echo "--- server 1 (the cached plan), log $slog1"
start_server "$slog1" || { kill_server $pid; exit 1; }
grep -E "n_ctx_seq|kv_unified|llama_kv_cache: size|llama_memory_recurrent: size|prompt cache is|context checkpoints|n_rs_seq" "$slog1" | sed -E 's/^[0-9.]+ [A-Z] //' | head -12
t0=$(date +%s)
python3 "$SD/ctx-class-cache-driver.py" --port $PORT --phase cached --log "$slog1" --out "$OUT/$TAG.cached.json" --k "$K" --n-predict "$NPRED"
echo "  wall $(( $(date +%s) - t0 )) s"
check_server "$slog1"
kill_server $pid

slog2="$OUT/$TAG.control.server.log"
echo; echo "--- server 2 (control: in-slot A1-A3, then the uncached reference), log $slog2"
start_server "$slog2" || { kill_server $pid; exit 1; }
t0=$(date +%s)
python3 "$SD/ctx-class-cache-driver.py" --port $PORT --phase control --log "$slog2" --ref "$OUT/$TAG.cached.json" --out "$OUT/$TAG.control.json" --k "$K" --n-predict "$NPRED" $([ "$CONTROL" = short ] && echo --no-uncached)
rc=$?
echo "  wall $(( $(date +%s) - t0 )) s"
check_server "$slog2"
kill_server $pid
echo; echo "results: $OUT/$TAG.{cached,control}.json, logs $OUT/$TAG.*.server.log"
exit $rc
