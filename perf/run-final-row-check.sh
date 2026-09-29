#!/bin/bash
set -euo pipefail
B=${B:-$(cd "$(dirname "$0")/.." && pwd)}
LINE=${LINE:-q4}
ARM=${ARM:-0}
DEPTH=${DEPTH:-3}
TAG=${TAG:-final-row-$LINE-d$DEPTH-$ARM}
PORT=${PORT:-8107}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/work-elimination-20260928}
export PICK_SPEC_EV=0 PICK_DEPTH=$DEPTH
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE"
pick_args "$LINE"
mkdir -p "$OUT"
if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "Port $PORT busy"; exit 1; fi
printf 'line=%s arm=%s depth=%s mode=%s\n' "$LINE" "$ARM" "$DEPTH" "${MODE:-suite}"
printf 'env: %s\n' "${PICK_ENV[*]} LLAMA_QWEN35_PRUNE_EMPTY_TAIL=$ARM"
printf 'args: %s\n' "-m $PICK_MODEL ${PICK_ARGS[*]} ${PICK_SPEC[*]} -np 1 -lv ${LV:-3} --port $PORT"
env "${PICK_ENV[@]}" LLAMA_QWEN35_PRUNE_EMPTY_TAIL="$ARM" GGML_METAL_SYNC_TIMEOUT=15 \
    "$B/build/bin/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" "${PICK_SPEC[@]}" \
    -np 1 -lv "${LV:-3}" --port "$PORT" > "$OUT/$TAG.server.log" 2>&1 &
pid=$!
cleanup() {
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
for ((i=0; i<200; i++)); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ready=1; break; fi
    kill -0 "$pid" || exit 1
    sleep 1
done
[ "$ready" = 1 ]
lsof -ti :"$PORT" | grep -qx "$pid"
python3 "$B/perf/final-row-driver.py" "$PORT" "$OUT/$TAG.json"
if rg -q 'sync-guard: backend|failed to process speculative batch' "$OUT/$TAG.server.log"; then exit 1; fi
