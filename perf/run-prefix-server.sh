#!/bin/bash
# perf/run-prefix-server.sh - one pick server for the prefix-slot-save work (perf/prefix-slot-saves.md), in the foreground:
# start it detached (nohup ... &), drive it with perf/prefix-save.py, stop it by pid. bash on purpose (pick.sh).
#   B=<tree> LINE=ud CTX=32768 DEPTH=3 SLOTDIR=<dir> PORT=8098 perf/run-prefix-server.sh
# DEPTH pins the verify width (no controller) so texts compare by sha; EXTRA_ENV / EXTRA_ARGS are passed through.
set -u
B=${B:?export B=<tree>}; BIN=${BIN:-$B/build/bin}
LINE=${LINE:-ud}; KV=${KV:-turbo4}; CTX=${CTX:-32768}; DEPTH=${DEPTH:-3}; PORT=${PORT:-8098}; LV=${LV:-3}
SLOTDIR=${SLOTDIR:-/Users/troff/play/kvquant-experiments/slots/prefix}; EXTRA_ENV=${EXTRA_ENV:-}; EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p "$SLOTDIR"
[ -n "$DEPTH" ] && export PICK_SPEC_EV=0
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"
SPEC=("${PICK_SPEC[@]}"); [ -n "$DEPTH" ] && SPEC=("${PICK_SPEC[@]:0:2}" --spec-type draft-dflash --spec-draft-n-max "$DEPTH")
if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
echo "=== prefix server: line=$LINE kv=$KV ctx=$CTX depth=${DEPTH:-controller} slots=$SLOTDIR commit $(git -C "$B" rev-parse --short HEAD) binary $(date -r "$BIN/llama-server" '+%m-%d %H:%M') extra: ${EXTRA_ENV:-none} ${EXTRA_ARGS}"
exec env "${PICK_ENV[@]}" $EXTRA_ENV caffeinate -dimsu "$BIN/llama-server" -m "$PICK_MODEL" -c "$CTX" "${PICK_ARGS[@]:2}" "${SPEC[@]}" \
    -np 1 --slot-save-path "$SLOTDIR" -lv "$LV" --port "$PORT" $EXTRA_ARGS
