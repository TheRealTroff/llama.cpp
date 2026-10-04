#!/bin/bash
# perf/run-session.sh - the session corpus harness (perf/session-corpus.md): record an agent session with the model
# itself, or replay a recorded script teacher-forced under several arms. bash on purpose (pick.sh).
#   MODE=record LINE=ud ARMS=d3 USER_SCRIPT=perf/session/pilot10.user.json SCRIPT=perf/session/pilot10.json perf/run-session.sh
#   MODE=replay LINE=ud ARMS="ev d3" PASSES=2 SCRIPT=perf/session/pilot10.json perf/run-session.sh
# Arms: ev = the line's pick (the LLAMA_SPEC_EV controller), dN = pinned draft depth N (verify width N+1, no controller),
#       nospec = no drafter. One server per arm and pass, arms interleaved inside a pass. TURNS=a-b replays a range.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:?export B=<tree>}; BIN=${BIN:-$B/build/bin}
MODE=${MODE:-replay}; LINE=${LINE:-ud}; KV=${KV:-turbo4}; ARMS=${ARMS:-ev d3}; PASSES=${PASSES:-1}
SCRIPT=${SCRIPT:?SCRIPT=<transcript json>}; USER_SCRIPT=${USER_SCRIPT:-}; TURNS=${TURNS:-}; NPRED=${NPRED:-0}
ROOT=${ROOT:-/Users/troff/play/kvquant-experiments/data/session-repo}
PORT=${PORT:-8098}; LV=${LV:-3}; COOL=${COOL:-5}; EXTRA_ENV=${EXTRA_ENV:-}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}; TAG=${TAG:-session-$LINE-$(date +%m%d-%H%M)}
mkdir -p "$OUT"
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
echo "=== session $MODE $TAG: line=$LINE kv=$KV ctx=${CTX:-pick} arms=[$ARMS] passes=$PASSES script=$SCRIPT turns=${TURNS:-all}"
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD); binary $BIN/llama-server $(date -r "$BIN/llama-server" '+%m-%d %H:%M'); extra: ${EXTRA_ENV:-none}"
pid=
stop_server() { if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; fi; pid=; }
trap stop_server EXIT INT TERM
start_server() {  # start_server <arm> <label>
  local arm=$1 label=$2 spec
  case "$arm" in
    ev)     export PICK_SPEC_EV=1 ;;
    d[1-7]|nospec) export PICK_SPEC_EV=0 ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"
  if [ -n "${CTX:-}" ]; then PICK_ARGS=(-c "$CTX" "${PICK_ARGS[@]:2}"); fi   # CTX=<n> replaces the pick's -c (a script longer than the pick's context)
  case "$arm" in
    ev)     spec=("${PICK_SPEC[@]}") ;;
    nospec) spec=() ;;
    *)      spec=(-md "$PICK_DRAFTER" --spec-type draft-dflash --spec-draft-n-max "${arm#d}") ;;
  esac
  if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
  slog=$OUT/$TAG-$label.server.log
  env "${PICK_ENV[@]}" $EXTRA_ENV LLAMA_SPEC_EV_TRACE="$OUT/$TAG-$label.picks" "$BIN/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" "${spec[@]}" \
      -lv "$LV" --port "$PORT" >"$slog" 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
    kill -0 "$pid" 2>/dev/null || { echo "[$label] server died:"; tail -5 "$slog"; exit 1; }
    sleep 2
  done
  echo "[$label] health timeout"; exit 1
}
if [ "$MODE" = record ]; then
  arm=${ARMS%% *}
  WORK=${WORK:-/Users/troff/play/kvquant-experiments/data/session-work}   # the tools write here: a fresh clone of the pinned tree per recording
  rm -rf "$WORK" && cp -cR "$ROOT" "$WORK" || exit 1
  start_server "$arm" record
  echo "--- record under arm $arm (${#PICK_ENV[@]} pick flags)"
  python3 "$B/perf/session.py" record --user "${USER_SCRIPT:?USER_SCRIPT=<user script json>}" --root "$WORK" --out "$SCRIPT" --port "$PORT" \
      --note "line=$LINE kv=$KV arm=$arm commit=$(git -C "$B" rev-parse --short HEAD)" ${EFFORT:+--effort "$EFFORT"} ${TEMPLATE_KWARGS:+--template-kwargs "$TEMPLATE_KWARGS"} ${CTX_LIMIT:+--ctx-limit "$CTX_LIMIT"} ${MAX_STEPS:+--max-steps "$MAX_STEPS"} ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"}
  stop_server
  exit 0
fi
files=()
for pass in $(seq 1 "$PASSES"); do
  for arm in $ARMS; do
    label=$arm-p$pass
    start_server "$arm" "$label"
    echo "--- [$label]"
    python3 "$B/perf/session.py" replay --script "$SCRIPT" --out "$OUT/$TAG-$label.json" --label "$label" --port "$PORT" \
        ${TURNS:+--turns "$TURNS"} --max-tokens "$NPRED" | tee "$OUT/$TAG-$label.txt" | grep -v "^arm \|^$label "
    grep -h "spec-ev" "$slog" | tail -3 | sed 's/^/    /'
    stop_server
    files+=("$OUT/$TAG-$label.json")
    sleep "$COOL"
  done
done
echo; echo "=== $TAG"
python3 "$B/perf/session.py" report "${files[@]}"
