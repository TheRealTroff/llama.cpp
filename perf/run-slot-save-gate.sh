#!/bin/bash
# perf/run-slot-save-gate.sh - prefill once, save the slot to disk, restore it in fresh server processes (2026-09-30,
# owner: "exercise the lovely write-slot-to-disk feature to not have to redo the prefill every time"; perf/slot-save-hybrid.md).
# Phase fresh : one server, the request twice (fresh prefill, then the in-process reuse = the verified RAM-cache path),
#               then POST /slots/0?action=save -> $SLOTDIR/$NAME{,.dft,.ckpt}.
# Phase restore: a fresh server per arm, POST /slots/0?action=restore, the same request; EXTRA_ENV per arm (GGML_METAL_PROFILE=1).
# Reports prompt_n (tokens actually prefilled), prompt ms, decode t/s, acceptance, spec-prof round, sha. bash on purpose.
#   B=<tree> LINE=ud CTX=16384 PROMPT=/Users/troff/play/benchprompt.txt NAME=ud-8k PHASES="fresh restore" perf/run-slot-save-gate.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:?export B=<tree>}; BIN=$B/build/bin
LINE=${LINE:-ud}; KV=${KV:-turbo4}; CTX=${CTX:-16384}; NPRED=${NPRED:-300}; DEPTH=${DEPTH:-}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
NAME=${NAME:-slot-$LINE}; SLOTDIR=${SLOTDIR:-/Users/troff/play/kvquant-experiments/slots}
PHASES=${PHASES:-fresh restore}; ARMS=${ARMS:-anchor}   # restore arms: anchor | metalprof (names; metalprof adds GGML_METAL_PROFILE=1)
PORT=${PORT:-8098}; LV=${LV:-3}; MAXTIME=${MAXTIME:-14400}
OUT=/Users/troff/play/kvquant-experiments/results; TAG=${TAG:-slotsave-$(date +%m%d-%H%M)}
EXTRA_ENV=${EXTRA_ENV:-}
mkdir -p "$OUT" "$SLOTDIR"
source "$B/perf/pick.sh"
PROMPT_FILE=$(pick_prompt "$PROMPT")
pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"
ARGS=(); skip=0
for a in "${PICK_ARGS[@]}"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  if [ "$a" = -c ]; then skip=1; continue; fi
  ARGS+=("$a")
done
SPEC=("${PICK_SPEC[@]}"); [ -n "$DEPTH" ] && SPEC=("${PICK_SPEC[@]:0:2}" --spec-type draft-dflash --spec-draft-n-max "$DEPTH")
echo "=== slot-save gate $TAG: line=$LINE kv=$KV ctx=$CTX npred=$NPRED prompt=$PROMPT_FILE name=$NAME phases=[$PHASES] arms=[$ARMS]"
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD); binary $(date -r "$BIN/llama-server" '+%m-%d %H:%M'); extra: ${EXTRA_ENV:-none}"
pid=
stop_server() { if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; fi; pid=; sleep 2; }
trap stop_server EXIT INT TERM
start_server() {  # start_server <label> [ENV=VAL ...]
  local label=$1; shift
  if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
  slog=$OUT/$TAG-$label.server.log
  env "${PICK_ENV[@]}" $EXTRA_ENV "$@" "$BIN/llama-server" -m "$PICK_MODEL" -c "$CTX" "${ARGS[@]}" "${SPEC[@]}" \
      --slot-save-path "$SLOTDIR" -lv "$LV" --port "$PORT" >"$slog" 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
    kill -0 "$pid" 2>/dev/null || { echo "[$label] server died:"; tail -5 "$slog"; exit 1; }
    sleep 2
  done
  echo "[$label] health timeout"; exit 1
}
request() {  # request <label>
  local label=$1 t0=$(date +%s)
  python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict': $NPRED, 'temperature': 0, 'id_slot': 0}))" \
    | curl -sS -H "Content-Type: application/json" -X POST "http://127.0.0.1:$PORT/completion" -d @- --max-time "$MAXTIME" > "$OUT/$TAG-$label.json"
  python3 - "$OUT/$TAG-$label.json" "$label" "$slog" $(( $(date +%s) - t0 )) <<'PY'
import json, sys, hashlib, re
d = json.load(open(sys.argv[1])); t = d.get("timings", {})
acc = 100.0*t.get("draft_n_accepted",0)/t["draft_n"] if t.get("draft_n") else 0.0
sha = hashlib.sha1(d.get("content","").encode()).hexdigest()[:12]
prof = {}
for ln in open(sys.argv[3], errors="replace"):
    m = re.search(r'spec-prof (\S+)\s+n =\s+(\d+), avg =\s+([\d.]+) ms', ln)
    if m: prof[m.group(1)] = (int(m.group(2)), float(m.group(3)))
ds = prof.get("dec_syn_tg", (0, 0.0)); dc = prof.get("draft_call", (0, 0.0))
print(f"  [{sys.argv[2]:<18}] prompt_n={t.get('prompt_n',0):>7} in {t.get('prompt_ms',0)/1000:8.1f} s | {t.get('predicted_per_second',0):7.3f} t/s acc={acc:5.1f}% n={t.get('predicted_n',0)} "
      f"round~={ds[1]+dc[1]:.1f} ms (dec_syn_tg {ds[1]:.1f} + draft {dc[1]:.1f}) | wall {sys.argv[4]} s | sha1={sha}")
PY
}
slot_action() {  # slot_action save|restore
  curl -sS -X POST "http://127.0.0.1:$PORT/slots/0?action=$1" -H "Content-Type: application/json" -d "{\"filename\":\"$NAME\"}"; echo
}
case " $PHASES " in *" fresh "*)
  start_server fresh
  request fresh-1
  echo -n "  save: "; slot_action save          # saved right after the first request: a restore then equals the in-process reuse below
  grep -h "slot saved" "$slog" | tail -1 | sed 's/^/  /'
  request fresh-2-reuse
  stop_server
  ls -la "$SLOTDIR/$NAME"* | awk '{print "  " $5 "  " $9}'
  ;;
esac
case " $PHASES " in *" roundtrip "*)   # restore NAME, save it again as NAME-rt with no request in between, byte-compare
  start_server roundtrip
  echo -n "  restore: "; slot_action restore
  echo -n "  re-save: "; curl -sS -X POST "http://127.0.0.1:$PORT/slots/0?action=save" -H "Content-Type: application/json" -d "{\"filename\":\"$NAME-rt\"}"; echo
  stop_server
  for sfx in "" .dft .ckpt; do
    if cmp -s "$SLOTDIR/$NAME$sfx" "$SLOTDIR/$NAME-rt$sfx"; then echo "  round trip $NAME$sfx: IDENTICAL ($(stat -f %z "$SLOTDIR/$NAME$sfx") bytes)"
    else echo "  round trip $NAME$sfx: DIFFERS  ($(stat -f %z "$SLOTDIR/$NAME$sfx") vs $(stat -f %z "$SLOTDIR/$NAME-rt$sfx") bytes, first diff byte $(cmp "$SLOTDIR/$NAME$sfx" "$SLOTDIR/$NAME-rt$sfx" 2>&1 | head -1 | grep -oE 'byte [0-9]+'), $(cmp -l "$SLOTDIR/$NAME$sfx" "$SLOTDIR/$NAME-rt$sfx" 2>/dev/null | wc -l | tr -d ' ') differing bytes)"; fi
  done
  ;;
esac
case " $PHASES " in *" restore "*)
  for arm in $ARMS; do
    case $arm in metalprof) start_server "restore-$arm" GGML_METAL_PROFILE=1 ;; *) start_server "restore-$arm" ;; esac
    echo -n "  restore ($arm): "; slot_action restore
    grep -h "slot restored" "$slog" | tail -1 | sed 's/^/  /'
    request "restore-$arm"
    stop_server
    if [ "$arm" = metalprof ]; then
      python3 "$B/perf/metalprof-buckets.py" "$slog" --top 40 > "$OUT/$TAG-restore-$arm-buckets.txt" 2>&1 && head -14 "$OUT/$TAG-restore-$arm-buckets.txt" | sed 's/^/    /'
    fi
  done
  ;;
esac
echo "logs: $OUT/$TAG-*"
