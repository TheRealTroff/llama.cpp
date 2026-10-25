#!/bin/bash
# perf/run-fa-prefill-200k.sh - the prefill FA kernel at a 200K cache without a 53-minute prefill (2026-10-25, owner: "I believe
# this kind of work is why we created the 200k slot saves?"). Restores kvquant-experiments/slots/ud-200k (Turbo4, -c 212992,
# 202066 tokens of the rendered War-and-Peace head) in a fresh server per arm, then sends the SAME rendered prompt with ~4K
# tokens of the book appended AFTER the assistant header (data/chat/longprompt-200k-plus4k.txt): the whole 202K prefix hits
# the restored cache (no checkpoint rollback), the appended tokens prefill in 512-row ubatches against the full cache - every
# prefill FA call runs at kv ~202-206K, the pick's >32K route (qt16w_turbo4 on ud). Arms: anchor (-lv 3, wall prefill t/s
# at 200K) and metalprof (GGML_METAL_PROFILE=1, per-op us: the FA rows at n=512 are the census input, PHASE=prefill).
#   BIN=<prod build/bin> B=<this tree> LINE=ud CTX=212992 DEPTH=3 NAME=ud-200k NPRED=64 ARMS="anchor metalprof" EXTRA_ENV=...
# Pinned depth (PICK_SPEC_EV=0) like the save; PICK_CHAT=0 because the prompt file is already rendered.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-fa200k}
BIN=${BIN:-/Users/troff/play/llama.cpp-prod/build/bin}   # no kernel change on this branch: the prod binary
LINE=${LINE:-ud}; KV=${KV:-turbo4}; CTX=${CTX:-212992}; NPRED=${NPRED:-64}; DEPTH=${DEPTH:-3}
PROMPT=${PROMPT:-/Users/troff/play/kvquant-experiments/data/chat/longprompt-200k-plus4k.txt}
NAME=${NAME:-ud-200k}; SLOTDIR=${SLOTDIR:-/Users/troff/play/kvquant-experiments/slots}
ARMS=${ARMS:-anchor metalprof}
PORT=${PORT:-8098}; LV=${LV:-3}; MAXTIME=${MAXTIME:-3600}
OUT=/Users/troff/play/kvquant-experiments/results; TAG=${TAG:-fa200k-$(date +%m%d-%H%M)}
EXTRA_ENV=${EXTRA_ENV:-}
export PICK_SPEC_EV=0 PICK_CHAT=0
source "$B/perf/pick.sh"
pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"
ARGS=(); skip=0
for a in "${PICK_ARGS[@]}"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  if [ "$a" = -c ]; then skip=1; continue; fi
  ARGS+=("$a")
done
SPEC=("${PICK_SPEC[@]:0:2}" --spec-type draft-dflash --spec-draft-n-max "$DEPTH")
echo "=== prefill FA at 200K $TAG: line=$LINE kv=$KV ctx=$CTX depth=$DEPTH npred=$NPRED save=$NAME arms=[$ARMS]"
echo "repo   : $B $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD); binary $BIN $(date -r "$BIN/llama-server" '+%m-%d %H:%M'); extra: ${EXTRA_ENV:-none}"
echo "prompt : $PROMPT ($(wc -c < "$PROMPT" | tr -d ' ') chars)"
pid=
stop_server() { if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; fi; pid=; sleep 2; }
trap stop_server EXIT INT TERM
for arm in $ARMS; do
  if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
  slog=$OUT/$TAG-$arm.server.log
  ARMENV=; [ "$arm" = metalprof ] && ARMENV=GGML_METAL_PROFILE=1   # a plain string: an empty array is "unbound" under set -u on bash 3.2
  env "${PICK_ENV[@]}" $EXTRA_ENV $ARMENV "$BIN/llama-server" -m "$PICK_MODEL" -c "$CTX" "${ARGS[@]}" "${SPEC[@]}" \
      --slot-save-path "$SLOTDIR" -lv "$LV" --port "$PORT" >"$slog" 2>&1 &
  pid=$!
  ok=0
  for _ in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || { echo "[$arm] server died:"; tail -5 "$slog"; exit 1; }
    sleep 2
  done
  [ $ok = 1 ] || { echo "[$arm] health timeout"; exit 1; }
  echo -n "  restore ($arm): "; curl -sS -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" -H "Content-Type: application/json" -d "{\"filename\":\"$NAME\"}"; echo
  grep -h "slot restored" "$slog" | tail -1 | sed 's/^/  /'
  t0=$(date +%s)
  python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0, 'id_slot': 0}))" \
    | curl -sS -H "Content-Type: application/json" -X POST "http://127.0.0.1:$PORT/completion" -d @- --max-time "$MAXTIME" > "$OUT/$TAG-$arm.json"
  python3 - "$OUT/$TAG-$arm.json" "$arm" "$slog" $(( $(date +%s) - t0 )) <<'PY'
import json, sys, hashlib, re
d = json.load(open(sys.argv[1])); t = d.get("timings", {})
if not t: print("  [%s] NO TIMINGS: %s" % (sys.argv[2], json.dumps(d)[:300])); sys.exit(0)
acc = 100.0*t.get("draft_n_accepted",0)/t["draft_n"] if t.get("draft_n") else 0.0
sha = hashlib.sha1(d.get("content","").encode()).hexdigest()[:12]
pn=t.get('prompt_n',0); pms=t.get('prompt_ms',0)
print(f"  [{sys.argv[2]:<10}] prompt_n={pn:>6} in {pms/1000:7.2f} s = {1000*pn/pms if pms else 0:6.1f} t/s prefill at 200K | decode {t.get('predicted_per_second',0):6.2f} t/s acc={acc:4.1f}% n={t.get('predicted_n',0)} | wall {sys.argv[4]} s | sha1={sha}", flush=True)
PY
  stop_server
  if [ "$arm" = metalprof ]; then
    python3 "$B/perf/metalprof-buckets.py" "$slog" --top 40 > "$OUT/$TAG-$arm-buckets.txt" 2>&1 && head -24 "$OUT/$TAG-$arm-buckets.txt" | sed 's/^/    /'
  fi
done
echo "logs: $OUT/$TAG-*"
