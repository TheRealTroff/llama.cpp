#!/bin/bash
# perf/run-w8-decomp.sh - where the UD width-8 verify round goes, against the q4 line's (2026-09-18,
# owner: "profile away"). spec-verify-narrow.md section 10 left the (7,7) round at ud 180 ms vs q4 131 ms
# with the width-4 rounds only 13 ms apart, and named two suspects for the ~35 ms: the generic skinny
# tile over the stored SoA formats (1.5-1.7x floor, issue-bound) and the FA leaving the GQA-reuse plan
# at widths 7-8 for the plain batched route. This pins the width (fixed depth = every verify at
# width depth+1) on both lines under the MANIFEST pick env (perf/pick.sh, Turbo4 cache, PICK_SPEC_EV=0)
# and runs the Sep 5 round-decomposition instruments per arm:
#   anchor    -lv 3         spec-prof per-round host split (round = dec_syn_tg GPU wait + draft_call; loop_body carries the prefill batch)
#   metalprof GGML_METAL_PROFILE=1  per-op GPU time, serialized encoders (shares and per-call us, not wall)
# then perf/metalprof-buckets.py per arm and a side-by-side of the buckets.
#   LINES="ud q4" DEPTHS="7 3" NPRED=300 KV=turbo4 PROMPT=<raw file, chat-rendered via pick_prompt>
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-w8decomp-$(date +%m%d-%H%M)}
LINES=${LINES:-ud q4}
DEPTHS=${DEPTHS:-7 3}
NPRED=${NPRED:-300}
KV=${KV:-turbo4}
LV=${LV:-3}
PORT=${PORT:-8093}
EXTRA_ENV=${EXTRA_ENV:-}
STEPS=${STEPS:-anchor metalprof}
export PICK_SPEC_EV=0   # fixed depth: the width under test on every round, never the controller
source "$B/perf/pick.sh"
PROMPT_FILE=$(pick_prompt "${PROMPT:-/Users/troff/play/benchprompt.txt}")
mkdir -p "$OUT"

echo "=== width decomposition by line: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "lines  : $LINES   depths: $DEPTHS (verify width = depth+1)   kv: $KV   npred: $NPRED   extra: ${EXTRA_ENV:-none}"
echo "prompt : $PROMPT_FILE (PICK_CHAT=$PICK_CHAT)"
if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "ABORT: port $PORT busy" >&2; exit 1; fi

server_pid=
cleanup_server() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap cleanup_server EXIT INT TERM

run_server() {  # run_server <line> <depth> <label> [ENV=VAL ...]
  local line=$1 depth=$2 label=$3; shift 3
  pick_env "$line" "$KV"; pick_args "$line" "$KV"
  local slog=$OUT/$TAG-$line-d$depth-$label.server.log
  env "${PICK_ENV[@]}" $EXTRA_ENV "$@" "$BIN/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" \
      -md "$PICK_DRAFTER" --spec-type draft-dflash --spec-draft-n-max "$depth" \
      -lv "$LV" --port "$PORT" >"$slog" 2>&1 &
  server_pid=$!
  local healthy=0
  for _ in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { healthy=1; break; }
    kill -0 "$server_pid" 2>/dev/null || { echo "[$line d$depth $label] server died:"; tail -8 "$slog"; return 1; }
    sleep 2
  done
  [ "$healthy" = 1 ] || { echo "[$line d$depth $label] health timeout"; return 1; }
  python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT_FILE').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
    | curl -sS -o "$OUT/$TAG-$line-d$depth-$label.json" -X POST "http://127.0.0.1:$PORT/completion" -d @-
  cleanup_server; sleep 3
  python3 - "$OUT/$TAG-$line-d$depth-$label.json" "$line d$depth $label" "$slog" <<'PY'
import json, sys, hashlib, re
d = json.load(open(sys.argv[1])); t = d.get("timings", {})
acc = 100.0*t.get("draft_n_accepted",0)/t["draft_n"] if t.get("draft_n") else 0.0
sha = hashlib.sha1(d.get("content","").encode()).hexdigest()[:12]
prof = {}
for ln in open(sys.argv[3], errors="replace"):
    m = re.search(r'spec-prof (\S+)\s+n =\s+(\d+), avg =\s+([\d.]+) ms', ln)
    if m: prof[m.group(1)] = (int(m.group(2)), float(m.group(3)))
# loop_body's average carries the prefill batch on a long prompt (~0.9 s over 65 rounds on the 8K benchprompt);
# the round is the GPU wait (dec_syn_tg) plus the drafter call, both per-round timers
ds = prof.get("dec_syn_tg", (0, 0.0)); dc = prof.get("draft_call", (0, 0.0)); de = prof.get("decode", (0, 0.0))
print(f"  [{sys.argv[2]:<16}] {t.get('predicted_per_second',0):7.3f} t/s  acc={acc:5.1f}%  n={t.get('predicted_n',0)}  "
      f"rounds={dc[0]}  round~={ds[1]+dc[1]:.1f} ms (dec_syn_tg {ds[1]:.1f} + draft_call {dc[1]:.1f}; decode {de[1]:.1f})  sha1={sha}")
PY
}

for line in $LINES; do
  for depth in $DEPTHS; do
    echo; echo "--- $line, depth $depth (verify width $((depth+1))) ---"
    case " $STEPS " in *" anchor "*)    run_server "$line" "$depth" anchor ;; esac
    case " $STEPS " in *" metalprof "*)
      run_server "$line" "$depth" metalprof GGML_METAL_PROFILE=1
      python3 "$B/perf/metalprof-buckets.py" "$OUT/$TAG-$line-d$depth-metalprof.server.log" --top 60 \
        > "$OUT/$TAG-$line-d$depth-buckets.txt" 2>&1 && head -16 "$OUT/$TAG-$line-d$depth-buckets.txt" | sed 's/^/    /'
      ;;
    esac
  done
done
echo; echo "logs: $OUT/$TAG-*"
