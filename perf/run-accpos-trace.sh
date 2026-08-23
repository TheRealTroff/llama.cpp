#!/bin/bash
# Per-position draft acceptance: is draft 7 a structural zero?
#
# `slope-sweep.md` shows n6/n7/n8 all commit exactly 3.75 tok/round over exactly 80 rounds,
# so the 7th draft is accepted ZERO times, not rarely. Marginal accepted drafts per added
# draft run +0.604 +0.447 +0.311 +0.256 +0.298 then +0.005 - a 60x cliff in one position,
# which prefix-gating cannot produce.
#
# server-context.cpp:4040-4045 already keeps slot.n_accepted_per_pos, printed at :710-716
# as "acc per pos". It is behind SLT_TRC, so it needs -v, and NO run in results/ has ever
# captured it. n_accepted_per_pos[i] counts rounds with at least i+1 drafts accepted, so
# the vector is a SURVIVAL CURVE: index 6 = fraction of rounds where all 7 drafts landed.
#
# NOT load-sensitive: acceptance is deterministic (all 15 slope-sweep runs emitted sha1
# 9ad7e023c6ab; routing and speculation change speed only). Timing here is throwaway - -v
# distorts it heavily - the sha and the acc-per-pos vector are the payload.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PORT=8094
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-accpos-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)

echo "=== acc-per-pos trace: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "note   : dirty files are docs/probes only - no source under ggml/ common/ tools/, so the build is current"
echo

run_one() {
  local label=$1 npred=$2; shift 2
  local slog="$OUT/$TAG-$label.server.log"
  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start"; return 1
  fi
  env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    -v "$@" --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }

  local clamp
  clamp=$(grep -o "clamping to [0-9]*" "$slog" | head -1)

  python3 -c "
import json
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': $npred, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
gen=t.get('predicted_n',0); dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
rounds = gen - da
acc = 100*da/dn if dn else 0
print('[%-10s] acc=%5.1f%%  drafted/rd=%4.2f  committed/rd=%4.2f  rounds=%3d  sha1=%s %s'
      % ('$label', acc, dn/rounds if rounds else 0, gen/rounds if rounds else 0, rounds,
         hashlib.sha1(c.encode()).hexdigest()[:12], '$clamp'))
"
  # give print_timings() a moment to land in the log
  sleep 2
  echo -n "  acc per pos = "
  grep -h "acc per pos" "$slog" | tail -1 | sed 's/.*acc per pos = //' || echo "(NOT FOUND)"

  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
}

# n7 is the question; n6 (prod pick) is the control - its vector should be a healthy
# survival curve out to position 6.
# Depths default to the n7/n6 pair; pass others to test how block depth reshapes the curve.
for n in "${@:-7 6}"; do
  run_one "dflash-n$n" 300 -md "$MD" --spec-type draft-dflash --spec-draft-n-max $n
done
