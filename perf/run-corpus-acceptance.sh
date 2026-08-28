#!/bin/bash
# Acceptance across the prompt corpus at the CURRENT pick (n4 + window stack), refreshing
# acceptance-by-prompt.md's n6-era table. Two arms per prompt: the pick's drafter window
# (LLAMA_DRAFT_WINDOW=1024) and a no-window control. Expectation to verify: the window is
# INERT on the tiny prompts (apply_window returns until n_past > sink+window = 1088, and
# prompt+300 stays under it), so shas/acceptance should match arm-for-arm there and differ
# only on benchprompt. Fused/async change nothing here (byte-identical, proven) and kernel
# env does not affect acceptance (slope-sweep record), so both stay out.
# Timing from these runs is throwaway (-v distorts): payload = acc, committed/rd,
# acc-per-pos survival vector, sha.
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
TAG=${TAG:-corpacc-$(date +%m%d-%H%M)}
NPRED=${NPRED:-300}
mkdir -p "$OUT"

BASE_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)
WIN_ENV=("${BASE_ENV[@]}" LLAMA_DRAFT_WINDOW=1024)

PROMPTS=(
  /Users/troff/play/benchprompt.txt
  "$B"/perf/prompts/01-code-explain.txt
  "$B"/perf/prompts/02-prose-creative.txt
  "$B"/perf/prompts/03-chat-support.txt
  "$B"/perf/prompts/04-math-derivation.txt
  "$B"/perf/prompts/05-json-boilerplate.txt
)

echo "=== corpus acceptance at the pick (n4, window vs none): $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo

run_one() {
  local label=$1 prompt=$2 envname=$3
  local slog="$OUT/$TAG-$label.server.log"
  local -a envv
  eval "envv=(\"\${$envname[@]}\")"
  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start"; return 1
  fi
  env "${envv[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    -v -md "$MD" --spec-type draft-dflash --spec-draft-n-max 4 --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }

  python3 -c "
import json
p = open('$prompt').read()
print(json.dumps({'prompt': p, 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
gen=t.get('predicted_n',0); dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
rounds = gen - da
acc = 100*da/dn if dn else 0
print('[%-22s] prompt_n=%5d  acc=%5.1f%%  committed/rd=%4.2f  rounds=%3d  sha1=%s'
      % ('$label', t.get('prompt_n',0), acc, gen/rounds if rounds else 0, rounds,
         hashlib.sha1(c.encode()).hexdigest()[:12]))
"
  sleep 2
  echo -n "  acc per pos = "
  grep -h "acc per pos" "$slog" | tail -1 | sed 's/.*acc per pos = //' || echo "(NOT FOUND)"

  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
}

for p in "${PROMPTS[@]}"; do
  name=$(basename "$p" .txt)
  run_one "$name-win"  "$p" WIN_ENV
  run_one "$name-base" "$p" BASE_ENV
done
