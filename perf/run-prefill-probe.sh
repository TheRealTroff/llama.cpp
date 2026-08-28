#!/bin/bash
# Prefill probe (cpu-round-overhead.md "Prefill submits: anomalous, unexplained"):
# benchprompt prefill is ~66 s wall at ~122 t/s, dec_sub_pp ~9.7 s/batch with syn at
# zero, ~18 s lands AFTER submit returns - and nobody has ever profiled prefill on
# this stack. First cut, non-perturbing only (submit-prof + decode-prof; the per-op
# GGML_METAL_PROFILE pass comes later if the timeline shape warrants it):
#   pick    - the full prod pick, drafter on
#   nodraft - spec off, isolates the drafter's prefill-side injection cost
# Payload: prompt eval wall + t/s, dec_sub_pp/dec_syn_pp, prefill-era submit-prof
# m1/m2 windows. n_predict is tiny - decode is not the subject.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=8095
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-prefill-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

# The prod pick env, copied from run-prod-pick.sh (keep in sync), plus the profilers.
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          GGML_METAL_SUBMIT_PROF=1 LLAMA_DECODE_PROF=1)

PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 4)
BASE_SPEC=(--spec-type none)

echo "=== prefill probe: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo

run_one() {
  local label=$1 specname=$2
  local slog="$OUT/$TAG-$label.server.log"
  local -a specv
  eval "specv=(\"\${$specname[@]}\")"
  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start"; return 1
  fi
  env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    "${specv[@]}" --port $PORT >"$slog" 2>&1 &
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
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': 8, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{})
print('[%-8s] prompt_n=%5d  prefill %8.1f ms  %6.1f t/s'
      % ('$label', t.get('prompt_n',0), t.get('prompt_ms',0), t.get('prompt_per_second',0)))
"
  sleep 2
  echo "  --- dec_sub_pp / dec_syn_pp ---"
  grep -hE "dec_sub_pp|dec_syn_pp" "$slog" | tail -4 | sed 's/^/    /'
  echo "  --- submit-prof windows (prefill era = first ones) ---"
  grep -h "submit-prof" "$slog" | head -6 | sed 's/^/    /'

  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

run_one "pick"    PICK_SPEC
run_one "nodraft" BASE_SPEC
