#!/bin/bash
# Does the K-split survive a pass and a round? (branch metal-mv-ext-ksplit)
#
# perf/run-ksplit-sweep.sh measured the kernel: -20% at width 3 and -9% at width 4 on
# ffn_down. Only the cells that leave the nxpsg heuristic alone are candidates here -
# forcing nxpsg globally would hit attn_q (+73% at width 4) and any shape that fails the
# ne00 % 256 guard, so the shippable form is kp with nxpsg left as the source picks it.
#
#   kp2  GGML_MV_EXT_KP=2      nsg stays 2, one row block per threadgroup, K in halves
#   kp4  GGML_MV_EXT_KP=4      nsg raised to 4 for the ks dispatch only, K in quarters
#
# The e2e arms run dflash at n3 (width 4, where the split engages) and n6 (width 7, the prod
# pick, which routes to skinny and must NOT move - that is the control).
# CORRECTNESS GATE: canonical output sha1 is 9ad7e023c6ab. Any arm that differs is a wrong
# answer, not a fast one.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PROMPT=/Users/troff/play/benchprompt.txt
PORT=8095
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ksplite2e-$(date +%m%d-%H%M)}
COOL=${COOL:-90}
mkdir -p "$OUT"

BASE_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)
CANON=9ad7e023c6ab

ARMS=(
  "base:GGML_MV_EXT_KP=1"
  "kp2:GGML_MV_EXT_KP=2"
  "kp4:GGML_MV_EXT_KP=4"
)

echo "=== K-split, pass and round: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-bench")"
echo "canon  : sha1 $CANON"
echo

echo "--- part 1: llama-bench ms/pass ---"
for arm in "${ARMS[@]}"; do
  label=${arm%%:*}; envs=${arm#*:}
  echo "--- $label ($envs) ---"
  env "${BASE_ENV[@]}" $envs "$BIN/llama-bench" -m "$M" -fa 1 -ctk f16 -ctv f16 \
      -n 0 -p 1,2,3,4,5,6,7,8 -r 3 2>&1 \
    | tee "$OUT/$TAG-bench-$label.log" | grep -E "^\|" | grep -vE "^\| *-"
  echo
done

run_e2e() {
  local label=$1; shift
  local envs=$1; shift
  local slog="$OUT/$TAG-$label.server.log"
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port busy"; return 1; fi
  env "${BASE_ENV[@]}" $envs "$BIN/llama-server" -m "$M" -c 10240 -fa on \
      -ctk f16 -ctv f16 "$@" --port $PORT >"$slog" 2>&1 &
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
print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': 300, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d: print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
gen=t.get('predicted_n',0); dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
rounds=gen-da; tps=t.get('predicted_per_second',0)
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
print('[%-16s] %6.3f t/s  acc=%5.1f%%  committed/rd=%4.2f  rounds=%3d  ms/rd=%6.1f  sha=%s%s'
      % ('$label', tps, 100*da/dn if dn else 0, gen/rounds if rounds else 0, rounds,
         1000*(gen/rounds)/tps if tps and rounds else 0, sha,
         '' if sha=='$CANON' else '  <<< SHA MISMATCH'))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep "$COOL"
}

echo "--- part 2: e2e, n3 is the treated width and n6 is the control ---"
for arm in "${ARMS[@]}"; do
  label=${arm%%:*}; envs=${arm#*:}
  run_e2e "n3-$label" "$envs" -md "$MD" --spec-type draft-dflash --spec-draft-n-max 3
done
for arm in "${ARMS[@]}"; do
  label=${arm%%:*}; envs=${arm#*:}
  run_e2e "n6-$label" "$envs" -md "$MD" --spec-type draft-dflash --spec-draft-n-max 6
done
