#!/bin/bash
# In-place repack (GGML_MV_REPACK=1) against the side-buffer probe (=3) and against no repack,
# on both memory paths. The point of the arm matrix is that repack's speed was never in doubt -
# its residency was - so every arm reports RSS as well as t/s.
#
#   mmap-off/none    control for the in-place arm (weights in a buffer we own)
#   mmap-off/inplace the new path: one layout, no second copy
#   mmap-on/side     the original probe, best known speed, +1 model in RAM
#   mmap-on/none     the prod pick today
#
# In-place needs --no-mmap: mmap-ed weights are PROT_READ pages of the model file and the
# conversion writes into the weights themselves.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=8093
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-inplace-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
REPS=${REPS:-1}
ARMS=${ARMS:-}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)
PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 6)

echo "=== repack in-place A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "npred  : $NPRED, $REPS repeats"
echo

# label, GGML_MV_REPACK value, extra server flags, extra env (optional)
run_one() {
  local label=$1 repack=$2 extra=$3 xenv=${4:-}
  if [ -n "$ARMS" ]; then
    case " $ARMS " in *" $label "*) ;; *) return 0 ;; esac
  fi
  local slog="$OUT/$TAG-$label.server.log"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  # shellcheck disable=SC2086
  # shellcheck disable=SC2086
  env "${PICK_ENV[@]}" GGML_MV_REPACK="$repack" $xenv "$BIN/llama-server" -m "$M" -c 10240 \
    -fa on -ctk f16 -ctv f16 "${PICK_SPEC[@]}" $extra --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }
  lsof -ti :$PORT 2>/dev/null | grep -qx "$pid" || {
    echo "[$label] ABORT: port $PORT is served by another process, not our server"
    kill -9 $pid 2>/dev/null; return 1; }

  python3 -c "
import json
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- > "$OUT/$TAG-$label.json"

  # RSS after the run, when every weight has been touched and any conversion has happened
  local rss
  rss=$(ps -o rss= -p $pid | tr -d ' ')

  local conv
  conv=$(grep -c "in place" "$slog" || true)

  python3 -c "
import json
d=json.load(open('$OUT/$TAG-$label.json'))
import hashlib
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); raise SystemExit
t=d.get('timings',{}); c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
print('[%-16s] %6.3f t/s  acc=%5.1f%%  sha1=%s  rss=%7.2f GiB  repack_log=%s'
      % ('$label', t.get('predicted_per_second',0), acc,
         hashlib.sha1(c.encode()).hexdigest()[:12], $rss/1024/1024, '$conv'))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

for r in $(seq 1 "$REPS"); do
  run_one "nommap-none"    0 "--no-mmap"
  run_one "nommap-inplace" 1 "--no-mmap"
  # the same deinterleaved kernels off a side buffer on the same memory path as the in-place
  # arm: this is what separates "in place vs side buffer" from "mmap vs no-mmap"
  run_one "nommap-side"    3 "--no-mmap"
  # in place with the weights in a private (not CPU-coherent) buffer: the side-buffer arms read
  # their deinterleaved copy out of exactly such a buffer, so this is the arm that says whether
  # the gap between them is the storage mode rather than the layout
  run_one "nommap-inplace-priv" 1 "--no-mmap" "GGML_METAL_SHARED_BUFFERS_DISABLE=1"
  run_one "mmap-side"      3 ""
  run_one "mmap-none"      0 ""
done
