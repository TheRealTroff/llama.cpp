#!/bin/bash
# Where is the small-batch slope right now, and where do we fall off the skinny window?
#
# The skinny gate (ggml-metal-ops.cpp) is `ne11 >= max(2, GGML_MM_SKINNY) && ne11 <= 8`.
# Speculation at depth d verifies d+1 columns, so ne11=9 (d=8) drops onto mul_mm. This
# sweep measures that boundary three ways:
#
#   part 1  llama-bench ms/pass for N=1..10, at the prod-pick env and at stock routing.
#           N=9 crossing the window is visible here with no speculation involved.
#   part 2  e2e dflash depth sweep. NOTE: dflash CANNOT reach ne11=9 - speculative.cpp:1008
#           clamps n_max to block_size-1 = 7 for this drafter, so `-n-max 8` silently runs
#           as 7. Any recorded "dflash n8" is a mislabelled n7. n8 is included here only to
#           document the clamp.
#   part 3  e2e MTP depth sweep, which CAN reach d=8 (clamped only to n_mtp_layers) and is
#           where the recorded 10.29 t/s collapse came from.
#
# Part 1 vs prod-baseline.md: that file ran llama-bench with NO env overrides, so its table
# is stock routing, not the prod pick. Both are measured here.
set -u

# Keep the machine awake for the whole harness. These runs spend more wall time idle in
# cooldowns than measuring, and on battery pmset is `sleep 1` / `displaysleep 2`, so a
# 120-180 s cooldown would idle-sleep the machine mid-run and the next arm would measure
# a cold cache and a ramping clock. On AC `sleep` is 0 and this is a no-op.
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PORT=8093
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-slope-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)

echo "=== slope sweep: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "env    : ${PICK_ENV[*]}"
echo

# ---------------------------------------------------------------- part 1: llama-bench
bench_one() {
  local label=$1; shift
  echo "--- llama-bench: $label ---"
  env "$@" "$BIN/llama-bench" -m "$M" -fa 1 -ctk f16 -ctv f16 \
      -n 0 -p 1,2,3,4,5,6,7,8,9,10 -r 3 2>&1 \
    | tee "$OUT/$TAG-bench-$label.log" | grep -E "^\|" | grep -vE "^\| *-"
  echo
}

bench_one "prodenv" "${PICK_ENV[@]}"
bench_one "stock"   PATH="$PATH"

# ---------------------------------------------------------------- e2e helper
run_e2e() {
  local label=$1 npred=$2; shift 2
  local slog="$OUT/$TAG-$label.server.log"
  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start"; return 1
  fi
  env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    "$@" --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }
  lsof -ti :$PORT 2>/dev/null | grep -qx "$pid" || {
    echo "[$label] ABORT: port served by another process"; kill -9 $pid 2>/dev/null; return 1; }

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
# each round commits 1 target token plus its accepted drafts
rounds = gen - da
acc = 100*da/dn if dn else 0
per  = dn/rounds if rounds else 0          # actual drafted per round = effective depth
tpr  = gen/rounds if rounds else 0         # committed tokens per round
tps  = t.get('predicted_per_second',0)
print('[%-14s] %6.3f t/s  acc=%5.1f%%  drafted/rd=%4.2f  committed/rd=%4.2f  rounds=%3d  ms/rd=%6.1f  sha1=%s %s'
      % ('$label', tps, acc, per, tpr, rounds, 1000*tpr/tps if tps else 0,
         hashlib.sha1(c.encode()).hexdigest()[:12], '$clamp'))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

# ---------------------------------------------------------------- part 2: dflash depth
echo "--- e2e dflash depth sweep (n_predict 300); n8 is clamped to 7 by design ---"
for n in 1 2 3 4 5 6 7 8; do
  run_e2e "dflash-n$n" 300 -md "$MD" --spec-type draft-dflash --spec-draft-n-max $n
done
echo

# ---------------------------------------------------------------- part 3: MTP depth
echo "--- e2e MTP depth sweep (n_predict 300); d8 = ne11 9 = off the skinny window ---"
for d in 1 2 4 6 7 8; do
  run_e2e "mtp-d$d" 300 --spec-type draft-mtp --spec-draft-n-max $d
done
echo

echo "--- batch-1 floor ---"
run_e2e "batch1" 300 --spec-type none
