#!/bin/bash
# The nxpsg=16 cutoff at the operating point it actually affects.
#
# Why this is a separate harness from run-nxpsg-gate.sh: the prod pick is dflash n6, and
# depth d verifies d+1 columns, so it runs at ne11 = 7 and routes to mul_mm_skinny. Widths
# 3-4 DO NOT OCCUR in the prod pick, so no gate change can move it. The width 3-4 work is
# about whether a DIFFERENT operating point becomes viable - oMLX's controller sits at
# width 4 for 82% of cycles (block4-shelf-probe.md) and ours sits at width 7.
#
# So the e2e question is: at dflash n3 (depth 3 = width 4, their operating point), does
# nxpsg=16 move end-to-end t/s?
#
# Arms, one binary, one flag (branch metal-mv-ext-nxpsg-w34):
#   GGML_MV_EXT_NXPSG16_MAX=3   shipping cutoff, widths 3-4 get nxpsg=8
#   GGML_MV_EXT_NXPSG16_MAX=5   widths 3-4 get nxpsg=16
#
# CONTROL: the same A/B at n6 (width 7, skinny). It must be flat - if n6 moves, the flag is
# reaching something it should not and the n3 number cannot be attributed to the gate.
#
# PRE-REGISTERED BOUND: run 3 measured -5.0%/-3.2% on ffn_down and -7.4%/+0.1% on
# ffn_gate/up at widths 3/4, against +2.5%/+8.3% on attn_q. run-nxpsg-gate.sh then measured
# the llama-bench pass at N=4. Verify is roughly half a speculative round at this depth, so
# expect e2e to move by appreciably less than the per-pass number, in the same direction.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PORT=8093
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-nxpsge2e-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

exec > >(tee "$OUT/$TAG.log") 2>&1

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)

echo "=== nxpsg=16 at width 4, end to end: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "env    : ${PICK_ENV[*]}"
echo "date   : $(date)"
echo

# label, n_predict, nmax(draft depth), NXPSG16_MAX
run_one() {
  local label=$1 npred=$2 nmax=$3 mx=$4
  local slog="$OUT/$TAG-$label.server.log"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  env "${PICK_ENV[@]}" GGML_MV_EXT_NXPSG16_MAX="$mx" \
    "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    -md "$MD" --spec-type draft-dflash --spec-draft-n-max "$nmax" \
    --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
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
print(json.dumps({'prompt': p, 'n_predict': $npred, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{})
c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
open('/tmp/nxpsge2e-$label.txt','w').write(c)
print('  [%-18s] nmax=%s max=%s n_predict=%-4s %6.3f t/s  acc=%5.1f%%  n=%d  sha1=%s'
      % ('$label', '$nmax', '$mx', '$npred', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

echo "--- WARMUP, discarded: the first run of a cold session reads high ---"
# The 2-rep version of this harness read 19.189 on its very first run and 17.967 on the same
# arm three runs later, an 6.8% within-arm spread that swamped the effect being measured.
run_one "warmup-discard" 600 3 3

echo
echo "--- dflash n3 (depth 3 = width 4, the affected point), n_predict 600 ---"
# alternate arm order between reps, per the ordering bias found in run-nxpsg-gate.sh
for r in 1 2 3 4; do
    if [ $((r % 2)) -eq 1 ]; then A=3; Bv=5; else A=5; Bv=3; fi
    run_one "n3-max$A-r$r" 600 3 $A
    run_one "n3-max$Bv-r$r" 600 3 $Bv
done

echo
echo "--- CONTROL: dflash n6 (depth 6 = width 7, skinny - the flag must not reach it) ---"
run_one "n6-max3-r1" 600 6 3
run_one "n6-max5-r1" 600 6 5
run_one "n6-max5-r2" 600 6 5
run_one "n6-max3-r2" 600 6 3

echo
echo "--- output identity (routing flags must change speed only) ---"
for f in /tmp/nxpsge2e-*.txt; do
    echo "  $(shasum "$f" | cut -c1-12)  $(wc -c <"$f" | tr -d ' ') B  $(basename "$f")"
done

echo
echo "=== done $(date) ==="
