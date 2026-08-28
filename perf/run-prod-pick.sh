#!/bin/bash
# THE canonical prod-pick benchmark. If you want to know "how fast are we right now",
# run this file and nothing else. See perf/README.md for the flag set it encodes.
#
# It exists because the prod pick used to live only in prose spread across perf/*.md,
# and every harness encoded its own partial subset of the env flags. RUN_DRAFTER_FINAL.sh
# and RUN_ROUND_DECOMP.sh set only GGML_MV_NC/GGML_MM_SKINNY, so they silently measure a
# config that predates the FA mm-split and the GDN writeback fusion. perf/prod-baseline.md
# reported 22.115 t/s that way and it read like a regression against ~25.
#
# Two traps this file is built to avoid:
#   1. Missing flags. All of them live in PICK_ENV below, in ONE place. Every flag defaults
#      to off/upstream in the source, so a forgotten one is silent, not an error.
#   2. n_predict units. Absolute t/s is NOT comparable across n_predict: generation grows
#      the KV cache, so the same config reads ~25 at 300 and ~23 at 600. Both are measured
#      here and always reported together.
#
# Fresh server per measurement, matching how every recorded number was taken.
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
# M/MD are overridable so this harness can measure a different target without anyone
# hand-rolling a server invocation - that is the trap the whole file exists to prevent.
# ARMS filters which labels run (substring match, space separated); default is all of them.
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
ARMS=${ARMS:-}
PORT=8093
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-prodpick-$(date +%m%d-%H%M)}
mkdir -p "$OUT"

# The prod pick, in one place. Moved 2026-08-28 (owner's decision) from n6+skinny to
# dflash n4 + the SoA scalar kernels (w4 v3 at draft-path width 4, w5r4h at verify
# width 5) + repack side buffer - see perf/m4-width5-crossover.md. SKINNY=6, not 5:
# skinny takes ne11 >= value and must not swallow width 5 ahead of the w5 route.
# WL_XL added 2026-08-28 (owner's decision): routes both 248320-vocab lm_heads to
# w5r4h, +3.04% e2e - see perf/shortk-head.md.
# GET_MEMCPY added 2026-08-28 afternoon (owner: "pick get_memcpy"): logits readback
# as memcpy-after-wait instead of a blit behind the graph, +3.3% e2e -
# see perf/cpu-round-overhead.md.
# Drafter stack added 2026-08-28 evening (owner: "do the others"): fused inject +
# async (process() submit-only) + attention window 1024 (acceptance improves),
# +1.87% e2e together, shas hold in every arm - see perf/drafter-graph-count.md.
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024)
# What the older harnesses set, kept to show the delta is the missing flags.
PART_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)

PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 4)
MTP_SPEC=(--spec-type draft-mtp --spec-draft-n-max 1)
BASE_SPEC=(--spec-type none)

echo "=== prod pick benchmark: $TAG ==="
echo "target : $M"
echo "drafter: $MD"
echo "repo   : $B"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "env    : ${PICK_ENV[*]}"
echo "spec   : ${PICK_SPEC[*]}"
echo

# label, n_predict, env-array-name, spec-array-name
run_one() {
  local label=$1 npred=$2 envname=$3 specname=$4
  if [ -n "$ARMS" ]; then
    case " $ARMS " in *" $label "*) ;; *) return 0 ;; esac
  fi
  local slog="$OUT/$TAG-$label.server.log"
  local -a envv specv
  eval "envv=(\"\${$envname[@]}\")"
  eval "specv=(\"\${$specname[@]}\")"

  if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "[$label] ABORT: port $PORT busy before start (stale server?)"; return 1
  fi

  env "${envv[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    "${specv[@]}" --port $PORT >"$slog" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }
  # a leftover server on this port would answer /health and we would measure ITS config
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
open('/tmp/prodpick-$label.txt','w').write(c)
print('[%-22s] n_predict=%-4s %6.3f t/s  acc=%5.1f%%  n=%d  sha1=%s'
      % ('$label', '$npred', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

echo "--- prod pick: dflash n6, full env, n_predict 300 (comparable to the 24.95 on record) ---"
run_one "pick-n6-300"     300 PICK_ENV PICK_SPEC
run_one "pick-n6-300-r2"  300 PICK_ENV PICK_SPEC

echo
echo "--- prod pick at n_predict 600 (low-variance units; ~166 rounds) ---"
run_one "pick-n6-600"     600 PICK_ENV PICK_SPEC
run_one "pick-n6-600-r2"  600 PICK_ENV PICK_SPEC

echo
echo "--- partial env (what RUN_DRAFTER_FINAL/RUN_ROUND_DECOMP/prod-baseline actually set) ---"
run_one "partial-n6-300"  300 PART_ENV PICK_SPEC

echo
echo "--- references ---"
run_one "mtp-d1-300"      300 PICK_ENV MTP_SPEC
run_one "batch1-300"      300 PICK_ENV BASE_SPEC

echo
echo "--- output identity (same n_predict must share a sha) ---"
for f in /tmp/prodpick-*.txt; do echo "  $(shasum "$f" | cut -c1-12)  $(wc -c <"$f" | tr -d ' ') B  $f"; done
