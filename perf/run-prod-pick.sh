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

B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
# M/MD are overridable so this harness can measure a different target without anyone
# hand-rolling a server invocation - that is the trap the whole file exists to prevent.
# ARMS filters which labels run (whole labels, space separated - not substrings); default is all of them.
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf}
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
# MM_ACC_HALF added 2026-08-28 night (owner: "I will absolutely take the prefill
# win"): half-accumulate mul_mm, +8.3% prefill wall, +0.006 mean KLD priced and
# accepted. NOTE: this changes prefill numerics, so it STARTS A NEW CANONICAL SHA
# LINEAGE - the old 9ad7e023c6ab/3776c0adb7ee gate pre-acch configs only. See
# perf/prefill-decomp.md.
# MM_N64 added 2026-08-30 (owner: "prod it"): 64x64 mul_mm tile on the measured
# width-512 short-K region, +1.27% full prefill with identical output.
# SOA_W3 adds the dedicated width-3 r4kp kernel: +20.09% at fixed DFlash depth 2.
# SOA_PIN + SKINNY_SOA complete the order-independent adaptive width frontier: widths
# 1-2 read the original weights, widths 3-5 use scalar SoA, and widths 6-8 consume the
# same persistent SoA layout with the skinny MMA kernel. The fixed depth-4 pick normally
# verifies width 5, so SKINNY_SOA is inert there; at fixed depth 5 it is +9.82% e2e.
# FA_VEC_MAX moved 5 -> 3 on 2026-09-02 (owner: "adjust the cutoff as you see fit"): the
# batched FA kernel beats the vector kernel at widths 3-4 at every context since the unroll
# fix (0.59x at width 4 on 8K, 0.26x at 100K); the vector kernel still wins at widths 1-2
# for f16 at <=8K and for Turbo4 everywhere. Inert at this depth-4 pick (verify width 5),
# -4.7% round at depth 3, where the output REJOINS the canonical sha. turbo4-filled-100k.md.
# 2026-09-07: the envs come from the manifest (perf/pick.sh, LINE=q4|ud). PICK_ENV is the line's f16-cache
# form (the reference arm), TURBO_PICK_ENV the Turbo4 form (the pick). Never copy the arrays here.
LINE=${LINE:-q4}
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" f16;    PICK_ENV=("${PICK_ENV[@]}")
pick_env "$LINE" turbo4; TURBO_PICK_ENV=("${PICK_ENV[@]}")
pick_args "$LINE" f16;   M=$PICK_MODEL; MD=$PICK_DRAFTER
TURBO=${TURBO:-0}
# LV=5 (with GGML_METAL_LOG_LEVEL=2 in the environment) makes the server log name every
# compiled pipeline, which is how a route is proved; default verbosity hides it.
LV=${LV:-}
TURBO_CTX=${TURBO_CTX:-102400}
M_TURBO=${M_TURBO:-$PICK_MODEL}
MD_TURBO=${MD_TURBO:-$PICK_DRAFTER}
# What the older harnesses set, kept to show the delta is the missing flags.
PART_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)

# Depths: the f16 pick is dflash n4 (verify width 5, m4-width5-crossover.md), the Turbo4 pick is the manifest's
# PICK_DEPTH (3). When the line's manifest carries the LLAMA_SPEC_EV controller (2026-09-17, q4), both arms run its
# block cap PICK_DEPTH_EV (7): the controller picks the verify width per round from that block.
# pick_args set PICK_DEPTH_LINE (PICK_DEPTH_EV for such a line). TRAP (the first sep17 mint): the global PICK_DEPTH is
# the fixed-depth default and confines the controller to width 3 silently - the server log's spec-ev summary shows a
# 3-entry cost table; the guard below refuses that.
F16_DEPTH=4
if printf '%s\n' "${TURBO_PICK_ENV[@]}" | grep -qx 'LLAMA_SPEC_EV=1'; then
  F16_DEPTH=$PICK_DEPTH_LINE
  [ "$PICK_DEPTH_LINE" -ge 7 ] || { echo "ABORT: the $LINE line picks LLAMA_SPEC_EV but the block cap is $PICK_DEPTH_LINE (PICK_DEPTH_EV=7 expected)"; exit 1; }
fi
PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$F16_DEPTH")
TURBO_SPEC=(-md "$MD_TURBO" --spec-type draft-dflash --spec-draft-n-max "$PICK_DEPTH_LINE")
MTP_SPEC=(--spec-type draft-mtp --spec-draft-n-max 1)
PROMPT_FILE=$(pick_prompt /Users/troff/play/benchprompt.txt)  # chat-templated since 2026-09-17 evening; PICK_CHAT=0 = the raw lineage
BASE_SPEC=(--spec-type none)

echo "=== prod pick benchmark: $TAG ==="
echo "target : $M"
echo "drafter: $MD"
echo "repo   : $B"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files dirty)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "env    : ${PICK_ENV[*]}"
[ -n "${EXTRA:-}" ] && echo "extra  : $EXTRA (appended after the pick env - a later assignment overrides an earlier one)"
echo "spec   : ${PICK_SPEC[*]}"
echo "prompt : $PROMPT_FILE (PICK_CHAT=$PICK_CHAT)"
echo

# label, n_predict, env-array-name, spec-array-name, [kv: f16 (default) | turbo4]
# kv=turbo4 switches the model, the allocation and the cache type to the Turbo4 line;
# the draft KV stays f16 (a Turbo4 draft KV is a memory-first option, +1.35% round).
run_one() {
  local label=$1 npred=$2 envname=$3 specname=$4 kv=${5:-f16}
  local model=$M ctx=10240
  local -a kvargs=(-ctk f16 -ctv f16)
  if [ "$kv" = turbo4 ]; then
    model=$M_TURBO; ctx=$TURBO_CTX
    kvargs=(-ctk turbo4 -ctv turbo4 -ctkd f16 -ctvd f16)
  fi
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

  env "${envv[@]}" ${EXTRA:-} "$BIN/llama-server" -m "$model" -c "$ctx" -fa on "${kvargs[@]}" \
    "${specv[@]}" ${LV:+-lv "$LV"} --port $PORT >"$slog" 2>&1 &
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
p = open('$PROMPT_FILE').read()
print(json.dumps({'prompt': p, 'n_predict': $npred, 'temperature': 0}))" \
  | curl -s -H "Content-Type: application/json" -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
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

if [ "$TURBO" = 1 ]; then
  echo
  echo "--- Turbo4 KV line: dflash n3 (verify width 4), 100K allocation, SOA-V1 files ---"
  echo "    reference 2026-09-01: 29.5 t/s, 104.9 ms/round, sha 12c3dc6bb2dd at 600"
  run_one "turbo4-n3-600"    600 TURBO_PICK_ENV TURBO_SPEC turbo4
  run_one "turbo4-n3-600-r2" 600 TURBO_PICK_ENV TURBO_SPEC turbo4
  run_one "turbo4-n3-300"    300 TURBO_PICK_ENV TURBO_SPEC turbo4
fi

if [ "${MULTISLOT:-1}" = 1 ]; then
  echo
  echo "--- multi-slot arm: 3 executors on 3 slots, f16, depth 1 (perf/run-multislot-gate.sh) - every arm above runs ONE"
  echo "    sequence and cannot see a multi-sequence-graph defect (perf/slot-mix.md, 2026-09-23); MULTISLOT=0 skips it;"
  echo "    then the long-extent arm (LONG=1: a 32K Turbo4 coordinator beside the executors, per-slot-ctx.md 2026-09-24) ---"
  LINE="$LINE" TAG="$TAG" B="$B" PORT=$(( ${PORT:-8093} + 1 )) LONG="${LONG:-1}" bash "$B/perf/run-multislot-gate.sh"
fi

echo
echo "--- output identity (same n_predict must share a sha) ---"
for f in /tmp/prodpick-*.txt; do echo "  $(shasum "$f" | cut -c1-12)  $(wc -c <"$f" | tr -d ' ') B  $f"; done
