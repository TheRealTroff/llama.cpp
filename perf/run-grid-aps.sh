#!/bin/bash
# Residency-vs-grid ladders on the width-8 skinny tiles WITH the resident-simdgroup counter (2026-10-30, perf/kernel-perf-model.md):
# arbitrary (type, M, K) at n=8 via the GGML_TEST_MM_SHAPES hook in test-backend-ops (branch exp/kernel-perf-model build).
# Per shape: REPS mirrored uncaptured timing passes (after one discarded warm-up run), then one capture replayed headlessly
# keeping raw/ so perf/aps-usc-values.py can read 'Compute SIMD Groups Inflight per Core'. Analyze with perf/grid-fit.py.
B=${B:-/Users/troff/play/llama.cpp-perfmodel}; BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/perfmodel-oct30/grid}; mkdir -p "$OUT"
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
APS=$(dirname "$0")/aps-usc-values.py
Q4ENV=${Q4ENV:-"GGML_MV_REPACK=2 GGML_MM_SKINNY_BSPLIT=2"}
UDENV_FILE=${UDENV_FILE:-/Users/troff/play/kvquant-experiments/results/perfmodel-oct30/ud-census-env.txt}
# SHAPES: "type:m:k" items, n = 8 throughout
SHAPES=${SHAPES:-"q4_0_soa:17408:5120"}; REPS=${REPS:-2}
MMSHAPES=$(for s in $SHAPES; do IFS=: read t m k <<<"$s"; printf '%s:%s:8:%s;' $t $m $k; done)
envfor() { case $1 in q4_0_soa|q4_0) echo "$Q4ENV";; *) cat "$UDENV_FILE";; esac; }
run() { # type m k extra... -> runs the perf binary for that one shape
  local t=$1 m=$2 k=$3; shift 3
  (cd "$B" && env $(envfor $t) GGML_TEST_MM_SHAPES="$MMSHAPES" "$@" "$BIN" perf -o MUL_MAT -b MTL0 -p "type_a=$t,type_b=f32,m=$m,n=8,k=$k,")
}
echo "binary $BIN ($(stat -f '%Sm' $BIN)); shapes $SHAPES; reps $REPS" | tee "$OUT/README.txt"
set -- $SHAPES; IFS=: read t m k <<<"$1"; run $t $m $k > "$OUT/warmup.log" 2>&1; echo "warm-up: $(grep -oE '[0-9.]+ us/run' $OUT/warmup.log | tail -1)" | tee -a "$OUT/README.txt"
for rep in $(seq 1 $REPS); do
  order="$SHAPES"; [ $((rep % 2)) -eq 0 ] && order=$(echo $SHAPES | tr ' ' '\n' | tail -r | tr '\n' ' ')
  for s in $order; do IFS=: read t m k <<<"$s"; id=$t-m$m-k$k
    run $t $m $k > "$OUT/time-$id-rep$rep.log" 2>&1
    us=$(grep -oE '[0-9.]+ us/run' "$OUT/time-$id-rep$rep.log" | tail -1); nrun=$(grep -c 'us/run' "$OUT/time-$id-rep$rep.log")
    echo "rep$rep $id $us (matches $nrun)" | tee -a "$OUT/README.txt"
  done
done
for s in $SHAPES; do IFS=: read t m k <<<"$s"; id=$t-m$m-k$k
  run $t $m $k MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 > "$OUT/$id.capture.log" 2>&1
  trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$OUT/$id.capture.log" | head -1)
  [ -n "$trace" ] && [ -d "$trace" ] && mv "$trace" "$OUT/$id.gputrace" || { echo "$id: NO TRACE" | tee -a "$OUT/README.txt"; continue; }
  python3 "$HEADLESS" "$OUT/$id.gputrace" "$OUT/$id.replay" > "$OUT/$id.replay.log" 2>&1 || { echo "$id: replay FAILED" | tee -a "$OUT/README.txt"; continue; }
  DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks "$PY" "$APS" "$OUT/$id.replay" > "$OUT/$id.aps.txt" 2>&1
  rm -rf "$OUT/$id.replay/raw" /tmp/com.apple.gputools.profiling
  python3 - "$OUT/$id.aps.txt" "$id" <<'PY' | tee -a "$OUT/README.txt"
import re,sys
act=[];tk=[]
for line in open(sys.argv[1]):
    m=re.search(r'usc\s+\d+.*acc/active=\s*([\d.]+)\s+ticks/sample=\s*([\d.]+)',line)
    if m: act.append(float(m.group(1))); tk.append(float(m.group(2)))
print(f"{sys.argv[2]}: inflight simdgroups/core (active) = {sum(act)/len(act)/(sum(tk)/len(tk)):.3f} over {len(act)} USCs" if act else f"{sys.argv[2]}: NO COUNTER DATA")
PY
done
echo "GRID DONE $(date +%T)" | tee -a "$OUT/README.txt"
