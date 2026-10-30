#!/bin/bash
# The threadgroup-memory pad ladder on the q4_0 SoA skinny tile WITH the resident-simdgroup counter (2026-10-30):
# TYPE=q5_K_soa etc. selects the ud skinny kernels (same tile geometry, their own chain); OUT must differ per kernel.
# uncaptured us/run per arm (mirrored order, REPS passes) + one capture per arm, replayed headlessly keeping raw/ so
# perf/aps-usc-values.py can read 'Compute SIMD Groups Inflight per Core'. Same knobs and filter as the Oct 26 probes.
B=${B:-/Users/troff/play/llama.cpp-prod}; BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/perfmodel-oct30/smem-pad}; mkdir -p "$OUT"
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
APS=$(dirname "$0")/aps-usc-values.py
M=${M:-17408}; K=${K:-5120}; TYPE=${TYPE:-q4_0_soa}; FILT="type_a=$TYPE,type_b=f32,m=$M,n=8,k=$K,"
PADS=${PADS:-"0 4096 8192 12288 16384 24576"}; REPS=${REPS:-2}
# EXTRA_ENV: the line's pick env for the kernel under test (the census.log "env:" string; ud kernels need GGML_MM_SKINNY_SOA=1 etc.)
ENV="GGML_MV_REPACK=2 GGML_MM_SKINNY_BSPLIT=2 ${EXTRA_ENV:-}"
echo "binary $BIN ($(stat -f '%Sm' $BIN)); filter $FILT" | tee "$OUT/README.txt"
for rep in $(seq 1 $REPS); do
  order="$PADS"; [ $((rep % 2)) -eq 0 ] && order=$(echo $PADS | tr ' ' '\n' | tail -r | tr '\n' ' ')
  for pad in $order; do
    (cd "$B" && env $ENV GGML_MM_SKINNY_SMEM_PAD=$pad "$BIN" perf -o MUL_MAT -b MTL0 -p "$FILT") > "$OUT/time-pad$pad-rep$rep.log" 2>&1
    us=$(grep -oE '[0-9.]+ us/run' "$OUT/time-pad$pad-rep$rep.log" | tail -1); k=$(grep -oE 'name = kernel_mul_mm[a-z0-9_]+' "$OUT/time-pad$pad-rep$rep.log" | tail -1)
    echo "rep$rep pad=$pad $us ($k)" | tee -a "$OUT/README.txt"
  done
done
for pad in $PADS; do
  id=pad$pad
  (cd "$B" && env $ENV MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 GGML_MM_SKINNY_SMEM_PAD=$pad "$BIN" perf -o MUL_MAT -b MTL0 -p "$FILT") > "$OUT/$id.capture.log" 2>&1
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
