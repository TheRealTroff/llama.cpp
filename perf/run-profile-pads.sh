#!/bin/bash
# Per-instruction profile of one skinny tile at several residencies (perf/kernel-perf-model.md, 2026-10-31): the pad ladder's
# arms captured and replayed with the per-instruction profiler, joined to the machine IR (the census row pipeline, one row per pad).
#   TYPE=q3_K_soa LINE=ud PADS="12288 24576" perf/run-profile-pads.sh        -> OUT/pad<pad>.{timing,capture.log,instr.json,join.json}
# Timing from an uncaptured run (one pass; the ladders have the repeated numbers), then capture -> headless replay -> stats ->
# shaderprof-table -> agx-nt-opt mir + align (kernel-census.py join_row). The translator metallib is built from THIS tree's source.
set -u
B=${B:-/Users/troff/play/llama.cpp-perfmodel}; BIN=$B/build/bin/test-backend-ops
TYPE=${TYPE:-q3_K_soa}; LINE=${LINE:-ud}; PADS=${PADS:-"12288 24576"}; M=${M:-17408}; K=${K:-5120}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/perfmodel-oct30/profile-pads-$TYPE}; mkdir -p "$OUT"
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
STATS=/Users/troff/.claude/skills/metal-gpu-profile/references/gpuprofiler-stats.py
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
FILT="type_a=$TYPE,type_b=f32,m=$M,n=8,k=$K,"
source "$B/perf/pick.sh"; pick_env "$LINE"; ENVS="GGML_MV_REPACK=2 ${PICK_ENV[*]}"
export CENSUS_METALLIB=$B/build-air/ggml.metallib
if [ ! -f "$CENSUS_METALLIB" ] || [ "$B/ggml/src/ggml-metal/ggml-metal.metal" -nt "$CENSUS_METALLIB" ]; then
    mkdir -p "$B/build-air"
    (cd "$B" && xcrun metal -fmodules-cache-path=/tmp/air-module-cache -DGGML_METAL_HAS_BF16=1 -DTURBO_USE_4MAG=1 -DTURBO_USE_PAIR_LUT=1 \
        -c ggml/src/ggml-metal/ggml-metal.metal -Iggml/src/ggml-metal -Iggml/src -o "$B/build-air/ggml.air" > "$B/build-air/ggml.log" 2>&1 \
        && xcrun metallib "$B/build-air/ggml.air" -o "$CENSUS_METALLIB") &
    LIBPID=$!
else LIBPID=; fi
echo "binary $BIN ($(stat -f '%Sm' "$BIN")); filter $FILT; line $LINE" | tee -a "$OUT/README.txt"
for pad in $PADS; do
    id=pad$pad
    (cd "$B" && env $ENVS GGML_MM_SKINNY_SMEM_PAD=$pad "$BIN" perf -o MUL_MAT -b MTL0 -p "$FILT" 2>&1) | grep -E 'us/run|loaded kernel_' > "$OUT/$id.timing.txt"
    echo "$id: $(grep -oE '[0-9.]+ us/run' "$OUT/$id.timing.txt" | tail -1) $(grep -oE 'loaded kernel_mul_mm[^ ]*' "$OUT/$id.timing.txt" | tail -1)" | tee -a "$OUT/README.txt"
    if [ ! -d "$OUT/$id.gputrace" ]; then
        (cd "$B" && env MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 $ENVS GGML_MM_SKINNY_SMEM_PAD=$pad "$BIN" perf -o MUL_MAT -b MTL0 -p "$FILT") > "$OUT/$id.capture.log" 2>&1
        trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$OUT/$id.capture.log" | head -1)
        [ -n "$trace" ] && [ -d "$trace" ] && mv "$trace" "$OUT/$id.gputrace" || { echo "$id: NO TRACE" | tee -a "$OUT/README.txt"; continue; }
    fi
    [ -f "$OUT/$id.replay/streamData" ] || python3 "$HEADLESS" "$OUT/$id.gputrace" "$OUT/$id.replay" > "$OUT/$id.replay.log" 2>&1 || { echo "$id: replay FAILED" | tee -a "$OUT/README.txt"; continue; }
    python3 "$STATS" --all "$OUT/$id.replay/streamData" > "$OUT/$id.stats.txt" 2>&1
    "$PY" "$B/perf/shaderprof-table.py" "$OUT/$id.replay/raw" --top 5 --json "$OUT/$id.instr.json" > "$OUT/$id.instr.txt" 2>&1
    rm -rf "$OUT/$id.replay/raw" /tmp/com.apple.gputools.profiling
done
[ -n "$LIBPID" ] && wait $LIBPID
for pad in $PADS; do
    id=pad$pad; [ -f "$OUT/$id.instr.json" ] || continue
    python3 - "$B/perf/kernel-census.py" "$OUT" "$id" <<'PY' | tee -a "$OUT/README.txt"
import importlib.util, json, sys, os
spec = importlib.util.spec_from_file_location('kc', sys.argv[1]); kc = importlib.util.module_from_spec(spec); spec.loader.exec_module(kc)
d, rid = sys.argv[2], sys.argv[3]
ks = [k for k in json.load(open(os.path.join(d, rid + '.instr.json'))) if k['role'] == 'main']
k = max(ks, key=lambda k: k['executed_total']); m = {'kernel_traced': k['kernel']}
kc.join_row(d, rid, m)
print(rid, 'join', m.get('join', 'FAILED: ' + m.get('join_error', '?')))
PY
done
echo "done $OUT" | tee -a "$OUT/README.txt"
