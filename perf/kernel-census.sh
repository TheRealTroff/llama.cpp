#!/bin/bash
# Kernel census driver (perf/kernel-census.md): profile log -> plan -> per kernel: uncaptured timing
# (2 reps), capture, headless replay, per-instruction decode -> metrics -> report + snapshot.
#   perf/kernel-census.sh <profiled server.log> [TAG]     env: B, ENVS (routing env of the pick), TOP, MIN_MS, PREV (snapshot to diff)
# Run it with the SAME routing env the profiled run used, or the captures measure other kernels.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
LOG=$1; TAG=${2:-census-$(date +%m%d-%H%M)}
B=${B:-/Users/troff/play/llama.cpp-ud-soa}
BIN=$B/build/bin/test-backend-ops
ENVS=${ENVS:-}
TOP=${TOP:-16}; MIN_MS=${MIN_MS:-0}
PREV=${PREV:-}
OUT=/Users/troff/play/kvquant-experiments/census/$TAG
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
STATS=/Users/troff/.claude/skills/metal-gpu-profile/references/gpuprofiler-stats.py
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
mkdir -p "$OUT"; exec > >(tee "$OUT/census.log") 2>&1
# the translator metallib for the per-row machine-IR join (perf/agx-nt-opt.py): same source and macros
# as the embedded library the binary runs (perf/agx-backend-access.md); rebuilt when the source is newer
export CENSUS_METALLIB=${CENSUS_METALLIB:-$B/build-air/ggml.metallib}
if [ ! -f "$CENSUS_METALLIB" ] || [ "$B/ggml/src/ggml-metal/ggml-metal.metal" -nt "$CENSUS_METALLIB" ]; then
    mkdir -p "$(dirname "$CENSUS_METALLIB")"
    (cd "$B" && xcrun metal -fmodules-cache-path=/tmp/air-module-cache -DGGML_METAL_HAS_BF16=1 -DTURBO_USE_4MAG=1 -DTURBO_USE_PAIR_LUT=1 \
        -c ggml/src/ggml-metal/ggml-metal.metal -Iggml/src/ggml-metal -Iggml/src -o "${CENSUS_METALLIB%.metallib}.air" > "${CENSUS_METALLIB%.metallib}.log" 2>&1 \
        && xcrun metallib "${CENSUS_METALLIB%.metallib}.air" -o "$CENSUS_METALLIB") || echo "translator metallib build FAILED (see ${CENSUS_METALLIB%.metallib}.log); rows get no join"
fi
echo "=== kernel census $TAG: $LOG"; echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)  env: $ENVS"
python3 "$B/perf/kernel-census.py" plan "$LOG" --top "$TOP" --min-ms "$MIN_MS" > "$OUT/plan.json"
n=$(python3 -c "import json;print(len(json.load(open('$OUT/plan.json'))))"); echo "rows: $n"
for i in $(seq 0 $((n-1))); do
    python3 -c "import json;print(json.dumps(json.load(open('$OUT/plan.json'))[$i]))" > "$OUT/row$i.json"
    id=$(python3 -c "import json;print(json.load(open('$OUT/row$i.json'))['id'])"); filt=$(python3 -c "import json;print(json.load(open('$OUT/row$i.json'))['filter'] or '')"); op=$(python3 -c "import json;print(json.load(open('$OUT/row$i.json'))['op'])")
    if [ -z "$filt" ]; then echo "$id: no filter for op $op (extend case_filter)"; python3 "$B/perf/kernel-census.py" metrics "$OUT" "$OUT/row$i.json"; continue; fi
    : > "$OUT/$id.timing.txt"
    for rep in 1 2; do (cd "$B" && env GGML_MV_REPACK=2 $ENVS "$BIN" perf -o "$op" -b MTL0 -p "$filt" 2>&1) | grep -E 'us/run|loaded kernel_' >> "$OUT/$id.timing.txt"; done
    if ! grep -q 'us/run' "$OUT/$id.timing.txt"; then echo "$id: NO PERF CASE for '$filt'"; python3 "$B/perf/kernel-census.py" metrics "$OUT" "$OUT/row$i.json"; continue; fi
    # a cached trace is stale when the routed kernel changed since it was captured (the timing run names the kernel)
    if [ -d "$OUT/$id.gputrace" ] && [ -f "$OUT/$id.capture.log" ]; then
        # the full specialization counts: nwg/nsg/gqah/qr/soa/bci are routing, and a cached trace of another
        # specialization is the wrong kernel (2026-09-16: an nwg change silently reused the nwg=8 capture)
        # every pipeline the timing run loaded (the main kernel and its reduce/aux) must be in the capture log
        for tk in $(grep -oE 'loaded kernel_[A-Za-z0-9_=,]+' "$OUT/$id.timing.txt" | grep -v 'cpy\|cvt' | sed 's/loaded //' | sort -u); do
            if ! grep -q "loaded $tk" "$OUT/$id.capture.log"; then echo "$id: kernel changed ($tk), recapturing"; rm -rf "$OUT/$id.gputrace" "$OUT/$id.replay" "$OUT/$id.instr.json"; break; fi
        done
    fi
    if [ ! -d "$OUT/$id.gputrace" ]; then
        (cd "$B" && env MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 GGML_MV_REPACK=2 $ENVS "$BIN" perf -o "$op" -b MTL0 -p "$filt") > "$OUT/$id.capture.log" 2>&1
        trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$OUT/$id.capture.log" | head -1)
        [ -n "$trace" ] && [ -d "$trace" ] && mv "$trace" "$OUT/$id.gputrace" || { echo "$id: NO TRACE"; python3 "$B/perf/kernel-census.py" metrics "$OUT" "$OUT/row$i.json"; continue; }
    fi
    [ -f "$OUT/$id.replay/streamData" ] || python3 "$HEADLESS" "$OUT/$id.gputrace" "$OUT/$id.replay" > "$OUT/$id.replay.log" 2>&1 || { echo "$id: replay FAILED"; continue; }
    python3 "$STATS" --all "$OUT/$id.replay/streamData" > "$OUT/$id.stats.txt" 2>&1
    "$PY" "$B/perf/shaderprof-table.py" "$OUT/$id.replay/raw" --top 5 --json "$OUT/$id.instr.json" > "$OUT/$id.instr.txt" 2>&1
    rm -rf "$OUT/$id.replay/raw" /tmp/com.apple.gputools.profiling   # each replay leaves ~1.7 GB in /tmp; two census passes filled the disk (2026-09-06)
    python3 "$B/perf/kernel-census.py" metrics "$OUT" "$OUT/row$i.json"
done
echo; python3 "$B/perf/kernel-census.py" report "$OUT" ${PREV:+--diff "$PREV"} --snapshot "$OUT/snapshot.json"
echo "snapshot: $OUT/snapshot.json"
