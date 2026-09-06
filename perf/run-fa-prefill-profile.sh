#!/bin/bash
# Prefill FA (perf/ud-model.md step 9): time + capture + headless replay + per-instruction decode of the
# mm flash-attention kernel at the prefill form (512 query rows, nwg=1, nsg=4) on the Qwen3.8 geometry
# (dk=dv=256, 24/4 heads, mask). Same steps as run-ud-soa-profile.sh; the pipeline name is read from
# each invocation's own stderr. Timing from the UNCAPTURED runs only.
set -u
B=${B:-/Users/troff/play/llama.cpp-ud-soa}
BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/profiles/fa-prefill-sep05}
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
STATS=/Users/troff/.claude/skills/metal-gpu-profile/references/gpuprofiler-stats.py
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
ENVS=${ENVS:-"GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8"}
ARMS=${ARMS:-"kv8448:8448 kv16384:16384"}
REPS=${REPS:-2}
mkdir -p "$OUT"
for arm in $ARMS; do
    label=${arm%%:*}; kv=${arm#*:}
    P="kv=$kv,nb=512"
    for rep in $(seq 1 $REPS); do
        out=$(cd "$B" && env $ENVS "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "$P" 2>&1)
        echo "$label rep$rep: $(echo "$out" | grep -oE '[0-9]+\.[0-9]+ us/run' | head -1)  $(echo "$out" | grep -oE 'loaded kernel_flash_attn_ext[A-Za-z0-9_=]+' | tail -1)"
    done
    if [ ! -d "$OUT/$label.gputrace" ]; then
        log=$OUT/$label.capture.log
        ( cd "$B" && env MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 $ENVS "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "$P" ) >"$log" 2>&1
        trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$log" | head -1)
        [ -n "$trace" ] && [ -d "$trace" ] || { echo "$label: NO TRACE (see $log)"; continue; }
        mv "$trace" "$OUT/$label.gputrace"; echo "$label: captured $(du -sh "$OUT/$label.gputrace" | cut -f1)"
    fi
    [ -f "$OUT/$label.replay/streamData" ] || python3 "$HEADLESS" "$OUT/$label.gputrace" "$OUT/$label.replay" >"$OUT/$label.replay.log" 2>&1 || { echo "$label: replay FAILED"; continue; }
    python3 "$STATS" --all "$OUT/$label.replay/streamData" >"$OUT/$label.stats.txt" 2>&1
    "$PY" "$B/perf/shaderprof-table.py" "$OUT/$label.replay/raw" --kernel flash_attn --top 80 --json "$OUT/$label.instr.json" >"$OUT/$label.instr.txt" 2>&1
    echo "$label: decoded, replay $(du -sh "$OUT/$label.replay" | cut -f1)"
done
echo "done: $OUT"
