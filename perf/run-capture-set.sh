#!/bin/bash
# Collect a matched set of GPU captures across the width 3-4 cliff, archive them out of
# /tmp, and dump each one headlessly.
#
# Why archive: on 2026-08-23 the previous session's replay output (/tmp/com.apple.gputools
# .profiling) and the oMLX capture (/tmp/dflash-b4.gputrace) were both gone - /tmp does not
# survive. Only the eight SUMMARY fields transcribed into width4-verify.md survived, and
# gpuprofiler-stats.py --all was never run on them. Captures are free and headless; the
# replay click is the expensive step, so keep every capture that a click might be spent on.
#
# What a capture gives without the click: structure - pipeline identity, function-constant
# values, buffer bindings and offsets, dispatch order and dispatch geometry. NOT timing and
# NOT counters (toolchain-isa-probe.md:237). Counters need step 2 of skills/metal-gpu-profile.
set -u

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin/test-backend-ops
PY=/Users/troff/play/.venv-convert/bin/python3
XFW=/Applications/Xcode.app/Contents/SharedFrameworks
DEST=${DEST:-/Users/troff/play/kvquant-experiments/traces/aug23}
mkdir -p "$DEST"

PROD_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5)

echo "=== capture set -> $DEST ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "date   : $(date)"
echo

# name m k n extra_env...
CASES=(
    "w1-ffn_down-mv          5120  17408 1"
    "w2-ffn_down-mvnc2       5120  17408 2"
    "w3-ffn_down-ext-nx8     5120  17408 3"
    "w4-ffn_down-ext-nx8     5120  17408 4"
    "w3-ffn_down-ext-nx16    5120  17408 3 GGML_MV_EXT_NXPSG=16"
    "w4-ffn_down-ext-nx16    5120  17408 4 GGML_MV_EXT_NXPSG=16"
    "w4-ffn_down-ext-nof16y  5120  17408 4 GGML_MV_EXT_F16Y=0"
    "w5-ffn_down-skinny      5120  17408 5"
    "w4-attn_q-ext-nx8       3072  5120  4"
    "w4-attn_q-ext-nx16      3072  5120  4 GGML_MV_EXT_NXPSG=16"
)

for c in "${CASES[@]}"; do
    read -r name m k n rest <<< "$c"
    # shellcheck disable=SC2206
    extra=($rest)
    echo "--- $name  (m=$m k=$k n=$n ${extra[*]:-}) ---"

    log=$(env "${PROD_ENV[@]}" ${extra[@]:+"${extra[@]}"} \
            MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 \
            "$BIN" perf -o MUL_MAT -b MTL0 \
            -p "type_a=q4_0,type_b=f32,m=$m,n=$n,k=$k," 2>&1)

    src=$(echo "$log" | grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' | head -1)
    if [ -z "$src" ] || [ ! -d "$src" ]; then
        echo "  NO CAPTURE - capture line was:"
        echo "$log" | grep -i capture | head -3
        continue
    fi

    # the pipelines that were actually compiled, so routing is recorded next to the trace
    echo "$log" | grep -oE "loaded kernel_[a-z0-9_]+(_nsg=[0-9]+_nxpsg=[0-9]+_nr0=[0-9]+)?" \
      | sed 's/loaded /  pipeline: /' | sort -u

    rm -rf "$DEST/$name.gputrace"
    mv "$src" "$DEST/$name.gputrace"
    echo "  archived: $(du -sh "$DEST/$name.gputrace" | cut -f1)"

    DYLD_FRAMEWORK_PATH=$XFW "$PY" "$B/perf/gputrace-dump.py" \
        "$DEST/$name.gputrace" "$DEST/$name.txt" 2>&1 | sed 's/^/  /'

    # dispatch geometry histogram: {threadgroups} x {threadsPerThreadgroup}, and how many
    echo "  geometry:"
    grep -oE "\{[0-9]+ul, [0-9]+ul, [0-9]+ul\}, \{[0-9]+ul, [0-9]+ul, [0-9]+ul\}" "$DEST/$name.txt" \
      | sort | uniq -c | sed 's/^/    /'
    echo
done

echo "=== done $(date) ==="
du -sh "$DEST"
