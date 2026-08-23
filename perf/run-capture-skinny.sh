#!/bin/bash
# The matched skinny width pair the tile question actually needs.
#
# width4-limiter.md refuted bandwidth-bound (width 4 sits at 47% of 273 GB/s while batch 1
# hits 92%) and established that tile waste CANNOT be tested from the aug23 set, for two
# reasons: every w3/w4 capture runs kernel_mul_mv_ext_*, and only w5 runs skinny with no
# matched partner. This fixes the first half - captures of kernel_mul_mm_skinny at MATCHED
# widths, which nothing routed to width 4 until GGML_MM_SKINNY=4 was measured today.
#
# The discriminator, already validated on mul_mv_ext (returns 1.247/1.251 against
# predictions of 1.333 and 1.000): instructions per weight byte, width 4 vs width 8.
#   1.000 => fixed 8-wide tile, work does not scale with columns  => tile waste is real
#   2.000 => work scales with columns                            => tile waste is not the story
# Normalise by DRAM read transactions, not wall time, so captures of different length compare.
#
# Both kernel variants are captured because they are different kernels: _di reads the
# GGML_MV_REPACK deinterleaved copy, plain does not. Both accumulate into simdgroup_half8x8,
# so both carry the fixed 8-wide column tile the question is about.
#
# Captures are free and headless. They give structure and dispatch geometry, NOT counters -
# counters need the replay, which is agent C's open work.
set -u

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin/test-backend-ops
PY=/Users/troff/play/.venv-convert/bin/python3
XFW=/Applications/Xcode.app/Contents/SharedFrameworks
DEST=${DEST:-/Users/troff/play/kvquant-experiments/traces/aug23-skinny}
mkdir -p "$DEST"

echo "=== skinny width captures -> $DEST ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "date   : $(date)"
echo

# name                 m     k     n  extra_env...
CASES=(
    "s4-ffn_down-w4-plain  5120 17408 4 GGML_MM_SKINNY=4"
    "s4-ffn_down-w6-plain  5120 17408 6 GGML_MM_SKINNY=4"
    "s4-ffn_down-w8-plain  5120 17408 8 GGML_MM_SKINNY=4"
    "s4-ffn_down-w4-di     5120 17408 4 GGML_MM_SKINNY=4 GGML_MV_REPACK=1"
    "s4-ffn_down-w6-di     5120 17408 6 GGML_MM_SKINNY=4 GGML_MV_REPACK=1"
    "s4-ffn_down-w8-di     5120 17408 8 GGML_MM_SKINNY=4 GGML_MV_REPACK=1"
)

for c in "${CASES[@]}"; do
    read -r name m k n rest <<< "$c"
    # shellcheck disable=SC2206
    extra=($rest)
    echo "--- $name  (m=$m k=$k n=$n ${extra[*]:-}) ---"

    log=$(env GGML_MV_NC=2 ${extra[@]:+"${extra[@]}"} \
            MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 \
            "$BIN" perf -o MUL_MAT -b MTL0 \
            -p "type_a=q4_0,type_b=f32,m=$m,n=$n,k=$k," 2>&1)

    src=$(echo "$log" | grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' | head -1)
    if [ -z "$src" ] || [ ! -d "$src" ]; then
        echo "  NO CAPTURE - capture line was:"; echo "$log" | grep -i capture | head -3; continue
    fi

    # ROUTING GATE: the whole point is that skinny ran. If mul_mv_ext shows up at n=4 the
    # capture is worthless for this question - say so loudly rather than archiving it quietly.
    pipes=$(echo "$log" | grep -oE "loaded kernel_[a-z0-9_]+(_nsg=[0-9]+_nxpsg=[0-9]+_nr0=[0-9]+)?" | sed 's/loaded //' | sort -u)
    echo "$pipes" | sed 's/^/  pipeline: /'
    if ! echo "$pipes" | grep -q "kernel_mul_mm_skinny"; then
        echo "  *** WARNING: no skinny pipeline loaded - this capture does NOT answer the question"
    fi

    rm -rf "$DEST/$name.gputrace"
    mv "$src" "$DEST/$name.gputrace"
    echo "  archived: $(du -sh "$DEST/$name.gputrace" | cut -f1)"

    DYLD_FRAMEWORK_PATH=$XFW "$PY" "$B/perf/gputrace-dump.py" \
        "$DEST/$name.gputrace" "$DEST/$name.txt" 2>&1 | sed 's/^/  /'

    echo "  geometry:"
    grep -oE "\{[0-9]+ul, [0-9]+ul, [0-9]+ul\}, \{[0-9]+ul, [0-9]+ul, [0-9]+ul\}" "$DEST/$name.txt" \
      | sort | uniq -c | sed 's/^/    /'
    echo
done

echo "=== done $(date) ==="
du -sh "$DEST"
