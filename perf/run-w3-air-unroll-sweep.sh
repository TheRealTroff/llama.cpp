#!/bin/bash
# Benchmark edited width-3 AIR libraries through a non-embedded Metal build.
set -euo pipefail

B=${B:-/Users/troff/play/llama.cpp-mv-w3-control-sweep}
BIN=${BIN:-$B/build-w3air/bin/test-backend-ops}
METALLIB_DST=${METALLIB_DST:-$B/build-w3air/bin/default.metallib}
AIR_ROOT=${AIR_ROOT:-/tmp/w3-air}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-w3-air-unroll-$(date +%m%d-%H%M)}
ORDER=${ORDER:-base 2 3 4 5 6 7 8}
PREFIX=${PREFIX:-unroll}

mkdir -p "$OUT"
log="$OUT/$TAG-perf.log"
: > "$log"

read -r -a variants <<< "$ORDER"
for variant in "${variants[@]}"; do
    if [[ $variant == base ]]; then
        source_lib="$AIR_ROOT/base/default.metallib"
    else
        source_lib="$AIR_ROOT/$PREFIX$variant/default.metallib"
    fi

    cp "$source_lib" "$METALLIB_DST"
    hash=$(shasum -a 256 "$source_lib" | awk '{print $1}')
    echo "ARM=$PREFIX-$variant SHA256=$hash" | tee -a "$log"
    env GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 \
        GGML_MV_SOA_W3=1 GGML_MV_SOA_W3_CTL=0 GGML_METAL_LOG_LEVEL=2 \
        "$BIN" perf -o MUL_MAT -b MTL0 \
        --test-file "$B/perf/w3-real-projections.ops" 2>&1 | tee -a "$log"
done

echo "log=$log"
