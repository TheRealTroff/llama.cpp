#!/bin/bash
# Benchmark exact applegpu-nt register budgets selected via MTLBinaryArchive.
set -euo pipefail

B=${B:-/Users/troff/play/llama.cpp-mv-w3-control-sweep}
BIN=${BIN:-$B/build-w3air/bin/test-backend-ops}
METALLIB_DST=${METALLIB_DST:-$B/build-w3air/bin/default.metallib}
AIR_ROOT=${AIR_ROOT:-/tmp/w3-air}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-w3-air-register-$(date +%m%d-%H%M)}
ORDER=${ORDER:-plain default 217 216 208 192}

mkdir -p "$OUT"
cp "$AIR_ROOT/base/default.metallib" "$METALLIB_DST"
log="$OUT/$TAG-perf.log"
: > "$log"

read -r -a variants <<< "$ORDER"
for variant in "${variants[@]}"; do
    archive=
    case "$variant" in
        plain) ;;
        default) archive="$AIR_ROOT/base/default.archive" ;;
        217|216|208|192) archive="$AIR_ROOT/reg$variant/pipelines.archive" ;;
        *) echo "unknown register arm: $variant" >&2; exit 2 ;;
    esac

    if [[ -n $archive ]]; then
        hash=$(shasum -a 256 "$archive" | awk '{print $1}')
        echo "ARM=reg-$variant SHA256=$hash" | tee -a "$log"
        env GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 \
            GGML_MV_SOA_W3=1 GGML_MV_SOA_W3_CTL=0 GGML_METAL_LOG_LEVEL=2 \
            GGML_METAL_BINARY_ARCHIVE="$archive" \
            "$BIN" perf -o MUL_MAT -b MTL0 \
            --test-file "$B/perf/w3-real-projections.ops" 2>&1 | tee -a "$log"
    else
        echo "ARM=reg-plain" | tee -a "$log"
        env GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 \
            GGML_MV_SOA_W3=1 GGML_MV_SOA_W3_CTL=0 GGML_METAL_LOG_LEVEL=2 \
            "$BIN" perf -o MUL_MAT -b MTL0 \
            --test-file "$B/perf/w3-real-projections.ops" 2>&1 | tee -a "$log"
    fi
done

echo "log=$log"
