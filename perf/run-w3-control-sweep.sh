#!/bin/bash
# Reproducible route/correctness or performance sweep for the width-3 control cells.
set -euo pipefail

B=${B:-/Users/troff/play/llama.cpp-mv-w3-control-sweep}
BIN=${BIN:-$B/build-w3ctl/bin/test-backend-ops}
MODE=${MODE:-perf}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-w3-control-$(date +%m%d-%H%M)}
ONLY=${ONLY:-.*}
ORDER=${ORDER:-}

mkdir -p "$OUT"

names=(base r4k2f r4k1h r4k1f r8k2h r8k2f r8k1h r8k1f r4k2hi r4k2fi r4k1hi r4k1fi r4i)
ids=(0 1 2 3 4 5 6 7 8 9 10 11 12)
base_env=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 GGML_MV_SOA_W3=1 GGML_METAL_LOG_LEVEL=2)

case "$MODE" in
    test) mode_args=(test -o MUL_MAT -b MTL0 --test-file "$B/perf/w3-real-projections.ops") ;;
    perf) mode_args=(perf -o MUL_MAT -b MTL0 --test-file "$B/perf/w3-real-projections.ops") ;;
    *) echo "MODE must be test or perf" >&2; exit 2 ;;
esac

log="$OUT/$TAG-$MODE.log"
: > "$log"
if [[ -n $ORDER ]]; then
    read -r -a run_names <<< "$ORDER"
else
    run_names=("${names[@]}")
fi

for name in "${run_names[@]}"; do
    if [[ ! $name =~ $ONLY ]]; then
        continue
    fi
    id=-1
    for i in "${!names[@]}"; do
        if [[ ${names[$i]} == "$name" ]]; then
            id=${ids[$i]}
            break
        fi
    done
    if [[ $id == -1 ]]; then
        echo "unknown arm: $name" >&2
        exit 2
    fi
    echo "ARM=$name ID=$id" | tee -a "$log"
    env "${base_env[@]}" GGML_MV_SOA_W3_CTL="$id" \
        "$BIN" "${mode_args[@]}" 2>&1 | tee -a "$log"
done

echo "log=$log"
