#!/bin/bash
# Isolated kernel timings only. Real-model validation remains a separate gate.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROD=${PROD:-/Users/troff/play/llama.cpp-prod}
MODEL=${MODEL:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf}
OUT=${OUT:-$ROOT/perf/results/ud-remaining-20260908}
mkdir -p "$OUT"
source "$PROD/perf/pick.sh"
pick_check ud
pick_env ud
{
    date -u
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" branch --show-current
    git -C "$ROOT" status --short
    printf 'MODEL=%s\n' "$MODEL"
    printf 'ENV=%s\n' "${PICK_ENV[*]}"
    printf 'ORDER=plain-1 soa-1 soa-2 plain-2\n'
    printf 'WIDTHS=1,2,3,4,5,6,7,8,9,32,512\n'
} > "$OUT/manifest.txt"
git -C "$ROOT" diff > "$OUT/source.patch"
for arm in plain-1 soa-1 soa-2 plain-2; do
    case "$arm" in
        plain-*) types='(iq4_nl|q3_K|q6_K|iq3_s)' ;;
        soa-*) types='(iq4_nl_soa|q3_K_soa|q6_K_soa|iq3_s_soa)' ;;
    esac
    printf '%s START %s\n' "$(date -u +%FT%TZ)" "$arm"
    env "${PICK_ENV[@]}" GGML_METAL_LOG_LEVEL=2 GGML_TEST_UD_GGUF="$MODEL" \
        "$ROOT/build/bin/test-backend-ops" perf -o MUL_MAT -b MTL0 \
        -p "type_a=$types,.*ud_remaining=1" > "$OUT/$arm.log" 2>&1
    count=$(grep -c 'runs -' "$OUT/$arm.log")
    if [ "$count" != 132 ]; then
        printf 'ERROR: %s produced %s timings, expected 132\n' "$arm" "$count" >&2
        exit 1
    fi
    printf '%s DONE %s (%s timings)\n' "$(date -u +%FT%TZ)" "$arm" "$count"
done
