#!/bin/bash
# Full-model long prefill only: unchanged UD versus Q3_K-only stored rows.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROD=${PROD:-/Users/troff/play/llama.cpp-prod}
BASE=${BASE:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf}
: "${CANDIDATE:?Set CANDIDATE to the verified Q3_K-only converted GGUF}"
OUT=${OUT:-$ROOT/perf/results/ud-q3k-e2e-20260908}
PROMPTS=${PROMPTS:-8192}
REPS=${REPS:-1}
mkdir -p "$OUT"
source "$PROD/perf/pick.sh"
pick_check ud
pick_env ud
ARGS=(-p "$PROMPTS" -n 0 -b 2048 -ub 512 -r "$REPS" -fa on -ctk turbo4 -ctv turbo4 -ngl 99 -o jsonl -v --progress)
{
    date -u
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" status --short
    printf 'BASE=%s\nCANDIDATE=%s\n' "$BASE" "$CANDIDATE"
    printf 'ENV=%s\n' "${PICK_ENV[*]}"
    printf 'ARGS=%s\n' "${ARGS[*]}"
    printf 'ORDER=plain-1 soa-1 soa-2 plain-2\n'
} > "$OUT/manifest-pp$PROMPTS.txt"
for arm in plain-1 soa-1 soa-2 plain-2; do
    output="$OUT/pp$PROMPTS-$arm"
    if [ -e "$output.jsonl" ] || [ -e "$output.log" ]; then
        printf 'Refusing to overwrite %s\n' "$output" >&2
        exit 1
    fi
    model=$BASE
    case "$arm" in soa-*) model=$CANDIDATE ;; esac
    printf '%s START pp%s %s\n' "$(date -u +%FT%TZ)" "$PROMPTS" "$arm"
    env "${PICK_ENV[@]}" GGML_METAL_LOG_LEVEL=2 \
        "$ROOT/build/bin/llama-bench" -m "$model" "${ARGS[@]}" > "$output.jsonl" 2> "$output.log"
    python3 - "$output.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows) == 1, rows
row = rows[0]
assert row['n_prompt'] > 0 and row['n_gen'] == 0 and row['avg_ts'] > 0, row
print('pp%d: %.3f t/s, %.3f seconds' % (row['n_prompt'], row['avg_ts'], row['avg_ns']/1e9))
PY
    printf '%s DONE pp%s %s\n' "$(date -u +%FT%TZ)" "$PROMPTS" "$arm"
done
