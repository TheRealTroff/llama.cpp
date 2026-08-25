#!/bin/bash
# Decompose the width-4 round with the landed R2 route: where do ~135 ms/round go?
#
# Three same-session instruments, one GPU at a time:
#   1. llama-bench pp4        - the bare width-4 pass, anchors the verify component
#   2. LLAMA_DECODE_PROF=1    - host-side decode split (apply/reuse/set_inputs/submit)
#   3. GGML_METAL_PROFILE=1   - per-op GPU time and counts, one encoder per op
#
# The metal profile SERIALIZES encoders (one per op) and inflates encode cost 6-8x
# (mlx-cycle-capture.md), so its totals are serialized-op time, not wall time. Use it for
# per-op shares and per-op us; use pp4 and the e2e round for wall anchors.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
PORT=${PORT:-8093}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-w4-gap-decomp-$(date +%m%d-%H%M)}
NPRED=${NPRED:-300}
# Speculation config: default is the DFlash n3 point this file first measured.
# For MTP: SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" (no -md).
SPEC_ARGS=${SPEC_ARGS:--md $MD --spec-type draft-dflash --spec-draft-n-max 3}
SKIP_BENCH=${SKIP_BENCH:-0}

COMMON_ENV=(
    GGML_MV_NC=2
    GGML_MM_SKINNY=5
    GGML_FA_VEC_MAX=5
    GGML_FA_MM_NWG=8
    GGML_GDN_FUSE_WB=1
    GGML_MV_REPACK=1
    GGML_MV_SOA_W4=1
    GGML_MV_SOA_W4_R2=1
)

mkdir -p "$OUT"
echo "=== width-4 gap decomposition: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "env    : ${COMMON_ENV[*]}"
echo "date   : $(date)"

if lsof -ti :"$PORT" >/dev/null 2>&1; then
    echo "ABORT: port $PORT busy" >&2
    exit 1
fi

echo "spec   : $SPEC_ARGS"

if [ "$SKIP_BENCH" != 1 ]; then
    echo
    echo "--- 1. llama-bench pp4: bare width-4 pass ---"
    env "${COMMON_ENV[@]}" "$BIN/llama-bench" -m "$M" -fa 1 -p 4 -n 0 -r 4 2>"$OUT/$TAG-bench.err" | tee "$OUT/$TAG-bench.txt"
fi

server_pid=
cleanup_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=
}
trap cleanup_server EXIT INT TERM

# label, extra env vars...
run_server() {
    local label=$1; shift
    local slog=$OUT/$TAG-$label.server.log
    env "${COMMON_ENV[@]}" "$@" \
        "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
        $SPEC_ARGS \
        --port "$PORT" >"$slog" 2>&1 &
    server_pid=$!
    local healthy=0
    for _ in $(seq 1 300); do
        curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { healthy=1; break; }
        kill -0 "$server_pid" 2>/dev/null || { echo "[$label] server died:"; tail -8 "$slog"; return 1; }
        sleep 2
    done
    [ "$healthy" = 1 ] || { echo "[$label] health timeout"; return 1; }
    python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
        | curl -sS -o "$OUT/$TAG-$label.json" -X POST "http://127.0.0.1:$PORT/completion" -d @-
    python3 - "$OUT/$TAG-$label.json" "$label" <<'PY'
import json, sys, hashlib
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
acc = 100.0*t.get("draft_n_accepted",0)/t["draft_n"] if t.get("draft_n") else 0.0
sha = hashlib.sha1(d.get("content","").encode()).hexdigest()[:12]
print(f"  [{sys.argv[2]}] {t.get('predicted_per_second',0):.3f} t/s  acc={acc:.1f}%  n={t.get('predicted_n',0)}  sha1={sha}")
PY
    cleanup_server
    sleep 5
}

if [ "${SKIP_DECODEPROF:-0}" != 1 ]; then
    echo
    echo "--- 2. decode-prof: host-side split ($SPEC_ARGS) ---"
    run_server decodeprof LLAMA_DECODE_PROF=1
fi

echo
echo "--- 3. metal-profile: per-op GPU time, serialized encoders ($SPEC_ARGS) ---"
run_server metalprof GGML_METAL_PROFILE=1

echo
echo "logs: $OUT/$TAG-*.server.log"
echo "=== done $(date) ==="
