#!/bin/bash
# Mirrored full-server prefill A/B for the width-512 n64 mul_mm kernel.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf}
PORT=${PORT:-8096}
COOLDOWN=${COOLDOWN:-30}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-mm-acch-n64-e2e-$(date +%m%d-%H%M)}

mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          GGML_MM_ACC_HALF=1)
PICK_SPEC=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max 4)

server_pid=
stop_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 25); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -KILL "$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=
}
trap stop_server EXIT

run_one() {
    local label=$1
    local use_n64=$2
    local slog=$OUT/$TAG-$label.server.log
    local response=$OUT/$TAG-$label.response.json
    local request=$OUT/$TAG-$label.request.json

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT is already in use"
        return 1
    fi

    if [ "$use_n64" = 1 ]; then
        env "${PICK_ENV[@]}" GGML_MM_N64=1 "$BIN/llama-server" \
            -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
            "${PICK_SPEC[@]}" --port "$PORT" >"$slog" 2>&1 &
    else
        env "${PICK_ENV[@]}" "$BIN/llama-server" \
            -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
            "${PICK_SPEC[@]}" --port "$PORT" >"$slog" 2>&1 &
    fi
    server_pid=$!

    local ready=0
    for _ in $(seq 1 200); do
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[$label] server exited during startup"
            tail -8 "$slog"
            return 1
        fi
        sleep 2
    done
    if [ "$ready" != 1 ]; then
        echo "[$label] health timeout"
        return 1
    fi
    if ! lsof -ti :"$PORT" 2>/dev/null | rg -qx "$server_pid"; then
        echo "[$label] ABORT: another process owns port $PORT"
        return 1
    fi

    python3 -c "
import json
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': 8, 'temperature': 0}))
" >"$request"

    local wall
    wall=$(curl -sS -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d @"$request" \
        -o "$response" -w '%{time_total}')

    LABEL="$label" WALL="$wall" python3 -c "
import hashlib, json, os
d = json.load(open('$response'))
if 'error' in d:
    raise SystemExit('[%s] ERROR %s' % (os.environ['LABEL'], json.dumps(d['error'])[:240]))
t = d.get('timings', {})
c = d.get('content', '')
draft_n = t.get('draft_n', 0)
acc = 100*t.get('draft_n_accepted', 0)/draft_n if draft_n else 0
print('[%-6s] prompt_n=%5d prefill=%8.1f ms %6.2f t/s request=%7.3f s gen=%6.2f t/s acc=%5.1f%% sha1=%s' % (
    os.environ['LABEL'], t.get('prompt_n', 0), t.get('prompt_ms', 0),
    t.get('prompt_per_second', 0), float(os.environ['WALL']),
    t.get('predicted_per_second', 0), acc,
    hashlib.sha1(c.encode()).hexdigest()[:12]))
"

    stop_server
    sleep "$COOLDOWN"
}

echo "=== n64 full-server e2e: $TAG ==="
echo "repo     : $B"
echo "commit   : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "dirty    : $(cd "$B" && git status --porcelain | wc -l | tr -d ' ') files"
echo "cooldown : $COOLDOWN s"
echo "env      : ${PICK_ENV[*]}"
echo

run_one base1 0
run_one n64-1 1
run_one n64-2 1
run_one base2 0

echo "logs: $OUT/$TAG-*"
