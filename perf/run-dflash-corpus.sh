#!/bin/bash
# Fixed-depth DFlash timing and acceptance across the five-prompt tiny corpus.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=${PORT:-8095}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
DEPTH=${DEPTH:-7}
# KV=turbo4 runs the Turbo4 line (export TURBO_AUTO_ASYMMETRIC=0 alongside for the sym pick).
KV=${KV:-f16}
TAG=${TAG:-dflash-corpus-$KV-n$DEPTH-$(date +%m%d-%H%M)}
NPRED=${NPRED:-300}
REPS=${REPS:-2}
COOL=${COOL:-5}
TSV=$OUT/$TAG.tsv

PROMPTS=(
    "$B/perf/prompts/01-code-explain.txt"
    "$B/perf/prompts/02-prose-creative.txt"
    "$B/perf/prompts/03-chat-support.txt"
    "$B/perf/prompts/04-math-derivation.txt"
    "$B/perf/prompts/05-json-boilerplate.txt"
)

COMMON_ENV=(
    GGML_MV_NC=2
    GGML_MM_SKINNY=6
    GGML_MM_SKINNY_SOA=1
    GGML_FA_VEC_MAX=3
    GGML_FA_MM_NWG=8
    GGML_GDN_FUSE_WB=1
    GGML_MV_REPACK=1
    GGML_MV_SOA_PIN=1
    GGML_MV_SOA_W3=1
    GGML_MV_SOA_W4=1
    GGML_MV_SOA_W4_R4KP=3
    GGML_MV_SOA_W5=4
    GGML_MV_SOA_W5_HALF=1
    GGML_MV_SOA_WL_XL=1
    GGML_METAL_GET_MEMCPY=1
    DFLASH_FUSED_INJECT=1
    DFLASH_ASYNC_INJECT=1
    LLAMA_DRAFT_WINDOW=1024
    GGML_MM_ACC_HALF=1
    GGML_MM_N64=1
)

mkdir -p "$OUT"
for path in "$BIN/llama-server" "$M" "$MD" "${PROMPTS[@]}"; do
    if [ ! -e "$path" ]; then
        echo "ABORT: required path is missing: $path" >&2
        exit 1
    fi
done
if lsof -ti :"$PORT" >/dev/null 2>&1; then
    echo "ABORT: port $PORT is already busy" >&2
    exit 1
fi

printf 'label\trep\tprompt\tprompt_n\tdraft_n_max\tverify_width\ttps\tpredicted_ms\tpredicted_n\tdraft_tokens\tdraft_accepted\taccept_pct\trounds\tround_ms\toutput_per_round\tsha1\tbytes\n' > "$TSV"

echo "=== DFlash tiny corpus at n=$DEPTH, KV $KV: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "env    : ${COMMON_ENV[*]}"
echo "run    : ${#PROMPTS[@]} prompts, $REPS mirrored fresh-process passes, n_predict $NPRED"
echo "date   : $(date)"
echo

server_pid=
cleanup_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 25); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=
}
trap cleanup_server EXIT INT TERM

run_one() {
    local rep=$1 prompt=$2
    local name label slog response
    name=$(basename "$prompt" .txt)
    label=$name-r$rep
    slog=$OUT/$TAG-$label.server.log
    response=$OUT/$TAG-$label.json

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT busy before start"
        return 1
    fi

    env "${COMMON_ENV[@]}" "$BIN/llama-server" \
        -m "$M" -c 10240 -fa on -ctk "$KV" -ctv "$KV" \
        -md "$MD" --spec-type draft-dflash --spec-draft-n-max "$DEPTH" \
        --port "$PORT" >"$slog" 2>&1 &
    server_pid=$!

    local healthy=0
    for _ in $(seq 1 300); do
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
            healthy=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[$label] server died during startup:"
            tail -12 "$slog"
            return 1
        fi
        sleep 2
    done
    if [ "$healthy" -ne 1 ]; then
        echo "[$label] health timeout"
        return 1
    fi
    if ! lsof -ti :"$PORT" 2>/dev/null | grep -qx "$server_pid"; then
        echo "[$label] ABORT: another process, not PID $server_pid, owns port $PORT"
        return 1
    fi

    local http_code
    http_code=$(python3 -c \
        "import json; print(json.dumps({'prompt': open('$prompt').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
        | curl -sS -o "$response" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/completion" -d @-)
    if [ "$http_code" != 200 ]; then
        echo "[$label] HTTP $http_code:"
        python3 -m json.tool "$response" 2>/dev/null || head -c 1000 "$response"
        echo
        return 1
    fi

    python3 - "$response" "$TSV" "$label" "$rep" "$name" "$DEPTH" <<'PY'
import hashlib
import json
import sys

response, tsv_path, label, rep, name, depth = sys.argv[1:]
d = json.load(open(response))
if "error" in d:
    raise SystemExit(f"[{label}] ERROR {json.dumps(d['error'])[:300]}")
t = d.get("timings", {})
content = d.get("content", "")
tps = float(t.get("predicted_per_second", 0))
predicted_ms = float(t.get("predicted_ms", 0))
predicted_n = int(t.get("predicted_n", 0))
prompt_n = int(t.get("prompt_n", 0))
draft_tokens = int(t.get("draft_n", 0))
accepted = int(t.get("draft_n_accepted", 0))
rounds = predicted_n - accepted
if predicted_n <= 0 or draft_tokens <= 0 or rounds <= 0:
    raise SystemExit(f"[{label}] invalid speculative counters: {t}")
accept_pct = 100.0 * accepted / draft_tokens
round_ms = predicted_ms / rounds
output_per_round = predicted_n / rounds
sha = hashlib.sha1(content.encode()).hexdigest()[:12]
width = int(depth) + 1
with open(tsv_path, "a") as f:
    f.write(f"{label}\t{rep}\t{name}\t{prompt_n}\t{depth}\t{width}\t{tps:.6f}\t{predicted_ms:.3f}\t"
            f"{predicted_n}\t{draft_tokens}\t{accepted}\t{accept_pct:.4f}\t{rounds}\t{round_ms:.6f}\t"
            f"{output_per_round:.6f}\t{sha}\t{len(content.encode())}\n")
print(f"  [{name:<19}] prompt_n={prompt_n:3d}  {tps:7.3f} t/s  round={round_ms:7.2f} ms  "
      f"out/round={output_per_round:4.2f}  acc={accept_pct:5.1f}%  sha1={sha}")
PY

    cleanup_server
    sleep "$COOL"
}

echo "--- warmup, discarded ---"
run_one 0 "${PROMPTS[0]}" >/dev/null 2>&1 || true

for rep in $(seq 1 "$REPS"); do
    if [ $((rep % 2)) -eq 1 ]; then
        order=("${PROMPTS[@]}")
    else
        order=()
        for ((i=${#PROMPTS[@]} - 1; i >= 0; --i)); do
            order+=("${PROMPTS[i]}")
        done
    fi
    echo "--- pass $rep ---"
    for prompt in "${order[@]}"; do
        run_one "$rep" "$prompt"
    done
done

echo
echo "--- summary by prompt ---"
python3 - "$TSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
groups = defaultdict(list)
for row in rows:
    if row["rep"] != "0":
        groups[row["prompt"]].append(row)

print("  prompt                p_n   t/s      round ms  out/round  acceptance  runs  sha1")
for name, rs in sorted(groups.items()):
    hashes = {r["sha1"] for r in rs}
    if len(hashes) != 1:
        raise SystemExit(f"ABORT: {name} is not deterministic across reps: {sorted(hashes)}")
    mean = lambda field: statistics.mean(float(r[field]) for r in rs)
    print(f"  {name:<21} {int(rs[0]['prompt_n']):>3}  {mean('tps'):7.3f}  {mean('round_ms'):10.2f}  "
          f"{mean('output_per_round'):9.3f}  {mean('accept_pct'):9.2f}%  {len(rs):>4}  {next(iter(hashes))}")
PY

echo
echo "results: $TSV"
echo "=== done $(date) ==="
