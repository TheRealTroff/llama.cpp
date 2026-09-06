#!/bin/bash
# Fresh-process DFlash depth sweep for the combined adaptive-width kernel set.
# Draft depth n verifies n+1 target columns: nc2, SoA w3/w4/w5, then skinny SoA w6-w8.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-skinny-soa}
BIN=${BIN:-$B/build/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
PORT=${PORT:-8096}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-dflash-full-gamut-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
REPS=${REPS:-2}
DEPTHS=${DEPTHS:-"1 2 3 4 5 6 7"}
COOL=${COOL:-5}
WARMUP=${WARMUP:-1}
TSV=$OUT/$TAG.tsv

mkdir -p "$OUT"

for path in "$BIN/llama-server" "$M" "$MD" "$PROMPT"; do
    if [ ! -e "$path" ]; then
        echo "ABORT: required path is missing: $path" >&2
        exit 1
    fi
done

if lsof -ti :"$PORT" >/dev/null 2>&1; then
    echo "ABORT: port $PORT is already busy" >&2
    exit 1
fi

COMMON_ENV=(
    GGML_MV_NC=2
    GGML_MM_SKINNY=6
    GGML_MM_SKINNY_SOA=1
    GGML_FA_VEC_MAX=5
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
    LLAMA_GDN_REPLAY=1
)

printf 'label\trep\tdraft_n_max\tverify_width\ttps\tpredicted_ms\tpredicted_n\tdraft_tokens\tdraft_accepted\taccept_pct\trounds\tround_ms\toutput_per_round\tsha1\tbytes\n' > "$TSV"

echo "=== DFlash full gamut: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "prompt : $(wc -c < "$PROMPT" | tr -d ' ') bytes, sha1 $(shasum "$PROMPT" | cut -c1-12)"
echo "env    : ${COMMON_ENV[*]}"
echo "depths : $DEPTHS; $REPS mirrored fresh-process passes; n_predict $NPRED"
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
    local rep=$1 depth=$2
    local label=n$depth-r$rep
    local slog=$OUT/$TAG-$label.server.log
    local response=$OUT/$TAG-$label.json

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT busy before start"
        return 1
    fi

    env "${COMMON_ENV[@]}" "$BIN/llama-server" \
        -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
        -md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth" \
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
        "import json; print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
        | curl -sS -o "$response" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/completion" -d @-)
    if [ "$http_code" != 200 ]; then
        echo "[$label] HTTP $http_code:"
        python3 -m json.tool "$response" 2>/dev/null || head -c 1000 "$response"
        echo
        return 1
    fi

    python3 - "$response" "$TSV" "$label" "$rep" "$depth" <<'PY'
import hashlib
import json
import sys

response, tsv_path, label, rep, depth = sys.argv[1:]
d = json.load(open(response))
if "error" in d:
    raise SystemExit(f"[{label}] ERROR {json.dumps(d['error'])[:300]}")
t = d.get("timings", {})
content = d.get("content", "")
tps = float(t.get("predicted_per_second", 0))
predicted_ms = float(t.get("predicted_ms", 0))
predicted_n = int(t.get("predicted_n", 0))
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
    f.write(f"{label}\t{rep}\t{depth}\t{width}\t{tps:.6f}\t{predicted_ms:.3f}\t{predicted_n}\t"
            f"{draft_tokens}\t{accepted}\t{accept_pct:.4f}\t{rounds}\t{round_ms:.6f}\t"
            f"{output_per_round:.6f}\t{sha}\t{len(content.encode())}\n")
print(f"  [{label:<7}] width={width}  {tps:7.3f} t/s  round={round_ms:7.2f} ms  "
      f"out/round={output_per_round:4.2f}  acc={accept_pct:5.1f}%  sha1={sha}")
PY

    cleanup_server
    sleep "$COOL"
}

if [ "$WARMUP" = 1 ]; then
    echo "--- warmup, discarded ---"
    run_one 0 4 >/dev/null 2>&1 || true
fi

for rep in $(seq 1 "$REPS"); do
    if [ $((rep % 2)) -eq 1 ]; then
        order=$DEPTHS
    else
        order=$(printf '%s\n' $DEPTHS | sort -rn | tr '\n' ' ')
    fi
    echo "--- pass $rep: $order ---"
    for depth in $order; do
        run_one "$rep" "$depth"
    done
done

echo
echo "--- summary by draft depth ---"
python3 - "$TSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
groups = defaultdict(list)
for row in rows:
    if row["rep"] != "0":
        groups[int(row["draft_n_max"])].append(row)

print("  n  width   t/s      round ms  out/round  acceptance  runs  sha1")
for depth, rs in sorted(groups.items()):
    hashes = {r["sha1"] for r in rs}
    if len(hashes) != 1:
        raise SystemExit(f"ABORT: n={depth} is not deterministic across reps: {sorted(hashes)}")
    mean = lambda field: statistics.mean(float(r[field]) for r in rs)
    print(f"  {depth:>1}  {depth + 1:>5}  {mean('tps'):7.3f}  {mean('round_ms'):10.2f}  "
          f"{mean('output_per_round'):9.3f}  {mean('accept_pct'):9.2f}%  {len(rs):>4}  {next(iter(hashes))}")
PY

echo
echo "results: $TSV"
echo "=== done $(date) ==="
