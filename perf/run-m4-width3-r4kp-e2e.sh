#!/bin/bash
# Fixed-depth DFlash A/B for the dedicated width-3 q4_0 SoA kernel.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
PORT=${PORT:-8097}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-m4-w3-r4kp-e2e-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
REPS=${REPS:-4}
COOL=${COOL:-5}
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

# Match the canonical production pick. The arm-level variable is only the width-3 route.
COMMON_ENV=(
    GGML_MV_NC=2
    GGML_MM_SKINNY=6
    GGML_FA_VEC_MAX=5
    GGML_FA_MM_NWG=8
    GGML_GDN_FUSE_WB=1
    GGML_MV_REPACK=1
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

printf 'label\tarm\ttps\taccept_pct\tpredicted_n\tsha1\tbytes\n' > "$TSV"

echo "=== M4 width-3 r4kp end-to-end A/B: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "prompt : $(wc -c < "$PROMPT" | tr -d ' ') bytes, sha1 $(shasum "$PROMPT" | cut -c1-12)"
echo "env    : ${COMMON_ENV[*]}"
echo "shape  : DFlash depth 2 -> target width 3"
echo "run    : $REPS A/B pairs, n_predict $NPRED, cooldown ${COOL}s"
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
    local label=$1 arm=$2
    local w3=0
    local slog=$OUT/$TAG-$label.server.log
    local response=$OUT/$TAG-$label.json
    local content=$OUT/$TAG-$label.txt

    [ "$arm" = candidate ] && w3=1

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT busy before start"
        return 1
    fi

    env "${COMMON_ENV[@]}" GGML_MV_SOA_W3="$w3" \
        "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
        -md "$MD" --spec-type draft-dflash --spec-draft-n-max 2 \
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

    python3 - "$response" "$content" "$TSV" "$label" "$arm" <<'PY'
import hashlib
import json
import pathlib
import sys

response, content_path, tsv_path, label, arm = sys.argv[1:]
d = json.load(open(response))
if "error" in d:
    raise SystemExit(f"[{label}] ERROR {json.dumps(d['error'])[:300]}")
t = d.get("timings", {})
content = d.get("content", "")
tps = float(t.get("predicted_per_second", 0))
predicted_n = int(t.get("predicted_n", 0))
draft_n = int(t.get("draft_n", 0))
accepted = int(t.get("draft_n_accepted", 0))
accept_pct = 100.0 * accepted / draft_n if draft_n else 0.0
sha = hashlib.sha1(content.encode()).hexdigest()[:12]
pathlib.Path(content_path).write_text(content)
with open(tsv_path, "a") as f:
    f.write(f"{label}\t{arm}\t{tps:.6f}\t{accept_pct:.4f}\t{predicted_n}\t{sha}\t{len(content.encode())}\n")
print(f"  [{label:<20}] {arm:<9} {tps:7.3f} t/s  "
      f"acc={accept_pct:5.1f}%  n={predicted_n}  sha1={sha}")
PY

    cleanup_server
    sleep "$COOL"
}

echo "--- WARMUP, discarded ---"
run_one warmup-discard baseline

echo
echo "--- DFlash depth 2: target width 3 ---"
for rep in $(seq 1 "$REPS"); do
    if [ $((rep % 2)) -eq 1 ]; then
        order=(baseline candidate)
    else
        order=(candidate baseline)
    fi
    for arm in "${order[@]}"; do
        run_one "d2-$arm-r$rep" "$arm"
    done
done

echo
echo "--- summary ---"
python3 - "$TSV" <<'PY'
import csv
import statistics
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["label"] != "warmup-discard"]
means = {}
for arm in ("baseline", "candidate"):
    arm_rows = [r for r in rows if r["arm"] == arm]
    values = [float(r["tps"]) for r in arm_rows]
    accepts = [float(r["accept_pct"]) for r in arm_rows]
    hashes = {r["sha1"] for r in arm_rows}
    if len(hashes) != 1:
        raise SystemExit(f"ABORT: {arm} is not internally deterministic: {sorted(hashes)}")
    means[arm] = statistics.mean(values)
    print(f"  {arm:<9}: " + ", ".join(f"{v:.3f}" for v in values)
          + f"  mean={means[arm]:.3f} t/s  acceptance={statistics.mean(accepts):.2f}%  sha1={next(iter(hashes))}")

delta = 100.0 * (means["candidate"] / means["baseline"] - 1.0)
print(f"  candidate delta: {delta:+.2f}%")
all_hashes = {r["sha1"] for r in rows}
print("  cross-arm output: " + ("identical" if len(all_hashes) == 1 else "DIFFERS"))
PY

echo
echo "results: $TSV"
echo "=== done $(date) ==="
