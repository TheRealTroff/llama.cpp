#!/bin/bash
# End-to-end A/B for the lm_head whitelist extension (GGML_MV_SOA_WL_XL).
#
# The 2026-08-28 "whitelist-XL refuted" synthetic was a phantom: its w5r4h arm ran with
# GGML_MV_REPACK=1, which declines test-backend-ops' non-weight buffers, so the SoA gate
# silently fell through and the A/B compared ext against ext (4963 vs 4964 us). Re-run
# with REPACK=2 the head engages w5r4h and reads 3175 us against ext-di's 4967 - the w5
# win transfers to the head after all. This script prices it end-to-end: both lm_heads
# (target verify + drafter, same [5120,248320] shape at width 5) run ~1.8 ms/call apart.
#
# Arms are the full prod-pick env, +-GGML_MV_SOA_WL_XL=1. The head emits LOGITS, so a
# different accumulation order can flip a near-tie argmax and change the trajectory: if
# the sha1 column differs across arms, acceptance is confounded (weight-quant-kld.md) and
# the t/s delta is NOT clean - read acceptance and the batch-1 control before believing it.
# The batch-1 control has no width>=2 mul_mv, so XL must be inert and byte-identical there.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
PORT=${PORT:-8098}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-shortk-head-e2e-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
REPS=${REPS:-4}
CONTROL_REPS=${CONTROL_REPS:-2}
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
)

arm_env() {
    case $1 in
        base) echo "" ;;
        xl)   echo "GGML_MV_SOA_WL_XL=1" ;;
    esac
}

printf 'label\tpoint\tarm\ttps\taccept_pct\tpredicted_n\tsha1\tbytes\n' > "$TSV"

echo "=== lm_head whitelist-XL e2e A/B: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "run    : n_predict $NPRED, $REPS order-balanced reps per point+arm"
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

# label, arm, spec (dflash|none), depth
run_one() {
    local label=$1 arm=$2 spec=$3 depth=$4
    local slog=$OUT/$TAG-$label.server.log
    local args=()
    case "$spec" in
        dflash) args=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth") ;;
        none)   args=(--spec-type none) ;;
    esac

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT busy before start"
        return 1
    fi

    # shellcheck disable=SC2046
    env "${COMMON_ENV[@]}" $(arm_env "$arm") "$BIN/llama-server" -m "$M" -c 10240 -fa on \
        -ctk f16 -ctv f16 "${args[@]}" --port "$PORT" >"$slog" 2>&1 &
    server_pid=$!

    local healthy=0
    for _ in $(seq 1 300); do
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
            healthy=1; break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[$label] server died during startup:"; tail -8 "$slog"; return 1
        fi
        sleep 2
    done
    if [ "$healthy" -ne 1 ]; then echo "[$label] health timeout"; return 1; fi
    if ! lsof -ti :"$PORT" 2>/dev/null | grep -qx "$server_pid"; then
        echo "[$label] ABORT: another process owns port $PORT"; return 1
    fi

    local response=$OUT/$TAG-$label.json
    python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
        | curl -sS -o "$response" -X POST "http://127.0.0.1:$PORT/completion" -d @-
    python3 - "$response" "$TSV" "$label" "$spec-$depth" "$arm" <<'PY'
import hashlib, json, sys
response, tsv, label, point, arm = sys.argv[1:]
d = json.load(open(response))
if "error" in d:
    raise SystemExit(f"[{label}] ERROR {json.dumps(d['error'])[:200]}")
t = d.get("timings", {})
tps = float(t.get("predicted_per_second", 0))
n = int(t.get("predicted_n", 0))
dn = int(t.get("draft_n", 0))
acc = 100.0*int(t.get("draft_n_accepted", 0))/dn if dn else 0.0
content = d.get("content", "")
sha = hashlib.sha1(content.encode()).hexdigest()[:12]
with open(tsv, "a") as f:
    f.write(f"{label}\t{point}\t{arm}\t{tps:.6f}\t{acc:.4f}\t{n}\t{sha}\t{len(content.encode())}\n")
print(f"  [{label:<22}] {point:<9} {arm:<5} {tps:7.3f} t/s  acc={acc:5.1f}%  sha1={sha}")
PY

    cleanup_server
    sleep "$COOL"
}

echo "--- WARMUP, discarded ---"
run_one warmup base dflash 4 >/dev/null 2>&1 || true

echo "--- dflash n4 (the pick), order-balanced ---"
for rep in $(seq 1 "$REPS"); do
    case $((rep % 2)) in
        1) order=(base xl) ;;
        0) order=(xl base) ;;
    esac
    for arm in "${order[@]}"; do
        run_one "dflash-n4-$arm-r$rep" "$arm" dflash 4
    done
done

echo "--- batch-1 control (no width>=2 mul_mv; XL must be inert, byte-identical) ---"
for rep in $(seq 1 "$CONTROL_REPS"); do
    if [ $((rep % 2)) -eq 1 ]; then order=(base xl); else order=(xl base); fi
    for arm in "${order[@]}"; do
        run_one "ctrl-b1-$arm-r$rep" "$arm" none 0
    done
done

echo
echo "--- summary (per point+arm means) ---"
python3 - "$TSV" <<'PY'
import csv, sys
from collections import defaultdict
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["label"] != "warmup"]
groups = defaultdict(list)
for r in rows:
    groups[(r["point"], r["arm"])].append(r)
for (point, arm), rs in sorted(groups.items()):
    tps = [float(r["tps"]) for r in rs]
    shas = {r["sha1"] for r in rs}
    print(f"  {point:<9} {arm:<5} mean {sum(tps)/len(tps):7.3f} t/s  ({', '.join(f'{t:.3f}' for t in tps)})  sha1={'/'.join(sorted(shas))}")
PY

echo
echo "results: $TSV"
echo "=== done $(date) ==="
