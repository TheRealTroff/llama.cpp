#!/bin/bash
# Depth re-sweep with the r4kp width-4 kernel on the board (m4-width4-r4kp.md open
# item 2). slope-sweep.md's optima were priced against kernels that no longer define
# the curve: width 4 is ~26-28% cheaper, so the dflash and MTP depth optima may move.
#
# One env for every arm: prod-pick flags + REPACK=1 + SOA_W4=1 + R4KP=3 (v3). Width 4
# (n3/d3) runs the new kernel; widths 5-8 stay on skinny, which the width-7 refutation
# says is correct; widths 2-3 stay on nc/ext. n_predict 600 throughout - do NOT compare
# these absolute numbers with the n_predict-300 tables (README trap 1), and note every
# number includes repack, which the prod pick does not.
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PROMPT=/Users/troff/play/benchprompt.txt
PORT=${PORT:-8097}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-depth-resweep-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
COOL=${COOL:-5}
TSV=$OUT/$TAG.tsv
mkdir -p "$OUT"

COMMON=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
        GGML_MV_REPACK=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3)

if lsof -ti :"$PORT" >/dev/null 2>&1; then
    echo "ABORT: port $PORT busy" >&2
    exit 1
fi

printf 'label\tspec\tdepth\ttps\taccept_pct\tpredicted_n\tsha1\n' > "$TSV"

echo "=== depth re-sweep (r4kp v3 on the board): $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "env    : ${COMMON[*]}"
echo "run    : n_predict $NPRED, one run per point, top contenders repeated at the end"
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

# label, spec (dflash|mtp|none), depth
run_one() {
    local label=$1 spec=$2 depth=$3
    local slog=$OUT/$TAG-$label.server.log
    local args=()
    case "$spec" in
        dflash) args=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth") ;;
        mtp)    args=(--spec-type draft-mtp --spec-draft-n-max "$depth") ;;
        none)   args=(--spec-type none) ;;
        *) echo "ABORT: unknown spec '$spec'" >&2; return 1 ;;
    esac

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT busy before start"
        return 1
    fi

    env "${COMMON[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
        "${args[@]}" --port "$PORT" >"$slog" 2>&1 &
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

    python3 -c "import json; print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
        | curl -sS -X POST "http://127.0.0.1:$PORT/completion" -d @- \
        | python3 - "$TSV" "$label" "$spec" "$depth" <<'PY'
import hashlib, json, sys
tsv, label, spec, depth = sys.argv[1:]
d = json.load(sys.stdin)
if "error" in d:
    raise SystemExit(f"[{label}] ERROR {json.dumps(d['error'])[:200]}")
t = d.get("timings", {})
tps = float(t.get("predicted_per_second", 0))
n = int(t.get("predicted_n", 0))
dn = int(t.get("draft_n", 0))
acc = 100.0*int(t.get("draft_n_accepted", 0))/dn if dn else 0.0
sha = hashlib.sha1(d.get("content", "").encode()).hexdigest()[:12]
with open(tsv, "a") as f:
    f.write(f"{label}\t{spec}\t{depth}\t{tps:.6f}\t{acc:.4f}\t{n}\t{sha}\n")
print(f"  [{label:<16}] {spec:<7} d={depth}  {tps:7.3f} t/s  acc={acc:5.1f}%  n={n}  sha1={sha}")
PY

    cleanup_server
    sleep "$COOL"
}

echo "--- WARMUP, discarded ---"
run_one warmup dflash 3 >/dev/null 2>&1 || true

echo "--- sweep: depth 1..7, both spec types interleaved ---"
for d in 1 2 3 4 5 6 7; do
    run_one "dflash-n$d" dflash "$d"
    run_one "mtp-d$d"    mtp    "$d"
done
run_one batch1 none 0

echo
echo "--- confirmation: top two points, one repeat each ---"
python3 - "$TSV" <<'PY' > /tmp/depth-resweep-top2
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["spec"] != "none" and r["label"] != "warmup"]
rows.sort(key=lambda r: -float(r["tps"]))
for r in rows[:2]:
    print(r["spec"], r["depth"])
PY
while read -r spec depth; do
    run_one "confirm-$spec-$depth" "$spec" "$depth"
done < /tmp/depth-resweep-top2

echo
echo "--- summary (sorted) ---"
python3 - "$TSV" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["label"] != "warmup"]
rows.sort(key=lambda r: -float(r["tps"]))
for r in rows:
    print(f"  {r['label']:<18} {float(r['tps']):7.3f} t/s  acc={float(r['accept_pct']):5.1f}%  sha1={r['sha1']}")
PY

echo
echo "results: $TSV"
echo "=== done $(date) ==="
