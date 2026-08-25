#!/bin/bash
# End-to-end width-4 A/B for the M4-specific barrier-free R2 Q4_0 kernel.
#
# DFlash depth 3 verifies four columns, which is the exact width selected by the R2 route.
# Both arms keep GGML_MV_REPACK=1 and GGML_MV_SOA_W4=1 so the persistent allocation, layout,
# and model-residency cost are controlled. The only arm-level change is the R2 kernel switch:
#
#   k2: two-simdgroup 4-row x 4-column SoA kernel with a terminal barrier and K-part add
#   r2: one-simdgroup 2-row x 4-column SoA kernel with direct output
#
# The DFlash depth-6 control verifies seven columns. The R2 selector is an exact-width-4 gate
# and must therefore be inert there; a moving control invalidates attribution of the n3 result.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-m4-width4}
BIN=${BIN:-$B/build-ilp/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
PORT=${PORT:-8093}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
TAG=${TAG:-m4-w4-r2-e2e-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
REPS=${REPS:-4}
CONTROL_REPS=${CONTROL_REPS:-2}
COOL=${COOL:-5}
PHASES=${PHASES:-all}
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

# REPACK and SoA are common deliberately. Comparing the generic layout against R2 would confound
# the kernel with layout policy and make the width-7 control move even when R2 itself is inactive.
COMMON_ENV=(
    GGML_MV_NC=2
    GGML_MM_SKINNY=5
    GGML_FA_VEC_MAX=5
    GGML_FA_MM_NWG=8
    GGML_GDN_FUSE_WB=1
    GGML_MV_REPACK=1
    GGML_MV_SOA_W4=1
)

printf 'label\tpoint\tarm\tnmax\ttps\taccept_pct\tpredicted_n\tsha1\tbytes\n' > "$TSV"

echo "=== M4 width-4 R2 end-to-end A/B: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary : $(stat -f '%Sm' "$BIN/llama-server")"
echo "prompt : $(wc -c < "$PROMPT" | tr -d ' ') bytes, sha1 $(shasum "$PROMPT" | cut -c1-12)"
echo "env    : ${COMMON_ENV[*]}"
echo "shape  : DFlash n3 -> width 4; DFlash n6 -> width 7 control"
echo "run    : phases $PHASES; $REPS A/B pairs, $CONTROL_REPS control pairs, n_predict $NPRED, cooldown ${COOL}s"
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

# label, point, arm, draft depth
run_one() {
    local label=$1 point=$2 arm=$3 nmax=$4
    local r2=0
    local slog=$OUT/$TAG-$label.server.log
    local response=$OUT/$TAG-$label.json
    local content=$OUT/$TAG-$label.txt

    case "$arm" in
        k2) ;;
        r2) r2=1 ;;
        *) echo "ABORT: unknown arm '$arm'" >&2; return 1 ;;
    esac

    if lsof -ti :"$PORT" >/dev/null 2>&1; then
        echo "[$label] ABORT: port $PORT busy before start"
        return 1
    fi

    env "${COMMON_ENV[@]}" GGML_MV_SOA_W4_R2="$r2" \
        "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
        -md "$MD" --spec-type draft-dflash --spec-draft-n-max "$nmax" \
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

    python3 - "$response" "$content" "$TSV" "$label" "$point" "$arm" "$nmax" <<'PY'
import hashlib
import json
import pathlib
import sys

response, content_path, tsv_path, label, point, arm, nmax = sys.argv[1:]
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
    f.write(f"{label}\t{point}\t{arm}\t{nmax}\t{tps:.6f}\t{accept_pct:.4f}\t{predicted_n}\t{sha}\t{len(content.encode())}\n")
print(f"  [{label:<20}] {point:<7} {arm:<8} {tps:7.3f} t/s  "
      f"acc={accept_pct:5.1f}%  n={predicted_n}  sha1={sha}")
PY

    cleanup_server
    sleep "$COOL"
}

if [ "$PHASES" = all ] || [ "$PHASES" = n3 ]; then
    echo "--- WARMUP, discarded ---"
    run_one warmup-discard warmup k2 3

    echo
    echo "--- DFlash n3: depth 3 = width 4, affected point ---"
    for rep in $(seq 1 "$REPS"); do
        if [ $((rep % 2)) -eq 1 ]; then
            first=k2; second=r2
        else
            first=r2; second=k2
        fi
        run_one "n3-$first-r$rep" n3 "$first" 3
        run_one "n3-$second-r$rep" n3 "$second" 3
    done
fi

if [ "$PHASES" = all ] || [ "$PHASES" = n6 ]; then
    echo
    echo "--- CONTROL: DFlash n6 = width 7, incompatible repack layouts must fall back safely ---"
    for rep in $(seq 1 "$CONTROL_REPS"); do
        if [ $((rep % 2)) -eq 1 ]; then
            first=k2; second=r2
        else
            first=r2; second=k2
        fi
        run_one "n6-$first-r$rep" n6 "$first" 6
        run_one "n6-$second-r$rep" n6 "$second" 6
    done
fi

echo
echo "--- summary ---"
python3 - "$TSV" <<'PY'
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
for point in ("n3", "n6"):
    means = {}
    for arm in ("k2", "r2"):
        values = [float(r["tps"]) for r in rows if r["point"] == point and r["arm"] == arm]
        if not values:
            continue
        means[arm] = statistics.mean(values)
        print(f"  {point} {arm:<8}: " + ", ".join(f"{v:.3f}" for v in values)
              + f"  mean={means[arm]:.3f} t/s")
    if len(means) == 2:
        delta = 100.0 * (means["r2"] / means["k2"] - 1.0)
        print(f"  {point} R2 delta: {delta:+.2f}%")

measured = [r for r in rows if r["point"] in ("n3", "n6")]
by_point_arm = {}
for row in measured:
    by_point_arm.setdefault((row["point"], row["arm"]), set()).add(row["sha1"])
for (point, arm), hashes in sorted(by_point_arm.items()):
    print(f"  {point} {arm} output hashes: {', '.join(sorted(hashes))}")
    if len(hashes) != 1:
        raise SystemExit(f"ABORT: {point} {arm} is not internally deterministic")
if "n6" in {point for point, _ in by_point_arm}:
    n6_hashes = set().union(*(hashes for (point, _), hashes in by_point_arm.items() if point == "n6"))
    if len(n6_hashes) != 1:
        raise SystemExit("ABORT: width-7 control output differs between nominally inert arms")
PY

echo
echo "results: $TSV"
echo "=== done $(date) ==="
