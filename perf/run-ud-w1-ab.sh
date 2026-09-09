#!/bin/bash
# Width-1 A/B for the remaining stored UD formats (perf/ud-remaining-quants.md): the shared scalar body
# (gen, GGML_MV_UD_W1=0) vs the kq-SoA body at NC = 1 (kq, GGML_MV_UD_W1=1), with the native kernels
# (plain) as the reference. Fresh processes, mirrored order, 12 shapes x width 1 per arm.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROD=${PROD:-/Users/troff/play/llama.cpp-prod}
MODEL=${MODEL:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf}
OUT=${OUT:-$ROOT/perf/results/ud-remaining-w1-$(date +%Y%m%d)}
ARMS=${ARMS:-"plain-1 gen-1 kq-1 kq-2 gen-2 plain-2"}
mkdir -p "$OUT"
source "$PROD/perf/pick.sh"
pick_check ud
pick_env ud
{
    date -u
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" status --short
    printf 'MODEL=%s\nENV=%s\nORDER=%s\n' "$MODEL" "${PICK_ENV[*]}" "$ARMS"
} > "$OUT/manifest.txt"
for arm in $ARMS; do
    case "$arm" in
        plain-*) types='(iq4_nl|q3_K|q6_K|iq3_s)'; w1=0 ;;
        gen-*)   types='(iq4_nl_soa|q3_K_soa|q6_K_soa|iq3_s_soa)'; w1=${GEN_W1:-0} ;;   # GEN_W1=2: the wide-load w1 form as the 'gen' arm
        kq-*)    types='(iq4_nl_soa|q3_K_soa|q6_K_soa|iq3_s_soa)'; w1=1 ;;
    esac
    [ -e "$OUT/$arm.log" ] && { echo "refusing to overwrite $OUT/$arm.log" >&2; exit 1; }
    printf '%s START %s\n' "$(date -u +%FT%TZ)" "$arm"
    env "${PICK_ENV[@]}" GGML_MV_UD_W1=$w1 GGML_METAL_LOG_LEVEL=2 GGML_TEST_UD_GGUF="$MODEL" \
        "$ROOT/build/bin/test-backend-ops" perf -o MUL_MAT -b MTL0 \
        -p "type_a=$types,.*n=1,k=.*ud_remaining=1" > "$OUT/$arm.log" 2>&1
    count=$(grep -c 'runs -' "$OUT/$arm.log")
    [ "$count" = 12 ] || { printf 'ERROR: %s produced %s timings, expected 12\n' "$arm" "$count" >&2; exit 1; }
    printf '%s DONE %s\n' "$(date -u +%FT%TZ)" "$arm"
done
python3 - "$OUT" <<'PY'
import re, sys, statistics, math
from pathlib import Path
out = Path(sys.argv[1])
pat = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=1,k=(\d+),[^\n]*?ud_remaining=1\):.*?(\d+) runs -\s+([0-9.]+) us/run", re.S)
def arm(name):
    t = re.sub(r"\x1b\[[0-9;]*m", "", (out/f"{name}.log").read_text())
    return {(q.removesuffix("_soa"), int(m), int(k)): float(us) for q, m, k, _, us in pat.findall(t)}
P = [arm("plain-1"), arm("plain-2")]; G = [arm("gen-1"), arm("gen-2")]; K = [arm("kq-1"), arm("kq-2")]
rows = []
print("| Type | M | K | plain us | gen us | kq us | gen vs plain | kq vs plain | kq vs gen | max spread |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for key in sorted(P[0]):
    p = statistics.mean(a[key] for a in P); g = statistics.mean(a[key] for a in G); k = statistics.mean(a[key] for a in K)
    sp = max(100*(max(a[key] for a in X)-min(a[key] for a in X))/statistics.mean(a[key] for a in X) for X in (P, G, K))
    rows.append((key, p, g, k))
    print(f"| {key[0]} | {key[1]} | {key[2]} | {p:.1f} | {g:.1f} | {k:.1f} | {100*(1-g/p):+.1f}% | {100*(1-k/p):+.1f}% | {100*(1-k/g):+.1f}% | {sp:.1f}% |")
print("\nGeometric mean per format (positive = faster):\n\n| Type | gen vs plain | kq vs plain | kq vs gen |\n|---|---:|---:|---:|")
for q in sorted({r[0][0] for r in rows}):
    gm = lambda f: 100*(1-math.exp(statistics.mean(math.log(f(r)) for r in rows if r[0][0]==q)))
    print(f"| {q} | {gm(lambda r: r[2]/r[1]):+.1f}% | {gm(lambda r: r[3]/r[1]):+.1f}% | {gm(lambda r: r[3]/r[2]):+.1f}% |")
PY
