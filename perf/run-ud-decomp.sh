#!/bin/bash
# UD-Q4_K_M round decomposition (perf/ud-model.md step 2): where does a UD round go, by
# kernel and format, at its best free-knob depth? Same three instruments as
# run-width4-gap-decomp.sh, but under the CURRENT pick env (that file's COMMON_ENV predates
# the SoA/acch/replay flags) and with DEPTH/MMMIN taken from run-ud-knobs.sh's winner.
#   1. llama-bench pp<width>   - bare verify-width pass
#   2. LLAMA_DECODE_PROF=1     - host-side split
#   3. GGML_METAL_PROFILE=1    - per-op GPU time (serialized encoders: shares, not wall)
# Then perf/metalprof-buckets.py on the profiled log.
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
M=${M:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf}
PROMPT=/Users/troff/play/benchprompt.txt
PORT=${PORT:-8093}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ud-decomp-$(date +%m%d-%H%M)}
NPRED=${NPRED:-300}
DEPTH=${DEPTH:-3}
MMMIN=${MMMIN:-8}
EXTRA_ENV=${EXTRA_ENV:-}  # plain string, word-split at use (bash 3.2 + set -u rejects an empty array)
# keep in sync with run-prod-pick.sh PICK_ENV
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MM_SKINNY_SOA=1
          GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_PIN=1 GGML_MV_SOA_W3=1
          GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          GGML_MM_ACC_HALF=1 GGML_MM_N64=1 LLAMA_GDN_REPLAY=1
          GGML_MM_MIN=$MMMIN)
SPEC_ARGS="-md $MD --spec-type draft-dflash --spec-draft-n-max $DEPTH"
mkdir -p "$OUT"
echo "=== UD round decomposition: $TAG ==="
echo "commit : $(git -C "$B" rev-parse --short HEAD) on $(git -C "$B" rev-parse --abbrev-ref HEAD)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "model  : $M   depth $DEPTH (verify width $((DEPTH+1)))   mm_min $MMMIN   extra: $EXTRA_ENV"
if lsof -ti :"$PORT" >/dev/null 2>&1; then echo "ABORT: port $PORT busy" >&2; exit 1; fi

if [ "${SKIP_BENCH:-0}" != 1 ]; then
  echo; echo "--- 1. llama-bench pp$((DEPTH+1)) and pp1: bare verify-width and b1 passes ---"
  env "${PICK_ENV[@]}" $EXTRA_ENV "$BIN/llama-bench" -m "$M" -fa 1 -p 1,$((DEPTH+1)) -n 0 -r 4 2>"$OUT/$TAG-bench.err" | tee "$OUT/$TAG-bench.txt"
fi

server_pid=
cleanup_server() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap cleanup_server EXIT INT TERM

run_server() {
  local label=$1; shift
  local slog=$OUT/$TAG-$label.server.log
  env "${PICK_ENV[@]}" $EXTRA_ENV "$@" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
      $SPEC_ARGS --port "$PORT" >"$slog" 2>&1 &
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
d = json.load(open(sys.argv[1])); t = d.get("timings", {})
acc = 100.0*t.get("draft_n_accepted",0)/t["draft_n"] if t.get("draft_n") else 0.0
sha = hashlib.sha1(d.get("content","").encode()).hexdigest()[:12]
print(f"  [{sys.argv[2]}] {t.get('predicted_per_second',0):.3f} t/s  acc={acc:.1f}%  n={t.get('predicted_n',0)}  sha1={sha}")
PY
  cleanup_server; sleep 5
}

echo; echo "--- 2. unprofiled anchor ---"
run_server anchor
echo; echo "--- 3. decode-prof: host-side split ---"
run_server decodeprof LLAMA_DECODE_PROF=1
echo; echo "--- 4. metal-profile: per-op GPU time, serialized encoders ---"
run_server metalprof GGML_METAL_PROFILE=1
echo; echo "--- 5. buckets ---"
python3 "$B/perf/metalprof-buckets.py" "$OUT/$TAG-metalprof.server.log" --top 40 | tee "$OUT/$TAG-buckets.txt"
echo; echo "logs: $OUT/$TAG-*"
