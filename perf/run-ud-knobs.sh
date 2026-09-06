#!/bin/bash
# UD-Q4_K_M free knobs (perf/ud-model.md step 1). The model has ZERO Q4_0 tensors, so every
# pick route that gates on Q4_0/Q4_0_SOA is inert and its verify/draft matmuls run the
# upstream per-column mul_mv, whose per-width cost is steeper than the SoA kernels'. Two
# knobs that need no code: DFlash depth (2/3/4; the depth-4 pick was chosen on Q4_0's cost
# curve) and GGML_MM_MIN (default 8: ne11 > 8 takes the generic simdgroup mul_mm; lowering
# it routes the verify width there instead of mul_mv). Full pick env otherwise, so the
# format-agnostic levers (FA, GDN, drafter stack, get_memcpy) are all on.
# n_predict 600 throughout (README trap 1: do not compare with 300-token numbers).
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=${PORT:-8093}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ud-knobs-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
DEPTHS=${DEPTHS:-"4 3 2"}
EXTRA_ENV=${EXTRA_ENV:-}  # plain string, word-split at use (bash 3.2 + set -u rejects an empty array)
EXTRA_ARGS=${EXTRA_ARGS:-} # extra llama-server args, word-split at use (e.g. "-lv 5" for pipeline-load lines)
RUN_B1=${RUN_B1:-1}
MMMINS=${MMMINS:-"8 4 2"}
TSV=$OUT/$TAG.tsv
mkdir -p "$OUT"
# keep in sync with run-prod-pick.sh PICK_ENV
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MM_SKINNY_SOA=1
          GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_PIN=1 GGML_MV_SOA_W3=1
          GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          GGML_MM_ACC_HALF=1 GGML_MM_N64=1 LLAMA_GDN_REPLAY=1 GGML_GDN_NR=4
          GGML_FA_QT=1 GGML_MM_F16B=1 GGML_FA_GQA_F16=1 GGML_MM_N64_KMAX=20000 GGML_FA_QR=8 GGML_FA_Q16=1)
printf 'label\tdepth\tmm_min\ttps\taccept_pct\tpredicted_n\tprompt_ms\tsha1\n' > "$TSV"
echo "=== UD free knobs: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "model  : $M"
echo "extra  : $EXTRA_ENV"

run_one() {
  local label=$1 depth=$2 mmmin=$3
  local slog="$OUT/$TAG-$label.server.log"
  local -a spec
  if [ "$depth" = 0 ]; then spec=(--spec-type none); else spec=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth"); fi
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port busy"; return 1; fi
  env "${PICK_ENV[@]}" GGML_MM_MIN=$mmmin $EXTRA_ENV "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    "${spec[@]}" $EXTRA_ARGS --port $PORT >"$slog" 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2
    kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
  done
  [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }
  python3 -c "
import json
p = open('/Users/troff/play/benchprompt.txt').read()
print(json.dumps({'prompt': p, 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
open('/tmp/udknobs-$label.txt','w').write(c)
print('[%-14s] depth=%s mm_min=%s  %6.3f t/s  acc=%5.1f%%  n=%d  prompt=%.0f ms  sha1=%s'
      % ('$label', '$depth', '$mmmin', t.get('predicted_per_second',0), acc, t.get('predicted_n',0), t.get('prompt_ms',0), sha))
open('$TSV','a').write('\t'.join(map(str,['$label','$depth','$mmmin',t.get('predicted_per_second',0),round(acc,2),t.get('predicted_n',0),round(t.get('prompt_ms',0)),sha]))+'\n')
"
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5
}

for mm in $MMMINS; do
  for d in $DEPTHS; do
    run_one "d${d}-mm${mm}" "$d" "$mm"
  done
done
[ "$RUN_B1" = 1 ] && run_one "b1-mm8" 0 8
echo; echo "tsv: $TSV"
