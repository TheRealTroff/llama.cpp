#!/bin/bash
# UD line, Turbo4 KV (perf/ud-model.md step 16): the stored UD file (IQ4_XS_SOA/Q4_K_SOA/Q5_K_SOA)
# under the pick env with the Turbo4 cache line (-ctk/-ctv turbo4, draft KV f16, the Turbo4 FA
# GQA-reuse flags of run-prod-pick.sh TURBO_PICK_ENV) vs the f16 cache. Fresh process per arm,
# mirrored order. Per arm: t/s, acceptance, sha of the text, prompt ms, footprint(1) and the
# wired+anonymous delta vs idle. Turbo4 moves the text (a quantized cache), so its sha is its
# own lineage - compare t/s only within a KV type across arms, and read the KLD row (run-quant-kld.sh
# KV=turbo4 on the step-15 q8_0 logits) for quality.
#   ARMS: space-separated label:kv:depth:ctx   (depth 0 = no spec)
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf}
PORT=${PORT:-8093}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ud-turbo4-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
ARMS=${ARMS:-"f16-d3-1:f16:3:10240 t4-d3-1:turbo4:3:10240 t4-d3-2:turbo4:3:10240 f16-d3-2:f16:3:10240 t4-d2:turbo4:2:10240 t4-d4:turbo4:4:10240 t4-d3-100k:turbo4:3:102400 f16-b1:f16:0:10240 t4-b1:turbo4:0:10240"}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
TSV=$OUT/$TAG.tsv
mkdir -p "$OUT"
# keep in sync with run-prod-pick.sh PICK_ENV + the UD SoA routes (run-ud-knobs.sh)
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MM_SKINNY_SOA=1
          GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_PIN=1 GGML_MV_SOA_W3=1
          GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          GGML_MM_ACC_HALF=1 GGML_MM_N64=1 LLAMA_GDN_REPLAY=1 GGML_GDN_NR=4
          GGML_FA_QT=1 GGML_MM_F16B=1 GGML_FA_GQA_F16=1 GGML_MM_N64_KMAX=20000 GGML_FA_QR=8 GGML_FA_Q16=1
          GGML_MV_SOA_IQ4XS=5 GGML_MV_SOA_KQ=2)
# NO_UD_SOA=1 drops the two UD SoA routes (the Q4_0 line's pick env exactly)
[ "${NO_UD_SOA:-0}" = 1 ] && PICK_ENV=("${PICK_ENV[@]:0:${#PICK_ENV[@]}-2}")
# keep in sync with run-prod-pick.sh TURBO_PICK_ENV
TURBO_ENV=(TURBO_AUTO_ASYMMETRIC=0 GGML_FA_GQA_HEADS=4,6 GGML_FA_GQA4_NWG=6 GGML_FA_GQA_W3_NWG=13 GGML_FA_TR=9)
printf 'label\tkv\tdepth\tctx\ttps\taccept_pct\tpredicted_n\tprompt_ms\tsha1\tfootprint\twired_anon_gib\n' > "$TSV"
echo "=== UD x Turbo4 KV A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "model  : $M"
echo "arms   : $ARMS"

vm_wired_anon_gib() {
  vm_stat | awk '/Pages wired down/ {w=$4} /Anonymous pages/ {a=$3} END {gsub(/\./,"",w); gsub(/\./,"",a); printf "%.3f", (w+a)*16384/1073741824}'
}

run_one() {
  local label=$1 kv=$2 depth=$3 ctx=$4
  local slog="$OUT/$TAG-$label.server.log"
  local -a spec envv kvargs
  if [ "$depth" = 0 ]; then spec=(--spec-type none); else spec=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth"); fi
  envv=("${PICK_ENV[@]}")
  kvargs=(-ctk f16 -ctv f16)
  if [ "$kv" = turbo4 ]; then
    envv+=("${TURBO_ENV[@]}")
    kvargs=(-ctk turbo4 -ctv turbo4 -ctkd f16 -ctvd f16)
  fi
  for i in $(seq 1 60); do lsof -ti :$PORT >/dev/null 2>&1 || break; sleep 2; done
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port busy"; return 1; fi
  sleep 5
  local base_gib=$(vm_wired_anon_gib)
  env "${envv[@]}" $EXTRA_ENV "$BIN/llama-server" -m "$M" -c "$ctx" -fa on "${kvargs[@]}" \
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
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- > "$OUT/$TAG-$label.json"
  local fp=$(footprint -p $pid 2>/dev/null | sed -n 's/.*Footprint: \([0-9.]* [KMG]B\).*/\1/p' | head -1 | tr -d ' ')
  local mem_gib=$(vm_wired_anon_gib)
  python3 - "$label" "$kv" "$depth" "$ctx" "$fp" "$base_gib" "$mem_gib" <<PY
import json,sys,hashlib
label,kv,depth,ctx,fp,base,mem=sys.argv[1:8]
d=json.load(open('$OUT/$TAG-$label.json'))
if 'error' in d:
    print('[%s] ERROR %s' % (label, json.dumps(d['error'])[:160])); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
open('$OUT/$TAG-$label.txt','w').write(c)
delta=float(mem)-float(base)
print('[%-11s] %-6s depth=%s ctx=%-6s %6.3f t/s  acc=%5.1f%%  n=%d  prompt=%.0f ms  sha1=%s  footprint=%s  wired+anon=%+.2f GiB'
      % (label, kv, depth, ctx, t.get('predicted_per_second',0), acc, t.get('predicted_n',0), t.get('prompt_ms',0), sha, fp, delta))
open('$TSV','a').write('\t'.join(map(str,[label,kv,depth,ctx,t.get('predicted_per_second',0),round(acc,2),t.get('predicted_n',0),round(t.get('prompt_ms',0)),sha,fp,round(delta,3)]))+'\n')
PY
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
}

for a in $ARMS; do
  IFS=: read -r label kv depth ctx <<<"$a"
  run_one "$label" "$kv" "$depth" "$ctx"
done
echo "results: $TSV"
