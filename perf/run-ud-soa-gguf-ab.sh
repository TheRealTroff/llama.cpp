#!/bin/bash
# UD line step 12 (perf/ud-model.md): the offline SoA GGUF (IQ4_XS_SOA / Q4_K_SOA / Q5_K_SOA, written
# by llama-gguf-repack) vs the original file with the runtime side buffers, fresh-process,
# mirrored order (orig stored stored orig). Same pick env in both arms (the SoA env vars are inert
# on stored types). Per arm: depth-3 DFlash at n_predict 600 (t/s, acceptance, sha of the text,
# prompt ms) and a no-spec b1 anchor, plus memory: the server's physical footprint after the
# request (footprint(1)) and vm_stat wired+anonymous deltas vs the idle baseline.
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-ud-soa-gguf}
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M.gguf}
MS=${MS:-/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=${PORT:-8093}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-ud-soa-gguf-$(date +%m%d-%H%M)}
NPRED=${NPRED:-600}
DEPTH=${DEPTH:-3}
ARMS=${ARMS:-"orig stored stored orig"}
RUN_B1=${RUN_B1:-1}
RUN_SPEC=${RUN_SPEC:-1}
EXTRA_ENV=${EXTRA_ENV:-}
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
printf 'label\tarm\tdepth\ttps\taccept_pct\tpredicted_n\tprompt_ms\tsha1\tfootprint_mib\twired_anon_gib\n' > "$TSV"
echo "=== UD offline SoA GGUF A/B: $TAG ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "binary : $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "orig   : $M"
echo "stored : $MS"

vm_wired_anon_gib() {  # wired + anonymous pages, GiB (16 KiB pages)
  vm_stat | awk '/Pages wired down/ {w=$4} /Anonymous pages/ {a=$3} END {gsub(/\./,"",w); gsub(/\./,"",a); printf "%.3f", (w+a)*16384/1073741824}'
}

run_one() {
  local label=$1 arm=$2 depth=$3
  local model=$M; [ "$arm" = stored ] && model=$MS
  local slog="$OUT/$TAG-$label.server.log"
  local -a spec
  if [ "$depth" = 0 ]; then spec=(--spec-type none); else spec=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth"); fi
  # the previous server's listening socket outlives its process teardown (Metal residency
  # sets, an in-flight prefill that kill -9 cannot interrupt): a second listener on the port
  # gets the next arm's POST routed to the dying process and an empty reply (2026-09-06)
  for i in $(seq 1 60); do lsof -ti :$PORT >/dev/null 2>&1 || break; sleep 2; done
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port busy"; return 1; fi
  sleep 5
  local base_gib=$(vm_wired_anon_gib)
  env "${PICK_ENV[@]}" $EXTRA_ENV "$BIN/llama-server" -m "$model" -c 10240 -fa on -ctk f16 -ctv f16 \
    "${spec[@]}" --port $PORT >"$slog" 2>&1 &
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
  python3 - "$label" "$arm" "$depth" "$fp" "$base_gib" "$mem_gib" <<PY
import json,sys,hashlib
label,arm,depth,fp,base,mem=sys.argv[1:7]
d=json.load(open('$OUT/$TAG-$label.json'))
if 'error' in d:
    print('[%s] ERROR %s' % (label, json.dumps(d['error'])[:160])); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
open('$OUT/$TAG-$label.txt','w').write(c)
delta=float(mem)-float(base)
print('[%-12s] %-6s depth=%s  %6.3f t/s  acc=%5.1f%%  n=%d  prompt=%.0f ms  sha1=%s  footprint=%s  wired+anon=%+.2f GiB'
      % (label, arm, depth, t.get('predicted_per_second',0), acc, t.get('predicted_n',0), t.get('prompt_ms',0), sha, fp, delta))
open('$TSV','a').write('\t'.join(map(str,[label,arm,depth,t.get('predicted_per_second',0),round(acc,2),t.get('predicted_n',0),round(t.get('prompt_ms',0)),sha,fp,round(delta,3)]))+'\n')
PY
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
}

# (n, not i: run_one's wait loops use i)
n=0
if [ "$RUN_SPEC" = 1 ]; then
  for arm in $ARMS; do
    n=$((n+1))
    run_one "d${DEPTH}-$arm-$n" $arm $DEPTH
  done
fi
if [ "$RUN_B1" = 1 ]; then
  n=0
  for arm in $ARMS; do
    n=$((n+1))
    run_one "b1-$arm-$n" $arm 0
  done
fi
echo "results: $TSV"
