#!/bin/bash
# Long-context baseline runner (perf/fa-long-context.md): the f16 pick env (keep PICK_ENV in sync with
# run-prod-pick.sh) plus the branch's byte-identical FA/mm levers, one fresh server per arm, a prompt file
# and context size of your choosing. Reports prompt eval s and t/s, decode t/s, acceptance, the spec-prof
# round time from the server log, and the output sha.
#   PROMPT=... CTX=40960 NPRED=300 DEPTH=3 TAG=... EXTRA_ENV="..." perf/run-longctx.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-gdn-scan}
BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PROMPT=${PROMPT:-/Users/troff/play/kvquant-experiments/data/longprompt-32k.txt}
CTX=${CTX:-40960}
NPRED=${NPRED:-300}
DEPTH=${DEPTH:-3}
KV=${KV:-f16}
PORT=${PORT:-8094}
LV=${LV:-}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-longctx-$(date +%m%d-%H%M)}
EXTRA_ENV=${EXTRA_ENV:-}
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MM_SKINNY_SOA=1
          GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_PIN=1 GGML_MV_SOA_W3=1
          GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1
          GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024
          GGML_MM_ACC_HALF=1 GGML_MM_N64=1 LLAMA_GDN_REPLAY=1 GGML_GDN_NR=4)
# the branch's byte-identical FA/mm levers (ud-model.md steps 9-11)
FA_ENV=(GGML_FA_QT=1 GGML_MM_F16B=1 GGML_FA_GQA_F16=1 GGML_MM_N64_KMAX=20000)
slog="$OUT/$TAG.server.log"
echo "=== longctx $TAG: ctx $CTX, prompt $(wc -c < "$PROMPT" | tr -d ' ') bytes, depth $DEPTH, kv $KV, n_predict $NPRED"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD); extra: $EXTRA_ENV"
if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
if [ "$DEPTH" = 0 ]; then spec=(--spec-type none); else spec=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$DEPTH"); fi
env "${PICK_ENV[@]}" "${FA_ENV[@]}" $EXTRA_ENV "$BIN/llama-server" -m "$M" -c "$CTX" -fa on -ctk $KV -ctv $KV \
  "${spec[@]}" ${LV:+-lv "$LV"} --port $PORT >"$slog" 2>&1 &
pid=$!; ok=0
for i in $(seq 1 300); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
  sleep 2; kill -0 $pid 2>/dev/null || { echo "server died:"; tail -4 "$slog"; exit 1; }
done
[ $ok = 1 ] || { echo "health timeout"; kill -9 $pid; exit 1; }
python3 -c "
import json
print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- --max-time 7200 | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d: print('ERROR', json.dumps(d['error'])[:200]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
sha=hashlib.sha1(c.encode()).hexdigest()[:12]
open('/tmp/longctx-$TAG.txt','w').write(c)
print('[%s] prompt %d tok  %.1f s  %.1f t/s | decode %.3f t/s  acc=%.1f%%  n=%d | sha1=%s' % ('$TAG', t.get('prompt_n',0), t.get('prompt_ms',0)/1000, t.get('prompt_per_second',0), t.get('predicted_per_second',0), acc, t.get('predicted_n',0), sha))
"
kill -TERM $pid 2>/dev/null; for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
grep -E 'spec-prof (round|loop_body)' "$slog" | tail -2 | sed -E 's/^[0-9.]+ I srv +operator\(\): //'
