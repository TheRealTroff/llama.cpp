#!/bin/bash
# Greedy agreement on the test model's OWN generated text - the measurement KLD-on-wikitext
# cannot make, because that one is teacher-forced on someone else's prose.
#
# At temperature 0 the token a model emits IS its argmax at that position. So if we generate
# a corpus with model Q and then ask "how often does reference P's argmax equal Q's argmax
# at each position of that corpus", the answer is exactly **the fraction of Q's own tokens
# that P would have accepted** - a greedy speculative-acceptance rate, measured on Q's real
# autoregressive trajectory rather than on wikitext.
#
# That factors into two passes with only ONE model resident at a time, which matters: both
# 27B models co-resident needs ~42.1 GiB against a 37.4 GiB Metal working set.
#
#   1. generate N tokens per prompt with Q, temperature 0            (this file)
#   2. score that corpus with run-quant-kld.sh, ref = q8_0           (existing harness)
#
# Read `Same top p` from the output. The other KLD statistics come along for free and are
# now measured on in-domain generated text instead of wikitext.
#
#   perf/run-agreement.sh                          # uniform-Q4_0, 2048 tok x 5 prompts
#   NPRED=512 perf/run-agreement.sh <model.gguf>
set -u

if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
GEN=${1:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
NPRED=${NPRED:-2048}
PORT=8094
TAG=${TAG:-agree-$(date +%m%d-%H%M)}
CORPUS=${CORPUS:-/Users/troff/play/kvquant-experiments/data/generated-$TAG.txt}
PROMPTS=("$B"/perf/prompts/*.txt

)
# Speculation is lossless (byte-identical output across configs, see README), so the prod
# pick is used purely to generate faster. It cannot change the text.
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)

echo "=== greedy agreement corpus: $TAG ==="
echo "generator : $GEN"
echo "prompts   : ${#PROMPTS[@]} x $NPRED tokens, temperature 0"
echo "corpus    : $CORPUS"
echo "commit    : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo

if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$GEN" -c 10240 -fa on -ctk f16 -ctv f16 \
  -md "$MD" --spec-type draft-dflash --spec-draft-n-max 6 --port $PORT \
  >"/tmp/agree-$TAG.server.log" 2>&1 &
PID=$!
ok=0
for i in $(seq 1 200); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
  sleep 2
  kill -0 $PID 2>/dev/null || { echo "server died:"; tail -5 "/tmp/agree-$TAG.server.log"; exit 1; }
done
[ $ok = 1 ] || { echo "health timeout"; kill -9 $PID; exit 1; }
lsof -ti :$PORT 2>/dev/null | grep -qx "$PID" || { echo "ABORT: port served by another pid"; kill -9 $PID; exit 1; }

: >"$CORPUS"
for pf in "${PROMPTS[@]}"; do
  python3 -c "
import json
print(json.dumps({'prompt': open('$pf').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d:
    print('  [$(basename "$pf")] ERROR', json.dumps(d['error'])[:120]); sys.exit(0)
c=d.get('content','')
open('$CORPUS','a').write(open('$pf').read() + c + '\n\n')
t=d.get('timings',{})
print('  %-24s generated %5d tok at %6.2f t/s' % ('$(basename "$pf")', t.get('predicted_n',0),
      t.get('predicted_per_second',0)))
"
done

kill -TERM $PID 2>/dev/null
for i in $(seq 1 25); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null

echo
echo "corpus: $(wc -c <"$CORPUS" | tr -d ' ') bytes"
echo "Now score it, e.g.:"
echo "  W=$CORPUS CHUNKS=5 TAG=$TAG perf/run-quant-kld.sh $GEN"
