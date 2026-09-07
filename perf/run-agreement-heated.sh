#!/bin/bash
# Agreement corpora at temperature 0 AND at temperature > 0 (perf/spec-heated.md): the same
# generator, the same prompts, one server, two corpora, so a KLD scored on each says whether the
# numerics price changes on the contexts a heated trajectory visits. Score each with
# run-quant-kld.sh (W=<corpus>, CHUNKS=8; the tool clamps to the chunks the corpus holds).
#   TEMP=0.7 SEED=1 NPRED=2048 LINE=q4 perf/run-agreement-heated.sh
# Generator = the pick (perf/pick.sh manifest, f16 cache), i.e. the contexts the served model
# visits. Speculation keeps the distribution at heat (rejection sampling), not the sample path,
# which is fine here: a corpus is a set of contexts, not a trajectory to reproduce.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi

B=${B:-/Users/troff/play/llama.cpp-active}
BIN=$B/build/bin
PORT=${PORT:-8096}
TEMP=${TEMP:-0.7}; SEED=${SEED:-1}; NPRED=${NPRED:-2048}; LINE=${LINE:-q4}
TAG=${TAG:-agreeh-$(date +%m%d-%H%M)}
DATA=/Users/troff/play/kvquant-experiments/data
C0=$DATA/generated-$TAG-t0.txt
CH=$DATA/generated-$TAG-t${TEMP/./}.txt
PROMPTS=("$B"/perf/prompts/*.txt)

source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" f16; pick_args "$LINE" f16

echo "=== agreement corpora, greedy + heated: $TAG ==="
echo "generator : $PICK_MODEL (line $LINE, f16 cache, depth $PICK_DEPTH)"
echo "prompts   : ${#PROMPTS[@]} x $NPRED tokens; heated arm temperature $TEMP seed $SEED, default chain"
echo "corpora   : $C0  /  $CH"
echo "commit    : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo
SLOG=/Users/troff/play/kvquant-experiments/results/$TAG.server.log
if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" "${PICK_SPEC[@]}" --port $PORT >"$SLOG" 2>&1 &
PID=$!
ok=0
for i in $(seq 1 200); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
  sleep 2
  kill -0 $PID 2>/dev/null || { echo "server died:"; tail -5 "$SLOG"; exit 1; }
done
[ $ok = 1 ] || { echo "health timeout"; kill -9 $PID; exit 1; }

gen() {  # gen <corpus> <temperature> <label>
  local corpus=$1 temp=$2 label=$3
  : >"$corpus"
  for pf in "${PROMPTS[@]}"; do
    python3 -c "
import json
print(json.dumps({'prompt': open('$pf').read(), 'n_predict': $NPRED, 'temperature': $temp, 'seed': $SEED}))" \
    | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d:
    print('  [$label $(basename "$pf")] ERROR', json.dumps(d['error'])[:120]); sys.exit(0)
c=d.get('content','')
open('$corpus','a').write(open('$pf').read() + c + '\n\n')
t=d.get('timings',{}); dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
print('  %-5s %-24s generated %5d tok at %6.2f t/s  acc %5.1f%%' % ('$label', '$(basename "$pf")', t.get('predicted_n',0),
      t.get('predicted_per_second',0), 100*da/dn if dn else 0))
"
  done
  echo "  $label corpus: $(wc -c <"$corpus" | tr -d ' ') bytes"
}
gen "$C0" 0 t0
gen "$CH" "$TEMP" "t${TEMP/./}"
kill -TERM $PID 2>/dev/null
for i in $(seq 1 25); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
grep -c 'spec-accept route: residual' "$SLOG" | sed 's/^/  residual-route lines in the server log: /'
echo "done: $TAG"
