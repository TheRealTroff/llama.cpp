#!/bin/bash
# First heated (temperature > 0) look at the DFlash pick (perf/spec-heated.md). Every prior
# harness sends temperature 0; this one runs the corpus at the pick through the server's
# sampled acceptance path (residual/rejection sampling, common/sampling.cpp) with a FIXED seed
# and reports per prompt x config: acceptance, committed/round, gen t/s, sha. One server per
# arm (timings are real: no -v); requests on one server reuse the prompt cache, so prompt_n
# after the first config of a prompt is the recurrent tail (n_rs_seq+1) and pp_s is only
# meaningful on the first. The server log must carry "spec-accept route: residual
# sampling" for every heated request - a silent fall to the greedy-match route would look like
# an acceptance collapse of the drafter (2 silent-routing phantoms on record).
#   LINE=q4|ud  KV=turbo4|f16  NPRED=300  CONFIGS="t0 t07 t10 t10pure"  SEEDS="1"  B1=0|1  SPEC_ARM=1|0
# Configs: t0 = greedy; t07/t10 = temperature 0.7/1.0 with the server's default chain
# (top_k 40, top_p 0.95, min_p 0.05 - what a client that only sets temperature gets);
# t10pure = temperature 1.0 with the chain open (top_k 0, top_p 1, min_p 0).
# B1=1 adds a no-speculation server (b1 anchor) at t0 and t07: prices the sampler's own CPU.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi

B=${B:-/Users/troff/play/llama.cpp-active}
BIN=$B/build/bin
PORT=${PORT:-8095}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-spech-$(date +%m%d-%H%M)}
NPRED=${NPRED:-300}
LINE=${LINE:-q4}; KV=${KV:-turbo4}
CONFIGS=${CONFIGS:-"t0 t07 t10 t10pure"}
SEEDS=${SEEDS:-"1"}
B1=${B1:-0}
mkdir -p "$OUT"

source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"

PROMPTS=(
  /Users/troff/play/benchprompt.txt
  "$B"/perf/prompts/01-code-explain.txt
  "$B"/perf/prompts/02-prose-creative.txt
  "$B"/perf/prompts/03-chat-support.txt
  "$B"/perf/prompts/04-math-derivation.txt
  "$B"/perf/prompts/05-json-boilerplate.txt
)

cfg_json() {  # cfg_json <config> -> extra sampling fields
  case "$1" in
    t0)      echo "'temperature': 0" ;;
    t07)     echo "'temperature': 0.7" ;;
    t10)     echo "'temperature': 1.0" ;;
    t10pure) echo "'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0" ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}

echo "=== spec heated sweep: $TAG ==="
echo "line=$LINE kv=$KV model=$PICK_MODEL depth=$PICK_DEPTH npred=$NPRED configs=[$CONFIGS] seeds=[$SEEDS] b1=$B1"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "env    : ${PICK_ENV[*]}"
echo

start_server() {  # start_server <label> <spec:0|1>
  local label=$1 spec=$2
  SLOG="$OUT/$TAG-$label.server.log"
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port $PORT busy"; return 1; fi
  local -a specargs=(); [ "$spec" = 1 ] && specargs=("${PICK_SPEC[@]}")
  # ${arr[@]+"${arr[@]}"}: macOS bash 3.2 treats an empty array as unbound under set -u
  env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" ${specargs[@]+"${specargs[@]}"} \
      --port $PORT >"$SLOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
    sleep 2
    kill -0 $SPID 2>/dev/null || { echo "[$label] server died:"; tail -4 "$SLOG"; return 1; }
  done
  echo "[$label] health timeout"; kill -9 $SPID; return 1
}
stop_server() {
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 25); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3
}

run_req() {  # run_req <label> <prompt> <config> <seed>
  local label=$1 prompt=$2 cfg=$3 seed=$4
  local extra; extra=$(cfg_json "$cfg") || return 1
  local nline_before; nline_before=$(wc -l < "$SLOG" 2>/dev/null | tr -d ' '); nline_before=${nline_before:-0}
  python3 -c "
import json
p = open('$prompt').read()
print(json.dumps({'prompt': p, 'n_predict': $NPRED, 'seed': $seed, $extra}))" \
  | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'error' in d:
    print('[$label] ERROR', json.dumps(d['error'])[:160]); sys.exit(0)
t=d.get('timings',{}); c=d.get('content','')
gen=t.get('predicted_n',0); dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
rounds = gen - da
acc = 100*da/dn if dn else 0
print('[%-26s] prompt_n=%5d gen=%3d acc=%5.1f%% committed/rd=%4.2f rounds=%3d tps=%6.2f pp_s=%6.1f sha1=%s'
      % ('$label', t.get('prompt_n',0), gen, acc, gen/rounds if rounds else 0, rounds,
         t.get('predicted_per_second',0), t.get('prompt_ms',0)/1000.0,
         hashlib.sha1(c.encode()).hexdigest()[:12]))
open('$OUT/$TAG-$label.txt','w').write(c)
"
  sleep 1
  # route proof: the newest route line since this request started
  local route; route=$(tail -n +$((nline_before+1)) "$SLOG" | grep -o 'spec-accept route: [a-z ]*(temp=[0-9.]*' | tail -1)
  echo "    route: ${route:-NONE (no verify round in this request)})"
}

# --- speculative arm (the pick); SPEC_ARM=0 skips it ---
if [ "${SPEC_ARM:-1}" = 1 ]; then
start_server spec 1 || exit 1
for p in "${PROMPTS[@]}"; do
  name=$(basename "$p" .txt)
  for cfg in $CONFIGS; do
    if [ "$cfg" = t0 ]; then run_req "$name-$cfg" "$p" "$cfg" 1; continue; fi
    for s in $SEEDS; do run_req "$name-$cfg-s$s" "$p" "$cfg" "$s"; done
  done
done
stop_server
fi

# --- b1 anchor (no speculation): sampler CPU price ---
if [ "$B1" = 1 ]; then
  start_server b1 0 || exit 1
  for p in "${PROMPTS[@]:1:2}"; do
    name=$(basename "$p" .txt)
    run_req "b1-$name-t0"  "$p" t0  1
    run_req "b1-$name-t07" "$p" t07 1
  done
  stop_server
fi
echo; echo "done: $TAG"
