#!/bin/bash
# YaRN beyond native context (2026-09-11, owner: "see if anything explodes"): Qwen3.8-27B's
# native context_length is 262144 (qwen35.context_length in the gguf, confirmed via gguf-py -
# NOT 32K). This pushes -c to 524288 (512K, 2x native) via --rope-scaling yarn --rope-scale 2
# --yarn-orig-ctx 262144. Without an explicit --rope-scale, "--rope-scaling yarn" alone is a
# NO-OP for context extension: cparams.rope_freq_scale defaults to hparams.rope_freq_scale_train
# (1.0) regardless of scaling type (src/llama-context.cpp:447), and with freq_scale=1 the yarn
# interp/extrap blend in rope_yarn() collapses to theta_extrap either way - so a "yarn" run without
# --rope-scale would report success while testing nothing. QWEN35 uses IMROPE (interleaved mrope,
# 4 position sections t/h/w/e - src/llama-model.cpp:2743) but both the CPU (ops.cpp:5864) and
# Metal (ggml-metal.metal:11780) rope kernels compute the yarn ramp/corr_dims once per i0
# independent of section routing, so yarn composes with imrope with no special-casing needed.
# Turbo4 KV (owner: "You'll want turbo4") - f16 KV at 512K would be ~35 GiB for 17 attn layers
# alone (65 blocks, 17 full-attn + 1 idle nextn head + 48 GDN recurrent - GDN state is O(1) in
# ctx, only the attn layers' cache scales) on a 48 GiB machine; Turbo4's 4-bit KV cuts that to
# single digits. No drafter (depth 0) for the first arm - fewer moving parts to attribute a
# crash to; add DEPTH=3 once the plain load+decode path is clean.
#   CTX=524288 DEPTH=0 NPRED=32 PROMPT=/path/to/prompt.txt TAG=... perf/run-yarn-512k.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
LINE=${LINE:-ud}
PORT=${PORT:-8095}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-yarn512k-$(date +%m%d-%H%M)}
CTX=${CTX:-524288}
YARN_ORIG_CTX=${YARN_ORIG_CTX:-262144}
ROPE_SCALE=${ROPE_SCALE:-} # default: computed as CTX/YARN_ORIG_CTX below
DEPTH=${DEPTH:-0}
NPRED=${NPRED:-32}
PROMPT=${PROMPT:-/Users/troff/play/benchprompt.txt}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p "$OUT"
slog="$OUT/$TAG.server.log"

source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" turbo4
pick_args "$LINE" turbo4
if [ -z "$ROPE_SCALE" ]; then
  ROPE_SCALE=$(python3 -c "print($CTX/$YARN_ORIG_CTX)")
fi
if [ "$DEPTH" = 0 ]; then spec=(--spec-type none); else spec=("${PICK_SPEC[@]:0:2}" --spec-type draft-dflash --spec-draft-n-max "$DEPTH"); fi

echo "=== yarn-512k $TAG: line=$LINE ctx=$CTX orig_ctx=$YARN_ORIG_CTX rope_scale=$ROPE_SCALE depth=$DEPTH ==="
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD)"
echo "model  : $PICK_MODEL"
for i in $(seq 1 60); do lsof -ti :$PORT >/dev/null 2>&1 || break; sleep 2; done
if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi

env "${PICK_ENV[@]}" $EXTRA_ENV "$BIN/llama-server" -m "$PICK_MODEL" -c "$CTX" -fa on \
  -ctk turbo4 -ctv turbo4 -ctkd f16 -ctvd f16 \
  --rope-scaling yarn --rope-scale "$ROPE_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX" \
  "${spec[@]}" $EXTRA_ARGS --port $PORT >"$slog" 2>&1 &
pid=$!; ok=0
echo "pid    : $pid  (log: $slog)"
for i in $(seq 1 300); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
  sleep 2
  kill -0 $pid 2>/dev/null || { echo "SERVER DIED during load:"; tail -40 "$slog"; exit 1; }
done
[ $ok = 1 ] || { echo "health timeout, still alive - check $slog"; tail -40 "$slog"; kill -9 $pid; exit 1; }
echo "loaded OK, sending a $NPRED-token completion from $PROMPT"
grep -E "n_ctx_orig_yarn|freq_scale|freq_base|rope scaling|n_ctx  *=|n_ctx_seq" "$slog" | tail -10

python3 -c "
import json
print(json.dumps({'prompt': open('$PROMPT').read(), 'n_predict': $NPRED, 'temperature': 0}))" \
  | curl -s -H "Content-Type: application/json" -X POST "http://127.0.0.1:$PORT/completion" -d @- --max-time "${MAXTIME:-14400}" > "$OUT/$TAG.json"
python3 -c "
import json,hashlib
d=json.load(open('$OUT/$TAG.json'))
if 'error' in d:
    print('COMPLETION ERROR', json.dumps(d['error'])[:300])
else:
    t=d.get('timings',{}); c=d.get('content','')
    sha=hashlib.sha1(c.encode()).hexdigest()[:12]
    print('OK: prompt %d tok  %.1f ms  |  decode %.3f t/s  n=%d  sha1=%s' % (t.get('prompt_n',0), t.get('prompt_ms',0), t.get('predicted_per_second',0), t.get('predicted_n',0), sha))
    print('text:', repr(c[:200]))
"
kill -TERM $pid 2>/dev/null; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
echo "server log: $slog"
