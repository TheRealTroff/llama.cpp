#!/bin/bash
# llama.cpp vs dflash_mlx, both sides re-measured in one run.
#
# Supersedes kvquant-experiments/RUN_HEAD_TO_HEAD.sh, which had two problems:
#   - it ran the llama.cpp side at MTP d1 with a PARTIAL env (GGML_MV_NC + GGML_MM_SKINNY
#     only), which is no longer the prod pick and is ~3 t/s slow;
#   - the writeup recorded a date but no commit sha, so 24 commits landed under the number
#     and the staleness was invisible. This script prints the sha into its own output.
#
# The prompt file for the MLX side is REGENERATED from benchprompt.txt on every run, so the
# two sides cannot drift apart. The original cross-framework error in results.md was exactly
# a prompt mismatch (18 tokens vs 8288).
#
# Protocol is kept identical to the archived run so the dflash side is comparable to the
# 29.55 t/s on record: 5 runs per side, 120 s between runs, 180 s cooldowns. Note that
# head-to-head-cooled.md found cooling changes nothing (-0.01% / +0.05%, under 0.6 pooled
# sd); the waits are kept for comparability, not because they are believed to matter.
set -u

B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf
MD=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf
PROMPT=/Users/troff/play/benchprompt.txt
PORT=8090
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-h2h-$(date +%m%d-%H%M)}
JSONL=$OUT/$TAG-prompt.jsonl
NRUN=${NRUN:-5}
PAUSE=${PAUSE:-120}
mkdir -p "$OUT"

PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1)

echo "=== head to head: $TAG ==="
echo "llama.cpp : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty), binary $(date -r "$BIN/llama-server" '+%Y-%m-%d %H:%M')"
echo "dflash    : $("/Users/troff/play/omlx/.venv/bin/dflash" --version 2>&1 | head -1)"
echo "env       : ${PICK_ENV[*]}"
echo "runs      : $NRUN per side, ${PAUSE}s between"

python3 -c "
import json, hashlib
p = open('$PROMPT').read()
json.dump({'id':'btree','suite':'btree','prompt':p}, open('$JSONL','w')); open('$JSONL','a').write('\n')
print('prompt    : %d chars, sha1 %s (both sides)' % (len(p), hashlib.sha1(p.encode()).hexdigest()[:12]))
"
echo

echo "### initial cooldown 180s ###"; sleep 180

echo "### llama.cpp: prod pick (uniform Q4_0 + pure-Q4_0 drafter + dflash n6, full env) ###"
: > "$OUT/$TAG-llama.txt"
for i in $(seq 1 $NRUN); do
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "  ABORT: port $PORT busy"; break; fi
  env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$M" -c 10240 -fa on -ctk f16 -ctv f16 \
    -md "$MD" --spec-type draft-dflash --spec-draft-n-max 6 \
    --port $PORT >"$OUT/$TAG-llama-srv-$i.log" 2>&1 &
  pid=$!; ok=0
  for t in $(seq 1 200); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2; kill -0 $pid 2>/dev/null || break
  done
  if [ $ok = 1 ] && lsof -ti :$PORT 2>/dev/null | grep -qx "$pid"; then
    python3 -c "
import json
p = open('$PROMPT').read()
print(json.dumps({'prompt': p, 'n_predict': 300, 'temperature': 0}))" \
    | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin); t=d.get('timings',{})
acc = 100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
print('%.4f %.1f %s' % (t.get('predicted_per_second',0), acc,
      hashlib.sha1(d.get('content','').encode()).hexdigest()[:12]))" >> "$OUT/$TAG-llama.txt"
  else
    echo "  run $i FAILED to start"
  fi
  kill -TERM $pid 2>/dev/null
  for t in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "  llama run $i: $(tail -1 "$OUT/$TAG-llama.txt")"
  [ "$i" -lt "$NRUN" ] && sleep $PAUSE
done

echo
echo "### pause 180s before switching framework ###"; sleep 180

echo "### dflash_mlx: block 5, w4:gs64 (same settings as the archived 29.55) ###"
cd /Users/troff/play
"/Users/troff/play/omlx/.venv/bin/dflash" benchmark \
  --model /Users/troff/play/mlx-models/mlx-community/Qwen3.8-27B-4bit \
  --draft /Users/troff/play/mlx-models/incoai/Qwen3.8-27B-DFlash2 \
  --prompt-file "$JSONL" \
  --max-tokens 300 --block-tokens 5 --no-chat-template --no-eos \
  --draft-quant w4:gs64 --only-dflash --repeat $NRUN --cooldown $PAUSE \
  > "$OUT/$TAG-dflash_raw.txt" 2>&1
tail -25 "$OUT/$TAG-dflash_raw.txt"

ART=$(ls -dt /Users/troff/play/.artifacts/dflash/benchmarks/*/ 2>/dev/null | head -1)
echo "ARTIFACTS: $ART"

echo
echo "=== summary ==="
python3 - "$OUT/$TAG-llama.txt" "$ART" <<'PY'
import sys, json, os, statistics as st

rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
tps  = [float(r[0]) for r in rows]
shas = {r[2] for r in rows if len(r) > 2}
def fmt(v):
    if not v: return "no runs"
    m = st.mean(v)
    s = st.stdev(v) if len(v) > 1 else 0.0
    return "%.3f +/- %.3f  (n=%d, min %.3f max %.3f)" % (m, s, len(v), min(v), max(v))

print("llama.cpp  : %s" % fmt(tps))
print("             acc %s, sha %s" % ({r[1] for r in rows}, shas if len(shas)==1 else "MIXED "+str(shas)))

d, meta = [], []
art = sys.argv[2].strip()
p = os.path.join(art, "results.json") if art else ""
if p and os.path.exists(p):
    j = json.load(open(p))
    for pr in j.get("prompts", []):
        for r in pr.get("runs", []):
            f = r.get("dflash") or {}
            v = f.get("generation_tps")
            if v:
                d.append(float(v))
                meta.append(f)
else:
    print("(no results.json at %r - check the ARTIFACTS path)" % p)

print("dflash_mlx : %s" % fmt(d))
if meta:
    m = meta[0]
    cyc = m.get("cycles") or 0
    acc = m.get("accepted_from_draft") or 0
    # their acceptance_ratio is accepted/COMMITTED, not accepted/attempted; and attempted is
    # NOT cycles*block because verify_mode defaults to adaptive (see acceptance-metric-conversion.md)
    print("             acceptance_ratio %s (= accepted/committed), accepted %s, cycles %s, tokens/cycle %.3f"
          % (m.get("acceptance_ratio"), acc, cyc, m.get("tokens_per_cycle") or 0))
    am = m.get("adaptive_metrics") or {}
    if am:
        print("             adaptive: %s" % json.dumps(am)[:200])
    print("             NOTE: converting acceptance for comparison needs the adaptive denominator,")
    print("             not cycles*block. See perf/acceptance-metric-conversion.md.")
if tps and d:
    print()
    print("gap        : %.3fx  (dflash_mlx / llama.cpp)" % (st.mean(d)/st.mean(tps)))
print()
print("Record the llama.cpp commit sha with this number.")
PY
