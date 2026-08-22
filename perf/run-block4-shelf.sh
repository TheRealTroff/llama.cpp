#!/bin/bash
# Is dflash_mlx's cheap block-4 cycle real, or lazy-eval misattribution?
# Task: perf/block4-shelf-probe.md. dflash_mlx side ONLY - nothing of ours is rebuilt or
# measured here, so this does not need the llama.cpp server at all.
#
# Their archived adaptive run spends 81/99 cycles at block 4 and derives 91.9 ms/cycle
# there against 140.2 at block 5. If that shelf is real, pinning block 4 should land near
# 33 t/s. If the split was an artifact of MLX lazy eval charging deferred block-4 work to
# the block-5 cycle that forces the sync, pinned block 4 lands near the adaptive 29.6.
#
# How the pinning works (engine/spec_epoch.py:340 and :343):
#   _AdaptiveBlockPolicy.from_runtime returns None - i.e. fixed block, no controller -
#   unless verify_mode == "adaptive", and ALSO returns None when block_tokens <= 4.
#   So: block 4 is inherently fixed, and block 5 needs DFLASH_VERIFY_MODE=dflash.
#
# Three configs on ONE protocol so the comparison is internally valid regardless of how it
# compares to the archived 29.613. The adaptive arm is the control: if it does not
# reproduce ~29.6 today, the environment drifted and the other two arms mean nothing.
set -u

VENV=/Users/troff/play/omlx/.venv
MODEL=/Users/troff/play/mlx-models/mlx-community/Qwen3.8-27B-4bit
DRAFT=/Users/troff/play/mlx-models/incoai/Qwen3.8-27B-DFlash2
PROMPT=/Users/troff/play/benchprompt.txt
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-shelf-$(date +%m%d-%H%M)}
JSONL=$OUT/$TAG-prompt.jsonl
NRUN=${NRUN:-3}
PAUSE=${PAUSE:-60}
mkdir -p "$OUT"

echo "=== block-4 shelf probe: $TAG ==="
echo "dflash    : dflash-mlx $("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('dflash-mlx'))" 2>/dev/null || echo unknown)"
echo "llama.cpp : $(cd /Users/troff/play/llama.cpp-prod && git rev-parse --short HEAD) (recorded for provenance; not measured here)"
echo "runs      : $NRUN per config, ${PAUSE}s cooldown"

# regenerate the prompt jsonl from the same source the head-to-head uses, so this cannot
# drift from the run we are comparing against
python3 -c "
import json, hashlib
p = open('$PROMPT').read()
json.dump({'id':'btree','suite':'btree','prompt':p}, open('$JSONL','w')); open('$JSONL','a').write('\n')
print('prompt    : %d chars, sha1 %s' % (len(p), hashlib.sha1(p.encode()).hexdigest()[:12]))
"
echo

run_cfg () {  # label, block_tokens, verify_mode
  local label=$1 block=$2 mode=$3
  echo "### $label: --block-tokens $block, DFLASH_VERIFY_MODE=$mode ###"
  cd /Users/troff/play
  env DFLASH_VERIFY_MODE="$mode" "$VENV/bin/dflash" benchmark \
    --model "$MODEL" --draft "$DRAFT" --prompt-file "$JSONL" \
    --max-tokens 300 --block-tokens "$block" --no-chat-template --no-eos \
    --draft-quant w4:gs64 --only-dflash --repeat $NRUN --cooldown $PAUSE \
    > "$OUT/$TAG-$label.raw.txt" 2>&1
  local art
  art=$(ls -dt /Users/troff/play/.artifacts/dflash/benchmarks/*/ 2>/dev/null | head -1)
  echo "$art" > "$OUT/$TAG-$label.artifact.txt"
  echo "  artifact: $art"
  python3 - "$art" "$label" <<'PY'
import json, os, sys, statistics as st
art, label = sys.argv[1].strip(), sys.argv[2]
p = os.path.join(art, "results.json")
if not os.path.exists(p):
    print("  (no results.json - run failed?)"); raise SystemExit
j = json.load(open(p))
tps, cyc, tpc, blocks, modes = [], [], [], {}, {}
for pr in j.get("prompts", []):
    for r in pr.get("runs", []):
        f = r.get("dflash") or {}
        if not f.get("generation_tps"): continue
        tps.append(float(f["generation_tps"])); cyc.append(f.get("cycles"))
        tpc.append(f.get("tokens_per_cycle"))
        am = f.get("adaptive_metrics") or {}
        for k, v in (am.get("cycles_by_block") or {}).items(): blocks[k] = blocks.get(k, 0) + v
        for k, v in (am.get("cycles_by_mode")  or {}).items(): modes[k]  = modes.get(k, 0) + v
m = st.mean(tps); s = st.stdev(tps) if len(tps) > 1 else 0.0
print("  t/s       : %.3f +/- %.3f  (n=%d, %s)" % (m, s, len(tps), ", ".join("%.3f" % v for v in tps)))
print("  cycles    : %s   tok/cycle: %s" % (cyc, ["%.4f" % v for v in tpc if v]))
print("  by_block  : %s   by_mode: %s" % (blocks or "(none - fixed block, as intended)", modes or "(none)"))
if tpc and tpc[0]:
    print("  ms/cycle  : %.2f" % (1000.0 * tpc[0] / m))
PY
  echo
}

run_cfg fixed-b4  4 dflash
run_cfg fixed-b5  5 dflash
run_cfg adaptive  5 adaptive

echo "=== done. Decision rule (block4-shelf-probe.md): fixed-b4 near 33 t/s = shelf real;"
echo "=== near 29-30 = by-block split was lazy-eval misattribution. Check adaptive arm"
echo "=== reproduces ~29.6 first, and that by_block is empty for the two fixed arms."
