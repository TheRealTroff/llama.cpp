#!/bin/bash
# perf/run-specev-pick-gate.sh - the gate for picking LLAMA_SPEC_EV=1 (adaptive verify depth) on both lines
# (2026-09-17, owner: "time we flipped the switch on adaptive depth spec"). Owner's condition of 2026-09-07:
# "KLD + agreement of its text vs the fixed-depth pick before any pick - its rounds verify at widths 1-8 and
# carry the union of the width families' decode numerics; then a new lineage."
#
# Every controller arm passes its flags explicitly (EV_ENV); every fixed arm opts out of the picked controller with
# PICK_SPEC_EV=0 and runs depth 3 (the q4 manifest picks LLAMA_SPEC_EV since 2026-09-17 afternoon). Sequential, one GPU. STEPS= selects (default all):
#   ab      8K corpus A/B per line: fixed 3 | ev hybrid | fixed 3 again (run-spec-ev-ab.sh on the PROD build, LV=3
#           = INFO: the spec-ev summary line per request is in the log, the per-round DBG lines are not - LV 1 is
#           errors only and left the first run without them; AB_ARMS= reruns a subset under the same TAG)
#   kldq4   the q4 line's width-6..8 route (Q4_0 skinny SoA tile, GGML_MM_SKINNY=6) priced pairwise vs the q4 width-4
#           decode base at -b 6 -ub 6 (q4-decode-kld.md method; routing proof from a 1-chunk -v run first). The ud
#           side of the union is priced already: width 4 = base, width 5 (w6-verify-cliff.md last section), widths
#           6-8 = the GEN=6 tile (same file, gated section); widths 1-3 are the exact-product readers (BI class).
#   agree   own-text agreement per line: 2048-token greedy completions on the free-form prompts, fixed 3 vs hybrid;
#           each corpus (prompt + completion) scored against fresh q8_0 reference logits through the line's f16 pick
#           (run-quant-kld.sh W=corpus) -> same-top / KLD of each text under the reference. Same statistics on both
#           texts = the controller's trajectory is as much the model's own as the fixed-depth one.
#   longctx 96K pair per line (run-longctx-pick.sh, 600 tokens): fixed 3 vs hybrid - the controller's cost seeds are
#           the 8K Turbo4 curve, the EMA has to relearn the 96K one within the request.
# Results: kvquant-experiments/results/specev-gate-sep17-* ; this log: results/specev-gate-sep17.log
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=/Users/troff/play/llama.cpp-prod
BIN=$B/build/bin
OUT=/Users/troff/play/kvquant-experiments/results
SCRATCH=/Users/troff/play/kvquant-experiments/logits
WIKI=/Users/troff/play/kvquant-experiments/data/wikitext-2-raw/wiki.test.raw
REF_Q8=/Users/troff/play/Qwen3.8-27B-conv-q8_0.gguf
STEPS=${STEPS:-"ab kldq4 agree longctx"}
LINES=${LINES:-"ud q4"}
AB_ARMS=${AB_ARMS:-"n3 hybrid n3b"}
DATE=sep17
EV_ENV="LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=hybrid LLAMA_SPEC_EV_WIDTHS=3,7"  # the picked form since the tax diagnosis (section 10); fixed arms opt out with PICK_SPEC_EV=0
export B
cd "$B"
source perf/pick.sh
echo "=== specev pick gate $DATE: steps [$STEPS] lines [$LINES]; prod $(git rev-parse --short HEAD), binary $(date -r "$BIN/llama-server" '+%m-%d %H:%M'); $(date '+%F %T') ==="
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
stamp() { echo "--- $(date '+%T') $* ---"; }

# 1. the 8K corpus A/B
if has ab; then
  for line in $LINES; do
    stamp "ab $line"
    LINE=$line LV=3 ARMS="$AB_ARMS" TAG=specev-gate-$DATE-$line perf/run-spec-ev-ab.sh 2>&1 | grep -v '^$'
    python3 perf/specev-ab-report.py specev-gate-$DATE-$line "n3 hybrid n3b" 2>&1 | tail -20
  done
fi

# 2. the q4 width-6..8 route, pairwise vs the q4 width-4 decode base
if has kldq4; then
  stamp "kldq4 routing proof (1 chunk, -b 6 -ub 6, -v)"
  pick_env q4 f16
  env "${PICK_ENV[@]}" "$BIN/llama-perplexity" -m "$PICK_MODEL_Q4" -f "$WIKI" -c 2048 --chunks 1 -b 6 -ub 6 -fa on \
    -ctk f16 -ctv f16 -v > "$OUT/specev-gate-$DATE-q4-b6-route.log" 2>&1
  grep -oE 'kernel_(mul_mm|mul_mv|flash_attn)[A-Za-z0-9_]*' "$OUT/specev-gate-$DATE-q4-b6-route.log" | sort | uniq -c | sort -rn | head -12
  grep -E 'Final estimate' "$OUT/specev-gate-$DATE-q4-b6-route.log"
  stamp "kldq4 pairwise (-b 6 -ub 6 vs kld-base-kld-pair-q4dec4-sep17, 24 chunks)"
  env "${PICK_ENV[@]}" KV=f16 PPL_EXTRA="-b 6 -ub 6" LABEL=-w6pick-dec6 TAG=kld-pair-q4dec4-sep17 \
    perf/run-quant-kld.sh "$PICK_MODEL_Q4" 2>&1 | grep -v '^$'
fi

# 3. own-text agreement corpora
score_corpus() {  # score_corpus <line> <corpus> <tag>
  local line=$1 corpus=$2 tag=$3 base chunks
  base=$SCRATCH/kld-base-$tag.dat
  chunks=$(python3 -c "import sys; b=len(open('$corpus','rb').read()); print(max(1, int(b/4.0/2048)))")
  pick_env "$line" f16; pick_args "$line" f16
  stamp "agree score $tag ($chunks chunks): q8_0 reference logits (clean env)"
  "$BIN/llama-perplexity" -m "$REF_Q8" -f "$corpus" -c 2048 --chunks "$chunks" -fa on -ctk f16 -ctv f16 \
    --kl-divergence-base "$base" > "$OUT/$tag-ref.log" 2>&1 || { echo "ref FAILED"; tail -3 "$OUT/$tag-ref.log"; return 1; }
  grep -E 'Final estimate' "$OUT/$tag-ref.log" | sed 's/^/  ref /'
  env "${PICK_ENV[@]}" W="$corpus" CHUNKS="$chunks" TAG="$tag" KV=f16 perf/run-quant-kld.sh "$PICK_MODEL" 2>&1 | grep -v '^$'
  rm -f "$base"
}
if has agree; then
  AGREE_PROMPTS="benchprompt 01-code-explain 02-prose-creative 03-chat-support 06-algorithms 08-story"
  for line in $LINES; do
    for arm in n3 hybrid; do
      tag=specev-agree-$DATE-$line-$arm
      stamp "agree generate $tag (2048 tokens x 6 prompts, Turbo4 pick)"
      if [ $arm = n3 ]; then
        PICK_SPEC_EV=0 LINE=$line KV=turbo4 NPRED=2048 LV=3 DEPTHS=3 PROMPTS="$AGREE_PROMPTS" TAG=$tag perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died'
        d=3
      else
        env $EV_ENV LINE=$line KV=turbo4 NPRED=2048 LV=3 DEPTHS=7 PROMPTS="$AGREE_PROMPTS" TAG=$tag perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died'
        d=7
      fi
      corpus=/Users/troff/play/kvquant-experiments/data/generated-$tag.txt
      : > "$corpus"
      for p in $AGREE_PROMPTS; do
        pf=$([ $p = benchprompt ] && echo /Users/troff/play/benchprompt.txt || echo "$B/perf/prompts/$p.txt")
        cat "$pf" "$OUT/$tag-n$d-$p-r1.txt" >> "$corpus"; printf '\n\n' >> "$corpus"
      done
      echo "  corpus $corpus: $(wc -c < "$corpus" | tr -d ' ') bytes"
      score_corpus "$line" "$corpus" "$tag"
    done
    # where the two texts diverge, per prompt
    python3 - "$OUT" specev-agree-$DATE-$line "$AGREE_PROMPTS" <<'PY'
import sys, os, hashlib
out, tag, prompts = sys.argv[1], sys.argv[2], sys.argv[3].split()
for p in prompts:
    a = open(f'{out}/{tag}-n3-n3-{p}-r1.txt').read(); b = open(f'{out}/{tag}-hybrid-n7-{p}-r1.txt').read()
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
    same = (a == b)
    print('  %-20s %s len %d/%d  first divergence at char %s  sha %s/%s' % (p, 'SAME' if same else 'forks', len(a), len(b),
          'none' if i is None else i, hashlib.sha1(a.encode()).hexdigest()[:8], hashlib.sha1(b.encode()).hexdigest()[:8]))
PY
  done
fi

# 4. 96K pair
if has longctx; then
  for line in $LINES; do
    stamp "longctx $line fixed 3"
    PICK_SPEC_EV=0 LINE=$line KV=turbo4 CTX=102400 NPRED=600 DEPTH=3 LV=3 TAG=specev-96k-$DATE-$line-n3 perf/run-longctx-pick.sh 2>&1 | grep -E '^\[|wall|spec-prof|ABORT|died|ERROR'
    stamp "longctx $line hybrid"
    LINE=$line KV=turbo4 CTX=102400 NPRED=600 DEPTH=7 LV=3 EXTRA_ENV="$EV_ENV" TAG=specev-96k-$DATE-$line-hybrid perf/run-longctx-pick.sh 2>&1 | grep -E '^\[|wall|spec-prof|ABORT|died|ERROR'
    grep -h 'spec-ev:' "$OUT/specev-96k-$DATE-$line-hybrid.server.log" | tail -1 | cut -c1-300
  done
fi
echo "=== GATE-COMPLETE $(date '+%F %T') ==="
