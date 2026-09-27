#!/bin/bash
# Vision reference/gate runner (exp/vision-gate, 2026-09-27). Walks perf/vision/prompts.tsv through llama-mtmd-cli:
# plain build, no pick env, no speculation (the CLI has no drafter), greedy, the CLI's own chat template
# (the model's template with the image marker; NOT the text lineage's hand-rendered template - a separate lineage).
#   LINE=q4|ud [B=<tree with build/bin>] [TAG=..] [ROWS=regex] bash perf/vision/run-vision-ref.sh
# Output: $OUT/$TAG-$LINE/<image>-<size>-<qid>.{txt,log} + summary.tsv (sha12 of the answer, encode ms, wall s)
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
LINE=${LINE:-q4}
TAG=${TAG:-vision-$(date +%m%d-%H%M)}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
IMG=/Users/troff/play/images/prepared
MMPROJ=${MMPROJ:-/Users/troff/play/qwen3.8-mmproj-F16.gguf}
PROMPTS=${PROMPTS:-$(dirname "$0")/prompts.tsv}
ROWS=${ROWS:-.}
case "$LINE" in
  q4) M=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf ;;
  ud) M=/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V2.gguf ;;
  *) echo "unknown LINE $LINE"; exit 1 ;;
esac
D=$OUT/$TAG-$LINE; mkdir -p "$D"
echo "tree $B  line $LINE  model $M  mmproj $MMPROJ  out $D"
printf 'image\tsize\tqid\tkind\tsha12\tencode_ms\twall_s\tn_out\tfile\n' > "$D/summary.tsv"
grep -v '^#' "$PROMPTS" | grep -E "$ROWS" | while IFS=$'\t' read -r image size qid kind n prompt expect; do
  [ -n "$image" ] || continue
  name="$image-$size-$qid"
  args=(-m "$M" --mmproj "$MMPROJ" -p "$prompt" -n "$n" --temp 0 -ngl 99 -c 8192 --no-warmup --jinja --chat-template "$(cat "$(dirname "$0")/chat-template-nothink.jinja")")
  if [ "$size" != none ]; then
    f=$(ls "$IMG/$image/$image-$size".* 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "MISSING $image $size"; continue; }
    args+=(--image "$f")
  fi
  t0=$(date +%s.%N)
  "$B/build/bin/llama-mtmd-cli" "${args[@]}" > "$D/$name.txt" 2> "$D/$name.log"
  rc=$?
  wall=$(python3 -c "print(round($(date +%s.%N)-$t0,1))")
  # answer = stdout minus the empty think block
  python3 - "$D/$name.txt" <<'PY'
import sys,re; p=sys.argv[1]; s=open(p).read()
s=re.sub(r'^\s*<think>\s*</think>\s*','',s,count=1).strip()+'\n'; open(p,'w').write(s)
PY
  sha=$(shasum -a 256 "$D/$name.txt" | cut -c1-12)
  enc=$(grep -o 'encoding done in [0-9]* ms' "$D/$name.log" | grep -o '[0-9]*' | paste -sd+ - | bc 2>/dev/null); enc=${enc:-0}
  nout=$(wc -w < "$D/$name.txt" | tr -d ' ')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$image" "$size" "$qid" "$kind" "$sha" "$enc" "$wall" "$nout" "$name.txt" >> "$D/summary.tsv"
  echo "[$LINE] $name rc=$rc sha=$sha enc=${enc}ms wall=${wall}s :: $(head -c 100 "$D/$name.txt" | tr '\n' ' ')"
done
echo "done: $D/summary.tsv"
