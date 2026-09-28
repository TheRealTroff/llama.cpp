#!/bin/bash
# ViT ffn_down tail-only K check, end-to-end arm (exp/vit-ffn-down): llama-mtmd-cli under the q4 pick env, prod's
# binary vs this tree's, interleaved; per rung the encoder ms and the sha of a 64-token answer.
#   [B=tree] [BASE=prod tree] [IMAGES=..] [RUNGS=..] [REPS=2] [OUT=dir] bash perf/vision/run-vit-ffn-down-e2e.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-vitffn}; BASE=${BASE:-/Users/troff/play/llama.cpp-prod}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/vit-ffn-down-e2e-$(date +%m%d-%H%M)}; mkdir -p "$OUT"
IMAGES=${IMAGES:-IMG_3334 IMG_2924}; RUNGS=${RUNGS:-1024 2048 full}; REPS=${REPS:-2}
source $B/perf/pick.sh; pick_env q4; pick_args q4; export "${PICK_ENV[@]}"
MMPROJ=/Users/troff/play/qwen3.8-mmproj-F16.gguf; TPL="$(cat $B/perf/vision/chat-template-nothink.jinja)"
printf 'rep\tarm\timage\trung\tencode_ms\tsha12\n' > "$OUT/summary.tsv"
for rep in $(seq 1 $REPS); do for arm in prod branch; do
  T=$B; [ $arm = prod ] && T=$BASE
  for img in $IMAGES; do for rung in $RUNGS; do
    f=$(ls /Users/troff/play/images/prepared/$img/$img-$rung.* 2>/dev/null | head -1); [ -n "$f" ] || continue
    n=$rep-$arm-$img-$rung
    $T/build/bin/llama-mtmd-cli -m $PICK_MODEL --mmproj $MMPROJ --image "$f" -p "Describe this image in detail." -n 64 --temp 0 -ngl 99 \
      -c 16384 --no-warmup -fa on -ctk turbo4 -ctv turbo4 --jinja --chat-template "$TPL" > "$OUT/$n.txt" 2> "$OUT/$n.log"
    enc=$(grep -o 'encoding done in [0-9]* ms' "$OUT/$n.log" | grep -o '[0-9]*' | head -1); sha=$(shasum -a 256 "$OUT/$n.txt" | cut -c1-12)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $rep $arm $img $rung "$enc" $sha >> "$OUT/summary.tsv"; echo "rep$rep $arm $img $rung enc=${enc}ms sha=$sha"
  done; done
done; done
python3 - "$OUT/summary.tsv" <<'PY'
import sys,csv,statistics as st
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
keys=sorted({(r['image'],r['rung']) for r in rows})
ok=0; bad=0
for k in keys:
    p=[r for r in rows if (r['image'],r['rung'])==k and r['arm']=='prod']; b=[r for r in rows if (r['image'],r['rung'])==k and r['arm']=='branch']
    shas={r['sha12'] for r in p+b}; same=len(shas)==1; ok+=same; bad+=not same
    pm=st.median(int(r['encode_ms']) for r in p); bm=st.median(int(r['encode_ms']) for r in b)
    print(f"  {k[0]} {k[1]}: prod {pm:.0f} ms -> branch {bm:.0f} ms ({100*(bm/pm-1):+.1f}%)  shas {'SAME' if same else 'DIFF '+str(shas)}")
print(f"sha-identical rows {ok}, differing {bad}")
PY
echo "done: $OUT"
