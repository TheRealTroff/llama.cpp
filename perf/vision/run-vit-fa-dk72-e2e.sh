#!/bin/bash
# dk=72 FA form, end-to-end arm (exp/vit-fa-dk72): llama-mtmd-cli under the q4 pick env, same binary, the encoder
# route flipped by GGML_FA_QT_DK72 (0 = generic kernel, 1 = the transposed-Q form); per rung the encoder ms and the
# sha of a 64-token answer - byte-identical answers are the e2e gate, the encoder ms the price.
#   [B=tree] [IMAGES="IMG_3334 IMG_2924"] [RUNGS="1024 2048 full"] [REPS=2] [OUT=dir] bash perf/vision/run-vit-fa-dk72-e2e.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-vitfa}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/vit-fa-dk72-e2e-$(date +%m%d-%H%M)}; mkdir -p "$OUT"
IMAGES=${IMAGES:-IMG_3334 IMG_2924}; RUNGS=${RUNGS:-1024 2048 full}; REPS=${REPS:-2}
source $B/perf/pick.sh; pick_env q4; pick_args q4; export "${PICK_ENV[@]}"
MMPROJ=/Users/troff/play/qwen3.8-mmproj-F16.gguf
TPL="$(cat $B/perf/vision/chat-template-nothink.jinja)"
echo "tree $B $(cd $B && git rev-parse --short HEAD) out $OUT"
printf 'rep\tarm\timage\trung\tencode_ms\tsha12\n' > "$OUT/summary.tsv"
for rep in $(seq 1 $REPS); do
  for arm in gen qt; do
    dk72=1; [ $arm = gen ] && dk72=0
    for img in $IMAGES; do
      for rung in $RUNGS; do
        f=$(ls /Users/troff/play/images/prepared/$img/$img-$rung.* 2>/dev/null | head -1); [ -n "$f" ] || continue
        name=$rep-$arm-$img-$rung
        GGML_FA_QT_DK72=$dk72 GGML_FA_DEBUG=1 $B/build/bin/llama-mtmd-cli -m $PICK_MODEL --mmproj $MMPROJ --image "$f" \
          -p "Describe this image in detail." -n 64 --temp 0 -ngl 99 -c 16384 --no-warmup -fa on -ctk turbo4 -ctv turbo4 \
          --jinja --chat-template "$TPL" > "$OUT/$name.txt" 2> "$OUT/$name.log"
        enc=$(grep -o 'encoding done in [0-9]* ms' "$OUT/$name.log" | grep -o '[0-9]*' | head -1)
        sha=$(shasum -a 256 "$OUT/$name.txt" | cut -c1-12)
        route=$(grep -o 'fa-route: kernel_flash_attn_ext[a-z0-9_]*dk72[^ ]*' "$OUT/$name.log" | head -1 | grep -o 'ext_[a-z0-9_]*dk72' )
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$rep" "$arm" "$img" "$rung" "$enc" "$sha" >> "$OUT/summary.tsv"
        echo "rep$rep $arm $img $rung enc=${enc}ms sha=$sha route=$route"
      done
    done
  done
done
echo "== sha check (gen vs qt, per rep/image/rung)"
python3 - "$OUT/summary.tsv" <<'PY'
import sys,csv
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
g={(r['rep'],r['image'],r['rung']):r for r in rows if r['arm']=='gen'}
ok=bad=0
for r in rows:
    if r['arm']!='qt': continue
    k=(r['rep'],r['image'],r['rung']); a=g.get(k)
    same = a and a['sha12']==r['sha12']; ok+=bool(same); bad+=not same
    print(f"  {k}: gen {a['encode_ms'] if a else '?'} ms -> qt {r['encode_ms']} ms  sha {'SAME' if same else 'DIFF'}")
print(f"identical {ok}, differ {bad}")
PY
echo "done: $OUT"
