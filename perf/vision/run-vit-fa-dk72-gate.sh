#!/bin/bash
# dk=72 FA form gate (exp/vit-fa-dk72, perf/vision/vit-fa-dk72.md): per-call timing of the encoder shapes under
# the pick env with the generic kernel (GGML_FA_QT_DK72=0) vs the transposed-Q form (qr 8 / 9 / 0), interleaved
# x REPS (ARMS= picks arms; q16 = the 16-row tile, GGML_FA_Q16_DK72=1); then the op-level byte gate (BARMS=) (GGML_TEST_DUMP of the hsk=72 eval cases, generic vs QT, cmp; GGML_TEST_SEED=1 -
# without the seed every process draws its own inputs and everything "differs", 2026-09-28). PART=byte skips the timing.
#   [B=tree] [REPS=2] [PART=all|byte] [OUT=dir] bash perf/vision/run-vit-fa-dk72-gate.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-vitfa}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/vit-fa-dk72-$(date +%m%d-%H%M)}; mkdir -p "$OUT"
REPS=${REPS:-2}; PART=${PART:-all}
source $B/perf/pick.sh; pick_env q4; export "${PICK_ENV[@]}"
T=$B/build/bin/test-backend-ops
echo "tree $B $(cd $B && git rev-parse --short HEAD)  out $OUT"
echo "== per-call timing (us/run, TFLOPS), arms: gen=generic dk72 kernel, qt8=QT qr 8, qt9=QT qr 9, qt0=QT qr 0"
[ "$PART" = byte ] && REPS=0
for rep in $(seq 1 $REPS | awk '$1>=1'); do
  for arm in ${ARMS:-gen qt8 qt9 qt0 q16}; do
    case $arm in gen) E="GGML_FA_QT_DK72=0";; qt8) E="GGML_FA_QR_DK72=8";; qt9) E="GGML_FA_QR_DK72=9";; qt0) E="GGML_FA_QR_DK72=0";; q16) E="GGML_FA_QR_DK72=0 GGML_FA_Q16_DK72=1";; esac
    for n in 3072 12288 16060; do
      line=$(env $E GGML_FA_DEBUG=1 $T perf -o FLASH_ATTN_EXT -b MTL0 -p "hsk=72,hsv=72,nh=16,nr23=\[1,1\],kv=$n,nb=$n," 2>&1 | grep -E 'us/run|fa-route' | tr '\n' ' ' | sed 's/\x1b\[[0-9;]*m//g')
      echo "rep$rep $arm kv=$n :: $line" | tee -a "$OUT/timing.txt"
    done
  done
done
echo "== op-level byte gate: eval cases hsk=72 (Metal vs CPU pass/fail per arm, then cmp of the dumped Metal outputs)"
for arm in ${BARMS:-gen qt8 qt9 q16}; do
  case $arm in gen) E="GGML_FA_QT_DK72=0";; qt8) E="GGML_FA_QR_DK72=8";; qt9) E="GGML_FA_QR_DK72=9";; q16) E="GGML_FA_QR_DK72=0 GGML_FA_Q16_DK72=1";; esac
  rm -rf "$OUT/dump-$arm"; mkdir -p "$OUT/dump-$arm"
  env $E GGML_TEST_SEED=1 GGML_TEST_DUMP="$OUT/dump-$arm" $T test -o FLASH_ATTN_EXT -b MTL0 -p "hsk=72" > "$OUT/test-$arm.log" 2>&1
  echo "$arm: $(grep -c 'OK' "$OUT/test-$arm.log") OK, $(grep -c 'FAIL' "$OUT/test-$arm.log") FAIL, $(ls "$OUT/dump-$arm" | wc -l | tr -d ' ') dumps; $(tail -1 "$OUT/test-$arm.log")"
done
for arm in ${BARMS:-gen qt8 qt9 q16}; do
  [ $arm = gen ] && continue
  nd=0; ne=0
  for f in "$OUT"/dump-gen/*.bin; do b=$(basename "$f"); if cmp -s "$f" "$OUT/dump-$arm/$b"; then ne=$((ne+1)); else nd=$((nd+1)); echo "  DIFF $arm $b"; fi; done
  echo "$arm vs gen: $ne identical, $nd differ"
done
echo "done: $OUT"
