#!/bin/bash
# Re-run of the 2026-09-16 op-level claim in perf/fa-decode-tile24.md ("bitwise identical to the pick's routes")
# with the fixed dump hook (before 2026-09-28 GGML_TEST_DUMP wrote the CPU reference, perf/vision/vit-fa-dk72.md).
# Per line: the pick env as is (24-row tile, GGML_FA_Q24=1 + ROWS) vs GGML_FA_Q24=0 (the 8-row routes), Turbo4
# head-256 GQA6 cases kv 512 / 8448, widths 1-6, seeded; cmp of the Metal outputs + the kernels each arm loaded.
set -u
B=${B:-/Users/troff/play/llama.cpp-vitfa}; T=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/tile24-recheck-$(date +%m%d-%H%M)}; mkdir -p $OUT
for L in q4 ud; do
  ( source $B/perf/pick.sh; pick_env $L; export "${PICK_ENV[@]}"
    echo "== $L: $(printf '%s\n' "${PICK_ENV[@]}" | grep -E 'GGML_FA_(Q24|TR)' | tr '\n' ' ')"
    for arm in tile row8; do
      E="X=1"; [ $arm = row8 ] && E="GGML_FA_Q24=0"
      rm -rf $OUT/$L-$arm; mkdir -p $OUT/$L-$arm
      env $E GGML_TEST_SEED=7 GGML_TEST_DUMP=$OUT/$L-$arm $T test -o FLASH_ATTN_EXT -b MTL0 -p "hsk=256,hsv=256,nh=4,nr23=\[6,1\]" > $OUT/$L-$arm.log 2>&1
      echo "  $arm: $(grep -c ': OK' $OUT/$L-$arm.log) OK $(grep -c ': FAIL' $OUT/$L-$arm.log) FAIL, $(ls $OUT/$L-$arm | wc -l | tr -d ' ') dumps; kernels: $(grep -o 'loaded kernel_flash_attn_ext_q[a-z0-9]*_turbo4' $OUT/$L-$arm.log | sort | uniq -c | tr '\n' ' ')"
    done
    s=0; d=0; for f in $OUT/$L-tile/*.bin; do b=$(basename $f); cmp -s $f $OUT/$L-row8/$b && s=$((s+1)) || { d=$((d+1)); echo "    differ $b" >> $OUT/$L-diff.txt; }; done
    echo "  $L tile vs 8-row: $s identical, $d differ" )
done
echo "done: $OUT"
