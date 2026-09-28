#!/bin/bash
# Vision prefill profile (2026-09-28, perf/vision/vision-profile-sep28.md): llama-mtmd-cli under a pick line's env
# (the served kernels; the CLI has no drafter), one image at three rungs, an unprofiled timing arm and a GGML_METAL_PROFILE=1
# arm each. Read the profiled dumps with perf/vision/metalprof-vision.py <rung>-prof.log [top]  (m1 = LLM ctx, m2 = clip ctx).
#   [LINE=q4|ud] [RUNGS="1024 2048 full"] [ARMS="time prof"] [OUT=dir] bash perf/vision/run-vision-profile.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/vision-prof-$(date +%m%d-%H%M)}; mkdir -p $OUT
LINE=${LINE:-q4}
source $B/perf/pick.sh
pick_env $LINE; pick_args $LINE
export "${PICK_ENV[@]}"
echo "pick env: ${PICK_ENV[*]}" > $OUT/env.txt
MMPROJ=/Users/troff/play/qwen3.8-mmproj-F16.gguf
TPL="$(cat $B/perf/vision/chat-template-nothink.jinja)"
IMGDIR=/Users/troff/play/images/prepared/IMG_3334
for arm in ${ARMS:-time prof}; do
  for rung in ${RUNGS:-1024 2048 full}; do
    f=$(ls $IMGDIR/IMG_3334-$rung.* | head -1)
    prof=0; [ $arm = prof ] && prof=1
    echo "=== $arm $rung $(date +%H:%M:%S)"
    GGML_METAL_PROFILE=$prof $B/build/bin/llama-mtmd-cli -m $PICK_MODEL --mmproj $MMPROJ --image "$f" \
      -p "Describe this image in one sentence." -n 1 --temp 0 -ngl 99 -c 16384 --no-warmup -fa on -ctk turbo4 -ctv turbo4 \
      --jinja --chat-template "$TPL" > $OUT/$rung-$arm.txt 2> $OUT/$rung-$arm.log
    echo "rc=$? $(grep -o 'encoding done in [0-9]* ms' $OUT/$rung-$arm.log) $(grep -E 'prompt eval time' $OUT/$rung-$arm.log | head -1)"
  done
done
echo "=== done $(date +%H:%M:%S)"
