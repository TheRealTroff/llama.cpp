#!/bin/bash
# Turbo4 FA port (perf/ud-model.md step 16): capture the Turbo4 batched FA kernels beside their f16
# twins at the same dispatch shape (decode width 4 gqah=6 nwg=8; prefill 512 rows), replay headlessly,
# dump register/instruction stats and the per-instruction table. skills/metal-gpu-profile: timing
# comes from an uncaptured test-backend-ops perf pass, never from the capture.
set -u
B=${B:-/Users/troff/play/llama.cpp-turbo4-fa}
BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/profiles/turbo4-fa-sep06}
KV=${KV:-8448}
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
STATS=/Users/troff/.claude/skills/metal-gpu-profile/references/gpuprofiler-stats.py
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
mkdir -p "$OUT"
COMMON="GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_FA_GQA_HEADS=4,6 GGML_FA_GQA4_NWG=6 GGML_FA_GQA_W3_NWG=13 TURBO_AUTO_ASYMMETRIC=0 GGML_FA_GQA_F16=1"
F16="GGML_FA_QT=1,GGML_FA_QR=8,GGML_FA_Q16=1"
# label:nb:type:extra-env(comma-separated; type as the perf case prints it: f16, turbo4)
ARMS=${ARMS:-"t4_dec_w4:4:turbo4: f16_dec_w4:4:f16:$F16 t4_pre:512:turbo4: f16_pre:512:f16:$F16"}
for arm in $ARMS; do
    IFS=: read -r label nb type envs <<<"$arm"; envs=${envs//,/ }
    if [ -d "$OUT/$label.gputrace" ]; then echo "$label: capture exists, skipping"; else
        log=$OUT/$label.capture.log
        ( cd "$B" && env MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 $COMMON $envs $EXTRA_ENV \
            "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=$KV,nb=$nb,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=$type,type_V=$type," ) >"$log" 2>&1
        trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$log" | head -1)
        kn=$(grep -oE 'loaded kernel_flash_attn_ext_[A-Za-z0-9_=]+' "$log" | grep -v reduce | tail -1)
        [ -n "$trace" ] && [ -d "$trace" ] || { echo "$label: NO TRACE (see $log)"; continue; }
        mv "$trace" "$OUT/$label.gputrace"
        echo "$label: $kn  $(du -sh "$OUT/$label.gputrace" | cut -f1)"
    fi
    if [ ! -f "$OUT/$label.replay/streamData" ]; then
        python3 "$HEADLESS" "$OUT/$label.gputrace" "$OUT/$label.replay" >"$OUT/$label.replay.log" 2>&1 \
            || { echo "$label: replay FAILED (see $OUT/$label.replay.log)"; continue; }
    fi
    python3 "$STATS" --all "$OUT/$label.replay/streamData" >"$OUT/$label.stats.txt" 2>&1
    "$PY" "$B/perf/shaderprof-table.py" "$OUT/$label.replay/raw" --kernel flash_attn_ext --top 80 --json "$OUT/$label.instr.json" >"$OUT/$label.instr.txt" 2>&1
    rm -rf "$OUT/$label.replay/raw"
    echo "$label: stats $(grep -c . "$OUT/$label.stats.txt") lines, instr $(grep -c '^   [0-9]' "$OUT/$label.instr.txt") rows"
done
rm -rf /tmp/com.apple.gputools.profiling
echo "done: $OUT"
