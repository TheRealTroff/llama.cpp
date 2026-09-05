#!/bin/bash
# UD line step 6 GPU trace (perf/ud-model.md): capture the width-4 SoA kernels for the three UD
# FFN formats plus their ext r1_4 incumbents and the q4_0 r4kp_v3 reference at the ffn_gate_up
# shape, replay headlessly, dump register/instruction stats and the per-instruction table.
# skills/metal-gpu-profile: do NOT read timing from a captured run; registers, spill and
# instruction counts are compile-time facts. test-backend-ops perf with GGML_MV_REPACK=2 (test
# buffers); the pipeline name of each capture is read from its own stderr.
set -u
B=${B:-/Users/troff/play/llama.cpp-ud-soa}
BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/profiles/ud-soa-w4-sep05}
M=${M:-17408}; K=${K:-5120}; N=${N:-4}
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}   # non-SIP python for the GT frameworks
STATS=/Users/troff/.claude/skills/metal-gpu-profile/references/gpuprofiler-stats.py
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
mkdir -p "$OUT"
ARMS=${ARMS:-"iq4xs_v5:iq4_xs:GGML_MV_SOA_IQ4XS=5 q4K_v2:q4_K:GGML_MV_SOA_KQ=2 q5K_v2:q5_K:GGML_MV_SOA_KQ=2 q40_r4kp3:q4_0:GGML_MV_SOA_W4=1,GGML_MV_SOA_W4_R4KP=3,GGML_MV_SOA_PIN=1 iq4xs_ext:iq4_xs: q4K_ext:q4_K: q5K_ext:q5_K:"}
for arm in $ARMS; do
    label=${arm%%:*}; rest=${arm#*:}; type=${rest%%:*}; envs=${rest#*:}; envs=${envs//,/ }
    if [ -d "$OUT/$label.gputrace" ]; then echo "$label: capture exists, skipping"; else
        log=$OUT/$label.capture.log
        ( cd "$B" && env MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 GGML_MV_REPACK=2 $envs \
            "$BIN" perf -o MUL_MAT -b MTL0 -p "type_a=$type,type_b=f32,m=$M,n=$N,k=$K," ) >"$log" 2>&1
        trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$log" | head -1)
        kn=$(grep -oE 'loaded kernel_mul_mv_[A-Za-z0-9_=]+' "$log" | grep -v cpy | tail -1)
        [ -n "$trace" ] && [ -d "$trace" ] || { echo "$label: NO TRACE (see $log)"; continue; }
        mv "$trace" "$OUT/$label.gputrace"
        echo "$label: $kn  $(du -sh "$OUT/$label.gputrace" | cut -f1)"
    fi
    if [ ! -f "$OUT/$label.replay/streamData" ]; then
        python3 "$HEADLESS" "$OUT/$label.gputrace" "$OUT/$label.replay" >"$OUT/$label.replay.log" 2>&1 \
            || { echo "$label: replay FAILED (see $OUT/$label.replay.log)"; continue; }
    fi
    python3 "$STATS" --all "$OUT/$label.replay/streamData" >"$OUT/$label.stats.txt" 2>&1
    "$PY" "$B/perf/shaderprof-table.py" "$OUT/$label.replay/raw" --kernel mul_mv --top 60 --json "$OUT/$label.instr.json" >"$OUT/$label.instr.txt" 2>&1
    echo "$label: stats $(grep -c . "$OUT/$label.stats.txt") lines, instr $(grep -c '^   [0-9]' "$OUT/$label.instr.txt") rows, replay $(du -sh "$OUT/$label.replay" | cut -f1)"
done
echo "done: $OUT"
