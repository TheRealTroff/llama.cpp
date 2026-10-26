#!/bin/bash
# capture + replay the three q4_0 SoA forms (probe 0/1/2) so the executed instruction counts verify what each probe changed
B=${B:-/Users/troff/play/llama.cpp-issueprobes}; BIN=$B/build/bin/test-backend-ops
OUT=${OUT:-/Users/troff/play/kvquant-experiments/census/issue-probes-oct26}; mkdir -p "$OUT"
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
STATS=/Users/troff/.claude/skills/metal-gpu-profile/references/gpuprofiler-stats.py
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
FILT="type_a=q4_0_soa,type_b=f32,m=17408,n=8,k=5120,"
for probe in 0 1 2; do
  id=q4soa-17408x5120-probe$probe
  (cd "$B" && env MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 GGML_MV_REPACK=2 GGML_MM_SKINNY_BSPLIT=2 GGML_MM_SKINNY_PROBE=$probe "$BIN" perf -o MUL_MAT -b MTL0 -p "$FILT") > "$OUT/$id.capture.log" 2>&1
  trace=$(grep -oE '/tmp/perf-metal-[0-9]+\.gputrace' "$OUT/$id.capture.log" | head -1)
  [ -n "$trace" ] && [ -d "$trace" ] && mv "$trace" "$OUT/$id.gputrace" || { echo "$id: NO TRACE"; continue; }
  python3 "$HEADLESS" "$OUT/$id.gputrace" "$OUT/$id.replay" > "$OUT/$id.replay.log" 2>&1 || { echo "$id: replay FAILED"; continue; }
  python3 "$STATS" --all "$OUT/$id.replay/streamData" > "$OUT/$id.stats.txt" 2>&1
  "$PY" "$B/perf/shaderprof-table.py" "$OUT/$id.replay/raw" --top 5 --json "$OUT/$id.instr.json" > "$OUT/$id.instr.txt" 2>&1
  rm -rf "$OUT/$id.replay/raw" /tmp/com.apple.gputools.profiling
  echo "== $id"; grep -E "Instruction count|ALU instruction|FP16 instr|INT32 instr|INT16 instr|Device load|Temporary register|Spilled" "$OUT/$id.stats.txt" | head -9
  python3 - "$OUT/$id.instr.json" <<'PY'
import json,sys
ks=[k for k in json.load(open(sys.argv[1])) if k.get('role')=='main']
for k in ks: print('   executed per dispatch: %.2f M  (%s, dispatches %s)'%(k['executed_total']/max(1,k['dispatches'])/1e6, k['kernel'][:60], k['dispatches']))
PY
done
