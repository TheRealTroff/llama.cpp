#!/bin/bash
# Replay every archived census gputrace headlessly (keeping raw/) and read the resident-simdgroup counter per row
# (2026-10-30). The census pipeline deleted replay/raw after the instruction tables, so the Oct 24 replays carry no
# counter data; the traces on offload do. ~8 s per replay + the SMB read. Output: <OUT>/<snapshot>/<row>.aps.txt
TR=${TR:-/Volumes/offload/play-archive/census-1024-traces}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results/perfmodel-oct30/census-aps}
PY=${PY:-/Users/troff/play/.venv-convert/bin/python3}
HEADLESS=/Users/troff/.claude/skills/metal-gpu-profile/references/metal-profile-headless.py
APS=$(dirname "$0")/aps-usc-values.py
SCR=${SCR:-/private/tmp/claude-501/-Users-troff-play/01d2d942-d821-44c1-a5d2-8a7938e1247c/scratchpad/census-replay}
mkdir -p "$OUT" "$SCR"
for snap in "$TR"/*/; do
  s=$(basename "$snap"); mkdir -p "$OUT/$s"
  for t in "$snap"/*.gputrace; do
    id=$(basename "$t" .gputrace); [ -s "$OUT/$s/$id.aps.txt" ] && continue
    rm -rf "$SCR/r"; python3 "$HEADLESS" "$t" "$SCR/r" > "$OUT/$s/$id.replay.log" 2>&1 || { echo "$s/$id replay FAILED"; continue; }
    DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks "$PY" "$APS" "$SCR/r" > "$OUT/$s/$id.aps.txt" 2>&1
    rm -rf "$SCR/r" /tmp/com.apple.gputools.profiling
    echo "$s/$id done $(date +%H:%M:%S)"
  done
done
echo ALL DONE
