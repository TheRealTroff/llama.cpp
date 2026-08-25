#!/bin/bash
# Archive GPU replay output the moment it appears, before /tmp eats it.
#
# Xcode writes replay statistics to /tmp/com.apple.gputools.profiling when you click
# "Profile GPU Trace" (skills/metal-gpu-profile step 2). On 2026-08-23 the previous
# session's entire replay output was gone by morning - only eight fields that had been
# transcribed by hand into perf/width4-verify.md survived, and `--all` had never been run.
# This watcher makes that unrepeatable: every replay is copied out of /tmp and dumped both
# ways as soon as it lands.
#
# Matching is automatic: the archive records its own `traceName`, so a replay names the
# capture it came from and the clicks can happen in any order.
#
# Run it in the background, then click through the traces. One line per archived replay.
#
# NOTE: macOS ships bash 3.2, which has no associative arrays. State lives in files under
# $DEST/.state instead - `declare -A` here exits the script instantly under `set -u`, which
# is exactly what happened the first time this was run.
set -u

B=/Users/troff/play/llama.cpp-prod
PY=python3
ROOT=/tmp/com.apple.gputools.profiling
DEST=${DEST:-/Users/troff/play/kvquant-experiments/traces/aug23/replays}
STATE=$DEST/.state
mkdir -p "$DEST" "$STATE"

echo "watching $ROOT -> $DEST (click 'Profile GPU Trace' in Xcode; ctrl-c to stop)"

while true; do
    # both nesting depths the stats reader knows about
    for sd in "$ROOT"/*.gpuprofiler_raw/streamData "$ROOT"/*/*.gpuprofiler_raw/streamData; do
        [ -f "$sd" ] || continue

        key=$(echo "$sd" | shasum | cut -c1-16)
        [ -e "$STATE/$key.done" ] && continue

        # the skill's oscillation gotcha: wait for the file to stop growing before reading
        sz=$(stat -f%z "$sd" 2>/dev/null || echo 0)
        prev=$(cat "$STATE/$key.size" 2>/dev/null || echo "")
        echo "$sz" > "$STATE/$key.size"
        { [ "$sz" = "0" ] || [ "$prev" != "$sz" ]; } && continue

        # traceName comes out of the archive itself, so the replay names its own capture
        dump=$("$PY" "$B/perf/gpuprofiler-stats.py" --all "$sd" 2>/dev/null)
        [ -z "$dump" ] && continue

        name=$(echo "$dump" | sed -n 's/^trace:  *//p' | head -1 \
               | sed 's/\.gputrace$//' | tr -c 'A-Za-z0-9._-' '_' | sed 's/_*$//')
        [ -z "$name" ] && name="unknown"

        # Xcode writes the SAME replay twice - ROOT/<name>_stream.gpuprofiler_raw and
        # ROOT/gtshaderprofiler/<name>.gputrace.gpuprofiler_raw - so dedup on traceName,
        # not on path, or every click archives two identical copies.
        out="$DEST/$name"
        if [ -e "$out/stats-all.txt" ]; then
            touch "$STATE/$key.done"
            continue
        fi
        mkdir -p "$out"

        # streamData ONLY (~24 MB). The rest of the .gpuprofiler_raw dir is Profiling_f_*.raw
        # frame data, ~1 GB per replay, which gpuprofiler-stats.py never reads. Copying the
        # whole dir cost 7.9 GB in the first four clicks.
        cp "$sd" "$out/streamData" 2>/dev/null
        echo "$dump" > "$out/stats-all.txt"
        "$PY" "$B/perf/gpuprofiler-stats.py" "$sd" > "$out/stats-summary.txt" 2>/dev/null

        touch "$STATE/$key.done"
        k=$(grep -c '^=== ' "$out/stats-all.txt" 2>/dev/null || echo 0)
        echo "archived: $name  ($k pipelines, $(du -sh "$out" | cut -f1 | tr -d ' ')) -> $out"
    done
    sleep 5
done
