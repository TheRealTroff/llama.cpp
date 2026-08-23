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
set -u

B=/Users/troff/play/llama.cpp-prod
PY=python3
ROOT=/tmp/com.apple.gputools.profiling
DEST=${DEST:-/Users/troff/play/kvquant-experiments/traces/aug23/replays}
mkdir -p "$DEST"

declare -A seen sized

echo "watching $ROOT -> $DEST (click 'Profile GPU Trace' in Xcode; ctrl-c to stop)"

while true; do
    # both nesting depths the stats reader knows about
    for sd in "$ROOT"/*.gpuprofiler_raw/streamData "$ROOT"/*/*.gpuprofiler_raw/streamData; do
        [ -f "$sd" ] || continue
        [ -n "${seen[$sd]:-}" ] && continue

        # the skill's oscillation gotcha: wait for the file to stop growing before reading
        sz=$(stat -f%z "$sd" 2>/dev/null || echo 0)
        if [ "$sz" = "0" ] || [ "${sized[$sd]:-}" != "$sz" ]; then
            sized[$sd]=$sz
            continue
        fi

        # traceName comes out of the archive itself, so the replay names its own capture
        dump=$("$PY" "$B/perf/gpuprofiler-stats.py" --all "$sd" 2>/dev/null)
        [ -z "$dump" ] && { sized[$sd]=""; continue; }

        name=$(echo "$dump" | sed -n 's/^trace:  *//p' | head -1 \
               | sed 's/\.gputrace$//' | tr -c 'A-Za-z0-9._-' '_' | sed 's/_*$//')
        [ -z "$name" ] && name="unknown"

        out="$DEST/$name"
        i=1; while [ -e "$out" ]; do out="$DEST/$name.$i"; i=$((i+1)); done
        mkdir -p "$out"

        # the raw archive, so a later session can re-read fields nobody thought to print
        cp -R "$(dirname "$sd")" "$out/" 2>/dev/null
        echo "$dump" > "$out/stats-all.txt"
        "$PY" "$B/perf/gpuprofiler-stats.py" "$sd" > "$out/stats-summary.txt" 2>/dev/null

        seen[$sd]=1
        k=$(grep -c '^=== ' "$out/stats-all.txt" 2>/dev/null || echo 0)
        echo "archived: $name  ($k pipelines, $(du -sh "$out" | cut -f1)) -> $out"
    done
    sleep 5
done
