#!/bin/bash
# perf/run-census-full.sh - the FULL kernel census of the current pick (2026-10-24, owner: "let's start by getting a full
# census"): one profiled run per line x verify depth (run-w8-decomp.sh STEPS=metalprof, manifest env, Turbo4, fixed depth so
# every round has the width under test), then kernel-census.sh over each log - a decode census per width and a prefill census
# per line - diffed against PREV snapshots (default: the Sep 18 width-8 censuses, the last ones with this row layout).
#   TAG=census-<mmdd> LINES="ud q4" DEPTHS="7 3" PORT=8097 PREV_DIR_SUFFIX=w8-sep18 perf/run-census-full.sh
# Needs the GPU alone: stop the serving servers first (perf/serve.sh <line> stop). Detach it (nohup ... & disown): hours.
set -u
export B=${B:-/Users/troff/play/llama.cpp-prod}
export CENSUS_KV=${CENSUS_KV:-turbo4}   # the FA rows match perf cases of THIS cache type; without it the census times the f16 FA kernel while the pick runs Turbo4 (the Oct 24 run did - re-run its FA rows)
TAG=${TAG:-census-$(date +%m%d)}
LINES=${LINES:-ud q4}; DEPTHS=${DEPTHS:-7 3}; PORT=${PORT:-8097}
PREV_DIR_SUFFIX=${PREV_DIR_SUFFIX:-w8-sep18}
TOP_DEC=${TOP_DEC:-16}; TOP_PRE=${TOP_PRE:-12}
RES=/Users/troff/play/kvquant-experiments/results
CEN=/Users/troff/play/kvquant-experiments/census
mkdir -p "$CEN"
echo "=== full census $TAG: lines [$LINES] depths [$DEPTHS] port $PORT  $(date)"
echo "commit : $(git -C "$B" rev-parse --short HEAD) ($(git -C "$B" status --porcelain | wc -l | tr -d ' ') files dirty)  binary: $(date -r "$B/build/bin/llama-server" '+%Y-%m-%d %H:%M')"
echo "toolchain: $(xcrun metal --version 2>&1 | head -1)"
if lsof -ti :8093 -ti :8094 >/dev/null 2>&1; then echo "WARNING: a serving server is up on 8093/8094 - it shares the GPU with every timing below"; fi

echo; echo "### 1. profiled runs"
TAG=$TAG LINES="$LINES" DEPTHS="$DEPTHS" STEPS=metalprof PORT=$PORT KV=turbo4 NPRED=300 "$B/perf/run-w8-decomp.sh" || echo "profiled runs: exit $?"

echo; echo "### 2. censuses"
source "$B/perf/pick.sh"
for line in $LINES; do
  pick_env "$line" turbo4; ENVS="${PICK_ENV[*]}"
  prev=$CEN/census-$line-$PREV_DIR_SUFFIX/snapshot.json; [ -f "$prev" ] || prev=
  for depth in $DEPTHS; do
    log=$RES/$TAG-$line-d$depth-metalprof.server.log; w=$((depth+1))
    [ -s "$log" ] || { echo "$line d$depth: no profiled log $log"; continue; }
    echo; echo "--- $line width $w decode census ($(date +%H:%M)) ---"
    ENVS="$ENVS" TOP=$TOP_DEC PHASE=decode PREV=$prev "$B/perf/kernel-census.sh" "$log" "$TAG-$line-w$w-dec" 2>&1 | tail -40
    if [ "$depth" = "$(echo $DEPTHS | awk '{print $1}')" ]; then
      echo; echo "--- $line prefill census ($(date +%H:%M)) ---"
      ENVS="$ENVS" TOP=$TOP_PRE PHASE=prefill PREV=$prev "$B/perf/kernel-census.sh" "$log" "$TAG-$line-pre" 2>&1 | tail -40
    fi
  done
done
echo; echo "=== full census $TAG done $(date): $CEN/$TAG-*"
