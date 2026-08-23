#!/bin/bash
# Capture what Xcode ACTUALLY does when it launches the replay service, at selector level.
#
# The open question (perf/headless-replay-probe.md): is `launchReplayService:` refusing an
# unentitled caller as policy, or is it failing because something is missing? Our attempt
# dies with a transport-shaped `Code=7 "Connection interrupted"`, nothing is logged anywhere,
# the agent has no launch constraints, and MTLReplayerTrampoline.app is not present on this
# system. Those are the fingerprints of a broken launch, not a denial - but it was written up
# as a security boundary, and that was an inference from an outcome.
#
# The decisive evidence is Xcode's own successful path. GPUTraceReplayController, in
# GPUDebugger.ideplugin, is the class that drives it (-setupAndStartReplayer:,
# -initiateReplaySession:, -postReplayMessage:withFuture:). The plugin runs INSIDE Xcode's
# process, so tracing Xcode traces it.
#
# Method: NSObjCMessageLoggingEnabled=YES makes libobjc log every message send to
# /tmp/msgSends-<pid>. Verified working on this machine 2026-08-23. It needs no injection,
# which matters because Xcode is signed with library-validation (flags=0x2000) and
# DYLD_INSERT_LIBRARIES is therefore refused.
#
# LIMITS, so the output is not over-read:
#   - Selectors only. No arguments, no return values, no dictionary keys. This answers
#     "which code path", never "which value".
#   - It cannot show file access, so it cannot by itself prove or disprove the trampoline
#     theory. Run the fs_usage line printed at the end for that.
#
# Usage: replay-trace-capture.sh [trace.gputrace]      (Xcode must be QUIT first)
set -u

B=/Users/troff/play/llama.cpp-prod
TRACES=/Users/troff/play/kvquant-experiments/traces/aug23
OUT=${OUT:-$TRACES/replay-trace}
PROF=/tmp/com.apple.gputools.profiling
TRACE=${1:-$TRACES/w4-ffn_down-ext-nx8.gputrace}
LOOKBACK=${LOOKBACK:-20}     # seconds of log to keep from before the replay was detected

mkdir -p "$OUT"

if pgrep -x Xcode >/dev/null; then
    echo "Xcode is running (pid $(pgrep -x Xcode | tr '\n' ' '))."
    echo "The env var can only be set at launch, so quit Xcode and re-run this."
    exit 1
fi
[ -d "$TRACE" ] || { echo "no such trace: $TRACE"; exit 1; }

echo "=== replay trace capture ==="
echo "trace : $TRACE"
echo "out   : $OUT"
echo

before=$(ls -1 "$PROF" 2>/dev/null | wc -l | tr -d ' ')
echo "replay dirs already present: $before"

# Exec the binary directly, NOT `open`: open hands off to LaunchServices, which does not
# inherit this shell's environment, so the var would silently never reach Xcode.
echo "launching Xcode with message logging..."
NSObjCMessageLoggingEnabled=YES /Applications/Xcode.app/Contents/MacOS/Xcode >/dev/null 2>&1 &
for i in $(seq 1 60); do
    XPID=$(pgrep -x Xcode | head -1)
    [ -n "$XPID" ] && break
    sleep 1
done
[ -n "${XPID:-}" ] || { echo "Xcode did not start"; exit 1; }
LOG=/tmp/msgSends-$XPID
echo "Xcode pid $XPID, log $LOG"

for i in $(seq 1 30); do [ -f "$LOG" ] && break; sleep 1; done
[ -f "$LOG" ] || { echo "no msgSends log appeared - the env var did not take"; exit 1; }
echo "message logging is live ($(du -h "$LOG" | cut -f1) so far)"

sleep 5
open "$TRACE"
echo
echo "--------------------------------------------------------------"
echo "  Xcode is open on the trace. Click \"Profile GPU Trace\" now."
echo "  This will sit here watching $PROF and snapshot the log"
echo "  the moment replay output appears. Ctrl-C to give up."
echo "--------------------------------------------------------------"
echo

# ring buffer of (elapsed, logsize) so we can slice back to before the click
: > "$OUT/.sizes"
start=$(date +%s)
while true; do
    now=$(( $(date +%s) - start ))
    sz=$(stat -f%z "$LOG" 2>/dev/null || echo 0)
    echo "$now $sz" >> "$OUT/.sizes"
    cur=$(ls -1 "$PROF" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$cur" -gt "$before" ]; then
        echo "replay output detected at t=${now}s (dirs $before -> $cur)"
        break
    fi
    sleep 1
done

# let the replay finish writing, then snapshot before /tmp can lose any of it
echo "waiting for the replay to settle..."
prev=-1
for i in $(seq 1 120); do
    n=$(find "$PROF" -name streamData 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "$prev" ] && [ "$n" != "0" ] && break
    prev=$n; sleep 2
done

END=$(stat -f%z "$LOG" 2>/dev/null || echo 0)
CUT=$(awk -v t="$(( $(date +%s) - start - LOOKBACK ))" '$1<=t{s=$2} END{print s+0}' "$OUT/.sizes")
echo "log is ${END} bytes; slicing from ${CUT} (t-${LOOKBACK}s)"

cp "$LOG" "$OUT/msgSends-full.log" 2>/dev/null
tail -c +$((CUT + 1)) "$LOG" > "$OUT/msgSends-click.log" 2>/dev/null
echo "  full  : $(du -h "$OUT/msgSends-full.log" 2>/dev/null | cut -f1)"
echo "  click : $(du -h "$OUT/msgSends-click.log" 2>/dev/null | cut -f1)"

echo
echo "=== GPUTraceReplayController and the launch chain, in order ==="
grep -nE "GPUTraceReplayController|GTLaunchService|GTMTLReplayService|GTServiceProvider|GTLocalXPCConnection|DYXPCTransport|launchReplayService|setupAndStartReplayer|initiateReplaySession|ReplayerTrampoline" \
    "$OUT/msgSends-click.log" 2>/dev/null | head -120 | tee "$OUT/replay-chain.txt"

echo
echo "=== distinct selectors on the replay classes ==="
awk '{print $2, $4}' "$OUT/msgSends-click.log" 2>/dev/null \
  | grep -E "^(GPUTraceReplayController|GTLaunchServiceXPCProxy|GTMTLReplayServiceXPCProxy|GTServiceProviderXPCProxy|GTLocalXPCConnection)" \
  | sort | uniq -c | sort -rn | head -40 | tee "$OUT/replay-selectors.txt"

echo
echo "archived to $OUT (out of /tmp, which does not survive)"
echo
echo "For the trampoline question, which selectors cannot answer, run this during a"
echo "second click and grep for MTLReplayerTrampoline:"
echo "  sudo fs_usage -w -f pathname \$(pgrep -x Xcode) | tee $OUT/fs_usage.log"
