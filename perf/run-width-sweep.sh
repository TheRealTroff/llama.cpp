#!/bin/bash
# perf/run-width-sweep.sh - the fixed-depth verify-width sweep on the current pick (2026-10-01, owner: "I haven't
# seen a width sweep in a while"). Turbo4 cache at the pick's 100K allocation, the manifest env of each line
# (perf/pick.sh through run-prod-pick.sh, one fresh server per measurement), chat-templated benchprompt:
#   width 1        the no-spec arm (turbo4-b1-300, 300 tokens only)
#   widths 2..8    fixed DFlash depth 1..7 with the controller's flags dropped (PICK_SPEC_EV=0 PICK_DEPTH=d),
#                  600 then 300 tokens (run-prod-pick.sh's arm order)
#   ev             the line's picked LLAMA_SPEC_EV controller (cap PICK_DEPTH_EV, widths {3,7}), 600 then 300
# One discarded warmup per line first (after a reboot the first server pays the pipeline compiles and file
# loads), then PASSES passes, each line in its own block. ~1 h 40 for two passes on both lines: every arm
# prefills the 8.3K-token prompt (~60 s).
#   STAMP=wsweep-<date>  PASSES=2  SWEEP_LINES="q4 ud"  DEPTHS="1 2 3 4 5 6 7"
# SWEEP_LINES, not LINES: zsh owns LINES (a `LINES=ud script` prefix from zsh arrives as the terminal height).
# Round costs per width come from the server logs afterwards (-lv 3 spec-prof: dec_syn_tg = the verify round's
# GPU wait, draft_call = the drafter): $OUT/$STAMP-<line>-d<depth>-p<pass>-turbo4-n3-{300,600}.server.log.
# Run it detached (nohup ... & disown); the log is the table.
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
export B=${B:-/Users/troff/play/llama.cpp-prod}
STAMP=${STAMP:-wsweep-$(date +%m%d-%H%M)}
PASSES=${PASSES:-2}
SWEEP_LINES=${SWEEP_LINES:-q4 ud}
DEPTHS=${DEPTHS:-1 2 3 4 5 6 7}

run() {  # run <line> <tag-suffix> <arms> [ENV=VAL ...]
  local line=$1 sfx=$2 arms=$3; shift 3
  env LINE="$line" TURBO=1 MULTISLOT=0 VISION=0 ARMS="$arms" TAG="$STAMP-$line-$sfx" "$@" \
    bash "$B/perf/run-prod-pick.sh" 2>&1 | grep -E '^\[|ABORT|commit|binary|^env|died|timeout'
}

echo "=== width sweep $STAMP start $(date) ==="
for line in $SWEEP_LINES; do
  echo "--- warmup $line (discarded) ---"
  run "$line" warm "turbo4-n3-300" PICK_SPEC_EV=0 PICK_DEPTH=3
done
for p in $(seq 1 "$PASSES"); do
  for line in $SWEEP_LINES; do
    echo "--- pass $p line $line ---"
    echo "## $line p$p width 1 (no spec)"
    run "$line" "w1-p$p" "turbo4-b1-300"
    for d in $DEPTHS; do
      echo "## $line p$p depth $d (width $((d+1)))"
      run "$line" "d$d-p$p" "turbo4-n3-300 turbo4-n3-600" PICK_SPEC_EV=0 PICK_DEPTH="$d"
    done
    echo "## $line p$p controller {3,7}"
    run "$line" "ev-p$p" "turbo4-n3-300 turbo4-n3-600"
  done
done
echo "=== width sweep done $(date) ==="
