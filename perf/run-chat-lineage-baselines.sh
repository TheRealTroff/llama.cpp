#!/bin/bash
# perf/run-chat-lineage-baselines.sh - the first baselines of the chat-templated benchmark lineage (2026-09-17 evening,
# owner: "we should either start using the chat template when we benchmark or add the role tokens ourselves" -> all
# prompts rendered by pick_prompt, PICK_CHAT=1 default). Sequential: mint q4 (controller pick) and ud, the corpus at the
# pick per line (+ a q4 fixed-3 reference arm), the 96K prompt at the pick per line. TAGs chat-sep17-*.
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=/Users/troff/play/llama.cpp-prod; cd "$B"; export B
stamp() { echo "--- $(date '+%T') $* ---"; }
echo "=== chat-lineage baselines: prod $(git rev-parse --short HEAD), $(date '+%F %T') ==="
for line in q4 ud; do
  stamp "mint $line"; LINE=$line TURBO=1 TAG=prodpick-sep17-chat-$line perf/run-prod-pick.sh 2>&1 | grep --line-buffered -E '^\[|^prompt|ABORT|died|ERROR'
done
stamp "corpus q4 pick (controller)"; LINE=q4 LV=3 DEPTHS=7 TAG=chat-sep17-corpus-q4-pick perf/run-depth-corpus.sh 2>&1 | grep --line-buffered -E '^\[n|ABORT|died'
stamp "corpus q4 fixed 3 (reference)"; PICK_SPEC_EV=0 LINE=q4 LV=3 DEPTHS=3 TAG=chat-sep17-corpus-q4-fixed3 perf/run-depth-corpus.sh 2>&1 | grep --line-buffered -E '^\[n|ABORT|died'
stamp "corpus ud pick (fixed 3)"; LINE=ud LV=3 DEPTHS=3 TAG=chat-sep17-corpus-ud-pick perf/run-depth-corpus.sh 2>&1 | grep --line-buffered -E '^\[n|ABORT|died'
stamp "96K q4 pick"; LINE=q4 KV=turbo4 CTX=102400 NPRED=600 DEPTH=7 TAG=chat-sep17-96k-q4 perf/run-longctx-pick.sh 2>&1 | grep --line-buffered -E '^\[|^prompt|wall|ABORT|died|ERROR'
stamp "96K ud pick"; LINE=ud KV=turbo4 CTX=102400 NPRED=600 DEPTH=3 TAG=chat-sep17-96k-ud perf/run-longctx-pick.sh 2>&1 | grep --line-buffered -E '^\[|^prompt|wall|ABORT|died|ERROR'
echo "=== BASELINES-COMPLETE $(date '+%F %T') ==="
