#!/bin/bash
# Interleaved e2e A/B of the LLAMA_SPEC_EV controller against fixed depth 3 on the corpus
# (perf/spec-verify-narrow.md section 7). Arms per prompt, in order: fixed n3 | ev hybrid | ev full
# | fixed n3 again (drift check). Everything else = the Turbo4 pick via run-depth-corpus.sh.
# Uses the ACTIVE tree's build (B=), which is prod + the controller. 300 tokens, LV=0. LINE=q4|ud picks the line.
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi
B=${B:-/Users/troff/play/llama.cpp-active}
TAG=${TAG:-specev-ab-$(date +%m%d-%H%M)}
NPRED=${NPRED:-300}
PROMPTS=${PROMPTS:-"benchprompt 01-code-explain 02-prose-creative 03-chat-support 04-math-derivation 05-json-boilerplate 06-algorithms 08-story"}
ARMS=${ARMS:-"n3 hybrid full n3b"}
export B NPRED LV=${LV:-0} KV=${KV:-turbo4} LINE=${LINE:-q4}
cd "$B"
for p in $PROMPTS; do
    for arm in $ARMS; do
        case $arm in
            n3|n3b) env -u LLAMA_SPEC_EV TAG=$TAG-$arm DEPTHS=3 PROMPTS=$p perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' | sed "s/^/[$arm] /" ;;
            hybrid) LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=hybrid TAG=$TAG-$arm DEPTHS=7 PROMPTS=$p perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' | sed "s/^/[$arm] /" ;;
            full)   LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=full   TAG=$TAG-$arm DEPTHS=7 PROMPTS=$p perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' | sed "s/^/[$arm] /" ;;
            tiered) LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_BLOCK=tiered TAG=$TAG-$arm DEPTHS=7 PROMPTS=$p perf/run-depth-corpus.sh 2>&1 | grep -E '^\[n|ABORT|died' | sed "s/^/[$arm] /" ;;
        esac
    done
done
echo AB-COMPLETE
