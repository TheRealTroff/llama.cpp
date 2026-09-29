#!/bin/bash
# Per-call timing of the pick's FA routes on the two K/V layouts test-backend-ops can build (perf/kv-layout.md,
# 2026-09-29): head-major (the harness default: each head's stream contiguous, permute=[0,1,2,3] kv_view=1) vs the
# cache's own cell-major layout (one row per cell, heads concatenated: permute=[0,2,1,3] kv_view=0), per line
# (q4 TR 7 / ud TR 9), per cache type, kv 8448 / 24576 / 98304, width 4 (decode) and 512 (prefill), interleaved reps.
#   B=<tree> TYPES="turbo4 f16" LINES="q4 ud" REPS=3 EXTRA="GGML_X=1" perf/run-fa-layout-timing.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin/test-backend-ops}
TYPES=${TYPES:-"turbo4 f16"}; LINES=${LINES:-"q4 ud"}; REPS=${REPS:-3}; EXTRA=${EXTRA:-}
KV=${KV:-"8448|24576|98304"}; NB=${NB:-"4|512"}
LAYOUTS=${LAYOUTS:-"headmajor cellmajor"}   # the K/V layouts to time
ARMS=${ARMS:-"base:"}                        # label:ENV=1,ENV2=2 per arm (interleaved inside each layout)
source "$B/perf/pick.sh"
LOG=${LOG:-/Users/troff/play/kvquant-experiments/results/fa-layout-timing-$(date +%m%d-%H%M).log}
echo "start $(date '+%F %T') binary $(date -r "$BIN" '+%F %H:%M') extra '$EXTRA' log $LOG"
for rep in $(seq 1 "$REPS"); do for type in $TYPES; do for line in $LINES; do
  pick_env "$line" "$type"
  for layout in $LAYOUTS; do for arm in $ARMS; do
    IFS=: read -r alabel envs <<<"$arm"; envs=${envs//,/ }
    if [ $layout = headmajor ]; then PERM='permute=\[0,1,2,3\],kv_view=1'; else PERM='permute=\[0,2,1,3\],kv_view=0'; fi
    FILT="hsk=256,hsv=256,nh=4,nr23=\\[6,1\\],kv=($KV),nb=($NB),mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=$type,type_V=$type,$PERM"
    (cd "$B" && env "${PICK_ENV[@]}" $EXTRA $envs "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "$FILT" 2>&1) | sed -E 's/\x1b\[[0-9;]*m//g' \
      | awk -v L="$type $line $layout-$alabel rep$rep" '/FLASH_ATTN_EXT\(/ { match($0, /kv=[0-9]+,nb=[0-9]+/); c = substr($0, RSTART, RLENGTH); sub(/,/, " ", c) }
          /us\/run/ { match($0, /[0-9.]+ us\/run/); print L, c, substr($0, RSTART, RLENGTH - 7), "us" }
          /fa-route:|loaded kernel_flash_attn_ext_/ { match($0, /kernel_flash_attn_ext_[A-Za-z0-9_=,\[\]]+/); print "  " L, "pipeline", substr($0, RSTART, RLENGTH) }' | sort -u
  done; done
done; done; done | tee "$LOG"
python3 - "$LOG" <<'PY'
import re, sys, collections
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r'(\S+) (\S+) (\S+) rep(\d+) kv=(\d+) nb=(\d+) ([\d.]+) us', l)
    if m: d[(m[1], m[2], m[5], m[6], m[3])].append(float(m[7]))
order = ['headmajor-base', 'cellmajor-base']
arms = sorted({k[4] for k in d}, key=lambda a: (order.index(a) if a in order else 99, a))
print(f"\n{'type':6} {'line':4} {'kv':>6} {'w':>3} | " + " | ".join(f"{a:>18}" for a in arms) + " | " + " ".join(f"{a}/{arms[0]}" for a in arms[1:]))
for (t, ln, kv, nb) in sorted({k[:4] for k in d}, key=lambda k: (k[0], k[1], int(k[2]), int(k[3]))):
    v = {a: d.get((t, ln, kv, nb, a), []) for a in arms}
    mean = {a: sum(x)/len(x) if x else float('nan') for a, x in v.items()}
    spr = {a: (max(x)-min(x))/mean[a]*100 if len(x) > 1 else 0 for a, x in v.items()}
    print(f"{t:6} {ln:4} {kv:>6} {nb:>3} | " + " | ".join(f"{mean[a]:12.1f} ±{spr[a]:3.0f}%" for a in arms) + " | " + " ".join(f"{mean[a]/mean[arms[0]]:6.3f}x" for a in arms[1:]))
PY
echo "SWEEP DONE $(date '+%F %T')"
