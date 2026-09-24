#!/bin/bash
# One-stream per-call timing of the width-1/2 decode FA routes (perf/per-slot-ctx.md, 2026-09-24 evening): the vec kernel
# (GGML_FA_VEC_MAX=3 sends widths 1-2 there) vs the GQA tile at every extent (GGML_FA_GQA_WMIN=1), per line (q4 TR 7 /
# ud TR 9) and per cache type (turbo4, f16), kv 8448 and 102400 (the shapes test-backend-ops has at widths 1-2),
# interleaved reps. Prints the per-arm means, the run-to-run spread and the tile/vec ratio. Run under bash.
#   B=<tree> TYPES="turbo4 f16" LINES="q4 ud" REPS=3 perf/run-fa-w12-timing.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin/test-backend-ops}
TYPES=${TYPES:-"turbo4 f16"}; LINES=${LINES:-"q4 ud"}; REPS=${REPS:-3}
ARMS=${ARMS:-"vec: gqa:GGML_FA_GQA_WMIN=1"}
source "$B/perf/pick.sh"
LOG=${LOG:-/Users/troff/play/kvquant-experiments/results/fa-w12-timing-$(date +%m%d-%H%M).log}
echo "start $(date '+%F %T') binary $(date -r "$BIN" '+%F %H:%M') log $LOG"
for rep in $(seq 1 "$REPS"); do for type in $TYPES; do for line in $LINES; do
  pick_env "$line" "$type"
  FILT="hsk=256,hsv=256,nh=4,nr23=\\[6,1\\],kv=(8448|102400),nb=(1|2),mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=$type,"
  for arm in $ARMS; do
    IFS=: read -r label envs <<<"$arm"; envs=${envs//,/ }
    (cd "$B" && env "${PICK_ENV[@]}" $envs "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "$FILT" 2>&1) | sed -E 's/\x1b\[[0-9;]*m//g' \
      | awk -v L="$type $line $label rep$rep" '/FLASH_ATTN_EXT\(/ { match($0, /kv=[0-9]+,nb=[0-9]+/); c = substr($0, RSTART, RLENGTH); sub(/,/, " ", c) }
          /us\/run/ { match($0, /[0-9.]+ us\/run/); print L, c, substr($0, RSTART, RLENGTH - 7), "us" }
          /loaded kernel_flash_attn_ext_/ { match($0, /kernel_flash_attn_ext_[A-Za-z0-9_=,\[\]]+/); print "  " L, "pipeline", substr($0, RSTART, RLENGTH) }' | sort -u
  done
done; done; done | tee "$LOG"
python3 - "$LOG" <<'PY'
import re, sys, collections
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r'(\S+) (\S+) (\S+) rep(\d+) kv=(\d+) nb=(\d+) ([\d.]+) us', l)
    if m: d[(m[1], m[2], m[5], m[6], m[3])].append(float(m[7]))
arms = sorted({k[4] for k in d}, key=lambda a: a != 'vec')
print(f"\n{'type':6} {'line':4} {'kv':>6} {'w':>2} | " + " | ".join(f"{a:>14}" for a in arms) + " | " + " ".join(f"{a}/vec" for a in arms[1:]))
for (t, ln, kv, nb) in sorted({k[:4] for k in d}, key=lambda k: (k[0], k[1], int(k[2]), int(k[3]))):
    v = {a: d.get((t, ln, kv, nb, a), []) for a in arms}
    mean = {a: sum(x)/len(x) if x else float('nan') for a, x in v.items()}
    spr = {a: (max(x)-min(x))/mean[a]*100 if len(x) > 1 else 0 for a, x in v.items()}
    print(f"{t:6} {ln:4} {kv:>6} {nb:>2} | " + " | ".join(f"{mean[a]:8.1f} ±{spr[a]:3.0f}%" for a in arms) +
          " | " + " ".join(f"{mean[a]/mean['vec']:6.2f}x" for a in arms[1:]))
PY
echo "SWEEP DONE $(date '+%F %T')"
