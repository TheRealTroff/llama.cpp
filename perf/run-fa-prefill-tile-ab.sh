#!/bin/bash
# perf/run-fa-prefill-tile-ab.sh - isolated per-call timing of the prefill FA tile forms at a long KV case (2026-10-25,
# perf/fa-prefill-200k.md): the pick's 16-row Turbo4 tile (qt16w), the 8-row Turbo4 tile (GGML_FA_Q16=0 -> qtl4w), and the
# f16 twins (no dequant: the dequant tax). test-backend-ops perf under the ud pick env, like kernel-census.sh's timing step
# (GGML_MV_REPACK=2, -b MTL0), REPS reps per arm, the kernel name from the 'loaded kernel_' line. Warm-cache, one op in a
# loop: a per-call ratio, not a wall number (README methodology rule 3).
#   BIN=<build/bin> B=<tree> KVS="98304 8448" NB=512 REPS=2 ARMS="base or"  (or = the O-resident forms, this branch's build)
set -u
B=${B:-/Users/troff/play/llama.cpp-fa200k}
BIN=${BIN:-/Users/troff/play/llama.cpp-prod/build/bin}
KVS=${KVS:-98304 8448}; NB=${NB:-512}; REPS=${REPS:-2}
export PICK_SPEC_EV=0
source "$B/perf/pick.sh"; pick_env ud turbo4
echo "=== prefill FA tile forms, isolated (binary $(date -r "$BIN/test-backend-ops" '+%m-%d %H:%M'))"
run() {  # run <label> <type> [ENV=VAL ...]
  local label=$1 typ=$2; shift 2
  local filt="kv=$kv,nb=$NB,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=$typ,type_V=$typ,"
  local out; out=$(for r in $(seq 1 "$REPS"); do (cd "$B" && env GGML_MV_REPACK=2 "${PICK_ENV[@]}" "$@" "$BIN/test-backend-ops" perf -o FLASH_ATTN_EXT -b MTL0 -p "$filt" 2>&1) | grep -E 'us/run|loaded kernel_flash'; done)
  local k; k=$(echo "$out" | grep -oE 'loaded kernel_flash_attn_ext_[A-Za-z0-9_]+' | grep -v '_blk' | head -1 | sed 's/loaded //')
  local us; us=$(echo "$out" | grep -oE '[0-9.]+ us/run' | awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n}')
  local tf; tf=$(echo "$out" | grep -oE '[0-9.]+ TFLOPS' | awk '{s+=$1; n++} END {if (n) printf "%.2f", s/n}')
  printf "  kv=%-6s %-22s %-48s %8s us/call  %5s TFLOPS  env=%s\n" "$kv" "$label" "$k" "$us" "$tf" "$*"
}
ARMS=${ARMS:-base or}
for kv in $KVS; do
  case " $ARMS " in *" base "*)
    run turbo4-16row turbo4
    run turbo4-8row  turbo4 GGML_FA_Q16=0
    run f16-16row    f16
    run f16-8row     f16    GGML_FA_Q16=0 ;;
  esac
  case " $ARMS " in *" or "*)   # needs the exp/fa-prefill-200k build as BIN (GGML_FA_Q16_OR route + the qt16wo instantiation)
    run turbo4-16row     turbo4
    run turbo4-16row-OR1 turbo4 GGML_FA_Q16_OR=1
    run turbo4-16row-OR2 turbo4 GGML_FA_Q16_OR=2
    run turbo4-16row     turbo4 ;;
  esac
done
