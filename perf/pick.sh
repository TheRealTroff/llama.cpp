#!/bin/bash
# perf/pick.sh - the two product lines' pick manifests. SOURCE this; never copy the arrays
# (2026-09-07, owner: "do the split" - the Q4 and UD lines carry different fidelity standards).
#
#   pick_env  <q4|ud> [f16]   -> PICK_ENV  (routing + numerics + cache flags of the line)
#   pick_args <q4|ud> [f16]   -> PICK_ARGS (model, context, cache types), PICK_SPEC (drafter, depth), PICK_MODEL
#   pick_check <q4|ud>        -> refuses a NUM/KV-class flag in the process environment that is not in the
#                                line's manifest (PICK_ALLOW_EXTRA=1 downgrades to a warning for experiments)
#   pick_print <q4|ud>        -> the manifest with classes and records, for a harness header
#
# Fidelity classes (what a flag CAN change, not how fast it is):
#   BI      byte-identical vs the kernel/route it replaced: free on both lines
#   SPEC    speculation-side: never changes the greedy argmax chain; forks shas (new lineage per adoption)
#   NUM-PP  prefill numerics: priced by the KLD line (perf/run-quant-kld.sh, q8_0 + f16-cache reference)
#   NUM-TG  decode numerics (half products, tile accumulation order): priced by the agreement / same-top
#           harness on the model's own greedy text, or by a sha gate only - the KLD line does NOT see them
#           (it scores prefill-shaped logits)
#   KV      cache quantization: prices both paths (turbo4-quality.md, ud-model.md step 16)
# Lines: q4 = uniform Q4_0 (willing to trade fidelity for speed, within a stated record), ud = unsloth
# UD-Q4_K_M (closer to the unquantized model: BI/SPEC only, plus what the owner has explicitly taken).
# Both lines run the Turbo4 cache by default (owner 2026-09-07: the context it buys is worth its cost,
# and that spend is already part of the UD budget).
#
# Entry: FLAG=VALUE|class|lines|status|record   (lines: q4, ud, both; status: pick | proposed | refused)

PICK_MANIFEST=(
  # --- routing and kernels, byte-identical ---
  "GGML_MV_NC=2|BI|both|pick|mv-nc-cliff-probe.md"
  "GGML_MM_SKINNY=6|BI|both|pick|skinny-soa.md"
  "GGML_MM_SKINNY_SOA=1|BI|both|pick|skinny-soa.md"
  "GGML_FA_VEC_MAX=3|BI|both|pick|turbo4-filled-100k.md (depth-3 sha rejoins canonical)"
  "GGML_FA_MM_NWG=8|BI|both|pick|flash-attn-mm-split.md"
  "GGML_GDN_FUSE_WB=1|BI|both|pick|gdn-writeback-fusion.md"
  "GGML_MV_REPACK=1|BI|both|pick|repack-inplace.md"
  "GGML_MV_SOA_PIN=1|BI|both|pick|shortk-head.md"
  "GGML_MV_SOA_W3=1|BI|both|pick|m4-width3-r4kp.md"
  "GGML_MV_SOA_W4=1|BI|both|pick|m4-width4-r4kp.md"
  "GGML_MV_SOA_WL_XL=1|BI|both|pick|shortk-head.md"
  "GGML_METAL_GET_MEMCPY=1|BI|both|pick|drafter-graph-count.md"
  "GGML_MM_N64=1|BI|both|pick|ud-model.md step 8"
  "GGML_MM_N64_KMAX=20000|BI|both|pick|ud-model.md step 8"
  "GGML_FA_QT=1|BI|both|pick|ud-model.md step 9"
  "GGML_MM_F16B=1|BI|both|pick|ud-model.md step 10 (byte-identical by construction)"
  "GGML_FA_GQA_F16=1|BI|both|pick|ud-model.md step 11"
  "GGML_FA_QR=8|BI|both|pick|fa-long-context.md"
  "GGML_FA_Q16=1|BI|both|pick|fa-long-context.md"
  "GGML_GDN_NR=4|BI|both|pick|gdn-prefill-scan.md"
  "LLAMA_GDN_REPLAY=1|BI|both|pick|gdn-replay-rollback.md (sha-identical, activation-traced)"
  "DFLASH_FUSED_INJECT=1|BI|both|pick|drafter-graph-count.md"
  "DFLASH_ASYNC_INJECT=1|BI|both|pick|drafter-graph-count.md"
  "GGML_MM_SKINNY_BSPLIT=2|BI|both|pick|spec-verify-narrow.md section 8 (-5% per width-8 round, byte-identical; inert at width 4-5; in both picks since 2026-09-07)"
  # --- UD SoA routes for the K-quant formats, byte-identical vs the block kernels; the stored file
  #     scored identical to the plain file on every KLD statistic under this env (ud-model.md step 15)
  "GGML_MV_SOA_IQ4XS=5|BI|ud|pick|ud-model.md step 6/13"
  "GGML_MV_SOA_KQ=2|BI|ud|pick|ud-model.md step 6/13"
  # --- decode numerics: half products in the width-4/5 scalar kernels. Same sha as the f32-product arms
  #     on the pick texts; NEVER priced by KLD (prefill-shaped) or the agreement harness - OPEN on both lines
  "GGML_MV_SOA_W4_R4KP=3|NUM-TG|both|pick|m4-width4-r4kp.md v3 (half product; sha-only; OPEN: agreement line)"
  "GGML_MV_SOA_W5=4|NUM-TG|both|pick|m4-width5-crossover.md w5r4h (sha-only; OPEN: agreement line)"
  "GGML_MV_SOA_W5_HALF=1|NUM-TG|both|pick|m4-width5-crossover.md w5r4h (sha-only; OPEN: agreement line)"
  # --- prefill numerics ---
  "GGML_MM_ACC_HALF=1|NUM-PP|q4|pick|kldacch-aug28: mean KLD 0.054->0.060 (+11.8%), same-top -0.86 pt; ON UD -2.51 pt (ud-model.md step 4) - q4 only"
  # --- speculation side ---
  "LLAMA_DRAFT_WINDOW=1024|SPEC|both|pick|draft-sink-window.md (acceptance improves; sha canonical)"
  "LLAMA_SPEC_EV=1|SPEC|both|proposed|spec-verify-narrow.md section 7 (+14.4% corpus mean). OWNER: KLD + agreement of its text vs the fixed-depth pick BEFORE any pick - its rounds verify at widths 1-8 and carry the union of the width families' decode numerics; then a new lineage"
  # --- Turbo4 cache line (KV) + its byte-identical FA forms ---
  "TURBO_AUTO_ASYMMETRIC=0|KV|both|pick|turbo4-fa-gqa-reuse.md (symmetric pick)"
  "GGML_FA_GQA_HEADS=4,6|BI|both|pick|turbo4-fa-gqa-reuse.md (Turbo4 GQA tile; hash moved at widths 3-4 = a lineage, not a numerics call)"
  "GGML_FA_GQA4_NWG=6|BI|both|pick|turbo4-fa-gqa-reuse.md"
  "GGML_FA_GQA_W3_NWG=13|BI|both|pick|turbo4-fa-gqa-reuse.md"
  "GGML_FA_TR=9|BI|both|pick|ud-model.md step 16 D (the byte-identical Turbo4 FA form; owner 2026-09-07)"
  # --- refused / declined, listed so pick_check knows them ---
  "GGML_FA_TR=6|NUM-TG|none|refused|ud-model.md step 16 C (folded norm, KLD a wash; owner took =9)"
  "GGML_KQ_SOA_EXACT=0|NUM-PP|none|refused|ud-model.md step 15 (=0 is the upstream half-division tile; the exact tile is the default, owner)"
)
# The Turbo4 cache itself (KV class), priced: q4 same-top 98.3% own text / KLD 0.006-0.008 (turbo4-quality.md);
# ud same-top 96.58 -> 95.83 (-0.75 pt), mean KLD 0.0135 -> 0.0173 (ud-model.md step 16). Owner 2026-09-07: on
# for both lines by default; the UD spend is the context it buys.

PICK_MODEL_Q4=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf
PICK_MODEL_UD=/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf
PICK_DRAFTER=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf
PICK_DEPTH=3            # DFlash depth of both lines (verify width 4, Turbo4's best width)
PICK_CTX_TURBO4=102400
PICK_CTX_F16=10240

_pick_line_has() {  # $1 = entry lines field, $2 = line
  case "$1" in both|"$2") return 0 ;; esac; return 1
}

pick_env() {  # pick_env <q4|ud> [f16]  -> PICK_ENV
  local line=$1 kv=${2:-turbo4} e cls lines status
  PICK_ENV=()
  for e in "${PICK_MANIFEST[@]}"; do
    IFS='|' read -r flag cls lines status _ <<<"$e"
    [ "$status" = pick ] || { [ "$status" = proposed ] && [ "${PICK_PROPOSED:-0}" = 1 ]; } || continue
    _pick_line_has "$lines" "$line" || continue
    if [ "$kv" = f16 ]; then
      case "$flag" in TURBO_AUTO_ASYMMETRIC=*|GGML_FA_GQA_HEADS=*|GGML_FA_GQA4_NWG=*|GGML_FA_GQA_W3_NWG=*|GGML_FA_TR=*) continue ;; esac
    fi
    PICK_ENV+=("$flag")
  done
}

pick_args() {  # pick_args <q4|ud> [f16] -> PICK_MODEL, PICK_ARGS, PICK_SPEC
  local line=$1 kv=${2:-turbo4}
  case "$line" in q4) PICK_MODEL=$PICK_MODEL_Q4 ;; ud) PICK_MODEL=$PICK_MODEL_UD ;; *) echo "pick_args: unknown line $line" >&2; return 1 ;; esac
  if [ "$kv" = f16 ]; then
    PICK_ARGS=(-c "$PICK_CTX_F16" -fa on -ctk f16 -ctv f16)
  else
    PICK_ARGS=(-c "$PICK_CTX_TURBO4" -fa on -ctk turbo4 -ctv turbo4 -ctkd f16 -ctvd f16)
  fi
  PICK_SPEC=(-md "$PICK_DRAFTER" --spec-type draft-dflash --spec-draft-n-max "$PICK_DEPTH")
}

pick_check() {  # pick_check <q4|ud> : the process environment must not carry a NUM/KV flag outside the line's manifest
  local line=$1 e flag cls lines status name val bad=0 known="" allowed="" lineflags="" refused=""
  for e in "${PICK_MANIFEST[@]}"; do
    IFS='|' read -r flag cls lines status _ <<<"$e"
    name=${flag%%=*}; known="$known $name:$cls"
    if [ "$status" = refused ]; then refused="$refused $flag"; continue; fi
    if _pick_line_has "$lines" "$line"; then allowed="$allowed $name"; lineflags="$lineflags $flag"; fi
  done
  while IFS='=' read -r name val; do
    case "$name" in GGML_*|LLAMA_*|DFLASH_*|TURBO_*) ;; *) continue ;; esac
    cls=$(printf '%s\n' $known | grep -m1 "^$name:" | cut -d: -f2)
    if printf '%s\n' $refused | grep -qx "$name=$val"; then
      echo "pick_check($line): $name=$val is a REFUSED form (see the manifest record)" >&2; bad=1
    elif [ -z "$cls" ]; then
      echo "pick_check($line): note: $name=$val is not in any manifest (experiment flag)" >&2
    elif ! printf '%s\n' $allowed | grep -qx "$name"; then
      case "$cls" in
        NUM-*|KV) echo "pick_check($line): $name=$val is class $cls and NOT in the $line manifest" >&2; bad=1 ;;
        *)        echo "pick_check($line): note: $name=$val ($cls) is not in the $line manifest" >&2 ;;
      esac
    elif ! printf '%s\n' $lineflags | grep -qx "$name=$val"; then
      echo "pick_check($line): note: $name=$val differs from the manifest value" >&2
    fi
  done < <(env | grep -E '^(GGML_|LLAMA_|DFLASH_|TURBO_)')
  if [ $bad = 1 ]; then
    if [ "${PICK_ALLOW_EXTRA:-0}" = 1 ]; then echo "pick_check($line): PICK_ALLOW_EXTRA=1, continuing" >&2; return 0; fi
    echo "pick_check($line): REFUSED - unset the flag or set PICK_ALLOW_EXTRA=1 for an experiment" >&2; return 1
  fi
  return 0
}

pick_print() {  # pick_print <q4|ud>
  local line=$1 e flag cls lines status rec
  echo "pick manifest, line $line (model $( [ "$line" = q4 ] && echo "$PICK_MODEL_Q4" || echo "$PICK_MODEL_UD" ), Turbo4 cache, DFlash depth $PICK_DEPTH):"
  for e in "${PICK_MANIFEST[@]}"; do
    IFS='|' read -r flag cls lines status rec <<<"$e"
    _pick_line_has "$lines" "$line" || [ "$status" = refused ] || continue
    printf '  %-28s %-7s %-9s %s\n' "$flag" "$cls" "$status" "$rec"
  done
}
