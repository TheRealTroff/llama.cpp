#!/bin/bash
# perf/pick.sh - the two product lines' pick manifests. SOURCE this; never copy the arrays
# (2026-09-07, owner: "do the split" - the Q4 and UD lines carry different fidelity standards).
#
#   pick_env  <q4|ud> [f16]   -> PICK_ENV  (routing + numerics + cache flags of the line)
#   pick_args <q4|ud> [f16]   -> PICK_ARGS (model, context, cache types), PICK_SPEC (drafter, depth: PICK_DEPTH, or
#                                PICK_DEPTH_EV when the line picks the LLAMA_SPEC_EV controller), PICK_DEPTH_LINE, PICK_MODEL
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
#   NUM-PP/TG  changes both prefill and decode arithmetic; both fidelity questions apply
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
  "GGML_MM_SKINNY_BSPLIT=2|BI|both|pick|spec-verify-narrow.md section 8 (-5% per width-8 round on q4, byte-identical; inert at width 4-5; in both picks since 2026-09-07 - but INERT on ud until 2026-09-18: the generic skinny tile ignored the constant; ported on exp/skinny-gen-bsplit = ud width-8 round -2.9%, sha canonical, w8-decomp-sep18.md)"
  "GGML_MM_SKINNY_Q5K=1|BI|ud|pick|w8-decomp-sep18.md levers 3+4 (the stored q5_K width-6..8 tile's plane folded into the in-place nibble integer: -2% per call, byte-identical; with KQ2 below: ud fixed-width-8 round -3.3%, GPU wait -2.7%, sha canonical, interleaved x2 2026-09-18; inert at width 4; branch exp/iq4xs-lut; owner decides)"
  "GGML_MM_SKINNY_KQ2=1|BI|ud|pick|w8-decomp-sep18.md levers 3+4 (the stored q4_K/q5_K width-6..8 tiles decode the superblock header once for the two tiles of a K-step: -6..-8% per call on both formats, byte-identical; see Q5K above for the e2e; owner decides)"
  "GGML_MM_SKINNY_Q6K=1|BI|ud|pick|w8-decomp-sep18.md the q6_K head (the native q6_K lm_head tile at widths 6-8 reads tile pairs with packed wide loads: -10.7% per head call, ud fixed-width-8 round -0.8%, sha canonical, interleaved x2 2026-09-18; inert at width 4; branch exp/q6k-head; owner decides)"
  "GGML_TOPK_STREAM=1|BI|both|pick|topk-stream.md (PICKED 2026-09-19, owner: "I would pick it" - on battery, mint pending AC; the drafter selector's TOP_K [248320, width] -> 16 as a strip scan + one merge in place of the bitonic block sort + 8-dispatch ladder: 837 -> 67 us at width 4, 1631 -> 116 at width 8; e2e draft_call -0.8 / -1.5 ms, ud +0.6 / +0.9%, q4 +1.0 / +1.9% at depth 3 / 7, shas canonical, ABAB x2 2026-09-19; the 2026-08-28 hold released by the owner; adoption = owner)"
  "GGML_MV_Y16_CVT=1|BI|both|pick|agx-backend-access.md (contiguous cast for the decode f32->f16 activation copy; +2.7% q4 / +2.3% ud e2e ABAB x2, shas identical; owner 2026-09-09: 'bring it in')"
  "GGML_SSM_CONV_WB=1|BI|both|pick|agx-backend-access.md (conv-state carry fused into the decode ssm_conv kernel; +0.2..0.7% alone, stacked with the cast +3.3/+3.1% q4, +2.8/+2.0% ud e2e ABAB x2, shas identical; owner 2026-09-09: 'bring it in')"
  "GGML_FUSE_SMALL=60|BI|both|pick|small-op-fusion.md (gated norm, add+norm, GDN gate chain, conv+carry+silu [8 carry copies since 2026-09-18: the q4 depth-7 pick had silently lost this bit, spec-verify-narrow.md section 11] - no f16 twins; ud +1.6/+1.8% (600/300), q4 +2.1/+2.0% e2e ABAB x2 on the final binary, shas canonical, acceptance identical; the twins (bits 1/2 -> mask 63) stay off: a drafter-side trace divergence under twins + add+norm, open - see the note; owner 2026-09-11: 'bring it into prod')"
  # --- UD SoA routes for the K-quant formats, byte-identical vs the block kernels; the stored file
  #     scored identical to the plain file on every KLD statistic under this env (ud-model.md step 15)
  "GGML_MV_SOA_IQ4XS=5|BI|ud|pick|ud-model.md step 6/13"
  "GGML_MV_SOA_KQ=2|BI|ud|pick|ud-model.md step 6/13"
  # --- decode numerics: half products in the width-4/5 scalar kernels. Same sha as the f32-product arms
  #     on the pick texts. UD line PRICED 2026-09-17 (w6-verify-cliff.md last section): width 4 = the decode-path
  #     base (+0.0003 mean KLD vs bf16), width 5 = 5e-6 / 99.943% same-top pairwise vs it at -b 5. q4 line PRICED
  #     2026-09-17 (q4-decode-kld.md): f32-product w4 vs the half-product base 5e-6 / 99.971%, w5 8e-6 / 99.976%,
  #     +0.00009 mean KLD vs bf16 - the KLD line (prefill-shaped) never saw them; the pairwise decode bases do
  "GGML_MV_SOA_W4_R4KP=3|NUM-TG|both|pick|m4-width4-r4kp.md v3 (half product; ud: the decode base, priced vs bf16; q4: 5e-6 / 99.971% vs its f32 form, +0.00009 vs bf16, q4-decode-kld.md)"
  "GGML_MV_SOA_W5=4|NUM-TG|both|pick|m4-width5-crossover.md w5r4h (ud: 5e-6 / 99.943% pairwise vs the width-4 base 2026-09-17; q4: 8e-6 / 99.976% vs the width-4 base, f32 form 1.3e-5, q4-decode-kld.md)"
  "GGML_MV_SOA_W5_HALF=1|NUM-TG|both|pick|m4-width5-crossover.md w5r4h (ud: 5e-6 / 99.943% pairwise vs the width-4 base 2026-09-17; q4: 8e-6 / 99.976% vs the width-4 base, f32 form 1.3e-5, q4-decode-kld.md)"
  # --- prefill numerics ---
  "GGML_MM_ACC_HALF=1|NUM-PP|q4|pick|kldacch-aug28: mean KLD 0.054->0.060 (+11.8%), same-top -0.86 pt; ON UD -2.51 pt (ud-model.md step 4) - q4 only"
  # --- speculation side ---
  "LLAMA_DRAFT_WINDOW=1024|SPEC|both|pick|draft-sink-window.md (acceptance improves; sha canonical)"
  "LLAMA_SPEC_EV=1|SPEC|q4|pick|spec-verify-narrow.md section 10 (owner 2026-09-17: 'time we flipped the switch'): the expected-value verify-depth controller, hybrid block rule, verify widths {3,7} only (the width-4..6 picks returned nothing on free-form and the width-1..2 picks lose; run-specev-tax.sh); q4 corpus +10.7% (math +25%, JSON +35%, free-form -7..+3%), the decode-kernel union priced pairwise at every width (q4 widths 6-8 = 9e-6 / 99.976%, q4-decode-kld.md); block cap PICK_DEPTH_EV=7; a SPEC lineage: free-form shas fork at the width crossings"
  "LLAMA_SPEC_EV_WIDTHS=3,7|SPEC|q4|pick|spec-verify-narrow.md section 10 (the tax diagnosis: hybrid +7.9% -> widths {3,7} +10.7% on the q4 corpus)"
  "LLAMA_SPEC_EV=1|SPEC|ud|proposed|spec-verify-narrow.md section 10: on the UD line the controller is a wash (widths {3,7}: corpus +1.1%, JSON +13%, math +5%, free-form -3..-7%; hybrid -2..-4%) because the UD width-8 round is 1.72x its width-4 round (178 vs 103 ms; q4 1.42x) - the lever on ud is the width-6..8 verify path, not the controller; owner decides (with LLAMA_SPEC_EV_WIDTHS=3,7)"
  # --- Turbo4 cache line (KV) + its byte-identical FA forms ---
  "TURBO_AUTO_ASYMMETRIC=0|KV|both|pick|turbo4-fa-gqa-reuse.md (symmetric pick)"
  "GGML_FA_GQA_HEADS=4,6|BI|both|pick|turbo4-fa-gqa-reuse.md (Turbo4 GQA tile; hash moved at widths 3-4 = a lineage, not a numerics call)"
  "GGML_FA_GQA4_NWG=6|BI|both|pick|turbo4-fa-gqa-reuse.md"
  "GGML_FA_GQA_W3_NWG=13|BI|both|pick|turbo4-fa-gqa-reuse.md"
  "GGML_FA_TR=9|BI|ud|pick|ud-model.md step 16 D (UD waits for decode-fidelity data; owner 2026-09-08)"
  "GGML_FA_Q24=1|BI|both|pick|fa-decode-tile24.md (the 24-row Turbo4 decode FA tile: the GQA6 route's 6 heads x 4 tokens in one threadgroup, each dequantized K/V tile feeds three query tiles, O register-resident; at the pick's split width (TURBO_NWG=20, the default it inherits) it is BYTE-IDENTICAL end to end - canonical sha reproduced - at -18% per decode FA call at 96K, -17% at 24K, -13% at 8K; 2026-09-16)"
  "GGML_FA_Q24_QR=0|BI|both|pick|fa-decode-tile24.md (no register head on the 24-row tile: three query tiles per dim tile make qr 8 a 144 B spill and slower)"
  "GGML_FA_Q24_ROWS=12|BI|both|pick|fa-decode-tile24.md widths section (owner 2026-09-16: 'bring this to prod', for adaptive speculation) (the tile at every GQA6 width, per-width plan: width 3 = one 24-row tile -17.5% per call at 96K / -14% at 8K, width 5 = two 16-row O-resident tiles -18% / -15%, width 6 = 24 + 16 -9% / -11%; bitwise identical to the pick's routes in both classes, depth-2/4/5 e2e shas equal; inert at the pick's width 4; 2026-09-16)"
  "GGML_FA_TURBO_NWG=20|BI|both|pick|longctx-inventory-sep15.md (split-K width for every Turbo4 batched FA route: -21% per decode FA call at 96K, verify round -6.8% at 96K / -0.8% at 8K; the reduce sums the partials with simd_sum so the Turbo4 shas move = a lineage, kernel numerics unchanged; f16 routes and prefill untouched; 2026-09-15; owner 2026-09-16: 'I'll take the flag' - in the pick, mint prodpick-sep16-nwg20)"
  "GGML_FA_TR=7|NUM-PP/TG|q4|pick|q4-fa-folded-pick.md (owner 2026-09-08: take the faster folded form for q4_0; accuracy mixed, new output lineage)"
  # --- refused / declined, listed so pick_check knows them ---
  "GGML_MM_SKINNY_GEN=6|NUM-TG|ud|pick|w6-verify-cliff.md (the generic skinny MMA tile over the stored SoA rows at verify widths 6-8: round -24% at 8K, -15% at 96K; FIXED 2026-09-16 night - the pipeline had left the stored q4_K reader's exact-scale constant unset (upstream's half-quotient form, 5e-4 pairwise); with it set the tile is the pick's own decode class: 2.5e-5 mean / 99.910% same-top vs the width-4 base = the reader's 99.914; not byte-identical (the half A tile); paired bf16 a wash; owner 2026-09-16 night: 'Pick it' - the depth-5 text = the depth-3 canonical sha 9128633c6cfa)"
  "GGML_FA_TR=6|NUM-TG|none|refused|ud-model.md step 16 C (folded norm, KLD a wash; owner took =9)"
  "GGML_KQ_SOA_EXACT=0|NUM-PP|none|refused|ud-model.md step 15 (=0 is the upstream half-division tile; the exact tile is the default, owner)"
)
# The Turbo4 cache itself (KV class), priced: q4 same-top 98.3% own text / KLD 0.006-0.008 (turbo4-quality.md);
# ud same-top 96.58 -> 95.83 (-0.75 pt), mean KLD 0.0135 -> 0.0173 (ud-model.md step 16). Owner 2026-09-07: on
# for both lines by default; the UD spend is the context it buys.

PICK_MODEL_Q4=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf
PICK_MODEL_UD=/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V2.gguf
PICK_DRAFTER=/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0-SOA-V1.gguf
PICK_DEPTH=${PICK_DEPTH:-3}  # DFlash depth of a fixed-depth line (verify width 4, Turbo4's best width)
PICK_DEPTH_EV=${PICK_DEPTH_EV:-7}  # the block cap of a line that picks the LLAMA_SPEC_EV controller (verify widths up to 8)
# PICK_SPEC_EV=0 leaves the controller's flags out of PICK_ENV and keeps PICK_DEPTH: the fixed-depth arm of a
# controller experiment (run-spec-ev-ab.sh, run-specev-tax.sh, the gate's fixed arms), never a pick measurement.
PICK_CTX_TURBO4=102400
PICK_CTX_F16=10240

# Benchmark prompts are chat-templated since 2026-09-17 evening (owner: "the benchprompt is a clear question and what
# we want is its answer"; perf/benchprompt-framing.md): raw greedy completion of an instruction + material is the
# regime where the instruct model restarts and loops, and the prompt's own positions score as unlikely text.
# pick_prompt <raw file> renders one user turn, thinking off, exactly as the server's /apply-template does for this
# model's template (verified byte-identical, the rendered file through /completion reproduces the chat endpoint's
# text), into $PICK_CHAT_DIR/<basename>, and echoes that path. PICK_CHAT=0 echoes the raw path (the pre-Sep-17 lineage).
PICK_CHAT=${PICK_CHAT:-1}
PICK_CHAT_DIR=${PICK_CHAT_DIR:-/Users/troff/play/kvquant-experiments/data/chat}
pick_prompt() {  # pick_prompt <raw prompt file> -> path of the prompt to send (rendered, or raw under PICK_CHAT=0)
  local raw=$1
  if [ "$PICK_CHAT" = 0 ]; then echo "$raw"; return 0; fi
  mkdir -p "$PICK_CHAT_DIR"
  local out="$PICK_CHAT_DIR/$(basename "$raw")"
  # the template trims the message content (Jinja trim = strip both ends); a file's trailing newline must go too
  { printf '<|im_start|>user\n'; python3 -c "import sys; sys.stdout.write(open(sys.argv[1]).read().strip())" "$raw"; printf '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'; } > "$out"
  echo "$out"
}

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
    case "$flag" in LLAMA_SPEC_EV=*|LLAMA_SPEC_EV_*) [ "${PICK_SPEC_EV:-1}" = 0 ] && continue ;; esac
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
  local depth=$PICK_DEPTH e flag cls lines status
  if [ "${PICK_SPEC_EV:-1}" != 0 ]; then
    for e in "${PICK_MANIFEST[@]}"; do
      IFS='|' read -r flag cls lines status _ <<<"$e"
      [ "$flag" = LLAMA_SPEC_EV=1 ] && [ "$status" = pick ] && _pick_line_has "$lines" "$line" && depth=$PICK_DEPTH_EV
    done
  fi
  PICK_SPEC=(-md "$PICK_DRAFTER" --spec-type draft-dflash --spec-draft-n-max "$depth")
  PICK_DEPTH_LINE=$depth
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
