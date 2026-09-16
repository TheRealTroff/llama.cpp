#!/bin/bash
# The 24-row Turbo4 decode FA tile (perf/longctx-inventory-sep15.md, branch exp/fa-decode-tile): per-call timing
# of the GQA6 width-4 decode shape, the pick's 8-row route (GGML_FA_TR=9, GGML_FA_TURBO_NWG=20) against the
# 24-row forms (GGML_FA_Q24=1 staged table / 2 constant table) over their split width and register head,
# interleaved reps. Run under bash (zsh does not word-split the env strings).
#   ARMS="label:envs ..." KVS="98304 24576 8448" REPS=2 perf/run-fa24-timing.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-fa24}
BIN=${BIN:-$B/build/bin/test-backend-ops}
PICK="GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_FA_GQA_HEADS=4,6 GGML_FA_GQA4_NWG=6 GGML_FA_GQA_W3_NWG=13 TURBO_AUTO_ASYMMETRIC=0 GGML_FA_GQA_F16=1 GGML_FA_QT=1 GGML_FA_QR=8 GGML_FA_Q16=1 GGML_FA_TR=9 GGML_FA_TURBO_NWG=20"
ARMS=${ARMS:-"base: q24s_n48_qr0:GGML_FA_Q24=1,GGML_FA_NWG_MAX=64,GGML_FA_Q24_NWG=48,GGML_FA_Q24_QR=0 q24s_n48_qr4:GGML_FA_Q24=1,GGML_FA_NWG_MAX=64,GGML_FA_Q24_NWG=48,GGML_FA_Q24_QR=4 q24s_n48_qr8:GGML_FA_Q24=1,GGML_FA_NWG_MAX=64,GGML_FA_Q24_NWG=48,GGML_FA_Q24_QR=8 q24c_n48_qr0:GGML_FA_Q24=2,GGML_FA_NWG_MAX=64,GGML_FA_Q24_NWG=48,GGML_FA_Q24_QR=0"}
KVS=${KVS:-"98304 24576 8448"}
REPS=${REPS:-2}
NB=${NB:-4}
kvre=$(echo $KVS | sed 's/ /|/g')
FILT="hsk=256,hsv=256,nh=4,nr23=\\[6,1\\],kv=($kvre),nb=($NB),mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=turbo4,"
for rep in $(seq 1 $REPS); do for arm in $ARMS; do
    IFS=: read -r label envs <<<"$arm"; envs=${envs//,/ }
    (cd "$B" && env $PICK $envs "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "$FILT" 2>&1) | sed -E 's/\x1b\[[0-9;]*m//g' \
      | awk -v L="$label rep$rep" '/FLASH_ATTN_EXT\(/ { match($0, /kv=[0-9]+,nb=[0-9]+/); c = substr($0, RSTART, RLENGTH); sub(/,/, " ", c) }
                                  /us\/run/ { match($0, /[0-9.]+ us\/run/); print L, c, substr($0, RSTART, RLENGTH - 7), "us" }
                                  /loaded kernel_flash_attn_ext_(qt|turbo|vec_reduce)/ { match($0, /kernel_flash_attn_ext_[A-Za-z0-9_=,]+/); print "  " L, "pipeline", substr($0, RSTART, RLENGTH) }' | sort -u
done; done
