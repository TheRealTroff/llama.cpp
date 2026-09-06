#!/bin/bash
# Turbo4 FA port (perf/ud-model.md step 16): per-call timing of the Turbo4 batched FA kernels, scratch
# form (GGML_FA_TR unset) vs the TR form (GGML_FA_TR=1), 2 interleaved reps, beside the f16 kernel at the
# same shapes. Decode widths 4/5 (gqah 6, nwg 8) at kv 8448/24576/98304, prefill 512 rows at the same.
set -u
B=${B:-/Users/troff/play/llama.cpp-turbo4-fa}
BIN=$B/build/bin/test-backend-ops
COMMON="GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_FA_GQA_HEADS=4,6 GGML_FA_GQA4_NWG=6 GGML_FA_GQA_W3_NWG=13 TURBO_AUTO_ASYMMETRIC=0 GGML_FA_GQA_F16=1 GGML_FA_QT=1 GGML_FA_QR=8 GGML_FA_Q16=1"
ARMS=${ARMS:-"t4_scratch:turbo4: t4_tr:turbo4:GGML_FA_TR=1 f16:f16:"}
FILT='hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=(8448|24576|98304),nb=(4|5|512),mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=TYPE,'
for rep in 1 2; do for arm in $ARMS; do
    IFS=: read -r label type envs <<<"$arm"; envs=${envs//,/ }
    (cd "$B" && env $COMMON $envs "$BIN" perf -o FLASH_ATTN_EXT -b MTL0 -p "${FILT/TYPE/$type}" 2>&1) | sed -E 's/\x1b\[[0-9;]*m//g' \
      | awk -v L="$label rep$rep" '/FLASH_ATTN_EXT\(/ { match($0, /kv=[0-9]+,nb=[0-9]+/); c = substr($0, RSTART, RLENGTH); sub(/,/, " ", c) }
                                  /us\/run/ { match($0, /[0-9.]+ us\/run/); print L, c, substr($0, RSTART, RLENGTH - 7), "us" }
                                  /loaded kernel_flash_attn_ext_(qt|turbo)/ { match($0, /kernel_flash_attn_ext_[a-z0-9_]+_dk256_dv256/); print "  " L, "pipeline", substr($0, RSTART, RLENGTH) }' | sort -u
done; done
