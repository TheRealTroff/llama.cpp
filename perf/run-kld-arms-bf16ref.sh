#!/bin/bash
# The same arms scored against the bf16 as-trained reference written on the Spark (ref_logits.py, window 32).
# The base file must already sit at logits/kld-base-kld-bf16ref-sep08.dat (scp from the Spark); the harness reuses it.
cd /Users/troff/play/llama.cpp-prod/perf || exit 1
export B=/Users/troff/play/llama.cpp-kldref TAG=kld-bf16ref-sep08 LLAMA_KLD_FLOOR=-32
run_arm() {  # run_arm <line> <kv:f16|turbo4> <model>
  local line=$1 kv=$2 model=$3
  ( source ./pick.sh >/dev/null 2>&1
    if [ "$kv" = f16 ]; then pick_env "$line" f16; else pick_env "$line"; fi
    export "${PICK_ENV[@]}"
    export KV=$kv LABEL="-${line}-${kv}pick"
    echo "=== arm $line $kv: ${PICK_ENV[*]}"
    ./run-quant-kld.sh "$model" )
}
Q4=/Users/troff/play/Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf
# 1. q8_0 itself as a TEST: the calibration number (how far the q8_0 reference sits from the trained model)
#    under a clean env, f16 cache
( export KV=f16 LABEL=-vs-bf16; echo "=== q8_0 vs bf16 (clean env)"; ./run-quant-kld.sh /Users/troff/play/Qwen3.8-27B-conv-q8_0.gguf )
UD=/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf
run_arm q4 f16 $Q4
run_arm ud f16 $UD
run_arm q4 turbo4 $Q4
run_arm ud turbo4 $UD
echo ARMS_DONE
