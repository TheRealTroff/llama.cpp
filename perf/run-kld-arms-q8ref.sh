#!/bin/bash
# The four pick arms scored against a fresh q8_0 reference (TAG kld-q8ref-sep08), so that the
# same arms can be re-scored against the bf16 as-trained reference from the Spark on the same build.
cd /Users/troff/play/llama.cpp-prod/perf || exit 1
export B=/Users/troff/play/llama.cpp-kldref TAG=kld-q8ref-sep08
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
# 1. the reference under a CLEAN env (no pick numerics can touch the q8_0 base); the q8_0 file is
#    scored against itself as the test, which also records the logits file's own floor
( export KV=f16 LABEL=-self; echo "=== reference + self-check (clean env)"; ./run-quant-kld.sh /Users/troff/play/Qwen3.8-27B-conv-q8_0.gguf )
UD=/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf
run_arm q4 f16 $Q4
run_arm ud f16 $UD
run_arm q4 turbo4 $Q4
run_arm ud turbo4 $UD
echo ARMS_DONE
