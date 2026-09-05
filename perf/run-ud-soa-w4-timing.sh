#!/bin/bash
# Same-session timings for the kernels captured by run-ud-soa-profile.sh (the per-instruction
# issue-cost arithmetic needs timing from an UNCAPTURED run). test-backend-ops perf,
# GGML_MV_REPACK=2, interleaved reps, pipeline name read from each invocation's stderr.
set -u
B=${B:-/Users/troff/play/llama.cpp-ud-soa}
BIN=$B/build/bin/test-backend-ops
M=${M:-17408}; K=${K:-5120}; N=${N:-4}; REPS=${REPS:-2}
ARMS=${ARMS:-"iq4xs_v5:iq4_xs:GGML_MV_SOA_IQ4XS=5 q4K_v2:q4_K:GGML_MV_SOA_KQ=2 q5K_v2:q5_K:GGML_MV_SOA_KQ=2 q40_r4kp3:q4_0:GGML_MV_SOA_W4=1,GGML_MV_SOA_W4_R4KP=3,GGML_MV_SOA_PIN=1 iq4xs_ext:iq4_xs: q4K_ext:q4_K: q5K_ext:q5_K:"}
printf '%-11s %-4s %9s  %s\n' arm rep us_run kernel
for rep in $(seq 1 $REPS); do
  for arm in $ARMS; do
    label=${arm%%:*}; rest=${arm#*:}; type=${rest%%:*}; envs=${rest#*:}; envs=${envs//,/ }
    out=$(cd "$B" && env GGML_MV_REPACK=2 $envs "$BIN" perf -o MUL_MAT -b MTL0 -p "type_a=$type,type_b=f32,m=$M,n=$N,k=$K," 2>&1)
    us=$(echo "$out" | grep -oE '[0-9]+\.[0-9]+ us/run' | head -1 | cut -d' ' -f1)
    kn=$(echo "$out" | grep -oE 'loaded kernel_mul_mv_[A-Za-z0-9_=]+' | grep -v cpy | tail -1 | sed 's/loaded //')
    printf '%-11s %-4s %9s  %s\n' "$label" "$rep" "$us" "$kn"
  done
done
