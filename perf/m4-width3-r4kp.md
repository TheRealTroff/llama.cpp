# M4 width-3 SoA r4kp

Measured 2026-08-30 on M4 Pro from committed base `301c0707a`, branch
`exp/mv-w3-r4kp`. The experiment is uncommitted.

## Result

The dedicated 4-row x 3-column, two-simdgroup K-split Q4_0 SoA kernel closes the
width-3 decode hole. It uses the width-4 r4kp v3 source form: signed integer indexing,
hoisted planar row pointers, half products, FP32 accumulators, and one final K-split
reduction. The route is behind `GGML_MV_SOA_W3=1` and requires `GGML_MV_REPACK=1`.

Route proof on `m=5120,n=3,k=17408` named
`kernel_mul_mv_q4_0_soa_w3_r4kp_v3`. The exact six real projections in
`perf/w3-real-projections.ops` pass 6/6 against the CPU with test-only repack mode:

```sh
GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 GGML_MV_SOA_W3=1 \
  ./build/bin/test-backend-ops test -o MUL_MAT -b MTL0 \
  --test-file perf/w3-real-projections.ops
```

The ordinary `test` case list does not contain these large real shapes. A parameter
filter against them reports a vacuous 0/0 pass; use the exported-op fixture above.

## Synthetic projection A/B

Three separate-process observations per arm, interleaved. Medians in microseconds per
call, `GGML_MV_REPACK=2`, width 3:

| projection | incumbent ext-di | w3 r4kp | delta |
|---|---:|---:|---:|
| ffn_gate/up | 298.74 | 210.51 | -29.5% |
| ffn_down | 328.64 | 231.04 | -29.7% |
| attn_output/ssm_out | 110.35 | 78.65 | -28.7% |
| attn_qkv | 176.97 | 129.81 | -26.7% |
| attn_gate | 108.07 | 77.80 | -28.0% |
| attn_q | 210.33 | 153.10 | -27.2% |

Weighted by the tagged calls per round (128/64/64/48/48/16), those six projections
fall from 83.38 to 59.18 ms, predicting 24.20 ms saved per width-3 target pass.

## Real model and server

Fresh-process Qwen3.8-27B `llama-bench -n 0 -p 3 -r 5`:

| arm | observations | mean | ms/pass |
|---|---|---:|---:|
| incumbent | 27.94, 28.04 t/s | 27.99 | 107.2 |
| w3 r4kp | 36.89, 37.48 t/s | 37.19 | 80.7 |

The measured 26.5 ms recovery matches the projection-weighted prediction. Width 3 is
no longer dominated by the 84.0 ms width-4 point.

`perf/run-m4-width3-r4kp-e2e.sh` fixes DFlash at depth 2, uses the full production
pick, generates 600 tokens, alternates process order, and records response hashes.
Results are in
`kvquant-experiments/results/m4-w3-r4kp-e2e-0830-0301.tsv`:

| arm | four runs | mean | acceptance | sha1 |
|---|---|---:|---:|---|
| incumbent | 19.531, 19.633, 19.638, 19.650 | 19.613 t/s | 74.07% | `6678b0507d41` |
| w3 r4kp | 23.522, 23.610, 23.547, 23.531 | 23.553 t/s | 68.18% | `885005326897` |

The candidate is **+20.09% end to end**. Each arm is internally byte-deterministic.
The cross-arm trajectory differs, as did the width-4 geometry variants; the six CPU
comparisons are clean.

## Compiler and hardware profile

Offline prescreen: 2880 text bytes, zero spill. Headless replay of
`/tmp/perf-metal-25075.gputrace` is archived at `/tmp/w3-profile-25075`:

- 61 temporary registers, 20 uniform registers, zero per-thread spill;
- 361 instructions: 315 ALU, 97 FP32, 64 FP16, 43 INT32, 11 device loads;
- per-instruction profile: issue cost 86.79, stall cost 6.92 over 187 dispatches.

The old ext family profile is 73 temporary registers and 453 instructions. The new
kernel's measured mechanism is a leaner scalar/SoA instruction stream, not occupancy
or a speculative source-size claim.

## Adaptive layout order

Repack layout is persistent per weight address. Without a policy pin, a width-2-first
process creates DI weights and locks the later SoA kernels out:

| first-use order | w1 | w2 | w3 | w4 | w5 |
|---|---:|---:|---:|---:|---:|
| SoA first (`3,1,2,4,5`) | 13.44 | 26.76 | 36.82 | 47.52 | 54.94 |
| DI first (`2,1,3,4,5`) | 13.48 | 26.64 | 29.06 | 35.18 | 39.67 |

`GGML_MV_SOA_PIN=1` makes eligible weights choose SoA even if width 1 or 2 consumes
them first. Those bandwidth-bound widths read the original weights, while widths 3-5
reuse SoA. The formerly bad order becomes 13.67/26.67/36.98/47.87/55.00 t/s for
widths 1/2/3/4/5: first-use order is gone.

Widths 6-8 remain a separate family. After SoA is pinned they fall to
48.75/56.73/64.51 t/s from isolated DI rates 55.21/63.02/71.04, about 9-12% down;
all three then cost roughly 123-124 ms per pass and are dominated by width 5. The safe
adaptive frontier is therefore widths 1-5. A skinny kernel consuming the SoA layout
would be required before extending the selector above 5. No such kernel exists in any
local worktree or git ref as of this experiment.
