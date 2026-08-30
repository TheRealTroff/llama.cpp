# M4 width-3 control sweep

Status: **complete negative experiment**, measured 2026-08-30 on an M4 Pro
(20-core GPU), Xcode 26.6 / Metal Toolchain 17.6.109. The experiment branch is
`exp/mv-w3-control-sweep`, based on production commit `9bf57e921`.

## Verdict

The production `kernel_mul_mv_q4_0_soa_w3_r4kp_v3` remains the winner. The sweep
exercised every identified source and below-Metal control: product precision, one/two
K partitions, four/eight output rows, row-major/ki-major arithmetic order, a
four-row-interleaved weight layout, every legal AIR partial-unroll factor, every
coarse row-loop block-order combination, and the native register-budget threshold.

No candidate produced a repeatable model-weighted improvement. Nothing from this
branch should be enabled in production. The negative variants are intentionally kept
and committed so their results remain discoverable if the compiler or hardware changes.

## Baseline construction and measurement

The production kernel computes a 4-row by 3-column output tile. Two simdgroups own
contiguous halves of K. Each lane walks Q4_0 `pack8` units, loads three `half8`
activation vectors, four row scales and four packed weights, forms the dequantized
weight product in half, and accumulates 12 FP32 outputs. It then performs simdgroup
reductions and one threadgroup-memory reduction across the two K partitions.

The common route was:

```sh
GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 \
GGML_MV_SOA_W3=1 GGML_METAL_LOG_LEVEL=2
```

Route proof named `kernel_mul_mv_q4_0_soa_w3_r4kp_v3`. All source variants passed
the six CPU comparisons in `perf/w3-real-projections.ops`; the ordinary generated
test list does not contain these large real shapes. `perf/summarize-w3-sweep.py`
weights the six projection times by 128/64/64/48/48/16 calls per target round.

## Source control matrix

The first screen held the arithmetic order row-major and varied row tile, K split,
and half/FP32 dequant product. Times are microseconds per projection; round time is
the model-weighted sum.

| arm | gate/up | down | attn out | QKV | attn gate | Q | round, ms | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production r4/k2/half | 209.37 | 231.05 | 77.23 | 125.72 | 76.03 | 150.89 | 58.628 | control |
| r4/k2/FP32 | 211.40 | 232.70 | 80.75 | 129.83 | 80.64 | 154.74 | 59.698 | +1.83% |
| r4/k1/half | 218.17 | 236.06 | 78.40 | 132.20 | 77.62 | 157.31 | 60.640 | +3.43% |
| r4/k1/FP32 | 220.09 | 238.45 | 81.59 | 133.02 | 81.11 | 159.71 | 61.488 | +4.88% |
| r8/k2/half | 217.84 | 236.45 | 77.99 | 132.28 | 78.08 | 155.90 | 60.599 | +3.36% |
| r8/k2/FP32 | 218.78 | 235.83 | 81.90 | 131.74 | 81.58 | 158.41 | 61.112 | +4.24% |
| r8/k1/half | 226.70 | 246.36 | 80.80 | 134.27 | 82.19 | 162.90 | 62.952 | +7.38% |
| r8/k1/FP32 | 223.59 | 246.11 | 84.13 | 134.91 | 84.41 | 161.97 | 62.874 | +7.24% |

All cells were spill-free offline. The R8 bodies were roughly twice the native size
of R4 and lost despite amortizing activation traffic across more rows. Removing the
K split lost 3.4-7.4%; occupancy/concurrency mattered more than the final barrier.

Changing the source order so each nibble updated all rows before advancing `ki`
also lost:

| arm | round, ms | delta |
|---|---:|---:|
| production row-major | 58.826 | control |
| r4/k2/half ki-major | 59.736 | +1.55% |
| r4/k2/FP32 ki-major | 62.711 | +6.60% |
| r4/k1/half ki-major | 61.319 | +4.24% |
| r4/k1/FP32 ki-major | 63.426 | +7.82% |

## Four-row-interleaved weight layout

`kernel_repack_q4_0_soa_r4i` stores a `half4` scale vector per block and `uint4`
weight vectors per pack. The matching compute kernel replaces four scalar scale loads
and four scalar packed-weight loads with one vector load of each. Its offline native
body fell from 2880 to 2508 bytes with no per-thread spill.

An eight-process interleaved A/B rejected it overall:

| arm | n | gate/up | down | attn out | QKV | attn gate | Q | round, ms | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 4 | 207.31 | 226.18 | 77.50 | 123.97 | 77.00 | 148.11 | 57.988 | control |
| row-interleaved | 4 | 207.42 | 224.06 | 78.79 | 125.72 | 77.78 | 148.59 | 58.078 | +0.15% |

It repeatedly helps FFN-down by about 0.9% but loses 1-1.7% on attention. It also
requires a distinct persistent repack identity, so the global form is both slower and
unsafe to mix with ordinary SoA consumers. A projection-specific copy would save only
about 0.1-0.2 ms per round and is not justified by these results.

Headless replay explains the miss:

| metric | production | row-interleaved |
|---|---:|---:|
| temporary registers | 61 | 63 |
| uniform registers | 20 | 20 |
| per-thread spill | 0 | 0 |
| instructions | 361 | 320 |
| ALU | 315 | 281 |
| FP32 / FP16 | 97 / 64 | 97 / 64 |
| INT32 | 43 | 21 |
| device loads | 11 | 5 |
| issue cost | 86.79 | 86.19 |
| stall cost | 6.92 | 7.52 |

The vector layout removes loads and integer address work, but issue cost barely moves
and stalls rise. This is another measured case where static instruction count does not
predict latency.

## AIR loop control

Compiling to AIR, printing to LLVM-like text, and reassembling it unchanged produced a
byte-identical metallib (`8e29daba...`). The width-3 function has four hot
eight-iteration loops. Replacing `llvm.loop.unroll.enable` on all four with each legal
partial-unroll count changed the native body and latency as follows:

| unroll | native text, B | spill B/thread | round, ms | delta |
|---:|---:|---:|---:|---:|
| compiler full | 2880 | 0 | 57.598 | control |
| 2 | 2962 | 0 | 136.288 | +136.62% |
| 3 | 3878 | 0 | 156.655 | +171.98% |
| 4 | 3350 | 0 | 95.723 | +66.19% |
| 5 | 3942 | 0 | 106.478 | +84.86% |
| 6 | 4040 | 0 | 94.656 | +64.34% |
| 7 | 4386 | 0 | 90.918 | +57.85% |
| 8 | 2880 | 0 | 57.963 | +0.63% single-run noise |

Count 8 and the compiler baseline have the same 376 structurally decoded instructions
and native stream hash `ef56d0ae...`. Full unrolling is mandatory.

## Native register budget

`AGC_TEMP_REGS_IN_BYTES` found three relevant allocation regimes:

| budget | native text, B | spill B/thread |
|---:|---:|---:|
| 217-228 B | 2868 | 0 |
| 216 B | 2864 | 16 |
| 208-212 B | 2908-2914 | 16 |
| 192 B | 3154 | 80 |
| default / at least 229 B | 2880 | 0 |

The exact translations were deployed with `MTLBinaryArchive` and
`MTLPipelineOptionFailOnBinaryArchiveMiss`, so a successful run proves native archive
selection. The spill-free 217-byte form was +2.2% in the grid. The closest boundary
case, 216 bytes, lost a balanced A/B by 0.24% (58.442 vs 58.300 ms). Forced lower
budgets lost by 1.2-3.6%. There is no occupancy rescue below the compiler allocation.

## AIR block order

Empty side-effect AIR barriers after each of the first three row loops perturb native
block placement without emitting an obvious instruction. All seven combinations were
spill-free and produced distinct native sizes (2864-2890 B).

| fences | round, ms | delta |
|---|---:|---:|
| none | 57.994 | control |
| 1 | 58.379 | +0.66% |
| 2 | 58.254 | +0.45% |
| 3 | 58.541 | +0.94% |
| 1+2 | 58.052 | +0.10% |
| 1+3 | 58.466 | +0.81% |
| 2+3 | 58.688 | +1.20% |
| 1+2+3 | 58.737 | +1.28% |

Fence 1+2 was the only near tie. A longer eight-process A/B resolved it at 58.449 vs
58.430 ms, +0.03%. It is noise, not a winner.

## Final validation and artifacts

The unchanged production route in this worktree measured `36.77 +/- 0.47 t/s` on the
real Qwen3.8-27B Q4_0 model at `llama-bench -n 0 -p 3 -r 5`, consistent with the
established 36.9-37.5 t/s range. No experimental arm advanced to a server E2E A/B:
none passed the balanced projection gate, and the only alternate persistent layout was
globally slower.

Reproduction helpers committed with the branch:

- `perf/run-w3-control-sweep.sh`: source cells, correctness and projection timing;
- `perf/run-w3-air-unroll-sweep.sh`: arbitrary edited AIR metallibs;
- `perf/run-w3-air-register-sweep.sh`: exact register-budget archives;
- `perf/summarize-w3-sweep.py`: per-projection and weighted-round summary; and
- `perf/w3-air-{target,pipelines}.mtlp-json`: single-target translation and live archive routes.

Durable profile artifacts:

- trace: `kvquant-experiments/traces/aug30-w3-control/w3-r4i-ffn_down.gputrace`;
- replay: `kvquant-experiments/profiles/aug30-w3-control/r4i-ffn_down`.

The principal raw timing logs are under `kvquant-experiments/results/` with prefixes
`w3-control-*` and `w3-air-*` dated `0830`.
