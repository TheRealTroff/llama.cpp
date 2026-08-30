# Skinny SoA Q4_0 matmul

Measured 2026-08-30 on M4 Pro from committed base `301c0707a`, branch
`exp/skinny-soa`. The experiment is uncommitted.

## Result

`kernel_mul_mm_skinny_q4_0_soa_f32` reuses the established 32-row x 8-column,
64-K-slice, two-simdgroup skinny MMA body while reading the persistent SoA Q4_0
layout used by the width-4/5 scalar kernels:

```
[half scale x nblk][uint nibble-planar pack8 x 4*nblk]
```

The final loader directly indexes the two Q4_0 blocks owned by each thread and
expands each pack8 with vector shifts. It is behind `GGML_MM_SKINNY_SOA=1`, requires
`GGML_MV_REPACK=1`, and routes widths 6 through 8 when `GGML_MM_SKINNY=6`.
The flag also pins eligible tensors to SoA at first repack, even if a smaller-width
consumer arrives first, so dispatch order does not strand a later skinny consumer on
the incompatible DI layout.

Route proof on `m=5120,n=6,k=17408` names both
`kernel_repack_q4_0_soa` and
`kernel_mul_mm_skinny_q4_0_soa_f32_ne12=1_r2=1_r3=1`.

## Correctness

`perf/skinny-soa-real-projections.ops` contains the exact six exported target
projections at each of widths 6, 7, and 8. Both the incumbent DI control and SoA pass
18/18 against CPU:

```sh
GGML_MM_SKINNY=6 GGML_MV_REPACK=2 GGML_MM_SKINNY_SOA=1 \
  ./build/bin/test-backend-ops test -o MUL_MAT -b MTL0 \
  --test-file perf/skinny-soa-real-projections.ops
```

The built-in perf list lacks several of these large width-6/7/8 shapes and can return
a vacuous 0/0 for a shape filter. Use the exported-op fixture for the full set.

## Real-projection microbenchmarks

Three separate-process observations per arm in A/B, B/A, A/B order. The table reports
medians in microseconds. `GGML_MV_REPACK=2`; DI and SoA use the same skinny MMA tile.

| width | projection | skinny DI | skinny SoA | delta |
|---:|---|---:|---:|---:|
| 6 | ffn gate/up | 320.05 | 320.05 | 0.00% |
| 6 | ffn down | 378.17 | 375.44 | -0.72% |
| 6 | attn output | 129.93 | 127.21 | -2.09% |
| 6 | attn qkv | 196.22 | 195.35 | -0.44% |
| 6 | attn gate | 124.66 | 124.97 | +0.25% |
| 6 | attn q | 233.80 | 233.09 | -0.30% |
| 7 | ffn gate/up | 323.90 | 323.11 | -0.24% |
| 7 | ffn down | 380.12 | 378.47 | -0.43% |
| 7 | attn output | 127.87 | 126.28 | -1.24% |
| 7 | attn qkv | 199.51 | 199.20 | -0.16% |
| 7 | attn gate | 126.55 | 125.95 | -0.47% |
| 7 | attn q | 235.74 | 233.68 | -0.87% |
| 8 | ffn gate/up | 327.14 | 330.12 | +0.91% |
| 8 | ffn down | 385.54 | 380.78 | -1.23% |
| 8 | attn output | 130.18 | 130.38 | +0.15% |
| 8 | attn qkv | 201.61 | 199.72 | -0.94% |
| 8 | attn gate | 126.71 | 127.94 | +0.97% |
| 8 | attn q | 237.47 | 237.53 | +0.03% |

Weighted by the tagged model calls per round (128/64/64/48/48/16), DI -> SoA is:

| width | skinny DI | skinny SoA | delta |
|---:|---:|---:|---:|
| 6 | 92.628 ms | 92.241 ms | -0.42% |
| 7 | 93.393 ms | 93.008 ms | -0.41% |
| 8 | 94.439 ms | 94.498 ms | +0.06% |

## Hardware profile

Matched width-6 captures at the gate/up projection, profiled by headless replay:

| metric | skinny DI | skinny SoA |
|---|---:|---:|
| temporary registers | 51 | 52 |
| uniform registers | 28 | 28 |
| per-thread spilled bytes | 0 | 0 |
| thread-invariant spilled bytes | 32 | 32 |
| instructions | 433 | 421 |
| ALU | 335 | 324 |
| FP32 / FP16 | 32 / 32 | 32 / 64 |
| INT32 / INT16 | 68 / 95 | 63 / 59 |
| device loads | 8 | 8 |
| issue / stall cost | 77.7808 / 22.2189 | 78.0259 / 21.9738 |

The planar loader removes 12 static instructions and 36 INT16 instructions without a
meaningful pressure or issue/stall change. Captures and replay archives:

- SoA: `/tmp/perf-metal-31188.gputrace`, `/tmp/skinny-soa-profile-31188`
- DI: `/tmp/perf-metal-31204.gputrace`, `/tmp/skinny-di-profile-31204`

`agx-spill-probe` cannot currently translate either this kernel or the unchanged skinny
DI kernel (`applegpu-nt: cannot find private metadata ...`), while it still translates a
known width-3 metallib. Therefore no offline spill inference is used here; the spill and
instruction claims above are hardware replay measurements.

## Real model: persistent-layout recovery

Qwen3.8-27B uniform Q4_0, `llama-bench -n 0 -p 4,5,6,7,8 -r 5`, two fresh-process
order-balanced runs per arm. Widths 4 and 5 establish the persistent SoA side copy
before widths 6 through 8. Without this kernel, those later widths must use plain skinny;
with it, they remain on SoA and recover DI-class throughput.

| width | incumbent mean | skinny SoA mean | delta | isolated DI control |
|---:|---:|---:|---:|---:|
| 4 | 47.42 t/s | 47.62 t/s | +0.41% | - |
| 5 | 54.39 t/s | 54.23 t/s | -0.30% | - |
| 6 | 48.71 t/s | 54.67 t/s | +12.24% | 55.02 t/s |
| 7 | 56.92 t/s | 63.51 t/s | +11.57% | 63.66 t/s |
| 8 | 62.94 t/s | 70.15 t/s | +11.46% | 70.67 t/s |

Standalone flag/order check: with no width-4/5 SoA flags, a fresh process running
`-p 1,6,7,8 -r 3` pins SoA when width 1 arrives and then records 13.45, 55.01,
63.14, and 70.50 t/s. Thus `GGML_MM_SKINNY_SOA=1` is order-safe on its own; it does
not depend on a width-4/5 consumer having created the layout first.

## End to end

Fixed DFlash depth 5 (verify width 6), 600 predicted tokens, four order-balanced fresh
servers per arm. The baseline carries the width-4/5 SoA flags and therefore uses the
plain-skinny fallback at width 6; the candidate adds only `GGML_MM_SKINNY_SOA=1`.

| arm | runs (t/s) | mean | acceptance | sha1 |
|---|---|---:|---:|---|
| pinned SoA + plain skinny | 22.111, 22.145, 22.107, 22.126 | 22.122 | 45.7% | `3776c0adb7ee` |
| pinned SoA + skinny SoA | 24.306, 24.283, 24.269, 24.324 | 24.295 | 46.0% | `3776c0adb7ee` |

That is **+9.82% end to end**, byte-identical. The depth-4 control, which has no
width-6 operation, is flat at 25.623 vs 25.636 t/s with identical 49.8% acceptance
and SHA. Raw results:
`/Users/troff/play/kvquant-experiments/results/skinny-soa-dflash-n5-0830.tsv`.

`perf/run-m4-width6-e2e.sh` now accepts `ARM_A`/`ARM_B`, a `skinny-soa` arm, and
`RUN_MTP=0`; defaults retain its original skinny-vs-scalar-W6 behavior.
