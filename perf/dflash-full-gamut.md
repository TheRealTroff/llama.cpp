# DFlash full gamut with the combined adaptive-width kernels

Measured 2026-08-30 on the M4 Pro from base `301c0707a`, branch
`exp/skinny-soa`, dirty experiment worktree. The combined build contains the dedicated
width-3 SoA r4kp kernel and the width-6-through-8 skinny SoA kernel behind their
experimental flags. No result in this file is from a captured/profiled run.

## Route and correctness proof

With the complete combined environment and `GGML_METAL_LOG_LEVEL=2`:

- width 3 compiled `kernel_mul_mv_q4_0_soa_w3_r4kp_v3` and passed all six real
  Qwen3.8 projections (`6/6`);
- widths 6 through 8 compiled `kernel_mul_mm_skinny_q4_0_soa_f32` and passed all
  eighteen width/projection cells (`18/18`).

Fixtures: `perf/w3-real-projections.ops` and `perf/skinny-soa-real-projections.ops`.

## Method

`perf/run-dflash-full-gamut.sh` ran fixed DFlash draft depths 1 through 7, which map
to target verify widths 2 through 8. Each observation used a fresh server, the
31,522-byte/8,288-token canonical prompt (`c0653ba4af5e`), 600 predicted tokens,
temperature zero, f16 KV, FA on, and the complete production environment plus:

- persistent SoA pinning;
- width-3 SoA r4kp;
- width-4 r4kp v3 and width-5 w5r4h;
- skinny SoA for widths 6 through 8.

Two passes were mirrored in depth order: ascending, then descending. Full-round time
is recovered from response counters as
`predicted_ms / (predicted_n - draft_n_accepted)`.

## Result

| DFlash n | verify width | runs (t/s) | mean t/s | mean round | output/round | acceptance | sha1 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 20.022, 19.982 | 20.002 | 89.13 ms | 1.786 | 79.04% | `885005326897` |
| 2 | 3 | 23.272, 23.511 | 23.392 | 100.42 ms | 2.353 | 68.18% | `885005326897` |
| 3 | 4 | 25.567, 25.621 | 25.594 | 106.87 ms | 2.740 | 58.35% | `885005326897` |
| 4 | 5 | 29.764, 29.773 | **29.769** | **111.17 ms** | 3.315 | 58.19% | `6678b0507d41` |
| 5 | 6 | 26.051, 26.045 | 26.048 | 133.70 ms | 3.488 | 50.18% | `6678b0507d41` |
| 6 | 7 | 26.191, 26.256 | 26.224 | 135.97 ms | 3.571 | 43.50% | `6678b0507d41` |
| 7 | 8 | 26.082, 26.389 | 26.235 | 140.08 ms | 3.681 | 38.74% | `6678b0507d41` |

Raw results and per-run JSON/server logs:
`/Users/troff/play/kvquant-experiments/results/dflash-full-gamut-0830-0916.tsv`.

## Interpretation

The round-latency increments are `+11.29, +6.45, +4.30, +22.53, +2.27, +4.11 ms`.
The width-5-to-6 kernel-family transition is the only large discontinuity left.

On this prompt, depths 5 through 7 are strictly dominated by depth 4: they have both
higher round latency and lower throughput. Depth 4 is 14.3% faster than depth 5 and
13.5% faster than depth 7. This conclusion is especially strong because depths 4
through 7 produced identical output bytes, so the comparison does not cross a sampling
trajectory change.

To equal depth 4 at the measured round costs, depths 5, 6, and 7 would need about
3.98, 4.05, and 4.17 output tokens per round respectively, versus the observed 3.49,
3.57, and 3.68. Their approximate token-acceptance break-even points are 60.1%, 51.6%,
and 45.8%, versus 50.2%, 43.5%, and 38.7% here. Those are useful policy thresholds,
not universal optima: acceptance is workload-dependent.
