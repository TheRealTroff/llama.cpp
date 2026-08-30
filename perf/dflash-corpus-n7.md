# DFlash n=7 on the tiny prompt corpus

Measured 2026-08-30 on the M4 Pro from base `301c0707a`, branch
`exp/skinny-soa`, dirty experiment worktree. This uses the combined adaptive-width build:
the width-3 SoA r4kp kernel, width-4/5 SoA kernels, persistent SoA pinning, and skinny
SoA for verify widths 6 through 8.

## Method

`perf/run-dflash-corpus.sh` fixed DFlash draft depth at 7 (target verify width 8) and
ran the five prompts from `perf/prompts/`. Every prompt generated 300 tokens at
temperature zero in two fresh-server observations. Prompt order was mirrored between
passes. KV was f16, FA was on, and verbose/profile timing was disabled.

Full-round latency is
`predicted_ms / (predicted_n - draft_n_accepted)`. Both runs of every prompt reproduced
the same response hash and speculative counters.

## Result

| prompt | prompt tokens | mean t/s | mean round | output/round | acceptance | sha1 |
|---|---:|---:|---:|---:|---:|---|
| code explanation | 181 | 26.703 | 130.20 ms | 3.488 | 36.21% | `dab0a2ca7f08` |
| creative prose | 67 | 23.021 | 129.88 ms | 3.000 | 29.28% | `d47965fc3927` |
| support chat | 86 | 22.877 | 130.62 ms | 3.000 | 28.92% | `9881460f5b50` |
| math derivation | 90 | 53.493 | 129.99 ms | 6.977 | 87.71% | `7272b5fe0b5d` |
| JSON boilerplate | 93 | **57.629** | **129.71 ms** | **7.500** | **95.94%** | `dee997046a27` |

Raw results and per-run JSON/server logs:
`/Users/troff/play/kvquant-experiments/results/dflash-corpus-n7-0830-1102.tsv`.

## Interpretation

Round latency is effectively workload-independent on these short contexts: all five
means fit in a 0.91 ms band around 130.08 ms. The 8,288-token benchmark prompt measured
140.08 ms at the same depth, so its long KV adds about 10 ms per round.

Throughput varies by 2.52x, from support chat to JSON, while output per round varies by
2.50x. Prediction accuracy is therefore the controlling variable once context length is
fixed. Depth 7 is extremely productive for math and JSON, but wastes most of its extra
verify columns on creative prose and support chat. Determining the per-prompt optimum
still requires the same short-context corpus at narrower depths; the long-context gamut's
round costs cannot be substituted for those measurements.
