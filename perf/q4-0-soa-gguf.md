# Offline Q4_0 SoA GGUF

Date: 2026-08-30

Hardware: Apple M4 Pro

## Result

`llama-gguf-repack` stores the existing Metal Q4_0 SoA row permutation in the
GGUF itself as `Q4_0_SOA_V1`. Metal reads those mapped rows directly, so the
production `GGML_MV_REPACK=1` flags no longer allocate a second persistent copy
for converted tensors.

The default converter policy selected 375 target tensors (13.54 GiB) and 26
draft tensors (0.88 GiB). It left token embeddings, DFlash selector lookup
tables, and matrices below 16M elements in ordinary Q4_0.

## Validation

- Converter `--verify` passed for both files.
- Forward + reverse of the drafter reproduced the original whole-file SHA-256:
  `f358ee600d8c19af8d1e6d78a07b8becedde4c59e1478e3fcd644baca74b46ea`.
- Metal versus CPU: all 19 reader cases passed, covering widths 1-8 and 32,
  exact Qwen FFN orientations at widths 1, 2, 4, and 7, and two additional
  cache-resident width-1 projections.
- Route logs selected the dedicated width-1 and width-2 kernels, width-3/4/5
  SoA kernels, skinny SoA kernel, and generic `mul_mm` with `soa=1`.
- The converted target + drafter loaded together and completed a deterministic
  100-token request with the same output, 118 draft tokens, and 69 accepted
  tokens as the original files.
- `test-quantize-fns` reached and passed `q4_0_soa`; its process exit remains 1
  because of the same six TurboQuant tolerance failures present at ground-truth
  commit `9bf57e921`.

## Unified memory

Measured from fresh server processes after one completion, using 16 KiB
`vm_stat` pages and reporting wired + anonymous memory:

| files | idle baseline | loaded + exercised | baseline-adjusted |
|---|---:|---:|---:|
| original Q4_0 + runtime repack | 8.479 GiB | 41.084 GiB | 32.605 GiB |
| offline Q4_0_SOA_V1 | 8.607 GiB | 26.853 GiB | 18.246 GiB |

The baseline-adjusted reduction is **14.359 GiB** (the raw steady-state
difference is 14.231 GiB), matching the converted weight volume.

## Balanced generation timing

Fresh-process order was original, offline, offline, original. Both arms used
the full production environment, DFlash depth 4, the same 22-token prompt, and
100 deterministic output tokens.

| storage | runs (tok/s) | mean |
|---|---:|---:|
| original + runtime repack | 30.907, 31.674 | 31.291 |
| offline SoA | 31.916, 31.896 | 31.906 |

Offline SoA was **+1.97%** in this short matched run. The important acceptance
criterion is that removing the duplicate allocation did not trade away the
repack kernel's throughput.

## Short-width correction

The first offline-reader build sent stored `Q4_0_SOA_V1` at verify width 2 to
the generic eight-column skinny kernel. This mistake was isolated to the new
stored type: ordinary `Q4_0` continued to use
`kernel_mul_mv_q4_0_f32_nc2`. On the seven converted target projection shapes,
the bad stored route was about 52% slower round-weighted than ordinary Q4_0.

The corrected reader has a dedicated two-column kernel that shares each SoA
weight unpack across both activation columns. Route logs select
`kernel_mul_mv_q4_0_soa_w2`; compact and exact FFN correctness pass at widths 1
and 2. A mirrored ordinary/stored/stored/ordinary microbenchmark over all seven
target projections measured:

| width-2 storage/reader | round-weighted projection time |
|---|---:|
| ordinary Q4_0 / NC2 | 60.432 ms |
| stored Q4_0_SOA_V1 / direct W2 | 59.142 ms |

The direct stored reader is **2.13% faster** in this isolated width-2 workload.
The comparison weights the measured projection cells by their observed calls
per target round (128, 64, 64, 48, 48, 16, and 1).

A fresh-process depth-1 real-model A/B/B/A check is deliberately reported
separately because the two storage formats follow different deterministic
token/acceptance trajectories. Acceptance-independent full-round means were
88.480 ms for ordinary Q4_0 and 89.384 ms for stored SoA. The remaining 1.02%
end-to-end deficit is consistent with the separately measured width-1 gap
(including drafter and last-token projections), while the corrected width-2
route wins in isolation. This A/B cannot attribute the entire round delta
because the output trajectories differ; width 1 already had a dedicated reader
and remains a separate parity problem.

## Current stored-layout width sweep

Commit `037fc5490` was measured with the converted target and drafter, the
31,522-byte/8,288-token canonical prompt, 600 output tokens, and the complete
production environment. Two fresh-process passes were mirrored in width order.

| verify width | DFlash depth | mean t/s | mean round | output/round | acceptance |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 20.480 | 89.44 ms | 1.835 | 84.00% |
| 3 | 2 | 24.722 | 100.54 ms | 2.490 | 74.79% |
| 4 | 3 | 27.008 | 107.66 ms | 2.913 | 64.17% |
| **5** | **4** | **30.138** | **111.03 ms** | **3.352** | **59.30%** |
| 6 | 5 | 25.532 | 134.06 ms | 3.429 | 49.08% |
| 7 | 6 | 26.535 | 136.81 ms | 3.636 | 44.52% |
| 8 | 7 | 26.310 | 140.54 ms | 3.704 | 39.21% |

Width 5 remains the optimum on this prompt. The width-5-to-6 transition adds
23.03 ms per round and loses 15.3% throughput; widths 6 through 8 are all
dominated by width 5. Raw results are in
`/Users/troff/play/kvquant-experiments/results/q4soa-full-gamut-0831.tsv`.

The matched pre-storage sweep used ordinary Q4_0 files plus runtime SoA side
buffers. Across widths 2 through 8, stored-layout round-time changes were
`+0.35%, +0.12%, +0.74%, -0.13%, +0.27%, +0.62%, +0.33%`. Throughput changes
cannot be read as pure layout effects because the deterministic output and
acceptance trajectories differ. The round-cost result is the relevant one:
offline storage removes 14.359 GiB without materially changing the width curve.

## Reuse

Start with `--plan` for every new checkpoint. Selection is structural and does
not assume a layer count or exact projection dimensions. `--min-elements` and
repeatable `--exclude` patterns let a new model version refine the conservative
policy without rebuilding the tool. See `tools/gguf-repack/README.md`.
