# Width 4: the un-run morphology cell, 2-row x K-split (R2K2)

Status: **answered 2026-08-25 - kernel-level win, e2e neutral; not adopted, R2 stays the
route.** Branch `m4-width4-r2k2` off prod `745fd2ce8`.

Opened from `m4-width4-ilp.md` open stub 1. The morphology matrix measured 4-row K2
(K split across two simdgroups, terminal barrier), 4-row K1, 3-row R3, and 2-row R2
(full-K, barrier-free), but never a 2x4 tile with two simdgroups splitting K. K-split won
at 4 rows, full-K won at 2 rows, and the R2 profile showed neither issue (1.93/tick) nor
DRAM (54% of peak) saturated, so the cell was not a priori dead.

## The kernel

`kernel_mul_mv_q4_0_soa_w4_r2k2` (`GGML_MV_SOA_W4_R2K2=1`): R2's exact loop body - same
SoA layout, same eight FP32 accumulators, same vector-dot inner expression - with K2's
`p = 32*sgitg + tiisg, p += 64` stride, an 8-value `threadgroup float partial[2][8]`,
one terminal barrier, and the two-way K-part add. Dispatch is `(ne01 + 1)/2` groups of
64 threads, so the launched-simdgroup count doubles R2's: at m=5120 that is 2560 groups
x 2 = 5120 simdgroups against R2's 2560.

Offline `applegpu_g16s` prescreen: **2096 bytes native text, zero spill** (R2: 2184/0,
K2: 3658/0). Exact `ffn_down` correctness (`m=5120,n=4,k=17408`, `GGML_MV_REPACK=2`,
MTL0) passes the CPU reference with the r2k2 pipeline confirmed in the log.

## Synthetic per-shape A/B (test-backend-ops perf, interleaved arms)

`GGML_MV_REPACK=2 GGML_MV_SOA_W4=1` common; arms differ only in `_R2=1` vs `_R2K2=1`.
Three interleaved reps for ffn_down, two for the rest; per-rep spread <1.5%.

| projection (m,k at n=4) | R2 mean, us | R2K2 mean, us | delta |
|---|---:|---:|---:|
| ffn_down (5120,17408) | 336.54 | 327.67 | **-2.6%** |
| ffn_gate/up (17408,5120) | 300.25 | 298.68 | -0.5% |
| attn_output/ssm_out (5120,6144) | 116.08 | 111.19 | **-4.2%** |
| attn_qkv (10240,5120) | 185.74 | 182.38 | -1.8% |
| attn_gate (6144,5120) | 115.00 | 110.89 | **-3.6%** |
| attn_q (12288,5120) | 217.52 | 214.24 | -1.5% |

R2K2 wins every routed shape. The win is largest where m is smallest, i.e. where R2's
grid is smallest (m=5120: 2560 groups), and shrinks toward noise at m=17408 (8704
groups) - consistent with the R2 profile's reading that the projections are
occupancy/latency-limited, not bandwidth-limited: splitting K doubles resident
simdgroups exactly where the grid is thin. The R2 arm reproduces the values recorded in
`m4-width4-ilp.md` within cross-session drift.

## End-to-end A/B

`perf/run-m4-width4-r2k2-e2e.sh`, the R2 runner with arms r2/r2k2: fresh server per arm,
`GGML_MV_REPACK=1` SoA common env, DFlash n3 target point, DFlash n6 width-7 inertness
control, 4 order-balanced n3 pairs + 2 control pairs, n_predict 600, temperature 0.
The warmup reproduced the landed R2 e2e byte-identically (20.620 t/s, sha `462183a49c4c`),
so the arms are directly comparable with `m4-width4-ilp.md`'s run.

| n3 arm | four runs, t/s | mean, t/s | delta |
|---|---|---:|---:|
| R2 | 20.709, 20.724, 20.723, 20.701 | 20.714 | - |
| R2K2 | 20.703, 20.658, 20.612, 20.635 | 20.652 | **-0.30%** |

The width-7 control is flat (+0.08%, 23.010 vs 23.027) and byte-identical across arms
(`3776c0adb7ee`), so the selector is inert and the run is valid.

## Verdict: the kernel wins, the trajectory takes it back

R2K2's n3 output sha is `3776c0adb7ee` - **exactly K2's recorded trajectory**, because the
two-part K reduction reproduces K2's floating-point association. That trajectory carries
K2's 59.3% acceptance against R2's luckier 60.2%, i.e. ~1% fewer committed tokens/round on
this prompt. Per round, R2K2 is faster: 2.779 tok / 20.652 t/s = **134.6 ms/round** against
R2's 2.806 / 20.714 = **135.5 ms/round**, a **-0.7%** pass-side win consistent with the
synthetic table. In t/s the trajectory penalty cancels it: **-0.30%**, negative.

By the decision standard used throughout this series (measured t/s), R2K2 is **not
adopted**. R2 remains simpler (no threadgroup memory, no barrier) and equal-or-better e2e.
The greedy-trajectory difference is luck, not quality - but the same luck applies to any
prompt, and there is no basis for preferring the arm that measured slower.

Two findings survive for the gap decomposition:

1. **K-split doubles resident simdgroups exactly where the grid is thin, and it works** -
   every m=5120/6144 projection gained 2.6-4.2% in isolation. The morphology matrix is now
   complete: {2,3,4} rows x {K1,K2} all measured, and 2-row is optimal in both columns.
2. **Occupancy recovery inside this kernel family is worth ~1% of the pass, not ~25%.**
   The verify pass's distance from its bytes floor (~109 vs 76-85 ms) is not going to be
   closed by more simdgroups on the projections alone.
