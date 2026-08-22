# Round decomposition, re-derived post FA-mm-split (2026-08-22)

The mm-split writeup ended with: FA is now 5.0% of generation GPU, so re-derive the round
decomposition before picking the next lever. This is that re-derivation, on the current
prod pick (uniform Q4_0 target + pure-Q4_0 drafter + `GGML_MV_NC=2 GGML_MM_SKINNY=5
GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8`, dflash n-max 6), branch tip 66cc4f39.

**The pick: small-ne01 matmul routing.** Every small-output matmul in the verify pass pays
an ~80 us threadgroup-starvation floor in the skinny kernel — `[5120,48]` costs 81.5 us at
N=7 vs 10.2 us at batch-1 (8.0x per call for 7x the work, on a 0.13 MB weight). Summed over
the round this is ~8 ms of a 158.6 ms cycle, ≈ +5% e2e, and the fix is routing-level (the
same starved-not-inefficient disease the FA mm kernel had). Runner-up: lazy GDN snapshot
writeback (~6–7 ms, more invasive).

## Method

Two `GGML_METAL_PROFILE=1` runs of the same binary (harness
`kvquant-experiments/RUN_ROUND_DECOMP.sh`, parser `parse_round_decomp.py`, logs
`results/rounddecomp-aug22-{n6,b1}.server.log`): the prod dflash-n6 config and a
`--spec-type none` batch-1 floor. 8288-token B-tree prompt, 300 tokens, temp 0.

Anchoring (everything cross-checks):
- n6 profiled 19.72 t/s, acc 46.9% (exactly yesterday's n6 per-draft figure).
- True rounds = **80**, not spec-prof's `n=57`: draft_n 469/80 = 5.86 ≈ n_max, accepted
  220 + 80 = 300 committed, and 80 × 189.8 ms round = 15.18 s = 300/19.76 t/s exactly.
  **Gotcha: the final spec-prof dump undercounts `n` (57) — its averages are right, its
  counts are not.** Committed/round = 3.75, matching 23.64 t/s × 158.6 ms unprofiled.
- Profiler inflation 1.20x on both runs (158.6/189.8 and 73.38/88.7): shares and per-call
  ratios carry over. "Real ms" below = ticks × 0.836. b1 ticks/token (80.2) ≈ profiled
  syn (79.4), so ticks ≈ profiled GPU ms.

## Round wall (n6, profiled ms → est. real ms)

| component | profiled | real | share |
|---|---|---|---|
| draft_call (drafter fwd + lattice) | 21.8 | 18.2 | 11.5% |
| dec_sub_tg (CPU graph build/submit) | 11.8 | 9.9 | 6.2% |
| dec_syn_tg (verify GPU wait) | 155.7 | 130.2 | 82.0% |
| accept + checkpoints | 0.46 | 0.4 | 0.2% |
| **round** | **189.8** | **158.6** | |

The four terms sum to the measured wall exactly (×80 = 15.18 s) — no overlap, no hidden
per-round cost. Batch-1: sub 9.26 + syn 79.4 profiled = 73.4 real. Note CPU submit is
~10 ms/round (~13% of the batch-1 floor at N=1 too) — a real, separate line item.

## Verify GPU by op class (ticks/round; b1 = ticks/token)

| op | n6/round | real ms | b1/token | slope |
|---|---|---|---|---|
| MUL_MAT | 136.1 | 113.8 | 63.5 | 2.14x |
| CPY | 13.1 | 10.9 | 1.75 | 7.5x |
| FLASH_ATTN_EXT | 9.2 | 7.7 | 3.87 | 2.4x |
| GATED_DELTA_NET | 5.6 | 4.7 | 1.02 | 5.5x |
| RMS/SILU/ADD/CONT/GET_ROWS/etc | ~18.8 | 15.7 | ~9.6 | ~2x |
| TOP_K (dflash lattice) | 1.4 | 1.2 | — | |
| **total** | **186.2** | **155.7** | **80.2** | **2.32x** |

(n6 ticks include the drafter's GPU work, which is why total slightly exceeds syn.)

Per-call verify slopes (N=7 vs N=1), the key table:

| matmul (us/call) | b1 | n6 | slope |
|---|---|---|---|
| ffn up/gate [5120,17408] | 203.9 | 352.9 | 1.73 |
| ffn down [17408,5120] | 205.2 | 427.9 | 2.09 |
| ssm in [5120,10240] | 122.3 | 225.6 | 1.84 |
| o_proj [6144,5120] | 85.0 | 161.2 | 1.90 |
| q_proj [5120,6144] | 86.7 | 148.1 | 1.71 |
| head [5120,248320] | 2805 | 4430 | 1.58 |
| **kv [5120,1024]** | **18.0** | **85.6** | **4.75** |
| **GDN a/dt [5120,48]** | **10.2** | **81.5** | **8.0** |
| FA (8448 KV) | 231.7 | 463.8 | 2.0 |
| GDN scan | 20.6 | 117.8 | 5.7 (linear in N — structural) |
| GDN writeback CPY | 25.8 | 175.4 | 6.8 (bandwidth-proportional, both ~250 GB/s) |

Big weight-streaming mats sit at 1.6–2.1x — microbench parity with MLX (closed line). The
outliers are the two small-output rows: **everything with ne01 ≤ ~1024 pays a flat ~80 us
at N=7 regardless of size** ([5120,48] 81.5, [5120,1024] 85.6, [5120,1280] 83.2,
[5120,256] 86.9 — while [5120,4096] at 114.5 us is fine at 230 GB/s).

CAUSE: the skinny gate (ggml-metal-ops.cpp:2578) has **no minimum on ne01**. Skinny tiles
32 dst rows per threadgroup, so [*,48] dispatches 2 TGs and [*,1024] 32 TGs — starved, the
FA-mm disease again. The mv path these rows use at batch-1 parallelizes rows/nr0=4 per TG
(12 and 256 TGs) and does [5120,48] in 10.2 us.

## Excess over floor (85.2 real ms = 158.6 − 73.4), attributed

| term | real ms | verdict |
|---|---|---|
| big-mat N=7 slope | ~43 | intrinsic-ish; at MLX microbench parity, little left |
| drafter round cost | ~18 | GPU ~9 (head-at-N=7 3.6 + eh_proj + small-N decodes), CPU/lattice ~8 |
| small-ne01 mat starvation | **~8** | **recoverable — next lever** |
| GDN snapshot writeback CPY | ~7 | recoverable via lazy/accepted-only writeback |
| elementwise/misc small ops | ~6 | launch-bound storm, incl. 416 tiny [3,10240] mask CPYs (~2.5 ms) |
| FA residual post-split | ~4.5 | per-call slope 2.0; diminishing |
| GDN scan slope | ~3.7 | recurrence is linear in N — structural |
| CPU submit delta | ~2.2 | see graph-reuse question below |

## Next levers, ranked

1. **Small-ne01 routing (~8 ms, ≈ +5% e2e → ~24.8 t/s).** Don't send ne01 ≤ ~2048 to
   skinny; give them an mv-style ne11 loop instead (GGML_MV_NC's structure already does
   exactly this for N≤4 — extend/reuse it for small-ne01 at N up to 8, taking precedence
   over skinny). Watch bit-identity: mv-loop vs skinny is the same not-bit-identical class
   as skinny-vs-ext, so expect an output-sha change unless the threshold is chosen to only
   move rows that never affect sampling order... it won't be — just re-baseline.
2. **Lazy GDN snapshot writeback (~7 ms).** delta-net-base.cpp writes all n_written
   per-position states (48 layers × 3.1 MB × 7 ≈ 2.1 GB/round at 250 GB/s — already on the
   fast CPY path, it's pure traffic). Only the accepted position's state is ever restored;
   writing one instead of seven needs acceptance-aware copy scheduling (post-accept copy or
   deferred GET_ROWS index), which touches the speculative flow — medium risk.
3. **Drafter round cost (~18 ms).** ~8 ms CPU/lattice + 4.4 ms full-vocab head at N=7 +
   several small-N ffn decodes (the N=2..6 rows). Architectural; profile draft_call's
   internals before touching.
4. **CPU submit (~10 ms/round, and ~9 ms/token at batch-1).** Constant-shape N=7 verify
   decodes should be hitting graph reuse — check why 10 ms of CPU per decode survives.

FA, GDN scan, and the big-mat slope are not worth further kernel work at current shares.
