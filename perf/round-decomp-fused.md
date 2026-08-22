# Round decomposition, re-derived with GDN writeback fusion enabled (2026-08-22)

Re-derivation of round-decomp-post-fa-split.md at the current prod pick, now WITH
`GGML_GDN_FUSE_WB=1` — and now trustworthy, because since the fusion commit the profiled
and production encoders execute the same work (the per-op profiling encoder derives the
fusion decision from the graph, same as the normal path).

**Headline: the two prior levers are spent (small-ne01 refuted, GDN writeback landed), and
the remaining recoverable costs are exactly two: CPU submit (~9.6 ms/round, and ~7.4
ms/token at the batch-1 floor) and the drafter round cost (18.0 ms). Verify GPU is now
81% of the round and almost entirely big-mat slope at MLX microbench parity.**

## Method

Harness `kvquant-experiments/RUN_ROUND_DECOMP.sh` (updated: `GGML_GDN_FUSE_WB=1` added to
the env set, port-busy + listener-pid guards ported from RUN_GDN_FUSE.sh), label
`aug22-fused`, logs `results/rounddecomp-aug22-fused-{n6,b1}.server.log`. Branch tip
e335471f. Same binary, same 8288-token B-tree prompt, 300 tokens, temp 0.

Anchoring (all cross-checks pass):
- n6 profiled 20.64 t/s (pre-fusion profiled run: 19.72), acc 46.9%, draft_n 469,
  sha 9abb1c6c6b16 = canonical bytes. b1 profiled 11.17 t/s.
- True rounds = 80 (draft_n 469/80 = 5.86 ≈ n_max; 300 committed → 3.75/round; spec-prof
  prints n=59/60 — stale counts again, averages valid).
- Profiled round = 300/20.64/80 = 181.7 ms; spec-prof terms sum to 181.0. Real round =
  300/24.95/80 = 150.3 ms (24.95 = unprofiled fused number from RUN_GDN_FUSE at the same
  n_predict 300). Inflation 1.209x → ticks×0.827. b1: 89.6 profiled vs 73.38 real =
  1.221x → ×0.819. (Same ~1.2x both runs, matching the pre-fusion derivation.)

## Round wall (n6, profiled → real ms)

| component | profiled | real | share | pre-fusion real |
|---|---|---|---|---|
| draft_call (drafter fwd + lattice) | 21.76 | 18.0 | 12.0% | 18.2 |
| dec_sub_tg (CPU graph build/submit) | 11.60 | 9.6 | 6.4% | 9.9 |
| dec_syn_tg (verify GPU wait) | 147.28 | 121.8 | 81.0% | 130.2 |
| accept + checkpoints | 0.42 | 0.3 | 0.2% | 0.4 |
| **round** | **181.0** | **149.7≈150.3** | | **158.6** |

The entire fusion win landed in dec_syn_tg: −8.4 ms, matching the WB-slots ceiling probe
(8.13 ms) to 0.3 ms. Draft and submit unchanged, as expected.

## Verify GPU by op class (ticks/round; pre-fusion in parens)

Total gen ticks/round 175.8 (was 186.2). MUL_MAT 136.4 (136.1), FLASH_ATTN_EXT 9.2 (9.2),
GATED_DELTA_NET 5.8 (5.6), **CPY 2.9 (13.1)** — the −10.2 ticks/round ≈ −8.5 real ms is
the fusion, and the remaining CPY is almost entirely the 416 tiny [3,10240] mask copies
(233.2 ticks total, 7.0 us/call). GDN scan per call 121.7 us (was 117.8, +3% — the kernel
now writes the state cache directly; cheap and expected). Per-call big-mat rates are
unchanged to <1% (ffn up/gate 354.4 vs 352.9, ffn down 432.4 vs 427.9), so the fusion
perturbed nothing else.

Batch-1 is bit-for-bit the pre-fusion profile: **the writeback CPY is still there at b1**
([128,128,48]→[786432,1], 25.8 us/call, 48/token ≈ 1.24 ms profiled ≈ 1.0 real = 1.4% of
the 73.4 ms floor). Not an oversight: `ggml_metal_gdn_wb_op` (ggml-metal-ops.cpp:37)
deliberately bails at n_written <= 1. Extending it to the b1 graph shape is a small,
optional floor lever.

## Where the gap stands

Cycle cost: 150.3/73.4 = **2.05 batch-1 floors** (was 2.16 post-FA-split, 2.30 before
that; oMLX 1.36–1.52). Verify GPU slope at N=7: 121.8 vs b1 syn 66.0 real = **1.85x**
(was 2.0). Gap to dflash_mlx 29.55: **1.18x** (was 1.25).

Excess over floor 150.3 − 73.4 = 76.9 ms: big-mat N=7 slope ~43 (MLX microbench parity —
closed), drafter 18.0, FA residual ~4.5, GDN scan slope ~4, elementwise/misc ~6 (profiler
overstates latency of ops hidden under concurrent dispatch — downgraded per
small-ne01-routing.md), small-ne01 starvation ~8 nominal (same caveat, refuted at e2e),
CPU submit delta ~2.2.

## Next levers, ranked

1. **CPU submit (~9.6 ms/round at n6, ~7.4 ms/token at b1).** The only line item that
   attacks BOTH the round and the floor: constant-shape N=7 verify decodes (and constant
   batch-1 decodes) should hit graph reuse — find out why ~10 ms of CPU per decode
   survives. Ceiling: round → ~141 ms (+6-7% e2e) and floor → ~66 ms = oMLX floor parity.
2. **Drafter round cost (18.0 ms = 12% of the round).** Prior attribution: ~8 ms
   CPU/lattice + ~10 GPU (4.4 full-vocab head at N=7 + small-N ffn decodes). Profile
   draft_call internals before touching anything.
3. **b1 GDN writeback fusion (~1.0 ms/token, floor only).** Extend the predicate past
   n_written <= 1 to match the batch-1 graph's contiguous writeback CPY.

Verify GPU beyond these is big-mat slope at parity — kernel work there is closed. If
drafter + submit-delta went to zero the cycle would be ~130 ms = 1.77 floors, still above
oMLX's 1.36–1.52: the residual is the N=7-vs-block-4 depth difference, i.e. structural
verify slope, not an inefficiency.
