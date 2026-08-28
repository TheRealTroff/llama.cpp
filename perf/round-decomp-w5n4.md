# Round decomposition at the n4+w5 pick - 40% of the round is now non-matmul

2026-08-28 night, harness `kvquant-experiments/RUN_ROUND_DECOMP_W5N4.sh` (label
`aug28`), branch `m4-width4-r4kp` tip `6000d74e8`. Method of `round-decomp-fused.md`:
unprofiled anchors give real t/s, profiled runs give spec-prof terms and the per-op
dump; profiled CPU terms deflate by the round-inflation ratio; **this dump printed
`ts factor = 1.000`, so per-op absolute ms are meaningless - op numbers below are
SHARES of the m1 generation ops anchored to the real verify-GPU term**, with the
standing caveat that the profiler serializes concurrent dispatch and therefore
overstates small ops (their ms are upper bounds; MUL_MAT's is correspondingly a
slight understatement - the synthetic slice says ~68 ms where the share-anchor
says 57+12).

Anchors (n_predict 300, temp 0, canonical sha `9abb1c6c6b16` in all four runs):
**anchor-n4 26.847 t/s real** - the missing 300-unit number for the new pick, vs the
old pick's 25.02 (+7.3%) - profiled 22.079 (inflation 1.216); anchor-b1 13.003 real
(76.9 ms floor), profiled 10.734. acc 55.7%, 3.23 committed/round, ~92 rounds,
round = 120.2 ms real (the 600-unit round is 116.7 at acc 49.8 - different KV and
acceptance mix, do not cross-compare absolutes; README trap 1).

## Round wall (real ms, deflated x0.823)

| component | real | share | n6 decomp (2026-08-22) |
|---|---:|---:|---:|
| verify GPU (dec_syn_tg) | 93.9 | 78.1% | 121.8 |
| draft_call (drafter fwd + lattice) | 16.6 | 13.8% | 18.0 |
| CPU graph build/submit (dec_sub_tg) | 9.4 | 7.8% | 9.6 |
| accept + checkpoints | 0.35 | 0.3% | 0.3 |
| **round** | **120.2** | | 150.3 |

Round = 1.56 b1 floors (was 2.05 at n6). Speculation buys 2.07x over the floor at
the 300-unit point.

## Verify GPU by op class (share-anchored to 93.9 ms)

| class | ms/rd | note |
|---|---:|---|
| MUL_MAT | 72.2 | six w5 projections 56.8 + lm_head 3.9 + other mm 11.5 (attn_k/v smalls, GDN-internal) |
| FLASH_ATTN_EXT | 6.3 | 16.8 calls/rd |
| GATED_DELTA_NET | 3.3 | scan, post-fusion |
| GET_ROWS | 2.2 | |
| CPY | 1.7 | residual mask copies |
| SSM_CONV | 1.5 | |
| norms + elementwise + misc | ~6.7 | RMS_NORM 1.1, ADD 1.1, SCALE 1.0, SILU 0.9, rest <1 each |

## The non-kernel ledger (the question this file answers)

Everything that is NOT a target mul_mat kernel: **48.0 ms of the 120.2 ms round = 40%.**

| bucket | ms/rd | share of round |
|---|---:|---:|
| verify non-matmul GPU (FA+GDN+gathers+copies+elementwise) | 21.7 | 18.0% |
| drafter round cost (draft_call) | 16.6 | 13.8% |
| CPU submit | 9.4 | 7.8% |
| accept/bookkeeping | 0.35 | 0.3% |

At the n6 pick this ledger was ~46 ms of 150.3 (31%). The absolute non-kernel cost
barely moved (-2 ms, mostly draft n-max 6 -> 4); the kernel work shrank around it, so
its SHARE rose 31% -> 40% - the Amdahl shift. If the mv kernels ever reach their
stream floor (~50 ms for the projections), the non-kernel 48 ms becomes HALF the
round. The three levers, largest first:

1. **Verify non-matmul, 21.7 ms** - biggest single line is FA at 6.3 (already
   post-mm-split; `GGML_FA_MM_NWG` landed), then a long tail of ~250 us ops. The
   serialization caveat applies: some of this tail is already hidden under
   concurrent dispatch and the profiler cannot see it (small-ne01 lesson) - treat
   line items as upper bounds, and treat only measured e2e deltas as real.
2. **Drafter, 16.6 ms** - a whole second model forward per round. ~~Untouched by
   width work~~ WRONG, corrected same night from this run's own m2 dump: the
   DFlash2 drafter shares the target's hidden dims and drafts at WIDTH 5, so its
   FFN projections (5120<->17408, 10 layers) already ride w5r4h - same us/call as
   the target's verify (248.9 vs 243.5 profiled). What is NOT covered is every
   drafter shape outside the whitelist: its 248320-vocab lm_head ~~(~3.7 ms/round
   real, one 715 MB stream at ~1.4x floor - the biggest single drafter op)~~
   (CORRECTED in `shortk-head.md`: the drafter head runs at ne11=2 on the nc
   route at 1.09x its stream floor - there was never a drafter-head lever), the
   small projections (1024/1280/4096/25600-row, ~2 ms combined), FA and the
   TOP_K/lattice tail. ~~Lever: extend the whitelist with the drafter ne01s~~
   ~~**TRIED AND REFUTED same night.** `GGML_MV_SOA_WL_XL=1` ... measures FLAT vs
   the ext-di incumbent: 4963+-2 vs 4964+-1 us ... both 1.89x the 2620 us stream
   floor. Mechanism: the head is a SHORT-K shape ... all scalar routes converge~~
   **THE REFUTATION WAS A PHANTOM (`shortk-head.md`, next morning): that
   synthetic ran `GGML_MV_REPACK=1`, which declines test-backend-ops' non-weight
   buffers, so the "w5r4h arm" silently measured ext - the A/B compared ext
   against ext.** With `GGML_MV_REPACK=2` the head engages w5r4h and reads
   **3175 us vs ext-di's 4967 = 1.21x floor, -36%, ~1.79 ms/call** - the w5 win
   transfers to the head at full size; there is no short-K wall (gate/up has the
   same k=5120 and the same 20 K-iterations/thread). The short-K-tuned cells
   were built anyway and rank r8rs -0.8% / r6 +4% / r8cs +52% vs w5r4h, closing
   the amortization question: no fixed cost worth chasing. A SECOND silent
   fallback (mixed-width repack cache conflict) then blocked the lever in-server;
   with it fixed (SoA creation pin), **XL measures +3.04% e2e at the pick,
   byte-identical, b1 control inert - adoption is the owner's call.** Full
   story, both fallbacks, and the pricing: `shortk-head.md`.
3. **CPU submit, 9.4 ms** - flat since 2026-08-22 across three picks; per-round graph
   build/encode. Fixed cost, so its share grows with every kernel win (7.8% now).

Artifacts: `results/rounddecomp-aug28-{anchor,prof}-{n4,b1}.server.log`. Harness
note: RUN_ROUND_DECOMP_W5N4.sh takes a second arg to filter runs (`anchor`/`prof`);
the first invocation hit the macOS bash-3.2 `set -u` empty-array trap and the
anchors were re-run with the fixed script - and do NOT edit a harness while a run
is live, bash re-reads the file by offset (one prof-n4 run was harmlessly repeated
this way; both runs agreed to 0.8%).
