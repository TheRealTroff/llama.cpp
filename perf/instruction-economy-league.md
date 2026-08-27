# The instruction-economy league - every archived replay, per-instruction

Status: **done 2026-08-27.** `perf/shaderprof-table.py` run over every archived replay
on disk (17 configs across `traces/aug23/replays/` and `profiles/aug25-*`; the aug23
layout keeps streamData beside `raw/` and the tool now handles that itself). Every
decode validated: executed sums match `instructionExecuted` bit-for-bit. This is the
fleet-wide per-instruction ground truth under the width-4 and skinny investigations.

Issue% / stall% are the kernel's OWN time split (issue+stall = 100 within the kernel).
exec/disp is dynamic instructions per dispatch (dispatch count = the main binary's
trace count, e.g. 141 for the w4 ffn_down captures, matching
`skinny-width-captures.md`'s counts). e/d/col divides by verify width. All ffn_down
rows are the same 5120x17408 shape and comparable; attn_q rows are a different shape -
compare their issue/stall, not their exec columns.

| capture | kernel | w | live | iss% | stall% | exec/disp | e/d/col |
|---|---|--:|--:|--:|--:|--:|--:|
| w1-ffn_down-mv | mul_mv_q4_0_f32 | 1 | 389 | 64.0 | 36.0 | 11.6M | 11.6M |
| w2 nc2 (aug23) | mul_mv_q4_0_f32_nc2 | 2 | 664 | 67.9 | 32.1 | 16.7M | 8.4M |
| w2 nc2 (aug25) | mul_mv_q4_0_f32_nc2 | 2 | 664 | 67.7 | 32.3 | 16.9M | 8.5M |
| w3 ext nx8 | mul_mv_ext_q4_0_f16_r1_3 | 3 | 379 | 79.9 | 20.1 | 27.4M | 9.1M |
| w3 ext nx16 | mul_mv_ext_q4_0_f16_r1_3 | 3 | 397 | 83.1 | 16.9 | 27.2M | 9.1M |
| w4 ext nx8 | mul_mv_ext_q4_0_f16_r1_4 | 4 | 453 | 84.0 | 16.0 | 32.0M | 8.0M |
| w4 ext nx16 | mul_mv_ext_q4_0_f16_r1_4 | 4 | 477 | 87.7 | 12.3 | 31.7M | 7.9M |
| w4 ext NO f16y | mul_mv_ext_q4_0_f32_r1_4 | 4 | 457 | 64.9 | 35.1 | 33.8M | 8.5M |
| w4 attn_q nx8 (f32) | mul_mv_ext_q4_0_f32_r1_4 | 4 | 457 | 69.4 | 30.6 | 6.1M | 1.5M |
| w4 attn_q nx16 (f32) | mul_mv_ext_q4_0_f32_r1_4 | 4 | 483 | 66.1 | 33.9 | 6.2M | 1.6M |
| w4 r2 (landed) | mul_mv_q4_0_soa_w4_r2 | 4 | 271 | 77.0 | 23.0 | 30.4M | 7.6M |
| w4 k2 | mul_mv_q4_0_soa_w4_k2 | 4 | 465 | 74.8 | 25.2 | 29.5M | 7.4M |
| w4 r2u2 | mul_mv_q4_0_soa_w4_r2u2 | 4 | 391 | 68.6 | 31.4 | 25.8M | 6.4M |
| w4 r2_sumy | ..._w4_r2_sumy | 4 | 315 | 84.3 | 15.7 | 38.0M | 9.5M |
| w4 r2_sumymin | ..._w4_r2_sumymin | 4 | 293 | 82.0 | 18.0 | 34.2M | 8.6M |
| w5 skinny | mul_mm_skinny_q4_0_f32 | 5 | 507 | 72.7 | 27.3 | 19.2M | 3.8M |
| w7 skinny ffn_down | mul_mm_skinny_q4_0_di_f32 | 7 | 433 | 77.3 | 22.7 | 22.1M | 3.2M |
| w7 skinny attn_q | mul_mm_skinny_q4_0_di_f32 | 7 | 433 | 86.3 | 13.7 | 15.8M | 2.3M |
| w7 skinny gate/up | mul_mm_skinny_q4_0_di_f32 | 7 | 433 | 88.6 | 11.4 | 22.3M | 3.2M |
| w7 f16b B-direct probe | mul_mm_skinny_q4_0_f16b_g16 | 7 | 443 | 82.9 | 17.1 | 20.2M | 2.9M |

## Readings

1. **Every kernel in the fleet is issue-dominated: 64-89% of its own time is spent
   issuing instructions.** No kernel shows a latency-bound signature (stall > 50%).
   The instruction-economy frame is not just the two walls - it is the whole fleet.

2. **The stall column behaves as a memory-pressure meter, and it validates itself.**
   The one kernel known to run near the DRAM roof (w1 plain mv, 81% of peak) has the
   highest stall, 36%. The nc2 pair replicates across sessions to 0.2 points
   (67.9/67.7) with identical 664 live instructions.

3. **The f16y win at width 4 is a STALL win, not an instruction win - now measured
   directly.** f32-y ext r1_4: 64.9% issue / 35.1% stall. Same kernel with f16 y:
   84.0/16.0. Live instructions barely move (457 vs 453) and dynamic instructions drop
   only 5%; what f16y buys is halved y-load traffic and a stall share cut by 2.2x.
   (The +17.3% e2e number is in `width4-verify.md` run 4.)

4. **The mv family barely amortizes its stream when widening; the MMA path is in a
   different economy class.** Dynamic instructions per column on ffn_down: plain mv
   11.6M -> nc2 8.4M (widening 1->2 buys only 1.4x) -> ext-f16/r2 at width 4 ~7.6-8.0M
   (1->4 buys only 1.5x) -> skinny at w7 3.2M (3.6x). The whole width-4 kernel series
   moved the per-column stream by ~7%/column-added; `simdgroup_multiply_accumulate`'s
   fixed 8-wide tile is what an actual amortization looks like. (At width 4 skinny
   still loses e2e - its fixed tile executes the same total for 4 real columns, ~5.6M
   per real column, and its issue rate differs - but the 2.4x class gap says a
   width-4-shaped kernel that amortizes like MMA is the only kind that could close the
   1.42x, consistent with `w4-ffn-scratch.md`'s arithmetic-rate ceiling.)

5. **The r2u2 unroll refinement:** it EXECUTES 15% fewer instructions than r2 (25.8M
   vs 30.4M/dispatch) and still lost, because stall rose 23.0 -> 31.4%. The unroll
   traded issue for stalls. Same lesson as sumy in reverse (`b745bff6d`): neither
   instruction count nor stall alone predicts time - the product of issue share and
   issue rate does.

6. **sumy decomposition confirmed dynamically:** r2_sumy executes 25% more
   instructions than r2 (38.0M vs 30.4M) at a BETTER issue share (84.3 vs 77.0) - more
   work, issued more smoothly, net slower. Matches the earlier replay finding that
   instruction count explained only half its slowdown.

7. **ffn_down is the stall-heavy skinny shape: 22.7% against gate/up's 11.4% and
   attn_q's 13.7% at identical code.** The ~11-point stall excess is the same
   magnitude as the ~12% ffn_down headroom the grid theory tried to claim
   (`skinny-grid-refuted.md` - grid change moved inflight, not time). The deficit is
   real and is stall; the grid was just not its cause. If ffn_down ever gets its own
   lever again, this is the number it must move.
   **Localized by per-instruction diff against gate/up** (same binary, offsets match
   1:1): the excess is NOT one site - every stall site roughly doubles. Barrier 0xafa
   +1.6 points, the 0x7f4 load +1.0, 0x86a +0.6, and a diffuse +0.18-0.29 on each MMA
   instruction; the sum of sites where gate/up stalls MORE is only 0.6 points. Uniform
   scaling of all waits is the signature of less latency hiding overall - ffn_down
   runs 160 threadgroups against gate/up's 544 for the same streamed bytes. Open
   tension with `skinny-grid-refuted.md`: doubling the grid raised inflight with zero
   time change, so either the extra inflight did not convert to hiding (issue-bound
   ceiling), or the stall metric and the time are decoupled here - a replay of the
   grid-fix arm would decide, and needs one GUI click on a new capture.

## Method notes

- Dispatch counts come from the main binary's `traceCount` (now in the tool's output
  and JSON as `dispatches`); captures repeat the op a shape-dependent number of times,
  so NEVER compare raw executed totals across captures - normalize per dispatch first.
- attn_q at width 4 never has f16y (16.78M-element gate, `width4-verify.md` run 5
  retraction), so its f32 rows double as the f32-y replication on a second shape.
- No per-family (dequant vs FMA vs load) split inside a loop body yet - that needs
  mnemonics (`agx-disasm.md`). The 14-byte encoding family tracks device loads and the
  12-byte family tracks the MMA lowering, which is enough for region-level reading
  (`skinny-stall-attribution.md`) but not for a strict ALU census.
- aug23-skinny's 6 matched-width captures were never GUI-replayed and have no raw
  bundles; replaying them would give this table skinny at w4/w6/w8 on one shape.
