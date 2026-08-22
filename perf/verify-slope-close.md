# Closing the 23.3 ms: does the verify-slope lever actually exist?

Status: **closed - the lever does not exist.** Task 1 and the free half of Task 2 done
2026-08-22 on branch `verify-slope-close` off prod `85ba62a5c`. No new runs: this is
arithmetic over `rounddecomp-aug22-{fused-b1,tagged-n6}`, `decodeprof-aug22-{b1,n6}`,
`slope-sweep.md`'s llama-bench table and the archived oMLX artifact
`.artifacts/dflash/benchmarks/20260822-172839-smoke-qwen3-8-27b-4bit/results.json`.

## Answer to Task 1: no. There is no 20 ms of removable overhead.

**Dense matmul alone fills the entire 1.5x budget.** The verify pass at N=7 costs 130.0 ms
against a 72.8 ms batch-1 floor (both unprofiled `dec_syn_tg`: 129.993 and 72.845). A 1.5x
slope would allow **109.3 ms**. MUL_MAT by itself is **105-108 ms** of the current 130.0.

That figure is the m1 (target-only) MUL_MAT tick total, 122.2 ticks/round, deflated by the
measured profiler factor 129.993/147.275 = 0.883; the low end drops the three server-startup
passes. Every other operation in the pass - flash attention over 8.4k KV, the GDN scan, all
norms, copies, gathers and the elementwise storm - would have to fit in the **~3 ms** left
over. They currently cost ~22 ms and none of them can be zero. **1.5x is not a target with
20 ms of slack in front of it; it is roughly the floor of the current matmul kernels.**

Deflation is the only soft step here, so note it is not load-bearing: at +/-5% the matmul
term is 100-113 ms against a 109.3 ms budget, and the conclusion survives either end.

### Independent cross-check, no profiler involved

The stub's llama-bench comparison works, and it is sharper as a *delta* than as a level
(comparing deltas cancels any constant offset in llama-bench's per-pass baseline, which the
stub was right to warn about):

| quantity | N=1 | N=7 | delta |
|---|--:|--:|--:|
| llama-bench, prod env, short KV, no spec | 73.0 | 123.1 | **+50.1** |
| in-graph verify GPU, 8.4k KV | 72.8 | 130.0 | **+57.2** |

The two slopes differ by **7.1 ms**, and that residue is fully attributed:

- FA growth over 8.4k KV: 3.72 -> 7.51 ticks/pass = +3.79, deflated **3.3 ms**. llama-bench
  at short KV has essentially no FA, so none of this is in its +50.1.
- The context-length mask copies (`CPY f32 s0=[3,10240]`, 416 calls/round at 7.0 us):
  2.94 ticks/pass, deflated **2.6 ms**. Also absent from a short-KV pass.
- Residual: **~1.2 ms**.

So everything that the verify pass does *beyond pure dense width scaling* - all the long-KV
cost and all the speculation-specific cost - is **about 7 ms per round, and 6 of it is two
already-named items**. The remaining 50.1 ms of the 57.2 ms excess is width scaling
measured independently, with no speculation and no KV, and that is the term already closed
at MLX microbench parity.

Two routes, one conclusion: the 1.79x is dense-op width scaling.

### Per-op slope, target only, width-7 passes vs batch-1 passes

Ticks/pass, `m1` context, clean width filter (77.2 and 299.6 passes; ratios are same-unit):

| op | b1 | N=7 | slope | share of N=7 pass |
|---|--:|--:|--:|--:|
| MUL_MAT | 62.20 | 120.29 | 1.93x | 88.8% |
| FLASH_ATTN_EXT | 3.72 | 7.51 | 2.02x | 5.5% |
| GATED_DELTA_NET | 0.98 | 5.86 | 5.99x | 4.3% |
| everything else (all ops) | ~12 | ~24 | ~2.0x | ~1.4% exposed |

All-ops totals are 79.1 ticks/pass at b1 and 157.9 at N=7 = 2.00x in ticks against 1.78x in
wall time, i.e. ~11% of summed ticks hide under concurrent dispatch at N=7 versus ~0% at
batch-1 - the small ops hide, the big matmuls are the critical path. GDN's 5.99x is the
documented linear-in-N scan and is structural.

### What is actually left on the verify side

Real ms/round, deflated, all of it bandwidth or copy work so the translation rule is
favourable:

| item | now | realistic recovery |
|---|--:|--:|
| FA at N=7 (post-mm-split residual) | 6.6 | ~2 |
| context mask copies `[3,10240]` x416 | 2.6 | ~2.5 |
| read-side GDN `GET_ROWS` gather | 2.5 | ~2 |

Call it **5-7 ms/round**: 149.8 -> ~144, which at 3.75 committed tokens/round is **~26.0
t/s, about +4%**. Worth doing, and nowhere near the 126.6 ms round that matching 29.6 t/s
requires. **Prod pick is unchanged.**

## Task 2: the trap is real, but the artifact does contain a usable measurement

**`phase_timings_us` is confirmed unusable, now on n=5 rather than n=1.** Across all five
runs the non-prefill phases sum to 847-1169 ms against ~10,120 ms of generation wall - 8.3%
if you count only verify+draft+replay+commit, 11% if you also count the draft subphases.
`prefill` matches `ttft_ms` exactly in every run. So the prefill timer brackets a sync and
the generation-loop timers do not. Do not build on them.

**But `adaptive_metrics` does reconcile, and nobody had used it.** Deriving ms/cycle from
`tokens_per_cycle_by_block / tokens_per_second_by_block` and weighting by `cycles_by_block`
gives 100.0 ms/cycle against 102.7 ms of wall per cycle - agreement to **2.7%**, unlike the
9x miss above. That is the first *measured* view of their cost curve:

| block | their cycles | their ms/cycle | their tok/cycle | our depth | our ms/round | our tok/round |
|---|--:|--:|--:|---|--:|--:|
| 1 | 1 | 72.4 | 1.00 | b1 | 74.1 | 1.00 |
| 4 | 81 | **91.9** | 3.049 | n4 | 144.9 | 3.19 |
| 5 | 17 | 140.2 | 3.059 | n5 | 147.5 | 3.45 |
| - | - | - | - | n6 | 149.8 | 3.75 |

Two things fall out, and they reframe the whole comparison:

1. **At matched depth 5 we are faster than they are.** Their block-5 cycle yields 21.82 t/s;
   our n5 yields 23.37. We commit more per cycle (3.45 vs 3.06) at a similar cost (147.5 vs
   140.2 ms). Our n6 at 25.04 beats their block 5 by 15%.
2. **Their entire 1.184x edge is a cheap shelf at block 4** that our curve does not have:
   91.9 ms vs our 144.9 at the same depth, and their controller sits there for 82% of
   cycles. Their block 4 -> 5 step costs **+48.3 ms**; our own width 5 -> 6 step costs
   **+1.9 ms** (llama-bench 119.0 -> 120.9). One of the two curves has a cliff, and it is
   not ours.

So "their slope is 1.5x and ours is 1.79x" was the wrong framing. The right one is: **we
are at or ahead of parity everywhere on the curve we have measured, and they have one
operating point we cannot reach.**

### The shelf has two possible explanations and they need separating first

(a) **It is real.** Their block-4 cycle buys 4 extra verify columns for 19.5 ms over their
block-1 cycle, where we pay 46.0 ms for the same four (llama-bench 73.0 -> 119.0). But this
contradicts the per-shape microbench parity already on record (`head-to-head-aug22.md`: MLX
n=5 slope 1.74x vs our 1.81x), so it would have to come from what their cycle computes, not
from their matmul kernels.

(b) **It is lazy-eval misattribution** - the same disease as `phase_timings_us`, one level
up. If the sync lands on full-mode (block-5) cycles, work deferred from preceding block-4
cycles is charged to block 5, making block 4 look cheap and block 5 expensive. That is
exactly the observed shape. **The 2.7% agreement on the weighted total does not rule this
out**, because total work is conserved under misattribution; only the split moves.

**Decisive test, cheap, on their CLI.** `DFLASH_VERIFY_MODE` takes
`dflash|adaptive|ddtree|off` (`runtime/config.py:759`, resolved from CLI or that env var);
`dflash` is fixed-block, non-adaptive. Run fixed block 4 and fixed block 5, 5 reps each, on
the same 8288-token prompt via `run-head-to-head.sh`:

- fixed block 4 lands near **33 t/s** -> the shelf is real, and building an equivalent cheap
  operating point is the whole remaining game.
- fixed block 4 lands near **29-30 t/s** -> the by-block split was distorted, their real
  advantage is the adaptive policy visiting cheap widths, and `slope-sweep.md`'s adaptive
  prerequisite (flatten widths 2-5 first) is the correct and already-written next step.

Either way this is one run and it decides which of the two remaining programmes to fund.
Do it before building anything.

## Consequences for the lever board

- **Verify slope is closed as a ~20 ms lever.** It was arithmetic against a target
  (`~1.5x`) that was never measured on their side and that our own matmul kernels cannot
  reach even with every other op in the pass set to zero.
- Remaining verify-side work is the 5-7 ms copy/FA/gather tail above, worth ~+4%.
- The `1.184x` gap is now localised to a single operating point of theirs, not to a
  property of our kernels.
- Unchanged: depth is finished as a lever, drafter quality is not the gap, CPU is not a
  lever anywhere, and the prod pick stays at ~25.0 t/s.
