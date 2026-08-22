# Closing the 23.3 ms: does the verify-slope lever actually exist?

Status: **closed - the lever does not exist.** Task 1 and the free half of Task 2 done
2026-08-22 on branch `verify-slope-close` off prod `85ba62a5c`. No new runs.

**Provenance (read this before quoting anything below).** These numbers come from four
sources and the first draft of this file mixed two builds. Corrected:

| quantity | source | build |
|---|---|---|
| wall anchors: b1 `dec_syn_tg` 72.053, n6 130.637, round costs | `slope-aug22-{batch1,dflash-n4,n5,n6}.server.log` | prod `7788371f`, 16:02 |
| llama-bench width curve 73.0 -> 123.1 | `slope-sweep.md` | prod `7788371f`, 16:02 |
| per-op ticks, target/drafter split | `rounddecomp-aug22-tagged-n6.server.log` | `be462dc70` |
| per-op ticks, batch-1 | `rounddecomp-aug22-fused-b1.server.log` | `e335471f6` |
| their cycle curve | `.artifacts/dflash/benchmarks/20260822-172839-.../results.json` | 0.1.10+omlx.6 |

**The profile builds are not stale: `git diff be462dc70 HEAD -- ggml src common include
tools` is empty**, and `e335471f6 -> HEAD` is instrumentation plus a `GGML_METAL_N_CB` env
that defaults to the previous constant. So the verify path is byte-identical code across
every source above, and the tick data is current.

**But `round-decomp-fused.md`'s anchors are from a different session than every headline
number.** Its b1 72.845 / n6 129.993 / slope 1.79x come from the 12:31 decode-prof run;
`README.md`, `slope-sweep.md` and `head-to-head-aug22.md` all quote the 16:02 build, where
the same code measures b1 **72.053** and n6 **130.637** - the floor got ~1% faster and the
verify pass ~0.5% slower. That is session drift, not a regression, but it moves the slope
from 1.79x to **1.813x**, so do not mix the two sets. Numbers below are all 16:02.

## Answer to Task 1: no. There is no 20 ms of removable overhead.

**Dense matmul alone fills the entire 1.5x budget, and then some.** The verify pass at N=7
costs **130.637 ms** against a **72.053 ms** batch-1 floor (unprofiled `dec_syn_tg`). A 1.5x
slope would allow **108.08 ms**. Target-only MUL_MAT is **106-108 ms** of the current
130.637.

That is the `m1` MUL_MAT tick total, 122.2 ticks/round over 80 rounds, deflated by the
measured profiler factor 130.637/147.451 = **0.886**; the low end also drops the three
server-startup passes. So matmul by itself is **98-100% of the whole 1.5x budget**, and
every other operation in the pass - flash attention over 8.4k KV, the GDN scan, all norms,
copies, gathers and the elementwise storm - would have to fit in what is left, which is
approximately nothing. They currently cost ~22 ms and none of them can be zero.

**1.5x is not a target with 20 ms of slack in front of it; it is at or below the floor of
the current matmul kernels.** Deflation is the only soft step, and it is not load-bearing:
at +/-5% the matmul term is 101-114 ms against a 108.1 ms budget.

### Independent cross-check, no profiler involved

Comparing the slope as a *delta* rather than a level cancels any constant offset in
llama-bench's per-pass baseline, which the stub was right to warn about:

| quantity | N=1 | N=7 | delta |
|---|--:|--:|--:|
| llama-bench, prod env, short KV, no spec | 73.0 | 123.1 | **+50.1** |
| in-graph verify GPU, 8.4k KV | 72.053 | 130.637 | **+58.6** |

The two differ by **8.5 ms**, and that residue is almost fully attributed:

- FA growth over 8.4k KV: 3.72 -> 7.51 ticks/pass, deflated 3.33 -> 6.65 = **+3.3 ms**.
  llama-bench at short KV has essentially no FA, so none of this is in its +50.1.
- Context-length mask copies (`CPY f32 s0=[3,10240]`, 416 calls/round at 7.0 us):
  2.94 ticks/pass = **2.6 ms**. Also absent from a short-KV pass.
- Residual: **~2.6 ms**.

So everything the verify pass does *beyond pure dense width scaling* - all the long-KV cost
and all the speculation-specific cost - is **about 8.5 ms per round, and 6 of it is two
already-named items**. The other 50.1 ms of the 58.6 ms excess is width scaling measured
independently, with no speculation and no KV, which is the term already closed at MLX
microbench parity.

Two routes, one conclusion: the 1.81x is dense-op width scaling.

### Per-op slope, target only, width-7 passes vs batch-1 passes

Ticks/pass, `m1` context, clean width filter (77.2 and 299.6 passes; same-unit ratios):

| op | b1 | N=7 | slope | share of N=7 pass |
|---|--:|--:|--:|--:|
| MUL_MAT | 62.20 | 120.29 | 1.93x | 88.8% |
| FLASH_ATTN_EXT | 3.72 | 7.51 | 2.02x | 5.5% |
| GATED_DELTA_NET | 0.98 | 5.86 | 5.99x | 4.3% |
| everything else (all ops) | ~12 | ~24 | ~2.0x | ~1.4% exposed |

All-ops totals are 79.1 ticks/pass at b1 and 157.9 at N=7 = 2.00x in ticks against 1.81x in
wall time, i.e. ~10% of summed ticks hide under concurrent dispatch at N=7 versus ~0% at
batch-1: the small ops hide, the big matmuls are the critical path. GDN's 5.99x is the
documented linear-in-N scan and is structural.

### What is actually left on the verify side

Real ms/round at the 16:02 build, all of it bandwidth or copy work, so the translation rule
is favourable:

| item | now | realistic recovery |
|---|--:|--:|
| FA at N=7 (post-mm-split residual) | 6.65 | ~2 |
| context mask copies `[3,10240]` x416 | 2.60 | ~2.5 |
| read-side GDN `GET_ROWS` gather | 2.55 | ~2 |

Call it **5-7 ms/round**: 149.9 -> ~144, which at 3.75 committed tokens/round is **~26.0
t/s, about +4%**. Worth doing, and nowhere near the 126.6 ms round that matching 29.6 t/s
requires. **Prod pick is unchanged.**

## Task 2: the trap is real, but the artifact does contain a usable measurement

**`phase_timings_us` is confirmed unusable, now on n=5 rather than n=1.** Across all five
runs the non-prefill phases sum to 847-1169 ms against ~10,120 ms of generation wall - 8.3%
counting only verify+draft+replay+commit, 11% including the draft subphases. `prefill`
matches `ttft_ms` exactly in every run. So the prefill timer brackets a sync and the
generation-loop timers do not. Do not build on them.

**But `adaptive_metrics` does reconcile, and nobody had used it.** Deriving ms/cycle from
`tokens_per_cycle_by_block / tokens_per_second_by_block` and weighting by `cycles_by_block`
gives 100.0 ms/cycle against 102.7 ms of wall per cycle - agreement to **2.7%**, unlike the
9x miss above. That is the first *measured* view of their cost curve:

| block | their cycles | their ms/cycle | their tok/cycle | ours | our ms/round | our tok/round |
|---|--:|--:|--:|---|--:|--:|
| 1 | 1 | 72.4 | 1.00 | b1 | 73.2 | 1.00 |
| 4 | 81 | **91.9** | 3.049 | n4 | 144.9 | 3.19 |
| 5 | 17 | 140.2 | 3.059 | n5 | 147.5 | 3.45 |
| - | - | - | - | n6 | 149.9 | 3.75 |

> **CORRECTION 2026-08-22 (later), from direct measurement in `block4-shelf-probe.md`:**
> the block-4 row is right (derived 91.9, measured **95.00** pinned), but **the block-5 row
> is not representative and point 1 below is WRONG.** Adaptive only enters block 5 when
> acceptance is already poor - those cycles accept 41.2%/draft against 53.2% at true fixed
> block 5 - so the 140.2 / 3.059 / 21.82 t/s figures describe bad cycles, not block 5.
> Measured fixed block 5 is **137.26 ms/cycle, 3.6585 tok/cycle, 26.654 t/s**, which is
> **faster than our n5's 23.37**, not slower. What survives is the cycle *cost* comparison:
> 137.26 vs our 147.5 is only 7% apart, so the two engines are near level at depth 5 and
> their whole advantage is the block-4 shelf. Deriving per-block behaviour from an adaptive
> run's own rows is not safe; the controller's choice of when to escalate is confounded
> with the thing being measured.

Two things fall out, and they reframe the comparison:

1. ~~**At matched depth 5 we are faster than they are.** Their block-5 cycle yields 21.82
   t/s; our n5 yields 23.37.~~ **REFUTED by measurement - see the correction above.** At
   matched depth 5 they are 1.14x faster (26.654 vs 23.371); what is true is that our
   *cycle cost* is within 7% of theirs there (147.5 vs 137.26).
2. **Their entire 1.184x edge is a cheap shelf at block 4** that our curve does not have:
   91.9 ms vs our 144.9 at the same depth, and their controller sits there for 82% of
   cycles. Their block 4 -> 5 step costs **+48.3 ms**; our own width 5 -> 6 step costs
   **+1.9 ms** (llama-bench 119.0 -> 120.9). One of the two curves has a cliff, and it is
   not ours.

So "their slope is 1.5x and ours is 1.79x" was the wrong framing twice over: the number is
1.81x, and we are at or ahead of parity everywhere on the curve we have measured. They have
one operating point we cannot reach.

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
  prerequisite (flatten widths 2-5 first) is the correct next step.

Either way it is one run and it decides which of the two remaining programmes to fund.
Do it before building anything.

## Consequences for the lever board

- **Verify slope is closed as a ~20 ms lever.** It was arithmetic against a target
  (`~1.5x`) that was never measured on their side and that our own matmul kernels cannot
  reach even with every other op in the pass set to zero.
- Remaining verify-side work is the 5-7 ms copy/FA/gather tail above, worth ~+4%.
- The 1.184x gap is localised to a single operating point of theirs, not to a property of
  our kernels.
- **Correction to `round-decomp-fused.md`:** its 130.0 / 72.8 / 1.79x are 12:31-session
  numbers; at the 16:02 build the same code reads 130.637 / 72.053 / 1.813x.
- Unchanged: depth is finished as a lever, drafter quality is not the gap, CPU is not a
  lever anywhere, and the prod pick stays at ~25.0 t/s.
