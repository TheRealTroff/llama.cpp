# Small-batch slope and the skinny cliff, re-measured (2026-08-22)

**STALE FOR WIDTHS 2-4 SINCE 2026-08-27 (`m4-width4-r4kp.md`): the width-4 verify is
~26-28% cheaper, dflash n3 reads 25.15 t/s and MTP d3 24.48 at n_predict 600 - both
depth optima need re-sweeping with GGML_MV_SOA_W4 + R4KP set.**

Where the slope sits at the current prod pick, and where speculation depth falls off the
skinny routing window. Harness `perf/run-slope-sweep.sh` (`TAG=slope-aug22`), prod
`7788371f`, build 2026-08-22 16:02. 15 e2e runs, zero failures.

**Every run in this sweep emitted the identical output sha1 `9ad7e023c6ab`** - all depths,
both drafters, batch-1, and the collapsed d8. Routing and speculation change speed only.

## Answer: the cliff is at ne11=9, which is spec depth 8

The skinny gate (ggml-metal-ops.cpp) is `ne11 >= max(2, GGML_MM_SKINNY) && ne11 <= 8`.
Depth d verifies d+1 columns, so d=8 is the first depth outside it.

**dflash_mlx counts the other way: their block *b* verifies *b* columns**
(`spec_epoch.py:2247-2257`). Our depth *d* == their block *d+1*. Never compare an `nN` row
with a `block N` row - match on the width column instead.

llama-bench ms/pass, f16 KV, `-n 0 -p 1..10 -r 3`:

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prod env | 73.0 | 73.8 | 101.5 | 111.5 | 119.0 | 120.9 | 123.1 | 124.1 | **277.6** | 279.0 |
| stock    | 72.6 | 83.6 | 100.4 | 112.1 | 124.7 | 156.8 | 186.7 | 188.0 | 278.7 | 279.7 |

The 8 -> 9 step costs **154 ms**; the 7 -> 8 step costs 1.0 ms. N=9 lands on 277.6, matching
the ~283 ms/pass `dflash-vs-mtp-uniform.md` recorded for mul_mm.

The stock column is the control and confirms the mechanism: prod env is 1.30x better at
N=6 and 1.51x at N=8, then the two **converge at N=9** (277.6 vs 278.7). Once off skinny
the routing flags buy nothing, which is exactly what a window-boundary effect predicts and
what a thermal or measurement artifact would not.

Note `prod-baseline.md`'s llama-bench table is stock routing (it runs with no env
overrides) and uses q8_0 KV, so it is not comparable to the prod-env row here.

## A second, smaller cliff at N=3

Marginal cost per added column, prod env: **+0.8, +27.7, +10.0, +7.5, +1.9, +2.2, +1.0**,
then +153.5.

N=2 is essentially free (mv-nc2). N=3 costs 27.7 ms, the largest step inside the window,
because `mv_nc_route` is gated to `ne11 <= min(GGML_MV_NC, 4)` = 2 while skinny starts at
5 - so columns 3 and 4 fall through to ext, covered well by neither kernel. Both halves of
that gap are already-closed lines (`mv-nc-cliff-probe.md`: NC>=3 pays a fixed ~112 us
penalty, parity not a win; `dflash-vs-mtp-uniform.md`: plain skinny at N=4 is 125 vs ext
119), but the sweep shows the residual cost clearly, and it is why both drafters dip at
depth 2.

## dflash depth sweep (n_predict 300)

| depth | t/s | acc | drafted/rd | committed/rd | rounds | ms/round |
|---|---:|---:|---:|---:|---:|---:|
| n1 | 20.821 | 84.0% | 0.99 | 1.83 | 164 | 87.9 |
| n2 | 19.975 | 72.5% | 1.98 | 2.44 | 123 | 122.1 |
| n3 | 20.462 | 63.6% | 2.96 | 2.88 | 104 | 141.0 |
| n4 | 22.023 | 55.7% | 3.94 | 3.19 | 94 | 144.9 |
| n5 | 23.371 | 49.9% | 4.91 | 3.45 | 87 | 147.5 |
| **n6** | **25.038** | 46.9% | 5.86 | 3.75 | 80 | 149.8 |
| n7 | 24.807 | 40.3% | 6.83 | 3.75 | 80 | 151.2 |
| n8 | 24.680 | 40.3% | 6.83 | 3.75 | 80 | 151.9 |

**Optimum stays n6** (25.038, reproducing the prod-pick run's 25.046/24.993). n7 commits
exactly the same 3.75 tokens/round as n6 - acceptance falls 46.9 -> 40.3%, cancelling the
extra draft - while costing 1.4 ms more, so the 7th draft token is pure loss.

Acceptance reproduced the values in `acceptance-metric-conversion.md` **exactly** at every
depth (72.5 / 63.6 / 55.7 / 49.9 / 46.9 / 40.3), confirming drafting is deterministic at
temp 0 and only timing varies.

### dflash cannot reach the cliff: n8 IS n7

`common/speculative.cpp:1008` clamps `n_max` to `block_size - 1`, and this drafter reports
`block_size=8`, so `--spec-draft-n-max 8` runs as 7 and logs
`requested draft size (n_max=8) exceeds the trained block size 8 -- clamping to 7`.

The n8 row above is byte-identical to n7 in every counter. **So any recorded "dflash n8" is
a mislabelled n7** - including the 23.13 in `flash-attn-mm-split.md`, which should not be
read as n8 beating n7.

That row is also a free same-config replicate: 24.807 vs 24.680 = **0.127 t/s spread
(0.51%)** at n_predict 300, consistent with the ~0.4 t/s control spread noted in
`gdn-writeback-fusion.md`. Differences below ~0.5% at this length are not real.

## MTP depth sweep - the optimum MOVED from d1 to d6

| depth | t/s | acc | drafted/rd | committed/rd | rounds | ms/round |
|---|---:|---:|---:|---:|---:|---:|
| d1 | 22.127 | 86.2% | 0.99 | 1.85 | 162 | 83.7 |
| d2 | 20.833 | 75.6% | 1.98 | 2.50 | 120 | 120.0 |
| d4 | 22.390 | 58.7% | 3.91 | 3.30 | 91 | 147.2 |
| **d6** | **24.215** | 49.0% | 5.91 | 3.90 | 77 | 160.9 |
| d7 | 23.000 | 41.8% | 6.81 | 3.85 | 78 | 167.2 |
| d8 | **11.927** | 38.7% | 7.75 | 4.00 | 75 | **335.4** |

**This supersedes the "MTP d1 stays optimal ... depth sweep complete, don't re-run" note.**
That conclusion predates the GDN writeback fusion. The flattened verify curve moved MTP's
optimum from d1 (22.127) to **d6 (24.215), +9.4%** - the same "optimum moves deeper as
verify cheapens" effect the CPY fix and the FA mm-split each produced.

MTP d6 out-drafts dflash n6 (3.90 vs 3.75 committed/round, 49.0% vs 46.9%) but is still
1.0 t/s slower, because its round costs 11 ms more (160.9 vs 149.8). dflash emits its whole
block in one decode (speculative.cpp:1293) whereas MTP runs d sequential chained head
passes, so dflash's drafter is cheaper per round despite MTP's head being individually
tiny. **The prod pick is unchanged: dflash n6.**

### d8: the collapse, and it is worse than not speculating

d8 is 11.927 t/s with the round cost doubling 167.2 -> 335.4 (2.01x) for one extra column.
Drafting *improved* - 4.00 committed tokens/round, the highest of any config measured - so
the loss is entirely the kernel. Against the batch-1 floor of **13.656 t/s** measured in
the same sweep, d8 is **13% slower than plain decoding**: you run a drafter, commit 4
tokens a round, and still lose.

Prior record was 10.29 t/s on the older stack; the collapse is milder now but still total.

## Which widths a run actually hits

A fixed-depth run is not a single kernel path, but it is close to one. From the per-op
profiles (`rounddecomp-aug22-tagged-n6`, plus the untagged pre/post-fusion pair), counting
MUL_MAT calls by the second dim of s1. One full target forward pass = **496** matmul calls
(calibrated on prefill: 8288 tokens = exactly 16 passes at width 512 plus one remainder
pass, and batch-1 generation = 299.6 passes at width 1 for 300 tokens).

dflash n6, per run:

| verify width | passes | kernel |
|---|--:|---|
| 7 | 77 | skinny |
| 6 | 1 | skinny |
| 3 | 1 | **ext** |

This reconciles exactly with the sweep counters: 77*6 + 5 + 2 = **469 drafted over 79
rounds = 5.86/round**, which is what the n6 row reports. The two odd rounds are almost
certainly the first round after prefill (no lattice yet) and the last round truncated by
the n_predict boundary. **Only one pass per run - the width-3 one - lands on a different
kernel than the rest.**

Two things that look like narrow rounds and are not:

- **Widths 2 and 4 are server startup, not verification.** They appear with identical call
  counts (992 and 496 = exactly 2 and 1 passes) in the batch-1 `--spec-type none` profile,
  which drafts nothing at all. Fixed per-run cost, present in every config, contaminating
  no comparison.
- **The drafter's width-16 row** (395 calls) is its lattice/top-k shape, not a verify batch.

**The mixture is deterministic.** The pre-fusion and post-fusion n6 profiles have
byte-identical width histograms (`{1:83, 2:1039, 3:508, 4:507, 6:508, 7:42829}`), and the
tagged run's per-model split sums to the untagged totals on every width (38269 + 4560 =
42829, 992 + 47 = 1039, ...). So the path mix is reproducible across runs and across a code
change, and is **not** a source of the 0.51% run-to-run spread - that is pure timing.

Consequence for debugging: because that single width-3 pass straddles a kernel boundary and
skinny/ext/mv-nc are not bit-identical, a routing change can flip the output sha while
changing throughput by nothing. That is the same mechanism as the `GGML_FA_VEC_MAX=4`-vs-`5`
trap in `flash-attn-mm-split.md`. **An unexpected sha change is not automatically a bug** -
check whether the routing of a rare narrow pass moved before assuming corruption.

## Consequences

1. **Effective depth must stay <= 7** for any speculation type on this kernel set. dflash
   enforces this itself via the block-size clamp; **MTP does not** - `--spec-draft-n-max 8`
   is accepted and silently costs 13% versus not speculating. A clamp to 7 would close a
   live foot-gun.

### Fixing the cliff is NOT a performance lever

Extending the skinny window past 8 would fix the foot-gun and buy no speed. Counterfactual
d8 at the measured 4.00 committed tokens/round:

| assumed d8 round cost | t/s |
|---|---:|
| flat from d7, zero marginal cost (most generous possible) | 23.92 |
| on the d6->d7 trend (+6.3 ms) | 23.06 |
| on the mean in-window marginal (+7.3 ms) | 22.92 |

All three are **below the measured MTP d6 (24.215)** and well below dflash n6 (25.038).

The reason is that **committed tokens/round has saturated at ~4.0**: d6 -> d7 *lost* 0.05
tokens/round and d7 -> d8 gained only 0.15, while every added column costs 6-7 ms even
inside the window. Depth is finished as a lever in both directions - the optimum is
interior, and the cliff sits past the point where extra depth stops paying.

**All remaining upside is round cost, not depth.** At a saturated 4.0 tokens/round:

| round cost | t/s |
|---|---:|
| 167.2 (today, MTP d7) | 23.92 |
| 160.9 (today, MTP d6) | 24.86 |
| 150.0 | 26.67 |
| 140.0 | 28.57 |
| 130.0 | 30.77 |

Concretely for the prod pick: dflash n6 commits 3.75 tokens in a 149.8 ms round. Matching
dflash_mlx's 29.55 t/s at that same committed/round needs a **126.9 ms** round, i.e. cutting
**22.9 ms**. ~~The verify-slope lever in `round-decomp-fused.md` is scoped at ~20 ms. Those
two numbers agreeing is the case for that lever being the whole remaining game.~~
**CORRECTED 2026-08-22: that lever does not exist** (`verify-slope-close.md`). The slope is
dense-matmul width scaling - matmul alone fills the whole 1.5x budget - so the 22.9 ms is
not available there. The verify side is worth ~5-7 ms; the rest of the gap is their cheap
block-4 operating point, and the adaptive prerequisite below is now the live question.
2. **MTP d6 is worth re-testing in any config where dflash is unavailable** - it is only
   1.0 t/s behind the prod pick and drafts better.
3. The N=3/N=4 gap is the one remaining structural dip inside the window. At fixed depth
   both routes into it are closed lines, so it is a note, not a lever - **but see below,
   because that is only true while we run one width.**

## Prerequisite if we ever go adaptive

At fixed depth, **the only point on the slope curve that matters is the width we actually
run** - 97.5% of rounds at width 7, one stray pass at 3. Everything else on the curve is
decoration, which is why the N=3/N=4 dip is currently harmless and why several kernel lines
were closed as "parity, not a win".

Adaptive speculation inverts that. A controller that shortens low-acceptance blocks visits
the whole curve, so **every width becomes hot and each one has to be driven as low as it
goes** before the policy can pay. dflash_mlx already works this way: `verify_mode` defaults
to adaptive and its archived counters spread over `cycles_by_block={1:1, 4:81, 5:17}`,
average depth 4.14 - i.e. it spends 80% of its cycles at a width we have never optimised.

Two closed lines would have to be reopened, and both were closed *because* we only run one
width:

- **The mv-nc NC>=3 cliff** (`mv-nc-cliff-probe.md`). Width 2 costs 73.8 ms - only 0.8 over
  batch-1, essentially free - because mv-nc2 covers it. Width 3 costs 101.5. The whole
  N=2->N=3 step is mv-nc's structure not extending past 2 columns. It was closed as "fixing
  it yields parity with ext, not a win", which is correct at fixed depth 7 and irrelevant
  under a controller that lives at widths 2-5.
- **`GGML_MV_NC_V2`** (branch `metal-mv-nc-spill`). `prod-baseline.md` measured +0.79% with
  disjoint ranges at MTP d1 (width 2) and nothing at dflash n6 (width 7), and concluded it
  "brings nothing to the config that actually ships". Under adaptive, width 2 *is* a
  shipping width. Note it was only ever A/B'd at `GGML_MV_NC=2`; **whether V2's lower
  register pressure also relieves the NC>=3 spill penalty is untested**, and that is the
  experiment that would decide whether widths 3-4 can be brought near width 2's cost.

So the ordering is: adaptive depth is not a lever on its own - `adaptive-spec` was closed
once as "doesn't beat best-fixed", and our shallow cycles are not cheap the way theirs are
(~~our n4 costs 2.15 batch-1 floors vs their block-4 at 1.36~~ **off by one - their block 4
is width 4, so the partner is our n3 at 1.99 floors; see below**). Flattening widths 2-5 is
the prerequisite that would make the policy worth having, not a follow-up to it.

> **CORRECTED 2026-08-22 (`mlx-cycle-capture.md`): their block *b* verifies *b* columns, ours
> *d+1*** (`spec_epoch.py:2247-2257` vs line 13 above). This section's llama-bench figures are
> already widths and are unaffected; only the `nN`-vs-`block N` comparison above was wrong.
> Their operating widths are **1, 4 and 5** (`cycles_by_block={1:1, 4:81, 5:17}`), which in
> our labels is **depths 0, 3 and 4**. So the widths that matter for an adaptive policy are:
>
> | width | our depth | our kernel | our ms/round | theirs, pinned |
> |---|---|---|--:|--:|
> | 4 | n3 | ext (`nxpsg=8, nr0=2, chpt=1`) | 141.0 | **95.00** (their block 4) |
> | 5 | n4 | skinny mm | 144.9 | 137.26 (their block 5) |
> | 7 | **n6 (prod pick)** | skinny mm | 149.8 | - |
>
> Two consequences. **Width 5 is already near parity (1.06x); the entire gap is width 4
> (1.48x)** - so "flatten widths 2-5" is really "fix width 4", and width 3 (our depth 2) is
> nobody's operating point. **And width 4 is on `ext` while width 5 is on `skinny`**, so the
> two need different kernel work; an `ext` change cannot reach width 5. Note our own prod
> pick runs width 7 on skinny, so none of this touches the shipping config - it buys an
> operating point we do not currently have, and today n3 is our *worst* depth (20.46 t/s).
