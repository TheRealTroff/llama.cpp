# Small-batch slope and the skinny cliff, re-measured (2026-08-22)

Where the slope sits at the current prod pick, and where speculation depth falls off the
skinny routing window. Harness `perf/run-slope-sweep.sh` (`TAG=slope-aug22`), prod
`7788371f`, build 2026-08-22 16:02. 15 e2e runs, zero failures.

**Every run in this sweep emitted the identical output sha1 `9ad7e023c6ab`** - all depths,
both drafters, batch-1, and the collapsed d8. Routing and speculation change speed only.

## Answer: the cliff is at ne11=9, which is spec depth 8

The skinny gate (ggml-metal-ops.cpp) is `ne11 >= max(2, GGML_MM_SKINNY) && ne11 <= 8`.
Depth d verifies d+1 columns, so d=8 is the first depth outside it.

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

## Consequences

1. **Effective depth must stay <= 7** for any speculation type on this kernel set. dflash
   enforces this itself via the block-size clamp; **MTP does not** - `--spec-draft-n-max 8`
   is accepted and silently costs 13% versus not speculating. A clamp to 7 (or extending
   the skinny window past 8) would close a live foot-gun.
2. **MTP d6 is worth re-testing in any config where dflash is unavailable** - it is only
   1.0 t/s behind the prod pick and drafts better.
3. The N=3/N=4 gap is the one remaining structural dip inside the window. Both routes into
   it are closed lines, so this is a note, not a lever.
