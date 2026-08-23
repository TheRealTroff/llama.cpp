# Width 4 on skinny: reproduces a known result, and moves the tile question (2026-08-23)

Status: **open**. Harness `perf/run-width4-ab.sh`, `TAG=width4ab-0823-1418`, prod `863675ea6`,
binary Aug 23 10:02. All seven e2e arms emitted canonical sha1 `9ad7e023c6ab`.

## Measurement quality: read relatives, not absolutes

Controls bracket the run: **ctrl-n6-pre 24.045, ctrl-n6-post 24.751** against the 25.038
reference - a **2.9% spread**, wider than the ~0.4 t/s usually seen. Absolutes here run ~3%
low. The run is NOT void (`width4-verify.md` run 6 died at -6.1% on a byte-identical
workload, and drift here is smaller and non-monotonic: arm 4 is worse than arm 2 despite
being later), but **quote no absolute from this file**. The headline effect reproduces in two
independent adjacent pairs, which is why it survives.

## e2e at width 4

| arm | t/s | ms/round | committed/rd |
|---|--:|--:|--:|
| dflash n3, skinny5 (ext today) | 19.678 | 146.6 | 2.88 |
| dflash n3, skinny4 **no** repack | 19.384 | 148.8 | 2.88 |
| dflash n3, skinny4 **+ repack** | **21.322** | **135.3** | 2.88 |
| MTP d3, skinny5 | 20.487 | 143.6 | 2.94 |
| MTP d3, skinny4 + repack | **21.773** | **135.1** | 2.94 |

**This reproduces `dflash-vs-mtp-uniform.md`, it does not discover it.** That file already has
ext 119 / skinny 125 / skinny+di 114 at N=4. Plain skinny loses to ext; skinny+di wins.
Confirmed.

**Two corrections to earlier readings in this repo's notes, mine included.** "Misroute" and
"only correct with repack" mean the wrong *routing choice*, not wrong output - every arm here
is canonical, so there is no output-correctness issue at `GGML_MM_SKINNY=4`. And **repack is
not free: +15 GB of Q4_0 weight residency** (`mtp-kv-results.md`), which is exactly why the
prod pick excludes it at a recorded ~0.4 t/s at d4.

**MTP d3, the cell the MTP sweep skipped, is now measured** and beats dflash n3 on both
routings (20.487 vs 19.678; 21.773 vs 21.322). The interpolation in `occupancy-next.md` was
right that this cell was worth filling.

## llama-bench ms/pass - the kernel without the round

| ne11 | skinny5 | skinny4+repack |
|--:|--:|--:|
| 1 | 72.6 | 74.3 |
| 2 | 74.9 | 74.4 |
| 3 | 101.4 | 108.3 |
| **4** | **112.0** | **107.4** |
| 5 | 121.0 | 106.2 |
| 6 | 121.7 | 108.7 |
| 7 | 124.3 | 111.1 |
| 8 | 125.6 | 111.7 |

ne11=3 regresses 6.8% with repack: at `SKINNY=4` the gate is `ne11 >= max(2,4)`, so width 3
still routes to ext, and repack changes the ext path too (`ggml-metal-ops.cpp:2851`).

## The tile question moved, in both directions

**For:** with skinny+di holding widths 4-8, cost runs 107.4 -> 111.7 - **four extra columns
for 4.3 ms**. Every previous statement of our "flat cost curve" spanned two different kernels,
which was the weak point in the argument. This is one kernel, and it is flat, which is what a
column tile fixed at 8 by `simdgroup_half8x8` predicts.

**Against, and this is probably the stronger reading:** flatness across ne11 4-8 is ALSO
exactly what a **weight-bandwidth-bound** matmul looks like. You stream the same 14.32 GiB of
weights regardless of column count, so columns are free until something else saturates. At
these batch sizes that is the more likely bottleneck. **If it is bandwidth, the discarded MMA
lanes cost nothing** - the units are not the constraint - and MLX's advantage is their data
movement (inline register dequant, no threadgroup staging, reuse 4 on both operands), not
their tile shape.

**The two are now distinguishable.** `agxps-counter-values` makes 137 counters readable. An
ALU/MMA-utilization counter against a memory-bandwidth counter at ne11=4 separates them
directly. **Take that measurement before writing any width-4 kernel.**

## Missing and next

- **The 2x2 is incomplete**: skinny5+repack was never run, so "repack helps" cannot yet be
  separated from "routing width 4 to skinny helps".
- n6 (prod width 7) + repack was not run either, though llama-bench says ne11=7 improves
  124.3 -> 111.1. Whether that survives e2e, and whether +15 GB is payable, is a separate call.
- Even the best width-4 config here is **135.3 ms/round against their 95.00** - still 1.42x,
  down from 1.48x. The gap is not closed by routing.
