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

---

# Second run: repack at the prod width is +9.3%, and the 2x2 closes

`TAG=repack2-0823-1452`, harness `perf/run-repack-cells.sh`, prod `e72397290`. All arms
canonical sha1 `9ad7e023c6ab`. **Controls: pre 24.708, post 24.779 - a 0.29% spread**, so
unlike the first run these absolutes are trustworthy.

| arm | t/s | ms/round |
|---|--:|--:|
| n6 control (prod pick, no repack) | 24.708 / 24.779 | 151.8 / 151.3 |
| **n6 + repack, SKINNY=5** | **27.014** | **138.8** |
| **n6 + repack, SKINNY=4** | **27.070** | **138.5** |
| n3 + repack, SKINNY=5 (ext+di) | 17.817 | 161.9 |

**+9.3% at the prod width**, on a flag that already exists. The README's current number is
25.02. Best-vs-best against MLX moves from **1.302x to 1.203x** (32.556 vs 27.070).
`SKINNY=4` vs `5` is immaterial at n6 - width 7 routes to skinny either way.

## The 2x2 at width 4, completed

| | no repack | repack |
|---|--:|--:|
| ext (`SKINNY=5`) | 146.6 | **161.9** |
| skinny (`SKINNY=4`) | 148.8 | **135.3** |

Repack **hurts ext by 10.4% and helps skinny by 9.1%**. It is not a general win: it pays
only where the kernel reads the deinterleaved `_di` copy. This also separates the two
variables the first run confounded - the width-4 win was repack-on-skinny, not the routing
change alone.

## The exclusion justification was stale, not wrong

`a559a52d9` recorded GGML_MV_REPACK as a **negative result**, and `mtp-kv-results.md` excludes
it as "+15 GB residency for ~0.4 t/s". That was measured at **MTP d4 on an older stack**,
before the skinny kernel, the GDN writeback fusion and the FA mm-split each flattened the
verify curve. At dflash n6 today it is worth **+2.3 t/s**. A refuted lever can come back when
what surrounded it changes - check the date and the config on any negative result before
trusting it.

Residency is still +15 GB and that cost is real; the owner has said to treat it as fixable
(a load-time transform that replaces rather than duplicates the weights) and probe anyway.

## What this says about `simdgroup_matrix` on M4

**M4 Pro has no GPU matrix hardware** - `simdgroup_matrix` lowers to ordinary FMAs on the
same SIMD ALUs; Apple did not ship per-core neural accelerators until M5. So "half the MMA
discarded at width 4" means half the **general ALU FMA throughput**, not a dedicated unit.

The consequence is sharper than the tile argument. The ONLY reason to accept
`dequant -> threadgroup -> simdgroup_load` is to reach dedicated matrix hardware. Without it
we pay a forced memory round trip, ~18 barriers per K-slice, and software pipelining to hide
that latency, in order to emit the same FMAs a register-tile kernel issues directly. MLX's
"no `simdgroup_matrix`" is not a stylistic alternative - on this hardware it is the correct
choice.

Repack is evidence for that reading: it changes the weight **load pattern** and no arithmetic
whatsoever, and it is worth 9.3% e2e. An arithmetic-limited kernel would barely notice.
INFERRED - the limiter counters (`width4-limiter.md`, in progress) are the measurement.
