# The prod-width pass runs at half the memory roof, and the FFN is most of it

Status: ~~open~~ **superseded 2026-08-25 by `verify-width-instruction-economy.md`**,
which counter-profiles the width-7 skinny pass for the first time (registers, issue,
inflight, DRAM, threadgroup-L1 rates). This file's "50% of both roofs / does not overlap
the two" reading now has a measured mechanism, distinct from the mv family's: skinny's
instruction economy is fine (~5-6 dynamic instr per weight byte via MMA); it is bound by
the threadgroup-memory staging round-trip (3.2-3.8 tg-L1 loads + 2.6-3.1 stores per tick,
vs ~0 for every mv kernel, and the faster shape shows the higher rate - a saturated
staging port). The mv kernels are instead instruction-economy-bound. One separable finding: skinny at ffn_down
(m=5120) is grid-starved (160 threadgroups, 1.79 simdgroups/core inflight vs ~3 for the
larger shapes), worth ~12% on that shape alone.
Previous status: **open. Runs 1-2 are in (2026-08-23) and they CORRECT this file's own diagnosis.**
The pass really does cost ~2x, and the FFN really is half the round - but it is not a memory
utilization failure. `kernel_mul_mm_skinny` is at ~50% of the memory roof and ~50% of the
**arithmetic** roof at the same time, because its fixed 8-column tile makes arithmetic a
first-class cost and the kernel does not overlap the two. **The follow-up is in and it is a
refutation: `skinny-nr0-refuted.md` closes dispatch geometry AND activation re-read as the
mechanism, so the remaining answer is a register-tile kernel, not a tuning flag.** Measured at prod `e5c08dd94`,
clean tree, build 2026-08-23 20:13, `test-backend-ops perf` on MTL0, caffeinated.
Tools added: `perf/skinny-width-util.py`, `perf/skinny-roofline.py`.

## The observation (as opened, and it is what led here)

Every projection in the verify pass reads its weight matrix **exactly once per call**,
whatever the width. So the achieved bandwidth of a call is a direct utilization number, and
at the prod width it is bad:

| projection | weights | width 1 | **width 7** | % of 273 GB/s peak | ms/round |
|---|--:|--:|--:|--:|--:|
| `ffn_gate` + `ffn_up` | 50.1 MB | 247 GB/s | **139** | **51%** | 46.2 |
| `ffn_down` | 50.1 MB | 238 GB/s | **116** | **42%** | 27.7 |
| `attn_qkv` | 29.5 MB | - | 129 | 47% | 11.0 |
| `attn_q` | 35.4 MB | - | 132 | 48% | 4.3 |
| `attn_gate` | 17.7 MB | - | 125 | 46% | 6.8 |
| `attn_output` + `ssm_out` | 17.7 MB | - | 121 | 44% | 9.4 |
| `output` (lm_head) | 715.2 MB | - | 158 | 58% | 4.5 |

**At batch 1 the same weights stream at 87-90% of peak. At width 7 nothing exceeds 58%.**
The width-7 pass therefore costs about **twice what streaming the matrix requires**.

The two FFN projections alone are **73.9 of the 120.3 ms of MUL_MAT per round**, and MUL_MAT
is 76% of verify ticks, so **the FFN is roughly half the round** - all of that still stands.

~~and that factor - not any extra traffic - is what the 1.81x verify slope is made of.~~
**CORRECTED 2026-08-23 (run 2, below): the factor is not a bandwidth shortfall.** Reading
achieved bandwidth alone is only a utilization number when arithmetic is negligible, which
is true of a batch-1 mv call and false of this kernel. See "The corrected diagnosis".

## What was already known before this file (read these first)

**This file opened without reconciling against three notes that had already reached most of
the way, and run 1 below is a hypothesis someone had already written down.** Recorded here
so the next session does not repeat that:

- **`occupancy-next.md`, "The standing hypothesis: we do not fit the width".** Read from
  source 2026-08-23, marked INFERRED: `NR1 = 8` and `sb` is `NK x 8`, so the column tile is
  fixed at 8 and at `ne11 = 4` the kernel "still issues full 8x8
  `simdgroup_multiply_accumulate`, so half of every MMA is discarded". It also states the
  consequence - "our prod pick sits at width 7 because that is where the 8-wide tile is
  nearly full; the width choice is a workaround for the tile shape" - and it lists the test
  under "Open, and deliberately not yet attempted": *"Measure MMA utilization at `ne11` = 4,
  5, 7, 8 on the skinny kernel. If it tracks `nr1/8`, the tile-waste account is confirmed and
  the fix is a narrow-tile kernel, not a tuning flag."* **Run 1 is that measurement.**
- **`width4-skinny-ab.md`, "What this says about `simdgroup_matrix` on M4".** No matrix
  hardware, so the `dequant -> threadgroup -> simdgroup_load` round trip buys nothing, and
  MLX's refusal to use `simdgroup_matrix` "is not a stylistic alternative - on this hardware
  it is the correct choice".
- **`width4-limiter.md` + `ksplit-width34.md`.** "Nothing is saturated in any capture",
  therefore latency and dependency stalls rather than either throughput ceiling - confirmed
  behaviourally by the K-split, since more independent work along K pays at identical
  traffic. **Run 2b is the same reading, on the skinny kernel.**

So what follows is not a new direction. It is the standing hypothesis measured, priced, and
extended to the kernel prod actually runs - which matters because the counter route is
closed on this hardware and `width4-limiter.md` had already noted that the captures on disk
cannot test it (`w5-ffn_down-skinny` is the only skinny capture and has no matched-width
partner). Behavioural measurement is the only route left, and it is what run 1 is.

## Run 1: the tile-waste account, confirmed - and it is worse than "half the MMA"

`perf/skinny-width-util.py`, widths 1-8, two arms: prod routing
(`GGML_MV_NC=2 GGML_MM_SKINNY=5`) and skinny forced everywhere it is legal
(`GGML_MM_SKINNY=2`; the kernel cannot take width 1). Kernel confirmed per cell from the
pipeline-compile line, not assumed.

```
ffn_gate+up (17408 x 5120, 50.1 MB)          ffn_down (5120 x 17408, 50.1 MB)
w  prod kernel      us   GB/s %pk   skinny    prod kernel      us   GB/s %pk   skinny
1  mv             232.9  215.6 79%   (n/a)    mv             210.9  238.1 87%   (n/a)
2  mv_nc2         236.6  212.7 78%   362.3    mv_nc2         220.8  227.9 83%   414.9
3  mv_ext r1_3    283.2  178.0 65%   357.6    mv_ext r1_3    337.2  149.5 55%   420.4
4  mv_ext r1_4    350.0  144.3 53%   359.5    mv_ext r1_4    361.3  139.8 51%   424.6
5  mm_skinny      360.8  140.2 51%   360.4    mm_skinny      430.1  117.6 43%   432.1
6  mm_skinny      363.7  139.3 51%   364.0    mm_skinny      433.4  116.9 43%   434.2
7  mm_skinny      369.0  137.6 50%   368.1    mm_skinny      437.1  116.1 43%   438.7
8  mm_skinny      370.6  137.2 50%   371.0    mm_skinny      441.7  115.1 42%   441.3
```

**Skinny is flat in width**: 362 -> 371 us on `ffn_gate/up` across widths 2-8 (2.4% spread),
415 -> 441 on `ffn_down` (6.4%). That is the code, not a coincidence - the grid is
`((ne11+7)/8, (ne01+31)/32, 1)`, so it is **one threadgroup column for every width 1..8**,
and neither the K loop nor the simdgroup MACs read `ne11`. `nr1` only clamps the B-tile
column index and the final store. **The kernel computes all 8 columns whatever you asked
for.**

So the answer to the file's question 1 is a **cliff**, and it is at the routing boundary:
the entire fall from 87% to 43% happens the instant you leave the per-column kernel. There
is no slope in width to chase.

**This confirms `occupancy-next.md`'s standing hypothesis and sharpens it.** That file said
half of every MMA is discarded at width 4, and that width 7 is where the 8-wide tile is
"nearly full". The measurement says the tile is **always** full of issued MACs and the cost
is **identical** at width 2 and width 8, so width 7 is not nearly-full - it costs exactly
what width 2 costs. The width choice is a workaround for the tile shape, as that file said,
but the tile is not partly wasted at low width: it is wholly issued at every width, and the
waste is whatever fraction of the columns you did not want. **`nr1/8` tracks discarded MMA
exactly, which is what that file said would confirm the account.** It is confirmed, and
without an MMA counter - which gen 16 does not have.

## Run 2: the missing half is arithmetic, and it is not overlapped

The step run 1 exposes is the one this file did not price. `kernel_mul_mm_skinny` always does
`ne01 x 8 x ne00` MACs. At width 7 on `ffn_gate/up` that is 713 M MACs per call, which is not
free on this machine.

**M4 has no matrix hardware - already established, see `width4-skinny-ab.md` and
`m4-verify-kernel-proposal.md`.** `simdgroup_matrix` lowers to ordinary FMAs on the same SIMD
ALUs; Apple shipped per-core neural accelerators only with M5. The proof on record is better
than a spec search: `MXU Utilization` / `MXU Limiter` / `MXUOpsIssued` are **undefined for
gen 16** in Apple's own counter catalogue and resolve only for gens 17.4-20.3
(`width4-limiter.md`). Nothing below re-establishes that; it prices it.

**Arithmetic roof, measured: 6.96 TFLOPS = 3.48 T MAC/s** - the peak this machine reaches on
the *same* `simdgroup_half8x8` primitive skinny uses (`mul_mm`,
`test-backend-ops perf -p n=512`, best of the whole MUL_MAT set). Third-party estimates for
the 20-core M4 Pro run 8.1-9.2 TFLOPS and Apple publishes nothing, so 6.96 is 76-86% of them:
the right ballpark for a sanity check, and every conclusion below is checked against all
three (see the sensitivity note) so none of them turns on the number.

`perf/skinny-roofline.py --width 7`. `stream` is that shape's own width-1 mv call with its
one column of arithmetic subtracted, so it carries the shape's real access pattern rather
than a spec-sheet number; `arith` is the 8-column tile at the measured roof.

```
shape             MB    TGs |   stream    arith      sum | measured  vs sum  vs max |  ms/rd
ffn_gate+up     50.1    544 |    187.1    204.9    392.0 |    368.4     94%    1.8x |   47.2
ffn_down        50.1    160 |    185.7    204.9    390.6 |    438.3    112%    2.1x |   28.1
attn_qkv        29.5    320 |    108.2    120.5    228.8 |    237.4    104%    2.0x |   11.4
attn_gate       17.7    192 |     56.9     72.3    129.2 |    148.4    115%    2.1x |    7.1
attn_out        17.7    160 |     56.3     72.3    128.6 |    151.9    118%    2.1x |    9.7
attn_q          35.4    384 |    132.0    144.6    276.6 |    273.7     99%    1.9x |    4.4
lm_head        715.2   7760 |   2533.9   2922.8   5456.7 |   4640.6     85%    1.6x |    4.6
```

Two columns say it. **`vs sum` is 85-118%: every shape takes about as long as streaming its
weights and then doing its arithmetic, back to back.** And **`vs max` is 1.6-2.1x: that is
the "2x" this file opened on.** It is not a bandwidth shortfall. The two costs are within
10% of each other on every shape, and the kernel pays them in series.

The corroborating A/B is the big `mul_mm` on the same bytes
(`GGML_MV_EXT_MAX=1 GGML_MM_SKINNY=0 GGML_MV_NC=0 GGML_MM_MIN=1`), which computes a
32-column tile - 4x skinny's MACs, identical weight traffic:

```
              width 2   width 4   width 8      MACs vs skinny
skinny         352.8     356.9     371.4       1x
mul_mm         910.2     912.6     915.5       4x
```

Flat in width, as expected of a fixed tile, and 2.5x skinny's cost for 4x the arithmetic on
the same bytes. Additive model predicts 187 + 4*205 = 1007 us against 912 measured (91%),
the same mild overlap the table shows. A bandwidth-limited pair of kernels would have been
within a few percent of each other.

**Sensitivity.** The whole plausible roof range was run (`--roof-tflops 6.96 / 8.10 / 9.20`).
`vs sum` moves 94/101/107% on `ffn_gate/up` and 112/120/127% on `ffn_down`. **A higher roof
makes the arithmetic cheaper and therefore makes the no-overlap conclusion stronger, not
weaker.** Nothing here depends on picking the right TFLOPS number.

## Run 2b: the overlap that does exist tracks threadgroup count

> **REFUTED AS A MECHANISM 2026-08-23, same day, see `skinny-nr0-refuted.md`.** The
> correlation below is real; the causation is not. `NR0` was made a function constant and
> swept, and doubling `ffn_down`'s threadgroups (160 -> 320, 8 -> 16 per core) is **flat**:
> 431.9 us against 434.3. The reason is that **total simdgroup count is invariant under
> `NR0`** - the loader pins rows per simdgroup at 16, so the knob only repacks the same
> simdgroups into different threadgroups. Total work covaries with `ne01` in the sweep below,
> which is noted as a caveat there and turns out to be the whole story. **Threadgroup count
> is not the lever.**

Same probe, `k` fixed at 5120, sweeping `ne01` - so the shape's arithmetic intensity is
constant and only the dispatch changes. `perf/skinny-roofline.py --width 7 --sweep-m ...`:

```
       m    TGs  TG/core |   stream    arith      sum | measured  vs sum  vs max
    1024     32      1.6 |      9.1     12.1     21.2 |     69.9    330%    5.8x
    1280     40      2.0 |     10.3     15.1     25.3 |     72.9    288%    4.8x
    4096    128      6.4 |     32.2     48.2     80.5 |    111.1    138%    2.3x
    6144    192      9.6 |     56.2     72.3    128.5 |    148.2    115%    2.0x
   10240    320     16.0 |    109.1    120.5    229.6 |    236.8    103%    2.0x
   12288    384     19.2 |    131.9    144.6    276.5 |    274.3     99%    1.9x
   17408    544     27.2 |    182.7    204.9    387.6 |    372.7     96%    1.8x
  248320   7760    388.0 |   2548.7   2922.8   5471.5 |   4659.3     85%    1.6x
```

Monotone across two decades of threadgroup count. The kernel dispatches **64 threads (2
simdgroups) per threadgroup** and 32 dst rows, so `ffn_down` gets 160 threadgroups = 8 per
core, and it is the worst of the two FFN shapes; `lm_head` gets 388 per core and is the best
thing in the table. This is what the file guessed from `ffn_down` having the fewest
threadgroups, now measured with `k` held fixed.

**Honest caveat: total work covaries with `m` in that sweep**, and the two smallest rows are
launch-overhead dominated. The trend is still clean from `m=4096` (111 us, overhead is a few
percent) upward, which is most of a decade. The confound-free version of this test is the
`NR0` knob below, at one fixed shape.

## The corrected diagnosis

`kernel_mul_mm_skinny` is **jointly limited**: it streams ~50 MB and does ~700 M MACs, those
two cost within 10% of each other, and it does them in series rather than under each other.
The structure that forces the serialization is visible in the kernel: per 64-element K slice
every thread dequantizes into registers, writes the A tile to threadgroup memory, crosses a
barrier, `simdgroup_load`s it back, MACs, crosses another barrier. There is a software
pipeline (slice t+1 is prefetched into registers during slice t's MACs), which is the right
idea and is presumably why the numbers are as good as they are - but with only 2 simdgroups
per threadgroup and 5120 B of threadgroup memory there is not enough independent work
resident per core to hide a memory system behind an ALU.

Consequences worth stating plainly:

- **The 87-90% number is not a target.** Batch 1 reaches it because a 1-column mv does 1/8
  the arithmetic - 26 us of the 233. You cannot get an 8-column tile there, whatever you do
  to the memory path.
- **The realistic ceiling is `max(stream, arith)`, and it is still large.** Round MUL_MAT at
  width 7 is 112.5 ms measured against 58.5 ms at that ceiling: **54 ms on the table**, of
  which the two FFN projections are ~36 ms. No kernel hits its roofline, but this is a much
  bigger prize than the 15 ms this file opened with, and it comes from **overlap**, not from
  traffic.
- **Tile width is now a first-class cost, and prod pays it worst at width 5.** The 8-column
  tile computes 8/7 of what width 7 needs (14% waste) but 8/5 at width 5 (60%). The file's
  "12.5% tile waste, far too small to explain a 2x" was right that it is not the 2x, and
  wrong that it is small - it is 14% of a cost that is half the kernel.
- **This is a lead on the cross-framework gap, and it is the one `occupancy-next.md`
  already named.** dflash_mlx's `verify_m4` uses no `simdgroup_matrix` at all - a 4x4 float
  register tile, dequant inline, never staged to threadgroup memory. On the pricing here that
  buys them two things at once: they issue only the MACs they want, and they skip the shmem
  round trip and its barriers. Now quantified: **the discarded columns alone are ~77 us per
  `ffn_gate/up` call at width 5** (3 of 8 columns, 205 us of arithmetic), about 10 ms/round.
  That is a number `occupancy-next.md` wanted for its narrow-tile argument and did not have.
- **Tension to resolve, and it is with `width4-skinny-ab.md`.** That file infers from repack
  being worth 9.3% e2e that "an arithmetic-limited kernel would barely notice", i.e. that
  skinny is *not* arithmetic-limited. This file says it is limited by both, additively. Those
  reconcile - if the costs are additive rather than overlapped, the stream half retains full
  leverage and a load-pattern change still pays - but the *inference* does not survive: a
  repack win is not evidence against an arithmetic component. Struck there.

## What this does NOT contradict

`verify-slope-close.md` retired a "~20 ms verify-slope lever", and that stands as written: it
showed the slope is not *overhead* sitting beside the matmuls - matmul alone fills the budget,
so there is nothing to delete around it. This file makes a different claim: **the matmuls
themselves cost about twice `max(stream, arith)`**, which is a statement about the kernel, not
about the scaffolding. Those are compatible, and only the second one is still open.

## What is now dead

- ~~"the matmuls run at half the *memory* roof"~~ - they run at ~50% of both roofs at once.
  Any experiment aimed only at weight traffic or coalescing is aimed at half the problem.
- ~~"a 20% cut in the FFN is ~15 ms"~~ - still true arithmetically, but the ceiling is
  larger and the mechanism is overlap. Quote 54 ms of MUL_MAT headroom, not 15.
- ~~"Utilization vs width for skinny, 1 through 8" (experiment 1)~~ - **done, run 1.** Cliff,
  not slope; skinny is flat in width by construction. Do not re-run it. Same measurement
  closes `occupancy-next.md`'s "Measure MMA utilization at `ne11` = 4, 5, 7, 8" item.
- ~~"M4 has no matrix hardware" as a finding of this file~~ - **it was already on record in
  three places before this file was opened** (`width4-skinny-ab.md`,
  `m4-verify-kernel-proposal.md`, `width4-limiter.md`, the last with the counter-catalogue
  proof). It is cited here, not established here.
- ~~"`ffn_down` has the fewest threadgroups and the worst utilization, which is suggestive
  and not yet evidence"~~ - **it is evidence now** (run 2b), with the covariance caveat.
- **`ksplit-width34.md`'s mechanism transfers, and for a reason this file did not have.**
  It recovered 20% on `ffn_down` at width 4 purely by adding K parallelism. The reading there
  was "total lanes along K"; the reading here is that more resident independent work is what
  buys overlap. Same lever, and `mul_mm_skinny` splits K not at all.

## Experiment order (revised)

1. ~~**`NR0` and `nsg` as function constants on `mul_mm_skinny`, then sweep.** Prediction
   from run 2b: `ffn_down` moves from 112% of sum toward ~95%, i.e. 438 -> ~370 us, about
   -4.4 ms/round.~~ **DONE AND REFUTED same day, `skinny-nr0-refuted.md`.** Built, correct at
   1154/1154 for NR0 = 16/32/64/128, and **NR0=32 was already the optimum**. Doubling
   threadgroups is flat on `ffn_down` and 4% worse on `ffn_gate/up`; total simdgroup count is
   invariant under the knob. That run also refutes the activation-re-read reading below:
   doubling B re-read to 178.3 MB at identical weight traffic moves `ffn_down` -1.3%, and the
   config with the *fewest* re-reads is the *slowest*. **There is no tuning win in this
   kernel**, which is the useful half - it was cheap and it closes the last alternative to
   rewriting.
2. ~~**Overlap directly: more simdgroups per threadgroup, or fewer barriers per K slice.**~~
   The `nsg` half is dead with 1 - it is the same function constant and it moves nothing.
   ~~The **barriers** half is not a tuning question and survives only inside 3.~~
   **REFUTED AND PARTLY VINDICATED 2026-08-24, `skinny-tpr-bsplit.md`.** The `nsg` half was
   reached properly at last (`GGML_MM_SKINNY_TPR`, the loader's threads-per-row, which is what
   sets rows per simdgroup): **4 simdgroups of 8 rows costs +7.7% on `ffn_gate+up`, 1 of 32
   costs +9.9%, and the shipped 2 of 16 is the optimum.** The barriers half did not need a new
   kernel after all - the **B-tile load between them was pinned at 32 threads** whatever the
   threadgroup size, and spreading it (`GGML_MM_SKINNY_BSPLIT`) is -1.4% to -2.6% per call on
   every projection and **+1.0 to +1.6% e2e**, the first measured win on this kernel.
3. ~~**THE ONE LEFT: the register-tile kernel.**~~ **REFUTED 2026-08-24 at the prod width -
   see `ext-at-width7-refuted.md`. That kernel already exists: it is `mul_mv_ext`, and at
   width 7 the best-tuned config is 596 us against skinny's 367, a 1.63x loss.** The register
   tile's accumulators and its `nr0*r1ptg*log2(nxpsg)` reduction both scale with verify width,
   so the family wins below width 5 and loses above it; the crossover is at 5 and is
   shape-dependent. Building a new kernel of that shape would reproduce the result. **Keep
   the original text below for the reasoning, which is still right about width 4:**
   *(Quant target ~~settled 2026-08-23~~
   **REOPENED 2026-08-24.** That day's call to build for Q4_0 rested on a clean Q4_K_M worth
   only +1.45 pp. `weight-quant-kld.md` then measured **UD-Q4_K_M at +5.82 pp for FEWER bytes**
   - a well-chosen 4-bit body is worth ~+4.6 pp, the largest quality lever in this project, and
   it contains **zero Q4_0 tensors**. So the format the kernel targets is now a real decision
   worth several points of top-token agreement, not a free choice. **Hybrid A resolved it
   2026-08-24: +3.89 of UD's +5.82 pp is in the FFN specifically**, and taking small tensors
   off the Q4_0 fast path is nearly free (-3.2% batch-1 at 82.8% coverage) while taking the FFN
   off it costs -39%. **So the kernel target is narrow, not wide: make `ffn_gate`/`ffn_up`/
   `ffn_down` fast on ONE good 4-bit format.** Three tensor shapes, ~3.9 pp of quality, and the
   rest of the model can stay K-quantized at ~3%.)* No `simdgroup_matrix`, inline dequant, never
   staged to threadgroup memory, K-split, narrow column tile. With 1-2 dead, dispatch geometry
   and activation re-read are both excluded and the design itself is what is left - the
   `dequant -> threadgroup -> simdgroup_load` round trip plus two barriers per K slice, paid
   to feed a primitive that lowers to ordinary FMAs on hardware with no matrix unit. This is
   `occupancy-next.md`'s narrow-tile item and `width4-skinny-ab.md`'s conclusion, and it is
   the shape of their `verify_m4`. **Read it, benchmark it, do not copy it.** Target width 4,
   not 7: that is where the arithmetic waste is 102 us of 205 rather than 26, where the stated
   goal is, and where ~32 ms of the 46 ms cross-framework gap sits in the FFN roofline.
4. **Price `verify_m4`'s arithmetic against ours.** We have the capture
   (`mlx-cycle-capture.md`). Their M=4 tile at their block 4 vs our 8-column tile at width 5
   is a 2x arithmetic difference on the same bytes, and this file's model says arithmetic is
   half the cost. If that survives contact with their trace it explains a large part of the
   95.00 vs 141.0 shelf.
5. **K-split**, still open, but as a feature *of* the kernel in 3 rather than a patch to
   skinny. `NR0` reached the same resource more cheaply and found nothing, so there is little
   reason to bolt K-split onto a design that is being replaced.

Dropped from the old order: "read the batch-1 kernel for what it does right". Run 2 answers
it - it does 1/8 the arithmetic. There is no transferable trick there.

## Trap found while opening this file (unchanged, still true)

**`GGML_MV_REPACK` is silently inert in `test-backend-ops`.** `try_repack_q4_0` requires
`src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
(`ggml-metal-ops.cpp:2532`), and `test_mul_mat` only overrides the one-argument
`build_graph(ctx)`, so its src0 never lands in `ctx_weights` and the buffer is never marked.
A repack A/B there returns a flat result **because the flag did nothing**, and the pipeline
name is the tell: no `_di`. That is why repack's only evidence is e2e (+9.3%,
`width4-skinny-ab.md`). To measure it per shape, `test_mul_mat` needs the two-argument
`build_graph` override first.

**This matters more after run 2 than before it.** Repack's `_di` layout changes only the
weight-load path, so on this file's model it can attack at most the `stream` half - and on
`ffn_gate/up` `stream` is 187 us of a 368 us call. That bounds what repack can be worth per
shape and makes the fix above worth doing before any further repack work.
