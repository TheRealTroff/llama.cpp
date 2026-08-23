# Width 4: the one operating point they have and we do not

Status: **open**. Opened 2026-08-22 from `mlx-cycle-capture.md` open stubs 1 and 2, plus a
new kernel-level measurement taken the same day (below).

Where it stands after runs 4-6 (2026-08-23), so a new session does not re-run these:

- **The shelf is still unexplained.** Three candidate causes are now measured and dead: the
  register tile (run 2), `nr0` (run 1), and the f16y convert dispatch (run 4). The tile is
  off the table, `ext` is the right family (run 3, and run 7 closes the family question
  against plain `mul_mv` too), and the convert is a win, not a tax.
- **`nxpsg=16` at widths 3-4 is the one live *tuning* lever**, worth **-1.5% to -1.7%** on a
  `llama-bench` pass at N=3/N=4 (run 6) - about a third of what run 3's per-shape table
  implied. ~~the one live lever~~ **Re-scoped 2026-08-23: it is a tuning lever against a
  1.48x deficit, so it cannot close the gap on its own.** See "The tile does not fit the
  width" below and the new section in `occupancy-next.md`. It needs the `ne00 % 256 == 0` guard kept; that condition is correctness, not a
  heuristic (run 6). Code lives on branch `metal-mv-ext-nxpsg-w34`, unmerged.
- **The e2e question is open and its 2026-08-23 attempt is invalid** - the n6 control failed
  at -6.1% on a byte-identical workload. Do not quote an e2e number from that run.
- **None of this can move the prod pick**, which sits at n6 / width 7 / skinny. See the last
  section of run 6.
- **Run 5 is retracted** (same day, by the replay counters): the f16y gate for q4_0 is
  **16.78M**, not 8M, so the "band where f16y does nothing" was just the gate working. The
  wrong-dimension hypothesis is untested, not refuted - do not treat the gate as known-bad.
  `attn_q` is below that gate, so it never had f16y and its run-4 row is a control. It does
  still lose on `nxpsg=16`, and that remains unexplained.
- Ten captures spanning the cliff are archived at
  `~/play/kvquant-experiments/traces/aug23/` with headless dumps, waiting on replay clicks.
  Replay output is auto-archived by `perf/watch-replays.sh` - do not let it die in `/tmp`
  again.

**Read the width convention first.** Their block *b* verifies *b* columns; our depth *d*
verifies *d+1* (`spec_epoch.py:2247-2257` vs `slope-sweep.md:13`). Everything here is stated
in **width**. Their block 4 == our depth 3.

## Where the gap actually is

Pinned measurements, both sides, matched by width:

| width | our depth | our kernel | our ms/round | theirs, pinned | ratio |
|---|---|---|--:|--:|--:|
| 4 | n3 | `mul_mv_ext` (`nxpsg=8, nr0=2, chpt=1`) | 141.0 | **95.00** | **1.48x** |
| 5 | n4 | `mul_mm_skinny` | 144.9 | 137.26 | 1.06x |
| 7 | **n6 (prod pick)** | `mul_mm_skinny` | 149.8 | - | - |

**The entire cross-framework gap is one width.** We are level at width 5. Their controller
sits at width 4 for 82% of cycles (`cycles_by_block={1:1, 4:81, 5:17}`); our prod pick sits
at width 7. Pinned, their best is **32.556 +/- 0.007 t/s** (`block4-shelf-probe.md`) against
our 25.04.

## What their width-4 kernel is

`custom_kernel_verify_m4_ksplit_np_kp{2,4}_gs64_bf16`, built by
`_build_kernel_m4_ksplit_np()` at `dflash_mlx/verify_qmm.py:193-334`. It is plain MSL with a
single template parameter `T`, no MLX dependency in the body, and **no `simdgroup_matrix`**:

- 4x4 register tile per thread: `BN=4` output columns x `M=4` activation rows = 16 `float`
  accumulators. Weight reuse factor 4, activation reuse factor 4.
- K split across simdgroups within the threadgroup (`K_PARTS` = 2 for N >= 4096, else 4);
  lanes stride by 32 packs, so both operands are coalesced.
- Dequant inline in registers, never staged to threadgroup memory. Per iteration per thread:
  32 dequants + 128 accumulate FMAs.
- Reduction is 16x `simd_sum`, then a `threadgroup float partial[]` reduce across parts.
- Routing: `m == 4` exactly, bits == 4, `N % 4 == 0`, `K % 32 == 0`, `N < 100_000`
  (`verify_linear.py:55-88`, `verify_qmm.py:30-31`). Only M=4 and M=16 have custom kernels;
  every other M falls to stock `mx.quantized_matmul`.

## The tile does not fit the width (INFERRED 2026-08-23, not measured)

Read from source, not from a run - label it as such until the counters land.

`kernel_mul_mm_skinny` (`ggml-metal.metal:11880`) accumulates into `simdgroup_half8x8`, so
its column tile is fixed at 8 by the hardware instruction: `sb` is `NK x 8`, `NR1 = 8`. At
`ne11 = 4` the kernel clamps `nr1 = 4` but still issues full 8x8
`simdgroup_multiply_accumulate`, and the dispatch is `((ne11 + 7)/8)`
(`ggml-metal-ops.cpp:2723`) - **we pay for 8 columns to compute 4, so half of every MMA is
discarded.** That is the likely reason `GGML_MM_SKINNY=5` excludes width 4 at all: at 4 the
alternative is `mul_mv_ext`, which loses weight reuse instead. **Both of our width-4 paths
are wrong-shaped**, and our prod pick sits at width 7 because that is where the 8-wide tile
is nearly full - the width choice is a workaround for the tile, not a property of the model.

Theirs routes on `m == 4` **exactly** and uses **no `simdgroup_matrix`**: 4x4 float register
tile, reuse 4 on both operands, dequant inline in registers. An exact fit for the width their
controller wants.

If this is right it predicts MMA utilization tracks `nr1/8` at `ne11` = 4, 5, 7, 8 - which is
a measurement, and is what the counter work exists to take. It also predicts the fix is a
narrow-tile width-4 kernel, not a flag. **Do not copy their kernel** - read it, benchmark it
(fork rule). Nothing here is confirmed; do not quote it as a finding.

## Kernel-level measurement, 2026-08-22 (new)

`test-backend-ops perf -o MUL_MAT -b MTL0 -p "k=14336"` against MLX at `K=14336, N=4096`
(33 MB of weights). **MLX absolute levels carry a ~140 us fixed per-call overhead** in this
harness (a 0.1 MB matmul measures 148 us), so compare *marginal* cost of widening, where the
offset cancels. Shape is at the SLC boundary, so this is not a valid absolute-bandwidth
measurement - but it does reproduce the real N=3 cliff, so it is a valid iteration proxy.

| width | ours, `mul_mv_ext` | stock MLX | their `verify_m4` |
|---|--:|--:|--:|
| 1 | 135.0 us | - | - |
| 2 | 156.5 us | - | - |
| 3 | 203.0 us | - | - (no m3 kernel) |
| 4 | 237.0 us | 342.7 us | 309.5 us |
| 5 | 279.2 us | - | - |
| 8 | 438.6 us | - | - |

**Cost of widening 1 -> 4: ours +102 us, stock MLX +85 us, their m4 kernel +52 us.** They
widen at about half our marginal cost, which is the shelf. Two things this also settles:

- **Their bespoke kernel is not exotic.** On real model shapes it is only **1.03-1.20x**
  over stock `mx.quantized_matmul` (ffn_gate/up 1.10x, ffn_down 1.20x, attn_q 1.03x,
  gdn_qkv 1.07x). Most of their width-4 advantage is that stock MLX widens cheaply.
- **Even at width 3, where they have no custom kernel at all**, stock MLX widens more
  cheaply than our ext does (+44 us vs our +68 us). Our 3-4 corner is bad independently of
  their kernel.

### Better: the same measurement on a real verify shape (2026-08-22, prod env)

The 33 MB shape above sits at the SLC boundary. `ffn_down` (`m=5120, k=17408`, 50 MB) is one
of the projections the verify pass actually runs, and it is now a perf case on prod. Under
**prod routing env** it reproduces the whole-model width curve shape exactly:

| width | ours, us | marginal | our kernel |
|---|--:|--:|---|
| 1 | 203.31 | - | `mul_mv` |
| 2 | 212.31 | +9.0 | `mul_mv` nc2 - nearly free |
| 3 | 325.75 | **+113.4** | ext `nr0=2, chpt=1` - **the cliff** |
| 4 | 348.11 | +22.4 | ext `nr0=2, chpt=1` |
| 5 | 421.58 | +73.5 | skinny mm |
| 6 | 419.58 | -2.0 | skinny mm |
| 7 | 424.80 | +5.2 | skinny mm |
| 8 | 427.01 | +2.2 | skinny mm |

Same signature as llama-bench's 73.0 / 73.8 / 101.5 / 111.5 / 119.0 / 120.9 / 123.1 / 124.1:
free at 2, cliff at 3, flat from 5. **So this one shape is a valid, ~2 minute proxy for the
whole-model curve** - iterate on it, confirm on llama-bench.

Widening 1 -> 4 at this shape: **ours +144.8 us, their `verify_m4` +58.1 us = 2.49x.** More
pronounced than at the 33 MB shape, and this is the shape that matters.

**Caffeination status of these numbers.** Taken with a plain `test-backend-ops` invocation,
not under `caffeinate` (the `perf/run-*.sh` harnesses only gained it later the same day).
Nothing was *suspended*: `pmset -g log` reports **`Total Sleep/Wakes since boot: 0`** over the
machine's full 2.5-day uptime, and the cold width-1 point reproduced its archived value to
1.3% (1185.6 vs 1201.8 us).

**That rules out sleep, and only sleep.** It does not address clock/power throttling with the
display off, which is a separate mechanism and which the display was, for most of the day.
Every number in this file was taken in that state, so treat them as provisional levels. The
*ratios* are the load-bearing part and are far more robust, since both sides of each
comparison were measured in the same state minutes apart.

~~**Re-take caffeinated when the width-4 work starts**, with the depth-3 round decomposition,
so baseline and change share one discipline.~~ **Settled 2026-08-22.** The `ffn_down` sweep was
re-taken under `caffeinate -dimsu` and reproduces every archived width to within 4%
(203.3/212.3/325.8/348.1/421.6 -> 212.3/221.1/329.6/347.3/415.5). Display-off throttling was
never a factor at this shape. The numbers above stand as levels, not just ratios.

### Two traps in this harness, both cost a run

- **`test-backend-ops` does not read the prod env by default.** With no env it routes
  everything to `ext` and the curve is a *different shape* (widths 5-8 keep climbing:
  399/515/637/638 instead of flattening). Always run it as
  `GGML_MV_NC=2 GGML_MM_SKINNY=5 ./build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p ...`
  or the numbers are not comparable to anything in this directory.
- **The 302 MB cold-streaming case cannot see the width 3-4 weakness.** Its `ne01` is 16384,
  so `nr0 = (ne11 >= 5 || ne01 >= 8192) ? 4 : 2` already gives **4** - verified from the
  pipeline names (`..._r1_3_nsg=2_nxpsg=8_nr0=4`). It is the right instrument for absolute
  DRAM bandwidth (width 1 measures 254.7 GB/s = 93% of peak) and the wrong one for this
  investigation. Use the model shapes.

### This retires a stale claim

~~`mv-bandwidth-probe.md` (branch `metal-mv-wideload`, 2026-08-21): "at n=4 we are already
ahead" (llama.cpp 1956.6 us vs MLX 2060.9 us, 5.3%).~~ **That benchmarked
`mx.quantized_matmul`, which MLX bypasses at M=4.** It was written 32 hours before the
capture found `custom_kernel_verify_m4`. Together with `mv-nc-cliff-probe.md`'s "parity, not
a win", it is why widths 3-4 were treated as a closed line. Reopen them.

## Why we are slow at width 4, precisely

> **Refuted 2026-08-22 by run 2 - read that section before this one.** The section below
> concludes "it is the register tile". We then built their tile (4x4, 16 accumulators, zero
> spill) and it is *slower* at width 4 than the 2x4 we ship. The parameter analysis below is
> still accurate as a description of what the kernel does; its *conclusion* is wrong.

**It is not weight traffic.** `mul_mv_ext` already reuses each loaded weight across all
`r1ptg` columns and streams the matrix exactly once (grid y-dim is 1 at ne11 <= 4). There is
no redundant DRAM read to remove.

It is the register tile. Confirmed empirically from the pipeline names at ne11=3 and 4:
`kernel_mul_mv_ext_q4_0_f16_r1_{3,4}_nsg=2_nxpsg=8_nr0=2`. Three parameter choices stack:

- `nxpsg=16` requires `ne11 < 3` (`ggml-metal-ops.cpp:2770-2776`), so widths 3-4 lose the
  wide variant that makes width 2 nearly free.
- `nr0 = (ne11 >= 5 || ne01 >= 8192) ? 4 : 2` (`:2805-2809`), so 2 rows per thread.
- `chpt` throttle: the f16y flavour is `(nr0*r1ptg >= 6) ? 1 : 2` (`ggml-metal.metal:4835`),
  so both widths land on `chpt=1`.

Net: ~8 dequants and ~32 FMAs per loop iteration, against their 32 and 128. **We do a
quarter of the work per iteration.** At width 1 we run at 92% of DRAM peak; at width 4 the
kernel is latency-bound, not bandwidth-bound.

## Experiments, in order

1. **Measure our depth 3 round cost properly** (open stub 1 from `mlx-cycle-capture.md`).
   One more arm on `run-slope-sweep.sh`. The 141.0 above is from the existing n3 row; what is
   missing is a round decomposition at that depth (verify / drafter / overhead split), which
   is what decides how much of the 46 ms gap is kernel and how much is drafter.
2. ~~**Free, no code: confirm the current routing is actually best at widths 3-4.**~~
   **Done - run 3 (ext vs mv-nc vs skinny) and run 7 (ext vs plain `mul_mv`). All four
   families measured, `ext` wins at both widths.** They are
   left to `ext` by *configuration*, not by code - `GGML_MV_NC` caps at `min(env,4)` with
   nc3/nc4 kernels already compiled, and `GGML_MM_SKINNY`'s floor is 2. A three-arm A/B
   (ext vs mv-nc vs skinny +/- repack) costs one run. Prior evidence says ext wins
   (`mv-nc-cliff-probe.md`, `dflash-vs-mtp-uniform.md:61-74`), so this is confirmation.
   It was not: run 3 found `nxpsg`, and run 7 found that plain mv ties `ext` on `ffn_down`
   at width 3 while moving half the bandwidth.
3. ~~**The real one: `nr0` 2 -> 4 at ne11=4.** One line in the heuristic at
   `ggml-metal-ops.cpp:2805-2809`.~~ **Done 2026-08-22 - refuted at width 4, see below.**
   It needed no code at all: `GGML_MV_EXT_NR0` has been a runtime override since the
   `metal-mv-ext-nr0` work (`ggml-metal-ops.cpp:2791`, `results.md:54`).
4. ~~**Screen the tile grid offline before building anything.**~~ **Done for this grid, see
   below.** `skills/metal-kernel-prescreen` + `perf/agx-spill-probe.py`. **Do not trust
   in-tree register comments while doing this**: `ggml-metal.metal:4311` is already
   demonstrated false.
5. **Toolchain, if 3-4 stall:** their kernel compiles standalone with `xcrun metal`, so
   `metal-objdump` / `metal-nm` will diff their register allocation against ours directly.

## Run 1 (2026-08-22, caffeinated): nr0 2 -> 4 is not the lever

Zero code. `GGML_MV_EXT_NR0=4` against default, on the four 27B verify projections at the
two widths that route to `ext` (n <= 2 is `mul_mv` nc2, n >= 5 is skinny - both unaffected,
which is the control):

| shape | width | nr0=2 | nr0=4 | delta |
|---|--:|--:|--:|--:|
| ffn_gate/up (m=17408) | 3 | 273.90 | 274.84 | +0.3% |
| ffn_gate/up (m=17408) | 4 | 325.67 | 321.31 | -1.3% |
| **ffn_down (m=5120,k=17408)** | **3** | 324.38 | 299.29 | **-7.7%** |
| **ffn_down (m=5120,k=17408)** | **4** | 351.62 | 385.97 | **+9.8%** |
| gdn_qkv (m=6144) | 3 | 106.82 | 104.37 | -2.3% |
| gdn_qkv (m=6144) | 4 | 124.90 | 122.60 | -1.8% |
| attn_q (m=3072) | 3 | 60.87 | 64.15 | +5.4% |
| attn_q (m=3072) | 4 | 74.09 | 73.81 | -0.4% |

`ffn_gate/up` has ne01 = 17408 >= 8192, so it is *already* nr0=4 in both arms - its +-1.3%
is the run-to-run noise floor. Only `ffn_down` moves past it, and it moves **both ways**:
better at width 3, worse at width 4. **Width 4, the target, is the one that regresses.**

### Why, from the spill probe

`kernel_mul_mv_ext_q4_0_f16_r1_{3,4}` at nsg=2, nxpsg=8 (spill bytes/thread):

| kernel | nr0=2 | nr0=4 |
|---|--:|--:|
| `r1_3` (width 3) | 0 | **16** |
| `r1_4` (width 4) | 0 | **32** |

`chpt` is 1 in all four cells (`nr0*r1ptg >= 6`), so this is not the documented chpt=2
cliff - `results.md:271-272` was right that "nr0=4 only works at chpt=1", and it is not the
constraint here. The 4x4 tile spills on its own. Width 3 pays 16 B and still nets -7.7%
because the extra rows/iteration outweigh it; width 4 pays 32 B and loses.

**The cause is addressing, not accumulators**, exactly as experiment 4 pre-registered.
`ggml-metal.metal:4851-4862` holds `xq[NR0MAX]` *and* `y8[r1ptg]` live as running device
pointers (`xq[k] += adv` in the inner loop, so they cannot be rematerialized). At
nr0=4, r1ptg=4 that is 8 live 64-bit pointers = ~16 GPRs of pure addressing, on top of 16
accumulators and 8 `float4` of `lx`.

## Run 2 (2026-08-22): the spill is fixed, and it does not buy width 4

Branch `metal-mv-ext-spill`. `kernel_mul_mv_ext_q4_f16y_impl_v2` + `GGML_MV_EXT_V2=1`, q4_0
f16y only. Same idea as `fe0429daf`: one base pointer per operand plus a shared running
index, instead of the live `xq[nr0]` / `y8[r1ptg]` arrays.

**The prescreen gate passed cleanly** - and the v2 code is *smaller*, so recomputing the row
offsets costs less code than maintaining the pointer arrays did:

| kernel | v1 text | v1 spill | v2 text | v2 spill |
|---|--:|--:|--:|--:|
| `r1_3` nr0=2 | 3226 | 0 | 3034 | 0 |
| `r1_3` nr0=4 | 5290 | **16** | 4946 | **0** |
| `r1_4` nr0=2 | 3850 | 0 | 3564 | 0 |
| `r1_4` nr0=4 | 6300 | **32** | 5824 | **0** |

Correct: 1154/1154 MUL_MAT tests pass on MTL0 at both nr0=2 and nr0=4.

`ffn_down` (5120 x 17408), mean of 3 interleaved reps, within-arm spread < 1%:

| width | v1 nr0=2 | v1 nr0=4 | v2 nr0=2 | v2 nr0=4 |
|---|--:|--:|--:|--:|
| 3 | 337.9 | **305.6** | 332.7 | 306.5 |
| 4 | **358.9** | 391.3 | 357.0 | 363.9 |

**Unspilling did exactly what it was supposed to and it was not enough.** At width 4 it
recovers most of the nr0=4 penalty (391.3 -> 363.9, so the 32 B spill was worth ~7%), but
the unspilled 4x4 tile still **loses to the nr0=2 baseline** (363.9 vs 358.9). At width 3
v2 adds nothing over v1 at nr0=4 (306.5 vs 305.6) - that 16 B spill was costing nothing
measurable.

### What this refutes

~~"It is the register tile."~~ **Refuted 2026-08-22.** We now have their tile shape - 4x4,
16 accumulators, zero spill, verified offline - and at width 4 it is *slower* than our 2x4.
The width-4 gap (they widen 1 -> 4 at +58 us, we at +145) is **not** explained by tile shape
or by register pressure. Whatever the shelf is, run 2 says it is somewhere else. Do not
reopen the tile as the explanation without new evidence.

The `nr0=4` win at width 3 is real and reproducible (-9.5%) but it long predates v2, is
`ffn_down`-only (gdn_qkv flat, attn_q worse), and does not touch the width the controller
actually sits at.

### v2 itself is a safe, small positive

At the shipping `nr0=2` it is -0.5% to -1.5% across widths 3-4 and never worse, with less
code and no spill anywhere in the grid. It is worth keeping on its own merits, but it is
**not** the width-4 answer and should not be sold as one.

### Methodology note: re-baseline per session

Within-session repeats agree to < 1%. *Across* sessions the same arm drifted ~3% (width 3
`v1 nr0=2` read 324-330 earlier in the day, 337.9 here). Always re-measure the baseline arm
in the same session as the change; do not diff against a number from another session.

## Run 3 (2026-08-22): `ext` is the right family, and `nxpsg` is the live lever

This is experiment 2, which the file had written off as "confirmation" and never ran. It was
not confirmation.

### Why we are on `ext` at widths 3-4

By *configuration*, not by measurement. `mv_nc_route` needs
`ne11 <= min(GGML_MV_NC, 4)` and skinny needs `ne11 >= max(2, GGML_MM_SKINNY)`
(`ggml-metal-ops.cpp:2633, 2683`). With the prod env - `GGML_MV_NC=2 GGML_MM_SKINNY=5` -
widths 3 and 4 fall in the gap between them and land on `ext` by default. Both thresholds
were set by whole-model sweeps; neither was ever A/B'd *at* widths 3-4.

### The A/B, and `ext` wins by a lot

`ffn_down`, 3 reps, all three families available and already compiled:

| width | `ext` (prod) | `mul_mv` nc | `mul_mm_skinny` |
|---|--:|--:|--:|
| 3 | **333** | 455 (+37%) | 417 (+25%) |
| 4 | **361** | 516 (+43%) | 420 (+16%) |

So the routing is correct and the gap is not a family-choice mistake. Two things fall out:

- **`GGML_MM_SKINNY=5` is at its optimum, and now we know why.** Skinny costs ~418 us at
  width 3 *and* width 4 *and* width 5 - it is nearly width-independent. It therefore wins
  only once `ext` climbs past ~420, which happens at width 5. The threshold is the crossover.
- **The nc cliff reproduces.** nc3 at 455 vs ext at 333 is +122 us, matching the fixed
  ~112 us at NC>=3 that `ead90eb62` diagnosed and declined to fix.

### The lever is `nxpsg`, not `nr0`

`nxpsg=16` is gated on `ne00 % 256 == 0 && ne11 < 3` (`:2770-2776`), so widths 3-4 lose the
wide variant that makes width 2 nearly free. Unlike `nr0` there is no kernel limit here -
`nxpsg` is a function constant and `GGML_MV_EXT_NXPSG` already forces it. Forcing 16:

| shape | width 3 | width 4 |
|---|--:|--:|
| ffn_gate/up | **-7.4%** | +0.1% |
| ffn_down | **-5.0%** | **-3.2%** |
| gdn_qkv | -0.6% | **-2.5%** |
| attn_q | +2.5% | **+8.3%** |

**Shape-dependent, and it is the first lever that helps at width 4 at all.** It wins on the
two large FFN projections and loses on `attn_q`, the smallest shape (8.8 MB, m=3072). So it
is a per-shape routing question, not a blanket flip of the `ne11 < 3` gate. This does *not*
contradict run 2: `nxpsg` changes how threads are laid out along the row, not the register
tile.

`nr0=4` does not stack with it: at width 3 `nxpsg=16` alone gets ~320 and adding `nr0=4`
gets ~314 (inside drift), and at width 4 `nr0=4` still costs ~12 us on top. `nr0` is done.

### Why `nxpsg=16` wins: it is occupancy, from the capture

Confirmed structurally by pointing our own GPU capture at the kernel for the first time
(`toolchain-isa-probe.md`, on-device section). Dispatch geometry at `ffn_down` width 4:

| config | threadgroups | threads/threadgroup | K iterations per thread |
|---|--:|--:|---|
| `nxpsg=8` (prod) | **320** | 64 | stride `chpt*nxpsg` = 8 |
| `nxpsg=16` | **640** | 64 | stride 16, so **half** |

`r0ptg = nypsg*nsg*nr0` and `nypsg = 32/nxpsg`, so doubling `nxpsg` halves rows per
threadgroup and doubles the grid: 5120/16 = 320 against 5120/8 = 640. Threads per
threadgroup is **unchanged at 64**. Same total work, twice the independent work units, and
each thread's serial K chain cut in half.

**That is the shape of a parallelism/latency limit, not a compute or register limit** - and
it is exactly consistent with run 2, where giving the kernel a *bigger* per-thread tile made
width 4 worse. The kernel does not want more work per thread; it wants more threads in
flight. The file's original "at width 4 the kernel is latency-bound, not bandwidth-bound"
line survives run 2 even though the register-tile conclusion attached to it did not.

**Confirmed 2026-08-23 by replay profiling** (`skills/metal-gpu-profile`), which measures
the register pressure this argument depends on rather than inferring it. Same kernel, same
width 4, only `nxpsg` differing:

| | nr0=2 nxpsg=8 | nr0=2 nxpsg=16 |
|---|--:|--:|
| Temporary register count | 73 | **73** |
| Spilled bytes | 0 | 0 |
| Instruction count | 453 | **477** |
| FP32 instructions | 124 | 132 |
| Device loads / stores | 8 / 8 | 8 / 8 |

**Register pressure is identical and the instruction count is slightly *higher*, yet
`nxpsg=16` is faster.** So the win is positively *not* register pressure or reduced work -
both are excluded by measurement now, not merely unmeasured. What is left is the dispatch
geometry: twice the threadgroups, half the serial K chain per thread. The occupancy reading
stands.

Caveat: this pins the mechanism, not the magnitude. A direct occupancy counter would size
it; that is still unavailable (see `toolchain-isa-probe.md`).

### Drift got worse over the session

`ffn_down` width 3 at the prod config read 336, 340, 362 and 363 across four blocks tonight,
an ~8% spread, on a machine that had been benchmarking continuously for hours. The
`nxpsg=16` arm was far steadier (318-320). **Only compare arms measured inside the same
block**, which is how the tables above are built. The ~3% figure recorded under run 2 is a
floor, not a bound.

### Next

~~Confirm on llama-bench before touching the gate - the per-shape split means the net effect
is an aggregation question that arithmetic over these tables will not settle honestly.~~
**Done 2026-08-23, see run 6.** The first attempt at it was invalid; read run 6 before
quoting any aggregate number for `nxpsg=16`.

## Measurement conditions for runs 4-6 (2026-08-23) - read before quoting a level

**The machine was standing in direct sun for all of these runs**, on AC and caffeinated.
Johan flagged it mid-session and moved it afterwards, so nothing here was taken in a
controlled thermal state. Two things follow, and the second is the one that matters:

- **Every absolute number in runs 4-6 is provisional.** The size of it: the n6 e2e control
  reproduced the archived prod-pick output sha (`3776c0adb7ee`) and acceptance (41.3%)
  *exactly* while running **4.5% slower** than the archived 22.899 t/s. Identical work,
  identical output, slower machine. That is the drift, measured rather than assumed.
- **The ratios are load-bearing and are built to survive it.** Every A/B here alternates its
  arms inside one block, and every one carries controls that must read flat - widths 1, 2, 5
  in run 4, the two below-gate ne01 rows in run 5, six control cells in run 6, n6 in the e2e.
  A thermal excursion that moved a verdict would have to move the treated cells without
  moving the controls sitting minutes away from them.

This is the same levels-vs-ratios split the "Caffeination status" note above draws, with a
stronger reason for it. When these get re-taken in a controlled state, expect the levels to
shift by a few percent and the deltas to hold.

## Run 4 (2026-08-23): the width 3-4 path encodes two dispatches, and the second one pays

Found by dumping the one capture that survived the previous session,
`/private/tmp/perf-metal-67662.gputrace` (50 MB, width 4 `ffn_down`, prod config).
**Everything else that session produced was in `/tmp` and is gone**: the replay output under
`/tmp/com.apple.gputools.profiling` and the 95 MB oMLX capture `/tmp/dflash-b4.gputrace`.
Only the eight SUMMARY fields transcribed into run 3 survive, and `gpuprofiler-stats.py
--all` was never run on them. That is why run 4's capture set is archived out of `/tmp`.

The dump shows each MUL_MAT node encoding **two** dispatches, not one:

```
"MUL_MAT" +- kernel_cpy_f32_f16               {272,1,1} x {256,1,1}
          +- kernel_mul_mv_ext_q4_0_f16_r1_4  {320,1,1} x {32,2,1}
```

272 = `nw0*ne11` = (17408/256)*4. It is the f16y activation convert
(`ggml-metal-ops.cpp:2856-2896`) with a `ggml_metal_op_concurrency_reset` between the two,
which is a real `memoryBarrier` (`:207-216`). The design is deliberate and documented at
`results.md:279-287`, size gate included.

What was **not** on record is the routing consequence. `use_f16y` needs `ne11 >= 2`, and
under prod env widths 1-2 go to `mul_mv`/`nc` and widths 5+ to skinny, so only widths 3-4
reach the ext f16y path at all. Measured, not inferred (`perf/run-f16y-ab.sh` logs it):

| width | ffn_down pipelines under prod env |
|---|---|
| 1 | `kernel_mul_mv_q4_0_f32_nsg` |
| 2 | `kernel_mul_mv_q4_0_f32_nc2_nsg` |
| **3** | **`kernel_cpy_f32_f16`** + `kernel_mul_mv_ext_q4_0_f16_r1_3_nsg=2_nxpsg=8_nr0=2` |
| **4** | **`kernel_cpy_f32_f16`** + `kernel_mul_mv_ext_q4_0_f16_r1_4_nsg=2_nxpsg=8_nr0=2` |
| 5 | `kernel_mul_mm_skinny_q4_0_f32_ne12` |

So the convert is a width-3/4-only mechanism in the shipping config, sitting exactly on the
cliff. Pre-registered as a possible hidden tax on it, bounded at under 20 us.

**Refuted, and in the direction the design predicted.** `GGML_MV_EXT_F16Y=0` against the
default, prod env, 3 interleaved reps, arms alternating inside one block:

| shape | w3 f16y=1 | w3 f16y=0 | delta | w4 f16y=1 | w4 f16y=0 | delta |
|---|--:|--:|--:|--:|--:|--:|
| ffn_gate/up | 279.39 | 307.69 | +10.1% | 353.44 | 370.90 | +4.9% |
| **ffn_down** | 336.73 | 338.96 | +0.7% | **359.61** | **421.82** | **+17.3%** |
| gdn_qkv | 108.94 | 117.85 | +8.2% | 126.28 | 147.42 | +16.7% |
| **attn_q** | 65.51 | 66.06 | +0.8% | **75.19** | **75.62** | **+0.6%** |

Controls - widths 1, 2 and 5, all four shapes, both arms - every cell within +-1.5%, which
is the single-rep noise floor. **The convert is not a tax on the cliff.** It is worth up to
62 us at width 4 `ffn_down` and removing it makes widths 3-4 worse. Do not reopen it as a
cliff component.

One arm is soft: width 4 ffn_gate/up at f16y=0 read 379.39 / 378.38 / 354.93, so its +4.9%
is really nearer +7.5% with one outlier rep. Every other cell has under 1% within-arm spread.

~~**What survives is `attn_q`**: +0.6% at width 4, inside noise, while ffn_down and gdn_qkv
take +17.3% and +16.7%. It passes the size gate comfortably (ne00*ne01 = 5120*3072 = 15.7M
against a gate of 8M). It is also the one shape that *loses* on `nxpsg=16` (+8.3%, run 3).
Same shape, both levers, no explanation on record. Run 5 is that explanation.~~

> **CORRECTED 2026-08-23 by the replay counters - the gate is 16M for q4_0, not 8M.**
> `(int64_t) ne00*ne01 >= (is_t4 ? 16 : 8)*1024*1024` (`:2799-2800`), and q4_0 **is** t4
> (`is_t4` excludes only the K-quants and IQ4_XS), so the threshold is **16.78M**. attn_q is
> 15.7M and therefore sits **below** it: f16y was never active for that shape, in either arm.
> Caught by the capture, which shows attn_q on `kernel_mul_mv_ext_q4_0_f32_r1_4` with no
> `kernel_cpy_f32_f16` next to it, and confirmed by compiling both shapes.
>
> So the attn_q row is **a control, not a treated cell**, and its +0.6% measures nothing
> about f16y - it is two identical kernels being compared, which is exactly why it reads
> flat. There is no attn_q anomaly here and nothing for run 5 to explain. The three real
> treated shapes are ffn_gate/up, ffn_down and gdn_qkv, all above 16.78M, and they win.
>
> attn_q losing on `nxpsg=16` (run 3) is untouched by this and remains unexplained.

## Run 5 (2026-08-23): ~~the f16y size gate is keyed on the wrong dimension~~ RETRACTED

> **RETRACTED the same day, by the replay counters. The conclusion below is wrong and the
> data below is fine.** The premise was that the gate admits shapes that gain nothing. It
> does not: the gate for q4_0 is **16.78M**, not the 8M this section assumed
> (`ne00*ne01 >= (is_t4 ? 16 : 8)*1024*1024`, and q4_0 is t4). At ne00 = 5120 that is
> ne01 >= 3277.
>
> Look at where the step lands: **+0.3% at ne01 3072, +14.5% at ne01 4096.** 3277 sits
> between them. **The step is the gate**, doing exactly what it was written to do - f16y is
> *off* at 3072 and *on* at 4096, so the "step" is just the treatment switching on. Verified
> by compiling both: m=3072 gives `kernel_mul_mv_ext_q4_0_f32_r1_4` alone, m=4096 gives
> `kernel_cpy_f32_f16` + `kernel_mul_mv_ext_q4_0_f16_r1_4`.
>
> So this run measured the gate boundary, not a flaw in it, and **three of its five rows are
> controls rather than two**. That is a clean confirmation that the gate is where the source
> says it is, and nothing more.
>
> **The wrong-dimension hypothesis is untested, not refuted.** The reasoning under it still
> stands on its own, but this sweep cannot test it: ne00 was held fixed, so ne01 and the
> product move together and the two hypotheses are indistinguishable. Testing it needs ne00
> and ne01 varied *independently at a fixed product* - e.g. (ne00 5120, ne01 4096) against
> (ne00 20480, ne01 1024), both 21M. If the win tracks ne01, the second should win less.
> Neither shape is a perf case today. **Do not treat the gate as known-wrong.**

Pre-registered from run 4, then measured (`perf/run-f16y-ne01-sweep.sh`). Hold ne00 at 5120
and sweep ne01, because:

```
convert cost   ~ ne00*ne11         one pass over the activations, independent of ne01
matmul saving  ~ ne01*ne00*ne11    halved y-loads, once per output row
ratio          ~ ne01              ne00 cancels
```

so a gate on `ne00*ne01` (`:2799-2800`) is wrong-dimensioned. Prediction: the win tracks
ne01 and is near zero at small ne01 whatever ne00 is.

| ne01 | ne00*ne01 | f16y? gate 16.78M | w3 f16y=1 | w3 f16y=0 | delta | w4 f16y=1 | w4 f16y=0 | delta |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| 1024 | 5.2M | **off - control** | 27.90 | 27.71 | -0.7% | 32.50 | 32.50 | +0.0% |
| 1280 | 6.6M | **off - control** | 28.39 | 28.40 | +0.0% | 32.80 | 32.91 | +0.3% |
| 3072 | 15.7M | **off - control** | 61.72 | 62.55 | +1.4% | 75.02 | 75.25 | +0.3% |
| 4096 | 21.0M | on | 76.65 | 81.95 | +6.9% | 88.87 | 101.80 | **+14.5%** |
| 6144 | 31.5M | on | 108.33 | 117.15 | +8.1% | 126.96 | 147.63 | **+16.3%** |
|  |  |  |  |  |  |  |  |  |

Read correctly: **three controls, all flat, and two treated rows, both winning.** The gate
column is what the retraction above adds - the original table called ne01 3072 "above gate",
and that was the error.

~~**Held.** ... At ne00 = 5120 the gate admits everything from ne01 = 1638 up, so it passes
a whole band where f16y does nothing, and `attn_q` (ne01 = 3072) sits in it.~~ **Wrong, see
the retraction at the top of this run.** The gate admits from ne01 = 3277; ne01 3072 is
below it; the step is the treatment switching on.

What the run does establish, and it is worth keeping:

- **The gate boundary is exactly where the source puts it**, confirmed by measurement and
  by compiling both sides of it.
- **f16y is worth +14.5% to +16.3% at width 4** on the two shapes that qualify, which is
  consistent with run 4's ffn_down and gdn_qkv and independent of them.
- **Three controls read flat**, so the harness is measuring f16y and not drift.

## The capture set, and what widths 1-2 were doing all along

`perf/run-capture-set.sh` takes ten captures across the cliff, archives them **out of
`/tmp`** (`~/play/kvquant-experiments/traces/aug23`, 424 MB) and dumps each headlessly.
Captures are free and need no GUI; the replay click is the expensive step, so keeping them
is what makes a click worth spending later.

Dispatch geometry at `ffn_down` (ne01 = 5120), threadgroups x threads:

| width | route | matmul grid | threads/tg | convert grid |
|---|---|--:|--:|---|
| 1 | `mul_mv` | **640** | 64 | - |
| 2 | `mul_mv` nc2 | **640** | 64 | - |
| 3 | ext nxpsg=8 | **320** | 64 | 204 x 256 |
| 4 | ext nxpsg=8 | **320** | 64 | 272 x 256 |
| 3 | ext nxpsg=16 | **640** | 64 | 204 x 256 |
| 4 | ext nxpsg=16 | **640** | 64 | 272 x 256 |
| 4 | ext, f16y=0 | 320 | 64 | - (single dispatch) |
| 5 | skinny | {1,160,1} | 64 | - |

**Widths 1-2 already run 640 threadgroups. Crossing to width 3 halves the grid to 320, and
`nxpsg=16` puts it back to 640.** Run 3 saw 320 vs 640 within width 4 only and read it as
occupancy; the captures show 640 is what the cheap widths were doing all along, so the
width-3 cliff coincides with a 2x grid collapse rather than with a new kernel that merely
happens to be narrow. Arithmetic: ext `r0ptg = (32/nxpsg)*nsg*nr0` is 16 at nxpsg=8 and 8 at
16, and 5120/16 = 320, 5120/8 = 640.

`attn_q` (ne01 = 3072) runs 192 threadgroups at nxpsg=8 and 384 at 16 - the smallest grid in
the set, and the one shape that loses on the lever.

Two cautions for anyone reading these dumps:

- Selector names come out `(null)` in this build, so read by shape. `memoryBarrierWithScope:`
  reads as `(null)(<enc>, 1ul)`; the f16y arm records 127 of them against 63 in every other
  arm, consistent with the extra barrier the convert inserts. It is **not** one per node -
  the baseline arms hold 63 across anywhere from 141 to 561 dispatches - so do not count
  barriers per iteration.
- There is still no timing and no counter anywhere in a dump (`toolchain-isa-probe.md:237`).

## Replay counters for the whole cliff (2026-08-23) - all ten captures

Johan clicked through all ten; `perf/watch-replays.sh` archived them. This is the first time
the cliff has been profiled at anything other than width 4. Spill is 0 everywhere, so it is
dropped from the table.

| capture | kernel | reg | instr | dev ld/st |
|---|---|--:|--:|--:|
| w1 ffn_down | `mul_mv_q4_0_f32` | 59 | 389 | 12/4 |
| w2 ffn_down | `mul_mv_q4_0_f32_nc2` | 77 | 664 | 16/8 |
| **w3 ffn_down nxpsg=8** | `mul_mv_ext_q4_0_f16_r1_3` | **64** | **379** | 7/6 |
| **w3 ffn_down nxpsg=16** | `mul_mv_ext_q4_0_f16_r1_3` | **64** | **397** | 7/6 |
| w4 ffn_down nxpsg=8 | `mul_mv_ext_q4_0_f16_r1_4` | 73 | 453 | 8/8 |
| w4 ffn_down nxpsg=16 | `mul_mv_ext_q4_0_f16_r1_4` | 73 | 477 | 8/8 |
| w4 ffn_down f16y=0 | `mul_mv_ext_q4_0_f32_r1_4` | 79 | 457 | **16**/8 |
| w4 attn_q nxpsg=8 | `mul_mv_ext_q4_0_f32_r1_4` | 79 | 457 | **16**/8 |
| w4 attn_q nxpsg=16 | `mul_mv_ext_q4_0_f32_r1_4` | 79 | 483 | **16**/8 |
| w5 ffn_down | `mul_mm_skinny_q4_0_f32` | 60 | 507 | 10/1 |

Three things fall out, none of them previously measured:

- **The nxpsg result replicates at width 3, which is where the cliff actually is.** Registers
  identical at 64, spill 0 both ways, and the instruction count *higher* at nxpsg=16 (397 vs
  379) - the same signature run 3 found at width 4 (73/73, 477 vs 453). Two widths, same
  pattern: `nxpsg=16` wins while doing measurably *more* work per thread with *identical*
  register pressure. Register pressure and instruction count are both excluded by
  measurement now, at both widths. What is left is the grid, which the captures show
  doubling from 320 to 640.
- **f16y halves the device loads, exactly as designed** - 16 down to 8 on `r1_4`
  (`results.md:281` says "reading 8 contiguous elements per 16B uint4 load"; this is that
  claim measured). It costs 6 registers (79 -> 73) and one extra dispatch, and run 4 shows it
  wins by up to 17.3% anyway.
- **`w4 attn_q nxpsg=8` and `w4 ffn_down f16y=0` are the same kernel with identical counters**
  (79 reg, 457 instr, 16/8). That is what exposed the gate error: attn_q was never on the
  f16y path. Two shapes, one kernel, and the "attn_q anomaly" of run 4 dissolves.

Raw `streamData` plus `--all` dumps are under
`~/play/kvquant-experiments/traces/aug23/replays/`. Only `streamData` is kept - the rest of
each `.gpuprofiler_raw` directory is ~1 GB of `Profiling_f_*.raw` frame data that
`gpuprofiler-stats.py` never reads, and archiving it whole cost 7.9 GB in four clicks before
that was noticed. Note also that Xcode writes each replay **twice**, at
`ROOT/<name>_stream.gpuprofiler_raw` and `ROOT/gtshaderprofiler/<name>.gputrace.gpuprofiler_raw`,
identical apart from the path - dedup on `traceName`, not on path.

## Run 6 (2026-08-23): the nxpsg cutoff, aggregated - and two invalid attempts first

Run 3's `### Next`. Branch `metal-mv-ext-nxpsg-w34` adds `GGML_MV_EXT_NXPSG16_MAX` (default
3, the shipping cutoff) so one binary runs both arms:

```
ne00 % 256 == 0 && ne11 < env_nx16_max     ggml-metal-ops.cpp:2769-2777
```

**Attempt 1 was invalid and its numbers must not be quoted.** It used the pre-existing
`GGML_MV_EXT_NXPSG=16`, which forces nxpsg for every ext call and thereby also bypasses the
`ne00 % 256 == 0` condition. That condition is **correctness, not a heuristic**: forced,
MUL_MAT fails with NaN on `kernel_mul_mv_ext_f16_f32_r1_2` at ne00 = 128 (two cases,
`type_a=f16`, m=64 and m=83 - `MTL0=nan CPU=-4.979980`). The harness that did this has been
deleted rather than struck: a note that is wrong can carry a correction, but a runnable
script that silently measures a NaN-producing kernel is a trap. Both arms of the real
experiment pass the full MUL_MAT suite, 3/3 backends, 0 failing cases.

**Attempt 2 was confounded.** It ran max=3 then max=5 in every rep, so the machine warming
inside a rep was charged to whichever arm ran second: the N=1 control - bit-identical
routing in both arms, so it must read flat - came out +2.1% while the N=2 control read
-0.3%. Alternating the arm order between reps fixes it and the controls then agree.

### Pass cost, 4 reps, arm order alternating

`llama-bench -n 0 -p 1..8`, prod-pick env, `perf/run-nxpsg-gate.sh`:

| N | max=3 (ship) | max=5 | delta | role |
|---|--:|--:|--:|---|
| 1 | 77.14 | 76.44 | -0.9% | control, identical routing |
| 2 | 78.13 | 77.91 | -0.3% | control, identical routing |
| **3** | **105.22** | **103.40** | **-1.7%** | affected |
| **4** | **116.13** | **114.36** | **-1.5%** | affected |
| 5 | 122.48 | 122.97 | +0.4% | control, skinny |
| 6 | 125.06 | 125.32 | +0.2% | control, skinny |
| 7 | 126.70 | 126.67 | -0.0% | control, skinny |
| 8 | 127.39 | 127.75 | +0.3% | control, skinny |

Six control cells inside +-0.9%, so that is the floor and the N=3/N=4 signal is about twice
it. Routing was logged, not assumed: at max=5 widths 3-4 load
`..._nsg=2_nxpsg=16_nr0=2` and widths 1, 2, 5 are untouched.

**The pre-registered bound was wrong and in the optimistic direction.** Weight-bytes
reasoning over run 3's per-shape table (-5.0%/-3.2% on ffn_down, -7.4%/+0.1% on
ffn_gate/up) predicted -3% to -5%. Measured: **-1.5% to -1.7%.** The per-shape wins dilute
by roughly a factor of three once everything else in a pass - attention, GDN, norms, FA, and
the `attn_q`-class shapes that lose - is included. This is exactly what run 3 refused to
settle by arithmetic over those tables, and it was right to refuse.

### The e2e arm is INVALID - its control failed

**No end-to-end number from 2026-08-23 should be quoted.** The design was: A/B at dflash n3
(depth 3 = width 4, the affected point and oMLX's operating point) with dflash n6 (width 7,
skinny) as a control that must read flat. `perf/run-nxpsg-e2e.sh`, n_predict 600, warmup
discarded, arms alternating.

| config | max=3 | max=5 | delta | arms overlap? |
|---|--:|--:|--:|---|
| n3, width 4 (affected) | 18.463 | 18.667 | +1.1% | yes |
| **n6, width 7 (CONTROL)** | **21.818** | **20.479** | **-6.1%** | **no** |

**The control moved six times further than the treated cell, and in the wrong direction.**
`GGML_MV_EXT_NXPSG16_MAX` cannot reach width 7 - skinny does not consult `nxpsg`, and the
n6 runs emit a byte-identical output sha and an identical 41.3% acceptance in both arms, so
they are provably doing the same work. The -6.1% is the machine, not the flag. Raw n6
readings across roughly ten minutes on that byte-identical workload: 21.600, 20.540, 20.418,
22.035 - a **7.9% spread**. The laptop was in direct sun.

**Why the pass-cost half survived the same conditions and this did not.** `llama-bench`
measures N=1..8 inside one invocation, so the controls (N=1,2,5..8) and the treated cells
(N=3,4) share whatever the machine is doing at that moment; drift between invocations moves
all eight together and the controls bound it. The e2e harness produces **one number per
server invocation**, so between-invocation drift *is* the measurement and no amount of
alternating fixes it. ABBA cancels a linear trend and is maximally confounded with a
non-monotone one, which is what the n6 block looks like (outer pair high, inner pair low).

Redo this in a controlled thermal state, with more reps, and keep the n6 control. Until it
reads flat, the e2e question is open.

### What the e2e run did establish

- **The two n3 arms produce different text, deterministically.** All four `max=3` runs give
  sha `a08f1b87121c` (2246 B) and all four `max=5` runs give `3776c0adb7ee` (2249 B), across
  the whole session. `nxpsg` changes the reduction order along the row, one near-tie flips,
  and the divergence is inside a degenerate repetition loop in the tail ("This is a" vs
  "This is"). Benign, but it means the arms run different accepted-token sequences, so e2e
  t/s is not comparing identical work even once the thermals are fixed.
- **Output identity is a within-depth invariant, not a cross-kernel one.** README's "all
  runs emit byte-identical text regardless of speculation config" holds for the flags tested
  there; it does not survive a kernel change that reorders a reduction. `3776c0adb7ee` is
  the archived n_predict-600 reference and n6 reproduces it exactly in both arms.
- **Acceptance is exactly reproducible per arm** - 60.2% / 59.7% / 41.3% every time - which
  is how the arms were confirmed to be doing the work they were supposed to.

### And none of this can move the prod pick

Worth stating plainly, because a -1.7% pass-cost win reads like a prod improvement and is
not one. **The prod pick is dflash n6, which verifies 7 columns and routes to
`mul_mm_skinny`. Widths 3-4 do not occur in it.** This lever only matters if width 4 becomes
a viable operating point, and today n3 is our *worst* depth (20.462 t/s at n_predict 300
against 25.038 at n6, `slope-sweep.md:56`). The width-4 work is about reaching oMLX's
operating point, not about speeding up ours.

## Run 7 (2026-08-23): plain `mul_mv` is the fourth family, and it loses too

Run 3 A/B'd `ext` against `mv_nc` and `mul_mm_skinny` and never tested the **plain batch-1
vector kernel** - the final `else` in `ggml_metal_op_mul_mat` (`:2985`), which dispatches
`ne11` in the grid y dim with `nr1 = 1` and therefore walks the weights once per column.

Zero code. `GGML_MV_EXT_MAX=2` drops widths 3-4 out of the ext gate
(`ne11 <= ne11_mv_max`, `:2617`); `GGML_MM_MIN` stays 8 so `mul_mm` does not claim them;
under the prod env nc holds width 2 and skinny holds 5-8. **Widths 3 and 4 are the only
thing that changes**, so widths 1/2/5 are within-run controls. Harness
`perf/run-width34-plainmv.sh`, caffeinated, 3 interleaved reps, `169635d6c`, binary 17:20.
Routing confirmed per width from the compiled pipeline names in both arms (part 0b), and the
plain-mv arm passes `test-backend-ops test -o MUL_MAT` on all 20 cases.

`test-backend-ops perf`, median of 3, within-arm spread <= 4.9% (typically < 1.5%):

| shape | width | `ext` (prod) | plain `mul_mv` | delta |
|---|--:|--:|--:|--:|
| ffn_gate/up | 3 | **282.58** | 323.26 | +14.4% |
| ffn_gate/up | 4 | **332.73** | 428.65 | **+28.8%** |
| ffn_down | 3 | 339.63 | **333.29** | **-1.9%** |
| ffn_down | 4 | **361.86** | 437.60 | **+20.9%** |
| gdn_qkv | 3 | **108.69** | 118.89 | +9.4% |
| gdn_qkv | 4 | **127.47** | 156.05 | +22.4% |
| attn_q | 3 | **61.82** | 64.02 | +3.6% |
| attn_q | 4 | **76.07** | 81.20 | +6.7% |

Controls: widths 1, 2 and 5 move -1.0% to +1.6% across all four shapes, which is the
within-session floor.

Whole model, `llama-bench -n 0 -p 1..8 -r 3`, same session, ms/pass:

| width | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `ext` (prod) | 74.2 | 77.5 | **103.9** | **116.0** | 124.1 | 125.1 | 127.1 | 128.5 |
| plain `mul_mv` | 76.0 | 77.3 | **108.0** | **137.2** | 123.9 | 124.8 | 127.7 | 128.1 |
| delta | +2.5% | -0.3% | **+3.9%** | **+18.3%** | -0.1% | -0.2% | +0.5% | -0.3% |

The untouched widths agree to <= 0.5% except pp1 at 2.5%, so 2.5% is the arm-to-arm floor for
`llama-bench` (separate processes, model reload) and pp3's +3.9% is small but real, pp4's
+18.3% overwhelming. The `ext` row itself sits 2-4% above the archived curve
(73.0/73.8/101.5/111.5/119.0/120.9/123.1/124.1), which is the documented ~3% cross-session
drift; both arms here were measured minutes apart.

**Verdict: `ext` is the right family at widths 3-4 against all three alternatives.** The
family question is closed - `mv_nc`, `mul_mm_skinny` (run 3) and plain `mul_mv` (here) have
each now been measured at these widths and each loses. Nothing here touches the prod pick,
which sits at n6 / width 7 / skinny.

### The one exception is the interesting number

`ffn_down` at width 3 is a **-1.9% win** for plain mv, and it should not be close. Plain mv
re-reads the weights per column, so at width 3 it should move 3 x 50.1 MB. 150 MB in 333 us
would be **451 GB/s, 1.65x the machine's 273 GB/s peak** - so it is not doing that: the three
column dispatches run concurrently over the same row blocks and the cache serves columns 2
and 3. Effective DRAM traffic is ~50 MB in 333 us = **150 GB/s**, against **247 GB/s** for
the same shape at width 1.

Per column, plain mv costs 111 us at width 3 and 109 us at width 4, about **half** the 210 us
it takes standing alone at width 1. So re-dispatching the batch-1 kernel per column recovers
roughly 2x of reuse from the cache for free, and still never approaches `ext`'s single
stream.

**A kernel with a completely different structure, at half our bandwidth, ties `ext` at
width 3.** That is independent of run 3's dispatch-geometry argument and points the same way:
at these widths `ext` is not spending its time on weight traffic. Consistent with run 2 (a
bigger per-thread tile made width 4 worse) and run 3 (`nxpsg=16`, twice the threadgroups,
wins).

## Superseded plan: port the V2 base-pointer rewrite to `ext`

`metal-mv-nc-spill` commit `fe0429daf` already did this for `mul_mv_nc`: keep one base
pointer per operand, recompute row/col offsets in the loop. It freed ~14 GPRs and took nc3
from 80 B spill to 0. The `ext` kernel has the identical anti-pattern, so the same rewrite
applies almost unchanged.

~~Order, and it is cheap at every step:~~ **All four steps ran - see run 2 above.** The
prescreen gate passed (spill 0) and the GPU measurement then refuted the premise. Step 4
(llama-bench) was not run: there is nothing to confirm, since v2 does not beat the width-4
baseline. The prescreen-before-build discipline is worth keeping regardless; it cost ~8 s to
prove the rewrite worked before spending a build on it.

**Still open, but the tile is off the table.** Nothing is committed from either run. Prod
routing, `GGML_MV_EXT_NR0` and `GGML_MV_EXT_V2` defaults are all unchanged, so prod behaves
exactly as before. What is left to explain the width-4 shelf is *not* the register tile -
candidates now are the `nxpsg=16` cutoff at `ne11 < 3` (widths 3-4 lose the wide variant
that makes width 2 nearly free) and the drafter serialization in
`drafter-pipelining.md`. Pick one and pre-register the bound before measuring.

## Sizing, honestly

**This does not speed up the prod pick.** We run width 7 on skinny for 97.5% of passes;
an `ext` change at width 4 buys the shipping config nothing. It buys an *operating point*,
and today n3 is our worst depth at 20.46 t/s.

The ceiling does close, though, which it did not appear to before the width correction. Our
141.0 ms width-4 round is roughly 118 ms in-graph verify + ~16 ms serialized drafter + ~7 ms
overhead. A width-4 verify at the bandwidth floor (76-85 ms, which their kernel demonstrates
is attainable) plus a pipelined drafter lands the round at **88-100 ms**, against their
measured 95.00. At our 2.88 committed tokens/round that is **~29-33 t/s**.

**It needs both levers.** A perfect kernel with today's serialized drafter gets the round to
about 118 ms = 24.4 t/s, still short of the prod pick. See `drafter-pipelining.md` (branch
`drafter-pipelining`), which is blocked on splitting the shared `MTLCommandQueue`.

And it only pays under a depth policy that would actually sit at width 4, so
`adaptive-spec` has to come back off the shelf afterwards - `slope-sweep.md`'s "flattening
widths 2-5 is the prerequisite" is right, but read it as **"fix width 4"**.

## Rule: their kernel is a number to beat, not a source to copy

**Decided 2026-08-22 by Johan. No code from `dflash_mlx` enters this tree, in any form -
not copied, not transliterated, not "adapted".** It is Apache-2.0 against llama.cpp's MIT,
and even inside a private fork that is a licence mismatch we do not want to carry, given
some of this work may get rebased onto upstream later.

What their kernel *is* for:

- **A performance target.** Their width-4 number is the bar. Measure it, benchmark against
  it, and treat "we widen 1 -> 4 at +102 us, they do it at +52" as the goal to close.
- **Evidence that the bar is reachable.** The value of reading it was learning that a 16-
  accumulator 4x4 tile fits without spilling on this hardware. That fact is what justifies
  experiment 3; the fact is not their code. **Refined 2026-08-22:** it fits in *their*
  kernel. Ours spills 32 B at the same tile (run 1 above), because our addressing holds 8
  device pointers live and theirs does not. The tile is reachable; our current addressing is
  what stops us reaching it.
- **A disassembly reference** (experiment 5). Comparing register allocation via
  `metal-objdump` is measurement, not reuse.

Everything we write is our own, against ggml's own layout - which would force a full rewrite
regardless: Q4_0 is 32-element blocks with **interleaved** nibbles (byte *j* holds value *j*
low, *j+16* high) and a scale-only affine, against their gs64 scale-plus-bias with sequential
nibbles in separate planar arrays. The algorithm description earlier in this file is here so
nobody needs to re-read their source; it is not a porting spec.
