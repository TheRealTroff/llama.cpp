# Width 4: the one operating point they have and we do not

Status: **open**. Opened 2026-08-22 from `mlx-cycle-capture.md` open stubs 1 and 2, plus a
new kernel-level measurement taken the same day (below).

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
2. **Free, no code: confirm the current routing is actually best at widths 3-4.** They are
   left to `ext` by *configuration*, not by code - `GGML_MV_NC` caps at `min(env,4)` with
   nc3/nc4 kernels already compiled, and `GGML_MM_SKINNY`'s floor is 2. A three-arm A/B
   (ext vs mv-nc vs skinny +/- repack) costs one run. Prior evidence says ext wins
   (`mv-nc-cliff-probe.md`, `dflash-vs-mtp-uniform.md:61-74`), so this is confirmation.
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
