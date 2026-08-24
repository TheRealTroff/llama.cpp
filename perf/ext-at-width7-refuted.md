# The register-tile kernel loses at width 7, and it is structural, not tuning

Status: **closed, refuted - and it redirects `ffn-utilization.md`'s experiment 3.** Measured
2026-08-24 at prod `2d21fe72b`, `test-backend-ops perf` on MTL0, zero code. **The TPR lever
this file left open at the bottom is now built and refuted too - `skinny-tpr-bsplit.md`,
same day - and the overlap win it was after turned out to be in the loader, not the MACs.**

## What this tests

`ffn-utilization.md` experiment 3 says the answer is "the register-tile kernel: no
`simdgroup_matrix`, inline dequant, never staged to threadgroup memory, K-split" - the shape of
dflash_mlx's `verify_m4`. **That kernel already exists in this repo. It is `mul_mv_ext`**
(`m4-verify-kernel-proposal.md`: "a matvec: threadgroup (32, nsg=2, 1), lanes `nxpsg x nypsg`,
reduction by simd shuffle" - no matrix primitive, no threadgroup tiling, no barriers).

Nobody had ever measured it at the prod width. The width sweep in `ffn-utilization.md`
compared prod routing against skinny-forced; ext at widths 5-8 was never an arm, because
**`GGML_MV_EXT_R1MAX` defaults to 5**, which disables the `r1_6`/`r1_8` variants and makes
width 7 take *two* passes over the weights (`r1ptg` 4, then 3).

## The numbers

`ffn_gate+up` (17408 x 5120) and `ffn_down` (5120 x 17408), us/call:

```
             width 5          width 6          width 7          width 8
             gate    down     gate    down     gate    down     gate    down
skinny      360.8   429.6    363.8   434.5    366.7   436.5    370.4   441.6
ext r1max8  374.5   410.2    546.6   777.0    953.6  1137.9    987.7  1208.1
ext r1max5  374.5   409.7    491.4   527.2    600.1   644.1    604.7   648.3
```

Tuned at width 7 on `ffn_gate+up`, sweeping every knob ext has:

| config | us/call | kernel |
|---|--:|---|
| `R1MAX=8` (nr0=4 default) | 919.0 | `r1_8` |
| `R1MAX=8 NR0=2` | 606.7 | `r1_8` |
| `R1MAX=8 NR0=2 NXPSG=16` | 599.1 | `r1_8` |
| **`R1MAX=8 NR0=2 NXPSG=4`** | **596.2** | `r1_8` |
| `R1MAX=8 NSG=4` | 926.5 | `r1_8` |
| `R1MAX=6` (two passes) | 588.6 | `r1_4` |
| **prod skinny** | **366.7** | `mm_skinny` |

**Best ext at width 7 is 1.63x worse than skinny, and it is not a tuning problem** - the knobs
were swept. `nr0` 4 -> 2 recovers 34% (919 -> 607), which is the register-pressure signature
`width4-verify.md` already found at width 4 (32 B spill at nr0=4/r1ptg=4 from 8 live device
pointers); at `r1ptg=8` the tile is 4x8 = 32 accumulators and it spills far worse. Halving to
16 recovers a third and no further knob helps.

## Why - and the crossover is the useful part

The two designs trade off in **opposite directions with width**:

- **Register tile (`ext`)**: no staging cost, but accumulators are `nr0 x r1ptg` and the
  simd-shuffle reduction costs `nr0*r1ptg*log2(nxpsg)` - **both scale with the verify width.**
  `ksplit-width34.md` already identified that reduction term; this is the same term at width 7,
  where it is twice what it was at width 4.
- **Staged + `simdgroup_matrix` (`skinny`)**: pays a fixed `dequant -> threadgroup ->
  simdgroup_load` round trip and two barriers per K slice, then **amortizes it over all 8
  columns.**

So the crossover is around **width 5**, and it is shape-dependent: at width 5 skinny wins on
`ffn_gate+up` (360.8 vs 374.5) while **ext wins on `ffn_down` (410.2 vs 429.6)**. Below that
ext wins outright; above it skinny wins and the gap widens fast.

## What this corrects

- **`ffn-utilization.md` experiment 3, as written, is wrong for the prod operating point.**
  ~~"THE ONE LEFT: the register-tile kernel"~~ - it exists, it is `mul_mv_ext`, and at width 7
  it is 1.63x slower than what we already run. Building a *new* one of the same shape would
  reproduce this result.
- **`width4-skinny-ab.md`'s conclusion needs its scope stated.** "The ONLY reason to accept
  `dequant -> threadgroup -> simdgroup_load` is to reach dedicated matrix hardware... MLX's
  no-`simdgroup_matrix` is not a stylistic alternative - on this hardware it is the correct
  choice." **That is correct at width 4, which is MLX's operating point, and it does not
  transfer to width 7.** At width 7 the staging round trip is what makes 8 columns affordable,
  and skipping it costs 63%. The file's reasoning was sound; its scope was unstated.
- The `nr0=2` win at `r1ptg=8` is a **third** independent sighting of the ext register cliff
  (after `width4-verify.md` runs 1-2 and the spill probe). The tile does not have room for 8
  columns on this hardware, at any `nsg`/`nxpsg`.

## What is actually left for saturating a roof

Both families are at ~50% of both roofs at width 7, and both now have a structural explanation.
The additive finding stands: `measured ~= stream + arith`, 187 + 205 us against 368 measured on
`ffn_gate+up`. **Neither roof can be saturated while both costs are paid in series**, and at
this width they are within 10% of each other, so the ceiling from perfect overlap is ~205 us -
still a 1.8x.

The one axis never tested, and the one the `NR0` sweep **structurally could not reach**:
`kernel_mul_mm_skinny` pins **rows per simdgroup = 32/TPR** where TPR is the A-tile loader's
threads-per-row, currently 2. So `nsg = TPR*NR0/32`, and varying `NR0` alone moves threadgroup
count while leaving total simdgroups and rows-per-simdgroup invariant - which is exactly why
`skinny-nr0-refuted.md` found nothing. **Changing TPR from 2 to 4 halves rows per simdgroup to
8 (`mc[1]` instead of `mc[2]`) and doubles simdgroups per threadgroup at fixed `NR0`**, giving
each barrier more independent MAC work to hide the A-tile loads behind. ~~That is the overlap
lever, it is a real code change, and it is the next thing to try.~~

**CORRECTED 2026-08-24, built and measured the same day: `skinny-tpr-bsplit.md`.** TPR is a
real knob (`GGML_MM_SKINNY_TPR`, 1154/1154 correct at 1/2/4) and **rows per simdgroup is
already at its optimum at 16**: TPR=4 costs +7.7% on `ffn_gate+up` and buys 0.3% on
`ffn_down`, TPR=1 costs +9.9% / +13.4%. The paragraph above is right that `NR0` could not
reach this resource and wrong that reaching it would pay. What did pay was found beside it:
the B-tile load was pinned at 32 threads whatever the threadgroup size, and spreading it over
all of them (`GGML_MM_SKINNY_BSPLIT`) is **-1.4% to -2.6% on every projection and +1.0 to
+1.6% e2e**. The overlap this file was looking for was in the loader, not in the MAC side.
