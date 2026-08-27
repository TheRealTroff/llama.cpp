# Width 5: the SoA scalar cell, and the scalar-vs-MMA crossover pinned between 5 and 7

Status: **answered 2026-08-28 - w5r4h wins 25-31% synthetic and +25% e2e, and dflash
n4+w5 (25.632 t/s) BEATS dflash n3 (25.282) on the same board, overturning the depth
re-sweep's deprioritization of this cell.** Branch `m4-width4-r4kp` (same branch as the
width-4 result). **Adopted as the prod pick 2026-08-28 (owner: "pick this for now") -
see README "The prod pick" for the flag set and the accepted caveats.**

Opened from `m4-width4-r4kp.md`'s depth re-sweep, which DEPRIORITIZED this cell for the
operating point (depth 4 is 4.7 t/s behind depth 3 and a ~20% kernel saving cannot close
that). This run is not chasing the operating point - it buys the learning: the width-4
scalar form won 26-29% and the width-7 transfer lost 1.5-1.7x, so width 5 is the cell
that pins the scalar-vs-`simdgroup_matrix` crossover and gives the first real datum for
the parked "why does MMA win above width ~5" question.

Prediction on record: scalar ~310 us vs skinny's flat ~385 on ffn_down (linear scaling
from the width-4 numbers).

## The kernels (staged like the w7 pair, v2 codegen form, SoA layout unchanged)

`kernel_mul_mv_q4_0_soa_w5_{r2,r4,r2h,r4h}`: R rows x 5 columns, full K, one simdgroup,
scalar broadcast dequant, signed-int indexing + hoisted planar row pointers. `_r2h/_r4h`
are the v3-style half-product bodies (`(half(q) - 8.h)*s`, product in half, f32
accumulate). Routed behind `GGML_MV_SOA_W5=<rows>` (+ `GGML_MV_SOA_W5_HALF=1`),
mirroring `GGML_MV_SOA_W7`; `try_repack`'s `use_soa` got the ne11==5 case. In the w5
arms `GGML_MM_SKINNY` must be 6, not 5 - skinny takes `ne11 >= value` and would
otherwise swallow width 5 before the mv path is reached.

All four: zero spill (offline prescreen, probe regression-checked against R2's recorded
2184/0; text 1956/3672/1974/3690), 1155/1155 `test-backend-ops -o MUL_MAT -b MTL0` vs
CPU per variant, pipeline engagement confirmed by name in the profile capture
(`kernel_mul_mv_q4_0_soa_w5_r4h`, 80 temp regs, 0 spilled).

> **Harness trap found on the way (now in README methodology): `test-backend-ops -b
> Metal` matches NO backend** - the device is named `MTL0` - and the run then prints
> `3/3 backends passed OK` with every backend "Skipping". A green run with the wrong
> `-b` is a vacuous pass, not a pass.

## Synthetic per-shape A/B (test-backend-ops perf, interleaved, GGML_MV_REPACK=2, n=5)

us/run means (3 reps, spread <=2.1%, worst spreads on the skinny arms;
`results/m4-w5-ab-aug28.tsv`). Floor = weight bytes / 273 GB/s.

| projection (m,k at n=5) | skinny | ext | w5r2 | w5r4 | w5r2h | **w5r4h** | vs skinny | best/floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ffn_down (5120,17408) | 380.9 | 409.0 | 339.6 | 282.7 | 386.6 | **263.8** | -30.8% | 1.44 |
| ffn_gate/up (17408,5120) | 323.0 | 378.7 | 299.2 | 263.6 | 351.8 | **239.8** | -25.7% | 1.31 |
| attn_output/ssm_out (5120,6144) | 132.4 | 148.5 | 128.3 | 104.8 | 144.4 | **97.8** | -26.1% | 1.51 |
| attn_qkv (10240,5120) | 198.8 | 231.0 | 184.7 | 161.0 | 214.8 | **148.9** | -25.1% | 1.38 |
| attn_gate (6144,5120) | 129.6 | 149.0 | 121.5 | 103.8 | 136.2 | **96.7** | -25.4% | 1.49 |
| attn_q (12288,5120) | 234.4 | 273.9 | 215.5 | 190.1 | 252.7 | **175.4** | -25.2% | 1.35 |

The prediction is confirmed and beaten: skinny's flat ~385 reproduced (380.9), and the
scalar cell came in at 264, under the ~310 the linear model promised. **w5r4h wins every
projection by 25-31%.**

Lever attribution (ffn_down): 4-row tile (r4 vs r2) -16.8%; half product (r4h vs r4)
-6.7%. **The half product INVERTS at 2 rows** (r2h vs r2 +13.8%, worse than skinny) -
the fold only pays when there are enough independent accumulator chains to cover the
half-pipe latency; at width 4 the same lever was -4.0% on a 4-row kernel. Do not apply
half-product to low-row-count bodies on trend.

## Per-instruction decode of the winner (ffn_down, per dispatch)

Capture `GGML_METAL_CAPTURE_COMPUTE=2` + headless replay; decode JSON
`profiles/shaderprof-decoded/w5-r4h-ffndown.json`, archive `profiles/aug28-w5-crossover.tar.zst`.

| | v3 (w4, record) | **w5r4h** | verify_m4 (w4, theirs) |
|---|--:|--:|--:|
| exec/dispatch | 24.92M | **27.50M** | 24.58M |
| exec per COLUMN | 6.23M | **5.50M** | 6.15M |
| issue/stall | 87.0/13.0 | **89.5/10.5** | 90/10 |
| hot-loop instructions | 283 | **314** | 280 |
| largest stall site | 1.08% | **0.91%** | 0.52% |

Two saturation reads:

1. **The kernel is issue-saturated.** 89.5/10.5 equals the best issue/stall ever
   observed on this board (their verify_m4's 90/10); the largest stall site is 0.91%.
   There is no stall headroom left to buy - any further win at this width must remove
   instructions, not hide latency.
2. **It is NOT bandwidth-saturated.** 1.31-1.51x the stream floor across the shapes
   (skinny sits at ~2.1x on ffn_down). The gap to floor is the issue-bound FMA stream.

Per column the w5 form is MORE economical than the width-4 winner (5.50M vs 6.23M
exec/col): the fixed per-dispatch overhead (dequant, addressing, reduction) amortizes
over more columns. Scalar efficiency IMPROVES with width - and still loses at 7.

## What this pins

- **The crossover is between widths 5 and 7.** Scalar wins width 5 by 25-31%
  (264 vs 381 on ffn_down); the same form loses width 7 by 1.5x (580 vs 386,
  `m4-width4-r4kp.md`). Width 6 is the one untested cell if an exact pin is ever needed.
- **The linear-scaling model survives at 5 and breaks by 7, in opposite directions.**
  It predicted 310 at width 5; measured 264 (better - per-column economy improves with
  width). Extrapolating that to width 7 predicts ~370; measured 580. Something turns
  SUPERLINEAR between 5 and 7 for the scalar form - and it is not spill (w7 prescreens
  clean) and not simply register count (w7_r2 with 2-row state is just as bad, 604).
  With w5r4h at 89.5% issue there is no stall slack anywhere: the working hypothesis
  for the parked "why" question is that past ~5 columns the scalar FMA stream's issue
  cost grows faster than the 8-wide MMA tile's fixed cost, while whatever inflates the
  w7 scalar stream superlinearly (encoding pressure? operand bank conflicts on 7 live
  half8 vectors?) is exactly the thing to decode when that question is opened. The
  w5-vs-w7 decode pair is now half-captured: this file has the w5 side.
- **MTP d4 skinny arm carries a layout conflict the w5 arm removes**: with
  `GGML_MV_SOA_W4=1` + `GGML_MM_SKINNY=5` + repack, MTP's width-4 draft-path ops want
  the SoA layout while width-5 skinny verify wants `_di`, same tensors, first use wins,
  the loser falls back to interleaved (the depth re-sweep's recorded MTP d4 already
  carries this). In the w5 arm both widths read SoA. Part of any MTP d4 delta is
  therefore layout, not kernel - the dflash n4 pair is the clean kernel read.

## End-to-end (2026-08-28, `run-m4-width5-e2e.sh`, TSV `results/m4-w5-e2e-aug28.tsv`)

Arms: skinny (`GGML_MM_SKINNY=5`) vs w5r4h (`GGML_MM_SKINNY=6` + `GGML_MV_SOA_W5=4` +
`GGML_MV_SOA_W5_HALF=1`), both over prod-pick flags + `GGML_MV_REPACK=1` +
`GGML_MV_SOA_W4=1` + `R4KP=3`, n_predict 600, 4 order-balanced reps per point, fresh
server per run.

| point | skinny (4 runs) | w5r4h (4 runs) | delta | sha1 |
|---|---:|---:|---:|---|
| dflash n4 | 20.369 (20.347-20.389) | **25.632** (25.606-25.660) | **+25.8%** | `3776c0adb7ee`, all 8 |
| MTP d4 | 19.585 (19.540-19.616) | **24.527** (24.494-24.562) | **+25.2%** | `3776c0adb7ee`, all 8 |
| ctrl dflash n3 | 25.282 (2 runs) | 25.262 (2 runs) | -0.08% | `a08f1b87121c`, all 4 |

- The control is inert and byte-identical - the selector touches nothing at n3.
- The synthetic slice transfers ~1:1 again: predicted 25.3 ms/round from the six mv
  shapes, observed ~30 at 2.99 committed/round (the excess is the also-rerouted
  lm_head + attn_k/v smalls).
- MTP transfers within 0.6 pp of dflash, so the MTP d4 layout conflict noted above is
  worth little at the round level - the kernel is the story.
- **Numerics note for the adoption call: the w5r4h arm is byte-identical to the skinny
  arm at BOTH depth-4 points** (same sha per point, acceptance unchanged at 49.8/49.0).
  The incumbent skinny route accumulates in `simdgroup_half8x8` - half products are not
  a numerics regression relative to what width 5 runs today. This is one trajectory,
  not a KLD study (`run-quant-kld.sh` is the tool if the owner wants it priced).

**The operating point is in question AGAIN, one day after the depth re-sweep.** On this
board, same session, same harness: dflash n4+w5 25.632 vs dflash n3 25.282 = **+1.4%**.
The re-sweep's item 4 said a ~20% width-5 saving could not close a 4.7 t/s gap -
measured at +25.8% it closes it and passes. MTP d4+w5 (24.527) also clears the re-sweep's
MTP d3 (24.09, other board). The best-known config is now **dflash n4 + w5r4h + v3 +
repack**, round ~117 ms at 2.99 committed vs n3's ~111 ms at 2.81 - depth 4's extra
committed tokens now outrun its extra round cost. Depth 5 (width 6) is the obvious next
cell: the w6 kernel does not exist, and the re-sweep's n5 (21.86 on skinny) would need
~+16% from a w6 kernel to reach 25.3 - inside the 25-31% this family has delivered
twice. The depth-optimum question is open until the width-6 cell is measured.

## Open

1. ~~**Adoption (owner's call)**~~ **TAKEN 2026-08-28: the pick moved to dflash n4 +
   w5r4h + v3 + repack ("pick this for now").** The caveats were accepted, not
   resolved: repack residency stays open (`repack-inplace.md` is the fix path) and
   the half-product numerics rest on the byte-identical note above plus the
   half-accumulate incumbent, not a KLD study.
2. **Width 6 (depth 5) cell** - promoted from "only if needed": two consecutive width
   cells beat their skinny arm by 25%+, and depth 5 needs only +16% to contend. Staging
   is mechanical from the w5/w7 template; prescreen r2/r4 first.
3. The "why" decode pair: capture skinny + scalar at width 7 and diff against this
   file's w5 decode when the owner reopens the parked question. The w5 datum says
   scalar economy IMPROVES per column through 5 and collapses superlinearly by 7.
