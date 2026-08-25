# Sumy-fold in the width-4 SoA kernel - refuted

Status: **refuted 2026-08-25.** Candidate 1 of `verify-width-instruction-economy.md`
("fold the -8 offset and scale via the sumy identity instead of two half4
convert+subtract chains per pack per row") is measured at **+15% to +32% per pass**
against the landed R2 kernel on both ffn shapes, in every variant tried, including the
4-row tile where the fold's y-side overhead amortizes best. Probe kernels + routing live
on branch **`m4-width4-sumy-fold`** (9a3534330, one commit ahead of prod, not pushed);
nothing lands on prod.

## What was built

Three probe kernels next to the SoA family in `ggml-metal.metal`, routed behind
`GGML_MV_SOA_W4_SUMY` (same gates as the other SoA variants: `GGML_MV_SOA_W4=1`,
ne11=4, f16y, repack, whitelisted row counts):

- `r2_sumy` (`GGML_MV_SOA_W4_SUMY=2`): full nc-style fold on the R2 tile. Nibbles stay
  at bit position (mask only, no shift, no -8, no half intermediate:
  `float4(uint4(q & 0xFFFF) & uint4(0xF, 0xF0, 0xF00, 0xF000))`), y is pre-scaled by
  exact powers of two (1, 1/16, 1/256, 1/4096) per column per pack, and the offset comes
  back as `-8*d*sumy` per accumulator. Products are bit-identical to R2's by the
  power-of-2 argument (n*16^k times y*16^-k rounds to n*y exactly); only the final
  combine order changes.
- `r2_sumymin` (`=3`): minimal delta - R2's shift-based expansion kept, ONLY the `-8.h`
  subtract moves into the sumy term.
- `r4_sumy` (`=4`): the full fold on the 4-row K1 tile, where the per-pack y-side fold
  work is amortized over twice as many rows.

All three pass test-backend-ops 1155/1155 (`GGML_MV_REPACK=2 GGML_MV_SOA_W4=1
GGML_MV_SOA_W4_SUMY=N ./build/bin/test-backend-ops -o MUL_MAT -b MTL0`).

## Prescreen, then measurement

Offline prescreen first (`skills/metal-kernel-prescreen`, regression-checked against
ext r1_4's known 32 B spill), then `test-backend-ops perf` on this build
(prod da2a137f0 + the probe commit), same session for every row. Floors are q4_0 weight
bytes at 273 GB/s; both shapes stream 50.14 MB = 183.7 us.

| kernel | text B | spill | ffn_down us (x floor) | vs base | gate/up us | vs base |
|---|---:|---:|---:|---:|---:|---:|
| R2 (base) | 2184 | 0 | 331.27 (1.80) | - | 295.96 (1.61) | - |
| r2_sumy | 2384 | 0 | 436.28 (2.38) | **+31.7%** | 391.76 (2.13) | **+32.4%** |
| r2_sumymin | 2286 | 0 | 381.56 (2.08) | **+15.2%** | 339.05 (1.85) | **+14.6%** |
| K1 (r4 base) | 3988 | 0 | 372.53 (2.03) | - | 324.22 (1.77) | - |
| r4_sumy | 4396 | 0 | 425.72 (2.32) | +14.3% vs K1 | 381.69 (2.08) | +17.7% vs K1 |

Controls: R2 today (331.27) reproduces the archived 334.9 from
`verify-width-instruction-economy.md` to -1%; `test-backend-ops perf` shape
`m=5120,n=4,k=17408` / `m=17408,n=4,k=5120`.

## Why it loses (and why that confirms the economy law)

The fold moves work from the weight side to the activation side. What it removes is per
row per pack: the shift/mask/convert/subtract chains, which the compiled numbers say are
nearly free (the sharpest cell: `r2_sumymin` deletes ONLY eight half-subtracts per row
per pack, buys four sumy reductions per pack plus one fma per accumulator, and loses
15%). What it adds is per **column** per pack: the sumy reduction and the pre-scaled y
copies. Y-side work per weight byte is exactly the term that scales with verify width -
the economy doc's own law - so at 4 columns the added y-side ops dominate the removed
row-side ops, and doubling the row amortization (r4) does not catch up.

Corollary worth keeping: **nc2's 20.8 instr/B does not come from its offset arithmetic;
it comes from having 2 columns.** The sumy identity's economics invert as columns grow,
so porting nc-style dequant "up" in width was never going to transfer.

Method note: the offline text-size proxy under-called the damage 3x (+9.2% text ->
+31.7% time for r2_sumy) but got the SIGN right on all three variants before any build.
As a gate it worked; do not read its magnitude as a prediction.

## What this retires and what stays open

- Retired: candidate 1 (sumy-fold dequant), all in-kernel forms. A fold can only pay if
  the y-side terms leave the kernel entirely - e.g. sumy and pre-scaled y emitted by the
  f16y convert pass and loaded per pack - which changes memory traffic and is a
  different experiment, unsized and unclaimed.
- Untouched and still live: candidate 2 (base-pointer + increment addressing in the SoA
  tile) and candidate 3 (bf16/f16-pair y with fp32 accumulation).
- Also still live: the skinny tg-L1 staging wall (task 2 of the 2026-08-25 board).
