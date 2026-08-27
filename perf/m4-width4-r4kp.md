# Width 4: the verify_m4 parity kernel - and the lever was codegen, not K-split

Status: **open 2026-08-27 - kernel-level result in, e2e running.** Branch `m4-width4-r4kp`
off prod `15bfa8cee`.

Opened from `omlx-verify-m4-decode.md`'s conclusion: a kernel combining r2's economy with
kp2-style K-split latency hiding, target 283 us on ffn_down. **The target is beaten -
240 us on ffn_down, 26-29% under R2 on every routed projection, faster than their
verify_m4's own best config on our own gs32 Q4_0 format.** But the attribution the target
came with was wrong, and the decomposition below corrects it: the K-split is worth ~1%,
the tile ~4%, and the payload is a source-level codegen change worth ~21% that every
kernel in the SoA family had been leaving on the table.

## The decomposition that found it

Step 1 - their morphology transplanted naively (`_r4kp`, env `GGML_MV_SOA_W4_R4KP=1`):
4x4 tile, scalar broadcast dequant, kp2 contiguous K halves, their exact dispatch
geometry (1280 tgs x 64 threads at ffn_down). **Slower than R2 on every shape (+2 to
+8%)**, and the per-instruction decode showed why: exec/dispatch 29.5M (R2: 30.4M - no
stream cut) and stall 20.8% (R2: 23.0 - no stall cut). Neither of their measured
advantages materialized from morphology alone, refuting `omlx-verify-m4-decode.md`'s
"the K-split lane structure is the demonstrated mechanism" by construction.

Step 2 - their kernel re-measured at OUR format point. `verify_matmul` at gs32:
**exec/dispatch 24.58M and 89.4/10.6 issue/stall, IDENTICAL to their gs64 numbers** -
the count edge and the stall edge are not format-coupled (corrects the "not all of the
19% is reachable without requantizing" caveat). gs32 does cost them +13.8% wall time
(310.1 vs 272.5 us chained, `bench_chain` method) at unchanged count and stall - the
gs64 win is issue-rate, not instructions. bf16 y costs them +16% when swapped to fp16
(their `float(v[ki])` converts; our f16 folds are free).

Step 3 - the encoding-size histograms. Their hot loop: 280 instructions, 162 of them
10-byte (all the arithmetic in compact wide-operand forms). Ours (R2, K2, scalar-R4KP
alike): ~335-346 per 4-row-equivalent iteration, flooded with 4/6-byte helper ops -
same Metal compiler (theirs is `mx.fast.metal_kernel`, compiled on this machine), so
the difference had to be source-shape-induced codegen. Offline `applegpu-nt` probes of
source variants found it: **signed-int indexing plus per-row planar pointers hoisted
out of the K loop** (`sp[block]` / `qp[p]` instead of `row + 2*block` / `row + 2*nblk +
4*p` recomputed per row per iteration) cuts static instructions 13% before touching
anything else.

## The kernels (all behind `GGML_MV_SOA_W4_R4KP=<n>`, SoA layout unchanged)

| n | kernel | tile | K | product | note |
|---|---|---|---|---|---|
| 1 | `_r4kp` | 4x4 | kp2 | f32 | naive morphology transplant, refuted |
| 2 | `_r4kp_v2` | 4x4 | kp2 | f32 | + int indexing, hoisted planar row pointers |
| 3 | `_r4kp_v3` | 4x4 | kp2 | half | v2 with the product in half |
| 4 | `_r4kp_v4` | 2x4 | full | f32 | v2 codegen at R2's geometry (isolates codegen) |
| 5 | `_r4kp_v5` | 4x4 | full | f32 | isolates the K-split |

All five: zero spill (offline prescreen; probe regression-checked against R2's recorded
2184/0), 1155/1155 `test-backend-ops -o MUL_MAT` vs CPU with the route forced, pipeline
engagement confirmed in the debug log (trap 3). v2..v5 hoist all four (two) row pointers
without a per-row guard - safe because the SoA whitelist only routes row counts
divisible by 4.

## Synthetic per-shape A/B (test-backend-ops perf, interleaved, GGML_MV_REPACK=2)

us/run means (2-3 reps, spread <1.5%):

| projection (m,k at n=4) | R2 | v4 | v5 | v2 | v3 | v3 vs R2 |
|---|---:|---:|---:|---:|---:|---:|
| ffn_down (5120,17408) | 335.4 | 265.5 | 253.3 | 250.4 | **240.3** | -28.4% |
| ffn_gate/up (17408,5120) | 300.3 | 242.4 | 235.6 | 237.3 | **223.0** | -25.7% |
| attn_output/ssm_out (5120,6144) | 116.0 | 99.8 | 92.9 | 92.4 | **86.9** | -25.1% |
| attn_qkv (10240,5120) | 186.2 | 148.5 | 145.4 | 144.6 | **136.0** | -27.0% |
| attn_gate (6144,5120) | 115.5 | 96.2 | 93.6 | 92.6 | **86.0** | -25.5% |
| attn_q (12288,5120) | 219.1 | 176.8 | 171.4 | 170.5 | **161.1** | -26.5% |

Lever attribution (ffn_down): codegen form (v4 vs R2) **-21%**; 4-row tile (v5 vs v4)
-4.6%; kp2 (v2 vs v5) -1.1% (consistent with `m4-width4-r2k2.md`'s ~1%); half product
(v3 vs v2) -4.0%.

Cross-framework, same shape, same session method: their verify_m4 gs64 bf16 chained
272.5-283.3 us. **v3 at 240 us is 1.13x faster than their best config; on gate/up
(223.0 vs 283.8) 1.27x.** Stream ceiling ~200 us puts v3 at ~1.2x ceiling where their
kernel sits at 1.42x (`omlx-verify-m4-decode.md`).

## The per-instruction account (ffn_down, per dispatch)

| | R2 | r4kp (naive) | **v3** | theirs (gs32 or gs64) |
|---|--:|--:|--:|--:|
| exec/dispatch | 30.37M | 29.48M | **24.92M** | 24.58M |
| issue/stall | 77.0/23.0 | 79.2/20.8 | **87.0/13.0** | 90/10 |
| hot-loop instructions | 173 (x2) | 335 | **283** | 280 |
| largest stall site | 4.63% | 1.99% | **1.08%** | 0.52% |

Both of `omlx-verify-m4-decode.md`'s levers landed - dynamic stream -18%, stall 23 -> 13
- and both came from the codegen form, not the K-split it predicted. At matched count
and stall we are FASTER than their kernel because the f16 operand folds keep our
per-instruction issue cost below their bf16 forms (their fp16 arm is +16% on their own
kernel; `width4-y-operand-width.md`'s fold finding, now demonstrated cross-framework).
Decode JSONs: `kvquant-experiments/profiles/shaderprof-decoded/{r4kp,v3,verify-m4-gs32}*.json`.

The law from `verify-width-instruction-economy.md` refines once more: R2 and v2 differ
~1% in STATIC per-work instruction count at 2-row scale (v4's static text is R2-sized),
yet v4 is -21% - because the removed instructions were the per-iteration 64-bit address
recomputation chains feeding the loads, i.e. the priced quantity is dynamic issue+stall
cost, and address-generation instructions were both numerous IN THE LOOP and attached to
the two load-consumer stall sites (R2's 0x44a/0x1e8 at 8.2 points, v3's worst site 1.1).

## Open

1. **e2e A/B running** (`run-m4-width4-r4kp-e2e.sh`, arms r2/v2/v3, DFlash n3 point,
   n6 inertness control). Expected order ~-15 to -20 ms on the ~135 ms n3 round if the
   ~57 us/pass serialized saving translates; trajectory sha will differ from R2's (the
   scalar accumulation order and kp2 partial add change FP association; v3 additionally
   rounds the product to half - `test-backend-ops` NMSE passes, KLD unmeasured).
2. **Numerics choice is the owner's**: v2 keeps f32 products (association-only change);
   v3 adds half-product rounding for ~4%. If v3 is adopted, price it with
   `run-quant-kld.sh` first.
3. **The codegen form should transfer to every kernel in this family** - width 2/3 (nc,
   ext), the drafter's projections, and above all **the width-7 skinny wall**: a 7-column
   scalar-broadcast mv kernel with this form was never buildable before because the
   economy wasn't there; at v3's instr/B it may now beat `mul_mm_skinny`'s staging
   round-trip at the prod width. That is the next experiment and the first one with a
   shot at moving the prod pick itself.
4. Whole-graph pp4 and the round decomposition after e2e completes.
