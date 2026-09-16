# The width-6 verify cliff on the UD line (2026-09-16 night, branch `exp/w6-verify-cliff`, worktree `llama.cpp-fa24`) - OPEN

Status: **opened on the owner's ask, nothing built yet.** Found by the depth-5 e2e sha gate of the FA tile widths
(`fa-decode-tile24.md` widths section): depth 5 (verify width 6) on the UD line runs at 16 t/s at 8K against 24-25
at depths 2-4; the q4 line at depth 5 runs 26.3.

## Evidence (prod `c85ad08e6`, Turbo4, 300 tokens, benchprompt at 8K, `PICK_DEPTH=5`)

| line | depth 5 t/s | acc | sha | the width-6 matmul route (`LV=5` pipeline names) |
|---|--:|--:|---|---|
| ud | 15.85 | 49.5% | `a409bb1b45df` | `kernel_mul_mm_q4_K_f16_bci`, `mul_mm_q5_K_f16_bci`, `mul_mm_q6_K_f16_bci`, `mul_mm_q4_0_f16_bci` (the prefill tile kernel at N = 6) |
| q4 | 26.29 | 46.4% | `04ada3a4de10` | the skinny SoA MMA kernel (`GGML_MM_SKINNY=6`, Q4_0_SOA widths 6-8) |

Why: the UD line's stored-SoA decode kernels (`kernel_mul_mv_{q4_K,q5_K}_soa_w{3,4}_v2`, `iq4_xs_soa_w{3,4}_v5`, the
w5 forms) cover widths 3-5 (`kq_soa_shape`: `ne11 >= 3 && ne11 <= 5`, ggml-metal-ops.cpp); the "remaining" formats
(q6_K / q3_K / iq4_nl / iq3_s) take the kq body as column groups at 6-8 (`w4cg`, `ud-remaining-quants.md`), but the
bulk of the UD tensors are Q4_K / Q5_K / IQ4_XS, and at width 6 those fall through to the mul_mm tile kernels - a
64-column MMA tile with 6 live columns, the kernel built for prefill. The skinny SoA route (`stored_soa_skinny`) is
Q4_0_SOA only.

**It is context-independent**: the weight matmuls cost the same at 96K as at 8K, so the absolute penalty per round
(~20 ms of a ~50 ms round at 8K) does not shrink at long context; only its share does, as the FA call grows. The FA
side of width 6 is already the 24 + 16 plan (-9% per call at 96K, -11% at 8K) and scales with context like the other
widths (a 24 + 16 plan is 1.87x the width-4 tile at 96K, 1.62x at 8K: the 16-row tile's fixed cost, not the stream).

## The lever

A width-6..8 form for the K-quant SoA kernels, two candidate shapes (both measured on other formats already):

1. the column-group form the remaining formats use (`w4cg`: ceil(ne11/4) groups of the 4-column body) applied to
   Q4_K / Q5_K / IQ4_XS - the cheapest port, expected at the w4 kernel's per-column cost x 2 groups (i.e. width 6 at
   ~2x width 4's matmul time, against the mul_mm cliff's ~4x);
2. a skinny MMA tile for the K-quant SoA layouts (the Q4_0_SOA skinny kernel's form, `skinny-soa.md`), the real
   width-6..8 kernel, a bigger build. **Prior art (owner's question, 2026-09-16): a K-quant skinny tile WAS built -
   2026-09-04, branch `ud-skinny-generic`, commit `c881ba34a`, `kernel_mul_mm_skinny_t<block_q, nl, dequantize_func>`
   over the AoS block dequantizers for q8_0/q3_K/q4_K/q5_K/q6_K/iq3_s/iq4_nl/iq4_xs, `GGML_MM_SKINNY_GEN=N`
   (`ud-model.md` step 5). Byte-identical, zero spill, REFUTED at width 4: e2e 17.82 -> 15.40 (-13.6%), q5_K
   [5120,17408] 455 -> 582 us/call - the incumbent there (the ext r1_4 mv family) already dequantizes once per pass,
   so the tile removed nothing and added the threadgroup round trip plus the generic dequant form. Not merged;
   not in prod.** Two things that verdict does not cover: (a) width 6-8, where the incumbent is the mul_mm cliff, not
   the ext family - the generic tile at ~1.25x the ext kernel's per-call time would still be well under the cliff
   (never timed at width 6: the measurement was depth 3); (b) the SoA planar layouts (Q4_K_SOA / Q5_K_SOA /
   IQ4_XS_SOA, built Sep 5-9 after the refutation) - the generic tile read AoS blocks through the 16-element
   `dequantize_*` form, which was half of what it lost. So the cheap first probe on this branch is to cherry-pick
   `c881ba34a`, route it at ne11 6-8 only (`GGML_MM_SKINNY_GEN=6`), and time depth 5; the SoA-reading tile is the
   real form if that probe lands short of the w4cg port.

Gate: depth-5 e2e sha `a409bb1b45df` must hold (the route is a BI change if the per-column arithmetic is the w4
kernel's); price = the depth-5 round at 8K and 96K, and whether depth 5 then beats depth 3 anywhere (the depth
sweep at long context favoured depth 3 with width 4's tile; adaptive speculation makes every width live).
