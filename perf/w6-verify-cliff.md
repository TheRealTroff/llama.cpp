# The width-6 verify cliff on the UD line (2026-09-16 night, branch `exp/w6-verify-cliff`, worktree `llama.cpp-fa24`) - OPEN

Status: **opened on the owner's ask, nothing built yet.** Found by the depth-5 e2e sha gate of the FA tile widths
(`fa-decode-tile24.md` widths section): depth 5 (verify width 6) on the UD line runs at 16 t/s at 8K against 24-25
at depths 2-4; the q4 line at depth 5 runs 26.3.

## Evidence (prod `c85ad08e6`, Turbo4, 300 tokens, benchprompt at 8K, `PICK_DEPTH=5`)

| line | depth 5 t/s | acc | sha | the width-6 matmul route (`LV=5` pipeline names) |
|---|--:|--:|---|---|
| ud | 15.85 | 49.5% | `a409bb1b45df` | ~~the mul_mm prefill tile kernels~~ **`kernel_mul_mv_ext_{q4_K,q5_K,iq4_xs}_soa_f16_r1_3` (nr0 2): the ext SoA reader at three columns per pass, two weight passes for the six columns** |
| q4 | 26.29 | 46.4% | `04ada3a4de10` | the skinny SoA MMA kernel (`GGML_MM_SKINNY=6`, Q4_0_SOA widths 6-8) |

Why: the UD line's stored-SoA decode kernels (`kernel_mul_mv_{q4_K,q5_K}_soa_w{3,4}_v2`, `iq4_xs_soa_w{3,4}_v5`, the
w5 forms) cover widths 3-5 (`kq_soa_shape`: `ne11 >= 3 && ne11 <= 5`, ggml-metal-ops.cpp); the "remaining" formats
(q6_K / q3_K / iq4_nl / iq3_s) take the kq body as column groups at 6-8 (`w4cg`, `ud-remaining-quants.md`), but the
bulk of the UD tensors are Q4_K / Q5_K / IQ4_XS, and at width 6 those fall through to the ext SoA readers
(`kernel_mul_mv_ext_soa_q4x4`, r1ptg 3: the six columns as two passes of three, each streaming the weights and
dequantizing them again). The skinny SoA route (`stored_soa_skinny`) is Q4_0_SOA only. **Correction (same night):
the first write-up named the mul_mm prefill tile kernels as the width-6 route, read off a pipeline list that also
carried the prefill pipelines of the 8K prompt - the `verify-before-generalizing` trap; the r1_3 ext pipelines
above are the ones that disappear when the skinny route engages.**

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

## Probe 1: the generic skinny tile over the stored SoA rows at widths 6-8 (2026-09-16 night, commits `bb54320ea` + `0b4bfac46`)

`c881ba34a` cherry-picked; `kernel_mul_mm_skinny_t` reads the stored rows through the mul_mm bodies' `dequantize_soa_mm`
under `FC_mul_mm_soa` (the host names the base type's kernel and sets the constant for `*_SOA`), `GGML_MM_SKINNY_GEN=6`
routes ne11 6-8 only - widths 2-5 keep their SoA kernels (the width-4 refutation stands). UD line, Turbo4, depth 5,
300 tokens, benchprompt at 8K, ABAB, `LV=5` (the r1_3 ext pipelines are gone in the tile arm; the
`kernel_mul_mm_skinny_{q4_K,q5_K,iq4_xs,q6_K,q3_K,iq4_nl,iq3_s}_f32_soa` pipelines are present):

| arm | decode t/s | acc | sha |
|---|--:|--:|---|
| base (the ext SoA reader, r1_3 x 2 passes) | 15.84 / 15.85 | 49.5% | `a409bb1b45df` |
| `GGML_MM_SKINNY_GEN=6` | **19.56 / 20.06 (+24..27%)** | 49.5% | `a409bb1b45df` (~~byte-identical~~ the sha held at 8K only - it moves at 96K, see below: NUM-TG) |

One weight pass with the MMA tile against two passes of the dequant-once reader: the same tile that lost 13.6% at
width 4 (where the incumbent was one pass) wins 24-27% at width 6. Depth 5 still trails depth 3 (27.8 at 8K) - this
is a widths lever for adaptive speculation, not a depth change. Not yet: widths 7-8 timed (same route), the 96K
number, the per-call table (`test-backend-ops` MUL_MAT cases for the `*_SOA` types), the q4 line (its skinny SoA
kernel already covers 6-8). Status: **built, priced at 8K and 96K, NOT byte-identical (96K sha moved); a NUM-TG lever needing the decode-path KLD before a pick; adoption = owner** (manifest entry to
follow as proposed if the owner wants it in a pick).

### Probe 1 at 96K (owner's ask, in this order: 96K, then widths 7-8) - the sha moves

`perf/run-longctx-pick.sh`, UD, Turbo4, `-c 102400`, `longprompt-96k.txt` (95508 tokens), depth 5, 300 tokens, one
pair, the fa24 build (`repo` line checked). Round time = predicted_ms / rounds with rounds = n / (1 + depth x acc):

| arm | prefill | decode t/s | acc | rounds | **verify round** | sha |
|---|--:|--:|--:|--:|--:|---|
| base (ext SoA reader, r1_3 x 2) | 1100.8 s | 11.20 | 37.7% | 104.0 | **257.6 ms** | `e867940fe47f` |
| `GGML_MM_SKINNY_GEN=6` | 1101.0 s | 14.16 | 41.7% | 97.2 | **217.9 ms (-15.4%)** | `12b7e25d7a6d` |

-15% per round at 96K (the matmul share of a depth-5 round is smaller there, as sized), +26% t/s of which part is the
forked text's better acceptance. **The sha moved: the tile is NOT byte-identical.** It held at 8K over 300 tokens
(`a409bb1b45df` in both arms, twice) and forks at 96K - the MMA tile sums the K dimension in a different order
from the scalar ext reader (half products into a float 8x8 accumulate, 64-wide slices), so it is a decode-numerics
change of the same kind as the skinny kernel on the Q4_0 line, not a routing change. **Class: NUM-TG**, which on
the UD line means the decode-path pairwise KLD (`-b 4 -ub 4` against the kept V1 decode base, `kld-reference-limits`)
before any pick, and the owner's explicit take. The 8K sha match was a 300-token coincidence, and this note's
"byte-identical" above is struck to "sha held at 8K only". Depth 5 at 96K itself is far off the depth-3 pick (11-14
vs ~20 t/s, acceptance 38-42% vs 57%): the lever's value is the width-6..8 rounds of adaptive speculation.
