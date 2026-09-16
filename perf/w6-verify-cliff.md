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

### Probe 1 at widths 7 and 8 (depth 6 and 7, UD, Turbo4, 300 tokens, benchprompt at 8K, ABAB, fa24 build)

| depth (width) | arm | decode t/s | acc | rounds | **verify round** | sha |
|---|---|--:|--:|--:|--:|---|
| 6 (7) | base | 15.75 / 16.00 | 46.7% | 78.9 | 239.4 ms | `9128633c6cfa` |
| 6 (7) | `GGML_MM_SKINNY_GEN=6` | **20.70 / 20.69 (+30%)** | 45.7% | 80.2 | **180.8 ms (-24.5%)** | `a409bb1b45df` |
| 7 (8) | base | 15.63 / 15.78 | 39.6% | 79.5 | 240.2 ms | `9128633c6cfa` |
| 7 (8) | `GGML_MM_SKINNY_GEN=6` | **20.70 / 20.65 (+31%)** | 39.9% | 79.1 | **183.5 ms (-23.6%)** | `a409bb1b45df` |

The same -24% per round as width 6 at 8K: the ext reader at widths 7-8 runs r1_4 x 2 passes, the tile one pass.
The sha moves here at 8K already (base = the depth-3 canonical text `9128633c6cfa`, the tile = the width-6 text
`a409bb1b45df`), which settles the class: **NUM-TG on the UD line at every width it touches**; at widths 7-8 the FA
runs the plain batched route in both arms (the GQA tile plan covers widths 3-6 only), so this is the matmul alone.
Depths 6-7 at 8K (16 -> 20.7 t/s) stay below depth 3 (27.8), as expected - the lever is per-round, for the wide
rounds of adaptive speculation.

**Status (2026-09-16 night): probe 1 priced at 8K (widths 6/7/8: round -24..-25%) and 96K (width 6: -15%); NOT
byte-identical (NUM-TG). Owner: "wait for that until picking". The pick prerequisite is the UD line's decode-path
pairwise KLD (`-b 4 -ub 4` vs the kept V1 decode base) at widths 6-8, then the owner's take; manifest entry
`GGML_MM_SKINNY_GEN=6` proposed, class NUM-TG, ud line (the q4 line's Q4_0 tensors keep their own skinny SoA
kernel; the route would touch only its q6_K/q8_0 tensors at widths 6-8, unmeasured there).** The SoA-layout-native
tile (candidate 2) stays open only if the KLD refuses this form.

### The pick prerequisite: pairwise decode-path KLD at width 6 (owner 2026-09-16: "we need the KLD"; launched 17:22)

Design (`kld-reference-limits`, `ud-remaining-quants.md` "Pairwise"): the same UD SOA-V2 file on both sides, the ud
f16 pick env, `-b 6 -ub 6` on both (every position's logits through the width-6 decode kernels: 342 six-token
steps per 2048-token chunk, 24 chunks, the wikitext positions of every KLD table), f16 cache. Base = the pick as
it runs today at width 6 (the ext SoA reader at r1_3), written with `--kl-divergence-base` (12.2 GB,
`logits/kld-base-kld-pair-v2dec6-sep16.dat`, kept as the width-6 decode base). Test 1 = the same file and env
(the self-score floor of the logits file). Test 2 = `GGML_MM_SKINNY_GEN=6` (the tile). Scale: q8_0 sits 0.0012
mean KLD / 99.08% same-top from the trained model; the four new UD formats' width-4 kernels cost 5e-6 pairwise.
Widths 7-8 run the same tile with the same per-column arithmetic (the column count only sets the B tile), so the
width-6 pair is the gate; a -b 8 pair can follow if the owner wants it stated. Driver: the session scratchpad's
`kld-w6.sh` (`run-quant-kld.sh` twice with the same TAG: the base is reused, LABEL distinguishes the test logs).

**Result (17:22-18:30, base 49 min, tests ~19 min each; logs `kld-pair-v2dec6-sep16-*-{self,gen6}.log`):**

| test arm vs the width-6 decode base (the pick's ext SoA reader) | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap (1-TV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| the same file and env (the logits file's floor) | 0.000000 | 0.000000 | 0.000037 | 0.000051 | 0.000073 | 100.000 | 99.936 |
| **`GGML_MM_SKINNY_GEN=6` (the skinny SoA tile)** | **0.000488 +/- 0.000370** | 0.000012 | 0.000535 | 0.012607 | **9.05** | **99.670 +/- 0.037** | 99.707 |
| scale: the four new UD formats' width-4 kernels vs V1 (`ud-remaining-quants.md`) | 0.000005 | 0.000000 | 0.000042 | 0.000247 | 0.032 | 99.935 | 99.900 |
| scale: the pick's decode path vs its prefill path | 0.000026 | 0.000001 | 0.000069 | 0.001866 | 0.177 | 99.914 | 99.861 |
| scale: the fork's native width-4 kernels vs the pick | 0.000445 | 0.000013 | 0.000570 | 0.011329 | 8.43 | 99.678 | 99.706 |
| scale: q8_0 vs the bf16 model | 0.0012 | 0.00018 | | 0.105 | 5.4 | 99.08 | |

Reading: **the tile is a real numerics perturbation, not a quality-free routing change.** Mean 4.9e-4 = 20x the
pick's own decode-vs-prefill gap and 100x the last decode kernel that was priced, 0.4x q8_0's distance from the
trained model; the mean is carried by the tail (median 1.2e-5, one position at 9 nats - an argmax flip at a
confident position, the same shape and size as the native-kernels arm's 8.4); same-top -0.33 pt = 81 of 24,552
positions, overlap -0.23 pt. It sits in the same class as "the fork's native width-4 kernels vs the pick" (4.5e-4,
99.68%), i.e. two different-but-legitimate decode arithmetics of the same weights - the pairwise cannot say which
side is closer to the model (against bf16 both would read ~0.0128 inside a 0.0015 error bar; the bf16 file lives on
the unmounted offload volume). What differs mechanically: the tile rounds the dequantized weights to half before
the MMA (the ext reader multiplies in float) and sums K in 64-wide 8x8 slices.

**Status: priced. Class NUM-TG at 4.9e-4 / 99.67% same-top pairwise; speed -24% per round at widths 6-8 (8K),
-15% at 96K (width 6). Adoption = owner** - this is above the UD line's "BI/SPEC only" standard and the owner's
explicit take is the gate. If refused: the SoA-native tile with float products (candidate 2, exact dequant into the
A tile as float, or a float8x8 A operand at half the MMA rate) is the form that would keep the width-6 speed on
the pick's numerics - unbuilt. The width-6 base (12.2 GB) stays at `logits/kld-base-kld-pair-v2dec6-sep16.dat`.

## Probe 2: the float-product form (owner: "investigate the float-product SoA tile"; commit `451bb6032`)

`GGML_MM_SKINNY_GEN_FP=1`: the same tile with `a_t = float` - the exact dequant lands in a float A tile, the
activations are rounded through half as the f16y readers do and held as float, float x float products into the
float 8x8 accumulate. Per product this is the ext reader's arithmetic; only the K-sum order (64-wide 8x8 slices)
differs. Prescreen: 48 B spill on every float instance (the doubled prefetch pair), the half form 0.

| depth 5, UD, Turbo4, 8K | decode t/s | acc | sha |
|---|--:|--:|---|
| base (ext SoA reader, r1_3 x 2) | 15.72 | 49.5% | `a409bb1b45df` |
| half tile (`GEN=6`) | 20.06 | 49.5% | `a409bb1b45df` |
| **float tile (`GEN=6 GEN_FP=1`)** | **14.67 / 14.95** | 50.5% | **`9128633c6cfa`** |

| vs the width-6 decode base (the ext reader) | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| half tile | 0.000488 | 0.000012 | 0.000535 | 0.012607 | 9.05 | 99.670 | 99.707 |
| **float tile** | **0.000462 +/- 0.000347** | 0.000012 | 0.000521 | 0.012797 | 8.49 | 99.666 | 99.709 |

Two findings, both against the hypothesis:

1. **Speed: the float tile is slower than the two-pass reader** (14.8 vs 15.7 t/s; the half tile 20.1). The float
   MMA runs at a fraction of the half rate on this GPU and the skinny tile at K = 5120-17408 turns out to be MMA-issue
   bound, not stream-bound; plus the spill. Dead as a form.
2. **Numerics: the float products change nothing** - the float tile sits at the same 4.6e-4 / 99.67% from the ext
   reader as the half tile. So the half rounding of the dequantized weights was NOT the mechanism of the 4.9e-4;
   the K-sum order (or something else shared by both tiles) is - OR the ext reader is the one that sits off, and
   both tiles are near the pick's numerics. The sha says the latter is worth checking: **the float tile at depth 5
   reproduces the depth-3 canonical text `9128633c6cfa`, which the width-6 base does not** (`a409bb1b45df`).

So the question moved: which width-6 kernel is closer to the pick's own decode numerics (the width-4 SoA kernels,
KLD-priced at 2.6e-5 from the prefill path)? Both width-6 arms are being scored at `-b 6 -ub 6` against the standing
V1 width-4 decode base of 2026-09-09 (`logits/kld-base-kld-pair-v1dec4-sep09.dat`, same positions; V2 vs V1 at
width 4 = 5e-6): if the ext reader reads ~4e-4 there and the tile ~3e-5, the "NUM-TG cost" of the tile is really
the removal of the ext reader's own deviation, and the pairwise table above had the sign backwards.

### Which side is off: both width-6 arms against the pick's own width-4 decode base (20:08)

Scored at `-b 6 -ub 6` against `logits/kld-base-kld-pair-v1dec4-sep09.dat` (the V1 file under the pick at width 4,
the decode base every later decode kernel is priced against; V2 vs V1 at width 4 = 5e-6):

| width-6 arm vs the width-4 pick base | mean KLD | median | 99.9% | max | same-top | overlap |
|---|---:|---:|---:|---:|---:|---:|
| the ext SoA reader (the pick's width-6 route) | **0.000025 +/- 0.000009** | 0.000001 | 0.001352 | 0.192 | 99.914 | 99.859 |
| the skinny tile (`GEN=6`) | 0.000505 +/- 0.000398 | 0.000013 | 0.012099 | 9.76 | 99.699 | 99.704 |
| scale: the pick's decode path vs its prefill path | 0.000026 | 0.000001 | 0.001866 | 0.177 | 99.914 | 99.861 |

Settled: **the ext reader at width 6 IS the pick's numerics** (2.5e-5 from the width-4 kernels = exactly the
decode-vs-prefill class, same-top 99.914 both), and **the tile is the outlier** at 5e-4 from both. The float-product
form deviating identically says the mechanism is neither the half rounding of the weights nor the MMA operand
type; the two candidates left are (a) something shared by both tiles' path - the K-slice summation, or the
remaining formats' reader (`dequantize_ud_soa_mm`, q6_K incl. the lm_head, q3_K, iq4_nl, iq3_s: a deviation in the
head's matmul lands on the logits unattenuated) - and (b) a plain bug at some shape. `test-backend-ops` MUL_MAT at
n = 6/7/8 passes vs the CPU (11/11) but only at m 16, k 256/1024 on the AoS types; the stored SoA types have no
test cases. Queued: the tile restricted to q4_K / q5_K / iq4_xs (`GGML_MM_SKINNY_GEN_TYPES=kq`) against the same
base - if that reads ~2.5e-5, the remaining formats' path carries the 5e-4; then the Fisher correlation of the two
tiles' deviations (`perf/kld-fisher.py`, D = the width-4 base; T, F = the tiles' logits written as base files) -
corr ~1 = one deterministic mechanism, ~0 = independent rounding.

### The geometry (`perf/kld-fisher.py`, 20:28-20:31; D = the width-4 pick base, E = the ext reader at width 6, T = the tile)

The script reproduces the tool's numbers from the files (E 2.5e-5 / 99.914%, T 4.67e-4 / 99.699% - the mean differs
from the tool's 5.05e-4 by the window renormalization) and its check holds: sum 1/2 Var_pD(d) / sum KL = 0.88-1.00,
i.e. KL is the Fisher norm of the deviation at these scales.

| Fisher correlation corr_pD(d_E, d_T) | median | mean | var-weighted | pooled | 10/25/75/90% |
|---|--:|--:|--:|--:|---|
| all 24,552 positions | 0.27 | 0.22 | 0.33 | **0.15** | -0.44 / -0.07 / 0.56 / 0.79 |
| ties (margin < 0.3 nat, n 3541) | 0.24 | | | | flips E 21, T 70 |
| margin 0.3-2 (n 10854) | 0.25 | | | | flips E 0, T 4 |
| confident (margin > 2, n 10157) | 0.30 | | | | flips E 0, T 0 |

**The two deviations are nearly independent** (pooled 0.15): the tile's deviation is not the reader's deviation
scaled up, it is its own noise. In amplitude the tile's per-position deviation is **3x the reader's** (median
Var_T / Var_E = 9.4, i.e. 9x in KL; medians 1.3e-5 vs 1e-6) at every margin, and ties flip 3x as often (70 vs 21
of 3541).

**The mean is one position.** Tile mean 4.67e-4 -> 1.06e-4 without position 6326 (chunk 6, offset 188: a 0.89-nat
margin the tile flips to a near-certain other token, 8.87 nats) -> 5.4e-5 without chunk 6 at all (the reader 9e-6
without it, 6x). Chunk 6 is a sensitive region for BOTH arms (E's mean there is 40x its other chunks, its own max
0.19 sits there too), and the tile's large positions in it are a cascade behind offset 188 (207, 211, 227, 240, 248,
356 ...: the teacher-forced cache carries the deviated K/V to every later position of the chunk). The Sep 9
native-kernels arm (median 1.3e-5, max 8.4, same-top 99.68 - the same profile as the tile to two digits) jumps at
the same chunk in its per-chunk record: chunk 6 flips under any kernel change of the ~1e-5-median class, and the
2.5e-5-class reader shows it as a 0.02 bump, not a flip.

So the fidelity statement for the owner is two numbers, not one: **the tile's own noise is ~1e-5 per position
(3x the reader's amplitude, the class of the fork's native kernels), and the 5e-4 mean is that noise meeting one
chaotic position.** Which side of that position is "right" is not a kernel question - against bf16 both arms read
0.0128 +/- 0.0015 there - but it is answerable per position with the bf16 file in a paired design (which arm the
trained model agrees with at 6326 and its cascade).

### Attribution and the tile-vs-tile correlation (20:32-21:22)

| arm vs the width-4 pick base | mean KLD | median | 99.9% | max | same-top |
|---|---:|---:|---:|---:|---:|
| the tile on every format (`GEN=6`) | 0.000505 | 0.000013 | 0.012099 | 9.76 | 99.699 |
| the tile on q4_K / q5_K / iq4_xs only (`GEN_TYPES=kq`; q6_K head, q3_K, iq4_nl, iq3_s, q8_0 keep their routes) | 0.000835 +/- 0.000722 | 0.000013 | 0.011983 | 17.7 | 99.690 |

The deviation lives in the three bulk formats' path, not in the remaining formats' reader or the lm_head (the
restricted arm has the same profile, and its chaotic position flips harder). And the two tiles' deviations from
the pick base are **the same vector**: Fisher corr(half tile, float tile) pooled **0.989**, median 0.987 at every
margin (ties 0.986, confident 0.989), the same 70/71 tie flips, KL medians 1.3e-5 both, both flipping position
6326 (to different tokens, 15 vs 73090, at 8.9 / 8.2 nats: a position where the model has no stable answer).

So the mechanism is **deterministic and shared by the half and float forms**: not the MMA operand precision, not
the half rounding of weights, not rounding noise. What the two forms share and the pick's kernels do not: the
tile's data path - the stored-row reader called per (block, 16-element tile) from the tile, the activations
rounded through half inside the tile, the 64-wide K slices summed by 8x8 MMA blocks. The Sep 4 refutation ran the
same tile through the AoS dequantizers on the plain file and was sha-identical to the pick at width 4 over
300/600 tokens, which points at the SoA-row path as the suspect; the prefill `mul_mm` bodies use the same
`dequantize_soa_mm` and price as exact, so it is the tile's use of it (indexing, `ne00`, the row pointer) or the
B side, not the reader itself. Unresolved tonight; an op-level dump of one real matmul under both routes (the
eval-callback dump, as `LLAMA_FA_DUMP` does for FA) is the tool that would settle it in one run.

### The paired run against the trained model (bf16 reference on `/Volumes/offload`, 21:47-22:01)

Reader and tile scored per position against the bf16 as-trained file (the reader's width-6 logits regenerated
first; the same 24,552 positions; `LLAMA_KLD_FLOOR`-class 32-nat window on the reference side):

| arm vs bf16 | mean KLD | median | 99.9% | max | same-top | ties (n 3563) | margin 0.3-2 (n 10800) | confident (n 10189) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the ext reader (the pick) | 0.010904 | 0.002636 | 0.793 | 13.27 | 96.583 | 79.905 | 98.972 | 99.882 |
| the tile | 0.011152 | 0.002647 | 0.870 | 12.09 | 96.546 | 79.736 | 98.917 | 99.912 |

Paired per position, KL(bf16||tile) - KL(bf16||reader): **mean +0.000247 +/- 0.000452 (sem), median 4e-7; without
chunk 6: -0.000026 +/- 0.000032**; the tile is the further arm at 52.7% of positions (a real but tiny tilt: the
excess over 50% is 8 sigma of a sign test, worth ~1e-5 per position, 1/1000 of the weights' distance). Fisher
corr(reader, tile) under bf16 = 0.988 pooled, 0.998 median: against the trained model the two arms are the same
kernel to three digits - both carry the Q4_K_M quantization's 0.011, and that is the whole picture; the tile's own
1.3e-5 rides on top of it at the 1/1000 level.

**At the chaotic positions the flips go both ways** - position 6326: bf16 says 6278 (margin 1.53), the reader
agrees (0.54), the tile flips to 15 (11.3 nats): the tile lost that one. Position 6494: bf16 says 67, the tile
agrees (0.70), the reader says 16 (2.12): the reader lost. 7066: both wrong. And the largest positions of chunk 6
(6863: 13.3 nats for BOTH arms, 6824: 11.5 both, 15418: 11.7 both) are the quantized weights' own catastrophes,
identical in both arms - chunk 6 is where the Q4 model is already wrong, and the kernel-level noise decides
which wrong answer.

**Status (2026-09-16 22:00): PRICED THREE WAYS.** (1) Pairwise vs the pick's width-6 route: 4.9e-4 mean, one
position; (2) vs the pick's width-4 decode base: the reader 2.5e-5 = the pick, the tile 5e-4 = 3x the reader's
amplitude (median 1.3e-5), deterministic, shared by the float form, in the bulk formats' tile path, mechanism
unidentified; (3) paired vs the trained model: +2.5e-4 +/- 4.5e-4 (not significant), same-top -0.04 pt, 52.7% of
positions further, one big flip lost and one won. Speed: round -24% at widths 6-8 (8K), -15% at 96K (width 6).
**Adoption = owner** (NUM-TG on the UD line; the trained-model view says the cost is below the resolution of
24K positions, the kernel view says it is a real 3x-amplitude deviation with an unknown cause). If wanted on the
pick's numerics instead: find the mechanism (an op-level dump of one real matmul under both routes) - the fix may
be one line.
