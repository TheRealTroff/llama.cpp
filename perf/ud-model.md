# UD-Q4_K_M as the target: optimizing the exact unsloth file (2026-09-04, OPEN; step 6 opened 2026-09-05)

Owner's decision 2026-09-04: run `Qwen3.8-27B-UD-Q4_K_M.gguf` **as-is** (no requant, no
hybrid), and see whether the Q4_0 stack's lessons carry. Agreed order: (1) free knobs,
(2) round decomposition, (4) acc-half mul_mm for UD's formats, (5) skinny MMA generalized
over the dequant block type. Steps (3) imatrix/hybrid requant and (6) per-format SoA scalar
kernels are explicitly NOT in scope unless asked.

Why the gap exists: every pick route at widths 3-8 gates on `GGML_TYPE_Q4_0` /
`GGML_TYPE_Q4_0_SOA` (SoA w3/w4/w5, repack, skinny, WL_XL), the acc-half prefill kernel gates
on Q4_0, and upstream Metal has no `mul_mv_ext` for K-quants. UD has **zero Q4_0 tensors**:
by streamed bytes ~29% IQ4_XS, ~29% Q5_K, ~26% Q4_K, ~10% Q6_K (the 248320-row head is Q6_K),
~5% Q3_K/IQ4_NL/IQ3_S/Q8_0. `weight-quant-kld.md` has the quality side (Same-top 96.56%
vs uniform-Q4_0's 90.75%). So UD's verify/draft matmuls run the upstream per-column
`mul_mv` at every width, and its batch-1 is already at its byte floor (-7% for +7.3% bytes).

**Sha discipline: UD is its own lineage.** batch-1 anchor `73ea53bbe98f` at 300,
`5e76afaba36c` at 600 (spec arms match both). Never compare UD t/s or shas with the Q4_0
pick's; compare UD arms with UD arms.

## Baseline on prod 7e4076e4a (the Q4_0 pick env, run-prod-pick.sh M=UD)

| arm (n_predict 300) | t/s | acc |
|---|--:|--:|
| DFlash depth 4 (the Q4_0 pick) | 15.60 | 49.9% |
| batch-1 no-spec | 13.07 | - |

Q4_0 pick the same day: 27.30 / b1 14.10. Speculation buys UD 1.19x over its floor
against Q4_0's 1.94x.

## Step 1: free knobs (perf/run-ud-knobs.sh, TAG ud-knobs-sep04, n_predict 600)

| depth | mm_min 8 | mm_min 4 | mm_min 2 | acc | mean accepted run |
|--:|--:|--:|--:|--:|--:|
| 4 | 16.02 | 16.12 | 16.10 | 53.5% | 3.14 |
| 3 | **17.82** | 17.81 | 17.68 | 60.9% | 2.83 |
| 2 | **17.86** | 17.85 | 17.62 | 72.7% | - |
| b1 | 12.67 | | | | |

- **Depth 2 and 3 tie at 17.8, +11.3% over the depth-4 pick.** The depth-4 optimum was
  priced on Q4_0's per-width cost curve (SoA w5r4h); on UD's per-column mul_mv the curve is
  steeper and the optimum moves down. Depth 3 is the working point for steps 2-5 (verify
  width 4, the width step 5 lifts first, and the Turbo4 line's depth).
- **`GGML_MM_MIN` is inert** (8 vs 4 within 0.6%, 2 slightly worse). Routing the verify
  width to the generic simdgroup `mul_mm` instead of `mul_mv` does not help: the generic
  tile is built for wide N and pays its A-tile dequant per 32 columns whether 4 or 32 are
  live. That is exactly the skinny kernel's reason to exist (step 5).
- Prefill is unchanged across arms (73.0-73.3 s; the Q4_0 pick is 62.5 s with acc-half,
  which UD does not get - step 4).
- Bare passes (llama-bench, depth-3 decomposition run): pp1 12.50 t/s (80.0 ms), pp4 30.21
  (132.4 ms). **The width-4 pass costs 1.65x the batch-1 pass** on UD; on the Q4_0 pick the
  w4 r4kp kernel holds it near 1.2x. That ratio is the whole step-5 target.

### Acceptance: UD is LOWER than Q4_0 with the same drafter (owner asked 2026-09-04)

Same pureQ4_0 DFlash drafter, same prompt: depth 4 @600 Q4_0 58.2% vs UD 53.5% (mean run
3.33 vs 3.14); @300 51.4% vs 49.9%; depth 3 @600 Q4_0 (Turbo4 KV line) 70.7% vs UD 60.9%.
UD is far closer to bf16 (mean KLD 0.014 vs 0.054) and the drafter was trained on bf16, so
the naive expectation was the reverse; the owner's reading was that a tiny drafter has a
harder time predicting a more capable target. **Corpus run (owner: "go ahead"), same session,
same pick env, depth 4, window arm, TAGs `corpacc-ud-sep04` / `corpacc-q40-sep04`:**

| prompt | UD acc | Q4_0 acc | UD committed/rd | Q4_0 committed/rd | UD acc per pos | Q4_0 acc per pos |
|---|--:|--:|--:|--:|---|---|
| `benchprompt` (8288 tok) | 49.9% | 51.4% | 2.97 | 3.03 | (.700 .550 .420 .320) | (.776 .612 .418 .245) |
| `01-code-explain` | **49.5%** | 37.4% | **2.94** | 2.48 | (.780 .560 .410 .230) | (.708 .433 .208 .142) |
| `02-prose-creative` | **54.7%** | 42.1% | **3.16** | 2.65 | (.840 .617 .415 .309) | (.676 .486 .315 .207) |
| `03-chat-support` | 32.8% | **47.1%** | 2.31 | **2.86** | (.756 .356 .156 .044) | (.816 .552 .345 .172) |
| `04-math-derivation` | 84.6% | **91.1%** | 4.29 | 4.55 | (.971 .897 .809 .706) | (.969 .923 .892 .815) |
| `05-json-boilerplate` | 95.2% | 97.1% | 4.69 | 4.76 | (.968 .952 .952 .935) | (1.00 .984 .967 .934) |
| **mean committed/rd** | | | **3.39** | **3.39** | | |

**Verdict: trajectory noise, not a drafter-target mismatch.** The mean committed/round is
identical to two decimals (3.39 vs 3.39); per prompt the sign flips both ways by 12-14 pt
(UD +12 on code-explain and +12.6 on prose, -14 on chat-support, where UD's generation hit
EOS after 45 rounds, so that cell is short). The same-prompt gap from the depth sweep (UD
5-10 pt under on benchprompt) is one draw from this spread. Neither "UD is harder to
predict" nor "UD accepts more because it is closer to bf16" survives the corpus; this
matches the Turbo4 finding that corpus acceptance is trajectory noise. Consequence for
the UD stub: acceptance is NOT a lever or a cost on this model; the entire gap to the Q4_0
pick is kernel time per width.

## Step 2: round decomposition at depth 3 (perf/run-ud-decomp.sh, TAG ud-decomp-sep04-d3)

Anchor 17.37 t/s @300 (acc 59.0%, sha `73ea53bbe98f` - the depth-3 arm shares the
depth-4/b1 sha at 300), decode-prof 17.43, profiled 15.20 (inflation 1.14, low because big
ops dominate). 108 rounds, 2.78 tokens/round, **~160 ms/round real**; serialized GPU
163.8 ms/round, so the profiler's sum is close to the wall here.

| bucket (serialized ms/round) | ms | share |
|---|--:|--:|
| **m1 MUL_MAT (target, width 4)** | **132.2** | **81%** |
| m1 flash_attn | 7.5 | 4.6% |
| m1 elementwise/other | 6.4 | 3.9% |
| m1 GDN | 3.2 | 2.0% |
| m2 drafter (q4_0 proj 5.9 + other mm 5.0 + misc 3.3 + FA 0.4) | 14.6 | 8.9% |

Per-call cost of the width-4 target projections against their byte floor (bytes at the
format's bpw, floor at 273 GB/s):

| tensor (format, shape) | us/call | floor us | x floor | ms/round |
|---|--:|--:|--:|--:|
| ffn_up/gate iq4_xs [5120,17408] | 411 | 173 | **2.37** | 25.3 |
| ffn_up/gate q5_K [5120,17408] | 455 | 224 | 2.03 | 12.9 |
| ffn_down q5_K [17408,5120] | 493 | 224 | 2.20 | 11.0 |
| ffn_up/gate q4_K [5120,17408] | 383 | 184 | 2.09 | 10.8 |
| ffn_down iq4_xs [17408,5120] | 404 | 173 | 2.33 | 9.8 |
| attn_qkv-ish q5_K [6144,5120] | 189 | 79 | 2.38 | 8.4 |
| q4_K [5120,10240] | 234 | 108 | 2.17 | 6.8 |
| lm_head q6_K [5120,248320] | 5026 | 3820 | 1.32 | 5.0 |
| ffn_down q4_K [17408,5120] | 407 | 184 | 2.21 | 4.1 |
| q3_K [5120,17408] | 402 | 140 | 2.87 | 2.4 |
| iq3_s [17408,5120] | 828 | 140 | **5.9** | 2.5 |
| drafter q4_0 [5120,17408] (r4kp, for reference) | 228 | 184 | **1.24** | 2.3 |

- **Every UD projection runs at 2.0-2.4x its byte floor at width 4** (iq3_s at 5.9x, but
  only 3 calls/round); the same shape in q4_0 on the r4kp kernel runs at 1.24x. If the
  generic skinny tile (step 5) brings UD's projections to ~1.3x, the m1 MUL_MAT bucket
  drops ~55 ms and the round goes ~160 -> ~105 ms, i.e. ~17.4 -> ~26 t/s at depth 3.
  That is the ceiling for step 5, not a prediction - the skinny tile on q4_0 measured
  ~1.5x floor at width 4 before the SoA kernels replaced it (m4-width4-*.md).
- The q6_K lm_head is already at 1.32x (one call, 5.0 ms); not a first-order lever.
- The drafter is 8.9% of the round and already on its fast path; the small q8_0
  [5120,48] calls (97/round, 4.4 ms) are the ssm vectors - dispatch-bound, hidden under
  concurrent encode in the real run (small-ne01 lesson).

## Step 4: acc-half mul_mm for UD's formats (branch ud-acch-types) - prefill -6.9%, KLD-priced, NOT recommended

Built: `kernel_mul_mm_acch_{q8_0,q3_K,q4_K,q5_K,q6_K,iq3_s,iq4_nl,iq4_xs}_f32` instances of
the same template as the q4_0 probe (half accumulate); host gate widened, n64 tile stays
q4_0. Same `GGML_MM_ACC_HALF=1` flag, so the pick env picks it up unchanged.

| depth 3, n_predict 300 | prompt (8288 tok) | decode t/s | acc | sha |
|---|--:|--:|--:|---|
| prod 7e4076e4a (4 arms + anchor) | 73.0-73.3 s | 17.37 | 59.0% | 73ea53bbe98f |
| acch build | **68.18 s (-6.9%)** | 18.39 | 64.6% | 10e1c40ab4bc |

Prefill numerics change, so this is a NEW UD sha lineage (`10e1c40ab4bc` @300); the decode
t/s and acceptance difference is trajectory (different text), not a decode effect - the
acch kernel only runs at ne11 > 8. Same size of win as on Q4_0 (+8.3% there, prefill-decomp.md).
Routing proven from the timing invocation's stderr: all eight `kernel_mul_mm_acch_*` pipelines
engage; bare pp512 119.26 -> 128.57 t/s (+7.8%).

**KLD pricing (TAG kld-ud-acch-sep04, q8_0 reference, 16 chunks x 2048 wikitext, f16 KV both
sides; the reference logits were deleted after the run - ~18 GB, disk at 94%):**

| UD vs q8_0 | control (f32 acc) | acch | Q4_0 line for scale (kldacch-aug28) |
|---|--:|--:|---|
| Mean KLD | 0.01654 | **0.02265 (+37%)** | 0.054 -> 0.060 (+11.8%) |
| Median KLD | 0.00262 | **0.00784 (3.0x)** | |
| 99% KLD | 0.0913 | 0.1269 | |
| 99.9% KLD | 1.328 | 1.494 | |
| Maximum KLD | 21.67 | 20.85 | no pathologies either side |
| RMS dp | 3.218% | **4.123%** | 6.29 -> ~6.8 |
| Same top p | 96.43% | **93.92% (-2.51 pt)** | -0.86 pt |

**Recommendation: do not adopt acch for UD.** The acch noise is the same absolute size on
both models - RMS dp adds in quadrature as ~2.6% (sqrt(3.22^2+2.6^2) = 4.14, sqrt(6.29^2+2.6^2)
= 6.8, both match the measurement) - but UD's own quant noise is half of Q4_0's, so the same
kernel costs UD three times the top-token agreement it cost Q4_0. It hands back ~44% of the
quality UD was chosen for (96.43 -> 93.92 against uniform-Q4_0's 90.75) for 6.9% of prefill
wall. Owner's call, as with the Q4_0 adoption; the branch stays built and measured. If a
cheaper form is wanted, the split-accumulate (f32 every N K-slices) that was never tried on
the Q4_0 line is the next probe, not a per-format gate - the noise is not format-specific.

## Step 5: skinny MMA generalized over the dequant block type - REFUTED at width 4 (branch ud-skinny-generic)

Built: `kernel_mul_mm_skinny_t<block_q, nl, dequantize_func>`, the q4_0 skinny tile (32
rows x 8 cols, K-slice 64, 2 simdgroups, slice t+1 dequantized in registers under slice
t's MACs) with the per-thread 32-element dequant done by the generic block dequantizers
(the ones `kernel_mul_mm` uses). Instances for q8_0/q3_K/q4_K/q5_K/q6_K/iq3_s/iq4_nl/iq4_xs;
host route `GGML_MM_SKINNY_GEN=N` sends ne11 in [max(2,N), 8] there. All eight pipelines
compile and engage (pipeline names read from the timing invocation's stderr); output is
**byte-identical** (sha `5e76afaba36c` @600, `73ea53bbe98f` @300).

**It is slower for every format.** Depth 3, GEN=2 vs off, same binary:

| | off | GEN=2 |
|---|--:|--:|
| e2e @600 | 17.82 | **15.40 (-13.6%)** |
| e2e @300 anchor | 17.37 | 15.09 |
| bare pp4 (llama-bench) | 30.05 | 26.19 |
| m1 MUL_MAT serialized ms/rd | 132.2 | 164.9 |

Per call (us, width 4, profiled): iq4_xs [5120,17408] 411 -> 451; q5_K [5120,17408] 455 ->
582; q5_K [17408,5120] 493 -> 628; q4_K [5120,17408] 383 -> 467; q6_K head 5026 -> 6968;
q8_0 [5120,48] 45 -> 92 (small-ne01, dispatch-starved - the route should exclude it anyway).
The drafter's q4_0 r4kp calls are unchanged (230 us), as expected.

~~Why (first write-up): the upstream K-quant `kernel_mul_mv_*_f32` runs ONE column per
threadgroup and re-streams the weights 4x at width 4.~~ **WRONG, corrected the same evening
(owner: "did you reference the work we did on reducing the number of redundant dequants?").**
The routing was then read from a width-4 timing invocation's own stderr (prod binary, pick
env, `GGML_METAL_LOG_LEVEL=2`): UD's projections run **`kernel_mul_mv_ext_{q4_K,q5_K,q6_K,
q3_K,iq4_xs}_f16_r1_4`** (f16y, nr0=2) and the `_f32_r1_4` variants for the smaller tensors;
only iq3_s (3 calls/round) and one q6_K call take the plain per-column mv. That is the x4
ext family from `results.md` - dequant-once-reuse-per-column, the nr0 rows-per-thread port
that removed the redundant src1 loads, f16y - and its record already says what today
re-measured: **K-quants sit ~1.6x behind Q4_0 per pass at widths 4-8 from heavier dequant
chains plus r1_4 register pressure, nr0=2 is the x4 ceiling (nr0=4 hits the register
cliff), and dequant granularity is not the mechanism.** So the generic skinny removed NO
redundant dequant relative to the incumbent - both dequantize each weight once per pass -
and only added the `dequant -> threadgroup -> simdgroup_load` round trip and the generic
16-element `dequantize_*` form (scale/min extraction and float conversion per call) on top.
Offline prescreen (agx-spill-probe, all cv set): zero spill on every skinny instance, text
4812 B (q4_0 skinny) vs 6262 q4_K / 8322 q5_K / 6644 q6_K / 5904 iq4_xs / 4590 q8_0, so
registers are not the cause either. `ffn-utilization.md` priced the same round trip on Q4_0
("buys nothing"). The 2.0-2.4x-floor figures in step 2 stand; they are measured against the
ext family, and its own gap to the SoA w4 kernel on Q4_0 (1.24x) is the layout lever
`results.md` named next and the Q4_0 line then built as SoA.

**What would actually move UD's width-4 plane** (NOT built - it is step 6-class work,
outside the agreed scope): the same move the Q4_0 line made after the ext family hit its
register ceiling - a weight-layout change (SoA-style planar scales/mins/nibbles) for the
three FFN formats q4_K/q5_K/iq4_xs, so the inner loop keeps the lean integer form with the
per-block bookkeeping hoisted. Ceiling from the decomposition: ~55 ms of the 160 ms round if
the projections reach ~1.3x floor. The generic skinny stays in the tree behind its env flag
as the negative control; do not put GGML_MM_SKINNY_GEN in any pick.

## Step 6: iq4_xs SoA layout + width-4 scalar kernel (2026-09-05, branch `ud-soa-iq4xs`, worktree `llama.cpp-ud-soa`)

Opened by the owner 2026-09-05 ("I absolutely want to see where we can get with this
substantially higher fidelity quant"). The move the Q4_0 line made after its ext family hit
the register ceiling, redone for iq4_xs first: it is the biggest line of the depth-3 round
(ffn up/gate 25.3 ms + down 9.8 + q/gate ~4 = ~39 ms of 132) and the format closest to q4_0
(4-bit indices into a 16-entry table, one 6-bit scale per 32, one f16 super-scale per 256).

**Layout** (runtime side buffer for now, per tensor, cached like the q4_0 repack; other widths
read the original weights, so no layout conflict exists): per row with nsb = ne00/256,
`[half d x nsb][int8 (ls-32) x 8*nsb][pad16][uint pack8 x 32*nsb]` (exact, 138 B/256 vs 136)
or `[half d*(ls-32) x 8*nsb][uint pack8 x 32*nsb]` (scale pre-rounded to half, 144 B/256,
+5.9% bytes). Pack p holds k = 8p..8p+7 in k order, so the q4_0 r4kp_v3 body transplants with
the nibble replaced by a table lookup. Repack kernels `kernel_repack_iq4_xs_soa{,h}`, mul_mv
`kernel_mul_mv_iq4_xs_soa_w4_v1..v6`, route `GGML_MV_SOA_IQ4XS=<v>` at ne11 == 4 on the
whitelisted row counts (all five UD row counts are on it), f16y activations as before.

| v | scale | product | lookup | text B | spill | ffn_gate_up us | ffn_down | attn_q | attn_gate |
|--:|---|---|---|--:|--:|--:|--:|--:|--:|
| base | ext r1_4 (incumbent) | | | | | 411-432 | 412 | 291 | 153 |
| 1 | exact d*int8 | f32 | constant table | 3582 | 16 | 300 | 323-340 | 217 | 117 |
| 2 | exact -> half | half | constant table | 3504 | 0 | 243 | 265 | 176 | 94 |
| 3 | exact -> half | half | simd_shuffle | 6646 | 0 | 439 | 454 | 316 | 166 |
| 4 | exact d*int8 | f32 | simd_shuffle | 6764 | 32 | 493 | 512 | 355 | 187 |
| **5** | **half planar** | **half** | **constant table** | 3240 | 0 | **227** | **249** | **165** | **86** |
| 6 | half planar | half | simd_shuffle | 6280 | 0 | 434 | 448 | 311 | 164 |

(`perf/run-ud-iq4xs-soa-ab.sh`, TAG `ud-iq4xs-soa-0905-1433`, 2 interleaved reps, spread
<1%, every arm's pipeline name read from its own stderr; 29/29 iq4_xs MUL_MAT cases vs CPU
for every variant, including the six UD shapes at widths 3/4/5 added to test-backend-ops.)

- **v5 is -40 to -45% per call and sits at ~1.31x the 273 GB/s byte floor** (ffn_gate_up
  47.4 MB -> 173 us floor; 1.23x on its own +5.9% bytes) - exactly the ~1.3x the step-2
  decomposition assumed for its ~55 ms ceiling. The q4_0 v3 kernel reads 1.24x on the same
  shape, so the table lookup costs ~7 points, not the 2x the K-quant record feared.
- **The lookup form decides everything**: `simd_shuffle` from a lane-held table doubles the
  text and the time (v3/v6 lose to the incumbent); the 16-entry `constant` table folds into
  the dequant chain. The f32-product forms spill (16/32 B) and land between.
- **v2 (exact scale) is 7% behind v5**: the pre-rounded half scale (<= 2^-11 relative per
  32-block) buys the per-pack `float(d)*float(ls)` plus a convert. This is a numerics
  decision the owner makes, priced by KLD like acch; v2 is the exact fallback.
- **e2e at depth 3, n_predict 600, `run-ud-knobs.sh` (B=llama.cpp-ud-soa), base / v5 / v2 / base:
  17.788 / 19.999 / 19.842 / 17.785 t/s = v5 +12.4%, v2 +11.6%, acceptance 60.9% on every arm,
  sha `5e76afaba36c` on every arm** (the UD 600-token lineage sha: the half-rounded scales moved
  no byte of this trajectory). The ~12% is the iq4_xs share of the round only (39 of 132 ms of
  width-4 matmul at -43%): the q4_K/q5_K lines below carry the rest.

### q4_K / q5_K, same day, same body (`GGML_MV_SOA_KQ=1|2`)

The scale-and-min form w = s*q - m with the min folded out of the element loop
(`acc -= m * sum8(v)`, the 8-element activation sums shared across the 4 rows); q5_K keeps a
byte-per-pack high-bit plane after the packs. Layouts: v1 exact `[half2 d,dmin x nsb][u8 sc x
8nsb][u8 mn x 8nsb][pad16][packs][hbits]` (148/180 B per 256 vs 144/176), v2 half planar
`[half d*sc x 8nsb][half dmin*mn x 8nsb][packs][hbits]` (160/192 B, +11%/+9% bytes). Both
zero spill (text 3392-4760 B); q4_K 55/55 and q5_K 23/23 vs CPU at the UD shapes, both variants,
route names read from the test run. `run-ud-iq4xs-soa-ab.sh` with `TYPE=q4_K|q5_K
ENVNAME=GGML_MV_SOA_KQ`, TAGs `ud-q4_K-soa-sep05` / `ud-q5_K-soa-sep05`, 2 interleaved reps:

| shape (us/call) | q4_K base | q4_K v1 | **q4_K v2** | q5_K base | q5_K v1 | **q5_K v2** |
|---|--:|--:|--:|--:|--:|--:|
| ffn_gate_up 17408x5120 | 376-390 | 272-275 | **244 (-36%)** | 451 | 318 | **292 (-35%)** |
| ffn_down 5120x17408 | 408 | 286 | **264 (-35%)** | 490 | 333 | **314 (-36%)** |
| attn_qkv 10240x5120 | 231 | 166 | **150 (-35%)** | 278-286 | 194 | **180 (-36%)** |
| attn_gate 6144x5120 | 146 | 105 | **94 (-36%)** | 178 | 121 | **110 (-38%)** |

Floors at 273 GB/s: q4_K 17408x5120 = 184 us (v2 1.33x), q5_K = 224 us (v2 1.30x). The
exact-scale v1 is 8-11% behind v2 (two scale products and two converts per pack per row);
v2 is the half-planar form, the numerics decision is the owner's, priced by sha/KLD at e2e.
**Combined e2e, depth 3, n_predict 600, `run-ud-knobs.sh`, base / IQ4XS=5+KQ=2 / same / base:
17.768 / 23.874 / 23.939 / 17.792 t/s = +34.5%, sha `5e76afaba36c` on every arm (byte-identical
text), acceptance 60.5 vs 60.9** (the drafts differ slightly because the drafter reads the
target's activations, the committed text does not). The step-2 decomposition's "~55 ms off a
160 ms round -> ~26 t/s" ceiling assumed 1.3x floor on every projection; the three formats
carry ~89% of the width-4 matmul bytes and land at 1.30-1.33x, so this is most of that ceiling
at the same operating point. Residency for now: three runtime side buffers (~12 GiB with the
half-planar layouts' +6-11% bytes) - the offline GGUF is the productization step, as on the
Q4_0 line.

### Widths 3 and 5, same body (2026-09-05 afternoon)

The kernels are templated on the column count with explicit named activation streams (an
array-of-half8 form changed the codegen: width-4 text shrank, width-5 ballooned to 18-20 KB and
q5_K spilled; the explicit form reproduces the measured width-4 text byte-for-byte). Widths 3
and 4 keep the two-simdgroup K split; width 5 is one simdgroup over the full K, the q4_0
w5_r4h geometry. All zero spill; 29/55/23 tests vs CPU at the UD shapes for every
width/variant, names read from the runs. `run-ud-iq4xs-soa-ab.sh N=3|5`, TAGs
`ud-<type>-soa-w<N>-sep05`, us/call, 2 interleaved reps:

| shape | iq4_xs w3 base -> v5 | q4_K w3 base -> v2 | q5_K w3 base -> v2 | iq4_xs w5 base -> v5 | q4_K w5 base -> v2 | q5_K w5 base -> v2 |
|---|---|---|---|---|---|---|
| ffn_gate_up | 324 -> 217 (-33%) | 318 -> 238 (-25%) | 390 -> 287 (-26%) | 520 -> 256 (-51%) | 414 -> 277 (-33%) | 490 -> 323 (-34%) |
| ffn_down | 348 -> 237 (-32%) | 366 -> 257 (-30%) | 432 -> 308 (-29%) | 508 -> 280 (-45%) | 449 -> 299 (-33%) | 527 -> 345 (-35%) |
| attn_qkv | 192 -> 133 (-31%) | 193-203 -> 146 (-26%) | 237 -> 174 (-27%) | 303 -> 159 (-48%) | 254 -> 171 (-33%) | 301 -> 198 (-34%) |
| attn_gate | 124 -> 76 (-39%) | 122 -> 85 (-30%) | 151 -> 108 (-29%) | 188-194 -> 102 (-47%) | 165 -> 110 (-33%) | 194 -> 126 (-35%) |

Width 5 is the big one because the incumbent there is the register-heavy `ext r1_5`; on the
SoA side width 5 costs only +13% over width 4 for +1 verify column (iq4_xs 256 vs 227), so
the depth optimum may move back up from 3. Depth sweep with all routes on (600 tokens, f16 KV,
TAG `ud-soa-all-depths-sep05`): **depth 4 21.87 / depth 3 23.98 / depth 2 21.99 t/s**, sha
`5e76afaba36c` on every arm - depth 3 stays the operating point (yesterday's base curve was
16.02 / 17.82 / 17.86). KV line held at f16 for this round (owner, 2026-09-05: "big enough
change for one round"); Turbo4 on UD is an untried follow-up, nothing in it depends on the
weight format.
Batch-1 anchor in the same sweep: 12.641 t/s (yesterday 12.67) - machine state healthy, and the
route does not touch width 1. **Engagement confirmed in the server run itself** (`EXTRA_ARGS="-lv 5"`,
TAG `ud-soa-engage-sep05`): `kernel_repack_{iq4_xs,q4_K,q5_K}_soah` and
`kernel_mul_mv_{iq4_xs,q4_K,q5_K}_soa_w4_v{5,2,2}` all load in the depth-3 run.

### Where this leaves the UD line (2026-09-05 evening)

| | t/s @600, depth 3 | vs Q4_0 pick (29.9) |
|---|--:|--:|
| UD, prod (yesterday) | 17.8 | 0.60x |
| UD, branch `ud-soa-iq4xs`, IQ4XS=5 + KQ=2 | **24.0** | 0.80x |

Adoption is the owner's call. What it costs: three runtime side buffers (~12 GiB; 48 GiB machine)
and the half-planar scale rounding, which moved no byte of any 600-token arm at any depth
(sha `5e76afaba36c` throughout) - the exact variants (`IQ4XS=2`, `KQ=1`) are 7-11% slower per
call and equally correct if the owner wants zero rounding on principle. Open, in order of value:

1. ~~**Offline GGUF storage** for the three layouts (the `Q4_0_SOA_V1` path: new ggml types,
   `llama-gguf-repack`, readers at widths 1-2 and the mm/prefill path) - removes the ~12 GiB
   and the first-call repack; the Q4_0 line measured that step at +2%.~~ **DONE, step 13
   (2026-09-06): -10.9 GiB resident, depth-3 parity, sha-identical; b1 -2.8% (q4_K width-1
   reader) is the open remainder.**
2. **Remaining width-4 formats** ~10 ms of the (now ~100 ms) round: q6_K attn/ffn tensors
   (0.6 GiB; the q6_K head is already 1.32x floor), q3_K, iq3_s (5.9x floor, 3 calls/round),
   iq4_nl (6 tensors; same table as iq4_xs with a per-32 scale - the cheapest to add).
3. **Turbo4 KV on UD** - untried, weight-format independent.
4. ~~A fresh round decomposition at the new point~~ **DONE, step 7 below**; then the acch prefill
   question again (unchanged: -6.9% wall for -2.5 pt same-top, not recommended).

## Step 7: round decomposition at the SoA point (2026-09-05 evening, `run-ud-decomp.sh` B=ud-soa, TAG `ud-decomp-sep05-soa`)

Same harness, same prompt, same depth 3 and pick env as step 2, plus `GGML_MV_SOA_IQ4XS=5
GGML_MV_SOA_KQ=2` (the engaged pipelines are on record from `ud-soa-engage-sep05`; the per-call
times below match the synthetic A/B within 4%, which is the in-graph proof). Anchor **23.45 t/s @300,
acc 59.0%, sha `73ea53bbe98f`** (step 2: 17.37, same acc, same sha, and the same 191/324 drafts -
the trajectory is byte-identical drafter-side too). 108 rounds, 2.78 tokens/round,
**118.1 ms/round real** (step 2: 159.4); decode-prof arm 23.39; profiled arm 19.94 (inflation
1.18, was 1.14 - the small ops are a larger share now). Bare passes: pp1 12.44 t/s (80.4 ms),
**pp4 42.23 (94.7 ms): the width-4 pass costs 1.18x the batch-1 pass**, against 1.65x in step 2 and
~1.2x on the Q4_0 pick - the ratio step 5 was chasing is closed.

**Real round (server spec-prof, cumulative over 86 rounds):** target verify sync 104.5 ms (step 2
141.9), drafter call 14.4 (14.2), target submit 2.8 (2.9), accept+post 0.75 (0.7). The first
verify round is 460 ms - it carries the one-time SoA repack of the three side buffers - and that
one round lifts the cumulative sync average by ~4 ms, so the steady-state round is ~100 ms of
target sync + ~18 ms of drafter/host, i.e. the 118 measured. Speculation now buys UD 1.88x over
its 12.44 batch-1 floor (step 1: 1.19x; the Q4_0 pick: 1.94x).

| bucket (serialized ms/round) | step 2 | **step 7** | share |
|---|--:|--:|--:|
| m1 MUL_MAT width 4, the three SoA formats (q5_K + iq4_xs + q4_K) | 110.5 | **71.1** | 58% |
| m1 MUL_MAT width 4, other formats (q6_K 3.0, q8_0 4.7, iq3_s 3.4, q3_K 2.9, iq4_nl 2.4) | 16.5 | 16.4 | 13% |
| m1 lm_head q6_K | 5.1 | 5.1 | 4% |
| m1 flash_attn | 7.5 | 7.5 | 6% |
| m1 elementwise/other | 6.4 | 6.0 | 5% |
| m1 GDN | 3.2 | 3.1 | 3% |
| m2 drafter (q4_0 proj 5.9 + head 5.0 + misc 2.9 + FA 0.4) | 14.6 | 14.1 | 11% |
| **total** | 163.8 | **123.3** | |

The whole drop is the three SoA formats (-39.4 ms serialized, -41.3 ms real); every other bucket
is unchanged to the tenth, so the routes touched nothing else. `metalprof-buckets.py` now floors
every quant type (bpw table) and prints a per-format summary; per format, calls/round and the
in-graph multiple of the 273 GB/s byte floor (the format's own bytes, not the side-buffer bytes):

| m1 width-4 format | ms/rd | floor | step 2 x | **step 7 x** | calls/rd | note |
|---|--:|--:|--:|--:|--:|---|
| q5_K | 27.00 | 18.20 | 2.19 | **1.48** | 134.6 | ffn 1.35-1.44x, attn_qkv 1.41, attn_gate [6144,5120] 1.66 |
| iq4_xs | 25.21 | 17.92 | 2.36 | **1.41** | 120.2 | ffn_up/gate 1.35x (234.6 us, A/B 227), ffn_down 1.46, [6144,5120] 1.77 |
| q4_K | 18.88 | 13.13 | 2.16 | **1.44** | 105.9 | ffn 1.36-1.46x, small shapes 1.52-1.81 |
| q6_K (incl. head) | 8.09 | 5.55 | 1.45 | 1.46 | 24.7 | ext r1_4 already ~1.5x; head 1.32 |
| q8_0 [5120,48] ssm vectors | 4.74 | 0.26 | 18.5 | 18.1 | 106.9 | dispatch-bound, hidden under concurrent encode in the real graph |
| iq3_s | 3.37 | 0.58 | 5.87 | 5.84 | 4.1 | plain per-column mv |
| q3_K | 2.92 | 1.01 | 2.88 | 2.90 | 7.2 | ext r1_4 |
| iq4_nl | 2.42 | 1.24 | 1.99 | 1.94 | 7.2 | ext r1_4 |

- **In-graph the big FFN shapes sit at 1.35-1.46x, the synthetic A/B said 1.30-1.35x** - the
  same shape-for-shape numbers within 4% (iq4_xs up/gate 234.6 vs 227 us, q4_K 249 vs 244, q5_K
  303 vs 292). The format aggregates are pulled up to 1.41-1.48x by the **short shapes**: the
  [6144,5120] and [5120,6144] attention projections run at 1.52-1.81x on all three formats (1280-1536
  threadgroups of 64 threads against 4352 for ffn_up/gate). The q4_0 r4kp_v3 shows the same
  geometry sensitivity in the drafter (1.24x at [5120,17408], 1.31 at [17408,5120], 3.8x at
  [5120,1024]); it is the kernel's fixed per-threadgroup cost against a short K-stream, not a
  format effect.
- **What is left in the width-4 matmul plane, priced from this table:** the three SoA formats at
  q4_0's 1.24x everywhere would be ~61 ms against 71.1 (-10 ms, mostly the short shapes); the
  remaining formats brought to ~1.4x (iq3_s 3.4 -> 0.8, q3_K 2.9 -> 1.4, iq4_nl 2.4 -> 1.7, q6_K
  non-head 3.0 -> 2.4) is ~-5 ms. Together ~15 ms of a 118 ms round (~13%), and the q8_0 vector
  calls are not real time. The rest of the round is now attention (7.5), elementwise (6.0), GDN
  (3.1) and the drafter (14.1 - 12% of the round, same absolute as on the Q4_0 pick, so the
  drafter-stack levers transfer unchanged).

### GPU trace at the SoA point: the three kernels against q4_0 r4kp_v3 and their ext incumbents

`perf/run-ud-soa-profile.sh`: seven captures at the ffn_gate_up shape (m=17408, k=5120, n=4,
`GGML_MV_REPACK=2`, each arm's pipeline name read from its own capture stderr), headless replay,
`gpuprofiler-stats.py --all`, `shaderprof-table.py --json`, and the new `perf/shaderprof-compare.py`
(per-dispatch normalization, hot loop = rows at >= 0.9 max executed, size fingerprint, stall sites).
Timings are from a separate uncaptured pass (`perf/run-ud-soa-w4-timing.sh`, 2 interleaved reps,
spread < 1%). Bundle: `kvquant-experiments/profiles/ud-soa-w4-sep05/` (gputraces, streamData,
stats, per-instruction JSON/text; the raw replay bundles were deleted for disk - ~3 min each to
regenerate from the trace). 87040 pack-iterations per dispatch (4352 threadgroups x 2 simdgroups
x 10); "/iter" is executed instructions per pack-iteration, i.e. per 4 rows x 8 k x 4 columns.

| kernel | us | x floor | regs | spill B | live | dev loads | exec/disp | /iter | issue/stall % | hot loop | us per M issued |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q4_0 soa_w4_r4kp_v3 (reference) | 221.3 | 1.21 | 66 | 0 | 402 | 12 | 25.62M | 294 | 88.1/11.9 | 283 | 7.61 |
| **iq4_xs soa_w4_v5** | 220.9 | 1.27 | 74 | 0 | 357 | **44** | **22.17M** | 255 | 86.5/**13.5** | 244 | 8.62 |
| **q4_K soa_w4_v2** | 236.9 | 1.29 | 70 | 0 | 406 | 16 | 26.44M | 304 | **95.7/4.3** | 293 | 8.58 |
| **q5_K soa_w4_v2** | 283.3 | 1.26 | 81 | 0 | 478 | 20 | 32.39M | 372 | 91.8/8.2 | 361 | 8.03 |
| ext_iq4_xs_f16_r1_4 (incumbent) | 403.0 | 2.32 | 96 | 16 | 620 | 46 | 30.88M | 355 | 87.2/12.8 | 348 | 11.38 |
| ext_q4_K_f16_r1_4 | 370.0 | 2.01 | 96 | 32 | 689 | 28 | 35.42M | 407 | 84.8/15.2 | 390 | 8.85 |
| ext_q5_K_f16_r1_4 | 442.9 | 1.97 | 96 | 64 | 831 | 36 | 46.75M | 537 | 85.1/14.9 | 514 | 8.06 |

Floors at 273 GB/s on the format's own bytes: iq4_xs 173.4 us, q4_0/q4_K 183.6, q5_K 224.5.
"us per M issued" = us x issue share / (exec per dispatch), the per-instruction issue cost.

1. **One economy point for all four SoA kernels, a different one for the three incumbents.**
   SoA: 66-81 registers, zero spill, 22-32M executed per dispatch, 86-96% issue, 1.21-1.29x
   floor. Ext r1_4: 96 registers on every format (the register cliff `results.md` recorded for
   nr0), 16/32/64 B spilled, 25-31% more dynamic instructions per dispatch, and **zero FP16
   instructions** - the ext kernels take f16 activations but convert and multiply in f32, while
   the SoA kernels carry the 32 products per pack in half (FP16 count 32). The layout change bought
   its -36..-45% as fewer instructions (iq4_xs -28%, q4_K -25%, q5_K -31%), a cheaper instruction
   for iq4_xs (11.4 -> 8.6 us/M), and 1-11 points of stall; nothing in it is a scheduling effect.

2. **The mix: three dequant characters at one price.**
   - **iq4_xs v5 executes the fewest instructions in the fleet** (255/iter against q4_0's 294:
     the table lookup replaces the nibble -> int -> float -> subtract chain) **but the 16-entry
     `constant` table compiles to a device load per nibble**: 44 loads in the loop against
     q4_0's 12, the difference being exactly 4 rows x 8 nibbles. Those loads set its character -
     13.5% stall, the top three stall sites all 12 B load consumers (0.92/0.87/0.47%), and the
     dearest per-instruction issue among the SoA kernels (8.62 vs q4_0's 7.61: a load issues
     dearer than an FMA). Net: the same 221 us as q4_0 for 5.6% fewer bytes, 1.27x vs 1.21x.
     The lane-held-table `simd_shuffle` forms (v3/v6) were the register-resident alternative and
     doubled the text and the time; this trace says the lookup's real cost is ~6 points of
     x-floor, and there is no cheaper form on the table (the iq4nl values are not arithmetic).
   - **q4_K v2 is the most issue-bound kernel measured on this machine: 95.7% issue, 4.3% stall,
     flat** (largest site 0.19%; the fleet range in `instruction-economy-league.md` was 64-89%).
     Its +10 instructions/iter over q4_0 are the min-fold (the 8-element activation sums shared
     by the 4 rows, then 4 FMAs), ~3%; the rest of its 1.29x is the half-planar layout's +11%
     bytes. There is nothing left in it at this geometry.
   - **q5_K v2 carries the high-bit plane: +78 instructions/iter over q4_0 (+27%), +68 over
     q4_K** - 4 byte loads and a shift/and/or per element (the 8 B:140 / 10 B:83 fingerprint of
     the 2-operand integer forms) - at 8.2% stall and 81 registers. Per byte it is the most
     efficient of the three (1.26x on a 22% larger floor); per instruction the heaviest. The
     one form lever the trace names is here: fold the high bit as a separate `16*s*(h . v)`
     term with a select per element instead of merging it into the nibble (~-8 of 372 ops, so
     ~2%, ~0.5 ms/round - priced, not worth a probe on its own).

3. **The instruction-economy law holds across the formats.** Per-instruction issue cost is
   7.6-8.6 us/M for every kernel in the table except ext_iq4_xs (11.4 - the same LUT loads on
   f32 arithmetic with spill), so time = executed instructions x that cost / issue share to a
   few percent, and the way to read a UD kernel is: count its loads, count its instructions per
   pack-iteration, and check the issue share; register pressure only enters through spill.

4. **Kernel headroom at width 4, from this table:** to the q4_0 reference's 1.21x, iq4_xs has 5%,
   q4_K 6%, q5_K 4% - ~3.5 ms of the round's 71 ms SoA bucket. **The width-4 kernel plane on UD
   is within ~5% of the Q4_0 line's kernel on every format.** The rest of the round's matmul
   time is geometry (the short attention shapes at 1.5-1.8x in the real graph, which the q4_0
   kernel shows too) and the ~16 ms of formats still on the ext/per-column paths (item 2 of the
   open list); the ~5% in-graph tax over synthetic (234.6 vs 220.9 us for iq4_xs up/gate; the
   drafter's q4_0 shows 227 vs 221) is common to all of them.

## Step 8: prefill - the f32 64-column mul_mm tile for the K-quant formats (2026-09-05 evening, branch `ud-soa-iq4xs`)

Opened after step 7 (owner: "If you can move prefill, it would be fableous"). The wall-clock case:
on the benchmark request prefill is 73.6 s against 12.8 s of decode, and the whole remaining
decode-kernel plane is worth ~1 s, while UD's prefill sat 5.3 s behind the Q4_0 line's non-acch
prefill (73.0 vs 67.7). Prefill is compute-bound, so the +7% bytes are not it.

**Measurement.** UD's prefill decomposes from the step-7 profile log the way Q4_0's did: m1
serialized 72.8 s against a 73.6 s wall - GPU-busy end to end - of which MUL_MAT 64.7 s (iq4_xs
22.7, q5_K 20.2, q4_K 16.2, q3_K 1.8, iq4_nl 1.5, q6_K 1.4, iq3_s 0.9), FA 3.0, GDN 2.1. Per format
the n=512 kernels run at iq4_xs 6.5-6.6 TFLOPS, q4_K 6.3-6.4, q5_K 5.7-5.9, q3_K 5.9 against
q4_0's 6.7-6.8 on the same shapes (`metalprof-buckets.py`'s parse, TFLOPS from the row's own
count and shape). The n=512 perf cases for the K-quant types were added to `test-backend-ops`
(both lists), and the kernels captured and replayed at the gate/up shape (`profiles/ud-mm-n512-sep05`,
same driver as step 7, `--kernel mul_m`); timings from an uncaptured pass, ms per call:

| kernel (n=512, [17408,5120]) | ms | live | regs | hot loop instr/K-step | issue/stall | exec/disp |
|---|--:|--:|--:|--:|--:|--:|
| mul_mm_q4_0_f32 | 13.28 | 364 | 56 | 174 | 99.0/1.0 | 488M |
| mul_mm_iq4_xs_f32 | 13.65 | 393 | 54 | 204 | 98.6/1.4 | 571M |
| mul_mm_q4_K_f32 | 14.02 | 418 | 53 | 218 | 97.9/2.1 | 626M |
| mul_mm_q3_K_f32 | 15.15 | 461 | 56 | 272 | 95.1/4.9 | 761M |
| mul_mm_q5_K_f32 | 15.47 | 479 | 54 | 279 | 95.5/4.5 | 796M |
| mul_mm_acch_n64_q4_0_f32 (the Q4_0 pick) | 12.40 | 442 | 69 | 234 (per 64 cols) | 98.8/1.2 | 328M |

- **Every mul_mm is issue-bound with zero spill and one hot K-loop; the K-quant deficit is
  dequant instruction count**, nothing else. Per 32-wide K-step each thread dequantizes 16
  weights once for a 32-column output tile: q4_0 pays ~48 instructions for it, iq4_xs +30,
  q4_K +44, q3_K +98, q5_K +105 (the scale/min unpack plus and/select/add/convert/FMA per
  element - ~6 ops per element, the format's arithmetic minimum in this form).
- **The MMA instructions dominate the cycles, so instruction counts undercount them.** The acch
  n64 kernel executes 33% fewer instructions than the f32 32-column kernel and is 6.7% faster.
  A two-term fit over q4_0/iq4_xs/q4_K (time = MMA cycles + c x non-MMA instructions per step)
  gives ~10.9 ms of MMA cycles and 0.017 ms per non-MMA instruction per K-step at this shape:
  the non-MMA share is 18% for q4_0 and 29% for q5_K, and that share is what a wider tile halves.

**Lever built: `kernel_mul_mm_n64_{q4_0,q4_K,q5_K,q6_K,q3_K,iq4_xs}_f32`** - the existing
`kernel_mul_mm` template instantiated at NR1=64 with f32 accumulators (the only in-tree n64 was
the acch q4_0 one), plus the f32 direct-store helper templated on the tile count. Same
accumulation order as the 32-column kernel, so the output is bit-identical. Offline prescreen:
all seven instantiations zero spill (text 4.9-6.0 KB vs 3.6-4.7 for the 32-column kernels).
Route in `ggml_metal_library_get_pipeline_mul_mm`: `GGML_MM_N64=1` (already in the pick env),
f32 types above, N=512, M>=4096, M%64==0, no bounds-check tile, K<=`GGML_MM_N64_KMAX` (default
6144, the acch tile's measured boundary); the acch q4_0 route is untouched (`GGML_MM_ACC_HALF`
still wins for q4_0, so the Q4_0 pick and the drafter are unchanged). `GGML_MM_N64_F32=0` is the
A/B off switch. Correctness: n=512 CPU-reference cases for all six types at both FFN shapes,
route names read from the test runs (`kernel_mul_mm_n64_*`).

**Synthetic (`perf/run-ud-mm-n64-ab.sh`, TAG `ud-mm-n64-sep05`, 2 interleaved reps, ms/call):**

| type | gate/up base -> n64 | ffn_down (K=17408) base -> n64 |
|---|---|---|
| q5_K | 16.02 -> 14.27 (**-10.9%**) | 16.19 -> 14.73 (-9.0%) |
| q3_K | 15.57 -> 14.07 (-9.6%) | 15.92 -> 14.62 (-8.2%) |
| q6_K | 14.72 -> 13.72 (-6.8%) | 14.98 -> 14.13 (-5.7%) |
| q4_K | 14.40 -> 13.67 (-5.1%) | 14.84 -> 14.15 (-4.6%) |
| iq4_xs | 14.01 -> 13.54 (-3.4%) | 14.29 -> 13.94 (-2.5%) |
| q4_0 (f32, reference) | 13.64 -> 13.36 (-2.1%) | 13.86 -> 13.80 (-0.5%) |

The gain orders exactly by dequant length, as the trace said it would. On q4_0 the f32 n64 tile is
still 7% behind the acch n64 (13.36 vs 12.40): the half accumulate is a real second effect on
q4_0, only the tile widening transfers to UD. ffn_down (K=17408, 640 threadgroups) gains on every
K-quant, so the K guard is a q4_0-acch boundary, not a general one; projected over the step-7
profile's prefill rows (61 of 65 s eligible): -3.8 s.

**E2e (`run-ud-knobs.sh` B=ud-soa, depth 3, n_predict 300, prompt 8288 tokens, TAG
`ud-mm-n64-e2e-sep05-*`, interleaved base / n64 K<=6144 / n64 K<=20000 / base):**

| arm | prefill | decode t/s | sha |
|---|--:|--:|---|
| base (`GGML_MM_N64_F32=0`) | 73.54 s | 23.45 | 73ea53bbe98f |
| n64 f32, K<=6144 | 70.70 s (-3.9%) | 23.47 | 73ea53bbe98f |
| **n64 f32, K<=20000** | **69.98 s (-4.8%)** | 23.54 | 73ea53bbe98f |
| base again | 73.40 s | 23.37 | 73ea53bbe98f |

Engagement confirmed in the server run (`-lv 5`, TAG `ud-mm-n64-engage-sep05`, prefill 70.15 s, same sha):
all five `kernel_mul_mm_n64_{iq4_xs,q4_K,q5_K,q6_K,q3_K}_f32` pipelines load, the drafter's q4_0 stays
on `kernel_mul_mm_acch_q4_0_f32`.

**Byte-identical (the canonical UD 300-token sha on every arm), decode untouched, -3.5 s of
prefill; the guard lifted to cover ffn_down is the better point.** UD prefill 73.5 -> 70.0 s
against the Q4_0 pick's 62.5 (acch) / 67.7 (f32). Recommendation: adopt with
`GGML_MM_N64_KMAX=20000` in the UD pick (or flip the default for the f32 tiles); no quality
question arises. The Q4_0 line could also take the f32 n64 tile instead of acch if the owner ever
wants the acch KLD cost back (13.36 vs 12.40 ms per call, i.e. ~-7% of its mm time).

**What the trace says is left in UD prefill mm** (not built): after n64 the K-quant kernels sit
3-7% behind q4_0's f32 tile per call (q5_K 14.27 vs 13.36), ~1.5 s of prefill, and the dequant
chains are at their per-element arithmetic minimum in the byte-load form - a uint-load +
nibble-spread form for q5_K's high bit is worth ~10% of its dequant, so ~0.5 s. The bigger items
are now FA (3.0 s), GDN (2.1 s) and the acch question (-6.9% wall for -2.5 pt same-top, still
not recommended). Open list order: adopt n64 f32 (owner), offline GGUF (another context),
remaining decode formats, FA ladder.

## Step 9: flash attention - the transposed-Q QK form (2026-09-05 night, owner: "then take a look at FA")

FA is 3.0 s of UD's prefill (3.1 on Q4_0, same kernel: `kernel_flash_attn_ext_f16_dk256_dv256` at
nsg=4, nwg=1, 8 queries x 64 keys per threadgroup, 16 calls per KV rung from 2560 to 8448, the top
rung 22.9 ms in-graph) and 7.5 ms of the 118 ms decode round (the same kernel at nwg=8, widths 3-8).
Quadratic in context, so its share grows with the prompt.

**Measurement** (`perf/run-fa-prefill-profile.sh`, `profiles/fa-prefill-sep05`, kv 8448 and 16384 at
512 rows; timings uncaptured): 972 live instructions, 90 registers, zero spill, 91% issue / 9%
stall, 896M instructions per dispatch at kv 8448 for 103.8M MMAs = **8.6 instructions per MMA
against mul_mm's 5.5**; at the test's FLOP count 5.1 TFLOPS against mul_mm's 6.8. The hot-loop
rule catches only the innermost QK body, so the read is by execution-count tier
(`shaderprof-compare.py --tiers`, new): per 64-key chunk each simdgroup executes **1105
instructions for 128 MMAs** -

| tier | per chunk per SG | instr | issue share | what it is |
|---|--:|--:|--:|---|
| innermost QK body (unroll-4 step, 8 MMAs) | 3.9 x 194 = 749 | 194 | 63% | 89 non-MMA per 8 MMAs: a ~40-instruction 10/12 B address block, then 24 load instructions for 8 K tiles + 8 Q tiles, then the 8 MMAs |
| per-chunk body (softmax + PV + barriers) | 320 | 320 | 27% | 64 PV MMAs at ~4 non-MMA each, 64 plain V loads, the online softmax for 2 rows |
| per-key-tile / per-chunk glue | ~36 | 43 | 1% | |

The QK side pays 11 non-MMA instructions per MMA, the PV side 4. The difference is the operand
form: PV loads V tiles untransposed (`simdgroup_load(mv, pv, NS20, 0, false)`); QK loads K tiles
**transposed from device memory** (`simdgroup_load(mk, pk, NS10, 0, true)`) - every lane needs two
column-strided addresses per tile, which is the address block. K^T is only needed because the
kernel computes S = Q K^T with Q as the left operand.

**Lever built: `kernel_flash_attn_ext_qt_f16_dk{128,256}` (`GGML_FA_QT=1`, "=0" off, f16 K/V only,
takes precedence below acch)** - compute S^T = K Q^T instead: the K tile loads untransposed
(contiguous rows), Q is staged transposed once per threadgroup (`sq[(4i+c)*Q + j]`, same threadgroup
bytes) and its tiles load with stride Q, and the score tile is stored with the transpose flag once
per key tile, so `ss` keeps its [query][key] layout and the softmax/PV code is untouched. Same
products in the same k order, so the output should be bit-identical; the quantized-K paths (Turbo4)
keep their form (`static_assert` pins QT to the f16 path, one 8-query tile, DK % 16 == 0). Template
parameter `QT` on the impl and the wrapper, default false; the existing pipelines compile unchanged.

- Offline prescreen: text 12580 vs 12586 B, spill 48 vs 32 B (threshold noise; replay read the
  baseline at 0/90 regs). **The text size did not move** - this form change is invisible to the
  prescreen and only timing or the per-tier decode sees it.
- Correctness: **2188/2188** f16 `FLASH_ATTN_EXT` cases against the CPU reference with `GGML_FA_QT=1`
  (113 QT pipelines loaded in the run: every nsg/nwg/gqah/mask form).
- Kernel timings (`results/fa-qt-sep05/timing.txt`, interleaved QT=0/1, 2 reps, us):

| form | QT=0 | QT=1 | delta |
|---|--:|--:|--:|
| prefill nwg=1, kv 8448, 512 rows | 20652 | 18925 | **-8.4%** |
| prefill nwg=1, kv 16384, 512 rows | 41080 | 37702 | **-8.2%** |
| decode nwg=8, kv 8448, width 4 | 414.0 | 385.7 | **-6.8%** |
| decode nwg=8, kv 8448, width 5 | 416.7 | 387.6 | -7.0% |

Both routes win, decode included (the width-3..8 verify FA is this kernel at nwg=8). E2e gate below.

**E2e (UD, depth 3, n_predict 300, n64 tile on in every arm, `run-ud-knobs.sh`, TAGs
`ud-fa-qt-e2e-sep05-*`, interleaved base / QT / QT / base):**

| arm | prefill | decode t/s | sha |
|---|--:|--:|---|
| base (`GGML_FA_QT=0`) | 70.30 s | 23.41 | 73ea53bbe98f |
| QT | 69.95 s | 23.66 | 73ea53bbe98f |
| QT | 69.68 s | 23.68 | 73ea53bbe98f |
| base | 70.04 s | 23.57 | 73ea53bbe98f |

**Byte-identical; prefill -0.5% (70.17 -> 69.82 s, the predicted 8% of a 3.0 s FA ladder), decode
+0.8% (23.49 -> 23.67, predicted ~+0.5% from FA's 7.5 ms of the round).** Small at an 8K prompt
because FA is 4% of prefill and 6% of the round here; it is a 7-8% kernel win on every f16 FA form
and grows with context (quadratic on the prefill side). Recommendation: adopt `GGML_FA_QT=1` on both
lines (the kernel is shared; the Q4_0 pick runs the same f16 forms; the Turbo4 line's quantized-K
kernels are untouched by it). Branch `ud-soa-iq4xs`; cherry-picks cleanly onto prod (FA kernel +
getter only).

**What is left in FA, from the tiers:** after QT the QK body still carries the Q tile reloads (32
threadgroup loads per key tile, ~2 instructions each) - the loop-order swap (load each Q tile once
per DK slice, accumulate into NC=2 key-tile accumulators) removes half of them, byte-identical, worth
maybe 3%; and the structural item is the 8-query tile - K/V are re-streamed from cache once per 8
queries (64x per head at 512 rows) and the per-chunk softmax/barrier body is paid per 8 queries. A
16-query prefill instantiation halves both but the QK/PV code assumes one 8-row Q tile, the
threadgroup memory lands exactly at the 32 KB limit, and registers are at 90: real surgery, prefill
only (decode widths would waste the wider tile). Not built.

## Step 10: prefill, what else moves (2026-09-05 late; owner: "anything that moves prefill is super-interesting")

**The "pre-batch-1 CPU gap" of `prefill-decomp.md` item 3 is closed - it is GPU, not CPU.** From
tonight's server log (`ud-fa-qt-e2e-sep05-base`): the slot launches at 0:09.100 and the spec-prof
`decode` bracket for batch 1 prints 16.714 s at 0:25.814, so the bracket opened at 0:09.100 - the
whole 16.7 s from launch is inside it; `dec_sub_pp` (llama_decode) is 12.455 s of that, and the
remaining 4.26 s is the fourth 512-token ubatch executing after llama_decode returned submit-only,
the same 4.2 s per ubatch as the other three (2048 tokens = 16.7 s = 8.16 ms/token throughout).
Nothing free there.

**f16 activations into the prefill mul_mm.** The tile converts its B operand to half at staging on
every K step (`(S1_2x4)(*yj)`) and re-reads the f32 activation block from cache once per 64-row
tile (2.85 GB per gate/up call). The `kernel_mul_mm_*_f16` instantiations exist for every format;
per call at n=512 gate/up, f32-B -> f16-B (2 reps, `results/ud-mm-f16b-sep05.txt`): q5_K 15.46 ->
15.18 ms (-1.8%), q4_K 14.03 -> 13.67 (-2.5%), iq4_xs 13.64 -> 13.37 (-2.0%), q4_0 13.29 -> 12.97
(-2.4%). Casting src1 to f16 once is the same round-to-nearest the tile applies, so it is
byte-identical by construction. Built backend-side like the decode f16y path: `GGML_MM_F16B=1`
reserves an f16 scratch after dst for mm-routed f32-B ops on the quant types with an f16 tile,
casts with `kernel_cpy_f32_f16` (0.06 ms against a 13 ms call), and runs the `_f16` tile; the n64
tiles got f16-B instantiations so the two compose; the acch q4_0 route (f32-B only) is excluded.
Expected ~-2% of the 61 s of prefill mm, ~1.2 s.

**GDN at prefill: 2.1 s, a latency-bound scan.** `kernel_gated_delta_net_f32_4` at 512 tokens runs
2.72 ms per layer call (768 calls): one threadgroup per (head, 4 state rows) walks all 512 tokens
sequentially with two `simd_sum`s per token - about 0.3 TFLOPS of useful work, starved, not at a
roof. A chunked delta-rule kernel (intra-chunk matmuls + inter-chunk state) would turn it into
MMA-parallel work; upstream has none for Metal. Up to ~2 s (3%) of prefill, several days, and not
byte-identical (different summation order). Parked as the next big item after the mm plane.

**Elementwise prefill passes** (RMS_NORM 0.39 s, SWIGLU 0.52, SILU 0.33, ADD 0.31, CONCAT 0.30,
SSM_CONV 0.24, MUL 0.17): all memory-bound at ~1.2x their bandwidth floor (SWIGLU on [17408,512]
f32: 107 MB per call, 0.51 ms measured vs 0.39 floor). Fusion could take ~0.5 s of the ~2.3 s;
the CONCAT (the SSM conv-state prepend, 0.33 ms x 768) is avoidable in principle.

**Result (fast cast built, same night).** The first route used the generic `kernel_cpy_f32_f16` and
LOST: e2e prefill 69.7 -> 73.4 s (+5.3%) at an identical sha. The profiled run put it in the mm rows
themselves (the cast is encoded inside the op): ffn_down calls went 14.1 -> 15.4 ms because
`kernel_cpy_t_t` runs one element per thread with four 64-bit index divisions per element - ~6 G
elements/s, 1.5 ms for an 8.9M-element activation, more than the tile saves. `kernel_cvt_f32_f16_cont`
(8 elements per thread, contiguous, no index math; route gated on contiguous src1) fixes it:

| n=512, n64 tile, f32-B -> fast-cast f16-B | gate/up [17408,5120] | ffn_down [5120,17408] |
|---|---|---|
| q5_K | 13.80 -> 13.41 ms (-2.8%) | 14.14 -> 13.72 (-3.0%) |
| q4_K | 13.27 -> 12.88 (-3.0%) | 13.65 -> 13.16 (-3.6%) |
| iq4_xs | 13.14 -> 12.77 (-2.8%) | 13.48 -> 13.07 (-3.1%) |
| q4_0 (f32 acc) | 12.93 -> 12.54 (-3.0%) | 13.33 -> 12.81 (-3.9%) |

(cast included; 12/12 n=512 CPU-reference cases with the route on, `kernel_cvt_f32_f16_cont` +
`kernel_mul_mm_n64_*_f16` named in the runs). **E2e UD depth 3 @300, interleaved base / f16b / f16b /
base (n64 + QT on): prefill 69.70 / 67.70 / 67.70 / 69.72 s = -2.0 s (-2.9%), sha `73ea53bbe98f` on
every arm, decode 23.5-23.7 either way.** `GGML_MM_F16B=1` recommended on both lines.

**Tonight's prefill stack on UD, all byte-identical:** 73.5 s -> n64 tiles 70.0 -> FA transposed-Q
69.8 -> f16 activations 67.7 s (**-7.9%**), against the Q4_0 pick's 65.4 (acch). Lesson for the
record: a backend-side cast must be a dedicated contiguous kernel; the generic cpy is index-math
bound and a plausible-looking +2% lever measured as a -5% loss until the profile said which row grew.

### Step 9 addendum (2026-09-06): the transposed-Q kernel traced - why the text size did not move

Owner asked whether we understand it. The QT kernel captured and decoded next to the baseline
(`profiles/fa-prefill-qt-sep05`, `shaderprof-compare.py --tiers`):

| | baseline | QT |
|---|--:|--:|
| live instructions (static) | 972 | 970 |
| device load instructions (static) | 101 | 85 |
| threadgroup stores (static) | 26 | 32 |
| executed per dispatch | 896.0M | 856.2M (-4.4%) |
| issue / stall | 91.1 / 8.9 | 92.7 / 7.3 |
| innermost QK body (per unroll-4 step) | 194 instr, 14 B loads 48, 10 B 96 | 182 instr, 14 B loads 32, 10 B 104 |
| per-chunk body | 320 | 327 |
| us per M issued | 21.0 | 20.5 |

**The mechanism is the load instruction count, not the address block I named.** A transposed 8x8
half tile load from device memory lowers to three load instructions per tile (each lane fetches its
two halves from two rows separately); the untransposed load is one. Eight K tiles per step: 24 -> 8
loads (-16, exactly the 14 B delta), while the address arithmetic stayed (10 B forms went UP by 8).
The step is -12 instructions net, -4.4% executed per dispatch; loads issue dearer than ALU, so the
per-instruction cost fell 2.4% as well, and the stall share fell 1.6 points: -4.4% x -2.4% x stall =
the measured -8.4%.

**Why the offline text size was flat (12580 vs 12586 B):** the static bytes sum the prologue, the
per-chunk body and the inner body with equal weight, and the change moved bytes between levels at
equal size - -16 x 14 B loads and +8 x 10 B ALU in the inner body, +7 instructions in the per-chunk
body (the transposed score store), plus the transposed Q staging in the prologue. The runtime
weights those levels 3.9 : 1 : 0.003 per chunk. Text size is a bad proxy exactly when a form change
swaps instruction classes at equal byte size or moves work between loop levels; it ranks
register-pressure and unrolling changes, not this.

## Step 11: the census's first lever - GQA-reuse FA for f16 KV at decode (2026-09-06)

`perf/kernel-census.md` ranked the width-4 decode FA (`flash_attn_ext_qt_f16` nwg=8) as the table's
largest per-work outlier: 16.9 instructions per useful GFLOP against the mul_mm's 3.8, 4x the loads,
2.2 TFLOPS. The cause is structural: an 8-query tile carrying 4 real queries (2x MMA padding) and
K/V re-streamed once per 4 rows. The gqah=6 GQA-reuse instantiation (six query heads per KV head
flattened into full 8-row tiles, each K/V chunk streamed once per 24 rows) already existed for the
Turbo4 line and was only ever routed for Turbo4 KV. `GGML_FA_GQA_F16=1` ("=0" off) routes it for
f16 KV too, widths 3-6, same conditions as the Turbo4 route.

| kv 8448, us/call (2 interleaved reps) | QT | QT + GQA f16 |
|---|--:|--:|
| width 3 | 379 | 214 (-44%) |
| width 4 | 385 | 219 (-43%) |
| width 5 | 388 | 307 (-21%) |
| width 6 | 391 | 328 (-16%) |

2452/2452 f16 `FLASH_ATTN_EXT` cases against the CPU reference with the gqah=6 pipelines engaged.
**E2e UD depth 3 @300, interleaved base / gqa / gqa / base: decode 24.38 / 25.04 / 25.06 / 24.38 t/s
(+2.8%), sha `73ea53bbe98f` on every arm, prefill unchanged (65.6 s all arms).** Same math per
(query, head) row in the same order, so byte-identical as expected. Recommended for both lines (the
Q4_0 pick verifies at width 5: -21% per FA call; gate below). The remaining width-5/6 gap to the
width-3/4 number is the second tile being half-empty (24 rows -> 3 full tiles at width 4, 30 rows ->
4 tiles with 2 empty rows at width 5).

## Step 12: the census's second lever - rows per simdgroup for the GDN prefill scan (2026-09-06)

Owner: "let's try the intermediate routes for now ... keep it byte-identical." Census flag #2 was the
prefill `kernel_gated_delta_net_f32_4` at 10.5x its byte floor / 25% stall: one simdgroup per state
row walking 512 tokens through a dependent chain (exp, dot, simd_sum, delta, fma, dot, simd_sum),
every row of a head re-loading the same q/k/g/beta. `GGML_GDN_NR=4` (`gdn-prefill-scan.md`) gives a
simdgroup 4 consecutive rows: the chains interleave, the loads amortize, same arithmetic per row in
the same order. Real shape (16 k-heads, 48 v-heads - the census's perf case had been a third of the
op, fixed) per call 2.10 -> 1.17 ms (K=1), the pick's ext form 2.38 -> 1.25 ms; the kernel goes
from 74/22 issue/stall to 93/6, i.e. from latency-bound to issue-bound. **E2e UD depth 3 @300, base /
nr4 / nr4 / base: prefill 66.87 / 65.80 / 65.64 / 66.71 s (-1.6%), sha `73ea53bbe98f` on every arm,
decode unchanged (gated at 32 tokens).** Not UD-specific (f32 kernel): Q4_0 pick prefill 64.2 -> 63.0 s (-1.9%), canonical sha `95eb7e65977e`; recommend on both lines.
Refuted on the way: loads pipelined a token ahead (+9.7%: nothing waits on memory after NR) and
32-bit token offsets in place of the five 64-bit pointer advances (+2.7%: the add moves to the loads).

## Step 13: the offline SoA GGUF - the ~12 GiB of side buffers go away (2026-09-06, owner: "let's do the offline repack")

The runtime SoA route (step 6) held a second copy of every iq4_xs/q4_K/q5_K weight: the width-3/4/5
kernels read the half-planar side buffers, widths 1-2 and the prefill tiles read the original rows,
so both had to stay resident (~12 GiB on top of the 15.3 GiB file, built during the first verify
round). The Q4_0 line's answer was a storage type the kernels read directly (`q4-0-soa-gguf.md`);
this is the same step for the three UD formats, branch `ud-soa-gguf`.

**The stored row is the side-buffer row with the original block header appended.** Per row of
`nsb = ne00/256` superblocks:

| type | row layout | B / superblock | vs plain |
|---|---|--:|--:|
| `IQ4_XS_SOA` | `[half d*(ls-32) x 8nsb][uint32 pack8 x 32nsb][u8 hdr8 x nsb]` | 152 | +11.8% |
| `Q4_K_SOA`   | `[half d*sc x 8nsb][half dmin*m x 8nsb][uint32 pack8 x 32nsb][u8 hdr16 x nsb]` | 176 | +22.2% |
| `Q5_K_SOA`   | as Q4_K_SOA with `[u8 hbits x 32nsb]` before the header plane | 208 | +18.2% |

The prefix (144/160/192 B per superblock) is byte-for-byte what `kernel_repack_*_soah` wrote, so
the pick's validated kernels (`kernel_mul_mv_{iq4_xs,q4_K,q5_K}_soa_w{3,4,5}_v5|v2`) read the model
buffer with no change and no repack - the decode path's numerics and streamed bytes are exactly the
step-6 ones (the header plane sits after the packs, untouched at widths 3-5). The header plane
holds the original `d, scales` bytes, which buys two things the half planes cannot: the prefill
tiles and the width-2/6-8 readers dequantize with the *exact* expressions of `dequantize_iq4_xs /
q4_K / q5_K` (`dequantize_{iq4_xs,kq}_soa_mm`), and the file reverses byte-for-byte
(`llama-gguf-repack --reverse`). The alternative - half planes only, 1.1 GiB less - would have
made prefill read rounded scales and the file one-way; the exact header costs 1.1 GiB of the
2.06 GiB growth and was the safer productization.

Readers, all new code, no existing hot kernel touched:
- **widths 3-5** on the row whitelist: the step-6 kernels, routed by type instead of env
  (`GGML_MV_SOA_IQ4XS`/`KQ` stay for the plain types' runtime side buffers, inert on stored).
- **widths 2, 6-8, and 3-5 off the whitelist**: `kernel_mul_mv_ext_soa_q4x4_f32_impl`, the q4x4 ext
  kernel with (row pointer, superblock, tile) addressing through a tag-dispatched
  `dequantize_soa_mm((device const block_q *) nullptr, row, ne00, blk, il, reg)`; 24
  instantiations `kernel_mul_mv_ext_{type}_soa_{f32,f16}_r1_{2..5}`.
- **prefill (mul_mm, incl. the n64 f32/f16 tiles)**: the existing `FC_mul_mm_soa` branch, which
  was `dequantize_q4_0_soa_mm` hard-wired, now calls the same tag dispatch, so every stored type
  rides the underlying type's tile instantiation (`kernel_mul_mm_n64_q4_K_f16` + `soa=1`).
- **width 1**: the step-6 kernel bodies gained a `TY` parameter and `NC` guards; an NC=1
  instantiation with f32 activations (`kernel_mul_mv_{type}_soa_w1`, 4 rows x 2 simdgroups)
  replaces the upstream block kernel. This is the one path whose numerics change (half-product
  form, new summation order): the no-spec b1 anchor is a new sha lineage, the spec arms are not.
- **CPU**: `ggml_soa_pack_* / unpack_*` in `ggml-quants.c` define the layout once; the quantize /
  dequantize refs, the `vec_dot` oracle `test-backend-ops` compares Metal against (f32 dot through
  the exact unpack) and the converter all use them.

Correctness: `test-backend-ops` 84/84 Metal cases for the three types (widths 1-8, 32, 512, the
17408/5120 FFN pair, an off-whitelist 1024-row shape, the GDN-style `[K,T,S]` fold);
`test-quantize-fns` passes the three. Conversion of the UD file with `--verify` (every row
reversed in memory and compared): 339 tensors, 12.14 -> 14.20 GiB (+16.9%); token_embd
(GET_ROWS) and the twelve 5120x1024 attn_k/v (below 16M elements) stay plain and keep their
upstream kernels.

**E2e, fresh processes, mirrored (orig stored stored orig), `run-ud-soa-gguf-ab.sh`, TAG
`ud-soa-gguf-sep06`, the 8288-token canonical prompt, n_predict 600, full pick env in both arms
(the runtime SoA env is inert on stored types):**

| arm | depth 3 t/s | acc | prompt | sha1 | wired+anon vs idle |
|---|--:|--:|--:|---|--:|
| orig + runtime side buffers | 24.757 / 24.738 | 60.5% | 66.32 / 66.66 s | `5e76afaba36c` | +35.02 / +34.44 GiB |
| stored SoA GGUF | 24.704 / 24.535 | 60.9% | 65.63 / 65.63 s | `5e76afaba36c` | +23.88 / +23.83 GiB |

Decode -0.5% (inside the two-run spread), prefill -1.3%, **-10.9 GiB resident**, the canonical
UD 600-token sha on every arm (acceptance moves because the drafter reads target activations, the
committed text does not - as in step 6). The route proof (`-lv 5`, stored file, depth 3): the
three `*_soa_w4_v5|v2` kernels, `kernel_mul_mv_ext_*_soa_f16_r1_2` at width 2, `kernel_mul_mm_*_f16
... soa=1` at prefill, `kernel_repack_*` only for the Q4_0 drafter.

**No-spec b1 anchor** (TAG `ud-soa-gguf-sep06-b1v2`, orig stored stored orig): 12.666 / 12.659 vs
12.324 / 12.292 t/s = **-2.8%**, same sha `5e76afaba36c` (the 600-token text survives the new
width-1 kernel), +30.0 vs +18.9 GiB. The first width-1 form read the half scale planes (+8..16 B
per superblock over the plain file) and measured -3.7%; the shipped form (`SM=2`) reads the
exact header plane and streams the plain file's bytes. What is left is per format, `test-backend-ops
perf` n=1 (us/call, plain -> stored):

| shape | iq4_xs | q4_K | q5_K |
|---|--:|--:|--:|
| 17408x5120 | 205.1 -> 200.7 (-2.2%) | 202.6 -> 232.8 (+14.9%) | 285.9 -> 274.2 (-4.1%) |
| 5120x17408 | 211.3 -> 205.9 (-2.6%) | 205.8 -> 243.0 (+18.1%) | 291.1 -> 283.7 (-2.5%) |

The iq4_xs and q5_K readers beat their upstream block kernels; the q4_K one loses 15-18% - the
upstream q4_K kernel runs at ~250 GB/s (0.9x the byte floor), the 4-row body at ~215 GB/s
(issue-bound: the per-pack scale/min math and the min-fold add ~1.3 instructions per element on a
~7-per-element budget). Weighted by streamed bytes that is the whole -2.8%. **Open:** port the
upstream q4_K width-1 form (16 elements per thread, packed scale masks) to the stored packs -
ceiling ~+1.5% b1 over plain, i.e. the stored file would win at every width. Not done: the pick
runs depth 3, where the file is at parity.

Harness trap (cost one run): a llama-server's listening socket outlives `kill -TERM` by the length
of its in-flight prefill plus the Metal teardown, and `kill -9` cannot interrupt the GPU wait; the
next arm's server binds the same port (`SO_REUSEPORT`), its POST lands on the dying process and
returns empty. `run-ud-soa-gguf-ab.sh` now waits for the pid and the port before launching.

**What this leaves.** The UD pick's memory is the file (17.4 GiB on disk, +2.06 GiB over the plain
15.3) plus KV/compute - no first-verify repack, no ~12 GiB of side buffers. The plain file is
redundant once the stored one verifies (`--reverse --verify` restores it byte-for-byte).
Adoption is the owner's: `-m Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf` with the unchanged pick env.
