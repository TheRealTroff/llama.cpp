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

1. **Offline GGUF storage** for the three layouts (the `Q4_0_SOA_V1` path: new ggml types,
   `llama-gguf-repack`, readers at widths 1-2 and the mm/prefill path) - removes the ~12 GiB
   and the first-call repack; the Q4_0 line measured that step at +2%.
2. **Remaining width-4 formats** ~10 ms of the (now ~100 ms) round: q6_K attn/ffn tensors
   (0.6 GiB; the q6_K head is already 1.32x floor), q3_K, iq3_s (5.9x floor, 3 calls/round),
   iq4_nl (6 tensors; same table as iq4_xs with a per-32 scale - the cheapest to add).
3. **Turbo4 KV on UD** - untried, weight-format independent.
4. A fresh round decomposition at the new point, then the acch prefill question again
   (unchanged: -6.9% wall for -2.5 pt same-top, not recommended).
