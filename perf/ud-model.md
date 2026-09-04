# UD-Q4_K_M as the target: optimizing the exact unsloth file (2026-09-04, OPEN)

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
the naive expectation was the reverse. Single-prompt acceptance is trajectory-dependent
(`acceptance-by-prompt.md`; the 2026-08-24 measurement on the same prompt had UD slightly
AHEAD, 43.0 vs 41.3) so this is not a verdict; `run-corpus-acceptance.sh` on UD is the
instrument. Not run yet (GPU-serialized behind steps 2-5).

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
