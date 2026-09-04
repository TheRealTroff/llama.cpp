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

(running)

## Step 4: acc-half mul_mm for UD's formats (branch ud-acch-types)

Built: `kernel_mul_mm_acch_{q8_0,q3_K,q4_K,q5_K,q6_K,iq3_s,iq4_nl,iq4_xs}_f32` instances of
the same template as the q4_0 probe; host gate widened (n64 tile stays q4_0). Same
`GGML_MM_ACC_HALF=1` flag. Changes prefill numerics -> needs its own KLD pricing on UD
(q8_0 reference logits, `run-quant-kld.sh`) before adoption. Not measured yet.

## Step 5: skinny MMA generalized over the dequant block type (branch ud-skinny-generic)

Built: `kernel_mul_mm_skinny_t<block_q, nl, dequantize_func>`, the q4_0 skinny tile (32
rows x 8 cols, K-slice 64, 2 simdgroups, slice t+1 dequantized in registers under slice
t's MACs) with the per-thread 32-element dequant done by the generic block dequantizer
(the ones `kernel_mul_mm` uses). Instances for the same eight formats. Host route
`GGML_MM_SKINNY_GEN=N` sends ne11 in [max(2,N), 8] there for those types. Not measured yet.
