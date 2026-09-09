# Remaining UD formats: implementation and isolated kernel gate

Experiment: `exp/ud-remaining-quants`, based on prod `16c3c84a6`, worktree
`/Users/troff/play/llama.cpp-ud-remaining-quants`. No production adoption or
real-model benchmark is part of this gate. The owner authorized correctness
and repeated kernel timings on 2026-09-08, with a pause immediately afterward.

## Scope and storage

The conversion plan on `Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf` selects 35 remaining
tensors, 1.37 GiB of native rows becoming 1.42 GiB (+3.9% for those tensors).
Embeddings, small matrices, and the Q6_K `output.weight` head remain native.
The file includes nextn-layer tensors; the shape census includes those too.
No converted model file was written for this test.

| Native | Stored enum | Bytes per block, native / stored | Contract |
|---|---:|---:|---|
| IQ4_NL | 50 | 18 / 18 per 32 weights | Original half scale plane, nibble-planar packs |
| Q3_K | 51 | 110 / 116 per 256 weights | Low-two-bit packs, high-bit masks, expanded exact signed scales, original half d |
| Q6_K | 52 | 210 / 212 per 256 weights | Low-nibble packs, high-two-bit packs, original signed scales and half d |
| IQ3_S | 53 | 110 / 136 per 256 weights | Two 9-bit grid indices and eight sign bits per pack, original scales and half d |

Planes span a complete logical row. Q3_K, Q6_K, and IQ3_S have two zero padding
bytes per superblock at the row tail. The CPU pack/unpack helpers retain every
native payload bit. No scale rounding or requantization is required.

The new width-1..8 kernels use four output rows per threadgroup, two SIMD groups
at widths 1..4 and one at widths 5..8. Each lane decodes eight contiguous K
values, reuses weights across token columns, and reuses activations across rows.
Products and accumulation use float. Wider batches use the existing matrix
tiles with new stored-row readers, including n64 and f16-B routes.

Lossless storage is not proof of identical model numerics: native kernels can
use different activation precision, dequantization expressions, and reduction
orders. A passing backend comparison is also not a decode-fidelity measurement.

## Correctness gate

Raw evidence is in `results/ud-remaining-20260908/`:

- `correctness-default.log`: 124/124 synthetic cases; all widths 1..8, tails,
  long K, generic prefill, and contiguous folded activation batches.
- `correctness-n64-f32.log`: 28/28 additional f32-B matrix-reader cases.
- `correctness-real-weight-slices.log`: 96/96 native/stored cases using the
  first 37 rows of actual tensors, at widths 1, 4, 8, and 512 with the prod UD
  environment. Each stored fixture reverses to byte-identical native rows.
- `correctness-final-f16b.log`: 40/40 after completing the new f16-B dispatch
  wiring. This includes real-weight slices and n64 cases. Exact f16-B pipeline
  names and `soa=1` are present in the log.

The backend MUL_MAT test uses a maximum normalized error of `5e-4`. This gate
does not assert bit-identical GPU results between native and stored layouts.

## Timing protocol

Run `bash perf/run-ud-remaining-kernels.sh`. It sources the ground-truth prod
`perf/pick.sh` UD environment, uses fresh processes, and runs uncaptured
`test-backend-ops perf` in mirrored `plain-1, soa-1, soa-2, plain-2` order.

Each arm contains 132 cases: all 12 native-format/matrix-shape combinations
found by the conversion census, at widths 1..8, 9, 32, and 512. Both arms read
the same actual weight bytes and use deterministic identical float activations.
The stored arm packs and reverse-verifies every selected matrix row before
timing. Fixture loading, conversion, and pipeline warmup are outside the timed
loop. This is isolated repeated-matrix timing, not full-model execution or a
whole-model cold-weight streaming test.

The ordinary harness runs each case for at least one second and retains the
number of operation repetitions. Complete pipeline base/name logs, the exact
environment, dirty source patch, and binary hashes accompany the results.
Pipeline messages can interleave with buffered stdout from the preceding case;
do not attach a newly compiled pipeline to a case merely because it appears
between that case's label and its `runs -` line.

Summarize with:

```sh
python3 perf/summarize-ud-remaining-kernels.py perf/results/ud-remaining-20260908
```

Positive percentages mean reduced latency. Aggregate percentages are geometric
means across distinct shapes, not tensor-frequency-weighted model speedups.
Repetition spread is reported separately and is not a confidence interval.

## Result

Completed at 15:44:37 UTC, 2026-09-08. The four timing arms took 9m39s; the
correctness/setup/timing window took about 15 minutes. All 528 timing samples
were present (132 cases per arm), with no unsupported cases counted as results.
Maximum within-arm repeat spread was 2.4% native and 1.7% stored. The complete
per-shape means and repetition spreads are in
[summary.md](results/ud-remaining-20260908/summary.md).

The following numbers are changes in latency relative to native rows; negative
is better. They are geometric means across each format's distinct shapes.

| Format | Single token | Width 4 | Width 512 prefill |
|---|---:|---:|---:|
| IQ4_NL | +17.9% | +135.0% | +0.1% |
| Q3_K | -24.0% | +109.9% | -4.7% |
| Q6_K | +9.8% | +125.4% | -1.1% |
| IQ3_S | +33.7% | +9.6% | -0.8% |

Q3_K single-token latency improves by about 21-27% across the two FFN
orientations, and width-512 prefill by about 4-5%. Its width-9/32 matrix readers
also improve (9.8% aggregate). Those are candidates for further investigation,
not a complete format-adoption result. The near-flat prefill changes for the
other formats are not a compelling reason to convert them.

The shared small-batch implementation loses at the pick's important width 4
for every format. Widths 5..8 are worse still; consult the per-shape table rather
than extrapolating from the width-4 values. The storage machinery passes its
checks, but this scalar multi-column kernel is not an acceptable blanket
replacement. No cause such as spilling or memory limitation has been profiled;
the timing results alone do not diagnose the bottleneck.

~~Decision: retain the work and evidence on the experiment branch, leave prod and
the UD model untouched, and stop here as requested. On resumption, separate the
Q3_K single-token/matrix-reader candidates from the losing multi-token tile.~~
Superseded the same evening by the port below (owner: "I strongly agree on 2").
Real-model validation, decode-fidelity measurements, and further tuning remain
outside this gate. Full converter integration tests and additional strided-view
coverage also remain before production readiness.

## Why the shared kernel lost, and the port to the kq-SoA form (2026-09-08 evening)

The cause was found offline before anything was rebuilt (`agx-spill-probe.py`, control
`kernel_mul_mv_ext_q4_0_f16_r1_4` at nr0=4/2 = 32/0 B as calibrated). The shared
`kernel_mul_mv_ud_soa<F,NC,KS>` body held its column streams in a `float x[NC][8]` array
under `#pragma unroll` with scalar f32 activation loads - the array-form trap already on
record in the prescreen skill:

| Width | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text bytes (q3_K) | 4848 | 6440 | 11428 | 7824 | 23294 | 11306 | 12560 | 13702 |
| Spill B/thread | 0 | 0 | 0 | 0 | 0 | 208 | 240 | 272 |

The odd widths balloon (the 7-19x cliffs), 6..8 spill, and width 4, with neither problem,
still competes against the ext `r1_4` f16 kernels, which already dequantize once and reuse
each weight across the columns with 16-byte activation loads. The same numbers hold for the
other three formats within a few hundred bytes.

The replacement is the pick's own q4_K/q5_K SoA body (`kernel_mul_mv_kq_soa_impl`) with the
per-format dequant swapped in (`kernel_mul_mv_ud_kq_impl`): 4 rows x NC columns per
threadgroup, per-row plane pointers hoisted out of the K loop, named `half8` streams v0..v4,
half products into f32 accumulators, the exact scale (original half d x original int8 or
nibble scale) rounded to half once per pack, and the Q3_K/Q6_K offsets (-4, -32) folded out
of the element loop as offset x scale x sum8(v) in f32. Widths 3/4 run two simdgroups
splitting K, width 5 one simdgroup, the kq-SoA dispatch geometry. Widths 2 and 6..8 take the
ext SoA readers (`kernel_mul_mv_ext_<type>_soa_f16_r1_N` over the existing
`dequantize_soa_mm` overloads; iq4_nl instantiated at 32 weights per block). Width 1 keeps the
shared body. Numerics class: the same as the pick's iq4_xs/q4_K/q5_K width 3..5 kernels (half
product, half-rounded exact scale), NOT bit-identical to the native kernels - a KLD gate is
still owed before any file conversion.

Prescreen of the port (zero spill everywhere but 16 B at q6_K width 5, near-threshold):

| Kernel (v1) | w3 text | w4 text | w5 text | Incumbent for scale |
|---|---:|---:|---:|---|
| iq4_nl | 2930 | 3196 | 3688 | iq4_xs w4_v5 3240 / w5 3666 |
| q3_K | 4146 | 4598 | 4974 | q4_K w4_v1 3918 / w5 4402 |
| q6_K | 4174 | 4612 | 5004 (16 B) | q5_K w4_v1 4760 |
| iq3_s | 3572 | 4098 | 4392 | |

Three host gates had to learn the new types, each found by a failing test: the ext small-batch
entry type list (width 2 fell through to the plain mv getter and aborted), the f16 activation
scratch reservation `ggml_metal_mul_mat_use_f16_src1_n` (encode forced f16y for the stored
types while nothing was reserved: 107 sentinel mismatches, the convert wrote over the next
tensor), and `ggml_metal_is_kq_soa_type` (IQ4_NL_SOA was missing, so it took f32 readers and
never reached the new kernels at small shapes). Test additions: a whitelisted 5120-row shape at
widths 2..8 so the kq-form route and the ext readers are exercised without the GGUF.

Correctness (`results/ud-remaining-kq-20260908/`): synthetic 202/202, real-weight slices 62/62,
the existing stored q4_K/q5_K/iq4_xs cases 86/86 unchanged (5e-4 gate). Every one of the 12
kq-form kernels and the 12 ext readers compiled in the test run's own stderr.

### Result

Same harness, same shapes, mirrored `plain-1, soa-1, soa-2, plain-2`, 20:32:58-20:42:09 UTC,
132 timings per arm. Geometric means of the latency change across each format's shapes;
positive = the stored form is faster. Per-shape table: [summary.md](results/ud-remaining-kq-20260908/summary.md).

| Format | w1 (shared body) | w2 (ext SoA) | w3 | w4 | w5 | w6..8 (ext SoA) | w9/32, w512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ4_NL | -18.8% | -23..-34% | +27..30% | **+32..37%** | +34..37% | -20..-32% | flat |
| Q3_K | +23.8% | -3% | +16..18% | **+22..24%** | +18..20% | +6..10% | +10% / +4.7% |
| Q6_K | -9.8% | -27..-44% | +6..13% | **+12..15%** | +38..47% | -3..-16% | +2% / +1% |
| IQ3_S | -33.6% | +45% | +61..63% | **+67..68%** | +69..70% | +54..61% | flat |

Read the widths against the pick: verify runs at 4 (f16 line) and 5 (dflash n4+w5), so the
bold column and w5 are the ones that move decode; w1 is the no-spec anchor; 2 and 6..8 arise
under variable depth and multi-slot only. The Q6_K w5 jump is a native cliff (the ext r1_5
q6_K kernel runs 1.6x its r1_4), not a stored-form virtue. Native IQ3_S has no small-batch
kernel at all (one `kernel_mul_mv_iq3_s_f32` pass per column, linear in width), which is why its
column reads +45..70% everywhere; that is still the incumbent the pick runs today. Widths
9/32/512 are the untouched matrix readers from the morning sweep and repeat within noise.

Spread: seven cases over 5% (six SoA, one plain), the worst q6_K 12288x5120 w5 at 15.6%; its
slower sample is still +35% against native, so no conclusion above turns on a noisy cell.

### Where it leaves the losers

- **Width 1** loses for three formats (-10..-34%) and wins for Q3_K (+24%). The incumbents have
  a dedicated w1 form (the kq body at NC=1 with f32 activations, `SM=2`); the same instantiation
  of `kernel_mul_mv_ud_kq_impl` is the obvious next probe, A/B against the shared body.
- **Width 2 and 6..8** lose for IQ4_NL and Q6_K because the generic `dequantize_soa_mm` q4x4
  reader competes with tuned native ext kernels (IQ4_NL is a t4 type there, nr0 4). A stored
  format is all-or-nothing per tensor, so these widths price the conversion under variable
  depth and multi-slot; under the pick they are not reached.
- ~~**E2e** is unmeasured.~~ Measured the same night, below.

## Width 1 on the kq body, and the converted file end to end (2026-09-08 night)

**Width 1.** The kq body at NC = 1 with f32 activations and f32 products on the exact scale (the
incumbents' `SM=2` class; kernels `kernel_mul_mv_<type>_soa_w1_v1`, 2.1-3.0 KB, zero spill, 26/26
+ 26/26 correctness) against the shared body and native, `perf/run-ud-w1-ab.sh`, mirrored
`plain gen kq kq gen plain`, 12 shapes, [results](results/ud-remaining-w1-20260908/):

| Format | shared body vs native | kq body vs native | kq vs shared |
|---|---:|---:|---:|
| IQ3_S | -35.3% | **+12.3%** | +35.1% |
| IQ4_NL | -16.7% | +1.2% | +15.4% |
| Q3_K | +23.5% | **+31.7%** | +10.7% |
| Q6_K | -9.3% | -5.5% | +3.4% |

The kq body wins every shape (+2..+36%, max spread 4.5%), so it is the width-1 default
(`GGML_MV_UD_W1=0` keeps the shared body for the record). Q6_K width 1 is the one remaining loser
against native, -5.5%.

**Converted file.** `llama-gguf-repack --verify --type iq4_nl --type q3_K --type q6_K --type iq3_s`
on the production `Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf` -> 35 tensors, 1.37 -> 1.42 GiB, every row
reversed byte-identical, three minutes, written to the session scratchpad
(`.../scratchpad/cand/Qwen3.8-27B-UD-Q4_K_M-SOA-V2cand.gguf`, 17.4 GiB, regenerable). The
`output.weight` Q6_K head stays native by the tool's exclusion. Then prod's
`perf/run-ud-soa-gguf-ab.sh` with this tree's binary (ff5e0fe10 + the width-1 default), the UD
f16 pick from `pick.sh`, depth-3 DFlash, fresh processes, mirrored `orig stored stored orig`, the
8K benchprompt, plus the no-spec b1 anchor at 600 ([evidence](results/ud-remaining-kq-20260908/e2e/)):

| Run | orig t/s | stored t/s | delta | acceptance | sha (all four arms) | prompt s |
|---|---:|---:|---:|---:|---|---:|
| depth 3, 600 | 25.35 / 25.40 | 26.32 / 26.11 | **+3.3%** | 60.9% both | `5e76afaba36c` | 63.6-64.0 |
| depth 3, 300 | 24.74 / 24.85 | 25.72 / 25.78 | **+3.8%** | 59.0% both | `73ea53bbe98f` | 63.6-64.0 |
| b1 (no spec), 600 | 13.02 / 12.96 | 13.06 / 13.07 | +0.6% | - | `5e76afaba36c` | 63.0-63.2 |

Every arm's text is byte-identical to the original file's, at both lengths and at b1: the
half-product width 3..5 kernels and the f32 width-1 kernels moved no byte of either trajectory.
The shas are the canonical UD lineage, so this file swap does not mint a new lineage. Prefill is
within noise (the n64/f16-B stored readers, measured -4.7% on Q3_K's 2.3% of the weights). The
resident footprint reads the same within the harness's own scatter (22.3-23.5 GiB deltas on both
arms).

**Attribution.** Before the e2e ran, the isolated per-shape timings times the converted tensor
list predicted the saving per verify round; after it, one profiled run per file
(`GGML_METAL_PROFILE=1`, 211 rounds each, `metalprof-buckets.py`) measured it, serialized decode
GPU ms per round, main model only:

| Format | tensors | predicted saving (w4) | measured orig -> stored ms/round | measured saving |
|---|---:|---:|---|---:|
| IQ3_S | 4 | -2.17 | 3.27 -> 1.10 | **-2.17** |
| IQ4_NL | 7 | -0.82 | 2.35 -> 1.58 | -0.77 |
| Q3_K | 7 | -0.60 | 2.85 -> 2.17 | -0.68 |
| Q6_K (converted part) | 16 | -0.49 | 2.23 -> 1.97 | -0.26 |
| all decode GPU | | -4.08 | 113.71 -> 109.57 | **-4.14 (-3.6%)** |

The isolated repeated-matrix timings predicted the cold-stream e2e within 2% in total and per format
within 0.25 ms; the hot-cache caveat did not bite at these sizes. Half the win is four IQ3_S tensors,
because native IQ3_S has no small-batch kernel (5.7x its byte floor in the original run). The
remaining unconverted q6_K/q5_K/q4_K rows (0.66/0.75/0.05 ms) are the small matrices the conversion
plan excludes. The four stored formats now read 1.1-2.2 ms per round each against byte floors of
0.6-1.2 ms - the same 1.3-1.9x band as the pick's q4_K/q5_K/iq4_xs planes, so no new outlier.

**The standard KLD gate (2026-09-09 00:25-01:45, `run-quant-kld.sh`, UD f16 pick env, this tree's
binary, 24 x 2048 wikitext chunks, reference logits from `/Volumes/offload/kld-references`):** V2 scores
identical to V1 to every printed digit on both references - bf16 as-trained: mean KLD 0.012537 +/- 0.001521,
median 0.002635, 99.0/99.9/max 0.0944/0.850/20.92, same-top 96.575 +/- 0.116%, overlap 96.814%; q8_0:
mean 0.013508, same-top 96.579%, overlap 96.806% - and V1 reproduces its 2026-09-08 q8_0 row exactly.
Expected: this gate runs 2048-token batches, so it scores the four new stored PREFILL readers (exact
dequant into the same tile), not the width 1..5 decode kernels (README rule "the KLD gate is a
prefill-path gate", stated the same night). Logs `kld-{bf16ref,q8ref}-sep08-*-{V1,V2cand}-ud-f16pick-sep09.log`.
(V1's bf16 row moved in the fourth digit vs 2026-09-08 - 0.012541 -> 0.012537, overlap 96.873 -> 96.814 -
while its q8_0 row is identical; the binary is not the variable, the bf16 base file's local copy is the
suspect; not chased.)

## The decode-path KLD: the first time any decode kernel's numerics were priced (2026-09-09 01:00-02:20)

Owner: "the second is really interesting in its own right". Same 24 x 2048 wikitext chunks and the same
bf16 as-trained reference, but the test arm runs `-b 4 -ub 4`: every position's logits come from 512
four-token steps per chunk through the width-4 DECODE kernels (mv, GQA decode FA, GDN decode, in-place
states) instead of one 2048-token prefill batch. Three arms, ~25 min each (`PPL_EXTRA`, this tree's
`run-quant-kld.sh`; the plain twin regenerated with `--reverse --verify`, 339 tensors, 16.5 GB):

| Arm (bf16 reference) | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap (1-TV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill path, V1 = V2, pick env (the standard gate) | 0.012537 +/- 0.00152 | 0.002635 | 0.0944 | 0.850 | 20.92 | 96.575 | 96.814 |
| decode w4, plain twin, CLEAN env (the fork's native kernels) | 0.012689 +/- 0.00153 | 0.002649 | 0.0943 | 0.873 | 21.12 | 96.530 | 96.813 |
| decode w4, V1, pick env (iq4_xs/q4_K/q5_K half-product SoA kernels) | 0.012852 +/- 0.00159 | 0.002639 | 0.0946 | 0.851 | 20.91 | 96.575 | 96.816 |
| decode w4, V2, pick env (+ the four new formats' kernels) | 0.012831 +/- 0.00160 | 0.002641 | 0.0935 | 0.847 | 20.89 | 96.546 | 96.817 |

Reading: the decode path costs +0.00015 (native, clean) to +0.00032 (the pick) of mean KLD on top of the
weights' 0.0125 - about 1/40 of the quantization cost, inside the printed error bar, with the tails, the
median and the overlap unmoved to three digits. Same-top moves by 0.03-0.045 pt = 7-11 of 24,552 positions,
the size of top-2 ties. V2 vs V1 on the decode path: mean -0.00002, 99.9% -0.004, same-top -0.03 pt: the
four new formats' kernels are quality-free at this test's resolution, and so, for the first time on record,
are the pick's own half-product width-4 kernels. What this does NOT yet say: these are all differences of
two noisy numbers against bf16; the kernel cost itself is the PAIRWISE KLD against the byte-identical twin
(the README's rule for a numerics form), which needs a decode-path base of V1 - queued next. Logs
`kld-bf16ref-sep08-*-{plain-twin-clean,SOA-V1-ud-f16pick,SOA-V2cand-ud-f16pick}-dec4.log`. Routing
caveat: `run-quant-kld.sh` runs perplexity at default verbosity, where the pipeline-compile lines
(`GGML_LOG_DEBUG`) are dropped, so these three logs do not name their kernels; batch_size=4 is in each log
and the routes at ne11 = 4 are deterministic in the shapes and the env - the proof runs (`-v`, one plain
chunk per arm) are part of the pairwise chain.

### Pairwise: the kernels' own cost, with the weights' quantization noise removed (02:40-03:56)

A decode-path BASE of V1 under the pick (`REF_EXTRA="-b 4 -ub 4"`, 12.2 GB, kept at
`kvquant-experiments/logits/kld-base-kld-pair-v1dec4-sep09.dat` as the standing reference for pricing any
future decode kernel form), then three arms scored against it, same 24,552 positions:

| Test arm vs V1 decode-path base (pick) | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 decode path, pick (the four new formats' kernels) | **0.000005 +/- 0.000002** | 0.000000 | 0.000042 | 0.000247 | 0.032 | 99.935 | 99.900 |
| V1 PREFILL path, pick (decode kernels vs prefill tiles, same file) | 0.000026 +/- 0.000010 | 0.000001 | 0.000069 | 0.001866 | 0.177 | 99.914 | 99.861 |
| plain twin decode path, CLEAN env (the fork's native width-4 kernels) | 0.000445 +/- 0.000344 | 0.000013 | 0.000570 | 0.011329 | 8.43 | 99.678 | 99.706 |
| scale: q8_0 vs the bf16 model (`kld-bf16-reference.md`) | 0.0012 | | | | | 99.08 | |

Reading, in order of what it settles:
- **The four new formats' kernels cost 5e-6 mean KLD against the pick's existing decode kernels - 1/240 of
  q8_0's own distance from the trained model, argmax identical on 99.935% of positions, 99.9% tail 0.00025.**
  Quality-free by any standard this project has used; adoption is a speed decision.
- **The pick's decode path sits 2.6e-5 from its prefill path** (half-product mv kernels, GQA f16 decode FA,
  GDN decode kernels vs the exact mm tiles and the batched FA) - 1/46 of q8_0's distance. This is the number
  the prefill-only KLD gate had been assuming was zero; it is not zero, and it is small.
- **The fork's native width-4 decode kernels differ from the pick's by 4.5e-4** (17x the decode-vs-prefill
  gap, still 1/3 of q8_0's distance), max 8.4 at one position, same-top 99.68%. The pairwise cannot say which
  side is closer to the model - against bf16 both read 0.0127-0.0129 inside a 0.0015 error bar - but the
  clean arm also swaps the FA/GDN routes (the env, not only the mv kernels), so it prices the whole pick's
  decode numerics against the fork's defaults, not the SoA kernels alone.

Routing proved for all three arms from one-chunk `-v` runs at `-b 4` (`ud-remaining-kld-dec4-routing-proof-*.log`):
V2 under the pick compiled exactly `kernel_mul_mv_{iq4_xs_soa_w4_v5, q4_K_soa_w4_v2, q5_K_soa_w4_v2}` plus
the four `kernel_mul_mv_{iq4_nl,q3_K,q6_K,iq3_s}_soa_w4_v1`, the plain twin under the pick the first three
(runtime repack = V1's numerics), the plain twin under the clean env the native `kernel_mul_mv_ext_*_f16_r1_4`
family (`_f32_` for the q4_K/q5_K/q6_K shapes below the f16y size gate, `kernel_mul_mv_iq3_s_f32` and
`kernel_mul_mv_q6_K_f32` per column) and `kernel_flash_attn_ext_vec_f16`; the pick arms ran the `qt_f16`
batched FA at width 4. The V2 decode arm against bf16 was run twice (the first proof attempt turned into a
full 24-chunk run because the base file fixes the chunk count) and reproduced to every digit: the decode
path is deterministic, one run is one sample.

## Adopted (2026-09-09, owner: "It's another home run for you - pick it")

The converted file is the UD line's model: `/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V2.gguf`
(V1 + the four remaining formats stored, 17.4 GiB, +0.05 GiB over V1). `perf/pick.sh` `PICK_MODEL_UD`,
`run-ud-knobs.sh`, `run-ud-decomp.sh`, `run-ud-turbo4-ab.sh` and `run-ud-soa-gguf-ab.sh` (stored arm) point
at it; `run-ud-remaining-kernels.sh`, `run-ud-w1-ab.sh` and `run-ud-q3k-prefill.sh` stay on V1 because their
native arm needs the native tensors. V1 stays on disk until the mint is recorded (`--reverse --type` on V2
regenerates it). Branch merged to prod (`b3e6e4857`). **Mint TAG `prodpick-sep09-udv2`** (`LINE=ud run-prod-pick.sh`, the
depth-4 pick config, prod binary 06:24): 22.54 / 22.70 at 300 (`73ea53bbe98f`), 23.70 / 23.74 at 600
(`5e76afaba36c`), partial env 20.81, MTP d1 17.76, batch-1 12.76 at 300, prompt eval 64.7 s / 8288 tokens;
all arms on the canonical UD shas. The depth-3 numbers for this file are the A/B above (26.32 / 26.11 at
600). This is the UD line's first canonical row under `run-prod-pick.sh`; earlier UD numbers were taken
with `run-ud-soa-gguf-ab.sh` at depth 3.

**Still open (not gates):** (1) ~~KLD~~ done: prefill-path gate identical, decode-path pairwise 5e-6; (2) the Q6_K width-1 and the IQ4_NL/Q6_K
width-2/6..8 losers, reached under variable depth and multi-slot only; (3) the 32K prefill pair
that the disk killed this morning, now unblocked; (4) the file name and manifest entry for the
pick if it goes in (a file swap is a routing change: prove routes, re-mint).

## Q6_K and the off-pick widths (2026-09-09 05:00-05:40, owner: "let's not get distracted from the q6_k stuff")

Branch `exp/q6k-soa-widths` off prod `a5ddd78a1`. The remaining losers after adoption were width 1 (Q6_K
-5.5%, the shared body's successor on the kq form) and widths 2 and 6..8 on the generic ext SoA reader
(Q6_K -27..-44% / -3..-16%, IQ4_NL -23..-34% / -20..-32%). Three probes, prescreened first, the pick's
width 3..5 kernels byte-identical throughout (3196/4598/4612/4098 at width 4):

1. **Width 1, scale once per pack instead of per element** (the native kernels' form: 8 unscaled f32
   products, then one FMA with the scale and the folded offset). Prescreen text moved by < 30 B; measured
   +0.3..+0.6% on every format (`results/ud-q6k-widths-20260909/w1`, mirrored, max spread 4.6%): inert -
   the compiler was already there. REVERTED; the width-1 kernel is prod's, byte for byte. Q6_K width 1
   stays -5.2% vs `kernel_mul_mv_q6_K_f32` (1.19x vs 1.11x its byte floor); the next lever is a
   per-instruction profile, not another source form.
2. **Width 2 on the kq body** (`kernel_mul_mv_<type>_soa_w2_v1`, NC = 2, K split, 2.6-3.6 KB, zero spill)
   instead of the ext SoA reader.
3. **Widths 6..8 as column groups of the 4-column body** (`*_w4cg_v1`: grid y = ceil(ne11/4), column
   pointers hoisted and clamped to the last valid column, the tail dropped at the store; a separate
   instantiation so the width-4 kernel's loop is untouched; q3_K/q6_K spill 16 B, near-threshold).

Full sweep, same harness and shapes (`results/ud-q6k-widths-20260909/sweep`, 132 timings x 4 mirrored arms,
max spread plain 3.2% / SoA 8.8%; the untouched width 3..5 rows repeat last night's within 0.3%):

| Format | w2 before -> after | w6 | w7 | w8 | w1 (unchanged) |
|---|---:|---:|---:|---:|---:|
| Q6_K | -27..-44% -> **-2..-7%** | +5..9% | +9..12% | +10..12% | -5.2% |
| IQ4_NL | -23..-34% -> **+5..+19%** | +15..16% | +30..31% | +31% | +3.6% |
| Q3_K | -3% -> **+9..11%** | +5..8% | +17..19% | +17..20% | +32% |
| IQ3_S | +45% -> **+48..49%** | +54..56% | +60..62% | +65..67% | +13% |

Every width 2..8 of every format is now at or ahead of native except Q6_K at width 2 (-2..-7%, four of five
shapes; the ext reader it replaces was -27..-44%). The only losers left on the whole map are Q6_K at
width 1 (-5%) and width 2 (-4% geomean). Numerics class: the same half-product kq body as widths 3..5
(pairwise decode-path KLD 5e-6 at width 4); width 2 and 6..8 are outside the pick and are the variable
depth / multi-slot widths. Correctness after the revert: synthetic 202/202, real-weight 62/62, the
`w2_v1` / `w4cg_v1` routes engaged in the test's own stderr. `GGML_MV_UD_KQ_ALL=0` = the ext readers.
Nothing in a pick changes; adoption = owner.

## Why Q6_K loses 5% at width 1: the per-instruction profile (2026-09-09 06:30-07:20, owner: "I don't think 5% on 16 tensors at batch are worth it, but I think the knowledge might be")

`run-ud-soa-profile.sh` on the 17408x5120 width-1 case, native `kernel_mul_mv_q6_K_f32` (nsg 2, nr0 2) beside
`kernel_mul_mv_q6_K_soa_w1_v1` (the kq body at NC = 1), headless replay, `shaderprof-compare.py`
(`kvquant-experiments/profiles/q6k-w1-sep09`; uncaptured timing 287.8 vs 300.0 us, +4.2%):

| | native | stored (kq NC=1) |
|---|---:|---:|
| static instructions / registers / spill | 400 / 51 / 0 | 302 / 69 / 0 |
| executed per dispatch | 19.44M | **16.26M (-16%)** |
| issue / stall share | 95.6 / 4.4% | 97.1 / 2.9% |
| hot loop rows, its issue share | 310, 92.1% | 211, 92.1% |
| hot-loop issue share by encoding class 8 B + 12 B | 55% | **68%** (12 B alone 37%) |
| issue cost per executed instruction (us x issue / M) | 14.2 | **17.9 (+27%)** |

Three facts, in the order they rule things out. **Not memory:** both kernels sit at 96-97% issue with < 5%
stall, so the byte-floor ratios (1.11x / 1.19x) describe nothing - the loop is issue-bound on both sides.
**Not instruction count and not spill:** the stored kernel executes 16% FEWER instructions per dispatch,
spills nothing, and still runs 4% longer - fewer instructions at a higher issue share and slower is the
signature of a fatter instruction MIX, not of more work. **The cost lives in the encoding class:** in both
kernels an 8 B or 12 B instruction carries ~4x the issue time of a 4/6/10 B one (per-instruction issue
share 0.67-0.84 vs 0.17-0.23 within each kernel), and the stored hot loop leans on that class harder - 68%
of its issue against native's 55%, the 12 B class (address arithmetic and load-consumer lowering, the
prescreen skill's fingerprint) alone 37%. The source says why: the kq body computes 16 addresses per
lane-iteration (4 rows x 4 planes: lo uint, hi ushort, int8 scale, half d, each indexed by the pack), where
native advances 10 pointers and reads its bytes at immediate offsets. Native's extra 100 instructions per
iteration are 10 B forms that cost a fifth each; the stored kernel traded them for 4 more fat ones.

**The probe that tests it** (branch `exp/q6k-w1-form`, `kernel_mul_mv_{q6_K,q3_K}_soa_w1_v2`,
`GGML_MV_UD_W1=2`): two packs per lane-iteration with wide loads (uint2 lo / ushort2 hi / one int8 scale
per 16 weights / d per 256) - 16 weights per row per lane-iteration like native, half the addresses per
element. Offline the text doubles (2942 -> 4502 B, 2x the work per iteration; per element 10.5 -> 8.0
instructions, 12 B 1.9 -> 1.6, 14 B 0.72 -> 0.42 - the prescreen cannot rank a form that changes the work
per iteration, only the timing can). Measured, mirrored plain/wide/current, 12 shapes: **q6_K -3.5% ->
-1.1% vs native (+2.4% over the routed kernel), q3_K +0.8%**; 15/15 + 16/16 correctness, v2 named in the
test's stderr. So the address class is about half of the gap; the other half is the remaining mix and the
69-vs-51 register count (fewer resident simdgroups issuing across the core's pipes - the profile cannot see
that directly on this hardware). NOT routed: +2.4% on 16 tensors at batch 1 is ~0.1% e2e and moves the f32
rounding order (per-16 scale), i.e. the b1 sha; the branch keeps the kernel behind the env for the record.

Method lesson (into `skills/metal-gpu-profile`): executed count x issue share does not predict time across
two kernels of different form - read the issue share by encoding SIZE first; the 8/12 B class costs ~4x,
and a kernel can win the instruction count and lose the clock on it.
