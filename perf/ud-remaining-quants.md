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
- **E2e** is unmeasured. 35 tensors, 1.37 GiB of the model; a converted file (`llama-gguf-repack
  --type` for the four types, ~17.4 GiB temporary, / has 92 GiB free tonight) through the UD
  pick at 300/600 tokens with the b1 anchor is the next gate, and a KLD gate follows it.
