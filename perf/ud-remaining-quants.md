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

Decision: retain the work and evidence on the experiment branch, leave prod and
the UD model untouched, and stop here as requested. On resumption, separate the
Q3_K single-token/matrix-reader candidates from the losing multi-token tile.
Real-model validation, decode-fidelity measurements, and further tuning remain
outside this gate. Full converter integration tests and additional strided-view
coverage also remain before production readiness.
