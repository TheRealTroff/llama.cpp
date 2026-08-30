# FINDINGS: controlling native AGX layout below Metal source

Status: **bounded experiment complete**, run 2026-08-30 on M4 Pro (20-core GPU),
Metal Toolchain 17.6.109. The experiment branch is `exp/air-layout-control`.

**Verdict:** AIR is a practical intermediate representation for controlling loop shape
and coarse block placement, and a GPU-specific `MTLBinaryArchive` can make ggml select
an exact `applegpu-nt` translation. It is not an exposed assembly language: the private
AGX backend still owns instruction selection and within-block scheduling. None of the
below-source variants beat the compiler's fully-unrolled width-5 baseline.

The measured baseline is `kernel_mul_mv_q4_0_soa_w5_r4h` at the real `ffn_down` shape
`m=5120,n=5,k=17408`. Runtime settings selected test-buffer repack mode and the width-5
SoA route:

```sh
GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MV_REPACK=2 \
GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1
```

## How much control exists

| Layer | Control demonstrated | Limit | Measured result |
|---|---|---|---|
| AIR text | Lossless round trip | LLVM-like AIR, not AGX assembly | Exact native instruction stream |
| Loop layout | Explicit full/partial unroll | Backend still lowers each body | Full unroll wins by at least 1.96x |
| Block placement | CFG split plus empty side-effect barrier perturbs order | No arbitrary within-block schedule | No repeatable win |
| Register allocation | Private byte budget forces the threshold and spills | Cannot choose individual physical registers | 80-register spill-free baseline wins |
| Native selection | Exact translated object selected through a binary archive | GPU/OS-specific and non-portable | Verified with fail-on-miss |
| Replay profiling | None for archive-forced native code | Replay recompiles the AIR | Must not attribute replay stats to archive variant |

This is enough control to answer layout hypotheses experimentally. It is not enough to
hand-schedule individual AGX instructions or assign registers as one could in a public
GPU assembly language.

## Reproducible AIR round trip

The compiler needs a writable module cache in this environment:

```sh
xcrun metal -fmodules-cache-path=/tmp/air-layout-module-cache -c \
  ggml/src/ggml-metal/ggml-metal.metal \
  -Iggml/src/ggml-metal -o /tmp/air-layout-base.air
xcrun metallib /tmp/air-layout-base.air -o /tmp/air-layout-base.metallib

air-opt -S /tmp/air-layout-base.air -o /tmp/air-layout-base.ll
air-as /tmp/air-layout-base.ll -o /tmp/air-layout-roundtrip.air
xcrun metallib /tmp/air-layout-roundtrip.air \
  -o /tmp/air-layout-roundtrip.metallib
```

`air-opt -S` produced editable LLVM IR. Reassembling it unchanged made a byte-identical
metallib. The outer native wrapper emitted by `applegpu-nt` had metadata differences,
but the decoded kernel stream was identical:

| metric | base | text round trip |
|---|---:|---:|
| native text | 3690 B | 3690 B |
| structurally decoded instructions | 488 | 488 |
| spill bytes/thread | 0 | 0 |
| decoded stream SHA-256 | `9dd33630a19e6d53…` | `9dd33630a19e6d53…` |

That no-op control is mandatory. Without it, an apparent AIR effect could be a wrapper
or relinking artifact.

## Loop layout: strong control, compiler baseline wins

The kernel has four hot eight-iteration inner loops. Replacing
`llvm.loop.unroll.enable` with explicit partial-unroll counts changed final native code
substantially. `full` and `count8` reproduced the baseline exactly.

| unroll | native text | decoded instructions | spill B/thread | runtime, µs |
|---:|---:|---:|---:|---:|
| 2 | 4050 | 468 | 0 | 813.05 |
| 3 | 5604 | 636 | 0 | 927.34 |
| 4 | 4532 | 532 | 0 | 534.31 |
| 5 | 5500 | 636 | 0 | 611.28 |
| 6 | 5588 | 652 | 0 | 545.68 |
| 7 | 6114 | 712 | 0 | 514.74 |
| 8 | 3690 | 488 | 0 | 260.34 |
| compiler full-unroll baseline | 3690 | 488 | 0 | 262.43 |

This is also a warning against static instruction-count reasoning: count 2 has fewer
decoded instructions than the baseline and is more than three times slower.

## Block placement: steerable, not hand-schedulable

Splitting selected row-loop CFG edges with empty LLVM side-effect inline assembly
produced different native stream hashes. The barriers did not add an obvious emitted
instruction, so this is a genuine coarse layout/scheduling perturbation.

The most promising single run was fence 3 at 260.32/261.36 µs versus a 262.32/262.69 µs
baseline. A balanced combination sweep rejected the apparent win:

| variant | run 1, µs | run 2, µs |
|---|---:|---:|
| baseline | 261.16 | 260.45 |
| fence 3 | 263.93 | 263.11 |
| fences 1+3 | 261.23 | 263.16 |
| fences 2+3 | 263.25 | 263.65 |
| fences 1+2+3 | 265.04 | 263.53 |

AIR can therefore influence final block layout, but the native backend retains the
fine schedule. This perturbation is useful for testing a scheduling hypothesis, not for
expressing an exact schedule.

## Register layout: exact budget threshold, no occupancy rescue

The private translator environment variable `AGC_TEMP_REGS_IN_BYTES` changes the native
allocation budget. It located the spill-free threshold precisely:

| budget | native text | spill B/thread | relation to baseline |
|---:|---:|---:|---:|
| 192 B | 4018 | 144 | +15.8% runtime |
| 256 B | 3746 | 80 | +0.9% runtime |
| 320 B | 3690 | 0 | exact baseline |
| 384 B | 3690 | 0 | exact baseline |
| 512 B | 3690 | 0 | exact baseline |

At four bytes per register, the threshold is 80 registers, agreeing with the independent
GPU profile. Lowering the budget is real and deployable, but spill cost exceeds any
occupancy benefit for this kernel.

Balanced runtime with exact archive selection was:

| archive | run 1, µs | run 2, µs |
|---|---:|---:|
| default | 263.07 | 262.42 |
| 256-byte budget | 264.79 | 265.50 |
| 192-byte budget | 304.11 | 304.18 |

## Generic scheduler controls do not reach the backend

The following controls either produced the exact baseline native stream or were rejected:

- `applegpu-nt -mllvm -misched=ilpmax`: accepted, inert;
- `applegpu-nt -mllvm -pre-RA-sched=source`: accepted, inert;
- `applegpu-nt -disable-optimizations`: accepted, inert for already-optimized AIR;
- `applegpu-nt -mtranslator -misched=ilpmax`: unsupported translator option;
- `AGX_FMA_SHFF_HOIST_DEPTH=0,1,2,4,8,16`: inert for this kernel; and
- `AGC_DISABLE_OPTIMIZATIONS=1`: inert for this kernel.

An accepted flag is not evidence of control. Only a changed decoded native stream counts.

## Exact native selection with `MTLBinaryArchive`

Direct `metal-tt` output is a GPU-specific binary archive Mach-O, not an `MTLLibrary`.
Loading it as `default.metallib` correctly fails with `Invalid library file`.

The experiment adds opt-in support for `GGML_METAL_BINARY_ARCHIVE`. ggml loads its normal
AIR library, attaches the archive to `MTLComputePipelineDescriptor.binaryArchives`, and
uses `MTLPipelineOptionFailOnBinaryArchiveMiss`. The tested archive included:

- `kernel_repack_q4_0_soa`;
- `kernel_cpy_f32_f16`; and
- `kernel_mul_mv_q4_0_soa_w5_r4h`.

The benchmark succeeds with fail-on-miss enabled, proving that all selected live pipelines
came from the archive. Extracted target objects independently show the expected code and
spill differences. This is the strongest practical control available below MSL: generate
the native object with a chosen translator configuration and require the runtime to use it.

## Important: trace replay recompiles the AIR

A GPU trace captured from the 256-byte archive-selected run replayed successfully, but
the profiler reported the default 488-instruction, spill-free layout instead of the
archive's 527-instruction, 80-byte-spill object. Replay reconstructed the pipeline from
the ordinary IR library rather than preserving the binary archive selection.

Consequences:

- uncaptured runtime timing plus fail-on-miss proves live archive selection;
- extracted archive objects provide offline instruction and spill properties; but
- replayed register, instruction-mix, issue, and stall data describe the default AIR
  compilation unless a matching native stream is independently demonstrated.

Do not claim per-instruction hardware attribution for an archive-forced variant from this
replay path.

## Outcome

The compiler's fully-unrolled, spill-free width-5 kernel remains the winner at about
262.4 µs. Going below Metal exposed useful controls and made exact native selection
possible, but did not uncover a faster layout in this pass. The experiment should remain
opt-in and branch-local unless a future AIR or translator variant wins end to end.
