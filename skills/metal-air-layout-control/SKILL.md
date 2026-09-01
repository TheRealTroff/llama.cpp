---
name: metal-air-layout-control
description: "Inspect and control Apple GPU native layout below Metal source by round-tripping AIR, translating AGX binaries, and selecting exact code with MTLBinaryArchive. Use after source-level Metal tuning stalls and a kernel needs measured loop, block-order, or register-allocation experiments; do not use native code size or replayed archive binaries as speed proof."
---

# Metal AIR layout control

Use this only after a representative source-level kernel and route have been established with the normal Metal benchmark and profile skills. AIR work is a scalpel: isolate one kernel, change one lowering property, prove that the native instruction stream changed, and benchmark the real route.

Read [references/workflow.md](references/workflow.md) before starting. It contains the commands, control matrix, archive deployment path, and the replay caveat.

## Required workflow

1. Work on a dedicated experiment branch and linked worktree. Do not switch the ground-truth checkout.
2. Record the Metal toolchain version, GPU architecture, source kernel, runtime route, shape, and uncaptured timing.
3. Compile MSL to AIR and create a textual AIR baseline with `air-opt -S`.
4. Reassemble the untouched text with `air-as`, link it, translate both versions with `applegpu-nt`, and require identical decoded native instruction streams. A byte-identical metallib alone is not enough.
5. Make one bounded AIR or translator change. Hash the structurally decoded instruction stream to distinguish a live control from an accepted-but-inert option.
6. Reject spilling variants unless a real benchmark shows an occupancy win. Report text bytes, decoded instruction count, and spill bytes, but never infer speed from them.
7. Benchmark balanced pairs through the actual ggml route. Include route proof and real-model checks in proportion to the claim.
8. If selecting a `metal-tt` binary archive, enable `MTLPipelineOptionFailOnBinaryArchiveMiss`. The archive must contain every pipeline touched by the benchmark, including copies and repacks.
9. Treat archive-selected timing as authoritative only for an uncaptured run. GPU trace replay may recompile the accompanying AIR and profile different native code.
10. Preserve negative findings. A mapped control boundary prevents later work from repeating inert compiler flags.

## Claim boundaries

- AIR loop metadata can strongly control unrolling and code shape.
- CFG boundaries and side-effect barriers can perturb block placement, but do not provide arbitrary within-block instruction scheduling.
- Private register-budget controls can force lower allocation and spills. Their availability and semantics are toolchain-specific.
- Generic LLVM scheduling flags may parse yet be inert because the AGX translator is statically linked and owns final scheduling.
- `metal-tt` output is a GPU- and OS-specific `MTLBinaryArchive`, not an `MTLLibrary`.
- Exact machine-code deployment is possible through a binary archive, but portability and replay-based profiling are weaker than for ordinary metallibs.
- **Known gap (2026-09-01):** `applegpu-nt` currently fails with `cannot find private metadata ...` on the flash-attention and skinny mul_mm kernels (seen in `perf/skinny-soa.md` and the Turbo4 Q16 probe, `perf/turbo4-fa-gqa-reuse.md`), while it still translates the mul_mv kernels. Step 4 of the workflow is therefore unavailable for those families until the failure is isolated. Do not substitute a translator failure for a spill or layout claim; see the matching rule in `metal-kernel-prescreen`. The findings this skill summarizes are in `perf/air-layout-control.md`.

Stop if the experiment has no repeatable end-to-end win. Keep the branch and report; do not propose a production change solely because the native layout differs.
