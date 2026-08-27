---
name: metal-kernel-prescreen
description: Measure register spilling AND rank source-level codegen forms of a Metal kernel offline, without building llama.cpp or running the GPU. Use when tuning Metal kernel shapes (tile sizes, unroll factors, rows/columns per threadgroup), when iterating a kernel's inner-loop source form (indexing, pointer hoisting, operand types), or when a perf claim rests on register-pressure or instruction-count reasoning.
---

# Pre-screen Metal kernels for register spilling

Apple's offline Metal toolchain can translate a `.metallib` to native AGX machine code on
the host. The resulting binary records how many bytes per thread the kernel spills. That
turns "does this shape spill?" into a **0.12 s** question instead of a build plus a
~130 s server start plus a benchmark.

Use this to kill bad kernel shapes cheaply and to check register-pressure claims that
were never measured. Do NOT use it to predict speed - see "What this does not tell you".

Harness: `references/agx-spill-probe.py`, next to this SKILL.md - resolve the path
against this skill's directory (inside the fork it is also `perf/agx-spill-probe.py`,
a symlink). Background and the calibration that validated it:
`perf/toolchain-isa-probe.md` in the fork.

**Validated against ground truth, 2026-08-23.** The spill number comes from an undocumented
FlatBuffer field, so it was worth checking against the compiler itself. For
`kernel_mul_mv_ext_q4_0_f16_r1_4` at `nr0=4`, this probe predicts **32 bytes/thread**, and
the Metal compiler's own `Spilled bytes`, read back via the `metal-gpu-profile` skill, is
**32** - with 0 predicted and 0 reported at `nr0=2`. The field is the real thing. Where that
skill needs an Xcode install and a full GPU replay per run, this stays a 0.12 s offline answer, so prefer it and
escalate only when you need register counts or the instruction mix too.

## Prerequisites

Xcode 26.x with the Metal Toolchain installed (`applegpu-nt` and `metal-arch` live next
to `metal`; find them with `dirname $(xcrun --find metal)`). Step 2's compile command
assumes the working directory is a checkout of the llama.cpp fork; the probe itself
(step 3) runs on any metallib from anywhere.

## Step 1 - Get the host GPU arch. Do not guess it.

```sh
"$(dirname $(xcrun --find metal))/metal-arch"     # e.g. applegpu_g16s
```

Guessing from the marketing name is how this goes wrong: an M4 Pro is `applegpu_g16s`,
not `applegpu_g16p`. The `*p` archs are legacy and are not even valid targets for
macOS 26. `agx-spill-probe.py` calls `metal-arch` for you; pass `--arch` only to
cross-target another chip.

## Step 2 - Build a metallib

```sh
xcrun metal -c ggml/src/ggml-metal/ggml-metal.metal -o /tmp/x.air \
    -I ggml/src/ggml-metal -I ggml/src
xcrun metallib /tmp/x.air -o /tmp/x.metallib
```

About 7.6 s for the whole ggml shader file. This is the only slow step, and you pay it
once per source variant, not once per kernel.

## Step 3 - Probe

```sh
python3 references/agx-spill-probe.py /tmp/x.metallib kernel_mul_mv_q4_0_f32_nc3 \
    --cv 600=2 --cv 602=1 --cv 603=1 --cv 604=1
```

`--cv IDX=VAL` sets a function constant (short-typed). **You must supply every function
constant the kernel reads**, or translation fails with "cannot lower module with
unresolved function constants". Get the indices and the values the runtime actually uses
from the pipeline getter in `ggml-metal-device.cpp` - for mul_mv nc that is
`ggml_metal_library_get_pipeline_mul_mv_nc`, which sets `FC_MUL_MV + 0/2/3/4` (base 600)
to nsg/ne12/r2/r3.

Function constants are specialized offline exactly as the Metal runtime specializes them
at pipeline creation, so the result reflects the real specialized kernel. Kernels behind
function constants are therefore in scope, including flash-attention shapes.

Output is code size (`text`) and `spill` bytes per thread. Zero means no spilling.

## Step 4 - Sweep a shape space in one compile

The fast way to map a design space is to instantiate the whole grid as extra kernels in
one source file, compile once, then probe each in 0.12 s:

```python
for R in range(1,9):
  for C in range(1,9):
    emit(f"kernel void probe_r{R}_c{C}(...) {{ my_impl<{R},{C}>(args, ...); }}")
```

**Insert the probe kernels inline next to the existing instantiations, not appended at
EOF.** Appending at end of file yields a metallib that `applegpu-nt` rejects with
"cannot find private metadata at offset N" for exactly the new functions. Inserting them
after the last real kernel of the same family works.

## Step 5 - Iterate codegen FORMS offline, not just shapes

This is the highest-value use of the pipeline, found 2026-08-27 (the width-4 parity
result, `perf/m4-width4-r4kp.md` in the fork): source-level FORM - how indexing,
pointers and operands are written - moved a kernel 21% where every schedule-level
lever (K-split, unroll, threadgroup packing) had measured +/-3%. The loop:

1. Write candidate variants in a STANDALONE .metal file with plain constant args
   (`constant int & ne00 [[buffer(3)]]` etc). You do not need the project's kargs
   struct to rank codegen - validated: a standalone probe body compiled
   byte-identical (3756 B) to the same body in-tree behind the real struct.
2. Compile + translate each variant (steps 2-3 above), read `text` size and spill.
3. For instruction-level detail, translate to a `.gpubin` with `applegpu-nt`
   (the probe's own `translate()` shows the invocation) and run the fork's
   `perf/agx-disasm.py --json` on it: exact per-instruction offsets, sizes and
   register pressure - no GPU, no mnemonics needed.
4. Compare **encoding-size histograms**, not just counts. On g16s the families are
   a fingerprint: ~6 B = f32 FMA short forms, ~10 B = compact wide-operand
   arithmetic, ~14 B tracks device loads, ~12 B load-consumers/MMA lowering. A hot
   loop flooded with 4/6 B helper ops next to a competitor dominated by 10 B forms
   means fat address/convert codegen, not more intrinsic work.
5. Transplant only the winning form in-tree and benchmark. Static text does NOT
   predict dynamic cost (see below) - the probe RANKS forms; the benchmark decides.

Forms measured to matter on AGX/g16s (each worth re-trying on any slow inner loop):

- **Signed-int indexing + per-row planar pointers hoisted out of the K loop**
  (`sp[block]` / `qp[p]` instead of recomputing `base + f(p)` byte offsets per row
  per iteration): -13% static instructions, **-21% measured time** on a q4_0 mv
  kernel. Per-iteration 64-bit address recomputation was both the instruction fat
  AND the load-consumer stall sites. This beat every schedule-level lever combined.
- **f16 sources fold into FMA operands for free; bf16 does not** (and an explicit
  `float(h)` cast does not block the fold). A scalar convert-per-element loop on
  bf16 cost a competitor kernel +16%.
- **Half-precision products** (`float(a_h * b_h)` accumulated in f32): ~-4 to -7%,
  but ONLY with enough independent accumulator chains - measured at width 5 the same
  form pays -6.7% on a 4-row body and INVERTS to +13.8% on the 2-row body
  (2026-08-28, `perf/m4-width5-crossover.md` in the fork). Do not apply it to
  low-row-count bodies on trend; benchmark the row-count pair. It changes rounding -
  a numerics decision, not a free lever - though where the incumbent route is
  `simdgroup_half8x8` MMA, the incumbent is already half-accumulate.
- Tile shape and K-split across simdgroups: single digits at best (~4.6% and ~1%
  respectively at width 4). Measure them AFTER the form is right.

One caution: the same-compiler assumption holds across frameworks. MLX
`mx.fast.metal_kernel` sources compile through the host's Metal compiler, so probing
a transliteration of a competitor's source form against yours is a valid controlled
comparison (respect any no-copying boundary - probe the FORM, not their code).

## Reading the numbers honestly

- **Always regression-check first.** The spill number comes from an unnamed field in an
  undocumented FlatBuffer. The harness locates it by vtable path (root field 0, subfield
  14) because the blob size varies with which fields are present. An earlier version read
  a fixed byte offset and silently reported false zeros for a whole variant family,
  producing a fake breakthrough. Before trusting a sweep, probe a kernel whose value you
  already know.
- **The metric is noisy near threshold.** Small values (16-48 bytes) sit at the edge and
  cells are not always monotonic - in one measured grid NR0=3 spilled 16 bytes at NC=2
  but zero at NC=3. Allocation and scheduling interact. Treat small numbers as "close to
  the limit", not as precise quantities.
- **Code size is not a substitute.** In the mul_mv nc sweep `text` grew smoothly across
  the whole range with no discontinuity at the shape where spilling starts. Only the
  spill field found it.

## What this does not tell you

- **Not speed.** It is a compile-time register fact. A kernel that stops spilling can
  still benchmark flat, especially if it is memory bound. Always confirm with a real
  benchmark. When it was validated on mul_mv nc the prediction held precisely - no
  measurable change at the shape where neither version spilled, +17.7% at the shape where
  the spill was removed - but the same run also showed the now-faster kernel still losing
  to the default path, so "stopped spilling" is not the same as "worth routing".
- **Not mnemonics.** ~~There is no AGX disassembly.~~ Since 2026-08-26 the fork's
  `perf/agx-disasm.py` decodes a `.gpubin` STRUCTURALLY - exact per-instruction
  offsets, sizes and register pressure (step 5 uses this) - but still no mnemonics:
  `metal-objdump --disassemble` registers the agx targets but ships no instruction
  printer, the translator plugin refuses `AIRNTEmitAssembly`, and the printer inside
  `libapplegpu-nt.dylib` exports no `LLVM*` symbols. Size-family histograms are the
  working substitute for a mnemonic census.
- **Static counts are not dynamic cost.** A 2-row variant with R2-equivalent static
  text measured -21% because the saved instructions were IN the hot loop and attached
  to stall sites; conversely an unroll that cut dynamic instructions 15% measured
  slower because stalls rose. Rank offline, then benchmark, then (if the result
  surprises) attribute per-instruction with the `metal-gpu-profile` skill.
- **Flat registers + better static economy can hide a stall cliff.** Measured
  2026-08-28 (`perf/m4-width5-crossover.md` in the fork): widening a zero-spill mv
  kernel 5->6 columns IMPROVED per-column instruction count and LOWERED the register
  count (80->78), yet wall time rose 40% - the allocator held pressure flat by
  shortening the software-pipelining distance for next-iteration loads, and diffuse
  load-consumer stall went 10.5% -> 22.1%. Single-simdgroup scalar kernels hide
  latency only with register-bounded intra-thread ILP (~3 simdgroups/core inflight is
  a fleet constant), so every added live load stream spends the same slack twice.
  The offline probe CANNOT see this; when live vector streams grow, benchmark and
  check the issue/stall pair before trusting any static ranking.
- **Not GPR counts or occupancy.** The plugin contains an AGX3 static performance model
  that reports `AvgGPRDynPressure` and `MeanOccupancyRequirement` into the (empty)
  `__GPU_STATS_MD` segment, but its options are unreachable: `-mllvm` only reaches the
  AIR-level stage and `-mtranslator` is a closed whitelist. `AGX3_TEMP_REG_LIMIT` is
  ignored by the offline tool (it is read by the in-driver runtime compiler only).

## Persistent-layout correctness is a separate gate

When a kernel consumes a transformed persistent buffer, treat the byte layout as cache
identity. A cache keyed only by source address or tensor can silently return an incompatible
layout when the same weights are used at another batch width. If residency permits only one
copy per tensor, record the cached layout and fall back to the original weights on a mismatch;
never reinterpret the existing buffer and never replace a buffer still referenced by an
unretained command buffer.

Fixed-shape CPU-reference tests cannot expose this class of defect. Add a mixed-width
end-to-end control that reuses one loaded model, requires stable output/acceptance, and selects
an unaffected width in both A/B arms. Treat corrupted text, collapsed speculative acceptance,
or a moving control as a correctness/routing failure before accepting a speed result.

## When a shape spills, what to try

Look for live state that carries no data. Arrays of `device` pointers are the usual
offender: each is 2 GPRs, and a `ptr[NR0]` plus `ptr[NC]` pair can be a dozen registers
of pure address bookkeeping. Replacing them with one base pointer per operand plus
recomputed offsets removed the spill entirely at the mul_mv nc3 shape. Accumulators are
irreducible; addressing usually is not.

For M4 width-4 q4_0 work, screen accumulator banking before building. On `applegpu_g16s`,
the `mul_mv_ext` 2-row x 4-column f16-src1 shape stayed at zero spill with two independent
accumulator banks, while four banks spilled 272 bytes/thread (`nr0=2`, `nxpsg=8`, `nsg=2`).
This is a measured register-allocation boundary, not a speed result. Keep two banks as the
first performance candidate; do not benchmark the four-bank shape unless its live state is
reduced and the probe returns to zero.
