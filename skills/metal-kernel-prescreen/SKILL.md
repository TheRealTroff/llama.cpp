---
name: metal-kernel-prescreen
description: Measure register spilling in a Metal kernel offline, without building llama.cpp or running the GPU. Use when tuning Metal kernel shapes (tile sizes, unroll factors, rows/columns per threadgroup) or when a perf claim rests on register-pressure reasoning.
---

# Pre-screen Metal kernels for register spilling

Apple's offline Metal toolchain can translate a `.metallib` to native AGX machine code on
the host. The resulting binary records how many bytes per thread the kernel spills. That
turns "does this shape spill?" into a **0.12 s** question instead of a build plus a
~130 s server start plus a benchmark.

Use this to kill bad kernel shapes cheaply and to check register-pressure claims that
were never measured. Do NOT use it to predict speed - see "What this does not tell you".

Harness: `perf/agx-spill-probe.py`. Background and the calibration that validated it:
`perf/toolchain-isa-probe.md`.

**Validated against ground truth, 2026-08-23.** The spill number comes from an undocumented
FlatBuffer field, so it was worth checking against the compiler itself. For
`kernel_mul_mv_ext_q4_0_f16_r1_4` at `nr0=4`, this probe predicts **32 bytes/thread**, and
the Metal compiler's own `Spilled bytes`, read back via `skills/metal-gpu-profile`, is
**32** - with 0 predicted and 0 reported at `nr0=2`. The field is the real thing. Where that
skill needs Xcode and ~1.7 GB per run, this stays a 0.12 s offline answer, so prefer it and
escalate only when you need register counts or the instruction mix too.

## Prerequisites

Xcode 26.x with the Metal Toolchain installed (`applegpu-nt` and `metal-arch` live next
to `metal`; find them with `dirname $(xcrun --find metal)`).

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
python3 perf/agx-spill-probe.py /tmp/x.metallib kernel_mul_mv_q4_0_f32_nc3 \
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
- **Not instructions.** There is no AGX disassembly. `metal-objdump --disassemble`
  registers the agx1/agx2/agx3 targets but ships no instruction printer, and the
  translator plugin refuses `AIRNTEmitAssembly`. The printer exists inside
  `libapplegpu-nt.dylib` but that library exports no `LLVM*` symbols, so it cannot be
  driven from outside.
- **Not GPR counts or occupancy.** The plugin contains an AGX3 static performance model
  that reports `AvgGPRDynPressure` and `MeanOccupancyRequirement` into the (empty)
  `__GPU_STATS_MD` segment, but its options are unreachable: `-mllvm` only reaches the
  AIR-level stage and `-mtranslator` is a closed whitelist. `AGX3_TEMP_REG_LIMIT` is
  ignored by the offline tool (it is read by the in-driver runtime compiler only).

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
