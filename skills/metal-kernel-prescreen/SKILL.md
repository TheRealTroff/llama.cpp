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

Cross-checked again 2026-09-02 on the two families that had been unreachable: skinny SoA
mul_mm probes 0 spill (replay: 0, 52 registers) and the Turbo4 GQA6 flash-attention
probes 0 spill (replay: 0, 60 registers). Instruction COUNTS do not cross-check the same
way - see "What this does not tell you". The same sweep found the f16 flash-attention
kernel spilling **400 B/thread** at its production specialization; replay confirmed 400
exactly, and the fix (K-loop unroll 4 instead of full) is a 5-10% kernel win
(`perf/fa-f16-spill.md`). That is the fourth family the spill field has been validated
on. Probe every production kernel once; a spilling one was hiding in the pick for weeks.

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
    -I ggml/src/ggml-metal -I ggml/src -DTURBO_USE_PAIR_LUT=1
xcrun metallib /tmp/x.air -o /tmp/x.metallib
```

Under Xcode 27 add `-mmacosx-version-min=26.0`: its `metal` emits AIR 2.9 and `applegpu-nt` targets 2.8 ("incompatible
module, module AIR version (2.9) is bigger than the one of the target (2.8)"); the Metal Toolchain is a separate
`xcodebuild -downloadComponent MetalToolchain` (2026-09-18). `-DTURBO_USE_PAIR_LUT=1` mirrors the runtime's preprocessor macro (`ggml-metal-device.m` sets it before
compiling the embedded source): without it the Turbo4 FA forms fail with "use of undeclared identifier
'turbo_pairs_4bit'" and no `.air` is written (2026-09-16, macOS 27 / Xcode 26.6). Check for other macros in
that file's `preprocessorMacros` block when a new define appears there.

About 7.6 s for the whole ggml shader file. This is the only slow step, and you pay it
once per source variant, not once per kernel.

## Step 3 - Probe

```sh
python3 references/agx-spill-probe.py /tmp/x.metallib kernel_mul_mv_q4_0_f32_nc3 \
    --cv 600=2 --cv 602=1 --cv 603=1 --cv 604=1
```

Function-constant options are typed and repeatable:

- `--cv IDX=VAL`: Metal `short` / `int16_t` (`ConstantShort`)
- `--cvi IDX=VAL`: Metal `int` / `int32_t` (`ConstantInt`)
- `--cvb IDX=VAL`: Metal `bool` (`ConstantBool`)

The option must match the Metal declaration; `applegpu-nt` rejects, for example, an i16
value for an int32 function constant. **You must supply every function constant the kernel
reads**, or translation fails with "cannot lower module with unresolved function constants".
Get the declared types from `ggml-metal.metal`, then get the indices and runtime values from
the pipeline getter in `ggml-metal-device.cpp`. For mul_mv nc that is
`ggml_metal_library_get_pipeline_mul_mv_nc`, which sets `FC_MUL_MV + 0/2/3/4` (base 600)
to short-typed nsg/ne12/r2/r3.

Function constants are specialized offline exactly as the Metal runtime specializes them
at pipeline creation, so the result reflects the real specialized kernel. Kernels behind
function constants are therefore in scope, including flash-attention shapes.

Output is code size (`text`), `spill` bytes per thread, and `via`: which applegpu-nt
route produced the binary. Zero spill means no spilling. `via=pkg` is the normal packaged
`.gpubin`; `via=stage` means the packager failed with "cannot find private metadata at
offset N" and the probe recovered the binary from `-stop-after translate` (see the
private-metadata section below - same native bytes). `--keep DIR` saves each native
binary as `DIR/<kernel>.gpubin` for `perf/agx-disasm.py`.

## Step 4 - Sweep a shape space in one compile

The fast way to map a design space is to instantiate the whole grid as extra kernels in
one source file, compile once, then probe each in 0.12 s:

```python
for R in range(1,9):
  for C in range(1,9):
    emit(f"kernel void probe_r{R}_c{C}(...) {{ my_impl<{R},{C}>(args, ...); }}")
```

Probe kernels can go anywhere in the file, including appended at EOF. ~~Appending at
end of file yields a metallib that `applegpu-nt` rejects with "cannot find private
metadata at offset N" for exactly the new functions; insert them inline instead.~~ That
was the packager bug described under "private metadata" below, and the probe now routes
around it (2026-09-02). Appended kernels simply show `via=stage`.

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
- **A tile does not beat an incumbent that already dequantizes once per weight; check
  what the incumbent IS before attributing a loss** (2026-09-04, `perf/ud-model.md` in the
  fork): the q4_0 skinny MMA tile generalized over the generic `dequantize_*` block
  functions (K-quants, iq4_xs) compiled with zero spill and ran 10-30% SLOWER per call.
  The first write-up blamed a per-column `mul_mv` that re-streams the weights 4x; the
  timing invocation's own stderr then showed the incumbent was the ext r1_4 family
  (dequant-once-reuse-per-column, nr0 rows/thread, f16y) - so the tile removed no
  redundant work and only added the threadgroup round trip plus a fatter dequant form.
  Read the pipeline names FIRST, then the fork's prior record for that kernel family
  (`results.md` had the K-quant ext ceiling on file), then explain the number.
- **A 16-entry lookup table belongs in a `constant` array, not in a lane-held register
  read with `simd_shuffle`** (2026-09-05, `perf/ud-model.md` step 6 in the fork): on the
  iq4_xs SoA kernel the constant table folds into the dequant chain (text 3240 B, 1.31x
  byte floor); the shuffle form doubles the text (6.6 KB) AND the time, losing to the
  incumbent it was meant to replace. The offline text size ranked it correctly before any
  GPU run.
- **A 16-entry `constant` table indexed per element is already in its best place; do not move it** (2026-09-18,
  `perf/w8-decomp-sep18.md` lever 2, seven arms on the iq4_xs width-8 tile): deleting the lookup outright is worth 4-9%
  of the call, and every exact replacement lost - the same 16 floats staged in threadgroup memory (flat to +1.8%: bank
  conflicts on a random gather), a byte-indexed 256-entry pair table with HALF the loads (+9..11% as `constant float2`,
  +2..4% as `constant half2`, +1..2% staged in threadgroup: the loss scales with the constant footprint, not the load
  count - form 3 had 32 fewer load instructions and -13% text), and an exact minimax quartic + `rint` (+10..17%: 320
  more 8 B instructions). The skill's "256-entry float2 table staged in threadgroup was -15%" (Turbo4 FA) was a win over
  a 2 KB CONSTANT table, i.e. over the losing form, not over a 64 B one. Prescreen text and 14 B counts ranked the arms
  by load count and got the order WRONG here; the deletion probe first, then time.
- **Decode a shared header once for the tiles that share it** (same day, lever 4): the skinny K-quant tile dequantizes
  two 16-element tiles of ONE superblock per K-step and decoded d/dmin/the 6-bit scale pair twice; a paired reader (one
  header decode, one `uint2` pack load and one `uint` plane load per tile pair) was -6..-8% per call on q4_K and q5_K,
  byte-identical (same expressions per tile), 0 spill, text -4.4/-6.6%. The prescreen ranked this one correctly.
- **Packed wide loads on a 2-byte-aligned block stream pay in the MMA tile and lose in the scalar ext reader** (same
  day, the q6_K lm_head): `sizeof(block_q6_K)` = 210, so upstream reads `ql`/`qh` as 16 ushort loads per tile; eight
  `packed_ushort4` loads per tile PAIR in the skinny tile were -10.7% per call (text -9%, 0 spill), the identical loads in
  the ext r1_4 reader +6% (it already spills 16 B; the wider live ranges cost more than the load count saved) and the
  ext shape knobs (nr0, nsg, nxpsg) found no better point. Prescreen the spill of the incumbent BEFORE porting a
  load-width win between kernel families: a register-bound reader takes stream levers only.
- **Do not write column streams as `half8 v[NC]` arrays indexed under `#pragma unroll`**
  (same day): templating a measured kernel on the column count with an array form changed
  the codegen - width-4 text shrank 3240 -> 2604, the width-5 instantiation ballooned to
  18-20 KB and the q5_K one started spilling 16 B. Explicit named streams (`v0..v4`, extra
  ones guarded by `if (NC > 3)`) reproduce the measured width-4 text byte-for-byte and give
  sane width-3/5 kernels. Text-size identity to the measured kernel is the cheap regression
  check for any template refactor of a tuned body.
  Reproduced 2026-09-08 on a fresh generic `float x[NC][8]` kernel (`exp/ud-remaining-quants`): w4 7.8 KB,
  w3 11.4 KB, w5 23.3 KB, and w6/7/8 spilled 208/240/272 B - the odd widths timed 7-19x slower than native.
  Probe every width of a templated column kernel before timing any of them. The fix was not tuning: porting
  the incumbent kq-SoA body (named streams, hoisted plane pointers, f16 activations) per format probed
  2.9-5.0 KB / 0 spill at every width and measured +12..+68% at width 4 where the array form had lost 2x.
- **A decode kernel's numerics are priced by a PAIRWISE decode-path KLD, not by the standard KLD and not by
  a sha** (2026-09-09, `perf/ud-remaining-quants.md` in the fork): `llama-perplexity --kl-divergence` scores
  2048-token batches, i.e. the prefill tiles - every KLD row before that date never ran an mv kernel. With
  `-b 4 -ub 4` the same positions go through the width-4 decode kernels (~25 min for 24 chunks); a base written
  the same way from the incumbent (`REF_EXTRA="-b 4 -ub 4"` in `run-quant-kld.sh`, kept at
  `kvquant-experiments/logits/kld-base-kld-pair-v1dec4-sep09.dat` for the UD pick) gives the kernel's own cost
  with the weights' quantization noise removed. Scale: q8_0 sits 0.0012 mean KLD from bf16; the four new
  UD-format kernels measured 5e-6 pairwise, the whole pick's decode path 2.6e-5 from its prefill path, the
  fork's native decode kernels 4.5e-4 from the pick's. Prove the arm's routing with a one-chunk `-v` run
  (pipeline-compile lines are DEBUG level and dropped at default verbosity); `--chunks` is ignored when a KL
  base file is given, so run the proof as plain perplexity.
- **Pre-rounding a per-block scale to half in the layout is free at e2e** where the
  product is already half: iq4_xs/q4_K/q5_K half-planar layouts were 7-11% faster per call
  than the exact d*int8 forms and moved no byte of a 600-token trajectory at any depth.
  Still record it as a numerics decision and keep the exact variant routable.
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
- **A translator failure shared by control and candidate is inconclusive.** If a
  known-good kernel and the experimental kernel both fail the same way, do not turn that
  into a spill, register-pressure, or codegen claim about the candidate. Reduce both to
  equivalent standalone probes; if that cannot be done, report the prescreen as
  unavailable and let uncaptured timing plus a survivor-only GPU profile decide.
- **"cannot find private metadata at offset N" is a bug in applegpu-nt's packaging step,
  and the probe routes around it (2026-09-02).** It blocked the skinny mul_mm and Turbo4
  flash-attention families for a week and was misread as a kernel property. Isolated
  with a four-kernel standalone file: specialization succeeds, translation succeeds
  (`-stop-after translate` emits the native Mach-O), only the final package step fails,
  and it fails by the function's position in the metallib's function list, not by
  anything in the kernel (reordering the same four kernels moves the failure with the
  position; in a fresh library only the first two functions package). For kernels that
  package fine, the stage output's native `__TEXT` bytes are byte-identical to the
  packaged `.gpubin`, so `via=stage` numbers are the same numbers. The exact packager
  rule (which original offsets it can map back) is not mapped; the workaround does not
  need it. If a run ever fails on BOTH routes, that is a real error again.

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
- **Offline instruction counts are not replay counts.** The host translator and the
  driver's runtime compiler are different builds. Measured 2026-09-02 on identical
  specializations: skinny SoA mul_mm decodes to 438 instructions offline vs 421 in the
  GPU replay; the Turbo4 GQA6 FA kernel 1,243 vs 992. Spill agreed (0/0) in both. Use
  offline counts to RANK forms against each other in the same run, never to quote a
  kernel's instruction count or compare against a profiled number.
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

## The benchmark arm must prove its own routing

A synthetic A/B is only as real as the kernel each arm actually ran, and env-gated
routes fail SILENTLY - a declined gate falls through to the default kernel and returns a
plausible number. Measured cost of this failure mode (2026-08-28, `perf/shortk-head.md`
in the fork): a `test-backend-ops` arm carrying `GGML_MV_REPACK=1` never engaged the SoA
kernel under test, because REPACK=1 only accepts weight buffers and test tensors are not
weights - the "A/B" compared the fallback against itself, measured FLAT, and a false
"all routes converge" mechanism was coined from it and briefly stood on record. Two
rules:

- In `test-backend-ops` runs, any repack-gated arm needs `GGML_MV_REPACK=2` (the
  test-buffer variant), not the production `=1`.
- Read the compiled pipeline name from the stderr OF THE TIMING INVOCATION ITSELF
  (`compiling pipeline: ... name = kernel_...`). An engagement check done as a separate
  run with "the same" env is exactly how the phantom slipped through; two arms that
  measure identical to the microsecond are a routing alarm, not a convergence finding.

The same lever then failed silently a SECOND way at the e2e level (same file): in-server
the tensor's first repack-eligible call arrived at a width outside the SoA set, cached
the di layout, and every later SoA-width call mismatched and fell back - a flat e2e that
was nearly recorded as "the win is absorbed by concurrent dispatch". The mixed-width
cache-conflict section below is not hypothetical; it fires on any tensor used at widths
on both sides of a layout decision (the lm_head: prompt-final logits at width 1, verify
at width 5). Before believing a flat e2e for a routed kernel, confirm the route engaged
IN THE SERVER RUN - a per-op profile of the target shape, or a temporary log on the
routing decision (note: ggml INFO logs are dropped at llama-server's default verbosity;
use `-lv 5`). Pipeline-compile lines cannot confirm this when the pipeline is shared
with other shapes.

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

## Latency-chain scans: rows per simdgroup, and what the residue is (2026-09-06)

The GDN prefill scan (`perf/gdn-prefill-scan.md` in the fork) is the worked case for a kernel
whose per-token work is a dependent chain (exp, dot, `simd_sum`, fma, dot, `simd_sum`) on a
single row of state per simdgroup: 44 instructions per token at 74% issue / 22% stall, 10x its
byte floor, and the profile's stall sites were the reduction and the `exp`, not loads. The lever
is **NR consecutive rows per simdgroup** with the same expressions per row in the same order
(byte-identical by construction, verified by e2e sha): NR=4 interleaves four chains and shares the
token's loads, 27 instructions per row-token, 93% issue / 6% stall, -44% per call. NR=8 buys 2%
more for 2x the state registers. Three things the follow-ups taught:

- **Prefetching the next token's inputs LOST 8-10%** once the chains were interleaved: nothing
  waited on memory any more, and the second live copy of the inputs cost more than it hid (the
  width-5/6 "every added live load stream spends the slack twice" rule, again).
- **Per-iteration 64-bit pointer advances are 12 B ops at 4-8 issue units each.** With five
  advancing pointers they were 22% of the NR=4 loop. Attribute them cheaply: compile throwaway
  variants (delete the advances; delete the store branch) and diff the loop's size sequence
  (`agx-spill-probe.py --keep` + `agx-disasm.py --json`) - the block that disappears is the
  culprit. The signed-32-bit-offset form that fixed the mv kernels did NOT fix this one (+2.7%):
  the add moved from the loop tail into the load block. What would: strides as function constants
  (load immediates) plus a token loop unrolled by U, so the pointers advance once per U tokens.
- **Instructions that do not scale with the row count are per-token overhead** - the NR=1 vs
  NR=4 profile delta separates per-row from per-token cost without any disassembly.

## FA tile forms that paid, and the two that did not (2026-09-06, `perf/fa-long-context.md`)

- **Read the loop nesting before the profile: a tile reloaded per inner iteration is the cheapest
  win there is.** The transposed-Q FA kernel reloaded all 32 Q^T tiles from threadgroup memory for
  EVERY score tile (an outer `cc` loop over score tiles, Q loads inside it). Making `cc` the inner
  loop so one Q load feeds both score tiles (two accumulators, +2 registers) was -3.4% on the kernel
  with a SMALLER text (9832 vs 10416 B); holding the first 8 tiles in registers on top was -7..-8%
  prefill / -6..-10% decode. Same MMAs in the same k order per tile: byte-identical, sha-gated.
- **A register array indexed in a loop needs the loop fully unrolled, and full unroll is what spills**
  (400 B on this kernel once before, 96 B here). The form that fits is a fully unrolled HEAD over the
  register-held tiles plus the existing partially unrolled tail (`#pragma unroll 4`) over the rest.
  Sweep the head length: 8 tiles 16 B spill and -7%, 12 tiles 32 B and slower than 8, 16 tiles 48 B
  and ~1% better than 8. A 16 B near-threshold spill did not show in the timing; 32 B did.
- **Register-resident accumulators are not automatically a win**: keeping the FA O tiles in registers
  across the KV loop (rescale via `thread_elements`) measured flat to -1% - the threadgroup round trip
  overlapped the MMAs. Refuted and removed; do not rebuild it on trend.
- `simdgroup_matrix::thread_elements()` lane map on M4 Pro (`simdgroup_float8x8`, measured with
  `perf/probe-thread-elements.{metal,swift}` in the fork): both elements of a lane share a row,
  row = ((lane >> 1) & 3) + 4*(lane >> 4), columns 2*(lane & 1) + 4*((lane >> 3) & 1). Measure, do not
  assume - and pyobjc is not installed here; a 30-line Swift host is the way to run a probe kernel.
- Timing sweeps run as bash scripts: `env $E cmd` in the zsh tool shell does not word-split `$E`
  (three silent no-op sweeps in one session, each with plausible numbers and the wrong kernel name).
- **A form that wins at one problem size can invert at another - time the kernel at the extremes of
  the shape range before gating it** (2026-09-06): the FA QR form was -7..-10% at 8K-24K caches, -2% at
  48K and +15..+29% at 96K, where the 8-query threadgroup's full-cache stream makes the kernel
  K/V-stream-bound (5.5 -> 4.8 TFLOPS with no code change) and a 16-48 B per-chunk spill turns into
  DRAM traffic. The route is gated by cache length; the two e2e 96K pairs had shown a 2-3% loss that
  single 20-minute arms could not have separated from noise without the kernel table.
- **Scaling a tile: add simdgroups before adding registers** (2026-09-06, the FA 16-row query tile):
  at 4 simdgroups the doubled per-simdgroup work (4 score accumulators, 4 P tiles) spilled 32 B and
  lost 20-25% everywhere; at 8 simdgroups (one score tile, 4 output tiles per simdgroup) it spilled
  nothing at any register-head length and was -21% at a 96K cache. The per-simdgroup load-per-MMA
  ratio did not improve (K loads halve, Q^T loads double) - what the bigger tile buys is half the
  cache stream per query, so it pays only where that stream is the bound (> ~32K here) and costs 7% at
  8K. Gate by the size that sets the bound; the sweep across 8K/24K/48K/96K is the whole measurement.
- The kernel's shared-memory formula is the first thing to evaluate for a bigger tile: at Q = 16 and
  head 256 it lands on exactly the 32 KB threadgroup limit, which is why the tile was possible at all
  without moving the accumulator to registers.

## Register-direct A tiles on the skinny MMA tile: what the count predicts and what it does not (2026-09-30, `perf/skinny-direct-mma.md`)

The q4_0 SoA skinny tile (32 x 8 x 64, 2 simdgroups) had its A stage (dequant -> threadgroup `sa` -> barrier ->
`simdgroup_load`) replaced by dequantizing each 8x8 K-tile straight into `simdgroup_half8x8::thread_elements()` by the
measured lane map. Twenty-four standalone forms prescreened in one compile, all 0 spill; seven timed. Lessons:

- **The direct `half2` dequant costs the same per ELEMENT as the staged `half4` one** (4 instructions/element: two
  variable shifts, two masks, convert, arithmetic). Deleting the stores, loads and barriers alone was -22 of 367
  instructions per K-step. `fma(half2(nib), d, -8d)` (exact: -8d is a power-of-two scale, the fma rounds the exact
  (nib-8)*d once) folded the subtract and convert and made the pair 6 instead of 8. Every other spelling of the
  extraction (`extract_bits`, exponent-bias `0x6400 | nib`, 16-bit shift vectors, the `* 0x1001` trick) compiled to the
  same shifts and masks or worse - the compiler canonicalizes; 4 extraction instructions per pair is the floor from
  source.
- **Fewer instructions, slower: -46 static per K-step, +3..+7% per call** for the fma form while the B stage stayed
  staged (two barriers per K-step). The direct kernel issues its pack loads after the barrier and nothing hoists a device
  load across `threadgroup_barrier`; the staged kernel had prefetched the next slice's words before its MMAs. The only
  form that paid was the fully direct one (B read straight from `src1` into the B tile too: no threadgroup memory in the
  loop, no barriers): -5.5 / -3.4 / -1.5% on the three verify shapes at width 8, reproduced four times, from a 30%
  smaller hot loop (228 -> 159 rows). The profile explains the ratio: issue 82.6 -> 86.0, stall 17.4 -> 14.0, the load
  stall gone (5.1 -> 0.4) and the residual on the MMAs waiting for their operands (4.9 -> 8.5). Prescreen counts rank
  the A-side forms; the barrier structure decides whether the saving shows, and only the timing pair sees it.
- **Source order around `simdgroup_multiply_accumulate` is not a lever**: dequantizing 8 or 16 tiles ahead of their MMAs
  compiled to a byte-identical instruction stream (or a register renaming) of the per-tile form. Check with
  `agx-disasm.py --json` before timing a "live tiles" or "software pipelining" spelling - the scheduler already did it.
- **Explicit next-block prefetch in the direct form: +40 instructions per K-step and +5..+20% per call.** The compiler's
  own placement of the plain loads beats the staged kernel's prefetch pattern once the barriers are gone; the doubled
  live word set costs more than it hides. (The same rule as the GDN scan's prefetch loss.)
- **A 1 KB `constant half2[256]` pair table by runtime byte lost again** (+6..+13% here, vs +2..+4% on the iq4_xs tile
  2026-09-18), with the fewest instructions of every form (3-4 per pair) - the third time the constant-footprint rule
  beat the count; a threadgroup copy of the table lost the same way. Stop proposing byte-indexed pair tables for the
  skinny tile.
- **Geometry: one simdgroup per threadgroup owning 4 row tiles (15 instructions per MMA instead of 21, 0 spill) was
  +10..+28%; four simdgroups per 64-row threadgroup timed the same as two per 32.** Fewer resident simdgroups lose;
  more do not win. `dispatch_threadgroups` must take `nr0`/`nsg` from the pipeline for such arms, and a getter that
  changes the threadgroup memory layout (a double-buffered `sb`) must size `smem` for it - the fixture passed with the
  second buffer past the allocation, the compact `m=256,k=512` test shapes caught it. Run the small shapes too.
- Harness traps that bit again: `env $E cmd` under zsh tested the staged kernel seven times with plausible output (the
  gate script is bash now, `perf/run-skinny-direct-gate.sh`); `-p "m=[0-9]+,n=[678],"` matched five of the built-in
  eval cases because the width-6..8 projection shapes live in the perf list - the 18-case fixture
  `perf/skinny-soa-real-projections.ops` is the coverage.

## Quantized K/V tiles dequantized straight into the simdgroup matrix (2026-09-06, `perf/ud-model.md` step 16 B)

- **The lane map holds for `simdgroup_half8x8` too, in both directions** (`perf/probe-thread-elements-half.*`
  in the fork: load -> `thread_elements()` matches the f32 map; write via `thread_elements()` -> `simdgroup_store`
  round-trips exactly). Each lane owns two ADJACENT columns of one row, so any format whose byte holds two
  adjacent values (Turbo4 nibble pairs) dequantizes into the matrix registers with one byte load + one table
  lookup + the arithmetic per tile per lane - no threadgroup scratch, no barrier, no transposed load. The
  "quantized K/V branch, not optimized yet" of the batched FA kernel was doing scratch + barrier + transposed
  loads + an O round trip per key tile: same static size as the f16 kernel, 3.2x its dynamic instructions.
  The register form is -38..-45% per call and byte-identical (same arithmetic, same k order).
- **A 32 B spill can be free and an unroll not**: the register-resident V loop at unroll 2 spilled 32 B and
  beat the zero-spill unroll 1 by 5-10%; unroll 4 (16 B) another 3-5%; unroll 8 (64-112 B) not tried on the
  GPU. Prescreen ranks the spill, the timing pair decides - always time the unroll pair around the threshold.
- **A constant-memory table indexed by a runtime byte is a device load per lookup; staging it in threadgroup
  memory once per threadgroup** (2 KB for a 256-entry float2 table, in scratch the kernel no longer used) was
  -15% on the prefill form and -2..-4% at decode. A half table with one packed multiply saved only 1-4% more
  and changed the numerics - refuted; check the table's values are half-exact before assuming a half form is
  free (the Turbo4 centroids are not).
- `#pragma clang loop unroll_count(N)` accepts a template parameter, so unroll factors can be routed as
  kernel variants (one metallib, A/B by pipeline name) instead of rebuilt.
- **Fold a per-row scale out of a per-element dequant into the accumulator** (2026-09-06, `perf/ud-model.md`
  step 16 C): the Turbo4 block norm multiplied every dequantized element (2 FMUL + 2 cvt per tile per lane);
  applying it once per key tile to the score tile instead (one accumulator per block on the K side, the P
  tile's key columns on the V side) was another -11..-14% per call. It changes the rounding (the scale lands
  in float instead of in the half operand) - a numerics decision, KLD-priced a wash against q8_0 (since 2026-09-08 a bf16 as-trained reference exists, `perf/kld-bf16-reference.md`: q8_0 sits 0.0012 mean KLD from the model and the 99.9% column moves 5-17% between references, so a numerics form's tail claim needs the bf16 file AND the pairwise KLD against its byte-identical twin). Loop per block so the
  accumulator index is a compile-time constant: the dynamic register-array index spilled 1280 B.
- **Fewer, wider loads beat lane-cooperative shuffles for a per-lane byte stream** (2026-09-07, `perf/ud-model.md`
  step 16 D): replacing 8 single-byte loads per chunk with four 2-byte-aligned 8-byte loads (`packed_ushort4`)
  plus a shift/mask per byte was -6..-9% per call; loading one slice per lane and fetching each byte from
  its holder by `simd_shuffle` was +15..+25%. The offline text size ranked them the same way (14.4 KB vs
  19.8 KB against a 16.0 KB base) - a shuffle-heavy form ballooning the text is the tell, as with the
  iq4_xs table. Vector loads need their natural alignment in device memory (`ushort4` = 8, `uint4` = 16);
  the `packed_` types are how a 2-byte-aligned stream takes a wide load.
