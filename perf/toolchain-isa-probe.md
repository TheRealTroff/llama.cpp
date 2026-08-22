# FINDINGS: offline ISA/register pre-screening

Status: **done**, run 2026-08-22 on M4 Pro (20-core GPU), Xcode 26.6 + Metal Toolchain
17.6.109.

**Verdict: the gate as written FAILS, but the underlying goal SUCCEEDS.** There is no
readable AGX disassembly offline. There *is* a usable offline register-spill number, and
it locates the mv-nc cliff exactly. Net: offline pre-screening is alive, and cheap
(8 s for a 7-variant sweep vs ~130 s server start plus a benchmark).

## The gate, step by step

1. `xcrun metal -c ...` - works, 7.8 s. Unchanged.
2. `xcrun metallib` - works.
3. `applegpu-nt` - works, but the recipe in the old doc was wrong in three ways:
   - **arch is `applegpu_g16s`, not `applegpu_g16p`.** Do not guess: `metal-arch` with
     no arguments prints the host arch. `applegpu_g16p` is not even valid for macOS 26
     (the `*p` archs are legacy); the M4 Pro/Max die is the `X`-tier part, `g16s`.
   - needs `-platform_version macos 26.0 26.0`, else "image AIR version (2.8) is bigger
     than the one of the target (2.7)".
   - needs a Metal pipelines script (`-N foo.mtlp-json`), else "cannot lower module with
     unresolved function constants". ggml-metal.metal has 64 of them. Format is
     documented: `man 5 metal-pipelines-script` (page ships in the toolchain, under
     `share/man/man5`, so pass the full path to `man`).
4. `metal-objdump --disassemble` - **FAILS**: "no instruction printer for target
   agx3---macho". Same for agx1 and agx2, i.e. every AGX generation.

So step 4 emits neither AIR bitcode nor readable instructions. What it emits is a real
native AGX Mach-O: outer file is `mach-o 64-bit apple gpu`, its `__compute` section is a
*nested* Mach-O whose `__TEXT` is AGX machine code with register allocation already done,
plus `__GPU_METADATA`, `__GPU_LD_MD`, and an empty `__GPU_STATS_MD`. Symbols are
`_agc.main` and `_agc.main.constant_program`.

## Is there a superset for the disassembler? (asked mid-run)

Yes, the printer exists. No, it is not reachable from the command line.

- `libapplegpu-nt.dylib` contains `LLVMInitializeAGX2InstPrinterTgt` and
  `LLVMInitializeAGX3InstPrinterTgt`. So Apple builds the AGX instruction printer, it is
  just not linked into `metal-objdump` (which registers the agx1/2/3 targets but has no
  AGX mnemonics and no printer). The plugin exports zero `LLVM*` symbols, so the printer
  cannot be driven from outside the dylib.
- `applegpu-nt -S` reaches the emit stage and the plugin refuses:
  "[AGX] Plugin interface not implemented: AIRNTEmitAssembly". Note `_AIRNTEmitAssembly`
  *is* an exported symbol of the plugin, so this is a deliberate stub, not an omission.
- Two genuine supersets exist but neither is CLI-drivable:
  `/System/Library/PrivateFrameworks/AGXCompilerCore.framework` exports the same `AIRNT*`
  API plus newer entry points (`AIRNTInitCompilationContext`,
  `AIRNTEmitPipelineImageWithModuleRef`), but `applegpu-nt -load <it>` registers no
  applegpu archs, so the driver cannot route through it. Xcode's
  `GPUToolsShaderProfiler.framework` has `disassembleBinary:` and `disassembly`
  properties - that is what renders AGX assembly in the Metal debugger - but it needs a
  capture/device session.

Getting text assembly would mean calling hidden symbols by address or driving an
Xcode ObjC framework headlessly. Not worth it: the metric below already answers the
register question.

## What we got instead: per-thread spill bytes

`__GPU_METADATA` is an undocumented FlatBuffer. One field tracks register pressure:
absent when the kernel does not spill (FlatBuffer default 0) and present and growing when
it does. Identified by differential analysis, not from a schema. Confidence is high but
not certain: these kernels allocate zero threadgroup memory, so a nonzero per-thread
allocation that appears only above a pressure threshold can only be spill/scratch.

**Read it by vtable path, not by byte offset.** The field is root field 0, subfield 14
(mirrored at subfield 41). The blob size varies with which fields are present (688, 696,
712, 720 seen), so a fixed offset silently reads the wrong word or reports a false zero.
A first version of the harness hardcoded offset 0xc8 and produced a fake result - a
whole variant family looked spill-free because its blobs were 712 bytes. Do not trust
any number from this metric without a regression check against the known baseline
(nc2/nc3/nc4/nc8 = 0/80/96/368).

Harness: `perf/agx-spill-probe.py` (reusable, takes a metallib, kernel names, and
function-constant values).

## Calibration against the mv-nc cliff: it hits exactly

`kernel_mul_mv_q4_0_f32_nc*`, NR0=4, runtime constants nsg=2 ne12=1 r2=1 r3=1:

| kernel | text bytes | spill bytes/thread |
|--------|-----------:|-------------------:|
| nc2    | 5828       | 0                  |
| nc3    | 7730       | 80                 |
| nc4    | 9382       | 96                 |
| nc5    | 11538      | 192                |
| nc6    | 13730      | 240                |
| nc7    | 15798      | 240                |
| nc8    | 18384      | 368                |

Spilling starts at NC=3. That is exactly where the known, previously unexplained ~112 us
cliff starts. Second sweep, holding NC=2 and varying NR0: 1/2/4/5 spill 0, then 6 -> 48,
7 -> 112, 8 -> 144, 12 -> 352, 16 -> 464. Smooth and monotonic once it starts.

Note that code size alone would NOT have found this - `text` grows smoothly across the
whole range with no discontinuity at NC=3. The spill field is doing the work.

## How to attack the cliff: a candidate that already tests clean

Full (NR0, NC) spill map, 64 kernels from one compile (probe kernels must be inserted
inline next to the other instantiations - appending at EOF produces a metallib that
applegpu-nt rejects with "cannot find private metadata at offset"):

```
baseline, spill bytes/thread          v2, spill bytes/thread
NR0\NC  1  2   3   4   5   6          NR0\NC  1  2   3   4   5   6
   1    .  .   .   .  32  48             1    .  .   .   .   .  48
   2    .  .  16  32  32  48             2    .  . 16  48  32  96
   3    . 16   .  32  80 144             3    .  .   .   .  48  96
   4    .  .  80  96 192 240             4    .  .   .  80 176 128
   5    .  .  96 144 272 368             5    .  . 64  80 208 272
   6    . 48 176 208 320 384             6    .  . 80  80 112 176
```

The working set at NR0=4, NC=3 is roughly: sumf[4][3] = 12 regs, ax[4] and yb[3] as
64-bit device pointers = 14 regs, q[4] as ushort4 = 8, d[4] = 4, yl[16] as half = 8. The
two pointer arrays are 14 registers of pure address bookkeeping, the fattest item that
carries no data.

**v2 replaces both arrays with one base pointer each plus recomputed offsets** (row r at
`ax0 + r*nb01`, column c at `yb0 + col*nb11`, element index `ib*QK4_0 + il` instead of an
advancing per-column pointer). At the shipping nc3 config this takes spill from **80 to 0**
and text from 7730 to 7386. The spill-free frontier moves from 10 outputs/simdgroup
(NR0=5,NC=2) to 12 (NR0=4,NC=3, and also NR0=3,NC=4 and NR0=6,NC=2).

Caveat that still applies: the metric is noisy near threshold. Small values (16-48) and
non-monotonic cells (baseline NR0=3: NC=2 spills 16 but NC=3 spills 0) mean the allocator
and scheduler interact. Treat small numbers as "near the edge", not as precise.

## Benchmarked: the spill was real and cost 18%

v2 is wired into the tree behind `GGML_MV_NC_V2=1` (kernels
`kernel_mul_mv_q4_0_f32_nc{2..8}_v2`, selected in
`ggml_metal_library_get_pipeline_mul_mv_nc`). Correctness first: all 1154 Metal MUL_MAT
cases in `test-backend-ops -o MUL_MAT` pass with `GGML_MV_NC=3 GGML_MV_NC_V2=1`, and the
log confirms the `_v2` pipelines are the ones built.

llama-bench, Qwen3.8-27B uniform Q4_0 (14.32 GiB, dense, 65 layers, no experts), M4 Pro,
r=20.

**Read pp2/pp3 t/s as "tokens amortized over one batched forward pass", not as generation
speed.** The pass streams all 15.4 GB of weights whatever the batch size, so t/s scales
with batch while pass time barely moves. Reference points: tg32 = 14.35 +/- 0.04 t/s, i.e.
69.7 ms/pass and 221 GB/s, which is 81% of the M4 Pro's 273 GB/s peak. Nothing here runs a
27B dense model at 29 tokens/s of generation.

| config                | pp2 t/s        | pp2 ms/pass | pp3 t/s        | pp3 ms/pass |
|-----------------------|---------------:|------------:|---------------:|------------:|
| nc path off (default) | 24.01 +/- 0.34 | 83.3        | 29.84 +/- 0.31 | 100.5       |
| nc on, baseline       | 27.04 +/- 0.36 | 74.0        | 22.39 +/- 0.18 | 134.0       |
| nc on, v2             | 27.44 +/- 0.19 | 72.9        | 26.36 +/- 0.27 | 113.8       |

Per-pass time is the honest unit. At NC=3 the spilling kernel costs 33.5 ms/pass over the
default path; v2 gives 20.2 ms of that back. A pass runs roughly 450 q4_0 matmuls
(65 layers x 7), so the baseline penalty is on the order of 70 us per matmul and v2
recovers about 45 us of it - the same order as the ~112 us cliff figure this file started
from.

The offline metric predicted exactly where the effect would land, and it did:

- **NC=2, neither version spills -> no speed change.** 27.04 to 27.44 is about one sigma.
- **NC=3, baseline spills 80 B/thread and v2 spills none -> +17.7%.** 22.39 to 26.36,
  many sigma apart. The cliff was spilling. The comment claiming it had been fixed was
  wrong in both directions: the spill was still there, and it was expensive.

Two things this also settles:

- **The old "parity, not a win" conclusion was measured on a kernel that was still
  spilling.** It needs revisiting for anything downstream of it.
- **nc3 still should not be routed.** Even spill-free, v2 at NC=3 (26.36) loses to the
  default kernel (29.84), while nc at NC=2 clearly beats it (27.04/27.44 vs 24.01). So
  the practical setting stays `GGML_MV_NC=2`. Removing the spill recovers about half the
  gap to the default path but does not close it, which says the remaining NC=3 deficit is
  something other than register pressure.

## Second result: a source comment in the tree is wrong

ggml-metal.metal:4311 claims the y-cache in half "halves the register footprint, which is
what breaks the NC>=3 spill cliff". Changing only `half yl[16]` to `float yl[16]`
produces **byte-identical** AGX text (same md5) and identical spill at nc3 (80) and nc4
(96). LLVM canonicalizes the fptrunc/fpext round trip, so the declaration changes nothing
in codegen. The cliff is not broken: nc3 still spills 80 bytes/thread. `git log -S` shows
the line arrived in 155914f1, the same commit that added the kernel, so this was never a
before/after measurement.

## Corrections to the two old warnings

- **Warning 1 was wrong.** Function constants ARE specializable offline, via
  `libraries.specialized_functions` + `constant_values` in the pipelines script. So
  `GGML_FA_NQ` is reachable after all, and so is any other function-constant kernel. This
  matters more than the mv-nc result: the FA NQ refutation rested on unmeasured
  register-pressure reasoning ("mqk[NQ][32] = 192 floats is hopeless") and can now be
  measured directly.
- Warning 2 still holds. This is a pre-screening tool, not a perf lever.

## Cost

7.6 s once for metal -> metallib, then **0.12 s per kernel variant**. The full 7-variant
NC sweep is 8.3 s end to end. The `xctrace` fallback is not needed for register work.

## Next lead, if someone wants actual GPR counts

The plugin contains a full AGX3 static performance model (`print-agx3-static-sim-stats`,
"AGX3 Static Performance Model and Simulator") that reports `AvgGPRDynPressure` in 32-bit
GPRs per thread, `MeanOccupancyRequirement`, `IntegralGPRPressure`, and `TotalIssueTime`
in cycles. That is what the empty `__GPU_STATS_MD` segment is for. It is unreachable
today: `-mllvm` only reaches the AIR-level materialize stage (the plugin statically links
its own LLVM with a separate cl::opt registry), and `-mtranslator` is a closed whitelist
that rejects every stats option tried. `AGX3_TEMP_REG_LIMIT` and `AGX3_FLAG_REG_LIMIT`
exist as env vars but are ignored by the offline tool (byte-identical output across the
whole range), so they are read by the in-driver runtime compiler only.

## On-device: capture and counters, pointed at our own kernels (2026-08-22)

The offline path above is static. This section is the device-side complement, and it was
tried on our kernels for the first time on 2026-08-22. Both instruments run headless.

### GPU capture works on ggml, and it is free

`GGML_METAL_CAPTURE_COMPUTE=<n>` is already in the tree
(`ggml-metal-context.m:293`): it captures the n-th `graph_compute` and writes
`/tmp/perf-metal-<pid>.gputrace`. `MTL_CAPTURE_ENABLED=1` must also be set. Then
`perf/gputrace-dump.py` (built for the MLX capture, never before aimed at us) reads it with
no Xcode GUI:

```sh
MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 GGML_MV_NC=2 GGML_MM_SKINNY=5 \
  ./build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p "m=5120,n=4,k=17408,"
DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks \
  ~/play/.venv-convert/bin/python3 perf/gputrace-dump.py /tmp/perf-metal-<pid>.gputrace out.txt
```

**What it gives: structure, not time.** Pipeline identity, buffer bindings and offsets,
dispatch order, and the dispatch geometry, which reads as
`(null)(<enc>, {<threadgroups>}, {<threadsPerThreadgroup>})`. Selector names come out
`(null)` in this build, so read by shape. There is **no** timing, duration or counter field
anywhere in the dump - a `.gputrace` document is a command-stream capture; Xcode derives
counters by replaying it under its profiler, which the headless dumper does not do.

### Metal System Trace records headless, but the default template has no counters

```sh
xcrun xctrace record --template "Metal System Trace" --output /tmp/mst.trace \
  --launch -- ./build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p "..."
xcrun xctrace export --input /tmp/mst.trace --toc
```

The schema list is exactly what this work wants: `gpu-counter-info`, `gpu-counter-value`,
`gpu-shader-profiler-interval`, `gpu-shader-profiler-sample`, `graphics-compiler-spill-events`,
`metal-application-encoders-list`. **The data is not.** On a width-4 `ffn_down` run:

| table | rows | why |
|---|--:|---|
| `gpu-counter-value` | 403958 | all samples of **one** counter |
| `gpu-counter-info` | **1** | and it is `RT Unit Active` - raytracing, useless here |
| `gpu-shader-profiler-sample` | 0 | shader timeline off |
| `metal-application-encoders-list` | 45 | real, and usable |

The recording settings in the TOC say why: `Counter Set: (null)` and
`Shader Timeline: Disabled`. The template selects no counter set, so no ALU/occupancy/
limiter counter is sampled, and no per-line attribution is collected.

### The templates are findable and editable. The counters are still not there.

Xcode ships 32 `.tracetemplate` files; the GPU ones are in
`Instruments.app/Contents/Packages/GPU.instrdst/Contents/Templates/` (Metal System Trace,
Game Performance, Game Performance Overview, Game Memory). They are **Apple binary property
lists**, so `plutil -convert xml1` opens them, and the relevant keys are right there and
unset: `counterprofile`, `counterscounterprofile`, `shaderprofiler`, `countersshaderprofiler`,
`gpuperformancestate`. `xctrace record --template <path>` accepts a file, so a hand-edited
copy is a legitimate route.

Two facts kill the easy version of that plan:

- **Neither stock GPU template helps.** `Game Performance` yields the same single
  `RT Unit Active`; `Game Performance Overview` yields zero counters. There is no
  shipped template to just borrow.
- **The device exposes one counter set, and it is useless for this.** Asked directly
  (`MTLDevice.counterSets`), an M4 Pro on macOS 26.6.2 reports exactly **one** set,
  `timestamp`, containing exactly one counter, `GPUTimestamp`. No ALU utilization, no
  occupancy, no memory limiter. So the **public** Metal counter API cannot answer "why is
  this kernel slow" at all, and `RT Unit Active` is arriving over Instruments' private
  path, not over `MTLCounterSet`.

The templates are also NSKeyedArchiver archives (`CF$UID` indirection into `$objects`), so
editing a value means fixing up object references, not a one-line plist poke.

~~**Corrected assessment.** ... the one-counter-set enumeration means it may be gated at the
driver and not by configuration.~~ **Wrong. Retracted the same day - see below.**

### The counters do exist. All 486 of them.

`MTLDevice.counterSets` returning only `timestamp` says what an **app** may sample itself.
It says nothing about what Instruments collects, which goes over a private path. The proof
was already sitting in the trace: `RT Unit Active` is not in the public set, so the public
set was never the binding constraint. Chasing that string finds the catalogue:

`Instruments.app/Contents/PlugIns/GPUPlugin.xrplugin/Contents/Resources/GPUCounterGraph.plist`

A plain XML plist - no NSKeyedArchiver indirection - defining **486 counters** in 13 groups
(`GPU`, **`Performance Limiters`**, `Memory`, `Compute Kernel`, `Shader Core`, `Texture`,
`Ray Tracing`, and the raster stages) plus 49 `timelineGroups`. Each entry maps a friendly
name to the driver-level `vendorCounters`, e.g.

```
Compute SIMD Groups Inflight per Core:
  counterType    Occupancy
  vendorCounters ["Compute Simdgroups Inflight Per Shader Core"]
  description    average number of simdgroups running concurrently per shader core
```

The ones this investigation has been asking for, by name:

| counter | answers |
|---|---|
| `Compute SIMD Groups Inflight per Core` | **occupancy** - directly tests run 3's nxpsg finding |
| `Top Occupancy Target Influence` | **what is capping occupancy** - registers? threadgroup memory? |
| `ALU Limiter`, `ALU Utilization` | are we compute bound |
| `Address Generation Limiter` / `Utilization` | the addressing-overhead theory, measured |
| `Buffer Read/Write Limiter`, `Buffer L1 Miss Rate` | memory side |
| `Compute Shader Launch Limiter` | launch/dispatch bound |
| `Average Kernel SIMD Group Latency` | latency per simdgroup |

`timelineGroups` includes `Occupancy`, `Occupancy Manager`, `Occupancy Target Influences`,
`Instruction Throughput`, `Shader Launch Limiter`, `Bandwidth`, `ALU`.

**So the templates do serve a purpose** - selecting which profile gets sampled - and the
stock GPU ones simply select none (`Counter Set: (null)`). That is a configuration gap, not
a driver wall. The `counterprofile` / `countersshaderprofiler` keys named above are the
lever, and the group names here are the values they are choosing between.

**Still untested:** actually setting one and getting rows back. That is the next step, and
it is now a well-posed one.

**What is usable today** without any of that: dispatch geometry from the GPU capture,
`GPUTimestamp` for GPU-side timing, and the offline spill probe.

Also checked and not useful: `metal-application-encoders-list` carries only
`Event Type = Encoding`, which is CPU-side encode duration, and its 45 rows do not include
the labelled `MUL_MAT` compute encoders.
