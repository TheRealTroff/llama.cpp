---
name: metal-gpu-profile
description: Get per-kernel register counts, spill bytes, instruction mix and PER-INSTRUCTION issue/stall attribution for a Metal kernel by capturing a GPU trace and replaying it headlessly. Use when tuning a Metal kernel and you need measured register pressure, instruction counts, or per-line stall sites; when a perf claim rests on "the kernel is register/ALU/memory bound"; or when comparing your kernel per-instruction against a third-party (e.g. MLX) kernel, which can be captured standalone without its engine.
---

# Profile a Metal kernel: registers, spill, instruction mix

Three steps. Capture, replay, and parsing are headless.

This gives **measured** per-thread register counts and the full instruction mix.
The `metal-kernel-prescreen` skill answers the narrower "does this shape spill?" offline in
0.12 s with no GPU and no Xcode - **use that first** when spill is the whole question.
Come here when you need to know how close to the limit you are, or what the kernel
actually executes.

The scripts live in `references/` next to this SKILL.md; every `references/` path below
is relative to this skill's directory, so resolve it against wherever this file loaded
from. Step 1 additionally needs a built checkout of the private llama.cpp fork as the
working directory; steps 2 and 3 work from anywhere. Inside the fork the same scripts
are also reachable as `perf/<name>` via symlinks, and the `perf/*.md` probe docs cited
below exist only in the fork.

Related tooling that is NOT in `references/` and was invisible to a fresh session until
2026-08-25 - check these before building anything similar:

- **Standalone kernel dispatch harness**:
  `~/play/rotorquant/turboquant/benchmark_metal.py` (pyobjc; loads a metallib from
  file, builds the pipeline, binds buffers, dispatches, times). Kernels are
  rotor-specific but the skeleton generalizes - no llama.cpp build or routing plumbing
  needed for a kernel A/B.
- **Measured roofline model for the skinny mm family**: `perf/skinny-roofline.py` +
  `perf/ffn-utilization.md` (arith roof 3.48 T MAC/s measured on the same
  `simdgroup_half8x8` primitive; per-shape stream roofs from each shape's own width-1
  call). Note the MXU utilization counters are undefined for gen 16 in the catalogue,
  so MMA occupancy cannot be read directly on M4.

## Step 1 - Capture (headless)

ggml already has capture built in. Both env vars are required:

```sh
MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 \
  ./build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p "m=5120,n=4,k=17408,"
```

`GGML_METAL_CAPTURE_COMPUTE=<n>` captures the n-th `ggml_metal_graph_compute` and writes
`/tmp/perf-metal-<pid>.gputrace`. Use 2, not 1, so warmup is not what you capture. The
path is logged: `ggml_metal_graph_compute: capturing graph in ...`.

Narrow the workload with `-p` first. The capture holds every buffer the graph touched, so
a whole model pass writes gigabytes; one perf case writes ~50 MB.

Synthetic `test-backend-ops` tensors are not model `WEIGHTS`. If the selected kernel depends
on a persistent weight repack, use the branch's test-only repack mode (`GGML_MV_REPACK=2`
for this fork) and reject the capture unless the log names the intended pipeline.

### Capturing a THIRD-PARTY kernel standalone (e.g. an MLX competitor)

Never capture a competitor's whole engine to profile one kernel: a full-cycle capture
of a resident-model engine came out at **17 GB and its replay wrote another 32 GB
before filling the disk** (2026-08-27). Instead drive the one kernel standalone on
synthetic tensors of the real shapes: import their package, monkeypatch whatever
debug/enable gate guards the kernel, build inputs with their own quantizer, set
`MTL_CAPTURE_ENABLED=1`, and wrap the calls in an `MTLCaptureManager` scope. Worked
example: `perf/capture-mlx-verify-kernel.py` in the fork (~600 MB capture, ~1 min,
correctness cross-checked against their stock op). This respects a no-copying
boundary - you run their code as-is and measure it; nothing is transplanted.

Two facts that make the comparison valid and cheap:

- MLX `mx.fast.metal_kernel` compiles through the SAME host Metal compiler as your
  kernels, so per-instruction differences are source-form differences, not compiler
  differences.
- MLX builds a pipeline per `mx.eval` batch, so on their captures `traceCount`
  counts eval batches and aux constant-programs appear once each; normalize per
  dispatch before comparing.

## Step 2 - Replay and profile (headless)

```sh
references/metal-profile-headless.py \
  /tmp/perf-metal-<pid>.gputrace /tmp/profile-output
```

The wrapper prefers Apple's supported `gpudebug` CLI when the selected Xcode provides it.
Apple documents it as scriptable and agent-friendly, with a live local replayer and a
`performance` subtree. Xcode 26.6 does not ship it; this was checked with `xcrun --find`,
the bundle contents, man pages and downloadable-component list. The documentation appears
to describe Xcode 27-era tooling but does not state a minimum version.

When `gpudebug` is absent, the wrapper automatically uses the Xcode 26 DY private-framework
path. This fallback is verified on Xcode 26.6: it launches `GPUToolsReplayService`, drives
the same client-side `DYMTLShaderProfiler` coordinator as Xcode, completes the hardware
passes, and saves a positive APS counter set plus the 20-USC raw streams without Xcode or a
human. Independent replays have produced 39-42 APS records, so the exact count is not an
invariant. Pass `--backend dy` to select it explicitly. `HEADLESS_DY_DIRECT_MESSAGES=1` is
only for reproducing the older, known-incomplete diagnostic path.

The output directory has the established reader contract: `streamData` plus `raw/`.
Treat a positive `APSCounterData` count as coordinator completion; its future subsumes the
separate replay-side raw-file notification used by direct-message diagnostics.

## Step 3 - Read it (headless)

```sh
python3 references/gpuprofiler-stats.py            # newest replay
python3 references/gpuprofiler-stats.py --all      # every field
python3 references/aps-dram-bandwidth.py <output>   # aggregate APS/RDE bandwidth counters
python3 references/aps-usc-values.py --list <output> # raw counters from every USC
python3 perf/shaderprof-table.py <output>/raw       # PER-INSTRUCTION exec counts + issue/stall shares
```

`shaderprof-table.py` (run it with the non-SIP python) is the per-line profile Xcode's
GUI shows, decoded headlessly: per instruction the offset, size, register pressure,
execution count and issue/stall time shares. See `perf/shaderprof-decode.md` for the
decode and `perf/skinny-stall-attribution.md` for a worked analysis.

Reading recipes that carried the width-4 parity investigation (`perf/m4-width4-r4kp.md`,
the fullest worked example - a cross-framework per-instruction diff that found a 21%
kernel win):

- **Normalize per dispatch** (`executed_total / dispatches`) before comparing captures;
  captures repeat ops a shape-dependent number of times.
- **Hot loop = rows with `executed >= 0.9 * max(executed)`.** Sum their `cost` (issue)
  and `cost2` (stall) for the loop's share; histogram their `size` field for the
  codegen fingerprint (6 B ~ f32 FMA short forms, 10 B ~ compact wide-operand
  arithmetic, 14 B ~ device loads, 12 B ~ load-consumers/MMA lowering on g16s).
- **issue share x issue rate, not instruction count or stall alone, predicts time.**
  Measured both failure directions: an unroll cut dynamic instructions 15% and lost
  (stall rose), a sumy variant issued 25% MORE instructions more smoothly and lost.
- A stall share concentrated in 1-2 load-consumer sites usually means per-iteration
  address recomputation feeding the loads - a SOURCE-form fix (see the
  `metal-kernel-prescreen` skill, step 5), not a scheduling fix.

M4 Pro (g16s) readings worth having before forming any hypothesis (all measured, see
the fork's `perf/verify-width-instruction-economy.md` and `instruction-economy-league.md`):
every mv/mm kernel in the measured fleet is issue-bound (64-89% issue share, stall > 50%
never observed); inflight sits at ~3 simdgroups/core regardless of grid size, registers
or family; there is no matrix hardware, `simdgroup_matrix` lowers to FMAs at ~2x the
plain-FMA rate and wins only where its fixed 8-wide tile amortizes (above ~5 columns -
scalar forms win below, measured both sides of the boundary); f16 sources fold into FMA
operands for free, bf16 does not.

For legacy Xcode GUI replay, **start `references/watch-replays.sh` before step 2** so output is
archived out of `/tmp`. The replay output lives in
`/tmp/com.apple.gputools.profiling` and does not survive. On 2026-08-23 a whole session of it
was gone by morning and only eight hand-transcribed fields were left, with `--all` never run.
The same applies to the `.gputrace` itself - move it somewhere durable before you rely on it.

Real output, `mul_mv_ext` at nr0=2 on an M4 Pro:

```
=== kernel_mul_mv_ext_q4_0_f16_r1_4 (pipeline 7) ===
   Temporary register count               73
   Uniform register count                 32
   Spilled bytes                          0
   Instruction count                      453
   ALU instruction count                  399
   FP32 instruction count                 124
   INT32 instruction count                113
   Device load instruction count          8
```

**`Temporary register count` is the per-thread GPR count that sets occupancy on AGX.**
`--all` adds threadgroup atomics, texture ops, `ComputeBufferPrefetch` promotion, the
compiler `Remarks` (unroll and prolog/epilog decisions), and compile timings.

## Where the data lives

The headless wrapper preserves replay output as `<output>/streamData` and `<output>/raw/`:

- `streamData` - `NSKeyedArchiver` plist, `GTMutableShaderProfilerStreamData`. Holds
  `pipelinePerformanceStatistics` (what step 3 reads), plus `shaderProfilerData`,
  `gpuTimelineData`, `encoderInfoData` and `batchIdFilteredCountersData`.
  `shaderProfilerData` is decoded by `perf/shaderprof-table.py` (via the whole `raw/`
  bundle, NOT this file alone - the standalone streamData's copy is empty); the others
  are still unread.
- `Counters_f_*.raw`, `Timeline_f_*.raw`, `Profiling_f_*.raw` - 20 each, undocumented
  binary. `aps-usc-values.py` also accepts stream archives that carry APS_USC bytes inline
  as `ShaderProfilerData` instead of using `APSTraceDataFile` references.

## Historical GUI path and dead ends

Before the DY launch chain was recovered, clicking **Profile GPU Trace** in Xcode was the
only complete path. The material below is retained to prevent repeating dead ends; it is
not the current workflow.

Traced with a process monitor across a real reopen. The replay is driven over **XPC**, not
by a command you can copy:

- `GPUToolsReplayService.xpc` (in `/System/Library/PrivateFrameworks/GPUToolsDeviceServices.framework`)
  starts with **ppid 1** - launchd-spawned, Xcode connects to it by service name.
- The only children Xcode spawns directly are
  `GTLLVMHelper <arch> Host 0 <xcode-pid> 0 /tmp/unixsocketipc_gtd` (a second one follows
  with arch suffix `-b1` and `/tmp/unixsocketipc_test`). Those are shader-compiler helpers.
- **No process anywhere receives the `.gputrace` path in argv.** It travels over XPC.

There **is** a command-line replay tool -
`/System/Library/CoreServices/MTLReplayer.app/Contents/MacOS/MTLReplayer archivePath
[options]`, with `--counters`, `--shader-profiling`, `-collectPipelinePerformanceStatistics`
and more. **It could not be made to work.** Direct exec dies instantly (`exit 137`,
launch-constraint kill); `open -a --args` does launch it and argv arrives intact, but it
then sits at 0.0% CPU for at least 5 minutes writing nothing, because it expects to be
driven over XPC rather than run standalone.

The gate for a client is `com.apple.private.gputools.client`, which `gputoolsserviced`
checks. Note that **Xcode itself has no GPU-tools entitlement** and drives the stack anyway,
via an entitled helper Apple ships inside its own plugin bundle - so the privileged
`agx.performance-spi` sits on the *service*, not on callers. Automating step 2 is therefore
not obviously impossible, just unattacked: it would mean satisfying that client check and
speaking a bespoke 89-message `DYMessage*` protocol over raw `libxpc` (there is no
`NSXPCConnection` interface to bind to). Full detail in `perf/toolchain-isa-probe.md`.

**Two cheaper shortcuts were tried on 2026-08-23 and both failed** - do not retry them,
see `perf/headless-replay-probe.md`:

- The plugin reads `GPUDebugger.ReplayOnOpen` and `GPUDebugger.ProfileOnTraceLoad` (and
  `GPUDebugger.ProfileAfterReplay`, already on). Setting them changes nothing: `open`ing a
  trace with all three true produced **no replayer process and zero profiling files in
  180 s**. The keys are read by the binary but not honoured on the file-open path.
- The plugin also declares a command "Replay GPU Frame Capture", but **that menu item does
  not appear in the UI** on a loaded trace, so there is nothing for AppleScript to click.

The modern GT XPC-proxy route was worked to the end on 2026-08-23 and is **closed**, though
the separate DY guest-app-session route now works.
An unentitled process can load `GPUToolsTransportAgents.framework`, open a
`DYXPCTransport`, and have launchd **spawn the entitled agent for it** - measured, with a
second agent pid appearing next to Xcode's. The replay path is six messages with known
kind values, and the modern object API is `GTMTLReplayServiceXPCProxy -load:`/`-profile:`.
**But `GTLaunchServiceXPCProxy -launchReplayService:error:` is refused instantly for an
unentitled caller.** Full detail is in `perf/headless-replay-probe.md`. The working DY route
instead launches `com.apple.DesktopReplayer` through `DYMTLGuestAppSession`; do not infer
from the failed GT proxy that headless replay is impossible.

## Gotchas

- **Wait for the replay to settle before parsing.** The file count under
  `/tmp/com.apple.gputools.profiling` oscillates while it works - measured
  0, 72, 92, 112, **40**, 112, 132, 112, 132, 152, **60**, 122 over about 5 s. It deletes
  and rewrites, so parsing mid-replay yields partial data. Watch until the count holds
  steady (122 files, ~1.7 GB, for a single-kernel capture).
- **Each replay writes ~1.7 GB to `/tmp`, and it survives quitting Xcode.** Tested
  properly: 366 files across three replays before the quit, 366 after, none removed, and
  the archives still parse. So you can replay several captures, quit, and analyse at
  leisure. Nothing cleans this up - delete `/tmp/com.apple.gputools.profiling` yourself.
  Three replays of one small kernel came to 7.8 GB.
- **Results are reproducible.** Two independent replays of the same capture gave
  byte-identical register counts and instruction mixes, so a surprising number is a real
  finding, not replay noise.
- **Replays accumulate and are keyed by pid**, so several stale directories pile up.
  `gpuprofiler-stats.py` picks the newest by mtime; pass a path to override.
- **Do not read timing from a captured run.** Capture distorts it. Registers, spill and
  instruction counts are compile-time facts and are unaffected; wall-clock is not.
  Take timings from `test-backend-ops perf` instead.
- **Do not use `xctrace` for this.** `--instrument "Metal GPU Counters"` fails with
  `Selected counter profile is not supported on target device` and records zero counters,
  and the stock `Metal System Trace` template samples exactly one counter (`RT Unit
  Active`, raytracing). Replay is the working path on macOS. Details, including what was
  already ruled out, are in `perf/toolchain-isa-probe.md`.
- **`MTLDevice.counterSets` returns only `timestamp`** on this hardware. That is the
  public API and is unrelated to what replay gives you; it is not a reason to stop.
- **Confirm which kernel you captured.** Pipeline names carry the config, e.g.
  `kernel_mul_mv_ext_q4_0_f16_r1_4_nsg=2_nxpsg=8_nr0=2`. Env routing flags change it, so
  set the same ones you benchmark with.

## CPU-side timing: what the GPU profiler CANNOT measure (2026-08-28)

`GGML_METAL_PROFILE=1` creates one encoder per op and inflates CPU encode 6-8x, and
that cost lands on the submit path specifically - **no deflation ratio can correct a
CPU term from a profiled run** (uniform tick-deflation just relabels profiler overhead
as "CPU submit"). This is not hypothetical: the prod round decompositions carried a
"9.4 ms CPU submit, flat across four picks" line for a week that was pure artifact -
the real, unprofiled number was 2.2-2.6 ms (`perf/cpu-round-overhead.md`). The
flatness itself was the tell: profiler encode inflation depends only on node count.

Measure CPU-side costs with these instead, both non-perturbing (canonical sha and
e2e t/s unchanged):

- **`LLAMA_DECODE_PROF=1`** (src/llama-context.cpp): per-context
  apply/reuse/set_inputs/submit/rest split of every small decode, printed every 64
  decodes. Separates target from drafter for free.
- **`GGML_METAL_SUBMIT_PROF=1`** (ggml-metal-context.m): per-graph GPU timeline vs
  the host encode window from MTLCommandBuffer GPUStartTime/GPUEndTime - per ctx,
  windowed every 64 graphs: `sub` (encode wall), `pre` (entry -> first GPU start),
  `gaps` (GPU idle between command buffers), `busy`, `tail`, `exposed`. This is the
  tool that says whether a CPU cost is ON the round or hidden under GPU execution -
  at the prod pick the whole 1.7 ms encode is hidden and only `pre` (~0.9 ms) is
  exposed. First window includes load/prefill warmup; read the later windows.
- Server side, `tools/server/server-context.cpp` spec-prof dump: `loop_gap` /
  `loop_body` prove whether any wall time escapes update_slots (at the pick: none,
  loop_gap 0.001 ms).

Two more traps caught by these tools the day they were built:
- **Count rounds from the run's own counters** (`draft acceptance ... mean len` /
  spec-prof `n =`), never from another run's acceptance rate - a wrong divisor
  manufactured a phantom "11 ms/round untimed" finding for an afternoon.
- **`ggml_metal_get_tensor_async` routes host-visible readbacks through the GPU
  queue** (fresh `newBufferWithBytesNoCopy` + blit command buffer queued behind the
  whole graph): the per-round logits readback cost ~3.4 ms of pure serial latency.
  `GGML_METAL_GET_MEMCPY=1` (branch `cpu-round-overhead`) defers to a plain memcpy
  after the sync wait: +3.3% e2e, byte-identical. When hunting CPU overhead, look
  for work that is QUEUED BEHIND the graph, not just work beside it.
