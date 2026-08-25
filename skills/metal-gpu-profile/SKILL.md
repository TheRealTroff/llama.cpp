---
name: metal-gpu-profile
description: Get per-kernel register counts, spill bytes and instruction mix for a Metal kernel by capturing a GPU trace and replaying it. Use when tuning a Metal kernel and you need measured register pressure or instruction counts, or when a perf claim rests on "the kernel is register/ALU/memory bound".
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
```

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
  `gpuTimelineData`, `encoderInfoData` and `batchIdFilteredCountersData`, none of which
  the script decodes yet.
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
