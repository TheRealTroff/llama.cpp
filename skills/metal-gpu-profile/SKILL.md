---
name: metal-gpu-profile
description: Get per-kernel register counts, spill bytes and instruction mix for a Metal kernel by capturing a GPU trace and replaying it. Use when tuning a Metal kernel and you need measured register pressure or instruction counts, or when a perf claim rests on "the kernel is register/ALU/memory bound".
---

# Profile a Metal kernel: registers, spill, instruction mix

Three steps. Capture is headless, replay needs Xcode once, parsing is headless.

This gives **measured** per-thread register counts and the full instruction mix.
`skills/metal-kernel-prescreen` answers the narrower "does this shape spill?" offline in
0.12 s with no GPU and no Xcode - **use that first** when spill is the whole question.
Come here when you need to know how close to the limit you are, or what the kernel
actually executes.

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

## Step 2 - Replay (Xcode, once per trace)

```sh
open /tmp/perf-metal-<pid>.gputrace
```

~~Xcode replays it and populates Shaders, Counters, Cost Graph and Heat Map. Nothing
further is needed in the GUI; the data is on disk from this point.~~ **Corrected
2026-08-23: `open` alone does nothing.** Measured - a trace left open and untouched for
180 s spawned no replayer and wrote zero profiling files, with Xcode idle at ~1.4% CPU.

You must **click "Profile GPU Trace"** in the loaded trace. That click is the replay, and
the replay - not the capture - is what produces the statistics. It populates **Shaders**,
**Counters**, **Cost Graph** and **Heat Map**; nothing further is needed in the GUI after
it, and the data is on disk from that point.

This click is the one manual step in the workflow and the reason it cannot be run from
away from the machine. `perf/headless-replay-probe.md` has what has been tried.

## Step 3 - Read it (headless)

```sh
python3 perf/gpuprofiler-stats.py            # newest replay
python3 perf/gpuprofiler-stats.py --all      # every field
```

**Start `perf/watch-replays.sh` before step 2 and this happens by itself**, archived out of
`/tmp` and dumped both ways. Do it: the replay output lives in
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

Replay writes to `/tmp/com.apple.gputools.profiling/<trace>_stream.gpuprofiler_raw/`:

- `streamData` - `NSKeyedArchiver` plist, `GTMutableShaderProfilerStreamData`. Holds
  `pipelinePerformanceStatistics` (what step 3 reads), plus `shaderProfilerData`,
  `gpuTimelineData`, `encoderInfoData` and `batchIdFilteredCountersData`, none of which
  the script decodes yet.
- `Counters_f_*.raw`, `Timeline_f_*.raw`, `Profiling_f_*.raw` - 20 each, undocumented
  binary.

## Why step 2 needs the GUI

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

The XPC route was worked to the end on 2026-08-23 and is **closed**, though not for the
reason the section above gives.
An unentitled process can load `GPUToolsTransportAgents.framework`, open a
`DYXPCTransport`, and have launchd **spawn the entitled agent for it** - measured, with a
second agent pid appearing next to Xcode's. The replay path is six messages with known
kind values, and the modern object API is `GTMTLReplayServiceXPCProxy -load:`/`-profile:`.
**But `GTLaunchServiceXPCProxy -launchReplayService:error:` is refused instantly for an
unentitled caller**, so the replay service never starts. Full detail, and the six things
ruled out before calling it a refusal, in `perf/headless-replay-probe.md`. The remaining
route is accessibility: click the button by AX title, after a one-time Accessibility grant
to the terminal app.

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
