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

Xcode replays it and populates **Shaders**, **Counters**, **Cost Graph** and **Heat Map**.
The replay - not the capture - is what produces the statistics. Nothing further is needed
in the GUI; the data is on disk from this point.

## Step 3 - Read it (headless)

```sh
python3 perf/gpuprofiler-stats.py            # newest replay
python3 perf/gpuprofiler-stats.py --all      # every field
```

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

## Gotchas

- **Each replay writes ~900 MB to `/tmp`.** It is not cleaned up. Delete
  `/tmp/com.apple.gputools.profiling` when done.
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
