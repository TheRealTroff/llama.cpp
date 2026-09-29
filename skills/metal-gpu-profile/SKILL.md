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

## Step 0 - Run the census first

Before opening any single kernel by hand, run the fork's `perf/kernel-census.sh <profiled
server.log>` with the pick's routing env (`perf/kernel-census.md`). It times, captures and decodes
every top kernel of a profiled run and ranks them by instructions and loads per GFLOP (MMA class)
or x byte floor (streaming class) against the class best, and diffs against the last snapshot.
Every 2026-09-05 kernel win came from that comparison done by hand on one kernel; the census does
it for all of them. Steps 1-3 below are what it runs per kernel, and what you use to go deeper on
a flagged row.

Read the census row in this order (2026-09-09, `perf/agx-backend-access.md`):
1. **`x floor@252`** first. It is time over bytes at the measured practical peak (252 GB/s, a 37 MB
   CPY; 273 is the LPDDR5X nominal). Under 1.1 the row is flagged FLOOR (the width-1 mul_mv kernels sit at 1.03-1.07; 1.1-1.3 is NEAR: stream-side levers only): the kernel streams at
   the DRAM limit and NO per-instruction reading applies, whatever issue/stall says (the
   profiler cannot see the scoreboard waits - they are bits in the instructions, `Wait
   instruction count 0`). Under 0.95 it is flagged CACHE: the perf loop re-reads a tensor that
   fits the system cache, so the number is not a DRAM number and the byte floor is not the bound.
2. Only then issue/stall, instructions per work unit, and the per-instruction join
   (`<row>.join.json`: final machine IR aligned to the native stream and the profile, `mem%` =
   executed share of loads/stores). The per-instruction `cost` column is executed x a static
   per-opcode table; `cost2` (stall) is the measured column.

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

> **Xcode 27 + Metal Toolchain (2026-09-18) restored the per-instruction profiler after the macOS 27 upgrade
> (2026-09-16) had taken it out.** The route is Apple's `gpudebug` v1.0: `profile run --exec serial --embed`
> replays the trace (8-30 s) and EMBEDS the shader-profiler bundle into it as
> `<trace>/emb_stream_N.gpuprofiler_raw/` - `streamData` plus 20 each of `Counters/Timeline/Profiling_f_N.raw`,
> byte-for-byte the `raw/` contract of step 3. The wrapper below does that and moves the bundle to
> `<outdir>/raw` + `<outdir>/streamData`; `kernel-census.sh` needed no change (verified on the width-8 census:
> executed sums = binary totals, issue/stall/regs populated). Facts that cost time: the Metal Toolchain is a
> separate download (`xcodebuild -downloadComponent MetalToolchain`, 839 MB - `xcrun metal` refuses without
> it); the license must be accepted (`sudo xcodebuild -license`); use `--oneshot`, a trailing `exit` leaves
> the session alive (`gpudebug -l`, `--terminate all`); in an interactive session the `shaders` leaf fills a
> few seconds AFTER "Profile data collected"; offline `metal` compiles for the prescreen/translator need
> `-mmacosx-version-min=26.0` (Xcode 27 emits AIR 2.9, `applegpu-nt` targets 2.8). Still down under Xcode 27:
> `--backend dy` (no DYDesktopDevice) and the lldb machine-IR join (`agx-nt-opt.py mir`, "cannot emit
> pipeline" from the re-signed debug copy) - census rows say "no join"; read sizes (14 B = loads) meanwhile.
> Under Xcode 26.6 on macOS 27 both backends were empty (`APSCounterData entries: 0`, empty performance leaves).

```sh
references/metal-profile-headless.py \
  /tmp/perf-metal-<pid>.gputrace /tmp/profile-output
```

The wrapper prefers Apple's supported `gpudebug` CLI (macOS 27's `/usr/bin/gpudebug`, functional with
Xcode 27 + the Metal Toolchain; empty under Xcode 26.6). It drives `profile run --exec serial --embed` and
archives the embedded bundle; pass `-c` commands to override.

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
  captures repeat ops a shape-dependent number of times. **And per OP INSTANCE across arms of one shape**
  (2026-09-18, `perf/w8-decomp-sep18.md` levers 3+4): `test-backend-ops perf` fills the captured graph with a
  speed-dependent number of copies of the op, so a faster variant shows MORE executed instructions per hot row
  (5/6/8/9 copies across four arms; the tool's `dispatches` stayed 71 in all of them). Divide a hot row's
  `executed` by its per-instance count (trip count x simdgroups x threadgroups; here 308992 in every arm) and
  compare static hot-row counts (391 -> 365 -> 339) - never `exec/disp` between arms of different speed.
- **Hot loop = rows with `executed >= 0.9 * max(executed)`.** Sum their `cost` (issue)
  and `cost2` (stall) for the loop's share; histogram their `size` field for the
  codegen fingerprint (6 B ~ f32 FMA short forms, 10 B ~ compact wide-operand
  arithmetic, 14 B ~ device loads, 12 B ~ load-consumers/MMA lowering on g16s).
- **The per-instruction `cost` (issue) column is NOT a measurement (2026-09-09,
  `perf/agx-backend-access.md`).** Joined against the final machine IR for 10,612 instructions
  across 34 profiles, `cost = executed x w(opcode) x k`: `w` is a fixed per-opcode weight
  (1 for f16 arithmetic/moves/branches/compares, 4 for 32-bit shifts, bit-field extracts,
  32-bit uniform ALU and f32 unary, 6 for one convert form, 8 for 64-bit pair ops, **0 for
  every load, store and stop**), `k` one constant per kernel, zero residual. A "hot
  instruction by issue" inside a kernel is count times that table. `cost2` (stall) DOES
  vary per site and per kernel and is the measured column; per-kernel issue vs stall shares
  remain measured. The table predicts time no better than plain counting on the mv fleet
  (5.3% vs 18.6% spread), so read it as the profiler's apportioning model, not hardware
  cycles. To see what a site actually is, dump the kernel's machine IR
  (`perf/agx-nt-opt.py mir`) and join (`perf/agx-mir-align.py --profile`).
- **Check x-the-byte-floor BEFORE the issue/stall split; the split cannot see encoded waits
  (2026-09-09 width series, `perf/agx-backend-access.md`).** On g16s the scoreboard waits are
  bits inside instructions (`Wait instruction count 0` in the stats), so a simdgroup waiting on
  DRAM is not "stall" to the profiler. q6_K and q4_0 mul_mv at 17408x5120, widths 1/2/4, all
  run at 1.09-1.21x the 273 GB/s floor (244 GB/s achieved) while reporting 87-98% issue; their
  time ratio is exactly their byte ratio, width 2 is free, and an ILP probe (N accumulator
  chains) moved nothing. The census computes `x_floor` for stream kernels: if it is under ~1.2,
  stop - no per-instruction reading (issue site, class mix, encoding size, us per M executed)
  applies. "us per M executed" fell 17.8 -> 8.6 from w1 to w4 only because more instructions
  fit into the same memory-bound time. The 7.6-8.6 us/M rule below is therefore a statement
  about the w4/w5 kernels' distance from the floor, not a hardware issue rate.
- **`test-backend-ops perf` overlaps iterations of an op whose inputs nothing writes (2026-09-09,
  `perf/agx-backend-access.md`).** ggml-metal only serializes dispatches on buffer hazards; a perf case's
  src tensors are never written, so back-to-back iterations of a streaming kernel run concurrently and
  contend for DRAM (+20..90% per op measured). Any variant that removes a barrier-bearing step (the
  decode f32->f16 activation copy ends in one) measures that contention, not the kernel. Keep the
  barrier (`ggml_metal_op_concurrency_reset`) in the variant, or compare in the real graph.
- **Per-op profile time is a span, not critical path, for small concurrent ops (2026-09-09,
  `perf/agx-backend-access.md`).** K copies between the same two barriers overlap K-fold and each
  span carries the dispatch front-end latency; the conv-state carry fusion removed "1.1-1.7% of GPU
  time" and delivered 0.2-0.7%. Estimate a group of small ops between shared barriers at dispatch
  count x ~1.5 us; only an op that owns its barrier (the activation copy did: reset after it) is
  worth its span, and even that delivered ~75%.
- **issue share x issue rate, not instruction count or stall alone, predicts time.**
  Measured both failure directions: an unroll cut dynamic instructions 15% and lost
  (stall rose), a sumy variant issued 25% MORE instructions more smoothly and lost.
- **Count `Device load instruction count` against what the source streams.** A `constant`
  table indexed by a runtime value compiles to one device load per lookup: the iq4_xs SoA
  kernel shows 44 loads where its q4_0 twin shows 12, the difference being exactly 4 rows x
  8 nibbles of LUT, and those loads are the kernel's whole stall (13.5%, three 12 B
  load-consumer sites). A `simd_shuffle` from a lane-held table is the register-resident
  alternative and doubled text and time (`perf/ud-model.md` step 6-7).
- **Several arms at once:** `perf/run-ud-soa-profile.sh` is the worked driver (capture ->
  headless replay -> stats -> per-instruction JSON per arm, pipeline name from each capture's
  own stderr, skips arms already captured), `perf/shaderprof-compare.py` reads the JSONs side
  by side (exec/dispatch, issue/stall, hot loop share, size fingerprint, stall sites), and
  time the same kernels in a separate uncaptured pass. Delete each `<out>/raw/` after decoding
  (0.7-1.8 GB per arm; the JSON/stats/streamData are what you keep).
- **Per-instruction issue cost, `us x issue share / executed per dispatch`, is 7.6-8.6 us/M
  across every mv kernel measured on the M4 Pro** (q4_0, iq4_xs, q4_K, q5_K, SoA and ext) -
  the outlier is a load-heavy f32 form at 11.4. So time = executed x cost / issue share, and the
  fleet's issue-share range is now 64-96% (q4_K SoA v2 at 95.7% is the high).
- **On `simdgroup_matrix` (mul_mm) kernels, instruction counts undercount the MMA ops:** each
  MMA instruction occupies the ALU for many cycles, so a 64-column tile that executes 33% fewer
  instructions than the 32-column kernel runs only 6.7% faster. Read these kernels as MMA cycles
  (fixed per FLOP) plus non-MMA instructions per K-step (dequant, staging, tg loads); fit the two
  terms across formats of the same shape (`perf/ud-model.md` step 8: ~10.9 ms MMA + 0.017 ms per
  non-MMA instruction/step at n=512 [17408,5120]) and the per-format dequant tax falls out. The
  lever for a long dequant chain is then tile width (paid once per NR1 columns), not the chain.
- **Kernels with several loops of different trip counts (flash-attention: an inner QK loop, a
  per-key-tile loop, a per-chunk softmax/PV body) need the execution-count TIER view, not the
  hot-loop rule** - `perf/shaderprof-compare.py --tiers` groups rows by executed count; divide
  each tier's exec/dispatch by (simdgroups x chunks) to get instructions per unit of work.
  Worked case (`perf/ud-model.md` step 9): the prefill FA kernel ran 1105 instructions per
  64-key chunk per simdgroup for 128 MMAs; the QK tier was 749 of them (89 non-MMA per 8 MMAs)
  and its size sequence showed a ~40-instruction 10/12 B address block ahead of the loads - the
  `simdgroup_load(..., transpose=true)` from device memory. Computing S^T = K Q^T instead
  (K tiles loaded plain, Q staged transposed once, score tile stored with the transpose flag) was
  -7..-8% on every FA form, byte-identical. Traced afterwards (2026-09-06): the win is the LOAD
  INSTRUCTION COUNT, not the address block - a transposed 8x8 half tile load lowers to 3 load
  instructions, the plain one to 1 (24 -> 8 per unrolled step, the exact 14 B delta), while the
  address arithmetic stayed; -4.4% executed, -2.4% per-instruction cost, -1.6 pt stall. The offline
  TEXT SIZE did not move (12580 vs 12586 B) because bytes moved between loop levels at equal size
  (-16 loads +8 ALU in the inner body, +7 in the per-chunk body, + the prologue staging) and static
  bytes weight the levels equally where the runtime weights them 3.9 : 1 : 0.003. Text size ranks
  register/unroll changes; it cannot see an instruction-class swap or a loop-level move. Count
  loads per MMA instead: FA QK was 2 per MMA where mul_mm pays ~0.5.
- **Fewer executed instructions at a higher issue share can still be SLOWER: read the issue share by
  encoding size before believing a count** (2026-09-09, `perf/ud-remaining-quants.md` Q6_K width 1): the
  stored q6_K width-1 kernel executed 16% fewer instructions than native at 97% issue / 3% stall and ran 4%
  longer. Within each kernel an 8 B or 12 B instruction carried ~4x the issue time of a 4/6/10 B one
  (per-instruction issue share 0.67-0.84 vs 0.17-0.23), and the stored hot loop put 68% of its issue in
  that class vs native's 55% - 16 per-lane-iteration addresses (4 rows x 4 planes) vs native's 10 pointer
  advances with immediate-offset byte loads. Recipe: `cost[size]/count[size]` over the hot rows; a kernel
  that trades cheap 10 B forms for fat 12 B ones wins the count and loses the clock. The wide-load form that
  halves the addresses per element recovered half the gap (+2.4%); the prescreen could not rank it because
  it doubles the work per iteration (text 2942 -> 4502 B) - per-element counts, then timing.
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

- **The FA perf cases are head-major; the served cache is cell-major (2026-09-29, `perf/kv-layout.md`).**
  `test_flash_attn_ext` allocates K/V `[hs, kv, nh]` (each head's stream contiguous) unless the case passes
  `permute={0,2,1,3}` + `kv_view=false`, which is the cache's own row-per-cell layout. The f16 FA kernels are ~10%
  slower on the cache's layout at every decode extent (2% at 96K prefill), Turbo4 0-2.5%. Time FA on the cache's
  layout: the Qwen3.8 shapes exist in both layouts in the perf list, `perf/run-fa-layout-timing.sh` runs them
  under the pick env per line. And price a layout lever's ceiling with a kernel-side load-stream probe (a function
  constant that issues the relayout's loads over the stored bytes, timing only) before building a ggml type: the
  Turbo4 norm-plane relayout came out flat (0.98-1.02x) that way in an hour.
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

Three more traps caught by these tools the day they were built:
- **Submit-prof prints 64-graph WINDOW AVERAGES, not per-graph facts** (`sum/n` at
  `sprof.n % 64`, no per-topology breakdown). A ctx that cycles unequal graphs
  per round (the dflash drafter: enc ~0.4 ms, inject ~0.5, draft decode ~13) shows
  cycling near-identical averages (busy 4.4-4.6) that LOOK like three uniform
  graphs - that misread cost the drafter-graph-count stub its premise for a day.
  Before building a lever on any counter, read its printing code for the
  aggregation (windowed? summed? serialized?), and get per-phase truth from the
  in-tree per-phase counters (`dflash-prof` enc/inject/lattice lines) or a per-op
  dump first.
- **Count rounds from the run's own counters** (`draft acceptance ... mean len` /
  spec-prof `n =`), never from another run's acceptance rate - a wrong divisor
  manufactured a phantom "11 ms/round untimed" finding for an afternoon.
- **`ggml_metal_get_tensor_async` routes host-visible readbacks through the GPU
  queue** (fresh `newBufferWithBytesNoCopy` + blit command buffer queued behind the
  whole graph): the per-round logits readback cost ~3.4 ms of pure serial latency.
  `GGML_METAL_GET_MEMCPY=1` (branch `cpu-round-overhead`) defers to a plain memcpy
  after the sync wait: +3.3% e2e, byte-identical. When hunting CPU overhead, look
  for work that is QUEUED BEHIND the graph, not just work beside it.

## Per-row vs per-token attribution by changing the work per simdgroup (2026-09-06)

When a kernel's hot loop has several kinds of work per iteration, profile two instantiations that
differ in the amount of one kind (e.g. 1 vs 4 state rows per simdgroup, `perf/gdn-prefill-scan.md`)
and diff the hot rows: the instructions whose count stays fixed are per-iteration overhead
(there: five 64-bit pointer advances at 4.06 issue units each = 22% of the loop), the rest scale
with the work. Confirm the identity offline with a deletion variant and `agx-disasm.py` size
sequences - no mnemonics needed. Also: the census's perf case must match the op's FULL shape;
the GDN case matched `head_count` but not the value-head repeat and timed a third of the op
(0.73 vs 2.64 ms) - ratios survived, absolutes did not (`perf/kernel-census.md`).

## Long-context census (2026-09-06, `perf/fa-long-context.md`)

FA rows scale with context (prefill quadratically, decode linearly) while mm/GDN rows do not, so the
8K census under-ranks them: FA was 4.1% of prefill / 3.7% of the round at 8K and 12.9% / 10.7% at 25K
(the 96K record has it near 40%). `perf/run-longctx.sh` runs the pick env on any prompt/context and a
profiled arm feeds `kernel-census.sh`; every ubatch and every round is its own KV length, so the census
FA filter snaps to the nearest perf case (512/8448/16384/24576, 12%) - an exact-match filter found no
case for a single FA row at 25K. Pass `B=<worktree>` to the census, always.

## Same static size, 3x the dynamic count (2026-09-06, `perf/ud-model.md` step 16 B)

The Turbo4 batched FA kernel profiled at 992 static instructions against the f16 kernel's 1067, 0 spill,
fewer registers - and 17.1M dynamic instructions per decode dispatch against 5.3M at 50% issue / 50% stall.
Static stats cannot see a loop that is not unrolled, a scratch round trip or a transposed load (3 load
instructions per tile): read `exec/disp` and the tier view first, then the hot loop's 14 B count against the
loads the source needs. Fixed by the register-resident dequant form (`metal-kernel-prescreen` skill).
