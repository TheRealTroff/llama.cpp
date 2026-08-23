# Getting an occupancy number: four ways in, pick one

Status: **open**. Written 2026-08-23 as a routing stub for a fresh session. Nothing here is
measured; it is the option board. The measured state lives in `aps-counters.md`,
`headless-replay-probe.md` and `width4-verify.md` - **read the Status block of each before
starting**, and read `skills/macos-reversing` before touching any private framework.

## The goal, and why it is worth a session

`width4-verify.md` has established the `nxpsg=16` mechanism **only by elimination**: at both
width 3 and width 4 the register count is identical (64/64, 73/73), spill is 0, the
instruction count is slightly *higher* at nxpsg=16, and the grid doubles 320 -> 640. Every
competing explanation is dead, which leaves dispatch geometry - but that is an argument, not
a measurement. `Compute SIMD Groups Inflight per Core` would size it.

Everything needed is already on disk: 10 captures, 10 replays, 99,478 parsed counter samples
and the 456-counter named catalogue. **What is missing is the join between the sample series
and the counter names.**

## The four options

### A. Unlock the counter names through the processor config (recommended first)

`XRGPUAPSDataProcessor` works: it constructs, and `-loadCounterGraphConfig` returns the full
named catalogue. `-loadCounters:` returns NO for every counterSet tried, so
`-apsRawCounterNames` and `-apsDerivedCounters` stay empty. **One config key is the blocker.**

`+[XRGPUAPSDataContainer configVariantFromConfig:]` is ~52 bytes of code and does a keyed
subscript on the dictionary you hand it (proven - passing an NSString threw
`-[NSTaggedPointerString objectForKeyedSubscript:]` from inside it). So:

1. **Swizzle `-[NSDictionary objectForKeyedSubscript:]` in our own process to log the key**,
   then call `configVariantFromConfig:` with the entry-0 dictionary. That prints the key
   being looked for, directly. This is the cheapest decisive step on the whole board.
2. Failing that, `otool -tV` the function and read the constant.
3. With the key known, feed the right shape to `-setConfig:` / `-initWithConfig:` /
   `-counterConfigForGRC:counterSet:` and `-loadCounters:` should populate.

Cost: low. Payoff: names on all 35 counters, which turns option B into a measurement.
Needs no clicks and no Xcode GUI.

### B. Identify counters by behaviour, without names

The values are already parsed (`perf/aps-samples.py`, 23 distinct `(slot, field3)` series).
The blocker recorded there is that the two arms return unequal sample counts (244 vs 81), so
a ratio of means is not a ratio of the underlying quantity.

1. **Normalise per unit time** using the two timestamp fields in each record rather than
   comparing raw means. That is the specific fix for the unequal-sampling confound.
2. Check whether series are cumulative or per-sample - slot 4 / field3 6 reading flat at
   ~8962 across both arms suggests per-sample, but verify rather than assume.
3. Diff across the four traces that form matched pairs (`w3`/`w4` x `nx8`/`nx16`), and match
   the surviving movers against `perf/ref/agx-counter-graph.json` on `unit` and
   `counterType`. A bounded 0-100 series is a utilization or limiter; an unbounded rising
   one is a cycle or instruction count.

Cost: low, data all in hand, no clicks. Payoff: sizes the effect, but an unnamed series is
weaker evidence than a named one. **Good as a cross-check on A, poor as a substitute.**

### C. Drive the replay over DY and remove the human

`headless-replay-probe.md` now knows the live path: `GPUTraceSession -setupAndStartReplayer:`
over `DYXPCTransport`, and **not** the `GTLaunchServiceXPCProxy` route that was mistaken for
a permission boundary. Two-way DY messaging already works for us and the kind values are
recovered.

1. Read `traces/aug23/replay-trace/extract-launch-window.log` - 10k lines around the launch -
   for the exact send order.
2. Aim the replayer-band kinds (4096+) at the **replayer's own endpoint**, not the agent
   transport, which is where they returned nil replies before.

Cost: medium to high. Payoff: the largest of the four, because it is a multiplier - every
counter measurement currently costs a human at the machine clicking once per trace.

### D. Attack the width-4 shelf without the counter

The perf question does not strictly need occupancy. `width4-verify.md`'s own board still has
the drafter-pipelining lever and the `nxpsg` per-shape routing question, and run 6 showed the
aggregate is only -1.5% to -1.7% anyway.

Cost: varies. Payoff: possibly the most *useful*, and it is worth asking honestly whether the
occupancy number would change any decision or is now mostly curiosity. **If it would not
change a decision, say so and close this stub rather than spending a session on it.**

## Suggested order

**A, then B as a cross-check.** A is cheap and decisive; B validates it against data already
on disk. C only if the click cost becomes the binding constraint - it is the biggest win but
also the biggest unknown. D is the honest alternative if the answer would not change what we
build.

## Do not repeat these

All measured, all recorded in the files above:

- Hashing the counter names. Not sha1/224/256/384, md5, blake2b or blake2s of the 535
  `vendorCounters` strings under 8 case/separator variants, and no mapping table exists on
  disk anywhere under the plugin, Instruments or the GPUTools frameworks.
- `XRGPUAPSDataContainer +fromData:error:` on the `APSCounterData` blobs - rejects all 41.
- Keyed-archive readers on `Counters_f_*.raw` / `Timeline_f_*.raw` / `Profiling_f_*.raw` -
  `NSCocoaErrorDomain 4864`, they are not archives.
- `GTMioKVDataStore -initWithURL:` on `streamData` - returns nil.
- Instruments/xctrace for GPU counters - 0 rows, `toolchain-isa-probe.md`.
- `-launchReplayService:` - the app never calls it; not a boundary, just not the path.
