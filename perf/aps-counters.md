# The runtime GPU counters are in the replay output, not behind Instruments

Status: **open. Container decoded, sample format decoded, names still unresolved.** The counter data four sessions chased through
Instruments/xctrace has been sitting in the replay output the whole time. The container is
decoded and readable in pure Python with no GUI, no Instruments, no entitlement and no ObjC.
Two things remain: the 35 counter names are hashed, and the sample payload is not decoded.

## Where they are

`APSCounterData`, a key in the **root of `streamData`** - the same file
`perf/gpuprofiler-stats.py` has been reading since 2026-08-23, sitting next to the
`pipelinePerformanceStatistics` it does read. APS = Apple Performance Statistics.

`streamData` is an NSKeyedArchiver plist; `APSCounterData` is an array of 41 **nested**
keyed-archive blobs (`bplist00`). `plistlib` opens all of it.

```
APSCounterData[0]     schema
  Limiter Counter List Map   hardware source -> its counter list
  limiter sample counters    the 35 sampled, as hashed names
  Counter Info               hash -> int
  Uarch Enabled              true
APSCounterData[1..40] samples
  Source / SourceIndex / RingBufferIndex + a raw ShaderProfilerData payload
```

Measured on `w3-ffn_down-ext-nx8`, and identical in structure across all ten captures:

| source | counters | sample buffers |
|---|--:|--:|
| `APS_USC` (unified shader core) | 10 | 20 |
| `RDE_0` | 13 | 10 |
| `BMPR_RDE_0` | 10 | 5 |
| `Firmware` | 2 | 1 |
| (unlabelled) | - | 4 |
| **total** | **35** | **40** |

Payloads are 365-480 KB each, ~16 MB per replay. `perf/aps-counters.py` prints all of this.

**`Uarch Enabled` is true and the counters are populated**, which is the part that matters:
`toolchain-isa-probe.md`'s conclusion that the counters are unreachable is wrong about the
replay path. It is right about Instruments - that route really does return 0 rows - but the
Xcode replay collects them anyway and writes them next to the compile statistics.

## What is not done

1. **The 35 names are hashed**, `_<64 hex>`. Ruled out, each measured:
   - Not sha256, sha1, md5 or sha512 of the `vendorCounters` strings in Instruments'
     `GPUCounterGraph.plist` (534 names x 6 case/separator variants, 0 hits).
   - No mapping table on disk: grepping a known hash across `GPUDebugger.ideplugin`,
     `Instruments.app` and every `GPUTools*` framework finds nothing.

   So resolution is a **runtime** step. The way in is Xcode's own object graph, where the
   names are already methods - `GTMioCounterData -name`, `-values`, `-sampleCount`,
   `-timestamps`, and `GTMioNonOverlappingCounters -encoderCounterNames`,
   `-counterValuesForPipelineStateId:encoderFunctionIndex:`. That last one is per-kernel
   counter values, which is exactly the shape this investigation wants. Enumerate with
   `perf/gtcounter-classdump.py`.

2. ~~**`ShaderProfilerData` is a raw sample buffer**, not an archive, so it needs a
   format.~~ **Format cracked - see "Round 3".** Still true and worth keeping: the sibling
   `Counters_f_*.raw` / `Timeline_f_*.raw` / `Profiling_f_*.raw` are not keyed archives
   (`GTMioTraceData +traceDataFromURL:error:` -> `NSCocoaErrorDomain 4864`), so do not point
   keyed-archive readers at them. **They are referenced by name from the archive**: the 20
   `APS_USC` records carry `APSTraceDataFile: Counters_f_<n>.raw` instead of an inline
   payload, which is why those files must be archived alongside `streamData`.

## Why this matters for width4-verify.md

`Compute SIMD Groups Inflight per Core` is the counter that would **size** the `nxpsg=16`
effect. Runs 3 and the replay counters have established the mechanism only by elimination:
register pressure identical (64/64 at width 3, 73/73 at width 4), spill 0, instruction count
slightly *higher* at nxpsg=16, and the grid doubling 320 -> 640. An occupancy number would
turn that from an argument into a measurement. `APS_USC` is the source it would live in.

## Tooling

The general technique is written up as `~/.claude/skills/macos-reversing` - the ctypes/ObjC
recipe, the traps, and how to read failure shapes. Read that before extending any of these.

- `perf/aps-counters.py` - decodes the container, prints schema, sources and payload sizes.
- `perf/gtcounter-classdump.py` - live ObjC class/method dump of the GPU tools frameworks,
  filtered by regex. This is how the reader classes above were found.
- `perf/gtcounter-probe.py` - tries Xcode's readers against a given file; how
  `GTShaderProfilerStreamData +dataFromArchivedDataURL:` was confirmed to work and
  `GTMioKVDataStore -initWithURL:` confirmed not to.

All three need `DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks` and a
non-SIP python (`~/play/.venv-convert/bin/python3`); `/usr/bin/python3` has `DYLD_*`
stripped. `aps-counters.py` alone needs neither - it is pure `plistlib`.

## Round 2 (2026-08-23, same day): the resolver is reachable, the last link is not

Everything below is measured, so nobody re-runs it.

**Works.** `XRGPUAPSDataProcessor -initWithGPUGeneration:variant:rev:config:options:` returns
a live processor (`gpuGeneration` is in the `streamData` root and is **2** on this M4 Pro -
not 16, despite `metalPluginName` being `AGXMetalG16X`). On it,
`-loadCounterGraphConfig` returns the **full named catalogue**: 456 counters with `name`,
`description`, `unit`, `vendorCounters`, `counterType`. Saved as
`perf/ref/agx-counter-graph.json` - it includes the two this investigation wants:

```
Compute SIMD Groups Inflight per Core   vendor: Compute Simdgroups Inflight Per Shader Core
Kernel Occupancy                        vendor: ShaderCoreComputeUtilization / ComputeOccupancy
```

**Does not work yet.** `-loadCounters:<n>` returns NO for n = 0..5 at both gpuGeneration 2
and 16, and `numAPSRawCounters` / `numAPSDerivedCounters` stay 0, so `-apsRawCounterNames`
and `-apsDerivedCounters` come back empty. The processor needs a config it has not been
given; `-setConfig:`, `-counterConfigForGRC:counterSet:` and
`-loadAPSCounters:counterSet:` are the untried levers.

**Ruled out this round, each measured:**

- `XRGPUAPSDataContainer +fromData:error:` rejects all 41 `APSCounterData` blobs (nil, no
  error object). The blobs are not a serialised container.
- `-initWithConfig:baseFolder:` takes a **dictionary**, not a name - passing an NSString
  throws `-[NSTaggedPointerString objectForKeyedSubscript:]` from
  `+configVariantFromConfig:`. Passing entry 0's unarchived dictionary returns nil, so the
  schema in the file is not the config this expects.
- `XRGPUAPSDataContainer` alloc/init then `-loadInstrumentsConfig` returns YES, but `-config`
  stays nil, so it loads nothing usable on its own. `-loadGTAConfig` and `-loadATRCConfig`
  return NO.
- **The hashes are still unmatched, and now against the runtime catalogue too.** 535
  `vendorCounters` strings x 8 case/separator variants x 7 digests (sha1/224/256/384,
  md5, blake2b, blake2s): zero hits. The counter-graph config contains no `_<64 hex>`
  strings at all, so it is not the mapping either. Whatever those hashes key, it is an
  internal driver namespace, not the friendly or vendor names.

## Round 3 (2026-08-23): the sample format, cracked

Johan's observation - that the shape of the contents should give away the identity of a
counter - is what prompted this, and it got the format open even though it has not yet
produced a name.

The samples are in the one `APSCounterData` entry carrying `Derived Counter Sample Data`
(a `list(16)` of `list(5)` of `list(1)` of bytes). Every byte blob is a run of **64-byte
records** behind an 8-byte magic:

```
 0   char[8]  "GPRWCNTR"
 8   u64      timestamp     strictly increasing within a blob
16   u64      value         the reading
24   u64      field3        small, 0..6 - candidate counter id within a slot
32   u64      sequence      +1 across the whole stream
40   u64      timestamp2    a second, coarser clock
48   u64      sample index  0,1,2,... within the blob
56   u64      slot          matches the list(5) position
```

99,478 records parse cleanly out of `w3-ffn_down-ext-nx8` with the magic checking on every
one. `(slot, field3)` gives **23 distinct series**. `perf/aps-samples.py` does this in pure
`plistlib` + `struct` - no frameworks, no GUI.

One series identifies itself by shape already: **slot 4 / field3 6** carries ~96k samples
and reads 8961.6 in the nxpsg=8 arm against 8962.3 in the nxpsg=16 arm - flat to 0.008%
across a change that moves wall-clock by 5%. That is a clock or fixed-rate tick, not a
workload counter.

**What this does NOT yet support, and the script says so in its own output.** The two arms
return very different sample counts for the same series (244 vs 81 on several), so the
aggregation windows differ and a ratio of means is *not* a ratio of the underlying
quantity. Several series do move a lot - `(2,3)` 0.405, `(1,3)` 0.449, `(3,3)` 0.524, and
`(1,5)` 2.505 - and a counter that halves when the grid doubles is exactly what a
per-threadgroup quantity would do, but with unequal sampling and no names that is a lead,
not a measurement. `aps-samples.py` flags every unsafe ratio rather than printing it plain.

So: still no occupancy number. What changed is that the values are now in hand, which makes
the identification problem a data problem rather than a reverse-engineering one.

## Next

Two live threads, in order of promise:

1. **Feed the processor a config so `-loadCounters:` succeeds.** Everything else on
   `XRGPUAPSDataProcessor` already works, including the named catalogue. Find what
   `+configVariantFromConfig:` reads out of the dictionary - that one key is the blocker -
   then hand the same shape to `-setConfig:` or `-initWithConfig:`.
2. **The GTMio path**, unchanged from round 1: build the object graph so `-name` resolves, starting from
`GTShaderProfilerStreamData +dataFromArchivedDataURL:` (works, verified) and looking for the
constructor that turns its `archivedAPSCounterData` into `GTMioTimelineCounters` /
`GTMioNonOverlappingCounters`. If the graph needs a `GTMioTraceData` that only the plugin can
build, the fallback is differential: the same 35 counters are sampled in every capture, so
diffing `w3-ffn_down-ext-nx8` against `w3-ffn_down-ext-nx16` identifies *which* counter
indices move with `nxpsg` even while unnamed, and `APS_USC` is only 10 of them.
