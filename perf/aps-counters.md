# The runtime GPU counters are in the replay output, not behind Instruments

Status: **open, and much closer than it was.** The counter data four sessions chased through
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

2. **`ShaderProfilerData` is a raw sample buffer**, not an archive, so it needs a format.
   The sibling `Counters_f_*.raw` / `Timeline_f_*.raw` / `Profiling_f_*.raw` are the same
   data unarchived: `GTMioTraceData +traceDataFromURL:error:` rejects all three with
   `NSCocoaErrorDomain 4864` (not a keyed archive). Do not spend time pointing keyed-archive
   readers at them.

## Why this matters for width4-verify.md

`Compute SIMD Groups Inflight per Core` is the counter that would **size** the `nxpsg=16`
effect. Runs 3 and the replay counters have established the mechanism only by elimination:
register pressure identical (64/64 at width 3, 73/73 at width 4), spill 0, instruction count
slightly *higher* at nxpsg=16, and the grid doubling 320 -> 640. An occupancy number would
turn that from an argument into a measurement. `APS_USC` is the source it would live in.

## Tooling

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
