# The runtime GPU counters are in the replay output, not behind Instruments

Status: **closed for the counter question. Names resolved, values read.** The counter data
four sessions chased through Instruments/xctrace has been sitting in the replay output the
whole time. ~~Two things remain: the 35 counter names are hashed, and the sample payload is
not decoded.~~ Both are done.

- **Round 4** named all 35: the `_<64 hex>` strings were never hashes to crack, they are
  **GRC enable strings**, and `libagxps` hands them out beside the plaintext counter names.
  `Compute SIMD Groups Inflight per Core` is `APS_USC` index **2**. `perf/agxps-probe.py`.
- **Round 5** read the values: `Counters_f_<n>.raw` parses, 137 counters x ~22k samples per
  USC x 20 USCs, in **0.8 s per capture**. The blocker was one descriptor field,
  `SystemTimePeriod`, silently rejected for not being a power of two.
  `perf/aps-usc-values.py`.

**The number this was all for:** compute simdgroups inflight per shader core rises
**2.46 -> 2.87 (+16.6%)** at width 3 and **2.48 -> 2.87 (+15.8%)** at width 4 when `nxpsg`
goes 8 -> 16, and is **flat between width 3 and width 4 at matched `nxpsg`** (2.461 vs 2.481,
2.869 vs 2.873). See "Round 5". The follow-on question - what *does* limit width 4 - is
answered in `perf/width4-limiter.md`.

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

1. ~~**The 35 names are hashed**, `_<64 hex>`.~~ **Resolved in Round 4 - they are GRC enable
   strings, not a digest.** The negative results stay, because they are why nobody should try
   cracking them again:
   - Not sha256, sha1, md5 or sha512 of the `vendorCounters` strings in Instruments'
     `GPUCounterGraph.plist` (534 names x 6 case/separator variants, 0 hits).
   - ~~No mapping table on disk: grepping a known hash across `GPUDebugger.ideplugin`,
     `Instruments.app` and every `GPUTools*` framework finds nothing.~~ Correct as far as it
     went, but the search was too narrow. The strings **are** on disk, in the AGX driver:
     `/System/Library/Extensions/AGXMetalG16X.bundle/Contents/Resources/AGXMetalPerfCountersExternal.plist`
     keys 3,906 of them to their hardware `{Partition, Select, Flag}`.

   ~~So resolution is a runtime step, via Xcode's object graph and `GTMioCounterData -name`.~~
   It is a runtime step, but a far shorter one: `agxps_counter_get_grc_enable_str` in
   `libagxps` returns these strings next to the plaintext names. See Round 4.

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
- `perf/agxps-probe.py` - **start here.** Drives `libagxps` from ctypes and names every
  counter a capture enabled. `--gpus`, `--find <substring>`, `<streamData> --json <out>`.
- `perf/aps-usc-values.py` - **the values.** Parses `Counters_f_<n>.raw` for all 20 USCs.
- `perf/aps-dram-bandwidth.py` - DRAM bandwidth from the `BMPR_RDE_0` blobs, pure plistlib.
- `perf/aps-samples.py` - **refuted as a counter source in Round 5**, and its record framing
  is wrong. Keep only as the record of that.

All of these need `DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks` and a
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

~~**Does not work yet.** `-loadCounters:<n>` returns NO ... The processor needs a config it
has not been given.~~ **Half wrong, see Rounds 4 and 5.** The processor was built with
gpuGeneration 2, so `-agxpsGPU` was **NULL** and there was no GPU to load counters for. With
gen 16 / variant 5 / rev 1 the handle is live and `-numValidUSCs` reads 20. The names did not
need `-loadCounters:` at all, and the values came from the C parser under it.

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

## Round 4 (2026-08-23): the names, resolved - the hashes were GRC enable strings

Measured; `perf/agxps-probe.py` reproduces all of it.

### The library under the ObjC wrapper is exported, so skip the wrapper

`XRGPUAPSDataProcessor` is a thin shell over a C library **statically linked into
GTShaderProfiler and fully exported**: `nm -gU` lists **384 `agxps_*` symbols**, callable
straight from ctypes. Round 2 spent itself on `-loadCounters:` when the data it wanted was
one `dlsym` away.

```
desc = agxps_derived_counter_gpu_descriptor_create(gen, variant, 0,0,0,0,0,0)
agxps_initialize(&desc, 1, NULL, NULL)      # without this every counter name is NULL
gpu  = agxps_gpu_create(gen, variant, rev, true)
```

### `gpuGeneration` in streamData is not the agxps generation

`streamData` says `gpuGeneration = 2`. That is a **different enum**. `agxps` numbers this M4
Pro **gen 16, variant 5, rev 1**, confirmed by its own property table: 20 physical USCs, 2
mGPUs, 4 MB L2. Within a generation the variant is the die width (4 = 10 USC, 5 = 20, 6 = 40,
7 = 80). `agxps-probe.py --gpus` prints the grid.

Gen and variant are **measured**. The rev is **inferred**: 1 is what the built-in table
carries for gen 16 / variant 5, and nothing used here reads it.

> The config was never the blocker, and both suspects were run down:
> - `+[XRGPUAPSDataContainer configVariantFromConfig:]`, read out of `otool -tV`, returns 2
>   if `cfg["APS"]["Binaries"]` or `cfg["APS"]["PreSiBinary"]` exists, 3 if `cfg` has all of
>   `Version`, `SourceConfigList` and `GPUConfigurationVariables`, 1 if `cfg["APSFiles"]`
>   exists, else 0. A variant tag, not a gate.
> - `-[XRGPUAPSDataProcessor setConfig:]` reads `CountPeriod`, `PulsePeriod`,
>   `SystemTimePeriod`, `ChunkSize` (default 0x1000), `CounterUarchBehaviour`, `Timestamp`,
>   `AcceleratorID`, `CounterMapping`, `RDERawCounters`, and `GPUConfigurationVariables` ->
>   `num_cores` / `num_gps` / `num_agcs` / `omu_eval_window`. Handing it those works.

### The `_<64 hex>` names are GRC enable strings, not a digest

Three sessions tried to crack these as hashes. They are the identifiers the GPU register
config uses to enable a counter, and **the library returns them beside the names it knows**:

```
agxps_counter_get_name(ident)              raw -> obfuscated, derived -> PLAINTEXT
agxps_counter_get_grc_enable_str(ident)    the _<64 hex> string, verbatim as in streamData
agxps_counter_get_raw_counters_used_by_derived_counters(gpu, ...)   derived -> its raw inputs
```

So `Limiter Counter List Map` is a list of GRC enable strings per hardware source, and
`(source, index)` is the join key. No mapping table needed.

The table `agxps_initialize` builds holds **197,040** counters across all GPUs: 15,363
plaintext (the derived ones), 172,848 `_<lowercase 64 hex>`, 8,829 `<UPPERCASE 64 hex>` (the
USC raws).

**The obfuscation map is absent and it no longer matters.**
`agxps_load_counter_obfuscation_map(NULL)` returns NO - it wants `RawCountersMapping.csv`
from bundle `com.apple.gpusw.AGXProfilingSupport`, which does not ship. `setConfig:`'s
`CounterMapping` default is `/AppleInternal/Library/AGX/AGXCounterMapping.csv`, also absent.
With no map loaded, `agxps_counter_obfuscated_name` and `agxps_counter_deobfuscate_name` are
identity functions. That is the wall the hash-cracking rounds were hitting.

### `Compute SIMD Groups Inflight per Core`: it is APS_USC index 2

| field | value |
|---|---|
| catalogue name | `Compute SIMD Groups Inflight per Core` |
| vendor / agxps name | `Compute Simdgroups Inflight Per Shader Core` |
| unit / counterType | `SIMD Groups` / `Occupancy` |
| agxps ident (gen 16 / variant 5) | **185896**, derived, group `One Pass` |
| description | "The number of compute simdgroups running concurrently per shader core." |
| raw inputs | idents 109123 / 109125 / 109127 |
| GRC enable string of all three | `_5fa064796fa00e51a16682635d496690f5bb01777755209762a8752a444bde93` |
| position in the capture | **`APS_USC` index 2** |

`Kernel Occupancy` (`Compute Occupancy`, ident 185828) needs the same single counter.

### The driver plist gives the hardware selector for the same strings

`/System/Library/Extensions/AGXMetalG16X.bundle/Contents/Resources/AGXMetalPerfCountersExternal.plist`
is a 3,906-entry dict keyed by exactly these strings, each mapping to
`{Partition, Select, Flag}`. The ten `APS_USC` counters are all `Partition 40` with
single-bit `Select`s: index 0 bit 45, 1 bit 46, **2 bit 33**, 3 bit 32, 4 bit 39, 5 bit 35,
6 bit 37, 7 bit 44, 8 bit 38, 9 multi-bit.

Previous rounds grepped Xcode, Instruments and the GPUTools frameworks.
**`/System/Library/Extensions` was never searched**, and that is where the driver keeps its
counter database.

### What this capture can answer

**137 of 332** named derived counters are computable from the 35 GRC counters these captures
enabled. `agxps-probe.py <streamData>` prints them with the `(source, index)` each needs. The
enabled set is **byte-identical across all ten replays**.

## Round 5 (2026-08-23): the values - one descriptor field was the whole blocker

Measured; `perf/aps-usc-values.py` reproduces every number in under a second per capture.

### The wall, and what it actually was

`agxps_aps_parser_create` returned NULL for **every** supported GPU generation, which looked
like "APS parsing is stubbed in the shipping Xcode". It is not. The per-generation factory
validates exactly four descriptor fields and returns NULL with no error if any fails:

| descriptor field | rule |
|---|---|
| `PulsePeriod` | power of two, 16 .. 2048 |
| `SystemTimePeriod` | power of two, 64 .. 8192 |
| `CountPeriod` | 0, or a power of two 128 .. 32768 |
| `ChunkSize` | exactly 1024, 4096 or 262144 |

**`SystemTimePeriod` is the one nobody supplies.** It is absent from the capture and from
`APS Options`, and defaults to 0 - not a power of two. That single field is why
`-loadCounters:` looked like a missing config key for three rounds.

> Reading the validator was cheaper than any experiment. `x ^ (x-1) > x-1` is a power-of-two
> test and `cmn`/`b.lo` pairs are range checks; ~50 instructions gave all four rules exactly.

### The chain, end to end

```
proc   = [[XRGPUAPSDataProcessor alloc] initWithGPUGeneration:16 variant:5 rev:1 config:cfg options:0]
parser = agxps_aps_parser_create(proc + 0x20)          # +0x20 is the descriptor -setConfig: fills
pd     = agxps_aps_parser_parse(parser, buf, len, 1, &err)
         agxps_aps_profile_data_get_counter_names / _counter_values / _counter_values_num
         agxps_aps_profile_data_get_usc_timestamps
```

`agxps_aps_descriptor_create` returns its struct in **x8**, the arm64 indirect-result
register, which ctypes cannot express - so borrow the descriptor `-setConfig:` already built
at `proc + 0x20`.

Per `Counters_f_<n>.raw`: **137 counters, 22,251 samples each, 460,691 tokens, err=0.** Names
come back as the same `<UPPERCASE 64 hex>` strings `agxps_counter_get_name` uses, so Round
4's join applies directly.

**Config sensitivity, measured** (w3-ffn_down-ext-nx8 USC 0, sha256 of the value array):

| field | effect |
|---|---|
| `SystemTimePeriod` | 64/128/512/2048/8192 all **byte-identical**. Gates the parser, changes nothing. |
| `PulsePeriod` | 1024 and 2048 identical; 16 and 256 shift the sample count by 2 |
| `CountPeriod` | **matters**: 4096 -> 22,251 samples, 32768 -> 2,782, 128 -> 22,252 |
| `ChunkSize` | 4096 and 262144 identical; 1024 truncates to 7,336 samples |

### The sampling is uniform, which is what makes the arms comparable

`agxps_aps_profile_data_get_usc_timestamps` gives **4096.0 ticks per sample in every arm**.
The counter is an accumulator over a fixed 4096-tick window, so a per-sample mean **is**
directly comparable across arms - the confound `aps-samples.py` could not clear.

### The measurement

Summed over all 20 USCs. Vertex and Fragment use **disjoint** raw sets from Compute, so this
triple is compute and nothing else; in these compute-only captures only `FD6F91B4...` carries
data.

| capture | samples | acc/active | **simdgroups/core (active)** | (all samples) |
|---|--:|--:|--:|--:|
| `w1-ffn_down-mv` | 679,540 | 12,416.0 | **3.031** | 2.823 |
| `w2-ffn_down-mvnc2` | 592,860 | 11,830.8 | **2.888** | 2.162 |
| `w3-ffn_down-ext-nx8` | 445,020 | 10,082.3 | **2.461** | 2.240 |
| `w3-ffn_down-ext-nx16` | 477,000 | 11,749.4 | **2.869** | 2.531 |
| `w4-ffn_down-ext-nx8` | 428,060 | 10,161.1 | **2.481** | 2.064 |
| `w4-ffn_down-ext-nx16` | 444,370 | 11,766.2 | **2.873** | 2.243 |
| `w4-ffn_down-ext-nof16y` | 485,380 | 10,642.7 | **2.598** | 2.326 |
| `w4-attn_q-ext-nx8` | 476,083 | 8,499.7 | **2.075** | 1.819 |
| `w4-attn_q-ext-nx16` | 535,590 | 10,547.5 | **2.575** | 2.231 |
| `w5-ffn_down-skinny` | 460,280 | 7,432.9 | **1.815** | 1.357 |

Per-USC figures vary by under 1% across the 20 cores, so the aggregate hides no imbalance.

**Measured vs inferred.** The accumulator values, the 4096-tick window and the ratios are
measured. Dividing by 4096 to get "simdgroups per core" assumes the accumulator adds
residency once per tick; that is **inferred** from the window matching `CountPeriod` exactly.
Every ratio below is independent of it.

### What it says

**1. `nxpsg=16` is a dispatch-geometry win, measured.**

| pair | nx8 | nx16 | ratio |
|---|--:|--:|--:|
| `w3-ffn_down-ext` | 2.461 | 2.869 | **1.166** |
| `w4-ffn_down-ext` | 2.481 | 2.873 | **1.158** |
| `w4-attn_q-ext` | 2.075 | 2.575 | **1.241** |

`width4-verify.md` had this by elimination. Doubling the grid puts ~16% more simdgroups in
flight per core.

**2. Occupancy is flat between width 3 and width 4.** At matched `nxpsg`: 2.461 vs 2.481
(+0.8%) at nx8, 2.869 vs 2.873 (+0.1%) at nx16. The width-4 cost is **not** an occupancy or
scheduling effect. `perf/width4-limiter.md` takes it from there.

### Refuted: `Derived Counter Sample Data` is not the GRC counter stream

The `GPRWCNTR` series `perf/aps-samples.py` parses are **not** the 35 enabled GRC counters:

- **Cardinality never matched and was never stable.** 19 series on `w5-ffn_down-skinny`, 22 on
  `w3-ffn_down-ext-nx8`, against a GRC set of 35 that is byte-identical across all ten
  captures.
- **`field3` only takes 0..6.** It cannot index per-source counter lists of 10, 13, 10 and 2.
- **The file carries no identity for them**: that entry's `Derived Counters Info Data` is an
  **empty dict** and its `Counter Info` has **215** keys, not 35.
- ~~Slot 0 carries only odd `f3` and slot 1 only even.~~ Only on `w5`; on
  `w3-ffn_down-ext-nx8` slot 1 carries all of 0..5. A small-sample artifact.

**And `aps-samples.py`'s framing is wrong.** Records are not a uniform 64 bytes: the stride is
constant within a blob but differs between blobs - 64, 128 or **352** - because a 64-byte
header can be followed by a payload (36 u64 for the 352-byte records, 8 u64 for the 128-byte
ones). The magic check keeps it from inventing records, so its series are real, but it drops
every payload.

**The same `GPRWCNTR` framing is, however, exactly right for the `RDE_0` / `BMPR_RDE_0`
blobs** - those are the non-USC GRC counter stream, 10 u64 of payload for BMPR_RDE_0, one
lane per counter. `perf/aps-dram-bandwidth.py` reads DRAM bandwidth straight out of it. So
the format was worth cracking; it was pointed at the wrong container.

## Next

The counter question is answered; what is left is optional depth.

1. **A named derived value rather than a raw accumulator.** `-loadCounters:` still returns NO:
   `parseData` succeeds and `numAPSRawCounters` reads **139**, but `-apsRawCounterNames` stays
   empty and `-aggregateAPSCounters:` returns NO. The C path already gives values, names and
   timestamps, so this only buys Apple's own normalization.
2. **Per-kick or per-encoder attribution** via `agxps_aps_timing_analyzer_*`, which would
   attribute occupancy to a single dispatch rather than a whole replay.
3. ~~**The GTMio path.**~~ Dead - names and values both came from `libagxps`.

### Issue and ALU raw counters, by name (recovered 2026-08-25)

The 2026-08-25 morning session read "instruction issue/tick" and the "sum of four ALU raw
inputs/tick" but did not record how; the join had to be re-derived the same day. Recipe,
so it does not get lost again: `agxps-probe.py --find "<derived name>"` gives the derived
ident; `raw_used_by(gpu, ident)` gives raw idents; `c_name(raw_ident)` gives the 64-hex
name `aps-usc-values.py --counter` accepts. Values divide by ticks/sample (~4096).

| counter | raw ident | name for --counter |
|---|---|---|
| Instruction Issue Utilization (derived 183014) | 102773 | `7FD8B674D9FE018B3D64EA31CB94787780CD12317B2764B9BAFB60C975CDC8EB` |
| ALU Utilization input 1 (derived 184162) | 102801 | `3476066F46CC277DE7616AAAD8FCDF2C28DA42293B231F74A62159EB6EDAC78C` |
| ALU Utilization input 2 | 102805 | `295D65BB175E4E4EEF9003E008E093043C9B8CE43190BE0A2D8F1771F9837033` |
| ALU Utilization input 3 | 102797 | `AA1E812506867A5F2C54D3BA3268DB5C4BB2C6B0E4F500340DD23C4E1E637D9D` |
| ALU Utilization input 4 | 102793 | `79E88035C9BC883D403F17831B8C9264E643C6B76E9B3C1451B49B0F672C32BF` |

| Threadgroup Memory L1 Load Bandwidth (derived 186473) | 109177 | `0D32627A10C5DB983D8E5DF466E154001CBB4E945D22A4E280A08C8D0F1701A7` |
| Threadgroup Memory L1 Store Bandwidth (derived 186512) | 109179 | `B6C42286FD0AB628E79B9D4F5F10DC31C2DFD460570800BEB5C9E53651363C7D` |

"Sum of four ALU raw inputs/tick" = the four ALU inputs added. First used in
`m4-width4-ilp.md`; cross-kernel table in `verify-width-instruction-economy.md`. The
tg-L1 rows are transaction rates (multiple lane accesses per instruction), so they are
not directly comparable to issue/tick.
