# Decoding the per-instruction shader profile - de-risked end to end

Status: **DONE 2026-08-27, except mnemonics.** The per-instruction profile decode is
complete and shipped as **`perf/shaderprof-table.py`**: for every kernel in an archived
replay it emits each native instruction's offset, size, per-type register pressure,
execution count, and the profiler's issue/stall time shares - headless, ~8 s per
archive. Validated on the r2_sumy capture: 315 live instructions, exactly the known
aggregate, and executed counts sum bit-for-bit to the binary's `instructionExecuted`.
First customer delivered: `perf/skinny-stall-attribution.md`. Readable ISA text remains
blocked; `perf/agx-disasm.py` and `perf/agx-disasm.md` record the working structural
decoder and the exhausted mnemonic routes.

## The chain, all verified live

1. `GTShaderProfiler.framework` (inside `GPUDebugger.ideplugin`) loads via ctypes. Its
   bundled LLVM recognizes AGX instruction boundaries and processing prints llvm-mca
   warnings for processor `g16s-b1`. It does **not** expose readable AGX3 instruction
   text on a public macOS host.
2. Our wrapper's `raw/` dir (streamData + 20x Counters/Timeline/Profiling `_f_N.raw`)
   is a complete `.gpuprofiler_raw` bundle. **Do NOT unarchive the top-level streamData
   file for this - its `shaderProfilerData` array is empty.** The whole bundle loads
   with `+[GTShaderProfilerStreamData dataFromArchivedDataURL:<raw dir URL>]`, which
   attaches the sample payloads.
3. `-[GTShaderProfilerStreamDataProcessor initWithStreamData:llvmHelperPath:]`
   (helper = `.../GPUToolsPlatform/PlugIns/GTLLVMHelper`), then `processStreamData` +
   `waitUntilFinished`, then `result` -> `GTShaderProfilerProcessedData`
   -> `shaderProfilerResult` -> `GTMioShaderProfilerResult`.
4. In the result: `pipelineStates` resolve to real kernels by NAME
   (`shaderFunctions` values respond to `name`: measured `kernel_cpy_f32_f16` = objectId
   8, `kernel_mul_mv_q4_0_soa_w4_r2_sumy` = objectId 9, matching
   `gpuprofiler-stats.py`'s pipeline numbers). `shaderBinaries` is a dict of 567
   `GTMioShaderBinaryData` objects of ~33-64 instructions each: **the profiler works by
   instrumenting the kernel into small trace segments**, not PC sampling. After the
   full-bundle load the segments carry data (`costCount` 68-195; one segment measured
   `instructionExecuted` = 480).

## The API surface on GTMioShaderBinaryData (the prize)

Per instruction: `isaForInstructionAtIndex:(count:)` (currently always `-`), `instructionCosts`,
`instructionCostsForPipelineState:/ForEncoder:/ForDraw:`, `addressForInstructionAtIndex:`,
`registerCountForInstructionAtIndex:type:`, `debugRangeForInstructionAtIndex:`.
Per line: `costForLine:fullPathIndex:scope:scopeIdentifier:cost:numInstructions:`,
`enumerateLinesForFile:enumerator:`, `debugLocations`/`debugStrings` (the compiler
Remarks in the stream carry `DebugLoc` -> source lines). Segment plumbing:
`usedInPipelineState:`, `enumerateTraces:`, `traces`. Sibling classes:
`GTShaderProfilerMCABinary generateAssemblyContent` / `generateAPSAssembly` (format a
pre-existing cached ISA vector; they do not generate mnemonics),
`GTShaderProfilerBinaryAnalysisResult
analyzeBinary:targetIndex:isaPrinter:` (instructions, clauses, branch targets,
per-instruction register pressure).

## The cost decode (2026-08-27, the part that was open)

All of it read straight off `method_getTypeEncoding` (`perf/shaderprof-typedump.py`
dumps any class's encodings; no guessed signatures - a wrong argtype kills the process
silently, see gotchas).

- `instructionCosts` returns a C array of 304-byte `GTMioCostInfo` structs, one per
  instruction: a 16-byte context `{uint16 scope, uint16, uint32 slot, uint64}`, then
  `double cost + double[10]`, `double cost2 + double[10]`, `uint64 samples +
  uint64[10]`, then 3 uint64. `costs` is the same struct with summary entries mixed in:
  entry 0 has scope=3 and holds the whole-binary totals; instruction i is the entry
  with slot=i+1 (scope=4). `instructionInfo` is 28-byte `{4x uint32, 6x uint16}`,
  `traces` 40-byte `{3x uint64, 2x uint32, uint16}`.
- **Semantics, established across three captures:** `samples` = times the instruction
  executed (sums exactly to `instructionExecuted`); `cost` = the instruction's share of
  TOTAL capture GPU time spent issuing/busy; `cost2` = its stalled share. cost+cost2
  over all kernels of a capture sums to 100.0. The 10-wide slot arrays hold the same
  value at one index (2 in every capture seen - the compute data master); the summary
  entry's trailing uint64s look like begin/end tick totals.
- **The join:** each pipeline state owns 3 binaries. `usedInPipelineState:` takes the
  PS **objectId** and works; `binaryKeys` is a DICTIONARY keyed by program type (its
  value was one aux binary in testing - do not use it to find the main binary);
  `allBinaryKeys` has all three. The main binary is the one with the largest
  `instructionExecuted`; the other two are ~51-64-instruction preamble programs
  (`_agc.main.constant_program`-shaped, executed once per threadgroup).
- **Traps:** `instructionCosts` on a segment with `costCount` 0 returns a pointer to
  UNINITIALIZED memory, not nil - check `costCount` first. Binary dict keys are NOT
  stable across processing runs of the same archive (384 one run, 391 the next) -
  identify binaries structurally, never by key. The master binaries' `traces[].q2`
  looks like a PS objectId on instrumented segments but is garbage on masters.
- `instructionExecuted > 0` selects the ~6 real binaries out of ~567 segments; the
  rest are the instrumentation trace segments the profiler splits kernels into.

Validation on r2_sumy: 315 of 333 decoded instructions live = the known
315-instruction aggregate; executed sums match `instructionExecuted` exactly on every
binary in three captures; cost totals per kernel (80.0 mv + 5.1 cpy, rest
uninstrumented) are consistent with the capture's time split.

## What remains

- Readable AGX3 mnemonics require a new decoder or a future Xcode helper that populates
  its ISA string table; further calls within the current framework do not generate it.
  The table's offsets are exact, so mnemonics can be joined on later.
- Unused surfaces if ever needed: `costForLine:`/`debugLocations` (source-line
  attribution; needs debug info in the capture), `instructionCostsForEncoder:/Draw:`
  (per-encoder/draw splits of the same structs).

## Gotchas already paid for

- ctypes introspection of the ObjC runtime needs explicit argtypes on
  `class_copyMethodList`/`method_getName`/`sel_getName` or pointers truncate to 32 bits
  and the process dies silently mid-probe. Archive artifacts BEFORE risky introspection.
- The saved top-level streamData unarchives fine with the legacy
  `unarchiveObjectWithFile:` and processes without error - it just yields empty costs
  and only the replayer's instrumentation binaries. A plausible-looking dead end.
- `-[GTShaderProfilerProcessedData archiveToURL:error:]` returns YES but wrote no file
  at the given path in the probe - do not rely on it yet.
