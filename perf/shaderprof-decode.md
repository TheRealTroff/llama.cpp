# Decoding the per-instruction shader profile - de-risked end to end

Status: **open, but the hard part is done (2026-08-25 late).** The per-line/
per-instruction profile that Xcode's GUI shows exists in our ARCHIVED replays and is
reachable headless. Every stage of the chain now runs from Python/ctypes with no Xcode
UI: load framework -> ingest archive -> process -> named kernels with per-segment cost
slots. What remains is mechanical (below). Probe: `perf/shaderprof-decode-probe.py`
(run with the non-SIP python + `DYLD_FRAMEWORK_PATH` incl.
`GPUDebugger.ideplugin/Contents/Frameworks`; see the script).

## The chain, all verified live

1. `GTShaderProfiler.framework` (inside `GPUDebugger.ideplugin`) loads via ctypes. It
   embeds LLVM with an AGX target: processing our archives prints llvm-mca warnings for
   processor `g16s-b1` - **the framework disassembles AGX machine code in-process**,
   which also makes it the engine for the disassembly tool (#1), not just the profile
   decode (#2).
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

Per instruction: `isaForInstructionAtIndex:(count:)`, `instructionCosts`,
`instructionCostsForPipelineState:/ForEncoder:/ForDraw:`, `addressForInstructionAtIndex:`,
`registerCountForInstructionAtIndex:type:`, `debugRangeForInstructionAtIndex:`.
Per line: `costForLine:fullPathIndex:scope:scopeIdentifier:cost:numInstructions:`,
`enumerateLinesForFile:enumerator:`, `debugLocations`/`debugStrings` (the compiler
Remarks in the stream carry `DebugLoc` -> source lines). Segment plumbing:
`usedInPipelineState:`, `enumerateTraces:`, `traces`. Sibling classes:
`GTShaderProfilerMCABinary generateAssemblyContent` / `generateAPSAssembly` (ISA text
via the embedded LLVM), `GTShaderProfilerBinaryAnalysisResult
analyzeBinary:targetIndex:isaPrinter:` (instructions, clauses, branch targets,
per-instruction register pressure).

## What remains (mechanical)

- ISA strings read `-` until generation is triggered: call
  `mcaBinaryForBinaryKey:` -> `generateAssemblyContent` (or set an isaPrinter /
  check `cachedISAFileURL`) and re-read.
- Read the cost VALUES: `instructionCosts` shape not yet dumped (likely NSArray or a
  C-array accessor); `costForLine:...` has out-params - prototype carefully.
- Join segments -> kernel with `usedInPipelineState:` and emit the deliverable: a
  per-instruction (ISA, cost, samples) table for one kernel, validated against the
  known aggregates (r2_sumy: 315 instructions, issue/tick 1.77).
- Then point it at a SKINNY capture: the single-stream-contention question from
  `skinny-staging-refuted.md` is the first customer.

## Gotchas already paid for

- ctypes introspection of the ObjC runtime needs explicit argtypes on
  `class_copyMethodList`/`method_getName`/`sel_getName` or pointers truncate to 32 bits
  and the process dies silently mid-probe. Archive artifacts BEFORE risky introspection.
- The saved top-level streamData unarchives fine with the legacy
  `unarchiveObjectWithFile:` and processes without error - it just yields empty costs
  and only the replayer's instrumentation binaries. A plausible-looking dead end.
- `-[GTShaderProfilerProcessedData archiveToURL:error:]` returns YES but wrote no file
  at the given path in the probe - do not rely on it yet.
