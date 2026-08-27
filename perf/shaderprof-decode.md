# Decoding the per-instruction shader profile - de-risked end to end

Status: **profile ingestion works; readable ISA attribution is blocked (updated
2026-08-26).** The per-line/per-instruction profile that Xcode's GUI shows exists in
our archived replays and is reachable headless. Every profile stage runs from
Python/ctypes with no Xcode UI: load framework -> ingest archive -> process -> named
kernels with per-segment cost slots. The earlier version of this note incorrectly
treated instruction-boundary analysis as readable disassembly. Xcode's public-host
helper returns `-` for every mnemonic; `perf/agx-disasm.py` and
`perf/agx-disasm.md` record the working structural decoder and the exhausted mnemonic
routes.

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

## What remains

- Read the cost VALUES: `instructionCosts` shape not yet dumped (likely NSArray or a
  C-array accessor); `costForLine:...` has out-params - prototype carefully. The
  helper-confirmed offsets and bytes from `perf/agx-disasm.py` can be used for the join,
  but there are no mnemonics.
- Join segments -> kernel with `usedInPipelineState:` and emit a per-instruction
  (offset, bytes, cost, samples) table for one kernel, validated against the
  known aggregates (r2_sumy: 315 instructions, issue/tick 1.77).
- Then point it at a SKINNY capture: the single-stream-contention question from
  `skinny-staging-refuted.md` is the first customer.
- Readable AGX3 mnemonics require a new decoder or a future Xcode helper that populates
  its ISA string table; further calls within the current framework do not generate it.

## Gotchas already paid for

- ctypes introspection of the ObjC runtime needs explicit argtypes on
  `class_copyMethodList`/`method_getName`/`sel_getName` or pointers truncate to 32 bits
  and the process dies silently mid-probe. Archive artifacts BEFORE risky introspection.
- The saved top-level streamData unarchives fine with the legacy
  `unarchiveObjectWithFile:` and processes without error - it just yields empty costs
  and only the replayer's instrumentation binaries. A plausible-looking dead end.
- `-[GTShaderProfilerProcessedData archiveToURL:error:]` returns YES but wrote no file
  at the given path in the probe - do not rely on it yet.
