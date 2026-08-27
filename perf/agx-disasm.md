# Native AGX instruction decoder

Status: **working structural decoder, validated 2026-08-26 on g16s (M4 Pro), Xcode
26.6 / Metal 32023.883.** The command identifies native instruction boundaries,
prints the exact bytes for each instruction, and reports Xcode's per-instruction
register-pressure value. It accepts either an `applegpu-nt` pipeline image or its
nested native Mach-O and can emit JSON.

It does **not** print AGX3 mnemonics. That is a limitation of the Xcode host API, not
an unfinished parser in this repository; the evidence and exhausted alternatives are
recorded below.

## Use

```sh
perf/agx-disasm.py /tmp/out.gpubin
perf/agx-disasm.py --json /tmp/out.gpubin
```

The script defaults to processor `g16s`. Override it with `--gpu` for another machine.
`XCODE_APP=/path/to/Xcode.app` overrides the default `/Applications/Xcode.app`.

The script automatically re-executes itself with Xcode's `SharedFrameworks` on
`DYLD_FRAMEWORK_PATH`. If macOS strips that variable, run it with a non-SIP Python
interpreter such as one from a virtual environment.

Example output from a 122-byte Metal compute kernel:

```text
# gpu=g16s text_bytes=122 instructions=33
# mnemonics=unavailable (Xcode host helper returns an empty ISA table)
000000: 03000700020000006000         R[ 1]  <unknown>
00000a: 0e000000                     R[ 1]  <unknown>
00000e: 0600                         R[ 1]  <unknown>
...
000076: 0e000000                     R[ 1]  <unknown>
```

Offsets are hexadecimal byte offsets in `__TEXT,__text`. The byte runs are exact and
contiguous; their sizes sum to `text_bytes`. `R[n]` is the register-pressure field from
Xcode's analyzer. In JSON, unavailable mnemonics are `null` and the top-level
`mnemonics_available` flag is `false`.

The input can be produced by the pipeline already used by
`perf/agx-spill-probe.py`: compile Metal to a metallib, translate one pipeline with
`applegpu-nt -N <pipeline.mtlp-json>`, then pass the resulting `.gpubin` here. The
decoder recognizes the outer `__compute` wrapper and extracts the native Mach-O itself.

## Implementation

`GTShaderProfiler.framework` starts Xcode's bundled `GTLLVMHelper` through a private
`GTLLVMConnectionManager` IPC connection. The tool sends the nested AGX Mach-O to the
helper, calls `dumpFileInstructionOutput:`, validates the returned offsets, locates the
native `__TEXT,__text` section, and joins each interval with its bytes. A deliberately
short `/tmp` socket name avoids Darwin's 104-byte Unix-socket path limit.

The private API is unsupported and tied to the installed Xcode build. Fail loudly if
the helper contract changes; do not silently guess instruction lengths.

## Why readable mnemonics are unavailable

All of these routes were checked against the same g16s binary:

- `metal-objdump --disassemble` registers the AGX3 target but reports that it has no
  instruction printer.
- `libapplegpu-nt.dylib` contains the AGX3 instruction-printer implementation, but its
  exported `AIRNTEmitAssembly` entry point is an intentional "not implemented" stub.
- `GTLLVMHelper` recognizes `g16s` and returns stable instruction boundaries. Its full
  `GTAPSBinaryInfo` result contains an ISA string table with one empty string, and its
  text formatter prints `-` for every instruction. Target indices 0-3 and 6, and
  generation values 0-7, produce the same result.
- Xcode's macOS platform analyzer (`DYPMTLShaderAnalyzer_OSX`) leaves its setup,
  disassembly-range, and direct instruction-info methods stubbed. The debugger only
  forwards a platform ISA printer when a private device runtime-state bit is present.
- Enabling the compiler's hidden `AGC_ENABLE_STATUS_FILE` logger writes AIR/LLVM IR but
  the offline translator exits with signal 11 immediately after `SimplifyGenericIR`,
  before its "compiler assembly" section. Single-threaded, stderr, per-object, and
  explicit-client modes do not avoid the fault, so this is not safe tooling.
- The public Mesa and Dougall Johnson decoders target the older AGX2/G13 encoding. They
  misdecode g16s AGX3 bytes and cannot be used as a fallback.

A future readable disassembler needs either an AGX3 decoder or an Xcode release that
populates the helper's ISA string table. Until then, this tool is useful for exact code
layout, instruction counts, byte-level diffs, profiler offset joins, and register-
pressure analysis, but it should not be represented as a mnemonic disassembler.
