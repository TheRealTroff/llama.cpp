# Reaching the AGX backend inside applegpu-nt (2026-09-09, OPEN)

Purpose: the GPU cost-model plan (why kernel speed stays unpredictable after ~50 kernels)
needs per-instruction *classes* for every instruction the profiler attributes stalls to.
The structural decoder (`perf/agx-disasm.py`) gives offsets and bytes but no mnemonics.
This note records what the Metal translator itself can be made to say. Branch
`exp/agx-cost-model`, tools `perf/agx-nt-debug.sh` and `perf/agx-nt-opt.py`.

## The registry split (why `-mllvm` never reached the backend)

- `usr/bin/air-nt` is a shim. The real launcher is `usr/metal/32023/bin/air-nt` (77 MB, the
  AIR-side LLVM). The AGX3 backend is `usr/metal/32023/lib/libapplegpu-nt.dylib` (110 MB) for
  macOS 26 targets; `libapplegpu23/24-nt` serve older OS versions (config in `lib/air-nt/config.yaml`).
- Each half is a separately linked LLVM with its own `cl::opt` registry. `-mllvm` parses in the
  launcher: backend-only options are "Unknown command line argument"; options present in both
  registries (`print-after-all`, `print-regusage`, `print-detailed-perf-diags`) are accepted and
  silently affect only the AIR side. `-mtranslator` is a whitelist that rejected every name tried,
  including `-help`. No environment variable reaches the backend registry (the `AGC_*`/`AGX_*`
  knobs are a different, hand-written mechanism).

## Method: flip the option object in memory under lldb

1. `perf/agx-nt-debug.sh` copies the launcher and the backend dylib to `~/.cache/agx-nt-debug`
   (symlinks for everything else) and ad-hoc re-signs the two. The shipped binaries are
   Apple-signed and lldb is refused ("Not allowed to attach"); the copy translates byte-
   identically (decoded stream SHA checked on the toy kernel and `kernel_mul_mv_q4_0_soa_w5_r4h`).
2. `perf/agx-nt-opt.py` launches it under lldb with a breakpoint on `^AIRNT` exports, stops at
   the first call into the backend module (`AIRNTGetLLVMVersion`, before any pass runs), finds
   the option's name string in `__cstring`, finds the one 8-byte pointer to it in `__DATA`
   (that is the `cl::Option::ArgStr` field), checks that the next StringRef is the help text,
   and writes the value.
3. **Value offset = ArgStr field + 112 bytes.** Calibrated by dumping the launcher's
   `print-after-all` object with the flag off and on: exactly two words differ, the
   occurrence counter (+56 from the object start) and the value byte (ArgStr + 112).
   Toolchain 17.6.109 / metal 32023; re-calibrate the same way on a new toolchain.
4. A hardware read/write watchpoint on the value byte tells whether the pipeline consults the
   option at all (`agx-nt-opt.py watch`).

## What came out

**Backend `print-after-all` works.** 280 dumps for the toy kernel: the AIR-level pipeline
(~80 passes) then GlobalISel (`irtranslator`, `legalizer`, `agx3-instexpand`,
`instruction-select`) and the machine pipeline: uniform folding, common-store backfiller,
machine-cse, sink, flag-def hoist, peephole, constant merger, **nopifier**, early block
placement, phi-elim, two-address, coalescing, `machine-scheduler`, rematerialize, LU placement,
interface-reg alloc, `greedy` + rewriter, flag-to-GPR spiller, machine-cp, prolog/epilog,
post-RA peephole, `post-RA-sched`, cfg-lower, expand-pseudos, dis2x2, RLD promotion,
fence placement, starvation-free execution, ROC/WB cache control, **clique scheduling hints**,
late SWWA. `agx-nt-opt.py passes` prints the list; `mir` extracts the last machine-function
dump.

**The final MIR is readable except for opcode names.** Registers (`$r64`, 16-bit halves
`$r70h`/`$r59l`, 64-bit pairs, quads), operand width immediates (16/32), memory operands
(`load (s8)`, `load (<8 x s16>)`, address space, tbaa), flags (`$flag0`), special registers
(`$sr_tg_x`, `$sr_simd_elem`), branch targets. Opcodes print as numbers: the TableGen
instruction-name table is stripped (no `G_ADD`, no `INLINEASM` strings anywhere in the dylib).
The numbering is one enum, stable across kernels in the same toolchain, so numbers are usable
as class labels once named by structure. First readings:

| opcode | reading | evidence |
|---:|---|---|
| 2210 | fma, f16 operands (`rNl/rNh`, width 16) | 160/446 of the q4_0 w5 kernel |
| 2190 | fma, f32 operands (width 32) | 35/302 of q6_K w1 |
| 998 / 862 / 776 | f32 mul / f16 mul / f16 add-immediate | dequant chains |
| 17013 / 17016 | bit-field extract (width, shift immediates 8,4 / 6,2) | q6_K high/low planes |
| 11179 / 11182 / 423 / 426 | shift/mask forms | |
| 12646 / 12682 / 12709 | device load s8/s16 / s32 / 8 x s16 | memory operands |
| 13324 / 17229 | stores | |
| 10282 / 10826 / 10793 | uniform-register moves/loads (`agx3-addrspace-uniform`) | 37/302 in q6_K w1 |
| 10268 / 10267 | 64-bit integer add (register pairs) | q6_K address math, 12 per loop |
| 554 / 555 | mov immediate 32 / 16 | |
| 14059 / 14060 | get special register | |
| 10370 / 10372 / 11322 | compare to flag | |
| 582 / 578 / 577 / 463 / 459 / 684 | branch-on-flag / pop / jmp / stop | |

**The final MIR is shorter than the native stream.** q4_0 w5: 446 MIR vs 488 decoded
(3690 B); q6_K w1: 302 vs 335; toy fma: 36 vs 63. The difference is inserted by the
assembly printer after the last dumped pass (scoreboard waits, most likely the 2- and 4-byte
encodings). Aligning MIR to decoder byte runs is the open item; the per-opcode encoding
sizes should make it a deterministic sequence alignment.

**q6_K w1 in this light** (the "fewer instructions, more issue" case from the census): per
loop, 12 single-byte loads (`load (s8)`), 12 64-bit address adds, 44 bit-field extracts,
37 uniform-register touches, against 35 f32 FMAs. The q4_0 w5 loop is 160 f16 FMAs against
4 s32 + 4 s16 loads. Class mix, not count, is the whole story here.

**Static simulator: present but not in the pipeline.** `agx3-static-sim` ("AGX3 Static
Performance Model and Simulator") is registered and would print TotalIssueTime,
TotalShaderLatency, WaitAndStallTime, TotalCycleCount per clique per TEC, F16/F32/Complex/
Immediate stall cycles, integral GPR/register/DL0 pressure, clause counts, result-bus
forwards, skid hits. With `print-agx3-static-sim-stats` set to 1 from the first backend call
to exit, a read watchpoint on the value never fired: nothing in the compute pipeline consults
it. It is reachable only by adding the pass (a code path we do not have) or through whatever
Xcode-side entry point uses it. Parked.

**Scheduling model.** ~80 `WriteRes` class names are in the dylib (WriteF16, WriteF32,
WriteF32Math, WriteINT, WriteINT64, WriteIMM, WriteMOV, WriteSEL, WriteCND, WriteBIT,
WriteConvert, WritePACK3/4, WriteGMEM{,StackImm,StackReg,UniformImm,UniformReg}, WriteLMEM,
WriteQUAD, WriteSIMD, WriteRCP/RSQRT/FEXP/FLOG/SINC, WriteITR1..16, WriteITRP1..16). The
latency/throughput tables behind them are not lifted yet (IDA job). They are Apple's prior
for the class costs we are about to fit.

## Other backend knobs now reachable (untested)

Any bool/int `cl::opt` in the backend: `agx3-new-scheduler`, `agx3-post-scheduler`,
`enable-cmw-allocator`, `disable-agx3-fma-contraction`, `agx3-tmp-reg-limit` (= the
`AGC_TEMP_REGS_IN_BYTES` route in registers), `print-after=<pass>` (string; needs a
StringRef write, not implemented), `print-regusage`. A patched dylib (the copy is ad-hoc
signed, so a byte patch of an option's default is loadable) would make any of them
persistent without lldb.

## Next

1. MIR <-> decoder alignment (per-opcode sizes), then join with the census `instr.json`
   per-instruction issue/stall rows: that is the regression dataset for class costs.
2. Name the opcode numbers by structure for the ~60 that the census kernels use.
3. Fit issue cycles = sum(class_i x cost_i) over the census kernels; validate on held-out
   kernels; compare with the WriteRes taxonomy.
