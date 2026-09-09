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

~~**The final MIR is shorter than the native stream.** q4_0 w5: 446 MIR vs 488 decoded;
the difference is inserted by the assembly printer (scoreboard waits).~~ Superseded the same
day: the difference is the preamble program plus nop padding, and the body maps one to one
(see "Alignment" below).

**q6_K w1 in this light** (the "fewer instructions, more issue" case from the census): per
loop, 12 single-byte loads (`load (s8)`), 12 64-bit address adds, 44 bit-field extracts,
37 uniform-register touches, against 35 f32 FMAs. The q4_0 w5 loop is 160 f16 FMAs against
4 s32 + 4 s16 loads. ~~Class mix, not count, is the whole story here.~~ Refuted by the
width series below: the class mix is worth 10-25%, the width-1 ILP regime the rest.

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

## Alignment and the first dataset (same day)

`perf/agx-mir-align.py`: after the preamble program and its run of 2-byte `0600` nops (body
starts at the next 64-byte boundary, 0xc0 on every ggml kernel so far), the native body maps
**one to one, in order** to the final MIR: 446/446 (q4_0 w5), 302/302 (q6_K w1), 402/402
(q4_0 w4). Nothing is inserted after the last dumped pass; the 4/6/8-byte encodings of one
opcode are compression forms and the wait/scoreboard state travels in the instruction words
(the large immediates such as 536870944 = 0x20000000). The census profile rows join by
offset (the profiler lists the final stop with size 0).

`perf/agx-cost-dataset.py` ran the join over the four Sep 06 census snapshots: 34 profiles
of 7 kernels (q4_0/iq4_xs/q4_K/q5_K SoA w4, cpy, rms_norm, swiglu). 69 profiled kernels
were skipped because the translator needs their function constants (mul_mm, FA, GDN, cvt,
ssm_conv, bin_fuse) - the census does not record the values; they have to come from
`ggml-metal-device.m`'s pipeline setup. Output in `build-air/dataset/` (not committed).

**What the join established (verified per instruction, 10,612 instructions, zero residual):**

1. **The profiler's per-instruction `cost` (issue) column is `executed x w(opcode) x k`.**
   `w` is a fixed per-opcode weight, identical in every kernel and every capture: 1 for f16
   arithmetic, moves, branches, compares and 32-bit uniform moves; 4 for 32-bit shifts,
   bit-field extracts, 32-bit uniform ALU and f32 unary ops; 6 for opcode 16842 (a
   16->32 convert form); 8 for the 64-bit register-pair ops (address adds, 64-bit uniform
   loads); **0 for every device load and store and for stop.** `k` is one constant per
   kernel. So within a kernel the "hot instruction by issue" ranking is a static statement -
   count times a table - not a measurement. The `cost2` (stall) column varies by kernel for
   the same opcode (0.01-2.4x for the same uniform move, 2.6-22x for 16842) and is the
   measured quantity. Per-kernel issue vs stall shares remain meaningful (the kernel's busy
   time is measured; only its apportioning over instructions is static). This corrects the
   reading recipes in `skills/metal-gpu-profile` and the census's "hot instruction" rows.
2. **The weight table is not a better time predictor than counting.** Across the 9 mv
   profiles with timings, `us per executed instruction x issue share` spreads 5.3% (the
   existing skill rule); `us per weighted unit` spreads 18.6%, because the iq4_xs kernel's
   LUT loads carry weight 0. Giving loads a weight of 4 brings it to 6.6%, uniform-plus-
   loads-4 to 8.7%: with 4 kernel families the data cannot rank these. The table is Apple's
   apportioning model, not a demonstrated hardware cost table. Do not use it as one until
   the dataset has the K-quant w1/w2, FA, mul_mm and GDN kernels in it.
3. **What q6_K w1 looks like at this level**: per loop 12 `load (s8)`, 12 64-bit address
   adds (weight 8 each under the profiler's table), 44 bit-field extracts, 37 uniform-register
   touches, 35 f32 FMAs. Under the table it is 2.5x the issue units of the q4_0 w4 loop per
   FMA; under plain counting it is fewer instructions. Its measured stall column per
   instruction (not yet captured on the current build) is the number that decides between
   the two readings.

## Width series (2026-09-09 afternoon): the per-instruction price is the ILP regime

Captured on the prod build (c19acef9a) at 17408x5120, `run-ud-soa-profile.sh`, uncaptured
timings from two `test-backend-ops perf` runs each
(`kvquant-experiments/profiles/width-series-sep09`, joins in `build-air/mir/*.join.json`):

| arm | kernel | us/run | executed/dispatch | **us per M executed** | issue / stall % |
|---|---|---:|---:|---:|---:|
| q4_0 w1 | mul_mv_q4_0_f32_di (native) | 205.7 | 11.53M | **17.8** | 68.9 / 31.1 |
| q4_0 w2 | mul_mv_ext_q4_0_di_f16_r1_2 | 258.0 | 23.01M | **11.2** | 86.3 / 13.7 |
| q4_0 w4 | mul_mv_q4_0_soa_w4_r4kp_v3 | 220.0 | 25.62M | **8.6** | 88.0 / 12.0 |
| q4_0 SoA w1 | mul_mv_q4_0_soa_w1 (Q4_0_SOA type) | 205.4 | 13.06M | **15.7** | 98.5 / 1.5 |
| q4_0 SoA w2 | mul_mv_q4_0_soa_w2 | 206.7 | 19.61M | **10.6** | 95.1 / 4.9 |
| q6_K w1 | mul_mv_q6_K_soa_w1_v1 | 299.5 | 15.30M | **19.6** | 97.3 / 2.7 |
| q6_K w2 | mul_mv_q6_K_soa_w2_v1 | 308.6 | 26.19M | **11.8** | 96.1 / 3.9 |
| q6_K w4 | mul_mv_q6_K_soa_w4_v1 | 324.6 | 34.19M | **9.5** | 87.0 / 13.0 |

1. **Time per executed instruction roughly halves from width 1 to width 4, for both
   formats.** q4_0 SoA 15.7 -> 10.6 -> 8.6 (native/ext forms 17.8 -> 11.2), q6_K 19.6 -> 11.8
   -> 9.5. Width adds independent accumulators per thread (ILP), nothing else; the instruction
   mix of each format barely changes with width. Width 2 is free for q4_0 (206.7 vs 205.4 us)
   and nearly so for q6_K (+3%).
2. **q6_K costs 10-25% more per instruction than q4_0 SoA** (1.25 at w1, 1.11 at w2, 1.10 at
   w4), although its stream is byte loads, 64-bit address adds and bit-field extracts where
   q4_0's is f16 FMAs. The instruction *class* mix is the second-order term; the regime term
   is 1.8-2.1x. The q6_K w1 gap to the fleet's 7.6-8.6 us/M rule is mostly the width-1
   regime (the rule was measured on w4/w5 kernels); the fat address mix is the remaining
   quarter at w1 - the September 9 morning reading, which put all of it on the mix, is
   corrected in `perf/ud-remaining-quants.md`.
3. **The profiler's "issue" bucket contains the dependency bubbles.** q6_K w1 shows 97.3%
   issue / 2.7% stall while running at 2x the per-instruction time of its own w4 form; stall
   *rises* with width (2.7 -> 3.9 -> 13.0%) as the ALU gets genuinely busy and memory waits
   surface. "Issue-bound at 97%" therefore means "not waiting on memory", not "the issue port
   is saturated". Where measured stall lands: on the first consumer of a load (shift/mask
   right after the byte load), 6.5% of it on load instructions themselves at w1.
4. Consequence for the cost model: time = executed x price(ILP regime) x (1 + small class
   term) + memory waits. The regime term (~2x between w1 and w4) dwarfs the class term
   (1.1-1.25x between the most different formats we have). The lever for the width-1/2 decode
   kernels is independent work per thread (more rows or columns per thread, software
   pipelining of the dequant chain), not instruction selection.

## Next

1. Confirm the ILP reading directly: q6_K w1 with 8 rows per thread instead of 4 (more
   independent accumulators, same instruction classes) should move us/M toward the w2 value.
2. Function constants for the skipped kernels (read them out of `ggml-metal-device.m` per
   pipeline name, or record them in the census plan) so mul_mm, FA and GDN enter the dataset.
3. With those in: fit per-class prices against measured kernel time directly (not against
   the profiler's issue column), with loads as a class; compare with the WriteRes taxonomy
   and with the profiler's 1/4/6/8 table.
4. Name the opcode numbers by structure for the ~80 seen so far (a `perf/agx-opcode-names.json`
   that `agx-mir-align.py --names` and the dataset tool already accept).
