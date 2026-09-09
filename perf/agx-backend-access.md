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

## Width series (2026-09-09 afternoon): these kernels sit on the DRAM floor

Captured on the prod build (c19acef9a) at 17408x5120, `run-ud-soa-profile.sh`, uncaptured
timings from two `test-backend-ops perf` runs each
(`kvquant-experiments/profiles/width-series-sep09`, joins in `build-air/mir/*.join.json`).
Bytes per dispatch = weights (q4_0 18 B/32, q6_K 210 B/256) + f32 activations + output; floor
at the census's 273 GB/s:

| arm | kernel | us/run | MB | x DRAM floor | executed/dispatch | us per M executed | issue / stall % |
|---|---|---:|---:|---:|---:|---:|---:|
| q4_0 w1 | mul_mv_q4_0_f32_di (native) | 205.7 | 50.2 | **1.12** | 11.53M | 17.8 | 68.9 / 31.1 |
| q4_0 SoA w1 | mul_mv_q4_0_soa_w1 | 205.4 | 50.2 | **1.12** | 13.06M | 15.7 | 98.5 / 1.5 |
| q4_0 w2 | mul_mv_ext_q4_0_di_f16_r1_2 | 258.0 | 50.3 | 1.40 | 23.01M | 11.2 | 86.3 / 13.7 |
| q4_0 SoA w2 | mul_mv_q4_0_soa_w2 | 206.7 | 50.3 | **1.12** | 19.61M | 10.6 | 95.1 / 4.9 |
| q4_0 w4 | mul_mv_q4_0_soa_w4_r4kp_v3 | 220.0 | 50.5 | **1.19** | 25.62M | 8.6 | 88.0 / 12.0 |
| q6_K w1 | mul_mv_q6_K_soa_w1_v1 | 299.5 | 73.2 | **1.12** | 15.30M | 19.6 | 97.3 / 2.7 |
| q6_K native w1 | mul_mv_q6_K_f32 | 291.3 | 73.2 | **1.09** | 19.44M | 15.0 | 95.6 / 4.4 |
| q6_K w2 | mul_mv_q6_K_soa_w2_v1 | 308.6 | 73.3 | **1.15** | 26.19M | 11.8 | 96.1 / 3.9 |
| q6_K w4 | mul_mv_q6_K_soa_w4_v1 | 324.6 | 73.5 | **1.21** | 34.19M | 9.5 | 87.0 / 13.0 |

1. **Every SoA arm is within 9-21% of the DRAM floor; q6_K w1 and q4_0 w1 both stream at
   244 GB/s.** q6_K/q4_0 time ratio 1.46 = their byte ratio 1.46. Width 2 is free because the
   bytes do not change. The "us per M executed" column falls with width only because more
   instructions are executed inside the same memory-bound time - it is not a per-instruction
   price and the first reading of this table (an "ILP regime") was wrong.
2. **The ILP probe confirmed it** (branch `exp/q6k-w1-ilp`, `GGML_MV_UD_W1_NA=2|4|8`, kernel
   `*_soa_w1_na<N>`): N independent f32 accumulator chains per row instead of one, same loads
   and dequant. NA=2: 300 us, NA=4: 302 us (349 vs 302 static instructions), NA=8: 315 us, all
   correct. More instructions, same time: the kernel is not waiting on its FMA chain.
3. **Why the profiler said "issue-bound".** Both q6_K kernels report `Wait instruction count 0`:
   on g16s the scoreboard waits are encoded in instruction bits, so there is nothing for the
   profiler to file under "stall" while a simdgroup waits for a load. Its issue/stall split is
   blind to exactly the wait that dominates a bandwidth-bound decode kernel. The morning's
   "not memory: 96-97% issue" (ud-remaining-quants.md) and the afternoon's ILP reading both
   overrode the census's own first metric for stream kernels, x the byte floor, which had the
   answer at 1.11x. Rule: when time is within ~1.2x of bytes/273 GB/s, the kernel is on the
   floor and no instruction-level reading applies, whatever the issue/stall split says.
4. What is left in these kernels is the 9-21% above the floor (the native q6_K w1 at 1.09x is
   the best of the set; the SoA form at 1.12x is the +4% the morning measured). The
   instruction-class question only becomes answerable on shapes or widths that leave the
   floor: mul_mm, FA, GDN, or these kernels at widths 5+ / small K where bytes stop dominating.

## Dataset with function constants (2026-09-09 evening)

`perf/agx-cost-dataset.py` now resolves function constants from the capture log's loaded
pipeline name (`<base>_key=val_...`, mapped to `FC_<family> + index` per
`ggml-metal-device.cpp`; table `FAMILIES` in the script) and records the census `x_floor`.
Over the four Sep 06 census snapshots plus today's profiles: **106 profiles joined, 7
skipped** (ssm_conv: a constant the name does not carry; the prefill FA at n=512: its
translation has no nop run, alignment rule needs a second form). Output `build-air/dataset2/`.

`perf/agx-cost-fit.py` fits per-class prices against measured time (kernel share of the
capture's cost+cost2 times us/run), excluding kernels under 1.3x the floor. Relative-weighted
fit, 4 classes survive (w1 1.7, load 67, store 304, other 13.8 us per M executed):

| family | profiles | pred/meas median (min-max) |
|---|---:|---|
| mul_mm (acch/n64, 6 formats, 6 shapes) | 33 | 1.01 (0.96-1.03) |
| flash_attn qt f16 | 4 | 1.04 (0.90-1.14) |
| mul_mv w4/w5 (mostly on the floor) | 9 | 1.09 (0.90-1.22) |
| elementwise (cpy, swiglu, rms_norm, bin_fuse) | 18 | 0.59 (0.19-1.20) |
| gated_delta_net | 6 | 0.16 (0.09-0.68) |

Reading: a linear instruction-class model holds within a few percent where the kernel is
throughput-bound with enough parallelism (mul_mm; within one family "us per M executed"
alone has 10% spread, per MMA count 16%). It is wrong by 2-10x for the small and serial
kernels (GDN is a recurrent scan: latency, not throughput; elementwise kernels at 512
tokens are launch-plus-bandwidth). The class prices themselves are not yet physics: the
"other" class holds every unnamed opcode including the real MMA instructions, and the 75
opcodes that appear only in mma-class kernels are unnamed. The integer-side opcode naming
(a solve across kernels against Xcode's INT16/INT32/FP counts) is the blocker for turning
this into a per-class table; the float side is named.

Practical floor calibration: a 37 MB `CPY` streams at 252 GB/s on this M4 Pro (92% of the
273 GB/s figure), so a decode mul_mv at 1.12x the nominal floor is at ~1.03x the practical
one - saturated; the w4 kernels at 1.19-1.21x nominal have ~8-10% left, which is the
compute/stream overlap (a prefetch/double-buffer form is the candidate, not instruction
selection).

## Next

1. Re-run the cost dataset on kernels that are NOT on the byte floor (mul_mm acch n64, FA,
   GDN: function constants via `agx-nt-opt.py --cv`, FC_MUL_MV = 600 nsg/nxpsg/ne12/r2/r3/nr0_v;
   FA and mul_mm constants from `ggml-metal-device.cpp`) and fit class prices against measured
   time there.
2. Add "x byte floor" to the join table output so a kernel on the floor is flagged before any
   per-instruction reading (census already computes it; the profile skill's recipes did not
   check it).
3. Name the opcode numbers (float side done via Xcode's FP32 counts; integer side needs a
   solve across kernels) - `perf/agx-opcode-names.json`.
