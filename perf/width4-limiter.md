# What actually limits us at width 4: not bandwidth, and not measurably the tile

Status: **one hypothesis refuted, the other not testable with the captures on disk.** Written
2026-08-23 from the ten aug23 replays, using `perf/aps-dram-bandwidth.py` and
`perf/aps-usc-values.py`. Everything below is measured unless it says otherwise.

> **SHAPE CORRECTION 2026-08-23:** the `w4-attn_q-*` captures here are of `m=3072, k=5120`,
> a shape **no tensor in this model has** - the real `blk.attn_q.weight` is (5120,12288).
> The capture pair is still a valid matched pair of a real kernel, but it is not "the attn_q
> projection". See the banner in `width4-verify.md` and the corrected shape set in
> `tests/test-backend-ops.cpp`.

## The verdict, up front

| | claim | outcome |
|---|---|---|
| **(b)** | weight-bandwidth bound - columns are free until memory saturates | **refuted** |
| **(a)** | tile waste - `kernel_mul_mm_skinny` issues 8-wide MMAs at width 4 | **not testable here** |

At width 4 the GPU sustains **127.6 GB/s against a 273 GB/s peak - 47%**. At batch 1, on the
same weights, it sustains **252.4 GB/s - 92%**. So the memory system demonstrably *can*
saturate on this workload, and at width 4 it is less than half busy. The flat cost curve in
`width4-skinny-ab.md` is not a bandwidth-saturation effect.

(a) survives only in the sense that nothing refuted it. Two hard obstacles, both measured:

1. **Every w3/w4 capture runs `kernel_mul_mv_ext_*`, not `kernel_mul_mm_skinny`.** Only
   `w5-ffn_down-skinny` runs the skinny kernel, and it has no matched-width partner. The
   width pairs on disk do not exercise the kernel the tile argument is about.
2. **The M4 has no matrix-unit counter.** `MXU Utilization`, `MXU Limiter`, `MXUOpsIssued`,
   `MxuInstructions` and `MXU Throttle Stall` all exist in the AGX catalogue but are
   **undefined for gen 16** - `agxps_counter_get_ident` returns invalid for every one of them
   at gen 16 / variant 5. They are defined only for gens 17-20. No capture on this machine
   can measure MMA lane utilization, whatever counter set it enables.

And a third reading is now on the table, because it follows from the same numbers: **nothing
is saturated in any capture.** That points at latency and dependency stalls rather than at
either throughput ceiling. See "What the numbers actually say".

> **CONFIRMED BEHAVIOURALLY 2026-08-23 (`ksplit-width34.md`), without a counter.** If the
> kernel is stalled rather than throughput-limited, giving it more independent work along K
> should pay - and it does: cost at widths 3-4 falls as a function of **total K lanes**
> (`nxpsg*kp`), 340 -> 273 us on ffn_down at width 3, saturating at 32-64 lanes and
> regressing at 128. Weight and activation traffic are identical in every cell, so the only
> variable is parallelism. That is the latency reading paying out; it does not tell us
> *which* stall, which is still what the counters are for.

## The kernels each capture ran

This decides which comparisons are legitimate, so it is worth stating plainly. Read out of
`streamData`'s string table:

| capture | kernel |
|---|---|
| `w1-ffn_down-mv` | `kernel_mul_mv_q4_0_f32` |
| `w2-ffn_down-mvnc2` | `kernel_mul_mv_q4_0_f32_nc2` |
| `w3-ffn_down-ext-nx8` / `nx16` | `kernel_mul_mv_ext_q4_0_f16_r1_3` |
| `w4-ffn_down-ext-nx8` / `nx16` | `kernel_mul_mv_ext_q4_0_f16_r1_4` |
| `w4-attn_q-ext-nx8` / `nx16` | `kernel_mul_mv_ext_q4_0_f32_r1_4` |
| `w4-ffn_down-ext-nof16y` | `kernel_mul_mv_ext_q4_0_f32_r1_4` |
| **`w5-ffn_down-skinny`** | **`kernel_mul_mm_skinny_q4_0_f32`** |

## 1. Achieved DRAM bandwidth

`perf/aps-dram-bandwidth.py`. Per-sample rates, so idle time in the replay window cannot
dilute them; "busy" is the mean over the busiest half of the recorded time.

| capture | median | p75 | p90 | p99 | **busy** | **% of 273** | window mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| `w1-ffn_down-mv` | 242.9 | 251.3 | 258.1 | 268.1 | **252.4** | **92%** | 217.9 |
| `w2-ffn_down-mvnc2` | 172.4 | 180.9 | 192.3 | 210.8 | **184.7** | **68%** | 130.2 |
| `w3-ffn_down-ext-nx8` | 146.0 | 150.4 | 152.0 | 160.0 | **151.4** | **55%** | 114.0 |
| `w3-ffn_down-ext-nx16` | 133.7 | 142.8 | 146.8 | 153.3 | **143.0** | **52%** | 106.9 |
| `w4-ffn_down-ext-nx8` | 122.6 | 126.1 | 128.0 | 135.6 | **127.6** | **47%** | 90.5 |
| `w4-ffn_down-ext-nx16` | 122.3 | 124.1 | 126.1 | 133.8 | **125.7** | **46%** | 87.2 |
| `w4-attn_q-ext-nx8` | 97.3 | 99.5 | 100.9 | 104.1 | **100.0** | **37%** | 80.2 |
| `w4-attn_q-ext-nx16` | 87.1 | 89.5 | 94.3 | 103.5 | **91.5** | **34%** | 70.9 |
| `w4-ffn_down-ext-nof16y` | 92.4 | 94.5 | 98.1 | 104.9 | **96.8** | **35%** | 80.0 |
| **`w5-ffn_down-skinny`** | 94.4 | 98.7 | 103.7 | 111.0 | **101.0** | **37%** | 68.1 |

### Why this counter, and why it can be trusted

The catalogue offers three families that could be called bandwidth:

- **`AF Bandwidth` / `BytesReadFromMainMemory` / `BytesWrittenToMainMemory` /
  `MainMemoryTraffic`**, all on `BMPR_RDE_0` indices 0-3 and 7, in groups **`System Memory
  Bandwidth`**. "AF" is the Apple Fabric, the port to DRAM. **This is the one used.**
- `L2 Bandwidth` (`BMPR_RDE_0` 4) and the whole `L1 * Bandwidth` family (`APS_USC` 8), both
  in group **`Internal Memory Bandwidth`**. These are on-chip traffic, not DRAM, and are
  larger than DRAM traffic in every capture - as they must be, which is itself a consistency
  check on the mapping.

Three things make the number trustworthy rather than assumed:

1. **The lane mapping is confirmed arithmetically.** The `BMPR_RDE_0` payload is 10 u64,
   exactly the 10 counters `Limiter Counter List Map` lists for that source. `agxps-probe.py`
   says `MainMemoryTraffic` is index 7 and read/write are 0,1 and 2,3. Measured:
   **lane 7 == lane 0 + lane 2** to within 0.006% on every capture, and to the unit on
   `w1-ffn_down-mv` (448,414,780 vs 448,414,782) and `w2-ffn_down-mvnc2` (exact). Lanes 1 and
   3 are zero throughout; lanes 8 and 9 are fixed clocks (11830 and 8962.5 per record in
   every capture).
2. **The granularity is calibrated, not guessed.** `w1-ffn_down-mv` is the batch-1 case that
   `mv-bandwidth-probe.md` independently measures at **251.3 GB/s**. At 64 B per transaction
   this script returns a busy-mean of **252.4** and a **p75 of 251.3** for that capture. At
   128 B it would return 505 GB/s, impossible against a 273 GB/s part.
3. **The clock is the capture's own.** Header timestamps are in units of 125/3 ns, the
   `Timebase [125, 3]` recorded in `APSCounterData[39]`. Cross-check: the BMPR span times
   125/3 reproduces the processor's own `firstAPSTimestamp`/`lastAPSTimestamp` delta to
   0.02% (100.73 ms vs 100.75 ms on `w3-ffn_down-ext-nx8`).

## 2. Arithmetic-unit rates

`perf/aps-usc-values.py` reads the USC stream; every counter is accumulated over the same
4096-tick window, so a per-tick rate is directly "events per shader-core cycle".

The obfuscated raw counters can be named exactly, by asking the library which raws each
derived counter reads and taking the ones with a single input:

| raw counter | is |
|---|---|
| `79E88035C9BC883D...` | **F16 ALU** (sole input to `F16 Utilization` and `ALU F16 Instructions`) |
| `AA1E812506867A5F...` | **F32 ALU** (sole input to `F32 Utilization` and `ALU F32 Instructions`) |
| `295D65BB175E4E4E...`, `3476066F46CC277D...` | the other two ALU classes (`ALU Utilization` = these four) |
| `7FD8B674D9FE018B...` | **Instruction Issue** (sole input to `Instruction Issue Utilization`) |
| `F89408CC4F2E499C...` | **Instruction Dispatch** (sole input to `Instruction Dispatch Utilization`) |

Per-tick rates, summed over all 20 USCs:

| capture | F32 ALU | F16 ALU | ALU cls3 | ALU cls4 | **ALU total** | Issue | Dispatch |
|---|--:|--:|--:|--:|--:|--:|--:|
| `w1-ffn_down-mv` | 0.847 | 0.000 | 0.785 | 1.178 | **2.809** | 1.256 | 1.241 |
| `w2-ffn_down-mvnc2` | 1.059 | 0.017 | 0.507 | 0.706 | **2.288** | 1.087 | 1.079 |
| `w3-ffn_down-ext-nx8` | 1.430 | 0.071 | 0.929 | 0.675 | **3.106** | 1.569 | 1.551 |
| `w3-ffn_down-ext-nx16` | 1.336 | 0.067 | 0.876 | 0.639 | **2.917** | 1.471 | 1.456 |
| `w4-ffn_down-ext-nx8` | 1.401 | 0.056 | 0.811 | 0.547 | **2.815** | 1.428 | 1.416 |
| `w4-ffn_down-ext-nx16` | 1.353 | 0.055 | 0.791 | 0.536 | **2.734** | 1.388 | 1.376 |
| `w4-attn_q-ext-nx8` | 1.256 | 0.100 | 0.686 | 0.425 | **2.467** | 1.360 | 1.352 |
| `w4-attn_q-ext-nx16` | 1.122 | 0.089 | 0.634 | 0.402 | **2.247** | 1.243 | 1.231 |
| `w4-ffn_down-ext-nof16y` | 1.236 | 0.099 | 0.661 | 0.402 | **2.397** | 1.336 | 1.323 |
| **`w5-ffn_down-skinny`** | 1.556 | 0.011 | 0.387 | 0.367 | **2.321** | 1.349 | 1.338 |

**F16 ALU issue is essentially zero everywhere, skinny included** - 0.011/tick against
1.556/tick of F32, a factor of 140. Whatever `simdgroup_half8x8` compiles to on this part, it
does not appear as F16 ALU issue. *Inferred, not measured:* either the accumulate dominates
and is F32, or matrix ops are counted in a class these counters do not separate. This is a
lead, not a conclusion, and with no MXU counter on gen 16 it cannot be chased further here.

**What is missing, and it is the important gap:** there is no absolute ceiling for any of
these rates in the shipping catalogue, because the `* Utilization` derived counters are
normalized by a constant the library will not hand out without evaluating its expression
graph. So "ALU total 2.815/tick" cannot be turned into "X% of ALU peak". The rates are
comparable *between captures* and not against an absolute ceiling.

## 3. Limiters

The catalogue has a real `Limiters` group and 20 of its members are computable from these
captures. But every one of them is a **normalized** derived counter over the raws above -
`F32 Limiter` and `F32 Utilization` read the identical single raw counter, and differ only in
the normalization. Without evaluating the expression graph there is no way to produce the
"this unit was the bottleneck N% of the time" number the group name promises.

So: **the limiter counters do not answer the question directly.** What they do give is the
mapping above, which is how the raw counters got named at all.

## What the numbers actually say

**(b) is dead.** Bandwidth at width 4 is 47% of peak on the ffn_down `mul_mv_ext` kernel and
37% on skinny, while the very same weights at batch 1 reach 92%. Columns are not free because
memory is saturated; memory is not saturated.

This also settles the point flagged as worth having: **MLX's 95.00 ms/round against our
135.3 cannot be a bandwidth story at width 4 either.** Neither of us can be near the memory
ceiling at that width if we are sitting at 47% of it.

**(a) is untested, not supported.** The width pairs on disk are the wrong kernel, and the
counter that would settle it does not exist on this GPU generation.

**Nothing is saturated.** Bandwidth 47%, ALU issue lower at width 4 than at width 3
(2.815 vs 3.106 per tick), instruction issue lower too (1.428 vs 1.569), and occupancy flat
between the widths (`aps-counters.md` Round 5: 2.481 vs 2.461 simdgroups/core). Every
throughput resource that can be measured is *less* busy at width 4 than at width 3, while the
work per pass is larger. That is the signature of a latency-bound kernel - stalled on
dependencies or on memory latency rather than on any unit's throughput - and it is a third
explanation that fits everything measured so far, including the flat cost curve.

## The experiment that would settle (a)

**A matched skinny width pair** - `kernel_mul_mm_skinny` captured at two widths, ideally 4 and
8. Then:

- if the column tile is fixed at 8, instructions per weight byte are **identical** at both
  widths (ratio 1.000);
- if work scales with width, the ratio is **2.000**.

**The test is already validated on data in hand.** Run against the `mul_mv_ext` kernel, where
width 3 and width 4 differ by construction, it gives ratio **1.247** (nx8) and **1.251**
(nx16) for F32 ALU per DRAM-read transaction, against 4/3 = 1.333 for proportional and 1.000
for a fixed tile. It cleanly separates the two, on the wrong kernel. Point it at skinny and it
answers the question.

Normalizing by DRAM read transactions rather than wall time is what makes this work: it
counts passes over the weights, so captures of different length and different iteration count
compare directly.

## Tooling

- `perf/aps-dram-bandwidth.py` - the bandwidth table above. Pure `plistlib` + `struct`, no
  frameworks, no ObjC. Runs in about a second per capture.
- `perf/aps-usc-values.py` - the USC counter values. Needs `libagxps`; see
  `perf/aps-counters.md`.
- `perf/agxps-probe.py` - names counters and prints which `(source, index)` each needs.

## Do not repeat these

- **Looking for a matrix-unit counter on the M4.** `MXU Utilization`, `MXU Limiter`,
  `MXUOpsIssued`, `MxuInstructions`, `MXU Throttle Stall` are undefined for gen 16. They
  resolve only for gens 17.4-20.3.
- **Using `perf/aps-samples.py` for counter values.** Its `GPRWCNTR` series come from
  `Derived Counter Sample Data`, which is not the GRC counter stream, and its 64-byte record
  stride is wrong - see `aps-counters.md`. The same `GPRWCNTR` framing *is* correct for the
  `BMPR_RDE_0` / `RDE_0` blobs, which is what `aps-dram-bandwidth.py` reads.
- **Feeding RDE buffers to the ObjC processor.** `-addBufferAtRDESourceIndex:...` leaves
  `-numRDESources` at 0 without an `RDERawCounters` config the captures do not carry. Reading
  the blobs directly is both simpler and enough.
- **`agxps_aps_parser_parse` on an RDE/BMPR blob.** It returns a profile-data handle and
  7 bogus counters with one value each, and sets err=2. Those blobs are not APS token streams.
