# The verify-width wall is instruction economy, not memory latency

Status: **measured 2026-08-25.** The first counter-profiled width-7 pass, plus a
cross-kernel comparison that reinterprets every width-4 result in this series. Replays
archived under `kvquant-experiments/profiles/aug25-w7-width-economy/` (new captures) and
`aug25-m4-width4-{profile,latency}/` (prior R2/K2/U2). Method: headless capture + replay
(`skills/metal-gpu-profile`), issue and ALU raws read by GRC name (recovered and now
recorded in `aps-counters.md` - the previous session's method was lost with its chat).

## The table

One machine, one instrument, seven captures. Floors are q4_0 weight bytes at 273 GB/s;
`instr/B` is compiled instructions per weight byte consumed per lane iteration
(mm-skinny's per-iteration byte count not derived - kernel not read at that depth).

| capture | width | us/run | x floor | DRAM busy | inflight (active) | issue/tick | ALU-inputs/tick | regs | instr/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nc2 ffn_down | 2 | 213.5 | **1.16** | **69%** | 2.89 | **1.39** | 3.18 | 77 | **20.8** |
| R2 ffn_down (SoA) | 4 | 334.9 | 1.82 | 54% | 3.29 | 1.95 | 4.11 | 43 | 33.9 |
| K2 ffn_down (SoA) | 4 | ~354 | ~1.9 | 53% | 3.13 | 1.85 | 4.00 | 55 | 29.1 |
| U2 ffn_down (SoA, unroll-2) | 4 | 329.2 | 1.79 | 45% | 3.12 | 1.43 | 3.07 | 73 | 34.4 |
| skinny ffn_down | 7 | 385.5 | 2.10 | 46% | **1.79** | 2.18 | 4.09 | 51 | - |
| skinny attn_q | 7 | 239.4 | 1.85 | 51% | 2.91 | 2.54 | 4.73 | 51 | - |
| skinny gate/up | 7 | 327.0 | 1.78 | 52% | 3.18 | 2.61 | 4.90 | 51 | - |

Reference points: batch-1/plain-mv streams at 87-92% of peak; w8 skinny costs the same
as w7 (+0.2-1.4%, the 8-column tile is full - padding is not the w7 problem).

## What the table rules out

- **Not residency.** Inflight is ~3 for every kernel regardless of grid size (320 to
  2560 simdgroups), register count (43 to 77), or family. The R2G probe
  (`m4-width4-latency.md`) confirmed causally: batching 4-8 tiles per threadgroup does
  not help. One real exception: skinny at ffn_down (m=5120) is **grid-starved** - 160
  threadgroups is not enough to reach even the ~3 plateau, and its 1.79 inflight / 46%
  DRAM / 2.10x floor line up. Worth ~12% there (to attn_q/gateup's level), no more.
- **Not memory latency or bandwidth.** DRAM never exceeds 69% below width 1; the best
  kernel has the SECOND-LOWEST outstanding-load proxy. U2 doubled per-lane loads in
  flight and DRAM went DOWN (54% -> 45%).
- **Not an issue-rate ceiling.** The best kernel issues at 1.39/tick, the worst at 2.61.
  Nobody is pinned at a common cap.
- **Not registers.** This file corrects `m4-width4-latency.md`'s U2 diagnosis: U2's
  inflight (3.12) equals R2's (3.29) despite 73 vs 43 registers. The register growth did
  not cost residency; U2 failed because it did not change instructions per byte.

## What it rules in

**For the mv family, time per weight byte tracks compiled instructions per weight
byte.** nc2 vs R2 vs U2:
instr/B 20.8 / 33.9 / 34.4 -> x floor 1.16 / 1.82 / 1.79. The nc2:R2 ratio in instr/B
(0.61) equals the ratio in x floor (0.63). K2 is the one anomaly (29.1 instr/B but no
faster than R2; its threadgroup traffic may not price like ALU). Reading: at ~3 resident
simdgroups the per-core pipes are saturated by the instruction stream itself - the
kernels are **throughput-bound on instructions, not latency-bound on memory** - so the
bytes/second a kernel can consume is (achievable instruction throughput)/(instr per
byte), and every schedule-level change that left instr/B unchanged (K-split, unroll,
threadgroup packing, more residency) measured ~0. That is the unifying explanation for
the entire `m4-width4-*` refutation series.

Width is expensive because the y-side work per weight byte scales with columns: each
added column adds dots, y loads and address work per byte of weights. That is why
width 2 -> 4 costs +50-77% in every family (`m4-width4-latency.md`) and why batch-1
saturates DRAM: at width 1 the instruction cost per byte is at its minimum.

## Skinny is NOT the same wall (asked and measured, same day)

The law does not extend to the mm kernel, and the counters say why. Skinny's dynamic
instruction economy is excellent - the K-slice-64 body consumes 18 weight bytes per lane
iteration in roughly 90-110 dynamic instructions, ~5-6 instr/B, four times better than
nc2 - because `simdgroup_multiply_accumulate` packs an 8x8x8 MAC block into one
instruction. What it buys with that, it spends on staging. Threadgroup L1 transaction
rates per tick (raws 109177/109179, names now in `aps-counters.md`):

| capture | tg-L1 loads/tick | tg-L1 stores/tick | x floor |
|---|---:|---:|---:|
| skinny gate/up w7 | 3.75 | 3.13 | 1.78 |
| skinny ffn_down w7 | 3.15 | 2.63 | 2.10 |
| R2 w4 | ~1.0 (act.) | ~0.2 | 1.82 |
| nc2 w2 | 0 | 0 | 1.16 |

The faster skinny shape shows the HIGHER tg-L1 rate - at healthy occupancy the staging
path runs flat out, the signature of a saturated port rather than an incidental cost.
Every K-slice pays two hard `threadgroup_barrier`s and a full A-tile + B-tile round trip
through shmem before `simdgroup_load` can feed the MMAs; the software pipeline overlaps
only the dequant, not the staging. So the two families hit two different walls with one
symptom (~50% DRAM): **mv = instruction stream per byte; mm = threadgroup-memory
round-trip**. This also sharpens `ffn-utilization.md`'s run-1/2 reading ("does not
overlap the two") into a named, counter-measured mechanism.

What would move skinny, in candidate order: (1) feed B without shmem - the f16y convert
already materializes y in half precision for the mv path, and a device-direct
`simdgroup_load` of a half B-tile would remove the B stage and one barrier phase;
(2) double-buffer sa/sb so A-staging overlaps the MMA block instead of barrier-
serializing; (3) the ffn_down grid fix (independent, ~12%).

## What would actually move width 4

Cut instructions per weight byte from ~34 toward nc2's ~21. In candidate order:

1. ~~**Fold the -8 offset and scale via the sumy identity** (nc-style) instead of two
   `half4` convert+subtract chains per pack per row - removes converts, keeps dots.~~
   **REFUTED 2026-08-25** (`width4-sumy-fold-refuted.md`, branch `m4-width4-sumy-fold`):
   +15 to +32% per pass in every variant, including the 4-row tile. The convert+subtract
   chains are nearly free; the fold's sumy/pre-scale terms are per-column y-side work,
   which is the very term that scales with width. nc2's instr/B edge is its 2 columns,
   not its offset arithmetic. The replay also refined the law itself: r2_sumy is +16%
   instructions but +32% time because its all-FP32 mix ALSO drops issue/tick 1.93 ->
   1.77 - instr/B predicts time only at comparable issue rates (same shape as the K2
   anomaly above, now with a measured mechanism).
2. **Amortize the q fetch and nibble expansion across all four columns** is already done;
   what is NOT amortized in R2 is address generation per row per pack (`AGen` feeds the
   ALU-input counters; INT ops are 49 of R2's 271). Base-pointer + increment addressing
   (the V2 rewrite pattern from `width4-verify.md` run 2) applied to the SoA kernel.
3. ~~**f16 dot pairs / bf16 activations** - halves y-side operand width per MAC; their
   winning kernel is `_bf16`. The earlier `HALF_PRODUCT` refutation changed rounding on
   the product path; a bf16/f16-pair probe that keeps FP32 accumulation is a different
   cell and unmeasured.~~ **CLOSED 2026-08-25, offline** (`width4-y-operand-width.md`):
   the compiler already folds f16 y into 16-bit FMA source operands (an fp32-y R2
   variant compiles to size-identical code), so there is nothing to halve; bf16 sources
   do NOT fold and cost extra ops per MAC, refuting the bf16 flavor before any build.
4. **Grid fix for skinny at m=5120** (ffn_down only): halve rows-per-tg or split K
   across threadgroups to lift 160 tgs above the residency knee. Bounded ~12% on that
   one shape; independent of the instr/B story.

Candidates 1-3 are one bounded prescreen-gated probe each: the readout is compiled
instruction count per byte FIRST (offline, free), then the benchmark.

## Instrument note

The issue and ALU raw counters are readable by GRC name via `aps-usc-values.py
--counter`; the names and the derivation chain (`agxps-probe.py --find` -> derived ident
-> `raw_used_by` -> `c_name`) are now recorded in `aps-counters.md`, so this table is
reproducible without re-deriving them.
