# Issue probes: is the width-8 skinny tile issue-bound? (2026-10-26, branch `exp/issue-probes`)

Owner: "I like testing predictions. It's how we learn." The per-type instruction table (`census-groups.md`, "Instructions per
type") read the q4_0 SoA skinny groups at 99-117 K issue-weighted units/us under Apple's per-opcode weights, ~90% of a
20-core x 4-issue port, and called them issue-bound; the K-quant tiles at 48-62 with 15-30% stall were called occupancy/latency
bound. Two predictions followed, each one env knob and one isolated `test-backend-ops perf` arm (`perf/run-issue-probes.sh`,
medians of 3, pick env `GGML_MM_SKINNY_BSPLIT=2`; q5_K under `GGML_MM_SKINNY_GEN=6 Q5K=1 KQ2=1 BSPLIT=2` on the stored
`q5_K_soa` rows = the census kernels by name):

- `GGML_MM_SKINNY_PROBE=1` - no dequant: the packed words reinterpreted as halves (same loads, same MMA, garbage output).
  Capture/replay: **373 -> 258 static instructions, 13.08 -> 6.40 M executed per dispatch (-51%)**, FP16 64 -> 0, 54 -> 55 regs.
- `GGML_MM_SKINNY_PROBE=2` - cached weights: every K slice re-reads blocks 0/1 of its row (same instructions: 375 static,
  13.19 M executed; the weight stream leaves DRAM).
- `GGML_MM_SKINNY_SMEM_PAD=<bytes>` - threadgroup memory the tile never touches: fewer resident threadgroups per core, kernel
  unchanged (host side only, `ggml-metal-ops.cpp` skinny dispatch). Tile smem is 5 KB (SoA) / 5.1 KB (generic).

## Results (isolated, us/run)

| q4_0 SoA, width 8 | 17408x5120 (ffn_down shape) | 5120x17408 (gate/up shape) |
|---|---|---|
| base | 318.5 | 365.4 |
| probe 1: half the instructions, same bytes | 324.3 (+2%) | 388.0 (+6%) |
| probe 2: same instructions, no DRAM weight stream | 286.1 (-10%) | 309.4 (-15%) |

| smem pad (bytes) | 0 | 4096 | 8192 | 12288 | 16384 | 24576 |
|---|---|---|---|---|---|---|
| q4_0 SoA 17408x5120 | 318.8 | 383.4 (1.20x) | 419.0 (1.31x) | 499.8 (1.57x) | 632.6 (1.98x) | 871.3 (2.73x) |
| q5_K generic 17408x5120 | 452.1 | 531.8 (1.18x) | 621.5 (1.37x) | 824.7 (1.82x) | 1080.3 (2.39x) | 1579.1 (3.49x) |

## Reading

1. **Prediction (a) REFUTED on both halves.** Removing half of the executed instructions - all the dequant ALU, 39% of the
   issue-weighted budget - changes the time by +2..+6%. Removing the DRAM weight stream with the instruction count unchanged
   gives -10..-15%. The q4_0 SoA tile at width 8 is not issue-bound, and Apple's per-opcode issue weights do not describe its
   limiter (what `agx-backend-access.md` finding 2 warned: the table is an apportioning model, not a cost table). The 91%
   "issue" share the profiler reports is time the kernel spends with instructions in flight, not a saturated port.
2. **Prediction (b) half right, for the wrong reason.** q5_K is the more occupancy-sensitive tile (3.5x vs 2.7x at the
   largest pad), but q4_0 SoA degrades from the first pad step too (1.2x at 4 KB). Both tiles are **latency-bound**: a
   threadgroup is one load -> dequant -> smem -> barrier -> MMA chain over 2 simdgroups, and the few threadgroups a core keeps
   resident do not hide the memory latency; the DRAM floor is not reached (the bytes probe says ~10-15% of the wall is the
   stream itself) and the issue port is idle half the time. The group-wall picture agrees: a tile that fills every threadgroup
   slot of every core alone cannot overlap with its neighbours (attention S/W 1.02), and a neighbour made faster by more
   threadgroups (split-K) has no slot to run in.
3. **The lever this names.** Latency hiding per core: more resident threadgroups (smaller smem / fewer registers per tile) or
   a deeper prefetch distance per threadgroup (two slices in flight, not one), not fewer instructions and not, first, fewer
   bytes. The direct-MMA skinny tile that gave ffn_down -4% in-graph (`skinny-direct-mma.md`) and the w8 decomposition items
   are this lever under other names; the pad ladder is the instrument to price any such form before it exists: a form that
   adds smem or registers moves down this curve.

## What stands in `census-groups.md`

The byte floor (needed bytes), the streamed-bytes column, the per-type table and the opcode map all stand - they are
measurements. The READING of the weighted rate as "issue-bound at 90%" is struck there; the per-type rates remain the input
they were ("a necessary input, not the answer"). Caveat: isolated kernels; the in-graph group walls under the same knobs are
the next arm if the owner wants the concurrent picture (`run-w8-decomp.sh STEPS=metalgroups` with `GGML_MM_SKINNY_PROBE=2`
and the pad as EXTRA env).

Files: `perf/run-issue-probes.sh`, `perf/run-issue-probes-capture.sh`; results `kvquant-experiments/results/issue-probes-oct26.txt`,
captures `kvquant-experiments/census/issue-probes-oct26/`. The probe knobs are inert unset; garbage output under PROBE=1|2.
