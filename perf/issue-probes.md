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

## In-graph (owner: "Run it") - the same knobs under the group profile, q4 line, pinned width 8, 2026-10-26

`run-w8-decomp.sh STEPS="metalprof metalgroups"` on this tree, port 8097, three controls interleaved with the arms (`ip-ctl{1,2,3}`,
`ip-probe2`, `ip-pad8k`, `ip-probe1`); per-group W us/call from `census-groups.py`, controls' spread in brackets. The probe arms
output garbage (acceptance 0.1%; pinned depth keeps every verify at width 8, so the per-call walls compare). PROBE=1 produced NaN
logits and the target emitted no tokens, so only the drafter's (m2) skinny groups carry that arm.

| group (W us/call) | controls | cached weights | pad 8 KB | no dequant (m2 only) |
|---|---|---|---|---|
| ffn gate+up (2 ops, S/W 1.09) | 560 (±1.4%) | 534 (-4.6%) | 715 (+27.6%) | m2: 600 (+8.9%) |
| ffn_down | 359 (±2.8%) | 303 (-15.6%) | 534 (+48.9%) | m2: 387 (+6.8%) |
| attn_out | 140 (±1.6%) | 116 (-17.5%) | 195 (+38.8%) | - |
| attn q+k+v (3 ops) | 125 (±0.2%) | 118 (-6.0%) | 149 (+19.2%) | - |
| GDN in_proj + 2x48 (S/W 1.61) | 205 (±1.0%) | 195 (-5.2%) | 235 (+14.4%) | - |
| lm_head | 3809 (±2.2%) | 3625 (-4.8%) | 5489 (+44.1%) | - |
| FA decode (control group, no skinny op) | 374 (±0.2%) | 374 (+0.2%) | 375 (+0.4%) | - |
| round (dec_syn_tg ms) | 110.3-111.6 | 102.9 (-7.5%) | 141.1 (+27%) | - |

The isolated reading holds in the concurrent graph: removing the DRAM weight stream pays 5-18% per skinny group (-7.5% of the
round); halving the resident threadgroups costs 14-49% per group (+27% of the round) and the ffn_down / attn_out / lm_head
single-op groups pay MORE in the graph (+39..+49%) than isolated (+31%); removing half the instructions costs +7..+9% on the
drafter's groups. The FA groups, which have no skinny op, do not move under any arm (the control that says the machine state held).
Two-op groups gain less from cached weights than single-op groups (gate+up -4.6% vs ffn_down -15.6%): in a latency-bound regime
the second concurrent kernel already hides part of the first's latency, which is what S/W 1.09 vs 1.00 was saying.

## Register ladder (owner: "I expect the w4 kernels to hide latency better. Would that be deducible?" - "Then run that")

`GGML_MM_REGPAD=N` (`FC_MUL_MM + 14`, declared beside the mul_mv constants because the width-4 kernel precedes the mul_mm block -
the first build compiled, the pipeline did not: a Metal error never fails the build): N extra floats loaded once before the K loop,
touched once per 16 slices, folded into the output under a runtime zero. Isolated medians of 3 on 17408x5120; registers and
spill from the capture of each step (`issue-probes-oct26/{skinny,w4}-rpN.stats.txt`):

| regpad | skinny w8: regs / us | w4: regs / us |
|---|---|---|
| 0 | 54 / 310.1 | 63 / 220.3 |
| 8 | 85 / 324.5 (+4.7%) | 74 / 234.4 (+6.4%) |
| 16 | 94 / 336.9 (+8.6%) | 80 / 246.1 (+11.7%) |
| 24 | 94 / 361.2 (+16.5%) | 93 / 263.5 (+19.6%) |
| 32 | 87 / 336.3 (+8.5%; executed 9.4 M vs 13.1 M - codegen moved) | 96 / 237.0 (+7.6%) |
| 48 | 96 + 32 B spill / 355.0 (+14.5%; executed 8.5 M) | 96 + 80 B spill / 252.8 (+14.8%) |

The clean rungs are 0/8/16 (executed counts within 7% of the base); from 24 up the pad perturbs codegen (the skinny's executed
count drops a third, the w4's 32-step is faster than its 24-step), so those rungs bound rather than measure.

**Reading.**
1. **Registers cost both kernels from the first rung, the w4 kernel more per register** (~0.7%/reg vs ~0.2%/reg over 0->16).
   The w4 kernel has no threadgroup memory, so its occupancy is register-bound and every register is a resident simdgroup; the
   skinny tile is smem-bound (the pad ladder: +20% for the first 4 KB) and registers only bind it past ~90.
2. **The w4 kernel does hide latency better - it reaches 0.88-0.95 of the byte floor in-graph where the skinny tile reaches
   0.55-0.71 - but with no surplus.** Eleven extra registers cost it 6%. It sits at the knee of its latency curve, exactly the
   "memory/issue crossover, nothing added, nothing removed pays" the width-4 ceiling probe found a month ago, now with the mechanism
   named: just enough bytes in flight, through structure (no barriers, lanes walking K in parallel, no smem), not through spare
   occupancy. The deduction in the preceding discussion ("at the floor at any plausible register-bound occupancy") was too
   generous by exactly this margin: Little's law with the pad ladder's ~1.1 us gives the right regime and the wrong slack.
3. **For the skinny tile the two ladders rank its levers:** resident threadgroups (smem) first, registers second. A form that
   halves the 5 KB per threadgroup (a 32-K slice, or the B stage out of smem) buys occupancy on the steep axis; a form that adds
   registers for a deeper prefetch pays on the shallow one. That trade is now priceable before the form exists.

Caveat: the capture's per-dispatch executed count for the w4 rows read 0.09 M (the profiler's "main" role matched an auxiliary
pipeline in those traces); the register counts come from the stats section matched by kernel name and are right.
