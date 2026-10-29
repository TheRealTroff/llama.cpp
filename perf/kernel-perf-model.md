# A kernel performance model for the M4 Pro fleet (2026-10-29, OPEN STUB - owner: "we should build a model that can predict with at least some accuracy how a kernel will perform")

Nothing built. This stub fixes the target, the dataset that already exists, the features, the first held-out split and
the scoring rule, so the first session on it starts by fitting, not by collecting.

## Target

The in-graph per-op time of a kernel row at the pick's widths (`us_call` in the census, one encoder per op under
`GGML_METAL_PROFILE=1`), and from it the group wall (`census-groups.py` bounds `[lo,hi]`, `GGML_METAL_PROFILE=2`). NOT the
isolated `test-backend-ops perf` number: that is a warm-cache upper bound (`percall-vs-ingraph`, 2026-09-30) and the Oct 27-29
campaign's qualified timer lives on that side. A prediction is a per-op ratio (candidate/incumbent) with a band, frozen before
the candidate's first GPU run, scored compatible / rejected / inconclusive, misses kept.

The fleet is the target, not one tile. The skinny width-8 tiles are the HARD CASE (below), not the prize: the owner's read is
that they may already be the best work in this area, and the ceiling probes agree (`ud-w4-ceiling`, `issue-probes`).

## What already predicts (keep, do not rediscover)

| Regime | Model | Measured accuracy | Where |
|---|---|---|---|
| DRAM-bound rows (width-4 decode mv, big prefill tiles) | bytes / 252 GB/s per concurrency group | rows sit at 1.1-1.3x the floor; a byte-saving form pays in proportion | `census-groups.md`, `byte-floor-per-group` |
| Register cliffs | offline spill bytes (`agx-spill-probe.py`) | 400 B = 5-10% of the kernel, 0 B vs 48 B = a ranking only | `metal-kernel-prescreen` |
| Table / shuffle forms | offline `text` size | the 2x text blowups timed 2x; ranked right every time it was the lever | prescreen ledger (iq4_xs table, shuffle gather) |
| Serialized delta -> wall | in-graph profiled ABAB, ctl-vs-ctl floor | the wall moved by exactly the serialized delta when nothing hid | `skinny-direct-mma.md` |

## What does not predict (measured, do not fit on it)

- Hot-loop instruction count on the width-8 skinny tiles: -46 static / K-step timed +3..+7% (2026-09-30); +9 / +16 timed
  +0.4 / +3.8% and a 9/16 scaling overpredicted (2026-10-29). Half the instructions +2..+6% (`issue-probes`).
- Register count alone: 80 -> 78 registers, +40% time (`m4-width5-crossover`); 54 -> 55 cost 0.4%, 54 -> 59 cost 3.8%.
- Apple's issue weights as a cost table (`issue-probes-state`: refuted 2026-10-26).
- Encoding-size families as opcode identities (prescreen step 4).

## Dataset (exists)

- `kvquant-experiments/census/census-1024-{q4,ud}-{w8-dec,w4-dec,pre}/snapshot.json` (2026-10-24/25, 16 rows each, 92 joined):
  per row `us_call`, `calls`, `bytes`, `x_floor`, `regs`, `spill`, `hot_instr`, `hot_issue`, `issue`, `stall`, `loads`,
  `load14_per_disp`, `exec_per_disp`, `exec_mem_share`, `opcodes` (-> `agx-opcode-classes.json` class counts), `cls`, `typ`,
  `s0`/`s1`/`dst` shapes, `join` (the `<row>.join.json` with the per-instruction profile). Traces on offload
  (`/Volumes/offload/play-archive/census-1024-traces/`).
- Group walls: `kvquant-experiments/results/cgroups-oct25-{q4,ud}-d7-groups.txt` + `-metalgroups.json`.
- Older snapshots for more rows at other shapes/lengths: `census-ud-96k-sep15*`, `census-ud-w8-sep18`, `fadec200k-oct25-ud`,
  `issue-probes-oct26` (the probe arms = the same kernel with one resource changed: the cleanest calibration pairs we have).
- Every mint adds rows (`run-census-full.sh` after any kernel change, `capture-method-into-skills`).

## Features to add (the two the Oct 29 work found mattered and nothing records)

1. **Load-use distance and dependency-chain depth per hot loop**, read off the final machine IR (the lldb MIR join,
   `metal-gpu-profile`; `agx-backend-access.md`). The Oct 29 prefetch forms differed in exactly this and in copies, and
   nothing else static separated them.
2. **Resident simdgroups per core**, the APS counter (`Compute SIMD Groups Inflight per Core`, `aps-counters.md`,
   `agxps-probe.py`): ~3 across the fleet so far, 1.79 on the grid-starved m=5120 rows. Record it per census row.
3. Threadgroup-memory bytes per threadgroup and threadgroups per core (the `issue-probes` lever that moved the skinny tiles 2x).

## First milestone (one session)

1. Load the 92 rows; features = bytes, class instruction counts (from `opcodes`), `regs`, `spill`, smem, grid; target `us_call`.
2. Fit the floor model alone (`us = max(bytes/BW, a + b * hot_instr * exec)`), then a per-class residual. Report the residual by
   `cls` (mv / mm_skinny / FA / small-op). Expect: DRAM rows inside 30%, skinny rows and FA rows not.
3. Held-out split by KERNEL, not by row: fit on ud, score q4 (same shapes, other quant types), then the reverse. A fit that
   needs the row to be seen is a table, not a model.
4. Add features 1-3 for the skinny and FA rows only (they are ~20 rows, the MIR join is per binary) and report the residual
   change. If it does not move with chain depth and residency, those are not the levers either - that is a result.
5. The first real forecast: pick a kernel change already queued (the w8 FA tier view: 1365/1175 instr per 32-key chunk, ~1 load
   per MMA, `census-oct24-state`), freeze its predicted per-op ratio and band, build, measure in-graph, score.

## Rules carried in

- Freeze before the first GPU run of a candidate; keep the misses in this file (the Oct 29 campaign's one rule worth keeping).
- Price in-graph; isolated numbers are upper bounds.
- A failed control invalidates the measurement, not the model; a passing fit on seen rows is calibration, not prediction.
- A deleted or added claim about the hardware needs a measurement in this file next to it.

## First fit (2026-10-30 night, `exp/kernel-perf-model`, tool `perf/kernel-perf-model.py`, data `kvquant-experiments/results/perfmodel-oct30/`)

Dataset: the six `census-1024-*` snapshots, 92 joined rows (`dataset.json`; bytes recomputed with the post-Oct-26 `work()`,
the Oct 24 snapshots still carried the SoA 2-byte bug). Target `us_call` (in-graph, one encoder per op). Repeatability: the
one row present twice (n=2 mv) agrees to 0.1%; the f32 ops shared by both lines agree to 1-3%. Model error is model error.

**Where each family sits (corrected floors):** mm prefill tiles at 0.93-1.03x the MMA roof (bytes irrelevant, 27-38x the
byte floor); mv decode at 1.04-1.4x the byte floor; skinny width-8 at 1.3-2.7x; FA decode 1.4-2.6x; FA prefill at the roof/1.35;
gdn 2-3.5x; small ops 1.1-10x with a ~10 us in-graph floor per op (add/swiglu at width 4-8 measure 4-6 us isolated, 10-11 in-graph).

**Candidate models, scored leave-one-kernel-out (|log(pred/actual)|, share of rows within 30%):**

| model | mm | mv | fa | skinny | gdn | small | all |
|---|---|---|---|---|---|---|---|
| M0b resource floor max(bytes/252 GB/s, gflop/6.96 TFLOPS) | 95% (med 3%) | 71% (med -20%) | 10% | 5% | 0% | 33% | 46% |
| M1 roofline c + max(floor, exec/K), K fitted | 95% | 83% (med 7%) | 70% | 18% | 50% | 50% | 63% |
| M2 class rates c + max(floor, sum exec_cls p_cls), NNLS | 95% | 62% | 60% | 64% | 50% | 33% | 65% |
| M4 M1 + chain floor L x trips x depth | 95% | 96% | 50% | 45% | 50% | 33% | 68% |
| M5 additive floor + sum exec_cls p_cls (no overlap) | 95% | 50% | 60% | 68% | 0% | 33% | 61% |
| M3 per-family log-linear (the table control) | 90% | 46% | 100% | 27% | 0% | 25% | 52% |

Fit on ud, score q4 (and the reverse): M1's K comes out 0.044 M SIMD-instr/us from q4 and 0.026 from ud - the "issue rate" is
the saturated rate of whichever kernels are in the fit (the census-groups table: q4 skinny groups 39-44 K/us, K-quant groups
22-35), not a chip constant; skinny scored cross-line falls to 0-5% within 30%. M3 fits fa/gdn perfectly in-sample and
collapses held out: a table, as predicted.

**The intervention scorecard (`intervention-scorecard.txt`) is the result.** Every model that fits the static rows mispredicts
the measured interventions on the q4_0 SoA skinny tile (issue-probes, 2026-10-26, isolated):

| intervention (m=17408,k=5120 / m=5120,k=17408) | measured | M1 roofline | M2 class rates | M5 additive |
|---|---|---|---|---|
| half the instructions, same bytes | +2% / +6% | -16% / -36% | -14% / -34% | -2% / -2% |
| same instructions, no DRAM weight stream | -10% / -15% | 0 / 0 | 0 / 0 | -47% / -47% |
| threadgroup-memory pad 8 KB | +31% (in-graph +28..+49%) | no term | no term | no term |
| register pad 16 (+40 regs) | +9% | no term | no term | no term |

A model with 68% of rows within 30% predicts the wrong sign or the wrong size of every lever actually pulled on this tile,
and has no variable for the two levers that moved it most. The floors are mechanism (mm, mv); the instruction terms are
tables. For the skinny class the static census features are the wrong inputs - which is what the stub predicted.

**Residency measured (APS `Compute SIMD Groups Inflight per Core`, read from headless replays; the census pipeline had deleted
`replay/raw`, so this needed the traces):**

- Oct 26 probe arms: base 3.11 simdgroups/core, half-instructions 3.28, no-stream 3.10. **Register pad ladder: skinny 3.12 ->
  2.75-2.79 from the first pad step and flat after (time +4.7 -> +16.5%); w4 kernel 3.11-3.13 FLAT across regpad 0..48 (time
  +6..+20%).** The regpad cost is not residency: time rises with the register count at constant occupancy, so the pad lengthens
  the chain (or perturbs codegen, as the record noted at rp32/48). The "~3 inflight" fleet reading holds on every arm.
- **Threadgroup-memory pad ladder with the counter (`run-smem-pad-aps.sh`, prod binary of Oct 26, m=17408 k=5120 n=8, two
  mirrored timing passes):**

| pad B | us/run (2 reps) | resident sg/core | waves (1088 sg / 20 / R) | t per k-step per sg | core throughput (k-steps/us) |
|---|---|---|---|---|---|
| 0 | 308.3 / 310.7 | 3.119 | 17.4 | 222 ns | 14.1 |
| 4096 | 361.9 / 353.8 | 1.813 | 30.0 | 149 ns | 12.2 |
| 8192 | 386.2 / 392.5 | 1.412 | 38.5 | 126 ns | 11.2 |
| 12288 | 460.2 / 480.3 | 0.991 | 54.9 | 107 ns | 9.3 |
| 16384 | 595.9 / 598.2 | 0.755 | 72.1 | 103 ns | 7.3 |
| 24576 | 863.3 / 837.3 | 0.507 | 107.3 | 99 ns | 5.1 |

  (k-step = one 64-wide K slice of the 32x8 tile, 80 per threadgroup here, 544 threadgroups x 2 simdgroups.) Residency falls
  6x, time rises 2.75x: T is NOT proportional to 1/R. Per-core throughput saturates (5 -> 14 k-steps/us) as R rises to ~3, and
  the uncontended k-step (R = 0.5) is ~99 ns - about the hot loop's 161 instructions at one issue per clock plus a partly
  hidden load. **This is the model shape for the latency-bound class: T = waves x k-steps x t_step(R), with t_step rising from
  the chain's own latency toward a shared per-core ceiling as residency grows.** Which resource sets the ceiling is still open:
  the saturated rate is ~2250 SIMD-instr/us/core, yet halving the instructions did not move the time, and caching the stream
  moved it only 10-15%.

What this leaves: (1) residency is a measurable per-kernel feature now (counter from a replay, 8 s per trace) - the census
rows are being replayed from offload for it; (2) a residency-aware latency model needs the per-kernel tile geometry (waves
and k-steps from the shapes) and a per-kernel t_step(R) - the hot loop's uncontended latency plus the contention curve;
(3) the first held-out test of that model is the ud skinny kernels (same tile family, 65-69 registers, different chains), then
the regpad ladder (R flat: the whole effect must come from t_step).

## Log

- 2026-10-29: stub opened after the Oct 27-29 prediction campaign (six `exp/m4-*` branches, one commit each; prefetch on the
  skinny tile refuted a third time; no model transferred to llama.cpp). Its timing host (`exp/m4-measurement-harness`
  41d5ffa8b) is a qualified ISOLATED timer; useful for calibration pairs, not for the target.
- 2026-10-30 night: first fit + intervention scorecard + residency ladder, section above. Branch exp/kernel-perf-model.
