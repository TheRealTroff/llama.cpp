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

  (k-step = one 64-wide K slice of the 32x8 tile per simdgroup: 80 per simdgroup, 544 threadgroups x 2 simdgroups = 87,040
  simdgroup-k-steps, 127 executed instructions each - 11.08 M per dispatch; the 161-instruction hot block is the dequant +
  stage half, run once per threadgroup-k-step.) Residency falls 6x, time rises 2.75x: T is NOT proportional to 1/R. Per-core
  throughput saturates (5 -> 14 k-steps/us) as R rises to ~3, and the uncontended k-step (R = 0.5) is ~99 ns - about 127
  instructions at one issue per clock plus a partly hidden load. **This is the model shape for the latency-bound class: T = waves x k-steps x t_step(R), with t_step rising from
  the chain's own latency toward a shared per-core ceiling as residency grows.** Which resource sets the ceiling is still open:
  the saturated rate is ~2000 SIMD-instr/us/core (15.9 k-steps/us x 127; ~1.35 per clock at 1.5 GHz), yet halving the instructions did not move the time, and caching the stream
  moved it only 10-15%.

**The ladder's calibration transfers (`smem-pad/tstep-fit.txt`, `transfer-q4soa-shapes.txt`).** Fitting t_step(R) on the six pad
points: linear 65 + 49 R ns (max error 9.7%), Little's-law (R + 1.48) x 46 ns (7.0%), hard max(95 ns, R/12.6) (11.6%), and
**smooth-max sqrt(91 ns^2 + (R / 15.9 k-steps/us/core)^2): 2.8%**. Taking that one calibration (one shape, one ladder) and
T = ceil(simdgroups / 20 cores / R) x k-steps x t_step(R) at the measured R = 3.12, with NO refit:

| q4_0 SoA skinny row | waves (ceil) | predicted us | isolated us/run | ratio |
|---|---|---|---|---|
| 5120x6144 | 6.16 -> 7 | 121.1 | 123.3 | 0.98 |
| 6144x5120 | 5.13 -> 6 | 124.6 | 123.5 | 1.01 |
| 5120x10240 | 10.26 -> 11 | 190.3 | 190.1 | 1.00 |
| 5120x12288 | 12.31 -> 13 | 224.9 | 225.7 | 1.00 |
| 5120x17408 (the calibration shape) | 17.44 -> 18 | 311.4 | 310.5 | 1.00 |
| 17408x5120 | 5.13 -> 6 | 352.9 | 356.5 | 0.99 |
| 5120x248320 (lm_head) | 248.8 -> 249 | 4307.6 | 3724.6 | 1.16 |

Five shapes never seen by the fit within 2%, with the wave count quantized (the continuous wave count misses by 6-15% on the
small grids: the last partial wave costs a full one). The lm_head miss (+16%) is the open row: at 7760 threadgroups the core
either keeps more resident or runs the k-step faster - its own counter reading (census replay) decides. The ud skinny kernels
(same tile geometry, 65-69 registers) run 1.24-1.72x what the q4 calibration gives: their own t0 / X, i.e. their chain and
ceiling, not the geometry - and the shape-to-shape spread of that ratio (q4_K 1.24 / 1.43 / 1.48) says their R is not 3.12.

**Register pad on the w4 kernel is an instruction cost law at constant residency**: executed instructions 7.68 -> 8.22 -> 8.70
-> 9.18 M per dispatch (+7 / +13 / +20%) for time +6.4 / +11.7 / +19.6%, R flat at 3.11-3.13 - at 1.1-1.2x the byte floor the
w4 kernel pays one-for-one for an added instruction. The skinny tile does not (+0.5% instr -> +4.7% time at rp8, -28% -> +8.5%
at rp32): registers cost it through the chain, not the count.

**Fleet residency (`census-aps-merge.py`, `census-aps-merge.txt`; every one of the 92 rows now carries `inflight` in
`dataset.json`, read from a fresh headless replay of its Oct 24 census trace).** Family medians (min-max): fa 1.98 (1.88-2.01),
gdn 5.75 (4.45-5.99), mm 3.21 (2.44-3.31), mv 3.19 (0.15-5.25), skinny 2.46 (0.30-3.85), small 8.77 (0.14-10.86). The prefill
mm tiles all sit at 3.1-3.3; FA at 1.9-2.0 on every shape, both lines. Within mv and skinny the spread is shape, not kernel:

| q4_0 SoA skinny row | threadgroups / core | in-graph R | implied t_step (us_call / waves / k-steps) | t_step(R) from the ladder law | T predicted at measured R | us_call | err |
|---|---|---|---|---|---|---|---|
| 6144x5120 | 8.0 | 1.59 | 130 ns | 135 | 143.0 | 136.9 | +4.4% |
| 17408x5120 | 8.0 | 1.75 | 128 | 143 | 389.1 | 349.1 | +11.5% |
| 5120x6144 | 9.6 | 1.80 | 137 | 145 | 127.7 | 120.2 | +6.2% |
| 5120x10240 | 16.0 | 2.72 | 192 | 194 | 185.8 | 184.0 | +1.0% |
| 5120x12288 | 19.2 | 2.82 | 194 | 199 | 223.3 | 217.4 | +2.7% |
| 5120x17408 (calibration shape) | 27.2 | 3.11 | 210 | 216 | 310.5 | 302.8 | +2.5% |
| 5120x248320 (lm_head) | 388 | 3.85 | 232 | 259 | 4179.0 | 3753.7 | +11.3% |

Three things this measures. (a) **R is a function of the grid, saturating near 3-4 simdgroups/core as threadgroups per core
grow** (8/core 1.6-1.8, 16/core 2.7, 27/core 3.1, 388/core 3.85); the same holds for the ud skinny kernels (27/core 2.80-2.92,
8/core 1.71-1.85) and for mv (the M=5120 shapes 2.8-3.0, the M=17408 shapes 4.7-5.2, lm_head mv 1.62 at 1.04x the byte floor).
The isolated ladder's R for the calibration shape (3.119) equals its in-graph reading (3.108): the counter is a property of the
kernel and grid, not of the host. (b) **The transfer table above assumed R = 3.12 for every shape; that assumption is wrong on
four of the seven shapes** (measured 1.6-1.8). With the measured R the same law lands +1..+12%, worse than the assumed-R
version: the 2% transfer was the ceil of the wave count on the wrong R compensating for a t_step 30% too high. The law itself
holds at the measured R on the implied t_step (130-137 ns measured vs 135-145 predicted at R 1.6-1.8, 192-210 vs 194-216 at R
2.7-3.1); the discrete-wave accounting is what breaks at 8 threadgroups per core (10 waves by R, 11 by ceil, +10%). (c) lm_head
runs at 16.5 k-steps/us/core, above the 14.1 the ladder saw at R 3.1 and above the fitted ceiling X = 15.9 - the ceiling is a
fit to six points below R 3.2, and at R 3.85 the chip gives more. Both (b) and (c) say the waves-x-t_step(R) form is right in
kind and wrong in its treatment of R: R has to be predicted from the grid (it is not a constant 3.12), and the counter's
time-average over a 10-wave kernel is not the steady-state concurrency the wave count wants.

**The ud skinny kernels, each on its own pad ladder (`run-smem-pad-aps.sh TYPE=…`, fitter `perf/tstep-fit.py`, dirs
`smem-pad-{q5_K,q4_K,iq4_xs}_soa/`, m=17408 k=5120, ud pick env; each ladder ~2.5 min).** Four kernels, one tile geometry,
four calibrations:

| kernel | regs | executed instr / sg-k-step | pad-0 R (isolated / in-graph) | t0 uncontended k-step | X per-core ceiling | X in SIMD-instr/us/core | smooth-max max err | linear / Little / hard-max |
|---|---|---|---|---|---|---|---|---|
| q4_0 SoA | 54 | 127 | 3.12 / 3.11 | 91 ns | 15.9 k-steps/us | 2020 | 2.9% | 9.7 / 7.0 / 11.6% |
| q4_K SoA | 65 | 95 | 2.91 / 2.92 | 150 ns | 15.4 | 1465 | 3.5% | 7.8 / 6.9 / 8.4% |
| iq4_xs SoA | 65 | 121 | 2.84 / 2.84 | 150 ns | 13.8 | 1660 | 3.8% | 8.0 / 8.0 / 7.6% |
| q5_K SoA | 69 | 122 | 2.81 / 2.80 | 170 ns | 13.1 | 1590 | 2.9% | 6.0 / 6.0 / 8.4% |

- **The padded residencies are identical across the four kernels** (pad 4096: 1.81 / 1.81 / 1.81 / 1.81; 8192: 1.41 / 1.41
  / 1.40 / 1.42; 24576: 0.51 / 0.50 / 0.50 / 0.50): with threadgroup memory padded, occupancy is the smem geometry and nothing
  else. At pad 0 it is not smem (5120 B would allow 6 threadgroups): R falls with the register count, 54 -> 65 -> 69 regs =
  3.12 -> 2.9 / 2.84 -> 2.81, the same step the q4_0 regpad ladder showed (3.12 -> 2.76 at +16 regs). The pad-0 cap is a
  register-file (or launch) effect and it is where every production skinny kernel runs.
- **Neither t0 nor X is shared.** The uncontended k-step is 91 ns on the q4_0 tile and 150-170 ns on the K-quant tiles with
  FEWER instructions per k-step (q4_K: 95 vs 127) - chain latency, not count. The ceiling X is 13.1-15.9 k-steps/us/core and,
  in instructions, 1465-2020 per us per core - so it is not an instruction-issue ceiling either (the half-instruction probe
  said the same). The one feature that separates 91 from 150-170 in the MIR hot loop is the load-use distance: q4_0 SoA
  consumes its device loads 138 instructions after issuing them (the hoisted stream), the K-quant tiles after 3-8
  (`load_use_min` in `dataset.json`; depth 5 vs 7-22 and ILP 32 vs 7-20 do not order the four). Four points, no fit -
  an observation to test with the next kernel.
- **Smooth-max is the form on all four** (max error 2.9-3.8%; linear and Little's law 6-10%, hard max 7.6-11.6%).

**Held-out scoring of the whole thing: every census skinny row of the four kernels, in-graph `us_call`, each at its own
calibration and its measured in-graph R, T = waves x k-steps x t_step(R), no refit (the 5120x17408 rows are the
calibration shapes; the other 13 are held out):**

| kernel: shape | TG/core | R | implied t_step / law | T (ceil waves) | T (fractional waves) | us_call | err ceil / frac |
|---|---|---|---|---|---|---|---|
| q4_0: 5120x6144 | 9.6 | 1.80 | 141 / 145 | 127.4 | 123.8 | 120.2 | +6.0 / +3.0% |
| q4_0: 6144x5120 | 8.0 | 1.59 | 142 / 135 | 142.7 | 130.2 | 136.9 | +4.2 / -4.9% |
| q4_0: 5120x10240 | 16.0 | 2.72 | 195 / 193 | 185.4 | 182.0 | 184.0 | +0.8 / -1.1% |
| q4_0: 5120x12288 | 19.2 | 2.82 | 200 / 199 | 222.7 | 216.6 | 217.4 | +2.5 / -0.4% |
| q4_0: 5120x17408 (cal) | 27.2 | 3.11 | 216 / 215 | 309.7 | 301.1 | 302.8 | +2.3 / -0.5% |
| q4_0: 17408x5120 | 8.0 | 1.75 | 141 / 143 | 388.4 | 354.1 | 349.1 | +11.2 / +1.4% |
| q4_0: 5120x248320 lm_head | 388 | 3.85 | 233 / 258 | 4167.6 | 4159.7 | 3753.7 | +11.0 / +10.8% |
| q4_K: 5120x6144 | 9.6 | 1.95 | 188 / 196 | 156.9 | 154.6 | 147.8 | +6.2 / +4.6% |
| q4_K: 5120x10240 | 16.0 | 2.55 | 231 / 223 | 232.2 | 224.5 | 231.9 | +0.1 / -3.2% |
| q4_K: 5120x17408 (cal) | 27.2 | 2.92 | 241 / 242 | 367.8 | 360.6 | 359.5 | +2.3 / +0.3% |
| q4_K: 17408x5120 | 8.0 | 1.82 | 184 / 191 | 466.7 | 456.9 | 440.2 | +6.0 / +3.8% |
| iq4_xs: 5120x17408 (cal) | 27.2 | 2.84 | 263 / 256 | 408.9 | 391.6 | 402.6 | +1.6 / -2.7% |
| iq4_xs: 17408x5120 | 8.0 | 1.85 | 189 / 202 | 494.4 | 474.6 | 443.9 | +11.4 / +6.9% |
| q5_K: 6144x5120 | 8.0 | 1.71 | 213 / 215 | 206.1 | 192.9 | 191.8 | +7.5 / +0.6% |
| q5_K: 5120x10240 | 16.0 | 2.46 | 261 / 254 | 263.9 | 263.9 | 270.9 | -2.6 / -2.6% |
| q5_K: 5120x17408 (cal) | 27.2 | 2.80 | 274 / 273 | 437.5 | 424.9 | 425.6 | +2.8 / -0.2% |
| q5_K: 17408x5120 | 8.0 | 1.81 | 210 / 219 | 537.1 | 527.9 | 505.0 | +6.4 / +4.5% |

With fractional waves: 13 of 17 rows within 5%, all 16 non-lm_head rows within 7%, the misses all on the 8-threadgroup-per-core
grids (+4..+7%, the law's t_step 4-7% above the implied one at R 1.6-1.9) and lm_head (+11%, the law's t_step 258 vs 233
implied at R 3.85: above the fitted ceiling). The ceil form is worse everywhere the grid is small (+6..+11%): a 10.5-wave
kernel does not cost 11 waves, which is what the time-averaged R was already saying. **The ud kernels, the stub's named
held-out test, land within 0-7% (median 3%) from their own ladders; the q4-calibrated version was 24-72% off.** The model
is: per-kernel (t0, X) from one ladder, R from the grid, geometry from the shapes. What it does not yet do is predict (t0, X)
or R without measuring them.

**Residency against the grid (2026-10-30 morning, owner: "run it"; `run-grid-aps.sh`, `grid-fit.py`, `grid/`; branch build
of test-backend-ops with the `GGML_TEST_MM_SHAPES` hook - the perf list is hard-coded; the branch binary gives 312 us on the
calibration shape vs 308-311 from the prod binary).** 23 shapes at n=8, pad 0: q4_0 SoA at K=5120 with 2, 4, 6, 8, 9, 10, 11, 12,
14, 16, 20, 24, 32, 64, 128 threadgroups per core (M = 640 x n), two unbalanced grids (8.5 and 12.5 per core), a K ladder at 8
per core (K = 1280 .. 20480), and q5_K at 8 and 16 per core. Two mirrored timing passes after a warm-up run, one capture each.

| q4_0 SoA, K=5120: TG/core | 2 | 4 | 6 | 8 | 8.5 | 9 | 10 | 11 | 12 | 12.5 | 14 | 16 | 20 | 24 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.43 | 0.89 | 1.13 | 1.55 | 1.59 | 1.59 | 1.76 | 1.91 | 2.15 | 2.19 | 2.36 | 2.71 | 2.85 | 2.94 | 3.18 | 3.53 | 3.71 |
| us/run | 55 | 64 | 89 | 102 | 112 | 123 | 134 | 146 | 150 | 158 | 180 | 195 | 232 | 285 | 350 | 664 | 1283 |
| implied t_step ns | 74 | 88 | 104 | 124 | 131 | 136 | 147 | 158 | 168 | 173 | 189 | 206 | 207 | 218 | 218 | 229 | 232 |
| k-steps/us/core | 5.8 | 10.1 | 10.8 | 12.5 | 12.1 | 11.7 | 12.0 | 12.1 | 12.8 | 12.7 | 12.5 | 13.1 | 13.8 | 13.5 | 14.6 | 15.4 | 16.0 |
| model (own t0/X, R measured, fractional waves) | +28% | +21% | +11% | +8% | +3% | 0% | -3% | -5% | -3% | -5% | -8% | -7% | -3% | -6% | +1% | +5% | +8% |

| K ladder, 8 TG/core | K 1280 | 2560 | 5120 | 10240 | 20480 |
|---|---|---|---|---|---|
| R | 1.38 | 1.47 | 1.55 | 1.68 | 1.69 |
| implied t_step | 129 | 121 | 124 | 147 | 141 |
| model | -3% | +7% | +8% | -6% | -1% |

q5_K: 8 per core R 1.70, 153 us, model +5%; 16 per core R 2.48, 290 us, model -9% (implied t_step 281 vs its ladder law 254).

What the design answered:

- **No staircase.** Time against the grid is a smooth curve; the unbalanced grids (8.5, 12.5 per core) sit between their
  neighbours, not on the next integer. The ceil-of-waves form is dead; the fractional form stands. The per-core counter reads
  the same value on all 20 cores to three digits from 4 threadgroups per core up (at 2 per core it spreads 0.23-0.66): the
  dispatcher spreads work evenly, and nothing in the model needs a max over cores.
- **R is a function of the grid, not of threadgroup lifetime.** K from 1280 to 20480 (time x14) moves R 1.38 -> 1.69 at 8
  per core while the grid moves it 0.43 -> 3.71. A fixed launch rate (Little's law) is refuted: resident threadgroups per
  lifetime would be 0.4-2 per us across the points. `R(n) = 3.89 (1 - exp(-n / 15.9))` fits the 17 q4_0 points within 9% and is
  the grid-to-residency predictor the model lacked.
- **The per-core ceiling is real and is reached only above ~64 threadgroups per core:** throughput rises monotonically to 16.0
  k-steps/us/core at 128 per core, the pad ladder's fitted X = 15.95. Every production shape (8-27 per core) runs at 70-90% of it.
- **The model is a +-8% model on this family at production grids** (17 of 23 points within 8%; all 15 points at 8-64 per core
  within 8%). The misses have a shape: small grids (2-6 per core) are over-predicted 11-28% because their k-step (74-104 ns)
  is below the pad ladder's floor t0 = 91 ns (the pad-24576 point, R 0.51, measured 99 ns; the 2-per-core grid, R 0.43, 74
  ns): **t_step at a given R depends on how that R came about** - fewer threadgroups queued vs less threadgroup-memory room
  differ by up to 9% at equal R (grid n=14-16: implied 189-206 vs law 174-192; pad: within 3%). R is the right first
  variable and not the whole story.
- **The counter's meaning is now the open question, not the model's form.** At 4 per core every core reads 0.89 simdgroups
  with 4 threadgroups queued on it - under one threadgroup in flight per core. The Oct 26 no-stream arm left the reading
  unchanged (3.10 vs 3.11), so it is not a count that excludes memory-stalled simdgroups; it says the chip keeps a small grid
  under-filled. Time per threadgroup-per-core is 9.7 us (= 160 sg-k-steps / X) from 16 per core up and cheaper below. Why a
  core with four threadgroups queued holds less than one is the mechanism to find; it is where 10-30% of a small-grid
  kernel's time lives, and the small-grid rows (projections at 5120 x 1024 and 5120 x 48, the drafter) are exactly the
  census rows no model has placed.

**Probe arms and five more kernels, each on its own pad ladder (2026-10-30 10:10-10:30, owner: "doing 1 and 2 first seems
obvious"; `smem-pad-q4_0_soa-{probe1,probe2,regpad16}/`, `smem-pad-{q6_K,q3_K,iq4_nl,iq3_s}_soa/`, `smem-pad-q8_0/`; the
probe arms on the prod binary like the baseline ladder, the five kernels on the branch binary via the shape hook).**

The probe arms on the q4_0 SoA tile, against its baseline ladder (t0 91 ns, X 15.95; times per pad step 0 / 4096 / 8192 /
12288 / 16384 / 24576 B):

| arm | pad-0 R | pad-0 time | time vs baseline at the five padded steps (R identical, 1.81 .. 0.51) | t0 | X |
|---|---|---|---|---|---|
| baseline | 3.12 | 309 us | - | 91 ns | 15.95 |
| PROBE=1 half the instructions (no dequant; same loads, same MMA) | 3.27 | 316 (+2%) | +9 / +11 / +21 / +22 / +24% | 114 ns (+25%) | 16.3 (+2%) |
| PROBE=2 no DRAM stream (same instructions, weights re-read from cache) | 3.10 | 283 (-8%) | -19 / -18 / -17 / -21 / -21% | 69 ns (-24%) | 17.15 (+7%) |
| REGPAD=16 (+16 live registers) | 2.75 | 338 (+9%) | -2 / 0 / -2 / -2 / -4% | 85 ns | 14.65 (fit compromise, see below) |

- **Halving the instructions lengthens the uncontended k-step by a quarter and leaves the ceiling where it was.** With the
  dequant ALU work removed, the same loads feed the same MMA 24% slower when nothing else is resident: the ALU instructions
  were filling the load-to-MMA latency, not costing time. At full residency the arm costs +2%: the ceiling (16.3 vs 15.95) does
  not see instruction count. Instruction issue is out as the ceiling, now by a ladder and not a single point.
- **The DRAM stream is 22 ns of the 91 ns chain and 7% of the ceiling.** Cached weights give t0 = 69 ns (the load-to-MMA chain
  through L1/L2) and X = 17.15. At production residency the arm is worth 8%, at low residency 21%: DRAM latency is mostly
  hidden by residency on this tile, and the ceiling is not DRAM.
- **REGPAD at equal residency is not slower - it is 0-4% faster at every padded step.** Its +9% at pad 0 is all residency
  (2.75 vs 3.12). But at R 2.75 it runs a 213 ns k-step where the baseline law gives 195 ns: the same R reached by registers
  instead of threadgroup memory is 9% slower per k-step. The fitted X = 14.65 is the compromise between the padded points (on
  the baseline curve) and that one. Same lesson as the grid: **t_step at a given R depends on how the R arose.**

The (t0, X) table, nine kernels of the one tile geometry (m=17408 k=5120 n=8, pad ladders, smooth-max within 1.6-4.7%):

| kernel | tile | regs | pad-0 R | t0 | X k-steps/us/core | executed instr / sg-k-step | X in instr/us/core | weight B / sg-k-step | X in KB/us/core | MIR load-use min |
|---|---|---|---|---|---|---|---|---|---|---|
| q4_0 SoA | SoA skinny | 54 | 3.12 | 91 | 15.95 | 127 | 2030 | 576 | 9.2 | 138 |
| q4_K SoA | generic | 65 | 2.91 | 150 | 15.35 | 95 | 1464 | 576 | 8.8 | 3 |
| iq4_nl SoA | generic | - | 2.83 | 146 | 13.80 | - | - | 576 | 8.0 | - |
| iq4_xs SoA | generic | 65 | 2.84 | 150 | 13.75 | 121 | 1663 | 544 | 7.5 | 8 |
| q5_K SoA | generic | 69 | 2.81 | 170 | 13.10 | 122 | 1594 | 704 | 9.2 | 6 |
| q3_K SoA | generic | 67 | 2.85 | 142 | 12.80 | 146 | 1871 | 440 | 5.6 | 44 |
| iq3_s SoA | generic | - | 2.78 | 166 | 11.95 | - | - | 440 | 5.3 | - |
| q6_K SoA (lm_head) | generic | 69 | 2.84 | 158 | 11.35 | 161 | 1822 | 840 | 9.5 | 4 |
| q8_0 | generic | 63 | 3.03 | 149 | 7.45 | ~204 (from the 5120x48 row, prologue included) | ~1517 | 1088 | 8.1 | 14 |

- **t0 is two values: 91 ns for the SoA tile, 142-170 ns for every generic-tile kernel**, whatever its instruction count (95-204),
  dependency depth (7-23) or ILP. With PROBE=2 saying the cached chain is 69 ns and DRAM adds 22 ns when the loads are
  consumed 138 instructions after issue, the generic tile's extra 50-80 ns is the DRAM latency it does not hide (load-use
  distance 3-44). **The t0 predictor is: ~70 ns of load-to-MMA chain, plus the fraction of DRAM latency the hot loop exposes,
  which the load-use distance sets.** The direct test is PROBE=2 on a generic-tile kernel (the knob is q4_0-only today).
- **X is not any instruction class.** No class count times X is constant across the seven kernels with MIR (threadgroup
  loads 111-212, other ALU 381-681, device loads 34-268 per us per core). Weight bytes times X is the tightest: 7.5-9.5 KB/us/core
  for seven kernels, but the two 3-bit formats sit at 5.3-5.6 and q8_0's generic tile reaches only 7.45 k-steps/us (its
  2x bytes per k-step, 8.1 KB/us). The ceiling scales with the weight stream through the core's load path at roughly 9 KB/us/core
  (180 GB/s chip-wide at full residency, cached or not), with the 3-bit formats paying more per byte. The load path
  (requests, not bytes: q3_K issues 12 device loads per hot block, iq3_s similar) is the candidate; a ladder on a format with
  the same bytes and a different load count would separate requests from bytes.

**The no-stream probe on the generic tile (2026-10-30 10:54-11:05; `GGML_MM_SKINNY_PROBE=2` ported to `kernel_mul_mm_skinny_t`
on the branch, f866c721e; ladders `smem-pad-{q4_K,q5_K,q3_K}_soa-probe2/`, same-binary control `smem-pad-q4_K_soa-branchbase/`
t0 154 X 15.6 vs the prod-binary 150 / 15.4).** The experiment that was to decide what t0 is a function of:

| kernel | t0 with the DRAM stream | t0 cached (PROBE=2) | exposed DRAM latency | uncontended k-step R=0.5, stream / cached | X stream / cached | pad-0 time |
|---|---|---|---|---|---|---|
| q4_0 SoA | 91 ns | 69 ns | 22 ns | 99 / 78 ns (-21%) | 15.95 / 17.15 | -8% |
| q4_K generic | 150-154 | 112 | 38-42 | 158 / 121 (-24%) | 15.4 / 15.1 | -7% |
| q5_K generic | 170 | 130 | 40 | 178 / 140 (-21%) | 13.1 / 13.6 | -10% |
| q3_K generic | 142 | 110 | 32 | 153 / 121 (-21%) | 12.8 / 12.8 | -6% |

**Outcome two.** The generic tile's cached chain is 110-130 ns, not 70: of the 50-80 ns between the SoA tile and the generic
tiles, 40-60 ns is the chain through the caches and only 10-20 ns is extra exposed DRAM latency. The dequant and staging
sequence of the generic tile is on the critical path. So t0 is NOT predicted by the load-use distance plus a shared chain; it
needs the critical-path latency of the k-step from the machine code, which means per-instruction latencies this tree does not
have (the profiler's stall column is where they would come from). Two facts the probe did settle: the exposed DRAM share is
a fifth of the uncontended k-step on all four kernels (21-24%, 22-42 ns), and the ceiling X does not move when the stream
leaves DRAM on any of them (now four kernels, two tiles). Within the generic cluster the cached chain spreads 110-130 ns and
no static feature in `dataset.json` orders it (q3_K: deepest dependency chain, most loads, fastest).

What this makes the model, honestly: **per kernel (t0, X) is a 2.5-minute ladder, not a formula.** t0 = 1.25 x the cached
chain, the cached chain is a property of the tile and format (69 ns SoA q4_0; 110-130 ns generic), and the ceiling is the
weight-stream load path. For a NEW kernel of a known tile the table gives t0 within +-10% before any measurement; for a new
tile one ladder calibrates it. That is where the first milestone ends.

**The t0 formula (2026-10-30 11:10-12:10; owner: "the t0 formula *is* the model"). Tools: `perf/agx-latency/` (bench.swift +
kernels.metal + latency.json), `perf/kstep-sim.py`, `perf/kstep-validate.py`; results `agx-latency.txt`, `kstep-validate.txt`.**

*Latency table.* Dependent-chain micro-kernels, one simdgroup, clocks warmed, ns per instruction (the chip constants):

| fma/add f32, f16 | int add/xor | int mul | shift / and / bfe | int<->half convert | L1 or threadgroup load | threadgroup store -> dependent load | L2 | DRAM (dependent chase) | MMA 8x8x8 f16->f32 | issue interval, one simdgroup | MMA issue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.6-1.7 | 1.65 | 3.5 | 2.0 | 2.5 | 37 | 51 | 80-95 | 300 | 11.7 | 0.9 | 3.2 |

The threadgroup barrier costs nothing for one simdgroup beyond the store-to-load it orders. The MIR opcode numbers of each
chain come from translating the micro-kernels with `agx-nt-opt.py mir` (fma.f32 = 2212, add.f32 = 3312, f16 fma/mul = 820 /
884 / 798, cvt = 12107 / 10226, bfe/shift = 17165 / 442 / 443, iadd = 11082, MMA = 2884, barrier = 466; loads and stores by
their memory operand and address space, not by opcode).

*Simulator.* `kstep-sim.py` takes a census MIR join, picks the loop body (basic blocks executed as often as the hottest), builds
register dependencies from the operands (pairs and 16-bit halves), carries uses across iterations, makes threadgroup loads wait
for the iteration's earlier threadgroup stores, and issues one simdgroup in order: an instruction issues at max(previous issue +
its issue interval, operands ready), its result is ready one latency later. Device loads: the activation loads (the two
`<4 x s32>` from a bare 64-bit base, read by every threadgroup) hit L1; of the weight loads the first per iteration pays the
stream latency and the rest (same block) hit L1. Output: the steady-state iteration = one 64-wide k-step of one simdgroup.

*Validation against the single-threadgroup k-step* (the pad-24576 ladder point: one 29.7 KB threadgroup per core, 27.2 of
them per core in sequence, 80 k-steps; `kstep-validate.py`):

| kernel | sim cached | measured cached (PROBE=2) | err | sim stream (L = 200 ns) | measured stream | err | pure issue per k-step |
|---|---|---|---|---|---|---|---|
| q4_0 SoA | ~~337~~ 325 | 307 | ~~+10%~~ +6% | ~~410~~ 398 | 390 | ~~+5%~~ +2% | ~~242~~ 205 ns |
| q4_K | ~~504~~ 493 | 482 | ~~+5%~~ +2% | ~~667~~ 656 | 636 | ~~+5%~~ +3% | ~~254~~ 217 |
| q5_K | ~~599~~ 587 | 560 | ~~+7%~~ +5% | ~~762~~ 750 | 710 | ~~+7%~~ +6% | ~~352~~ 315 |
| q3_K | ~~469~~ 458 | 492 | ~~-5%~~ -7% | ~~632~~ 620 | 626 | ~~+1%~~ -1% | ~~327~~ 290 |
| iq4_xs | ~~423~~ 412 | (485, 12:40) | -15% | ~~586~~ 575 | 650 | ~~-10%~~ -12% | ~~252~~ 215 |
| q6_K | ~~490~~ 478 | (510) | -6% | ~~653~~ 641 | 661 | ~~-1%~~ -3% | ~~323~~ 286 |
| q8_0 (join from the 5120x48 row) | ~~374~~ 362 | (443) | -18% | ~~537~~ 525 | 627 | ~~-14%~~ -16% | ~~218~~ 181 |

(Struck values: the 12:10 table, which priced an MMA at 3.2 ns of issue. The evening's clean MMA probes put the back-to-back MMA
interval at 11.7 ns and showed the 3.2 came from a folded probe; the re-score prices an MMA at one issue slot, which is what
the measured k-steps support - see "Where the mul_mm tile's three quarters went" below, and `kstep-validate.txt` for both rules.)

**The cached chain is predicted from the machine code within 10% on all four probed kernels with nothing fitted**: table
latencies, in-order issue, the dependency graph. The weight stream needs one chip constant, an effective latency of 200 ns
(the dependent chase measures 300; a streaming slice gets some overlap), and with it seven kernels land within 14%, six within
10%. This is t0 as a function of the code. It also says what the chain is made of: pure issue is 60-75% of the k-step (~~the
MMAs at 3.2 ns each are 51 of it~~ the 16 MMAs are 14 of it at one slot each; what the matrix pipe costs the simdgroup beyond
that is the evening's open question), the rest is the load-to-use wait and the store-to-load through threadgroup memory. The
generic tile's 110-130 ns "cached chain" over the SoA tile's 69 is, in these terms, 12-110 more instructions to issue plus
a dequant sequence that waits on its loads instead of hiding them.

*Six cached points (12:40; owner asked whether the simulator's error rose with bits on the first three: q3_K -5%, q4_K +5%,
q5_K +7%).* PROBE=2 ladders added for iq4_xs, q6_K, q8_0 (`smem-pad-{iq4_xs_soa,q6_K_soa,q8_0}-probe2/`; cached t0 / X:
108 / 13.3, 117 / 13.0, 105 / 14.5 - q8_0's ceiling doubles from 7.45 when the stream leaves DRAM, the load-path cap in person).
Cached k-step, simulator vs measured, by bits per weight: q3_K (3.4) -5%, iq4_xs (4.25) -13%, q4_K (4.5) +4%, q5_K (5.5) +7%,
q6_K (6.6) -4%, q8_0 (8.5) -16%. **No trend with bits**; the two ladder repeats agree to 0.1%, so the errors are structural
and kernel-specific. The two misses: iq4_xs under-predicted 13% both cached and streamed - its dequant gathers from a
threadgroup-memory lookup table (32 `s32` loads per iteration, lane-indexed), which the simulator prices as plain 37 ns loads
and which bank conflicts would make slower (a gather micro-kernel would price it); q8_0 under-predicted 16% from a join taken
on the 5120x48 census row (two threadgroups; rejoin at 17408x5120 before reading anything into it). The four other generic
kernels and the SoA tile sit within -5..+10%.

*What the simulator says about the ceiling.* If a core issues from four simdgroups at once (four 32-lane pipes), the per-core
ceiling in k-steps/us is 4 / (issue time per k-step), capped by the load path:

| kernel | issue per k-step | 4 ports / issue | 9 KB/us/core / weight bytes | X predicted | X measured (ladder) | err |
|---|---|---|---|---|---|---|
| q4_0 SoA | 242 ns | 16.5 | 15.6 | 15.6 | 15.95 | -2% |
| q4_K | 254 | 15.8 | 15.6 | 15.6 | 15.35 | +2% |
| q5_K | 352 | 11.4 | 12.8 | 11.4 | 13.10 | -13% |
| q3_K | 327 | 12.2 | 20.5 | 12.2 | 12.80 | -4% |
| iq4_xs | 252 | 15.9 | 16.5 | 15.9 | 13.75 | +15% |
| q6_K | 323 | 12.4 | 10.7 | 10.7 | 11.35 | -6% |
| q8_0 | 218 | 18.4 | 8.3 | 8.3 | 7.45 | +11% |

Two chip constants (four issue ports at 0.9 ns, a load path of ~9 KB/us/core of weight bytes) place the ceiling of seven
kernels within 15%, no per-kernel fit. It also explains the two probe arms that looked contradictory: halving q4_0's
instructions did not move X because q4_0 sits on the load-path cap, not the issue cap; caching the stream raised X 7% because
the load path fills faster from L2. The four-port figure is inferred from the fit, not measured: a micro-kernel with 2, 4, 8
simdgroups per threadgroup would measure it directly.

*Dual issue and the core's ports (13:00, owner: "the dual issue question had some implications, so let's settle that";
`agx-dual-issue.txt`, kernels `ctl_*`, `dual_*`, `mma4_*` in `perf/agx-latency/kernels.metal`).* Throughput-bound chains (4 or
8 independent) of two classes interleaved in ONE simdgroup, against 8 chains of one class:

| one simdgroup | ns per iteration | reading |
|---|---|---|
| 8 x f32 fma (64 ops) | 48.8 | 0.76 ns per instruction |
| 8 x f16 fma (64) | 48.8 | same pipe rate |
| 4 f32 + 4 f16 (32 + 32) | 49.4 | = the 64-op control: **no FP32+FP16 dual issue** |
| 4 f32 + 4 int (32 + 64 int ops) | 41.8 | = 28.5 + 15.8 (each alone): **no FP32+INT dual issue** |
| 4 f16 + 4 int | 42.5 | same |
| ~~32 MMA~~ 8 MMA + 128 f32 fma | 180.0 | vs 105.8 + 89.4 alone: ~~**MMA and ALU do not overlap** (8%)~~ RETRACTED (evening): the "32 MMA" probe held 8 (the compiler folded four identical accumulator chains into one), and its 105.8 is a dependent chain of 8 at 13 ns - see below |

A simdgroup issues one instruction at a time whatever its class, MMA included: the simulator's in-order single-issue assumption
is right. (The int chain, shift+xor, ran at 0.25 ns per op - half the FP rate - so either the pair fused or the integer pipe
issues faster from one simdgroup; unresolved, and immaterial to the loops, whose int work is a minority.)

Then the same kernels with 1, 2, 4, 8, 16, 32 simdgroups in one threadgroup (one core):

| simdgroups per core | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| f32 fma, core throughput (sg-instr/ns) | 1.12 | 2.25 | 4.5 | 5.4 | 5.8 | 6.0 |
| f16 fma | 1.3 | - | 5.2 | 5.8 | 6.05 | - |
| f32 + f16 mixed streams | 1.3 | - | 5.2 | 6.85 | **7.7** | - |
| ~~MMA (8x8x8 half -> float), MMA/ns~~ (folded probe, 4x too high) | ~~0.30~~ | - | ~~1.21~~ | ~~1.41~~ | ~~1.49~~ | - |
| MMA, clean probe (evening: `mma16_regs`, 16 distinct products from operands loaded once) | 0.086 | - | 0.34 | 0.37 | **0.38** | - |
| ~~32 MMA + 128 fma per iteration~~ (folded) | - | - | - | - | ~~0.77 MMA/ns + 3.07 fma/ns~~ | - |

- **Four simdgroups run at full single-simdgroup speed; the core saturates at 6.0 scalar instructions/ns for one class.** At
  1.5 GHz that is four instructions per clock = 128 lanes: the four ports the ceiling fit inferred are real, and a port issues
  one instruction per clock while a single simdgroup manages one per ~1.2 clocks (0.76-0.89 ns). Two simdgroups per port
  saturate it.
- **FP32 and FP16 are separate pipes behind the ports.** Mixed streams from different simdgroups reach 7.7/ns against 6.0 for
  either class alone: the 6.0 is a pipe limit, and the port/issue limit is above 7.7. Dequant-heavy loops that mix FP16
  converts with FP32 accumulation get some of this at residency; the ceiling model does not yet account for it.
- ~~**MMA shares the ports and costs four slots.** 1.49 MMA/ns per core = one 8x8x8 per clock; the mixed stream's shares sum
  to 103% (52% of MMA capacity + 51% of scalar), so the two compete for the same issue resource, one MMA = four scalar slots -
  the 3.2 ns single-simdgroup MMA issue is the port's four clocks.~~ RETRACTED the same day (evening): the MMA probes
  (`mma_f16_indep4_u16`, `mma4_u8`, `mma4_f32x16_u8`) had four accumulator chains with identical inputs and the compiler
  folded them into one - 8 MMAs per iteration where 32 were counted, so every MMA figure above was 4x too good. Measured
  cleanly (operands loaded once before the loop, 16 distinct products): **one MMA costs a port 16 clocks (10.5-10.7 ns), the
  core does 0.38 MMA/ns, and one simdgroup starts back-to-back MMAs 11.7 ns apart.** An 8x8x8 MMA is 512 lane-FMAs = 16 per
  lane: it runs at the FMA lanes' rate, i.e. this GPU has no matrix unit (consistent with Apple first adding one in the M5).
- ~~**The MMA pipe is far above what any tile reaches.** 1.49 MMA/ns/core x 1024 flop x 20 cores = 30 TFLOPS of half->float
  matrix work, against the 7 TFLOPS the mul_mm prefill tiles sit at. The "MMA roof" the mm rows were scored against is the
  best kernel, not the hardware: the prefill tiles are issue-, stage- and load-bound at a quarter of the matrix pipe. That
  reopens the mm family (95% of a roof that is itself 25% of the pipe).~~ RETRACTED: 0.38 MMA/ns/core x 1024 x 20 = 7.8 TFLOPS,
  the same number as the FP32 FMA roof (6.0 instr/ns x 32 lanes x 2 x 20 = 7.7), and the mul_mm acch n64 tile runs at 7.54
  isolated: **the tiles are at 97-98% of the hardware's matrix rate**. The mm family stays closed by the roof. Section below.

For the ceiling model the refinement is: per k-step, slots per pipe (FP32-pipe instructions, FP16-pipe instructions, MMA at
~~four~~ **sixteen** slots on the FP16 pipe, loads on the load path), ceiling = min over the binding pipe of 4 per clock, and the
load path - instead of one undifferentiated issue sum. Not yet done; the one-sum version placed seven kernels within 15%. For
the skinny tiles the MMA term is not binding (16 MMAs x 10.6 ns per simdgroup k-step -> 23.6 k-steps/us/core against the
load path's 15.6); for the mul_mm tiles it is the whole story.

*What R was.* At pad 24576 the core holds exactly one threadgroup (two simdgroups) and the counter reads 0.51: the APS "inflight"
value is about a quarter of the resident simdgroups, so the t0 = 91 ns and X of the earlier sections are the physical 390 ns
per threadgroup k-step and 4 simdgroups per core seen through that proxy. Nothing above changes; the physical model is cleaner:
T = threadgroups per core x k-steps x t_TG(resident) with t_TG(1) from the simulator, and the contention curve from how
several resident simdgroups share four issue ports and the load path - which the same simulator can produce by running N
instruction streams against shared ports.

*Where the mul_mm tile's three quarters went (evening; owner: "now I can't wait to see where the other 3/4 go";
`agx-mm-units.txt`, `mm-isolated.txt`, `perf/agx-latency/agx-mm-*-m4pro-2026-10-30.txt`).* Nowhere: there were no three
quarters. The chase ran like this. The `kernel_mul_mm_acch_n64_q4_0_f16` tile (64x64 output per threadgroup, 32-wide k per
iteration, 4 simdgroups; per simdgroup-iteration 64 MMAs, 32 `simdgroup_load`s of 8x8 half tiles from threadgroup memory,
16 scattered 2-byte staging stores + 2 vector stores, ~100 dequant ALU ops, two threadgroup barriers) runs its census join
through the simulator at 310 ns per simdgroup-iteration - but its MMA is opcode 860 (half accumulate), which the table did not
have, so it was priced as a 1.6 ns FP16 op. Unit costs were then measured one at a time (one simdgroup, then 1-16 simdgroups
per core; ns):

| unit (probe) | one simdgroup | per port at 4 simdgroups per port | note |
|---|---|---|---|
| MMA half->half 860, dependent chain (`mma_h_u16`) | 11.7 latency | | same as half->float |
| MMA, 16 distinct products, operands in registers (`mma16_regs_u16`) | **11.7 per MMA** back-to-back | **10.5** | half->float (`mmaf16_regs`): 11.9 / 10.7 |
| 8 tile loads + 16 MMAs, the tile's 8-wide k sub-step (`substep_h_u16`) | 257 | 191 (= 16 x 11.9) | MMA-bound at 2 simdgroups per port already |
| tile load -> ALU use (`ld1_te1`) / -> MMA use (`ld1_mma1`) | 58 / ~72 latency | | 2-3 ns per extra load; core >= 1.4 tile loads/ns |
| scattered 2-byte staging store, the tile's exact pattern (`tst16_u16`) | 3.3 each | 0.7 each (1.44/ns/core) | a 16-byte vector store: 11.7 / 1.25 |
| threadgroup barrier with 4 simdgroups in lockstep (`barrier_u16` at 128 threads) | +25 per barrier | | +30 at 8, +37 at 16 simdgroups |
| 16 MMAs + 64 FP32 FMAs (`mma16_f32x64`) | 223 = 15 x 11.7 + 64 x 0.76 | 749 vs 675 + 170 apart | at the core 56% of the FMAs hide under the MMAs |
| 16 MMAs + 64 FP16 FMAs | 223 | 833 vs 844 apart | additive: FP16 FMAs and MMAs share a pipe |
| 16 MMAs + 64 INT ops | 200 vs 219 apart | 694 vs 767 apart | INT hides 59% in the simdgroup, 79% at the core |
| 16 MMAs + 32 tile loads | 301 vs 332 apart | 802 vs 1078 apart | loads hide 21% / 68% |

The first probe of the set (`mma_h_indep4_u32`, the half->half copy of the morning's `mma_f16_indep4_u16`) gave the morning's
3.3 ns per MMA again, and `mma4_h_ctl_u4` - four MMAs on one register pair - ran at 25 ns per iteration with ONE MMA in its
machine code: the compiler had folded four accumulators with identical inputs (same a, same b, all zero) into one chain. The
morning's MMA probes have the same shape, and their MIR shows 8 MMAs per iteration where 32 were counted (checked for all four).
With operands loaded once before the loop and 16 distinct products nothing folds, and the number is 11.7 ns between
back-to-back MMAs from one simdgroup, 10.5 ns per MMA per port - **16 clocks at 1.5 GHz, one MMA = 512 lane-FMAs = 16 FMA
slots per lane. The matrix work runs on the FMA lanes; the MMA roof is the FMA roof, 7.7-7.8 TFLOPS.**

The tile against that roof: in the census graph 11994 us per call = 689 ns per threadgroup-iteration per core (isolated:
12107 us, 7.54 TFLOPS; the f32-accumulate n64 tile 7.15, the acch n32 tile 7.35). One threadgroup-iteration is four
simdgroup-iterations on four ports, 64 MMAs each: 64 x 10.5-10.7 = 672-685 ns of the port's MMA capacity against 689 measured.
**The acch n64 tile is at 97-99% of the matrix rate. Everything else in its iteration - the 32 tile loads, 18 stores,
~100 dequant instructions, two barriers - hides under the MMAs at the core level** (the mixed probes above say how: FP32 and
INT work from other simdgroups overlaps the pipe, the dequant is FP32 + bitfield + convert). The single-simdgroup chain (the
simulator says 955 ns per iteration with the 11.7 spacing enforced: 659 of it MMA-to-MMA, the rest the once-per-sub-step tile
load latency of ~60-70 ns exposed because the sub-step's loads sit behind the previous sub-step's MMAs in program order) is
hidden by the 3.3 resident threadgroups per core. So: the mm family is not reopened. Its rows sit on the hardware, the
remaining 1-3% are not worth a tile, and the only lever that could move an mm row is fewer MMAs (a different accumulation
shape or a sparse/skipped block), not a better schedule.

Two things this changes for the model. (a) *The MMA rule in the simulator.* Measured: an MMA holds the simdgroup's matrix pipe
11.7 ns and the FP16 pipe with it; FP32, INT and load instructions issue beside it to a varying degree (26%, 59%, 21% hidden
in one simdgroup when placed after a 16-MMA group - the compiler groups MMAs and drops `simdgroup_barrier(mem_none)` fences,
so an interleaved program order could not be forced). Enforcing the 11.7 spacing in program order over-predicts the skinny
k-steps by 16-27% on q4_0/q4_K/q5_K (their 16 MMAs are grouped at the end of the k-step and the measured k-step, 307 ns for
230 instructions, is below 16 x 11.7 + the rest); pricing an MMA at one issue slot lands within 7% on five of seven (the table
above, re-scored). The one-slot rule is the default (`latency.json` mma_pipe 0, `--mma-pipe 11.7` enforces the spacing;
`kstep-validate.txt` carries both); how a real tile overlaps its MMA group with the next k-step's dequant is the open
mechanism, and it is the next thing a per-instruction probe of a real tile should settle. (b) *The harness has a trap.*
Independent chains must differ in their inputs or the compiler folds them; the dependent-chain latencies and the ALU
dual-issue probes (distinct initial values per chain) were not affected, every MMA throughput number from the morning was.
The folded kernels stay in `kernels.metal` marked as such.

*The class matrix: what runs beside what on a core (night; owner: "it is precisely the kind of mixed execution I was
looking for" / "you have the machine"; `mix-matrix*.txt`, `op-costs*.txt`, `pipe-ceiling.txt`; tools `perf/agx-latency/mix-matrix.py`,
`mir-check.py`, `perf/pipe-ceiling.py`).* Nine instruction classes, each as a loop body on its own simdgroups; pure runs with every
simdgroup of one threadgroup on one class (4, 8, 16 per core), mixed runs with half the simdgroups on class A and half on B.
Two things had to be fixed before the numbers meant anything, and both are now tooling:

- **The port is picked by simdgroup index.** The first split (even simdgroups A, odd B) made every MMA pair come out at the
  16-simdgroup pure-MMA time: 16 MMAs on the even half alone = 673 ns, on the (sg & 4) half = 347. Even/odd lands one class on
  two of the four ports. The matrix uses (sg & 4).
- **Three class bodies were not what their names said** (owner: "are you double checking the compiler output?"). The shift/xor
  integer chain folded algebraically to 21 ops of 128 (the same fold made this morning's "int issues at 0.25 ns" anomaly);
  consuming one half of a tile through `thread_elements()[0]` made the compiler replace the 128-byte tile load with a 2-byte
  load (so the earlier "1.4 tile loads/ns" was a 2-byte-load figure; the MMA-fed tile loads were real); `float(half(v))` became
  one round op, not the dequant's convert. `mir-check.py <kernels.metal> <kernel>...` now prints every loop body's opcode
  histogram from the backend MIR, and every probe below was checked with it before it was timed.

Pure costs (ns of port time per instruction at four simdgroups per port; the single-simdgroup issue interval in brackets):

| class / opcode | port cost | clocks | note |
|---|---|---|---|
| f32 fma (2212), f16 fma (820), f16 mul (884), f32->half (1038) | 0.67 | 1 | |
| int add / and / select (11082, 443, 12300) | 0.72-0.78 | ~1.1 | shl by an immediate folds into the add's operand shifter for free |
| shift, standalone (442, 17168) | ~1.3 | ~2 | inferred from the 96-op int class (0.9 average); not probed alone - the compiler keeps absorbing it |
| **int->float, int->half, half->int, float->int (12104/12107/12108/10226/10225)** | **2.4-2.6** | **3.5-4** | a quarter-rate path; one simdgroup alone issues them at 1.3-1.5 (2 clocks) |
| **bitfield extract (17165/17166)** | **2.33** | **3.5** | same path |
| MMA (860, 2884) | 10.6 | 16 | |
| 8x8 half tile load from threadgroup memory (13201) | 2.6 | 4 | |
| 2-byte threadgroup store (14161) / 16-byte (14245) | 2.8 / 5.0 | 4 / 7.5 | |
| 1-byte table gather, 16-byte table (13519) | 0.8 at the port | 1.2 | 37 ns per step in the chain: iq4 lookups cost latency, not throughput |
| 16-byte coalesced device load | latency-bound | | one core streams 37 KB/us from DRAM alone (26 at 16 simdgroups), 70 from L2, 130-220 from L1 |

The last line relocates a constant: the skinny ceiling's 9 KB/us/core (180 GB/s over 20 cores, 66% of DRAM peak) is the
chip's DRAM share, not a per-core path; a core alone pulls four times that.

Mixed pairs, one simdgroup of each class per port (4 + 4; cell = share of the smaller class's time that disappeared):

| | f32 | f16 | int | round | deq | tile ld | tg st | mma | dev ld |
|---|---|---|---|---|---|---|---|---|---|
| f32 | - | 0.98 | 0.80 | **0.18** | 0.95 | 0.78 | 0.97 | 0.57 | 1.00 |
| f16 | | - | 0.95 | 0.98 | 0.40 | 0.76 | 0.82 | **0.21** | 0.99 |
| int | | | - | 0.98 | **0.15** | 0.78 | 0.95 | 0.97 | 1.01 |
| round-to-half | | | | - | 0.97 | 0.90 | 1.00 | 0.71 | 0.81 |
| deq (cvt, hmul, cvt, and) | | | | | - | 0.68 | 0.99 | 0.90 | 1.00 |
| tile load | | | | | | - | 0.90 | 0.94 | 0.96 |
| tg store | | | | | | | - | 0.97 | 0.94 |
| mma | | | | | | | | - | 1.01 |

So the pipes behind a port are: **FP32** (f32 math and the float->half round), **FP16** (f16 math and the MMA), **INT** (integer
ops, converts and bitfield extracts - deq hides only 15% against int and 95% against f32), tile loads, threadgroup stores, device
loads; classes on different pipes hide 76-100% of each other, the leakage being the port's own issue (which exceeds one per
clock: f32 + f16 from two simdgroups reach 1.75 per clock). f32 against MMA hides only 57%: the MMA takes something from the
FP32 side too. The design rules this gives for a dequant-plus-MMA tile: converts and bit extracts are the expensive instructions
(four clocks each, on the pipe that nothing else needs), FP16 math competes with the MMAs, FP32 and integer work does not.

*Per-pipe ceiling* (`pipe-ceiling.py --skinny`: per simdgroup k-step, port time per pipe from the join with the costs above;
X = min(4 ports / the longest pipe, 9 KB/us/core / weight bytes)):

| kernel | FP32 | FP16+MMA | INT (int / shift / cvt / bfe) | tile ld | tg st | binding | X pred | X meas | err |
|---|---|---|---|---|---|---|---|---|---|
| q4_0 SoA | 5 | 212 | 169 (24 / 12 / 32 / 24) | 62 | 34 | stream | 15.6 | 15.95 | -2% |
| q4_K | 33 | 170 | 158 (66 / 16 / 34 / 2) | 62 | 34 | stream | 15.6 | 15.35 | +2% |
| q5_K | 32 | 170 | 255 (162 / 18 / 34 / 13) | 62 | 34 | stream (INT 15.7) | 12.8 | 13.10 | -2% |
| q3_K | 29 | 170 | 240 (126 / 35 / 36 / 6) | 62 | 34 | INT | 16.7 | 12.80 | **+30%** |
| iq4_xs | 27 | 170 | 149 (85 / 17 / 1 / 27) | 62 | 34 | stream | 16.5 | 13.75 | **+20%** |
| q6_K | 28 | 170 | 301 (101 / 24 / 34 / 48) | 62 | 34 | stream (INT 13.3) | 10.7 | 11.35 | -6% |
| q8_0 | 27 | 170 | 136 (34 / 21 / 32 / 2) | 62 | 34 | stream | 8.3 | 7.45 | +11% |

Five of seven within 6%, with the converts and extracts now the second-largest pipe on every kernel and the binding one on
q5_K/q6_K if the stream were not. The two misses are the kernels whose measured ceiling sits between the longest pipe and the
sum of all pipes (q3_K: pipes 240/170/62/34/29, measured 312 ns of port time per k-step; iq4_xs: 170/149/62/34/27, measured
291): some cross-pipe serialization the pair matrix (76-100% hidden) does not show at this scale, or an opcode still mispriced
(q3_K's 33 standalone shifts are the inferred 1.3; its 32 selects measured one clock, so not those). That is the open item of
this section, and the N-stream simulator with per-pipe occupancy is the tool that would show which.

*The contention curve, first attempt (small hours of 10-31; owner: "do that, since that has direct bearing on our model";
`nstream-sim.txt`, tools `perf/nstream-sim.py`, `perf/agx-pipes.py`).* The pad ladders already hold the target: t_step on the ladder
is a throughput per threadgroup, so t_step x resident threadgroups (4R/2) is each simdgroup's own k-step at 1, 2, 3, 4
simdgroups per port (pad 24576 -> 12288 -> 4096 -> 0 and the ceiling 4000/X):

| per-simdgroup k-step, ns | 1 per port | 2 | 3 | 4 (= ceiling) | binding at 4 |
|---|---|---|---|---|---|
| q4_0 SoA | 396 | 596 | 887 | 1003 | DRAM share |
| q4_K | 633 | 747 | 1000 | 1042 | DRAM share |
| q5_K | 712 | 855 | 1126 | 1221 | DRAM share (INT pipe 15.7 just above) |
| q3_K | 614 | 771 | 1106 | 1250 | **neither: pipes allow 16.7** |
| iq4_xs | 643 | 782 | 1062 | 1164 | **neither: pipes allow 16.5** |
| q6_K | 670 | 880 | 1204 | 1410 | DRAM share |

A lone simdgroup is unaffected by one neighbour on another port; two per port cost 15-50%; four per port cost 1.6-2.1x.
`nstream-sim.py` runs N in-order streams per port on four ports against the per-port pipes, a core-wide convert unit and the
latency table (shape: event-driven, earliest-ready-first, ~150 lines). Base settings reproduce N = 1 within +3..+25% (too slow:
a lone stream should not pay the port's pipe occupancies) and N = 2 within 10%, and for the DRAM-bound kernels the measured
N = 3, 4 points are the stream term, which the simulator does not contain. For q3_K and iq4_xs it predicts 18 and 22
k-steps/us against 12.8 and 13.75: **the real core overlaps about 1.7 pipes where the simulator overlaps 2.4.** One knob
reproduces q3_K's whole curve (a port issue slot of 0.9 ns = one instruction per 1.35 clocks: 734/764/973/1249), but the same
knob breaks q4_0 and the direct probe refutes it: a single stream alternating FP32 and three integer pipes with back-to-back
dependencies (`mixdep_u128`) issues at 1.1 instructions per clock per port at four simdgroups, INT-pipe-bound exactly as the
costs say. Also ruled out along the way, each by a probe: instruction fetch (8-byte FMAs issue at one per clock like 6-byte ones),
register count (FMA streams are flat from 24 to 80 live registers; the compiler spills at 96, and a spilled loop is 2.4x slower
per instruction - the skinny tiles sit at 54-69 registers, no spill), selects (one clock), the lookup gather (one clock at the
port). Two findings that stay: **the convert/bitfield unit is one per core** (the single-opcode probes slow from one simdgroup to
four when each has its own port, no other class does), and the per-port pipe view and the core-wide view give the same
ceiling (4 x 1/2.5 = 1/0.63 ns), so `pipe-ceiling.py` stands.

What is left as the mechanism for q3_K/iq4_xs: the two things the simulator does not model and the probes have not isolated -
threadgroup memory under load (24 tile loads + 12 staging stores per k-step per simdgroup, 16 simdgroups, the load unit's
core capacity measured at 1.23 tile loads/ns with latency growing from 37 to ~130 ns under 16 streams; the stores' bank pattern
on the skinny tile is not the mul_mm one that was probed) and the two per-k-step barriers between a threadgroup's two
simdgroups, which sit on different ports and so wait on each other's port load. Both are probe-able; both are tomorrow's.

What this leaves: (1) **the contention curve from the simulator** - run N simdgroup streams against four shared issue
ports and the load path and compare with the pad ladders (R 0.5 -> 3.1) and the grid curve; if it reproduces them, the whole
skinny family is predicted from code plus five chip constants (latency table, 200 ns stream, 4 ports, 9 KB/us/core, the
resident count from smem/registers); (2) **the four ports are measured** (6.0 sg-instr/ns/core, four simdgroups at full speed); ~~still to measure: the
load-path figure (a streaming micro-kernel at 1-6 resident threadgroups), and the per-pipe ceiling (FP32 / FP16 / MMA at
sixteen slots) against the pad ladders~~ MEASURED (night): the per-core load path is 37 KB/us from DRAM (the 9 is a chip share),
the per-pipe ceiling with measured per-opcode port costs places 5 of 7 skinny kernels within 6%; open: q3_K +30% and iq4_xs
+20% sit between longest-pipe and sum-of-pipes; ~~(2b) **the mm family is reopened**: the MMA pipe is ~30 TFLOPS and the prefill tiles
reach 7 - the simulator on a mul_mm tile's loop body says where the other three quarters go~~ (2b) RETRACTED: the MMA rate
is 7.8 TFLOPS and the mm tiles sit at 97-99% of it (evening section); what remains open there is the single-simdgroup
MMA/ALU overlap rule for the simulator; (3) the simulator's misses: iq4_xs -13% (the lookup-table gathers: price them with a gather micro-kernel) and q8_0 -16%
(its join is the 5120x48 row - rejoin at 17408x5120); the loop-body rule and the activation-load heuristic are the fragile parts; (4) the small-grid under-fill (why four queued threadgroups leave a core at a quarter of
the counter) is unchanged; (5) the other latency-bound families get the same treatment: a join, the simulator, a ladder to
check - fa decode first.

## Log

- 2026-10-29: stub opened after the Oct 27-29 prediction campaign (six `exp/m4-*` branches, one commit each; prefetch on the
  skinny tile refuted a third time; no model transferred to llama.cpp). Its timing host (`exp/m4-measurement-harness`
  41d5ffa8b) is a qualified ISOLATED timer; useful for calibration pairs, not for the target.
- 2026-10-30 night: first fit + intervention scorecard + residency ladder, section above. Branch exp/kernel-perf-model.
- 2026-10-31 small hours: contention curve from the pad ladders (t_step x resident TGs = per-sg k-step at 1-4 sg/port: lone sg
  unaffected, 4/port = 1.6-2.1x); nstream-sim.py (N streams x 4 ports, per-port pipes, core-wide convert unit) matches N=1-2 and the
  DRAM-bound kernels, over-predicts q3_K/iq4_xs by 40-60% (real core overlaps ~1.7 pipes, sim 2.4). Refuted as the cause by probes:
  port issue rate (mixed dependent stream issues 1.1/clock), fetch (8-B FMAs at 1/clock), registers (flat to 80; spill cliff at 96:
  2.4x per instruction), selects, gathers. Found: convert/bfe unit is one per core. Open: tg memory under load + the per-k-step
  barrier between a TG's two simdgroups on different ports.
- 2026-10-30 night: the class matrix (owner: "you have the machine"). Port = f(simdgroup index), even/odd splits collide; three
  probe bodies were not what they claimed (folded int chain, 2-byte loads for tile loads, round for convert) -> mir-check.py
  checks every loop body against the backend MIR before timing. Pipes behind a port: FP32, FP16+MMA, INT+converts+bfe, tile
  load, tg store, device load; different pipes hide 76-100% of each other. Converts and bitfield extracts cost 3.5-4 clocks of
  port time (quarter-rate path), selects/adds/f16 mul/f32->half one, MMA 16, tile load 4, 2-byte tg store 4. Per-core load path
  37 KB/us from DRAM (the 9 KB/us/core is the chip share). pipe-ceiling.py: 5 of 7 skinny X within 6%; q3_K +30%, iq4_xs +20% open.
- 2026-10-30 evening: the mul_mm chase. The morning's MMA probes were folded by the compiler (8 MMAs where 32 were counted);
  clean probes: an MMA costs a port 16 clocks (10.5 ns), 0.38 MMA/ns/core = 7.8 TFLOPS = the FMA roof - no matrix unit on this
  GPU. The acch n64 tile is at 97-99% of it; the mm family stays closed. Unit costs for tile loads, staging stores and barriers
  measured; simulator MMA rule = one slot (the 11.7 spacing enforced over-predicts the skinny k-steps), validation re-scored
  (5 of 7 within 7%); the single-simdgroup MMA/ALU overlap is the open mechanism. Section above.
- 2026-10-30 13:00: dual issue settled - one simdgroup never dual-issues (FP32/FP16/INT ~~/MMA~~ all serial), the core has four ports
  at one instruction per clock with separate FP32 and FP16 pipes (mixed 7.7/ns vs 6.0), ~~MMA = four slots; the MMA pipe is ~4x
  what the mul_mm tiles reach - the mm 'roof' was the best kernel, not the hardware~~ (the MMA half RETRACTED the same evening:
  folded probes; MMA = sixteen slots, the tiles are at the roof).
- 2026-10-30 12:40: six cached points - no bits trend, repeats to 0.1%; misses are iq4_xs (LUT gathers) and q8_0 (wrong-shape join).
- 2026-10-30 12:10: THE T0 FORMULA: latency table from micro-kernels (perf/agx-latency), in-order k-step simulator over the
  MIR join (perf/kstep-sim.py); cached chain within 10% on four kernels with nothing fitted, stream within 14% on seven with one
  constant (200 ns); ceiling X = min(4 ports / issue time, 9 KB/us/core / bytes) within 15% on seven. Section above.
- 2026-10-30 11:05: PROBE=2 ported to the generic tile; its cached chain is 110-130 ns (not 70): t0 is a per-tile table,
  the exposed DRAM share a fifth of the k-step on all four kernels, X unmoved by the stream on all four. First milestone ends here.
- 2026-10-30 10:30: probe-arm ladders (half instructions: t0 +25%, X unchanged; no stream: t0 -24%, X +7%; regpad:
  residency only) and five more kernels: t0 is 91 ns (SoA tile) vs 142-170 ns (generic tile) = exposed DRAM latency; X tracks
  the weight stream (~9 KB/us/core), not any instruction class. Nine-kernel table above.
- 2026-10-30 09:12: residency-vs-grid design run (23 shapes): no staircase, R = f(grid) not lifetime, R(n) fitted, model
  +-8% at 8-64 TG/core, small grids over-predicted, the counter's small-grid reading is the open mechanism. Section above.
- 2026-10-30 02:15: fleet residency on all 92 rows (census traces replayed), the ud skinny ladders (own t0/X each), held-out
  scoring 13/17 within 5%; the earlier 'transfers within 2%' row corrected (assumed R was wrong on four shapes). Open items
  rewritten under 'What this leaves'.
