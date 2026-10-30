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

What this leaves: (1) **R has to come from the grid** - the seven q4_0 shapes and the ten ud shapes draw one saturating
curve of R against threadgroups per core (8 -> 1.6-1.9, 16 -> 2.5-2.7, 27 -> 2.8-3.1, 388 -> 3.3-3.85) with the pad-0 cap
set by the register count; two more grid sizes per kernel (e.g. 32 and 64 threadgroups per core via M) would pin the curve
and settle whether the 8-per-core misses are R's time-average or a real ramp; (2) **t0 from the MIR** - the load-use distance
is the candidate (4 points); the regpad ladder on q4_0 (R flat 2.76, time +5..+17%) is the direct test of what moves t0 at
fixed geometry; (3) **the ceiling X** - 13-16 k-steps/us/core, not instructions, not DRAM (183 GB/s of weight bytes at the q4_0
ceiling); the MMA pipe and the threadgroup-memory path are the candidates, and the Oct 26 no-stream arm (-10..-15%, R
unchanged at 3.10) is the one existing measurement that bears on it; (4) the other latency-bound families (fa at R 1.9-2.0,
gdn, mv at 1.04-1.4x the floor) have not been given this treatment: fa decode is the obvious next ladder since every shape
sits at the same R.

## Log

- 2026-10-29: stub opened after the Oct 27-29 prediction campaign (six `exp/m4-*` branches, one commit each; prefetch on the
  skinny tile refuted a third time; no model transferred to llama.cpp). Its timing host (`exp/m4-measurement-harness`
  41d5ffa8b) is a qualified ISOLATED timer; useful for calibration pairs, not for the target.
- 2026-10-30 night: first fit + intervention scorecard + residency ladder, section above. Branch exp/kernel-perf-model.
- 2026-10-30 02:15: fleet residency on all 92 rows (census traces replayed), the ud skinny ladders (own t0/X each), held-out
  scoring 13/17 within 5%; the earlier 'transfers within 2%' row corrected (assumed R was wrong on four shapes). Open items
  rewritten under 'What this leaves'.
