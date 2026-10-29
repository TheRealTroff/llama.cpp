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

## Log

- 2026-10-29: stub opened after the Oct 27-29 prediction campaign (six `exp/m4-*` branches, one commit each; prefetch on the
  skinny tile refuted a third time; no model transferred to llama.cpp). Its timing host (`exp/m4-measurement-harness`
  41d5ffa8b) is a qualified ISOLATED timer; useful for calibration pairs, not for the target.
