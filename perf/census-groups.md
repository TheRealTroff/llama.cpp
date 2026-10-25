# From profile to wall: the concurrency-group profile (GGML_METAL_PROFILE=2) and per-op exposed-time bounds - BUILT + CALIBRATED + MERGED 2026-10-25 evening (owner: "Merge"; prod `9fd97533c`)

Owner (after the decode FA at 200K landed): "What is really going on with the 'shadowing'? Can we do better in our predictions
going from profiling to real perf?" -> "Let's see what it gives us." Branch `exp/census-groups`, worktree `llama.cpp-censusgroups`,
off prod `89d67cf26`. The discussion deferred on 2026-10-25 morning (`census-shadow-tooling-discussion`) is this note.

## What is going on

- The real encoder is ONE concurrent compute pass (`MTLDispatchTypeConcurrent`). `ggml_metal_op_concurrency_check` puts a memory
  barrier only where a node writes a range an earlier node of the group touched, or reads an earlier node's output; the graph
  becomes a chain of **concurrency groups**, and inside a group the hardware scheduler fills the 20 cores with threadgroups from
  any dispatch whose inputs are ready.
- A profiled op time (`GGML_METAL_PROFILE=1`: one SERIAL encoder per op, timestamps at the encoder boundaries) is the op's
  latency run alone on an idle GPU. A 48-row GDN projection is a serial K reduction on a handful of threadgroups: it holds a few
  cores for 70-85 us while the rest of the machine would idle; in its group the idle cores run `ssm_in`, so that latency costs
  nothing on the wall. A group's wall is roughly max(longest latency chain in the group, group work / machine throughput); an op
  reaches the wall only through one of those two terms. Hence the row-specific shadow: 48-row projections 85-100% hidden, the
  1024-row k/v rows ~45% exposed next to the throughput-bound `attn_q` (`gdn-proj-shadow.md`, deletion probes).
- Why the profiler cannot see it: the M4 Pro supports counter sampling at **stage (encoder) boundaries only** - `MTLCounterSamplingPoint`
  DispatchBoundary is unsupported (checked 2026-10-25 with `supportsCounterSampling:`; Draw/Dispatch/TileDispatch/Blit all 0,
  StageBoundary 1). Timestamps inside a concurrent encoder do not exist, so the only way to time an op is its own serial encoder,
  which destroys the information the census needs.

## What was built

1. **`GGML_METAL_PROFILE=2` - the group-timed profile.** The barrier IS an encoder boundary: at every node-level
   `ggml_metal_op_concurrency_reset` the op context ends the current timed concurrent encoder and starts the next one (same
   `MTLComputePassDescriptor` timing as mode 1, `MTLDispatchTypeConcurrent`), recording the raw graph indices of the nodes encoded
   in each group. Execution is the unprofiled schedule up to the encoder overhead per group (a barrier already sits at every
   boundary). A barrier taken INSIDE a node's encode (fused GDN paths, the mul_mat src1 copy) stays a memory barrier within the
   group - the op functions hold the encoder pointer - and is counted ("in-node barriers"). The dump prints per distinct group
   (FNV of its member list): total ms, count, us/call, `gid`, n, then one `ggml_metal_prof_gmember` line per member in the
   mode-1 key format, so a mode-1 log of the same configuration joins on the keys. Sample buffer: 4096 samples = 2048 groups per
   command buffer; past that the groups run untimed (counted). Files: `ggml-metal-context.m` (mode branch, group table, dump),
   `ggml-metal-ops.cpp` (`prof_group*`, the split in `concurrency_reset`, registration after the check), `ggml-metal-device.m`
   (`ggml_metal_encoder_init_timed_concurrent`).
2. **`perf/census-groups.py <mode1.log> <mode2.log>`** - per group: W = measured wall, S = sum of member mode-1 times; per member
   op i with M_i = max of the OTHER members:
   - `exposed_hi = min(t_i, max(0, W - M_i))` - the group cannot be shorter than its longest other op;
   - `exposed_lo = max(0, W - (S - t_i))` - even with every other op back to back, op i must fill the rest of W.
   Aggregated per op key over its groups into ms/round beside the serialized charge; plus the run-level hiding ratio
   (sum of S vs sum of W over the phase's groups) and the largest groups.
3. `run-w8-decomp.sh STEPS=metalgroups` runs mode 2 and the join after the metalprof step.

## Calibration (the two rows measured by deletion on 2026-10-25)

Pinned width 8 (depth 7), Turbo4, chat benchprompt, 300 tokens, both lines, `cgroups-oct25-*` (anchor / metalprof / metalgroups
per line). Expected from `gdn-proj-shadow.md`: 5120x48 GDN projections 0.5-1.3 ms exposed of 6.8 ms charged on q4, ~0 of 8.1 on
ud; 5120x1024 k/v 1.07 ms of 2.4 on q4, ~0 on ud.

### Runs (`cgroups-oct25-{q4,ud}-d7-{anchor,metalprof,metalgroups}`, 20:50-21:10)

| line | anchor round (dec_syn_tg + draft) | mode 1 round | mode 2 round | sha (all three) |
|---|--:|--:|--:|---|
| q4 | 121.6 (106.9 + 14.7) | 143.7 (1.18x) | 129.1 | `86213d038a29` |
| ud | 155.3 (138.1 + 17.2) | 179.1 (1.15x) | 157.5 | `ce826d8a3cbd` |

Mode 2 runs the real schedule: the sha is the anchor's and its round is 1.5-6% over the anchor (1068 / 1116 timed groups per
decode round, ~2-7 us of encoder start/end each - the same per-encoder cost every mode-1 op time carries). Mode 1 inflates the
round 15-18%.

### The join (decode phase, ms per round)

| | q4 | ud |
|---|--:|--:|
| serialized sum of parts (mode 1 over the groups) | 139.5 | 178.4 |
| sum of group walls (mode 2) | 125.9 | 160.9 |
| hiding factor as printed / after the encoder overhead | 1.11x / ~1.18x | 1.11x / ~1.15x |

After the split overhead (mode 2 round minus anchor) comes off the group walls, the hiding factor equals the anchor-vs-mode-1
ratio on both lines - the group walls add up to the GPU round.

**Where the hiding is.** Almost nowhere in the big rows: the ffn pairs (gate+up in one group, `S/W` 1.09-1.10), every single-op
group (`down`, `o_proj`, the heads, the FA calls, the GDN kernel: `S/W` 1.00). The one group with real overlap is the GDN input
block: `in_proj` 5120x10240 + the two 5120x48 projections (+ their q8 twins on ud), `S/W` 1.61 (q4) / 1.69 (ud) - the shadow
row. **Prefill: no hiding at all** - every large group is one stream-bound 512-wide matmul (`S/W` 1.00 across the board on both
lines); for prefill the per-op profile already is the wall, which is why the prefill levers of this month translated 1:1.

### Calibration against the deletion probes

| row (width 8) | charged (serialized ms/rd) | bound [lo, hi] | deletion probe (`gdn-proj-shadow.md`) |
|---|--:|--:|--:|
| q4 `5120x48` GDN projections (q4_0, 8 groups) | 7.04 | **[0.07, 1.91]** | 0.5-1.3 ms |
| ud `5120x48` GDN projections (q8_0, 10 groups) | 8.45 | **[0.03, 0.95]** | ~0 (-0.2 / +0.4) |
| q4 `5120x1024` attn_k/v (q4_0, 6 groups) | 2.48 | **[0.03, 1.45]** | 1.07 ms |
| ud `5120x1024` attn_k/v (q5_K+q6_K+q8_0, 3 rows) | 3.31 | [0, 2.03] | ~0 (+0.2) |

Every measurement falls inside its bound, and the bound orders the rows the way the probes did (q4 projections above ud,
q4 k/v the most exposed). What it would have said before anyone built: the Aug 22 and Oct 24 #1 census row is worth at most
1.9 ms (q4) / 1.0 ms (ud) of its 7-8 ms charge - not a 5% lever. The bound is loose where several small ops share a group
whose wall is set by a big neighbour (ud k/v: hi 2.0 ms against a measured 0): `lo = 0` means "possibly fully hidden"; such a
row is a skip-arm candidate, never a lever, until the arm says otherwise. The exact instrument for a surviving row stays the
deletion probe.

### What it gives us

- A **measured** per-group wall next to the serialized per-op time, from one extra profiled run of the same configuration
  (~2 min per line), with the real schedule (sha = anchor) and the GPU round reproduced to the encoder overhead.
- Two columns the census was missing: `exposed_lo` / `exposed_hi` per row. Reading rule: rank levers by `exposed_hi`, not by the
  serialized charge; a row with `hi/ser` under ~0.3 is a shadow row; prefill rows need no correction.
- The group table itself names the neighbour that hides a row (the heaviest member of its group), which is the thing a
  replacement kernel or a fusion would have to beat.
- **Checked against a real kernel the same night (`kv-splitk.md`, branch `exp/kv-splitk`):** a split-K form of the q4_0 skinny
  tile cut the k/v rows 40% and the 48-row projections 48-64% per call (serialized round -6 ms); the group walls moved -14..-17 us
  per layer in the GDN input group and only -3..-7 us in the attention group = 0.4-0.7 ms/round in all, which the anchor confirmed
  once three controls had set the per-group floor (0.06 ms/round on these groups, 0.5-2 ms/round of machine-state drift on the big
  stream groups that no kernel touched). ~~The k/v rows' exposure is the tail of a chain that starts behind q's threadgroups~~ -
  wrong: a 40% shorter k/v chain gave back almost none of it; the exposure is contention with `attn_q` (threadgroup slots, the DRAM
  queue), which a faster k/v kernel does not remove and a deletion probe does. The per-group diff is the attribution instrument.
- Not captured: cross-group interaction through the cache/DRAM, CPU-exposed gaps between command buffers (the submit profiler's
  job), and the drafter/target split (both are in the decode phase here, keyed `m1`/`m2`).

### The step that was missing (owner 2026-10-25 night: "we're still not great at predictions")

The bound is a cap, not a forecast: the k/v and projection rows had near-identical bounds and paid 14x apart when both were made
40-60% faster per call (`kv-splitk.md`). What a profile cannot tell apart - a group longer than its biggest member because of a
CHAIN (start position, latency; a shorter chain fixes it) or because of CONTENTION (slots, the DRAM queue; only removing the op
fixes it) - a probe ladder can, before any kernel is written: (a) the deletion probe = the cap for anything built for the row;
(b) a SHORT-CHAIN probe = the same kernel with its K walk truncated to one slice (garbage output; same threadgroups, same slots,
no chain) = the cap for a faster kernel of the same footprint, and (a) minus (b) = the contention share; (c) a smaller-footprint
probe (fewer threadgroups/threads, full chain) = the cap for an occupancy change. One env knob and one group-profiled arm each,
read against the row's own group wall over >= 3 controls. Run tonight, (b) would have priced the k/v split at ~0.1 ms and the
projections at a few tenths before the kernel existed. Not built: the truncation knob (a `GGML_MM_SHORTK_NE01=<rows>` gate in the
skinny route that sets the kernel's ne00 to one K-slice) is the next tooling item if the owner wants the ladder.

### Done at the merge / open

1. MERGED prod `9fd97533c` (inert unless `GGML_METAL_PROFILE=2`; the mode-1 and unprofiled paths are unchanged - post-merge sha
   sanity in the section below). `run-census-full.sh` runs `metalprof metalgroups` and prints the bounds above each decode census.
2. OPEN: the two columns inside `kernel-census.py`'s own table (today the join is a separate `-groups.txt` beside the census).
3. The skip arm stays the exact step for rows whose `hi` is still worth building for.

### Post-merge sanity on the prod build (`cgroups-merge-oct25-{ud,q4}`, prod `9fd97533c`, unprofiled)

| line | pick-n6-300 (f16, canonical) | turbo4-n3-300 |
|---|---|---|
| ud | 31.70 t/s acc 69.0% `9c53aaade052` = record | 32.33 t/s acc 72.4% `d180ae89f168` = record |
| q4 | 32.92 t/s acc 63.5% `d2953fccfb41` = record | 34.94 t/s acc 67.8% `f07b0f8c58e6` = record |

The unprofiled encode path is unchanged by construction (`prof_group` false takes the old branches); the shas say the same.
