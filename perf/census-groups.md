# From profile to wall: the concurrency-group profile (GGML_METAL_PROFILE=2) and per-op exposed-time bounds - 2026-10-25

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

(results below)
