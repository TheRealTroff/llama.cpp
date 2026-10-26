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

### The byte floor per group (2026-10-25 night, owner: "the other resources are just as shared within each group. Even so, let's start with bytes")

The owner's three points: (a) designs should be planned against ops running concurrently, (b) the memory traffic is exactly
predictable, (c) it feels wrong that a faster kernel does not at least leave room for the others. (b) gives the first analytic
column: every member's bytes per call follow from its shape (weights at the stored format's bpw, `kernel-census.work`; the FA rows
= the K and V cache slices at the run's KV bpw x 4 KV heads + q/o), so a group has a floor F = bytes / 252 GB/s that no kernel of
the same bytes can beat. `census-groups.py` now prints per group `floor us`, `F/W` (byte utilization) and the aggregate `GB/s`, per
op its `MB/rd`, `floor`, `own` GB/s (bytes / t_i alone) and `grp` GB/s (its group's aggregate). `kernel-census.py`'s command dispatch
went under a main guard so the group script imports its byte model (`python3 perf/kernel-census.py plan|metrics|report` unchanged).
Formats named `*_soa` read the stored format's bytes. Both lines are Turbo4 KV (4.125 bits/elem; `--kv f16|q8_0` for other runs).

Same logs as the join above (pinned width 8, depth 7, `cgroups-oct25-*`):

| line | GB moved / round | byte floor ms/rd | sum of group walls | F/W | aggregate GB/s |
|---|---|---|---|---|---|
| q4 | 18.75 | 74.4 | 125.9 | 0.59 | 149 |
| ud | 20.85 | 82.7 | 160.9 | 0.51 | 130 |

Per group (W = wall, F/W = byte utilization, GB/s = aggregate through the group):

| group (width 8) | q4 F/W | q4 GB/s | S/W | ud F/W | ud GB/s |
|---|---|---|---|---|---|
| ffn gate+up (2 ops) | 0.71 | 180 | 1.09 | 0.51-0.62 (iq4_xs / q5_K) | 129-157 |
| ffn_down | 0.55 | 140 | 1.00 | 0.42-0.48 (iq4_xs / q5_K), iq3_s 0.28 | 71-122 |
| attn q+k+v (3 ops) | 0.57 | 145 | 1.02 | 0.47 (q4_K) | 119 |
| attn_out | 0.51 | 128 | 1.00 | 0.45 (q5_K) | 114 |
| GDN in_proj + 2x48 (3 ops) | 0.59 | 147 | 1.61 | 0.50 (q4_K) | 127 |
| lm_head | 0.74 | 187 | 1.00 | 0.74 (q6_K) | 186 |
| ffn gate+up at WIDTH 2 | 0.79 | 200 | 1.03 | - | - |
| GATED_DELTA_NET | 0.27 | 68 | 1.01 | 0.28 | 69 |
| FLASH_ATTN_EXT (8.5K cache) | 0.10 | 25 | 1.00 | 0.10 | 26 |
| ADD 5120x8 | 0.17 | 43 | 1.04 | - | - |

**What the column says.**

1. **No width-8 decode matmul group is at the byte floor.** Utilization is 0.51-0.79 on q4 and 0.28-0.74 on ud; the rest of the
   wall is not DRAM. The same ffn weights run at 200 GB/s at width 2 and 180 at width 8 (q4), and the width-1 mul_mv sits at
   1.1-1.2x the floor (`agx-backend-access.md`): the per-byte cost grows with width, which a DRAM limit cannot do. This is the
   width-4 memory/issue crossover (`ud-w4-ceiling`) seen from the other side: at the census width the groups are on the
   instruction side. FA (0.10) and GDN (0.27) are compute-class as known; the elementwise rows are launch-bound (ADD at 11 us).
2. **The attention group is saturated on something that is not bytes.** Three independent ops (q, k, v) with S/W 1.02 - they do
   not overlap at all - while 43% of the byte budget is unused. That is the measurement behind (c): split-K k/v was not a
   kernel that consumed less of the shared resource, it was the same instruction stream spread over more threadgroups (plus the
   partial sums), so it had nothing to give back. A kernel leaves room only by consuming less of whatever the group is saturated
   on; at width 8 that is not DRAM, and the candidate is the instruction stream (the census has instructions per MB for every
   traced kernel - the next column, see below).
3. **`own` vs `grp` GB/s is the chain signature, S/W says whether it matters.** The 48-row projections run alone at 4-5 GB/s
   inside a group that moves 67-99 GB/s, and that group has S/W 1.6-1.7 (slack): a latency chain the wall can see, and it paid
   when shortened (-14..-17 us/layer). The k/v rows run alone at 41 GB/s inside a group at 114-171 GB/s, but that group has S/W
   1.02 (no slack): dragged at the group's rate, they gave back 3-7 us. Same per-op signature, opposite outcome, separated by the
   group's S/W.

**The prediction rule this gives (bytes version).** For a lever on an op in group G with wall W, bytes B, serialized S:
- F/W >= ~0.9: G is at the floor; only fewer bytes pay, priced at 252 GB/s (bytes removed / 252).
- F/W < 0.9 and S/W ~ 1: G is saturated on a non-byte resource. More parallelism for one member pays nothing (split-K). Fewer
  bytes still pay, but at the GROUP's measured rate (bytes removed x W/B), because for a mul_mv the instructions go with the
  bytes; a kernel that cuts instructions per byte is the lever, and its cap is the short-chain probe (b) of the ladder above.
- S/W >> 1: G has slack; a member with `own` << `grp` is a chain; shortening it pays up to its `exposed_hi`.

**Caveats.** Pinned width 8 is the census width; the picks run widths 3-7, where utilization is higher (the width-2 row). The FA
bytes take K and V both at 4.125 bits and 4 KV heads (right for both lines; the drafter's FA, 0.3 ms/round, uses the same
assumption). 252 GB/s is the CPY stream figure; ~230 is what a mul_mv has reached, so an F/W of 0.9 is "at the floor".

**Next column (not built): instructions.** The owner's premise is that the other resources are shared the same way. The
census's per-kernel instruction counts give each group an instruction total; against an aggregate issue rate that is a second
floor, I/W, and a group below both floors is the one whose wall is set by occupancy, dispatch order or a latency chain - exactly
where the short-chain probe (b) is worth running first. Bytes were the first column because they need no trace.

### Instructions per type (2026-10-26, owner: "instructions per se is interesting, but per type would be even more so")

The owner's reason for wanting the instruction stream decoded: execution units differ, so a group's wall may be set by one
type, not by the count. Built on the same branch:

- `perf/agx-opcode-classes.py`: an integer solve assigning each of the 403 MIR opcode numbers seen in the census joins to one
  of Xcode's static classes (FP16 / FP32 / INT16 / INT32 / untyped ALU / device load / device store / threadgroup load / store /
  spill / branch / none) so that all 63 distinct joined binaries' static counters come out. Hard pins from the MIR text: a memory
  operand pins load/store by address space (1 and 2 device, 3 threadgroup, `%stack` spill - Xcode counts spill traffic under
  neither); fast-math flags on a 16-bit destination pin FP16. Both are exact against Xcode on every binary. Residual 130
  instructions over 182 rows, all in the FA kernels; **280/403 opcodes pinned across 3 tie-broken solves, 94.1% of executed
  instructions**. Output `perf/agx-opcode-classes.json` (the 2026-09-09 pass pinned 36 of 159 from 15 binaries). The profiler's
  fixed issue weight per opcode (`agx-backend-access.md`: cost = executed x w x k) is consistent across every kernel and splits
  the untyped ALU bucket further: w4 = 32-bit shifts, bit-field extracts, uniform ALU; w6 = 16->32 converts; w8 = 64-bit pair
  ops; w16 = the simdgroup matrix ops (filed under FP32 by Xcode).
- `census-groups.py --census <dir>`: per concurrency group, executed instructions per us of group wall by class (K/us), the total,
  and the issue-weighted total ("units", loads and stores at 0).

**The table (width 8, `census-1024-*-w8-dec` joins over the `cgroups-oct25` walls):**

| group | S/W | F/W | K instr/us | units/us | FP16 | FP32 | INT32 | OALU | DLD | TLD | stall % | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q4 ffn gate+up (q4_0 SoA) | 1.09 | 0.71 | 39.1 | 103 | 10.8 | 4.1 | 2.5 | 12.5 | 0.7 | 4.1 | 9 | 54 |
| q4 ffn_down | 1.00 | 0.55 | 41.1 | 108 | 11.5 | 4.3 | 2.6 | 13.1 | 0.7 | 4.3 | 18 | 54 |
| q4 lm_head | 1.00 | 0.74 | 44.5 | 117 | 12.3 | 4.6 | 2.9 | 14.2 | 0.8 | 4.7 | 4 | 54 |
| q4 ffn gate+up at width 2 | 1.03 | 0.79 | 11.7 | 22 | 0 | 5.9 | 0.7 | 3.8 | 0.6 | 0 | 4 | 71 |
| ud ffn gate+up (iq4_xs) | 1.10 | 0.51 | 28.2 | 62 | 0 | 6.6 | 3.2 | 6.6 | 4.5 | 2.8 | 22 | 65 |
| ud ffn_down (q5_K) | 1.00 | 0.48 | 23.6 | 54 | 0 | 4.4 | 4.8 | 10.1 | 0.6 | 1.6 | 25 | 69 |
| ud attn_out (q5_K) | 1.00 | 0.45 | 22.1 | 50 | 0 | 4.1 | 4.5 | 9.4 | 0.6 | 1.5 | 25 | 69 |
| ud lm_head (q6_K) | 1.00 | 0.74 | 35.2 | 86 | 0 | 6.4 | 9.5 | 12.3 | 0.7 | 2.7 | 9 | 69 |
| FA decode 8.5K (both) | 1.00 | 0.10 | 17 | 59 | 0 | 4.1 | 4.2 | 2.5 | 0.7 | 2.4 | 14 | 96 (+48 B spill) |

**What it says.**

1. **No class has a common ceiling across the saturated groups.** Each class maximum is set by whichever kernel uses that class
   most (FP16 and OALU by the q4_0 SoA kernel, INT32 by q6_K, device loads by iq4_xs's LUT reads at 16% of its stream). With
   Xcode's taxonomy alone the walls do not line up on one unit, so "which unit" is not answered by counting per type - the
   per-type table is a necessary input, not the answer.
2. ~~**Apple's issue weights separate the two kernel families by 2x.** The q4_0 SoA groups run at 99-117 K weighted units/us;
   20 cores x 4 issue/clock x ~1.5 GHz is ~120 K/us, so under Apple's apportioning model those groups are at 85-95% of a single
   issue port per SIMD - issue-bound, which is also what their 4-18% stall says.~~ **TESTED 2026-10-26 and REFUTED
   (`exp/issue-probes`, `perf/issue-probes.md`):** halving the executed instructions of the q4_0 SoA tile (no dequant, same bytes)
   changed its time +2..+6%; removing its DRAM weight stream (same instructions) gave -10..-15%; halving the resident threadgroups
   doubled it (q5_K: 2.4x). Both tiles are LATENCY-bound (occupancy / prefetch distance), the issue port is not the limiter and
   Apple's weights do not describe it. The weighted rate stays in the table as a measurement; it is not a floor. The rest of this
   point, as written at the time: The K-quant groups sit at 48-62 (q6_K head 86) with
   15-30% stall and 65-69 registers against q4's 54: neither the byte floor (F/W 0.42-0.62) nor the issue port binds them. In the
   q4_0 SoA stream the weighted budget is 42% simdgroup-matrix ops (w16), 39% w4 dequant shifts/extracts, 11% f16 arithmetic: the
   dequant integer work costs as much issue as the matrix multiply itself. The weight table is Apple's apportioning model, not a
   demonstrated hardware cost table (`agx-backend-access.md` finding 2) - this is a reading to test, and it makes two testable
   predictions: (a) a q4_0 SoA change that removes w4 instructions per byte pays, one that removes bytes alone does not; (b) a K-quant
   kernel responds to the footprint probe (fewer registers / more resident simdgroups), not to instruction removal.
3. **The width-2 row is the control.** Same weights, same format, 1.03x the byte floor, 22 K units/us: at width 2 the kernel is on
   the DRAM floor and the issue port is idle; by width 8 the per-byte instruction cost has moved the wall to the issue side.

**Caveats.** The 403-opcode map has 123 ambiguous opcodes, 5.9% of executed instructions, mostly INT16-vs-OALU splits that do not
move the table. Drafter (m2) rows have no census join (0% cover). The absolute executed counts are the profiler's; the census's own
UNDERCOUNT check applies to mma rows only. The clock in the 120 K/us figure is nominal.

**Tested (2026-10-26).** Both arms ran the same night, `perf/issue-probes.md` on `exp/issue-probes`: (a) refuted on both halves,
(b) half right - q5_K is the more occupancy-sensitive tile but q4_0 SoA degrades from the first pad step too. The lever both
name is latency hiding per core. ~~**Next.** The two predictions above are each one arm:~~ (a) is a kernel-form experiment the prescreen skill ranks offline; (b) is
probe (c) of the ladder (fewer threadgroups / lower register budget via `-fmax-register`-style limits or a smaller tile) read
against the K-quant groups' walls. A width-4 pair of the census + group profile would show the pick's own regime.

**Needed vs streamed (owner 2026-10-26, on the FA row's F/W of 0.10):** the floor F counts the bytes the problem NEEDS - each K and V
element once - and that is the right definition: it is the optimality bound no implementation can beat, and the width-8 FA call's
0.10 is a true statement about how far that kernel is from it. What the kernel STREAMS (two 24-row tiles per KV head = two K/V
passes, the per-row mask fetches) is the implementation's choice, so it is a separate measured column: `str/F` in the `--census`
table = issued device+constant load bytes from the joins over F. FA 8.7K streams ~5x its need (2x from the two passes), the q4_0
SoA skinny kernels 1.5-2.1x (the activation panel re-read per 32-row tile), the K-quant SoA kernels 0.9-1.4x, iq4_xs 5-6x (the
LUT reads from constant memory, cache hits). A first version of this section folded the FA pass count into F - reverted: that hid
the lever inside the floor.
