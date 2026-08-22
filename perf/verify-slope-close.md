# Closing the 23.3 ms: does the verify-slope lever actually exist?

Status: **open**

Opened 2026-08-22 after `head-to-head-aug22.md`. This is the whole remaining game - every
other item on the lever board is worth single digits next to it.

## The target, stated exactly

Matching dflash_mlx's measured 29.613 t/s at our own 3.75 committed tokens/round needs a
**126.6 ms round** against today's **150.0** - a **23.3 ms** cut. `round-decomp-fused.md`
scopes the verify-slope lever at ~20 ms, i.e. ~86% of the gap. Round today is 130.0 ms
verify GPU + 16.4 drafter GPU + 2.7 CPU.

## Task 1: does the 20 ms exist? (do this first - it is free)

**The lever's own named components no longer add up, and nobody has re-checked.**
`round-decomp-fused.md` composes the ~20 ms as: big-mat slope (*explicitly closed - at MLX
microbench parity*), FA residual 4.5, GDN scan ~4, elementwise/misc ~6 (*downgraded by the
concurrency argument*), small-ne01 ~8 (*refuted at e2e*). Strike the closed and refuted
items and it is not obviously 20 ms of anything.

New data from `slope-sweep.md` sharpens this. llama-bench width scaling at the prod env
(short KV, no speculation) is **73.0 -> 123.1 ms for N=1->7 = 1.686x**, while the in-graph
verify slope at 8.4k KV is **130.0/72.8 = 1.79x**. The gap between those two is only
**~6.9 ms**, and the known post-split FA residual is ~4.5 of it. So the long-KV and
spec-specific overhead on top of pure width scaling looks *small*, which would mean the
1.79x is mostly dense-op width scaling - the thing already declared at parity.

If that holds, the 1.5x target is not reachable by removing overhead, and the honest
conclusion is that the remaining gap is structural. **Establish this before building
anything.** Cheap: both numbers already exist, this is arithmetic plus one careful
re-derivation of what the in-graph N=7 pass contains that a short-KV pp7 does not.

Caveat to respect: llama-bench pp7 runs with a near-empty KV cache, so its FA cost is
nothing like the verify pass's 8.4k-KV FA. The two slopes are comparable in *shape*, not
absolutely. Do not treat 123.1 vs 130.0 as a like-for-like difference.

## Task 2: their side, and a trap in it

The oMLX "implied ~1.5x slope" everything is measured against was never measured - it is
arithmetic from their floor, throughput and tokens/cycle. Their artifacts contain a
`phase_timings_us` block that looks like it would settle this directly:

    commit 3933.6, draft 321528.6, draft_incremental 260296.7, draft_prefill 61232.0,
    prefill 67577129.9, replay 34060.7, verify 488100.8      (us, one 300-token run)

**It does not, and this is a trap worth documenting before someone builds on it.** For one
run: generation wall is 300/29.613 = **10130 ms**, but verify + draft + replay + commit =
488.1 + 321.5 + 34.1 + 3.9 = **847.6 ms, only ~8% of it**. Meanwhile their `prefill`
67577 ms does match `ttft_ms` and 8288/122.7 tok/s exactly.

So the prefill timer captures wall time and the generation-phase timers do not - almost
certainly MLX lazy evaluation: nothing inside the decode loop forces a sync, so those
timers measure submission, not GPU execution. Same class of error as our
profiler-inflated `dec_sub_tg`, inverted: theirs *under*-measures.

**Verify this before using any of it** (sum the phases across all 5 runs against wall time;
check whether dflash_mlx has a sync/eval option for the generation loop). If it can be made
to sync, it hands over their verify-vs-draft split directly and turns "implied ~1.5x" into
a measured number. If it cannot, that framing stays unfalsifiable and Task 1's arithmetic
is the only honest read.

## If the lever is real, candidates in order

Only pursue after Task 1. Per `round-decomp-fused.md`, and remembering the rule that
bandwidth costs translate to e2e while latency/occupancy costs hidden under concurrent
dispatch do not:

1. read-side GDN state fusion - GET_ROWS still gathers 48 x 3 MB into scratch before the
   kernel (~2 ms/round), the mirror of the writeback fusion that paid +6.2%. Pure traffic,
   so it should translate.
2. b1 GDN writeback fusion (~1.0 ms/token, floor only) - the predicate bails at
   `n_written <= 1` by design, ops.cpp:37.
3. drafter selector head at 161 GB/s + TOP_K, ~4.8 ms nominal, realistic ~2.

## Do not reopen

`slope-sweep.md` and `head-to-head-aug22.md` closed these: depth (committed/round saturates
at ~4.0, and the ne11=9 cliff is a foot-gun not a lever), drafter quality (we accept 55.7%
at depth 4 vs their 49.0%), CPU anywhere (2.7 ms/round total, measured unprofiled).

Adaptive depth is not a shortcut: flattening verify widths 2-5 is its prerequisite, not a
follow-up - see the adaptive section in `slope-sweep.md`.
