# The recurrent-state view slide: reuse the decode graph across rollbacks (2026-09-29)

Branch `exp/graph-reuse-cache`, worktree `/Users/troff/play/llama.cpp-graph-reuse`, off prod `c7f560114`. Owner: "see if there is
better reuse to be had" after the finding in `cpu-round-overhead.md`'s 2026-09-29 addendum: `graphs reused` = 32 of ~98 rounds at
fixed depth 3, because `llm_graph_input_rs::can_reuse_rs` compares `view_row0` / `view_row0_ss` (the recurrent state's source
rows, which follow the previous round's rollback) as graph topology, and a miss costs `build_graph` + `sched_alloc_graph` on
the CPU while the GPU has nothing queued.

## The rebuild, priced (decode-prof split, `LLAMA_DECODE_PROF=1`, q4 fixed depth 3, 8K benchprompt)

Per miss: **build 0.37 ms, alloc 2.2-2.5 ms** (the sched split + galloc re-plan of a 4245-node graph) - the allocation is the
cost, not the ggml build. 45 misses per 128 decodes; target ctx `reuse` bracket 1.8 ms/decode average.

## The mechanism

A view resolves to the cache tensor (`view_src`) with an absolute byte offset, and the Metal backend takes every buffer offset
from `tensor->data` at encode time - nothing about the offset is baked into the encoded work across rounds. So a cached graph
whose source rows moved can be kept: `can_reuse_rs` accepts a row0 change when both the built and the requested forms are views
(`>= 0`) and the graph holds them; `set_input` then slides, by the row delta, the `build_rs` output view and every view DERIVED
from it (the tensor a view aliases: `src[1]` for CPY, `src[0]` for every other view-producing op), patching `view_offs` and
`data`. `LLAMA_RS_SLIDE=0` restores the strict check. `n_keep` / `xk_gather` / a gather form on either side stay topology.

**The first form was WRONG and is recorded because it is the obvious form:** "every view of the same root inside the seed's byte
range". The write-back views (`conv_state_update` per snapshot group, the delta-net snapshot `dst`) are direct one-row views of
the cache at `(group*size + head)*row_size`; at rollback 0 the group-0 write view has the read view's bytes, so the range rule
slid it too and the new state went to the rollback group. A/B `rsslide-0929-q4`: sha `69eee5ef0973` (canonical `86213d038a29`),
acceptance 66.8 -> 34.5%, 32.7 -> 22.7 t/s, reused 146 - the mechanism worked, the arithmetic did not. The derivation-chain
form keeps the write views where they are; the reverse case (built after a rollback, reused at rollback 0) puts the read view on
the write row = the identity case every non-rollback round already runs.

## A/B, fixed depth 3, Turbo4 pick arm, 8K benchprompt, 300 tokens, ABBA fresh servers (`perf/run-rs-slide-ab.sh`)

| q4, TAG `rsslide2-0929` | sha | t/s | graphs reused | dec_sub_tg | target reuse bracket | misses (build / alloc each) |
|---|---|--:|--:|--:|--:|---|
| slide off (A1) | `86213d038a29` | 32.72 | 34 | 2.96 | 1.85 | 45: 0.37 / 2.25 |
| slide on (B1) | `86213d038a29` | 33.34 | 98 | 1.28 | 0.18 | 4: 0.41 / 2.52 |
| slide on (B2) | `86213d038a29` | 33.43 | 98 | 1.29 | 0.19 | 4 |
| slide off (A2) | `86213d038a29` | 32.75 | 34 | 2.92 | 1.82 | 45 |

Byte-identical (the canonical fixed-depth sha, acceptance 199/298 in every arm), **+2.0% e2e** (32.74 -> 33.39), the target's
submit -1.65 ms/round; the 4 remaining misses are the prefill-to-decode transitions. The slide itself costs ~0.04 ms
(`set_inputs` 0.012 -> 0.05).

| ud, TAG `rsslide2-0929` | sha | t/s | graphs reused | dec_sub_tg | target reuse bracket | misses |
|---|---|--:|--:|--:|--:|---|
| slide off (A1) | `ce826d8a3cbd` | 30.33 | 36 | 2.99 | 1.71 | 42: 0.37 / 2.23 |
| slide on (B1) | `ce826d8a3cbd` | 30.74 | 91 | 1.35 | 0.19 | 4 |
| slide on (B2) | `ce826d8a3cbd` | 30.84 | 91 | 1.39 | 0.19 | 4 |
| slide off (A2) | `ce826d8a3cbd` | 30.25 | 36 | 2.96 | 1.70 | 42 |

ud: byte-identical, **+1.7% e2e** (30.29 -> 30.79), submit -1.6 ms/round.

## The pick's own gates on the branch binary (slide on by default, TAGs `rsslidegate-0929-{q4,ud}`)

- **Controller arm `turbo4-n3-300`:** q4 34.05 t/s, `graphs reused` 77 (22 on prod this morning; the remaining misses are the
  width-4/8 changes = real topology), sha `f07b0f8c58e6` = the sha the PROD binary produced this morning under the profiler
  (`secondop-0929`), i.e. a known controller fork of the statistical arm (`mint-controller-arm-is-statistical`); the replay
  gate (below) is the byte proof under the controller. ud 30.95 t/s, reused 61, sha `ce826d8a3cbd` canonical.
- **`batch1-300`:** q4 `d2953fccfb41`, ud `9c53aaade052` - canonical (no rollback ever happens at batch 1: the slide is inert).
- **Multi-slot gate short + long: PASS both lines** (3 executors + coordinator; non-uniform rollbacks across seqs give a gather
  form (`view_row0 = -1`) and stay strict).
- **Vision arm (one slot + multi-slot): PASS both lines.**
- **Replay gate PASS** (`run-specev-replay-gate.sh` TAG `rsslide-replay-0929`, q4, record with `LLAMA_RS_SLIDE=0`, 2 replays with it
  on, 945 tokens / 317 picks): record `35abc5a8312e` at 31.47 t/s, replays `35abc5a8312e` x2 at 31.93 / 31.96 (0 desync, 0 past
  trace) - byte-identical under the controller on the same picks, +1.5% on the day.

Adoption = owner. Proposed form: code default (the slide on, `LLAMA_RS_SLIDE=0` the off-switch), no manifest line - a CPU-only
change with no numerics class; worth +2.0% q4 / +1.7% ud at fixed depth, and more rounds reused under the controller.
