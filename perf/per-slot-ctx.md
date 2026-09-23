# Per-slot context sizes: the size-class work (2026-09-24)

**Status: IN PROGRESS on branch `exp/ctx-classes` (tree `~/play/llama.cpp-ctxclass`, off prod cdcb10e86).** Owner's use
case (memory `per-slot-context-sizes-state`): one resident 96K coordinator + short-lived executors, deterministic, no
starvation. The plan's order (2026-09-23 night): single-class byte-identical gate first, then per-stream KV length in
the decode FA kernels, then the memory layout. Baseline = `slot-mix.md` "The per-slot context baseline"
(TAG `slotmix-baseline-sep23`, prod adea1cc69 + 582cae336 = today's prod binary).

## The baseline, decomposed into rounds

Every phase of the baseline runs at **fixed depth 1 for 3+ generating slots** - the slot budget
(`LLAMA_SPEC_SLOT_BUDGET`, default 8: depth <= 8/n_gen - 1) clamps the executors-alone phase (3 slots) and the mix (4)
to depth 1; only the solo coordinator runs the controller (widths 3/7). ms per round = predicted_ms / draft_n:

| phase (split arm) | columns in the round | ms/round | tok/round |
|---|---|---|---|
| 3 executors alone | 3 x 2 = 6 | 123-130 (JSON 43, math 52) | 1.6-1.8 |
| coordinator solo, 96K | 3-7 (controller) | ~118 | ~2.5 |
| mix, 4 slots | 4 x 2 = 8 | 216-244 (unified 165-207) | 1.7-1.9 |

So the mix costs ~110 ms per round over the executors alone, and the plan's assumption was that most of it is the
three executors reading (scanning the mask of) the coordinator's 96K extent, since a ubatch's FA runs to
`n_kv = max over its streams`. The slot trace (`-lv 5`, the new `decode: ubatch` line) confirms one ubatch per round
with every slot in it (`n_tokens = 8, n_seqs = 4, n_seq_tokens = 2`): the round is one graph run, not one per slot.

## Lever 1: per-stream KV extent for flash attention - BUILT, BYTE-IDENTICAL, NO GAIN (refuted as a speed lever)

Commit e0ce639a3. An I32 `[n_stream]` input `attn_inp_kv_len` (`ggml_flash_attn_ext_set_kv_len`, src[5] of the FA op),
filled by `llama_kv_cache::set_input_kv_len` with each stream's padded `used_max_p1` (the per-stream term of
`get_n_kv()`, so `kv_len[s] <= n_kv` and every cell of stream s at or beyond it is empty = masked -inf). The Metal
tile and vec kernels stop stream `iq3`'s KV loop at `kvlen[iq3]`, and the blk map classifies chunks beyond it as
all-masked without reading them. Exact by construction: the skipped chunks are exactly the ones the blk map
(`blk == 0 -> continue`) and the vec kernel's `-INF` check already skip. The input exists only when the cache has
`n_stream > 1`, so unified and one-slot graphs are the old graphs (their route names carry no `_kvl=1`). Kill
switches: `GGML_FA_KVLEN=0` (kernels ignore it), `LLAMA_ATTN_KV_LEN=0` (the input is not created).

**Trap found by the first smoke (shas moved, acceptance 40/56/27%):** `llm_graph_input_mem_hybrid::set_input` sets
the attention inputs itself instead of delegating to `llm_graph_input_attn_kv::set_input`, so the extent was never
filled and the kernels read an unset buffer (a truncated attention). `GGML_FA_KVLEN=0` on the same binary gave the
reference shas, which split "graph change" from "kernel change" in one run. Fixed; the multislot gate
(`run-multislot-gate.sh` config) then passed (fa07afbb6c44 / b5639c4c0996 / 68e5283468ff).

**Gate at 32K** (`kvlen32k-{ref,new}`, coordinator `longprompt-32k` 25.6K tokens, 3 executors, `PICK_SPEC_EV=0`):
every executors-alone and solo sha identical, ms/round identical (mix 146-158 both). Two mix-phase shas forked
(coordinator, and slot 1's second prompt) - see "mix-phase determinism" below.

**Gate at 96K** (`kvlen96k-new` vs the Sep 23 baseline, the baseline config exactly): **all 14 shas identical**
(execs, solo 318524e3ecaa, mix incl. the coordinator 7a2e58f5669a), **mix round 224.1 vs 225.5 ms, coordinator 7.61
vs 7.57 t/s, executors 7.0-8.9 both**. The executors' extent scans cost nothing measurable: the lever is
byte-identical and inert. It stays on the branch as infrastructure (the size-class layout needs per-stream extents
and offsets in the same place), not as a pick.

## Where the +110 ms per round is, then

In the mix the coordinator verifies at width 2 (the budget), and its 96K attention takes the 8-row Turbo4 tile
**without GQA reuse** (`use_gqa_reuse` needs `ne01 >= 3`): route `qtnw_turbo4 ... nwg=20 gqah=1` in the 96K log,
one threadgroup per query head, each streaming the whole KV head = 6x the traffic of the `qtnw24 ... gqah=6` route
the solo coordinator takes at widths 3/7. That is the FA cost that appears only in the mix (the executors' own
FA is short either way).

## Lever 2: the GQA tile at width 1-2 for multi-stream FA calls - `GGML_FA_GQA_WMIN_MS=<w>` (pending)

`ggml_metal_op_flash_attn_ext`: the smallest width that takes the GQA tile when `ne03 > 1` (default 3 = the
single-stream rule, which is untouched so the one-slot lineage holds). At width 2 with `GGML_FA_Q24_ROWS=12` the
12 GQA rows take one 16-row tile per KV head per split; width 1 (6 rows) takes the 8-row tile at `gqah=6`.
A lineage move for multi-slot width-1/2 text (new kernel family at those widths) - multi-slot shas are already
per width class (`slot-mix.md`).

**Trial `gqaw2-96k` (the 96K baseline config + `GGML_FA_GQA_WMIN_MS=2`, 00:36-00:55): the width-2 rounds take the
16-row O-resident tile per KV head (`qtnw16o ... nwg=20 gqah=6`) and the mix round drops 225 -> 146 ms (-35%).**

| mix phase, 96K coordinator + 3 executors | baseline (split) | unified baseline | `GQA_WMIN_MS=2` (split) |
|---|---|---|---|
| coordinator, overlap window | 7.53 t/s | 8.68 | **11.48** (+52%) |
| executors per stream / aggregate | 7.78 / 19.55 | 9.51 / 22.47 | **12.08 / 29.74** (+55%) |
| ms per round (coordinator / executors) | 225 / 216-244 | 193 / 165-208 | **146 / 140-159** |
| executors alone (3 slots, also width 2 now on the tile) | 17.94 | 17.73 | 17.53 (-2%, ms/round +2-4%) |
| coordinator solo (single stream, untouched) | 21.12, sha 318524e3ecaa | 21.09 | 21.22, sha 318524e3ecaa |

Shas: the solo coordinator and the 96K mix coordinator (7a2e58f5669a) held; 4 of 6 executors-alone texts and 2 of
7 mix texts moved (the width-2 kernel family changed: a NUM-TG lineage move on multi-stream width-2 text, to be
priced pairwise like the other decode-route moves if adopted). The executors-alone phase (short extents) gives
2-4% per round back: at 512-cell extents the 16-row tile's per-KV-head pass is not cheaper than six 8-row
passes. A route rule by extent (the tile from ~8K cells up, like `GGML_FA_Q16_KVMIN`) would keep both.

## Mix-phase determinism

At 96K the mix phase reproduced every sha across two binaries and two days (7 requests). At 32K two of seven mix
shas forked between the two binaries while every deterministic phase matched. The mix phase's ubatch composition
depends on when each executor's next request lands relative to a round, so a fork there is a timing fork unless
the picks (kernel family) differed first (memory `owner-race-evidence-bar`); `kvlen32k-new2` = the new binary run
again at 32K to see whether it forks from itself.

## Open

- The slot budget (depth 1 at 3+ slots) is the policy that puts the coordinator on the width-2 route and caps
  tok/round at ~1.8; `LLAMA_SPEC_SLOT_BUDGET_WIDE=16` + `GGML_MM_SKINNY_N16=1` exists (parallel-streams work) and is
  the next thing to price for this use case once the FA route is fixed.
- The memory layout (size classes): today one `[embd, kv_size, n_stream]` tensor per layer with a uniform stride
  (`get_k/get_v` view a contiguous stream range, `cpy_k/cpy_v` use global indices). Per-class sizes need a per-stream
  offset table in the FA kernels (the same input as `kv_len`, `[2, n_stream]`) and offset-based indices; the value on
  this 48 GB box is memory (4 x 1.6 GB Turbo4 at 96K today vs 1.6 + 3 x 0.13), not speed.
