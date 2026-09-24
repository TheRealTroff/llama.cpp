# Per-slot context sizes: the size-class work (2026-09-24)

**Status 2026-09-24 morning: ADOPTED ON THE q4 LINE (owner: "Cache is fine but at the end of the day the quality of the output
is what matters ... we've been using the reference KLD as the proxy. But for now, I'll go with your recommendations" = adopt on
q4 as NUM-TG, price the ud line before pinning its reference).** `GGML_FA_GQA_WMIN_MS=1 GGML_FA_GQA_WMIN_KVMIN=8192` are in
`perf/pick.sh` for q4 (class NUM-TG) and proposed for ud; `run-multislot-gate.sh` carries `REF_LONG_Q4` = the flags row and
`REF_LONG_UD` = the old route's row (below, "Gate arm references"); the mint's multi-slot call passes `LONG=1`. Merged to prod
this session; the post-merge gate on the prod binary and the ud pricing pair (pairwise + f16-cache, the `kld-w2-1s` /
`kld-w2-f16ref` recipe on the UD model) are recorded at the end ("Post-merge"). Owner's framing for the record: output quality
is the real target, ill-defined; the reference KLD is the proxy this project uses for it, and the f16-cache pair is the cache
form of that proxy.

~~**Status 2026-09-24 06:45: PROPOSED (NUM-TG class), owner decides.**~~ The lever is `GGML_FA_GQA_WMIN_MS=1 GGML_FA_GQA_WMIN_KVMIN=8192`
(the GQA-reuse FA tile at widths 1-2 on multi-stream calls over 8K cells): at the 96K baseline config the mix round
225 -> 146 ms, coordinator 7.57 -> 11.56 t/s, executors 7.78 -> 12.14 per stream; at 5 slots (width 1) 6.13 -> 8.25 /
6.50 -> 8.94; wins at every extent from 7.5K up. One-slot and short-extent shas are untouched by construction; multi-slot
width-1/2 text moves. NOT byte-identical and NOT a mere lineage: the pairwise decode KLD vs the old width-2 route is
0.0022 / 98.9% same-top, but the float64 reference of the dumped node and the exact-f16-cache KLD both put the new
route CLOSER to the truth than the old one (below). Lever 1 (per-stream KV extent) is byte-identical and inert; the slot
budget at 16 is a wash; the size-class layout is a memory lever only. Earlier status lines kept below for the record.
PROPOSED, owner decides - branch `exp/ctx-classes` (tree `~/play/llama.cpp-ctxclass`, off prod
cdcb10e86). The lever is `GGML_FA_GQA_WMIN_MS=2 GGML_FA_GQA_WMIN_KVMIN=8192` (the GQA-reuse FA tile at width 2 on
multi-stream calls whose extent exceeds 8K): at the 96K baseline config the mix round 225 -> 146 ms, coordinator 7.57 ->
11.56 t/s, executors 7.78 -> 12.14 per stream (29.8 aggregate); executors alone and the solo coordinator byte-identical
to the baseline (all 8 shas), the mix 5/7 shas held and 2 moved on the width-2 route - a multi-slot width-2 lineage
move, recurring run to run. Lever 1 (per-stream KV extent) is byte-identical and inert; the slot budget at 16 is a
wash. See "The proposed configuration" at the end.** Owner's use
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

## The extent crossover of the width-2 tile (`xover-{8k,16k,24k}-{old,tile}`, 02:10-02:45)

Coordinators of 7.5K / 16.4K / 24.3K tokens (`longprompt-{8,16,24}k.txt`, cuts of the 32K prompt), 3 executors, the
tile at EVERY extent (`GGML_FA_GQA_WMIN_MS=2 GGML_FA_GQA_WMIN_KVMIN=0`) vs the old route, q4 Turbo4, fixed depth:

| coordinator | executors alone ms/round old / tile | mix coordinator t/s old / tile | mix round ms old / tile | mix executors t/s old / tile |
|---|---|---|---|---|
| 7.5K | 99.1 / 98.4 | 12.79 / 13.59 (+6%) | 139 / 130 | 13.25 / 14.19 |
| 16.4K | 99.1 / 98.4 | 11.88 / 13.21 (+11%) | 148 / 133 | 12.46 / 13.94 |
| 24.3K | 99.2 / 98.3 | 11.25 / 12.96 (+15%) | 156 / 134 | 11.80 / 13.76 |
| 96K (above) | 128 / 126 | 7.53 / 11.52 (+53%) | 225 / 146 | 7.78 / 12.14 |

The tile wins at every extent, including the 512-cell executors alone (three pairs, +0.7% each: the "-2%" of the
first 96K trial was run-to-run noise). With the tile the mix round is nearly flat in the coordinator's length
(130 -> 146 ms from 7.5K to 96K) where the old route grew 139 -> 225. `GGML_FA_GQA_WMIN_KVMIN` is therefore a
choice, not a need: 8192 keeps the mint's short-extent multi-slot arm (f16, 512 cells) on its recorded references,
0 takes the +0.7% at short extents and moves those references. The proposal keeps 8192.

## Width 1: five slots, speculation off by the budget (`w1-96k-{old,w1}`, 02:47-03:27)

The 96K coordinator + 4 executors (5 generating slots: the budget turns speculation off, every round is width 1),
`GGML_FA_GQA_WMIN_MS=1 GGML_FA_GQA_WMIN_KVMIN=8192` vs the old route (the 8-row tile at `gqah=1`; the new route is the
same 8-row tile at `gqah=6`, six GQA rows per KV head):

| | old | width-1 tile |
|---|---|---|
| mix coordinator | 6.13 t/s | **8.25** (+35%) |
| mix executors per stream / aggregate | 6.50 / 21.7 | **8.94 / 29.3** (+38% / +35%) |
| executors alone (4 slots, 512-cell extents, old route under KVMIN) | 11.88, 8/8 shas identical | 11.88 |
| solo coordinator | 21.30, 318524e3ecaa | 21.19, 318524e3ecaa |

Every mix sha moved (the width-1 multi-stream kernel family), the coordinator's to 7a2e58f5669a - the same text the
width-2 tile produces in the 4-slot mix. So `GGML_FA_GQA_WMIN_MS=1` is the general setting: widths 1 and 2 on
multi-stream calls over the threshold take the GQA tile, widths 3-6 already did.

## The slot budget at 16 on top of lever 2 - a wash in the mix (refuted for this workload)

`budget16-96k` = lever 2 + `LLAMA_SPEC_SLOT_BUDGET=16 LLAMA_SPEC_SLOT_BUDGET_WIDE=16 GGML_MM_SKINNY_N16=1` (3 slots
-> depth 4, 4 slots -> depth 3, the fused 16-column projection tile; `parallel-streams.md`):

| | lever 2 (budget 8) | + budget 16 |
|---|---|---|
| mix coordinator / acceptance | 11.53 t/s / 76% | 10.89 / 48% |
| mix executors per stream / aggregate | 12.08 / 29.7 | 12.03 / 28.1 |
| executors alone per stream | 17.53 (JSON 29, prose 12) | 20.23 (JSON 62, prose 7.6, math 18.6) |
| solo coordinator | 21.22, 318524e3ecaa | 21.11, 318524e3ecaa |

At 4 slots the deeper drafts halve the coordinator's acceptance and the 16-column round is ~2x the 8-column one:
the extra tokens per round pay for the round, no more. Executors alone gain 15% on average with a per-prompt spread
of 8x (JSON near 100% acceptance flies, prose at depth 4 crawls). Not a lever for the coordinator + executors
workload at 96K; the parallel-streams numbers (8K contexts, 4-8 equal slots) stand as their own case.

## Mix-phase determinism

At 96K the mix phase reproduced every sha across two binaries and two days (7 requests). At 32K two of seven mix
shas forked between the two binaries while every deterministic phase matched. The mix phase's ubatch composition
depends on when each executor's next request lands relative to a round, so a fork there is a timing fork unless
the picks (kernel family) differed first (memory `owner-race-evidence-bar`). **Resolved 01:07:** the new binary
reproduced itself at 32K three times (`kvlen32k-new`, `-new2`, and `-new-k0` = kernels ignoring the extent via
`GGML_FA_KVLEN=0`: all 14 shas identical, so the extent path is inert on the same composition), and the prod
binary's rerun (`kvlen32k-ref2`) produced the NEW binary's mix shas - the first reference run was the timing fork
(its coordinator 9fe3be25e872 and 04-math 0749107642e8 never recurred). Lever 1 is byte-identical on every
comparison that shares a composition; a 32K mix fork of this harness is a race between the executors' second
requests and a round boundary, and a mix-phase sha is only evidence when it recurs.

## Open

- The slot budget (depth 1 at 3+ slots) is the policy that puts the coordinator on the width-2 route and caps
  tok/round at ~1.8; `LLAMA_SPEC_SLOT_BUDGET_WIDE=16` + `GGML_MM_SKINNY_N16=1` exists (parallel-streams work) and is
  the next thing to price for this use case once the FA route is fixed.
- The memory layout (size classes): today one `[embd, kv_size, n_stream]` tensor per layer with a uniform stride
  (`get_k/get_v` view a contiguous stream range, `cpy_k/cpy_v` use global indices). Per-class sizes need a per-stream
  offset table in the FA kernels (the same input as `kv_len`, `[2, n_stream]`) and offset-based indices; the value on
  this 48 GB box is memory (4 x 1.6 GB Turbo4 at 96K today vs 1.6 + 3 x 0.13), not speed.

## The proposed configuration, gated (`gqaw2k8-96k`, 01:32-01:52)

`GGML_FA_GQA_WMIN_MS=2 GGML_FA_GQA_WMIN_KVMIN=8192` on the ctxclass binary (f68eeb468), the Sep 23 baseline config:

| phase | baseline (split, Sep 23) | proposed | shas |
|---|---|---|---|
| executors alone, per stream / aggregate | 17.94 / 39.98 | 18.18 / 40.53 | 6/6 identical (extent < 8K = the old route) |
| coordinator solo (96K prefill 1045 s) | 21.12 | 21.23 | 318524e3ecaa both |
| mix coordinator (overlap window) | 7.53 | **11.52** (+53%) | 7a2e58f5669a both |
| mix executors per stream / aggregate | 7.78 / 19.55 | **12.14 / 29.83** (+56% / +53%) | 5/7 held; 04-math 0749107642e8 -> 92977c8bec89 (its execs-phase text), 02-prose 92fd53479c9d -> a4f1c47dfd61 |
| mix round, coordinator / executors | 225 / 216-244 ms | 146 / 140-157 | |

The mix shas are identical between the two lever-2 runs (`gqaw2-96k`, `gqaw2k8-96k`: 7/7), so the two moves are the
kernel family, not timing. Against the UNIFIED baseline (8.68 / 9.51) the proposed split arm is +33% / +28%: the
size-class question "can split mode beat unified" is answered before any layout work.

**Adoption.** Multi-stream only, extent > 8K only: every one-slot arm of the mint and the multislot gate arm (f16,
512-cell extents) run the old routes - their shas cannot move. What moves is multi-slot text whose long stream
verifies at width 2: the same `qtnw16o gqah=6` tile family the pick already runs at widths 5-6, at a new width.
Owner's call on the lineage; a pairwise decode-path KLD on multi-slot width-2 text (the per-width recipe of
`ud-width5-decode-kld`) is the formal gate if wanted. If adopted: both flags into `perf/pick.sh` PICK_ENV (BI class
with a multi-stream lineage note), and `run-multislot-gate.sh` gains a long-extent arm (a 32K coordinator + 3
executors, `PHASES=execs,solo,mix`, ~6 min) so this route is gated from now on.

**Open after this:**
- width 1 (5+ slots, the budget turns speculation off): `GGML_FA_GQA_WMIN_MS=1` puts the 6 GQA rows in one 8-row
  tile per KV head; untested (the owner's use case is 1 + 3).
- the size-class memory layout: memory only (48 GB box: 4 x 1.6 GB Turbo4 at 96K + the drafter's f16 per slot);
  design = a per-stream offset table beside `kv_len` in the FA op, offset-based `k_idxs`/`v_idxs`, per-stream cell
  counts, a per-slot cap list in the server. Start it when the owner wants more executor slots than the box holds.
- the slot budget policy for 2-3 generating slots (depth 1 today, 8/n - 1): the width-3 GQA route at 2 slots
  (budget 8 -> depth 3, width 4) is already the single-stream route; nothing to do there.

## Pricing the route: NOT a summation-order move (`kld-w2ms-route`, 03:27-04:40)

Pairwise decode-path KLD, two streams of 16K cells (`llama-perplexity -c 16384 -b 32768 -ub 4` = 2 tokens per stream per
ubatch, the positions scored are cells 8192-16384, all under the tile with `KVMIN=8192`), Turbo4 KV both sides, the q4
pick env, base = the old route, 4 chunks (32K scored positions):

| row | mean KLD | median | 99.9% | max | same top | overlap (1-TV) |
|---|---|---|---|---|---|---|
| control (same config as the base) | 0.000000 | 0 | 0.00005 | 0.00006 | 100.000% | 99.945% |
| `GGML_FA_GQA_WMIN_MS=2` (the 16-row GQA tile, 12 rows) | **0.002156** | 0.00018 | 0.21 | 8.06 | **98.944%** | 99.00% |

Per chunk 0.0005 / 0.0036 / 0.0026 / 0.0022 (chunk 1 = stream 0 of the first pair is the low one; no clean per-stream
pattern). For scale: the priced decode-route moves are 5e-6 to 2.5e-5 mean / 99.9x% same-top, the whole Turbo4 cache
costs 0.006-0.008 vs f16, and the q4 line's folded-norm FA form (TR=7) was a 0.0019 perturbation. So this is either a
defect in the tile when it runs with `ne03 > 1` (it never had before this session - every prior GQA-tile call was one
stream) or a numerics class of the 12-row `qtnw16o` tile that the width-6 single-stream pricing did not see.
`test-backend-ops -o FLASH_ATTN_EXT` passes with the route on (0 fails incl. the multi-stream `nr23=[x,2]` cases), which
bounds it: not a gross indexing error at test shapes (short KV, no split-K at 20 workgroups).

Discriminator queued (`kld-w2-1s`, `kld-w6-1s`): the tile at width 2 on ONE stream (`GGML_FA_GQA_WMIN_ALL=1`) vs the 8-row
route, and the pick's width-6 24+16 plan vs 8-row GQA tiles (`GGML_FA_Q24_ROWS=0`). Tile-at-12-rows ~0.002 on one stream =
the tile's own class (then the pick's width 6 carries it too, to be re-priced); ~5e-6 on one stream = a multi-stream
defect in the tile (then hunt with `LLAMA_FA_DUMP` on a two-stream ubatch).

**Op level (05:20, `LLAMA_FA_DUMP` with the new `LLAMA_FA_DUMP_NS=2 LLAMA_FA_DUMP_KVMIN=1536` selectors, the KL-divergence
configuration at 2K):** the first attention layer's FA node on a two-stream, 2-token, 1536-cell ubatch has byte-identical
q/k/v/mask under both routes and outputs that differ by **relative RMS 2.7e-4, max 9.4e-4, on every head (5e-5..7e-4),
both tokens and both streams** (stream 0 3.4e-4, stream 1 1.7e-4); the later layers drift to 1-4e-2 through the
residual. Not a row or stream mapping error (that would be O(1) on some rows) - a precision-class difference between
the 8-row `qtnw` tile at `gqah=1` and the 16-row O-resident `qtnw16o` tile at `gqah=6`, at 12 valid rows. The three
Turbo4 tiles share `FA_TYPES` and the template flags (QT, TRM 3, VU 4, LD 1), so the difference is in the tile's
data path, not its declared types. 2.7e-4 relative per layer at 1.5K cells, growing with the extent, matches the KLD
0.0003 (2K) -> 0.002 (16K).

**Localized (05:30-06:00):** the deviation is the GQA flattening at width 2 itself, not the streams, the extent input,
the split-K partials or the O-resident tile: the same two-stream node gives byte-identical output under every GQA
variant (`qtnw16o` nsg 8, the plain 8-row `qtnw gqah=6` via `GGML_FA_Q24_REM=8`, `nwg=1` vs 20, `GGML_FA_KVLEN=0`), all
2.68e-4 from the `gqah=1` route, and a ONE-stream width-2 node (`-b 2 -ub 2`, `GGML_FA_GQA_WMIN_ALL=1`) shows the same
3.0e-4. The single-stream KLD pair (`kld-w2-1s`, 12 chunks at 2K) = **0.00223 mean / 98.83% same-top / max 4.2**, the
two-stream number again.

**Which route is right - a float64 reference of the dumped node (`perf/fa-dump-ref.py`: Turbo4 dequant from the block
layout and the 16 centroids, f64 softmax, scale 1/16):**

| route (same inputs, layer 3, 2 streams x 2 tokens, 1536 cells) | relRMS vs exact | max abs | per stream |
|---|---|---|---|
| old: 8-row `qtnw` at `gqah=1` (2 valid rows of 8) | 3.42e-4 | 1.44e-3 | 4.2e-4 / 2.5e-4 |
| new: any GQA tile at width 2 (12 rows) | **1.85e-4** | **5.0e-4** | 2.1e-4 / 1.6e-4 |

Both are the q4 line's folded-norm (TR 7) class - a few 1e-4 per layer - and the GQA route is the MORE accurate of the
two (half the error, a third of the max). So the 0.002 pairwise KLD is the distance between two TR-7-class kernels, the
same size as the TR 7 perturbation itself (0.0019 when it was introduced), in the favourable direction. It is a NUM-TG
class move on multi-slot width-1/2 text, not a bug and not a free lineage. Queued: each route against the exact f16-cache
logits at one stream (`kld-w2-f16ref`) to state the direction at the logit level; `kld-w6-1s` prices the pick's own
width-6 tile plan against 8-row GQA tiles (expected ~1e-5: both are GQA paths). Why the 8-row `gqah=1` tile is worse at
width 2 than the GQA tiles is open (its arithmetic should be per row; at width 4 the two agree byte for byte).

**`kld-w6-1s` (06:00):** the pick's width-6 plan (`GGML_FA_Q24_ROWS=12`: a 24-row + a 16-row tile) vs 8-row GQA tiles
(`GGML_FA_Q24_ROWS=0`), one stream, 12 chunks: **mean KLD 0.000000, median 0, max 6.8e-5 (the uint16 base floor),
same-top 100.000%** - the GQA tiles are one arithmetic at every row count, exactly as the width-2 probes showed. The
pick's width-6 route is unaffected by anything here.

## The direction, at the logits (`kld-w2-f16ref`, 05:58-06:40)

Each width-2 route (one stream, `-b 2 -ub 2`, 12 chunks at 2K) against the EXACT cache: the same model with f16 K/V:

| route | mean KLD vs f16 cache | median | 99.9% | max | same top |
|---|---|---|---|---|---|
| old: 8-row `qtnw` at `gqah=1` | 0.01071 ± 0.0020 | 0.00127 | 1.25 | 15.2 | 97.46 ± 0.14% |
| new: the GQA tile at width 2 | **0.00895 ± 0.0018** | 0.00129 | **0.95** | 15.8 | 97.45 ± 0.14% |

The new route is 16% closer to the exact cache in mean KLD and 24% in the 99.9% tail, same-top equal within error: the
op-level verdict (1.9e-4 vs 3.4e-4 from exact per layer) holds at the logits. (Both numbers are the Turbo4 cache's own
price on the width-2 decode path, larger than the 0.006-0.008 prefill-path figure.) So the move is a NUM-TG class change
in the favourable direction on multi-slot width-1/2 text. Why the 8-row `gqah=1` tile is the less accurate one at width
2 (at width 4 the two agree byte for byte) stays open - a kernel question, not a blocker.

## Gate arm references (`run-multislot-gate.sh LONG=1`, 04:41-04:58, prod route vs the flags)

| line | route | execs 1/2/3 | solo | mix coord / 1 / 2 / 3 |
|---|---|---|---|---|
| q4 | prod (old) | c8522a40c1e8 / 28ff51768e4d / 914119d97178 | d0d8cd0eb2d8 | eeffe5ac0857 / c8522a40c1e8 / 36715962b9e7 / db5040e9fdac |
| q4 | flags | same | same | 64a49312d01f / 67b0b590dd7b / d5fa80109900 / eae23bbebec0 |
| ud | prod (old) | 36529d9fb3fe / 039bf7ad9b41 / 9c7f73d13fb8 | d6c3f3372554 | cf057877480d / 36529d9fb3fe / 039bf7ad9b41 / 9c7f73d13fb8 |
| ud | flags | same | same | ef89fa0c0a9c / 36529d9fb3fe / 039bf7ad9b41 / 9c7f73d13fb8 |

Executors alone and solo identical on both lines (short extents / one stream); the mix moves on the width-2 texts
(the UD line's executors' mix texts held, its coordinator moved). Whichever route the owner picks, its row becomes
`REF_LONG_Q4` / `REF_LONG_UD` in `run-multislot-gate.sh` and `LONG=1` joins the mint.

## If adopted

- `GGML_FA_GQA_WMIN_MS=1` and `GGML_FA_GQA_WMIN_KVMIN=8192` into `perf/pick.sh` PICK_ENV, class NUM-TG, both lines
  (the UD line's Turbo4 tile is its own instantiation, `qtl4w16o`; its long arm above ran it; its pairwise KLD is not
  measured - the same two runs as `kld-w2-1s` on the UD model if wanted).
- `REF_LONG_{Q4,UD}` = the flags rows above; `LONG=1` in `run-prod-pick.sh`'s multi-slot call.
- Lever 1 stays as merged infrastructure (inert, byte-identical), the kill switches documented.

## Post-merge (prod e12330da1, binary 11:45, 2026-09-24)

`run-multislot-gate.sh LONG=1` on the merged prod binary, the flags supplied by `pick.sh` (no override), both lines
(TAGs `postmerge-0924-{q4,ud}`):

| line | short arm (f16, 3 executors) | long arm (32K Turbo4 coordinator + 3 executors) |
|---|---|---|
| q4 | PASS fa07afbb6c44 / b5639c4c0996 / 68e5283468ff | **PASS, all 8 = the flags row** (mix coordinator 64a49312d01f at 17.60 t/s, executors 12.8-13.1; the route log names `qtnw16o ... gqah=6` at width 2) |
| ud | PASS fa07afbb6c44 / a3c90139bbfd / 68e5283468ff | PASS, all 8 = the old route's row (mix coordinator cf057877480d; the flags are proposed on ud, so `pick_env ud` does not carry them) |

The ud pricing pair (`kld-w2-ud-1s`: control + tile vs the old-route Turbo4 base at `-b 2 -ub 2`; `kld-w2-ud-f16ref`: each
route vs the f16 cache; 12 chunks at 2K, the q4 recipe on the UD model, `pick_env ud` exported) started 11:55 on the prod
binary - results below when they land.
