# Per-slot context sizes: the size-class work (2026-09-24)

**Status 2026-09-24 evening: ADOPTED ON BOTH LINES, MERGED TO PROD (e12330da1), MINTED (`prodpick-sep24-gqawmin-{q4,ud}`, a hot-ambient
SHA mint: every one-slot sha = Sep 23's on both lines, both multi-slot arms PASS on both lines, the long arms on the flags rows).** The ud
line was priced the same afternoon ("The ud line priced" + "Paired verdict" at the end) and the owner took the flags on ud too
("Take them on ud"): both entries are `pick` on both lines, `REF_LONG_UD` = the flags row. The q4 Turbo4 600 controller arm forked on
its first run (`ae44d18ca4a9`, block histogram [3:106 5:1 7:81]) and gave the replay-gated text on its second (`9e49b3d13b31`,
[2:1 3:90 7:89]): a pick diff before any text diverged = the controller, gated by `run-specev-replay-gate.sh` TAG
`replaygate-0924-gqawmin` (result at the very end). Open after this: the size-class memory layout (memory only), the 8-row
`gqah=1` tile's width-2 accuracy question ~~(a kernel question)~~ [CLOSED in the evening, see "Resolution" at the end: the old route was the VEC kernel, the gqah=1 tile is byte-identical to the GQA tile], and - if the owner wants the route's behaviour outside the 2K
wikitext regime - a paired pair on the model's own chat-templated text at 16-32K and on a code prompt.

~~**Status 2026-09-24 morning: ADOPTED ON THE q4 LINE (owner: "Cache is fine but at the end of the day the quality of the output
is what matters ... we've been using the reference KLD as the proxy. But for now, I'll go with your recommendations" = adopt on
q4 as NUM-TG, price the ud line before pinning its reference).** `GGML_FA_GQA_WMIN_MS=1 GGML_FA_GQA_WMIN_KVMIN=8192` are in
`perf/pick.sh` for q4 (class NUM-TG) and proposed for ud; `run-multislot-gate.sh` carries `REF_LONG_Q4` = the flags row and
`REF_LONG_UD` = the old route's row (below, "Gate arm references"); the mint's multi-slot call passes `LONG=1`. Merged to prod
this session; the post-merge gate on the prod binary and the ud pricing pair (pairwise + f16-cache, the `kld-w2-1s` /
`kld-w2-f16ref` recipe on the UD model) are recorded at the end ("Post-merge"). Owner's framing for the record: output quality
is the real target, ill-defined; the reference KLD is the proxy this project uses for it, and the f16-cache pair is the cache
form of that proxy.~~

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
**without GQA reuse** (`use_gqa_reuse` needs `ne01 >= 3`): ~~route `qtnw_turbo4 ... nwg=20 gqah=1` in the 96K log~~ [evening correction: at width 2 the call goes to the VEC kernel `kernel_flash_attn_ext_vec_turbo4` (`GGML_FA_VEC_MAX=3`, no `fa-route:` line); the `gqah=1` line in that log belongs to a width >= 3 call without reuse - see "Resolution"],
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
`GGML_FA_GQA_WMIN_MS=1 GGML_FA_GQA_WMIN_KVMIN=8192` vs the old route (~~the 8-row tile at `gqah=1`~~ [the VEC kernel, see "Resolution"]; the new route is the
8-row tile at `gqah=6`, six GQA rows per KV head):

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
- ~~The memory layout (size classes): today one `[embd, kv_size, n_stream]` tensor per layer with a uniform stride
  (`get_k/get_v` view a contiguous stream range, `cpy_k/cpy_v` use global indices). Per-class sizes need a per-stream
  offset table in the FA kernels (the same input as `kv_len`, `[2, n_stream]`) and offset-based indices; the value on
  this 48 GB box is memory (4 x 1.6 GB Turbo4 at 96K today vs 1.6 + 3 x 0.13), not speed.~~ BUILT 2026-09-25, see
  "Per-slot context sizes: the packed layout" at the end of this note (branch `exp/kv-size-classes`).

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
- ~~the size-class memory layout: memory only (48 GB box: 4 x 1.6 GB Turbo4 at 96K + the drafter's f16 per slot);
  design = a per-stream offset table beside `kv_len` in the FA op, offset-based `k_idxs`/`v_idxs`, per-stream cell
  counts, a per-slot cap list in the server. Start it when the owner wants more executor slots than the box holds.~~
  BUILT 2026-09-25 (owner: "what's the point of a smaller max context size if I allocate the same amount anyway?"),
  the section at the end of this note.
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
~~the 8-row `qtnw` tile at `gqah=1`~~ [the VEC kernel `kernel_flash_attn_ext_vec_turbo4`, see "Resolution"] and the 16-row O-resident `qtnw16o` tile at `gqah=6`, at 12 valid rows. The three
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
| old: ~~8-row `qtnw` at `gqah=1` (2 valid rows of 8)~~ [the VEC kernel, reproduced 3.42e-4 / 1.44e-3 in the evening with no `fa-route:` line] | 3.42e-4 | 1.44e-3 | 4.2e-4 / 2.5e-4 |
| new: any GQA tile at width 2 (12 rows) | **1.85e-4** | **5.0e-4** | 2.1e-4 / 1.6e-4 |

Both are the q4 line's folded-norm (TR 7) class - a few 1e-4 per layer - and the GQA route is the MORE accurate of the
two (half the error, a third of the max). So the 0.002 pairwise KLD is the distance between two TR-7-class kernels, the
same size as the TR 7 perturbation itself (0.0019 when it was introduced), in the favourable direction. It is a NUM-TG
class move on multi-slot width-1/2 text, not a bug and not a free lineage. Queued: each route against the exact f16-cache
logits at one stream (`kld-w2-f16ref`) to state the direction at the logit level; `kld-w6-1s` prices the pick's own
width-6 tile plan against 8-row GQA tiles (expected ~1e-5: both are GQA paths). Why the 8-row `gqah=1` tile is worse at
width 2 than the GQA tiles ~~is open (its arithmetic should be per row; at width 4 the two agree byte for byte)~~ [CLOSED: it is not - the 8-row tile at gqah=1 forced at width 2 is byte-identical to the GQA tile; the less accurate route was the VEC kernel. "Resolution" below].

**`kld-w6-1s` (06:00):** the pick's width-6 plan (`GGML_FA_Q24_ROWS=12`: a 24-row + a 16-row tile) vs 8-row GQA tiles
(`GGML_FA_Q24_ROWS=0`), one stream, 12 chunks: **mean KLD 0.000000, median 0, max 6.8e-5 (the uint16 base floor),
same-top 100.000%** - the GQA tiles are one arithmetic at every row count, exactly as the width-2 probes showed. The
pick's width-6 route is unaffected by anything here.

## The direction, at the logits (`kld-w2-f16ref`, 05:58-06:40)

Each width-2 route (one stream, `-b 2 -ub 2`, 12 chunks at 2K) against the EXACT cache: the same model with f16 K/V:

| route | mean KLD vs f16 cache | median | 99.9% | max | same top |
|---|---|---|---|---|---|
| old: ~~8-row `qtnw` at `gqah=1`~~ [the VEC kernel] | 0.01071 ± 0.0020 | 0.00127 | 1.25 | 15.2 | 97.46 ± 0.14% |
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

## The ud line priced (`kld-w2-ud-1s`, `kld-w2-ud-f16ref`, 11:55-13:39, prod e12330da1, hot ambient)

The q4 recipe on the UD model (`pick_env ud` exported: its Turbo4 tile is the TR=9 instantiation), one stream, `-b 2 -ub 2`,
12 chunks at 2K. Pairwise vs the OLD route (Turbo4 base at width 2):

| row | mean KLD | median | 99.0% | 99.9% | max | same top |
|---|---|---|---|---|---|---|
| control (the base rerun) | 0.000000 | 0 | - | 0.00005 | 0.00006 | 99.984 ± 0.012% |
| the GQA tile at width 2 (`GGML_FA_GQA_WMIN_ALL=1 MS=2`) | **0.002384 ± 0.00062** | 0.00018 | 0.0088 | 0.36 | 6.4 | **98.770 ± 0.10%** |

= the q4 pair (0.00223 / 0.00018 / 98.83%) on every statistic: the TR=9 instantiation moves by the same amount as TR=7 when
the width-2 flattening changes. Each route vs the EXACT f16 cache (same weights, f16 K/V base at width 2):

| route | mean KLD vs f16 cache | median | 99.0% | 99.9% | max | same top |
|---|---|---|---|---|---|---|
| old: 8-row tile at `gqah=1` | 0.010484 ± 0.0022 | 0.00116 | 0.0426 | 1.43 | 17.8 | 97.638 ± 0.14% |
| new: the GQA tile at width 2 | **0.009117 ± 0.0016** | 0.00116 | 0.0424 | 1.56 | 11.1 | 97.442 ± 0.14% |

The mean (the project's proxy) moves the q4 way: -13% (q4 -16%), median and 99.0% equal, max better. Two secondary
statistics lean the other way, both inside their error: same-top -0.20 pt (two ± 0.14 bars) and the 99.9% quantile +9%
(a single quantile over ~24 positions; on q4 it improved -24%). Whether the same-top slip is real is a PAIRED question
(both arms score the same positions): three base files (D = f16 cache, X = old, Y = GQA) and `perf/kld-fisher.py`
(per-position KL, top flips per margin bin, the Fisher correlation of the two routes' deviations) are running
(`kld-w2-ud-fisher-{D,X,Y}`, 13:45); verdict below. Speed on ud at 32K (the long arm, old vs flags rows above):
mix coordinator 12.30 -> 12.95 t/s (+5%), executors 9.31 -> 10.30 per stream (+11%); the 96K figure is unmeasured on ud
(q4's +6% at 7.5K grew to +53% at 96K).

**Paired verdict (`kld-w2-ud-fisher-{D,X,Y}` + `perf/kld-fisher.py`, rows in `logits/kld-w2-ud-fisher-rows.npz`, 14:37):** the
two routes are statistically indistinguishable at the logits on this corpus, and neither is worse.

| paired over the same 12,276 positions | old | GQA tile | paired difference |
|---|---|---|---|
| KL vs the f16 cache (common-window form) | 0.008780 | 0.008665 | -0.000114 ± 0.000567 (t = -0.2) |
| ... without the 0.1% largest positions (12) | 0.004817 | 0.004235 | the body favours the tile by 12% |
| ... the 12 largest positions, summed KL | 48.7 | 54.4 | chaotic positions decide the sign of any mean |
| top flips vs the f16 cache | 292 | 314 | discordant: old-only 59, tile-only 81 (McNemar chi2 3.5, p ~ 0.06) |
| discordant flips' base top-2 margin | | | median 0.11 nat, 90% < 0.37 nat = ties; at margin > 1 nat: 3 vs 3 |
| Fisher corr(old, tile) under p_D | | | median 0.93, pooled 0.85: both deviate together (the cache), the kernel term is small |

So the perplexity tool's -13% (its full-logit tail treatment) and this tool's -1% (common window) are the same data read through
different tail conventions; the body of the distribution favours the tile on ud as it did on q4, the dozen chaotic positions go
either way, and the same-top slip is 22 net coin-flip positions at p ~ 0.06. Nothing here says the new route is worse; nothing
here proves it better on ud the way the q4 f64 node reference did (the ud TR=9 instantiation was not dumped at the op level).
**The ud manifest rule (`pick.sh`: ud = BI/SPEC only, plus what the owner has explicitly taken) makes a NUM-TG flag on ud the
owner's explicit call - the entries stay `proposed` with this record until then.** Speed on ud at 32K: +5% coordinator / +11%
executors in the mix (above); at 96K the q4 curve suggests several times that.

## Replay gate on the q4 controller fork (`replaygate-0924-gqawmin`, 15:50-16:00)

The mint's q4 Turbo4 600 arm forked on its first run (`ae44d18ca4a9`, block histogram [3:106 5:1 7:81]) and gave the
replay-gated text on its second (`9e49b3d13b31`, [2:1 3:90 7:89]). `run-specev-replay-gate.sh` on the merged prod binary
(record once, replay x3, 949 tokens): record `b981f7376af5`, replays `b981f7376af5` x3, **322 picks, 0 desync, 0 past trace**
each. The same picks on the same tokens reproduce bit for bit: the fork is the controller's timing-dependent pick (a hot room
moves the cost EMA), not a kernel change - the one-slot graph does not carry the new route by construction.

## Resolution of the width-2 accuracy question (2026-09-24 evening): the old route was the vec kernel

Owner: "How would you go about finding the answer?" then "do 1 and 2 and we'll reassess", then "Do the sweep", then "Just go
for it" and "I don't see the point in reserving it for turbo4". Branch `exp/fa-gqa-w12-ss`, tree `llama.cpp-gqaw12`.

**Step 2 (the premise) answered it.** One stream, `-b W -ub W`, q4 pick env, Turbo4 cache, 2K wikitext, the first
attention layer's node at 1536 cells (inputs route-independent: layers 0-2 are GDN), float64 reference = `fa-dump-ref.py`:

| width | route | kernel (`fa-route:` line) | relRMS from exact | max abs |
|---|---|---|---|---|
| 2 | the pick (old) | NONE printed = the vec getter: `kernel_flash_attn_ext_vec_turbo4_dk256_dv256` | 3.66e-4 | 1.32e-3 |
| 2 | 8-row tile forced (`GGML_FA_VEC_MAX=1 GGML_FA_GQA_HEADS=1 GGML_FA_Q16=0 GGML_FA_Q24=0`) | `qtnw_turbo4 ... nsg=4 nwg=20 gqah=1` | 2.01e-4 | 4.01e-4 |
| 2 | GQA tile (`GGML_FA_GQA_WMIN_ALL=1 KVMIN=0`) | `qtnw24_turbo4 ... gqah=6` | 2.01e-4, **byte-identical to the row above** | 4.01e-4 |
| 4 | 8-row tile gqah=1 (`GGML_FA_GQA_HEADS=1`) | `qtnw_turbo4 ... gqah=1` | 2.08e-4, **byte-identical to the 24-row GQA tile** | 5.29e-4 |
| 8 | 8-row tile gqah=1 | `qtnw_turbo4 ... gqah=1 bcm=0` | 2.06e-4 | 8.81e-4 |
| 1 | the pick (old) | none = vec | 3.75e-4 | 7.98e-4 |
| 1 | GQA tile | `qtnw_turbo4 ... gqah=6` | 2.09e-4 | 4.28e-4 |

The two-stream config of the 05:20 section (`-c 2048 -b 4096 -ub 4`) reproduces its "old route" row exactly (3.42e-4 /
1.44e-3) and prints no route line either. `GGML_FA_VEC_MAX=3` sends `ne01 < 3` to the vec kernel whatever the stream count
or extent (`ggml_metal_op_flash_attn_ext_use_vec` looks at ne01 and ne00 only). The morning's "8-row tile at gqah=1" was a
label taken from the routing code, not from a route line - and the `fa-route:` print lives in the tile getter only, so the
absence of a line IS the vec route (rule captured in the skill, step 4b).

**Step 1 (the residual's structure, `perf/fa-residual.py`)** on the same node: both routes' residuals align with the
heaviest-weighted keys (|cos| with a single key direction ~0.9, no key index shared across heads); every explicit
off-by-one hypothesis (the other token's mask row, a dropped boundary key, an added masked key) is 1e-3..1e-1 from the
kernel output, orders of magnitude worse than exact. Rounding of the dominant terms, not a mapping error. The vec kernel's
extra share has a visible source: `dequantize_turbo4_0_t4` = half centroid table x half norm, K/V staged as `half4` - a half
rounding per cache element; the tile's TR forms keep a float table with the norm folded out (TR 7) or staged (TR 9).

**The speed pick that put widths 1-2 on the vec kernel (turbo4-filled-100k.md F, 2026-09-02) is stale.** It measured the
vec kernel against the 8-row tile at gqah=1 (no reuse below width 3) before the TR forms, split-K 20 and the GQA tile at
widths 1-2 existed. Per call, one stream, `test-backend-ops perf`, GQA 6, interleaved x3 (spread <= 2%), HOT AMBIENT 27C
(the vec numbers = the Sep 2 table within 2%, so no throttling at this load), `perf/run-fa-w12-timing.sh`:

| line | cache | kv | width | vec (the pick) | GQA tile (`GGML_FA_GQA_WMIN=1`) | tile / vec |
|---|---|---:|---:|---:|---:|---:|
| q4 | turbo4 | 8448 | 1 | 287 us | 74 | 0.26x |
| q4 | turbo4 | 8448 | 2 | 537 | 125 | 0.23x |
| q4 | turbo4 | 102400 | 1 | 3752 | 800 | 0.21x |
| q4 | turbo4 | 102400 | 2 | 6972 | 1335 | 0.19x |
| ud | turbo4 | 8448 | 1 | 287 | 79 | 0.28x |
| ud | turbo4 | 8448 | 2 | 540 | 129 | 0.24x |
| ud | turbo4 | 102400 | 1 | 3750 | 871 | 0.23x |
| ud | turbo4 | 102400 | 2 | 6965 | 1411 | 0.20x |
| q4 | f16 | 8448 | 1 | 213 | 163 | 0.76x |
| q4 | f16 | 8448 | 2 | 378 | 192 | 0.51x |
| q4 | f16 | 102400 | 1 | 6225 | 1823 | 0.29x |
| q4 | f16 | 102400 | 2 | 10254 | 2255 | 0.22x |
| ud | f16 | 8448 | 1 | 216 | 157 | 0.73x |
| ud | f16 | 8448 | 2 | 382 | 187 | 0.49x |
| ud | f16 | 102400 | 1 | 6218 | 1781 | 0.29x |
| ud | f16 | 102400 | 2 | 10804 | 2149 | 0.20x |

(f16 rows and the re-timed Turbo4 rows: the second sweep on the branch binary, 17:10, spread up to 8% on the tile arm and
15% on one f16 vec cell - the sun was lower but the room was still 27C; the Turbo4 ratios repeat the 16:40 sweep within
0.02x. The f16 tile at width 1 is the 8-row `qt_f16 ... nwg=8 gqah=6` tile: the f16 line has no `GGML_FA_TURBO_NWG=20`,
so its split width is the generic `GGML_FA_MM_NWG=8` - a further lever, not taken here.) Owner on f16: "I don't see the
point in reserving it for turbo4" - the rule is cache-agnostic.

At width 2 the pick's 24-row form (`GGML_FA_Q24_ROWS=12`) beats `GGML_FA_Q24=0` by ~10%. 16 attention layers: per token at
width 1 the vec route costs 3.4 ms more at 8K and 47 ms more at 100K; at width 2, 6.6 / 90 ms per round.

**Adopted (owner: "Just go for it"): `GGML_FA_GQA_WMIN=1`** - the GQA tile from width 1 on every cache, stream count and
extent (ggml-metal-ops.cpp; `GGML_FA_GQA_WMIN_MS/_KVMIN/_ALL` stay as the narrower rules and the probe). Both lines,
NUM-TG. What moves: one-slot width-1/2 calls = the mint's `batch1-300` (no-spec, f16) and `mtp-d1-300` (width 2, f16) arms
and the new `turbo4-b1-300` arm (the Turbo4 no-spec anchor, added to `run-prod-pick.sh`); the depth-3+ pick arms verify at
widths 3-7 and never see it. Gates below as they land.

**Gates (17:13-17:53, branch binary dd1369295):** `test-backend-ops test -o FLASH_ATTN_EXT` under `GGML_FA_GQA_WMIN=1`:
4869/4869 on q4 turbo4, q4 f16, ud turbo4, ud f16. E2e `run-prod-pick.sh` ABAB x2 per line on the three arms that run
widths 1-2 (`EXTRA=GGML_FA_GQA_WMIN=3` = the old routes; 300-token prompt + 300 generated, so the calls sit under 1K
cells where the per-call win is smallest; hot ambient, interleaved):

| arm (width, cache) | line | old t/s (x2) | new t/s (x2) | delta | sha |
|---|---|---|---|---|---|
| `batch1-300` (no-spec = width 1, f16) | q4 | 14.29 / 14.76 | 14.52 / 14.90 | +1.0..1.6% | `d2953fccfb41` both = HELD |
| | ud | 12.95 / 13.34 | 13.55 / 13.55 | +1.5..4.6% | `9c53aaade052` both = HELD |
| `mtp-d1-300` (width 2, f16) | q4 | 22.68 / 23.35 | 23.48 / 24.31 | +3.5..4.1% | `d2953fccfb41` = HELD, acc 86.2% |
| | ud | 18.88 / 19.41 | 19.39 / 20.03 | +2.7..3.2% | `9c53aaade052` = HELD, acc 87.4% |
| `turbo4-b1-300` (no-spec = width 1, Turbo4; NEW arm) | q4 | 13.67 / 14.06 | 14.32 / 14.79 | +4.7..5.2% | `86213d038a29` -> `7c5254d01b12` = MOVES (the TR 7 tile vs the half-dequant vec kernel) |
| | ud | 12.42 / 12.85 | 13.40 / 13.39 | +4.3..7.9% | `d180ae89f168` both = HELD |

So on the f16 cache the move is text-identical on both lines at this length (f16 K/V are exact in both kernels; only the
summation order differs), and on Turbo4 only the q4 line's no-spec text moves (its `turbo4-b1-300` reference is
`7c5254d01b12` from here on; the arm is new to the mint, so no minted sha changes). Every depth-3+ pick arm verifies at
widths 3-7 and is untouched by construction. KLD pricing of the width-1 route (pairwise vs the vec route, 12 chunks at 2K,
`-b 1 -ub 1`): below when it lands.

**KLD pricing of the width-1 route (`kld-w1-turbo4-{q4,ud}-1s`, 17:53-, one stream `-b 1 -ub 1`, 12 chunks at 2K,
Turbo4 both arms, base = the vec route (`GGML_FA_GQA_WMIN=3`), test = the tile):**

| line | mean KLD | median | 99.0% | 99.9% | max | same top | RMS dp |
|---|---|---|---|---|---|---|---|
| q4 | 0.00354 ± 0.00158 | 0.000194 | 0.0101 | 0.319 | 18.7 | 98.925 ± 0.093% | 1.85% |
| (the morning's width-2 pair, same base kind) | 0.00223 ± 0.00044 | 0.000205 | | 0.299 | 4.18 | 98.83% | |
| ud | 0.00319 ± 0.00153 | 0.000167 | 0.0071 | 0.359 | 18.5 | 98.957 ± 0.092% | 1.36% |
| (the afternoon's ud width-2 pair) | 0.00238 | | | | | 98.77% | |

The width-1 pairs are the width-2 class on both lines: the medians and the same-tops are the same numbers, the means are
tails (one ~18.5 position on each line, error bars 0.0015 - the "a mean KL can be one chaotic position" rule). Its direction was settled at the node:
the tile is 2.09e-4 from exact at width 1, the vec kernel 3.75e-4. The f16 pairs were not run: the e2e text is
identical on both f16 arms and f16 K/V are exact in both kernels (summation order only) - open if anyone wants the number.

**Minted (`prodpick-sep24-gqaw12-{q4,ud}`, 19:52-20:34, prod `d4c0d1b64`, HOT AMBIENT 27 C = a SHA mint; numbers in the README
merge log):** every one-slot sha on both lines = the recorded lineage; the new `turbo4-b1-300` arm records `7c5254d01b12` (q4)
and `d180ae89f168` (ud). Multi-slot: q4 short PASS, long = the executors-alone shas moved onto the mix-phase shas (their 512-cell
width-2 calls left the vec kernel too - the executors' text no longer depends on a long coordinator beside them; `REF_LONG_Q4`
re-pinned), solo and mix held; ud long PASS, short = slot 2 (f16, 16 tokens) `a3c90139bbfd` -> `ff519f555a75`, stable on 3
reruns, re-pinned. Worktree `llama.cpp-gqaw12` removed, branch `exp/fa-gqa-w12-ss` kept.

Open after this: (1) the f16 GQA tile at widths 1-2 runs its generic split (`GGML_FA_MM_NWG=8`; the Turbo4 tile has
`GGML_FA_TURBO_NWG=20`) - an untimed lever; (2) the f16 width-1/2 KLD pairs were not run (text identical on every f16 arm);
(3) `GGML_FA_VEC_MAX=3` is now inert for GQA-6 shapes on both caches (widths 1-2 go to the tile before the vec rule is
consulted) and only routes non-GQA or hsk >= 512 shapes; (4) `GGML_FA_GQA_WMIN_MS/_KVMIN/_ALL` are subsumed by `_WMIN=1` in
the pick and stay as the narrower rules.

## Per-slot context sizes: the packed layout (2026-09-25, branch `exp/kv-size-classes`, tree `~/play/llama.cpp-kvclass`)

**Ask (owner):** "what's the point of a smaller max context size if I allocate the same amount anyway?" - split mode gave
every slot the coordinator's cache. **Built:** `--ctx-seq-sizes 98304,8192,8192,8192` (server: per slot; sets `-np` to the
count and `-c` to the sum; suffix `k`; needs split mode and flash attention; C API `llama_context_params.ctx_seq_sizes` +
`llama_n_ctx_seq_id()`). Each stream gets its own cell count and the streams are packed back to back per layer.

**Measured, q4 line, 96K + 3 x 8K vs 4 x 96K** (`-ctk/-ctv turbo4` outside the pick env, so auto-asymmetric K = q8_0;
the pick's Turbo4 K is smaller in absolute terms, the ratio is the layout's):

| layout | cells | KV buffer | recurrent state (unchanged) |
|---|---|---|---|
| split, 4 x 96K (before) | 393216 | 9696 MiB | 598.5 MiB |
| packed, 96K + 3 x 8K | 122880 | **3030 MiB (-69%)** | 598.5 MiB |

With the pick's Turbo4 K and V (`TURBO_AUTO_ASYMMETRIC=0`): 4 x 96K = 6336 MiB (3168 K + 3168 V), 96K + 3 x 8K = **1980 MiB**
(990 + 990), the same -69%; 16.5 MiB per 1K cells, so an 8K executor slot costs 132 MiB.

**Gate = byte identity against the split arm** (this is the whole point: the same text, a third of the memory). q4 line,
prod `49bda3039` references, both on the new binary: `run-multislot-gate.sh` split arm PASS (3/3 short, 8/8 long),
`ARM=classes` (the new arm: `--ctx-seq-sizes 32768,8192,8192,8192`, a 32K coordinator beside three 8K executors) PASS
3/3 short + 8/8 long - every sha (execs, solo, mix coordinator, mix executors) equals the recorded split references.
`test-backend-ops FLASH_ATTN_EXT` 4869/4869. Speed unchanged (mix coordinator 17.2 vs 17.1 t/s, executors 12.5-12.9
vs 12.4-12.9): the executors already read only their own cells through the per-stream `kv_len` extent.

**Design as built.** `llama_kv_cache`: one 2D `[n_embd, sum of the sizes]` tensor per layer, `v_offs` prefix sums,
`v_cells[s]` sized per stream; `k_stream/v_stream` views, `set_input_k_idxs/v_idxs` (global rows = `v_offs[strm] + idx`),
the K-shift input and graph, the mask (`n_kv` is the ubatch's longest stream, a shorter stream's missing cells are
dropped), state I/O and cross-stream `seq_cp` (row-prefix copy, the source's used cells must fit) all run off the
offsets. The uniform layout is the old bytes and the old graph (no new input, no kernel variant) - hence the split arm's
byte identity. A non-uniform cache has no uniform stream stride, so `get_k/get_v` return a view at the buffer base with
a zero stream stride, widened to `ns` streams in place (`ggml_view_4d` sizes a view as if contiguous - the packed tensors
carry `GGML_TENSOR_FLAG_LOOSE_VIEWS`, which relaxes that one check; the Metal per-view bounds check on the real strided
extent stays), and a new input `attn_inp_kv_off` (`ggml_flash_attn_ext_set_kv_off`, src[6], beside `kv_len` src[5])
carries each stream's first cell; the Metal `ext`/`vec`/`pad` kernels take a `kvoff` table under function constant +6
(`_kvo=1` in the route name) and replace `ikv3*nb13` by `kvoff[ikv3]*nb11`; the CPU FA asserts it away. The reserve
graph sizes `n_kv` by the largest stream. The drafter's SWA cache (`llama_kv_cache_iswa`) clamps each size to its window.
Server: slot i = sequence i; an unpinned task goes to the smallest idle slot whose context holds its prompt (LRU inside
that size class; an LCP-similar slot must also hold it); `id_slot` pinning unchanged; a prompt no slot holds reports the
old context error; if every slot that could hold it is busy the task is deferred. Exercised on `16384,4096,4096`: a 5.6K
prompt took slot 0, two short ones slots 1-2, the next short one slot 2 (LRU), a 20K prompt got HTTP 400.

**Not covered:** the transposed (non-FA) V layout and non-Metal backends refuse a mixed list; `llama_kv_cache_msa/dsa/dsv4`
refuse it (create_memory guard); `--kv-unified` refuses it. ~~Adoption = owner (merge to prod; the gate's `ARM=classes`
arm can join the mint if the pick starts using the list).~~ **MERGED TO PROD 2026-09-25 (dc02ff4d6, owner: "Bring it onto
prod")**; post-merge gates on the prod binary: split arm PASS 3/3 + LONG 8/8, `ARM=classes` PASS 3/3 + 8/8 (tags
`prodmerge-0925-{split,classes}`). No re-mint: the uniform layout builds the identical graph and every one-slot arm is
that graph. ~~Open: put `ARM=classes` into the mint's multi-slot call once the pick uses a size list.~~ DONE 2026-09-25
(owner: "Make it so", prod `8ccffca2e`): `run-prod-pick.sh` runs the classes arm after the split arm; first mint with it
`prodpick-sep25-kvclass-{q4,ud}` (cool day): all 16 multi-slot shas PASS on both lines, one-slot shas canonical, t/s
at the Sep 18 level (README "The prod pick"). Worktree `llama.cpp-kvclass` removed, branch kept.
