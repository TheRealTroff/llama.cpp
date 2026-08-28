# Drafter graph count: ~~3.1 weight-streams per round, the largest open lever on the board~~ PREMISE REFUTED - one full stream plus two slivers

Status: **the opening premise fell 2026-08-28 evening, before any runs, from evidence
already on record** (same-morning logs + code reading). The corrected decomposition and
the reframed open questions are below. Opened 2026-08-28 (owner: "stub it") from the
CPU-round-overhead session's per-context submit-prof data (`cpu-round-overhead.md`).

> ## THE CORRECTION - read this before the struck text below
>
> The opening measurement read `GGML_METAL_SUBMIT_PROF`'s **64-graph window averages**
> as per-graph facts. The profiler prints `sum/n` every 64 graphs and has no per-graph
> or per-topology breakdown (`ggml-metal-context.m:322-329`). The drafter submits a
> strict per-round cycle [enc, inject, draft-decode]; 64 is not divisible by 3, so each
> window's composition shifts by one graph and the *averages* cycle through three nearby
> values (nodes 306/307/320, busy 4.4-4.6) - manufactured uniformity. Three independent
> same-config sources give the real split:
>
> 1. **In-tree dflash-prof, same log as the submit-prof lines**
>    (`results/cpuovh-aug28e-sprof-n4.server.log`, steady state at the pick):
>    enc 0.62-0.68 ms wall (sync-inclusive), inject+sync ~0.55, noise-decode submit
>    0.72, **lattice sync 13.0-13.8 ms**. The drafter's GPU cost is ONE graph - the
>    draft decode. `drafter-pipelining.md`'s method note ("grep the in-tree dflash-prof
>    counters before scoping a lever") applied verbatim and was not followed - third
>    over-generalization event after the two in the drafter-head story; the full dump
>    section was one grep away in the same file.
> 2. **The graphs themselves** (`src/models/dflash.cpp`): the enc graph is one fc
>    matmul + RMS norm, a handful of nodes (`graph<true>`, :268-281); the inject graph
>    builds only per-layer wk/wv + KV copies - no attention, no FFN, no head
>    (`graph<false>` ubatch.embd branch, :474-535). Neither can stream ~1 GB.
> 3. **The m2 per-op dump** (`results/rounddecomp-aug28-prof-n4.server.log`, 93 rounds,
>    pre-XL pick) prices the draft decode's inside, per round: lm_head 5.01 ms
>    (post-XL ~3.2, `shortk-head.md`), FFN mv ~3.8 (gate/up 249 us x 10 + down
>    266 us x 5), TOP_K 1.09 on [248320,5], FLASH_ATTN ~1.6 over 8448-8704 KV, small
>    projections ~2, elementwise tail ~1+. Profiled shares (GGML_METAL_PROFILE
>    serializes) - treat as upper bounds, not walls.
>
> Also corrected: `draft_call` (13.4-13.9 ms) wraps only `draft()`
> (`server-context.cpp:3105`); enc+inject (~1.2 ms wall incl. their syncs) run in
> `process()` outside it - drafter total is ~14.6-15 ms/round, ~12.5% of the round.

## The measurement ~~and what it meant~~ (original text, struck where wrong)

At the prod pick (dflash n4 + w5 + XL + GET_MEMCPY, 600 units, round 115.5 ms):

- `draft_call` = 13.4-13.9 ms/round = **11.6% of the round**, second only to verify GPU.
- The drafter ctx runs **~3.1 graphs per round** ~~, three distinct topologies cycling
  (window node-averages repeat 306/307/320), each 4.4-4.6 ms of GPU busy~~ - the count
  is right (enc + inject + draft decode + ~0.1 window-flush), the uniformity was the
  windowing artifact above. Real split: draft decode ~13 ms, enc ~0.4 GPU / 0.63 wall,
  inject <=0.55 wall.
- ~~Each graph moves at **~235 GB/s against the 273 peak** - the drafter is cleanly
  bandwidth-bound~~ REFUTED: 235 GB/s assumed 3.1 full streams. The decode streams the
  weights ONCE in ~13 ms. Its mv calls individually ride w5r4h at the known
  1.2-1.4x-of-floor mv wall (lm_head 3175 vs 2620 us floor; ffn 249 vs ~183) - the mv
  plane stays closed - but **~35-40% of the decode is non-mv tail** (head TOP_K, FA,
  elementwise) that the "near peak" framing hid.
  CPU exposure is ~0.45 ms/round; there is nothing to win on the drafter's CPU side
  (the old "~8 ms drafter CPU/lattice" was struck in `round-decomp-fused.md`).

~~So the drafter's cost is **weight-stream count x bandwidth**, per-stream cost is
already near peak, and the count is 3.1.~~ The drafter's cost is one full forward
whose inside decomposes as above, plus ~1.2 ms of enc/inject slivers.

## ~~Why this is the largest open lever~~ What survives, what falls

- **FALLS**: ~~each graph removed saves ~4.4 ms = +3.8% e2e; a single-forward drafter
  ~+8%~~. Fusing enc+inject (or folding them into the draft decode) has a **~1.2 ms
  ceiling (~1% e2e)** - they were never weight streams. The "fewer graphs" question
  this stub was opened to ask is answered: yes, mechanically easy, and nearly worthless.
- **SURVIVES**: the graph count itself; the dependency anatomy and section-3 queue
  blocker in `drafter-pipelining.md`; and the invariant that makes drafter work safe:
  **speculation is lossless.** The canonical shas (`9ad7e023c6ab` at 300,
  `3776c0adb7ee` at 600) gate any drafting change; acceptance (49.8%/55.7% at 600/300,
  mean len ~3) is the quantity to watch.

## Open questions, reframed (in order)

1. **TOP_K, 1.09 ms/round on [248320,5]** - a fixed per-round cost ~1% of e2e on its
   own. **Scoped 2026-08-28 evening (owner: "what's hiding behind those 2 doors?"):**
   the Metal op is NOT a naive full argsort - it is block-bitonic top-k (1024-wide
   blocks, keep 16 -> 243 blocks -> 3888 candidates) followed by a merge ladder of
   **8 serialized dispatches** (`ggml-metal-ops.cpp` ggml_metal_op_top_k, each round
   a concurrency reset, tail rounds single-threadgroup latency-bound). The data is
   only ~5 MB (~20 us at peak), so the 1.09 ms is dispatch/serialization structure.
   Design on the shelf: streaming two-dispatch top-k (each threadgroup scans a strip
   keeping a local top-16, one merge of the ~1-4k candidates) - expect ~0.1-0.15 ms,
   **prize ~0.9 ms/round ~ +0.8% e2e**. Confirm attribution with the per-instruction
   profiler before building. Algorithmic sub-door (does the selector need exact
   full-vocab top-16?) is drafter design - owner's.
   **ON HOLD 2026-08-28 (owner: "hold off on top K") - do not build unprompted.**
2. ~~**Drafter FA runs over the full ~8.4k KV**~~ **MEASURED same evening
   (`run-draft-window.sh`, interleaved 600 units, sha canonical in EVERY arm - text
   is verify-gated, so windowing is sha-safe by construction):** the already-built
   `LLAMA_DRAFT_WINDOW` machinery was simply never benchmarked. Sweep: w512 +1.25%
   (acc 49.6), **w1024 +1.94% e2e (acc 50.1 - acceptance IMPROVES over full-context
   49.8)**, w2048 +0.94% (acc 49.8); drafter lattice 13.1-13.4 -> 11.7-11.8 ms.
   Clean knee at 1024. **The largest single lever measured on the drafter plane.**
   ~~Two follow-ups if adopted: (a) window mode currently falls back off
   `DFLASH_FUSED_INJECT` (ring needs g rows on host)~~ **(a) DONE same evening
   (owner: "do the others"): the ring stores enc-width FEATURE rows in fused mode
   (host already has them for free in process(); a rebuild replays them through the
   fused wide graph - KV at p is a pure function of the injected row at p either
   way; costs ~5x ring memory, ~111 MB/seq at sink64+w1024). All three flags
   compose: **stack = +1.87% e2e vs ctrl (26.216/26.279 vs 25.80/25.731 at 600),
   +0.73% on top of the window alone, acc 50.1, every sha canonical including the
   300-unit gate that exercises the ring rebuild through the fused graph.**
   **ADOPTED into the pick 2026-08-28 evening (owner: "do the others"):
   `DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024` in
   run-prod-pick.sh PICK_ENV + README pick block; mint TAG
   `prodpick-aug28-drafter`.** Still open: (b) the window's win should GROW with
   context (FA scales linearly), re-check at longer prompts; sink=64 default and
   `full_ctx_min` untouched (a full_ctx_min sweep was never run).
3. **The elementwise/REPEAT/ADD tail** (~1-2 ms profiled). Serialization caveat in
   full force (small-ne01 lesson): line items are upper bounds, some of this hides
   under concurrent dispatch. Only e2e deltas count.
4. ~~**enc+inject fusion + dropping the inp_g host round-trip**~~ **BUILT AND MEASURED
   same evening (owner: "just do it"; branch `drafter-fused-inject`, harness
   `run-fused-inject.sh`).** `DFLASH_FUSED_INJECT=1`: the encoder fc+norm runs inside
   the injection graph (`llama_set_dflash_inject_wide` routes encoder-width embd
   through llama_decode; `src/models/dflash.cpp` graph<false> wide branch), one decode
   per process() chunk, no host round-trip for g. Alone it is FLAT (+0.04% - the sync
   still waits for the same GPU work), but it removes the mandatory enc->inject
   readback sync that blocked `DFLASH_ASYNC_INJECT`, and the pair goes submit-only
   (process wall 1.15 -> 0.135 ms): **+0.76% e2e at 600 units (ctrl 26.454/26.402 vs
   fasync 26.632/26.627, interleaved, repeats +-0.005), byte-identical at 300 AND 600
   - both canonical shas hold.** Falls back to the legacy path when the drafter window
   is on (ring_push needs g rows on host). Adoption into the pick = owner's call:
   `DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1` in PICK_ENV.
   (`drafter-pipelining.md` correction (a) said this fusion was impossible - that was
   true only for fusing the two EXISTING graphs; moving the fc+norm into the inject
   graph sidesteps it. Struck there.)
5. **The head as a whole**: lm_head + TOP_K ~ 4.3 ms/round post-XL, ~30% of the
   drafter. Anything structural here (smaller draft vocab, head factorization) is
   drafter design - owner's area, priced here for when that conversation happens.

## Method notes

- e2e judge: `run-prod-pick.sh` unchanged; shas must hold (losslessness gate).
- Size any delta from interleaved same-harness arms only - same-config absolutes
  wander +/-2-4% within a day, unattributed (README pick block, 2026-08-28).
- Acceptance is a property of the generated text and differs 300 vs 600 units
  (README trap 1 and `weight-quant-kld.md`) - compare acceptance at matched units.
- **Profiler literacy, the lesson this stub now embodies**: submit-prof prints
  windowed averages (`%64`), dflash-prof prints per-phase walls, GGML_METAL_PROFILE
  serializes. Cross-check any new counter's aggregation in the code before building
  levers on its numbers, and grep the in-tree per-phase counters first.

## Cross-links

**`drafter-pipelining.md` - READ FIRST** (the serialization anatomy, the
target-hidden-state dependency, the section-3 pipelining blocker, and the retired
ASYNC_INJECT step; merged from its branch 2026-08-28), `cpu-round-overhead.md` (the
submit-prof measurement this stub over-read; its "adjacent observation" carries the
same correction), `round-decomp-fused.md` (draft_call attribution history),
`round-decomp-w5n4.md` (the m2 dump's home), `shortk-head.md` (drafter lm_head on
w5r4h), `m4-width5-crossover.md` (the mv wall that ranked this lever first).
