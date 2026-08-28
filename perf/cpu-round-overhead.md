# CPU-side per-round overhead: ~~17~~ ~10 ms nobody has ever attacked

Status: **OPEN, first diagnostic session 2026-08-28.** Opened 2026-08-28 (owner: "draft
a stub for the CPU work... a win's a win"), after the WL_XL adoption moved the pick to
27.7 t/s at 300 units. The round decomposition re-run at the extended pick is deferred
until this work lands - do both together.

> **Corrected before the first run (same day): the headline was ~17 ms, and ~7 of it
> was a stale claim.** The "drafter CPU ~8 ms" line below came from the attribution
> `round-decomp-fused.md` itself struck: its same-day follow-up measured draft_call =
> 16.4 ms drafter GPU wait + 0.7 CPU submit + ~0 lattice at n6 ("the prior '~8 ms
> CPU/lattice' attribution was wrong"). And the per-context profiler tag this stub
> queued as the first move ALREADY LANDED that day (`m<N>` prefix,
> `ggml-metal-context.m:72`; the drafter is attributed per-op in that file's "Drafter
> forward attributed per-op" section). The real uncharted CPU line is the **9.4 ms
> submit** plus the drafter's ~0.7 ms - about 10 ms, not 17. Code branch:
> `cpu-round-overhead`, off prod (prod fast-forwarded to the shortk tip `4defdc3b6`
> the same day).

## Why this is the frontier

At the extended pick the round is ~116.5 ms and the two CPU-side lines are ~17 ms of it:

- ~~**CPU graph build/submit (`dec_sub_tg`): 9.4 ms/round, 8%.** Flat across FOUR picks
  since 2026-08-22 (n6+skinny, n3+v3, n4+w5, n4+w5+XL) while the GPU side shrank ~25% -
  it is pure fixed cost and its share grows with every kernel win. Never decomposed,
  never attacked.~~ **THE 9.4 WAS A PROFILER ARTIFACT (first diagnostic run, same
  day).** Every recorded dec_sub_tg came from a `GGML_METAL_PROFILE=1` run deflated
  by the round ratio - the exact method the README's own rule says cannot correct
  the submit path ("it just relabels profiler overhead as CPU submit"; raw profiled
  ~11.4 x0.823 = 9.4, real 2.56, ~4.5x inflation, in the documented 6-8x encode
  band). Unprofiled at the pick (`run-cpu-overhead.sh`, LLAMA_DECODE_PROF=1,
  non-perturbing, sha canonical): **dec_sub_tg = 2.56 ms/round at 300 units / 2.17
  at 600**, of which the ggml-metal encode window is ~1.7-2.0 and the rest is
  reuse-check 0.10 + set_inputs 0.01 + llama_decode bookkeeping ~0.5. "Flat across
  four picks" was the artifact being flat: profiler encode inflation depends only
  on node count, which never changed.
- ~~**Drafter CPU (~8 ms/round est.)** - `round-decomp-fused.md`'s prior attribution of
  draft_call was ~8 ms CPU/lattice + ~10 GPU at the n6 point; the GPU half has since
  shrunk (FFN + head ride w5r4h) but the CPU half has never been measured separately.
  **Per-op attribution is blocked by the profiler key collision**: drafter and target
  are both q4_0 with shared dims and `g_prof_entries` is global across the two Metal
  contexts (`round-decomp-fused.md` already queued the fix: a per-context tag on the
  profiler key).~~ **STALE ON BOTH COUNTS (see Status blockquote): drafter CPU is
  ~0.7 ms, and the tag landed 2026-08-22.** The drafter's remaining CPU interest is
  whether graph reuse engages for ctx_dft (its `n_reused` is printed nowhere) and
  whether its many small decodes pay a fixed submit floor - owner: "the draft model
  also needs this."

~~Ceiling honesty: submit to zero is +8.7% e2e; realistic is a fraction of that.~~
With the artifact corrected, submit-to-zero is +2.3% - and the submit-prof run below
shows most of it is already hidden under GPU execution, so the honest submit ceiling
is under 1%. But the same diagnostics found an UNTIMED ~11 ms/round at 600 units
(below) - larger than everything the stub originally listed.

## Findings, first diagnostic session (2026-08-28, branch `cpu-round-overhead`)

Tools built: `perf/run-cpu-overhead.sh` (decode-prof + submit-prof arms at the pick),
`GGML_METAL_SUBMIT_PROF=1` (new, ggml-metal-context.m: per-graph GPU timeline vs the
host encode window from MTLCommandBuffer GPUStartTime/GPUEndTime, windowed per-ctx
averages every 64 graphs - no encoders added, non-perturbing, drafter ctx included),
and `loop_gap`/`loop_body` timers in the server spec-prof dump. All runs hold the
canonical shas; anchors this session read ~+2.5% over the record (cross-session
drift, within the ~3% documented band).

**The verify submit is real but mostly hidden (steady windows, 600 units).** Target
graph = 4230 nodes: encode window (`sub`) 1.7-1.8 ms, GPU busy 96.4-96.6 ms,
inter-chunk gaps 0.002 ms (the n_main-first commit scheme never starves the GPU at
n_cb=1), and **exposed GPU idle inside graph_compute = 0.86-1.02 ms/graph, all of it
`pre`** - the entry -> first-GPU-start latency (encode n_main=423 nodes + commit +
schedule). Batch-1: sub 1.34, busy 73.5, exposed = pre = 0.37-0.58. The encode
LENGTH is a non-lever: cutting it moves nothing while gaps are 0 and tail is 95.7
(the GPU keeps running 95.7 ms after the CPU finished encoding). The only submit
lever is the fixed per-graph `pre`, worth at most ~1 ms/round on the target.

**The drafter is GPU-bound too (owner asked; measured per-ctx).** ~3.1 graphs/round
(306-320 nodes each after warmup): sub 0.125, pre 0.10, gaps 0.04, busy 4.4-4.6 -
3.1 x 4.4 = 13.6 ms = the whole 13.4 ms draft_call. Graph reuse engages for ctx_dft
(reuse-check 0.16-0.19 ms, no build spikes). Drafter CPU exposure: ~0.45 ms/round.

**Prefill submits: anomalous, unexplained, out of scope for the round work.**
dec_sub_pp reads ~9.7 s per prefill batch (48 s total for the ~10k benchprompt) with
dec_syn_pp at zero - the wait is inside llama_decode, before the syn timer. A
plausible alloc-waits-on-previous-GPU story does not survive the arithmetic (the
DEBUG-only synchronize in the server decode loop drains each batch, yet every batch
still pays ~9.7 s), and t_decode totals imply prefill wall ~66 s, ~18 s of it after
submit returns. Prefill is compute-bound GPU work of roughly this magnitude on this
machine, so it may be nothing - but WHERE the wait surfaces is not understood, and
nobody has profiled prefill on this stack. Open, separate from the decode round.

~~**OPEN - the untimed ~11 ms/round at 600 units.**~~ **RESOLVED - it was my own
divisor error, not missing time.** Rounds at 600 units were derived as 600/3.28
committed (the 300-unit acceptance); the run's own `draft acceptance` line says mean
len 2.98 -> 200 rounds, and 23.09 s / 200 = 115.5 ms/round, which the component sum
(115.8) and the loop windows (114.7-115.0, flat across the whole run) both match.
The `loop_gap`/`loop_body` timers proved the stronger statement along the way:
**update_slots iterations account for 100.0% of wall in every 5 s window
(loop_gap = 0.001 ms/round), so there is NO untimed server cost anywhere in the
decode loop.** (Same trap as ever: `verify-before-generalizing-one-line` - count
rounds from the run's own counters, never from another run's acceptance.)

**The corrected CPU-exposed budget/round (600 units, round 115.5 ms):** GPU busy
~110.2 (target 96.6 + drafter 3.1 x 4.4), CPU-exposed ~5.3 ms:

| item | ms/rd | note |
|---|---:|---|
| post-GPU tail inside dec_syn | ~3.6 | dec_sub+dec_syn end 101.3 ms from decode start vs GPU last-buffer end 97.5 - see the get_tensor_async finding below |
| target `pre` (entry -> first GPU start) | 0.85 | n_main encode + commit + schedule; b1 reads 0.4-0.6 |
| drafter pre+gaps (~3.1 graphs) | 0.45 | |
| accept_blk | 0.36 | sampler clone + sample_and_accept_n + rollback |
| post_decode | 0.37 | |
| dec_sub bookkeeping (reuse-check/set_inputs/rest) | ~0.4 | |

**The single largest item has a named mechanism: `ggml_metal_get_tensor_async`.**
The per-round logits readback (5 rows x 248320 f32 = 4.97 MB) wraps the HOST
destination in a fresh `newBufferWithBytesNoCopy` MTLBuffer, encodes a blit into an
extra command buffer queued BEHIND the whole graph, and synchronize waits for that
extra GPU roundtrip - buffer wrap + blit + commit + completion, all serial after the
graph. On unified memory a plain memcpy after the wait is equivalent.
**Experiment `GGML_METAL_GET_MEMCPY=1`** (ggml-metal-context.m): get_tensor_async
defers {dst, src buffer, offs, size}; synchronize performs the memcpys after the
wait; a safety drain at the next graph_compute entry preserves queue-order semantics
for any get-without-sync caller.

**MEASURED: +3.3% e2e at the pick, byte-identical (TAG `cpuovh-aug28f`, interleaved
A/B at 600 units, all four arms sha `3776c0adb7ee`):**

| arm | t/s | dec_sub_tg | dec_syn_tg |
|---|---:|---:|---:|
| ctrl-a | 25.304 | 2.391 | 101.694 |
| **on-a** | **26.064** | 1.990 | 98.869 |
| ctrl-b | 25.534 | 2.285 | 100.824 |
| **on-b** | **26.452** | 2.043 | 97.310 |

ctrl mean 25.42 -> on mean 26.26 = **+3.3%**, each ON beats BOTH surrounding CTRLs
(effect >> today's ~1% run spread). dec_syn -3.1 ms (the blit roundtrip + wait),
dec_sub -0.3 (the newBufferWithBytesNoCopy wrap + commit inside llama_decode) -
together the predicted ~3.4 ms. Larger e2e than the WL_XL lever adopted this
morning. Cost: none - no residency, no quality, no output change; the copy always
ran on CPU-visible memory, upstream just routed it through the GPU queue.
(Today's ctrl arms read ~1.5% under the canonical 25.79 - cross-session drift;
within-harness deltas are the measurement.) Batch-1 control (`cpuovh-aug28g`):
13.397 -> 13.452 (+0.4%), sha canonical - coherent, the b1 copy is one row (~1 MB)
so the saving scales with copy size as the mechanism predicts.
~~**Unmerged, branch `cpu-round-overhead` - adoption into the pick is the owner's
call** (like BSPLIT it is a pure code-default question, no trade-off found).~~
**ADOPTED 2026-08-28 afternoon (owner: "pick get_memcpy")**: in `run-prod-pick.sh`
PICK_ENV and the README pick block; branch merged to prod; canonical numbers TAG
`prodpick-aug28-gmc`.

**Adjacent non-CPU observation, for the drafter plane:** ~~the drafter streams its
~1 GB of weights ~3.1 times per round (3.1 graphs x 4.4 ms at ~235 GB/s - near
peak). Its 13.6 ms/round is bandwidth x graph COUNT; a lattice that drafted in
fewer forwards would save ~4.4 ms per graph removed~~ **REFUTED same evening
(`drafter-graph-count.md` correction block): those were this profiler's 64-graph
WINDOW AVERAGES read as per-graph facts.** Real split (dflash-prof, same log):
draft decode ~13 ms, enc ~0.6, inject ~0.55 - one full stream plus two slivers.
The submit-prof numbers here stay valid for what they claim (per-ctx totals and
CPU exposure); only the per-graph uniformity inference was wrong.

## Open questions, in order

> All five answered in the first session (2026-08-28) - see Findings. 1: decomposed
> (artifact + encode split). 2: reuse already engages every steady round, both ctxs
> (91 reuses / 92 rounds; drafter reuse-check 0.16 ms, no build spikes). 3: submit is
> NOT serialized with GPU idle - encode is fully hidden, only `pre` (~0.9 ms) and the
> post-GPU readback (~3.6 ms, now the GET_MEMCPY lever) are exposed. 4: struck
> (answered 2026-08-22). 5: loop_gap/loop_body prove no harness/server noise inside
> the measured round. What remains open: GET_MEMCPY adoption (owner), the ~2 ms of
> micro-items (pre, accept_blk, post_decode), the drafter graph-count observation
> (owner's plane), the prefill anomaly, and the round-decomp re-run when this lands.

1. **Decompose the 9.4 ms.** llama.cpp rebuilds the ggml graph and ggml-metal re-encodes
   it every decode: graph build vs metal encode vs concurrency analysis vs command-buffer
   commit/wait - which dominates? Method: `LLAMA_DECODE_PROF=1` gives host-side per-call
   terms (`width4-gap-decomposition.md` used it: "target decode 2.26 ms/call (1.89
   submit)" at the old point); a `sample`/Instruments profile of llama-server during
   steady-state decode attributes the submit path by symbol.
2. **Is any of it reusable across rounds?** At a fixed operating point the verify graph
   has the same topology every round (widths 5/1, same tensors) - graph caching,
   pre-encoded command buffers, or memoized concurrency analysis would turn per-round
   cost into per-request cost. Check what upstream's graph-reuse machinery (if any) does
   on this path before building anything.
3. **Is submit serialized with GPU idle?** If the GPU drains while the CPU builds the
   next graph, the 9.4 ms is on the critical path in full; if build overlaps the GPU
   tail, only part of it is. Measure the inter-round GPU gap (timeline from the replay
   tooling, or bracket dispatch timestamps) before pricing any fix.
4. ~~**Split the drafter's ~8 ms.** Land the per-context profiler tag first (small,
   already-specified change), then decompose draft_call: lattice bookkeeping vs
   tokenization vs its own submit vs GPU wait. The lattice is pure CPU work that has
   never been profiled.~~ DONE 2026-08-22 in `round-decomp-fused.md` (tag landed,
   draft_call = 16.4 GPU + 0.7 CPU + ~0 lattice). Remaining drafter question folded
   into the submit work: does ctx_dft reuse graphs, and what is its per-decode
   submit floor x its decodes/round?
5. **Sanity: subtract harness noise.** Confirm server-side logging, health polling and
   the completion-endpoint bookkeeping are not inside the measured round (they should
   not be, but nobody checked).

## Method notes

- The e2e judge is `run-prod-pick.sh` unchanged; any CPU fix must hold sha
  `9ad7e023c6ab` at 300 units - CPU-side changes have no business changing output.
- Profiled runs: `GGML_METAL_PROFILE=1` serializes concurrency and inflates the round
  (ts-factor caveat, `round-decomp-w5n4.md`) - use it for shares, use unprofiled anchors
  for real ms, per the standing method.
- ggml INFO logs are invisible at llama-server's default verbosity; diagnose with
  `-lv 5` (learned the hard way, `shortk-head.md`).

## Cross-links

`round-decomp-w5n4.md` (the ledger this attacks), `round-decomp-fused.md` (drafter
attribution + the queued per-context profiler tag), `shortk-head.md` (why the non-kernel
share keeps growing), `m4-width5-crossover.md` (the mv wall that makes this the open
frontier).
