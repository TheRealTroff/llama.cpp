# Pipeline the drafter under the verify

Status: **open - scoped 2026-08-22, not started. Read the blocker in section 3 before
writing any code; the naive version makes the round SLOWER.**

Opened from `mlx-cycle-capture.md`, which established that dflash_mlx launches the *next*
cycle's draft with `async_launch=True` (`spec_epoch.py:2490`) before the current cycle's
tokens are yielded, while ours is 16.4 ms of fully serialized drafter GPU wait
(`round-decomp-fused.md`), ~11% of the round. "If they can do it, so can we" is the right
instinct - but the two engines do not have the same execution model underneath, and that is
where the cost is.

## 1. Where the serialization actually is

One loop serves both the `-md` and MTP/dflash paths: `update_slots()`,
`tools/server/server-context.cpp:2804`. One call = one round, strictly ordered:

| step | file:line | cost |
|---|---|--:|
| drafter forward | `server-context.cpp:3084-3089` | **16.4 ms** |
| draft ids -> verify batch | `server-context.cpp:3142` -> `:549-580` | ~0 |
| target verify submit | `server-context.cpp:3725-3728` | 9.6 ms CPU |
| verify wait | `server-context.cpp:3730-3731` | 121.8 ms |
| drafter KV re-seed | `server-context.cpp:3792-3794` | - |

**`llama_decode` is already non-blocking** - it submits via
`ggml_backend_sched_graph_compute_async` (`src/llama-context.cpp:2558`), and the output sync
is deliberately deferred (`src/llama-context.cpp:2066-2067` literally has `//synchronize();`
with the comment "wait for the computation to finish (automatically done when obtaining the
model output)").

So the drafter's 16.4 ms is **not** a synchronous decode. It is the host readback of the
selector lattice, `common/speculative.cpp:1366-1368`, whose own comment says so:

```cpp
// this call synchronizes, so it carries the drafter's GPU wait
const int64_t t_lat0 = ggml_time_us();
const float * lattice = llama_get_embeddings_nextn(ctx_dft);
```

`llama_get_embeddings_nextn` -> `ctx->synchronize()` (`src/llama-context.cpp:3880-3884`) ->
`ggml_backend_metal_synchronize` -> `[cmd_buf_last waitUntilCompleted]`
(`ggml-metal-context.m:374-379`). The lattice walk after it is ~0 ms.

## 2. The dependency is real, and it is worse than "next token"

The verify batch is built from **host-side token ids** (`server-context.cpp:576`), so the
verify graph cannot be built until the lattice lands in host memory and is walked.

Worse, the drafter is not an independent model - it is conditioned on the *target's* hidden
states. `common/speculative.cpp:1220`:

```cpp
const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
```

which also synchronizes (`src/llama-context.cpp:3892-3895`). Those rows are copied
GPU->host (`extract_layer_inputs`, `src/llama-context.cpp:2252-2277`) and pushed back
host->GPU as `batch.embd` for `llama_encode(ctx_dft, ...)` (`speculative.cpp:1243`) and
`llama_decode(ctx_dft, batch_inject)` (`:1269`), each followed by its own
`llama_synchronize(ctx_dft)` (`:1162`, `:1276`).

**This is why MLX gets the overlap for free and we do not.** Their
`feature_store.require_current_hidden()` hands back a lazy array and the draft graph is
enqueued as its consumer - same device, no materialization, no host round trip. ggml has no
way to express "this context's graph consumes that context's tensor", so every drafter/target
handoff round-trips through host memory.

## 3. THE BLOCKER: both contexts share one MTLCommandQueue

The two `llama_context`s have separate `ggml_backend_sched`s and separate `ggml_metal_t`s,
but they resolve to the **same** queue. `ggml-metal-context.m:225-228`:

```objc
// TODO: would it be better to have one queue for the backend and one queue for the device?
//res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
```

The queue is created once per device (`ggml-metal-device.m:757`) and the device objects are
created exactly once at registry init, so both contexts get the same queue.

Two consequences, and the second one is a trap:

1. `ggml_metal_graph_compute` reserves ordering with `[cmd_buf enqueue]`
   (`ggml-metal-context.m:646`, `:673`). **Command buffers on one queue execute in enqueue
   order.** Submitting draft(N+1) before verify(N) finishes buys the CPU bubble only - the
   GPU still runs them back to back. You do not recover the 16.4 ms of occupancy.
2. `ggml_metal_synchronize` waits on its own `cmd_buf_last` (`:376-379`), but FIFO means a
   drafter buffer enqueued *behind* a verify buffer cannot start until the verify finishes.
   So `llama_synchronize(ctx_dft)` would block for **verify + draft**. **A prefetch with the
   ordering wrong makes the round slower than today.**

`n_cb` does not help - it splits one graph across command buffers on the same queue, in
order (`ggml-metal-context.m:782`).

True concurrency needs the commented-out `[TAG_QUEUE_PER_BACKEND]` line. The cross-queue
ordering primitive already exists: `MTLSharedEvent` as `ggml_metal_event`
(`ggml-metal-device.m:1125-1134`), exposed as `event_record`/`event_wait`
(`ggml-metal.cpp:592-593`, impls `ggml-metal-context.m:764-796`), already used for
cross-context copies (`:566`).

## 4. Temper the expected win before spending a week on it

~~"Worth up to ~16 ms/round on its own, and it needs no kernel change"~~ - **that framing,
from `mlx-cycle-capture.md`, is too optimistic and is corrected here.** Two reasons:

- `round-decomp-fused.md` establishes the drafter graph runs with **zero concurrency slack**
  ("Drafter (m2) = 19.76 ticks/round x 0.827 = 16.34 real ms - matches the 16.43 lattice-sync
  wall EXACTLY"), and the verify is bandwidth-bound big matmul. **Two GPU-saturating graphs
  co-scheduled on one AGX land somewhere between `max(a,b)` and `a+b`, not at `max`.**
- The certain part of the win is the CPU submit/readback bubble, which is only ~2.7 ms/round.

Budget the expected win at well under half of 16.4 ms until step 0 measures it.

## 5. Plan, cheapest first

**Step 0 - measure whether the queue split does anything (~5 lines, do this first).**
Un-comment `[TAG_QUEUE_PER_BACKEND]` (`ggml-metal-context.m:227`, restore the matching
`[queue release]` at `:361`), change nothing else, and see whether round wall moves. If it
does not, the split alone is inert and the win must come from real co-scheduling. This is the
prerequisite for everything below.

> ## STEP 3 NEEDS NO SPECULATION - and the premise of this whole note is weaker than it looked
>
> **Corrected 2026-08-22 (Johan's question: "might that not be why they pass one existing
> token to the next round?" - yes, and it changes the design).**
>
> Their prefetch inputs are **exact**, not guessed:
> `staged_first_next = best.posterior[acceptance_len : acceptance_len + 1]`
> (`spec_epoch.py:1818`). Because verify column 0 is an **already-known** token, the verify
> always commits at least one token regardless of acceptance
> (`commit_count = 1 + acceptance_len`), so the next draft's starting token is known the
> moment the verify lands. **There is no miss risk and no wasted draft.** So step 3's
> "speculate `id_last`, speculate `n_past=1`, ~47% acceptance means it misses most of the
> time" framing below is **wrong** - delete that risk from the design.
>
> We already build the batch the same way: `batch.add(id, sampled, ...)` then the drafts
> (`server-context.cpp:576-578`). Their block *b* = `1 known + (b-1) drafted`; our depth *d* =
> `1 known + d drafted`.
>
> **But the same reading also deflates the lever.** Their prefetch sits at the *end* of cycle
> N after acceptance, and is consumed by cycle N+1's verify - so `draft(N+1)` -> `verify(N+1)`
> is a hard serial GPU chain for them too. The async launch hides **host** work (token
> yielding), not a verify. Their drafter is on their critical path just like ours.
>
> Using their block-1 cycle as a no-draft baseline (block 1 drafts nothing: capacity
> `block_len - 1 = 0`, prefetch guarded `next_block_len > 1` at `spec_epoch.py:2480`):
> theirs spends **22.6 ms** on draft + 3 extra verify columns (95.0 - 72.4); ours spends
> **55.0 ms** (16.5 drafter + 38.5 for llama-bench 73.0 -> 111.5). **Even with a free drafter
> their extra columns cost at most 22.6 against our 38.5.** The gap is the width-2..4 verify
> kernels, not the pipelining. **Treat this note as the secondary lever and
> `mlx-cycle-capture.md`'s width-4 kernel finding as the primary one.**
>
> ## STEP 1 IS DEAD. Its ceiling is 0.5 ms, and the in-tree profiler already said so.
>
> Measured 2026-08-22, `run-async-inject.sh` arm `off-r1`, steady state at n_predict 600:
>
> ```
> spec-prof draft_call        avg = 17.18 ms
> dflash-prof draft:          noise decode avg  0.694 ms
> dflash-prof lattice sync:   avg 16.479 ms     <-- THE 16.4 ms IS HERE, in draft()
> dflash-prof process:        enc avg 0.871 ms, inject+sync avg 0.517 ms
> ```
>
> **Step 1 aimed at the wrong function.** It removes the sync inside `process()`, whose
> *entire* inject+sync cost is **0.517 ms** - 0.35% of a ~150 ms round, i.e. noise. The 16.5 ms
> everyone has been quoting is the **lattice sync in `draft()`**
> (`llama_get_embeddings_nextn(ctx_dft)`), and that one is a real GPU wait for the drafter's
> forward which cannot be dropped: the lattice values must be on the host to build the verify
> batch.
>
> Whole drafter cost decomposes as 16.48 (lattice sync) + 0.69 (noise decode) + 0.87 (encode)
> + 0.52 (inject+sync) = 18.6 ms, of which **only the 0.52 was ever reachable by step 1**.
>
> `DFLASH_ASYNC_INJECT` is kept, default off, because it is harmless and the flag documents
> the finding - but **do not expect it to move the needle, and do not spend more time on
> step 1.** The 16.5 ms is reachable only through step 3.
>
> **Method note for next time: grep the in-tree `dflash-prof` counters before scoping a
> lever.** They were already in the server log and would have killed this in one minute.
>
> **CORRECTIONS FROM ACTUALLY IMPLEMENTING THIS (2026-08-22, branch `drafter-pipelining`).**
> Two of the steps below were scoped wrong, and a third fact turned up that changes step 0's
> risk profile.
>
> **(a) ~~Step 1 cannot "fuse encode+inject into one graph".~~ DONE 2026-08-28 via a route
> this correction missed** (branch `drafter-fused-inject`, `perf/drafter-graph-count.md`
> item 4): the claim was true for fusing the two EXISTING graphs (the inp_g host read is
> forced by the two-call API), but moving the encoder fc+norm INTO the injection graph
> sidesteps it - `DFLASH_FUSED_INJECT=1` feeds raw encoder-width features to one
> llama_decode and g never exists on the host. Fused alone is flat; fused + ASYNC_INJECT
> makes process() submit-only, **+0.76% e2e, shas hold**. The original text follows for
> the API anatomy, which is still accurate. What is available at step-1 risk without the
> fusion is only **dropping the two redundant `llama_synchronize(ctx_dft)` calls**
> (after the inject decode in `process()`, and in `apply_window()`'s `flush()`). That is
> implemented behind **`DFLASH_ASYNC_INJECT=1`**, default off, harness
> `perf/run-async-inject.sh`. It does not remove the wait - it moves it to the next read of
> the drafter output - so the win is bounded by the CPU work between `process()` and the next
> `draft()`, i.e. the accept/sampling path. Expect small (measured +0.38% alone;
> its real value arrived with the fusion above).
>
> **(b) Step 2's reorder is impossible without step 3.** Verified in
> `server-context.cpp`: the draft runs at `:3083-3088` and the verify at `:3723-3733` **in
> the same round**, and the verify batch is built from the draft's tokens. So the drafter
> submit cannot move after the verify submit until there are *speculated* inputs to submit.
> Only the `_submit`/`_collect` API split is doable today, and on its own it is a pure
> refactor with no measurable effect. **Do it as part of step 3, not before it.**
>
> **(c) In steady state, the shared Metal queue is load-bearing for CORRECTNESS, not just a
> perf limiter.** `llama_encode`/`llama_decode` do *not* unconditionally synchronize on entry
> - both syncs are conditional on `!embd_seq.empty()` (`llama-context.cpp:1470`, `:1761`) and
> `sched_reserve()` early-returns unless `sched_need_reserve` (`:601-605`). Inside ggml,
> `ggml_backend_sched_graph_compute_async` only synchronizes when the graph is *reallocated*
> (`ggml-backend.cpp`, the "synchronize without ggml_backend_sched_synchronize" path). So
> back-to-back graphs on one context reuse the same galloc addresses with **no explicit
> sync** - what stops the new graph from clobbering the in-flight one is Metal's in-order
> execution on the shared queue. Splitting the queue per backend keeps same-context ordering
> (one queue per `ggml_metal_t`), so this is probably still safe, but it is no longer
> *obviously* safe, and step 0 should check output identity, not just wall time.

**Step 1 - remove the host bubble, no speculation (safe, ~2-3 ms).** The round pays *two*
GPU->host->GPU round trips through the drafter: the lattice readback (`speculative.cpp:1367`)
and the target-feature re-injection (`:1220` -> `:1243` -> `:1269`). On unified memory these
copies are gratuitous. Fuse the drafter's encode + inject + noise-block decode into one
submitted graph and defer the sync to the single point where the lattice is read, dropping
the intermediate syncs at `:1162` and `:1276`. No correctness risk.

**Step 2 - reorder so the drafter submit precedes the verify wait.** Split
`common_speculative_draft` into `_submit` / `_collect` halves; the natural cut is exactly
`speculative.cpp:1365` (everything before is submit, everything from
`llama_get_embeddings_nextn` on is collect). Then move the submit between
`llama_decode(ctx_tgt, ...)` (`server-context.cpp:3727`) and `llama_synchronize(ctx_tgt)`
(`:3731`).

**Step 3 - the actual speculation, and the real project.** A prefetched draft for cycle N+1
needs three things that only exist after verify N: `dp.id_last` (speculate the greedy argmax
of the verify's first column - the direct analogue of MLX's `staged_first_next`), `dp.n_past`
(speculate "1 accepted"; with ~47% acceptance a full-block guess misses most of the time),
and **the target hidden-state rows for the speculated prefix** - the one with no cheap
answer. Matching MLX means letting the drafter's graph read the target's
`res->get_layer_inp(il)` **in place** (same device, unified memory, so physically a pointer)
with an `event_wait` on the target's completion. That needs either a "foreign input tensor +
backend event" concept plumbed through `llm_graph_result` / `ggml_backend_sched`, or merging
the drafter into the target's sched as a second sub-graph.

## Note for whoever picks this up

A plain `-md` drafter (`common_speculative_impl_draft_simple`) has **no** target-feature
dependency and would be far easier to pipeline - but it is not the path being optimised, so
that ease is not available here.
