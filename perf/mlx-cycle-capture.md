# What does dflash_mlx's block-4 cycle actually compute?

Status: **ANSWERED 2026-08-22. Hypotheses 1 and 2 confirmed from source; the capture was then
taken and dumped headlessly, and it confirms both plus the kernel mechanism. Hypothesis 3
(the microbench parity claim) is the only piece left, and it is now the open stub.**

Artifacts: `perf/capture-mlx-cycle.py` (capture), `perf/gputrace-dump.py` (headless
`.gputrace` -> text, no Xcode GUI), trace at `/tmp/dflash-b4.gputrace` (17 GB, volatile),
dump at `/tmp/dflash-b4-trace.txt` (95 MB).

Opened 2026-08-22 by `block4-shelf-probe.md`, which measured their block-4 cycle at
**95.00 ms** - a drafter *and* a verify - against our five-column verify pass alone at
**126.3 ms** in-graph. The microbench comparison on record says we are at parity with MLX at
n=5 (ours 1.81x, theirs 1.74x). Both cannot be true. This task settles which.

---

## What the source settled (2026-08-22, no capture required)

### 1. Their block *b* verifies *b* columns. Ours verifies *d+1*. Every cross-framework depth comparison on record is off by one.

`engine/config.py:21-28` and `spec_epoch.py:2247-2257`:

```python
def resolve_verify_len_cap(runtime_config, block_tokens):
    requested = int(getattr(runtime_config, "verify_len_cap", 0) or 0)
    if requested <= 0:
        return int(block_tokens)          # default 0 => cap == block size, no clamp
    return max(1, min(int(block_tokens), requested))

def verify_token_count_for_block(block_len, verify_len_cap):
    return max(1, min(int(block_len), int(verify_len_cap)))
```

```python
verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)   # == 4
verify_token_ids = mx.concatenate(
    [current_staged_first[:1], drafted[: verify_token_count - 1]], axis=0,     # 1 + 3
)
```

So **block 4 = 4 verify columns** (1 staged/current + 3 drafted), not 5. Their `block_tokens`
*includes* the current token; draft capacity is `block_len - 1` (`spec_epoch.py:2052`,
`_draft_capacity` at :373). Ours is the other convention - "depth d verifies d+1 columns"
(`slope-sweep.md:13`). The bonus token is not an extra column for them: it falls out of the
last column's logits, `commit_count = 1 + acceptance_len` (`spec_epoch.py:2385`), so a 4-wide
pass commits up to 4 tokens.

Nothing is algorithmically skipped inside that 4-wide pass, so this is a width correction and
not a "they cheat" finding: `forward_with_hidden_capture` (`target_qwen_gdn.py:840`) runs
every layer, and the lm_head is a **full-vocab** projection over **all** columns
(`target_qwen_gdn.py:886`, `logits_from_hidden` at :775). `logits_last_only` exists but
`verify_block` never passes it. There is no gathered/candidate-narrowed lm_head. Their
`verify_qmm.py` / `verify_linear.py` are small-M quantized GEMM kernels for the q/k/v/o and
mlp projections only - `is_verify_eligible` rejects `N >= 100_000` (`verify_linear.py:74`),
so a ~151k vocab never routes there.

**Corrections this forces on notes already written** (originals struck, not deleted):

- `verify-slope-close.md`: ~~"Their block 4 -> 5 step costs +48.3 ms; our own width 5 -> 6
  step costs +1.9 ms (llama-bench 119.0 -> 120.9)"~~ - their block 4 -> 5 is a **width 4 -> 5**
  step, so the number to compare it against is **our width 4 -> 5 step, +7.5 ms**
  (llama-bench 111.5 -> 119.0). The asymmetry is still enormous (+42.3 vs +7.5) and the
  conclusion "one of the two curves has a cliff and it is not ours" survives - but **their
  cliff sits between width 4 and 5, not 5 and 6.**
- `verify-slope-close.md`: ~~"their block-4 cycle buys 4 extra verify columns for 19.5 ms
  over their block-1 cycle, where we pay 46.0 ms for the same four (73.0 -> 119.0)"~~ - it
  buys **3** extra columns, and the like-for-like figure for us is **38.5 ms** (73.0 -> 111.5).
- `block4-shelf-probe.md`'s depth table compares their block *b* against our depth *b*, i.e.
  their width *b* against our width *b+1*. Matched by width instead:

| verify width | theirs, ms/cycle (draft+verify) | ours, ms/round | ours, llama-bench verify only |
|---|--:|--:|--:|
| 4 | 95.00 (their block 4) | **not measured - our depth 3** | 111.5 |
| 5 | 137.26 (their block 5) | 144.9 (our depth 4) | 119.0 |

  So "1.53x at depth 4" was their width-4 against our width-5. At matched width 5 they are
  **1.06x** (137.26 vs 144.9) - which lands in the same place as the old "1.07x at depth 5"
  by coincidence, since our depth-4 and depth-5 rounds are 2.6 ms apart.

**The contradiction shrinks but does not vanish.** Corrected, their 95.00 ms buys a drafter
plus a 4-wide verify, against our 4-wide verify *alone* at 111.5 ms on llama-bench (~118 ms
scaled to in-graph, using the 126.3/119.0 in-graph-vs-bench ratio at width 5). They are still
ahead by ~23 ms *and* getting their drafter for free. Which leads to:

### 2. Their drafter is launched async. But it overlaps HOST work, not the verify.

> **CORRECTED 2026-08-22, same session.** The heading below used to read "their drafter
> overlaps the next verify", and that overstates it. The prefetch at `spec_epoch.py:2469` sits
> at the **end of cycle N's loop body, after acceptance is computed**, and its result is
> consumed by cycle **N+1's** verify at the top of the next iteration. So `draft(N+1)` ->
> `verify(N+1)` is still a hard serial chain on the GPU; what the async launch hides is the
> **host** work in between (token yielding, `spec_epoch.py:2513-2548`), not a verify.
> Their drafter GPU time is on their critical path too. See "what this actually implies"
> below - the corrected arithmetic points at their *kernels*, not at overlap.

`spec_epoch.py:2469-2495`, immediately after a cycle commits:

```python
if not profile_cycles:
    ...
    next_drafted, next_probs, next_indices = _draft_for_block(
        staged_first_next, next_block_len,
        feature_store.require_current_hidden(),
        async_launch=True,
    )
    state.prefetched_draft = {...}
```

The **next** cycle's draft is launched with `async_launch=True` before the current cycle's
tokens are even yielded, so under MLX lazy eval those draft kernels are in flight while the
next verify is being built. Ours is 16.4 ms of fully serialized drafter GPU wait
(`round-decomp-fused.md`), ~11% of the round.

The `if not profile_cycles:` gate is the important half: **turning on their per-cycle
profiling disables the prefetch**, and their `phase_timings_us` therefore can never show this
overlap. That is a second, independent way their counters mislead, on top of the
submission-vs-execution problem. It also means any future capture must run with
`profile_cycles` OFF or it destroys the very structure being measured.

This is hypothesis 2, confirmed without a trace, and it is engine structure.

**What this actually implies, with the overlap claim corrected.** Their prefetch inputs are
**exact, not speculated**: `staged_first_next = best.posterior[acceptance_len : acceptance_len + 1]`
(`spec_epoch.py:1818`) is the target's own output at the column after the last accepted draft
token - the guaranteed bonus token. Because column 0 of the verify is an **already-known**
token, the verify always commits at least one token regardless of acceptance
(`commit_count = 1 + acceptance_len`), so the next draft's starting token is known the instant
the verify lands. **We do the identical thing** - `batch.add(id, sampled, ...)` then the drafts
(`server-context.cpp:576-578`). Their block *b* is `1 known + (b-1) drafted`; our depth *d* is
`1 known + d drafted`. Same shape, different label - which *is* the off-by-one in finding 1.

So the drafter is not where their edge is. The arithmetic, using their own block-1 cycle as
the no-draft baseline (block 1 drafts nothing: `_draft_capacity = block_len - 1 = 0`, and the
prefetch is guarded `next_block_len > 1` at `spec_epoch.py:2480`):

| | theirs | ours |
|---|--:|--:|
| verify-only baseline, width 1 | 72.4 ms (block 1) | 73.0 ms (llama-bench b1) |
| draft + 3 extra verify columns | **22.6 ms** | **55.0 ms** (16.5 drafter + 38.5 for 73.0->111.5) |

**Even if their drafter were entirely free, their three extra verify columns cost at most
22.6 ms against our 38.5.** The gap is the width-2..4 verify kernels, which is exactly what
the capture found (`custom_kernel_verify_m4_*`). Pipelining is a real but secondary lever.

### 3. Hypothesis 3 is now the live one

With widths corrected, the `1.81x / 1.74x` microbench parity claim needs re-reading before it
can be trusted: those slopes are quoted as "n=5" but `verify-slope-close.md:27` derives our
1.813x from **b1 72.053 vs N=7 130.637**, i.e. a width-7 ratio. If their 1.74x is a width-5
ratio, the two slopes were never measured at the same width and the "parity" claim compares
different points on two curves. **Not yet verified - this is the open stub below.**

---

## What the capture settled (2026-08-22, 3 steady-state block-4 cycles)

Captured with `perf/capture-mlx-cycle.py`, dumped with `perf/gputrace-dump.py`. Clean window:
`copyspec_hits=0`, `adaptive_block_cycles=0` (block genuinely pinned), acceptance 0.6739,
cycles 12-15. Counts below are references to a resolved pipeline object across all 3 cycles,
so treat them as a **proxy for dispatch count** (one dispatch may contribute more than one
reference), and note `kp2`/`kp4` are k-partitioned, so one logical matmul can be several
dispatches.

### Their quantized matmul is NOT `qmv_fast` and NOT `qmm_t`. It is a custom M=4 kernel.

| kernel | refs (3 cycles) |
|---|--:|
| `custom_kernel_verify_m4_ksplit_np_kp2_gs64_bf16` | **1180** |
| `custom_kernel_verify_m4_ksplit_np_kp4_gs64_bf16` | **454** |
| `affine_dequantize_bfloat16_t_gs_64_b_4` | 24 |
| `affine_qmv_wide_..._nv_3_kl_8_batch_0` | 17 |
| `affine_qmv_fast_..._batch_0` | 11 |
| `affine_qmv_wide_..._nv_4_kl_8_batch_0` | 3 |

`qmm` appears **3 times in 95 MB of trace text**. The stock `qmv` family totals 31 refs
against 1634 for the custom kernel. So the whole premise of "which of `qmv_fast` / `qmm_t`
does it dispatch, and where does it switch" is answered: **neither - their verify runs on
`verify_qmm.py`'s bespoke kernel**, the one `is_verify_eligible` routes everything except
lm_head to (`verify_linear.py:74`, `N >= 100_000` rejected). The stock `qmv` refs are
consistent with the lm_head and a few odd shapes falling back.

**`m4` in the kernel name is independent confirmation of the width-4 finding above** - their
verify kernel is specialised for M=4, compiled per group-size (`gs64`) and k-partitioning
(`kp2`/`kp4`), in bf16.

1634 refs / 3 cycles = ~545 per cycle, against our 496 MUL_MAT dispatches per full target
pass (`slope-sweep.md`). Same order - so **hypothesis 1's dispatch-count test independently
agrees with the source reading: they run a full pass, not a truncated one.**

### The rest of the cycle is vector-family too

`sdpa_vector_2pass_2_bfloat16_t_128` / `_256` (48 refs) - they use the **vector** SDPA path at
width 4, not a batched attention kernel. Linear-attention/GDN is
`custom_kernel_gated_delta_tape` (144) with `custom_kernel_tape_replay` (96) for rollback.
Everything is `bfloat16` and `gs_64`.

### Why this matters, and where it points

Their cheap shelf is a **custom M=4 quantized kernel plus vector SDPA at width 4** - exactly
the width where our own routing is weakest. Per `slope-sweep.md`, `mv_nc_route` is gated to
`ne11 <= 2` and skinny starts at 5, so **widths 3 and 4 fall through to `ext`, covered well by
neither** - the N=3 step alone costs +27.7 ms, the largest inside the window. So their
operating point sits precisely in our worst-covered region. That is the strongest argument yet
for the "flatten widths 2-5" programme, and it is no longer speculative: we now know what a
good width-4 kernel looks like, because we can read theirs.

---

## The three candidate explanations (original framing, kept for provenance)

1. ~~**Their block-4 cycle does not run a full 5-column forward pass.**~~ **CONFIRMED, with a
   twist**: it runs a full *4*-column pass. Full stack, full-vocab lm_head, nothing skipped -
   it is one column narrower than we assumed, not cheaper per column.
2. ~~**Their drafter overlaps the verify on the GPU timeline.**~~ **CONFIRMED** at
   `spec_epoch.py:2490` (`async_launch=True`).
3. **The microbench parity claim was measured wrongly.** Still open - see above.

## Why a GPU capture is the right instrument

Everything in this investigation so far has been inferred from counters, and **both sides'
counters are known to measure the wrong thing in opposite directions**: our
`GGML_METAL_PROFILE` creates one encoder per op, inflating CPU encode 6-8x and serializing
dispatch that is normally concurrent (which is why small-ne01 measured 2.3x per-call and
0.0% e2e); their `phase_timings_us` measures submission rather than execution under MLX lazy
eval, summing to 8-11% of generation wall. A GPU capture reads the device timeline, so it is
immune to both failure modes by construction. It also directly answers a question no counter
can: **what overlaps what**.

Both of those questions have now been answered from source instead. A capture is still the
only way to get **per-kernel attribution** - which kernel eats the 23 ms at width 4, and
whether `qmv_fast` or `qmm_t` is running there.

## How, without sudo - VALIDATED 2026-08-22

- **Their side: works.** Harness committed as **`perf/capture-mlx-cycle.py`**, config matched
  to `run-block4-shelf.sh` (same model/draft/prompt/`w4:gs64`/no-chat-template/no-EOS/
  `verify_mode=dflash`/block 4). Verified end to end: model load 2.6 s, prefill 68.0 s
  (matches the recorded ~67 s), capture opens cleanly at cycle 12 with `copyspec_hits=0`.
  - The API is **`mx.metal.start_capture` / `mx.metal.stop_capture`**, not top-level
    `mx.start_capture` as this note originally said.
  - `MTL_CAPTURE_ENABLED=1` must be set **before** mlx initialises Metal.
  - Confirmed by direct test that capture **does not need developer mode**.
- **Our side:** already wired, and the env var is a **countdown, not a flag**:
  `GGML_METAL_CAPTURE_COMPUTE=N` captures the **Nth** graph compute (read at
  `ggml-metal-context.m:293`, decremented at :604, fires at :607, writes
  `/tmp/perf-metal-<pid>.gputrace` at :618). That is how you skip prefill.
- **Instruments is confirmed blocked**: `DevToolsSecurity -status` reports developer mode
  disabled, so `xctrace` needs a one-time `sudo DevToolsSecurity -enable` from Johan. Not
  required for either in-process path.

Capture both sides at **matched width** - **their block 4 against our depth 3**, not our
depth 4. That off-by-one is the whole point of finding 1 above.

## Cautions

- **Capture distorts timing.** Use it for structure - kernel names, dispatch counts, shapes,
  and overlap - not for headline throughput. Same discipline as `GGML_METAL_PROFILE`.
- **`profile_cycles` must stay OFF** or the drafter prefetch at `spec_epoch.py:2469`
  disappears and the overlap cannot be seen. This also rules out gating the capture on
  `CycleCompleteEvent`, which only fires in profiling mode.
- Do not run a capture while a benchmark is in flight; it perturbs the numbers.
- **A 27B trace is ~16 GB and takes many minutes**, because the capture serializes all
  53,447 weight `MTLBuffer`s. Watch free disk. Three cycles is already more than enough.
- ~~**The trace can only be read in the Xcode GUI.**~~ **WRONG - retracted 2026-08-22.**
  `strings` does not work on the bundle, but Xcode's archive reader is just a framework:
  **`perf/gputrace-dump.py`** dumps a `.gputrace` to text headlessly, no GUI and no developer
  mode. It drives three private Xcode frameworks through the ObjC runtime with ctypes:
  `DYCaptureArchive` (GPUTools) opens the bundle, `DYFunctionTracer` (GPUToolsCore) formats
  calls, `DYCaptureArchiveTraceToFile` (GPUToolsServices) walks the stream. Needs
  `DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks` and a non-SIP
  python (the omlx venv works; `/usr/bin/python3` has `DYLD_*` stripped). **Gotcha:
  `-performTrace` alone writes an empty file - the visitor must be driven via
  `-visitCaptureArchive:`, which returns false even on a good dump, so check the output file
  and not the return value.** The 17 GB / 906,021-file trace dumped to 95 MB of text in about
  6 minutes. Metal API call names come out as `(null)` (the API symbol table is not
  registered) but **kernel names and arguments are intact**, which is what matters.
- ~~Their kernels to look for: **`qmv_fast`** and **`qmm_t`**.~~ **ANSWERED - and it is
  neither.** See the capture results below.

## What each outcome implies

- ~~**Full pass, no overlap, cheap kernels**~~ -> superseded: the pass is full but 4 wide,
  and the overlap exists.
- **Drafter overlaps verify** -> CONFIRMED. Pipeline our drafter under the verify. ~~Worth up
  to ~16 ms/round on its own, and it needs no kernel change.~~ **CORRECTED 2026-08-22 after
  scoping (`drafter-pipelining.md`, on branch `drafter-pipelining`): the 16.4 ms is
  optimistic and it is NOT free.** Both
  `llama_context`s share one `MTLCommandQueue` (`ggml-metal-context.m:227-228`, the
  `[TAG_QUEUE_PER_BACKEND]` line is commented out), so command buffers execute in enqueue
  order and a naive prefetch **makes the round slower** - `llama_synchronize(ctx_dft)` would
  block for verify + draft. The queue must be split first. And our drafter is conditioned on
  the *target's* hidden states (`speculative.cpp:1220`), which ggml round-trips through host
  memory, where MLX passes a lazy array on-device - so prefetching means speculating on
  feature rows, not just on the next token. The certain part of the win is the ~2.7 ms CPU
  bubble. **Still the most promising lever, but budget well under half of 16.4 ms until
  step 0 of `drafter-pipelining.md` (branch `drafter-pipelining`) measures it.**
- **Not a full pass** -> it is a full pass, one column narrower than assumed.

## Open stubs this leaves

1. **Measure our depth 3 (width 4) round cost.** Until that exists there is no like-for-like
   number against their 95.00 ms block-4 cycle, and the headline "1.53x" is a width-4-vs-
   width-5 comparison. Cheapest experiment on the board - it is one more arm on
   `run-slope-sweep.sh`.
2. **Re-derive the 1.81x / 1.74x slopes at a stated, matched width** (hypothesis 3). Ours is
   a width-7 ratio; theirs is quoted at "n=5" with the convention unstated, and their
   convention is now known to differ from ours by one. Until this is redone, treat
   "kernel parity" as unverified rather than established.
3. **Pipeline our drafter under the verify** (from finding 2). Scoped out into its own stub,
   **`drafter-pipelining.md`, which lives on branch `drafter-pipelining`, not here** -
   engine work, but blocked on splitting the shared Metal
   command queue, and worth less than the 16.4 ms headline. Start at its step 0.
