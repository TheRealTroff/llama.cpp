---
name: metal-divergence-hunt
description: Localize a wrong-output, NaN, zero-acceptance or GPU-hang defect in the fork's Metal backend to the node and the code gate behind it, without a commit bisect - a known-good anchor build under the current pick, pick-flag stripping, single-slot shape probes, op-level matmul dumps, the per-node activation-trace diff, and the sync guard for hangs. Use when a sha moves, drafts collapse, text goes garbage, or a server hangs the GPU under a configuration a one-slot mint never exercised (several slots, unusual batch shapes, a new KV type or file), or when "which kernel is wrong" has to be proven rather than guessed.
---

# Hunt a divergence in the Metal backend

The method that took the 2026-09-23 multi-slot defect from "speculation at 3 slots hangs the GPU
or emits garbage" to one wrong gate in `ggml-metal-ops.cpp` in an afternoon, with about forty
40-second trials and no commit bisect. Numbers and the full trial table: `perf/slot-mix.md`
("Resolution"). Each step below answers one question; stop as soon as the answer names the code.

Assume a fork checkout as cwd; the harness is `perf/run-slot-mix.sh` (multi-slot) or
`perf/run-prod-pick.sh` (single slot), both source `perf/pick.sh` and must run under **bash**.

## Step 0 - Make the machine and the symptom trustworthy

- `ps -axo pid,stat,command | grep llama-server`: a pid in state `E` holds GPU memory and a
  wedged context - reboot before any GPU trial. A killed server that is NOT `E` has released
  everything (SIGTERM may be ignored for 30 s; SIGKILL frees it; check `ioreg -r -c
  AGXAccelerator -d 1 | grep 'Alloc system memory'`).
- Arm the sync guard: `GGML_METAL_SYNC_TIMEOUT=12` (the harness's `SYNC_TIMEOUT`). A hung
  command buffer stalls WindowServer too and launchd kills the login session after 40 s; the
  guard dumps the graph in flight and SIGKILLs the server first. Fresh `PORT` per trial.
- Make garbage visible: `GGML_TOPK_STREAM=0` (the streaming top-k turns garbage drafter logits
  into `Invalid input batch` rejections - a symptom of a symptom); record each stream's TEXT,
  not only its sha (`slot-mix-driver.py` writes `text`). Sixteen tokens in sixteen characters,
  a repeated syllable, or a first token that is `,` on every slot is garbage; a first-token
  flip on `01-code-explain` alone is the known tie, not a bug (check the top-2 margin first).
- Determinism: rerun the reproducer; then `GGML_METAL_CONCURRENCY_DISABLE=1`. Same shas =
  deterministic, not a race - the rest of this skill applies. Shas that move between runs =
  the race skills (`perf/small-op-fusion.md`, the alias guard) instead.
- `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` and equal-vs-unequal prompt lengths change the graph's
  memory layout: a symptom that flips between garbage and a hang under them is a bad ADDRESS
  (an unreserved scratch, an over-read), not arithmetic.

## Step 1 - Find a good anchor under the CURRENT pick

Build an older prod commit in its own worktree and run it under today's pick env and prompts:
`BIN=<old tree>/build/bin bash perf/run-slot-mix.sh ...` (the old tree has no `pick.sh`; the
harness takes this tree's). A clean anchor turns a "bug" into a "regression with a date" and
bounds every later step. The Sep 6 build (0623a06a5) was clean; the Sep 9 build hung.

Do NOT reach for `git bisect run` next: its steps run guard-less builds, one hang costs 30
minutes, and the answer it gives (a merge commit) is coarser than Step 2's (a flag). Keep it
as the last resort, with a watchdog step (build, wait for the GPU to be idle, 300 s limit,
`exit 125` on timeout) - the pattern is in `perf/slot-mix.md`.

## Step 2 - Strip the pick one flag at a time

`UNSET_ENV="GGML_MM_F16B ..." bash perf/run-slot-mix.sh ...` drops names from the pick env.
Presence-based flags cannot be turned off with `=0` (README trap 2026-09-02), so unsetting is
the only clean A/B. Strip the whole group of suspects first (the flags that landed after the
anchor date, or that route the ops the symptom implicates), then one at a time. Read the
result on the TEXT: a flag whose removal changes the shas but keeps the garbage is numerics
noise (acc-half), not the cause; the cause is the one whose removal restores the anchor's shas.

### Step 2b - A new lever ships with one kill switch per layer it touches

A lever that adds a graph input AND a kernel path (the per-stream KV extent, 2026-09-24: an I32
input on the FA op, kernels that stop at it) gets two env switches before its first run: one
that leaves the graph exactly as it was (`LLAMA_ATTN_KV_LEN=0`, the input is not created) and
one that keeps the graph but makes the kernels ignore it (`GGML_FA_KVLEN=0`). When the first
smoke moved every sha, the kernel switch alone restored the reference: the graph change was
harmless, the kernels read an UNSET buffer - `llm_graph_input_mem_hybrid::set_input` sets the
attention inputs itself instead of delegating to `llm_graph_input_attn_kv::set_input`, so a new
attention input has to be set in BOTH (a `-lv 5` DEBUG line in the setter, grep count 0, named
it). One run split the hypothesis space; without the switches it is a bisect of the lever.

### Step 2c - A mix-phase sha only counts when it recurs

`run-slot-mix.sh`'s mix phase composes each round from whichever executors' next requests have
landed, so its ubatch sequence (and with it the width class each marginal text is verified
under) is a race against the round period. At 32K the prod binary forked from its own rerun on
2 of 7 mix texts and the "new" binary matched itself three times; at 96K (slower rounds) both
matched on all 14. Compare binaries on the deterministic phases (executors alone, solo), and
read a mix fork as evidence only when it recurs on the same binary (memory `owner-race-evidence-bar`).

## Step 3 - Shape probes on one slot

Any prefill column count N can be reproduced on ONE slot with `EXTRA_ARGS="-ub N"`. If the
single-slot text is unchanged at every N the multi-slot graph produces (12, 19, 20, ...),
the kernel at that width is fine and the defect is an interaction between ops or a layout
problem. Equal-length prompts across slots (`EXEC_PROMPTS` with the same file three times)
remove the unequal-split ubatches and are the second axis.

## Step 4 - Op-level: is this matmul route wrong?

`LLAMA_MM_DUMP=<dir> LLAMA_MM_DUMP_NT=<N> LLAMA_MM_DUMP_TYPE=q4_0_soa LLAMA_MM_DUMP_N=12` dumps
the first 12 matching MUL_MAT nodes (a, b, dst) from the eval callback; run it under both
configurations and `references/mm-dump-diff.py dirA dirB`. `b` differs = the corruption
entered upstream of this op; `dst` differs with `b` equal = this route's kernel. Byte-identical
everywhere = the route is innocent, go to Step 5. (`perf/mm-dump-compare.py` is the f64
reference form for pricing one route's numerics; this is the two-route A/B.)

### Step 4b - Two routes disagree and neither is "the bug": which one is right?

A pairwise KLD between two kernel routes (0.002 here) says how far apart they are, not which one is
wrong. Answer that with a reference that is exact by construction, at two levels (per-slot-ctx.md
2026-09-24, the width-2 GQA route): (1) op level - `LLAMA_FA_DUMP=<dir>` (with `_NT`, `_NS`, `_KVMIN`
selectors for the token count, stream count and extent) dumps one FA node's inputs and output under
each route; `perf/fa-dump-ref.py` recomputes it in float64 from the dumped Turbo4 blocks and reports
each route's relative RMS from exact (old 3.4e-4, new 1.9e-4: the NEW route was the accurate one);
(2) logit level - `run-quant-kld.sh` with `REF=<the same model> REF_KV=f16` and each route as a test
arm under `KV=turbo4` (same `-b W -ub W` on both arms) gives each route's distance from the exact
cache; the direction must agree with (1). Only then is the move a numerics class with a sign, not a
defect. Do not price a decode route with the prefill-shaped KLD line (README rule).

## Step 5 - Node-level: the first divergent computed op

`LLAMA_TRACE_DUMP=<dir> LLAMA_TRACE_MAX_MB=64` under both configurations (about 2 minutes per
run for a 400-token workload: the callback syncs every node), then
`references/trace-nodediff.py dirA dirB [first_graph] [last_graph]`. Rules the tool encodes:

- skip VIEW/RESHAPE/TRANSPOSE/PERMUTE: they hash memory nobody wrote yet, which differs
  between layouts (the first false alarm of the hunt);
- skip GATED_DELTA_NET outputs by default: the tensor carries unwritten padding behind the
  state and kept-input regions, so its hash differs while every consumer view is identical
  (the second false alarm); `NODEDIFF_SKIP_OPS=` to include it;
- `graphs.tsv` tells which graph is which (n_tokens, n_seqs, seq ids, positions): read the
  prefill graphs of the failing request, not the warm-up.

The first differing computed op is the victim. The culprit is usually the op right BEFORE it
in encode order whose writes are not part of any node - a scratch behind dst, a twin, a
fused kernel's carry - and the `data` column of the idx gives the addresses to check. Then
open the two places that decide such a route: the alloc-size query
(`ggml_metal_op_mul_mat_extra_src1f16` via `ggml-metal.cpp` `get_alloc_size`) and the encode
route (`ggml_metal_op_mul_mat` / `_impl`), and check they see the SAME tensor shape.

## Step 6 - Reading a hang

The guard dump lists every node of the graph in flight and each command buffer's status. A
buffer that is `scheduled` with GPU start 0.000 while the GPU is cool and idle never launched:
that is a bad address the driver refused (an unreserved scratch reaching unmapped memory, an
over-read), not a spinning kernel. Hundreds of zero-row nodes (`GET_ROWS ne=[.., 0]`, the
replay tensors at n_rep = 0) are normal and skipped by the encoder. A live hung process (no
guard) sits in `waitUntilCompleted` (`sample <pid>`), still answers `/health`, and reads 100%
Device Utilization - treat it as a specimen, not an emergency, unless it goes to state `E`.

## The rule that came out of it

**A route that takes scratch behind dst must be decided by ONE function, on the same
(folded) shape, at alloc time and at encode time.** The mv side learned it on 2026-09-04
(7415209e2: the [K,1,S] fold and the f16-y scratch); the mm side on 2026-09-23 (6ed433a2f: the
[K,T,S] fold and the GGML_MM_F16B scratch). Single-sequence graphs never fold, so a one-slot
mint cannot see this class: every gate of a scratch-taking route needs a multi-slot arm.

## After the fix

Gate the configurations that showed each symptom (unequal prompts, equal prompts, the KV type
that hung, the full pick with the controller and top-k on, and one slot as the unchanged
control), write the resolution at the TOP of the perf note with the evidence table and strike
the stale status, add the arm to the mint, and update this skill.
