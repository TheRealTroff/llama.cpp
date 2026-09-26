# Prompt caching under per-slot context sizes: the gate (2026-09-25)

**Status 2026-09-26: THE FOUR BEHAVIOURS ARE CODE DEFAULTS on the branch (owner: "2, while accepting loss of acceptance"):
save-on-overwrite at 64 decoded tokens (`LLAMA_CACHE_SAVE_TAIL=0` = upstream's rule), checkpoint-aware cache scoring
on a recurrent/SWA context (`LLAMA_CACHE_LOAD_CKPT=0` = raw prefix), no drafter blob in checkpoints when the drafter
can truncate (`LLAMA_CKPT_NO_DFT=0` keeps it; the deep-restore acceptance dip is accepted), and no idle-slot saves in
split mode (`LLAMA_CACHE_IDLE_SPLIT=1` restores the copies; unchanged under `--kv-unified`). Gates on the default
binary: the cache gate with no flags (must equal arm D) + the multi-slot gate short and long arms on both lines -
see "Defaults" at the end. Earlier status lines below stand as the record.**

**Status: VERIFIED on the q4 line, prod `ed2215f3e` (binary `dc02ff4d6`, no code change between them), four slots of
4K / 8K / 16K / 32K tokens (`--ctx-seq-sizes 4096,8192,16384,32768`), the Turbo4 pick at fixed DFlash depth 3.
Every layer of the server's prompt cache works with the size classes, and the cached path reproduces the
uncached path: 16 of 16 sequential requests land in their class, 10 of 10 continuations reuse >= 99% of their
prompt, all 16 cached texts equal the uncached reference except one 48th-token tie, the host-RAM round trip is
byte-identical. One behavioural finding (not a bug): a RAM-cache entry loaded on a partial prefix match is
destroyed by the recurrent reset that follows it on this model.** Tool: `run-ctx-class-cache-gate.sh` +
`ctx-class-cache-driver.py`, TAG `ctxcache-sep25`, results and both `-lv 5` server logs under
`kvquant-experiments/results/ctxcache-sep25.*`. Branch `exp/ctx-class-cache-gate`.

## What the prompt cache is on Qwen3.8 (hybrid GDN + attention)

Three layers, in `tools/server/server-context.cpp` (the request path) and `server-task.cpp` (`server_prompt_cache`):

1. **Attention-KV prefix reuse in the slot.** The longest common prefix of the slot's token list and the new
   prompt is kept (`n_past`), the tail is prefilled. `--cache-reuse` (KV shifting of later chunks) is refused on
   this model at startup: the hybrid memory cannot shift.
2. **Recurrent checkpoints.** The GDN state is one snapshot per sequence and can only be rolled back through the
   `n_rs_seq` ring (= the draft depth, 3 here; it exists for draft rejection). So the server stores checkpoints
   with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` - for the hybrid memory that is the recurrent state only, **170-190
   MiB each on this model**, in host memory - at the start of user messages (spaced `--checkpoint-min-step`,
   8192, apart; always at the last one) and at `4 + n_ubatch` and `4` tokens before the end of the prompt. The
   hybrid memory reports the recurrent state's position as the sequence's `pos_min`; when that is at or past the
   resume point the newest checkpoint before the divergence is restored and `n_past` drops to its token count;
   with none, "forcing full prompt re-processing" and `n_past = 0`. Checkpoints past the resume point are erased.
3. **The host-RAM prompt cache** (`--cache-ram`, default 8192 MiB; `--cache-idle-slots` on). A slot about to be
   repurposed (LRU pick, or a similarity pick that would keep < 50% of its context) serializes its whole
   sequence state (target KV + recurrent + drafter + checkpoints) to a list; every idle slot is re-saved on
   every launch. The incoming prompt then takes the entry that beats the slot on both similarity and keep
   fraction (entries under 0.25 keep are never taken); loading removes the entry from the list.

On top: slot selection by prefix similarity (> 0.1 of the prompt) then LRU, filtered by the size classes - a
task goes only to a slot that holds it, and the LRU fallback stays in the smallest idle class that fits.

A detail that matters for reading `cache_n`: the last sampled token of a finished request is never decoded, so
a continuation whose prefix ends there needs neither the ring nor a checkpoint (A2/B2/C2/D2 below show no
restore line: `cache_n` = the tokens decoded so far). A continuation that diverges *inside* the previous request's
tail (T3, a different follow-up) restores the last-user-message checkpoint that the previous prefill created.

## The plan (`ctx-class-cache-driver.py`)

Five multi-turn chat streams (Qwen3 template, thinking off, greedy, 48 tokens): A and A' (~2.5K tokens, the 4K
class), B (~5K, 8K), C (~10K, 16K), D (~20K, 32K); documents are distinct slices of `longprompt-96k`.

| turn | what it exercises |
|---|---|
| T1 | fresh prefill, class routing |
| T2 | continuation after the model's own reply (KV prefix reuse; the recurrent state is already at the resume point) |
| T3 | a different follow-up in place of T2's: the last-user-message checkpoint created during T2's prefill |
| A3 | runs after A' has displaced A from slot 0: the RAM prompt cache restores A, then T3's checkpoint |
| T4 (A, B) | a word changed 40% into the first user message: no checkpoint precedes it = full re-processing |
| T5 (A-D) | four continuations of T3 fired concurrently: routing under contention, B from the RAM cache |

Server 2 (control) replays A1-A3 cached without displacement (A3 in slot vs A3 RAM-restored) and then every
sequential prompt string with `cache_prompt = false` - the uncached reference, same slot routing.

## Result (`ctxcache-sep25`, cached server 409 s, control server 1014 s, no error, no sync-guard)

| req | slot | cache_n / total | acc% | vs uncached | max abs dlogprob (token 0) | mechanism (server log) |
|---|---|---|---|---|---|---|
| A1 / A'1 / B1 / C1 / D1 | 0 / 0 / 1 / 2 / 3 | 0 | 41-66 | identical | 0 | LRU into the class |
| A2 / A'2 | 0 | 2530 / 2558, 2532 / 2560 | 67 / 61 | identical | 3.2e-2, 4.7e-3 | similarity 0.989; no restore needed |
| B2 | 1 | 5011 / 5039 | 58 | **47/48 tokens, tie at token 47 (p 0.18 vs 0.16)** | 1.2e-1 (p 0.27) | as above |
| C2 / D2 | 2 / 3 | 9945 / 9973, 19957 / 19985 | 100 | identical | 1.2e-3, 3.7e-2 | as above |
| A3 | 0 | 2530 / 2561 | 100 | identical, **= A3 in slot byte for byte** | 5.7e-4 | LRU; RAM entry f_keep 0.990; checkpoint 2530 |
| A'3 | 0 | 2532 / 2563 | 90 | identical | 8.5e-3 | LRU; RAM entry 0.985; checkpoint 2532 |
| B3 / C3 / D3 | 1 / 2 / 3 | 5011, 9945, 19957 | 82-94 | identical | 2.8e-3, 5.6e-3, 3.2e-4 | checkpoint at the user-message start |
| A4 / B4 | 0 / 1 | 0 | 60 / 38 | identical | 0 | full re-processing (no checkpoint before the edit) |
| A5 (concurrent) | 0 | 0 / 2639 | 100 | - | - | reset: A's RAM entry was consumed by A4 (below) |
| B5 / C5 / D5 (concurrent) | 1 / 2 / 3 | 5089, 10023, 20035 | 78-100 | - | - | B from the RAM cache (f_keep 1.000), C/D in slot |

Reading the numerics column: every cached-vs-uncached delta sits on the **first generated token** (the one whose
logits come out of the tail prefill: 28-31 tokens through the small-batch kernels vs the same tokens inside a
512-wide `mul_mm` in the reference) and is largest where the top-1 probability is lowest; from token 1 on both
arms run the same decode kernels from states that agree to 1e-3 in log-probability. That is the prefill-shape
numerics class, not a state error - a wrong recurrent state would not agree to 3e-4 on D3's 48 tokens. B2's fork
is a 48th-token near-tie (`check-logit-margin-before-hunting`). The RAM round trip (A3 vs A3c: identical text,
zero logprob delta) is exact save/restore.

Costs seen: a 20K-token entry is 492 MiB (Turbo4 KV ~16.5 KB/token + checkpoints); nine saved states filled 7.3 GB
of the 8 GB default during this 5-stream run, so a busier mix evicts by age. Each recurrent checkpoint is
170-190 MiB of host memory, up to 32 per slot (`--ctx-checkpoints`), spaced 8192 tokens apart except near the
prompt end.

## The finding: a partial-match RAM load is wasted, and the entry with it (hybrid models)

A4 (the edited document) selected slot 0 by LRU (it held A'), so the server saved A' and looked for a better
entry: A's T3 state matched 42% of the prompt (`f_keep 0.406 >= 0.25`), was loaded (a ~200 MiB restore), and was
**erased from the list**. The hybrid model then had no checkpoint before the divergence and re-processed from
zero - the loaded state bought nothing. When A5 (the exact continuation of A's T3) arrived, its entry was gone
and it re-processed 2639 tokens; B5, whose T4 had selected its slot by similarity (the slot itself held B's T3
state until B4 saved it), came back from the cache with 5089 tokens reused. On a KV-only model the 42% prefix would
have been usable, which is why upstream's `server_prompt_cache::load` takes it. Possible fix: on a
`SEQ_RM_TYPE_FULL/RS` context, take an entry only if one of its checkpoints covers the common prefix (or keep the
entry in the list when the restore ends in a reset). Not built; the owner decides whether the case matters
(an edited earlier turn on a stream that was displaced from its slot in between).

Also seen, benign: A'1 into a slot holding A hit "forcing full prompt re-processing" for a 2-token common prefix
(the template header) - a reset that costs nothing.

## Running it again

    B=/Users/troff/play/llama.cpp-prod K=4096 LINE=q4 bash perf/run-ctx-class-cache-gate.sh   # ~25 min, TAG=ctxcache-<date>

`K=` scales the four classes; the documents scale with them (0.6 of each class), so K=8192 is ~2 h of prefill.
`--phase verdict --ref <cached.json> --out <control.json>` re-prints the table from the two result files.

## The slot/RAM duplicate, and two rules that replace it (2026-09-25 evening, owner: "Let's see where it brings us")

In split mode (what the size classes need) `--cache-idle-slots` serializes every idle slot into the RAM cache on
each launch and leaves the slot untouched: the copy costs the full state (KV + recurrent + checkpoints) a second
time and, as the arms below show, is never read. The cache is only consulted when the slot being taken is saved
(LRU pick, or a similarity pick keeping under half of its context); a similarity pick that keeps 99% never saves,
so it never loads either - the T2 branch of C sat in the cache in arm A and C6 resumed from a checkpoint in the
slot regardless. Two opt-in rules on this branch make the idle-slot save unnecessary and fix the partial-match
waste found in the morning:

- **`LLAMA_CACHE_SAVE_TAIL=N`** (`get_available_slot`): save the chosen slot's state whenever the task will
  overwrite at least N *decoded* tokens of it (`slot tokens - 1 - common prefix`; the last sampled token is
  never decoded, so a plain continuation counts 0), whichever way the slot was picked. Costs one serialization
  (~100 ms, the entry's size) per real overwrite; the cache drops an entry contained in a newer one, so it stays
  one entry per branch.
- **`LLAMA_CACHE_LOAD_CKPT=1`** (`server_prompt_cache::effective_reuse`): on a context whose partial state cannot
  be truncated (recurrent / SWA), score a cached prompt by the tokens the server can actually resume from - the
  newest checkpoint at or before the common prefix, or the prefix itself when it reaches the entry's decoded
  end - instead of the raw prefix. An entry with nothing to resume from is left in the list.

Three arms, same plan plus T6 (the T2 branch of a stream returns after T3 overwrote it in the slot), short
control (in-slot A1-A3 only). Every T1-T4 request gave the morning's shas in every arm: the unflagged path is
unchanged and the flags change what is saved and loaded, not what is computed.

| arm | config | RAM cache peak | A5 (entry consumed by A4?) | T5 round, 4 slots | T6 branch return |
|---|---|---|---|---|---|
| A | today's defaults | 10 entries, 7879 MiB | reset, 0 / 2639 | 22 s (A5's full prefill in the batch) | C6 from a slot checkpoint (9382 / 10003); the cache copy existed, never loaded |
| B | `--no-cache-idle-slots` | 3 entries, 2110 MiB | reset, 0 / 2639 | 22 s | C6 from a slot checkpoint, same |
| C | B + `SAVE_TAIL=32` + `LOAD_CKPT=1` | 5 entries, 3601 MiB | **restored, 2608 / 2639** | **3.1-3.9 s** | C6 from a slot checkpoint: C2's reply was 4 tokens, 28 lost < 32 (rule correct, test wrong) |
| C2 | arm C, T6 on stream B | 4 entries, 3042 MiB | restored, 2608 / 2639 | 3.1-3.9 s | **B6 from the RAM cache, 5086 / 5114** (`found better prompt f_keep 1.000`; saved at B3: 68 tokens overwritten) |

TAGs `ctxcache-arm{A,B,C,C2}`, all PASS, A3's RAM round trip byte-identical in each. The tail rule fired exactly
where it should: B3 (68 decoded tokens overwritten) and the T6 request itself (120); not on C3/D3 (28). Arm B
alone already removes the duplicate with no loss on this plan (displacement is covered by the eviction-path
save); arm C is arm B plus the two cases upstream's rules miss: the branch overwritten in place (B6) and the
partial match with no checkpoint under it (A5, whose 2.6K-token prefill also stalled the concurrent round 22 -> 3 s).

Recommendation for the size-class servers: `--no-cache-idle-slots`, `LLAMA_CACHE_SAVE_TAIL=64` (32 was the test
value; a 64-token tail is ~0.5 s of prefill against a ~100 ms save), `LLAMA_CACHE_LOAD_CKPT=1`. Both flags are
inert without `--cache-ram`. Owner decides; not on prod.

## What a checkpoint is made of (owner: "Can you dump a checkpoint to disk?", 2026-09-25 evening)

Two ways exist. `--slot-save-path` + `POST /slots/<id>?action=save` writes a slot's whole sequence state to a
file (KV + recurrent + drafter, `llama_state_seq_save_file`) and `action=restore` reads it back - the
persistent form of a RAM-cache entry, not a checkpoint. A checkpoint is three byte vectors in host memory; the
branch adds **`LLAMA_CKPT_DUMP=<dir>`**, which writes them as `ckpt-slot<id>-n<tokens>.{tgt,dft,spec}` whenever
one is created, and `perf/ckpt-dump-decode.py` walks the target blob (magic, seq, cells, then per non-null layer
a type + row size header and the rows). One 2224-token prompt under the q4 pick:

| blob | bytes | content |
|---|---|---|
| `.tgt` | 156,894,364 (149.6 MiB), fixed | the recurrent state only (hybrid `PARTIAL_ONLY`): 48 GDN layers x (S: 786,432 f32 = 3.000 MiB + R: 30,720 f32 = 0.117 MiB), 1 cell at pos 2223 |
| `.dft` | 35 -> 40 MiB, **grows ~20 KB per token up to the drafter's 2048 cells, then flat** | **the DFlash drafter's entire KV cache** (n_stream 4, 2048 cells of 20 KB; still 40.0 MiB at a 19906-token checkpoint): a plain KV cache ignores `PARTIAL_ONLY` and writes every cell |
| `.spec` | 0 | the drafter's speculative stash (empty for DFlash2) |

So the "170-190 MiB" of the morning = 150 MiB of GDN state + 20-40 MiB of drafter KV (bounded by the drafter's
2048-cell context, so a checkpoint is at most ~190 MiB at any position - an earlier reading of the two dumps as
"13 KB per prompt token, 1.3 GB at 96K" was wrong: the log's 19906-token checkpoint is 189.65 MiB, the same
drafter part as at 2K). The drafter's memory is a plain KV cache (`SEQ_RM_TYPE_PART`): it needs nothing checkpointed, the normal
`seq_rm` to `n_past` truncates it on restore exactly as on every cached prefix, and `load_dft` already returns on an
empty blob. **`LLAMA_CKPT_NO_DFT=1`** (`create_checkpoint`) leaves `data_dft` empty when the drafter's memory
supports partial removal - a byte-identical lever by construction (the cells below `n_past` are the same cells).

On the f16 idea for the S state: the largest row has max |x| 41.9, every element nonzero, and 7.05% of the
elements below f16's normal minimum (6.1e-5) - f16 would push them subnormal (bf16 keeps the range at 8 mantissa
bits). Not free; a numerics lever to price if the 144 MiB per checkpoint ever matters after the drafter fix.

**Arm D** (`ctxcache-armD`: arm C2's config + `LLAMA_CKPT_NO_DFT=1`): every one of the 44 checkpoints is 149.626
MiB (arm C2: 169.7-189.7), every request's sha, `cache_n` and acceptance identical to arm C2, the RAM cache peak
3042 -> 2723 MiB (the saved entries carry their checkpoints). Byte-identical as expected: -21% per checkpoint,
a fixed size at every position. Proposed with the arm C flags; owner decides.

**Caveat on `LLAMA_CKPT_NO_DFT` (owner's question, same evening: "doesn't the drafter only look at the last 1024
tokens anyway?").** The drafter's cache is its own f16 context (`-ctkd f16 -ctvd f16`; the Turbo4 type is the
target's), and `LLAMA_DRAFT_WINDOW=1024` is a sink + window: during decode the drafter frees cells older than the
last 1024 each round and keeps a ring of the last 1024 injected feature rows. The 2048 cells in the dump are the
prefill high-water mark, up to twice what the drafter attends. Target output is unaffected by the flag (drafts are
proposals), but a restore to a checkpoint more than ~1024 tokens behind the drafter's current end finds those
window cells already freed and the feature ring rolled past them: with the drafter blob they come back, without it
the drafter drafts from a short view until it refills - an acceptance dip for up to a window of tokens, no text
change. Every restore in the gate was within 120 tokens of the end (acceptance identical), so arm D did not price
this. The precise form: checkpoint only the sink + last `n_window` cells (~20 MiB) instead of nothing. Not built.

## The lossy-checkpoint experiment (owner: "Hypothesize away ... Let's try it and see what we learn", 2026-09-25 night)

Is the delta-net state compressible by a change of basis? Per-head SVD of the dumped 128x128 states says the
energy is: effective rank (exp of the spectral entropy, median over heads) 1.0-3.7 per layer at 2.2K tokens,
1.0-6.8 at 8.5K; the top 8 singular triplets hold 88-100% of the energy. Lossless is impossible either way (the
mantissas are full-entropy: zstd 92.7% of raw, ~27.5 bits of entropy per f32; the conv state R is ~14 bits per
value, f16-shaped). So the question is what the readout tolerates. Tools: `LLAMA_CKPT_LOAD_DIR=<dir>` (the
server restores `<dir>/ckpt-slot<id>-n<tokens>.tgt` in place of the in-memory blob when the file exists),
`perf/ckpt-lowrank.py` (rank-k truncation / f16 round trip of every head's state, layout byte-exact),
`perf/ckpt-lowrank-driver.py` (per variant: a fresh full re-prefill of the slot, then a request that restores the
n-4-ubatch checkpoint = the variant blob, 522 tokens of re-prefill, 128 greedy tokens with top-20 probs; the
`exact` variant is the file round trip of the untouched dump and reproduces the in-memory restore byte for byte).
KL = KL(exact || variant) over the union of the two top-20 sets, mean over the tokens before the first fork.

| variant | S storage | S rel. error | 2228-token prompt: text / fork / mean KL / max KL | 8991-token prompt: text / fork / mean KL / max KL |
|---|---|---|---|---|
| rank 32 | 50% | 2.5-3.1% | differs @32 / 1.0e-4 / 3.2e-3 | differs @66 / 5.6e-4 / 3.7e-2 |
| rank 16 | 25% | 4.5-5.3% | **identical** / 8.2e-5 / 1.1e-2 | differs @66 / 8.3e-4 / 5.5e-2 |
| rank 8 | 12.5% | 6.8-7.7% | **identical** / 2.2e-4 / 2.9e-2 | differs @66 / 1.9e-3 / 1.2e-1 |
| rank 4 | 6.3% | 9.3-10% | **identical** / 7.1e-5 / 9.1e-3 | differs @29 / 6.0e-3 / 1.8e-1 |
| rank 2 | 3.1% | 12-13% | differs @60 / 4.1e-4 / 2.5e-2 | differs @26 / 8.2e-3 / 2.1e-1 |
| rank 1 | 1.6% | 15-16% | differs @0 | differs @26 / 7.8e-3 / 2.0e-1 |
| f16 | 50% | 0.02% | differs @32 / 2.2e-4 / 6.9e-3 | **identical** / 2.4e-5 / 2.1e-3 |

What we learned: (1) the energy criterion is the wrong one - at 2.2K tokens rank 4 keeps 90% of the energy and
the exact text, at 8.5K rank 32 keeps 99% and forks at token 66 with a 5.6e-4 mean KL; the query readout weights
directions by relevance, not by singular value, and the state accumulates directions with context. (2) The price
grows with context: 5-50x between 2.2K and 8.5K at every rank. For scale, the pick's decode-kernel classes are
5e-6 pairwise and the Turbo4 cache itself 0.006 vs f16: rank 32 at 8.5K sits between them, rank 4 at 8.5K equals
the Turbo4 cache's price for a 16x smaller checkpoint. (3) f16 is not a free half: it forks the 2.2K case (the
same near-tie token 32 as rank 32; 11% of the elements are below f16's normal range) and is benign at 8.5K -
noise-level, not a trend, and the fork positions are ties (`check-logit-margin-before-hunting`). (4) Every fork
here is a single flipped near-tie; no variant produced garbage above rank 1. One prompt per size, 128 tokens,
top-20 KL: a first look, not a pricing. The pricing recipe if any of this is wanted: the paired KLD pair over the
model's own text with restores at several depths.

**Regenerate mode** (`--mode near`: the variant request repeats the reset prompt exactly, so the server restores
the n-4 checkpoint and re-prefills 4 tokens before generating; 2228-token prompt, 128 tokens of a low-confidence
"describe in detail" answer):

| variant | text | fork | mean KL | max KL |
|---|---|---|---|---|
| rank 32 | identical | - | 6.9e-7 | 8.8e-5 |
| rank 16 | differs | 44 | 5.3e-7 | 2.3e-5 |
| rank 8 | differs | 20 | 1.8e-5 | 3.7e-4 |
| rank 4 | differs | 20 | 6.0e-5 | 1.2e-3 |
| rank 2 | differs | 20 | 7.3e-5 | 1.5e-3 |
| rank 1 | differs | 20 | 4.5e-4 | 9.0e-3 |
| f16 | identical | - | 1.7e-7 | 2.2e-5 |

The forks sit at tokens 20 and 44 with mean KLs of 1e-7..1e-5: ties in a low-confidence text, not damage. The
surprise is the direction: with 4 tokens of re-prefill the truncation costs 10-100x LESS than with 522 (far
mode, same prompt). The 522-token re-prefill does not wash the state error out, it compounds it - the delta rule
updates the state against its own (wrong) prediction at every step. One sample per mode; a hypothesis to test
with restores at several depths before it is believed. Results: `kvquant-experiments/results/ckpt-lowrank-sep25.*`
(far) and `ckpt-lowrank-near-sep25.*`; the dumps used are in the session scratch, not kept.

Bottom line for the checkpoint-memory question: a per-head rank-8..16 factorization is a 4-8x lever on the 144
MiB S state whose price is a NUM-TG class between the decode kernels and the Turbo4 cache at 2K context and grows
with context; f16 is a 2x lever at noise level in two of three cases and a tie-flip in the third. Neither is
adopted; the drafter fix (`LLAMA_CKPT_NO_DFT`) is the only checkpoint lever that is free.

## Longer lengths: 12K / 24K / 48K / 96K classes (owner: "Should we see what happens at longer lengths?", 2026-09-25 night)

`K=12288 MATERIAL=longprompt-yarn-486k.txt CONTROL=short`, streams 7.2K / 7.1K / 14.4K / 28.5K / 57.3K tokens, two
arms: A = today's defaults, P = the proposed set (`--no-cache-idle-slots LLAMA_CACHE_SAVE_TAIL=64
LLAMA_CACHE_LOAD_CKPT=1 LLAMA_CKPT_NO_DFT=1`). TAGs `ctxcache-long-arm{A,P}`, both PASS (the driver's acceptance
guard now applies only to replies of >= 24 tokens - a 6-token reply at 2/9 accepted drafts is not a signal), A3's
RAM round trip byte-identical in both, every T1-T5 sha identical between the arms, D1 = 57K tokens in 541 s.

| | arm A (defaults) | arm P (proposed) |
|---|---|---|
| RAM cache peak | 7 entries, 7984 MiB (**at the 8 GB cap, 2 evictions**) | 4 entries, 2875 MiB, no eviction |
| a 57K-token saved entry | 1095 MiB | 1095 MiB (minus the drafter blobs in its checkpoints) |
| eviction-path save + load, max | 103 ms (11 saves) | 105 ms (12 saves) |
| idle-slot save attempts | 60 | 0 |
| checkpoint at 57K tokens | 169.7-189.7 MiB | 149.6 MiB (fixed) |
| A5 (entry consumed by A4?) | reset: 7379 tokens re-prefilled | restored, 7350 / 7379 |
| **the concurrent T5 round** | **59 s** (A5's re-prefill holds all four streams) | **4-5 s** |
| B6 (branch overwritten at B3) | slot checkpoint (13907 / 14530) | slot checkpoint: B2's reply was 6 tokens, 30 lost < 64, the rule correctly did not save |

What the length changes: nothing in the mechanisms (every restore, including D3 from a 57K state in 0.9 s, and A3
through the RAM cache, behaves as at 4K), and the costs of the defaults get worse exactly where predicted - the
duplicates push the cache to its cap and start evicting, and the partial-match consumption turns a 22 s stall
into a 59 s one because the re-prefill it forces is a 7K-token one. A save is still ~100 ms even at 1.1 GB.
Checkpoints stay at 2 per single-message slot at any length (the spacing rule), so the checkpoint memory of a
long slot is 300 MiB, not the 32 x 190 MiB worst case; the RAM cache is where a long slot's memory goes (1.1 GB
per saved 57K state), and with the defaults that cache was full.

## Defaults (2026-09-26, owner: "2, while accepting loss of acceptance")

The owner chose code defaults over flags. `server-context.cpp`: `save_tail` defaults to 64, `ckpt_aware` defaults to on
for a context whose partial state is fixed, `ckpt_no_dft` defaults to on when the drafter's memory supports partial
removal, and the idle-slot save pass runs only under `--kv-unified` (where it is a move that frees shared cells) or
`LLAMA_CACHE_IDLE_SPLIT=1`. The startup line `prompt cache: checkpoint-aware load = 1 ..., idle-slot saves = off (split
mode ...)` says what is in force. `--cache-idle-slots` keeps its meaning under unified KV. Nothing else in the server
changed; the four off-switches restore upstream's behaviour one rule at a time.

Accepted with it: a restore further back than the drafter's window (1024) from its current end finds the drafter's
window cells freed and drafts from a short view until it refills - an acceptance dip, no text change. The precise
form (checkpoint the sink + last 1024 cells, ~20 MiB) was not built.

Gates on the default binary (2026-09-26, no flags): the cache gate `ctxcache-defaults` = arm D line for line (A5 restored
2608/2639, B6 from the cache 5086/5114, peak 4 entries / 2723 MiB, all shas, A3 round trip identical) - PASS; the
multi-slot gate `defaults-0926` q4 short + long PASS, ud short + long PASS (every reference sha held: the defaults touch
only what is saved and loaded). Merged to prod the same morning; post-merge gates on the prod binary recorded below.

Post-merge on the prod binary (prod `4770657a2`, rebuilt 2026-09-26 morning): cache gate `ctxcache-postmerge` PASS
(= arm D line for line, peak 4 entries / 2723 MiB, B6 from the cache, A3 round trip identical), multi-slot gate
`postmerge-0926` q4 short arm PASS on its references. The defaults are live on prod.

## Cold state on disk (2026-09-26 morning, owner: "move everything except the kv cache to disk", branch `exp/cold-state-spill`)

The owner's framing, which is the right one: three copies of sequence state exist and only one is hot. The live KV in
the slot sits in Metal buffers and is read every decode step. A **checkpoint** is a host-side snapshot of the GDN state
at a mid-prompt position, written once and read only on a rollback. A **RAM prompt-cache entry** is a host-side copy
of a displaced slot's whole state (Turbo4 KV + end-of-prompt GDN state + drafter KV + the slot's checkpoint list),
written once and read at most once, if that conversation returns. On unified memory every cold MiB held in host RAM
is a MiB the model, the KV and the size classes cannot use. So: the cold copies go to disk, the live KV stays.

**Build** (commit `c0ca0afba`): `common_cold_blob` (common.h) = a blob written to an `mkstemp` file that is **unlinked
at creation** - it lives exactly as long as the last `shared_ptr` (a slot's list and a cache entry share one file), a
crash leaves nothing behind, `ls` never shows it, and its pages sit in the OS buffer cache until memory pressure moves
them to flash (so the OS runs the tier). `common_prompt_checkpoint::spill()` moves `data_tgt`/`data_dft` to files right
after `create_checkpoint` captures them (after the `LLAMA_CKPT_DUMP` hook); `load_tgt`/`load_dft` read a spilled blob
into a temporary buffer for `llama_state_seq_set_data`. `server_prompt_data::spill()` does the same for a cache entry
in `prompt_save`; `server_prompt_cache::load` reads it back. The per-round `spec_ckpt` (hot) is untouched. `size()`
counts cold bytes, so `--cache-ram` and `--ctx-checkpoints` keep their meaning (a byte cap on the cache wherever it
lives). **Default on**; `LLAMA_COLD_STATE=0` keeps everything in RAM, `LLAMA_COLD_STATE_DIR=<dir>` moves the files off
the OS temp dir (`$TMPDIR`, the same SSD). Startup INF line `cold state: ...`; `-lv 5` shows `spill N ms` on every
checkpoint line, `cached state spilled to disk` / `read from disk` and `checkpoint restored from disk` with ms.

**Smoke** (q4 pick, `--ctx-seq-sizes 4096,4096`, seven chat requests: a 1.5K prompt, a continuation, a different
follow-up = rollback, two fresh chats that displace it, its return = cache load + checkpoint, a rollback on the loaded
entry), RAM arm vs disk arm: all seven texts byte-identical. Costs on this SSD:

| operation | size | time |
|---|---|---|
| checkpoint spill (write, no fsync) | 149.6 MiB | 14.2-15.3 ms |
| checkpoint restore from disk (read + set_data) | 149.6 MiB | 10.7-11.2 ms |
| cache entry spill | 196 MiB (KV 175 + drafter 21) | 18.7-19.2 ms |
| cache entry read back | 175 MiB | 9.7-10.1 ms |

(The reads are page-cache hits: the file was written seconds before. A cold-from-flash read of 150 MiB measured
~50 ms with `cat` on this box.) The `TMPDIR` listing showed 0 `llama-cold-*` files during and after the run.

Two pre-existing behaviours seen in the smoke, unrelated to the tier, noted for whoever reads the log: (1) with 4K
classes the min-step rule (8192) erases every checkpoint of an earlier task except the first, so a returning chat's
second rollback goes to the system-prompt checkpoint; (2) a fresh 70-token chat is placed by LCP similarity on the
1.5K slot (f_sim 0.74 on the shared system prompt, f_keep 0.03) rather than the empty slot - the save-on-overwrite
rule then saves the 1.5K state, which is what makes its return cheap.

**Gate**: `run-ctx-class-cache-gate.sh` (LINE=q4 K=4096, full control) twice on the branch binary, disk arm
`coldspill-q4-disk` and RAM arm `coldspill-q4-ram` (`EXTRA_ENV=LLAMA_COLD_STATE=0`); the script now samples the
server's RSS every 2 s and prints the peak per phase plus the disk-path counts. **Both arms PASS** (2026-09-26,
09:04-09:51): all 22 cached-phase shas identical between the arms, cached = uncached in each, A3's RAM round trip
byte-identical (max |dlogprob| 0), cache peak 4 entries / 2723 MiB in both (the same bytes, in files vs in RAM).

| | disk arm | RAM arm (`LLAMA_COLD_STATE=0`) |
|---|---|---|
| server RSS peak, cached phase (5 streams, 4 entries) | **20.30 GiB** | 24.34 GiB |
| server RSS peak, control phase | 19.81 GiB | 23.02 GiB |
| spills / reads from disk | 84 checkpoints (mean 15.7 ms, max 59.6), 16 entries (mean 30 ms, max 141 at 252 MiB) | 0 |
| restores from disk | 6 checkpoints mean 11.4 ms; 7 entry reads mean 14.5 ms, max 23.4 | - |

The spill takes 4.0 GiB off the process at the peak - the cache (2.7 GiB) plus the in-slot checkpoints. The fixed
part (model 14.8 + drafter 1.0 + GDN state 1.2 + KV 1.2 + compute arenas 2.3 GiB) is what remains, see the memory
breakdown discussion above. The mid-prefill costs are 15 ms per checkpoint and 30 ms per entry save, and all reads
were page-cache hits; a cold-from-flash read is ~50 ms per 150 MiB. **MERGED TO PROD 2026-09-26 (owner: 'Yes do it')** as a default like the other
four; off-switch `LLAMA_COLD_STATE=0`. **OPEN: the prod binary was NOT rebuilt at the merge (owner packing the box) - run
`cmake --build build --target llama-server -j 12` in llama.cpp-prod, then the multi-slot gate (run-multislot-gate.sh) on
both lines; expected: every reference sha held (the change touches only what is saved and loaded) plus the
`cold state:` startup line.**
