# Prompt caching under per-slot context sizes: the gate (2026-09-25)

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
