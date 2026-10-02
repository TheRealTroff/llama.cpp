# Prefix slot saves: an agent's base prompt restored from disk instead of prefilled (2026-10-02, owner: "I would like to have some slot saves for common prefixes, e.g. the base prompts of the agent software that I run (right now pi and opencode)")

Status: **MERGED to prod 2026-10-02 evening (`2ed5f4a4e`, owner: "Move the test saves to offload and merge"; was `exp/prefix-slot-saves`). Off unless `LLAMA_PREFIX_DIR` is set; `LLAMA_PREFIX_AUTO=1` makes the server write the saves itself, in layers, byte-identical to a fresh prefill (section "Saves made by the server"). Post-merge on the rebuilt prod binary: multi-slot gate PASS both lines (shas held), prefix restart + size-class arms 18/18 at dlogprob 0 against the pre-merge references (`prefixmerge-oct02-*`). The two restore fixes are on the default path. Test saves archived on offload (`play-archive/kvquant-experiments/slots/prefix-*`). Open: the controller arm, `LLAMA_PREFIX_CUT=user` / `LLAMA_PREFIX_NO_DFT=1` / the eviction path ungated, the container's real prompts. The sections below are in the order the work went; the first ones describe hand-made saves.**

## What the clients send (captures, no GPU)

`kvquant-experiments/data/agent-prompts/`: `capture.py` is a fake OpenAI endpoint that records request bodies; pi 1.0.0 (+ goal-x, 15 idasql skills via `--skill`) and opencode 1.18.34 were run against it from two scratch projects (`pi-*.json`, `oc-*.json`; `pi -p` needs `</dev/null`). Host installs, not the owner's container: the elements are there, the exact text is not.

The Qwen3.8 template renders the reasoning-effort line, then the tools JSON, then the client's system text. Tokens on the ud server:

| | pi | opencode |
|---|--:|--:|
| prompt up to the first user message | 3630 | 10712 |
| shared with another project | 2084 (57%) | 7288 (68%) |
| first project-specific token | path of the project's AGENTS.md | working directory in `<env>` |
| after it | project file, skills list, cwd | git / platform / today's date, project file, skills list |

The effort line is token 1 (a thinking-level change invalidates a save). opencode sends a title request first (593 tokens, no tools, 41 tokens shared with the main prompt). The skills list follows the project file in both clients, so it is not shareable across projects.

## Two kinds of save

- **head**: the tokens every project of a client shares. The cut is mid-system-message, where the server makes no checkpoint, so the save is made by prefilling exactly those tokens (`/completion` with a token-array prompt, `n_predict` 0) and saving the slot: the recurrent state then sits at the cut and a longer prompt continues from it with no checkpoint.
- **project**: a slot saved after a real first request. Its checkpoint at the first user message resumes any new session of that project.

## The lookup (server-context.cpp)

`LLAMA_PREFIX_DIR=<dir>`: the saves in that directory are listed at startup and whenever the directory changes. When a completion task has its slot, the save that resumes the most tokens of the prompt is restored into it if that beats what the slot itself resumes by `LLAMA_PREFIX_MIN_GAIN` (256) tokens; what the slot held goes to the RAM prompt cache first. Resumable tokens = the whole save when the prompt contains it, else its newest checkpoint inside the common prefix (the rule of `server_prompt_cache::effective_reuse`). Clients send nothing special.

`SLOT_SAVE` now writes `<file>.meta`: model file name and size, KV types, `LLAMA_KV_HEAD_MAJOR`. The lookup lists only saves whose line equals the server's, so a save of the other line or KV type is skipped (logged), never loaded. A prompt that no longer matches a save simply does not pick it. `SLOT_RESTORE` and the lookup share `slot_restore_file`.

## Gate (`prefix-oct02-*`, ud, Turbo4, `-c 32768`, depth 3, opencode captures, 64 tokens, thinking on)

| request | prefilled | reused | prefill s |
|---|--:|--:|--:|
| fresh | 10725 | 0 | 84.9 |
| other project, no save (in-process) | 10727 | 0 | 84.9 |
| other project, head save (restart, lookup) | 3439 | 7288 | 28.9 |
| same project, project save (restart, lookup) | 16 | 10709 | 1.2 |
| project A again after the title request took the slot | 4 | 10721 | 0.1 |

- Text: every restored arm produced the fresh arm's 64 tokens. Project save vs in-process reuse: logprobs equal to 4 decimals. Head save vs fresh prefill: max |dlogprob| 0.004 (the tail is prefilled in a different batch shape; `perf/prefix-fork.py` reports the fork and its margin when there is one).
- Without a save a second project re-prefills everything (the divergence is inside the system message, no checkpoint before it).
- Cost: head 280 MiB + 314 MiB `.ckpt` (2 checkpoints) + 153 MiB drafter pair; project 339 + 628 (4 checkpoints) + 132 MiB. Restore 100-210 ms.
- A foreign `.meta` was skipped with a warning.

## Both lines, several slots (2026-10-02 afternoon, `perf/run-prefix-gate.sh`, `prefix-oct02-{q4,q4b,ud,ud-r1,ud-r2}`)

The gate makes the references and the four saves (head + project A, both clients) on a one-slot server, then restarts and sends the captures with no slot id: one slot; `-np 4`; `--ctx-seq-sizes 32768,16384,8192,8192`; each multi-slot layout sequentially and with the four requests at once. A save made in slot 0 of a one-slot server restores into any slot of any layout.

| prefill s | fresh | head (other project) | project (same project) |
|---|--:|--:|--:|
| q4 opencode (10.7K) | 79.2 | 26.8 | 0.5 |
| q4 pi (3.6K) | 26.2 | 11.7 | 0.4 |
| ud opencode | 84.9 | 28.9 | 0.5 |
| ud pi | 28.1 | 12.6 | 0.5 |

Every arm restored the intended save (`prompt_n` 16 for a project save, 3439 / 1560 for a head) on both lines in all five layouts.

Text against the one-slot references:
- **Project saves: the in-process text in every arm**, logprobs equal sequentially, within 0.06 in the concurrent arms (batch composition).
- **Head saves: the fresh text except at near-ties.** A head restore prefills the tail in another batch shape than a fresh prefill; logprobs move by up to 0.004 (ud) / 0.08 (q4). Forks seen, each priced with a no-drafter replay (`NOSPEC=1`, every token then carries logprobs; a draft-accepted token has none):
  - q4 pi project A through the head: token 4, ` to` -0.936 / ` me` -1.040 fresh, -0.985 / -0.975 through the head.
  - q4 opencode other project, sequential arms: token 22, ` working` -0.756 / ` directory` -0.803 fresh, -0.786 / -0.789 through the head. The concurrent arms landed on the fresh text.
  - ud pi other project, concurrent arms only (4 of 4 runs): token 9, ` here` -0.720 / ` in` -0.726.
  So an unaligned head save is not byte-identical to a fresh prefill (the aligned one below is): it is the cached-vs-uncached delta of `ctx-class-cache.md` (first-token shape numerics), and a project save is exact.

### An aligned head is byte-identical (owner: "The tail is prefilled in a different batch shape - why? Does it have to be that way?")

It does not. A prefill is decoded in micro-batches of 512 tokens (`n_ubatch`), and the Metal kernels are not bit-stable across how tokens are grouped. A fresh opencode prefill groups `[0,512) ... [7168,7680) ...` up to its own end-of-prompt breaks. The first head save moved three boundaries: it was cut at 7288 (not a multiple of 512), and the server broke its last batches at `n-4-512` and `n-4` (6772, 7284) to place the near-end checkpoints, so tokens 6144-7288 and everything after 7288 were grouped differently from the fresh run.

Head saves cut at the last multiple of 512 inside the shared prefix (`--backoff`: opencode 7168 of 7288, pi 2048 of 2084) and made on a server with `--ctx-checkpoints 0` (no near-end breaks; a head needs no checkpoint, the prompt contains all of it): **all 8 comparisons (both lines, both clients, project A and B) give the fresh run's tokens with max |dlogprob| 0.0000**, the three near-tie forks included (`prefix-oct02-{q4,ud}-*-al`). Cost: 120 / 36 tokens more to prefill (27.8 vs 26.8 s on q4 opencode), and the save loses its 314 MiB `.ckpt` (opencode head 412 MiB in all, pi 329 MiB).

Rule: cut a head on an `n_ubatch` boundary and prefill it without checkpoint breaks. Not yet in the tooling as a default (the driver takes `--backoff`, the server flag is manual).

### Two restore bugs this found (both fixed on the branch)

1. **A slot save was tied to the server's slot count.** `llama_kv_cache::state_read` refused a file whose stream count differs ("n_stream mismatch"): a save from `-np 1` did not restore under `-np 4`, the lookup logged it and the prompt was prefilled in full. A single-sequence state has cells in one stream and is read into the target sequence's stream, so the check now applies to whole-context states only.
2. **Out-of-bounds read after a single-sequence restore of the recurrent state.** `llama_memory_recurrent::state_read` left `head` at the restored sequence's cell; the server sets a sampler when it launches a task, the next decode re-reserves the graph, and the reserve reads `n_seqs` cells from `head` (`s_copy_view_row0`): past the end when the restored slot is not the first. ud concurrent arms segfaulted 2 of 2 (crash reports `llama-server-2026-10-02-1439/1440`), q4 read garbage and survived. `head = 0` after the restore: 0 of 4 concurrent runs crash. This is reachable from a plain `SLOT_RESTORE` or a RAM prompt-cache load into a slot other than 0 on a multi-slot server, with or without prefix saves.

## Saves made by the server, in layers (2026-10-02 evening, owner: "Do both"; `LLAMA_PREFIX_AUTO=1`, gate `perf/run-prefix-auto-gate.sh`)

With `LLAMA_PREFIX_DIR=<dir> LLAMA_PREFIX_AUTO=1` nothing is captured or prefilled by hand: the server writes saves while it prefills ordinary requests, at no extra prefill (a batch break at the cut and a 50-70 ms write).

- **Project cut**: the last 512-token (`n_ubatch`) boundary that every prompt with this system block passes before its end-of-prompt batch breaks: `floor((U + 4 - 512) / 512) * 512`, U = the first user message (opencode 9728 of 10709, pi 3072 of 3627). `LLAMA_PREFIX_CUT=user` cuts at U instead (a shorter tail, not the fresh grouping; not gated).
- **Head cut**: when the prompt shares a prefix with a save of another prompt (another project, or the same project on another date), the last 512 boundary inside that shared prefix (opencode 7168, pi 2048). The head is written on the way and the project save becomes a **layer** on it.
- **Layer**: the attention cells from the parent's end on, plus the recurrent state (fixed size, always whole) and the drafter pair. `llama_state_seq_set_tail(ctx, p0)` (llama-ext.h) makes the KV writer emit only cells at positions >= p0 and the reader append; restore = parent chain, oldest first. `.meta` line 2: `auto parent=<name|-> start=<n>`; a layer whose parent is missing or does not match is skipped.
- No checkpoints in these saves: a server-made save resumes only a prompt that contains all of it (scored that way).
- A save is written only from a prefill grouped like a fresh one: it starts at 0 or at the end of a restored server-made save, and the slot is alone in each of its batches (else "cancelled" / "none made" in the log).
- Names `auto-<tokens>-<fnv64 of the tokens>`. `LLAMA_PREFIX_MIN_TOKENS` (1024: opencode's 593-token title request makes nothing), `LLAMA_PREFIX_MAX_MIB` (16384: over it the least recently used server-made saves that nothing builds on are deleted; a restore touches `.meta`), `LLAMA_PREFIX_NO_DFT=1` drops the drafter pair (-153 MiB per save; not gated).

What the server did with an empty directory (ud, one slot; q4 the same sequence):

| request | server action | prefilled | prefill s |
|---|---|--:|--:|
| opencode project A | writes A whole at 9728 | 10725 | 84.9 |
| opencode project B | writes head at 7168 + B as a layer | 10727 | 85.1 |
| project A, next day's date | restores the head, writes A' as a layer | 3557 | 29.9 |
| pi project A / B | whole at 3072; head at 2048 + layer | 3642 / 3644 | 28.2 |
| after a restart: any of them | restores head + layer (58-86 ms) | 997-999 (opencode), 571-572 (pi) | 9.2-9.3 / 5.1 |

Sizes: head 412 MiB (opencode) / 329 (pi); project layer 337 / 313 MiB (of which the recurrent state ~150 and the drafter pair 153; the cells themselves 41 / 16 MiB); a whole project save 453 / 346 MiB. Seven saves 2.5 GB.

**Text: every arm equals a fresh prefill of the same request on a server with no prefix saves and no RAM cache** (`*-ref`): the make phase, the restart, per-slot context sizes (layers restored into the 16K and 8K slots) with max |dlogprob| 0.0000, and four requests at once on `-np 4` with the same tokens and |dlogprob| <= 0.002 (batch composition). ud 20/20 comparisons. q4 (`prefix-auto-oct02-q4`): the 16 sequential comparisons at 0.0000 (restart prefill 8.4 s opencode / 4.6 s pi); the concurrent arm 3 of 4 on the fresh text (|dlogprob| <= 0.008) and opencode project B forks at token 22, the ` working` / ` directory` tie priced above at 0.05 logits fresh - four streams in one batch move a logprob more than that, with or without saves.

Against the hand-made project save (checkpoint at the user message, 0.5 s) the aligned cut pays ~1000 tokens per new session (9.2 s opencode, 5.1 s pi on ud) for byte-identity with a fresh prefill and a third of the disk.

## Usage

```
B=<tree> LINE=ud CTX=32768 DEPTH=3 EXTRA_ENV="LLAMA_PREFIX_DIR=<dir>" SLOTDIR=<dir> perf/run-prefix-server.sh   # detached: nohup ... &
perf/prefix-save.py tok a.json b.json            # token counts, common prefixes
perf/prefix-save.py head a.json b.json oc-head   # prefill to the common prefix of two projects, save
perf/prefix-save.py chat a.json label            # send a capture; prompt_n / cache_n / sha of the generated tokens
perf/prefix-save.py save oc-projA
B=<tree> LINE=q4 TAG=prefix-<date>-q4 perf/run-prefix-gate.sh      # PHASES="make one np4 classes", SKIP=seq, REFTAG=<run with the references>
```
`NP=4` / `SIZES=...` / `NOSPEC=1` on `run-prefix-server.sh`; `--slot -1` and `multi` on the driver.
`--slot-save-path` and `LLAMA_PREFIX_DIR` are the same directory here. zsh: call the driver through a function, not `$VAR args`.

## Open

- The controller arm (the gate pins depth 3). A second pass of the crash fix to the owner's bar (x8).
- Slot choice: a new project of a client goes to the slot holding that client's other conversation (similarity pick) even when an empty slot is idle; the displaced conversation goes to the RAM cache.
- The first project of a client is saved whole (no head is known yet) and stays whole; it is not re-layered when a head appears.
- `LLAMA_PREFIX_CUT=user` and `LLAMA_PREFIX_NO_DFT=1` are built, not gated.
- Refresh: a changed prompt just stops matching; stale files are not removed. opencode's date makes a project save one day long, its head is not affected.
- Checkpoints are 150 MiB each and a save carries 2-4; a head needs none when the prompt contains all of it.
- A save without `.spec` leaves the previous conversation's drafter ring in place (acceptance only).
- The container's real prompts.
