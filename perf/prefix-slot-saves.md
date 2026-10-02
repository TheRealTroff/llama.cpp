# Prefix slot saves: an agent's base prompt restored from disk instead of prefilled (2026-10-02, owner: "I would like to have some slot saves for common prefixes, e.g. the base prompts of the agent software that I run (right now pi and opencode)")

Status: **BUILT + GATED on `exp/prefix-slot-saves` (ud line, Turbo4, one slot, pinned depth 3). Not merged; merge = owner. Open: the q4 line, several slots / size classes, the container's real prompts, who makes and refreshes the saves, checkpoint trimming.**

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

## Usage

```
B=<tree> LINE=ud CTX=32768 DEPTH=3 EXTRA_ENV="LLAMA_PREFIX_DIR=<dir>" SLOTDIR=<dir> perf/run-prefix-server.sh   # detached: nohup ... &
perf/prefix-save.py tok a.json b.json            # token counts, common prefixes
perf/prefix-save.py head a.json b.json oc-head   # prefill to the common prefix of two projects, save
perf/prefix-save.py chat a.json label            # send a capture; prompt_n / cache_n / sha of the generated tokens
perf/prefix-save.py save oc-projA
```
`--slot-save-path` and `LLAMA_PREFIX_DIR` are the same directory here. zsh: call the driver through a function, not `$VAR args`.

## Open

- q4 line (its own saves), more than one slot, `--ctx-seq-sizes` classes, the controller arm.
- Who makes the saves: today the driver, from captures. Candidates: the server saves a project prefix itself at the first user message of an unseen prompt; a head is found as the common prefix of two saved prompts.
- Refresh: a changed prompt just stops matching; stale files are not removed. opencode's date makes a project save one day long, its head is not affected.
- Checkpoints are 150 MiB each and a save carries 2-4; a head needs none when the prompt contains all of it.
- A save without `.spec` leaves the previous conversation's drafter ring in place (acceptance only).
- The container's real prompts.
