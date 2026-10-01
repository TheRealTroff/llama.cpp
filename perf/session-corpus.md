# The session corpus: a recorded agent session, replayed teacher-forced (2026-10-01, owner: "time we created a more realistic corpus ... a series of back and forths with some discussion, some code explains, some tool calls, some doc reads"; "absolutely with thinking on")

Status: **BUILT + MEASURED on `exp/session-corpus` (both lines): the read-only pilot (`pilot10`) and the edit session (`edit11`, write/edit/compile tools). Scripts only, no server change; runs on prod's binary. In progress: the long three-language recording (`long29`). Open: slot saves at chosen turns, the k histogram per segment. Merge = owner.**

## Why

The prompt corpus (`perf/prompts/`, `run-depth-corpus.sh`) is eight cold single-turn prompts of 200-500 bytes plus benchprompt, 300 tokens each, thinking off. The controller's corpus number (+10.7% q4, +5.7% ud) is its mix: math +25%, JSON +35%, free-form -7..+3%. A real agent session changes regime inside one response (thinking, prose, code block, tool call), grows its context, and runs with thinking on. None of that was measured.

## What it is

- `perf/session.py record`: the model itself drives three read-only tools (`list_dir`, `read_file`, `grep`) over a pinned tree (`kvquant-experiments/data/session-repo` = `git archive 7ac100447`), user turns from a user script (`perf/session/pilot10.user.json`). Thinking on at the template's default effort (xhigh). Output = a frozen transcript (`perf/session/pilot10.json`: tools + messages, reasoning kept).
- `perf/session.py replay`: for every assistant turn, one `/v1/chat/completions` request whose history is the SCRIPT up to that turn; the generation is measured and discarded. Every arm sees identical prompts at every turn whatever its own text did. The model's template keeps `reasoning_content` in history (`preserve_thinking` undefined = true), so a turn that reproduces the script is fully cached for the next request.
- Segments: each stream chunk is booked as think / prose / code (inside a ``` fence) / tool from the server's cumulative `timings_per_token`. A round's cost lands on the first token of its burst, so a round that straddles a boundary is booked to the earlier segment.
- `perf/run-session.sh`: one server per arm and pass from `pick.sh`; arms `ev` (the pick), `dN` (pinned depth N), `nospec`; `TURNS=a-b` replays a range; `LLAMA_SPEC_EV_TRACE` per arm in `results/<TAG>-<arm>.picks`.

```
B=~/play/llama.cpp-session BIN=~/play/llama.cpp-prod/build/bin MODE=replay LINE=ud ARMS="ev d3" PASSES=2 SCRIPT=$B/perf/session/pilot10.json perf/run-session.sh
MODE=record LINE=ud ARMS=d3 USER_SCRIPT=$B/perf/session/pilot10.user.json SCRIPT=... perf/run-session.sh
```

## The pilot script (`pilot10`, recorded on the ud pick at pinned depth 3)

10 user turns -> 37 assistant turns (27 with tool calls), 80 messages, context 631 -> 42.6K tokens, ~16.1K generated: 52% thinking, 30% prose, 8% code, 10% tool calls. User turn 2 hit the 10-step cap (the model kept grepping); the script keeps it as recorded.

## Result (prod `7ac100447`, binary 09-30 21:34, Turbo4, two interleaved passes, TAGs `session-pilot10-{ud,q4}-a`)

Decode t/s, pinned depth 3 -> the pick's controller {3,7} (p1 / p2 where they differ):

| segment | ud d3 | ud controller | | q4 d3 | q4 controller | | d3 acceptance ud / q4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| thinking | 27.2 | 27.8 / 27.6 | +2% | 30.9 / 31.0 | 30.5 / 30.6 | -1% | 64.7 / 65.3% |
| prose | 27.8 | 28.7 | +3% | 31.7 / 31.8 | 32.7 / 33.3 | +3..5% | 67.1 / 67.5% |
| code blocks | 35.2 | 39.6 / 42.1 | +12..20% | 40.2 / 40.3 | 45.9 / 46.6 | +14..16% | 93.8 / 95.5% |
| tool calls | 35.0 | 39.0 | +11% | 39.0 / 39.1 | 47.6 / 47.3 | +21..22% | 86.4 / 84.5% |
| **all** | **28.51 / 28.53** | **29.58 / 29.47** | **+3.5%** | **32.34 / 32.39** | **33.43 / 33.68** | **+3.4..4.0%** | 69.0 / 69.3% |

- **Thinking decodes like prose** (ud 27.2 vs 27.8, q4 30.9 vs 31.7 at pinned depth 3; acceptance 2 points lower). The owner's guess; no separate regime.
- **The controller's value is the code and tool-call segments** (acceptance 85-95%, the deep round pays); on thinking and prose it is level to +3%, not the -2..-7% the cold free-form prompts showed.
- **The mix sets the headline**: thinking + prose = 83% of this session's tokens, so the session number is +3.5..4%, under the prompt corpus's +5.7% (ud) / +10.7% (q4). A session with more code written or more tool calls sits higher.
- **By context** (ud d3 -> controller): 0-8K 32.1 -> 32.4/33.0, 8-16K 30.7 -> 33.6/32.5 (670 tokens), 16-32K 28.1 -> 29.0, 32-64K 28.1 -> 29.5/29.1. No fade to 43K. q4: +7% / +10% / +0.5% / +6%.
- **Repeatability**: pinned depth 3 reproduces the script on 37/37 turns in both ud passes and is equal to two decimals; the controller's two passes are within 0.4% (ud) / 0.7% (q4). k hist ud 3:3661 7:927, q4 3:3562 7:1099 (depth 7 on ~20-24% of rounds).

## The edit session (`edit11`, 2026-10-01 night; owner: the 16-32K plateau "is more related to the content than the length ... start with 4 and then do 1")

The pilot's context buckets split by segment confirm the owner's reading: q4 8-16K is 53% tool calls (+10%), 16-32K is ~50% thinking + 13% code (+0.5%: the controller arm's forked thinking ran 29.5 vs 30.9 t/s and cancelled the code gain), 32-43K is thinking + prose (+6%). The buckets are the mix, not the length.

Tools added: `write_file`, `edit_file`, `compile`. Recordings run on a scratch clone of the pinned tree (`kvquant-experiments/data/session-work`, `cp -cR` per recording). `edit11`: 11 user turns (create a header, tests, extend, compile-and-fix x3, a rename across files, a doc, a JSON summary) -> 46 assistant turns, context to 37K, ~24K generated: 61% thinking, 11% prose, 3% code blocks, 26% tool calls (single write/edit calls of 350-1780 tokens).

Decode t/s, pinned depth 3 -> controller (TAGs `session-edit11-{ud,q4}-a`, two interleaved passes):

| segment | ud d3 | ud controller | | q4 d3 | q4 controller | |
|---|--:|--:|--:|--:|--:|--:|
| thinking | 30.3 | 31.7 | +5% | 33.4 | 35.7 / 36.8 | +7..10% |
| prose | 29.7 | 29.9 | +1% | 31.7 | 33.9 | +7% |
| code blocks | 32.7 | 34.8 / 37.6 | +6..15% | 36.5 | 40.2 / 41.3 | +10..13% |
| tool calls | 35.7 | 43.1 | +21% | 40.3 | 50.2 / 50.7 | +25% |
| **all** | **31.49 / 31.48** | **33.32 / 33.35** | **+5.8%** | **34.94 / 34.96** | **38.97 / 39.52** | **+11.5..13%** |

- **Code-bearing tool calls carry the gain** (93% acceptance at depth 3; 26% of the tokens against 10% in the read-only pilot). Thinking gains too in an edit session (the model drafts the code in its reasoning).
- The session number moves with what the session does: +3.5..4% reading and discussing, +6% (ud) / +12% (q4) editing.
- In `edit11` 10 of 11 compiles returned "no diagnostics" (the model's C++ was right the first time): it does not exercise diagnostic parsing. Hence the next section.

## The compile tool and its leash (owner: "Compile would be nice, because then it has to parse the output ... keep the script on a leash"; "rustc is pretty holier-than-thou")

`compile(path)` is the one tool that starts a process on model-written input. Fixed command lines (the model gives a path, never a flag); check-only (`c++ -fsyntax-only -Wall -Wextra` for C/C++, `rustc --emit=metadata` on a crate root for Rust, in-process `ast.parse` for Python - no object, no link, nothing is run, nothing imported); rustc directly, never cargo (no build scripts, no proc-macro crates, no external crates); the path must resolve inside the scratch tree; `sandbox-exec` with network and file writes denied (rustc gets one throwaway output dir); 120 s timeout; output capped. Not stopped: a model-written `#include`/`include!` of an absolute path is read by the compiler and can surface in a diagnostic in the transcript.

`long29` (`perf/session/long29.user.json`): 29 user turns across orientation, C++ (templates, a breaking signature change), Rust (a crate from scratch: generics, a slot pool with `&mut` returns, an error type, an iterator API, a rename) and Python (a gguf-py script, a refactor, unittest), with discussion, a derivation and JSON turns between. The recorder stops at `--ctx-limit` (default 92K, under the pick's 102400).

## Traps

- **Wall and prefill seconds are not comparable between arms.** A turn whose text left the script makes the next request prefill the scripted turn (ud controller: 395 s prefill vs 262 s for the arm that reproduces the script; it forked on 22 / 18 of 37 turns). On q4 every turn of both arms differs from the ud-recorded script, so both pay the same 380 s. Compare decode t/s.
- Segment token counts differ between arms whose text forked (the generation runs to its own stop); the per-segment t/s is the comparison, not the seconds.
- The `fork` column is "differs from the script", not a defect: expected for the controller and for any line other than the recording's.
- Do not edit `session.py` / `run-session.sh` while a run is in flight (one python start per arm).

## Open

1. The long recording `long29`: record, then replay on both lines.
2. Slot saves at chosen turns (`slot-save-hybrid.md`) so an arm can start at turn N without the prefix prefill.
3. The controller's k / block histogram per segment (join the `.picks` trace to the stream).
