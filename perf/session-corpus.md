# The session corpus: a recorded agent session, replayed teacher-forced (2026-10-01, owner: "time we created a more realistic corpus ... a series of back and forths with some discussion, some code explains, some tool calls, some doc reads"; "absolutely with thinking on")

Status: **PILOT BUILT + MEASURED on `exp/session-corpus` (both lines). Scripts only, no server change; runs on prod's binary. Open: a long recording, slot saves at chosen turns, the k histogram per segment. Merge = owner.**

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

## Traps

- **Wall and prefill seconds are not comparable between arms.** A turn whose text left the script makes the next request prefill the scripted turn (ud controller: 395 s prefill vs 262 s for the arm that reproduces the script; it forked on 22 / 18 of 37 turns). On q4 every turn of both arms differs from the ud-recorded script, so both pay the same 380 s. Compare decode t/s.
- Segment token counts differ between arms whose text forked (the generation runs to its own stop); the per-segment t/s is the comparison, not the seconds.
- The `fork` column is "differs from the script", not a defect: expected for the controller and for any line other than the recording's.
- Do not edit `session.py` / `run-session.sh` while a run is in flight (one python start per arm).

## Open

1. A long recording (100+ assistant turns, 100K+ context, more code writing and edits) - the pilot stops at 43K.
2. Slot saves at chosen turns (`slot-save-hybrid.md`) so an arm can start at turn N without the prefix prefill.
3. The controller's k / block histogram per segment (join the `.picks` trace to the stream).
4. Tool mix: the pilot's tools are read-only; an edit/write tool would add long code-bearing tool calls.
