# Reasoning-effort line position: can the template move it off token 1? (2026-10-03, owner: "The goal is to be able to switch")

Status: **DONE 2026-10-03 evening, owner decides adoption** (branch `exp/effort-line-position`, worktree `llama.cpp-effortpos`; serving
binary = prod `089c2367e`, no C++ change). The tail template works: the model reads the effort line at the end of the system block
as at the head (Q&A calibration 16 prompts x 3 levels, and a recorded tool session at xhigh and low), and a level change or a
thinking toggle inside a session then costs ~500 tokens / 5 s of prefill instead of the whole prompt (85 s at 10.7K). To serve
it: `--chat-template-file perf/qwen3.8-effort-tail.jinja` (env `LLAMA_ARG_CHAT_TEMPLATE_FILE`); the pick mints are unaffected
(pick_prompt renders no system block), server-made prefix saves rebuild themselves under the new token hashes, the session-corpus
replays should carry the same template (`run-session.sh EXTRA_ENV`) once it is the serving default.

## Why

The Qwen3.8 template renders `Reasoning effort is set to xhigh. ...` (or the `low` sentence; `medium` renders no line) as the
first tokens of the system block, before the tools JSON and the client's system text. Both agent clients send
`reasoning_effort` on every request once a level is chosen (opencode "variants", pi thinking level; perf/prefix-slot-saves.md).
So a level change or a thinking toggle inside a session changes token 1: the conversation re-prefills from zero and every
prefix save (`LLAMA_PREFIX_DIR`) stops matching. Moving the line to the end of the system block keeps the head and project
layers valid across levels; the question is whether the model still reads the instruction there, since it never saw it
anywhere but token 1.

## Templates

- stock: the template embedded in the GGUF (`perf/qwen3.8-stock.jinja`, dumped for the diff).
- tail: `perf/qwen3.8-effort-tail.jinja` - the identical template with the `reasoning_instructions` emission moved from the
  head of the system block to its end, in both the tools branch (after `merged_system`) and the plain branch. With no system
  message and no tools the two render identically (the line is the whole system block), so every test request carries a
  system prompt. `--chat-template-file <file>` selects it; `reasoning_effort` in the request body sets the level.

## Method (`perf/run-effort-pos.sh`, `perf/effort-pos.py`, prompts `perf/effort-pos-prompts.json`)

The readout is a calibration, not a scalar: per prompt and level, the thinking-token count under tail against stock at
the same level. The stock xhigh/low spread is the yardstick; stock medium (no line) is the "line ignored" control.

- ud line, Turbo4, one slot `-c 32768`, depth 3 pinned (texts compare by sha), `run-prefix-server.sh`, one server per template.
- 16 prompts behind one 60-token system prompt: design/explanation (btree, asyncio-leak, ratelimit, tcp, git, sql-2nd,
  rust-borrow, c-race, regex) and checkable answers (tank 12 h, bird 200 km, pyout, dice 5/12, pow 49, heapify O(n), mpg 31.4).
- Arms: stock-{xhigh,low,medium}, tail-{xhigh,low,medium}. temp 0, `max_tokens` 16384 (a cap would truncate the count).
  tail-medium renders the same prompt as stock-medium: it is the server-restart determinism floor.
- `/apply-template` renders of every prompt and level are saved per server (`render-<tmpl>.json`): the diff must be the moved line only.
- Pass: tail-xhigh tracks stock-xhigh and tail-low tracks stock-low on the prompts where stock xhigh and low separate.
  tail-low ~ tail-xhigh ~ medium = the model ignores the moved line. Anything else, or a correctness drop = the position perturbs it.

Results: `kvquant-experiments/effortpos/<TAG>/` (one JSON per prompt x arm with the full thinking and answer text; `report.md`).

## Results (`effortpos-oct03`, ud, prod `089c2367e`, 2026-10-03 15:06-17:50)

thinking tokens (answer tokens); `*` = hit the 8192 cap (stock-xhigh btree ran before the cap was lowered from 16384, it hit that); Y/N = checkable answer right/wrong

| prompt | stock-xhigh | tail-xhigh | stock-low | tail-low | stock-medium | tail-medium |
|---|---:|---:|---:|---:|---:|---:|
| asyncio-leak | 8192* (0) | 8192* (0) | 1985 (2227) | 1548 (2550) | 2114 (3107) | 2114 (3107) |
| bird | 111 (108) Y | 126 (118) Y | 159 (134) Y | 207 (152) Y | 185 (264) Y | 185 (264) Y |
| btree | 16385* (0) | 8192* (0) | 3103 (2873) | 2869 (2599) | 2276 (2821) | 2276 (2821) |
| c-race | 8192* (0) | 8192* (0) | 678 (932) | 1149 (1069) | 996 (1110) | 996 (1110) |
| dice | 358 (137) Y | 225 (52) Y | 536 (293) Y | 479 (365) Y | 629 (334) Y | 629 (334) Y |
| git | 8192* (0) | 8192* (0) | 966 (1676) | 1018 (1607) | 1621 (1833) | 1621 (1833) |
| heapify | 2835 (590) Y | 8192* (0) N | 491 (889) Y | 1785 (655) Y | 1050 (834) Y | 1050 (834) Y |
| mpg | 741 (149) Y | 428 (176) Y | 775 (193) Y | 1119 (217) Y | 1003 (246) Y | 1003 (246) Y |
| pow | 458 (124) Y | 378 (123) Y | 454 (220) Y | 534 (199) Y | 531 (272) Y | 531 (272) Y |
| pyout | 997 (179) N | 1072 (175) N | 590 (251) N | 520 (270) N | 741 (427) N | 741 (427) N |
| ratelimit | 8192* (0) | 8192* (0) | 1232 (2974) | 1145 (2655) | 1061 (3772) | 1061 (3772) |
| regex | 346 (622) | 468 (536) | 281 (595) | 291 (540) | 308 (595) | 308 (595) |
| rust-borrow | 8192* (0) | 8192* (0) | 2662 (591) | 4993 (485) | 1967 (844) | 1967 (844) |
| sql-2nd | 3672 (559) | 1218 (698) | 777 (565) | 734 (520) | 880 (701) | 880 (701) |
| tank | 119 (112) Y | 119 (112) Y | 114 (136) Y | 116 (134) Y | 147 (157) Y | 147 (157) Y |
| tcp | 6175 (662) | 8016* (174) | 3292 (803) | 1053 (598) | 1071 (883) | 1071 (883) |
| **sum think** | **73157** | **69394** | **18095** | **19560** | **16580** | **16580** |
| capped, no answer | 6/16 | 8/16 | 0 | 0 | 0 | 0 |

- **Determinism floor is zero**: tail-medium reproduced stock-medium byte for byte on all 16 prompts (the rendered prompts are
  identical at medium) across a server restart. Every tail-vs-stock difference at xhigh/low is the line's position.
- **The yardstick**: stock xhigh/low = 4.0x in thinking tokens (73.2K vs 18.1K; 65.0K vs 18.1K with btree at the same cap),
  xhigh capped without an answer on 6 of 16 prompts, low and medium never. low vs medium: 1.09x - on this prompt set the
  low sentence is not a brake relative to no line; the separation that matters is xhigh against the other two.
- **tail-xhigh behaves as xhigh**: 69.4K (1.07x stock at equal caps), 8 of 16 capped (stock's 6 plus heapify and tcp),
  xhigh/low separation 3.5x. The "line ignored" signature would have been ~16.6K (the medium sum); it is 4x that.
- **tail-low behaves as low**: 19.6K vs 18.1K (1.08x), 0 capped, every checkable answer right.
- **Correctness unchanged** except heapify tail-xhigh, which is a runaway cap (no answer), not a wrong one. pyout is wrong in
  all six arms (the model believes `print(f(1), f(2))` shows `[1] [1, 2]`; both arguments are the same list, it is `[1, 2] [1, 2]`).
- **Per-prompt counts are greedy forks, not signal**: the moved line changes the prompt tokens, the greedy text forks within a
  few hundred tokens, and the thinking length then wanders (heapify low 491 -> 1785, tcp low 3292 -> 1053, sql-2nd xhigh
  3672 -> 1218). The content was identical only on tank at xhigh (the 7/16 "identical" count includes 6 pairs of empty capped
  answers) and on no prompt at low. Read the sums and the capped counts, not a row.

Verdict on the Q&A arms: **the model reads the effort sentence at the end of the system block as it reads it at the head.**
Side finding: at the template's default xhigh the model runs away on 6 of 9 open design/debug prompts (8K+ tokens of thinking,
no answer) under both templates; medium (no line) finishes all of them in 1-2.3K. Same failure mode perf/sharp-template.md
saw at a 4K cap; Sharp's default is medium.

## Agentic confirmation (`perf/session.py record`, pilot10 user script, ud, depth 3, 2026-10-03 17:51-18:16)

The line sits after a 1.7K-token tools block and the system text here, 3-11K tokens in for the real clients. pilot10 recorded
under the tail template at xhigh and at low (`effortpos-oct03/pilot10-tail-{xhigh,low}.json`, `LLAMA_ARG_CHAT_TEMPLATE_FILE`
through `run-session.sh EXTRA_ENV`, `EFFORT=`), against the stock-xhigh recording `perf/session/pilot10.json`:

| recording | assistant turns | tool-call turns | generated | thinking | prose / code / tool | user turn 2 | final ctx |
|---|--:|--:|--:|--:|---|---|--:|
| stock-xhigh | 37 | 28 | 16115 | 8419 (52%) | 4892 / 1251 / 1553 | 10 steps (cap) | 43.5K |
| tail-xhigh | 38 | 29 | 16470 | 7800 (47%) | 5312 / 1270 / 2088 | 10 steps (cap) | 45.2K |
| tail-low | 29 | 19 | 12952 | 5175 (40%) | 4928 / 1275 / 1574 | 5 steps | 34.1K |

Thinking per user turn, stock-xhigh vs tail-xhigh: u3 1677/1322, u4 855/1265, u5 1666/1560, u6 1036/1068, u8 1201/1239 -
the same session. tail-low: -34% thinking, a third fewer tool calls, the step-cap turn resolved in 5 steps, 22% less context at
the end. A recording is one trajectory (the model drives the tools), so these are regimes, not measurements to the token.

**Verdict: the effort line works at the end of the system block, in Q&A and in a tool session, at both levels.**

## What the switch costs (`perf/effort-switch-cost.py`, `effortpos-oct03/run-switch.sh`, ud, one slot `-c 32768`, depth 3)

One captured agent request sent four times with `reasoning_effort` xhigh, low, none (thinking off), xhigh; 8 tokens generated each.

| opencode `oc-a1-006` (10.7K) | stock: prompt_n / cache_n / prefill s | tail: prompt_n / cache_n / prefill s |
|---|---|---|
| xhigh, first request | 10725 / 0 / 85.2 | 10725 / 0 / 85.3 |
| -> low | 10713 / 0 / 85.0 | 504 / 10209 / 4.9 |
| -> none | 10689 / 0 / 84.7 | 480 / 10209 / 4.6 |
| -> xhigh again | 4 / 10721 / 0.1 | 516 / 10209 / 4.9 |

pi `pi-a1-001` (2.2K): stock 2198 / 2186 / 2162 tokens, 17 s each, then 4 tokens on the return; tail 504 / 480 / 516 after the first, 4.5 s.

- Stock: every level not seen before in the slot is a full re-prefill (token 1 differs, the common prefix is 3 tokens). The return
  to xhigh is a RAM prompt-cache hit: the displaced state was saved because the new prompt shared nothing with it.
- Tail: any switch costs ~500 tokens, the distance from the last recurrent-state checkpoint to the line (the hybrid model resumes at a
  checkpoint, `--ctx-checkpoints`), ~5 s at 10.7K or at 2.2K. The return to xhigh costs the same 500 (the slot's state was continued,
  not displaced, so nothing went to the RAM cache) - 5 s instead of stock's 0.1 s on that one path, 5 s instead of 85 on the others.
- With prefix saves (`LLAMA_PREFIX_DIR`): under stock a save matches one level only; under tail the head and project layers are
  shared by every level (the line is in the unsaved tail: the project cut is >= 508 tokens before the first user message).
