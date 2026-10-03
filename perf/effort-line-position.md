# Reasoning-effort line position: can the template move it off token 1? (2026-10-03, owner: "The goal is to be able to switch")

Status: **RUNNING** (branch `exp/effort-line-position`, worktree `llama.cpp-effortpos`; serving binary = prod `089c2367e`).

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

## Agentic confirmation

(running: pilot10 recorded under the tail template at xhigh and at low, `effortpos-oct03/pilot10-tail-{xhigh,low}.json`,
against the stock-xhigh recording `perf/session/pilot10.json`)
