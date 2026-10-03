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

## Results

(pending)
