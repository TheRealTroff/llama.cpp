# Reasoning-effort line position: can the template move it off token 1? (2026-10-03, owner: "The goal is to be able to switch")

Status: **DONE 2026-10-04 (mechanism test added), owner decides adoption** (sections in the order the questions came) (branch `exp/effort-line-position`, worktree `llama.cpp-effortpos`; serving
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
- **Correctness unchanged** except heapify tail-xhigh, which hit the cap (no answer), not a wrong one. pyout is wrong in
  all six arms (the model believes `print(f(1), f(2))` shows `[1] [1, 2]`; both arguments are the same list, it is `[1, 2] [1, 2]`).
- **Per-prompt counts are greedy forks, not signal**: the moved line changes the prompt tokens, the greedy text forks within a
  few hundred tokens, and the thinking length then wanders (heapify low 491 -> 1785, tcp low 3292 -> 1053, sql-2nd xhigh
  3672 -> 1218). The content was identical only on tank at xhigh (the 7/16 "identical" count includes 6 pairs of empty capped
  answers) and on no prompt at low. Read the sums and the capped counts, not a row.

Verdict on the Q&A arms: **the model reads the effort sentence at the end of the system block as it reads it at the head.**
Side finding: at the template's default xhigh the model exhausts an 8K thinking budget without answering on 6 of 9 open
design/debug prompts, under both templates; medium (no line) finishes all of them in 1-2.3K. It is not stuck (owner: "it just
thinks every problem has spin 12"): 8-gram repetition in the capped traces is 0.3-7% (a looping model is at 30%+), and the btree
trace is still adding correct nuance at the cap (PostgreSQL leaf linkage, fan-out arithmetic for an 8 KB page, a worked 100M-row
height). The sentence is a prior on how hard the problem is; a mismatch costs in either direction (owner: at too high a level it
builds bells and whistles nobody asked for) - which is the case for switching per task rather than per session. Same failure mode
perf/sharp-template.md saw at a 4K cap; Sharp's default is medium.

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

## Sharp + xhigh (owner: "have you tried sharp+xhigh?"; `sharp-{xhigh,medium}` arms, 2026-10-03 18:45-19:33)

perf/sharp-template.md compared stock at its xhigh default with Sharp at its medium default plus the terseness block, so "Sharp
reins in thinking" was confounded. Same 16 prompts under `perf/sharp_chat_template.jinja` (froggeric v22.1: the terse block at the
end of the system text, the effort line still at the head):

| | stock-xhigh | sharp-xhigh | stock-medium | sharp-medium |
|---|---:|---:|---:|---:|
| thinking tokens | 73157 (64964 at equal cap) | 59068 | 16580 | 15339 |
| answer tokens | 3242 | 2188 | 18200 | 10525 |
| capped, no answer | 6/16 | 6/16 (the same six) | 0 | 0 |

- The terse block does not rescue one of the six open prompts at xhigh; the effort sentence wins. On the mid prompts that finish it
  trims thinking (heapify 2835 -> 1270, sql-2nd 3672 -> 1374, tcp 6175 -> 2788).
- At medium it leaves thinking alone (0.93x) and cuts answers 42%: an instruction about the answer acts on the answer.
- So the September gain was the medium default, not the block. pyout: sharp-xhigh is the only arm of eight that got it right
  (2981 thinking tokens, caught the evaluation order on a re-check) - one trajectory, the shape of the owner's anecdata that xhigh
  pays on some problems.

## The sentence in the user turn (owner: "would that be fundamentally different from just saying ... in a message?"; `umsg-*` arms, 19:34-20:39)

Same 16 prompts on the stock template at level medium (no system line), the template's own xhigh or low sentence prepended to the
user message (`effort-pos.py VARIANTS`): the one change against stock-xhigh/-low is where the trained string sits.

| | stock-xhigh | tail-xhigh | umsg-xhigh | stock-low | tail-low | umsg-low | stock-medium |
|---|---:|---:|---:|---:|---:|---:|---:|
| thinking tokens | 73157 (64964 eq-cap) | 69394 | 72271 | 18095 | 19560 | 17497 | 16580 |
| capped, no answer | 6/16 | 8/16 | 8/16 (tail's eight) | 0 | 0 | 0 | 0 |

The sentence works from the user turn as from the system block, at both levels. pyout was right under umsg-xhigh (2756 thinking
tokens) as under sharp-xhigh (2981): the two trajectories that thought ~3K on it got it, the seven at <= 1.1K did not.

So there are two switches: the level in the system block (what the clients emit; the tail template makes a change cost ~500
tokens) and the sentence typed into a message (per task, zero prefix cost, no template change; stays in the history for later
turns, and nothing in opencode/pi emits it from their level setting).

### Natural language instead of the trained sentence (`nl-*` arms, 20:39-21:17)

Same setup, the user message prefixed with "I want you to think really hard about this." (`nl-hard`) or "Keep it simple, don't
overthink it." (`nl-simple`). open 9 = the design/debug prompts, easy 7 = the checkable ones.

| arm | thinking | open 9 | easy 7 | capped | answer tokens |
|---|---:|---:|---:|---:|---:|
| stock-xhigh (trained sentence, system block) | 73157 | 70027 | 3130 | 6/16 | 3242 |
| umsg-xhigh (trained sentence, user turn) | 72271 | 66631 | 5640 | 8/16 | 1661 |
| **nl-hard** | **26749** | 20418 | 6331 | **0/16** | 21335 |
| stock-medium (nothing) | 16580 | 13036 | 3544 | 0 | 18200 |
| stock-low / umsg-low (trained sentence) | 18095 / 17497 | 15186 / 14654 | 2909 / 2843 | 0 | 15352 / 13358 |
| **nl-simple** | **9469** | 6336 | 3133 | 0 | 8876 |

Every arm: the same 7 checkable answers right, pyout wrong (the two ~3K-thinking xhigh trajectories excepted).

- "Don't overthink it" is twice the brake the trained low sentence is (9.5K vs 18.1K; open prompts 6.3K vs 15.2K); the trained
  low sentence is barely distinguishable from saying nothing. It halves the answers too (btree 1024 tokens vs 2821 at medium).
- "Think really hard" is a graded push: 1.6x medium, nothing capped, every prompt answered. The trained xhigh sentence is 4x and
  puts 6-8 prompts past an 8K budget with no answer.
- Ladder in thinking tokens: don't-overthink 9.5K < nothing 16.6K ~ low 18K < think-hard 27K << xhigh 65-73K. The two phrases fill
  the middle the template leaves empty.
- Reading: for a human in a session the phrases are the better per-task knobs (graded, zero prefix cost, no template); the trained
  xhigh sentence is the "may not finish in budget" setting. The agent clients emit the level, so the tail template keeps its place.

### Phrases inside a tool session (owner: "how would I switch back to medium after having said whatever xhigh expands to?"; 21:30-22:05)

Two more pilot10 recordings at level medium on the stock template: A plain (the medium agent baseline), B with phrases in user turns
(`perf/effort-pos-pilot10-phrases.user.json`): u3 the trained xhigh sentence, u4 plain (persistence), u5 "Reasoning effort is set to
medium." (a switch-back candidate; the template has no medium sentence), u6 "Keep it simple, don't overthink it.", u7 plain, u8 "I want
you to think really hard about this.", u9 "Normal effort from here on - think as much as the problem needs, no more." B is byte-identical
to A through u2 (deterministic until the first phrase). Thinking tokens per user turn:

| user turn | phrase in B | xhigh recordings | low | medium (A) | phrases (B) | B/A |
|---|---|---:|---:|---:|---:|---:|
| u3 | xhigh sentence | 1322-1677 | 1188 | 603 | 2104 | 3.5x |
| u4 | (plain) | 855-1265 | 654 | 518 | 796 | 1.5x |
| u5 | "effort is set to medium." | 1560-1666 | 562 | 995 | 1166 | 1.2x |
| u6 | "don't overthink it" | 1036-1068 | 211 | 221 | 574 | 2.6x |
| u7 | (plain) | 317-342 | 172 | 198 | 150 | 0.8x |
| u8 | "think really hard" | 1201-1239 | 1350 | 1038 | 1628 | 1.6x |
| u9 | natural reset | 218-264 | 180 | 35 | 259 | (A is the outlier) |
| total | | 7800-8419 | 5175 | 4271 | 7340 | |

- **Medium is the leanest agent regime** (A: 29 turns, 19 tool calls, 12.7K generated, 4271 thinking = 34%; low 5175, xhigh 7800-8419).
- The xhigh sentence works mid-session from a user turn (3.5x), carries into the next plain turn (1.5x) and is gone two turns later
  (0.8x at u7). **Switching back = say nothing; it decays within ~2 turns.** The "set to medium" sentence landed at 1.2x - compatible
  with helping, not separable from the decay; it did not hurt.
- "Think really hard" is 1.6x in-session, its Q&A ratio. "Don't overthink it" did NOT brake here (2.6x A at that turn vs 0.57x
  single-turn). Replication C below says why.

### Replication with the phrases on other turns (C, `perf/effort-pos-pilot10-phrases2.user.json`, 22:35-22:55)

u2 "don't overthink it" with a clean medium history, u3 plain, u4 the xhigh sentence, u5 "set to medium" at distance 1, u6 plain,
u7 "don't overthink it" at distance 3 from the xhigh turn. Identical to A through u1.

| turn | B phrase | C phrase | xhigh recordings | A | B | C |
|---|---|---|---:|---:|---:|---:|
| u2 | - | don't overthink (clean) | 449-1089 | 477 | 477 | 109 (0.2x) |
| u3 | xhigh sentence | plain | 1322-1677 | 603 | 2104 (3.5x) | 84 (0.1x) |
| u4 | plain | xhigh sentence | 855-1265 | 518 | 796 (1.5x) | 419 (0.8x) |
| u5 | "set to medium" | "set to medium" (d1) | 1560-1666 | 995 | 1166 (1.2x) | 704 (0.7x) |
| u6 | don't overthink | plain (d2) | 1036-1068 | 221 | 574 (2.6x) | 238 (1.1x) |
| u7 | plain | don't overthink (d3) | 317-342 | 198 | 150 (0.8x) | 156 (0.8x) |
| u8 | think hard | plain | 1201-1239 | 1038 | 1628 (1.6x) | 699 (0.7x) |
| u9 | reset | plain | 218-264 | 35 | 259 | 222 |
| total | | | 7800-8419 | 4271 | 7340 | 2817 |

**Phrases steer the session, not the turn (history anchoring).** The same sentence, two histories: the xhigh sentence after a medium
history 3.5x (B u3), after a terse history 0.8x (C u4). "Don't overthink it" after a medium history 0.2x and the next plain turn 0.1x
(C u2-u3); after an xhigh-regime history 2.6x (B u6). One phrase at u2 ran C's whole session at 66% of medium; one sentence at u3 ran
B's at 172%. The likely mechanism: the stock template keeps every earlier turn's `reasoning_content` in the history
(`preserve_thinking` defaults on), so the model sees its own recent thinking lengths as the local norm; a phrase moves the norm, later
turns inherit it, a counter-phrase one turn later fights the visible history. B's "decay" was the anchor drifting back as the tasks
got smaller, not the sentence wearing off.

Use: say it once, early; expect it to stick; to switch back say the opposite (not "medium") and allow a turn or two.

### The mechanism test: `preserve_thinking=false` (D, 2026-10-04; owner: "You'll track the memory and amount of prefill?")

C's script again with `chat_template_kwargs: {"preserve_thinking": false}` (`run-session.sh TEMPLATE_KWARGS`, `session.py
--template-kwargs`): assistant turns before the latest user message render without their `<think>` block (the current tool loop keeps
its own). Identical to C through u0.

| turn | phrase | A think | C think (retained) | D think (stripped) | C prefill tok (s) | D prefill tok (s) | C ctx end | D ctx end |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| u0 | | 90 | 90 | 90 | 3479 (29) | 3479 (29) | 4351 | 4351 |
| u1 | | 96 | 96 | 96 | 34 (1) | 3800 (29) | 4678 | 4576 |
| u2 | don't overthink (clean) | 477 | 109 | 193 | 4491 (37) | 3267 (27) | 9735 | 8060 |
| u3 | plain | 603 | 84 | 487 | 37 (1) | 3621 (29) | 10237 | 8701 |
| u4 | xhigh sentence | 518 | 419 | 754 | 1466 (13) | 2443 (21) | 13083 | 12134 |
| u5 | "set to medium" | 995 | 704 | 1290 | 112 (2) | 3585 (30) | 14957 | 14210 |
| u6 | plain | 221 | 238 | 679 | 5607 (51) | 8079 (71) | 21648 | 20352 |
| u7 | don't overthink | 198 | 156 | 169 | 1873 (18) | 11266 (99) | 24045 | 22687 |
| u8 | plain | 1038 | 699 | 1241 | 67 (1) | 3376 (31) | 25300 | 24235 |
| u9 | plain | 35 | 222 | 604 | 30 (1) | 452 (5) | 26118 | 24226 |
| total | | 4271 | 2817 | 5603 | 17196 (153 s) | 43368 (372 s) | | |

- **Prefill 2.5x**: every user turn re-prefills the previous loop (2.4-3.8K tokens on ordinary turns, 11.3K / 99 s after the big
  tool-output loop) where C re-prefilled 30-112. Prefix saves (system block) are untouched; the slot's own cache is discarded per turn.
- **Context only 7% smaller** (24.2K vs 26.1K): this session retained little thinking (C 2.8K) and D generated a third more (12.0K vs
  8.9K). The saving scales with the thinking a session carries (an xhigh pilot10 would shed ~20%); it never pays for the prefill.
- **Anchoring confirmed**: with the model's prior reasoning out of the history the terse regime does not form (u3 0.1x -> 0.8x) and
  the xhigh sentence takes where C's terse history suppressed it (u4 0.8x -> 1.5x). The phrase text itself stays in the history and,
  with nothing to anchor against it, the xhigh sentence lifts the two following turns (u5 1.3x, u6 3.1x): phrases become closer to
  per-turn, not independent. D thought 2x C: without its notes the model re-derives (retention is trained behaviour).
- Not a serving option; the mechanism question is closed.

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
