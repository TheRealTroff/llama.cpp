# Benchprompt: the framing line, the restart, and the chat template (2026-09-17)

Status: **the benchmark lineage moved on it (section 6); the picks themselves unchanged.** Found while scoring the adaptive-depth agreement corpora
(`spec-verify-narrow.md` section 10): the shared benchprompt made both corpora score mean KLD 0.28 with chunk-1 PPL
~1000 under q8_0 and the pick alike.

## 1. The instruction line makes the code unlikely text

`llama-perplexity -c 2048`, q4 f16 pick, chunk 1 = the first 2048 tokens of the file (scored half = the code at
positions 1024-2047, context = the file start). The same whisper.cpp `command.cpp` source under different first lines:

| first line | chunk-1 PPL |
|---|--:|
| none (line removed) | 1.26 |
| `// file: examples/command/command.cpp` | 1.26 |
| `Here is a file from the whisper.cpp repository.` | 1.25 |
| `What does the following C++ code do?` + blank line | 2.20 |
| `What does the following C++ code do?` + newline | 4.43 |
| `Explain the following code in detail.` + blank line | 238 |
| `Summarize what this does:` + newline | 359 |
| **`Summarize what this does: ` + newline (the benchprompt, trailing space)** | **985** |

The model knows the file nearly verbatim (1.26). A prose line that announces a file leaves that intact; a question
that announces code costs 2-4x; an instruction that announces nothing about what follows makes every code token
~1e-3 likely - the model's mass sits on ending the code and starting the answer. The exact tokens matter 3x (the
trailing space), the same surface sensitivity as the owner's EOS-first finding on 01-code-explain (a trailing newline).
At `-c 512` (no instruction in most contexts) the region scores 5-100: the code is "easy" only as a recalled file.
Consequence for the harnesses: prompts stay out of agreement corpora (`run-specev-pick-gate.sh` defaults to the four
free-form prompts); a raw-completion prompt is unlikely text under the model, and its positions swamp the completion's.

## 2. The greedy raw completions are degenerate, and the stack is not the cause

600-token greedy completions on raw `/completion` (q4 pick, Turbo4 + the controller; and the no-spec f16 control):

| prompt | pick (Turbo4, controller) | no spec, f16 cache |
|---|---|---|
| benchprompt as is | 33.5 t/s, acc 65.1%, sha `b40a84e252af` (canonical): the summary **restarts once** at "### 2." ("Here is a summary of the provided C++ code...") | sha `5f32a6b9d371` (canonical f16 600): the same restart |
| `What does the following C++ code do?` + blank line + code | 36.5 t/s, acc 65.8%: **a repetition loop** after ~100 tokens ("This is a C++ program that implements..." x17, one stray CJK token) | the same loop, x17 |

Every fixed-depth text on disk since Aug 28 (b1 f16 300, f16 600, Turbo4 600) has the single restart: **the canonical
benchmark text has been a restarting summary since the acch mint.** The August text (the partial arm's
`9ad7e023c6ab`, pre-acch lineage) has no restart.

## 3. The restart is the half-accumulate lineage (the bisection, one step)

Same binary, f16 cache, no speculation, the pick env with `GGML_MM_ACC_HALF` unset (presence-based flag):
600 tokens, **no restart**, sha `b6b839196da8`. With it: `5f32a6b9d371`, the restart. On the f16 line acch is the
only numerics move since August, so this is the whole bisection: the +11.8% mean KLD / -0.86 pt same-top of
`kldacch-aug28` includes tipping this trajectory into a restart before token 250. One sample of one prompt, the
owner's "small differences yield dramatically different outcomes" made visible; the acch decision (q4 only, taken
2026-08-28) is the owner's and is not reopened here.

## 4. The chat template gives clean summaries under both questions

`/v1/chat/completions`, `enable_thinking: false` (a thinking model: with thinking on, all 600 tokens go to
`reasoning_content` and `content` is empty), greedy, the q4 pick:

| user message | t/s | acceptance | text |
|---|--:|--:|---|
| "Summarize what this does:" + code | 31.4 | 61.9% | complete structured summary, no restart, no loop (`3bf7fcb7beec`) |
| "What does the following C++ code do?" + code | 35.5 | 67.7% | complete structured summary, no restart, no loop (`f1a194448e36`) |

So "attending to the right things" is the template: the raw-completion regime is where the instruct model degenerates
under greedy decoding, and the prefix only picks the failure mode. Speculation accepts as well or better there (62-68%).

## What this changes

- Nothing in the picks or the shas: the benchmark prompt is a byte string with a lineage; it stays.
- Quality judgments on generated text use the chat template (thinking off) or sampling, never raw greedy completion.
- Agreement corpora exclude prompts.
- Open (owner's): whether the q4 line's acch is worth a second look given that one greedy trajectory it tips; a
  benchmark prompt file under the chat template would be a new lineage.

## 5. Controls (owner: "before the half-acc we were fine without the chat template?" - no)

| arm | result |
|---|---|
| q4 f16 no-spec, acch UNSET, benchprompt raw, 2048-token cap | 1396 tokens then EOS, no "### 2." restart, but the finished summary is **repeated ~15 times** from ~600 tokens on (`059d3a6fd092`) |
| q4 f16 no-spec, acch unset, question framing raw, 600 | 505 tokens then EOS, the summary repeated once (`4766ad025da3`) - milder than the x17 loop with acch |
| q4 Turbo4 pick, acch unset, chat template "Summarize" | clean, 33.4 t/s, acc 63.4% |
| ud f16 no-spec, benchprompt raw, 600 | **clean** (`5e76afaba36c` = the UD f16 canonical text): no restart, no repeat through 600 tokens |
| ud f16 no-spec, question framing raw, 600 | clean, ends at EOS after 151 tokens |
| ud Turbo4 pick, chat template, both questions | clean, 29.4 / 30.7 t/s, acc 68.9 / 72.6% |

Reading: on q4 the raw greedy regime degenerates with or without acch - acch moves the first failure from a
repeated summary after ~600 tokens to a restart at ~250 and a full loop; it does not create the fragility. On UD
the raw greedy text is clean through 600 tokens on both framings (no acch there by design, and a different
quantization - not a controlled pair with q4). The chat template is clean on both lines, with and without acch,
at 63-73% acceptance. "Before acch we were fine without the template" holds only for the first 600 tokens of one
prompt on q4; "with the template we are fine" holds everywhere measured.

## 6. The switch: every benchmark prompt chat-templated (owner, 2026-09-17 evening)

"The benchprompt is a clear question and what we want is its answer." `pick_prompt` in `perf/pick.sh` renders
`<|im_start|>user\n` + content.strip() + `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` - byte-identical
to the server's `/apply-template` (thinking off) on benchprompt, the corpus prompts and the 96K prompt (the first
wrapper kept the file's trailing newline and was one byte off on every file: the template trims). The rendered
file through `/completion` reproduces the chat endpoint's text exactly (3bf7fcb7beec). For a single user turn with
thinking off the template contributes nothing else; with thinking on it also injects a system turn ("Reasoning effort
is set to xhigh. Please think carefully...") and ends at an open `<think>` - a different lineage if ever benchmarked.
Harnesses: run-prod-pick.sh, run-depth-corpus.sh, run-longctx-pick.sh (`PICK_CHAT=0` = raw). First baselines
(`run-chat-lineage-baselines.sh`): the README pick block entry of 2026-09-17 evening. Highlights: both corpus
prompts that emitted EOS first on raw prompts (01, 07) answer; acceptance up on every arm (ud Turbo4 300 73.5%);
the q4 controller +8.0% on the nine-prompt corpus with free-form no worse than -2.5% and 7 of 9 texts identical to
fixed depth; ud's JSON text byte-identical to q4's (`0b9f71ff9941`, the first cross-line sha match on record).
