# Sharp chat template eval (peculiar-ragdoll/Qwen-Sharp-Chat-Templates)

Qwen3.8-27B Q4_0, llama-server (metal-mv-ext-nr0 kernels, MTP n-max 2), temp 0, max_tokens 4096.
Stock = template embedded in the unsloth GGUF; Sharp = `--chat-template-file ~/play/sharp_chat_template.jinja` (v22.1).
Applied-check: /props showed the terseness block present. Counts are deterministic (reproduced exactly across reruns).

| prompt | stock tokens | sharp tokens | delta | notes |
|---|---:|---:|---|---|
| btree vs b+tree | 3057 | 1374 | -55% | both answer; sharp answer is complete and accurate |
| asyncio mem leak | 4096 | 2456 | -40% | stock burned the WHOLE budget thinking, produced NO answer; sharp thought 4.3k chars then wrote a 5.4k-char answer |
| tank fill math | 358 | 420 | +17% | both correct (2.4 h); sharp leads with the answer; small tasks have no fat to trim (matches the repo's own caveat) |

Quality read (sharp answers, full text in tmp/tmpl_sharp_full.json):
- btree: correct structural table (data-in-internal-nodes, linked leaves, fan-out/height), correct
  engine facts (InnoDB/PostgreSQL/Oracle/SQL Server/SQLite use B+ trees), sound selection guidance.
  Nothing a 2x longer answer would have added.
- debug: ranked causes are the real ones (unbounded collections; unreferenced/unreaped tasks incl.
  exception tracebacks pinning frames; unclosed transports; __del__ cycles; call_later closures;
  queue backpressure) plus a concrete gc.collect/RSS diagnostic workflow with code. Genuinely good.
- math: equivalent to stock, terser framing.

Verdict: the headline claim replicates on this setup. The big win is not style - it is that the
terseness instruction reins in runaway thinking: the stock template's failure mode (budget
exhausted before any answer) simply did not occur under Sharp. Cost: a few percent overhead on
tasks that were already terse.

## Planted-error (error persistence) test

Design: a fabricated prior assistant turn carries plausible-but-wrong reasoning in
reasoning_content (arithmetic slip 900 vs 1000 L/h; overconfident wrong debug hypothesis;
wrong M1 launch year) while its visible answer stays vague; the next user turn asks a
dependent question. Both templates, temp 0, Qwen3.8-27B Q4_0.

Surprise finding that reframes the retention concern: the STOCK Qwen3.8 template also renders
prior-turn reasoning_content (its removal is gated on a `preserve_thinking` flag that callers
never set, defaulting to retain-all). Thinking retention is official Qwen3.8 behavior, not a
Sharp addition - froggeric's retention fix mattered for 3.6. So on 3.8, Sharp adds NO new
retention risk, and the context-burn cost applies to stock equally.

Anchoring result: zero anchoring in all 6 runs. Both templates explicitly caught the planted
arithmetic slip ("my earlier 900 L/h was an arithmetic slip"), both answered 2020/6 for the M1
despite retained "2019" reasoning, and Sharp pivoted cleanly off the planted pool-exhaustion
theory to the correct worker-crash diagnosis. Qwen3.8 appears trained to re-verify retained
reasoning rather than trust it (consistent with retention-by-default + multi-step MTP training).

Sharp recovered with fewer tokens every time (656 vs 861; 54 vs 324) and was the only template
to complete the debug answer - stock burned its entire 2048-token budget thinking without
answering (third sighting of that stock failure mode).

Caveats: three scenarios, one-turn gap, temp 0. Subtler errors, deeper histories, or
sampling could still surface persistence; the acute version does not reproduce.

Not yet tested: multi-turn cache hits / TTFT in practice, and tool-calling.
Timing note: wall times in the first eval were contaminated by a concurrent benchmark session
on the same GPU - token counts are the reliable metric in this file.

To use routinely: add `--chat-template-file /Users/troff/play/sharp_chat_template.jinja` to
llama-server, or bake it into the GGUF with gguf-new-metadata (see the repo README).
