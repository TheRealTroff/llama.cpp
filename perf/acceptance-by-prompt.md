# Acceptance is a property of the workload, not the engine (2026-08-23)

Status: **open** - an indication, not a distribution study. Five short prompts, one depth
(n6), one run each. Enough to size the effect and kill two wrong claims; not enough to set
a policy.

Harness `perf/run-accpos-trace.sh` (`PROMPT=` selects the workload), prompts in
`perf/prompts/`. Acceptance is kernel-independent - every routing and depth emits the same
output sha - so these numbers stay valid across kernel work and are NOT load-sensitive.

## The numbers

`acc per pos` is a survival curve: entry i is the fraction of rounds with at least i+1
drafts accepted, so the sum is mean accepted drafts and committed/rd is that plus one bonus.

| prompt | prompt_n | acc | committed/rd | acc per pos | sum |
|---|--:|--:|--:|---|--:|
| `benchprompt.txt` (code-summary) | 8288 | 46.9% | 3.75 | (.899 .709 .519 .304 .215 .139) | 2.79 |
| `01-code-explain` | 181 | 35.9% | 3.12 | (.779 .547 .337 .242 .147 .095) | 2.15 |
| `02-prose-creative` | 67 | 30.5% | 2.80 | (.708 .425 .264 .179 .151 .094) | 1.82 |
| `03-chat-support` | 86 | 30.2% | 2.79 | (.723 .455 .317 .178 .079 .059) | 1.81 |
| `04-math-derivation` | 90 | 89.7% | 6.25 | (.979 .936 .936 .894 .851 .766) | 5.36 |
| `05-json-boilerplate` | 93 | 98.8% | **6.67** | (1.00 1.00 1.00 .977 .977 .977) | 5.93 |

**Committed tokens per round span 2.79 to 6.67** on the same model, drafter and depth.

## What extra width is worth, by workload

Sum of positions 4-6 - what you buy going from width 4 to width 7:

| workload | tokens gained |
|---|--:|
| `03-chat-support` | **0.32** |
| `02-prose-creative` | 0.42 |
| `01-code-explain` | 0.48 |
| `benchprompt` | 0.66 |
| `04-math-derivation` | 2.51 |
| `05-json-boilerplate` | **2.93** |

**A ~9x spread in the value of depth.** `05` commits 6.67 of a possible 7 at n6 - saturated,
and it wants to go deeper. `03` buys a third of a token for three extra columns and wants to
be narrow. There is no single correct width across this set.

## Two claims this kills

1. ~~The benchmark prompt is a high-acceptance sample.~~ **Wrong.** At 46.9% it is mid-range:
   below math and JSON, above all three free-form prompts. Its position-1 of 0.899 looked
   high only because nothing else had been measured.
2. ~~Adaptive width is a mistake of theirs, not a lever for us.~~ **Scope error.** True on the
   single prompt `block4-shelf-probe.md` measured, where pinning beat adaptive by 9.7% at no
   acceptance cost. Across this spread the optimal depth runs from ~3 to beyond 7, so on
   mixed traffic a controller is worth real money. MLX's adaptive looks less like a mistake
   and more like a bet on varied workloads our single benchmark cannot see.

Note this does NOT rehabilitate `LLAMA_SPEC_ADAPTIVE` as written: it shortens `n_draft_max`
(`server-context.cpp:3031-3037`), and per `accept-per-pos-curves` shortening the *draft*
walks the drafter out of distribution. The lever is still draft-deep / verify-narrow.

## Confounds and what is missing

- **Context length is confounded with style.** These five are 67-181 prompt tokens against
  benchprompt's 8288. The five compare cleanly to *each other*; comparing them to benchprompt
  mixes two variables. Re-run with length-matched prompts before quoting any cross-comparison
  with benchprompt.
- One run per prompt, one depth. No repeats, no error bars.
- No cost data: these are acceptance only. Optimal width per workload needs the round-cost
  curve, and for short-context prompts that curve is not the one in `slope-sweep.md` (tiny KV,
  much cheaper rounds).
- The set is a guess at "dissimilar", not a sample of real traffic. If the actual workload is
  mostly code, `benchprompt` is closer to representative than this spread implies.

## Why it matters for width 4

The low-acceptance end (chat, prose) is exactly where narrow width is correct - and exactly
where our kernel is worst (`occupancy-next.md`: width 4 costs 141.0 ms/round against their
95.00). The high-acceptance end wants width 7-8, where our 8-wide tile is nearly full and we
are fine. **So the tile problem specifically taxes conversational and creative traffic**, and
a width-4 kernel is worth more on that traffic than the benchmark suggests.
