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

## Refreshed at the adopted pick: n4 + drafter window (2026-08-28, TAG `corpacc-aug28`)

Harness `run-corpus-acceptance.sh`: every prompt run twice at depth 4, with the pick's
`LLAMA_DRAFT_WINDOW=1024` and without. n_predict 300, same method as above.

| prompt | acc | committed/rd | acc per pos (n4) | vs n6 committed/rd |
|---|--:|--:|---|--:|
| `benchprompt` window | **57.5%** | **3.26** | (.890 .692 .429 .275) | 3.75 |
| `benchprompt` no-win | 55.7% | 3.19 | (.849 .667 .430 .269) | |
| `01-code-explain` | 49.8% | 2.97 | (.760 .540 .400 .290) | 3.12 |
| `02-prose-creative` | 40.1% | 2.59 | (.713 .443 .270 .174) | 2.80 |
| `03-chat-support` | 42.6% | 2.69 | (.733 .486 .324 .162) | 2.79 |
| `04-math-derivation` | 91.1% | 4.55 | (.969 .923 .892 .815) | 6.25 |
| `05-json-boilerplate` | 97.1% | 4.76 | (1.00 .984 .967 .934) | 6.67 |

Three findings:

1. **The window is INERT on the whole tiny corpus - measured, not assumed.** Win and
   no-win arms are byte-identical (sha, acc, survival curve) on all five prompts:
   `apply_window` returns until n_past > sink+window = 1088, and prompt+300 never gets
   there. The window's acceptance gain exists only at benchprompt scale, where it is
   real: 55.7 -> 57.5, committed/rd 3.19 -> 3.26, and the improvement is concentrated
   at POSITION 1 (.849 -> .890) - trimming stale context helps the first draft most.
2. **Depth 4 taxes the saturated workloads hard.** Math and JSON, which committed
   6.25/6.67 per round at n6, commit 4.55/4.76 at n4 (the ceiling is 5) - ~35% more
   rounds on exactly the traffic where rounds are cheapest to amortize. The n4
   optimum was measured on benchprompt-style text; on a math/JSON-heavy workload the
   depth re-sweep would likely land deeper. Reinforces this file's adaptive-depth
   conclusion; the pick stays n4 for the benchmark workload.
3. **The n4 tail survives better than n6's tail did.** At matched positions 3-4,
   n4's curves sit above n6's on every free-form prompt (e.g. 01: .400/.290 vs
   .337/.242) - the shorter noise block conditions the tail positions better. Part
   of why n4 wins e2e despite committing less per round.
