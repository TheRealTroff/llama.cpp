# Is dflash_mlx's cheap block-4 cycle real, or a lazy-eval artifact?

Status: **open**

Opened 2026-08-22 by `verify-slope-close.md`, which closed the verify-slope lever and moved
the whole remaining gap onto this one question. **Do this before building anything** - it
decides which of the two remaining programmes is worth funding, and it is a single run on
their side with no code changes on ours.

## The question

Their archived `adaptive_metrics` (unlike `phase_timings_us`, which is unusable - see the
Task 2 section of `verify-slope-close.md`) reconciles to 2.7% of wall and gives their cost
curve:

| block | their cycles | their ms/cycle | their tok/cycle | ours | our ms/round | our tok/round |
|---|--:|--:|--:|---|--:|--:|
| 1 | 1 | 72.4 | 1.00 | b1 | 73.2 | 1.00 |
| 4 | 81 | **91.9** | 3.049 | n4 | 144.9 | 3.19 |
| 5 | 17 | 140.2 | 3.059 | n5 | 147.5 | 3.45 |

At matched depth 5 **we are the faster engine** (23.37 vs 21.82 t/s). Their entire 1.184x
edge is the block-4 row, where their controller spends 82% of cycles. But their block 4 ->
5 step costs +48.3 ms while our own width 5 -> 6 step costs +1.9 ms, and a cheap 5-column
verify contradicts the per-shape microbench parity on record (`head-to-head-aug22.md`: MLX
n=5 slope 1.74x vs our 1.81x). So either:

- **(a) the shelf is real** - their cycle computes less than a full 5-column verify, and
  building an equivalent cheap operating point is the whole remaining game; or
- **(b) it is lazy-eval misattribution** - if the sync lands on full-mode (block-5) cycles,
  work deferred from preceding block-4 cycles is charged to block 5, producing exactly this
  shape. **The 2.7% agreement on the weighted total does not rule this out**, because
  misattribution conserves the total and only moves the split.

## The run

`DFLASH_VERIFY_MODE` takes `dflash|adaptive|ddtree|off` (`runtime/config.py:759`, resolved
from CLI or that env var). `adaptive` is the default and is what every archived run used;
**`dflash` is fixed-block, non-adaptive**, which pins the cycle mix and removes the
controller from the comparison.

Run fixed block 4 and fixed block 5, 5 reps each, same 8288-token prompt, via
`run-head-to-head.sh` (it regenerates the MLX-side prompt jsonl from `benchprompt.txt`
every run, so the two sides cannot drift - that mismatch was the original `results.md`
error). Compare against the adaptive baseline of 29.613 already measured there.

Decision rule:

- fixed block 4 near **33 t/s** -> **(a)**, the shelf is real. Next question becomes what
  their block-4 cycle skips, and whether our width 2-5 costs can be brought down to match.
- fixed block 4 near **29-30 t/s** -> **(b)**, the by-block split was distorted. Their real
  advantage is the adaptive policy visiting cheap widths, and `slope-sweep.md`'s adaptive
  prerequisite - flatten verify widths 2-5 first - is the correct programme. Two closed
  lines reopen under that reading, both closed *because* we only ever run one width: the
  mv-nc NC>=3 cliff, and `GGML_MV_NC_V2`, which was only ever A/B'd at `GGML_MV_NC=2` so
  whether its lower register pressure relieves the NC>=3 spill penalty is still untested.

Either outcome is worth recording; there is no null result here.

## Gotchas that apply to this run

- The `dflash` CLI has **no `--version`** (it prints usage into the log). Read the installed
  dist instead: `~/play/omlx/.venv/bin/python -c "import importlib.metadata as m;
  print(m.version('dflash-mlx'))"`. Expect 0.1.10+omlx.6.
- **Record a commit sha with every number**, ours and theirs.
- `n_predict` is not comparable across harnesses (300 here, 600 in the GDN harnesses).
- A leftover `llama-server` answers `/health` and the next run then silently measures *that*
  server's config; llama-server ignores SIGTERM during Metal teardown.
- Their raw per-run counters live in
  `~/play/.artifacts/dflash/benchmarks/<ts>/results.json` under
  `prompts[0].runs[i].dflash`. Check `cycles_by_block` actually collapses to a single block
  under `dflash` mode - if it does not, the mode did not take.
