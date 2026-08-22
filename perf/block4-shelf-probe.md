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

`verify_mode` takes `dflash|adaptive|ddtree|off`. `adaptive` is the default and is what
every archived run used; **`dflash` is fixed-block, non-adaptive** (`engine/spec_epoch.py:340`
returns no policy unless the mode is `adaptive`).

> **MEASURED 2026-08-22, do not repeat my mistake: the `DFLASH_VERIFY_MODE` env var is
> IGNORED by `dflash benchmark`.** `_resolve_verify_mode` (`runtime/config.py:759`) checks
> the CLI value first and only falls back to the env var when that value is `None`, and the
> benchmark CLI always supplies one. My first fixed-block-5 arm silently ran adaptive and
> reproduced the adaptive number exactly. **Use the `--verify-mode dflash` CLI flag**
> (`benchmark.py:1787`), and **always verify the mode took** by checking
> `adaptive_metrics.cycles_by_block` in the artifact: it must be absent/empty. If it reads
> `{1:1, 4:81, 5:17}` per run, the controller was live and the arm is not fixed-block.
>
> Note `--block-tokens 4` is fixed regardless, for an unrelated reason: `from_runtime` also
> bails at `full_block_tokens <= 4` (`spec_epoch.py:343`). That is why the block-4 arm was
> genuinely pinned even though the env var did nothing.

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

## Pre-registered bound: (b) cannot fully rescue us

Written before the run. Misattribution has to conserve the total, which is what makes the
2.7% reconciliation useless as evidence - but conservation also *constrains* it. Let `d` be
the ms per block-4 cycle deferred into the block-5 cycles that force the sync. Then
`T4 = 91.9 + d` and, spreading 81 cycles of deferral over 17 block-5 cycles,
`T5 = 140.2 - (81/17)d`. A block-5 cycle cannot be cheaper than a block-4 one, so
`T5 >= T4` gives `48.3 >= 5.76d`, i.e. **`d <= 8.4 ms`** and therefore
**`T4 <= 100.3 ms`**.

So even under the most misattribution the data permits, their depth-4 cycle costs between
**91.9 and 100.3 ms** against our n4's **144.9**. At their 3.049 tokens/cycle that is
**30.4 to 33.2 t/s**. Predictions:

- **33 t/s** -> shelf real as measured, `d ~ 0`.
- **30.4-33 t/s** -> shelf real but partly misattributed; the cheap cycle still stands and
  the programme is the same.
- **below 30.4** -> the cost model itself is wrong, not just the split. Most likely cause
  would be that fixed block 4 accepts worse than adaptive block 4 (the controller may be
  picking its moments), so check `accepted_from_draft` and `tokens_per_cycle` before
  concluding anything about cycle cost.

Note this bound means **(b) was never a way for the gap to be illusory** - it only changes
how much of the shelf is real, not whether their shallow cycle is far cheaper than ours.

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
