# Acceptance, converted properly: we are at or ahead of oMLX (2026-08-21)

Answers "how does acceptance compare to MLX, compensating for the different metric?"
Supersedes the ~0.71-vs-0.67 note in `draft-sink-window.md`, which was directionally
right but used a wrong denominator for the oMLX side.

> **CORRECTED 2026-08-22 (`mlx-cycle-capture.md`). Every row-to-row comparison in this
> file is off by one column.** Their block *b* verifies *b* columns
> (`spec_epoch.py:2247-2257`); our depth *d* verifies *d+1* (`slope-sweep.md:13`). So the
> width-matched partner of their `blk 4` is our **n3**, not our n4, and of their `blk 5` is
> our **n4**, not our n5. Matched by width the table below reads **their blk 4 (1.36x
> floors) vs our n3 (1.99x)** and **their blk 5 (2.07x) vs our n4 (2.15x)** - i.e. we are
> nearly level at width 5 and the whole gap is at width 4, which is what
> `block4-shelf-probe.md` later measured directly. The acceptance conclusion is unaffected
> (it compares drafters, not widths); the *cost* ratios at lines 68-72 are all off by one
> and need re-deriving at a stated width.
>
> **Also: their per-block rows here are derived from `adaptive_metrics`, and the block-4 row
> is superseded by direct measurement.** `block4-shelf-probe.md` pins block 4 and measures
> **95.00 ms/cycle, 3.0928 tok/cycle, 32.556 +/- 0.007 t/s**, against the 91.9 / 3.049 /
> 33.17 derived below. The 3.1 ms difference is MLX lazy-eval deferral. **Quote 32.556, not
> 33.17.**

## Their metric, verified from raw counters

`.artifacts/dflash/benchmarks/20260821-182712-.../results.json`, run 0 (all 5 runs
identical): `accepted_from_draft=201`, `cycles=99`, `tokens_per_cycle=3.0303`,
`acceptance_ratio=0.67`.

201/300 = 0.670 exactly, so **their metric is accepted / committed**. Ours
(`draft_n_accepted / draft_n`) is **accepted / attempted**. Confirmed, not assumed.

## The denominator trap: their attempted is NOT cycles x 5

`verify_mode` defaults to **`adaptive`** (`runtime/config.py:95`, `:229` -- "Default
adaptive shortens low-acceptance blocks"). Despite `--block-tokens 5`, the archived run
records `cycles_by_block = {1: 1, 4: 81, 5: 17}` and `cycles_by_mode = {full: 7,
probe: 12, reduced: 80}`.

So attempted = 1x1 + 81x4 + 17x5 = **410**, not 99x5 = 495, and their average draft
depth is **4.14**, not 5. Using 495 understates their per-draft acceptance as 40.6%;
the true figure is 201/410 = **49.0%**.

## Both metrics, both sides

Ours: pure-Q4_0 drafter, 300 committed tokens per run, `GGML_MV_NC=2 GGML_MM_SKINNY=5`,
8288-token prompt. Cycles derived as committed - accepted (each cycle commits the
accepted drafts plus one target token); the implied attempted/cycle recovers n_max to
within rounding, which validates the derivation. Floors: ours 73.38 ms/token measured
today (`--spec-type none`); theirs 67.66 ms from the cooled head-to-head, not re-measured.

| side            |  n / blk | att/cyc | acc/att | acc/com | tok/cyc | cycle ms | /floor | t/s   |
|-----------------|---------|---------|---------|---------|---------|----------|--------|-------|
| llama.cpp       | 2       | 1.98    | 72.5%   | 0.590   | 2.439   | 125.7    | 1.71x  | 19.40 |
| llama.cpp       | 3       | 2.96    | 63.6%   | 0.653   | 2.885   | 146.2    | 1.99x  | 19.74 |
| llama.cpp       | 4       | 3.94    | **55.7%** | 0.687 | 3.191   | 157.7    | 2.15x  | 20.24 |
| llama.cpp       | 5       | 4.91    | 49.9%   | 0.710   | 3.448   | 163.0    | 2.22x  | 21.16 |
| llama.cpp       | 6       | 5.86    | 46.9%   | 0.733   | 3.750   | 169.1    | 2.30x  | 22.17 |
| llama.cpp       | 7       | 6.83    | 40.3%   | 0.733   | 3.750   | 175.2    | 2.39x  | 21.41 |
| dflash_mlx      | blk 4   | 4.00    | **51.2%** | 0.672 | 3.049   |  91.9 [d] | 1.36x | 33.17 [d] |
| dflash_mlx      | blk 5   | 5.00    | 41.2%   | 0.673   | 3.059   | 140.1 [d] | 2.07x | 21.83 [d] |

[d] = derived from `adaptive_metrics`, **superseded by pinned measurement**
(`block4-shelf-probe.md`): block 4 is **95.00 ms / 3.0928 tok / 32.556 t/s**, block 5 is
**137.26 ms / 3.6585 tok / 26.654 t/s**. Their block *b* is width *b*; our `n` rows are
depth, i.e. width n+1.
| **dflash_mlx**  | **ALL** | **4.14**| **49.0%** | **0.670** | **3.030** | **102.5** | **1.52x** | **29.55** |

oMLX per-block rows are reconstructed from their own `adaptive_metrics`
(`cycles_by_block` x `tokens_per_cycle_by_block`, timed via `tokens_per_second_by_block`);
they reproduce the ALL row exactly, but the block-5 row is only 17 cycles and is
mode-biased (mostly probe/full, which carry controller overhead), so treat it as
indicative rather than a clean depth-5 cost.

## Answer

**At matched draft depth we accept slightly better, not worse.** Depth ~4: ours 55.7%
vs theirs 51.2% per draft (0.687 vs 0.672 on their metric). Their headline 49.0% sits
between our n5 (49.9%) and n6 (46.9%) because their adaptive controller averages 4.14
attempts/cycle.

The naive headline comparison -- our 0.733 vs their 0.670 -- **overstates our advantage**.
Our accepted/committed rises with depth (0.590 at n2 to 0.733 at n6) purely because
deeper drafting commits more per cycle; theirs is nearly flat at 0.67 across blocks.
Comparing the two headline numbers compares depth choices, not drafter quality.

**So drafter quality is not the gap. The entire 1.33x is cycle cost**, and the exact
decomposition confirms it (predicted = observed to three decimals):

    matched depth 5:  cycle 1.589x against us  /  committed-per-cycle 1.138x for us  = 1.397x  (obs 1.397x)
    our best n6:      cycle 1.649x against us  /  committed-per-cycle 1.238x for us  = 1.333x  (obs 1.333x)

Sharpest framing: ~~**a speculation cycle costs us 2.15-2.30 batch-1 passes; theirs costs
1.36-1.52.** We buy roughly the same tokens per cycle, and pay ~1.6x more wall time for
them.~~ **OFF BY ONE - the two ranges are different stretches of curve.** Ours is n4..n7 =
widths 5..8; theirs is blk 4..5 = widths 4..5. Width-matched, our n3..n4 costs **1.99-2.15**
floors against their **1.36-2.07**, so we are level at width 5 and behind only at width 4.

## Lead this re-opens: adaptive depth

Their adaptive controller spent 80/99 cycles at block 4 rather than the requested 5, and
that is where their throughput lives (~~their block-4 cycles run at an effective 33.17 t/s
vs 21.83 at block 5~~ **superseded: pinned block 4 measures 32.556 t/s and pinned block 5
measures 26.654, `block4-shelf-probe.md`; the 21.83 was mode-biased**) -- while
`tokens_per_cycle` barely moves (3.049 vs 3.059). They get the same tokens for a materially
cheaper cycle by drafting shallower.

We have `LLAMA_SPEC_ADAPTIVE` (branch `adaptive-spec`, default off), previously judged
"doesn't beat best-fixed on stable text -- needs heterogeneous-prompt eval". That verdict
predates the mv-nc, skinny, CPY and drafter-requant work, all of which reshaped the
cycle-cost-vs-depth curve. Worth re-running: our curve above is far from flat
(125.7 ms at n2 to 175.2 at n7), so there is real money in picking depth per cycle.

Caveat before chasing it: our shallow cycles are not cheap the way theirs are
(~~our n4 = 2.15 floors vs their block 4 = 1.36~~ **width-matched: our n3 = 1.99 floors vs
their block 4 = 1.36**), so adaptive depth alone will not close
1.33x. ~~The verify slope is still the root cause.~~ **CORRECTED 2026-08-22: the verify
slope is dense-matmul width scaling and is not recoverable** (`verify-slope-close.md`). The
root cause is the one this paragraph already gestures at - their cheap shallow cycle. That
makes the cost of our own shallow widths the thing to attack, not the slope at width 7.
