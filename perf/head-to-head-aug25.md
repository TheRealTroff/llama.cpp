# Head-to-head 2026-08-25: the pinned 95.00 is real, best-vs-best is 1.32x

Status: **measured 2026-08-25, all three arms in one session.** This closes the
"deferred same-session head-to-head" item: their pinned block-4 number - the target
every gap claim routes through - had never been re-measured in the same session as our
side. It reproduces. Harness: `perf/run-head-to-head.sh` (TAG=h2h-aug25, NRUN=3,
PAUSE=60) plus a `--block-tokens 4` arm (auto-pinned; `cycles_by_block` absent in the
artifact confirms fixed mode per `block4-shelf-probe.md`'s gotcha). Raw:
`kvquant-experiments/results/h2h-aug25-*`, artifacts
`.artifacts/dflash/benchmarks/20260825-2032*/-2038*`.

## Numbers

prod `6549de807` (clean), same 31522-char prompt both sides (sha1 c0653ba4af5e):

| arm | t/s | detail |
|---|---:|---|
| llama.cpp prod pick (dflash n6, full env) | **24.386 +/- 0.032** | acc 46.9%, sha 9ad7e023c6ab (canonical) |
| dflash_mlx block 5 (adaptive default) | 28.882 +/- 0.062 | 0.67 acc/committed, 99 cycles, 3.030 tok/cycle |
| dflash_mlx block 4 (pinned) | **32.26 +/- 0.02** | 97 cycles, 3.0928 tok/cycle = **95.9 ms/cycle** |

- **The whole machine is ~2% slower than the Aug-22 session** (llama.cpp 24.39 vs
  25.02 recorded, dflash block5 28.88 vs 29.61, block4 32.26 vs 32.556) - the adaptive
  gap is **1.184x, IDENTICAL to Aug 22 to three digits**, so this is machine state,
  not a regression on either side. Session-relative ratios are the stable quantity.
- **Their pinned cycle validates: 95.9 ms today vs 95.00 archived** (consistent with
  the ~1-2% machine drift). Their side of the ledger is measured, not pinned, from
  today.
- **Best-vs-best: 32.26 / 24.39 = 1.323x** (archived 32.556/25.04 = 1.302x - same
  number through the drift). Per token: theirs 31.0 ms, ours 41.0 ms at 3.75
  committed/round and ~154 ms/round.

## What this settles

Combined with today's six kernel-side refutations (`width4-sumy-fold-refuted.md`,
`width4-y-operand-width.md`, `width4-addressing-refuted.md`,
`skinny-staging-refuted.md`, `skinny-grid-refuted.md`): the 1.30x best-vs-best gap is
real, cross-checked in one session at a recorded sha, and **not available from any
kernel-local lever that has been proposed so far**. The remaining board is round
structure: the drafter's 5.3 ms full-vocab head, operating points that avoid the
width-4 regime (MTP d1 arithmetic in `width4-gap-decomposition.md`), and whatever a
per-kernel decode of their block-4 capture says their 95.9 ms is actually spent on
(`mlx-cycle-capture.md` infrastructure exists for this).
