# Head to head, both sides re-measured (2026-08-22)

Supersedes `head-to-head-cooled.md`, which was flagged stale: 24 commits had landed under
it and the dflash side had never been re-run. Harness `perf/run-head-to-head.sh`
(`TAG=h2h-aug22`), 5 runs per side, same 8288-token B-tree prompt regenerated from
`benchprompt.txt` for both sides (**sha1 c0653ba4af5e, verified identical**).

llama.cpp: prod `814eaf37`, binary 2026-08-22 16:02, prod pick = uniform Q4_0 target +
pure-Q4_0 drafter + `GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8
GGML_GDN_FUSE_WB=1` + dflash n6.
dflash_mlx 0.1.10+omlx.6, block 5, `w4:gs64`, adaptive verify (its default).

| side | t/s | spread | n |
|---|---:|---|---:|
| llama.cpp | **25.004** | +/- 0.015 (24.983 - 25.019) | 5 |
| dflash_mlx | **29.613** | +/- 0.060 (29.507 - 29.651) | 5 |
| **gap** | **1.184x** | | |

Both sides are deterministic: our 5 runs share output sha1 `9ad7e023c6ab` and 46.9%
acceptance; their 5 runs report **identical** `cycles=99` and `accepted_from_draft=201`
every time.

**Their number did not drift.** 29.613 against the archived 29.55 (+0.2%), with the
adaptive counters reproducing exactly (`cycles_by_block={1:1, 4:81, 5:17}`,
`cycles_by_mode={full:7, probe:12, reduced:80}`, `tokens_per_cycle` 3.0303,
`acceptance_ratio` 0.67). So the staleness risk that motivated this re-run was real in
principle but did not materialise - the 1.18x quoted from stale halves happens to be right,
and is now verified on both sides simultaneously at a recorded sha.

## Decomposition

Round/cycle cost from committed tokens and throughput:

| | committed/cycle | cycle ms | ms per committed token |
|---|---:|---:|---:|
| llama.cpp (n6) | 3.75 | 150.0 | 40.00 |
| dflash_mlx | 3.0303 | 102.3 | 33.77 |

    gap 1.184x = (their tokens/cycle / ours) x (our ms/cycle / theirs)
               = 0.808 x 1.466

**We commit 1.238x more per cycle; their cycle is 1.466x cheaper.** Same shape as the
2026-08-21 decomposition, with the cycle-cost deficit improved from **1.589x to 1.466x**.
Our cycle is 2.05 batch-1 floors (floor re-measured at 13.656 t/s = 73.2 ms in
`slope-sweep.md`).

**The whole gap is one number.** Matching 29.613 t/s at our own 3.75 committed/round needs
a **126.6 ms** round against today's 150.0 - a **23.3 ms** cut. `round-decomp-fused.md`
scopes the verify-slope lever at ~20 ms, i.e. **~86% of the entire remaining gap**.
Everything else on the lever board is rounding error next to it.

## Drafter quality is still not the problem

Their attempted drafts are **not** cycles x block, because adaptive verify shortens blocks:
1x1 + 81x4 + 17x5 = **410 over 99 cycles, average depth 4.14**. Per-draft acceptance is
201/410 = **49.0%**.

Ours is 46.9% at n6 - but at matched depth 4 the sweep measured **55.7%**. So we draft
better at comparable depth, confirming `acceptance-metric-conversion.md` on fresh data.

Matched-depth cost, which is the cleanest apples-to-apples available:

| at depth ~4 | cycle ms | committed | ms per committed token |
|---|---:|---:|---:|
| llama.cpp n4 | 144.9 | 3.19 | 45.42 |
| dflash_mlx (avg 4.14) | 102.3 | 3.03 | 33.77 |

**Our cycle costs 1.42x theirs at comparable depth**, while committing slightly more. The
deficit is cycle cost, everywhere, at every depth.

## Caveat: this compares a fixed-depth config against an adaptive one

We run fixed n6 - 97.5% of rounds at verify width 7 (`slope-sweep.md`). They run adaptive
and spend 81 of 99 cycles at block 4. So part of this gap is depth *policy*, not per-round
efficiency, and their cheap shallow cycles are a policy we cannot currently copy profitably:
our n4 cycle is 144.9/73.2 = **1.98 batch-1 floors**, while theirs averages 102.3/67.66 =
**~1.51** (their floor is the 2026-08-21 figure and was not re-measured here, so treat that
one as approximate).

That is why adaptive depth is not a shortcut here. Per `slope-sweep.md`, flattening verify
widths 2-5 is the prerequisite that would make an adaptive policy pay; the verify-slope
lever above is the thing that shrinks the gap regardless of policy.
