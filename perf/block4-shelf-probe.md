# Is dflash_mlx's cheap block-4 cycle real, or a lazy-eval artifact?

Status: **closed - the shelf is real, and it is their whole advantage.** Measured
2026-08-22, dflash_mlx 0.1.10+omlx.6, prompt sha1 `c0653ba4af5e` (identical to the
head-to-head), harness `perf/run-block4-shelf.sh` (`TAG=shelf-aug22`), 3 reps per config.
Our numbers are the 16:02-build figures from `slope-sweep.md`; nothing of ours was rebuilt
or re-measured. llama.cpp side recorded at prod `90e0bcd08`.

## Answer: real. Their derived 91.9 ms/cycle measures 95.00 pinned.

| their config | ms/cycle | tok/cycle | t/s | `cycles_by_block` |
|---|--:|--:|--:|---|
| **fixed block 4** | **95.00** | 3.0928 | **32.556 +/- 0.007** | empty (pinned) |
| adaptive (their default) | 102.15 | 3.0303 | 29.666 +/- 0.097 | {1:1, 4:81, 5:17} |
| fixed block 5 | 137.26 | 3.6585 | 26.654 +/- 0.058 | empty (pinned) |

Hypothesis (a) confirmed: the deferral term `d` implied by 95.00 against the derived 91.9
is **3.1 ms**, inside the pre-registered `d <= 8.4` bound. Hypothesis (b) never could have
made the gap illusory - conservation capped `T4` at 100.3 ms either way - and it turns out
to account for 3 ms of a 50 ms difference.

**Controls both passed.** Adaptive reproduces its archived 29.613 three separate times
(29.561, 29.613, 29.666), so their side has not drifted and the window is trustworthy.
Pinning costs no acceptance: fixed block 4 commits 3.0928 tok/cycle against adaptive's
block-4 rows at 3.049, marginally *better*, which kills the "the controller picks its
moments" reading.

## Three findings the question did not ask for

**1. Their adaptive controller is leaving ~10% on the table here.** Fixed block 4 (32.556)
beats their own default (29.666) by 9.7% on this prompt, because adaptive spends 17 of 99
cycles at the expensive block 5 plus 12 probe cycles. **Every cross-framework number on
record therefore compares our best config against a suboptimal config of theirs.**
Best-vs-best is **25.04 vs 32.56 = 1.302x**, not 1.184x.

**2. Their controller escalates exactly when acceptance is bad.** Fixed block 5 accepts
53.2%/draft; adaptive's block-5 cycles accept 41.2%. So the block-5 rows in the archived
adaptive artifact are not representative of block 5 - they are the cycles where drafting was
already going badly. This is why the earlier arithmetic predicting fixed block 5 at
21.8-24.4 t/s was wrong: it used adaptive's 3.0588 tok/cycle where the true fixed figure is
3.6585. The *cycle cost* prediction (125-140 ms) was right at 137.26.

**3. At matched depth 5 the two engines are nearly level - the advantage is all at
block <= 4.**

> **CORRECTED 2026-08-22 (`mlx-cycle-capture.md`): the table below is off by one column.**
> Their block *b* verifies *b* columns (`spec_epoch.py:2247-2257`); our depth *d* verifies
> *d+1* (`slope-sweep.md:13`). So each row compares their width *b* against our width *b+1*.
> Matched by width: their block 5 (width 5, 137.26) vs our depth 4 (width 5, 144.9) =
> **1.06x**, and their block 4 is a **width-4** cycle for which **we have no measurement** -
> that would be our depth 3. The "1.53x at depth 4" below is width 4 vs width 5. The
> qualitative reading - flat-but-high vs cheap-shelf-below-a-cliff - survives; their cliff
> just sits between width 4 and 5, not 5 and 6.

| depth | their ms/cycle | our ms/round | ratio | their tok | our tok |
|---|--:|--:|--:|--:|--:|
| 4 | 95.00 | 144.9 | **1.53x** | 3.0928 | 3.19 |
| 5 | 137.26 | 147.5 | **1.07x** | 3.6585 | 3.45 |

**Our curve is flat but high (144.9 / 147.5 / 149.8 at n4/n5/n6); theirs is steep with a
cheap shelf below a cliff.** So "our verify slope is too steep" was the wrong diagnosis
twice over - `verify-slope-close.md` showed the slope is dense-op width scaling, and this
shows we are level with them at depth 5 anyway. The gap is one operating point.

## The contradiction this leaves, and it is the next task

Their block-4 cycle runs a drafter **and** a verify in 95.00 ms. Our five-column verify pass
*alone* costs 126.3 ms in-graph (119.0 on llama-bench). Yet the microbench comparison on
record (`head-to-head-aug22.md`) says we are at parity with MLX at n=5: ours 1.81x, theirs
1.74x. **Those cannot both be true.** Exactly three ways out:

1. their block-4 cycle does not run a full 5-column forward pass;
2. their drafter **overlaps** the verify on the GPU timeline, where ours is 16.4 ms of fully
   serialized drafter wait (`round-decomp-fused.md`);
3. the microbench parity claim was measured wrongly.

Mechanism evidence already in hand: `mlx.metallib` contains **`qmv_fast` and `qmm_t`**, so
they have the same class of small-batch quantized-matmul routing cliff we do. Theirs sits
between block 4 and 5 (verify width 5 -> 6); ours sits at `ne11=9`. Their fast path is
simply much faster than ours at the same width, which is what needs explaining.

Follow-up opened as `mlx-cycle-capture.md`.

## Gotchas, both of which cost a run

- **`DFLASH_VERIFY_MODE` is ignored by `dflash benchmark`.** `_resolve_verify_mode`
  (`runtime/config.py:759`) reads the env var only when the CLI value is `None`, and the
  benchmark CLI always supplies one. Use **`--verify-mode`** (`benchmark.py:1787`). My first
  block-5 arm set the env, silently ran adaptive, and reproduced the adaptive number exactly
  - no error anywhere. **Always confirm `adaptive_metrics.cycles_by_block` is absent before
  believing an arm is fixed-block**; the config block in `results.json` also echoes
  `verify_mode`.
- `--block-tokens 4` is pinned regardless of the flag, because `from_runtime` bails at
  `full_block_tokens <= 4` (`spec_epoch.py:343`). That is the only reason the block-4 arm was
  valid despite the env doing nothing.
- Each config takes ~6 min wall (3 reps, 60 s cooldown, ~67 s prefill + ~10 s generation per
  rep). Budget ~25 min for the four arms.
