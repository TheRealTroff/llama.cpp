# Variable speculation depth: draft deep, verify narrow (2026-09-07)

Status: **OPEN - measuring.** Owner: "let's see what we can uncover." Branch `spec-verify-narrow`
(worktree `llama.cpp-active`), no code change yet; this file prices the policy first.

## The question

Acceptance is a property of the workload (`acceptance-by-prompt.md`: committed/round spans 2.8
to 6.7 at one depth). The Turbo4 pick verifies width 4 (depth 3). Arithmetic on the corpus
survival curves and the Sep-1 Turbo4 width curve said: free-form text is already at its
optimum at width 4; math/JSON want width 7-8 (+22..43%). A per-workload depth is worth money
only if the traffic has saturated content, and it only pays on the deep end, where Turbo4's FA
is outside the GQA-reuse guard (widths 7-8 take the plain Q8 route).

Design on the table: keep the drafter's block deep (DFlash2 is trained at block 8), verify
only k columns. `LLAMA_SPEC_ADAPTIVE` as written shortens the BLOCK (walks the drafter out of
distribution, `occupancy-next.md`); the decoupled form has never been built. In the code the
block is sized from `dp.n_max` at `common/speculative.cpp:1368` and the result is truncated to
the same cap at `:2922` - splitting the two fields is the whole change.

What is NOT known and this sweep measures:

1. Does a deep block's k-prefix accept better or worse than a block-k draft? Benchprompt
   (Aug 23) said better at every position; the Aug 28 corpus said n4's tail beats n6's on
   free-form prompts. Unresolved, and the whole premise.
2. The drafter's cost per block size on the pick (its matmuls, not its FA - the window caps
   its KV at ~1088; the pick's draft KV is f16, so the Turbo4 GQA4 tile is inert).
3. Today's Turbo4 round cost per verify width (the Sep-1 table predates TR=9 and the vector fix).
4. The sha map across widths: three FA families (vector 1-2, GQA6 tile 3-6, plain Q8 7-8).
5. With those: what a fixed-per-workload width, a per-round oracle and the existing EMA
   controller (observing only the verified prefix) would each buy, per prompt.

## Method

`perf/run-depth-corpus.sh` (KV=turbo4, depths 1-7, 9 prompts = benchprompt + perf/prompts 01-08,
300 tokens, `-lv 5` for the per-round `accepted a/n` lines; pick env read from
`run-prod-pick.sh`, single source of truth). One row per (depth, prompt): t/s, round ms,
survival curve over full-block rounds, drafter lattice ms, sha; per-round sequence in `.rounds`.
Quiet arm (`LV=0`) on benchprompt/01/05 to calibrate the verbose arm's timing distortion.
`perf/depth-corpus-sim.py` prices the policies from the rows (fixed = measured; everything
else = arithmetic on measured components, said so in the output).

TAGs: `depthcorpus-turbo4-lv5-sep07`, `depthcorpus-turbo4-lv0-sep07`
(`kvquant-experiments/results/`).

## Results

### 1. Block depth does not change the prefix acceptance - the decoupling premise is NULL

Three prompts emit one sha at every depth 1-7 (01-code-explain, 04-math, 05-json), so their
curves are matched-trajectory. On them the k-prefix of the block-7 survival curve vs the
block-k curve's own sum: 01: -4.0 / +0.9 / -3.2 / -0.2 / -1.9 / -0.1 %, 04: +1.1 / +0.5 / +0.2 /
+1.9 / -0.8 / 0.0, 05: -0.7 / -1.6 / -0.6 / +0.3 / +0.1 / -0.2. Position-1 survival across
depths on 01: .768 .724 .734 .710 .742 .707 .697 - flat, no trend. The Aug-23 benchprompt
finding ("every position improves with block depth", `occupancy-next.md`) does not reproduce
on today's stack (benchprompt position 1: .822 .823 .840 .827 .841 .829 .812); that stack had
no draft window, no fused inject and the drafter still ignored the per-seq depth - the
discrepancy is recorded, not explained.

Consequence: **"draft deep, verify narrow" buys nothing and costs the drafter**: block 8
drafts at 14.1 ms vs 10.2 at block 4 (the drafter's own matmuls leave the SoA family at 6+
columns), so the decoupled estimate sits 3-8% BELOW fixed depth 3 on every free-form prompt.
Do not build the split. The objection to `LLAMA_SPEC_ADAPTIVE` shortening the block is void
on this data; a coupled controller (block = verify) is the right form.

### 2. Fixed depth per workload: 3 everywhere except saturated text, where 7 is +33..37%

Turbo4 line, 8K benchprompt / short corpus prompts, 300 tokens, verbose arm (quiet-arm
calibration below), t/s per depth:

| prompt | d1 | d2 | **d3 (pick)** | d4 | d5 | d6 | d7 | best |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| benchprompt | 19.08 | 24.77 | **27.94** | 26.93 | 24.89 | 25.57 | 25.59 | 3 |
| 01-code-explain | 20.73 | 24.17 | **28.09** | 27.30 | 24.01 | 23.52 | 23.83 | 3 |
| 02-prose-creative | 21.02 | 25.53 | **29.00** | 28.80 | 24.91 | 25.25 | 25.14 | 3 |
| 03-chat-support | 20.45 | 25.14 | **26.34** | 25.26 | 22.75 | 22.57 | 23.41 | 3 |
| 06-algorithms | 21.86 | 26.65 | **28.96** | 25.89 | 21.96 | 21.84 | 22.10 | 3 |
| 08-story | 21.42 | 23.48 | **25.80** | 25.30 | 21.59 | 22.05 | 21.66 | 3 |
| 04-math-derivation | 23.06 | 31.92 | 40.23 | 44.28 | 44.03 | 48.34 | **53.35** | 7 (+32.6%) |
| 05-json-boilerplate | 23.41 | 32.60 | 41.17 | 45.27 | 44.83 | 50.96 | **56.40** | 7 (+37.0%) |

Round ms by verify width (short prompts): 84.7 / 91 / 94.7 / 105.6 / 127 / 130.5 / 131.7 for
depths 1-7; benchprompt +5..10 ms. The cliff is depth 4 -> 5 (width 5 -> 6: +20%, skinny
family), depths 5-6 are dominated by 7 (+3 ms for two more columns). Drafter lattice ms by
block: 8.8 / 9.5 / 10.2 / 11.4 / 13.5 / 14.0 / 14.1. 07-shell-script emits EOS as its first
token on the raw completion endpoint at temperature 0 and is excluded.

### 3. Controllers on the recorded sequences (`depth-policy-sim.py`)

Replayed over the block-7 per-round sequences with the measured cost table; per-prompt best
fixed depth (knows the workload) = the target, mean over 8 prompts:

| policy | mean t/s | vs best-fixed | free-form | saturated |
|---|--:|--:|---|---|
| best fixed per prompt | 34.90 | 0 | | |
| fixed 3 (the pick) | 31.05 | -11.0% | 0..-1% | -25 / -29% |
| fixed 7 | 32.37 | -7.3% | -7..-18% | 0 |
| `spec_adaptive_t` as written, prefix-only observation, explore/16 | 31.64 | -9.3% | -1..-2% | -18 / -24% |
| same + extrapolate unobserved positions from the deepest seen | 32.12 | -8.0% | 0 | -17 / -21% |
| AIMD (+1 on full accept, -1 on a miss) | 32.57 | -6.7% | -6..-10% | -3 / -4% |
| **per-round oracle** (knows this round's a) | **39.33** | **+12.7%** | **+13..+22%** | +2 / +5% |

Nothing that tracks the workload gets there: the EMA (time constant ~12 rounds) does not climb
inside a 40-round saturated run and the explore rounds tax free-form text; AIMD climbs but
oscillates into the width-6 cliff on free-form text. Mixed stream (all prompts back to back,
one controller state): fixed 3 29.30, best-fixed-per-prompt 32.37, every rule 29.2-29.9,
oracle 36.97. Caveat: the 300-token runs make the transient dominate; a long saturated request
would let the EMA converge. **The money is in the per-round variance, not the workload
level**: the oracle is worth +13..22% on exactly the free-form traffic where depth 3 is already
optimal. Only a per-round signal can reach it - the drafter's own confidence is the candidate
(next section).

### 4. The sha map

```
01-code-explain   one sha, depths 1-7
04-math           one sha, depths 1-7
05-json           one sha, depths 1-7
02-prose          1-3 | 4-7
03-chat           1-3 | 4 | 5-7
06-algorithms     1 | 2-3 | 4 | 5-7
08-story          1 | 2-3 | 4 | 5-7
benchprompt       1 | 2 | 3 | 4 | 5-7
```
Forks sit on the kernel-family edges (width 2->3 vector->GQA tile, 4->5 SoA w4->w5, 5->6
skinny); a depth controller cannot be sha-gated, its price is a KLD against the fixed-depth text.

### 5. Verbose-arm timing

Quiet arm (`LV=0`) on benchprompt/01/05: depth-1 rounds 94.85 / 84.98 / 84.72 ms vs verbose
94.97 / 84.84 / 84.03 - the `-lv 5` distortion is within noise at 300 tokens. The three points outside 2% (depth-2 benchprompt/01, which overlapped the
worktree build, and depth-4 05) re-ran at 94.61 / 90.81 / 103.10 vs verbose 95.80 / 90.96 /
103.21 (TAGs `depthcorpus-turbo4-lv0-rerun-sep07-*`): 21/21 points within 1.3%, the verbose
arm's timing stands.

### 6. The drafter's confidence IS the per-round signal (`depth-conf-sim.py`, TAG `depthconf-turbo4-n7-sep07`)

`DFLASH_CONF_LOG=1` (this branch) logs, per drafted position on the greedy path, the softmax
top-1 over the selector's top-k lattice scores (p) and the top-2 margin (m). Block 7 on the 8
prompts, same shas and survival curves as the sweep (the worktree build reproduces prod).

Calibration, observed positions only, all prompts: p in [0.9,1) accepts 93.7% (n=1378), [0.8,0.9)
65%, [0.7,0.8) 60%, [0.5,0.7) ~47%, [0.3,0.5) ~35%, below 0.3 ~20%. Monotone; the signal is real.

Policies, priced per round on the recorded block-7 sequences with the sweep's cost table
(arithmetic on measured components, LOPO = calibration bins fitted on the other 7 prompts):

| policy | mean t/s | vs fixed 3 | free-form (6 prompts) | math / json |
|---|--:|--:|---|---|
| fixed 3 (pick) | 31.05 | 0 | 0 | 0 |
| best fixed per prompt (knows the workload) | 34.90 | +12.4% | 0 | +33 / +41% |
| leading-run threshold p >= 0.6 | 34.85 | +12.2% | -1..+4% | +31 / +41% |
| **EV(p), coupled cost** (k = argmax (1+sum survival)/cost, survival from calibrated p) | **36.25** | **+16.7%** | **+2.6..+10%** | +34 / +42% |
| EV(p), honest decoupled cost (block 7 every round, +4 ms drafter) | **35.39** | **+14.0%** | **-1.0..+4.3%** | +34 / +41% |
| EV(p), hybrid block = max(4, k_prev+2) | 35.01 | +12.8% | -0.6..+2.7% | +30 / +41% |
| per-round oracle | 39.33 | +26.7% | +13..+22% | +5 / +2% |

Reading: **a per-round verify depth chosen from the drafter's own confidence beats a
workload-oracle fixed depth without knowing the workload** (35.4 vs 34.9 mean). On saturated
text it takes the whole fixed-7 win; on free-form text the confidence gain (+2.6..+10% at
coupled cost) is halved by the deep block's drafter tax (+4 ms/round, block 8 vs 4), net
-1..+4%. The per-round oracle says +13..22% is there on free-form text; EV(p) captures ~40% of
it, the rest is the calibration's ~65% accuracy in the 0.5-0.9 bins. The margin m carries the
same information (EV(m) 36.25).

What a build would need (not started - owner's call):
1. `common/speculative.cpp`: a block-depth field separate from the verify cap (the block from
   `params.n_max`, the truncation at `:2922` from the per-round pick) - ~10 lines.
2. Expose the per-position p to the server (a vector on the draft params, filled on the greedy
   path; temperature > 0 already has `dists`).
3. Server: replace `spec_adaptive_t`'s per-position EMA with a 10-bin acceptance-by-p table
   (seeded from this file, updated online from the verify result) and the per-width cost EMA
   seeded from the sweep; per round k = argmax over the efficient widths {1,2,3,4,7} of
   (1 + sum survival)/cost. Candidate widths 5-6 are dominated on Turbo4 (+3 ms for 7).
4. Recover the drafter tax: the block-8 draft runs the drafter's projections on the skinny
   family (14.1 ms vs 10.2 at block 4); a drafter-side width-8 SoA kernel or the hybrid block
   rule is worth up to +4% on free-form text.
5. Gate: KLD vs the fixed-depth text (the widths cross three kernel families, section 4), then
   an interleaved e2e A/B on the corpus against fixed depth 3, per prompt, plus the 2- and
   4-slot points under `LLAMA_SPEC_SLOT_BUDGET` and a mixed stream.

## Open

- **Build the EV(p) controller?** Owner's call. Priced at +14% mean over this corpus
  (+34/+41% saturated, -1..+4% free-form) with the honest deep-block cost; every number above
  is replay arithmetic, the e2e A/B is the proof.
- Drafter tax of the block-8 draft (4 ms/round) - the free-form half of the win.
- Width 7-8 GQA-reuse tile: target-side prerequisite for the deep end on Turbo4
  (`turbo4-fa-gqa-reuse.md` open item 4). Not a drafter lever (the pick's draft KV is f16,
  the window caps the drafter's KV at ~1088).
- Calibration accuracy in the 0.5-0.9 bins (~65%): a second feature (position index, the
  previous round's outcome) may lift EV(p) toward the oracle. 300-token runs, one rep each,
  8 prompts - no error bars; math/JSON have 39-41 rounds.
- Long context: every number here is 8K or shorter. At a filled 96K the width curve flattens
  (batched FA amortizes the KV stream over the query tile) and the optimum should move deeper;
  unmeasured.
